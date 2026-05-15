"""
run1/05_determ_beliefs/run.py — deterministic-belief ablation (DetBTAgent).

Combines the coverage sweep (02_data_efficiency) and the arity sweep
(022_data_efficiency_arity) in one script, swapping DlbtAgent for
DetBTAgent throughout.

DetBTAgent replaces Monte Carlo Dirichlet integration with a deterministic
forward pass through the Dirichlet mean:

    μ_x  = α_x / Σ_k α_xk
    p̃_xt = σ(μ_x · ΔU_t)

SLDA is always computed on all eligible tasks as a reference baseline.

Usage (from repo root):
    python experiments/behavior/run1/05_determ_beliefs/run.py [--sweep {coverage,arity,both}]
"""

import argparse
import gc
import pickle
import sys
import warnings
from pathlib import Path

# Suppress benign third-party warnings that flood the log.
# open_clip: architectural mismatch between model config and pretrained tag.
warnings.filterwarnings("ignore", message="QuickGELU mismatch")
# sklearn RidgeCV: degenerate gram matrix at tiny budgets (1–2 samples / task).
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.agents.detbt import DetBTAgent
from dlbt.agents.dlbt import DlbtAgent       # used only for CLIP feature extraction
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--sweep", choices=["coverage", "arity", "both"],
                     default="both",
                     help="Which sweep to run (default: both).")
args = _parser.parse_args()

RUN_COVERAGE = args.sweep in ("coverage", "both")
RUN_ARITY    = args.sweep in ("arity", "both")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[4]
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"Sweep: {args.sweep}  |  RUN_TAG: {cfg.RUN_TAG}")

np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict   = load_image_refs(_REPO_ROOT / cfg.METADATA)
all_refs    = image_refs_as_list(refs_dict)
refs_by_uid = {r.uid: r for r in all_refs}
print(f"Loaded {len(refs_dict)} image refs.")

# ---------------------------------------------------------------------------
# Load + filter behavioural data
# ---------------------------------------------------------------------------
print("\nLoading behavioural data...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
print(f"  Raw trials: {len(df_raw):,}  "
      f"({df_raw['assignment_id'].nunique()} assignments)")

df_filtered, _ = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
print(f"  Filtered: {df_filtered['assignment_id'].nunique()} assignments remain.")

all_tasks_ordered = cfg.eligible_tasks(df_filtered)
n_all_tasks       = len(all_tasks_ordered)
print(f"  Eligible tasks: {n_all_tasks}")

# Arity breakdown
tasks_by_arity = cfg.tasks_by_arity(all_tasks_ordered)
for a, ts in tasks_by_arity.items():
    print(f"    arity {a}: {len(ts)} tasks")

# N_TASKS_PER_ARITY: use config or fall back to min across arities
n_tasks_per_arity = cfg.N_TASKS_PER_ARITY
if n_tasks_per_arity is None:
    n_tasks_per_arity = min(len(ts) for ts in tasks_by_arity.values() if ts)
print(f"  N_TASKS_PER_ARITY = {n_tasks_per_arity}")

_beh_id_eligible = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in set(all_tasks_ordered)}
full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _beh_id_eligible,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)
print(f"  Aggregated: {len(full_ds):,} cells  "
      f"({full_ds.df['task_name'].nunique()} tasks, "
      f"{full_ds.df['uid'].nunique()} images)")

# ---------------------------------------------------------------------------
# Probe matrix ordering
# ---------------------------------------------------------------------------
probe_refs_ordered = sorted(
    [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
probe_uids_ordered = [r.uid for r in probe_refs_ordered]
n_probe            = len(probe_uids_ordered)
uid_to_row         = {uid: i for i, uid in enumerate(probe_uids_ordered)}
task_to_col        = {t: j for j, t in enumerate(all_tasks_ordered)}
print(f"  Probe images: {n_probe}")

# ---------------------------------------------------------------------------
# Ground truth probe matrix  [n_probe × n_tasks]
# ---------------------------------------------------------------------------
probe_cells_df = full_ds.df[full_ds.df["uid"].isin(probe_uids)].copy()
true_matrix    = np.full((n_probe, n_all_tasks), np.nan)
count_matrix   = np.zeros((n_probe, n_all_tasks), dtype=np.int32)
for row in probe_cells_df.itertuples(index=False):
    i     = uid_to_row.get(row.uid)
    j     = task_to_col.get(row.task_name)
    total = row.count_0 + row.count_1
    if i is not None and j is not None and total > 0:
        true_matrix[i, j]  = row.count_1 / total
        count_matrix[i, j] = total
n_filled = int((~np.isnan(true_matrix)).sum())
print(f"  Ground truth probe matrix: {n_filled}/{n_probe * n_all_tasks} cells filled.")

_nf_mask = count_matrix > 1
if _nf_mask.any():
    _p = true_matrix[_nf_mask]
    _n = count_matrix[_nf_mask].astype(float)
    probe_noise_floor = float(np.mean(_p * (1 - _p) / (_n - 1)))
else:
    probe_noise_floor = 0.0

_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = (float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2))
                   - probe_noise_floor)
print(f"  Probe NF: {probe_noise_floor:.5f}  "
      f"random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Fixed 10 % eval split of main cells (for early stopping)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split     = np.random.default_rng(cfg.SEED)
n_eval_cells  = max(1, int(len(main_cells_df) * 0.10))
eval_idx      = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask     = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df  = main_cells_df[eval_mask].reset_index(drop=True)
pool_df  = main_cells_df[~eval_mask].reset_index(drop=True)
print(f"\n  Eval cells (early stopping): {len(eval_df)}")
print(f"  Train pool cells (90 %%):    {len(pool_df)}")

# ---------------------------------------------------------------------------
# Per-task trial pools
# ---------------------------------------------------------------------------
task_trial_pools: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in pool_df.itertuples(index=False):
    tn = row.task_name
    if tn not in task_trial_pools:
        continue
    pool = task_trial_pools[tn]
    for _ in range(int(row.count_0)):
        pool.append((row.uid, 0))
    for _ in range(int(row.count_1)):
        pool.append((row.uid, 1))

pool_sizes      = {t: len(task_trial_pools[t]) for t in all_tasks_ordered}
global_min_pool = min(pool_sizes.values())
print(f"\n  Trial pool sizes — global_min: {global_min_pool}, "
      f"max: {max(pool_sizes.values())}, "
      f"total: {sum(pool_sizes.values()):,}")

# ---------------------------------------------------------------------------
# CLIP feature cache (frozen throughout; extracted with a temporary DlbtAgent)
# ---------------------------------------------------------------------------
_agent_tmp  = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                        device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
_cache_path = _REPO_ROOT / cfg.CACHE_PATH
if _cache_path.exists():
    _agent_tmp.load_cache(str(_cache_path))
else:
    _agent_tmp.precompute_features(all_refs)
    _agent_tmp.save_cache(str(_cache_path))
frozen_clip = {uid: feat.clone() for uid, feat in _agent_tmp._cache.items()}
del _agent_tmp
print(f"CLIP cache ready ({len(frozen_clip)} images).")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _budget_series(tasks: list) -> list[int]:
    """
    Valid budget points for a trace over `tasks`.

    min_budget = n_tasks               (1 trial per task)
    max_budget = global_min_pool × n   (global cap so all seeds share the same
                                        endpoint; never exceeds any pool size)
    """
    n     = len(tasks)
    min_b = n
    max_b = global_min_pool * n
    if max_b < min_b:
        return []
    pts = {min_b, max_b}
    for b in cfg.TRIAL_BUDGETS:
        if min_b <= b <= max_b:
            pts.add(b)
    return sorted(pts)


def _uniform_sample(tasks: list, budget: int,
                    rng: np.random.Generator) -> BehavioralDataset:
    """
    Allocate `budget` trials uniformly across `tasks` without replacement.

    Every task receives q = floor(budget / n) trials.
    A randomly chosen subset of r = budget % n tasks each receive q+1.
    """
    n     = len(tasks)
    q     = budget // n
    r     = budget % n
    extra = set(int(i) for i in rng.choice(n, size=r, replace=False))
    rows  = []
    for i, task_name in enumerate(tasks):
        k    = q + (1 if i in extra else 0)
        pool = task_trial_pools[task_name]
        if k == 0:
            continue
        chosen = rng.choice(len(pool), size=k, replace=False)
        for idx in chosen:
            uid, outcome = pool[idx]
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _eval_ds_for_tasks(tasks: list) -> BehavioralDataset:
    """Slice the fixed eval set to the current training tasks."""
    sub = eval_df[eval_df["task_name"].isin(tasks)].reset_index(drop=True)
    if len(sub) > 0:
        return BehavioralDataset(sub)
    fb = probe_cells_df[probe_cells_df["task_name"].isin(tasks)].head(20).reset_index(drop=True)
    return BehavioralDataset(fb) if len(fb) > 0 else BehavioralDataset(
        pd.DataFrame(columns=["uid", "task_name", "count_0", "count_1"]))


def _init_agent() -> DetBTAgent:
    """Fresh DetBTAgent with frozen CLIP cache and initialised mapper bias."""
    torch.manual_seed(cfg.SEEDS[0])
    agent = DetBTAgent(
        freeze_encoder    = True,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    if cfg.INIT_MODE == "random":
        rng_init  = np.random.default_rng(cfg.INIT_SEED)
        alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                     size=(_linear.bias.shape[0],)).astype(np.float32)
        b_init    = np.log(np.exp(alpha_rnd) - 1.0)
        with torch.no_grad():
            _linear.bias.copy_(torch.from_numpy(b_init).to(device))
    else:
        bv = float(np.log(np.exp(cfg.INIT_ALPHA) - 1.0))
        with torch.no_grad():
            _linear.bias.fill_(bv)
    return agent


def _train_one(agent: DetBTAgent,
               train_ds: BehavioralDataset,
               eval_ds_local: BehavioralDataset):
    """Single-phase training (frozen encoder — no attnpool fine-tuning)."""
    return train_dlbt(
        agent, train_ds, eval_ds_local, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )


@torch.no_grad()
def _det_probe_matrix(agent: DetBTAgent) -> np.ndarray:
    """[n_probe × n_tasks] DetBTAgent predicted P(right) for all probe × tasks."""
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    for j, task_name in enumerate(all_tasks_ordered):
        task  = get_task(task_name)
        probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
        pred[:, j] = probs
    return pred


def _probe_cmse_net(pred_mat: np.ndarray) -> float:
    """cMSE − probe noise floor over cells with empirical data."""
    valid = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    if not valid.any():
        return float("nan")
    return float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor


def _save_ckpt(agent: DetBTAgent, tag: str) -> Path:
    p = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_{tag}.pt"
    torch.save({"mapper": agent.mapper.state_dict()}, p)
    return p


def _trials_per_task(tasks: list, train_ds: BehavioralDataset) -> dict:
    return {t: int((train_ds.df[train_ds.df["task_name"] == t]
                    [["count_0", "count_1"]].sum(axis=1)).sum())
            for t in tasks}


# ---------------------------------------------------------------------------
# SLDA helpers  (identical to 02 and 022 — SLDA is agent-independent)
# ---------------------------------------------------------------------------

def _fit_slda(tasks: list, train_ds: BehavioralDataset):
    scalers, models, temps = {}, {}, {}
    for task_name in tasks:
        group = train_ds.df[train_ds.df["task_name"] == task_name]
        uids  = [uid for uid in group["uid"].tolist() if uid in frozen_clip]
        if len(uids) < 1:
            continue
        X       = np.array([frozen_clip[uid].cpu().numpy() for uid in uids])
        g_sub   = group[group["uid"].isin(uids)]
        totals  = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_right = g_sub["count_1"].values / np.clip(totals, 1, None)
        scaler  = StandardScaler(with_mean=(len(uids) >= 5),
                                 with_std=(len(uids) >= 5))
        X_sc    = scaler.fit_transform(X)
        model   = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
        model.fit(X_sc, p_right)
        p_pred  = np.clip(model.predict(X_sc), 1e-6, 1 - 1e-6)
        logits  = np.log(p_pred / (1 - p_pred))

        def _nll(log_tau, logits=logits, y=p_right):
            p = 1 / (1 + np.exp(-logits / np.exp(log_tau)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        opt = minimize_scalar(_nll, bounds=(-3.0, 3.0), method="bounded")
        scalers[task_name] = scaler
        models[task_name]  = model
        temps[task_name]   = float(np.exp(opt.x))
    return scalers, models, temps


def _slda_probe_matrix(scalers: dict, models: dict, temps: dict) -> np.ndarray:
    pred            = np.full((n_probe, n_all_tasks), np.nan)
    probe_uids_clip = [uid for uid in probe_uids_ordered if uid in frozen_clip]
    probe_X         = np.array([frozen_clip[uid].cpu().numpy()
                                 for uid in probe_uids_clip])
    for j, task_name in enumerate(all_tasks_ordered):
        if task_name not in models:
            continue
        X_sc   = scalers[task_name].transform(probe_X)
        p_pred = np.clip(models[task_name].predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))
        p_cal  = 1 / (1 + np.exp(-logits / temps[task_name]))
        for i_clip, uid in enumerate(probe_uids_clip):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_cal[i_clip])
    return pred


# ---------------------------------------------------------------------------
# Shared SLDA sweep  (used by both coverage and arity analysis)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"SLDA sweep — all {n_all_tasks} tasks")

_slda_budget_series = _budget_series(all_tasks_ordered)
print(f"  Budget series: {_slda_budget_series}")

slda_results: dict = {
    "tasks":      all_tasks_ordered,
    "n_tasks":    n_all_tasks,
    "min_budget": _slda_budget_series[0]  if _slda_budget_series else None,
    "max_budget": _slda_budget_series[-1] if _slda_budget_series else None,
    "budgets":    {},
}

rng_slda = np.random.default_rng(cfg.SEED + 100)
for _b in _slda_budget_series:
    print(f"\n  SLDA  budget={_b}")
    _ds_b           = _uniform_sample(all_tasks_ordered, _b, rng_slda)
    _sc, _md, _tp   = _fit_slda(all_tasks_ordered, _ds_b)
    _pm_b           = _slda_probe_matrix(_sc, _md, _tp)
    _mse_b          = _probe_cmse_net(_pm_b)
    print(f"  Fitted {len(_md)}/{n_all_tasks} tasks  probe_cmse_net={_mse_b:.5f}")
    slda_results["budgets"][str(_b)] = {
        "n_trials":    _b,
        "probe_cmse_net": _mse_b,
        "pred_matrix": _pm_b,
    }
    del _ds_b, _sc, _md, _tp

# ===========================================================================
# Random-init DetBT baseline
# ===========================================================================
print("\nComputing random-init DetBT baseline...")
_agent_rand          = _init_agent()
_pred_rand           = _det_probe_matrix(_agent_rand)
random_init_cmse_net = _probe_cmse_net(_pred_rand)
del _agent_rand
print(f"  Random-init DetBT cMSE−NF: {random_init_cmse_net:.5f}")

# ===========================================================================
# Coverage sweep
# ===========================================================================
coverage_results: dict = {}

if RUN_COVERAGE:
    for seed_i, seed_val in enumerate(cfg.SEEDS):
        seed_key = f"seed_{seed_i}"
        print(f"\n{'=' * 60}")
        print(f"DetBT coverage sweep — seed {seed_i}  (seed_val={seed_val})")

        rng_order     = np.random.default_rng(seed_val)
        task_ordering = list(rng_order.permutation(all_tasks_ordered))
        coverage_results[seed_key] = {
            "task_ordering": task_ordering,
            "coverage":      {},
        }

        for frac in cfg.COVERAGE_FRACS:
            n_frac       = max(1, round(frac * n_all_tasks))
            task_subset  = task_ordering[:n_frac]
            frac_key     = f"{frac:.2f}"
            budgets_frac = _budget_series(task_subset)

            print(f"\n  Coverage {frac:.0%} — {n_frac} tasks  "
                  f"budgets: {budgets_frac}")

            cov_entry: dict = {
                "tasks":      task_subset,
                "n_tasks":    n_frac,
                "min_budget": budgets_frac[0]  if budgets_frac else None,
                "max_budget": budgets_frac[-1] if budgets_frac else None,
                "budgets":    {},
            }

            rng_run = np.random.default_rng(seed_val * 1000 + int(frac * 100))

            for budget in budgets_frac:
                print(f"\n  --- cov={frac:.0%}  budget={budget} ---")
                train_ds_b = _uniform_sample(task_subset, budget, rng_run)
                eval_ds_b  = _eval_ds_for_tasks(task_subset)
                n_cells    = len(train_ds_b)
                n_trials   = int((train_ds_b.df["count_0"] +
                                  train_ds_b.df["count_1"]).sum())
                print(f"  cells={n_cells}  trials={n_trials}")

                if n_cells == 0:
                    print("  Empty dataset — skipping.")
                    continue

                agent  = _init_agent()
                result = _train_one(agent, train_ds_b, eval_ds_b)
                print(f"  best_epoch={result.best_epoch}  "
                      f"eval_mse={result.best_val_mse:.5f}")

                pred_mat = _det_probe_matrix(agent)
                mse      = _probe_cmse_net(pred_mat)
                print(f"  probe_cmse_net={mse:.5f}")

                ckpt_tag  = f"cov{frac:.2f}_seed{seed_i}_budget{budget}"
                ckpt_path = _save_ckpt(agent, ckpt_tag)
                print(f"  Saved ckpt → {ckpt_path.name}")

                cov_entry["budgets"][str(budget)] = {
                    "n_trials":        n_trials,
                    "n_cells":         n_cells,
                    "trials_per_task": _trials_per_task(task_subset, train_ds_b),
                    "probe_cmse_net":  mse,
                    "pred_matrix":     pred_mat,
                    "best_epoch":      result.best_epoch,
                    "best_val_mse":    result.best_val_mse,
                    "ckpt_path":       str(ckpt_path),
                }

                del agent, train_ds_b
                gc.collect()
                torch.cuda.empty_cache()

            coverage_results[seed_key]["coverage"][frac_key] = cov_entry

    cov_summary = {
        "run_tag":              cfg.RUN_TAG,
        "coverage_fracs":       cfg.COVERAGE_FRACS,
        "trial_budgets":        cfg.TRIAL_BUDGETS,
        "seeds":                cfg.SEEDS,
        "all_tasks_ordered":    all_tasks_ordered,
        "probe_uids_ordered":   probe_uids_ordered,
        "true_matrix":          true_matrix,
        "probe_noise_floor":    probe_noise_floor,
        "random_cmse_net":      random_cmse_net,
        "random_init_cmse_net": random_init_cmse_net,
        "slda":                 slda_results,
        "dlbt":                 coverage_results,
    }
    cov_out = cfg.RESULTS_DIR / f"coverage_sweep_{cfg.RUN_TAG}.pkl"
    with open(cov_out, "wb") as f:
        pickle.dump(cov_summary, f)
    print(f"\nSaved coverage sweep → {cov_out}")

# ===========================================================================
# Arity sweep
# ===========================================================================
arity_results: dict = {}

if RUN_ARITY:
    _shared_budgets = _budget_series([None] * n_tasks_per_arity)
    print(f"\nShared arity budget series ({n_tasks_per_arity} tasks/arity): "
          f"{_shared_budgets}")

    for seed_i, seed_val in enumerate(cfg.SEEDS):
        seed_key = f"seed_{seed_i}"
        print(f"\n{'=' * 60}")
        print(f"DetBT arity sweep — seed {seed_i}  (seed_val={seed_val})")

        rng_order = np.random.default_rng(seed_val)
        arity_results[seed_key] = {"arity": {}}

        for arity in cfg.ARITIES:
            arity_tasks = tasks_by_arity[arity]
            if len(arity_tasks) < n_tasks_per_arity:
                print(f"  [warn] arity {arity}: only {len(arity_tasks)} tasks < "
                      f"{n_tasks_per_arity} required — skipping.")
                continue

            task_subset   = list(rng_order.choice(arity_tasks, size=n_tasks_per_arity,
                                                  replace=False))
            arity_key     = str(arity)
            budgets_arity = _budget_series(task_subset)

            print(f"\n  Arity {arity} — tasks: {task_subset}")
            print(f"  Budget series: {budgets_arity}")

            arity_entry: dict = {
                "tasks":      task_subset,
                "n_tasks":    n_tasks_per_arity,
                "min_budget": budgets_arity[0]  if budgets_arity else None,
                "max_budget": budgets_arity[-1] if budgets_arity else None,
                "budgets":    {},
            }

            rng_run = np.random.default_rng(seed_val * 1000 + arity * 10)

            for budget in budgets_arity:
                print(f"\n  --- arity={arity}  budget={budget} ---")
                train_ds_b = _uniform_sample(task_subset, budget, rng_run)
                eval_ds_b  = _eval_ds_for_tasks(task_subset)
                n_cells    = len(train_ds_b)
                n_trials   = int((train_ds_b.df["count_0"] +
                                  train_ds_b.df["count_1"]).sum())
                print(f"  cells={n_cells}  trials={n_trials}")

                if n_cells == 0:
                    print("  Empty dataset — skipping.")
                    continue

                agent  = _init_agent()
                result = _train_one(agent, train_ds_b, eval_ds_b)
                print(f"  best_epoch={result.best_epoch}  "
                      f"eval_mse={result.best_val_mse:.5f}")

                pred_mat = _det_probe_matrix(agent)
                mse      = _probe_cmse_net(pred_mat)
                print(f"  probe_cmse_net={mse:.5f}")

                ckpt_tag  = f"arity{arity}_seed{seed_i}_budget{budget}"
                ckpt_path = _save_ckpt(agent, ckpt_tag)

                arity_entry["budgets"][str(budget)] = {
                    "n_trials":        n_trials,
                    "n_cells":         n_cells,
                    "trials_per_task": _trials_per_task(task_subset, train_ds_b),
                    "probe_cmse_net":  mse,
                    "pred_matrix":     pred_mat,
                    "best_epoch":      result.best_epoch,
                    "best_val_mse":    result.best_val_mse,
                    "ckpt_path":       str(ckpt_path),
                }

                del agent, train_ds_b
                gc.collect()
                torch.cuda.empty_cache()

            arity_results[seed_key]["arity"][arity_key] = arity_entry

    arity_summary = {
        "run_tag":              cfg.RUN_TAG,
        "arities":              cfg.ARITIES,
        "n_tasks_per_arity":    n_tasks_per_arity,
        "tasks_by_arity":       tasks_by_arity,
        "trial_budgets":        cfg.TRIAL_BUDGETS,
        "seeds":                cfg.SEEDS,
        "all_tasks_ordered":    all_tasks_ordered,
        "probe_uids_ordered":   probe_uids_ordered,
        "true_matrix":          true_matrix,
        "probe_noise_floor":    probe_noise_floor,
        "random_cmse_net":      random_cmse_net,
        "random_init_cmse_net": random_init_cmse_net,
        "slda":                 slda_results,
        "dlbt":                 arity_results,
    }
    arity_out = cfg.RESULTS_DIR / f"arity_sweep_{cfg.RUN_TAG}.pkl"
    with open(arity_out, "wb") as f:
        pickle.dump(arity_summary, f)
    print(f"\nSaved arity sweep → {arity_out}")

print("\nDone.")
