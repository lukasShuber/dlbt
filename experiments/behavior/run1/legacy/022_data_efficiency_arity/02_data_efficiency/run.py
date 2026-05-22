"""
run1/02_data_efficiency/run.py — coverage sweep for DLBT and SLDA.

Protocol
--------
1.  Load + filter run0+run1 data; identify all eligible tasks.
2.  Separate probe images (held-out evaluation) from main images (training).
3.  Hold out a fixed 10 % of main cells for early-stopping (eval set).
    Expand the remaining 90 % into per-task individual trial pools.
4.  Build ground truth probe matrix from probe-image count cells.
5.  Pre-compute frozen CLIP features.

6.  SLDA sweep (always all eligible tasks as training set):
      For each valid budget B in [n_tasks, …, min_pool × n_tasks]:
        - Uniformly allocate B trials (q or q+1 per task, no replacement).
        - Fit per-task RidgeCV + temperature calibration on training images.
        - Predict probe images; record probe matrix MSE.

7.  DLBT coverage sweep:
      For each seed:
        - Draw a random task ordering over all eligible tasks.
        - For each coverage fraction [10 %, 25 %, 50 %, 75 %, 100 %]:
            * task_subset = first (frac × n_tasks) tasks in the ordering.
            * Compute valid budget series for this subset.
            * For each valid budget B:
                - Uniform allocation across task_subset.
                - Train DLBT (phase 1 + optional attnpool phase 2).
                - Predict all probe images × all tasks → probe matrix MSE.
                - Save lightweight checkpoint.

8.  Save summary dict as coverage_sweep_<RUN_TAG>.pkl.

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/run.py
"""

import gc
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit as _sigmoid
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.train_slda_attnpool import finetune_slda_attnpool

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[4]   # repo root regardless of CWD
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict    = load_image_refs(_REPO_ROOT / cfg.METADATA)
all_refs     = image_refs_as_list(refs_dict)
refs_by_uid  = {r.uid: r for r in all_refs}
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

# All eligible tasks (sorted by arity then name — consistent ordering)
all_tasks_ordered = cfg.eligible_tasks(df_filtered)
n_all_tasks       = len(all_tasks_ordered)
print(f"  Eligible tasks: {n_all_tasks}")

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
# Probe matrix ordering (rows = probe images sorted by latent state)
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

# Probe noise floor: mean binomial sampling variance over cells with n > 1
_nf_mask = count_matrix > 1
if _nf_mask.any():
    _p = true_matrix[_nf_mask]
    _n = count_matrix[_nf_mask].astype(float)
    probe_noise_floor = float(np.mean(_p * (1 - _p) / (_n - 1)))
else:
    probe_noise_floor = 0.0

# Random-guesser cMSE−NF (constant reference)
_valid_rg = ~np.isnan(true_matrix)
random_cmse_net = (float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2))
                   - probe_noise_floor)
print(f"  Probe noise floor: {probe_noise_floor:.5f}  "
      f"random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Fixed 10 % eval split of main cells (for DLBT early stopping)
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
# Per-task trial pools  task_name -> [(uid, outcome), …]
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
# CLIP feature cache (frozen throughout)
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

    min_budget = n_tasks               (q=1 for every task)
    max_budget = global_min_pool × n   (global cap so all seeds share the same
                                        endpoint at each coverage fraction;
                                        global_min_pool ≤ min_pool(subset)
                                        for any subset, so this is always
                                        achievable without replacement)

    The fixed TRIAL_BUDGETS series is intersected with [min, max];
    both endpoints are always included.
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
    Allocate `budget` trials uniformly across `tasks`.

    Every task receives q = floor(budget / n_tasks) trials.
    A randomly chosen remainder r = budget % n_tasks tasks each receive q+1.
    Sampling is without replacement from each task's individual trial pool.
    """
    n     = len(tasks)
    q     = budget // n
    r     = budget % n
    # r randomly chosen task indices receive the extra trial
    extra = set(int(i) for i in rng.choice(n, size=r, replace=False))
    rows  = []
    for i, task_name in enumerate(tasks):
        k    = q + (1 if i in extra else 0)
        pool = task_trial_pools[task_name]
        if k == 0:
            continue
        # k ≤ pool size by design (budget ≤ max_budget)
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
    # Fallback: a handful of probe cells (so training doesn't crash)
    fb = probe_cells_df[probe_cells_df["task_name"].isin(tasks)].head(20).reset_index(drop=True)
    return BehavioralDataset(fb) if len(fb) > 0 else BehavioralDataset(
        pd.DataFrame(columns=["uid", "task_name", "count_0", "count_1"]))


def _init_agent() -> DlbtAgent:
    """Fresh DlbtAgent with frozen CLIP cache and initialised mapper bias."""
    torch.manual_seed(cfg.SEEDS[0])
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
        median_correction = cfg.MEDIAN_CORRECTION,
        neutral_alpha     = cfg.NEUTRAL_ALPHA,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    if cfg.INIT_MODE == "uniform":
        bv = float(np.log(np.exp(cfg.INIT_ALPHA) - 1.0))
        with torch.no_grad():
            _linear.bias.fill_(bv)
    elif cfg.INIT_MODE == "random":
        rng_init   = np.random.default_rng(cfg.INIT_SEED)
        alpha_rnd  = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                      size=(_linear.bias.shape[0],)).astype(np.float32)
        b_init     = np.log(np.exp(alpha_rnd) - 1.0)
        with torch.no_grad():
            _linear.bias.copy_(torch.from_numpy(b_init).to(device))
    else:
        raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")
    return agent


def _train_one(agent: DlbtAgent,
               train_ds: BehavioralDataset,
               eval_ds_local: BehavioralDataset):
    """Phase 1 mapper warmup + optional phase 2 attnpool fine-tuning."""
    phase1 = train_dlbt(
        agent, train_ds, eval_ds_local, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )
    if not cfg.FREEZE_ENCODER:
        gc.collect()
        torch.cuda.empty_cache()
        for p in agent.mapper.parameters():
            p.requires_grad_(False)
        for p in agent.encoder.attnpool.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = False
        agent._cache.clear()
        opt2   = torch.optim.Adam(agent.encoder.attnpool.parameters(),
                                  lr=cfg.LR_ATTNPOOL)
        phase2 = train_dlbt(
            agent, train_ds, eval_ds_local, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            optimizer = opt2,
        )
        # Repopulate CLIP cache with fine-tuned features
        agent.eval()
        agent.precompute_backbone_features(all_refs)
        with torch.no_grad():
            for i in range(0, len(all_refs), 16):
                batch   = all_refs[i : i + 16]
                spatial = torch.stack(
                    [agent._backbone_cache[r.uid] for r in batch]
                ).to(agent.device)
                feats = agent.encoder.attnpool(spatial).float()
                for ref, feat in zip(batch, feats):
                    agent._cache[ref.uid] = feat.cpu()
        return phase2
    return phase1


@torch.no_grad()
def _dlbt_probe_matrix(agent: DlbtAgent) -> np.ndarray:
    """[n_probe × n_tasks] DLBT predicted P(yes) for all probe images × all tasks."""
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
    raw_mse = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2))
    return raw_mse - probe_noise_floor


def _save_ckpt(agent: DlbtAgent, tag: str) -> Path:
    ckpt = {"mapper": agent.mapper.state_dict()}
    if not cfg.FREEZE_ENCODER:
        ckpt["attnpool"] = agent.encoder.attnpool.state_dict()
    p = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_{tag}.pt"
    torch.save(ckpt, p)
    return p


# ---------------------------------------------------------------------------
# SLDA helpers
# ---------------------------------------------------------------------------

def _fit_slda(tasks: list,
              train_ds: BehavioralDataset) -> tuple[dict, dict, dict]:
    """Fit per-task RidgeCV + temperature calibration. Returns (scalers, models, temps)."""
    scalers, models, temps = {}, {}, {}
    for task_name in tasks:
        group = train_ds.df[train_ds.df["task_name"] == task_name]
        uids  = [uid for uid in group["uid"].tolist() if uid in frozen_clip]
        if len(uids) < 1:
            continue
        X = np.array([frozen_clip[uid].cpu().numpy() for uid in uids])
        g_sub   = group[group["uid"].isin(uids)]
        totals  = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_right = g_sub["count_1"].values / np.clip(totals, 1, None)

        # With very few samples, centering kills all signal (X - X = 0).
        # Use raw features below n=5; standard scaling above.
        scaler = StandardScaler(with_mean=(len(uids) >= 5),
                                with_std=(len(uids) >= 5))
        X_sc   = scaler.fit_transform(X)
        model  = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
        model.fit(X_sc, p_right)

        p_pred = np.clip(model.predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))

        def _nll(log_tau, logits=logits, y=p_right):
            p = _sigmoid(logits / np.exp(log_tau))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        opt = minimize_scalar(_nll, bounds=(-3.0, 3.0), method="bounded")
        scalers[task_name] = scaler
        models[task_name]  = model
        temps[task_name]   = float(np.exp(opt.x))
    return scalers, models, temps


def _slda_probe_matrix(scalers: dict, models: dict, temps: dict,
                       probe_features: dict | None = None) -> np.ndarray:
    """[n_probe × n_tasks] SLDA predicted P(yes) for all probe images × all tasks.
    probe_features: uid → np.array override; used after SLDA Stage 2."""
    pred            = np.full((n_probe, n_all_tasks), np.nan)
    feat_src        = probe_features if probe_features is not None else {
        uid: frozen_clip[uid].cpu().numpy() for uid in frozen_clip
    }
    probe_uids_clip = [uid for uid in probe_uids_ordered if uid in feat_src]
    probe_X         = np.array([feat_src[uid] for uid in probe_uids_clip])
    for j, task_name in enumerate(all_tasks_ordered):
        if task_name not in models:
            continue
        X_sc   = scalers[task_name].transform(probe_X)
        p_pred = np.clip(models[task_name].predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))
        p_cal  = _sigmoid(logits / temps[task_name])
        for i_clip, uid in enumerate(probe_uids_clip):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_cal[i_clip])
    return pred


def _init_slda_attnpool_agent() -> DlbtAgent:
    ag = DlbtAgent(freeze_encoder=False, n_mc_samples=1,
                   device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
    ag.precompute_backbone_features(all_refs)
    return ag


@torch.no_grad()
def _slda_probe_features(agent_slda: DlbtAgent) -> dict:
    agent_slda.eval()
    return {uid: agent_slda._encode([refs_by_uid[uid]])[0].cpu().numpy()
            for uid in probe_uids_ordered if uid in refs_by_uid}


def _run_slda(tasks: list, train_ds: BehavioralDataset,
              eval_ds_slda: BehavioralDataset):
    """Stage 1 always; Stage 2 attnpool fine-tuning when FREEZE_ENCODER=False."""
    scalers, models, temps = _fit_slda(tasks, train_ds)
    if not cfg.FREEZE_ENCODER_SLDA:
        agent_slda = _init_slda_attnpool_agent()
        finetune_slda_attnpool(
            agent_slda, scalers, models, temps,
            train_ds, eval_ds_slda, refs_dict,
            n_epochs = cfg.N_EPOCHS_PHASE2,
            patience = cfg.PATIENCE_PHASE2,
            lr       = cfg.LR_ATTNPOOL,
        )
        pred = _slda_probe_matrix(scalers, models, temps,
                                  probe_features=_slda_probe_features(agent_slda))
        del agent_slda
        gc.collect(); torch.cuda.empty_cache()
    else:
        pred = _slda_probe_matrix(scalers, models, temps)
    return pred, scalers, models, temps


def _trials_per_task(tasks: list, train_ds: BehavioralDataset) -> dict:
    tpt = {}
    for t in tasks:
        sub    = train_ds.df[train_ds.df["task_name"] == t]
        tpt[t] = int((sub["count_0"] + sub["count_1"]).sum()) if len(sub) > 0 else 0
    return tpt


# ===========================================================================
# Random-init DLBT baseline  (no training)
# ===========================================================================
print("\nComputing random-init DLBT baseline...")
_agent_rand          = _init_agent()
_pred_rand           = _dlbt_probe_matrix(_agent_rand)
random_init_cmse_net = _probe_cmse_net(_pred_rand)
del _agent_rand
print(f"  Random-init DLBT cMSE−NF: {random_init_cmse_net:.5f}")

# ===========================================================================
# SLDA sweep  (all eligible tasks, varying budget)
# ===========================================================================
print("\n" + "=" * 60)
print(f"SLDA sweep — all {n_all_tasks} tasks")

slda_budgets = _budget_series(all_tasks_ordered)
print(f"  Budget series: {slda_budgets}")

slda_results: dict = {
    "tasks":      all_tasks_ordered,
    "n_tasks":    n_all_tasks,
    "min_budget": slda_budgets[0]  if slda_budgets else None,
    "max_budget": slda_budgets[-1] if slda_budgets else None,
    "budgets":    {},
}

rng_slda = np.random.default_rng(cfg.SEED + 100)
for budget in slda_budgets:
    print(f"\n  SLDA  budget={budget}")
    train_ds_b                      = _uniform_sample(all_tasks_ordered, budget, rng_slda)
    eval_ds_slda_b                  = _eval_ds_for_tasks(all_tasks_ordered)
    pred_mat_b, scalers_b, mods_b, temps_b = _run_slda(
        all_tasks_ordered, train_ds_b, eval_ds_slda_b)
    mse_b              = _probe_cmse_net(pred_mat_b)
    print(f"  Fitted {len(mods_b)}/{n_all_tasks} tasks  probe_cmse_net={mse_b:.5f}")
    slda_results["budgets"][str(budget)] = {
        "n_trials":        budget,
        "trials_per_task": _trials_per_task(all_tasks_ordered, train_ds_b),
        "n_fitted":        len(mods_b),
        "probe_cmse_net":       mse_b,
        "pred_matrix":     pred_mat_b,
    }
    del train_ds_b, scalers_b, mods_b, temps_b

# ===========================================================================
# DLBT coverage sweep
# ===========================================================================
dlbt_results: dict = {}

for seed_i, seed_val in enumerate(cfg.SEEDS):
    seed_key = f"seed_{seed_i}"
    print(f"\n{'=' * 60}")
    print(f"DLBT coverage sweep — seed {seed_i}  (seed_val={seed_val})")

    rng_order     = np.random.default_rng(seed_val)
    task_ordering = list(rng_order.permutation(all_tasks_ordered))
    dlbt_results[seed_key] = {
        "task_ordering": task_ordering,
        "coverage":      {},
    }

    for frac in cfg.COVERAGE_FRACS:
        n_frac      = max(1, round(frac * n_all_tasks))
        task_subset = task_ordering[:n_frac]
        frac_key    = f"{frac:.2f}"
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

        # Separate RNG per (seed, frac) so budget-order doesn't affect sampling
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

            pred_mat = _dlbt_probe_matrix(agent)
            mse      = _probe_cmse_net(pred_mat)
            print(f"  probe_cmse_net={mse:.5f}")

            ckpt_tag  = f"cov{frac:.2f}_seed{seed_i}_budget{budget}"
            ckpt_path = _save_ckpt(agent, ckpt_tag)
            print(f"  Saved ckpt → {ckpt_path.name}")

            cov_entry["budgets"][str(budget)] = {
                "n_trials":        n_trials,
                "n_cells":         n_cells,
                "trials_per_task": _trials_per_task(task_subset, train_ds_b),
                "probe_cmse_net":       mse,
                "pred_matrix":     pred_mat,
                "best_epoch":      result.best_epoch,
                "best_val_mse":    result.best_val_mse,
                "ckpt_path":       str(ckpt_path),
            }

            del agent, train_ds_b
            gc.collect()
            torch.cuda.empty_cache()

        dlbt_results[seed_key]["coverage"][frac_key] = cov_entry

# ===========================================================================
# Save
# ===========================================================================
summary = {
    "run_tag":               cfg.RUN_TAG,
    "coverage_fracs":        cfg.COVERAGE_FRACS,
    "trial_budgets":         cfg.TRIAL_BUDGETS,
    "seeds":                 cfg.SEEDS,
    "all_tasks_ordered":     all_tasks_ordered,
    "probe_uids_ordered":    probe_uids_ordered,
    "true_matrix":           true_matrix,
    "probe_noise_floor":     probe_noise_floor,
    "random_cmse_net":       random_cmse_net,
    "random_init_cmse_net":  random_init_cmse_net,
    "slda":                  slda_results,
    "dlbt":                  dlbt_results,
}
out_path = cfg.RESULTS_DIR / f"coverage_sweep_{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
