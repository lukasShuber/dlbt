"""
run1/07_task_generalization/run.py — task generalization experiment.

For each training condition (arity category) and seed:
  1. Sample k tasks from the arity pool  (k = number of eligible 1-arity tasks)
  2. Train DLBT on all available trials from those k tasks
  3. Evaluate on probe images × HELD-OUT tasks (not in training)
  4. Record cMSE-NF and Spearman ρ on held-out tasks only

Reference lines (fresh, N_SEEDS runs):
  Full DLBT — all tasks, all data → probe matrix on ALL tasks
  Full SLDA — all tasks, all data → probe matrix on ALL tasks

Run from repo root:
    python experiments/behavior/run1/07_task_generalization/run.py
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
from scipy.stats import spearmanr

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.train_slda import fit_slda_logreg, slda_probe_matrix

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[4]
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

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

# Arity groups
arity_groups = cfg.tasks_by_arity(all_tasks_ordered)
for a in sorted(arity_groups):
    print(f"  Arity {a}: {len(arity_groups[a])} tasks")

_beh_id_eligible = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in set(all_tasks_ordered)}
full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _beh_id_eligible,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)

# Budget parameter: k = number of 1-arity tasks
k_tasks = len(arity_groups.get(1, []))
print(f"\n  k_tasks (budget) = {k_tasks}  (number of eligible 1-arity tasks)")

# ---------------------------------------------------------------------------
# Probe matrix (ground truth)
# ---------------------------------------------------------------------------
probe_refs_ordered = sorted(
    [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
probe_uids_ordered = [r.uid for r in probe_refs_ordered]
n_probe            = len(probe_uids_ordered)
uid_to_row         = {uid: i for i, uid in enumerate(probe_uids_ordered)}
task_to_col        = {t: j for j, t in enumerate(all_tasks_ordered)}

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

_nf_mask = count_matrix > 1
probe_noise_floor = (
    float(np.mean(true_matrix[_nf_mask] * (1 - true_matrix[_nf_mask])
                  / (count_matrix[_nf_mask].astype(float) - 1)))
    if _nf_mask.any() else 0.0
)
_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Spearman ρ noise ceiling  (split-half + Spearman-Brown)
# ---------------------------------------------------------------------------
def _rho_noise_ceiling(cells_df: pd.DataFrame,
                       n_splits: int = 200, seed: int = 0) -> float:
    df = cells_df.copy()
    df["total"] = df["count_0"] + df["count_1"]
    df = df[df["total"] >= 2].reset_index(drop=True)
    if len(df) < 2:
        return float("nan")
    totals  = df["total"].values.astype(int)
    count1s = df["count_1"].values.astype(int)
    n1s     = totals // 2
    n2s     = totals - n1s
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        k1 = np.array([rng.hypergeometric(c1, t - c1, n1)
                       for c1, t, n1 in zip(count1s, totals, n1s)], dtype=float)
        p1 = k1 / n1s
        p2 = (count1s - k1) / n2s
        rh, _ = spearmanr(p1, p2)
        if not np.isnan(rh) and rh > -1:
            vals.append((2 * rh) / (1 + rh))
    return float(np.mean(vals)) if vals else float("nan")

rho_noise_ceiling = _rho_noise_ceiling(probe_cells_df)
print(f"  Spearman ρ noise ceiling: {rho_noise_ceiling:.4f}")

# ---------------------------------------------------------------------------
# 10 % eval split of main cells  (for DLBT early stopping + SLDA val)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split    = np.random.default_rng(cfg.SEED)
n_eval_cells = max(1, int(len(main_cells_df) * 0.10))
eval_idx     = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask    = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df   = main_cells_df[eval_mask].reset_index(drop=True)
pool_df   = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds   = BehavioralDataset(eval_df)
eval_ds_global = eval_ds   # alias used for reference line training
print(f"\n  Eval cells (early stopping): {len(eval_df)}")
print(f"  Train pool cells (90 %%):    {len(pool_df)}")

# ---------------------------------------------------------------------------
# Per-task trial pools  (from the 90 % main pool)
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
total_pool_size = sum(pool_sizes.values())
print(f"\n  Trial pool — total: {total_pool_size:,}  "
      f"min/task: {min(pool_sizes.values())}  max/task: {max(pool_sizes.values())}")

# ---------------------------------------------------------------------------
# CLIP feature cache
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

# Probe features for SLDA (numpy format)
probe_features_np = {uid: frozen_clip[uid].cpu().numpy()
                     for uid in probe_uids_ordered if uid in frozen_clip}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tasks_to_ds(task_subset: list[str]) -> BehavioralDataset:
    """Build a BehavioralDataset from all pooled trials for the given tasks."""
    rows = []
    for task_name in task_subset:
        for uid, outcome in task_trial_pools[task_name]:
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _set_mapper_bias(agent, seed: int):
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    rng_init  = np.random.default_rng(seed)
    alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                 size=(_linear.bias.shape[0],)).astype(np.float32)
    with torch.no_grad():
        _linear.bias.copy_(
            torch.from_numpy(np.log(np.exp(alpha_rnd) - 1.0)).to(device))


def _init_dlbt(seed: int) -> DlbtAgent:
    torch.manual_seed(seed)
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _set_mapper_bias(agent, seed)
    return agent


def _train_dlbt(agent: DlbtAgent, train_ds: BehavioralDataset,
                val_ds: BehavioralDataset) -> None:
    train_dlbt(
        agent, train_ds, val_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )


@torch.no_grad()
def _probe_matrix_full(agent) -> np.ndarray:
    """Probe matrix for ALL tasks (for reference line evaluation)."""
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    for j, task_name in enumerate(all_tasks_ordered):
        task  = get_task(task_name)
        probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
        pred[:, j] = probs
    return pred


@torch.no_grad()
def _probe_matrix_subset(agent, held_out_tasks: list[str]) -> np.ndarray:
    """Probe matrix restricted to held-out tasks (for generalization eval)."""
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    for task_name in held_out_tasks:
        j = task_to_col.get(task_name)
        if j is None:
            continue
        task  = get_task(task_name)
        probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
        pred[:, j] = probs
    return pred


def _probe_stats_full(pred_mat: np.ndarray) -> tuple[float, float]:
    """cMSE-NF and ρ on all tasks with valid data."""
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


def _probe_stats_subset(pred_mat: np.ndarray,
                        held_out_tasks: list[str]) -> tuple[float, float]:
    """cMSE-NF and ρ restricted to held-out task columns."""
    cols    = [task_to_col[t] for t in held_out_tasks if t in task_to_col]
    if not cols:
        return float("nan"), float("nan")
    sub_pred = pred_mat[:, cols]
    sub_true = true_matrix[:, cols]
    valid    = ~np.isnan(sub_pred) & ~np.isnan(sub_true)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    cmse_nf = float(np.mean((sub_pred[valid] - sub_true[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(sub_pred[valid], sub_true[valid])
    return cmse_nf, float(rho)


def _run_slda_full(all_tasks: list[str],
                   train_ds: BehavioralDataset,
                   val_ds:   BehavioralDataset) -> np.ndarray:
    """Fit SLDA on all tasks, return full probe matrix."""
    scalers, models, use_base = fit_slda_logreg(
        all_tasks, train_ds, val_ds,
        clip_features = frozen_clip,
        Cs            = cfg.SLDA_Cs,
        max_iter      = cfg.SLDA_MAX_ITER,
    )
    return slda_probe_matrix(
        scalers, models, use_base,
        probe_features = probe_features_np,
        tasks_ordered  = all_tasks,
        uid_to_row     = uid_to_row,
        n_probe        = n_probe,
    )


# ===========================================================================
# Reference lines: full DLBT + full SLDA on all tasks, all data
# ===========================================================================
print("\n" + "="*60)
print("Reference lines: full DLBT + SLDA (all tasks, all data)")

all_ds_full = _tasks_to_ds(all_tasks_ordered)

ref_dlbt_cmse = np.full(len(cfg.SEEDS), np.nan)
ref_dlbt_rho  = np.full(len(cfg.SEEDS), np.nan)
ref_slda_cmse = np.full(len(cfg.SEEDS), np.nan)
ref_slda_rho  = np.full(len(cfg.SEEDS), np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n  Ref seed {s_i+1}/{len(cfg.SEEDS)}  (seed_val={seed_val})")

    # Full DLBT
    agent = _init_dlbt(seed_val)
    _train_dlbt(agent, all_ds_full, eval_ds_global)
    pred  = _probe_matrix_full(agent)
    ref_dlbt_cmse[s_i], ref_dlbt_rho[s_i] = _probe_stats_full(pred)
    print(f"    Full DLBT  cMSE−NF={ref_dlbt_cmse[s_i]:+.5f}  ρ={ref_dlbt_rho[s_i]:.4f}")
    del agent, pred
    gc.collect(); torch.cuda.empty_cache()

    # Full SLDA (uses global eval_ds for model selection)
    pred_s = _run_slda_full(all_tasks_ordered, all_ds_full, eval_ds_global)
    ref_slda_cmse[s_i], ref_slda_rho[s_i] = _probe_stats_full(pred_s)
    print(f"    Full SLDA  cMSE−NF={ref_slda_cmse[s_i]:+.5f}  ρ={ref_slda_rho[s_i]:.4f}")
    del pred_s

# ===========================================================================
# Main experiment: task generalization sweep
# ===========================================================================
# Results: dict condition → [n_seeds] arrays (no budget grid — one value per seed)
n_seeds    = len(cfg.SEEDS)
conditions = cfg.ARITY_CONDITIONS   # ["1-arity", "2-arity", "3-arity", "4-arity", "random"]

gen_cmse = {c: np.full(n_seeds, np.nan) for c in conditions}
gen_rho  = {c: np.full(n_seeds, np.nan) for c in conditions}
# Record which tasks were used per seed per condition
gen_train_tasks = {c: [] for c in conditions}

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    rng = np.random.default_rng(seed_val + 300_000)  # isolated RNG stream

    for cond in conditions:
        # ---- Sample k training tasks from the arity pool -------------------
        if cond == "random":
            pool = all_tasks_ordered
        else:
            arity = int(cond[0])
            pool  = arity_groups.get(arity, [])

        if not pool:
            print(f"  [{cond}] no eligible tasks — skipping")
            continue

        if len(pool) <= k_tasks:
            # Fewer tasks than k: use all of them (no sampling, no replacement)
            train_tasks = list(pool)
        else:
            # Sample k tasks without replacement from the pool
            idx         = rng.choice(len(pool), size=k_tasks, replace=False)
            train_tasks = [pool[i] for i in sorted(idx)]

        held_out = [t for t in all_tasks_ordered if t not in set(train_tasks)]
        gen_train_tasks[cond].append(train_tasks)

        print(f"\n  [{cond}] training on {len(train_tasks)} tasks, "
              f"held-out: {len(held_out)} tasks")

        if not held_out:
            print(f"    No held-out tasks — skipping evaluation")
            continue

        # ---- Build training dataset (all trials for selected tasks) --------
        train_ds = _tasks_to_ds(train_tasks)
        # Use eval_ds_global (10% global split) for DLBT early stopping
        # Note: eval_ds may contain cells from training tasks only — this is
        # intentional; early stopping monitors held-out cells from training tasks.

        # ---- Train DLBT on selected tasks ----------------------------------
        agent = _init_dlbt(seed_val)
        _train_dlbt(agent, train_ds, eval_ds_global)

        # ---- Evaluate on held-out tasks ------------------------------------
        pred = _probe_matrix_subset(agent, held_out)
        gen_cmse[cond][s_i], gen_rho[cond][s_i] = _probe_stats_subset(pred, held_out)
        print(f"    cMSE−NF={gen_cmse[cond][s_i]:+.5f}  ρ={gen_rho[cond][s_i]:.4f}")

        del agent, train_ds, pred
        gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":              cfg.RUN_TAG,
    "seeds":                cfg.SEEDS,
    "k_tasks":              k_tasks,
    "all_tasks_ordered":    all_tasks_ordered,
    "arity_groups":         {k: v for k, v in arity_groups.items()},
    "conditions":           conditions,
    "probe_uids_ordered":   probe_uids_ordered,
    "true_matrix":          true_matrix,
    "count_matrix":         count_matrix,
    "probe_noise_floor":    probe_noise_floor,
    "random_cmse_net":      random_cmse_net,
    "rho_noise_ceiling":    rho_noise_ceiling,
    # Generalization results [n_seeds] per condition
    "gen_cmse":             gen_cmse,
    "gen_rho":              gen_rho,
    "gen_train_tasks":      gen_train_tasks,
    # Reference lines [n_seeds]
    "ref_dlbt_cmse":        ref_dlbt_cmse,
    "ref_dlbt_rho":         ref_dlbt_rho,
    "ref_slda_cmse":        ref_slda_cmse,
    "ref_slda_rho":         ref_slda_rho,
    "total_pool_size":      total_pool_size,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
