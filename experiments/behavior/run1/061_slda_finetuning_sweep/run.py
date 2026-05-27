"""
run1/061_slda_finetuning_sweep/run.py — budget sweep: frozen vs. finetuned SLDA.

Protocol
--------
For each seed × budget (trials per task):
  1. Sample tpt trials per task (bootstrap if pool < tpt).  Same RNG for both
     conditions (frozen and finetuned share the same train/val split).
  2. Phase 1: fit LogisticRegressionCV (frozen CLIP features) → scalers, models.
  3. Frozen condition: evaluate Phase-1 models directly on probe matrix.
  4. Finetuned condition: run Phase 2 (attnpool fine-tuning), then re-evaluate.

All-data point: uses entire 90 % pool; fixed cell-level 10 % eval split.

Run from repo root:
    python experiments/behavior/run1/061_slda_finetuning_sweep/run.py
"""

import gc
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.training.train_slda import fit_slda_logreg, slda_probe_matrix
from dlbt.training.train_slda_attnpool import finetune_slda_attnpool

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
# Probe matrix
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
if _nf_mask.any():
    _p = true_matrix[_nf_mask]
    _n = count_matrix[_nf_mask].astype(float)
    probe_noise_floor = float(np.mean(_p * (1 - _p) / (_n - 1)))
else:
    probe_noise_floor = 0.0

_valid_rg      = ~np.isnan(true_matrix)
random_cmse_nf = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE−NF: {random_cmse_nf:.5f}")

# ---------------------------------------------------------------------------
# Spearman rank-correlation noise ceiling  (split-half + Spearman-Brown)
# ---------------------------------------------------------------------------
def _rho_noise_ceiling(cells_df: pd.DataFrame,
                       n_splits: int = 200,
                       seed: int = 0) -> float:
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
    rho_full_values = []
    for _ in range(n_splits):
        k1 = np.array([
            rng.hypergeometric(c1, t - c1, n1)
            for c1, t, n1 in zip(count1s, totals, n1s)
        ], dtype=float)
        p1    = k1 / n1s
        p2    = (count1s - k1) / n2s
        valid = (n1s > 0) & (n2s > 0)
        if valid.sum() < 2:
            continue
        rho_half, _ = spearmanr(p1[valid], p2[valid])
        if np.isnan(rho_half) or rho_half <= -1:
            continue
        rho_full_values.append((2 * rho_half) / (1 + rho_half))
    return float(np.mean(rho_full_values)) if rho_full_values else float("nan")


rho_noise_ceiling = _rho_noise_ceiling(probe_cells_df)
print(f"  Spearman ρ noise ceiling (split-half): {rho_noise_ceiling:.4f}")

# ---------------------------------------------------------------------------
# Per-task individual trial pools  (main images only)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))

task_trial_pools: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in main_cells_df.itertuples(index=False):
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
print(f"\n  Trial pool (full) — min: {min(pool_sizes.values())}  "
      f"max: {max(pool_sizes.values())}  "
      f"total: {total_pool_size:,}")

# ---------------------------------------------------------------------------
# Fixed cell-level 10 % eval split for the all-data point
# ---------------------------------------------------------------------------
rng_split    = np.random.default_rng(cfg.SEED)
n_eval_cells = max(1, int(len(main_cells_df) * 0.10))
eval_idx     = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask    = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df        = main_cells_df[eval_mask].reset_index(drop=True)
pool_df        = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds_global = BehavioralDataset(eval_df)

# 90 % pool for all-data training
task_trial_pools_all: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in pool_df.itertuples(index=False):
    tn = row.task_name
    if tn not in task_trial_pools_all:
        continue
    p = task_trial_pools_all[tn]
    for _ in range(int(row.count_0)):
        p.append((row.uid, 0))
    for _ in range(int(row.count_1)):
        p.append((row.uid, 1))

pool_all_size     = sum(len(v) for v in task_trial_pools_all.values())
avg_pool_per_task = pool_all_size / n_all_tasks
print(f"  Trial pool (all-data 90%) — total: {pool_all_size:,}  "
      f"avg/task: {avg_pool_per_task:.1f}"
      f"  eval cells: {len(eval_df)}")

trials_per_task = list(cfg.TRIALS_PER_TASK)
if cfg.FAST_PASS:
    trials_per_task = [trials_per_task[0]]
    print("  FAST_PASS=True → min tpt only (all-data always runs)")
print(f"  Trials-per-task grid ({len(trials_per_task)} points): {trials_per_task}")

# ---------------------------------------------------------------------------
# CLIP feature cache  (frozen — used for Phase 1 and as backbone for Phase 2)
# ---------------------------------------------------------------------------
_agent_tmp  = DlbtAgent(freeze_encoder=True, n_mc_samples=1, device=device)
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
# Helpers — data
# ---------------------------------------------------------------------------

def _trials_to_ds(trials: list) -> BehavioralDataset:
    if not trials:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    rows = [{"uid": uid, "task_name": tn,
             "count_0": 1 - out, "count_1": out}
            for uid, tn, out in trials]
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _sample_and_split(
    tasks: list,
    tpt: int,
    rng: np.random.Generator,
) -> tuple[BehavioralDataset, BehavioralDataset]:
    """
    Sample `tpt` trials per task (bootstrap if pool < tpt).
    Aggregate to (uid, task) cells, then split 90/10 at cell level.
    Returns (train_ds_90, val_ds_10).
    """
    all_trials = []
    for task_name in tasks:
        pool = task_trial_pools[task_name]
        if tpt == 0 or len(pool) == 0:
            continue
        replace = len(pool) < tpt
        chosen  = rng.choice(len(pool), size=tpt, replace=replace)
        for idx in chosen:
            uid, outcome = pool[int(idx)]
            all_trials.append((uid, task_name, outcome))

    if not all_trials:
        empty = BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
        return empty, empty

    rows = [{"uid": uid, "task_name": tn, "count_0": 1 - out, "count_1": out}
            for uid, tn, out in all_trials]
    cell_df = (pd.DataFrame(rows)
               .groupby(["uid", "task_name"])[["count_0", "count_1"]]
               .sum().reset_index())

    n_cells  = len(cell_df)
    n_val    = max(1, int(n_cells * 0.10))
    perm     = rng.permutation(n_cells)
    val_mask = np.zeros(n_cells, dtype=bool)
    val_mask[perm[:n_val]] = True

    return (BehavioralDataset(cell_df[~val_mask].reset_index(drop=True)),
            BehavioralDataset(cell_df[ val_mask].reset_index(drop=True)))


def _all_data_ds(tasks: list) -> BehavioralDataset:
    """Build training dataset from the fixed 90 % pool."""
    all_trials = []
    for task_name in tasks:
        for uid, outcome in task_trial_pools_all[task_name]:
            all_trials.append((uid, task_name, outcome))
    return _trials_to_ds(all_trials)


# ---------------------------------------------------------------------------
# Helpers — SLDA
# ---------------------------------------------------------------------------

def _slda_features_for_probe(clip_feats: dict) -> dict:
    return {uid: clip_feats[uid].cpu().numpy()
            for uid in probe_uids_ordered if uid in clip_feats}


def _run_slda_frozen(
    tasks: list,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
) -> tuple[np.ndarray, dict]:
    """
    Phase 1 only: LogReg on frozen CLIP features.
    Returns (probe_pred_matrix, phase1_artifacts).
    """
    scalers, models, ub = fit_slda_logreg(
        tasks, train_ds, val_ds, frozen_clip,
        Cs=cfg.SLDA_Cs, max_iter=cfg.SLDA_MAX_ITER,
    )
    probe_feats = _slda_features_for_probe(frozen_clip)
    pred = slda_probe_matrix(scalers, models, ub, probe_feats,
                             all_tasks_ordered, uid_to_row, n_probe)
    n_fitted = sum(1 for t in tasks if t in models and not ub.get(t, False))
    n_base   = len(tasks) - n_fitted
    print(f"    Phase1 model sel: fitted={n_fitted}/{len(tasks)}  base={n_base}/{len(tasks)}")
    artifacts = {"scalers": scalers, "models": models, "use_base": ub}
    return pred, artifacts


def _run_slda_finetuned(
    tasks: list,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
) -> np.ndarray:
    """
    Phase 1 + Phase 2: LogReg then attnpool fine-tuning.
    Returns probe_pred_matrix for finetuned condition.
    """
    # Phase 1 on frozen CLIP
    scalers, models, ub = fit_slda_logreg(
        tasks, train_ds, val_ds, frozen_clip,
        Cs=cfg.SLDA_Cs, max_iter=cfg.SLDA_MAX_ITER,
    )

    # Phase 2 — fine-tune attnpool
    slda_agent = SldaAgent(freeze_encoder=False, device=device)
    slda_agent.precompute_backbone_features(all_refs)
    finetune_slda_attnpool(
        slda_agent, scalers, models,
        train_ds, val_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS_PHASE2,
        patience = cfg.PATIENCE_PHASE2,
        lr       = cfg.LR_ATTNPOOL,
    )
    finetuned_feats = slda_agent.extract_features(all_refs)
    del slda_agent
    gc.collect(); torch.cuda.empty_cache()

    probe_feats = _slda_features_for_probe(finetuned_feats)
    pred = slda_probe_matrix(scalers, models, ub, probe_feats,
                             all_tasks_ordered, uid_to_row, n_probe)
    return pred


# ---------------------------------------------------------------------------
# Helpers — evaluation
# ---------------------------------------------------------------------------

def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
n_tpt   = len(trials_per_task)
n_seeds = len(cfg.SEEDS)

frozen_cmse = np.full((n_seeds, n_tpt), np.nan)
frozen_rho  = np.full((n_seeds, n_tpt), np.nan)
ft_cmse     = np.full((n_seeds, n_tpt), np.nan)
ft_rho      = np.full((n_seeds, n_tpt), np.nan)

frozen_all_cmse = np.full(n_seeds, np.nan)
frozen_all_rho  = np.full(n_seeds, np.nan)
ft_all_cmse     = np.full(n_seeds, np.nan)
ft_all_rho      = np.full(n_seeds, np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    # Both conditions share the same RNG (identical train/val split)
    rng = np.random.default_rng(seed_val)

    # ---- Budget grid -------------------------------------------------------
    for b_i, tpt in enumerate(trials_per_task):
        print(f"\n  tpt={tpt:>5,}  (total≈{tpt*n_all_tasks:,})  [{b_i+1}/{n_tpt}]")

        # Sample once — both conditions use same data
        train_ds, val_ds = _sample_and_split(all_tasks_ordered, tpt, rng)

        # Frozen SLDA
        pred_fr, _ = _run_slda_frozen(all_tasks_ordered, train_ds, val_ds)
        frozen_cmse[s_i, b_i], frozen_rho[s_i, b_i] = _probe_stats(pred_fr)
        print(f"    Frozen  cMSE−NF={frozen_cmse[s_i,b_i]:+.5f}  ρ={frozen_rho[s_i,b_i]:.3f}")
        del pred_fr

        # Finetuned SLDA
        pred_ft = _run_slda_finetuned(all_tasks_ordered, train_ds, val_ds)
        ft_cmse[s_i, b_i], ft_rho[s_i, b_i] = _probe_stats(pred_ft)
        print(f"    Finetuned cMSE−NF={ft_cmse[s_i,b_i]:+.5f}  ρ={ft_rho[s_i,b_i]:.3f}")
        del pred_ft, train_ds, val_ds

    # ---- All-data point  ---------------------------------------------------
    print(f"\n  [All data — {pool_all_size:,} train trials, avg {avg_pool_per_task:.0f}/task]")

    all_tr = _all_data_ds(all_tasks_ordered)

    # Frozen all
    pred_fr_all, _ = _run_slda_frozen(all_tasks_ordered, all_tr, eval_ds_global)
    frozen_all_cmse[s_i], frozen_all_rho[s_i] = _probe_stats(pred_fr_all)
    print(f"    Frozen all  cMSE−NF={frozen_all_cmse[s_i]:+.5f}  ρ={frozen_all_rho[s_i]:.3f}")
    del pred_fr_all

    # Finetuned all
    pred_ft_all = _run_slda_finetuned(all_tasks_ordered, all_tr, eval_ds_global)
    ft_all_cmse[s_i], ft_all_rho[s_i] = _probe_stats(pred_ft_all)
    print(f"    Finetuned all  cMSE−NF={ft_all_cmse[s_i]:+.5f}  ρ={ft_all_rho[s_i]:.3f}")
    del pred_ft_all, all_tr

    gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
    "lr_attnpool":         cfg.LR_ATTNPOOL,
    "trials_per_task":     trials_per_task,
    "avg_pool_per_task":   avg_pool_per_task,
    "total_pool_size":     total_pool_size,
    "seeds":               cfg.SEEDS,
    "all_tasks_ordered":   all_tasks_ordered,
    "probe_uids_ordered":  probe_uids_ordered,
    "true_matrix":         true_matrix,
    "count_matrix":        count_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_nf":      random_cmse_nf,
    "rho_noise_ceiling":   rho_noise_ceiling,
    # Budget sweep [n_seeds × n_tpt]
    "frozen_cmse":         frozen_cmse,
    "frozen_rho":          frozen_rho,
    "ft_cmse":             ft_cmse,
    "ft_rho":              ft_rho,
    # All-data point [n_seeds]
    "frozen_all_cmse":     frozen_all_cmse,
    "frozen_all_rho":      frozen_all_rho,
    "ft_all_cmse":         ft_all_cmse,
    "ft_all_rho":          ft_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
