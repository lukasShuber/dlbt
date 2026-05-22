"""
run1/06_slda_finetuning/run.py — budget sweep comparing frozen vs. attnpool SLDA.

Protocol
--------
1.  Load + filter run0+run1 data; identify all eligible tasks.
2.  Separate probe images (evaluation) from main images (training).
3.  10 % of main cells held out for Phase 2 early stopping.
4.  Build ground-truth probe matrix + noise floor.
5.  Pre-compute frozen CLIP features (loaded from shared cache).

6.  For each seed:
      Separate RNGs for frozen and attnpool conditions.
      a. Budget grid (TRIAL_BUDGETS):
           For each budget B:
             i.  Frozen SLDA:
                   Phase 1 — fit per-task ridge + τ on frozen CLIP features.
                   Evaluate on probe set.
             ii. Attnpool SLDA:
                   Phase 1 — fit per-task ridge on frozen CLIP features (no τ).
                   Phase 2 — fine-tune attnpool through fixed decoders (τ=1).
                   Phase 3 — re-optimize τ per task on fine-tuned features.
                   Evaluate on probe set.
      b. All-data point (no sampling, full pool) — same protocol.

7.  Save summary dict as results/slda_finetuning.pkl.

Run from repo root:
    python experiments/behavior/run1/06_slda_finetuning/run.py
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
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
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

_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Spearman rank-correlation noise ceiling
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
    rng     = np.random.default_rng(seed)
    rho_vals = []
    for _ in range(n_splits):
        k1 = np.array([rng.hypergeometric(c1, t - c1, n1)
                       for c1, t, n1 in zip(count1s, totals, n1s)], dtype=float)
        k2  = count1s - k1
        p1  = k1 / n1s
        p2  = k2 / n2s
        valid = (n1s > 0) & (n2s > 0)
        if valid.sum() < 2:
            continue
        rho_half, _ = spearmanr(p1[valid], p2[valid])
        if np.isnan(rho_half) or rho_half <= -1:
            continue
        rho_vals.append((2 * rho_half) / (1 + rho_half))
    return float(np.mean(rho_vals)) if rho_vals else float("nan")

rho_noise_ceiling = _rho_noise_ceiling(probe_cells_df)
print(f"  Spearman ρ noise ceiling: {rho_noise_ceiling:.4f}")

# ---------------------------------------------------------------------------
# 10 % eval split (for Phase 2 early stopping)
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
eval_ds  = BehavioralDataset(eval_df)

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
total_pool_size = sum(pool_sizes.values())
print(f"\n  Trial pool — min: {min(pool_sizes.values())}  "
      f"max: {max(pool_sizes.values())}  "
      f"total: {total_pool_size:,}")

trial_budgets = [b for b in cfg.TRIAL_BUDGETS if b <= total_pool_size]
if not trial_budgets:
    trial_budgets = cfg.TRIAL_BUDGETS[:1]
if cfg.FAST_PASS:
    trial_budgets = [trial_budgets[0]]
    print("  FAST_PASS=True → min budget only")
print(f"  Budget grid ({len(trial_budgets)} points): {trial_budgets}")

# ---------------------------------------------------------------------------
# Frozen CLIP feature cache  (loaded via SldaAgent — no DlbtAgent needed)
# ---------------------------------------------------------------------------
_agent_tmp  = SldaAgent(freeze_encoder=True, device=device)
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

def _bootstrap_sample(tasks: list, budget: int,
                       rng: np.random.Generator) -> BehavioralDataset:
    n     = len(tasks)
    q     = budget // n
    r     = budget % n
    extra = set(int(i) for i in rng.choice(n, size=r, replace=False))
    rows  = []
    for i, task_name in enumerate(tasks):
        k    = q + (1 if i in extra else 0)
        pool = task_trial_pools[task_name]
        if k == 0 or len(pool) == 0:
            continue
        replace = len(pool) < k
        chosen  = rng.choice(len(pool), size=k, replace=replace)
        for idx in chosen:
            uid, outcome = pool[int(idx)]
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _all_data_ds(tasks: list) -> BehavioralDataset:
    rows = []
    for task_name in tasks:
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


def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


# ---------------------------------------------------------------------------
# SLDA helpers
# ---------------------------------------------------------------------------

def _fit_slda_mapper(
    tasks:    list,
    train_ds: BehavioralDataset,
) -> tuple[dict, dict]:
    """
    Phase 1: fit per-task ridge decoders on frozen CLIP features.
    Returns (scalers, models) — no temperature.
    """
    scalers, models = {}, {}
    for task_name in tasks:
        group = train_ds.df[train_ds.df["task_name"] == task_name]
        uids  = [uid for uid in group["uid"].tolist() if uid in frozen_clip]
        if not uids:
            continue
        X       = np.array([frozen_clip[uid].cpu().numpy() for uid in uids])
        g_sub   = group[group["uid"].isin(uids)]
        totals  = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_right = g_sub["count_1"].values / np.clip(totals, 1, None)

        scaler = StandardScaler(with_mean=(len(uids) >= 5),
                                with_std  =(len(uids) >= 5))
        X_sc   = scaler.fit_transform(X)
        model  = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
        model.fit(X_sc, p_right)

        scalers[task_name] = scaler
        models[task_name]  = model
    return scalers, models


def _refit_temperatures(
    tasks:    list,
    train_ds: BehavioralDataset,
    scalers:  dict,
    models:   dict,
    features: dict,            # uid → np.ndarray [1024]
) -> dict:
    """
    Phase 3: optimize τ per task given fixed ridge decoders and current features.

    Can be used for both the frozen path (features = frozen_clip) and the
    attnpool path (features = fine-tuned agent features).
    """
    temps = {}
    for task_name in tasks:
        if task_name not in models or task_name not in scalers:
            continue
        group  = train_ds.df[train_ds.df["task_name"] == task_name]
        uids   = [uid for uid in group["uid"].tolist() if uid in features]
        if not uids:
            continue
        X      = np.array([features[uid] for uid in uids])
        g_sub  = group[group["uid"].isin(uids)]
        totals = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_right = g_sub["count_1"].values / np.clip(totals, 1, None)

        X_sc   = scalers[task_name].transform(X)
        p_pred = np.clip(models[task_name].predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))

        def _nll(log_tau, logits=logits, y=p_right):
            p = _sigmoid(logits / np.exp(log_tau))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        opt = minimize_scalar(_nll, bounds=(-3.0, 3.0), method="bounded")
        temps[task_name] = float(np.exp(opt.x))
    return temps


def _slda_probe_matrix(
    scalers:        dict,
    models:         dict,
    temps:          dict,
    probe_features: dict | None = None,
) -> np.ndarray:
    """
    Predict p_right for all probe images × tasks.

    probe_features: uid → np.ndarray override.  If None, uses frozen_clip.
    """
    pred      = np.full((n_probe, n_all_tasks), np.nan)
    feat_src  = probe_features if probe_features is not None else {
        uid: frozen_clip[uid].cpu().numpy() for uid in frozen_clip
    }
    p_uids    = [uid for uid in probe_uids_ordered if uid in feat_src]
    probe_X   = np.array([feat_src[uid] for uid in p_uids])

    for j, task_name in enumerate(all_tasks_ordered):
        if task_name not in models or task_name not in temps:
            continue
        X_sc   = scalers[task_name].transform(probe_X)
        p_pred = np.clip(models[task_name].predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))
        p_cal  = _sigmoid(logits / temps[task_name])
        for i_uid, uid in enumerate(p_uids):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_cal[i_uid])
    return pred


@torch.no_grad()
def _extract_agent_features(
    agent:       SldaAgent,
    uids:        list,
    batch_size:  int = 64,
) -> dict:
    """Extract current features for a list of UIDs from a SldaAgent."""
    agent.eval()
    result = {}
    present = [uid for uid in uids if uid in refs_by_uid]
    for i in range(0, len(present), batch_size):
        batch_uids = present[i : i + batch_size]
        batch_refs = [refs_by_uid[uid] for uid in batch_uids]
        feats = agent._encode(batch_refs)   # [B, 1024]
        for uid, feat in zip(batch_uids, feats):
            result[uid] = feat.cpu().numpy()
    return result


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def _run_frozen(tasks: list, train_ds: BehavioralDataset):
    """
    Frozen SLDA:
      Phase 1 — fit ridge mapper on frozen CLIP.
      Phase 3 — optimize τ on frozen CLIP features.
    """
    scalers, models = _fit_slda_mapper(tasks, train_ds)
    frozen_feats    = {uid: frozen_clip[uid].cpu().numpy() for uid in frozen_clip}
    temps           = _refit_temperatures(tasks, train_ds, scalers, models, frozen_feats)
    pred            = _slda_probe_matrix(scalers, models, temps)
    return pred


def _run_attnpool(tasks: list, train_ds: BehavioralDataset):
    """
    Attnpool SLDA:
      Phase 1 — fit ridge mapper on frozen CLIP (no τ).
      Phase 2 — fine-tune attnpool through fixed decoders.
      Phase 3 — optimize τ on fine-tuned features.
    """
    # Phase 1
    scalers, models = _fit_slda_mapper(tasks, train_ds)

    # Phase 2
    agent = SldaAgent(freeze_encoder=False, device=device)
    agent.precompute_backbone_features(all_refs)
    finetune_slda_attnpool(
        agent, scalers, models,
        train_ds, eval_ds, refs_dict,
        n_epochs   = cfg.N_EPOCHS_ATTNPOOL,
        patience   = cfg.PATIENCE_ATTNPOOL,
        lr         = cfg.LR_ATTNPOOL,
        batch_size = cfg.BATCH_SIZE_ATTNPOOL,
    )

    # Phase 3
    train_uids   = train_ds.df["uid"].unique().tolist()
    train_feats  = _extract_agent_features(agent, train_uids)
    temps        = _refit_temperatures(tasks, train_ds, scalers, models, train_feats)

    # Predict on probe set
    probe_feats  = _extract_agent_features(agent, probe_uids_ordered)
    pred         = _slda_probe_matrix(scalers, models, temps,
                                      probe_features=probe_feats)
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return pred


# ---------------------------------------------------------------------------
# Result arrays
# ---------------------------------------------------------------------------
n_budgets = len(trial_budgets)
n_seeds   = len(cfg.SEEDS)

frozen_cmse   = np.full((n_seeds, n_budgets), np.nan)
frozen_rho    = np.full((n_seeds, n_budgets), np.nan)
attnpool_cmse = np.full((n_seeds, n_budgets), np.nan)
attnpool_rho  = np.full((n_seeds, n_budgets), np.nan)

frozen_all_cmse   = np.full(n_seeds, np.nan)
frozen_all_rho    = np.full(n_seeds, np.nan)
attnpool_all_cmse = np.full(n_seeds, np.nan)
attnpool_all_rho  = np.full(n_seeds, np.nan)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    # Independent RNGs so the two conditions draw different trial samples
    rng_frozen   = np.random.default_rng(seed_val)
    rng_attnpool = np.random.default_rng(seed_val + 100_000)

    # ---- Budget grid -------------------------------------------------------
    for b_i, budget in enumerate(trial_budgets):
        print(f"\n  Budget {budget:>7,}  [{b_i+1}/{n_budgets}]")

        # Frozen SLDA
        train_ds_f = _bootstrap_sample(all_tasks_ordered, budget, rng_frozen)
        pred_f     = _run_frozen(all_tasks_ordered, train_ds_f)
        frozen_cmse[s_i, b_i], frozen_rho[s_i, b_i] = _probe_stats(pred_f)
        print(f"    Frozen   cMSE−NF={frozen_cmse[s_i,b_i]:+.5f}  "
              f"ρ={frozen_rho[s_i,b_i]:.3f}")
        del train_ds_f, pred_f

        # Attnpool SLDA
        train_ds_a = _bootstrap_sample(all_tasks_ordered, budget, rng_attnpool)
        pred_a     = _run_attnpool(all_tasks_ordered, train_ds_a)
        attnpool_cmse[s_i, b_i], attnpool_rho[s_i, b_i] = _probe_stats(pred_a)
        print(f"    Attnpool cMSE−NF={attnpool_cmse[s_i,b_i]:+.5f}  "
              f"ρ={attnpool_rho[s_i,b_i]:.3f}")
        del train_ds_a, pred_a

    # ---- All-data point ----------------------------------------------------
    print(f"\n  [All data — {total_pool_size:,} trials]")
    all_ds = _all_data_ds(all_tasks_ordered)

    pred_fa = _run_frozen(all_tasks_ordered, all_ds)
    frozen_all_cmse[s_i], frozen_all_rho[s_i] = _probe_stats(pred_fa)
    print(f"    Frozen all   cMSE−NF={frozen_all_cmse[s_i]:+.5f}  "
          f"ρ={frozen_all_rho[s_i]:.3f}")
    del pred_fa

    pred_aa = _run_attnpool(all_tasks_ordered, all_ds)
    attnpool_all_cmse[s_i], attnpool_all_rho[s_i] = _probe_stats(pred_aa)
    print(f"    Attnpool all cMSE−NF={attnpool_all_cmse[s_i]:+.5f}  "
          f"ρ={attnpool_all_rho[s_i]:.3f}")
    del pred_aa, all_ds

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
    "trial_budgets":       trial_budgets,
    "total_pool_size":     total_pool_size,
    "seeds":               cfg.SEEDS,
    "all_tasks_ordered":   all_tasks_ordered,
    "probe_uids_ordered":  probe_uids_ordered,
    "true_matrix":         true_matrix,
    "count_matrix":        count_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_net":     random_cmse_net,
    "rho_noise_ceiling":   rho_noise_ceiling,
    # Budget sweep [n_seeds × n_budgets]
    "frozen_cmse":         frozen_cmse,
    "frozen_rho":          frozen_rho,
    "attnpool_cmse":       attnpool_cmse,
    "attnpool_rho":        attnpool_rho,
    # All-data [n_seeds]
    "frozen_all_cmse":     frozen_all_cmse,
    "frozen_all_rho":      frozen_all_rho,
    "attnpool_all_cmse":   attnpool_all_cmse,
    "attnpool_all_rho":    attnpool_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
