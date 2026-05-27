"""
run1/021_efficiency_main/run.py — trials-per-task budget sweep.

Protocol
--------
1.  Load + filter run0+run1 data; identify eligible tasks.
2.  Separate probe images (held-out evaluation) from main images (training pool).
3.  Build ground-truth probe matrix and noise floor.
4.  Pre-compute frozen CLIP features.

5.  For each seed × budget (trials per task):
      a. Sample tpt trials per task (bootstrap if pool < tpt).
      b. Split sampled trials 90/10 → train_ds / val_ds.
      c. DLBT:
           - Train mapper on train_ds, early-stop on val_ds.
           - Compare trained model vs. base model (α=1000) on val_ds by MSE.
           - Use whichever is better to predict probe matrix.
      d. SLDA (per task):
           - Fit L2 logistic regression on train_ds cells for this task.
           - Compare fitted vs. base model (P=0.5) on val_ds cells.
           - Use whichever is better for probe predictions.
      e. Anti-human DLBT: same as DLBT but labels are flipped.
      f. All-data point: uses entire pool (no budget cap), same 90/10 logic.

6.  Save summary dict as results/efficiency_main_021.pkl.

Run from repo root:
    python experiments/behavior/run1/021_efficiency_main/run.py
"""

import gc
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)    # sklearn convergence
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.train_slda import fit_slda_logreg, slda_probe_matrix
from dlbt.training.train_slda_attnpool import finetune_slda_attnpool

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).parents[4]
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
models_dir  = cfg.RESULTS_DIR / "models" / cfg.RUN_TAG
models_dir.mkdir(parents=True, exist_ok=True)

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
# (mirrors 02_efficiency_main: same seed, same cell-level strategy)
# ---------------------------------------------------------------------------
rng_split    = np.random.default_rng(cfg.SEED)
n_eval_cells = max(1, int(len(main_cells_df) * 0.10))
eval_idx     = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask    = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df   = main_cells_df[eval_mask].reset_index(drop=True)
pool_df   = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds_global = BehavioralDataset(eval_df)

eval_df_anti = eval_df.copy()
eval_df_anti["count_0"], eval_df_anti["count_1"] = (
    eval_df["count_1"].copy(), eval_df["count_0"].copy()
)
eval_ds_anti_global = BehavioralDataset(eval_df_anti)

# 90 % pool used for all-data training (task_trial_pools used for budget grid)
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
avg_pool_per_task = pool_all_size / n_all_tasks   # x-position for all-data marker
print(f"  Trial pool (all-data 90%) — total: {pool_all_size:,}  "
      f"avg/task: {avg_pool_per_task:.1f}"
      f"  eval cells: {len(eval_df)}")

trials_per_task = list(cfg.TRIALS_PER_TASK)
if cfg.FAST_PASS:
    trials_per_task = [trials_per_task[0]]
    print("  FAST_PASS=True → min tpt only (all-data always runs)")
print(f"  Trials-per-task grid ({len(trials_per_task)} points): {trials_per_task}")

# ---------------------------------------------------------------------------
# CLIP feature cache
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

# Pre-build probe feature matrix for SLDA (ordering matches probe_uids_ordered)

# ---------------------------------------------------------------------------
# DLBT base agent  (symmetric Dirichlet α = BASE_CONCENTRATION)
# ---------------------------------------------------------------------------
# Under normalised utility the SEU logit is exactly 0 → P(right) = 0.5.
base_agent = DlbtAgent(
    freeze_encoder    = True,
    n_mc_samples      = cfg.N_MC,
    device            = device,
    normalize_utility = cfg.NORMALIZED_UTILITY,
)
base_agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
with torch.no_grad():
    _lin = base_agent.mapper[0]   # Linear(1024, K)
    _lin.weight.zero_()
    _lin.bias.fill_(cfg.BASE_CONCENTRATION)  # softplus(1000) ≈ 1000 = α_k
base_agent.eval()
print(f"Base agent ready (α = {cfg.BASE_CONCENTRATION}).")

# ---------------------------------------------------------------------------
# Helpers — data
# ---------------------------------------------------------------------------

def _trials_to_ds(trials: list) -> BehavioralDataset:
    """Aggregate (uid, task_name, outcome) tuples into a BehavioralDataset."""
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
    flip: bool = False,
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
            if flip:
                outcome = 1 - outcome
            all_trials.append((uid, task_name, outcome))

    if not all_trials:
        empty = BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
        return empty, empty

    # Aggregate trials → cells, then split at cell level
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


def _all_data_ds(tasks: list, flip: bool = False) -> BehavioralDataset:
    """
    Build a training dataset from the fixed 90 % pool (task_trial_pools_all).
    Used for the all-data point; the matching eval split is eval_ds_global /
    eval_ds_anti_global (cell-level 10 %, fixed at startup — same strategy as 02).
    """
    all_trials = []
    for task_name in tasks:
        for uid, outcome in task_trial_pools_all[task_name]:
            if flip:
                outcome = 1 - outcome
            all_trials.append((uid, task_name, outcome))
    return _trials_to_ds(all_trials)


# ---------------------------------------------------------------------------
# Helpers — DLBT
# ---------------------------------------------------------------------------

def _init_agent(seed: int) -> DlbtAgent:
    """Fresh DlbtAgent with CLIP cache and random mapper bias init."""
    torch.manual_seed(seed)
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    rng_init  = np.random.default_rng(seed)
    alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                 size=(agent.mapper[0].bias.shape[0],)).astype(np.float32)
    b_init    = np.log(np.exp(alpha_rnd) - 1.0)
    with torch.no_grad():
        agent.mapper[0].bias.copy_(torch.from_numpy(b_init).to(device))
    return agent


def _base_mse_on_ds(val_ds: BehavioralDataset) -> float:
    """MSE of base model (P=0.5 everywhere) on val_ds — analytical."""
    if val_ds.df.empty:
        return float("nan")
    totals = (val_ds.df["count_0"] + val_ds.df["count_1"]).values.astype(float)
    p_obs  = val_ds.df["count_1"].values / np.clip(totals, 1, None)
    return float(np.mean((0.5 - p_obs) ** 2))


@torch.no_grad()
def _ds_mse_dlbt(agent: DlbtAgent, val_ds: BehavioralDataset) -> float:
    """MSE of agent predictions on val_ds."""
    if val_ds.df.empty:
        return float("nan")
    agent.eval()
    pred_list, true_list = [], []
    for task_name, group in val_ds.df.groupby("task_name"):
        task = get_task(task_name)
        uids = [uid for uid in group["uid"].tolist() if uid in refs_by_uid]
        if not uids:
            continue
        refs  = [refs_by_uid[uid] for uid in uids]
        probs = agent.choice_probs(refs, task)[:, 1].cpu().numpy()
        g_sub = group[group["uid"].isin(uids)]
        tot   = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_obs = g_sub["count_1"].values / np.clip(tot, 1, None)
        pred_list.extend(probs.tolist())
        true_list.extend(p_obs.tolist())
    if not pred_list:
        return float("nan")
    return float(np.mean((np.array(pred_list) - np.array(true_list)) ** 2))


def _dlbt_ckpt(agent: DlbtAgent, used_base: bool) -> dict:
    """Snapshot trained DLBT weights (mapper + attnpool if fine-tuned)."""
    ckpt = {
        "used_base":          used_base,
        "mapper_state_dict":  {k: v.cpu() for k, v in agent.mapper.state_dict().items()},
    }
    if not cfg.FREEZE_ENCODER_DLBT:
        ckpt["attnpool_state_dict"] = {
            k: v.cpu() for k, v in agent.encoder.attnpool.state_dict().items()
        }
    return ckpt


def _save_ckpt(label: str, dlbt: dict, slda: dict) -> None:
    """
    Write per-(seed, budget) model checkpoints.

    DLBT  → <label>_dlbt.pt   (torch.save — state dicts only)
    SLDA  → <label>_slda.pkl  (pickle    — sklearn objects + optional attnpool sd)
    """
    dlbt_path = models_dir / f"{label}_dlbt.pt"
    slda_path = models_dir / f"{label}_slda.pkl"
    torch.save(dlbt, dlbt_path)
    with open(slda_path, "wb") as fh:
        pickle.dump(slda, fh)
    print(f"    Ckpt → {dlbt_path.name}  {slda_path.name}")


def _run_dlbt(
    agent: DlbtAgent,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
) -> DlbtAgent:
    """
    Train DLBT and return whichever of (trained, base) wins on val_ds.

    Phase 1: train mapper with frozen CLIP encoder.
    Phase 2: fine-tune attnpool jointly  [if FREEZE_ENCODER_DLBT=False].
    """
    # Phase 1 — mapper, frozen encoder
    train_dlbt(
        agent, train_ds, val_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )

    if not cfg.FREEZE_ENCODER_DLBT:
        # Phase 2 — unfreeze attnpool, freeze mapper
        gc.collect(); torch.cuda.empty_cache()
        for p in agent.mapper.parameters():
            p.requires_grad_(False)
        for p in agent.encoder.attnpool.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = False
        agent._cache.clear()
        opt2 = torch.optim.Adam(
            agent.encoder.attnpool.parameters(), lr=cfg.LR_ATTNPOOL_DLBT)
        train_dlbt(
            agent, train_ds, val_ds, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            optimizer = opt2,
        )
        # Repopulate CLIP cache with fine-tuned attnpool features
        agent.eval()
        agent.precompute_backbone_features(all_refs)
        with torch.no_grad():
            for i in range(0, len(all_refs), 16):
                batch   = all_refs[i : i + 16]
                spatial = torch.stack(
                    [agent._backbone_cache[r.uid] for r in batch]
                ).to(device)
                feats = agent.encoder.attnpool(spatial).float()
                for ref, feat in zip(batch, feats):
                    agent._cache[ref.uid] = feat.cpu()
        # Re-enable mapper gradients for any future use
        for p in agent.mapper.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = True

    if val_ds.df.empty:
        return agent
    trained_mse = _ds_mse_dlbt(agent, val_ds)
    base_mse    = _base_mse_on_ds(val_ds)
    if not np.isnan(base_mse) and base_mse < trained_mse:
        return base_agent
    return agent


def _dlbt_probe_matrix(agent: DlbtAgent) -> np.ndarray:
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    with torch.no_grad():
        for j, task_name in enumerate(all_tasks_ordered):
            task  = get_task(task_name)
            probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
            pred[:, j] = probs
    return pred


# ---------------------------------------------------------------------------
# Helpers — SLDA
# ---------------------------------------------------------------------------

def _slda_features_for_probe(clip_feats: dict) -> dict:
    """Extract probe image features as np.ndarray dict for slda_probe_matrix."""
    return {uid: clip_feats[uid].cpu().numpy()
            for uid in probe_uids_ordered if uid in clip_feats}


def _run_slda(
    tasks: list,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
    clip_feats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full SLDA pipeline → (probe prediction matrix, artifacts dict).

    Phase 1: fit LogReg per task on frozen (or provided) CLIP features.
    Phase 2: fine-tune attnpool through fixed Phase-1 decoders  [if FREEZE_ENCODER_SLDA=False].

    clip_feats defaults to frozen_clip.

    artifacts dict keys:
        scalers, models, use_base       — Phase-1 sklearn objects
        attnpool_state_dict             — attnpool weights after Phase 2 (None if frozen)
    """
    if clip_feats is None:
        clip_feats = frozen_clip

    # Phase 1
    scalers, models, ub = fit_slda_logreg(
        tasks, train_ds, val_ds, clip_feats,
        Cs=cfg.SLDA_Cs, max_iter=cfg.SLDA_MAX_ITER,
    )

    attnpool_sd = None
    if not cfg.FREEZE_ENCODER_SLDA:
        # Phase 2 — fine-tune attnpool through fixed Phase-1 decoders
        slda_agent = SldaAgent(freeze_encoder=False, device=device)
        slda_agent.precompute_backbone_features(all_refs)
        finetune_slda_attnpool(
            slda_agent, scalers, models,
            train_ds, val_ds, refs_dict,
            n_epochs   = cfg.N_EPOCHS_PHASE2,
            patience   = cfg.PATIENCE_PHASE2,
            lr         = cfg.LR_ATTNPOOL_SLDA,
        )
        attnpool_sd = {k: v.cpu()
                       for k, v in slda_agent.encoder.attnpool.state_dict().items()}
        clip_feats = slda_agent.extract_features(all_refs)
        del slda_agent
        gc.collect(); torch.cuda.empty_cache()

    probe_feats = _slda_features_for_probe(clip_feats)
    pred = slda_probe_matrix(scalers, models, ub, probe_feats,
                             all_tasks_ordered, uid_to_row, n_probe)

    # Model-selection summary: how many tasks chose fitted vs base
    n_fitted = sum(1 for t in tasks if t in models and not ub.get(t, False))
    n_base   = sum(1 for t in tasks if t not in models or ub.get(t, False))
    print(f"    SLDA model sel: fitted={n_fitted}/{len(tasks)}  base={n_base}/{len(tasks)}")

    artifacts = {
        "scalers":             scalers,
        "models":              models,
        "use_base":            ub,
        "attnpool_state_dict": attnpool_sd,
    }
    return pred, artifacts


# ---------------------------------------------------------------------------
# Helpers — evaluation
# ---------------------------------------------------------------------------

def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    """Return (cMSE−NF, Spearman ρ) for a predicted probe matrix."""
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
n_tpt   = len(trials_per_task)
n_seeds = len(cfg.SEEDS)

dlbt_cmse = np.full((n_seeds, n_tpt), np.nan)
dlbt_rho  = np.full((n_seeds, n_tpt), np.nan)
slda_cmse = np.full((n_seeds, n_tpt), np.nan)
slda_rho  = np.full((n_seeds, n_tpt), np.nan)
anti_cmse = np.full((n_seeds, n_tpt), np.nan)
anti_rho  = np.full((n_seeds, n_tpt), np.nan)

dlbt_all_cmse = np.full(n_seeds, np.nan)
dlbt_all_rho  = np.full(n_seeds, np.nan)
slda_all_cmse = np.full(n_seeds, np.nan)
slda_all_rho  = np.full(n_seeds, np.nan)
anti_all_cmse = np.full(n_seeds, np.nan)
anti_all_rho  = np.full(n_seeds, np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    rng_dlbt = np.random.default_rng(seed_val)
    rng_slda = np.random.default_rng(seed_val + 100_000)
    rng_anti = np.random.default_rng(seed_val + 200_000)

    # ---- Budget grid -------------------------------------------------------
    for b_i, tpt in enumerate(trials_per_task):
        print(f"\n  tpt={tpt:>5,}  (total≈{tpt*n_all_tasks:,})  [{b_i+1}/{n_tpt}]")

        # DLBT
        train_ds, val_ds = _sample_and_split(all_tasks_ordered, tpt, rng_dlbt)
        agent  = _init_agent(seed_val)
        chosen = _run_dlbt(agent, train_ds, val_ds)
        pred   = _dlbt_probe_matrix(chosen)
        dlbt_cmse[s_i, b_i], dlbt_rho[s_i, b_i] = _probe_stats(pred)
        used_base_dlbt = (chosen is base_agent)
        print(f"    DLBT   cMSE−NF={dlbt_cmse[s_i,b_i]:+.5f}  ρ={dlbt_rho[s_i,b_i]:.3f}"
              f"  (base={'yes' if used_base_dlbt else 'no'})")
        dlbt_ck = _dlbt_ckpt(agent, used_base_dlbt)
        del chosen, train_ds, val_ds, pred, agent
        gc.collect(); torch.cuda.empty_cache()

        # SLDA
        train_ds_s, val_ds_s = _sample_and_split(all_tasks_ordered, tpt, rng_slda)
        pred_s, slda_art = _run_slda(all_tasks_ordered, train_ds_s, val_ds_s)
        slda_cmse[s_i, b_i], slda_rho[s_i, b_i] = _probe_stats(pred_s)
        print(f"    SLDA   cMSE−NF={slda_cmse[s_i,b_i]:+.5f}  ρ={slda_rho[s_i,b_i]:.3f}")
        _save_ckpt(f"seed{seed_val}_tpt{tpt}", dlbt_ck, slda_art)
        del train_ds_s, val_ds_s, pred_s, dlbt_ck, slda_art

        # Anti-human DLBT (label-flipped)
        train_ds_a, val_ds_a = _sample_and_split(all_tasks_ordered, tpt, rng_anti, flip=True)
        agent_a  = _init_agent(seed_val)
        chosen_a = _run_dlbt(agent_a, train_ds_a, val_ds_a)
        pred_a   = _dlbt_probe_matrix(chosen_a)
        anti_cmse[s_i, b_i], anti_rho[s_i, b_i] = _probe_stats(pred_a)
        print(f"    Anti   cMSE−NF={anti_cmse[s_i,b_i]:+.5f}  ρ={anti_rho[s_i,b_i]:.3f}"
              f"  (base={'yes' if chosen_a is base_agent else 'no'})")
        del agent_a, chosen_a, train_ds_a, val_ds_a, pred_a
        gc.collect(); torch.cuda.empty_cache()

    # ---- All-data point  (fixed cell-level 10 % eval split, same as 02) -----
    print(f"\n  [All data — {pool_all_size:,} train trials, avg {avg_pool_per_task:.0f}/task]")

    # DLBT all  — train on 90 % pool, val on fixed cell-level 10 % eval split
    all_tr = _all_data_ds(all_tasks_ordered)
    agent_all  = _init_agent(seed_val)
    chosen_all = _run_dlbt(agent_all, all_tr, eval_ds_global)
    pred_all   = _dlbt_probe_matrix(chosen_all)
    dlbt_all_cmse[s_i], dlbt_all_rho[s_i] = _probe_stats(pred_all)
    used_base_all = (chosen_all is base_agent)
    print(f"    DLBT all  cMSE−NF={dlbt_all_cmse[s_i]:+.5f}  ρ={dlbt_all_rho[s_i]:.3f}"
          f"  (base={'yes' if used_base_all else 'no'})")
    dlbt_ck_all = _dlbt_ckpt(agent_all, used_base_all)
    del chosen_all, all_tr, pred_all, agent_all
    gc.collect(); torch.cuda.empty_cache()

    # SLDA all  — train on 90 % pool, model selection on fixed 10 % eval split
    all_tr_s = _all_data_ds(all_tasks_ordered)
    pred_sa, slda_art_all = _run_slda(all_tasks_ordered, all_tr_s, eval_ds_global)
    slda_all_cmse[s_i], slda_all_rho[s_i] = _probe_stats(pred_sa)
    print(f"    SLDA all  cMSE−NF={slda_all_cmse[s_i]:+.5f}  ρ={slda_all_rho[s_i]:.3f}")
    _save_ckpt(f"seed{seed_val}_all", dlbt_ck_all, slda_art_all)
    del all_tr_s, pred_sa, dlbt_ck_all, slda_art_all

    # Anti all  — train on 90 % pool (flipped), val on fixed anti eval split
    all_tr_a = _all_data_ds(all_tasks_ordered, flip=True)
    agent_aa  = _init_agent(seed_val)
    chosen_aa = _run_dlbt(agent_aa, all_tr_a, eval_ds_anti_global)
    pred_aa   = _dlbt_probe_matrix(chosen_aa)
    anti_all_cmse[s_i], anti_all_rho[s_i] = _probe_stats(pred_aa)
    print(f"    Anti all  cMSE−NF={anti_all_cmse[s_i]:+.5f}  ρ={anti_all_rho[s_i]:.3f}"
          f"  (base={'yes' if chosen_aa is base_agent else 'no'})")
    del agent_aa, chosen_aa, all_tr_a, pred_aa
    gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
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
    "dlbt_cmse":           dlbt_cmse,
    "dlbt_rho":            dlbt_rho,
    "slda_cmse":           slda_cmse,
    "slda_rho":            slda_rho,
    "anti_cmse":           anti_cmse,
    "anti_rho":            anti_rho,
    # All-data point [n_seeds]
    "dlbt_all_cmse":       dlbt_all_cmse,
    "dlbt_all_rho":        dlbt_all_rho,
    "slda_all_cmse":       slda_all_cmse,
    "slda_all_rho":        slda_all_rho,
    "anti_all_cmse":       anti_all_cmse,
    "anti_all_rho":        anti_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
