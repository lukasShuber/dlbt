"""
run1/06_slda_finetuning/run.py — SLDA Phase 2 diagnostic sandbox.

Protocol
--------
1.  Load + filter run0+run1 data; identify all eligible tasks.
2.  Separate probe images (evaluation) from main images (training).
3.  90/10 split of main cells → train_ds (fixed) / eval_ds (early stopping).
4.  Build ground-truth probe matrix + noise floor.
5.  Pre-compute frozen CLIP features (loaded from shared cache).

6.  Phase 1: fit_slda_logreg on frozen CLIP features (matching 021_efficiency_main).
    Evaluate on probe set → phase1_cmse, phase1_rho.

7.  For each learning rate in LR_ATTNPOOL_VARIANTS:
      a. Phase 2: fine-tune attnpool through fixed Phase-1 decoders.
         Epoch hook: compute probe cMSE every EVAL_EVERY epochs using
         current fine-tuned features + frozen Phase-1 scalers/models.
      b. After convergence: compute probe cMSE/rho with fine-tuned features
         (frozen scalers — raw Phase 2 output).
      c. Refit-scaler variant: refit StandardScaler per task on fine-tuned
         training features (keep LogReg weights), compute probe cMSE/rho.

8.  Save results/slda_sandbox.pkl.

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
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

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
print(f"  Aggregated: {len(full_ds.df):,} cells  "
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
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE-NF: {random_cmse_net:.5f}")

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
        k2   = count1s - k1
        p1   = k1 / n1s
        p2   = k2 / n2s
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
# 90 / 10 train / eval split of main cells
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split     = np.random.default_rng(cfg.SEED)
n_eval_cells  = max(1, int(len(main_cells_df) * 0.10))
eval_idx      = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask     = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

train_df = main_cells_df[~eval_mask].reset_index(drop=True)
eval_df  = main_cells_df[eval_mask].reset_index(drop=True)
train_ds = BehavioralDataset(train_df)
eval_ds  = BehavioralDataset(eval_df)

print(f"\n  Train cells (90%%): {len(train_df)}")
print(f"  Eval  cells (10%%): {len(eval_df)}")

# ---------------------------------------------------------------------------
# Frozen CLIP feature cache
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

def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


@torch.no_grad()
def _extract_probe_features(agent: SldaAgent) -> dict[str, np.ndarray]:
    """Extract current attnpool features for all probe images."""
    agent.eval()
    feats = agent._encode(probe_refs_ordered)   # [n_probe, D]
    return {r.uid: feats[i].cpu().numpy()
            for i, r in enumerate(probe_refs_ordered)}


@torch.no_grad()
def _extract_train_features(agent: SldaAgent) -> dict[str, np.ndarray]:
    """Extract current attnpool features for all training images."""
    agent.eval()
    train_uids = train_ds.df["uid"].unique().tolist()
    present    = [uid for uid in train_uids if uid in refs_by_uid]
    result     = {}
    for uid in present:
        feat = agent._encode([refs_by_uid[uid]])   # [1, D]
        result[uid] = feat[0].cpu().numpy()
    return result


def _refit_scalers(
    agent:         SldaAgent,
    scalers_orig:  dict,
) -> dict:
    """
    Refit StandardScaler per task on fine-tuned features, keeping the
    original LogReg weights unchanged.

    Returns new_scalers dict (same keys as scalers_orig; tasks with no
    training images keep the original scaler).
    """
    train_uids = set(train_ds.df["uid"].unique())
    new_scalers = {}
    for task_name, orig_sc in scalers_orig.items():
        group = train_ds.df[train_ds.df["task_name"] == task_name]
        uids  = [uid for uid in group["uid"].tolist()
                 if uid in train_uids and uid in refs_by_uid]
        if len(uids) < 2:
            new_scalers[task_name] = orig_sc   # keep original
            continue
        with torch.no_grad():
            feats = agent._encode([refs_by_uid[uid] for uid in uids])   # [N, D]
        new_sc = StandardScaler()
        new_sc.fit(feats.cpu().numpy())
        new_scalers[task_name] = new_sc
    return new_scalers


# ---------------------------------------------------------------------------
# Phase 1 — fit LogReg on frozen CLIP features
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Phase 1 — fitting LogReg on frozen CLIP features...")

frozen_feat_dict = {uid: t.clone() for uid, t in frozen_clip.items()}

scalers_p1, models_p1, use_base_p1 = fit_slda_logreg(
    tasks         = all_tasks_ordered,
    train_ds      = train_ds,
    val_ds        = eval_ds,
    clip_features = frozen_feat_dict,
    Cs            = cfg.SLDA_Cs,
    max_iter      = cfg.SLDA_MAX_ITER,
)
n_fitted = len(models_p1)
n_base   = sum(v for v in use_base_p1.values())
print(f"  Fitted: {n_fitted} tasks  |  model selection chose base: {n_base}")

probe_feats_frozen = {uid: frozen_clip[uid].cpu().numpy()
                      for uid in probe_uids_ordered if uid in frozen_clip}
pred_p1 = slda_probe_matrix(
    scalers_p1, models_p1, use_base_p1,
    probe_feats_frozen, all_tasks_ordered, uid_to_row, n_probe,
)
phase1_cmse, phase1_rho = _probe_stats(pred_p1)
print(f"  Phase 1  cMSE-NF={phase1_cmse:+.5f}  ρ={phase1_rho:.4f}")

# ---------------------------------------------------------------------------
# Phase 2 — LR variants
# ---------------------------------------------------------------------------
variants = []

for lr_val in cfg.LR_ATTNPOOL_VARIANTS:
    print(f"\n{'='*60}")
    print(f"Phase 2 — lr={lr_val:.0e}")

    agent = SldaAgent(freeze_encoder=False, device=device)
    agent.precompute_backbone_features(all_refs)

    # Epoch hook: compute probe cMSE with frozen scalers + fine-tuned features
    def _make_hook(sc, mo, ub):
        @torch.no_grad()
        def _hook(epoch: int, ag: SldaAgent) -> float:
            pf = _extract_probe_features(ag)
            pr = slda_probe_matrix(sc, mo, ub, pf,
                                   all_tasks_ordered, uid_to_row, n_probe)
            cmse_nf, _ = _probe_stats(pr)
            return cmse_nf
        return _hook

    hook = _make_hook(scalers_p1, models_p1, use_base_p1)

    result = finetune_slda_attnpool(
        agent      = agent,
        scalers    = scalers_p1,
        models     = models_p1,
        train_ds   = train_ds,
        eval_ds    = eval_ds,
        refs_dict  = refs_dict,
        n_epochs   = cfg.N_EPOCHS_ATTNPOOL,
        patience   = cfg.PATIENCE_ATTNPOOL,
        lr         = lr_val,
        batch_size = cfg.BATCH_SIZE_ATTNPOOL,
        epoch_hook = hook,
        eval_every = cfg.EVAL_EVERY,
    )
    print(f"  Best epoch: {result.best_epoch}  best val NLL: {result.best_val_nll:.1f}")

    # ── Phase 2 probe evaluation (frozen scalers, fine-tuned features) ────
    probe_feats_ft = _extract_probe_features(agent)
    pred_p2 = slda_probe_matrix(
        scalers_p1, models_p1, use_base_p1,
        probe_feats_ft, all_tasks_ordered, uid_to_row, n_probe,
    )
    phase2_cmse, phase2_rho = _probe_stats(pred_p2)
    print(f"  Phase 2 (frozen sc)  cMSE-NF={phase2_cmse:+.5f}  ρ={phase2_rho:.4f}")

    # ── Refit-scaler variant ───────────────────────────────────────────────
    scalers_refit = _refit_scalers(agent, scalers_p1)
    pred_rs = slda_probe_matrix(
        scalers_refit, models_p1, use_base_p1,
        probe_feats_ft, all_tasks_ordered, uid_to_row, n_probe,
    )
    refit_cmse, refit_rho = _probe_stats(pred_rs)
    print(f"  Phase 2 (refit sc)   cMSE-NF={refit_cmse:+.5f}  ρ={refit_rho:.4f}")

    variants.append({
        "lr":                lr_val,
        "best_epoch":        result.best_epoch,
        "best_val_nll":      result.best_val_nll,
        "train_nll":         result.train_nll,
        "val_nll":           result.val_nll,
        "hook_epochs":       result.hook_epochs,
        "hook_cmse":         result.hook_results,
        "phase2_cmse":       phase2_cmse,
        "phase2_rho":        phase2_rho,
        "refit_scaler_cmse": refit_cmse,
        "refit_scaler_rho":  refit_rho,
    })

    del agent
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
    "seeds":               cfg.SEEDS,
    "all_tasks_ordered":   all_tasks_ordered,
    "probe_uids_ordered":  probe_uids_ordered,
    "true_matrix":         true_matrix,
    "count_matrix":        count_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_net":     random_cmse_net,
    "rho_noise_ceiling":   rho_noise_ceiling,
    # Phase 1 reference
    "phase1_cmse":         phase1_cmse,
    "phase1_rho":          phase1_rho,
    # Phase 2 variants (list, one entry per LR)
    "variants":            variants,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  {'Condition':<30}  {'cMSE-NF':>10}  {'ρ':>8}")
print("  " + "-" * 52)
print(f"  {'Phase 1 (frozen CLIP)':<30}  {phase1_cmse:+10.5f}  {phase1_rho:8.4f}")
for v in variants:
    lr_str = f"lr={v['lr']:.0e}"
    print(f"  {'Phase 2 ('+lr_str+', frozen sc)':<30}  "
          f"{v['phase2_cmse']:+10.5f}  {v['phase2_rho']:8.4f}")
    print(f"  {'Phase 2 ('+lr_str+', refit sc)':<30}  "
          f"{v['refit_scaler_cmse']:+10.5f}  {v['refit_scaler_rho']:8.4f}")
print("=" * 60)
