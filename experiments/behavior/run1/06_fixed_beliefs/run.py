"""
run1/06_fixed_beliefs/run.py — oracle belief ablation evaluation.

No training.  OracleBTAgent uses fixed Dirichlet beliefs peaked at each
image's true latent state (from metadata), bypassing the CLIP encoder and
learned mapper entirely.

Protocol
--------
1.  Load + filter run0+run1 behavioural data; identify eligible tasks.
2.  Build ground-truth probe matrix from probe-image count cells.
3.  Evaluate OracleBTAgent on every probe image × every task in one
    forward pass — no CLIP cache needed.
4.  Compute cMSE−NF and Spearman ρ.
5.  Save results/oracle_beliefs.pkl.

Run from repo root:
    python experiments/behavior/run1/06_fixed_beliefs/run.py
"""

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

from dlbt.agents.oracle_bt import OracleBTAgent
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

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
print(f"Device: {device}")
print(f"RUN_TAG: {cfg.RUN_TAG}  |  concentration={cfg.CONCENTRATION}  background={cfg.BACKGROUND}")

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
print(f"  Raw trials: {len(df_raw):,}  ({df_raw['assignment_id'].nunique()} assignments)")

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
full_ds, probe_uids, _ = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _beh_id_eligible,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)
print(f"  Aggregated: {len(full_ds):,} cells  "
      f"({full_ds.df['task_name'].nunique()} tasks, "
      f"{full_ds.df['uid'].nunique()} images)")

# ---------------------------------------------------------------------------
# Probe matrix ordering (sorted by latent_state — consistent with 02/022)
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
# Ground-truth probe matrix  [n_probe × n_tasks]
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
print(f"  Ground truth: {n_filled}/{n_probe * n_all_tasks} cells filled.")

# Probe noise floor
_nf_mask = count_matrix > 1
if _nf_mask.any():
    _p = true_matrix[_nf_mask]
    _n = count_matrix[_nf_mask].astype(float)
    probe_noise_floor = float(np.mean(_p * (1 - _p) / (_n - 1)))
else:
    probe_noise_floor = 0.0

_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  "
      f"random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Oracle evaluation — single forward pass, no training
# ---------------------------------------------------------------------------
print(f"\nEvaluating OracleBTAgent  "
      f"(concentration={cfg.CONCENTRATION}, background={cfg.BACKGROUND}) ...")

agent = OracleBTAgent(
    concentration     = cfg.CONCENTRATION,
    background        = cfg.BACKGROUND,
    device            = device,
    normalize_utility = cfg.NORMALIZED_UTILITY,
)
agent.eval()

pred_matrix = np.full((n_probe, n_all_tasks), np.nan)

with torch.no_grad():
    for j, task_name in enumerate(all_tasks_ordered):
        task  = get_task(task_name)
        probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
        pred_matrix[:, j] = probs

# cMSE−NF
valid           = ~np.isnan(pred_matrix) & ~np.isnan(true_matrix)
oracle_cmse_net = float(np.mean((pred_matrix[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor

# Spearman ρ
r, _       = spearmanr(pred_matrix[valid], true_matrix[valid])
oracle_rho = float(r)

print(f"  Oracle cMSE−NF : {oracle_cmse_net:.5f}")
print(f"  Oracle Spearman ρ : {oracle_rho:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":           cfg.RUN_TAG,
    "concentration":     cfg.CONCENTRATION,
    "background":        cfg.BACKGROUND,
    "all_tasks_ordered": all_tasks_ordered,
    "probe_uids_ordered": probe_uids_ordered,
    "true_matrix":       true_matrix,
    "pred_matrix":       pred_matrix,
    "probe_noise_floor": probe_noise_floor,
    "random_cmse_net":   random_cmse_net,
    "oracle_cmse_net":   oracle_cmse_net,
    "oracle_rho":        oracle_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
