"""
02_data_efficiency/run.py — data-efficiency sweep for DLBT on real human data.

Protocol:
  1. Load data and apply the same 10% cell-level eval split as 01_fit.
  2. Expand all (main × TRAIN_TASKS) training trials into a flat pool of
     individual trials (uid, task_name, outcome).
  3. For each trial budget B in TRIAL_BUDGETS:
       - Sample B trials uniformly from the pool (without replacement; with
         replacement if B > pool size).
       - Re-aggregate into count cells per (uid, task_name).
       - Train DLBT with early stopping on the fixed eval_ds.
       - Evaluate on all 4 regions.
  4. Save a summary dict indexed by budget label.

The eval_ds uses the same 10% cell split (full counts, not subsampled) so
that the stopping signal is always comparably noisy across budgets.

Run from repo root:
    python experiments/behavior/run0/02_data_efficiency/run.py
"""

import gc
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.metrics import corrected_mse

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Load + preprocess behavioural data
# ---------------------------------------------------------------------------
print("\nLoading behavioural data...")
full_ds, probe_uids, main_uids, diag = load_and_preprocess(
    cfg.BEHAVIOR_CSV,
    beh_id_to_task    = cfg.BEH_ID_TO_TASK,
    min_catch_perf    = cfg.MIN_CATCH_PERF,
    main_perf_quantile= cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds   = cfg.USE_TRIAL_KINDS,
    seed              = cfg.SEED,
)
train_uids = set(main_uids)
test_uids  = set(probe_uids)

# ---------------------------------------------------------------------------
# Cell-level eval split (identical to 01_fit)
# ---------------------------------------------------------------------------
main_train_mask = (
    full_ds.df["uid"].isin(main_uids) &
    full_ds.df["task_name"].isin(cfg.TRAIN_TASKS)
)
main_train_df = full_ds.df[main_train_mask].copy().reset_index(drop=True)

rng_split = np.random.default_rng(cfg.SEED)
n_eval    = max(1, int(len(main_train_df) * cfg.EVAL_CELL_FRAC))
eval_idx  = rng_split.choice(len(main_train_df), size=n_eval, replace=False)
eval_mask = np.zeros(len(main_train_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df      = main_train_df[eval_mask].reset_index(drop=True)
all_train_df = main_train_df[~eval_mask].reset_index(drop=True)

eval_ds = BehavioralDataset(eval_df)
print(f"\nEval  cells: {len(eval_df)}")
print(f"Train cells (90%%): {len(all_train_df)}  (pool for subsampling)")

# ---------------------------------------------------------------------------
# Expand training cells to individual trials
# ---------------------------------------------------------------------------
# Each (uid, task_name, count_0, count_1) cell expands to
#   count_0 trials with outcome=0, count_1 trials with outcome=1.
rows_uid  = []
rows_task = []
rows_out  = []
for row in all_train_df.itertuples(index=False):
    for _ in range(int(row.count_0)):
        rows_uid.append(row.uid)
        rows_task.append(row.task_name)
        rows_out.append(0)
    for _ in range(int(row.count_1)):
        rows_uid.append(row.uid)
        rows_task.append(row.task_name)
        rows_out.append(1)

trial_uid  = np.array(rows_uid)
trial_task = np.array(rows_task)
trial_out  = np.array(rows_out, dtype=np.int32)
N_POOL = len(trial_uid)
print(f"Total training trial pool: {N_POOL} individual trials")

# ---------------------------------------------------------------------------
# Generalization evaluation regions (probe images only)
# ---------------------------------------------------------------------------
def _slice(ds: BehavioralDataset, task_names, uids) -> BehavioralDataset:
    sub = ds.df[
        ds.df["task_name"].isin(task_names) & ds.df["uid"].isin(uids)
    ].copy()
    return BehavioralDataset(sub)

stim_gen_ds  = _slice(full_ds, cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = _slice(full_ds, cfg.VAL_TASKS,   train_uids)
joint_gen_ds = _slice(full_ds, cfg.VAL_TASKS,   test_uids)

# ---------------------------------------------------------------------------
# CLIP feature cache
# ---------------------------------------------------------------------------
_agent_for_cache = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                             device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    _agent_for_cache.load_cache(str(cache_path))
else:
    _agent_for_cache.precompute_features(list(refs_dict.values()))
    _agent_for_cache.save_cache(str(cache_path))
frozen_clip = {uid: feat.clone() for uid, feat in _agent_for_cache._cache.items()}
del _agent_for_cache

# ---------------------------------------------------------------------------
# Noise floors (constant across budgets)
# ---------------------------------------------------------------------------
noise_floors = {
    "eval":      eval_ds.noise_floor(),
    "stim_gen":  stim_gen_ds.noise_floor(),
    "task_gen":  task_gen_ds.noise_floor(),
    "joint_gen": joint_gen_ds.noise_floor(),
}
print(f"Noise floors: {noise_floors}")

# ---------------------------------------------------------------------------
# Empirical truth lookup
# ---------------------------------------------------------------------------
_emp_lookup: dict = {}
for row in full_ds.df.itertuples(index=False):
    total = row.count_0 + row.count_1
    p     = row.count_1 / total if total > 0 else np.nan
    _emp_lookup[(row.uid, row.task_name)] = (p, total)

def emp_p(uid, tn): v = _emp_lookup.get((uid, tn)); return v[0] if v else np.nan
def emp_n(uid, tn): v = _emp_lookup.get((uid, tn)); return v[1] if v else 0


def _collect_preds(agent, ds_list):
    """Collect predictions over a list of (label, ds) pairs."""
    out = {}
    agent.eval()
    for label, ds in ds_list:
        out[label] = {}
        for task_name, group in ds.iter_tasks():
            task       = TASKS[task_name]
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
            totals     = np.array([emp_n(r.uid, task_name)  for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
            out[label][task_name] = {
                "pred": pred, "true": true_p, "totals": totals,
                "uids": [r.uid for r in batch_refs],
            }
    return out


def _build_subsampled_ds(B: int, rng: np.random.Generator) -> BehavioralDataset:
    """
    Sample B trials uniformly from the full training pool and re-aggregate
    into a BehavioralDataset.  Uses replacement if B > N_POOL.
    """
    replace = B > N_POOL
    idx     = rng.choice(N_POOL, size=B, replace=replace)
    sub_uid  = trial_uid[idx]
    sub_task = trial_task[idx]
    sub_out  = trial_out[idx]

    # Aggregate back to counts
    sub_df = pd.DataFrame({"uid": sub_uid, "task_name": sub_task, "outcome": sub_out})
    grp = (
        sub_df.groupby(["uid", "task_name"])["outcome"]
        .agg(count_1="sum", n="count")
        .reset_index()
    )
    grp["count_0"] = grp["n"] - grp["count_1"]
    grp = grp.drop(columns="n")
    return BehavioralDataset(grp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


# ---------------------------------------------------------------------------
# Data-efficiency sweep
# ---------------------------------------------------------------------------
results_per_budget = {}

rng_run = np.random.default_rng(cfg.SEED + 1)  # separate rng for subsampling

# Process budgets in order of actual trial count:
#   without-replacement first (B ≤ N_POOL), then "full", then with-replacement (B > N_POOL).
# This keeps the sweep semantically ordered even when n_pool < max(integer budgets).
_int_budgets_in  = sorted([b for b in cfg.TRIAL_BUDGETS if b != "full" and b <= N_POOL])
_int_budgets_out = sorted([b for b in cfg.TRIAL_BUDGETS if b != "full" and b >  N_POOL])
_has_full        = "full" in cfg.TRIAL_BUDGETS
_budgets_ordered = _int_budgets_in + (["full"] if _has_full else []) + _int_budgets_out

for budget in _budgets_ordered:
    budget_label = "full" if budget == "full" else str(budget)
    print(f"\n{'='*60}")
    print(f"Budget: {budget_label}")

    # Build training dataset for this budget
    if budget == "full":
        train_ds_b = BehavioralDataset(all_train_df.copy())
    else:
        train_ds_b = _build_subsampled_ds(int(budget), rng_run)

    n_cells_b  = len(train_ds_b)
    n_trials_b = int((train_ds_b.df["count_0"] + train_ds_b.df["count_1"]).sum())
    print(f"  cells={n_cells_b}  trials={n_trials_b}")

    if n_cells_b == 0:
        print("  → Empty dataset, skipping.")
        continue

    # Fresh agent — always start phase 1 with frozen encoder
    torch.manual_seed(cfg.SEEDS[0])
    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                      device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}

    # Phase 1: mapper warmup (frozen encoder)
    print("  Phase 1 — mapper warmup...")
    phase1 = train_dlbt(
        agent, train_ds_b, eval_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
        extra_val_datasets = {
            "stim_gen":  stim_gen_ds,
            "task_gen":  task_gen_ds,
            "joint_gen": joint_gen_ds,
        },
    )
    print(f"  Phase 1 best epoch: {phase1.best_epoch}  eval_mse: {phase1.best_val_mse:.4f}")

    # Phase 2: attnpool fine-tuning (optional)
    phase2 = None
    if not cfg.FREEZE_ENCODER:
        print("  Phase 2 — attnpool fine-tuning...")
        gc.collect()
        torch.cuda.empty_cache()

        for p in agent.mapper.parameters():
            p.requires_grad_(False)
        for p in agent.encoder.attnpool.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = False
        agent._cache.clear()

        optimizer2 = torch.optim.Adam(
            agent.encoder.attnpool.parameters(), lr=cfg.LR_ATTNPOOL
        )
        phase2 = train_dlbt(
            agent, train_ds_b, eval_ds, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            optimizer = optimizer2,
            extra_val_datasets = {
                "stim_gen":  stim_gen_ds,
                "task_gen":  task_gen_ds,
                "joint_gen": joint_gen_ds,
            },
        )
        print(f"  Phase 2 best epoch: {phase2.best_epoch}  eval_mse: {phase2.best_val_mse:.4f}")

        print("  Repopulating feature cache...")
        agent.eval()
        all_refs_list = list(refs_dict.values())
        with torch.no_grad():
            for i in tqdm(range(0, len(all_refs_list), 16), desc="  caching", unit="batch"):
                batch   = all_refs_list[i : i + 16]
                spatial = torch.stack(
                    [agent._backbone_cache[r.uid] for r in batch]
                ).to(agent.device)
                feats = agent.encoder.attnpool(spatial).float()
                for ref, feat in zip(batch, feats):
                    agent._cache[ref.uid] = feat.cpu()

    result = phase2 if phase2 is not None else phase1
    print(f"  best epoch: {result.best_epoch}  eval_mse: {result.best_val_mse:.4f}")

    # Collect predictions — agent is at best-checkpoint weights here.
    # train_dlbt restores best_state via agent.load_state_dict(best_state)
    # before returning, so these preds reflect the early-stopped model.
    agent.eval()
    preds = _collect_preds(agent, [
        ("train",     train_ds_b),
        ("eval",      eval_ds),
        ("stim_gen",  stim_gen_ds),
        ("task_gen",  task_gen_ds),
        ("joint_gen", joint_gen_ds),
    ])

    # Compute aggregate corrected-MSE minus noise-floor per region
    def _region_cmse_net(label, task_list, nf_key):
        pt = preds[label]
        all_preds  = np.concatenate([pt[t]["pred"]   for t in task_list if t in pt])
        all_trues  = np.concatenate([pt[t]["true"]   for t in task_list if t in pt])
        all_totals = np.concatenate([pt[t]["totals"] for t in task_list if t in pt])
        valid = all_totals > 0
        if not valid.any():
            return float("nan")
        p = all_preds[valid]
        t = all_trues[valid]
        raw = float(np.mean((p - t) ** 2))
        # MC correction
        if cfg.N_MC > 1:
            raw -= float(np.mean(p * (1 - p))) / (cfg.N_MC - 1)
        nf  = noise_floors.get(nf_key, 0.0)
        return raw - nf

    metrics = {
        "n_trials":           n_trials_b,
        "n_cells":            n_cells_b,
        "best_epoch":         result.best_epoch,
        "eval_mse":           result.best_val_mse,
        "train_cmse_net":     _region_cmse_net("train",     cfg.TRAIN_TASKS, "train"),
        "stim_gen_cmse_net":  _region_cmse_net("stim_gen",  cfg.TRAIN_TASKS, "stim_gen"),
        "task_gen_cmse_net":  _region_cmse_net("task_gen",  cfg.VAL_TASKS,   "task_gen"),
        "joint_gen_cmse_net": _region_cmse_net("joint_gen", cfg.VAL_TASKS,   "joint_gen"),
        "preds":              preds,
        "curves":             dict(
            train_mses  = _concat(phase1.train_mses,
                                  phase2.train_mses  if phase2 else None),
            eval_mses   = _concat(phase1.val_mses,
                                  phase2.val_mses    if phase2 else None),
            stim_mses   = _concat(phase1.extra_val_mses.get("stim_gen",  []),
                                  phase2.extra_val_mses.get("stim_gen",  []) if phase2 else None),
            task_mses   = _concat(phase1.extra_val_mses.get("task_gen",  []),
                                  phase2.extra_val_mses.get("task_gen",  []) if phase2 else None),
            joint_mses  = _concat(phase1.extra_val_mses.get("joint_gen", []),
                                  phase2.extra_val_mses.get("joint_gen", []) if phase2 else None),
        ),
    }
    for k in ["train_cmse_net", "stim_gen_cmse_net", "task_gen_cmse_net", "joint_gen_cmse_net"]:
        print(f"  {k}: {metrics[k]:.4f}")

    results_per_budget[budget_label] = metrics

# ---------------------------------------------------------------------------
# Random-guesser baseline for joint_gen
# ---------------------------------------------------------------------------
# MSE of predicting 0.5 for every probe × val-task cell, corrected for NF.
_jg_trues  = np.concatenate([
    full_ds.df.loc[
        full_ds.df["uid"].isin(test_uids) & (full_ds.df["task_name"] == t),
        ["count_0","count_1"]
    ].eval("count_1 / (count_0 + count_1)").values
    for t in cfg.VAL_TASKS
    if len(full_ds.df.loc[
        full_ds.df["uid"].isin(test_uids) & (full_ds.df["task_name"] == t)
    ]) > 0
])
_jg_totals = np.concatenate([
    (full_ds.df.loc[
        full_ds.df["uid"].isin(test_uids) & (full_ds.df["task_name"] == t),
        ["count_0","count_1"]
    ].sum(axis=1)).values
    for t in cfg.VAL_TASKS
    if len(full_ds.df.loc[
        full_ds.df["uid"].isin(test_uids) & (full_ds.df["task_name"] == t)
    ]) > 0
])
valid_jg = _jg_totals > 0
random_guesser_joint_gen_cmse_net = float(
    np.mean((0.5 - _jg_trues[valid_jg]) ** 2)
) - noise_floors["joint_gen"]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = dict(
    trial_budgets      = cfg.TRIAL_BUDGETS,
    noise_floors       = noise_floors,
    results            = results_per_budget,
    random_guesser_joint_gen_cmse_net = random_guesser_joint_gen_cmse_net,
    n_pool             = N_POOL,
    eval_cell_frac     = cfg.EVAL_CELL_FRAC,
)
out_path = cfg.RESULTS_DIR / f"data_efficiency_{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved -> {out_path}")
