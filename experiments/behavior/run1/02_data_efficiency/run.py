"""
run1/02_data_efficiency/run.py — data-efficiency sweep for DLBT on run0+run1 data.

Protocol:
  1. Concatenate run0 + run1 CSVs, apply QC filtering, aggregate counts.
  2. Use the same SPLIT_MODE task split as 01_fit (arity or random).
  3. Cell-level 10% eval split on (main × TRAIN_TASKS).
  4. Expand all training cells to a flat pool of individual trials.
  5. For each budget B in TRIAL_BUDGETS:
       - Sample B trials uniformly (without replacement; with if B > pool).
       - Re-aggregate into count cells.
       - Train DLBT (phase 1 + optional phase 2 attnpool).
       - Evaluate on all 4 regions; optionally run h_n threshold correction
         on val tasks (task_gen + joint_gen).
  6. Save summary dict indexed by budget label.

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/run.py
"""

import gc
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as scipy_beta
from torch.distributions import Dirichlet
from tqdm import tqdm

from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.data.dataset import BehavioralDataset
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

print(f"\nTask split [{cfg.SPLIT_MODE}]: "
      f"{len(cfg.TRAIN_TASKS)} train / {len(cfg.VAL_TASKS)} val  "
      f"(MIN_TASK_ASSIGNMENTS={cfg.MIN_TASK_ASSIGNMENTS})")

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Load + concatenate + preprocess behavioural data
# ---------------------------------------------------------------------------
print("\nLoading behavioural data...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
print(f"  Combined raw trials: {len(df_raw):,}  "
      f"({df_raw['assignment_id'].nunique()} assignments)")

df_filtered, diag = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)

# Only include eligible tasks in full_ds
_eligible_names  = set(cfg.TRAIN_TASKS + cfg.VAL_TASKS)
_eligible_beh_id = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in _eligible_names}

full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _eligible_beh_id,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)
print(f"  Aggregated cells: {len(full_ds):,}  "
      f"({full_ds.df['task_name'].nunique()} tasks, "
      f"{full_ds.df['uid'].nunique()} images)")

train_uids = set(main_uids)
test_uids  = set(probe_uids)

# ---------------------------------------------------------------------------
# Cell-level eval split — 10% of (main × TRAIN_TASKS) cells
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
# Generalization regions
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
# Expand training cells to individual trials
# ---------------------------------------------------------------------------
rows_uid, rows_task, rows_out = [], [], []
for row in all_train_df.itertuples(index=False):
    for _ in range(int(row.count_0)):
        rows_uid.append(row.uid); rows_task.append(row.task_name); rows_out.append(0)
    for _ in range(int(row.count_1)):
        rows_uid.append(row.uid); rows_task.append(row.task_name); rows_out.append(1)

trial_uid  = np.array(rows_uid)
trial_task = np.array(rows_task)
trial_out  = np.array(rows_out, dtype=np.int32)
N_POOL     = len(trial_uid)
print(f"Total training trial pool: {N_POOL} individual trials")

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
            task       = get_task(task_name)
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
            totals     = np.array([emp_n(r.uid, task_name) for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
            out[label][task_name] = {
                "pred": pred, "true": true_p, "totals": totals,
                "uids": [r.uid for r in batch_refs],
            }
    return out


def _build_subsampled_ds(B: int, rng: np.random.Generator) -> BehavioralDataset:
    replace = B > N_POOL
    idx     = rng.choice(N_POOL, size=B, replace=replace)
    sub_df  = pd.DataFrame({
        "uid": trial_uid[idx], "task_name": trial_task[idx], "outcome": trial_out[idx]
    })
    grp = (
        sub_df.groupby(["uid", "task_name"])["outcome"]
        .agg(count_1="sum", n="count").reset_index()
    )
    grp["count_0"] = grp["n"] - grp["count_1"]
    return BehavioralDataset(grp.drop(columns="n"))


# ---------------------------------------------------------------------------
# Arity-adjusted threshold helpers
# ---------------------------------------------------------------------------
def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _h(task_name: str) -> float:
    """h_n = 2·median(Beta(K₊, K₋)) − 1 in b·Δu space (h=0 for 1-way)."""
    from dlbt.constants import K as _K
    n       = _arity(task_name)
    k_plus  = _K // (2 ** n)
    k_minus = _K - k_plus
    return 2.0 * scipy_beta.median(k_plus, k_minus) - 1.0


def _collect_h_preds(agent, ds, h_tasks):
    """MC inference with per-task arity-adjusted threshold h_n."""
    out = {}
    agent.eval()
    for task_name, group in ds.iter_tasks():
        if task_name not in h_tasks:
            continue
        h_val   = _h(task_name)
        task    = get_task(task_name)
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=agent.device)
        uids    = group["uid"].tolist()
        batch_refs = [refs_dict[uid] for uid in uids]
        true_p  = np.array([emp_p(uid, task_name) for uid in uids])
        totals  = np.array([emp_n(uid, task_name) for uid in uids])
        with torch.no_grad():
            alpha  = agent.get_alpha(batch_refs).clamp(min=0.1)
            b      = Dirichlet(alpha).sample((cfg.N_MC,))
            logit  = torch.einsum("nbk,k->nb", b, delta_u)
            p_corr = (logit > h_val).float().mean(dim=0).cpu().numpy()
        out[task_name] = {
            "pred": p_corr, "true": true_p, "totals": totals,
            "uids": uids, "h": h_val, "n_way": _arity(task_name),
        }
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


def _region_cmse_net(preds_label, task_list, nf_key, preds_dict):
    pt = preds_dict.get(preds_label, {})
    tasks_present = [t for t in task_list if t in pt]
    if not tasks_present:
        return float("nan")
    all_preds  = np.concatenate([pt[t]["pred"]   for t in tasks_present])
    all_trues  = np.concatenate([pt[t]["true"]   for t in tasks_present])
    all_totals = np.concatenate([pt[t]["totals"] for t in tasks_present])
    valid = all_totals > 0
    if not valid.any():
        return float("nan")
    p   = all_preds[valid]; t_v = all_trues[valid]
    raw = float(np.mean((p - t_v) ** 2))
    if cfg.N_MC > 1:
        raw -= float(np.mean(p * (1 - p))) / (cfg.N_MC - 1)
    return raw - noise_floors.get(nf_key, 0.0)


# ---------------------------------------------------------------------------
# Data-efficiency sweep
# ---------------------------------------------------------------------------
results_per_budget = {}
rng_run = np.random.default_rng(cfg.SEED + 1)

_int_budgets_in  = sorted([b for b in cfg.TRIAL_BUDGETS if b != "full" and b <= N_POOL])
_int_budgets_out = sorted([b for b in cfg.TRIAL_BUDGETS if b != "full" and b >  N_POOL])
_has_full        = "full" in cfg.TRIAL_BUDGETS
_budgets_ordered = _int_budgets_in + (["full"] if _has_full else []) + _int_budgets_out

_h_tasks = set(cfg.VAL_TASKS)   # tasks eligible for threshold correction

for budget in _budgets_ordered:
    budget_label = "full" if budget == "full" else str(budget)
    print(f"\n{'='*60}")
    print(f"Budget: {budget_label}")

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

    # Phase 1: mapper warmup
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

    # Phase 2: attnpool fine-tuning
    phase2 = None
    if not cfg.FREEZE_ENCODER:
        print("  Phase 2 — attnpool fine-tuning...")
        gc.collect(); torch.cuda.empty_cache()

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
        agent.precompute_backbone_features(all_refs_list)
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

    agent.eval()
    preds = _collect_preds(agent, [
        ("train",     train_ds_b),
        ("eval",      eval_ds),
        ("stim_gen",  stim_gen_ds),
        ("task_gen",  task_gen_ds),
        ("joint_gen", joint_gen_ds),
    ])

    # Threshold-corrected predictions on val tasks
    if cfg.THRESHOLD_CORRECTION:
        print("  Running h_n threshold-corrected inference...")
        preds["task_gen_h"]  = _collect_h_preds(agent, task_gen_ds,  _h_tasks)
        preds["joint_gen_h"] = _collect_h_preds(agent, joint_gen_ds, _h_tasks)

    metrics = {
        "n_trials":               n_trials_b,
        "n_cells":                n_cells_b,
        "best_epoch":             result.best_epoch,
        "eval_mse":               result.best_val_mse,
        "train_cmse_net":         _region_cmse_net("train",     cfg.TRAIN_TASKS, "train",     preds),
        "stim_gen_cmse_net":      _region_cmse_net("stim_gen",  cfg.TRAIN_TASKS, "stim_gen",  preds),
        "task_gen_cmse_net":      _region_cmse_net("task_gen",  cfg.VAL_TASKS,   "task_gen",  preds),
        "joint_gen_cmse_net":     _region_cmse_net("joint_gen", cfg.VAL_TASKS,   "joint_gen", preds),
        "preds": preds,
        "curves": dict(
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

    if cfg.THRESHOLD_CORRECTION:
        metrics["task_gen_h_cmse_net"]  = _region_cmse_net(
            "task_gen_h",  cfg.VAL_TASKS, "task_gen",  preds)
        metrics["joint_gen_h_cmse_net"] = _region_cmse_net(
            "joint_gen_h", cfg.VAL_TASKS, "joint_gen", preds)

    for k in ["train_cmse_net", "stim_gen_cmse_net",
              "task_gen_cmse_net", "joint_gen_cmse_net"]:
        print(f"  {k}: {metrics[k]:.4f}")
    if cfg.THRESHOLD_CORRECTION:
        print(f"  task_gen_h_cmse_net:  {metrics['task_gen_h_cmse_net']:.4f}")
        print(f"  joint_gen_h_cmse_net: {metrics['joint_gen_h_cmse_net']:.4f}")

    # Lightweight checkpoint (mapper + attnpool only)
    ckpt = {"mapper": agent.mapper.state_dict()}
    if not cfg.FREEZE_ENCODER:
        ckpt["attnpool"] = agent.encoder.attnpool.state_dict()
    ckpt_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_budget_{budget_label}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"  Saved checkpoint -> {ckpt_path}")

    results_per_budget[budget_label] = metrics

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = dict(
    split_mode         = cfg.SPLIT_MODE,
    trial_budgets      = cfg.TRIAL_BUDGETS,
    noise_floors       = noise_floors,
    results            = results_per_budget,
    n_pool             = N_POOL,
    eval_cell_frac     = cfg.EVAL_CELL_FRAC,
    train_tasks        = cfg.TRAIN_TASKS,
    val_tasks          = cfg.VAL_TASKS,
    train_uids         = train_uids,
    test_uids          = test_uids,
    threshold_correction = cfg.THRESHOLD_CORRECTION,
)
out_path = cfg.RESULTS_DIR / f"data_efficiency_{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved -> {out_path}")
