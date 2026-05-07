"""
run1/01_fit/run.py — DLBT fit on combined run0 + run1 behavioural data.

Key differences from run0/01_fit/run.py:
  1. Concatenates two raw CSVs (run0 + run1) before preprocessing.
  2. TRAIN_TASKS / VAL_TASKS come from a seeded 80/20 random task split
     computed in config.py at import time.
  3. No threshold correction.
  4. Uses get_task() for all task lookups so that new run1 tasks (not in
     the original TASKS registry) are built on-the-fly from their names.

Run from repo root:
    python experiments/behavior/run1/01_fit/run.py
"""

import gc
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
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

model_label = "DLBT (frozen)" if cfg.FREEZE_ENCODER else "DLBT (attnpool)"

print(f"\nTask split: {len(cfg.TRAIN_TASKS)} train / {len(cfg.VAL_TASKS)} val")
print(f"  TRAIN: {cfg.TRAIN_TASKS}")
print(f"  VAL:   {cfg.VAL_TASKS}")

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"\nLoaded {len(refs)} images.")

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

# Only pass eligible task IDs to aggregate_counts so that ineligible tasks
# (those below MIN_TASK_ASSIGNMENTS) are excluded from full_ds, the
# probe/main image split, and all diagnostics — not just from the sliced
# training / evaluation datasets.
_eligible_names = set(cfg.TRAIN_TASKS + cfg.VAL_TASKS)
_eligible_beh_id = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in _eligible_names}

full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _eligible_beh_id,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)

# Diagnostics
diag["n_raw_trials"]      = len(df_raw)
diag["n_filtered_trials"] = len(df_filtered)
diag["n_cells"]           = len(full_ds)
diag["n_unique_images"]   = full_ds.df["uid"].nunique()
diag["n_unique_tasks"]    = full_ds.df["task_name"].nunique()
diag["n_probe_uids"]      = len(probe_uids)
diag["n_main_uids"]       = len(main_uids)
in_probe = full_ds.df["uid"].isin(probe_uids)
diag["trials_per_cell_mean"]  = float((full_ds.df["count_0"] + full_ds.df["count_1"]).mean())
diag["trials_per_cell_probe"] = float((full_ds.df[in_probe]["count_0"] + full_ds.df[in_probe]["count_1"]).mean()) if in_probe.any() else 0.0
diag["trials_per_cell_main"]  = float((full_ds.df[~in_probe]["count_0"] + full_ds.df[~in_probe]["count_1"]).mean()) if (~in_probe).any() else 0.0

print(f"  After QC filtering:  {diag['n_filtered_trials']:>7,}")
print(f"  Assignments passed:  {diag['n_pass_both']} / {diag['n_total_assignments']}"
      f"  (catch≥{cfg.MIN_CATCH_PERF:.2f}  &  main≥{diag['main_perf_threshold']:.2f})")
print(f"  Aggregated cells:    {diag['n_cells']:>7,}")
print(f"  Unique images:       {diag['n_unique_images']:>7,}"
      f"  ({diag['n_main_uids']} main + {diag['n_probe_uids']} probe)")
print(f"  Unique tasks:        {diag['n_unique_tasks']:>7,}")
print(f"  Trials/cell (mean):  {diag['trials_per_cell_mean']:.2f}"
      f"  (main={diag['trials_per_cell_main']:.2f}"
      f", probe={diag['trials_per_cell_probe']:.2f})")

# Sanity check
missing = [t for t in (cfg.TRAIN_TASKS + cfg.VAL_TASKS)
           if t not in full_ds.df["task_name"].unique()]
if missing:
    raise ValueError(f"Tasks missing from behavioural data: {missing}")

# ---------------------------------------------------------------------------
# Image split: probe (eval only) vs main (train + in-dist eval split)
# ---------------------------------------------------------------------------
train_uids = set(main_uids)
test_uids  = set(probe_uids)
assert not (train_uids & test_uids), "probe and main UIDs should be disjoint"
print(f"\nImage split: {len(train_uids)} main / {len(test_uids)} probe")

# ---------------------------------------------------------------------------
# Cell-level eval split — 10% of (main × TRAIN_TASKS) cells
# ---------------------------------------------------------------------------
main_train_mask = (
    full_ds.df["uid"].isin(main_uids) &
    full_ds.df["task_name"].isin(cfg.TRAIN_TASKS)
)
main_train_df = full_ds.df[main_train_mask].copy()

rng_split = np.random.default_rng(cfg.SEED)
n_eval    = max(1, int(len(main_train_df) * cfg.EVAL_CELL_FRAC))
eval_idx  = rng_split.choice(len(main_train_df), size=n_eval, replace=False)
eval_mask = np.zeros(len(main_train_df), dtype=bool)
eval_mask[eval_idx] = True

train_ds = BehavioralDataset(main_train_df[~eval_mask].reset_index(drop=True))
eval_ds  = BehavioralDataset(main_train_df[eval_mask].reset_index(drop=True))

print(f"\nCell split (main × TRAIN_TASKS): "
      f"{len(train_ds)} train  /  {len(eval_ds)} eval  "
      f"(total {len(main_train_df)})")

def _slice(ds: BehavioralDataset, task_names, uids) -> BehavioralDataset:
    sub = ds.df[
        ds.df["task_name"].isin(task_names) & ds.df["uid"].isin(uids)
    ].copy()
    return BehavioralDataset(sub)

stim_gen_ds  = _slice(full_ds, cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = _slice(full_ds, cfg.VAL_TASKS,   train_uids)
joint_gen_ds = _slice(full_ds, cfg.VAL_TASKS,   test_uids)

for name, ds in [("train",    train_ds),    ("eval",      eval_ds),
                 ("stim_gen", stim_gen_ds),  ("task_gen",  task_gen_ds),
                 ("joint_gen",joint_gen_ds)]:
    print(f"  {name:12s}: {ds}")

# ---------------------------------------------------------------------------
# CLIP feature cache
# ---------------------------------------------------------------------------
_agent_for_cache = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                             device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    print(f"\nLoading CLIP feature cache from {cache_path}")
    _agent_for_cache.load_cache(str(cache_path))
else:
    print(f"\nPrecomputing CLIP features -> {cache_path}")
    _agent_for_cache.precompute_features(list(refs_dict.values()))
    _agent_for_cache.save_cache(str(cache_path))

frozen_clip      = {uid: feat.clone() for uid, feat in _agent_for_cache._cache.items()}
frozen_clip_copy = {uid: feat.clone() for uid, feat in frozen_clip.items()}
del _agent_for_cache

# ---------------------------------------------------------------------------
# Empirical truth
# ---------------------------------------------------------------------------
_emp_lookup: dict = {}
for row in full_ds.df.itertuples(index=False):
    total = row.count_0 + row.count_1
    p = row.count_1 / total if total > 0 else np.nan
    _emp_lookup[(row.uid, row.task_name)] = (p, total)

def emp_p(uid: str, task_name: str) -> float:
    v = _emp_lookup.get((uid, task_name))
    return v[0] if v is not None else np.nan

def emp_n(uid: str, task_name: str) -> int:
    v = _emp_lookup.get((uid, task_name))
    return v[1] if v is not None else 0

# ---------------------------------------------------------------------------
# SLDA (fitted on train_ds cells, one model per TRAIN task)
# ---------------------------------------------------------------------------
def clip_features(uids) -> np.ndarray:
    return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])

print(f"\nFitting SLDA on {len(cfg.TRAIN_TASKS)} train tasks...")
slda_scalers, slda_models, slda_temps = {}, {}, {}

for task_name in cfg.TRAIN_TASKS:
    group = train_ds.df[train_ds.df["task_name"] == task_name]
    if len(group) == 0:
        continue
    uids    = group["uid"].tolist()
    X       = clip_features(uids)
    totals  = (group["count_0"] + group["count_1"]).values.astype(float)
    p_right = group["count_1"].values / np.clip(totals, 1, None)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
    model.fit(X_scaled, p_right)

    p_pred = np.clip(model.predict(X_scaled), 1e-6, 1 - 1e-6)
    logits = np.log(p_pred / (1 - p_pred))

    def _nll_tau(log_tau, logits=logits, targets=p_right):
        p = 1.0 / (1.0 + np.exp(-logits / np.exp(log_tau)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

    opt = minimize_scalar(_nll_tau, bounds=(-3.0, 3.0), method="bounded")
    slda_scalers[task_name] = scaler
    slda_models[task_name]  = model
    slda_temps[task_name]   = float(np.exp(opt.x))

print(f"  Fitted {len(slda_models)} SLDA models.")

def slda_predict(task_name: str, uids) -> np.ndarray:
    X        = clip_features(uids)
    X_scaled = slda_scalers[task_name].transform(X)
    p_pred   = np.clip(slda_models[task_name].predict(X_scaled), 1e-6, 1 - 1e-6)
    logits   = np.log(p_pred / (1 - p_pred))
    tau      = slda_temps[task_name]
    return 1.0 / (1.0 + np.exp(-logits / tau))

# ---------------------------------------------------------------------------
# SLDA predictions (train + stim_gen only — covers TRAIN_TASKS)
# ---------------------------------------------------------------------------
print("\nCollecting SLDA predictions...")
slda_preds: dict = {cond: {} for cond in ["train", "stim"]}

for cond, ds in [("train", train_ds), ("stim", stim_gen_ds)]:
    for task_name, group in ds.iter_tasks():
        if task_name not in slda_models:
            continue
        uids   = group["uid"].tolist()
        true_p = np.array([emp_p(u, task_name) for u in uids])
        totals = np.array([emp_n(u, task_name) for u in uids])
        pred   = slda_predict(task_name, uids)
        slda_preds[cond][task_name] = {
            "pred": pred, "true": true_p, "totals": totals, "uids": uids,
        }

# ---------------------------------------------------------------------------
# Multi-seed DLBT training loop
# ---------------------------------------------------------------------------
print(f"\nTraining DLBT ({model_label}) — {cfg.N_SEEDS} seed(s)")
print(f"Early stopping on eval_ds ({len(eval_ds)} cells, main images)")

dlbt_preds:     dict = {cond: {} for cond in ["train", "eval", "stim", "task", "joint"]}
dlbt_preds_end: dict = {cond: {} for cond in ["train", "eval", "stim", "task", "joint"]}

phase1 = phase2 = result = None
curves = None
phase_boundary = best_epoch_offset = 0
agent  = None


def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


def _collect_preds(target_dict, ds_list):
    """Collect predictions for all (cond, ds) pairs into target_dict."""
    agent.eval()
    for cond, ds in ds_list:
        for task_name, group in ds.iter_tasks():
            task       = get_task(task_name)
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
            totals     = np.array([emp_n(r.uid, task_name) for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
            if task_name not in target_dict[cond]:
                target_dict[cond][task_name] = {
                    "pred": [], "true": true_p, "totals": totals,
                    "uids": [r.uid for r in batch_refs],
                }
            target_dict[cond][task_name]["pred"].append(pred)


cond_ds_list = [
    ("train", train_ds),
    ("eval",  eval_ds),
    ("stim",  stim_gen_ds),
    ("task",  task_gen_ds),
    ("joint", joint_gen_ds),
]

for seed_idx, seed in enumerate(cfg.SEEDS):
    print(f"\n--- Seed {seed_idx + 1}/{cfg.N_SEEDS}  (seed={seed}) ---")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                      device=device, mapper_hidden=cfg.MAPPER_HIDDEN,
                      normalize_utility=cfg.NORMALIZED_UTILITY)
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip_copy.items()}

    # -- Mapper initialisation --
    # Get the last Linear layer before Softplus (mapper[0] or mapper[2] with hidden).
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    if cfg.INIT_MODE == "uniform":
        _bias_val = float(np.log(np.exp(cfg.INIT_ALPHA) - 1.0))  # softplus_inv
        with torch.no_grad():
            _linear.bias.fill_(_bias_val)
    elif cfg.INIT_MODE == "random":
        _init_rng  = np.random.default_rng(cfg.INIT_SEED)
        _alpha_rnd = _init_rng.uniform(
            cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
            size=(_linear.bias.shape[0],),
        ).astype(np.float32)
        _bias_init = np.log(np.exp(_alpha_rnd) - 1.0)  # softplus_inv
        with torch.no_grad():
            _linear.bias.copy_(torch.from_numpy(_bias_init).to(device))
    else:
        raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")

    # -- Phase 1: mapper warmup --
    print("  Phase 1 — mapper warmup...")
    phase1 = train_dlbt(
        agent, train_ds, eval_ds, refs_dict,
        n_epochs  = cfg.N_EPOCHS_PHASE1,
        lr        = cfg.LR,
        patience  = cfg.PATIENCE_PHASE1,
    )
    print(f"    best epoch: {phase1.best_epoch}  eval_mse: {phase1.best_val_mse:.4f}")

    # -- Phase 2: attnpool fine-tuning --
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
            agent, train_ds, eval_ds, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            optimizer = optimizer2,
        )
        print(f"    best epoch: {phase2.best_epoch}  eval_mse: {phase2.best_val_mse:.4f}")

        # Repopulate feature cache with final attnpool features
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
    print(f"  Final best eval_mse: {result.best_val_mse:.4f}")

    # -- Best-checkpoint predictions --
    _collect_preds(dlbt_preds, cond_ds_list)

    # -- End-of-training predictions --
    if result is not None and result.end_state:
        best_state_backup = {k: v.clone() for k, v in agent.state_dict().items()}
        agent.load_state_dict(result.end_state)
        _collect_preds(dlbt_preds_end, cond_ds_list)
        agent.load_state_dict(best_state_backup)

    # -- Learning curves --
    n_phase1       = len(phase1.train_nlls)
    phase_boundary = n_phase1 - 1
    curves = dict(
        train_nlls  = _concat(phase1.train_nlls,  phase2.train_nlls  if phase2 else None),
        eval_nlls   = _concat(phase1.val_nlls,    phase2.val_nlls    if phase2 else None),
        train_mses  = _concat(phase1.train_mses, phase2.train_mses if phase2 else None),
        eval_mses   = _concat(phase1.val_mses,   phase2.val_mses   if phase2 else None),
    )
    best_epoch_offset = result.best_epoch + (phase_boundary if phase2 else 0)

# Stack per-seed predictions
for preds_dict in [dlbt_preds, dlbt_preds_end]:
    for cond in preds_dict:
        for task_name in preds_dict[cond]:
            preds_dict[cond][task_name]["pred"] = np.stack(
                preds_dict[cond][task_name]["pred"]
            )

# ---------------------------------------------------------------------------
# Save agent weights
# ---------------------------------------------------------------------------
agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
torch.save(agent.state_dict(), agent_path)
print(f"\nSaved best agent weights -> {agent_path}")

end_state      = result.end_state if (result is not None and result.end_state) else agent.state_dict()
agent_end_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_end.pt"
torch.save(end_state, agent_end_path)
print(f"Saved end  agent weights -> {agent_end_path}")

# ---------------------------------------------------------------------------
# Noise floors
# ---------------------------------------------------------------------------
noise_floors = {
    "train":     train_ds.noise_floor(),
    "eval":      eval_ds.noise_floor(),
    "stim_gen":  stim_gen_ds.noise_floor(),
    "task_gen":  task_gen_ds.noise_floor(),
    "joint_gen": joint_gen_ds.noise_floor(),
}
print(f"\nNoise floors: {noise_floors}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = dict(
    model_label    = model_label,
    run_tag        = cfg.RUN_TAG,
    n_seeds        = cfg.N_SEEDS,
    seeds          = cfg.SEEDS,
    phase_boundary = phase_boundary,
    best_epoch     = best_epoch_offset,
    noise_floors   = noise_floors,
    curves         = curves,
    dlbt           = dlbt_preds,
    slda           = slda_preds,
    train_tasks    = cfg.TRAIN_TASKS,
    val_tasks      = cfg.VAL_TASKS,
    train_uids     = train_uids,
    test_uids      = test_uids,
    main_uids      = main_uids,
    probe_uids     = probe_uids,
    eval_uids      = set(eval_ds.df["uid"].unique()),
    diag           = diag,
    eval_cell_frac = cfg.EVAL_CELL_FRAC,
    split_seed     = cfg.SPLIT_SEED,
)

results_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(results_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results -> {results_path}")

results_end = {**results, "dlbt": dlbt_preds_end}
results_end_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}_end.pkl"
with open(results_end_path, "wb") as f:
    pickle.dump(results_end, f)
print(f"Saved end results -> {results_end_path}")
