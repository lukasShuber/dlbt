"""
Behavior run0 — DLBT + SLDA on real human data (pendant to simulation 01).

Identical training loop to simulations/01_four_dim_generalization/run.py,
but with two key differences:

  - Behavioural counts come from the filtered AMT data rather than a
    synthetic Dirichlet observer.
  - The "true" P(right) used for evaluation is the empirical proportion
    count_1 / (count_0 + count_1) observed in the data (not an oracle).

Run from repo root:
    python experiments/behavior/run0/run.py
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
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

# config.py and preprocess.py live next to this script
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
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

model_label = "DLBT (frozen)" if cfg.FREEZE_ENCODER else "DLBT (attnpool)"

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
print(f"  Raw trials:             {diag['n_raw_trials']:>7d}")
print(f"  After QC filtering:     {diag['n_filtered_trials']:>7d}")
print(f"  Assignments:            {diag['n_pass_both']} / {diag['n_total_assignments']}"
      f"  (catch≥{cfg.MIN_CATCH_PERF:.2f}  &  main≥{diag['main_perf_threshold']:.2f})")
print(f"  Aggregated cells:       {diag['n_cells']:>7d}")
print(f"  Unique images:          {diag['n_unique_images']:>7d}"
      f"   ({diag['n_main_uids']} main + {diag['n_probe_uids']} probe)")
print(f"  Unique tasks:           {diag['n_unique_tasks']:>7d}")
print(f"  Mean trials/cell:       {diag['trials_per_cell_mean']:.2f}"
      f"   (main={diag.get('trials_per_cell_main', 0):.2f}"
      f", probe={diag.get('trials_per_cell_probe', 0):.2f})")

# Sanity check: every task in the config should have data
missing = [t for t in (cfg.TRAIN_TASKS + cfg.VAL_TASKS)
           if t not in full_ds.df["task_name"].unique()]
if missing:
    raise ValueError(f"Tasks missing from behavioural data: {missing}")

# ---------------------------------------------------------------------------
# Image split: probe images (high-rep eval) vs main images (bulk training)
# ---------------------------------------------------------------------------
# The behavioural experiment collected ~20 trials per (uid, task) on 16 probe
# images and ~3 trials per (uid, task) on ~980 main images.  We use probe
# images as the held-out evaluation set (clean, low-noise targets) and main
# images as the training set.  The two UID sets are disjoint by design.
train_uids = set(main_uids)
test_uids  = set(probe_uids)
assert not (train_uids & test_uids), "probe and main UIDs should be disjoint"
print(f"\nImage split: {len(train_uids)} main (train) / {len(test_uids)} probe (eval)")

# ---------------------------------------------------------------------------
# Build the 4 evaluation datasets by slicing full_ds
# ---------------------------------------------------------------------------
def _slice(ds: BehavioralDataset, task_names, uids) -> BehavioralDataset:
    sub = ds.df[
        ds.df["task_name"].isin(task_names) & ds.df["uid"].isin(uids)
    ].copy()
    return BehavioralDataset(sub)

train_ds     = _slice(full_ds, cfg.TRAIN_TASKS, train_uids)
stim_gen_ds  = _slice(full_ds, cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = _slice(full_ds, cfg.VAL_TASKS,   train_uids)
joint_gen_ds = _slice(full_ds, cfg.VAL_TASKS,   test_uids)

for name, ds in [("train", train_ds), ("stim_gen", stim_gen_ds),
                 ("task_gen", task_gen_ds), ("joint_gen", joint_gen_ds)]:
    print(f"  {name:12s}: {ds}")

# ---------------------------------------------------------------------------
# CLIP feature cache (shared, precomputed once)
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
# Empirical "truth" — count_1 / total per (uid, task_name)
# ---------------------------------------------------------------------------
# Used as the regression target for SLDA and the y-axis in pred-vs-true plots.
# Build a lookup dict for quick access.
_emp_lookup: dict = {}
for row in full_ds.df.itertuples(index=False):
    total = row.count_0 + row.count_1
    p = row.count_1 / total if total > 0 else np.nan
    _emp_lookup[(row.uid, row.task_name)] = (p, total)


def emp_p(uid: str, task_name: str) -> float:
    """Empirical P(right) for a given (image, task) cell. NaN if no data."""
    v = _emp_lookup.get((uid, task_name))
    return v[0] if v is not None else np.nan


def emp_n(uid: str, task_name: str) -> int:
    """Total trials for a given (image, task) cell. 0 if no data."""
    v = _emp_lookup.get((uid, task_name))
    return v[1] if v is not None else 0


# ---------------------------------------------------------------------------
# SLDA (fitted once on TRAIN cells, same recipe as sim 01)
# ---------------------------------------------------------------------------
def clip_features(uids) -> np.ndarray:
    return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])


print("\nFitting SLDA...")
slda_scalers, slda_models, slda_temps = {}, {}, {}

for task_name in cfg.TRAIN_TASKS:
    group = train_ds.df[train_ds.df["task_name"] == task_name]
    if len(group) == 0:
        continue
    uids    = group["uid"].tolist()
    X       = clip_features(uids)
    totals  = (group["count_0"] + group["count_1"]).values.astype(float)
    p_right = (group["count_1"].values / np.clip(totals, 1, None))

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
# SLDA predictions
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
print(f"\nTraining DLBT with {cfg.N_SEEDS} seeds: {cfg.SEEDS}")

dlbt_preds: dict = {cond: {} for cond in ["train", "stim", "task", "joint"]}

phase1 = phase2 = result = None
curves = None
phase_boundary = best_epoch_offset = 0
agent = None


def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


for seed_idx, seed in enumerate(cfg.SEEDS):
    print(f"\n--- Seed {seed_idx + 1}/{cfg.N_SEEDS}  (seed={seed}) ---")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                      mapper_hidden=cfg.MAPPER_HIDDEN)
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip_copy.items()}

    # -- Phase 1: mapper warmup --
    print("  Phase 1 — mapper warmup...")
    phase1 = train_dlbt(
        agent, train_ds, stim_gen_ds, refs_dict,
        n_epochs=cfg.N_EPOCHS_PHASE1, lr=cfg.LR, patience=cfg.PATIENCE_PHASE1,
        extra_val_datasets={"task_gen": task_gen_ds, "joint_gen": joint_gen_ds},
    )
    print(f"    best epoch: {phase1.best_epoch}  stim_gen_mse: {phase1.best_val_mse:.4f}")

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
            agent, train_ds, stim_gen_ds, refs_dict,
            n_epochs=cfg.N_EPOCHS_PHASE2, patience=cfg.PATIENCE_PHASE2,
            optimizer=optimizer2,
            extra_val_datasets={"task_gen": task_gen_ds, "joint_gen": joint_gen_ds},
        )
        print(f"    best epoch: {phase2.best_epoch}  stim_gen_mse: {phase2.best_val_mse:.4f}")

        # Repopulate _cache with final attnpool features
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
    print(f"  Final best stim_gen_mse: {result.best_val_mse:.4f}")

    # -- Collect DLBT predictions for this seed --
    agent.eval()
    for cond, ds in [("train", train_ds), ("stim", stim_gen_ds),
                     ("task", task_gen_ds), ("joint", joint_gen_ds)]:
        for task_name, group in ds.iter_tasks():
            task       = TASKS[task_name]
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([emp_p(r.uid, task_name) for r in batch_refs])
            totals     = np.array([emp_n(r.uid, task_name) for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()

            if task_name not in dlbt_preds[cond]:
                dlbt_preds[cond][task_name] = {
                    "pred":   [],
                    "true":   true_p,
                    "totals": totals,
                    "uids":   [r.uid for r in batch_refs],
                }
            dlbt_preds[cond][task_name]["pred"].append(pred)

    # -- Learning curves (overwritten each seed; last seed kept) --
    n_phase1       = len(phase1.train_nlls)
    phase_boundary = n_phase1 - 1
    curves = dict(
        train_nlls  = _concat(phase1.train_nlls,  phase2.train_nlls  if phase2 else None),
        val_nlls    = _concat(phase1.val_nlls,    phase2.val_nlls    if phase2 else None),
        train_mses  = _concat(phase1.train_mses,  phase2.train_mses  if phase2 else None),
        val_mses    = _concat(phase1.val_mses,    phase2.val_mses    if phase2 else None),
        task_nlls   = _concat(phase1.extra_val_nlls["task_gen"],
                              phase2.extra_val_nlls["task_gen"]  if phase2 else None),
        joint_nlls  = _concat(phase1.extra_val_nlls["joint_gen"],
                              phase2.extra_val_nlls["joint_gen"] if phase2 else None),
        task_mses   = _concat(phase1.extra_val_mses["task_gen"],
                              phase2.extra_val_mses["task_gen"]  if phase2 else None),
        joint_mses  = _concat(phase1.extra_val_mses["joint_gen"],
                              phase2.extra_val_mses["joint_gen"] if phase2 else None),
    )
    best_epoch_offset = result.best_epoch + (phase_boundary if phase2 else 0)

# Stack per-seed predictions -> [n_seeds, n_pts]
for cond in dlbt_preds:
    for task_name in dlbt_preds[cond]:
        dlbt_preds[cond][task_name]["pred"] = np.stack(
            dlbt_preds[cond][task_name]["pred"]
        )

# Save agent weights
agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
torch.save(agent.state_dict(), agent_path)
print(f"\nSaved agent weights (last seed) -> {agent_path}")

# ---------------------------------------------------------------------------
# Noise floors per evaluation region (for plotting reference lines)
# ---------------------------------------------------------------------------
noise_floors = {
    "train":     train_ds.noise_floor(),
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
    noise_floor    = noise_floors["train"],   # for sim-01-style plot_02
    noise_floors   = noise_floors,            # per-region, for the real-data plots
    curves         = curves,
    dlbt           = dlbt_preds,
    slda           = slda_preds,
    train_uids     = train_uids,     # = main UIDs
    test_uids      = test_uids,      # = probe UIDs
    main_uids      = main_uids,
    probe_uids     = probe_uids,
    diag           = diag,
)

results_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(results_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results -> {results_path}")
