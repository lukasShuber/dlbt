"""
Simulation 011 — emulate behavioural data constraints.

Fork of sim 01 with three surgical changes (marked [011]):
  1. Per-dim sigmoid sharpness (BETA_PER_DIM) in gt_alpha
  2. Per-region trial counts (N_TRIALS_MAIN / N_TRIALS_PROBE)
  3. Explicit probe/main image partition instead of random fraction

Run from repo root:
    python experiments/simulations/011_emp_sim/run.py
"""

import gc
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dlbt.constants import (
    K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

import config as cfg

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

cont_meta: dict = {}
with open(cfg.METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        z   = rec["z"]
        cont_meta[rec["id"]] = dict(
            x            = z["pos_xy"][0],
            transparency = z["transparency"],
            glossiness   = z["glossiness"],
            scale        = z["scale"],
        )

# ---------------------------------------------------------------------------
# Ground-truth Dirichlet observer  [011] per-dim BETA
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    z = cont_meta[uid]
    # [011] per-dim sharpness from BETA_PER_DIM
    p_right  = _sigmoid(cfg.BETA_PER_DIM["lr"] * (z["x"]            - X_THRESHOLD))
    p_transp = _sigmoid(cfg.BETA_PER_DIM["tr"] * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(cfg.BETA_PER_DIM["gl"] * (z["glossiness"]   - GLOSS_THRESH))
    p_large  = _sigmoid(cfg.BETA_PER_DIM["sl"] * (z["scale"]        - SCALE_THRESH))

    q = np.empty(K, dtype=np.float64)
    for k in range(K):
        k_right  = (k >> DIM_LEFT_RIGHT)  & 1
        k_transp = (k >> DIM_TRANSP)      & 1
        k_glossy = (k >> DIM_GLOSS)       & 1
        k_large  = (k >> DIM_SMALL_LARGE) & 1
        q[k] = (
            (p_right  if k_right  else 1.0 - p_right)  *
            (p_transp if k_transp else 1.0 - p_transp) *
            (p_glossy if k_glossy else 1.0 - p_glossy) *
            (p_large  if k_large  else 1.0 - p_large)
        )

    clarity = (abs(p_right  - 0.5) * 2.0 *
               abs(p_transp - 0.5) * 2.0 *
               abs(p_glossy - 0.5) * 2.0 *
               abs(p_large  - 0.5) * 2.0)
    lam = cfg.BASE_CONCENTRATION + cfg.PEAK * clarity
    return 1e-6 + lam * q


def gt_p_right(uid: str, task, n_mc: int = 2000, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(uid)
    beliefs = rng.dirichlet(alpha, size=n_mc)
    return float((beliefs @ task.delta_u > 0).mean())


# [011] n_trials now passed per call (main vs probe)
def sample_behavior(ref, task, rng, n_trials: int) -> tuple:
    alpha   = gt_alpha(ref.uid)
    beliefs = rng.dirichlet(alpha, size=n_trials)
    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return n_trials - count_1, count_1

# ---------------------------------------------------------------------------
# [011] Image split — explicit probe/main partition
# ---------------------------------------------------------------------------
all_uids_sorted = sorted(refs_dict.keys())

probe_uids = set(all_uids_sorted[:cfg.N_PROBE_IMAGES])
main_uids  = set(all_uids_sorted[cfg.N_PROBE_IMAGES:])

train_uids = main_uids
test_uids  = probe_uids

print(f"Image split: {len(train_uids)} main / {len(test_uids)} probe")

# ---------------------------------------------------------------------------
# [011] Synthetic datasets with per-region trial counts
# ---------------------------------------------------------------------------
rng = np.random.default_rng(cfg.SEED)


def make_dataset(task_names: list, allowed_uids: set, n_trials: int) -> BehavioralDataset:
    avail   = [r for r in refs if r.uid in allowed_uids]
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in avail:
            c0, c1 = sample_behavior(ref, task, rng, n_trials)
            records.append(Observation(
                uid=ref.uid, task_name=task_name, count_0=c0, count_1=c1,
            ))
    return BehavioralDataset.from_records(records)


train_ds     = make_dataset(cfg.TRAIN_TASKS, train_uids, cfg.N_TRIALS_MAIN)
stim_gen_ds  = make_dataset(cfg.TRAIN_TASKS, test_uids,  cfg.N_TRIALS_PROBE)
task_gen_ds  = make_dataset(cfg.VAL_TASKS,   train_uids, cfg.N_TRIALS_MAIN)  if cfg.VAL_TASKS else BehavioralDataset.from_records([])
joint_gen_ds = make_dataset(cfg.VAL_TASKS,   test_uids,  cfg.N_TRIALS_PROBE) if cfg.VAL_TASKS else BehavioralDataset.from_records([])

for name, ds in [("train", train_ds), ("stim_gen", stim_gen_ds),
                 ("task_gen", task_gen_ds), ("joint_gen", joint_gen_ds)]:
    print(f"  {name:12s}: {ds}")

# ---------------------------------------------------------------------------
# CLIP feature cache
# ---------------------------------------------------------------------------
_agent_for_cache = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                              mapper_hidden=cfg.MAPPER_HIDDEN)

cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    print(f"Loading CLIP feature cache from {cache_path}")
    _agent_for_cache.load_cache(str(cache_path))
else:
    print(f"Precomputing CLIP features → {cache_path}")
    _agent_for_cache.precompute_features(list(refs_dict.values()))
    _agent_for_cache.save_cache(str(cache_path))

frozen_clip      = {uid: feat.clone() for uid, feat in _agent_for_cache._cache.items()}
frozen_clip_copy = {uid: feat.clone() for uid, feat in frozen_clip.items()}
del _agent_for_cache

# ---------------------------------------------------------------------------
# SLDA baseline (frozen CLIP features)
# ---------------------------------------------------------------------------
def clip_features(uids: list) -> np.ndarray:
    return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])


print("\nFitting SLDA...")
slda_scalers, slda_models, slda_temps = {}, {}, {}

for task_name in cfg.TRAIN_TASKS:
    group = train_ds.df[train_ds.df["task_name"] == task_name]
    if len(group) == 0:
        continue
    uids    = group["uid"].tolist()
    X       = clip_features(uids)
    p_right = (group["count_1"] / (group["count_0"] + group["count_1"])).values

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


def slda_predict(task_name: str, uids: list) -> np.ndarray:
    X        = clip_features(uids)
    X_scaled = slda_scalers[task_name].transform(X)
    p_pred   = np.clip(slda_models[task_name].predict(X_scaled), 1e-6, 1 - 1e-6)
    logits   = np.log(p_pred / (1 - p_pred))
    tau      = slda_temps[task_name]
    return 1.0 / (1.0 + np.exp(-logits / tau))

# ---------------------------------------------------------------------------
# Ground-truth cache
# ---------------------------------------------------------------------------
rng_gt    = np.random.default_rng(cfg.SEED + 1)
_gt_cache: dict = {}


def get_true_p(uid: str, task_name: str) -> float:
    key = (uid, task_name)
    if key not in _gt_cache:
        _gt_cache[key] = gt_p_right(uid, TASKS[task_name], n_mc=1000, rng=rng_gt)
    return _gt_cache[key]

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
        true_p = np.array([get_true_p(uid, task_name) for uid in uids])
        totals = (group["count_0"] + group["count_1"]).values.astype(float)
        emp_p  = (group["count_1"] / totals.clip(min=1)).values
        pred   = slda_predict(task_name, uids)
        slda_preds[cond][task_name] = {
            "pred": pred, "true": true_p, "emp": emp_p, "totals": totals, "uids": uids,
        }

# ---------------------------------------------------------------------------
# DLBT multi-seed training loop
# ---------------------------------------------------------------------------
print(f"\nTraining DLBT with {cfg.N_SEEDS} seeds: {cfg.SEEDS}")

dlbt_preds: dict = {cond: {} for cond in ["train", "stim", "task", "joint"]}

phase1 = phase2 = result = None
curves = None
phase_boundary = best_epoch_offset = 0
agent  = None


def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


def _extra(res, key):
    return res.extra_val_nlls.get(key, []) if res else []


def _extra_mse(res, key):
    return res.extra_val_mses.get(key, []) if res else []


for seed_idx, seed in enumerate(cfg.SEEDS):
    print(f"\n--- Seed {seed_idx + 1}/{cfg.N_SEEDS}  (seed={seed}) ---")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                      mapper_hidden=cfg.MAPPER_HIDDEN)
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip_copy.items()}

    extra_val = ({"task_gen": task_gen_ds, "joint_gen": joint_gen_ds}
                 if cfg.VAL_TASKS else {})

    print("  Phase 1 — mapper warmup...")
    phase1 = train_dlbt(
        agent, train_ds, stim_gen_ds, refs_dict,
        n_epochs=cfg.N_EPOCHS_PHASE1, lr=cfg.LR, patience=cfg.PATIENCE_PHASE1,
        extra_val_datasets=extra_val,
    )
    print(f"    best epoch: {phase1.best_epoch}  stim_gen_mse: {phase1.best_val_mse:.4f}")

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
            extra_val_datasets=extra_val,
        )
        print(f"    best epoch: {phase2.best_epoch}  stim_gen_mse: {phase2.best_val_mse:.4f}")

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

    agent.eval()
    for cond, ds in [("train", train_ds), ("stim", stim_gen_ds),
                     ("task", task_gen_ds), ("joint", joint_gen_ds)]:
        for task_name, group in ds.iter_tasks():
            task       = TASKS[task_name]
            batch_refs = [refs_dict[uid] for uid in group["uid"]]
            true_p     = np.array([get_true_p(r.uid, task_name) for r in batch_refs])
            with torch.no_grad():
                pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()

            totals = (group["count_0"] + group["count_1"]).values.astype(float)
            emp_p  = (group["count_1"] / totals.clip(min=1)).values

            if task_name not in dlbt_preds[cond]:
                dlbt_preds[cond][task_name] = {
                    "pred":   [],
                    "true":   true_p,   # GT probability (continuous)
                    "emp":    emp_p,    # empirical p̂ from simulated trials
                    "totals": totals,
                    "uids":   [r.uid for r in batch_refs],
                }
            dlbt_preds[cond][task_name]["pred"].append(pred)

    n_phase1       = len(phase1.train_nlls)
    phase_boundary = n_phase1 - 1
    curves = dict(
        train_nlls = _concat(phase1.train_nlls, phase2.train_nlls if phase2 else None),
        val_nlls   = _concat(phase1.val_nlls,   phase2.val_nlls   if phase2 else None),
        train_mses = _concat(phase1.train_mses, phase2.train_mses if phase2 else None),
        val_mses   = _concat(phase1.val_mses,   phase2.val_mses   if phase2 else None),
        task_nlls  = _concat(_extra(phase1, "task_gen"),
                             _extra(phase2, "task_gen") if phase2 else None),
        joint_nlls = _concat(_extra(phase1, "joint_gen"),
                             _extra(phase2, "joint_gen") if phase2 else None),
        task_mses  = _concat(_extra_mse(phase1, "task_gen"),
                             _extra_mse(phase2, "task_gen") if phase2 else None),
        joint_mses = _concat(_extra_mse(phase1, "joint_gen"),
                             _extra_mse(phase2, "joint_gen") if phase2 else None),
    )
    best_epoch_offset = result.best_epoch + (phase_boundary if phase2 else 0)

# Stack predictions
for cond in dlbt_preds:
    for task_name in dlbt_preds[cond]:
        dlbt_preds[cond][task_name]["pred"] = np.stack(
            dlbt_preds[cond][task_name]["pred"]
        )

agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
torch.save(agent.state_dict(), agent_path)
print(f"\nSaved best agent weights (last seed) → {agent_path}")

# Also save end-of-training weights
end_state  = result.end_state if (result is not None and result.end_state) else agent.state_dict()
agent_end_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_end.pt"
torch.save(end_state, agent_end_path)
print(f"Saved end agent weights  (last seed) → {agent_end_path}")

# Collect end-agent predictions (last seed only)
print("Collecting end-agent predictions...")
dlbt_preds_end: dict = {cond: {} for cond in ["train", "stim", "task", "joint"]}
agent.load_state_dict(end_state)
agent.eval()
for cond, ds in [("train", train_ds), ("stim", stim_gen_ds),
                 ("task", task_gen_ds), ("joint", joint_gen_ds)]:
    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        true_p     = np.array([get_true_p(r.uid, task_name) for r in batch_refs])
        totals     = (group["count_0"] + group["count_1"]).values.astype(float)
        emp_p      = (group["count_1"] / totals.clip(min=1)).values
        with torch.no_grad():
            pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
        dlbt_preds_end[cond][task_name] = {
            "pred":   pred,
            "true":   true_p,
            "emp":    emp_p,
            "totals": totals,
            "uids":   [r.uid for r in batch_refs],
        }

# ---------------------------------------------------------------------------
# Noise floors per region
# ---------------------------------------------------------------------------
noise_floors = {
    "train":    train_ds.noise_floor(),
    "stim_gen": stim_gen_ds.noise_floor(),
}
if len(task_gen_ds.df)  > 0: noise_floors["task_gen"]  = task_gen_ds.noise_floor()
if len(joint_gen_ds.df) > 0: noise_floors["joint_gen"] = joint_gen_ds.noise_floor()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = dict(
    model_label    = model_label,
    run_tag        = cfg.RUN_TAG,
    n_seeds        = cfg.N_SEEDS,
    seeds          = cfg.SEEDS,
    n_trials_main  = cfg.N_TRIALS_MAIN,
    n_trials_probe = cfg.N_TRIALS_PROBE,
    beta_per_dim   = cfg.BETA_PER_DIM,
    phase_boundary = phase_boundary,
    best_epoch     = best_epoch_offset,
    noise_floors   = noise_floors,
    noise_floor    = train_ds.noise_floor(),   # kept for analysis.py compat
    curves         = curves,
    dlbt           = dlbt_preds,
    slda           = slda_preds,
    train_uids     = train_uids,
    test_uids      = test_uids,
)

results_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(results_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results → {results_path}")

# Separate end-agent pkl
results_end = dict(
    model_label    = f"{model_label} (end)",
    run_tag        = f"{cfg.RUN_TAG}_end",
    n_seeds        = 1,
    seeds          = cfg.SEEDS[-1:],
    n_trials_main  = cfg.N_TRIALS_MAIN,
    n_trials_probe = cfg.N_TRIALS_PROBE,
    beta_per_dim   = cfg.BETA_PER_DIM,
    phase_boundary = 0,
    best_epoch     = 0,
    noise_floors   = noise_floors,
    noise_floor    = train_ds.noise_floor(),
    curves         = curves,
    dlbt           = dlbt_preds_end,
    slda           = slda_preds,
    train_uids     = train_uids,
    test_uids      = test_uids,
)
results_end_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}_end.pkl"
with open(results_end_path, "wb") as f:
    pickle.dump(results_end, f)
print(f"Saved end-agent results → {results_end_path}")
