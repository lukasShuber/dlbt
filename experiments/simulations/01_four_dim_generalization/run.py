"""
Simulation 01 — four-dimensional generalization.

Trains DLBT (two-phase if FREEZE_ENCODER=False) and a per-task SLDA baseline
on synthetic behavioral data, then saves all predictions and training curves
to results/ for offline analysis (analysis.py).

Run from repo root:
    python experiments/simulations/01_four_dim_generalization/run.py
"""

import gc
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from tqdm import tqdm

from dlbt.constants import (
    K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.metrics import corrected_mse

import config as cfg

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
# Load stimuli + continuous metadata
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
# Ground-truth Dirichlet observer
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    z = cont_meta[uid]
    p_right  = _sigmoid(cfg.BETA       * (z["x"]            - X_THRESHOLD))
    p_transp = _sigmoid(cfg.BETA       * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(cfg.BETA       * (z["glossiness"]   - GLOSS_THRESH))
    p_large  = _sigmoid(cfg.SCALE_BETA * (z["scale"]        - SCALE_THRESH))

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


def sample_behavior(ref, task, rng) -> tuple:
    alpha   = gt_alpha(ref.uid)
    beliefs = rng.dirichlet(alpha, size=cfg.N_TRIALS)
    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return cfg.N_TRIALS - count_1, count_1

# ---------------------------------------------------------------------------
# Image split — stratified by latent state
# ---------------------------------------------------------------------------
all_uids  = sorted(refs_dict.keys())
rng_split = np.random.default_rng(cfg.SEED)

state_to_uids: dict = defaultdict(list)
for uid in all_uids:
    state_to_uids[refs_dict[uid].latent_state].append(uid)

train_uids: set = set()
test_uids:  set = set()
for state_uids in state_to_uids.values():
    arr    = np.array(state_uids)
    rng_split.shuffle(arr)
    n_test = max(1, round(len(arr) * cfg.IMG_TEST_FRAC))
    test_uids.update(arr[:n_test].tolist())
    train_uids.update(arr[n_test:].tolist())

print(f"Image split: {len(train_uids)} train / {len(test_uids)} test")

# ---------------------------------------------------------------------------
# Synthetic datasets (four evaluation regions)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(cfg.SEED)


def make_dataset(task_names: list, allowed_uids: set) -> BehavioralDataset:
    avail   = [r for r in refs if r.uid in allowed_uids]
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, avail, rng=rng):
            c0, c1 = sample_behavior(ref, task, rng)
            records.append(Observation(
                uid=ref.uid, task_name=task_name, count_0=c0, count_1=c1,
            ))
    return BehavioralDataset.from_records(records)


train_ds     = make_dataset(cfg.TRAIN_TASKS, train_uids)
stim_gen_ds  = make_dataset(cfg.TRAIN_TASKS, test_uids)
task_gen_ds  = make_dataset(cfg.VAL_TASKS,   train_uids)
joint_gen_ds = make_dataset(cfg.VAL_TASKS,   test_uids)

for name, ds in [("train", train_ds), ("stim_gen", stim_gen_ds),
                 ("task_gen", task_gen_ds), ("joint_gen", joint_gen_ds)]:
    print(f"  {name:12s}: {ds}")

# ---------------------------------------------------------------------------
# Train DLBT — phase 1 (mapper warmup, encoder always frozen)
# ---------------------------------------------------------------------------
agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                  mapper_hidden=cfg.MAPPER_HIDDEN)

cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    print(f"Loading CLIP feature cache from {cache_path}")
    agent.load_cache(str(cache_path))
else:
    print(f"Precomputing CLIP features → {cache_path}")
    agent.precompute_features(list(refs_dict.values()))
    agent.save_cache(str(cache_path))

# Snapshot frozen CLIP features — SLDA always uses these, even in attnpool runs,
# so that it is never evaluated on fine-tuned representations.
frozen_clip: dict = {uid: feat.clone() for uid, feat in agent._cache.items()}

print("\nPhase 1 — mapper warmup...")
phase1 = train_dlbt(
    agent, train_ds, stim_gen_ds, refs_dict,
    n_epochs=cfg.N_EPOCHS_PHASE1, lr=cfg.LR, patience=cfg.PATIENCE_PHASE1,
    extra_val_datasets={"task_gen": task_gen_ds, "joint_gen": joint_gen_ds},
)
print(f"  best epoch: {phase1.best_epoch}  stim_gen_mse: {phase1.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Phase 2 — attnpool fine-tuning (only when FREEZE_ENCODER=False)
# ---------------------------------------------------------------------------
phase2 = None
if not cfg.FREEZE_ENCODER:
    print("\nPhase 2 — attnpool fine-tuning...")
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
    print(f"  best epoch: {phase2.best_epoch}  stim_gen_mse: {phase2.best_val_mse:.4f}")

    # Repopulate _cache with final attnpool features for DLBT predictions
    print("Repopulating feature cache...")
    agent.eval()
    all_refs_list = list(refs_dict.values())
    with torch.no_grad():
        for i in tqdm(range(0, len(all_refs_list), 16), desc="caching", unit="batch"):
            batch   = all_refs_list[i : i + 16]
            spatial = torch.stack(
                [agent._backbone_cache[r.uid] for r in batch]
            ).to(agent.device)
            feats = agent.encoder.attnpool(spatial).float()
            for ref, feat in zip(batch, feats):
                agent._cache[ref.uid] = feat.cpu()

result = phase2 if phase2 is not None else phase1
print(f"\nFinal best stim_gen_mse: {result.best_val_mse:.4f}")

# Save agent weights
agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
torch.save(agent.state_dict(), agent_path)
print(f"Saved agent weights → {agent_path}")

# ---------------------------------------------------------------------------
# Fit per-task SLDA  (always on frozen CLIP features)
# ---------------------------------------------------------------------------
def clip_features(uids: list) -> np.ndarray:
    return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])


print("\nFitting SLDA...")
# GT least-squares decoder: W maps CLIP features → one-hot latent states.
# Temperature only is tuned from behavioral data.
_all_refs = list(refs_dict.values())
_X_all    = np.stack([frozen_clip[r.uid].cpu().numpy() for r in _all_refs])
_Y_oh     = np.zeros((len(_all_refs), K), dtype=np.float32)
for _i, _r in enumerate(_all_refs):
    _Y_oh[_i, _r.latent_state] = 1.0
W_slda, _, _, _ = np.linalg.lstsq(_X_all, _Y_oh, rcond=None)  # [1024, K]

slda_temps = {}
for task_name in cfg.TRAIN_TASKS:
    group = train_ds.df[train_ds.df["task_name"] == task_name]
    if len(group) == 0:
        slda_temps[task_name] = 1.0
        continue
    uids    = group["uid"].tolist()
    X       = np.stack([frozen_clip[uid].cpu().numpy() for uid in uids])
    p_right = (group["count_1"] / (group["count_0"] + group["count_1"])).values
    delta_u = TASKS[task_name].delta_u.astype(np.float64)
    logits  = (X @ W_slda) @ delta_u

    def _nll_tau(log_tau, logits=logits, targets=p_right):
        p = 1.0 / (1.0 + np.exp(-logits / np.exp(log_tau)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

    opt = minimize_scalar(_nll_tau, bounds=(-3.0, 3.0), method="bounded")
    slda_temps[task_name] = float(np.exp(opt.x))

print(f"  Fitted SLDA temperatures for {len(slda_temps)} tasks.")


def slda_predict(task_name: str, uids: list) -> np.ndarray:
    X       = np.stack([frozen_clip[uid].cpu().numpy() for uid in uids])
    delta_u = TASKS[task_name].delta_u.astype(np.float64)
    logits  = (X @ W_slda) @ delta_u
    tau     = slda_temps.get(task_name, 1.0)
    return 1.0 / (1.0 + np.exp(-logits / tau))

# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------
agent.eval()
rng_gt   = np.random.default_rng(cfg.SEED + 1)
_gt_cache: dict = {}


def get_true_p(uid: str, task_name: str) -> float:
    key = (uid, task_name)
    if key not in _gt_cache:
        _gt_cache[key] = gt_p_right(uid, TASKS[task_name], n_mc=1000, rng=rng_gt)
    return _gt_cache[key]


def collect_dlbt(ds: BehavioralDataset) -> dict:
    out = {}
    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        true_p     = np.array([get_true_p(r.uid, task_name) for r in batch_refs])
        with torch.no_grad():
            pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
        raw_mse = float(np.mean((pred - true_p) ** 2))
        mc_corr = float(np.mean(pred * (1 - pred))) / (cfg.N_MC - 1)
        rho_val, _ = spearmanr(pred, true_p)
        out[task_name] = dict(pred=pred, true=true_p,
                              cmse=raw_mse - mc_corr, rho=float(rho_val))
    return out


def collect_slda(ds: BehavioralDataset) -> dict:
    out = {}
    for task_name, group in ds.iter_tasks():
        if task_name not in slda_temps:
            continue
        uids   = group["uid"].tolist()
        true_p = np.array([get_true_p(uid, task_name) for uid in uids])
        pred   = slda_predict(task_name, uids)
        raw_mse = float(np.mean((pred - true_p) ** 2))
        rho_val, _ = spearmanr(pred, true_p)
        out[task_name] = dict(pred=pred, true=true_p, cmse=raw_mse, rho=float(rho_val))
    return out


print("\nCollecting predictions...")
dlbt_train  = collect_dlbt(train_ds)
dlbt_stim   = collect_dlbt(stim_gen_ds)
dlbt_task   = collect_dlbt(task_gen_ds)
dlbt_joint  = collect_dlbt(joint_gen_ds)
slda_train  = collect_slda(train_ds)
slda_stim   = collect_slda(stim_gen_ds)

# ---------------------------------------------------------------------------
# Build learning-curve arrays
# ---------------------------------------------------------------------------
n_phase1       = len(phase1.train_nlls)
phase_boundary = n_phase1 - 1


def _concat(p1_list, p2_list):
    if p2_list is None:
        return list(p1_list)
    return list(p1_list) + list(p2_list)[1:]


curves = dict(
    train_nlls  = _concat(phase1.train_nlls,  phase2.train_nlls  if phase2 else None),
    val_nlls    = _concat(phase1.val_nlls,     phase2.val_nlls    if phase2 else None),
    train_mses  = _concat(phase1.train_mses,   phase2.train_mses  if phase2 else None),
    val_mses    = _concat(phase1.val_mses,     phase2.val_mses    if phase2 else None),
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

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = dict(
    # metadata
    model_label       = model_label,
    run_tag           = cfg.RUN_TAG,
    phase_boundary    = phase_boundary,
    best_epoch        = best_epoch_offset,
    noise_floor       = train_ds.noise_floor(),
    # curves
    curves            = curves,
    # predictions
    dlbt_train        = dlbt_train,
    dlbt_stim         = dlbt_stim,
    dlbt_task         = dlbt_task,
    dlbt_joint        = dlbt_joint,
    slda_train        = slda_train,
    slda_stim         = slda_stim,
    # split info (for balance plot)
    state_to_uids     = dict(state_to_uids),
    train_uids        = train_uids,
    test_uids         = test_uids,
)

results_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(results_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results → {results_path}")
