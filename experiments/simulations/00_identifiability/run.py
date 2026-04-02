"""
Simulation 00 — identifiability.

Generates behavioral data from a ground-truth observer whose alpha parameters
are a linear function of frozen CLIP features — exactly DLBT's hypothesis
class.  Trains DLBT on all images × all tasks (no splits) and checks whether
it recovers the true parameters.

Run from repo root:
    python experiments/simulations/00_identifiability/run.py
"""

import pickle
import random
from pathlib import Path

import numpy as np
import torch

from dlbt.constants import K
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS as ALL_TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

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

# ---------------------------------------------------------------------------
# Load image refs
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Load CLIP cache  (needed before dataset generation)
# ---------------------------------------------------------------------------
_boot = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                  mapper_hidden=cfg.MAPPER_HIDDEN)

cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    print(f"Loading CLIP cache from {cache_path}")
    _boot.load_cache(str(cache_path))
else:
    print(f"Precomputing CLIP features → {cache_path}")
    _boot.precompute_features(list(refs_dict.values()))
    _boot.save_cache(str(cache_path))

frozen_clip: dict = {uid: feat.clone() for uid, feat in _boot._cache.items()}
frozen_clip_copy  = {uid: feat.clone() for uid, feat in frozen_clip.items()}
D_clip = next(iter(frozen_clip.values())).shape[0]
print(f"CLIP feature dim: {D_clip}")
del _boot

# ---------------------------------------------------------------------------
# Ground-truth linear observer
# alpha*(uid) = softplus(W* @ clip(uid) + b*) + floor
# Same functional form as DLBT's mapper → target is realizable.
# ---------------------------------------------------------------------------
_rng_gt = np.random.default_rng(cfg.GT_SEED)
W_star  = _rng_gt.normal(0.0, cfg.ALPHA_SCALE, size=(K, D_clip)).astype(np.float32)
b_star  = _rng_gt.normal(0.0, 0.1,             size=(K,)).astype(np.float32)
print(f"Sampled W* ({K}×{D_clip}), b* ({K},)  [GT_SEED={cfg.GT_SEED}]")


def gt_alpha(uid: str) -> np.ndarray:
    feat = frozen_clip[uid].cpu().numpy().astype(np.float32)
    raw  = W_star @ feat + b_star
    return np.log1p(np.exp(raw)).astype(np.float64) + 1e-6   # softplus + floor


def gt_p_right(uid: str, task, n_mc: int = 2000, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(uid)
    beliefs = rng.dirichlet(alpha, size=n_mc)
    return float((beliefs @ task.delta_u > 0).mean())


def sample_behavior(ref, task, rng):
    alpha   = gt_alpha(ref.uid)
    beliefs = rng.dirichlet(alpha, size=cfg.N_TRIALS)
    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return cfg.N_TRIALS - count_1, count_1


# ---------------------------------------------------------------------------
# Single dataset — all images × all tasks
# ---------------------------------------------------------------------------
rng_data = np.random.default_rng(cfg.SEED)
records  = []
for task_name in cfg.TASKS:
    task = ALL_TASKS[task_name]
    for ref in refs:
        c0, c1 = sample_behavior(ref, task, rng_data)
        records.append(Observation(uid=ref.uid, task_name=task_name,
                                   count_0=c0, count_1=c1))
ds = BehavioralDataset.from_records(records)
print(f"Dataset: {ds}")

# ---------------------------------------------------------------------------
# Ground-truth cache (true P(right) for every image × task)
# ---------------------------------------------------------------------------
rng_gt_p = np.random.default_rng(cfg.SEED + 1)
_gt_cache: dict = {}


def get_true_p(uid: str, task_name: str) -> float:
    key = (uid, task_name)
    if key not in _gt_cache:
        _gt_cache[key] = gt_p_right(uid, ALL_TASKS[task_name], n_mc=2000, rng=rng_gt_p)
    return _gt_cache[key]


# Pre-compute true alpha for all images
true_alphas = {uid: gt_alpha(uid) for uid in refs_dict}

# ---------------------------------------------------------------------------
# Multi-seed DLBT training
# ---------------------------------------------------------------------------
print(f"\nTraining DLBT ({cfg.N_SEEDS} seeds)...")

# {task_name: {pred: list, true: array, uids: list}}
dlbt_preds: dict = {}
# {uid: list of [K] arrays, one per seed}
alpha_preds: dict = {uid: [] for uid in refs_dict}

curves       = None
phase1       = None


def _make_agent():
    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=device,
                      mapper_hidden=cfg.MAPPER_HIDDEN)
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip_copy.items()}
    return agent


for seed_idx, seed in enumerate(cfg.SEEDS):
    print(f"\n--- Seed {seed_idx+1}/{cfg.N_SEEDS}  (seed={seed}) ---")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = _make_agent()

    # Use the same dataset for train and val — we care about fitting, not gen
    phase1 = train_dlbt(
        agent, ds, ds, refs_dict,
        n_epochs=cfg.N_EPOCHS, lr=cfg.LR, patience=cfg.PATIENCE,
    )
    print(f"  best epoch: {phase1.best_epoch}  train_mse: {phase1.best_val_mse:.5f}")
    curves = dict(train_nlls=phase1.train_nlls, val_nlls=phase1.val_nlls,
                  train_mses=phase1.train_mses, val_mses=phase1.val_mses)

    # -- Collect behavioral predictions --
    agent.eval()
    for task_name, group in ds.iter_tasks():
        task       = ALL_TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        true_p     = np.array([get_true_p(r.uid, task_name) for r in batch_refs])
        with torch.no_grad():
            pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()

        if task_name not in dlbt_preds:
            dlbt_preds[task_name] = {"pred": [], "true": true_p,
                                     "uids": [r.uid for r in batch_refs]}
        dlbt_preds[task_name]["pred"].append(pred)

    # -- Collect alpha predictions --
    with torch.no_grad():
        for uid, ref in refs_dict.items():
            alpha_preds[uid].append(agent.get_alpha([ref])[0].cpu().numpy())

# Stack to [n_seeds, n_pts]
for tn in dlbt_preds:
    dlbt_preds[tn]["pred"] = np.stack(dlbt_preds[tn]["pred"])
alpha_preds = {uid: np.stack(v) for uid, v in alpha_preds.items()}

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results = dict(
    n_seeds     = cfg.N_SEEDS,
    seeds       = cfg.SEEDS,
    n_trials    = cfg.N_TRIALS,
    best_epoch  = phase1.best_epoch,
    noise_floor = ds.noise_floor(),
    curves      = curves,
    dlbt        = dlbt_preds,    # {task: {pred:[n_seeds,n_pts], true, uids}}
    W_star      = W_star,
    b_star      = b_star,
    true_alphas = true_alphas,   # {uid: [K]}
    alpha_preds = alpha_preds,   # {uid: [n_seeds, K]}
)

path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved → {path}")
