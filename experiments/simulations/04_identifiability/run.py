"""
Simulation 04 — identifiability.

Tests whether the DLBT mapper can recover the ground-truth Dirichlet observer
from behavioral data alone, bypassing the CLIP encoder entirely.

Two oracle feature modes:
  onehot  — K=16 one-hot state vector: strict identifiability.
  latent  — 4D continuous GT latent coords, linear mapper: expressivity test.

No train/test split. All stimuli and all tasks used for fitting and evaluation.
Main output: recovered Dirichlet means vs GT means per (stimulus, state) pair.

Run from repo root:
    python experiments/simulations/04_identifiability/run.py
"""

import gc
import json
import pickle

import numpy as np
import torch
from scipy.stats import spearmanr

from dlbt.constants import (
    K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

import config as cfg

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

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


# ---------------------------------------------------------------------------
# Oracle feature functions
# ---------------------------------------------------------------------------
def make_onehot(uid: str) -> np.ndarray:
    """K=16 one-hot encoding of the discrete latent state."""
    feat          = np.zeros(K, dtype=np.float32)
    feat[refs_dict[uid].latent_state] = 1.0
    return feat


def make_latent(uid: str) -> np.ndarray:
    """4D continuous GT latent coordinates."""
    z = cont_meta[uid]
    return np.array(
        [z["x"], z["transparency"], z["glossiness"], z["scale"]],
        dtype=np.float32,
    )


FEATURE_FNS = {"onehot": make_onehot, "latent": make_latent}

# ---------------------------------------------------------------------------
# Behavioral dataset — all images, all tasks
# ---------------------------------------------------------------------------
print("Generating behavioral dataset...")
_rng = np.random.default_rng(42)

records = []
for task_name in cfg.ALL_TASKS:
    task = TASKS[task_name]
    for ref in balanced_refs(task, refs, rng=_rng):
        alpha   = gt_alpha(ref.uid)
        beliefs = _rng.dirichlet(alpha, size=cfg.N_TRIALS)
        count_1 = int((beliefs @ task.delta_u > 0).sum())
        records.append(Observation(
            uid=ref.uid, task_name=task_name,
            count_0=cfg.N_TRIALS - count_1, count_1=count_1,
        ))

ds = BehavioralDataset.from_records(records)
print(f"  {len(ds.df)} (image, task) pairs  ·  "
      f"{ds.df['task_name'].nunique()} tasks  ·  "
      f"{ds.df['uid'].nunique()} images")

# Pre-compute GT alpha and means for all stimuli (fixed, reused across modes)
all_refs  = list(refs_dict.values())
alpha_gt  = np.array([gt_alpha(r.uid) for r in all_refs])          # [N, K]
q_gt      = alpha_gt / alpha_gt.sum(axis=1, keepdims=True)         # [N, K]
states    = np.array([refs_dict[r.uid].latent_state for r in all_refs])  # [N]

# ---------------------------------------------------------------------------
# Main loop: train one agent per feature mode
# ---------------------------------------------------------------------------
results = {}

for mode in cfg.FEATURE_MODES:
    print(f"\n{'='*60}")
    print(f"Mode: {cfg.MODE_LABELS[mode]}")
    print(f"{'='*60}")

    feat_fn  = FEATURE_FNS[mode]
    feat_dim = cfg.FEATURE_DIMS[mode]

    # Build oracle feature cache (no CLIP, no image loading)
    oracle_cache = {
        uid: torch.tensor(feat_fn(uid)).to(device) for uid in refs_dict
    }

    # Agent: mapper only, oracle features injected via _cache
    torch.manual_seed(42)
    agent = DlbtAgent(
        freeze_encoder=True,
        n_mc_samples=cfg.N_MC,
        device=device,
        mapper_hidden=None,
        feature_dim=feat_dim,
    )
    agent._cache.update(oracle_cache)

    # Train — use same dataset for train and val (monitoring convergence)
    result = train_dlbt(
        agent, ds, ds, refs_dict,
        n_epochs=cfg.N_EPOCHS,
        lr=cfg.LR,
        patience=cfg.PATIENCE,
    )
    print(f"  Best epoch {result.best_epoch:4d}  "
          f"MSE={result.best_val_mse:.5f}")

    # Extract recovered alpha for all stimuli
    agent.eval()
    with torch.no_grad():
        alpha_rec = agent.get_alpha(all_refs).cpu().numpy()     # [N, K]

    q_rec = alpha_rec / alpha_rec.sum(axis=1, keepdims=True)    # [N, K]

    # Flatten over (stimulus × state) for scatter
    q_gt_flat  = q_gt.flatten()
    q_rec_flat = q_rec.flatten()

    rho, _ = spearmanr(q_gt_flat, q_rec_flat)
    mse    = float(np.mean((q_gt_flat - q_rec_flat) ** 2))
    print(f"  Recovery  ρ={rho:.4f}  MSE={mse:.6f}")

    results[mode] = dict(
        q_gt       = q_gt,
        q_rec      = q_rec,
        q_gt_flat  = q_gt_flat,
        q_rec_flat = q_rec_flat,
        alpha_gt   = alpha_gt,
        alpha_rec  = alpha_rec,
        states     = states,
        uids       = [r.uid for r in all_refs],
        rho        = rho,
        mse        = mse,
        best_epoch = result.best_epoch,
    )

    del agent
    gc.collect()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = cfg.RESULTS_DIR / "results.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved → {out_path}")
