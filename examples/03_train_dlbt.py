"""
Train DlbtAgent on a synthetic behavioral dataset.

Synthetic data generation (model-matched ground truth):
  - Ground truth: a DLBT agent with known Dirichlet parameters α*(x).
    α*(x) is peaked on the true latent state of image x:
        α*_k = peak(x)           if k == latent_state(x)
        α*_k = BASE_CONCENTRATION  otherwise
    where peak(x) ~ Uniform(PEAK_MIN, PEAK_MAX) is drawn once per image.
    PEAK_MIN=2 keeps beliefs genuinely uncertain (only 2/17 mass on true state),
    spreading P(right) continuously across [0, 1] for both simple and composite
    tasks. PEAK_MAX=20 allows confident-but-not-extreme beliefs at the upper end.
  - Behavior: N_TRIALS independent draws of argmax SEU given b̃ ~ Dirichlet(α*(x)).
  - Because the training model has the same functional form, train MSE should
    converge to the noise floor (≈0) in the limit of sufficient data and epochs.
    If it doesn't, something is fundamentally broken.

Train/val split (7 train / 3 val):
  - All 4 simple tasks in train (only per-dimension signal).
  - Val spans two distinct composite types: tests whether learned beliefs
    generalise to new task compositions (the tomography claim).

After training, plots:
  - NLL and MSE learning curves with noise floor.
  - Predicted vs true P(right) scatter (ground truth vs model).
  - Predicted vs empirical P(right) scatter (train and val tasks).

Run from repo root:
    python examples/03_train_dlbt.py
"""

import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from dlbt.constants import K
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.metrics import corrected_mse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METADATA   = "stimuli/imgs/metadata.jsonl"
CACHE_PATH = "stimuli/imgs/clip_rn50_features.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Device: {DEVICE} (no GPU — training will be slow)")

SEED               = 42
N_TRIALS           = 100    # SEU decisions per (image, task)
PEAK_MIN           = 2.0    # minimum peak concentration — genuinely uncertain beliefs
PEAK_MAX           = 20.0   # maximum peak concentration — confident but not extreme
BASE_CONCENTRATION = 1.0    # α* on all other latent states
N_EPOCHS           = 10000
LR                 = 1e-4  # conservative: attnpool finetuning needs small LR to stay stable
N_MC               = 200    # MC samples for choice_probs during training

# 7 train / 3 val task split.
# All 4 simple tasks stay in train (they are the only per-dimension signal).
# Val spans two distinct dimension combinations for a broader generalization test.
TRAIN_TASKS = [
    # simple — one per dimension, must all be in train
    "front_back", "triangular", "transparent", "glossy",
    # composites
    "front_and_transparent",
    "triangular_and_transparent",
    "nontriangular_and_glossy",
]
VAL_TASKS = [
    "back_and_glossy",         # location × material
    "triangular_and_front",    # shape × location
    "nontriangular_and_front", # shape × location (flipped)
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Per-image peak concentration (drawn once, reused everywhere)
# ---------------------------------------------------------------------------
# Sample PEAK_MIN ≤ peak(x) ≤ PEAK_MAX uniformly for each image.
# This creates a continuous spread of ground-truth P(right) values and avoids
# the three-band artefact that arises from a fixed PEAK_CONCENTRATION.
rng_peaks = np.random.default_rng(SEED)
peak_per_uid: dict[str, float] = {
    ref.uid: float(rng_peaks.uniform(PEAK_MIN, PEAK_MAX))
    for ref in refs
}

# ---------------------------------------------------------------------------
# Ground truth Dirichlet agent
# ---------------------------------------------------------------------------

def gt_alpha(latent_state: int, peak: float) -> np.ndarray:
    """
    Ground truth Dirichlet concentration vector for a given latent state.
    Peaked on the true state, uniform background elsewhere.
    """
    alpha = np.full(K, BASE_CONCENTRATION, dtype=np.float64)
    alpha[latent_state] = peak
    return alpha


def gt_p_right(latent_state: int, peak: float, task, n_mc: int = 2000, rng=None) -> float:
    """
    Estimate the ground truth P(right | latent_state, task) via MC integration
    over the ground truth Dirichlet.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(latent_state, peak)
    beliefs = rng.dirichlet(alpha, size=n_mc)   # [n_mc, K]
    logits  = beliefs @ task.delta_u             # [n_mc]
    return float((logits > 0).mean())


def sample_behavior(
    ref,
    task,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Sample N_TRIALS binary choices from the ground truth Dirichlet agent.
    Returns (count_0, count_1).
    """
    peak    = peak_per_uid[ref.uid]
    alpha   = gt_alpha(ref.latent_state, peak)
    beliefs = rng.dirichlet(alpha, size=n_trials)  # [n_trials, K]
    logits  = beliefs @ task.delta_u               # [n_trials]
    count_1 = int((logits > 0).sum())
    return n_trials - count_1, count_1


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def make_synthetic_dataset(task_names: list[str]) -> BehavioralDataset:
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, refs, rng=rng):
            count_0, count_1 = sample_behavior(ref, task, N_TRIALS, rng)
            records.append(Observation(
                uid=ref.uid,
                task_name=task_name,
                count_0=count_0,
                count_1=count_1,
            ))
    return BehavioralDataset.from_records(records)


train_ds = make_synthetic_dataset(TRAIN_TASKS)
val_ds   = make_synthetic_dataset(VAL_TASKS)
print(f"Train: {train_ds}  |  Val: {val_ds}")
print(f"Noise floor — train: {train_ds.noise_floor():.4f}  "
      f"val: {val_ds.noise_floor():.4f}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
agent = DlbtAgent(freeze_encoder=True, n_mc_samples=N_MC, device=DEVICE)

# Load or compute CLIP feature cache
if Path(CACHE_PATH).exists():
    print(f"Loading cached CLIP features from {CACHE_PATH}")
    agent.load_cache(CACHE_PATH)
else:
    print(f"Precomputing CLIP features → {CACHE_PATH}")
    agent.precompute_features(list(refs_dict.values()))
    agent.save_cache(CACHE_PATH)
    print("Saved.")

print("\nTraining DlbtAgent...")
result = train_dlbt(
    agent, train_ds, val_ds, refs_dict,
    n_epochs=N_EPOCHS, lr=LR, patience=N_EPOCHS,  # patience=N_EPOCHS disables early stopping
)
print(f"\nBest epoch: {result.best_epoch}  best_val_mse: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
noise_floor_train = train_ds.noise_floor()
noise_floor_val   = val_ds.noise_floor()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
epochs = range(len(result.train_nlls))

axes[0].plot(epochs, result.train_nlls, label="train", color="#d95f02")
axes[0].plot(epochs, result.val_nlls,   label="val",   color="#7570b3")
axes[0].axvline(result.best_epoch, ls=":", color="gray")
axes[0].set(xlabel="epoch", ylabel="NLL", title="Negative log-likelihood")
axes[0].legend()

axes[1].plot(epochs, result.train_mses, label="train", color="#d95f02")
axes[1].plot(epochs, result.val_mses,   label="val",   color="#7570b3")
axes[1].axvline(result.best_epoch, ls=":", color="gray")
axes[1].axhline(noise_floor_train, ls="--", color="#d95f02", alpha=0.5, lw=1,
                label=f"train floor ({noise_floor_train:.4f})")
axes[1].axhline(noise_floor_val,   ls="--", color="#7570b3", alpha=0.5, lw=1,
                label=f"val floor ({noise_floor_val:.4f})")
axes[1].set(xlabel="epoch", ylabel="cMSE", title="Mean squared error (corrected)")
axes[1].legend()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("examples/plots/03_learning_curves.png", dpi=150)
print("Saved: examples/plots/03_learning_curves.png")
plt.close()

# ---------------------------------------------------------------------------
# Scatter: predicted vs ground truth P(right)
# ---------------------------------------------------------------------------
agent.eval()
rng_gt = np.random.default_rng(SEED + 1)  # separate rng for gt_p_right estimates

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

for ax, (task_names, ds, label) in zip(axes, [
    (TRAIN_TASKS, train_ds, "Train tasks"),
    (VAL_TASKS,   val_ds,   "Val tasks"),
]):
    pred_all = []
    true_all = []

    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]

        with torch.no_grad():
            probs = agent.choice_probs(batch_refs, task)
        pred = probs[:, 1].cpu().numpy()

        true_p = np.array([
            gt_p_right(r.latent_state, peak_per_uid[r.uid], task, n_mc=1000, rng=rng_gt)
            for r in batch_refs
        ])

        pred_all.append(pred)
        true_all.append(true_p)

    pred_all = np.concatenate(pred_all)
    true_all = np.concatenate(true_all)

    # Corrected MSE: subtract MC variance of the predictor.
    # The ground truth target is noise-free, so only the predictor's
    # MC variance needs correcting (not finite-sample noise from counts).
    raw_mse  = float(np.mean((pred_all - true_all) ** 2))
    mc_corr  = float(np.mean(pred_all * (1 - pred_all))) / (N_MC - 1)
    cmse     = raw_mse - mc_corr

    rho, _ = spearmanr(pred_all, true_all)

    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    ax.scatter(pred_all, true_all, alpha=0.4, s=12, color="#2ca02c")
    ax.set(
        xlabel="Predicted P(right)",
        ylabel="True P(right)",
        title=f"{label}\ncMSE={cmse:.4f}   ρ={rho:.3f}",
        xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
    )

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("examples/plots/03_pred_vs_emp.png", dpi=150)
print("Saved: examples/plots/03_pred_vs_emp.png")
plt.close()
