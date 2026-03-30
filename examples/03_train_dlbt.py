"""
Train DlbtAgent on a synthetic behavioral dataset.

Synthetic data generation:
  - Ground truth: for each (image, task), the optimal action is read directly
    from the image's latent state and the task's delta_u.
  - Each (image, task) observation is n_trials Bernoulli draws with
    P(correct) = noise_level.
  - Train/val split is on tasks (7 train / 3 val).
    All 4 simple tasks are in train; val spans two distinct composite types.

After training, plots:
  - NLL and MSE learning curves.
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

from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METADATA     = "stimuli/imgs/metadata.jsonl"
CACHE_PATH   = "stimuli/imgs/clip_rn50_features.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
SEED         = 42
N_TRIALS     = 100    # Bernoulli draws per (image, task)
NOISE_LEVEL  = 0.85   # P(correct action)  — 1.0 = noise-free oracle
N_EPOCHS     = 500
LR           = 3e-3
N_MC         = 200    # MC samples for choice_probs during training

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
# Synthetic data generator
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

def make_synthetic_dataset(task_names: list[str]) -> BehavioralDataset:
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, refs, rng=rng):
            optimal = task.optimal_action(ref.latent_state)
            # count_1 = number of "right" choices
            p_right = NOISE_LEVEL if optimal == 1 else (1.0 - NOISE_LEVEL)
            count_1 = int(rng.binomial(N_TRIALS, p_right))
            count_0 = N_TRIALS - count_1
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
    print(f"Saved.")

print(f"\nTraining DlbtAgent (frozen encoder)...")
result = train_dlbt(
    agent, train_ds, val_ds, refs_dict,
    n_epochs=N_EPOCHS, lr=LR, patience=40,
)
print(f"\nBest epoch: {result.best_epoch}  best_val_mse: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
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
axes[1].set(xlabel="epoch", ylabel="MSE", title="Mean squared error")
axes[1].legend()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("examples/plots/03_learning_curves.png", dpi=150)
print("Saved: examples/plots/03_learning_curves.png")
plt.close()

# ---------------------------------------------------------------------------
# Predicted vs empirical P(right) scatter
# ---------------------------------------------------------------------------
agent.eval()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

for ax, (task_names, ds, label) in zip(axes, [
    (TRAIN_TASKS, train_ds, "Train tasks"),
    (VAL_TASKS,   val_ds,   "Val tasks"),
]):
    pred_all = []
    emp_all  = []
    for task_name, group in ds.iter_tasks():
        task      = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        with torch.no_grad():
            probs = agent.choice_probs(batch_refs, task)
        pred = probs[:, 1].cpu().numpy()

        totals = (group["count_0"] + group["count_1"]).values.clip(1)
        emp    = group["count_1"].values / totals
        pred_all.append(pred)
        emp_all.append(emp)

    pred_all = np.concatenate(pred_all)
    emp_all  = np.concatenate(emp_all)
    mse_val  = float(np.mean((pred_all - emp_all) ** 2))

    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    ax.scatter(pred_all, emp_all, alpha=0.4, s=12)
    ax.set(
        xlabel="Predicted P(right)",
        ylabel="Empirical P(right)",
        title=f"{label}  (MSE={mse_val:.4f})",
        xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
    )

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("examples/plots/03_pred_vs_emp.png", dpi=150)
print("Saved: examples/plots/03_pred_vs_emp.png")
plt.close()
