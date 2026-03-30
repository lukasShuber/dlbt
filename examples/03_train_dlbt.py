"""
Train DlbtAgent on a synthetic behavioral dataset.

Synthetic data generation (model-matched ground truth):
  - Ground truth: a DLBT agent with known Dirichlet parameters α*(x).
    α*(x) is peaked on the true latent state of image x:
        α*_k = PEAK              if k == latent_state(x)
        α*_k = BASE_CONCENTRATION  otherwise
    Soft beliefs: each binary dimension is "softened" by a sigmoid applied
    to the distance of the image's continuous property from its threshold.
    Images near a boundary (e.g. y ≈ 0.5) get mixed beliefs across states;
    images far from any boundary get sharply peaked beliefs. This creates a
    continuous spread of P(right) values that is encoded in the image content
    (and hence potentially recoverable from CLIP features), unlike per-image
    variable peak which is an invisible synthetic artefact.
  - Behavior: N_TRIALS independent draws of argmax SEU given b̃ ~ Dirichlet(α*(x)).
  - Because the training model has the same functional form, train MSE should
    converge to the noise floor in the limit of sufficient data and epochs.

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

from dlbt.constants import (
    K, DIM_FRONT_BACK, DIM_SHAPE, DIM_TRANSP, DIM_GLOSS,
    Y_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, NON_TRIANGULAR_SHAPES,
)
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.slda import SldaAgent
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
PEAK               = 15.0   # peak concentration added to matching latent states
BASE_CONCENTRATION = 1.0    # base concentration on all latent states
BETA               = 5.0    # sigmoid sharpness for continuous dimensions;
                             # higher = sharper boundary, lower = more perceptual ambiguity
N_EPOCHS           = 100
LR                 = 1e-3   # lower than frozen: attnpool weights are sensitive
N_MC               = 200    # MC samples for choice_probs during training
FREEZE_ENCODER     = False  # True → DLBT-frozen; False → DLBT-attnpool

# 7 train / 3 val task split.
# All 4 simple tasks stay in train (they are the only per-dimension signal).
# Val spans two distinct dimension combinations for a broader generalization test.
TRAIN_TASKS = [
    # simple — one per dimension, must all be in train
    "front_back", "triangular", "transparent", "glossy",
    # composites
    "front_and_transparent",
    "nontriangular_and_glossy",
    "triangular_and_front",
    "nontriangular_and_front"
]
VAL_TASKS = [
    "back_and_glossy",
    "triangular_and_transparent"
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
# Continuous metadata (used for soft belief generation)
# ---------------------------------------------------------------------------
# Load the raw continuous properties for each image directly from metadata.
# These drive the sigmoid softening of the ground truth Dirichlet.

def _load_continuous_metadata(metadata_path: str) -> dict[str, dict]:
    """Return uid -> {y, transparency, glossiness, is_nontri} for every image."""
    import json
    result = {}
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            z   = rec["z"]
            result[rec["id"]] = dict(
                y            = z["pos_xy"][1],
                transparency = z["transparency"],
                glossiness   = z["glossiness"],
                is_nontri    = z["shape_name"] in NON_TRIANGULAR_SHAPES,
            )
    return result

cont_meta = _load_continuous_metadata(METADATA)

# ---------------------------------------------------------------------------
# Ground truth Dirichlet agent — soft beliefs
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    """
    Soft Dirichlet concentration vector for an image.

    Each binary dimension is softened by a sigmoid applied to the image's
    continuous property distance from its threshold:

        p_dim(x) = σ(BETA · (value − threshold))

    The match score for latent state k is the product of per-dimension
    probabilities, giving high α_k for clearly-matching states and low α_k
    for mismatching ones. Images near a boundary get genuinely mixed beliefs.

    Shape dimension is discrete (triangular / non-triangular) so it is not
    softened — it is kept as a hard 0/1 probability.
    """
    z = cont_meta[uid]

    p_back   = _sigmoid(BETA * (z["y"]            - Y_THRESHOLD))   # P(back)
    p_nontri = float(z["is_nontri"])                                  # discrete
    p_transp = _sigmoid(BETA * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(BETA * (z["glossiness"]   - GLOSS_THRESH))

    alpha = np.empty(K, dtype=np.float64)
    for k in range(K):
        k_back   = (k >> DIM_FRONT_BACK) & 1
        k_nontri = (k >> DIM_SHAPE)      & 1
        k_transp = (k >> DIM_TRANSP)     & 1
        k_glossy = (k >> DIM_GLOSS)      & 1

        match = (
            (p_back   if k_back   else (1.0 - p_back))   *
            (p_nontri if k_nontri else (1.0 - p_nontri)) *
            (p_transp if k_transp else (1.0 - p_transp)) *
            (p_glossy if k_glossy else (1.0 - p_glossy))
        )
        alpha[k] = BASE_CONCENTRATION + PEAK * match

    return alpha


def gt_p_right(uid: str, task, n_mc: int = 2000, rng=None) -> float:
    """
    Estimate P(right | image, task) via MC integration over the soft Dirichlet.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(uid)
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
    Sample N_TRIALS binary choices from the soft ground truth Dirichlet agent.
    Returns (count_0, count_1).
    """
    alpha   = gt_alpha(ref.uid)
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
# Train — DLBT (frozen or attnpool, controlled by FREEZE_ENCODER)
# ---------------------------------------------------------------------------
model_label = "DLBT (frozen)" if FREEZE_ENCODER else "DLBT (attnpool)"
agent = DlbtAgent(freeze_encoder=FREEZE_ENCODER, n_mc_samples=N_MC, device=DEVICE)

# Full CLIP feature cache is only used in frozen mode.
# In attnpool mode, train_dlbt precomputes backbone (pre-attnpool) features instead.
if FREEZE_ENCODER:
    if Path(CACHE_PATH).exists():
        print(f"Loading cached CLIP features from {CACHE_PATH}")
        agent.load_cache(CACHE_PATH)
    else:
        print(f"Precomputing CLIP features → {CACHE_PATH}")
        agent.precompute_features(list(refs_dict.values()))
        agent.save_cache(CACHE_PATH)
        print("Saved.")

print(f"\nTraining {model_label}...")
result = train_dlbt(
    agent, train_ds, val_ds, refs_dict,
    n_epochs=N_EPOCHS, lr=LR, patience=N_EPOCHS,
)
print(f"Best epoch: {result.best_epoch}  best_val_mse: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Train — SLDA baseline
# ---------------------------------------------------------------------------
slda = SldaAgent(device=DEVICE)

# Load cached CLIP features if available; train_dlbt will precompute them
# otherwise (since slda.freeze_encoder=True). Save after to reuse next run.
if Path(CACHE_PATH).exists():
    print(f"Loading cached CLIP features from {CACHE_PATH}")
    slda.load_cache(CACHE_PATH)

print("\nTraining SldaAgent...")
slda_result = train_dlbt(
    slda, train_ds, val_ds, refs_dict,
    n_epochs=N_EPOCHS, lr=LR, patience=100,
)
print(f"Best epoch: {slda_result.best_epoch}  best_val_mse: {slda_result.best_val_mse:.4f}")
print(f"Learned temperature τ = {slda.log_temperature.exp().item():.3f}")

# Persist freshly-computed features for future runs.
if not Path(CACHE_PATH).exists():
    slda.save_cache(CACHE_PATH)
    print(f"Saved fresh CLIP features → {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Learning curves — DLBT and SLDA side by side
# ---------------------------------------------------------------------------
noise_floor_train = train_ds.noise_floor()
noise_floor_val   = val_ds.noise_floor()

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)

for row, (res, lbl) in enumerate([(result, model_label), (slda_result, "SLDA")]):
    epochs = range(len(res.train_nlls))
    ax_nll, ax_mse = axes[row]

    ax_nll.plot(epochs, res.train_nlls, label="train", color="#d95f02")
    ax_nll.plot(epochs, res.val_nlls,   label="val",   color="#7570b3")
    ax_nll.axvline(res.best_epoch, ls=":", color="gray")
    ax_nll.set(ylabel="NLL", title=f"{lbl} — NLL")
    ax_nll.legend(fontsize=8)

    ax_mse.plot(epochs, res.train_mses, label="train", color="#d95f02")
    ax_mse.plot(epochs, res.val_mses,   label="val",   color="#7570b3")
    ax_mse.axvline(res.best_epoch, ls=":", color="gray")
    ax_mse.axhline(noise_floor_train, ls="--", color="#d95f02", alpha=0.5, lw=1,
                   label=f"train floor ({noise_floor_train:.4f})")
    ax_mse.axhline(noise_floor_val,   ls="--", color="#7570b3", alpha=0.5, lw=1,
                   label=f"val floor ({noise_floor_val:.4f})")
    ax_mse.set(ylabel="cMSE", title=f"{lbl} — cMSE")
    ax_mse.legend(fontsize=8)

for ax in axes[1]:
    ax.set_xlabel("epoch")

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("examples/plots/03_learning_curves.png", dpi=150)
print("Saved: examples/plots/03_learning_curves.png")
plt.close()

# ---------------------------------------------------------------------------
# Scatter: predicted vs ground truth P(right)
# ---------------------------------------------------------------------------
agent.eval()
slda.eval()
rng_gt = np.random.default_rng(SEED + 1)

# Model colors: train/val split within each model
DLBT_TRAIN  = "#d95f02"   # orange
DLBT_VAL    = "#7570b3"   # purple
SLDA_TRAIN  = "#1b9e77"   # teal
SLDA_VAL    = "#e7298a"   # magenta

# Collect per-task predictions for both models; reuse true_p across models.
# per_task[task_name] -> {pred, true, cmse, rho, color}  (DLBT)
# per_task_slda[task_name] -> same shape                 (SLDA)

per_task:      dict = {}
per_task_slda: dict = {}

all_ds = [(TRAIN_TASKS, train_ds), (VAL_TASKS, val_ds)]
for task_names, ds in all_ds:
    is_train     = task_names is TRAIN_TASKS
    dlbt_color   = DLBT_TRAIN  if is_train else DLBT_VAL
    slda_color   = SLDA_TRAIN  if is_train else SLDA_VAL

    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]

        # Ground truth — computed once, shared by both models
        true_p = np.array([
            gt_p_right(r.uid, task, n_mc=1000, rng=rng_gt)
            for r in batch_refs
        ])

        # ---- DLBT predictions -----------------------------------------------
        with torch.no_grad():
            pred_d = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()

        raw_mse = float(np.mean((pred_d - true_p) ** 2))
        mc_corr = float(np.mean(pred_d * (1 - pred_d))) / (N_MC - 1)
        rho_d, _ = spearmanr(pred_d, true_p)
        per_task[task_name] = dict(
            pred=pred_d, true=true_p,
            cmse=raw_mse - mc_corr, rho=rho_d,
            color=dlbt_color,
        )

        # ---- SLDA predictions (deterministic — no MC correction) ------------
        with torch.no_grad():
            pred_s = slda.choice_probs(batch_refs, task)[:, 1].cpu().numpy()

        raw_mse_s = float(np.mean((pred_s - true_p) ** 2))
        rho_s, _  = spearmanr(pred_s, true_p)
        per_task_slda[task_name] = dict(
            pred=pred_s, true=true_p,
            cmse=raw_mse_s, rho=rho_s,   # no MC correction for deterministic model
            color=slda_color,
        )

# ---------------------------------------------------------------------------
# Plot 1: summary scatter — train/val × DLBT/SLDA (2×2 grid)
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.35, "wspace": 0.12})

for col, (task_names, split_label) in enumerate([
    (TRAIN_TASKS, "Train"),
    (VAL_TASKS,   "Val"),
]):
    for row, (pt, scatter_lbl, mc_n) in enumerate([
        (per_task,      model_label, N_MC),
        (per_task_slda, "SLDA",      None),
    ]):
        ax    = axes[row, col]
        is_train = task_names is TRAIN_TASKS

        pred_all = np.concatenate([pt[t]["pred"] for t in task_names])
        true_all = np.concatenate([pt[t]["true"] for t in task_names])
        color    = (DLBT_TRAIN if row == 0 else SLDA_TRAIN) if is_train \
                   else (DLBT_VAL if row == 0 else SLDA_VAL)

        raw_mse = float(np.mean((pred_all - true_all) ** 2))
        if mc_n is not None:
            mc_corr = float(np.mean(pred_all * (1 - pred_all))) / (mc_n - 1)
            cmse    = raw_mse - mc_corr
        else:
            cmse    = raw_mse
        rho, _ = spearmanr(pred_all, true_all)

        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8, zorder=0)
        ax.scatter(pred_all, true_all, alpha=0.35, s=10, color=color,
                   linewidths=0)
        ax.set(
            title=f"{scatter_lbl} — {split_label}\ncMSE={cmse:.4f}   ρ={rho:.3f}",
            xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
        )
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("True P(right)", fontsize=9)
        if row == 1:
            ax.set_xlabel("Predicted P(right)", fontsize=9)

sns.despine(fig=fig, trim=True)
plt.savefig("examples/plots/03_pred_vs_true.png", dpi=150, bbox_inches="tight")
print("Saved: examples/plots/03_pred_vs_true.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: per-task grid — DLBT and SLDA overlaid in each panel
# ---------------------------------------------------------------------------
ALL_TASKS = TRAIN_TASKS + VAL_TASKS   # 10 tasks
N_COLS    = 5
N_ROWS    = 2

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(12, 5),
    sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.48, "wspace": 0.08},
)

for idx, (ax, task_name) in enumerate(zip(axes.flat, ALL_TASKS)):
    d_dlbt = per_task[task_name]
    d_slda = per_task_slda[task_name]
    true_p = d_dlbt["true"]   # shared

    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8, zorder=0)
    ax.scatter(d_dlbt["pred"], true_p, alpha=0.5, s=6,
               color=d_dlbt["color"], linewidths=0, label="DLBT")
    ax.scatter(d_slda["pred"], true_p, alpha=0.5, s=6,
               color=d_slda["color"], linewidths=0, marker="s", label="SLDA")

    # clean task name
    nice = (task_name
            .replace("nontriangular", "nontri.")
            .replace("_and_", " & ")
            .replace("_", "/"))
    ax.set_title(nice, fontsize=7.5, pad=3)

    # ρ annotations: DLBT top-left, SLDA below it
    ax.text(0.05, 0.93, f"D ρ={d_dlbt['rho']:.2f}",
            transform=ax.transAxes, fontsize=6.5,
            color=d_dlbt["color"], va="top")
    ax.text(0.05, 0.78, f"S ρ={d_slda['rho']:.2f}",
            transform=ax.transAxes, fontsize=6.5,
            color=d_slda["color"], va="top")

    row, col = divmod(idx, N_COLS)
    if row == N_ROWS - 1:
        ax.set_xlabel("Pred", fontsize=8)
    if col == 0:
        ax.set_ylabel("True", fontsize=8)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=6)

# shared super-labels
fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)

# legend: model × split
is_train_mask = {t: True for t in TRAIN_TASKS}
is_train_mask.update({t: False for t in VAL_TASKS})

fig.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DLBT_TRAIN,
               markersize=6, label="DLBT (train)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DLBT_VAL,
               markersize=6, label="DLBT (val)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=SLDA_TRAIN,
               markersize=6, label="SLDA (train)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=SLDA_VAL,
               markersize=6, label="SLDA (val)"),
    ],
    loc="lower right", bbox_to_anchor=(1.0, 0.02),
    fontsize=7.5, frameon=False, ncol=2,
)

sns.despine(fig=fig, trim=True)
plt.savefig("examples/plots/03_per_task.png", dpi=150, bbox_inches="tight")
print("Saved: examples/plots/03_per_task.png")
plt.close()
