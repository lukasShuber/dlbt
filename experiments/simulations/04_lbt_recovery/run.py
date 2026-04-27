"""
Simulation 04 — LBT recovery sanity check.

GT_MODE = "peaked": each image's Dirichlet is peaked at its true latent state
    with concentration c, uniform (1.0) elsewhere:
        alpha_true[k] = c   if k == true_state
        alpha_true[k] = 1.0 otherwise

GT_MODE = "random": each image's α_k ~ Uniform(GT_ALPHA_LOW, GT_ALPHA_HIGH)
    drawn independently; CONCENTRATIONS is ignored.

Simulates N_TRIALS behavioral trials per (image × task) cell across all 80
tasks (8 one-way + 24 two-way + 32 three-way + 16 four-way), then fits a bare
LbtAgent (no encoder — just a learnable α table) using train_lbt with the same
NLL loss and training loop as DLBT.

Goal: verify that the training pipeline correctly recovers the ground-truth α
when the model class is correct (oracle setting).

Outputs (per run):
  plot_01_alpha_true_{tag}.png      — ground-truth α heatmap
  plot_01_alpha_recovered_{tag}.png — recovered α heatmap after fitting
  plot_02_scatter_{tag}.png         — predicted P(right) vs oracle P(right)
  plot_03_curves_{tag}.png          — training / val NLL and cMSE curves

Run from repo root:
    python experiments/simulations/04_lbt_recovery/run.py
"""

import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

# Make sure dlbt is importable from repo root
sys.path.insert(0, str(Path(__file__).parents[3]))

from dlbt.agents.lbt import LbtAgent
from dlbt.constants import K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_lbt import train_lbt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

METADATA       = cfg.METADATA
RESULTS_DIR    = cfg.RESULTS_DIR
SEED           = cfg.SEED
N_TRIALS       = cfg.N_TRIALS
N_MC           = cfg.N_MC
N_EPOCHS       = cfg.N_EPOCHS
LR             = cfg.LR
GRAD_CLIP      = cfg.GRAD_CLIP
GT_MODE        = cfg.GT_MODE
TRAIN_TASKS    = cfg.TRAIN_TASKS
VAL_TASKS      = cfg.VAL_TASKS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def state_label(k: int) -> str:
    lr = (k >> DIM_LEFT_RIGHT)  & 1
    tr = (k >> DIM_TRANSP)      & 1
    gl = (k >> DIM_GLOSS)       & 1
    sl = (k >> DIM_SMALL_LARGE) & 1
    return (f"{'R' if lr else 'L'} "
            f"{'Tr' if tr else 'Op'} "
            f"{'Gl' if gl else 'Mt'} "
            f"{'Lg' if sl else 'Sm'}")


STATE_LABELS = [state_label(k) for k in range(K)]


def _arity(task) -> int:
    return task.name.count("_and_") + 1


def make_gt_alpha(true_state: int, concentration: float) -> np.ndarray:
    """α peaked at true_state with given concentration; 1.0 elsewhere."""
    alpha = np.ones(K, dtype=np.float64)
    alpha[true_state] = concentration
    return alpha


def oracle_p_right(alpha: np.ndarray, task, n_mc: int = 5000,
                   rng: np.random.Generator = None) -> float:
    """Exact P(right) under Dir(alpha) via MC (large n_mc → low noise)."""
    if rng is None:
        rng = np.random.default_rng(0)
    b = rng.dirichlet(alpha, size=n_mc)          # [n_mc, K]
    return float((b @ task.delta_u > 0).mean())


def simulate_cell(alpha: np.ndarray, task, n_trials: int,
                  rng: np.random.Generator):
    """Draw n_trials choices from Dir(alpha) for a given task."""
    b        = rng.dirichlet(alpha, size=n_trials)   # [n_trials, K]
    count_1  = int((b @ task.delta_u > 0).sum())
    count_0  = n_trials - count_1
    return count_0, count_1


def plot_alpha_heatmap(alpha_matrix: np.ndarray,
                       image_labels: list,
                       title: str,
                       out_path: Path,
                       true_states: list = None):
    """
    Heatmap: rows = images, cols = latent states.
    If true_states is provided, draw a blue box around each diagonal cell.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        alpha_matrix,
        ax=ax,
        xticklabels=STATE_LABELS,
        yticklabels=image_labels,
        cmap="YlOrRd",
        cbar_kws={"label": "αₖ"},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("Latent state", fontsize=10)
    ax.set_ylabel("Probe image",  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    # Draw blue outline at true-state diagonal
    if true_states is not None:
        for row_i, col_i in enumerate(true_states):
            ax.add_patch(plt.Rectangle(
                (col_i, row_i), 1, 1,
                fill=False, edgecolor="#1a5ccc", lw=2.5, zorder=5,
            ))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plots_dir = RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng    = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load image refs; pick one representative image per latent state
# ---------------------------------------------------------------------------
print("Loading image refs...")
refs_dict = load_image_refs(METADATA)
refs_all  = image_refs_as_list(refs_dict)
print(f"  Total images: {len(refs_all)}")

# Group by latent state and take first of each
by_state = defaultdict(list)
for r in refs_all:
    by_state[r.latent_state].append(r)

probe_refs = [by_state[k][0] for k in range(K) if by_state[k]]
assert len(probe_refs) == K, f"Expected {K} probe images, got {len(probe_refs)}"

probe_refs_dict = {r.uid: r for r in probe_refs}
image_labels    = [state_label(r.latent_state) for r in probe_refs]
true_states     = [r.latent_state for r in probe_refs]

print(f"  Probe images selected: {len(probe_refs)}  (one per latent state)")

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
all_tasks_list = cfg.all_tasks()
print(f"\nSplit mode: {cfg.SPLIT_MODE}")
print(f"  Train tasks: {len(TRAIN_TASKS)}  "
      f"(arities: {sorted(set(_arity(t) for t in TRAIN_TASKS))})")
if VAL_TASKS:
    print(f"  Val tasks:   {len(VAL_TASKS)}  "
          f"(arities: {sorted(set(_arity(t) for t in VAL_TASKS))})")

# ---------------------------------------------------------------------------
# Build ground-truth α for each probe image
# ---------------------------------------------------------------------------
if GT_MODE == "peaked":
    tag       = f"c{cfg.CONCENTRATION:g}"
    gt_alphas = {
        r.uid: make_gt_alpha(r.latent_state, cfg.CONCENTRATION)
        for r in probe_refs
    }
elif GT_MODE == "random":
    tag    = "random"
    gt_rng = np.random.default_rng(cfg.GT_SEED)
    gt_alphas = {
        r.uid: gt_rng.uniform(cfg.GT_ALPHA_LOW, cfg.GT_ALPHA_HIGH, size=K)
        for r in probe_refs
    }
else:
    raise ValueError(f"Unknown GT_MODE {GT_MODE!r}")

print(f"GT_MODE={GT_MODE}  tag={tag}")

# ---------------------------------------------------------------------------
# Compute oracle P(right) for all (image, task) pairs
# ---------------------------------------------------------------------------
print("Computing oracle P(right)...")
oracle_rng = np.random.default_rng(SEED + 1)
oracle = {}    # (uid, task_name) -> float
for ref in probe_refs:
    alpha = gt_alphas[ref.uid]
    for task in all_tasks_list:
        oracle[(ref.uid, task.name)] = oracle_p_right(
            alpha, task, n_mc=5000, rng=oracle_rng
        )

# ---------------------------------------------------------------------------
# Simulate behavioral data — training tasks only
# ---------------------------------------------------------------------------
print(f"Simulating {N_TRIALS} trials per cell "
      f"({len(probe_refs)} images × {len(TRAIN_TASKS)} train tasks = "
      f"{len(probe_refs) * len(TRAIN_TASKS)} cells)...")
sim_rng = np.random.default_rng(SEED + 2)
records = []
for ref in probe_refs:
    alpha = gt_alphas[ref.uid]
    for task in TRAIN_TASKS:
        c0, c1 = simulate_cell(alpha, task, N_TRIALS, sim_rng)
        records.append({
            "uid":       ref.uid,
            "task_name": task.name,
            "count_0":   c0,
            "count_1":   c1,
        })

df_all   = pd.DataFrame(records)
train_ds = BehavioralDataset(df_all)
print(f"Training cells: {len(train_ds)}")

# ---------------------------------------------------------------------------
# Initialise and train LbtAgent
# ---------------------------------------------------------------------------
if cfg.INIT_MODE == "uniform":
    agent = LbtAgent(
        uid_list     = [r.uid for r in probe_refs],
        n_mc_samples = N_MC,
        device       = device,
        init_alpha   = cfg.INIT_ALPHA,
    )
elif cfg.INIT_MODE == "random":
    import torch.nn.functional as _F2
    agent = LbtAgent(
        uid_list     = [r.uid for r in probe_refs],
        n_mc_samples = N_MC,
        device       = device,
    )
    init_rng   = np.random.default_rng(cfg.INIT_SEED)
    n          = len(probe_refs)
    alpha_rand = init_rng.uniform(
        cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH, size=(n, K)
    ).astype(np.float32)
    alpha_t = torch.tensor(alpha_rand)
    with torch.no_grad():
        agent.log_alpha.copy_(torch.log(torch.exp(alpha_t) - 1.0))
else:
    raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")

print("Training LbtAgent...")
result = train_lbt(
    agent,
    train_ds,
    train_ds,          # val = train → monitors train loss, runs to N_EPOCHS
    probe_refs_dict,
    n_epochs  = N_EPOCHS,
    lr        = LR,
    patience  = N_EPOCHS,  # effectively no early stopping
    grad_clip = GRAD_CLIP,
)
print(f"Best epoch: {result.best_epoch}  |  Best val cMSE: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Extract recovered α
# ---------------------------------------------------------------------------
import torch.nn.functional as _F
agent.eval()
with torch.no_grad():
    recovered_alpha = _F.softplus(agent.log_alpha).cpu().numpy()

# ---------------------------------------------------------------------------
# Compute predicted P(right) from fitted agent
# ---------------------------------------------------------------------------
print("Computing predicted P(right) from fitted agent...")
agent.eval()
pred_probs, oracle_probs, task_arities, is_train = [], [], [], []
train_task_names = {t.name for t in TRAIN_TASKS}
with torch.no_grad():
    for task in all_tasks_list:
        probs = agent.choice_probs(probe_refs, task)[:, 1].cpu().numpy()
        for i, ref in enumerate(probe_refs):
            pred_probs.append(probs[i])
            oracle_probs.append(oracle[(ref.uid, task.name)])
            task_arities.append(_arity(task))
            is_train.append(task.name in train_task_names)

pred_probs   = np.array(pred_probs)
oracle_probs = np.array(oracle_probs)
task_arities = np.array(task_arities)
is_train     = np.array(is_train)

for split_label, mask_split in [("train", is_train), ("val", ~is_train)]:
    if not mask_split.any():
        continue
    rho_s, _ = spearmanr(pred_probs[mask_split], oracle_probs[mask_split])
    mse_s    = float(np.mean((pred_probs[mask_split] - oracle_probs[mask_split]) ** 2))
    print(f"[{split_label}]  ρ={rho_s:.3f}  MSE={mse_s:.4f}")
    for n in range(1, 5):
        mask_n = mask_split & (task_arities == n)
        if not mask_n.any():
            continue
        rho_n, _ = spearmanr(pred_probs[mask_n], oracle_probs[mask_n])
        mse_n    = float(np.mean((pred_probs[mask_n] - oracle_probs[mask_n]) ** 2))
        print(f"  {n}-way  ρ={rho_n:.3f}  MSE={mse_n:.4f}")

# ---------------------------------------------------------------------------
# Plot 01a — ground-truth α heatmap
# ---------------------------------------------------------------------------
gt_matrix = np.stack([gt_alphas[r.uid] for r in probe_refs])   # [16, 16]
gt_title  = (f"Ground-truth α  [concentration={cfg.CONCENTRATION}]"
             if GT_MODE == "peaked"
             else f"Ground-truth α  [random U({cfg.GT_ALPHA_LOW},{cfg.GT_ALPHA_HIGH})]")
plot_alpha_heatmap(
    gt_matrix, image_labels,
    title       = gt_title,
    out_path    = plots_dir / f"plot_01_alpha_true_{tag}.png",
    true_states = true_states if GT_MODE == "peaked" else None,
)

# ---------------------------------------------------------------------------
# Plot 01b — recovered α heatmap
# ---------------------------------------------------------------------------
rec_title = (f"Recovered α  [concentration={cfg.CONCENTRATION}  epoch={result.best_epoch}]"
             if GT_MODE == "peaked"
             else f"Recovered α  [random GT  epoch={result.best_epoch}]")
plot_alpha_heatmap(
    recovered_alpha, image_labels,
    title       = rec_title,
    out_path    = plots_dir / f"plot_01_alpha_recovered_{tag}.png",
    true_states = true_states if GT_MODE == "peaked" else None,
)

# ---------------------------------------------------------------------------
# Plot 02 — scatter: predicted vs oracle P(right), coloured by arity
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

splits_to_plot = [("train", is_train)]
if VAL_TASKS:
    splits_to_plot.append(("val", ~is_train))
n_panels = len(splits_to_plot)

fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), squeeze=False)
for ax, (split_label, mask_split) in zip(axes[0], splits_to_plot):
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    for n in range(1, 5):
        mask_n = mask_split & (task_arities == n)
        if not mask_n.any():
            continue
        ax.scatter(pred_probs[mask_n], oracle_probs[mask_n],
                   s=18, alpha=0.55, color=ARITY_COLOR[n],
                   label=f"{n}-way", zorder=2)
    rho_s, _ = spearmanr(pred_probs[mask_split], oracle_probs[mask_split])
    mse_s    = float(np.mean((pred_probs[mask_split] - oracle_probs[mask_split])**2))
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted P(right)  [fitted LbtAgent]", fontsize=9)
    ax.set_ylabel("Oracle P(right)  [true α]", fontsize=9)
    ax.set_title(f"{split_label}  [{tag}]\nρ={rho_s:.3f}   MSE={mse_s:.4f}", fontsize=9)
    ax.legend(fontsize=8, frameon=False, title="arity", title_fontsize=8)

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_02_scatter_{tag}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 03 — learning curves
# ---------------------------------------------------------------------------
epochs = range(len(result.train_nlls))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].plot(epochs, result.train_nlls, color="#E76F51")
axes[0].set(xlabel="epoch", ylabel="NLL", title=f"Train NLL  [{tag}]")
axes[0].set_ylim(bottom=0)

axes[1].plot(epochs, result.train_mses, color="#E76F51")
axes[1].set(xlabel="epoch", ylabel="cMSE", title=f"Train cMSE  [{tag}]")
axes[1].set_ylim(bottom=0)

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_03_curves_{tag}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print(f"\nDone. All plots saved to {plots_dir}")
