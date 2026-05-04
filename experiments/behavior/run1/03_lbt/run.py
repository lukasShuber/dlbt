"""
run1/03_lbt/run.py — LbtAgent fit on combined run0 + run1 behavioral data.

Fits a bare LbtAgent (no encoder — just a learnable α table, one row per probe
image) directly to the empirical choice counts from the real experiment.  No
image features are used.  This is the behavioral analogue of the simulation in
experiments/simulations/04_lbt_recovery/.

Only probe images are used (those shown in the forced-choice probing phase of
the experiment), giving one Dirichlet per probe image.

Outputs (all written to results/plots/):
  plot_01_alpha_{RUN_TAG}.png    — recovered α heatmap (probe images × latent states)
  plot_02_scatter_{RUN_TAG}.png  — predicted vs empirical P(right), coloured by arity
  plot_03_curves_{RUN_TAG}.png   — training / val NLL and cMSE curves

Run from repo root:
    python experiments/behavior/run1/03_lbt/run.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[4]))   # repo root

from dlbt.agents.lbt import LbtAgent
from dlbt.constants import K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_lbt import train_lbt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parents[2] / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_label(k: int) -> str:
    lr = (k >> DIM_LEFT_RIGHT)  & 1
    tr = (k >> DIM_TRANSP)      & 1
    gl = (k >> DIM_GLOSS)       & 1
    sl = (k >> DIM_SMALL_LARGE) & 1
    return (f"{'R' if lr else 'L'} "
            f"{'Tr' if tr else 'Op'} "
            f"{'Gl' if gl else 'Mt'} "
            f"{'Lg' if sl else 'Sm'}")


STATE_LABELS = [_state_label(k) for k in range(K)]


def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _image_label(ref) -> str:
    """Short label for a probe image: its true latent-state label."""
    return _state_label(ref.latent_state)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

print(f"Device: {device}")
print(f"Run tag: {cfg.RUN_TAG}")
print(f"Split mode: {cfg.SPLIT_MODE}  "
      f"({len(cfg.TRAIN_TASKS)} train / {len(cfg.VAL_TASKS)} val tasks)")

# ---------------------------------------------------------------------------
# Load image refs
# ---------------------------------------------------------------------------
print("\nLoading image refs...")
refs_dict = load_image_refs(cfg.METADATA)
refs_all  = image_refs_as_list(refs_dict)
print(f"  Total images: {len(refs_all)}")

# ---------------------------------------------------------------------------
# Load + preprocess behavioral data
# ---------------------------------------------------------------------------
print("\nLoading behavioral data...")
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
print(f"  After QC: {len(df_filtered):,} trials  "
      f"({diag['n_pass_both']} / {diag['n_total_assignments']} assignments passed)")

# Aggregate all eligible tasks (train + val)
_eligible_names  = set(cfg.TRAIN_TASKS + cfg.VAL_TASKS)
_eligible_beh_id = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in _eligible_names}

full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _eligible_beh_id,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)
print(f"  Aggregated cells: {len(full_ds):,}  "
      f"({len(probe_uids)} probe + {len(main_uids)} main images)")

# ---------------------------------------------------------------------------
# Restrict to probe images only
# ---------------------------------------------------------------------------
probe_df = full_ds.df[full_ds.df["uid"].isin(probe_uids)].copy()
probe_df["n_trials"] = probe_df["count_0"] + probe_df["count_1"]
avg_n = probe_df["n_trials"].mean()
print(f"  Probe-only cells: {len(probe_df):,}  |  avg trials/cell: {avg_n:.1f}")

refs_by_uid  = {r.uid: r for r in refs_all}
probe_refs   = sorted(
    [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
probe_refs_dict = {r.uid: r for r in probe_refs}
image_labels    = [_image_label(r) for r in probe_refs]

print(f"  Probe image refs found: {len(probe_refs)}")

# ---------------------------------------------------------------------------
# Train / val dataset split
# ---------------------------------------------------------------------------
train_names = set(cfg.TRAIN_TASKS)
val_names   = set(cfg.VAL_TASKS)

train_df = probe_df[probe_df["task_name"].isin(train_names)].copy()
val_df   = probe_df[probe_df["task_name"].isin(val_names)].copy()

train_ds = BehavioralDataset(train_df)
val_ds   = BehavioralDataset(val_df) if len(val_df) > 0 else None

print(f"\n  Train cells: {len(train_ds)}")
if val_ds:
    print(f"  Val   cells: {len(val_ds)}")

# ---------------------------------------------------------------------------
# Compute empirical P(right) for scatter plot (all eligible tasks)
# ---------------------------------------------------------------------------
# Empirical P(right) = count_1 / (count_0 + count_1) per (uid, task_name) cell
emp_pright = {}
for _, row in probe_df.iterrows():
    total = row["count_0"] + row["count_1"]
    if total > 0:
        emp_pright[(row["uid"], row["task_name"])] = row["count_1"] / total

# ---------------------------------------------------------------------------
# Initialise LbtAgent
# ---------------------------------------------------------------------------
if cfg.INIT_MODE == "uniform":
    agent = LbtAgent(
        uid_list           = [r.uid for r in probe_refs],
        n_mc_samples       = cfg.N_MC,
        device             = device,
        init_alpha         = cfg.INIT_ALPHA,
        normalize_utility  = cfg.NORMALIZED_UTILITY,
        use_prior_trick    = cfg.USE_PRIOR_TRICK,
    )
elif cfg.INIT_MODE == "random":
    agent = LbtAgent(
        uid_list           = [r.uid for r in probe_refs],
        n_mc_samples       = cfg.N_MC,
        device             = device,
        normalize_utility  = cfg.NORMALIZED_UTILITY,
        use_prior_trick    = cfg.USE_PRIOR_TRICK,
    )
    init_rng   = np.random.default_rng(cfg.INIT_SEED)
    n          = len(probe_refs)
    alpha_rand = init_rng.uniform(
        cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH, size=(n, K)
    ).astype(np.float32)
    alpha_t = torch.tensor(alpha_rand)
    with torch.no_grad():
        # softplus inverse: log(exp(x) - 1)
        agent.log_alpha.copy_(torch.log(torch.exp(alpha_t) - 1.0))
else:
    raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\nTraining LbtAgent...")
result = train_lbt(
    agent,
    train_ds,
    val_ds if val_ds is not None else train_ds,
    probe_refs_dict,
    n_epochs  = cfg.N_EPOCHS,
    lr        = cfg.LR,
    patience  = cfg.N_EPOCHS,   # no early stopping
    grad_clip = cfg.GRAD_CLIP,
)
print(f"Best epoch: {result.best_epoch}  |  Best val cMSE: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Extract recovered α
# ---------------------------------------------------------------------------
agent.eval()
with torch.no_grad():
    recovered_alpha = F.softplus(agent.log_alpha).cpu().numpy()   # [n_probe, K]

# ---------------------------------------------------------------------------
# Compute predicted P(right) and collect empirical counterpart
# ---------------------------------------------------------------------------
print("\nComputing predicted P(right)...")
all_task_names = list(_eligible_names)
all_tasks_obj  = {n: get_task(n) for n in all_task_names}

pred_probs, emp_probs, arities, is_train_flag = [], [], [], []
agent.eval()
with torch.no_grad():
    for task_name, task_obj in all_tasks_obj.items():
        probs = agent.choice_probs(probe_refs, task_obj)[:, 1].cpu().numpy()
        for i, ref in enumerate(probe_refs):
            key = (ref.uid, task_name)
            if key not in emp_pright:
                continue
            pred_probs.append(probs[i])
            emp_probs.append(emp_pright[key])
            arities.append(_arity(task_name))
            is_train_flag.append(task_name in train_names)

pred_probs    = np.array(pred_probs)
emp_probs     = np.array(emp_probs)
arities       = np.array(arities)
is_train_flag = np.array(is_train_flag)

for split_label, mask_split in [("train", is_train_flag),
                                 ("val",   ~is_train_flag)]:
    if not mask_split.any():
        continue
    rho, _ = spearmanr(pred_probs[mask_split], emp_probs[mask_split])
    mse    = float(np.mean((pred_probs[mask_split] - emp_probs[mask_split]) ** 2))
    print(f"[{split_label}]  ρ={rho:.3f}  MSE={mse:.4f}")
    for n in range(1, 5):
        mask_n = mask_split & (arities == n)
        if not mask_n.any():
            continue
        rho_n, _ = spearmanr(pred_probs[mask_n], emp_probs[mask_n])
        mse_n    = float(np.mean((pred_probs[mask_n] - emp_probs[mask_n]) ** 2))
        print(f"  {n}-way  ρ={rho_n:.3f}  MSE={mse_n:.4f}")

# ---------------------------------------------------------------------------
# Plot 01 — recovered α heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    recovered_alpha,
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
ax.set_title("Recovered α", fontsize=11)
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=7, rotation=0)
plt.tight_layout()
out = plots_dir / f"plot_01_alpha_{cfg.RUN_TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# Plot 02 — scatter: predicted vs empirical P(right), coloured by arity
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

splits_to_plot = [("train", is_train_flag)]
if val_ds is not None:
    splits_to_plot.append(("val", ~is_train_flag))

n_panels = len(splits_to_plot)
fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), squeeze=False)

for ax, (split_label, mask_split) in zip(axes[0], splits_to_plot):
    if not mask_split.any():
        ax.set_visible(False)
        continue
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    for n in range(1, 5):
        mask_n = mask_split & (arities == n)
        if not mask_n.any():
            continue
        ax.scatter(pred_probs[mask_n], emp_probs[mask_n],
                   s=18, alpha=0.55, color=ARITY_COLOR[n],
                   label=f"{n}-way", zorder=2)
    rho, _ = spearmanr(pred_probs[mask_split], emp_probs[mask_split])
    mse    = float(np.mean((pred_probs[mask_split] - emp_probs[mask_split]) ** 2))
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted P(right)  [fitted LbtAgent]", fontsize=9)
    ax.set_ylabel("Empirical P(right)  [human]", fontsize=9)
    ax.set_title(f"{split_label}  [{cfg.RUN_TAG}]\nρ={rho:.3f}   MSE={mse:.4f}", fontsize=9)
    ax.legend(fontsize=8, frameon=False, title="arity", title_fontsize=8)

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_02_scatter_{cfg.RUN_TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 03 — learning curves
# ---------------------------------------------------------------------------
epochs = range(len(result.train_nlls))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].plot(epochs, result.train_nlls, color="#E76F51", label="train")
if val_ds is not None and result.val_nlls:
    axes[0].plot(epochs, result.val_nlls, color="#2a6fb5", label="val")
    axes[0].legend(fontsize=8, frameon=False)
axes[0].set(xlabel="epoch", ylabel="NLL",
            title=f"NLL  [{cfg.RUN_TAG}]")
axes[0].set_ylim(bottom=0)

axes[1].plot(epochs, result.train_mses, color="#E76F51", label="train")
if val_ds is not None and result.val_mses:
    axes[1].plot(epochs, result.val_mses, color="#2a6fb5", label="val")
    axes[1].legend(fontsize=8, frameon=False)
axes[1].set(xlabel="epoch", ylabel="cMSE",
            title=f"cMSE  [{cfg.RUN_TAG}]")
axes[1].set_ylim(bottom=0)

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_03_curves_{cfg.RUN_TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print(f"\nDone. All plots saved to {plots_dir}")
