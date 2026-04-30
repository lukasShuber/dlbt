"""
run1/04_lbt_on_noisy/run.py — LbtAgent fit on main images only.

Trains on MAIN images only (probe images are never used for training).
Probe images are evaluated post-hoc to assess stimulus-generation quality.

Scatter panels produced:
  1. Train         — main images  × training tasks
  2. Stim-gen      — probe images × training tasks   (always)
  3. Held-out val  — main images  × val tasks         (if val tasks exist)
  4. Joint         — probe images × val tasks         (if val tasks exist)

Outputs (all written to results/plots/):
  plot_01_alpha_{RUN_TAG}.png    — recovered α heatmap (probe images × latent states)
  plot_02_scatter_{RUN_TAG}.png  — scatter panels
  plot_03_curves_{RUN_TAG}.png   — training NLL and cMSE curves

Run from repo root:
    python experiments/behavior/run1/04_lbt_on_noisy/run.py
"""

import sys
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


def _scatter_panel(ax, pred, emp, arity_arr, title):
    """Draw one scatter panel with arity colouring and stats."""
    ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    for n in range(1, 5):
        mask = arity_arr == n
        if not mask.any():
            continue
        ax.scatter(pred[mask], emp[mask],
                   s=12, alpha=0.45, color=ARITY_COLOR[n],
                   label=f"{n}-way", zorder=2)
    if len(pred) > 1:
        rho, _ = spearmanr(pred, emp)
        mse    = float(np.mean((pred - emp) ** 2))
        ax.set_title(f"{title}\nρ={rho:.3f}  MSE={mse:.4f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted P(right)  [LbtAgent]", fontsize=8)
    ax.set_ylabel("Empirical P(right)  [human]",    fontsize=8)
    ax.legend(fontsize=7, frameon=False, title="arity", title_fontsize=7)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

has_val = len(cfg.VAL_TASKS) > 0

print(f"Device: {device}")
print(f"Run tag: {cfg.RUN_TAG}")
print(f"Split mode: {cfg.SPLIT_MODE}  "
      f"({len(cfg.TRAIN_TASKS)} train / {len(cfg.VAL_TASKS)} val tasks)")

# ---------------------------------------------------------------------------
# Load image refs
# ---------------------------------------------------------------------------
print("\nLoading image refs...")
refs_dict   = load_image_refs(cfg.METADATA)
refs_all    = image_refs_as_list(refs_dict)
refs_by_uid = {r.uid: r for r in refs_all}
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
# Split into main-image df (training) and probe-image df (evaluation)
# Apply minimum-trials filter to main images
# ---------------------------------------------------------------------------
full_df = full_ds.df.copy()
full_df["n_trials"] = full_df["count_0"] + full_df["count_1"]

main_df  = full_df[full_df["uid"].isin(main_uids)].copy()
main_df  = main_df[main_df["n_trials"] >= cfg.MIN_TRIALS_PER_CELL].copy()

probe_df = full_df[full_df["uid"].isin(probe_uids)].copy()

avg_n_main  = main_df["n_trials"].mean()
avg_n_probe = probe_df["n_trials"].mean() if len(probe_df) > 0 else 0
print(f"  Main  cells: {len(main_df):,}  ({main_df['uid'].nunique()} images, "
      f"avg trials/cell: {avg_n_main:.1f})")
print(f"  Probe cells: {len(probe_df):,}  ({probe_df['uid'].nunique()} images, "
      f"avg trials/cell: {avg_n_probe:.1f})")

# ---------------------------------------------------------------------------
# Build image ref lists
# ---------------------------------------------------------------------------
main_refs = sorted(
    [refs_by_uid[uid] for uid in main_df["uid"].unique() if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
probe_refs = sorted(
    [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
all_refs_dict = {r.uid: r for r in main_refs + probe_refs}

print(f"  Main  image refs: {len(main_refs)}")
print(f"  Probe image refs: {len(probe_refs)}")

# ---------------------------------------------------------------------------
# Empirical P(right) — stored separately for main and probe
# ---------------------------------------------------------------------------
def _emp_pright(df):
    out = {}
    for _, row in df.iterrows():
        total = row["count_0"] + row["count_1"]
        if total > 0:
            out[(row["uid"], row["task_name"])] = row["count_1"] / total
    return out

emp_main  = _emp_pright(main_df)
emp_probe = _emp_pright(probe_df)

# ---------------------------------------------------------------------------
# Train / val split — MAIN images only
# ---------------------------------------------------------------------------
train_names = set(cfg.TRAIN_TASKS)
val_names   = set(cfg.VAL_TASKS)

train_df = main_df[main_df["task_name"].isin(train_names)].copy()
train_ds = BehavioralDataset(train_df)

print(f"\n  Train cells (main × train tasks): {len(train_ds)}")

# ---------------------------------------------------------------------------
# Initialise LbtAgent on MAIN images only
# ---------------------------------------------------------------------------
if cfg.INIT_MODE == "uniform":
    agent = LbtAgent(
        uid_list          = [r.uid for r in main_refs],
        n_mc_samples      = cfg.N_MC,
        device            = device,
        init_alpha        = cfg.INIT_ALPHA,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
elif cfg.INIT_MODE == "random":
    agent = LbtAgent(
        uid_list          = [r.uid for r in main_refs],
        n_mc_samples      = cfg.N_MC,
        device            = device,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    init_rng   = np.random.default_rng(cfg.INIT_SEED)
    alpha_rand = init_rng.uniform(
        cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH, size=(len(main_refs), K)
    ).astype(np.float32)
    alpha_t = torch.tensor(alpha_rand)
    with torch.no_grad():
        agent.log_alpha.copy_(torch.log(torch.exp(alpha_t) - 1.0))
else:
    raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")

# ---------------------------------------------------------------------------
# Train (on main images × training tasks only)
# ---------------------------------------------------------------------------
print("\nTraining LbtAgent on main images...")
result = train_lbt(
    agent,
    train_ds,
    train_ds,             # no val during training
    all_refs_dict,
    n_epochs  = cfg.N_EPOCHS,
    lr        = cfg.LR,
    patience  = cfg.N_EPOCHS,
    grad_clip = cfg.GRAD_CLIP,
)
print(f"Best epoch: {result.best_epoch}  |  Best cMSE: {result.best_val_mse:.4f}")

# ---------------------------------------------------------------------------
# Helper: predict P(right) for a set of (refs, tasks, emp_dict)
# ---------------------------------------------------------------------------
def _collect_predictions(refs, task_names, emp_dict):
    """Returns arrays: pred, emp, arity — for all (ref, task) pairs with data."""
    all_tasks_obj = {n: get_task(n) for n in task_names}
    pred_list, emp_list, arity_list = [], [], []
    agent.eval()
    with torch.no_grad():
        for task_name, task_obj in all_tasks_obj.items():
            refs_with_data = [r for r in refs if (r.uid, task_name) in emp_dict]
            if not refs_with_data:
                continue
            probs = agent.choice_probs(refs_with_data, task_obj)[:, 1].cpu().numpy()
            for i, ref in enumerate(refs_with_data):
                pred_list.append(probs[i])
                emp_list.append(emp_dict[(ref.uid, task_name)])
                arity_list.append(_arity(task_name))
    return np.array(pred_list), np.array(emp_list), np.array(arity_list)

# ---------------------------------------------------------------------------
# Collect predictions for all four panels
# ---------------------------------------------------------------------------
print("\nComputing predictions...")
pred_train, emp_train, ar_train = _collect_predictions(
    main_refs, cfg.TRAIN_TASKS, emp_main)
print(f"  Train panel:    {len(pred_train)} cells")

pred_stim, emp_stim, ar_stim = _collect_predictions(
    probe_refs, cfg.TRAIN_TASKS, emp_probe)
print(f"  Stim-gen panel: {len(pred_stim)} cells")

if has_val:
    pred_val_main, emp_val_main, ar_val_main = _collect_predictions(
        main_refs, cfg.VAL_TASKS, emp_main)
    pred_val_probe, emp_val_probe, ar_val_probe = _collect_predictions(
        probe_refs, cfg.VAL_TASKS, emp_probe)
    print(f"  Held-out main panel:  {len(pred_val_main)} cells")
    print(f"  Joint panel:          {len(pred_val_probe)} cells")

# ---------------------------------------------------------------------------
# Print stats
# ---------------------------------------------------------------------------
for label, pred, emp in [
    ("train  (main  × train tasks)",  pred_train,     emp_train),
    ("stimgen (probe × train tasks)", pred_stim,       emp_stim),
    *([("val main  (main  × val tasks)",  pred_val_main,  emp_val_main),
       ("joint    (probe × val tasks)",   pred_val_probe, emp_val_probe)]
      if has_val else []),
]:
    if len(pred) < 2:
        continue
    rho, _ = spearmanr(pred, emp)
    mse    = float(np.mean((pred - emp) ** 2))
    print(f"[{label}]  ρ={rho:.3f}  MSE={mse:.4f}")

# ---------------------------------------------------------------------------
# Plot 01 — recovered α heatmap (probe images)
# ---------------------------------------------------------------------------
# For probe images, we don't have α from the agent directly (they weren't
# trained). We show the main-image α as a heatmap instead, grouped by
# latent state using mean α per latent state.
agent.eval()
with torch.no_grad():
    main_alpha = F.softplus(agent.log_alpha).cpu().numpy()   # [n_main, K]

# Average α per latent state across main images
from collections import defaultdict
alpha_by_state = defaultdict(list)
for i, ref in enumerate(main_refs):
    alpha_by_state[ref.latent_state].append(main_alpha[i])

state_mean_alpha = np.zeros((K, K))
for s in range(K):
    if alpha_by_state[s]:
        state_mean_alpha[s] = np.mean(alpha_by_state[s], axis=0)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    state_mean_alpha,
    ax=ax,
    xticklabels=STATE_LABELS,
    yticklabels=STATE_LABELS,
    cmap="YlOrRd",
    cbar_kws={"label": "mean αₖ"},
    linewidths=0.3,
    linecolor="white",
)
ax.set_xlabel("Latent state", fontsize=10)
ax.set_ylabel("Image latent state (mean over main images)", fontsize=9)
ax.set_title("Recovered α  [main images, averaged by latent state]", fontsize=10)
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=7, rotation=0)
plt.tight_layout()
out = plots_dir / f"plot_01_alpha_{cfg.RUN_TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# Plot 02 — scatter panels
# ---------------------------------------------------------------------------
panels = [
    (pred_train,     emp_train,     ar_train,     "Train\n(main × train tasks)"),
    (pred_stim,      emp_stim,      ar_stim,      "Stim-gen\n(probe × train tasks)"),
]
if has_val:
    panels += [
        (pred_val_main,  emp_val_main,  ar_val_main,  "Held-out val\n(main × val tasks)"),
        (pred_val_probe, emp_val_probe, ar_val_probe, "Joint\n(probe × val tasks)"),
    ]

n_panels = len(panels)
fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), squeeze=False)

for ax, (pred, emp, arity_arr, title) in zip(axes[0], panels):
    _scatter_panel(ax, pred, emp, arity_arr, f"{title}\n[{cfg.RUN_TAG}]")

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

axes[0].plot(epochs, result.train_nlls, color="#E76F51")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("NLL")
axes[0].set_title(f"NLL  [{cfg.RUN_TAG}]")

axes[1].plot(epochs, result.train_mses, color="#E76F51")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cMSE")
axes[1].set_title(f"cMSE  [{cfg.RUN_TAG}]")

sns.despine()
plt.tight_layout()
out = plots_dir / f"plot_03_curves_{cfg.RUN_TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
