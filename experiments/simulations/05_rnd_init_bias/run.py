"""
experiments/simulations/05_rnd_init_bias/run.py
Diagnoses why the randomly-initialised DlbtAgent slightly beats the random
guesser in cMSE−NF before any training.

Hypothesis: real CLIP features have latent structure that the random linear
mapper accidentally exploits, producing systematically biased predictions
(mean p̃ ≠ 0.5).  We isolate the contribution of each component through the
conditions below.

Data
----
Loads ground-truth probe matrix (true_matrix, probe_noise_floor) and the
frozen CLIP feature cache from an existing run1/02_data_efficiency pkl.

Conditions  (each → one scatter p̂_xt vs p̃_xt)
----------
Figure 1 — main diagnostic grid (2 × 4)
  [0] Random guesser                  p̃ = 0.5 for all cells
  [1] Rand-init DLBT  (norm)          02 settings, NORMALIZED_UTILITY=True
  [2] Rand-init DLBT  (no norm)       same but NORMALIZED_UTILITY=False
  [3] Rand-init DLBT  (random CLIP)   real mapper, random Gaussian input
  [4] Uniform α = 0.5                 Dir(0.5,...,0.5) for all images
  [5] Uniform α = 1                   Dir(1,...,1)     — classic uniform
  [6] Uniform α = 5
  [7] Uniform α = 10

Figure 2 — arity stratification (1 × 4)
  Same rand-init DLBT (norm) agent; scatter restricted per task arity.

Run from repo root:
    python experiments/simulations/05_rnd_init_bias/run.py [--pkl PATH]
"""

import argparse
import sys
import warnings
from pathlib import Path
import pickle

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Repo root on sys.path so dlbt imports work regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from dlbt.agents.dlbt import DlbtAgent
from dlbt.constants import K
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to run1/02 coverage_sweep_*.pkl. "
                         "Default: auto-discover frozen_coverage_norm pkl.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants (mirror run1/02 config)
# ---------------------------------------------------------------------------
SEEDS          = [42, 43, 44, 45, 46]
SEED           = 42
N_MC           = 1000
MAPPER_HIDDEN  = None
INIT_SEED      = 0
INIT_ALPHA_LOW = 0.6
INIT_ALPHA_HIGH= 0.7
CACHE_PATH     = _REPO_ROOT / "stimuli/imgs/clip_rn50_features_v2.pt"
METADATA_PATH  = _REPO_ROOT / "stimuli/imgs/metadata.jsonl"

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load run1/02 results pkl
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_path = Path(args.pkl)
else:
    _results_dir = (_REPO_ROOT / "experiments/behavior/run1"
                    / "02_data_efficiency/results")
    candidates = sorted(_results_dir.glob("coverage_sweep_frozen*.pkl"))
    if not candidates:
        candidates = sorted(_results_dir.glob("coverage_sweep_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No coverage_sweep_*.pkl found in {_results_dir}. "
            "Pass --pkl explicitly.")
    pkl_path = candidates[0]

print(f"Loading from: {pkl_path.name}")
with open(pkl_path, "rb") as f:
    summary = pickle.load(f)

true_matrix       = summary["true_matrix"]          # [n_probe × n_tasks]
probe_noise_floor = summary["probe_noise_floor"]
all_tasks         = summary["all_tasks_ordered"]
probe_uids        = summary["probe_uids_ordered"]
n_probe, n_tasks  = true_matrix.shape
print(f"  Probe matrix: {n_probe} × {n_tasks}  NF={probe_noise_floor:.5f}")

# ---------------------------------------------------------------------------
# Image refs for probe images
# ---------------------------------------------------------------------------
refs_dict   = load_image_refs(METADATA_PATH)
refs_by_uid = {r.uid: r for r in image_refs_as_list(refs_dict)}
probe_refs  = [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid]
assert len(probe_refs) == n_probe, "Probe ref count mismatch"

# ---------------------------------------------------------------------------
# CLIP feature cache
# ---------------------------------------------------------------------------
_tmp = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                 device=device, mapper_hidden=MAPPER_HIDDEN)
_tmp.load_cache(str(CACHE_PATH))
frozen_clip = {uid: feat.clone() for uid, feat in _tmp._cache.items()}
del _tmp
print(f"CLIP cache loaded ({len(frozen_clip)} images).")

# ---------------------------------------------------------------------------
# Task arity helper
# ---------------------------------------------------------------------------
def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1

task_arities = np.array([_arity(t) for t in all_tasks])   # [n_tasks]

# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------

def _init_dlbt(normalize_utility: bool = True) -> DlbtAgent:
    """Random-init DlbtAgent with frozen CLIP cache (mirrors 02/run.py)."""
    torch.manual_seed(SEED)
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = N_MC,
        device            = device,
        mapper_hidden     = MAPPER_HIDDEN,
        normalize_utility = normalize_utility,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _linear = agent.mapper[0]   # Linear(1024, K) with MAPPER_HIDDEN=None
    rng  = np.random.default_rng(INIT_SEED)
    a    = rng.uniform(INIT_ALPHA_LOW, INIT_ALPHA_HIGH,
                       size=(_linear.bias.shape[0],)).astype(np.float32)
    b    = np.log(np.exp(a) - 1.0)
    with torch.no_grad():
        _linear.bias.copy_(torch.from_numpy(b).to(device))
    agent.eval()
    return agent


class _UniformBTAgent(DlbtAgent):
    """DlbtAgent with fixed uniform Dirichlet beliefs (all α = alpha_val)."""
    def __init__(self, alpha_val: float, **kwargs):
        super().__init__(**kwargs)
        self._alpha_val = float(alpha_val)

    def get_alpha(self, image_refs):
        return torch.full(
            (len(image_refs), K), self._alpha_val, device=self.device)


def _init_uniform(alpha_val: float) -> _UniformBTAgent:
    torch.manual_seed(SEED)
    agent = _UniformBTAgent(
        alpha_val         = alpha_val,
        freeze_encoder    = True,
        n_mc_samples      = N_MC,
        device            = device,
        mapper_hidden     = MAPPER_HIDDEN,
        normalize_utility = True,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Probe matrix helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pred_matrix(agent: DlbtAgent) -> np.ndarray:
    """Compute [n_probe × n_tasks] predicted P(right) for all probe cells."""
    pred = np.full((n_probe, n_tasks), np.nan)
    for j, task_name in enumerate(all_tasks):
        task       = get_task(task_name)
        probs      = agent.choice_probs(probe_refs, task)[:, 1].cpu().numpy()
        pred[:, j] = probs
    return pred


# ---------------------------------------------------------------------------
# Compute all conditions
# ---------------------------------------------------------------------------
print("\nComputing conditions...")

# [0] Random guesser
pred_random = np.full((n_probe, n_tasks), 0.5)
print("  [0] Random guesser done.")

# [1] Rand-init DLBT, NORMALIZED_UTILITY=True
agent_norm   = _init_dlbt(normalize_utility=True)
pred_dlbt_norm = _pred_matrix(agent_norm)
del agent_norm
print("  [1] Rand-init DLBT (norm) done.")

# [2] Rand-init DLBT, NORMALIZED_UTILITY=False  (no prior correction)
agent_nonorm   = _init_dlbt(normalize_utility=False)
pred_dlbt_nonorm = _pred_matrix(agent_nonorm)
del agent_nonorm
print("  [2] Rand-init DLBT (no norm) done.")

# [3] Rand-init DLBT, RANDOM CLIP features  (same mapper, Gaussian input)
agent_rndclip = _init_dlbt(normalize_utility=True)
rng_clip      = np.random.default_rng(SEED + 1)
agent_rndclip._cache = {
    uid: torch.from_numpy(
        rng_clip.standard_normal(1024).astype(np.float32)
    )
    for uid in frozen_clip
}
pred_dlbt_rndclip = _pred_matrix(agent_rndclip)
del agent_rndclip
print("  [3] Rand-init DLBT + random CLIP done.")

# [4-7] Uniform initialization at various concentrations
uniform_alphas = [0.5, 1.0, 5.0, 10.0]
pred_uniform   = {}
for av in uniform_alphas:
    agent_u        = _init_uniform(av)
    pred_uniform[av] = _pred_matrix(agent_u)
    del agent_u
    print(f"  Uniform α={av} done.")

print("\nAll conditions computed.")

# ---------------------------------------------------------------------------
# Scatter helper — style mirrors run1/01_fit analysis.py
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}


def _scatter(ax, true_mat, pred_mat, title,
             noise_floor=probe_noise_floor,
             task_arities_sub=None,   # [n_tasks] int array for per-cell colouring
             single_color=None):      # override: one colour for all points
    """
    Scatter plot following 01_fit conventions.

    x-axis : Predicted P(yes)  = p̃_xt
    y-axis : Human P(yes)      = p̂_xt

    Points coloured by task arity when task_arities_sub is provided.
    """
    valid = ~np.isnan(true_mat) & ~np.isnan(pred_mat)

    # Build per-point arity colour array
    if single_color is not None:
        colors_pt = single_color
    elif task_arities_sub is not None:
        # Expand task_arities_sub to match (n_probe × n_tasks) layout
        # valid is a boolean 2-D mask; we need the arity of each valid cell's task
        _, n_t   = true_mat.shape
        n_p      = true_mat.shape[0]
        arity_grid = np.broadcast_to(task_arities_sub[np.newaxis, :], (n_p, n_t))
        colors_pt = np.array(
            [ARITY_COLOR.get(a, "#888") for a in arity_grid[valid]]
        )
    else:
        colors_pt = "#4C72B0"

    # x = predicted, y = human  (01_fit convention)
    x = pred_mat[valid]
    y = true_mat[valid]

    ax.scatter(x, y, c=colors_pt, alpha=0.5, s=3, linewidths=0, rasterized=True)

    # Reference diagonal (same style as 01_fit pooled: dashed gray, lw=1.2)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)

    # Vertical reference at predicted = 0.5 (no-bias marker)
    ax.axvline(0.5, color="gray", lw=0.8, ls=":", zorder=0)

    # Statistics
    rho, _  = spearmanr(x, y)
    cmse_nf = float(np.mean((x - y) ** 2)) - noise_floor
    bias    = float(np.mean(x)) - 0.5        # bias in predicted values

    # Title with metrics (01_fit pattern)
    ax.set_title(
        f"{title}\n(−NF)={cmse_nf:+.4f}   ρ={rho:.3f}   bias={bias:+.4f}",
        fontsize=8, pad=4,
    )

    ax.set_xlabel("Predicted P(yes)", fontsize=9)
    ax.set_ylabel("Human P(yes)",     fontsize=9)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal", adjustable="box")
    sns.despine(ax=ax, trim=True)


# ===========================================================================
# Figure 1 — Main diagnostic grid (2 × 4)
# ===========================================================================
fig1, axes1 = plt.subplots(2, 4, figsize=(13, 7.0),
                            gridspec_kw={"hspace": 0.55, "wspace": 0.35})

conditions = [
    (axes1[0, 0], pred_random,       "Random guesser"),
    (axes1[0, 1], pred_dlbt_norm,    "Rand-init DLBT\n(prior norm.)"),
    (axes1[0, 2], pred_dlbt_nonorm,  "Rand-init DLBT\n(no prior norm.)"),
    (axes1[0, 3], pred_dlbt_rndclip, "Rand-init DLBT\n(random CLIP)"),
    *[(axes1[1, i], pred_uniform[av], f"Uniform α = {av}")
      for i, av in enumerate(uniform_alphas)],
]

for ax, pred, title in conditions:
    _scatter(ax, true_matrix, pred, title, task_arities_sub=task_arities)

# Arity legend (upper left of first panel — same as 01_fit)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=c, markersize=5, label=f"{a}-way")
           for a, c in ARITY_COLOR.items()
           if a in set(task_arities)]
axes1[0, 0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")

fig1.suptitle(
    "Bias diagnosis — randomly initialised DLBT vs baselines",
    fontsize=11, y=1.01,
)
out1 = PLOTS_DIR / "fig1_main_diagnostic.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"\nSaved: {out1}")

# ===========================================================================
# Figure 2 — Arity stratification (2 × 4): norm vs no-norm rows
# ===========================================================================
arities = [1, 2, 3, 4]
fig2, axes2 = plt.subplots(2, 4, figsize=(13, 7.0),
                            gridspec_kw={"hspace": 0.55, "wspace": 0.35})

rows_cfg = [
    (axes2[0], pred_dlbt_norm,   "prior norm."),
    (axes2[1], pred_dlbt_nonorm, "no prior norm."),
]

for ax_row, pred_full, row_label in rows_cfg:
    for ax, arity in zip(ax_row, arities):
        cols  = np.where(task_arities == arity)[0]
        n_col = len(cols)
        if n_col == 0:
            ax.set_visible(False)
            continue
        _scatter(
            ax,
            true_matrix[:, cols],
            pred_full[:, cols],
            f"{arity}-way tasks  [{row_label}]  (n={n_col})",
            task_arities_sub=np.full(n_col, arity, dtype=int),
            single_color=ARITY_COLOR[arity],
        )

fig2.suptitle(
    "Arity stratification — rand-init DLBT\n"
    "Top: prior normalisation   Bottom: no prior normalisation",
    fontsize=11, y=1.01,
)
out2 = PLOTS_DIR / "fig2_arity_stratified.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

# ===========================================================================
# Print summary table
# ===========================================================================
print("\n" + "=" * 65)
print(f"{'Condition':<35}  {'mean p̃':>8}  {'bias':>8}  {'cMSE-NF':>10}  {'ρ':>6}")
print("-" * 65)

def _stats(pred, true=true_matrix, nf=probe_noise_floor):
    valid = ~np.isnan(true) & ~np.isnan(pred)
    x, y  = true[valid], pred[valid]
    rho, _= spearmanr(x, y)
    return float(np.mean(y)), float(np.mean(y))-0.5, float(np.mean((x-y)**2))-nf, float(rho)

def _row_stats(pred_sub, true_sub):
    valid  = ~np.isnan(true_sub) & ~np.isnan(pred_sub)
    x, y   = true_sub[valid], pred_sub[valid]
    rho, _ = spearmanr(x, y)
    mean_p = float(np.mean(y))
    return mean_p, mean_p - 0.5, float(np.mean((x - y)**2)) - probe_noise_floor, float(rho)

# (name, pred_sub, true_sub)
rows = [
    ("Random guesser",               pred_random,        true_matrix),
    ("Rand-init DLBT (norm)",        pred_dlbt_norm,     true_matrix),
    ("Rand-init DLBT (no norm)",     pred_dlbt_nonorm,   true_matrix),
    ("Rand-init DLBT (random CLIP)", pred_dlbt_rndclip,  true_matrix),
    *[(f"Uniform α={av}",            pred_uniform[av],   true_matrix)
      for av in uniform_alphas],
]
for arity in arities:
    cols = np.where(task_arities == arity)[0]
    if len(cols):
        rows.append((f"Rand-init DLBT arity={arity}",
                     pred_dlbt_norm[:, cols],
                     true_matrix[:, cols]))

for name, pred_sub, true_sub in rows:
    mean_p, bias, cmse, rho = _row_stats(pred_sub, true_sub)
    print(f"{name:<35}  {mean_p:>8.4f}  {bias:>+8.4f}  {cmse:>10.5f}  {rho:>+6.3f}")

print("=" * 65)
print(f"\nAll plots saved to {PLOTS_DIR}")

# ===========================================================================
# Figure 3 — SEU distribution by arity (2 rows × 4 cols: unnorm / norm)
# ===========================================================================
# For each (arity, normalization) combination we pool raw SEU scores
#   s_t(b) = b · ΔU_t  (b ~ Dir(α), one sample per MC draw per image per task)
# and show the distribution, marking mean, median, and the zero-threshold.
# This illustrates that normalization centres the mean at 0 but leaves the
# distribution skewed, so the median stays negative for arity > 1.

from scipy.stats import gaussian_kde  # noqa: E402 (late import, already in env)

N_MC_SEU    = 2000   # Dirichlet samples per (image, task)
N_IMGS_SEU  = 30     # probe images to use
N_TASKS_SEU = 8      # tasks per arity (capped; use all if fewer)

rng_fig3  = np.random.default_rng(SEED + 99)
_img_idx  = rng_fig3.choice(n_probe, size=min(N_IMGS_SEU, n_probe), replace=False)
refs_seu  = [probe_refs[i] for i in _img_idx]


def _seu_pool(normalize_utility: bool, arity: int) -> np.ndarray:
    """
    Return a 1-D array of raw SEU scores pooled across a subset of images
    and tasks of the given arity.
    """
    torch.manual_seed(SEED)
    ag = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = N_MC_SEU,
        device            = device,
        mapper_hidden     = MAPPER_HIDDEN,
        normalize_utility = normalize_utility,
    )
    ag._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _lin = ag.mapper[0]
    rng_init = np.random.default_rng(INIT_SEED)
    a_init   = rng_init.uniform(INIT_ALPHA_LOW, INIT_ALPHA_HIGH,
                                size=(_lin.bias.shape[0],)).astype(np.float32)
    with torch.no_grad():
        _lin.bias.copy_(
            torch.from_numpy(np.log(np.exp(a_init) - 1.0)).to(device))
    ag.eval()

    task_names_ar = [t for t, ar in zip(all_tasks, task_arities) if ar == arity]
    if len(task_names_ar) > N_TASKS_SEU:
        task_names_ar = list(
            np.random.default_rng(SEED + arity).choice(
                task_names_ar, size=N_TASKS_SEU, replace=False))

    pool = []
    with torch.no_grad():
        alpha_t = ag.get_alpha(refs_seu)                    # [N_imgs, K]
        dist    = torch.distributions.Dirichlet(alpha_t)   # batched over images

        for tname in task_names_ar:
            task    = get_task(tname)
            delta_u = ag._delta_u(task)                     # [K]

            # samples: [N_MC_SEU, N_imgs, K]
            samples = dist.sample((N_MC_SEU,))
            seu     = (samples * delta_u[None, None, :]).sum(-1)  # [N_MC_SEU, N_imgs]
            pool.append(seu.cpu().numpy().ravel())

    return np.concatenate(pool) if pool else np.array([0.0])


print("\nComputing SEU distributions for Figure 3 ...")
seu_data = {}   # (normalize_utility, arity) → 1-D array
for norm in (False, True):
    for ar in arities:
        key = (norm, ar)
        seu_data[key] = _seu_pool(norm, ar)
        tag = "norm" if norm else "unnorm"
        print(f"  arity={ar} [{tag}]  n={len(seu_data[key]):,}  "
              f"mean={seu_data[key].mean():+.4f}  "
              f"median={np.median(seu_data[key]):+.4f}")

# --- plot ---
fig3, axes3 = plt.subplots(
    2, 4, figsize=(13, 6.5),
    gridspec_kw={"hspace": 0.55, "wspace": 0.35},
)

row_labels = ["no prior normalisation", "prior normalisation"]
row_norms  = [False, True]

for row_i, (norm, row_lbl) in enumerate(zip(row_norms, row_labels)):
    for col_i, ar in enumerate(arities):
        ax  = axes3[row_i, col_i]
        seu = seu_data[(norm, ar)]

        # KDE — fixed x-axis per row
        # unnorm: evaluate KDE on wider range to avoid edge artefacts, then trim display
        kde    = gaussian_kde(seu, bw_method="scott")
        if not norm:
            x_grid  = np.linspace(-1.1, 1.0, 500)
            xlim    = (-1.0, 1.0)
        else:
            x_grid  = np.linspace(-0.2, 0.2, 500)
            xlim    = (-0.2, 0.2)
        y_kde  = kde(x_grid)

        color = ARITY_COLOR[ar]
        ax.fill_between(x_grid, y_kde, alpha=0.25, color=color)
        ax.plot(x_grid, y_kde, color=color, lw=1.5)

        mu  = float(seu.mean())
        med = float(np.median(seu))

        ymax = y_kde.max()
        # zero threshold
        ax.axvline(0, color="gray", lw=1.2, ls="-",  zorder=2,
                   label="zero (threshold)")
        # mean
        ax.axvline(mu,  color="steelblue", lw=1.5, ls="--", zorder=3,
                   label=f"mean={mu:+.3f}")
        # median
        ax.axvline(med, color="firebrick",  lw=1.5, ls=":",  zorder=3,
                   label=f"median={med:+.3f}")

        ax.set_title(
            f"{ar}-way tasks  [{row_lbl}]\n"
            f"mean={mu:+.4f}   median={med:+.4f}",
            fontsize=8, pad=4,
        )
        ax.set_xlim(*xlim)
        ax.set_xlabel("SEU(A=right)", fontsize=9)
        ax.set_ylabel("Density",         fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax, trim=True)

        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")

fig3.suptitle(
    "SEU score distribution — random-init DLBT\n"
    "Prior normalisation centres the mean but leaves the distribution skewed "
    "(median ≠ 0 for arity > 1)",
    fontsize=10, y=1.02,
)
out3 = PLOTS_DIR / "fig3_seu_distribution.png"
fig3.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")
