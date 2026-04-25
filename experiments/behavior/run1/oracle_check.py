"""
oracle_check.py  (run1)
-----------------------
Two-stage Dirichlet upper-bound check on probe-image human data
for the combined run0 + run1 dataset (up to 4-way conjunctions).

Stage 3 — Per-image unconstrained Dirichlet MLE  ("DLBT on steroids")
    For each probe image, directly optimise K=16 free Dirichlet parameters
    to maximise the multinomial log-likelihood of observed choice counts
    across ALL eligible tasks.  No metadata structure assumed — this is the
    tightest possible Dirichlet upper bound.

Stage 4 — Held-out task generalisation
    Same optimisation, but fit only on TRAIN_TASKS observations.  Evaluated
    on VAL_TASKS to measure how well the fitted Dirichlet extrapolates to
    unseen task conjunctions (including 4-way tasks).

Decision rule: original argmax (τ=0) throughout — no threshold correction.

Usage:
    cd <repo root>
    python experiments/behavior/run1/oracle_check.py
"""

import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_RUN1_DIR = Path(__file__).parent
_RUN0_DIR = _RUN1_DIR.parent / "run0"

sys.path.insert(0, str(_RUN1_DIR / "01_fit"))
sys.path.insert(0, str(_RUN0_DIR))

import config as cfg           # run1/01_fit/config.py  (TRAIN_TASKS, VAL_TASKS, …)
from preprocess import filter_assignments, aggregate_counts

from dlbt.constants import K
from dlbt.data.task import get_task

try:
    import torch
    from torch.distributions import Dirichlet as TorchDirichlet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available — stages 3 & 4 will be skipped.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MC        = 2000
N_MC_FIT    = 2000
N_STEPS_FIT = 5000
LR_FIT      = 0.05
TEMPERATURE = 50.0
RNG         = np.random.default_rng(42)

PLOTS_DIR = _RUN1_DIR / "01_fit" / "results" / "plots" / "oracle"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


# ---------------------------------------------------------------------------
# Ground-truth state helpers (from metadata)
# ---------------------------------------------------------------------------
from dlbt.constants import X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH

_cont_meta: dict = {}
with open(cfg.METADATA) as _f:
    for _line in _f:
        _line = _line.strip()
        if not _line:
            continue
        _rec = json.loads(_line)
        _cont_meta[_rec["id"]] = _rec["z"]


def _gt_state_idx(uid: str) -> int:
    z = _cont_meta.get(uid, {})
    return (int(z.get("pos_xy",      [0])[0] > X_THRESHOLD) * 8 +
            int(z.get("transparency", 0)     > TRANSP_THRESH) * 4 +
            int(z.get("glossiness",   0)     > GLOSS_THRESH)  * 2 +
            int(z.get("scale",        0)     > SCALE_THRESH))


def _gt_state_label(uid: str) -> str:
    z = _cont_meta.get(uid, {})
    return (("R"  if z.get("pos_xy", [0])[0] > X_THRESHOLD  else "L")  + " " +
            ("Tr" if z.get("transparency", 0) > TRANSP_THRESH else "Op") + " " +
            ("Gl" if z.get("glossiness",   0) > GLOSS_THRESH  else "Mt") + " " +
            ("Lg" if z.get("scale",        0) > SCALE_THRESH  else "Sm"))


def _state_col_label(k: int) -> str:
    return (("R"  if (k >> 3) & 1 else "L")  + "\n" +
            ("Tr" if (k >> 2) & 1 else "Op") + "\n" +
            ("Gl" if (k >> 1) & 1 else "Mt") + "\n" +
            ("Lg" if  k & 1       else "Sm"))


STATE_LABELS = [_state_col_label(k) for k in range(K)]


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
THUMB = 128


def _load_thumb(ref):
    try:
        img = PILImage.open(ref.path).convert("RGB").resize(
            (THUMB, THUMB), PILImage.LANCZOS)
        return np.array(img)
    except Exception:
        return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)


def _add_thumbs(ax, refs, zoom=0.18):
    n = len(refs)
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    for i, ref in enumerate(refs):
        oi = OffsetImage(_load_thumb(ref), zoom=zoom)
        oi.image.axes = ax
        ax.add_artist(AnnotationBbox(
            oi, (0.5, i), xycoords="data",
            frameon=False, pad=0, box_alignment=(0.5, 0.5)))

# ---------------------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------------------
print("Loading behavioural data (run0 + run1)...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
df_filtered, diag = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
_eligible_names  = set(cfg.TRAIN_TASKS + cfg.VAL_TASKS)
_eligible_beh_id = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in _eligible_names}
ds_full, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _eligible_beh_id,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)
print(f"  {len(ds_full.df)} (uid, task) cells  |  "
      f"{ds_full.df['uid'].nunique()} images  |  "
      f"{ds_full.df['task_name'].nunique()} tasks")

df = ds_full.df[ds_full.df["uid"].isin(probe_uids)].copy()
df["emp_p"]  = df["count_1"] / (df["count_0"] + df["count_1"])
df["totals"] = df["count_0"] + df["count_1"]

# Restrict to eligible tasks (respects MIN_TASK_ASSIGNMENTS)
eligible = set(cfg.TRAIN_TASKS) | set(cfg.VAL_TASKS)
df = df[df["task_name"].isin(eligible)].copy()

print(f"  Probe-only (eligible tasks): {len(df)} cells  |  "
      f"{df['uid'].nunique()} images  |  "
      f"{df['task_name'].nunique()} tasks")

_uids       = df["uid"].values
_task_names = df["task_name"].values
_emp_p      = df["emp_p"].values
_valid      = np.isfinite(_emp_p) & (df["totals"].values > 0)
_delta_us   = np.stack([get_task(t).delta_u for t in _task_names])   # [N, K]

probe_uid_list = sorted(probe_uids)

# Per-image task data  uid -> [(delta_u [K], count_0, count_1), ...]
uid_task_data: dict = {uid: [] for uid in probe_uid_list}
for _, row in df.iterrows():
    uid = row["uid"]
    c0, c1 = int(row["count_0"]), int(row["count_1"])
    if c0 + c1 > 0:
        uid_task_data[uid].append(
            (get_task(row["task_name"]).delta_u, c0, c1))

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def mc_p_yes(alpha: np.ndarray, delta_u: np.ndarray,
             rng: np.random.Generator, n: int = N_MC) -> float:
    b = rng.dirichlet(alpha, size=n)
    return float((b @ delta_u > 0).mean())


def evaluate_model(alphas_by_uid: dict, label: str) -> dict:
    preds = np.zeros(len(_uids))
    for i, (uid, du) in enumerate(zip(_uids, _delta_us)):
        preds[i] = mc_p_yes(alphas_by_uid[uid], du, RNG)
    mask = _valid
    mse    = float(np.mean((preds[mask] - _emp_p[mask]) ** 2))
    rho, _ = spearmanr(preds[mask], _emp_p[mask])
    dpa    = float(np.mean(
        (preds[mask] > 0.5) == (_emp_p[mask] > 0.5)))
    return dict(label=label, mse=mse, rho=rho, dpa=dpa,
                preds=preds, emp_p=_emp_p, valid=mask,
                task_names=_task_names)


# ---------------------------------------------------------------------------
# Optimisation: Adam per image
# ---------------------------------------------------------------------------

def fit_alpha(uid: str, task_data: list = None,
              n_steps: int = N_STEPS_FIT) -> np.ndarray:
    if task_data is None:
        task_data = uid_task_data[uid]
    if not task_data:
        return np.ones(K, dtype=np.float64)

    delta_us_t = torch.tensor(
        np.stack([du for du, c0, c1 in task_data]),
        dtype=torch.float32)
    counts_t = torch.tensor(
        [[c0, c1] for du, c0, c1 in task_data],
        dtype=torch.float32)

    log_alpha = torch.full((K,), np.log(10.0), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([log_alpha], lr=LR_FIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-4)

    for _ in range(n_steps):
        optimizer.zero_grad()
        alpha  = torch.exp(log_alpha).clamp(min=1e-3)
        b      = TorchDirichlet(alpha).rsample((N_MC_FIT,))   # [n_mc, K]
        logits = b @ delta_us_t.T                              # [n_mc, T]
        p_yes  = torch.sigmoid(TEMPERATURE * logits).mean(0)  # τ=0
        probs  = torch.stack([1 - p_yes, p_yes], dim=1).clamp(1e-7, 1 - 1e-7)
        nll    = -(counts_t * probs.log()).sum()
        nll.backward()
        optimizer.step()
        scheduler.step()

    return torch.exp(log_alpha).detach().clamp(min=1e-6).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 3 — Unconstrained MLE on ALL tasks
# ---------------------------------------------------------------------------
print("\n[Stage 3] Per-image Dirichlet MLE (all tasks)...")

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    fitted_all = None
    res3 = None
else:
    fitted_all: dict = {}
    for uid in probe_uid_list:
        print(f"  Fitting {uid} ...", end="  ", flush=True)
        fitted_all[uid] = fit_alpha(uid)
        print("done")

    res3 = evaluate_model(fitted_all, "MLE all tasks")
    print(f"\nStage 3:  MSE={res3['mse']:.4f}  ρ={res3['rho']:.3f}  DPA={res3['dpa']:.3f}")


# ---------------------------------------------------------------------------
# Stage 4 — MLE on TRAIN_TASKS only, evaluate on VAL_TASKS
# ---------------------------------------------------------------------------
print("\n[Stage 4] Held-out task generalisation (fit on TRAIN_TASKS only)...")

train_task_set = set(cfg.TRAIN_TASKS)
val_task_set   = set(cfg.VAL_TASKS)

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    fitted_train = None
    res4 = None
else:
    uid_train_data: dict = {uid: [] for uid in probe_uid_list}
    for _, row in df.iterrows():
        if row["task_name"] not in train_task_set:
            continue
        uid = row["uid"]
        c0, c1 = int(row["count_0"]), int(row["count_1"])
        if c0 + c1 > 0:
            uid_train_data[uid].append(
                (get_task(row["task_name"]).delta_u, c0, c1))

    fitted_train: dict = {}
    for uid in probe_uid_list:
        n_tr = len(uid_train_data[uid])
        print(f"  Fitting {uid}  ({n_tr} train tasks) ...", end="  ", flush=True)
        fitted_train[uid] = fit_alpha(uid, task_data=uid_train_data[uid])
        print("done")

    res4 = evaluate_model(fitted_train, "MLE train tasks only")

    def _split_metrics(result, task_set):
        mask = np.array([t in task_set for t in result["task_names"]]) & result["valid"]
        if mask.sum() < 2:
            return float("nan"), float("nan"), float("nan")
        p, e = result["preds"][mask], result["emp_p"][mask]
        mse = float(np.mean((p - e) ** 2))
        rho, _ = spearmanr(p, e)
        dpa = float(np.mean((p > 0.5) == (e > 0.5)))
        return mse, rho, dpa

    trn_mse, trn_rho, trn_dpa = _split_metrics(res4, train_task_set)
    val_mse, val_rho, val_dpa = _split_metrics(res4, val_task_set)
    print(f"\nStage 4:")
    print(f"  train tasks: MSE={trn_mse:.4f}  ρ={trn_rho:.3f}  DPA={trn_dpa:.3f}")
    print(f"  VAL tasks:   MSE={val_mse:.4f}  ρ={val_rho:.3f}  DPA={val_dpa:.3f}  ← generalisation")
    print(f"  (Stage 3 upper bound: MSE={res3['mse']:.4f}  ρ={res3['rho']:.3f})")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Image refs (for thumbnails)
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs_list  = image_refs_as_list(refs_dict)
refs_by_uid = {r.uid: r for r in refs_list}

# ---------------------------------------------------------------------------
# Alpha heatmap helper
# ---------------------------------------------------------------------------

def _plot_alpha_heatmap(alphas: dict, probe_uid_list: list,
                        subtitle: str, out_path: Path) -> None:
    """Single heatmap: fitted α for all probe images."""
    sorted_uids = sorted(probe_uid_list, key=_gt_state_idx)
    probe_refs  = [refs_by_uid[u] for u in sorted_uids if u in refs_by_uid]
    sorted_uids = [r.uid for r in probe_refs]
    n_rows = len(sorted_uids)

    gt_states  = [_gt_state_idx(u)   for u in sorted_uids]
    row_labels = [_gt_state_label(u) for u in sorted_uids]
    mat        = np.stack([alphas[u] for u in sorted_uids])

    fig = plt.figure(figsize=(14, max(5, n_rows * 0.55)))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[1.2, 10, 0.4], wspace=0.18)
    ax_thumb = fig.add_subplot(gs[0])
    ax_heat  = fig.add_subplot(gs[1])
    ax_cbar  = fig.add_subplot(gs[2])

    _add_thumbs(ax_thumb, probe_refs, zoom=0.18)
    ax_thumb.set_title("image", fontsize=8)
    ax_thumb.set_ylabel("probe image  (sorted by ground-truth state)", fontsize=8)

    im = ax_heat.imshow(mat, aspect="auto", cmap="YlOrRd",
                        extent=[-0.5, K - 0.5, n_rows - 0.5, -0.5])
    ax_heat.set_xticks(range(K))
    ax_heat.set_xticklabels(STATE_LABELS, rotation=90, fontsize=6, va="top")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=7)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)
    ax_heat.set_xlabel("latent state", labelpad=4)
    ax_heat.set_title(subtitle, fontsize=9)
    for row_i, k in enumerate(gt_states):
        ax_heat.add_patch(plt.Rectangle(
            (k - 0.5, row_i - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=2, zorder=5))
    fig.colorbar(im, cax=ax_cbar, label="α_k")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


ARITY_COLORS = {1: "#457B9D", 2: "#E76F51", 3: "#9B5DE5", 4: "#43AA8B"}


def _task_color(name):
    return ARITY_COLORS.get(_arity(name), "gray")


def _plot_per_task(result: dict, out_path: Path,
                   val_tasks: set = None) -> None:
    """Per-task scatter: oracle vs human P(yes)."""
    all_tasks = sorted(df["task_name"].unique(),
                       key=lambda t: (_arity(t), t))
    n_cols = 12
    n_rows = int(np.ceil(len(all_tasks) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, task_name in enumerate(all_tasks):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        is_val = val_tasks is not None and task_name in val_tasks
        mask = (result["task_names"] == task_name) & result["valid"]
        x, y = result["preds"][mask], result["emp_p"][mask]
        col_c = _task_color(task_name)

        ax.scatter(x, y, s=12, alpha=0.8, color=col_c, linewidths=0)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, zorder=0)
        rho_t, _ = spearmanr(x, y) if len(x) > 2 else (float("nan"), None)
        ax.text(0.05, 0.95, f"ρ={rho_t:.2f}", transform=ax.transAxes,
                fontsize=6, va="top", color=col_c)

        title_str = task_name.replace("_and_", "&").replace("_", "/")
        if is_val:
            title_str += " ★"
        ax.set_title(title_str, fontsize=5.5, pad=2,
                     color="#CC2222" if is_val else "black")
        if is_val:
            for spine in ax.spines.values():
                spine.set_edgecolor("#CC2222")
                spine.set_linewidth(1.4)
            ax.set_facecolor("#FFF5F5")
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect="equal")
        ax.tick_params(labelsize=5)

    for idx in range(len(all_tasks), n_rows * n_cols):
        r2, c2 = divmod(idx, n_cols)
        axes[r2, c2].set_visible(False)

    fig.suptitle(
        f"{result['label']}  |  "
        f"MSE={result['mse']:.4f}  ρ={result['rho']:.3f}  DPA={result['dpa']:.3f}\n"
        "colour: 1-way (blue)  2-way (orange)  3-way (purple)  4-way (green)  "
        "★ = held-out",
        fontsize=8, y=1.01)
    fig.text(0.5,  -0.01, "Oracle P(yes)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5,  "Human P(yes)",  va="center",
             rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_summary(result: dict, train_tasks: set,
                  val_tasks: set, out_path: Path) -> None:
    """Two-panel summary scatter: train (left) vs val (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    splits = [
        ("train tasks",    train_tasks, axes[0], "#457B9D"),
        ("held-out tasks", val_tasks,   axes[1], "#CC2222"),
    ]
    for split_label, task_set, ax, base_col in splits:
        mask = (np.array([t in task_set for t in result["task_names"]])
                & result["valid"])
        x, y = result["preds"][mask], result["emp_p"][mask]
        tasks_here = result["task_names"][mask]

        unique_tasks = sorted(set(tasks_here), key=lambda t: (_arity(t), t))
        for tname in unique_tasks:
            tm = tasks_here == tname
            ax.scatter(x[tm], y[tm], s=14, alpha=0.7,
                       color=_task_color(tname), linewidths=0, zorder=3)

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, zorder=0)

        if len(x) > 2:
            rho_s, _ = spearmanr(x, y)
            mse_s    = float(np.mean((x - y) ** 2))
            dpa_s    = float(np.mean((x > 0.5) == (y > 0.5)))
        else:
            rho_s = mse_s = dpa_s = float("nan")

        ax.text(0.05, 0.95, f"ρ={rho_s:.3f}",    transform=ax.transAxes,
                fontsize=9, va="top", fontweight="bold", color=base_col)
        ax.text(0.05, 0.86, f"MSE={mse_s:.4f}",  transform=ax.transAxes,
                fontsize=9, va="top", color=base_col)
        ax.text(0.05, 0.77, f"DPA={dpa_s:.3f}",  transform=ax.transAxes,
                fontsize=9, va="top", color=base_col)
        ax.text(0.05, 0.68, f"n={mask.sum()}",    transform=ax.transAxes,
                fontsize=8, va="top", color="gray")

        ax.set_title(split_label, fontsize=10,
                     color=base_col, fontweight="bold")
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect="equal")
        ax.tick_params(labelsize=8)

    fig.text(0.5,  0.01, "Oracle P(yes)",  ha="center", fontsize=10)
    fig.text(0.01, 0.5,  "Human P(yes)",   va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle(f"Oracle summary — {result['label']}", fontsize=9, y=1.02)
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Produce plots
# ---------------------------------------------------------------------------
print("\nPlotting...")

if res3 is not None:
    _plot_per_task(res3, PLOTS_DIR / "oracle_s3_per_task.png")

if res4 is not None:
    _plot_per_task(res4, PLOTS_DIR / "oracle_s4_per_task.png",
                   val_tasks=val_task_set)
    _plot_summary(res4, train_task_set, val_task_set,
                  PLOTS_DIR / "oracle_s4_summary.png")

if fitted_all is not None:
    _plot_alpha_heatmap(fitted_all, probe_uid_list,
                        "Stage 3 — all tasks (upper bound)",
                        PLOTS_DIR / "oracle_alpha_heatmap_s3.png")
if fitted_train is not None:
    _plot_alpha_heatmap(fitted_train, probe_uid_list,
                        "Stage 4 — train tasks only (generalisation)",
                        PLOTS_DIR / "oracle_alpha_heatmap_s4.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
if res3 is not None:
    all_tasks = sorted(df["task_name"].unique(), key=lambda t: (_arity(t), t))
    print(f"\n{'='*75}")
    print(f"{'task':<45}  {'N':>4}  {'S3 ρ':>7}  {'S4 ρ':>7}")
    print("=" * 75)
    for task_name in all_tasks:
        mask = (_task_names == task_name) & _valid
        if mask.sum() < 2:
            continue
        tag = " ★" if task_name in val_task_set else "  "
        n_w = _arity(task_name)
        y   = _emp_p[mask]
        rhos = []
        for res in [res3, res4]:
            if res is None:
                rhos.append(float("nan"))
                continue
            x = res["preds"][mask]
            r, _ = spearmanr(x, y) if len(x) > 2 else (float("nan"), None)
            rhos.append(r)
        print(f"[{n_w}w]{tag} {task_name:<43}  {mask.sum():>4}  "
              + "  ".join(f"{r:>7.3f}" for r in rhos))

    print(f"\n{'='*75}")
    print("Overall:")
    for res in [res3, res4]:
        if res is not None:
            print(f"  {res['label']:<30}  "
                  f"MSE={res['mse']:.4f}  ρ={res['rho']:.3f}  DPA={res['dpa']:.3f}")
