"""
oracle_check.py
---------------
Two-stage Dirichlet upper-bound check on probe-image human data.

Stage 3 — Per-image unconstrained Dirichlet MLE  ("DLBT on steroids")
    For each probe image, directly optimise K=16 free Dirichlet parameters to
    maximise the multinomial log-likelihood of observed choice counts across
    ALL tasks.  No metadata structure assumed — this is the tightest possible
    Dirichlet upper bound.

Stage 4 — Held-out task generalisation
    Same optimisation, but fit only on TRAIN_TASKS observations.  Evaluated
    on VAL_TASKS to measure how well the fitted Dirichlet extrapolates to
    unseen task conjunctions.

Usage:
    cd <repo root>
    python experiments/behavior/run0/oracle_check.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent / "01_fit"))
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from preprocess import load_and_preprocess

from dlbt.constants import K, X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH
from dlbt.data.task import TASKS

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
N_MC        = 2000    # MC samples for final evaluation
RNG         = np.random.default_rng(42)
PLOTS_DIR   = cfg.RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Optimisation
N_MC_FIT    = 2000    # MC samples per gradient step
N_STEPS_FIT = 1000    # gradient steps per image
LR_FIT      = 0.05
TEMPERATURE = 50.0    # sigmoid sharpness for soft indicator

# ---------------------------------------------------------------------------
# Load + preprocess behavioural data
# ---------------------------------------------------------------------------
print("Loading behavioural data...")
ds_full, probe_uids, main_uids, diag = load_and_preprocess(
    cfg.BEHAVIOR_CSV,
    beh_id_to_task     = cfg.BEH_ID_TO_TASK,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds    = cfg.USE_TRIAL_KINDS,
    seed               = cfg.SEED,
)
print(f"  {len(ds_full)} (uid, task) cells  |  "
      f"{ds_full.df['uid'].nunique()} images  |  "
      f"{ds_full.df['task_name'].nunique()} tasks")

df = ds_full.df[ds_full.df["uid"].isin(probe_uids)].copy()
df["emp_p"]  = df["count_1"] / (df["count_0"] + df["count_1"])
df["totals"] = df["count_0"] + df["count_1"]
print(f"  Probe-only: {len(df)} cells  |  {df['uid'].nunique()} images  |  "
      f"{df['task_name'].nunique()} tasks")

# Flat arrays for vectorised evaluation
_uids       = df["uid"].values
_task_names = df["task_name"].values
_emp_p      = df["emp_p"].values
_valid      = np.isfinite(_emp_p) & (df["totals"].values > 0)
_delta_us   = np.stack([TASKS[t].delta_u for t in _task_names])   # [N_cells, K]

probe_uid_list = sorted(probe_uids)

# Ground-truth latent state per probe uid
cont_meta: dict = {}
with open(cfg.METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cont_meta[rec["id"]] = rec["z"]


def _gt_state(uid: str) -> int:
    z  = cont_meta[uid]
    lr = int(z["pos_xy"][0]      > X_THRESHOLD)
    tr = int(z["transparency"]   > TRANSP_THRESH)
    gl = int(z["glossiness"]     > GLOSS_THRESH)
    sl = int(z["scale"]          > SCALE_THRESH)
    return lr * 8 + tr * 4 + gl * 2 + sl


gt_states = {uid: _gt_state(uid) for uid in probe_uid_list}

# Per-image task data  uid -> [(delta_u [K], count_0, count_1), ...]
uid_task_data: dict = {uid: [] for uid in probe_uid_list}
for _, row in df.iterrows():
    uid = row["uid"]
    c0, c1 = int(row["count_0"]), int(row["count_1"])
    if c0 + c1 > 0:
        uid_task_data[uid].append((TASKS[row["task_name"]].delta_u, c0, c1))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def mc_p_yes_batch(alphas: np.ndarray,
                   delta_us: np.ndarray,
                   rng: np.random.Generator,
                   n: int = N_MC) -> np.ndarray:
    """Estimate P(yes_i) for each row i via MC. Returns [N]."""
    preds = np.zeros(len(alphas))
    for i in range(len(alphas)):
        b = rng.dirichlet(alphas[i], size=n)
        preds[i] = float((b @ delta_us[i] > 0).mean())
    return preds


def evaluate_model(alphas_by_uid: dict, label: str = "") -> dict:
    """Evaluate any model given a dict uid -> alpha [K]. Returns a result dict."""
    alphas = np.stack([alphas_by_uid[u] for u in _uids])
    preds  = mc_p_yes_batch(alphas, _delta_us, RNG)
    mse    = float(np.mean((preds[_valid] - _emp_p[_valid]) ** 2))
    rho, _ = spearmanr(preds[_valid], _emp_p[_valid])
    return dict(label=label, mse=mse, rho=rho,
                preds=preds, emp_p=_emp_p, valid=_valid, task_names=_task_names)


# ---------------------------------------------------------------------------
# Optimisation: fit alpha per image via Adam
# ---------------------------------------------------------------------------

def fit_alpha_for_image(uid: str,
                        task_data: list = None,
                        n_steps: int = N_STEPS_FIT) -> np.ndarray:
    """
    Optimise K=16 Dirichlet parameters for one image via Adam, maximising
    the soft multinomial NLL across a set of tasks.

    task_data: list of (delta_u [K], count_0, count_1).
               If None, uses all tasks for this image.
    """
    if task_data is None:
        task_data = uid_task_data[uid]
    if not task_data:
        return np.ones(K, dtype=np.float64)

    delta_us_t = torch.tensor(
        np.stack([du for du, c0, c1 in task_data]),
        dtype=torch.float32)                         # [T, K]
    counts_t   = torch.tensor(
        [[c0, c1] for du, c0, c1 in task_data],
        dtype=torch.float32)                         # [T, 2]

    # Start from uniform (log_alpha = 0 → alpha = 1 for all dims)
    log_alpha = torch.zeros(K, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([log_alpha], lr=LR_FIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-4)

    for _ in range(n_steps):
        optimizer.zero_grad()
        alpha  = torch.exp(log_alpha).clamp(min=1e-3)
        b      = TorchDirichlet(alpha).rsample((N_MC_FIT,))   # [n_mc, K]
        logits = b @ delta_us_t.T                              # [n_mc, T]
        p_yes  = torch.sigmoid(TEMPERATURE * logits).mean(0)  # [T]
        probs  = torch.stack([1 - p_yes, p_yes], dim=1).clamp(1e-7, 1 - 1e-7)
        nll    = -(counts_t * probs.log()).sum()
        nll.backward()
        optimizer.step()
        scheduler.step()

    return torch.exp(log_alpha).detach().clamp(min=1e-6).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 3 — Unconstrained MLE on all tasks
# ---------------------------------------------------------------------------
print("\n[Stage 3] Per-image unconstrained Dirichlet MLE (all tasks)...")

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    fitted_alphas_by_uid = None
    best_fitted          = None
else:
    fitted_alphas_by_uid: dict = {}
    for uid in probe_uid_list:
        print(f"  Fitting {uid}  (gt_state={gt_states[uid]:2d}) ...", end="  ", flush=True)
        alpha_fit = fit_alpha_for_image(uid)
        fitted_alphas_by_uid[uid] = alpha_fit
        task_data = uid_task_data[uid]
        if task_data:
            du_arr = np.stack([du for du, c0, c1 in task_data])
            preds  = mc_p_yes_batch(
                np.tile(alpha_fit, (len(du_arr), 1)), du_arr, RNG, n=500)
            mse_q  = float(np.mean([
                (p - c1 / (c0 + c1)) ** 2
                for p, (du, c0, c1) in zip(preds, task_data) if c0 + c1 > 0
            ]))
            print(f"img-MSE={mse_q:.4f}")
        else:
            print("no data")

    best_fitted = evaluate_model(fitted_alphas_by_uid,
                                 label="unconstrained MLE (all tasks)")
    print(f"\nStage 3: MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f}")


# ---------------------------------------------------------------------------
# Stage 4 — MLE on TRAIN_TASKS only, evaluate on VAL_TASKS
# ---------------------------------------------------------------------------
print("\n[Stage 4] Held-out task generalisation (fit on TRAIN_TASKS only)...")

val_task_set   = set(cfg.VAL_TASKS)
train_task_set = set(cfg.TRAIN_TASKS)

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    fitted_train_alphas = None
    best_fitted_train   = None
    trn_mse = trn_rho = val_mse = val_rho = float("nan")
else:
    uid_train_task_data: dict = {uid: [] for uid in probe_uid_list}
    for _, row in df.iterrows():
        if row["task_name"] not in train_task_set:
            continue
        uid = row["uid"]
        c0, c1 = int(row["count_0"]), int(row["count_1"])
        if c0 + c1 > 0:
            uid_train_task_data[uid].append(
                (TASKS[row["task_name"]].delta_u, c0, c1))

    fitted_train_alphas: dict = {}
    for uid in probe_uid_list:
        n_train = len(uid_train_task_data[uid])
        print(f"  Fitting {uid}  (gt={gt_states[uid]:2d}, "
              f"{n_train} train tasks) ...", end="  ", flush=True)
        alpha_fit = fit_alpha_for_image(uid, task_data=uid_train_task_data[uid])
        fitted_train_alphas[uid] = alpha_fit
        val_rows = [
            (TASKS[row["task_name"]].delta_u, int(row["count_0"]), int(row["count_1"]))
            for _, row in df[df["uid"] == uid].iterrows()
            if row["task_name"] in val_task_set and
               int(row["count_0"]) + int(row["count_1"]) > 0
        ]
        if val_rows:
            du_v  = np.stack([du for du, c0, c1 in val_rows])
            pv    = mc_p_yes_batch(
                np.tile(alpha_fit, (len(du_v), 1)), du_v, RNG, n=500)
            mse_v = float(np.mean([(p - c1 / (c0 + c1)) ** 2
                                   for p, (du, c0, c1) in zip(pv, val_rows)]))
            print(f"val-MSE={mse_v:.4f}")
        else:
            print("no val data")

    best_fitted_train = evaluate_model(fitted_train_alphas,
                                       label="MLE (train tasks only)")

    val_mask   = np.array([t in val_task_set   for t in _task_names]) & _valid
    train_mask = np.array([t in train_task_set for t in _task_names]) & _valid
    preds_tr   = best_fitted_train["preds"]

    if val_mask.sum() > 1:
        val_mse    = float(np.mean((preds_tr[val_mask]   - _emp_p[val_mask])   ** 2))
        val_rho, _ = spearmanr(preds_tr[val_mask],   _emp_p[val_mask])
    else:
        val_mse = val_rho = float("nan")
    if train_mask.sum() > 1:
        trn_mse    = float(np.mean((preds_tr[train_mask] - _emp_p[train_mask]) ** 2))
        trn_rho, _ = spearmanr(preds_tr[train_mask], _emp_p[train_mask])
    else:
        trn_mse = trn_rho = float("nan")

    print(f"\nStage 4 — all tasks:   MSE={best_fitted_train['mse']:.4f}  rho={best_fitted_train['rho']:.3f}")
    print(f"        — train tasks: MSE={trn_mse:.4f}  rho={trn_rho:.3f}")
    print(f"        — VAL tasks:   MSE={val_mse:.4f}  rho={val_rho:.3f}  ← generalisation")
    print(f"  (Stage 3 full MLE:   MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f})")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _task_color(name):
    n = name.count("_and_") + 1
    return {1: "#457B9D", 2: "#E76F51", 3: "#9B5DE5"}.get(n, "gray")


def _plot_scatter(result: dict, out_path: Path, val_tasks: set = None) -> None:
    """Per-task scatter: oracle P(yes) vs human P(yes)."""
    all_tasks = sorted(df["task_name"].unique())
    n_cols    = 8
    n_rows    = int(np.ceil(len(all_tasks) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, task_name in enumerate(all_tasks):
        row, col = divmod(idx, n_cols)
        ax       = axes[row, col]
        is_val   = val_tasks is not None and task_name in val_tasks
        mask     = (result["task_names"] == task_name) & result["valid"]
        x, y     = result["preds"][mask], result["emp_p"][mask]
        col_c    = _task_color(task_name)

        ax.scatter(x, y, s=10, alpha=0.8, color=col_c, linewidths=0)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, zorder=0)
        rho_t, _ = spearmanr(x, y) if len(x) > 2 else (float("nan"), None)
        mse_t    = float(np.mean((x - y) ** 2)) if len(x) > 0 else float("nan")
        ax.text(0.05, 0.95, f"r={rho_t:.2f}", transform=ax.transAxes,
                fontsize=6, va="top", color=col_c)
        ax.text(0.05, 0.82, f"e={mse_t:.3f}", transform=ax.transAxes,
                fontsize=6, va="top", color=col_c)

        title_str = task_name.replace("_and_", " & ").replace("_", "/")
        if is_val:
            title_str += "\n(held-out)"
        ax.set_title(title_str, fontsize=6, pad=2,
                     color="#CC2222" if is_val else "black")
        if is_val:
            for spine in ax.spines.values():
                spine.set_edgecolor("#CC2222")
                spine.set_linewidth(1.4)
            ax.set_facecolor("#FFF5F5")

        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    for idx in range(len(all_tasks), n_rows * n_cols):
        r2, c2 = divmod(idx, n_cols)
        axes[r2, c2].set_visible(False)

    extra = "  |  red frame = held-out val task" if val_tasks else ""
    fig.suptitle(
        f"Oracle ({result['label']})  |  "
        f"MSE={result['mse']:.4f}  rho={result['rho']:.3f}{extra}\n"
        "colour: 1-way (blue)  2-way (orange)  3-way (purple)",
        fontsize=8, y=1.01)
    fig.text(0.5,  -0.01, "Oracle P(yes)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5,  "Human P(yes)", va="center", rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------
print("\nPlotting per-task scatters...")
if best_fitted is not None:
    _plot_scatter(best_fitted,
                  PLOTS_DIR / "oracle_scatter_fitted.png")
if best_fitted_train is not None:
    _plot_scatter(best_fitted_train,
                  PLOTS_DIR / "oracle_scatter_fitted_train.png",
                  val_tasks=val_task_set)


def _plot_summary_scatter(result: dict,
                          train_tasks: set,
                          val_tasks: set,
                          out_path: Path) -> None:
    """Two-panel summary scatter: all train tasks (left) vs all val tasks (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    splits = [
        ("train tasks",    train_tasks, axes[0], "#457B9D"),
        ("held-out tasks", val_tasks,   axes[1], "#CC2222"),
    ]

    for split_label, task_set, ax, base_col in splits:
        mask = (
            np.array([t in task_set for t in result["task_names"]])
            & result["valid"]
        )
        x = result["preds"][mask]
        y = result["emp_p"][mask]
        tasks_here = result["task_names"][mask]

        unique_tasks = sorted(set(tasks_here))
        cmap = plt.cm.get_cmap("tab20", max(len(unique_tasks), 1))
        for ti, tname in enumerate(unique_tasks):
            tm = tasks_here == tname
            ax.scatter(x[tm], y[tm], s=12, alpha=0.7,
                       color=cmap(ti), linewidths=0,
                       label=tname.replace("_and_", " & ").replace("_", "/"),
                       zorder=3)

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8, zorder=0)

        if len(x) > 2:
            rho_s, _ = spearmanr(x, y)
            mse_s    = float(np.mean((x - y) ** 2))
        else:
            rho_s = mse_s = float("nan")

        ax.text(0.05, 0.95, f"ρ={rho_s:.3f}", transform=ax.transAxes,
                fontsize=9, va="top", fontweight="bold", color=base_col)
        ax.text(0.05, 0.86, f"mse={mse_s:.4f}", transform=ax.transAxes,
                fontsize=9, va="top", color=base_col)
        ax.text(0.05, 0.77, f"n={mask.sum()}", transform=ax.transAxes,
                fontsize=8, va="top", color="gray")

        ax.set_title(split_label, fontsize=10,
                     color=base_col, fontweight="bold")
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=8)

        legend = ax.legend(fontsize=6, loc="lower right",
                           framealpha=0.7, ncol=2)
        for lh in legend.legend_handles:
            lh.set_alpha(1.0)

    fig.text(0.5,  0.01, "Oracle P(yes)", ha="center", fontsize=10)
    fig.text(0.01, 0.5,  "Human P(yes)", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle(
        f"Oracle summary — Stage 4 (fit on train tasks only)\n"
        f"{result['label']}",
        fontsize=9, y=1.02)
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if best_fitted_train is not None:
    _plot_summary_scatter(
        best_fitted_train,
        train_tasks=train_task_set,
        val_tasks=val_task_set,
        out_path=PLOTS_DIR / "oracle_summary_scatter.png",
    )


# ---------------------------------------------------------------------------
# Bar plots: rho and MSE per task, stages 3 & 4
# ---------------------------------------------------------------------------
print("Plotting comparison bar charts...")

if best_fitted is None:
    print("Skipping bar plots (torch not available).")
else:
    models       = [best_fitted, best_fitted_train]
    model_labels = ["MLE — all tasks", "MLE — train tasks only"]
    model_colors = ["#43AA8B", "#9B5DE5"]

    all_tasks = sorted(df["task_name"].unique())
    rho_data  = [[] for _ in models]
    mse_data  = [[] for _ in models]
    labels    = []
    bar_cols  = []

    for task_name in all_tasks:
        mask = (_task_names == task_name) & _valid
        if mask.sum() < 3:
            continue
        y = _emp_p[mask]
        for mi, m in enumerate(models):
            x = m["preds"][mask]
            rho_data[mi].append(spearmanr(x, y)[0])
            mse_data[mi].append(float(np.mean((x - y) ** 2)))
        labels.append(task_name.replace("_and_", " & ").replace("_", "/"))
        bar_cols.append(_task_color(task_name))

    n_models = len(models)
    x_pos    = np.arange(len(labels))
    width    = 0.8 / n_models
    figw     = max(12, len(labels) * 0.65)
    offset   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    def _barplot(data_list, ylabel, title, out_path):
        fig, ax = plt.subplots(figsize=(figw, 4.5))
        for mi, (dat, lbl, col) in enumerate(zip(data_list, model_labels, model_colors)):
            for xi, (d, task_lbl) in enumerate(zip(dat, labels)):
                is_val  = any(vt.replace("_and_", " & ") == task_lbl
                              for vt in val_task_set)
                is_pred = (lbl == "MLE — train tasks only")
                hatch   = "//" if (is_pred and is_val) else None
                ax.bar(xi + offset[mi], d, width,
                       color=col, alpha=0.82, zorder=3,
                       hatch=hatch,
                       edgecolor="black" if hatch else "white",
                       linewidth=0.4,
                       label=lbl if xi == 0 else None)
        ax.set_xticks(x_pos)
        tick_labels = [
            (" " if any(vt.replace("_and_", " & ") == tl for vt in val_task_set) else "")
            + tl for tl in labels
        ]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        for tick, col in zip(ax.get_xticklabels(), bar_cols):
            tick.set_color(col)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        sns.despine(ax=ax, trim=True)
        ax.grid(axis="y", lw=0.3, color="lightgray", zorder=0)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    _barplot(rho_data, "Spearman rho",
             "Dirichlet MLE: all tasks vs train-tasks-only  "
             "|  ★ = held-out val task  |  hatching = held-out prediction",
             PLOTS_DIR / "oracle_rho_comparison.png")
    _barplot(mse_data, "MSE",
             "Dirichlet MLE: all tasks vs train-tasks-only  "
             "|  ★ = held-out val task  |  hatching = held-out prediction",
             PLOTS_DIR / "oracle_mse_comparison.png")


# ---------------------------------------------------------------------------
# Alpha heatmaps (stages 3 & 4 side by side)
# ---------------------------------------------------------------------------
if fitted_alphas_by_uid is not None:
    print("Plotting alpha heatmaps...")

    sorted_uids = sorted(probe_uid_list, key=lambda u: gt_states[u])
    gt_sorted   = [gt_states[u] for u in sorted_uids]

    def _state_label(k):
        lr = "R" if k & 8 else "L"
        tr = "Tr" if k & 4 else "Op"
        gl = "Gl" if k & 2 else "Mt"
        sl = "Lg" if k & 1 else "Sm"
        return f"{lr}·{tr}·{gl}·{sl}"

    state_labels = [_state_label(k) for k in range(K)]
    row_labels   = [f"k={gt_states[u]:2d}  {_state_label(gt_states[u])}"
                    for u in sorted_uids]

    alpha_mat       = np.stack([fitted_alphas_by_uid[u] / fitted_alphas_by_uid[u].sum()
                                for u in sorted_uids])
    alpha_mat_train = np.stack([fitted_train_alphas[u]  / fitted_train_alphas[u].sum()
                                for u in sorted_uids])
    vmax = max(alpha_mat.max(), alpha_mat_train.max())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, mat, subtitle in [
        (axes[0], alpha_mat,       "Stage 3 — fit on ALL tasks (upper bound)"),
        (axes[1], alpha_mat_train, "Stage 4 — fit on TRAIN tasks only (generalisation)"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="E[b_k]", shrink=0.7)
        ax.set_xticks(range(K))
        ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(sorted_uids)))
        ax.set_yticklabels(row_labels, fontsize=7)
        for row_i, k in enumerate(gt_sorted):
            ax.add_patch(plt.Rectangle((k - 0.5, row_i - 0.5), 1, 1,
                                       fill=False, edgecolor="blue",
                                       linewidth=2, zorder=5))
        ax.set_xlabel("Latent state k")
        ax.set_title(subtitle, fontsize=9)

    fig.suptitle(
        "Fitted Dirichlet E[b] per probe image  |  blue border = ground-truth state\n"
        f"Stage 3: MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f}    "
        f"Stage 4: MSE={best_fitted_train['mse']:.4f}  rho={best_fitted_train['rho']:.3f}",
        fontsize=9)
    axes[0].set_ylabel("Probe image (sorted by ground-truth state)")
    plt.tight_layout()
    out = PLOTS_DIR / "oracle_alpha_heatmap_dual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
if best_fitted is not None:
    all_tasks = sorted(df["task_name"].unique())
    print(f"\n{'='*70}")
    print(f"{'task':<39}  {'N':>5}  {'MSE(all)':>9}  {'rho':>5}  "
          f"{'MSE(train)':>10}  {'rho':>5}")
    print("=" * 70)
    for task_name in all_tasks:
        mask = (_task_names == task_name) & _valid
        if mask.sum() < 2:
            continue
        n_way = task_name.count("_and_") + 1
        tag   = " *" if task_name in val_task_set else "  "
        y     = _emp_p[mask]
        row   = f"[{n_way}-way]{tag}{task_name:<33}  {mask.sum():>5}"
        for m in [best_fitted, best_fitted_train]:
            x       = m["preds"][mask]
            mse_v   = float(np.mean((x - y) ** 2))
            rho_v,_ = spearmanr(x, y)
            row    += f"  {mse_v:>9.4f}  {rho_v:>5.3f}"
        print(row)

    print(f"\n{'='*70}")
    print("Overall (all tasks):")
    print(f"  Stage 3 (all tasks):   MSE={best_fitted['mse']:.4f}  rho={best_fitted['rho']:.3f}")
    print(f"  Stage 4 (train only):  MSE={best_fitted_train['mse']:.4f}  rho={best_fitted_train['rho']:.3f}")
    print(f"\nGeneralisation (Stage 4, train→val):")
    print(f"  train tasks:  MSE={trn_mse:.4f}  rho={trn_rho:.3f}")
    print(f"  VAL tasks:    MSE={val_mse:.4f}  rho={val_rho:.3f}  ← key number")
    print("\n  * = val task (held-out from Stage 4 fitting)")
