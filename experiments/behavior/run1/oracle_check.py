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

Results include both raw (τ=0) and τ_n-corrected predictions.

Usage:
    cd <repo root>
    python experiments/behavior/run1/oracle_check.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from scipy.special import betainc

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_RUN1_DIR = Path(__file__).parent
_RUN0_DIR = _RUN1_DIR.parent / "run0"

sys.path.insert(0, str(_RUN1_DIR / "01_fit"))
sys.path.insert(0, str(_RUN0_DIR))

import config as cfg           # run1/01_fit/config.py  (TRAIN_TASKS, VAL_TASKS, …)
from preprocess import load_and_preprocess

from dlbt.constants import K
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
N_MC        = 2000
N_MC_FIT    = 2000
N_STEPS_FIT = 1000
LR_FIT      = 0.05
TEMPERATURE = 50.0
RNG         = np.random.default_rng(42)

PLOTS_DIR = _RUN1_DIR / "01_fit" / "results" / "plots" / "oracle"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# τ_n correction helpers
# ---------------------------------------------------------------------------
def _k_plus(n: int) -> int:
    return K // (2 ** n)


def _tau_n(n: int) -> float:
    """τ_n = 2·median(Beta(K₊, K₋)) − 1  for n-way conjunction."""
    kp = _k_plus(n)
    km = K - kp
    # median of Beta(kp, km) via incomplete beta inverse (binary search)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if betainc(kp, km, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return 2 * ((lo + hi) / 2) - 1


def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


TAU = {n: _tau_n(n) for n in range(1, 5)}   # precompute for n=1..4

# ---------------------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------------------
print("Loading behavioural data (run0 + run1)...")
ds_full, probe_uids, main_uids, diag = load_and_preprocess(
    cfg.BEHAVIOR_CSV_RUN0,
    beh_id_to_task     = cfg.BEH_ID_TO_TASK,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds    = cfg.USE_TRIAL_KINDS,
    seed               = cfg.SEED,
    extra_csv          = cfg.BEHAVIOR_CSV_RUN1,
)
print(f"  {len(ds_full)} (uid, task) cells  |  "
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
_delta_us   = np.stack([TASKS[t].delta_u for t in _task_names])   # [N, K]
_arities    = np.array([_arity(t) for t in _task_names])
_taus       = np.array([TAU[a] for a in _arities])

probe_uid_list = sorted(probe_uids)

# Per-image task data  uid -> [(delta_u [K], count_0, count_1, arity), ...]
uid_task_data: dict = {uid: [] for uid in probe_uid_list}
for _, row in df.iterrows():
    uid = row["uid"]
    c0, c1 = int(row["count_0"]), int(row["count_1"])
    if c0 + c1 > 0:
        n = _arity(row["task_name"])
        uid_task_data[uid].append(
            (TASKS[row["task_name"]].delta_u, c0, c1, n))

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def mc_p_yes(alpha: np.ndarray, delta_u: np.ndarray,
             tau: float, rng: np.random.Generator,
             n: int = N_MC) -> float:
    b = rng.dirichlet(alpha, size=n)
    return float((b @ delta_u > tau).mean())


def evaluate_model(alphas_by_uid: dict, label: str,
                   tau_corrected: bool = False) -> dict:
    preds = np.zeros(len(_uids))
    for i, (uid, du, tau) in enumerate(zip(_uids, _delta_us, _taus)):
        t = tau if tau_corrected else 0.0
        preds[i] = mc_p_yes(alphas_by_uid[uid], du, t, RNG)
    mask = _valid
    mse    = float(np.mean((preds[mask] - _emp_p[mask]) ** 2))
    rho, _ = spearmanr(preds[mask], _emp_p[mask])
    dpa    = float(np.mean(
        (preds[mask] > 0.5) == (_emp_p[mask] > 0.5)))
    return dict(label=label, mse=mse, rho=rho, dpa=dpa,
                preds=preds, emp_p=_emp_p, valid=mask,
                task_names=_task_names, tau_corrected=tau_corrected)


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
        np.stack([du for du, c0, c1, n in task_data]),
        dtype=torch.float32)
    counts_t = torch.tensor(
        [[c0, c1] for du, c0, c1, n in task_data],
        dtype=torch.float32)
    taus_t = torch.tensor(
        [TAU[n] for du, c0, c1, n in task_data],
        dtype=torch.float32)

    log_alpha = torch.zeros(K, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([log_alpha], lr=LR_FIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-4)

    for _ in range(n_steps):
        optimizer.zero_grad()
        alpha  = torch.exp(log_alpha).clamp(min=1e-3)
        b      = TorchDirichlet(alpha).rsample((N_MC_FIT,))   # [n_mc, K]
        logits = b @ delta_us_t.T                              # [n_mc, T]
        # Use τ_n-corrected threshold during fitting
        shifted = logits - taus_t.unsqueeze(0)
        p_yes   = torch.sigmoid(TEMPERATURE * shifted).mean(0)
        probs   = torch.stack([1 - p_yes, p_yes], dim=1).clamp(1e-7, 1 - 1e-7)
        nll     = -(counts_t * probs.log()).sum()
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
    res3_raw = res3_corr = None
else:
    fitted_all: dict = {}
    for uid in probe_uid_list:
        print(f"  Fitting {uid} ...", end="  ", flush=True)
        fitted_all[uid] = fit_alpha(uid)
        print("done")

    res3_raw  = evaluate_model(fitted_all, "MLE all tasks  [τ=0]",      tau_corrected=False)
    res3_corr = evaluate_model(fitted_all, "MLE all tasks  [τ_n]",       tau_corrected=True)
    print(f"\nStage 3 (τ=0):  MSE={res3_raw['mse']:.4f}  ρ={res3_raw['rho']:.3f}  "
          f"DPA={res3_raw['dpa']:.3f}")
    print(f"Stage 3 (τ_n):  MSE={res3_corr['mse']:.4f}  ρ={res3_corr['rho']:.3f}  "
          f"DPA={res3_corr['dpa']:.3f}")


# ---------------------------------------------------------------------------
# Stage 4 — MLE on TRAIN_TASKS only, evaluate on VAL_TASKS
# ---------------------------------------------------------------------------
print("\n[Stage 4] Held-out task generalisation (fit on TRAIN_TASKS only)...")

train_task_set = set(cfg.TRAIN_TASKS)
val_task_set   = set(cfg.VAL_TASKS)

if not HAS_TORCH:
    print("  Skipping (torch not available).")
    fitted_train = None
    res4_raw = res4_corr = None
    trn_mse = trn_rho = val_mse = val_rho = float("nan")
else:
    uid_train_data: dict = {uid: [] for uid in probe_uid_list}
    for _, row in df.iterrows():
        if row["task_name"] not in train_task_set:
            continue
        uid = row["uid"]
        c0, c1 = int(row["count_0"]), int(row["count_1"])
        if c0 + c1 > 0:
            n = _arity(row["task_name"])
            uid_train_data[uid].append(
                (TASKS[row["task_name"]].delta_u, c0, c1, n))

    fitted_train: dict = {}
    for uid in probe_uid_list:
        n_tr = len(uid_train_data[uid])
        print(f"  Fitting {uid}  ({n_tr} train tasks) ...", end="  ", flush=True)
        fitted_train[uid] = fit_alpha(uid, task_data=uid_train_data[uid])
        print("done")

    res4_raw  = evaluate_model(fitted_train, "MLE train only  [τ=0]", tau_corrected=False)
    res4_corr = evaluate_model(fitted_train, "MLE train only  [τ_n]", tau_corrected=True)

    def _split_metrics(result, task_set):
        mask = np.array([t in task_set for t in result["task_names"]]) & result["valid"]
        if mask.sum() < 2:
            return float("nan"), float("nan"), float("nan")
        p, e = result["preds"][mask], result["emp_p"][mask]
        mse = float(np.mean((p - e) ** 2))
        rho, _ = spearmanr(p, e)
        dpa = float(np.mean((p > 0.5) == (e > 0.5)))
        return mse, rho, dpa

    for tag, res in [("τ=0", res4_raw), ("τ_n", res4_corr)]:
        trn_mse, trn_rho, trn_dpa = _split_metrics(res, train_task_set)
        val_mse, val_rho, val_dpa = _split_metrics(res, val_task_set)
        print(f"\nStage 4 ({tag}):")
        print(f"  train tasks: MSE={trn_mse:.4f}  ρ={trn_rho:.3f}  DPA={trn_dpa:.3f}")
        print(f"  VAL tasks:   MSE={val_mse:.4f}  ρ={val_rho:.3f}  DPA={val_dpa:.3f}  ← generalisation")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
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

if res3_raw is not None:
    _plot_per_task(res3_raw,  PLOTS_DIR / "oracle_s3_raw_per_task.png")
    _plot_per_task(res3_corr, PLOTS_DIR / "oracle_s3_corr_per_task.png")

if res4_raw is not None:
    _plot_per_task(res4_raw,  PLOTS_DIR / "oracle_s4_raw_per_task.png",
                   val_tasks=val_task_set)
    _plot_per_task(res4_corr, PLOTS_DIR / "oracle_s4_corr_per_task.png",
                   val_tasks=val_task_set)
    _plot_summary(res4_raw,  train_task_set, val_task_set,
                  PLOTS_DIR / "oracle_s4_raw_summary.png")
    _plot_summary(res4_corr, train_task_set, val_task_set,
                  PLOTS_DIR / "oracle_s4_corr_summary.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
if res3_raw is not None:
    all_tasks = sorted(df["task_name"].unique(), key=lambda t: (_arity(t), t))
    print(f"\n{'='*80}")
    print(f"{'task':<45}  {'N':>4}  "
          f"{'S3τ=0 ρ':>8}  {'S3τ_n ρ':>8}  "
          f"{'S4τ=0 ρ':>8}  {'S4τ_n ρ':>8}")
    print("=" * 80)
    for task_name in all_tasks:
        mask = (_task_names == task_name) & _valid
        if mask.sum() < 2:
            continue
        tag  = " ★" if task_name in val_task_set else "  "
        n_w  = _arity(task_name)
        y    = _emp_p[mask]
        rhos = []
        for res in [res3_raw, res3_corr, res4_raw, res4_corr]:
            x = res["preds"][mask]
            r, _ = spearmanr(x, y) if len(x) > 2 else (float("nan"), None)
            rhos.append(r)
        print(f"[{n_w}w]{tag} {task_name:<43}  {mask.sum():>4}  "
              + "  ".join(f"{r:>8.3f}" for r in rhos))

    print(f"\n{'='*80}")
    print("Overall:")
    for res in [res3_raw, res3_corr, res4_raw, res4_corr]:
        print(f"  {res['label']:<35}  "
              f"MSE={res['mse']:.4f}  ρ={res['rho']:.3f}  DPA={res['dpa']:.3f}")
