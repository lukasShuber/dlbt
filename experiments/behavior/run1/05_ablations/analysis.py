"""
run1/05_ablations/analysis.py — belief-ablation budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trial budget):

  Traces (mean ± SEM across seeds):
    • DLBT    — red solid   (full model: MC Dirichlet integration)
    • DetBT   — blue solid  (deterministic mean, same mapper)
    • SLDA    — purple solid (ridge decoder baseline)

  Reference lines (no SEM):
    • Oracle beliefs  — orange dashed horizontal (fixed beliefs from latent state)
    • Random (P=0.5)  — gray dashed horizontal   (cMSE only)

  Noise ceiling:
    • Spearman ρ noise ceiling — dark gray dotted horizontal (ρ plot only)

  All-data markers:
    • Filled marker for DLBT, DetBT, SLDA (disconnected from trace)

Run from repo root:
    python experiments/behavior/run1/05_ablations/analysis.py
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to ablations.pkl. Default: auto-discover.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis on cMSE plot (also set via cfg.LOG_Y).")
args = parser.parse_args()

_log_y = cfg.LOG_Y or args.log_y

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_path = Path(args.pkl)
else:
    pkl_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
    if not pkl_path.exists():
        candidates = sorted(cfg.RESULTS_DIR.glob("ablations*.pkl"))
        matching = [p for p in candidates if cfg.RUN_TAG in p.stem]
        if matching:
            pkl_path = matching[-1]
        elif candidates:
            pkl_path = candidates[-1]
        else:
            raise FileNotFoundError(f"No pkl found in {cfg.RESULTS_DIR}")

print(f"Loading: {pkl_path.name}")
with open(pkl_path, "rb") as f:
    d = pickle.load(f)

budgets          = np.array(d["trial_budgets"])    # [n_budgets]
total_pool_size  = d["total_pool_size"]
seeds            = d["seeds"]

dlbt_cmse  = d["dlbt_cmse"]    # [n_seeds, n_budgets]
dlbt_rho   = d["dlbt_rho"]
detbt_cmse = d["detbt_cmse"]
detbt_rho  = d["detbt_rho"]
slda_cmse    = d["slda_cmse"]
slda_rho     = d["slda_rho"]
randont_cmse = d["randont_cmse"]
randont_rho  = d["randont_rho"]

dlbt_all_cmse    = d["dlbt_all_cmse"]    # [n_seeds]
dlbt_all_rho     = d["dlbt_all_rho"]
detbt_all_cmse   = d["detbt_all_cmse"]
detbt_all_rho    = d["detbt_all_rho"]
slda_all_cmse    = d["slda_all_cmse"]
slda_all_rho     = d["slda_all_rho"]
randont_all_cmse = d["randont_all_cmse"]
randont_all_rho  = d["randont_all_rho"]

oracle_cmse    = d["oracle_cmse"]       # scalar
oracle_rho     = d["oracle_rho"]        # scalar
oracle_conc    = d.get("oracle_concentration", cfg.ORACLE_CONCENTRATION)

random_cmse_nf    = d["random_cmse_net"]
rho_noise_ceiling = d.get("rho_noise_ceiling", float("nan"))

# Fallback: recompute noise ceiling from count_matrix if missing
if np.isnan(rho_noise_ceiling) and "count_matrix" in d:
    from scipy.stats import spearmanr as _spearmanr

    def _rho_nc_from_counts(true_mat, count_mat, n_splits=200, seed=0):
        mask    = count_mat > 1
        totals  = count_mat[mask].astype(int)
        count1s = np.round(true_mat[mask] * totals).astype(int)
        n1s     = totals // 2
        n2s     = totals - n1s
        if len(totals) < 2:
            return float("nan")
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_splits):
            k1 = np.array([rng.hypergeometric(c1, t - c1, n1)
                           for c1, t, n1 in zip(count1s, totals, n1s)], dtype=float)
            p1 = k1 / n1s
            p2 = (count1s - k1) / n2s
            rh, _ = _spearmanr(p1, p2)
            if not np.isnan(rh) and rh > -1:
                vals.append((2 * rh) / (1 + rh))
        return float(np.mean(vals)) if vals else float("nan")

    print("Computing Spearman ρ noise ceiling from count_matrix...")
    rho_noise_ceiling = _rho_nc_from_counts(d["true_matrix"], d["count_matrix"])
    print(f"  ρ noise ceiling: {rho_noise_ceiling:.4f}")
elif np.isnan(rho_noise_ceiling):
    print("[warn] rho_noise_ceiling not in pkl — re-run run.py to generate it.")

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    """Mean and SEM across axis-0 (seeds)."""
    mu  = np.nanmean(arr, axis=0)
    n   = np.sum(~np.isnan(arr), axis=0).astype(float)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mu, sem


def _mean_sem_scalar(arr: np.ndarray):
    """Mean and SEM for a 1-D [n_seeds] array."""
    n   = float(np.sum(~np.isnan(arr)))
    mu  = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(max(n, 1)))
    return mu, sem


def _plot_trace(ax, budgets, mu, sem, color, label, ls="-", zorder=3):
    """Line + SEM band + open circular markers."""
    ax.fill_between(budgets, mu - sem, mu + sem,
                    color=color, alpha=0.15, zorder=zorder - 1)
    ax.plot(budgets, mu, color=color, lw=2.0, ls=ls, zorder=zorder,
            label=label)
    ax.plot(budgets, mu, "o", color=color, ms=5, mfc="none",
            mew=1.4, zorder=zorder + 1)


def _plot_all_data_marker(ax, x, mu, sem, color, zorder=5):
    """Filled marker (not connected to trace) for the all-data point."""
    ax.errorbar(x, mu, yerr=sem, fmt="o", color=color,
                ms=7, mfc=color, mew=1.4, capsize=3,
                elinewidth=1.2, zorder=zorder)


def _xaxis_setup(ax):
    """Log x-axis from 10^2 to 10^5."""
    ax.set_xscale("log")
    ax.set_xlim(70, 1.5e5)
    ax.set_xticks([100, 1_000, 10_000, 100_000])
    ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Total trial budget", fontsize=11)


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def _make_figure(metric: str):
    """metric: 'cmse' or 'rho'"""
    is_cmse = metric == "cmse"

    dlbt_mu,    dlbt_sem    = _mean_sem(dlbt_cmse    if is_cmse else dlbt_rho)
    detbt_mu,   detbt_sem   = _mean_sem(detbt_cmse   if is_cmse else detbt_rho)
    slda_mu,    slda_sem    = _mean_sem(slda_cmse    if is_cmse else slda_rho)
    randont_mu, randont_sem = _mean_sem(randont_cmse if is_cmse else randont_rho)

    dlbt_all_mu,    dlbt_all_sem    = _mean_sem_scalar(
        dlbt_all_cmse    if is_cmse else dlbt_all_rho)
    detbt_all_mu,   detbt_all_sem   = _mean_sem_scalar(
        detbt_all_cmse   if is_cmse else detbt_all_rho)
    slda_all_mu,    slda_all_sem    = _mean_sem_scalar(
        slda_all_cmse    if is_cmse else slda_all_rho)
    randont_all_mu, randont_all_sem = _mean_sem_scalar(
        randont_all_cmse if is_cmse else randont_all_rho)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # ---- Reference lines ----
    if is_cmse:
        ax.axhline(random_cmse_nf, color=cfg.C_RNDINI, lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)

    oracle_val = oracle_cmse if is_cmse else oracle_rho
    ax.axhline(oracle_val, color=cfg.C_ORACLE, lw=1.5,
               ls="--", label=f"Oracle (c={oracle_conc:.0f})", zorder=2)

    if not is_cmse and not np.isnan(rho_noise_ceiling):
        ax.axhline(rho_noise_ceiling, color="#555555", lw=1.5,
                   ls=(0, (2, 2)), label="Noise ceiling", zorder=2)

    # ---- Budget-grid traces ----
    _plot_trace(ax, budgets, dlbt_mu,    dlbt_sem,    cfg.C_DLBT,    "DLBT",    zorder=6)
    _plot_trace(ax, budgets, detbt_mu,   detbt_sem,   cfg.C_DETBT,   "DetBT",   zorder=5)
    _plot_trace(ax, budgets, slda_mu,    slda_sem,    cfg.C_SLDA,    "SLDA",    zorder=4)
    _plot_trace(ax, budgets, randont_mu, randont_sem, cfg.C_RANDONT, "RandOnt", zorder=3)

    # ---- All-data markers (filled, disconnected) ----
    _plot_all_data_marker(ax, total_pool_size, dlbt_all_mu,    dlbt_all_sem,    cfg.C_DLBT,    zorder=7)
    _plot_all_data_marker(ax, total_pool_size, detbt_all_mu,   detbt_all_sem,   cfg.C_DETBT,   zorder=7)
    _plot_all_data_marker(ax, total_pool_size, slda_all_mu,    slda_all_sem,    cfg.C_SLDA,    zorder=7)
    _plot_all_data_marker(ax, total_pool_size, randont_all_mu, randont_all_sem, cfg.C_RANDONT, zorder=7)

    # ---- Axes ----
    _xaxis_setup(ax)

    if is_cmse:
        ax.set_ylabel("cMSE − noise floor", fontsize=11)
        if _log_y:
            ax.set_yscale("log")
            ax.set_ylim(0.01, 1.0)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
        else:
            ax.set_ylim(0, 0.34)
        legend_loc = "upper right"
    else:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
        ax.set_ylim(0, 1)
        legend_loc = "lower right"

    ax.legend(fontsize=8, frameon=False, loc=legend_loc)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()

    tag = "cmse" if is_cmse else "rho"
    out = plots_dir / f"plot_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
_make_figure("cmse")
_make_figure("rho")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"{'Model':<20}  {'budget':>10}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}")
print("-" * 72)

for label, cmse_arr, rho_arr, cmse_all, rho_all in [
    ("DLBT",    dlbt_cmse,    dlbt_rho,    dlbt_all_cmse,    dlbt_all_rho),
    ("DetBT",   detbt_cmse,   detbt_rho,   detbt_all_cmse,   detbt_all_rho),
    ("SLDA",    slda_cmse,    slda_rho,    slda_all_cmse,    slda_all_rho),
    ("RandOnt", randont_cmse, randont_rho, randont_all_cmse, randont_all_rho),
]:
    for b_i, b in enumerate(budgets):
        mu_c, sem_c = _mean_sem_scalar(cmse_arr[:, b_i])
        mu_r, sem_r = _mean_sem_scalar(rho_arr[:, b_i])
        print(f"  {label:<18}  {b:>10,}  "
              f"{mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}")

    mu_c, sem_c = _mean_sem_scalar(cmse_all)
    mu_r, sem_r = _mean_sem_scalar(rho_all)
    print(f"  {label:<18}  {'all data':>10}  "
          f"{mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}")
    print()

print(f"  {'Oracle':<18}  {'(fixed)':>10}  "
      f"{oracle_cmse:+.5f}            "
      f"{oracle_rho:+.4f}")
print("=" * 72)
