"""
run1/023_efficiency_main/analysis.py — budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trial budget):

  Traces (mean ± SEM across seeds):
    • DLBT          — blue solid
    • SLDA          — purple solid
    • Anti-human    — red dashed  (DLBT trained on label-flipped data)

  Reference lines (no SEM):
    • Random guesser       — gray dashed horizontal
    • Random-init DLBT     — gray dotted horizontal

  Markers:
    • Budget grid points   — open markers, connected by lines
    • All-data point       — filled marker, plotted separately (not connected)

Run from repo root:
    python experiments/behavior/run1/023_efficiency_main/analysis.py
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
                    help="Path to efficiency_main.pkl. Default: auto-discover.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis on cMSE plot.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_path = Path(args.pkl)
else:
    pkl_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
    if not pkl_path.exists():
        candidates = sorted(cfg.RESULTS_DIR.glob("efficiency_main*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No pkl found in {cfg.RESULTS_DIR}")
        pkl_path = candidates[-1]

print(f"Loading: {pkl_path.name}")
with open(pkl_path, "rb") as f:
    d = pickle.load(f)

budgets          = np.array(d["trial_budgets"])          # [n_budgets]
total_pool_size  = d["total_pool_size"]
seeds            = d["seeds"]
n_seeds          = len(seeds)

dlbt_cmse = d["dlbt_cmse"]    # [n_seeds, n_budgets]
dlbt_rho  = d["dlbt_rho"]
slda_cmse = d["slda_cmse"]
slda_rho  = d["slda_rho"]
anti_cmse = d["anti_cmse"]
anti_rho  = d["anti_rho"]

dlbt_all_cmse = d["dlbt_all_cmse"]   # [n_seeds]
dlbt_all_rho  = d["dlbt_all_rho"]
slda_all_cmse = d["slda_all_cmse"]
slda_all_rho  = d["slda_all_rho"]
anti_all_cmse = d["anti_all_cmse"]
anti_all_rho  = d["anti_all_rho"]

random_cmse_nf      = d["random_cmse_nf"]
random_init_cmse_nf = d["random_init_cmse_nf"]
probe_noise_floor   = d["probe_noise_floor"]

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    """Mean and SEM across axis-0 (seeds)."""
    mu  = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return mu, sem


def _mean_sem_scalar(arr: np.ndarray):
    """Mean and SEM for a 1-D [n_seeds] array."""
    mu  = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr))))
    return mu, sem


def _plot_trace(ax, budgets, mu, sem, color, label, ls="-", zorder=3):
    ax.fill_between(budgets, mu - sem, mu + sem,
                    color=color, alpha=0.15, zorder=zorder - 1)
    ax.plot(budgets, mu, color=color, lw=1.8, ls=ls, zorder=zorder,
            label=label)
    ax.plot(budgets, mu, "o", color=color, ms=5, mfc="none",
            mew=1.4, zorder=zorder + 1)


def _plot_all_data_marker(ax, x, mu, sem, color, marker="o", zorder=5):
    """Filled marker (not connected) for the all-data point."""
    ax.errorbar(x, mu, yerr=sem, fmt=marker, color=color,
                ms=7, mfc=color, mew=1.4, capsize=3,
                elinewidth=1.2, zorder=zorder)


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def _make_figure(metric: str):
    """
    metric: "cmse" or "rho"
    """
    is_cmse = metric == "cmse"

    dlbt_mu,  dlbt_sem  = _mean_sem(dlbt_cmse if is_cmse else dlbt_rho)
    slda_mu,  slda_sem  = _mean_sem(slda_cmse if is_cmse else slda_rho)
    anti_mu,  anti_sem  = _mean_sem(anti_cmse if is_cmse else anti_rho)

    dlbt_all_mu, dlbt_all_sem = _mean_sem_scalar(dlbt_all_cmse if is_cmse else dlbt_all_rho)
    slda_all_mu, slda_all_sem = _mean_sem_scalar(slda_all_cmse if is_cmse else slda_all_rho)
    anti_all_mu, anti_all_sem = _mean_sem_scalar(anti_all_cmse if is_cmse else anti_all_rho)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # ---- Reference lines ----
    if is_cmse:
        ax.axhline(random_cmse_nf, color="#aaaaaa", lw=1.2, ls="--",
                   label="Random guesser", zorder=1)
        ax.axhline(random_init_cmse_nf, color="#aaaaaa", lw=1.2, ls=":",
                   label="Random-init DLBT", zorder=1)

    # ---- Budget-grid traces ----
    _plot_trace(ax, budgets, dlbt_mu, dlbt_sem, cfg.C_DLBT, "DLBT")
    _plot_trace(ax, budgets, slda_mu, slda_sem, cfg.C_SLDA, "SLDA")
    _plot_trace(ax, budgets, anti_mu, anti_sem, cfg.C_ANTI,
                "Anti-human DLBT", ls="--")

    # ---- All-data markers (filled, disconnected) ----
    _plot_all_data_marker(ax, total_pool_size, dlbt_all_mu, dlbt_all_sem, cfg.C_DLBT)
    _plot_all_data_marker(ax, total_pool_size, slda_all_mu, slda_all_sem, cfg.C_SLDA)
    _plot_all_data_marker(ax, total_pool_size, anti_all_mu, anti_all_sem, cfg.C_ANTI)

    # Vertical dashed line separating the budget grid from the all-data point
    ax.axvline(total_pool_size, color="#cccccc", lw=0.8, ls="--", zorder=0)

    ax.set_xscale("log")
    if is_cmse and args.log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Trial budget", fontsize=11)
    ylabel = "cMSE − NF" if is_cmse else "Spearman ρ"
    ax.set_ylabel(ylabel, fontsize=11)

    ax.set_title(
        f"Budget efficiency — full task coverage\n"
        f"(N={n_seeds} seeds,  NF={probe_noise_floor:.4f})",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax, trim=True)

    tag  = "cmse" if is_cmse else "rho"
    out  = plots_dir / f"plot_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
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
print("\n" + "=" * 70)
print(f"{'Model':<20}  {'budget':>10}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}")
print("-" * 70)

for label, cmse_arr, rho_arr in [
    ("DLBT",         dlbt_cmse, dlbt_rho),
    ("SLDA",         slda_cmse, slda_rho),
    ("Anti-human",   anti_cmse, anti_rho),
]:
    for b_i, b in enumerate(budgets):
        mu_c, sem_c = float(np.nanmean(cmse_arr[:, b_i])), \
                      float(np.nanstd(cmse_arr[:, b_i], ddof=1) / np.sqrt(n_seeds))
        mu_r, sem_r = float(np.nanmean(rho_arr[:, b_i])), \
                      float(np.nanstd(rho_arr[:, b_i],  ddof=1) / np.sqrt(n_seeds))
        print(f"  {label:<18}  {b:>10,}  "
              f"{mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}")
    # All-data row
    c_arr = dlbt_all_cmse if label == "DLBT" else \
            slda_all_cmse if label == "SLDA" else anti_all_cmse
    r_arr = dlbt_all_rho  if label == "DLBT" else \
            slda_all_rho  if label == "SLDA" else anti_all_rho
    mu_c, sem_c = float(np.nanmean(c_arr)), float(np.nanstd(c_arr, ddof=1)/np.sqrt(n_seeds))
    mu_r, sem_r = float(np.nanmean(r_arr)), float(np.nanstd(r_arr, ddof=1)/np.sqrt(n_seeds))
    print(f"  {label:<18}  {'all data':>10}  "
          f"{mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}")
    print()

print("=" * 70)
