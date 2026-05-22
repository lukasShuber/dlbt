"""
run1/021_efficiency_main/analysis.py — trials-per-task budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trials per task) for every
efficiency_main_021*.pkl found in results/.

  Traces (mean ± SEM across seeds):
    • DLBT         — red solid
    • SLDA         — purple solid
    • Anti-human   — gray solid

  Reference lines:
    • chance (P=0.5) — gray dashed, annotated at right edge (cMSE only)
    • Noise ceiling  — dark gray dotted, annotated at left edge (ρ only)

  Markers:
    • Budget grid points — open circular markers, connected by lines
    • All-data point     — filled marker, plotted separately (not connected)

Run from repo root:
    python experiments/behavior/run1/021_efficiency_main/analysis.py
    python experiments/behavior/run1/021_efficiency_main/analysis.py --pkl PATH
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
from scipy.stats import spearmanr as _spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to a specific pkl. Default: all efficiency_main_021*.pkl.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis on cMSE plot (also set via cfg.LOG_Y).")
args = parser.parse_args()

_log_y = cfg.LOG_Y or args.log_y

# ---------------------------------------------------------------------------
# Collect pkl paths
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_paths = [Path(args.pkl)]
else:
    pkl_paths = sorted(cfg.RESULTS_DIR.glob("efficiency_main_021*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No efficiency_main_021*.pkl found in {cfg.RESULTS_DIR}")

print(f"Processing {len(pkl_paths)} pkl(s):")
for p in pkl_paths:
    print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    mu  = np.nanmean(arr, axis=0)
    n   = np.sum(~np.isnan(arr), axis=0).astype(float)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mu, sem


def _mean_sem_scalar(arr: np.ndarray):
    n   = float(np.sum(~np.isnan(arr)))
    mu  = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(max(n, 1)))
    return mu, sem


def _plot_trace(ax, tpt_grid, mu, sem, color, ls="-", zorder=3):
    ax.fill_between(tpt_grid, mu - sem, mu + sem,
                    color=color, alpha=0.15, zorder=zorder - 1)
    ax.plot(tpt_grid, mu, color=color, lw=2.0, ls=ls, zorder=zorder)
    ax.plot(tpt_grid, mu, "o", color=color, ms=5, mfc="none",
            mew=1.4, zorder=zorder + 1)


def _plot_all_data_marker(ax, x, mu, sem, color, zorder=5):
    ax.errorbar(x, mu, yerr=sem, fmt="o", color=color,
                ms=7, mfc=color, mew=1.4, capsize=3,
                elinewidth=1.2, zorder=zorder)


def _xaxis_setup(ax, tpt_grid, avg_pool_per_task):
    ax.set_xscale("log")
    x_upper = max(avg_pool_per_task * 2.5, float(tpt_grid[-1]) * 3.0)
    ax.set_xlim(7, x_upper)
    ax.set_xticks([10, 100, 1_000])
    ax.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$"])
    ax.set_xlabel("Trials per task", fontsize=11, fontweight="bold")


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


# ---------------------------------------------------------------------------
# Per-pkl processing
# ---------------------------------------------------------------------------

def process_pkl(pkl_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading: {pkl_path.name}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    tpt_grid          = np.array(d["trials_per_task"])
    avg_pool_per_task = d["avg_pool_per_task"]
    seeds             = d["seeds"]
    n_seeds           = len(seeds)

    dlbt_cmse = d["dlbt_cmse"]
    dlbt_rho  = d["dlbt_rho"]
    slda_cmse = d["slda_cmse"]
    slda_rho  = d["slda_rho"]
    anti_cmse = d["anti_cmse"]
    anti_rho  = d["anti_rho"]

    dlbt_all_cmse = d["dlbt_all_cmse"]
    dlbt_all_rho  = d["dlbt_all_rho"]
    slda_all_cmse = d["slda_all_cmse"]
    slda_all_rho  = d["slda_all_rho"]
    anti_all_cmse = d["anti_all_cmse"]
    anti_all_rho  = d["anti_all_rho"]

    random_cmse_nf    = d["random_cmse_nf"]
    rho_noise_ceiling = d.get("rho_noise_ceiling", float("nan"))

    if np.isnan(rho_noise_ceiling) and "count_matrix" in d:
        print("  Computing ρ noise ceiling from count_matrix...")
        rho_noise_ceiling = _rho_nc_from_counts(d["true_matrix"], d["count_matrix"])
        print(f"  ρ noise ceiling: {rho_noise_ceiling:.4f}")

    print(f"  Seeds: {n_seeds}  TPT grid: {list(tpt_grid)}  "
          f"avg pool/task: {avg_pool_per_task:.0f}")

    plots_dir = cfg.RESULTS_DIR / "plots" / pkl_path.stem
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _make_figure(metric: str):
        is_cmse = metric == "cmse"

        dlbt_mu, dlbt_sem = _mean_sem(dlbt_cmse if is_cmse else dlbt_rho)
        slda_mu, slda_sem = _mean_sem(slda_cmse if is_cmse else slda_rho)
        anti_mu, anti_sem = _mean_sem(anti_cmse if is_cmse else anti_rho)

        dlbt_all_mu, dlbt_all_sem = _mean_sem_scalar(
            dlbt_all_cmse if is_cmse else dlbt_all_rho)
        slda_all_mu, slda_all_sem = _mean_sem_scalar(
            slda_all_cmse if is_cmse else slda_all_rho)
        anti_all_mu, anti_all_sem = _mean_sem_scalar(
            anti_all_cmse if is_cmse else anti_all_rho)

        fig, ax = plt.subplots(figsize=(5.0, 4.5))

        # ── Reference lines ──────────────────────────────────────────────────
        if is_cmse:
            ax.axhline(random_cmse_nf, color=cfg.C_RNDINI, lw=1.5,
                       ls=(0, (4, 3)), zorder=1)
            ax.annotate("chance (P=0.5)",
                        xy=(1.0, random_cmse_nf),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 5), textcoords="offset points",
                        color=cfg.C_RNDINI, fontsize=8, style="italic",
                        va="bottom", ha="right", zorder=6)
        else:
            if not np.isnan(rho_noise_ceiling):
                ax.axhline(rho_noise_ceiling, color="#555555", lw=1.5,
                           ls=(0, (2, 2)), zorder=2)
                ax.annotate("noise ceiling",
                            xy=(0.0, rho_noise_ceiling),
                            xycoords=("axes fraction", "data"),
                            xytext=(4, 5), textcoords="offset points",
                            color="#555555", fontsize=8, style="italic",
                            va="bottom", ha="left", zorder=6)

        # ── Traces ───────────────────────────────────────────────────────────
        _plot_trace(ax, tpt_grid, anti_mu, anti_sem, cfg.C_ANTI, zorder=2)
        _plot_trace(ax, tpt_grid, slda_mu, slda_sem, cfg.C_SLDA, zorder=3)
        _plot_trace(ax, tpt_grid, dlbt_mu, dlbt_sem, cfg.C_DLBT, zorder=4)

        # ── All-data markers ─────────────────────────────────────────────────
        _plot_all_data_marker(ax, avg_pool_per_task,
                              anti_all_mu, anti_all_sem, cfg.C_ANTI, zorder=4)
        _plot_all_data_marker(ax, avg_pool_per_task,
                              slda_all_mu, slda_all_sem, cfg.C_SLDA, zorder=5)
        _plot_all_data_marker(ax, avg_pool_per_task,
                              dlbt_all_mu, dlbt_all_sem, cfg.C_DLBT, zorder=6)

        _xaxis_setup(ax, tpt_grid, avg_pool_per_task)

        # ── Y axis ───────────────────────────────────────────────────────────
        if is_cmse:
            ax.set_ylabel("cMSE − noise floor", fontsize=11, fontweight="bold")
            if _log_y:
                ax.set_yscale("log")
                ax.set_yticks([0.01, 0.1, 1])
                ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
                ax.yaxis.set_minor_locator(plt.NullLocator())
                ax.set_ylim(0.008, 1.0)
            else:
                ax.set_ylim(0, 0.34)
        else:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=11, fontweight="bold")
            ax.set_ylim(-1, 1)

        # ── Stacked annotations bottom-left (cMSE only) ─────────────────────
        if is_cmse:
            for k, (lbl, col) in enumerate([
                ("anti-human DLBT", cfg.C_ANTI),
                ("SLDA",            cfg.C_SLDA),
                ("DLBT",            cfg.C_DLBT),
            ]):
                ax.text(0.03, 0.03 + k * 0.045, lbl,
                        transform=ax.transAxes,
                        color=col, fontsize=8, fontweight="bold", style="italic",
                        va="bottom", ha="left", zorder=6)

        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.tight_layout()

        tag = "cmse" if is_cmse else "rho"
        out = plots_dir / f"plot_{tag}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.relative_to(cfg.RESULTS_DIR)}")

    _make_figure("cmse")
    _make_figure("rho")

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print(f"  {'Model':<20}  {'tpt':>8}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}")
    print("  " + "-" * 70)

    for label, cmse_arr, rho_arr, cmse_all, rho_all in [
        ("DLBT",       dlbt_cmse, dlbt_rho, dlbt_all_cmse, dlbt_all_rho),
        ("SLDA",       slda_cmse, slda_rho, slda_all_cmse, slda_all_rho),
        ("Anti-human", anti_cmse, anti_rho, anti_all_cmse, anti_all_rho),
    ]:
        for b_i, tpt in enumerate(tpt_grid):
            mu_c, sem_c = _mean_sem_scalar(cmse_arr[:, b_i])
            mu_r, sem_r = _mean_sem_scalar(rho_arr[:, b_i])
            print(f"  {label:<20}  {int(tpt):>8,}  "
                  f"{mu_c:+.5f} ± {sem_c:.5f}  "
                  f"{mu_r:+.4f} ± {sem_r:.4f}")
        mu_c, sem_c = _mean_sem_scalar(cmse_all)
        mu_r, sem_r = _mean_sem_scalar(rho_all)
        print(f"  {label:<20}  {'all data':>8}  "
              f"{mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}")
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for pkl_path in pkl_paths:
    process_pkl(pkl_path)
