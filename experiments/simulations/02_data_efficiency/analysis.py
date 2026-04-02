"""
Simulation 02 — data efficiency analysis.

Auto-detects available results_{tag}.pkl files saved by run.py and produces
per-tag plots:
  plot_efficiency_cmse_{tag}.png  — cMSE vs. budget
  plot_efficiency_rho_{tag}.png   — ρ vs. budget

Run from repo root:
    python experiments/simulations/02_data_efficiency/analysis.py
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# Auto-detect available result files
# ---------------------------------------------------------------------------
available_results = sorted([
    p for tag in ("frozen", "attnpool")
    for p in [cfg.RESULTS_DIR / f"results_{tag}.pkl"]
    if p.exists()
])

if not available_results:
    sys.exit(f"No results files found in {cfg.RESULTS_DIR}.\nRun run.py first.")

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

C_TRAIN, C_STIM, C_TASK, C_JOINT = cfg.C_TRAIN, cfg.C_STIM, cfg.C_TASK, cfg.C_JOINT

for results_path in available_results:
    run_tag = results_path.stem[len("results_"):]   # e.g. "frozen" or "attnpool"
    print(f"\n--- Analysing {results_path.name} (tag={run_tag}) ---")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    budgets = res["budgets"]
    dlbt    = res["dlbt"]
    slda    = res["slda"]

    model_label = "DLBT (frozen)" if run_tag == "frozen" else "DLBT (attnpool)"

    # ---------------------------------------------------------------------------
    # Plotting helper
    # ---------------------------------------------------------------------------
    def _plot_metric(ax, metric: str, ylabel: str):
        """Draw all conditions for both models on ax."""
        x = np.array(budgets)

        # DLBT — 4 conditions, solid lines
        for cond, color, label in [
            ("train", C_TRAIN, f"{model_label} train"),
            ("stim",  C_STIM,  f"{model_label} stim gen"),
            ("task",  C_TASK,  f"{model_label} task gen"),
            ("joint", C_JOINT, f"{model_label} joint gen"),
        ]:
            vals = dlbt[cond][metric]          # [n_seeds, n_budgets]
            mean = np.nanmean(vals, axis=0)
            std  = np.nanstd(vals, axis=0)
            ax.plot(x, mean, color=color, lw=2.5, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        # SLDA — 2 conditions, dashed lines
        # Use nanmean/nanstd: at very low budgets some seeds may have no fitted tasks
        # (too few observations per task), so NaN is expected and should not mask
        # the seeds that do have a valid fit.
        for cond, color, label in [
            ("train", C_TRAIN, "SLDA train"),
            ("stim",  C_STIM,  "SLDA stim gen"),
        ]:
            vals = slda[cond][metric]
            mean = np.nanmean(vals, axis=0)
            std  = np.nanstd(vals, axis=0)
            ax.plot(x, mean, color=color, lw=2.5, ls="--", label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

        ax.set_xscale("log")
        ax.set_xticks(budgets)
        tick_labels = [f"{b:,}" if b < 1_000_000 else "1M" for b in budgets]
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("Total training trials", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if metric == "cmse":
            ax.set_ylim(bottom=0)
        elif metric == "rho":
            ax.set_ylim(-0.1, 1)
        ax.legend(fontsize=9, ncol=2, frameon=False)

    # ---------------------------------------------------------------------------
    # Plot 1 — cMSE
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _plot_metric(ax, "cmse", "cMSE")
    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_efficiency_cmse_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ---------------------------------------------------------------------------
    # Plot 2 — ρ
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _plot_metric(ax, "rho", "Spearman ρ")
    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_efficiency_rho_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)
