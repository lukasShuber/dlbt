"""
Simulation 02 — data efficiency analysis.

Loads results_frozen.pkl / results_attnpool.pkl saved by run.py and produces:
  plot_efficiency_cmse_{tag}.png  — cMSE vs. budget
  plot_efficiency_rho_{tag}.png   — ρ vs. budget
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# Colours / line styles
# ---------------------------------------------------------------------------
C_DLBT = "#C44F52"
C_SLDA = "#7D6EAE"

DLBT_CONDITIONS = [
    ("stim",  "dotted", "DLBT — stim gen"),
    ("task",  "dashed", "DLBT — task gen"),
    ("joint", "solid",  "DLBT — joint gen"),
]
SLDA_CONDITIONS = [
    ("stim",  "dotted", "SLDA — stim gen"),
]

# ---------------------------------------------------------------------------
# Auto-detect available result files
# ---------------------------------------------------------------------------
plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

available = sorted(
    [p for p in [cfg.RESULTS_DIR / "results_frozen.pkl",
                 cfg.RESULTS_DIR / "results_attnpool.pkl"]
     if p.exists()]
)
if not available:
    raise FileNotFoundError(f"No results_*.pkl found in {cfg.RESULTS_DIR}. Run run.py first.")

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_metric(ax, res, metric: str, ylabel: str, run_tag: str):
    budgets = res["budgets"]
    dlbt    = res["dlbt"]
    slda    = res["slda"]
    x       = np.array(budgets)

    model_label = "DLBT (frozen)" if run_tag == "frozen" else "DLBT (attnpool)"

    for cond, ls, label in DLBT_CONDITIONS:
        if cond not in dlbt:
            continue
        vals = dlbt[cond][metric]
        mean = np.nanmean(vals, axis=0)
        std  = np.nanstd(vals, axis=0)
        lbl  = label.replace("DLBT", model_label)
        ax.plot(x, mean, color=C_DLBT, lw=2.5, ls=ls, label=lbl)
        ax.fill_between(x, mean - std, mean + std, color=C_DLBT, alpha=0.15)

    for cond, ls, label in SLDA_CONDITIONS:
        if cond not in slda:
            continue
        vals = slda[cond][metric]
        mean = np.nanmean(vals, axis=0)
        std  = np.nanstd(vals, axis=0)
        ax.plot(x, mean, color=C_SLDA, lw=2.5, ls=ls, label=label)
        ax.fill_between(x, mean - std, mean + std, color=C_SLDA, alpha=0.10)

    # Dummy — flat reference line (stim gen condition, constant across budgets)
    dummy = res.get("dummy", {})
    if "stim" in dummy and metric in dummy["stim"]:
        vals = dummy["stim"][metric]   # [n_seeds]
        mean = float(np.nanmean(vals))
        std  = float(np.nanstd(vals))
        ax.axhline(mean, color="gray", lw=1.5, ls=(0, (3, 3)), label="Chance (P=0.5)")
        ax.axhspan(mean - std, mean + std, color="gray", alpha=0.08)

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b:,}" for b in budgets], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Total training trials", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if metric == "cmse":
        ax.set_ylim(bottom=0)
    elif metric == "rho":
        ax.set_ylim(-0.1, 1)
    ax.legend(fontsize=9, frameon=False)


# ---------------------------------------------------------------------------
# Loop over available result files
# ---------------------------------------------------------------------------
for results_path in available:
    run_tag = results_path.stem[len("results_"):]

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    for metric, ylabel, fname in [
        ("cmse", "cMSE",        f"plot_efficiency_cmse_{run_tag}.png"),
        ("rho",  "Spearman ρ",  f"plot_efficiency_rho_{run_tag}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        _plot_metric(ax, res, metric, ylabel, run_tag)
        sns.despine(trim=False)
        plt.tight_layout()
        out = plots_dir / fname
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

print("\nAll plots saved to", plots_dir)
