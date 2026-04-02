"""
Simulation 03 — belief distribution robustness analysis.

Loads results_frozen.pkl / results_attnpool.pkl and produces:
  plot_robustness_cmse_{tag}.png   — cMSE across distributions
  plot_robustness_rho_{tag}.png    — ρ    across distributions
  plot_robustness_reldeg_{tag}.png — relative cMSE degradation vs Dirichlet baseline
"""

import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# Colours / bar styles
# ---------------------------------------------------------------------------
C_DLBT = "#C44F52"
C_SLDA = "#7D6EAE"

# (cond, facecolor, edgecolor, hatch, linewidth, alpha, label, model)
COND_STYLES = [
    ("train", "none", C_DLBT, "",    1.5, 1.00, "DLBT — train",     "dlbt"),
    ("task",  C_DLBT, C_DLBT, "",    1.0, 0.85, "DLBT — task gen",  "dlbt"),
    ("stim",  C_DLBT, C_DLBT, "///", 0.5, 0.70, "DLBT — stim gen",  "dlbt"),
    ("joint", C_DLBT, C_DLBT, "xxx", 0.5, 0.70, "DLBT — joint gen", "dlbt"),
    ("train", "none", C_SLDA, "",    1.5, 1.00, "SLDA — train",     "slda"),
    ("stim",  C_SLDA, C_SLDA, "///", 0.5, 0.70, "SLDA — stim gen",  "slda"),
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
def _plot_metric(ax, dlbt, slda, metric: str, ylabel: str, distributions, dist_labels):
    n_dist  = len(distributions)
    n_bars  = len(COND_STYLES)
    width   = 0.16
    spacing = 1.6
    x       = np.arange(n_dist) * spacing
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    for i, (cond, fc, ec, hatch, lw, alpha, label, model) in enumerate(COND_STYLES):
        src  = dlbt[cond][metric] if model == "dlbt" else slda[cond][metric]
        mean = np.nanmean(src, axis=0)
        std  = np.nanstd(src, axis=0)

        ax.bar(x + offsets[i], mean, width,
               facecolor=fc, edgecolor=ec, hatch=hatch,
               linewidth=lw, alpha=alpha, label=label)
        ax.errorbar(x + offsets[i], mean, yerr=std,
                    fmt="none", color="black", capsize=2.5, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels[d] for d in distributions],
                       rotation=0, ha="center", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    if metric == "cmse":
        ax.set_ylim(bottom=0)
    elif metric == "rho":
        ax.set_ylim(-0.1, 1)


def _legend_handles():
    handles = []
    for _, fc, ec, hatch, lw, alpha, label, _ in COND_STYLES:
        handles.append(mpatches.Patch(
            facecolor=fc, edgecolor=ec, hatch=hatch,
            linewidth=lw, alpha=alpha, label=label,
        ))
    return handles


# ---------------------------------------------------------------------------
# Loop over available result files
# ---------------------------------------------------------------------------
for results_path in available:
    run_tag = results_path.stem[len("results_"):]

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    distributions = res["distributions"]
    dist_labels   = res["dist_labels"]
    dlbt          = res["dlbt"]
    slda          = res["slda"]

    handles = _legend_handles()

    # -----------------------------------------------------------------------
    # Plot 1 — cMSE
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3))
    _plot_metric(ax, dlbt, slda, "cmse", "cMSE", distributions, dist_labels)
    ax.legend(handles=handles, fontsize=9, ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_cmse_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2 — ρ
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3))
    _plot_metric(ax, dlbt, slda, "rho", "Spearman ρ", distributions, dist_labels)
    ax.legend(handles=handles, fontsize=9, ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_rho_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3 — relative cMSE degradation vs Dirichlet baseline
    # -----------------------------------------------------------------------
    if "dirichlet" not in distributions:
        print("No dirichlet baseline found — skipping reldeg plot.")
        continue

    alt_dists = [d for d in distributions if d != "dirichlet"]
    d0_idx    = distributions.index("dirichlet")
    x_rel     = np.arange(len(alt_dists))
    n_bars    = len(COND_STYLES)
    width_rel = 0.16
    offsets_rel = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width_rel

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (cond, fc, ec, hatch, lw, alpha, label, model) in enumerate(COND_STYLES):
        src  = dlbt[cond]["cmse"] if model == "dlbt" else slda[cond]["cmse"]
        base = src[:, d0_idx][:, None]
        rel  = (src - base) / (base + 1e-10)

        for j, d in enumerate(alt_dists):
            d_idx = distributions.index(d)
            vals  = rel[:, d_idx]
            mean  = float(np.nanmean(vals))
            std   = float(np.nanstd(vals))
            ax.bar(x_rel[j] + offsets_rel[i], mean, width_rel,
                   facecolor=fc, edgecolor=ec, hatch=hatch,
                   linewidth=lw, alpha=alpha,
                   label=label if j == 0 else "")
            ax.errorbar(x_rel[j] + offsets_rel[i], mean, yerr=std,
                        fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x_rel)
    ax.set_xticklabels([dist_labels[d] for d in alt_dists], fontsize=10)
    ax.set_ylabel("Relative cMSE change\n(vs Dirichlet baseline)", fontsize=10)
    ax.legend(handles=handles, fontsize=8, ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_reldeg_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)
