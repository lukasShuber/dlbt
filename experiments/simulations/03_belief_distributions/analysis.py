"""
Simulation 03 — belief distribution robustness analysis.

Auto-detects available results_{tag}.pkl files and produces per-tag plots:
  plot_robustness_cmse_{tag}.png        — cMSE across distributions, all conditions
  plot_robustness_rho_{tag}.png         — ρ    across distributions, all conditions
  plot_robustness_reldeg_{tag}.png      — relative cMSE degradation vs Dirichlet baseline

Run from repo root:
    python experiments/simulations/03_belief_distributions/analysis.py
"""

import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

for results_path in available_results:
    run_tag = results_path.stem[len("results_"):]   # e.g. "frozen" or "attnpool"
    print(f"\n--- Analysing {results_path.name} (tag={run_tag}) ---")

    # ---------------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------------
    with open(results_path, "rb") as f:
        res = pickle.load(f)

    distributions = res["distributions"]
    dist_labels   = res["dist_labels"]
    dist_colors   = res["dist_colors"]
    dlbt          = res["dlbt"]
    slda          = res["slda"]

    n_dist  = len(distributions)
    width   = 0.16                          # individual bar width
    spacing = 1.6                           # gap between distribution groups
    x       = np.arange(n_dist) * spacing  # group centres: 0, 1.6, 3.2, 4.8

    # ---------------------------------------------------------------------------
    # Plotting helper
    # ---------------------------------------------------------------------------
    COND_STYLES = [
        # (cond,   task_names_key,  color,      label,           hatch, model)
        ("train", "train", cfg.C_TRAIN, "DLBT train",     "",   "dlbt"),
        ("stim",  "stim",  cfg.C_STIM,  "DLBT stim gen",  "",   "dlbt"),
        ("task",  "task",  cfg.C_TASK,  "DLBT task gen",  "",   "dlbt"),
        ("joint", "joint", cfg.C_JOINT, "DLBT joint gen", "",   "dlbt"),
        ("train", "train", cfg.C_TRAIN, "SLDA train",     "//", "slda"),
        ("stim",  "stim",  cfg.C_STIM,  "SLDA stim gen",  "//", "slda"),
    ]

    N_BARS = len(COND_STYLES)
    offsets = np.linspace(-(N_BARS - 1) / 2, (N_BARS - 1) / 2, N_BARS) * width

    def _plot_metric(ax, metric: str, ylabel: str):
        for i, (cond, _, color, label, hatch, model) in enumerate(COND_STYLES):
            data = dlbt[cond][metric] if model == "dlbt" else slda[cond][metric]
            mean = data.mean(axis=0)   # [n_dist]
            std  = data.std(axis=0)

            ax.bar(x + offsets[i], mean, width,
                   color=color, alpha=0.85 if model == "dlbt" else 0.5,
                   hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=label)
            ax.errorbar(x + offsets[i], mean, yerr=std,
                        fmt="none", color="black", capsize=2.5, linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([dist_labels[d] for d in distributions],
                           rotation=0, ha="center", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        if metric == "cmse":
            ax.set_ylim(bottom=0)

    # ---------------------------------------------------------------------------
    # Plot 1 — cMSE
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3))
    _plot_metric(ax, "cmse", "cMSE")

    handles = [
        mpatches.Patch(facecolor=color, alpha=0.85 if model == "dlbt" else 0.5,
                       hatch=hatch, label=label, edgecolor="gray")
        for _, _, color, label, hatch, model in COND_STYLES
    ]
    ax.legend(handles=handles, fontsize=9, ncol=2, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_cmse_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ---------------------------------------------------------------------------
    # Plot 2 — ρ
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3))
    _plot_metric(ax, "rho", "Spearman ρ")

    ax.legend(handles=handles, fontsize=9, ncol=2, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.despine(trim=False)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_rho_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ---------------------------------------------------------------------------
    # Plot 3 — relative cMSE degradation vs Dirichlet baseline
    # (cMSE_dist - cMSE_dirichlet) / cMSE_dirichlet
    # Dirichlet itself is omitted (always 0 by definition).
    # ---------------------------------------------------------------------------
    alt_dists = [d for d in distributions if d != "dirichlet"]
    d0_idx    = distributions.index("dirichlet")

    x_rel    = np.arange(len(alt_dists))
    width_rel = 0.20
    n_conds   = 6   # 4 DLBT + 2 SLDA

    offsets_rel = np.linspace(-(n_conds - 1) / 2,
                               (n_conds - 1) / 2, n_conds) * width_rel

    REL_STYLES = [
        ("train", cfg.C_TRAIN, "DLBT train",     "",   "dlbt"),
        ("stim",  cfg.C_STIM,  "DLBT stim gen",  "",   "dlbt"),
        ("task",  cfg.C_TASK,  "DLBT task gen",  "",   "dlbt"),
        ("joint", cfg.C_JOINT, "DLBT joint gen", "",   "dlbt"),
        ("train", cfg.C_TRAIN, "SLDA train",     "//", "slda"),
        ("stim",  cfg.C_STIM,  "SLDA stim gen",  "//", "slda"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (cond, color, label, hatch, model) in enumerate(REL_STYLES):
        src       = dlbt[cond]["cmse"] if model == "dlbt" else slda[cond]["cmse"]
        base      = src[:, d0_idx][:, None]          # [n_seeds, 1] Dirichlet reference
        rel       = (src - base) / (base + 1e-10)    # relative degradation per seed

        for j, d in enumerate(alt_dists):
            d_idx  = distributions.index(d)
            vals   = rel[:, d_idx]
            mean   = float(np.nanmean(vals))
            std    = float(np.nanstd(vals))
            ax.bar(x_rel[j] + offsets_rel[i], mean, width_rel,
                   color=color, alpha=0.85 if model == "dlbt" else 0.5,
                   hatch=hatch, edgecolor="white", linewidth=0.4, label=label if j == 0 else "")
            ax.errorbar(x_rel[j] + offsets_rel[i], mean, yerr=std,
                        fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x_rel)
    ax.set_xticklabels([dist_labels[d] for d in alt_dists], fontsize=10)
    ax.set_ylabel("Relative cMSE change\n(vs Dirichlet baseline)", fontsize=10)
    ax.set_title(
        f"Relative degradation under observer misspecification  "
        f"({res['n_seeds'] if 'n_seeds' in res else cfg.N_SEEDS} seeds ± 1 SD)",
        fontsize=11,
    )

    rel_handles = [
        mpatches.Patch(facecolor=color, alpha=0.85 if model == "dlbt" else 0.5,
                       hatch=hatch, label=label, edgecolor="gray")
        for _, color, label, hatch, model in REL_STYLES
    ]
    ax.legend(handles=rel_handles, fontsize=8, ncol=2, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_robustness_reldeg_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)
