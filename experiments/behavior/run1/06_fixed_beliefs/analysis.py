"""
run1/06_fixed_beliefs/analysis.py — oracle overlay plots.

Loads the oracle_beliefs.pkl produced by run.py, then reproduces the
02_data_efficiency and 022_data_efficiency_arity budget-sweep figures with
the oracle cMSE and Spearman ρ added as horizontal reference lines.

Figures produced
----------------
  plot_cov_cmse_<cov_tag>_c<conc>.png   — coverage cMSE plot + oracle line
  plot_cov_rho_<cov_tag>_c<conc>.png    — coverage ρ plot   + oracle line
  plot_arity_cmse_<arity_tag>_c<conc>.png — arity cMSE plot + oracle line
  plot_arity_rho_<arity_tag>_c<conc>.png  — arity ρ plot    + oracle line

Run from repo root:
    python experiments/behavior/run1/06_fixed_beliefs/analysis.py [--log-y]
           [--oracle-pkl PATH]   # default: auto-find oracle_beliefs.pkl
           [--cov-pkl PATH]      # default: auto-find 02 coverage_sweep_*.pkl
           [--arity-pkl PATH]    # default: auto-find 022 arity_sweep_*.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--log-y", action="store_true", default=False)
parser.add_argument("--oracle-pkl", default=None,
                    help="Path to oracle_beliefs.pkl (default: auto-discover).")
parser.add_argument("--cov-pkl", default=None,
                    help="Path to 02 coverage_sweep_*.pkl (default: auto-discover).")
parser.add_argument("--arity-pkl", default=None,
                    help="Path to 022 arity_sweep_*.pkl (default: auto-discover).")
args = parser.parse_args()

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load oracle results
# ---------------------------------------------------------------------------
if args.oracle_pkl:
    oracle_path = Path(args.oracle_pkl)
else:
    candidates = sorted(cfg.RESULTS_DIR.glob("oracle_beliefs*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle_beliefs*.pkl in {cfg.RESULTS_DIR}. Run run.py first.")
    oracle_path = candidates[-1]

with open(oracle_path, "rb") as f:
    oracle = pickle.load(f)

oracle_cmse = oracle["oracle_cmse_net"]
oracle_rho  = oracle["oracle_rho"]
conc        = oracle["concentration"]
print(f"Oracle loaded: c={conc}  cMSE−NF={oracle_cmse:.5f}  ρ={oracle_rho:.4f}")

_conc_tag = f"c{conc:.0f}"


# ---------------------------------------------------------------------------
# Helpers shared between coverage and arity
# ---------------------------------------------------------------------------

def _rho(pred_mat: np.ndarray, true_mat: np.ndarray) -> float:
    valid = ~np.isnan(pred_mat) & ~np.isnan(true_mat)
    if valid.sum() < 2:
        return float("nan")
    r, _ = spearmanr(pred_mat[valid], true_mat[valid])
    return float(r)


def _set_x_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1, 1.0e5)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels(["0", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Total trial budget", fontsize=11)


def _set_log_y(ax):
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1.0)
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])


def _oracle_line(ax, value, label):
    """Add oracle as a dashed red horizontal reference line."""
    ax.axhline(value, color=cfg.C_ORACLE, lw=1.5, ls="--",
               label=label, zorder=2)


def _save(fig, fname):
    out = plots_dir / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Inset helper (delta bars) — same as 02/022 analysis
# ---------------------------------------------------------------------------
def _add_delta_inset(ax, keys, dlbt_traces, slda_dict, colors,
                     inset_bounds=(0.13, 0.42, 0.30, 0.22)):
    all_dlbt_budgets: set[int] = set()
    for k in keys:
        all_dlbt_budgets |= set(dlbt_traces[k].keys())
    bar_budgets = sorted(
        b for b in slda_dict
        if not np.isnan(slda_dict.get(b, np.nan)) and b in all_dlbt_budgets
        and b >= 10 and np.log10(b) % 1 == 0
    )
    if not bar_budgets:
        return
    n_groups = len(bar_budgets)
    n_bars   = len(keys)
    group_w  = 0.80
    bar_w    = group_w / n_bars
    axins = ax.inset_axes(inset_bounds)
    axins.set_facecolor("none")
    for bar_i, key in enumerate(keys):
        trace = dlbt_traces[key]
        xs, ys = [], []
        for g_i, b in enumerate(bar_budgets):
            if b not in trace:
                continue
            dlbt_mean = float(np.nanmean(trace[b]))
            slda_val  = slda_dict[b]
            if np.isnan(dlbt_mean) or np.isnan(slda_val):
                continue
            xs.append(g_i - group_w / 2 + (bar_i + 0.5) * bar_w)
            ys.append(dlbt_mean - slda_val)
        if xs:
            axins.bar(xs, ys, width=bar_w * 0.85, color=colors[key], alpha=0.85)
    axins.axhline(0, color="black", lw=0.7, zorder=5)
    axins.set_xticks(range(n_groups))
    axins.set_xticklabels(
        [r"$10^{" + str(int(np.log10(b))) + r"}$" for b in bar_budgets],
        fontsize=7,
    )
    axins.set_xlim(-0.5, n_groups - 0.5)
    axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, symmetric=True))
    axins.tick_params(axis="y", labelsize=7, labelrotation=90)
    axins.set_ylabel(r"$\Delta$ cMSE$-$NF", fontsize=8, labelpad=2)
    sns.despine(ax=axins, top=True, right=True)


# ===========================================================================
# Coverage sweep overlay
# ===========================================================================
if args.cov_pkl:
    cov_candidates = [Path(args.cov_pkl)]
else:
    _cov_dir = Path(__file__).parents[1] / "02_data_efficiency" / "results"
    cov_candidates = sorted(_cov_dir.glob("coverage_sweep_*.pkl"))

for cov_path in cov_candidates:
    cov_tag = cov_path.stem[len("coverage_sweep_"):]
    print(f"\n=== Coverage: {cov_path.name} ===")

    with open(cov_path, "rb") as f:
        summary = pickle.load(f)

    all_tasks   = summary["all_tasks_ordered"]
    true_matrix = summary["true_matrix"]
    slda_res    = summary["slda"]
    dlbt_res    = summary["dlbt"]
    cov_fracs   = summary["coverage_fracs"]

    # Build traces {frac: {budget: [val_per_seed]}}
    dlbt_traces:     dict[float, dict[int, list]] = {f: {} for f in cov_fracs}
    dlbt_rho_traces: dict[float, dict[int, list]] = {f: {} for f in cov_fracs}

    for seed_data in dlbt_res.values():
        for frac in cov_fracs:
            frac_key = f"{frac:.2f}"
            if frac_key not in seed_data["coverage"]:
                continue
            for bstr, bdata in seed_data["coverage"][frac_key]["budgets"].items():
                b = int(bstr)
                dlbt_traces[frac].setdefault(b, []).append(
                    bdata.get("probe_cmse_net", float("nan")))
                pm = bdata.get("pred_matrix")
                dlbt_rho_traces[frac].setdefault(b, []).append(
                    _rho(pm, true_matrix) if pm is not None else float("nan"))

    slda_budgets = sorted(int(b) for b in slda_res["budgets"])
    slda_y       = [slda_res["budgets"][str(b)].get("probe_cmse_net", float("nan"))
                    for b in slda_budgets]
    slda_dict    = {b: slda_y[i] for i, b in enumerate(slda_budgets)}

    random_cmse_net = oracle.get("random_cmse_net",
                                  summary.get("random_cmse_net", float("nan")))

    # --- cMSE plot ---
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for frac in cov_fracs:
        trace = dlbt_traces[frac]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.cov_color(frac)
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DLBT {frac:.0%} coverage", zorder=4)
        if any(s > 0 for s in sems):
            lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]
            hi = [m + s            for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    ax.plot(slda_budgets, slda_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    _inset_bounds = [0.13, 0.09, 0.30, 0.22] if args.log_y else [0.13, 0.42, 0.30, 0.22]
    _add_delta_inset(ax, cov_fracs, dlbt_traces, slda_dict,
                     colors={f: cfg.cov_color(f) for f in cov_fracs},
                     inset_bounds=_inset_bounds)

    if not np.isnan(random_cmse_net):
        ax.axhline(random_cmse_net, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)

    _oracle_line(ax, oracle_cmse, f"Oracle beliefs (c={conc:.0f})")

    _set_x_axis(ax)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if args.log_y:
        _set_log_y(ax)
    else:
        ax.set_ylim(0, 0.34)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _save(fig, f"plot_cov_cmse_{cov_tag}_{_conc_tag}.png")

    # --- ρ plot ---
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for frac in cov_fracs:
        trace = dlbt_rho_traces[frac]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.cov_color(frac)
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DLBT {frac:.0%} coverage", zorder=4)
        if any(s > 0 for s in sems):
            lo = [m - s for m, s in zip(means, sems)]
            hi = [m + s for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    slda_rho_y = [_rho(slda_res["budgets"][str(b)].get("pred_matrix"), true_matrix)
                  for b in slda_budgets]
    ax.plot(slda_budgets, slda_rho_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5, zorder=3)

    ax.axhline(0, color="#999999", lw=1.5, ls=(0, (4, 3)), zorder=1)
    _oracle_line(ax, oracle_rho, f"Oracle beliefs (c={conc:.0f})")

    _set_x_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
    ax.set_ylim(0, 1)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _save(fig, f"plot_cov_rho_{cov_tag}_{_conc_tag}.png")


# ===========================================================================
# Arity sweep overlay
# ===========================================================================
if args.arity_pkl:
    arity_candidates = [Path(args.arity_pkl)]
else:
    _arity_dir = Path(__file__).parents[1] / "022_data_efficiency_arity" / "results"
    arity_candidates = sorted(_arity_dir.glob("arity_sweep_*.pkl"))

for arity_path in arity_candidates:
    arity_tag = arity_path.stem[len("arity_sweep_"):]
    print(f"\n=== Arity: {arity_path.name} ===")

    with open(arity_path, "rb") as f:
        summary = pickle.load(f)

    true_matrix = summary["true_matrix"]
    slda_res    = summary["slda"]
    dlbt_res    = summary["dlbt"]
    arities     = summary["arities"]

    random_cmse_net = oracle.get("random_cmse_net",
                                  summary.get("random_cmse_net", float("nan")))

    # Build traces {arity: {budget: [val_per_seed]}}
    dlbt_traces:     dict[int, dict[int, list]] = {a: {} for a in arities}
    dlbt_rho_traces: dict[int, dict[int, list]] = {a: {} for a in arities}

    for seed_data in dlbt_res.values():
        for arity in arities:
            arity_key = str(arity)
            if arity_key not in seed_data["arity"]:
                continue
            for bstr, bdata in seed_data["arity"][arity_key]["budgets"].items():
                b = int(bstr)
                dlbt_traces[arity].setdefault(b, []).append(
                    bdata.get("probe_cmse_net", float("nan")))
                pm = bdata.get("pred_matrix")
                dlbt_rho_traces[arity].setdefault(b, []).append(
                    _rho(pm, true_matrix) if pm is not None else float("nan"))

    slda_budgets = sorted(int(b) for b in slda_res["budgets"])
    slda_y       = [slda_res["budgets"][str(b)].get("probe_cmse_net", float("nan"))
                    for b in slda_budgets]
    slda_dict    = {b: slda_y[i] for i, b in enumerate(slda_budgets)}

    # --- cMSE plot ---
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for arity in arities:
        trace = dlbt_traces[arity]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.ARITY_COLOR[arity]
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DLBT {arity}-way tasks", zorder=4)
        if any(s > 0 for s in sems):
            lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]
            hi = [m + s            for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    ax.plot(slda_budgets, slda_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    _inset_bounds = [0.13, 0.09, 0.30, 0.22] if args.log_y else [0.13, 0.42, 0.30, 0.22]
    _add_delta_inset(ax, arities, dlbt_traces, slda_dict,
                     colors=cfg.ARITY_COLOR,
                     inset_bounds=_inset_bounds)

    if not np.isnan(random_cmse_net):
        ax.axhline(random_cmse_net, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)

    _oracle_line(ax, oracle_cmse, f"Oracle beliefs (c={conc:.0f})")

    _set_x_axis(ax)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if args.log_y:
        _set_log_y(ax)
    else:
        ax.set_ylim(0, 0.34)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _save(fig, f"plot_arity_cmse_{arity_tag}_{_conc_tag}.png")

    # --- ρ plot ---
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for arity in arities:
        trace = dlbt_rho_traces[arity]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.ARITY_COLOR[arity]
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DLBT {arity}-way tasks", zorder=4)
        if any(s > 0 for s in sems):
            lo = [m - s for m, s in zip(means, sems)]
            hi = [m + s for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    slda_rho_y = [_rho(slda_res["budgets"][str(b)].get("pred_matrix"), true_matrix)
                  for b in slda_budgets]
    ax.plot(slda_budgets, slda_rho_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5, zorder=3)

    ax.axhline(0, color="#999999", lw=1.5, ls=(0, (4, 3)), zorder=1)
    _oracle_line(ax, oracle_rho, f"Oracle beliefs (c={conc:.0f})")

    _set_x_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
    ax.set_ylim(0, 1)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _save(fig, f"plot_arity_rho_{arity_tag}_{_conc_tag}.png")

if not cov_candidates and not arity_candidates:
    print("\nNo coverage or arity pkls found. Run 02 and 022 first.")
else:
    print(f"\nAll plots saved to {plots_dir}")
