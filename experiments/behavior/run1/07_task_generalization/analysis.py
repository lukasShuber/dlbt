"""
run1/07_task_generalization/analysis.py — task generalization plots.

Produces cMSE−NF and Spearman ρ figures with:
  X-axis: training condition (1-arity, 2-arity, 3-arity, 4-arity, random)
  Y-axis: performance on held-out tasks (not seen during training)

  Bars:   mean per condition (coloured by arity)
  Error:  ±1 SEM across seeds

  Reference lines (horizontal):
    Chance (P=0.5)      — gray dashed
    Full DLBT           — red  dashed  (same colour as DLBT in 021)
    Full SLDA           — purple dashed (same colour as SLDA in 021)

  Annotation: approximate % of training data used across all conditions.

Run from repo root:
    python experiments/behavior/run1/07_task_generalization/analysis.py
    python experiments/behavior/run1/07_task_generalization/analysis.py --pkl PATH
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to a specific pkl. Default: task_generalization*.pkl.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis for cMSE plot.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Collect pkl paths
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_paths = [Path(args.pkl)]
else:
    pkl_paths = sorted(cfg.RESULTS_DIR.glob("task_generalization*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(
            f"No task_generalization*.pkl found in {cfg.RESULTS_DIR}")

print(f"Processing {len(pkl_paths)} pkl(s):")
for p in pkl_paths:
    print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Colors (local override — sequential red scale for arity, green for random)
# ---------------------------------------------------------------------------
COND_COLOR = {
    "1-arity": "#fcbba1",
    "2-arity": "#fb6a4a",
    "3-arity": "#cb181d",
    "4-arity": "#67000d",
    "random":  "#1b9e77",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), 0.0
    mu  = float(np.mean(valid))
    sem = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
    return mu, sem


# ---------------------------------------------------------------------------
# Per-pkl processing
# ---------------------------------------------------------------------------

def process_pkl(pkl_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading: {pkl_path.name}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    conditions      = d["conditions"]          # ["1-arity", "2-arity", ...]
    gen_cmse        = d["gen_cmse"]            # dict cond → [n_seeds]
    gen_rho         = d["gen_rho"]
    ref_dlbt_cmse   = d["ref_dlbt_cmse"]       # [n_seeds]
    ref_dlbt_rho    = d["ref_dlbt_rho"]
    ref_slda_cmse   = d["ref_slda_cmse"]
    ref_slda_rho    = d["ref_slda_rho"]
    random_cmse_nf  = d["random_cmse_net"]
    rho_nc          = d.get("rho_noise_ceiling", float("nan"))
    k_tasks         = d.get("k_tasks", "?")
    n_all_tasks     = len(d.get("all_tasks_ordered", []))
    total_pool_size = d.get("total_pool_size", None)

    # ---- Data-volume annotation -----------------------------------------------
    # Approximate fraction of data used for each arity condition:
    # k tasks out of n_all_tasks, trials roughly uniform across tasks.
    if isinstance(k_tasks, int) and n_all_tasks > 0:
        pct = k_tasks / n_all_tasks * 100
        data_label = f"Models trained on ~{pct:.0f}% of training data"
    else:
        data_label = f"Models trained on {k_tasks} tasks"

    # ---- Reference line scalars (mean over seeds) -----------------------------
    ref_dlbt_mu_cmse, ref_dlbt_sem_cmse = _mean_sem(ref_dlbt_cmse)
    ref_dlbt_mu_rho,  ref_dlbt_sem_rho  = _mean_sem(ref_dlbt_rho)
    ref_slda_mu_cmse, ref_slda_sem_cmse = _mean_sem(ref_slda_cmse)
    ref_slda_mu_rho,  ref_slda_sem_rho  = _mean_sem(ref_slda_rho)

    plots_dir = cfg.RESULTS_DIR / "plots" / pkl_path.stem
    plots_dir.mkdir(parents=True, exist_ok=True)

    bar_w = 0.55

    def _make_figure(metric: str):
        is_cmse = metric == "cmse"
        data    = gen_cmse if is_cmse else gen_rho

        arity_conds  = [c for c in conditions if c != "random"]
        has_random   = "random" in conditions

        # ── Two-panel broken x-axis (arity | random) ─────────────────────────
        n_arity   = len(arity_conds)
        r_ratio   = 1.1 if has_random else 0     # width ratio of random panel
        ratios    = [n_arity, r_ratio] if has_random else [n_arity]
        ncols     = 2 if has_random else 1

        fig, axes = plt.subplots(
            1, ncols,
            figsize=(6.5, 4.5),
            gridspec_kw={"width_ratios": ratios, "wspace": 0.07},
            sharey=True,
        )
        ax_l = axes[0] if has_random else axes
        ax_r = axes[1] if has_random else None

        def _axhline_both(val, **kw):
            ax_l.axhline(val, **kw)
            if ax_r is not None:
                ax_r.axhline(val, **kw)

        # ── Reference lines ──────────────────────────────────────────────────
        if is_cmse:
            _axhline_both(random_cmse_nf, color=cfg.C_CHANCE, lw=1.5,
                          ls=(0, (4, 3)), zorder=1)
            ax_l.annotate("chance (P=0.5)",
                          xy=(1.0, random_cmse_nf),
                          xycoords=("axes fraction", "data"),
                          xytext=(-4, -5), textcoords="offset points",
                          color=cfg.C_CHANCE, fontsize=8, style="italic",
                          va="top", ha="right", zorder=6)

        ref_dlbt_val = ref_dlbt_mu_cmse if is_cmse else ref_dlbt_mu_rho
        if not np.isnan(ref_dlbt_val):
            _axhline_both(ref_dlbt_val, color="#C0392B", lw=1.5, ls="--",
                          zorder=2, label="Full DLBT (all tasks)")

        ref_slda_val = ref_slda_mu_cmse if is_cmse else ref_slda_mu_rho
        if not np.isnan(ref_slda_val):
            _axhline_both(ref_slda_val, color="#7D3C98", lw=1.5, ls="--",
                          zorder=2, label="Full SLDA (all tasks)")

        if not is_cmse and not np.isnan(rho_nc):
            _axhline_both(rho_nc, color="#555555", lw=1.5,
                          ls=(0, (2, 2)), zorder=2)
            ax_l.annotate("noise ceiling",
                          xy=(1.0, rho_nc),
                          xycoords=("axes fraction", "data"),
                          xytext=(-4, 5), textcoords="offset points",
                          color="#555555", fontsize=8, style="italic",
                          va="bottom", ha="right", zorder=6)

        # ── Bars — arity conditions on ax_l ──────────────────────────────────
        for x_i, cond in enumerate(arity_conds):
            mu, sem = _mean_sem(data[cond])
            if np.isnan(mu):
                continue
            color = COND_COLOR.get(cond, "#888888")
            ax_l.bar(x_i, mu, width=bar_w, color=color, alpha=0.88,
                     zorder=3, linewidth=0.8, edgecolor="white")
            ax_l.errorbar(x_i, mu, yerr=sem, fmt="none", color="#333333",
                          capsize=5, capthick=1.4, elinewidth=1.4, zorder=5)

        # ── Bar — random on ax_r ──────────────────────────────────────────────
        if ax_r is not None and "random" in data:
            mu, sem = _mean_sem(data["random"])
            if not np.isnan(mu):
                color = COND_COLOR["random"]
                ax_r.bar(0, mu, width=bar_w, color=color, alpha=0.88,
                         zorder=3, linewidth=0.8, edgecolor="white")
                ax_r.errorbar(0, mu, yerr=sem, fmt="none", color="#333333",
                              capsize=5, capthick=1.4, elinewidth=1.4, zorder=5)

        # ── Broken-axis styling ───────────────────────────────────────────────
        ax_l.spines["right"].set_visible(False)
        sns.despine(ax=ax_l, top=True, right=True)

        if ax_r is not None:
            ax_r.spines["left"].set_visible(False)
            sns.despine(ax=ax_r, top=True, right=True, left=True)
            ax_r.tick_params(left=False)

            # Diagonal break marks — bottom only
            d    = 0.022
            bkw  = dict(color="k", clip_on=False, lw=1.2,
                        transform=ax_l.transAxes)
            ax_l.plot([1 - d, 1 + d], [-d, +d], **bkw)
            bkw2 = dict(color="k", clip_on=False, lw=1.2,
                        transform=ax_r.transAxes)
            ax_r.plot([-d, +d], [-d, +d], **bkw2)

        # ── X axes ───────────────────────────────────────────────────────────
        ax_l.set_xticks(range(n_arity))
        # Short numeric labels (1, 2, 3, 4) instead of "1-arity" etc.
        ax_l.set_xticklabels([str(i + 1) for i in range(n_arity)], fontsize=10)
        ax_l.set_xlim(-0.6, n_arity - 0.4)
        # X-label centred under the arity bars only
        ax_l.set_xlabel("Task complexity", fontsize=11, fontweight="bold")

        if ax_r is not None:
            ax_r.set_xticks([0])
            ax_r.set_xticklabels(["random"], fontsize=10)
            ax_r.set_xlim(-0.6, 0.6)

        # ── Y axis ───────────────────────────────────────────────────────────
        if is_cmse:
            ax_l.set_ylabel("cMSE − noise floor\n(held-out tasks)",
                            fontsize=11, fontweight="bold")
            if args.log_y:
                ax_l.set_yscale("log")
            else:
                ax_l.set_ylim(bottom=0)
        else:
            ax_l.set_ylabel(r"Spearman $\rho$" + "\n(held-out tasks)",
                            fontsize=11, fontweight="bold")
            ax_l.set_ylim(0, 1)

        # ── Legend & title ────────────────────────────────────────────────────
        leg_anchor = (0.01, 0.88) if is_cmse else (0.01, 1.00)
        ax_l.legend(loc="upper left", bbox_to_anchor=leg_anchor,
                    fontsize=8, frameon=False, handlelength=2.0)

        ax_l.set_title(data_label, fontsize=8, color="#666666",
                       style="italic", pad=6)

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
    print(f"  {data_label}")
    print(f"  {'Condition':<12}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}  seeds")
    print("  " + "-" * 62)
    for cond in conditions:
        mu_c, sem_c = _mean_sem(gen_cmse[cond])
        mu_r, sem_r = _mean_sem(gen_rho[cond])
        n           = int(np.sum(~np.isnan(gen_cmse[cond])))
        print(f"  {cond:<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}  n={n}")
    print()
    mu_c, sem_c = _mean_sem(ref_dlbt_cmse)
    mu_r, sem_r = _mean_sem(ref_dlbt_rho)
    print(f"  {'Full DLBT':<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}  (ref, all tasks)")
    mu_c, sem_c = _mean_sem(ref_slda_cmse)
    mu_r, sem_r = _mean_sem(ref_slda_rho)
    print(f"  {'Full SLDA':<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}  (ref, all tasks)")
    print(f"  {'Chance':<12}  {random_cmse_nf:+.5f}  (ref)")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for pkl_path in pkl_paths:
    process_pkl(pkl_path)
