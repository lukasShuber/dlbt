"""
run1/07_task_generalization/analysis.py — task generalization plots.

Produces cMSE−NF and Spearman ρ figures with:
  X-axis: training condition (1-arity, 2-arity, 3-arity, 4-arity, random)
  Y-axis: performance on held-out tasks (not seen during training)

  Per seed: small translucent dot (shows variance)
  Mean:     fat filled dot ± SEM error bar

  Reference lines (horizontal):
    Chance (P=0.5)            — gray dashed
    Full DLBT (all tasks)     — dark blue dashed
    Full SLDA (all tasks)     — purple dashed

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
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem_scalar(arr: np.ndarray):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), 0.0
    mu  = float(np.mean(valid))
    sem = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
    return mu, sem


def _jitter(n: int, width: float = 0.08, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(-width, width, size=n)


# ---------------------------------------------------------------------------
# Per-pkl processing
# ---------------------------------------------------------------------------

def process_pkl(pkl_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading: {pkl_path.name}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    conditions     = d["conditions"]       # ["1-arity", "2-arity", ...]
    gen_cmse       = d["gen_cmse"]         # dict cond → [n_seeds]
    gen_rho        = d["gen_rho"]
    ref_dlbt_cmse  = d["ref_dlbt_cmse"]   # [n_seeds]
    ref_dlbt_rho   = d["ref_dlbt_rho"]
    ref_slda_cmse  = d["ref_slda_cmse"]
    ref_slda_rho   = d["ref_slda_rho"]
    random_cmse_nf = d["random_cmse_net"]
    rho_nc         = d.get("rho_noise_ceiling", float("nan"))
    k_tasks        = d.get("k_tasks", "?")

    # Reference line scalars (mean over seeds)
    ref_dlbt_mu_cmse, _ = _mean_sem_scalar(ref_dlbt_cmse)
    ref_dlbt_mu_rho,  _ = _mean_sem_scalar(ref_dlbt_rho)
    ref_slda_mu_cmse, _ = _mean_sem_scalar(ref_slda_cmse)
    ref_slda_mu_rho,  _ = _mean_sem_scalar(ref_slda_rho)

    plots_dir = cfg.RESULTS_DIR / "plots" / pkl_path.stem
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_ticks  = np.arange(len(conditions))
    x_labels = conditions

    def _make_figure(metric: str):
        is_cmse = metric == "cmse"
        data    = gen_cmse if is_cmse else gen_rho

        fig, ax = plt.subplots(figsize=(6.0, 4.5))

        # ── Reference lines ──────────────────────────────────────────────────
        if is_cmse:
            ax.axhline(random_cmse_nf, color=cfg.C_CHANCE, lw=1.5,
                       ls=(0, (4, 3)), zorder=1)
            ax.annotate("chance (P=0.5)",
                        xy=(1.0, random_cmse_nf),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 5), textcoords="offset points",
                        color=cfg.C_CHANCE, fontsize=8, style="italic",
                        va="bottom", ha="right", zorder=6)

        ref_dlbt_val = ref_dlbt_mu_cmse if is_cmse else ref_dlbt_mu_rho
        ref_slda_val = ref_slda_mu_cmse if is_cmse else ref_slda_mu_rho

        if not np.isnan(ref_dlbt_val):
            ax.axhline(ref_dlbt_val, color=cfg.C_DLBT_REF, lw=1.5,
                       ls="--", zorder=2, label="Full DLBT (all tasks)")
        if not np.isnan(ref_slda_val):
            ax.axhline(ref_slda_val, color=cfg.C_SLDA_REF, lw=1.5,
                       ls="--", zorder=2, label="Full SLDA (all tasks)")

        if not is_cmse and not np.isnan(rho_nc):
            ax.axhline(rho_nc, color="#555555", lw=1.5,
                       ls=(0, (2, 2)), zorder=2)
            ax.annotate("noise ceiling",
                        xy=(0.0, rho_nc),
                        xycoords=("axes fraction", "data"),
                        xytext=(4, 5), textcoords="offset points",
                        color="#555555", fontsize=8, style="italic",
                        va="bottom", ha="left", zorder=6)

        # ── Per-condition scatter + mean dot ─────────────────────────────────
        for x_i, cond in enumerate(conditions):
            vals = data[cond]
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) == 0:
                continue

            n    = len(valid_vals)
            jit  = _jitter(n, width=0.12, rng_seed=x_i + 7)

            # Colour by condition arity (or "random")
            if cond == "random":
                color = cfg.ARITY_COLOR["random"]
            else:
                color = cfg.ARITY_COLOR.get(int(cond[0]), "#888888")

            # Seed dots (small, translucent)
            ax.scatter(x_i + jit, valid_vals,
                       color=color, alpha=0.40, s=30, zorder=3,
                       linewidths=0)

            # Mean ± SEM (fat dot + error bar)
            mu, sem = _mean_sem_scalar(vals)
            ax.errorbar(x_i, mu, yerr=sem, fmt="o",
                        color=color, ms=10, mfc=color, mew=1.5,
                        capsize=4, elinewidth=1.5, zorder=5)

        # ── X axis ───────────────────────────────────────────────────────────
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Training task type", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6, len(conditions) - 0.4)

        # ── Y axis ───────────────────────────────────────────────────────────
        if is_cmse:
            ax.set_ylabel("cMSE − noise floor\n(held-out tasks)",
                          fontsize=11, fontweight="bold")
            if args.log_y:
                ax.set_yscale("log")
        else:
            ax.set_ylabel(r"Spearman $\rho$" + "\n(held-out tasks)",
                          fontsize=11, fontweight="bold")
            ax.set_ylim(-0.04, 1)

        # ── Annotation: k tasks ───────────────────────────────────────────────
        ax.set_title(f"Task generalization  (k={k_tasks} training tasks / condition)",
                     fontsize=9, color="#555555", pad=6)

        sns.despine(top=True, right=True)
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
    print(f"  {'Condition':<12}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}  seeds")
    print("  " + "-" * 62)
    for cond in conditions:
        mu_c, sem_c = _mean_sem_scalar(gen_cmse[cond])
        mu_r, sem_r = _mean_sem_scalar(gen_rho[cond])
        n           = int(np.sum(~np.isnan(gen_cmse[cond])))
        print(f"  {cond:<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}  n={n}")
    print()
    mu_c, sem_c = _mean_sem_scalar(ref_dlbt_cmse)
    mu_r, sem_r = _mean_sem_scalar(ref_dlbt_rho)
    print(f"  {'Full DLBT':<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}  (ref, all tasks)")
    mu_c, sem_c = _mean_sem_scalar(ref_slda_cmse)
    mu_r, sem_r = _mean_sem_scalar(ref_slda_rho)
    print(f"  {'Full SLDA':<12}  {mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}  (ref, all tasks)")
    print(f"  {'Chance':<12}  {random_cmse_nf:+.5f}            "
          f"{'n/a':>16}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for pkl_path in pkl_paths:
    process_pkl(pkl_path)
