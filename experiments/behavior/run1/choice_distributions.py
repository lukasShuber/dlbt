"""
experiments/behavior/run1/choice_distributions.py

Diagnostic: distribution of probe-trial choice frequencies (proportion "yes")
before and after participant filtering, plus mirror-task symmetry analysis.

Figures
  1. overall_before.png  — entire probe matrix, raw (no filtering)
  2. overall_after.png   — entire probe matrix, filtered participants only
  3. arity_before.png    — per-task arity (1-4), raw
  4. arity_after.png     — per-task arity (1-4), filtered
  5. mirror_before.png   — mirror-task symmetry analysis, raw
  6. mirror_after.png    — mirror-task symmetry analysis, filtered

Mirror analysis (figs 5-6):
  For every task t, its mirror t' is obtained by flipping every dimension
  condition (right↔left, large↔small, transparent↔opaque, glossy↔matte).
  If the ontology and stimulus sampling were perfectly symmetric one would
  expect P(yes|t) + P(yes|t') ≈ 1, i.e. the pair average ≈ 0.5.
  Departures indicate perceptual or benchmark asymmetry rather than a
  generic human "no-bias".

Each histogram marks the mean (green) and median (orange).

Run from repo root:
    python experiments/behavior/run1/choice_distributions.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
_RUN1_DIR  = Path(__file__).parent
_REPO_ROOT = _RUN1_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_RUN1_DIR.parent / "run0"))   # for preprocess.py

import config as cfg                                  # run1/config.py
from preprocess import filter_assignments, aggregate_counts

PLOTS_DIR = _RUN1_DIR / "results" / "choice_distributions"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
C_MEAN      = "#27AE60"   # green  — mean marker
C_MEDIAN    = "#E67E22"   # orange — median marker
C_BAR       = "#555577"   # neutral bar color for overall figures

# ── helpers ──────────────────────────────────────────────────────────────────

def _task_arity(task_name: str) -> int:
    """Arity = number of '_and_' separators + 1."""
    return task_name.count("_and_") + 1


def _build_probe_freqs(df: pd.DataFrame, all_tasks: list[str]) -> dict:
    """
    Aggregate probe-trial choice frequencies from *df* (already filtered or raw).

    Returns a dict:
        freqs_by_task[task_name] -> np.ndarray of shape [n_cells] with values
        in [0, 1]  (proportion "yes" / count_1 / total).
    """
    probe_df = df[df["trial_kind"] == "probe"].copy()
    probe_df = probe_df[probe_df["task_id"].isin(cfg.BEH_ID_TO_TASK)]
    probe_df["task_name"] = probe_df["task_id"].map(cfg.BEH_ID_TO_TASK)
    probe_df = probe_df[probe_df["task_name"].isin(set(all_tasks))]
    probe_df["uid"]    = probe_df["stimulus"].str.slice(0, 6)
    probe_df["action"] = (probe_df["choice"] == "yes").astype(int)

    grp = (
        probe_df
        .groupby(["uid", "task_name"])["action"]
        .agg(count_1="sum", n="count")
        .reset_index()
    )
    grp["freq"] = grp["count_1"] / grp["n"]

    freqs_by_task = {
        t: grp.loc[grp["task_name"] == t, "freq"].values
        for t in all_tasks
        if t in grp["task_name"].values
    }
    return freqs_by_task


# ── data loading ─────────────────────────────────────────────────────────────

print("Loading behavioural data...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
print(f"  Raw: {len(df_raw):,} trials  ({df_raw['assignment_id'].nunique()} assignments)")

df_filtered, diag = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
print(f"  After filtering: {df_filtered['assignment_id'].nunique()} assignments remain "
      f"(removed {diag['n_total_assignments'] - diag['n_pass_both']})")

# Eligible tasks (based on filtered data)
all_tasks = sorted(cfg.eligible_tasks(df_filtered, min_assignments=0))
print(f"  Eligible tasks: {len(all_tasks)}")

# Build probe-frequency dicts for both raw and filtered data
freqs_raw      = _build_probe_freqs(df_raw,      all_tasks)
freqs_filtered = _build_probe_freqs(df_filtered, all_tasks)

# Flatten to single arrays + per-arity arrays
def _flatten(freqs_by_task: dict) -> np.ndarray:
    parts = [v for v in freqs_by_task.values() if len(v) > 0]
    return np.concatenate(parts) if parts else np.array([])

def _by_arity(freqs_by_task: dict) -> dict[int, np.ndarray]:
    result = {}
    for task, vals in freqs_by_task.items():
        a = _task_arity(task)
        if a not in result:
            result[a] = []
        result[a].append(vals)
    return {a: np.concatenate(vs) for a, vs in result.items() if vs}

all_raw      = _flatten(freqs_raw)
all_filtered = _flatten(freqs_filtered)
arity_raw      = _by_arity(freqs_raw)
arity_filtered = _by_arity(freqs_filtered)

# ── plot helpers ─────────────────────────────────────────────────────────────
_BINS = np.linspace(0, 1, 26)   # 25 bins of width 0.04


def _hist_ax(ax, freqs: np.ndarray, color: str, title: str) -> None:
    """Draw histogram + mean/median markers on *ax*."""
    if len(freqs) == 0:
        ax.set_visible(False)
        return

    mu  = float(np.mean(freqs))
    med = float(np.median(freqs))

    ax.hist(freqs, bins=_BINS, color=color, alpha=0.75, edgecolor="white",
            linewidth=0.4, zorder=2)

    ax.axvline(0.5, color="black",  lw=1.2, ls=":",  zorder=3,
               label="0.5")
    ax.axvline(mu,  color=C_MEAN,   lw=1.8, ls="-",  zorder=4,
               label=f"mean = {mu:.3f}")
    ax.axvline(med, color=C_MEDIAN, lw=1.8, ls="--", zorder=4,
               label=f"median = {med:.3f}")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Choice frequency (P(yes))", fontsize=10)
    ax.set_ylabel("Cell count", fontsize=10)
    ax.set_title(f"{title}  (n={len(freqs):,})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax, top=True, right=True)


# ── Figure 1 & 2: overall before / after ─────────────────────────────────────

for tag, freqs, label in [
    ("before", all_raw,      "Before filtering"),
    ("after",  all_filtered, "After filtering"),
]:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    _hist_ax(ax, freqs, C_BAR, label)
    plt.tight_layout()
    out = PLOTS_DIR / f"overall_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(_RUN1_DIR)}")

# ── Figure 3 & 4: per-arity before / after ───────────────────────────────────

arities = sorted(set(arity_raw) | set(arity_filtered))   # [1, 2, 3, 4]
n_cols  = 2
n_rows  = (len(arities) + n_cols - 1) // n_cols

for tag, arity_freqs, label in [
    ("before", arity_raw,      "Before filtering"),
    ("after",  arity_filtered, "After filtering"),
]:
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten()

    for idx, a in enumerate(arities):
        ax    = axes_flat[idx]
        freqs = arity_freqs.get(a, np.array([]))
        _hist_ax(ax, freqs, ARITY_COLOR.get(a, "#888888"),
                 f"Arity {a}")

    # Hide any unused panels
    for idx in range(len(arities), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(label, fontsize=12, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / f"arity_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(_RUN1_DIR)}")

# ── Console summary ───────────────────────────────────────────────────────────

print()
print(f"{'':30s}  {'n cells':>8}  {'mean':>7}  {'median':>7}  {'std':>7}")
print("─" * 65)

for label, arity_freqs, all_freqs in [
    ("BEFORE FILTERING", arity_raw,      all_raw),
    ("AFTER FILTERING",  arity_filtered, all_filtered),
]:
    print(f"\n{label}")
    for a in arities:
        v = arity_freqs.get(a, np.array([]))
        if len(v) == 0:
            continue
        print(f"  Arity {a}  {'':<22}  {len(v):>8,}  "
              f"{np.mean(v):>7.4f}  {np.median(v):>7.4f}  {np.std(v):>7.4f}")
    print(f"  {'Overall':<26}  {len(all_freqs):>8,}  "
          f"{np.mean(all_freqs):>7.4f}  {np.median(all_freqs):>7.4f}  "
          f"{np.std(all_freqs):>7.4f}")

# ── Mirror / complement symmetry analysis ────────────────────────────────────
#
# For every task t, its mirror t' flips every dimension condition:
#   right ↔ left,  large ↔ small,  transparent ↔ opaque,  glossy ↔ matte
#
# If the benchmark and perceptual encoding were perfectly symmetric,
# P(yes|t) + P(yes|t') = 1  for every pair, so the pair-average = 0.5.
# Departures from 0.5 indicate asymmetry in the benchmark or perception.

_FLIP = {
    "right": "left",  "left": "right",
    "large": "small",  "small": "large",
    "transparent": "opaque",  "opaque": "transparent",
    "glossy": "matte",  "matte": "glossy",
}


def _find_mirror_pairs(tasks: list[str]) -> list[tuple[str, str]]:
    """
    Return unique (t, t') pairs where t' is the dimension-flip of t
    and both tasks are in *tasks*.  Each pair appears once (t < t' by name).
    """
    cond_to_task = {frozenset(t.split("_and_")): t for t in tasks}
    seen, pairs  = set(), []
    for t in tasks:
        conditions = frozenset(t.split("_and_"))
        flipped    = frozenset(_FLIP[c] for c in conditions)
        t_mirror   = cond_to_task.get(flipped)
        if t_mirror and t_mirror != t:
            key = tuple(sorted([t, t_mirror]))
            if key not in seen:
                pairs.append((t, t_mirror))
                seen.add(key)
    return pairs


def _task_mean_freq(freqs_by_task: dict) -> dict[str, float]:
    """Mean P(yes) across all probe cells for each task."""
    return {t: float(np.mean(v))
            for t, v in freqs_by_task.items() if len(v) > 0}


def _mirror_figure(freqs_by_task: dict, all_tasks: list[str],
                   label: str, filename: str) -> None:
    pairs     = _find_mirror_pairs(all_tasks)
    task_mean = _task_mean_freq(freqs_by_task)

    # Keep only pairs where we have data for both tasks
    valid  = [(t, tm) for t, tm in pairs
              if t in task_mean and tm in task_mean]
    n_pair = len(valid)

    xs        = [task_mean[t]  for t, tm in valid]
    ys        = [task_mean[tm] for t, tm in valid]
    pair_arities = [_task_arity(t) for t, tm in valid]
    pair_avgs = [(x + y) / 2 for x, y in zip(xs, ys)]

    fig, (ax_sc, ax_hi) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # ── Left: scatter P(yes|t) vs P(yes|t') ──────────────────────────────────
    # Anti-diagonal: x + y = 1  ↔  pair average = 0.5
    ax_sc.plot([0, 1], [1, 0], color="#bbbbbb", lw=1.2, ls="--", zorder=1,
               label=r"$P_t + P_{t'} = 1$")
    ax_sc.axhline(0.5, color="#dddddd", lw=0.8, ls=":", zorder=1)
    ax_sc.axvline(0.5, color="#dddddd", lw=0.8, ls=":", zorder=1)

    for a in sorted(set(pair_arities)):
        idx = [i for i, ar in enumerate(pair_arities) if ar == a]
        ax_sc.scatter(
            [xs[i] for i in idx], [ys[i] for i in idx],
            color=ARITY_COLOR.get(a, "#888888"), label=f"Arity {a}",
            s=45, zorder=3, alpha=0.85, edgecolors="none",
        )

    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel(r"$P(\mathrm{yes} \mid t)$",  fontsize=11)
    ax_sc.set_ylabel(r"$P(\mathrm{yes} \mid t')$", fontsize=11)
    ax_sc.set_title(f"Mirror-task pairs  ({label},  {n_pair} pairs)", fontsize=10)
    ax_sc.legend(fontsize=8, frameon=False, loc="upper right")
    sns.despine(ax=ax_sc, top=True, right=True)

    # ── Right: histogram of pair averages ─────────────────────────────────────
    mu_avg  = float(np.mean(pair_avgs))
    med_avg = float(np.median(pair_avgs))

    ax_hi.axvline(0.5, color="#bbbbbb", lw=1.2, ls="--", zorder=1, label="0.5")
    ax_hi.hist(pair_avgs, bins=np.linspace(0, 1, 21),
               color=C_BAR, alpha=0.75, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax_hi.axvline(mu_avg,  color=C_MEAN,   lw=1.8, ls="-",  zorder=3,
                  label=f"mean = {mu_avg:.3f}")
    ax_hi.axvline(med_avg, color=C_MEDIAN, lw=1.8, ls="--", zorder=3,
                  label=f"median = {med_avg:.3f}")

    ax_hi.set_xlim(0, 1)
    ax_hi.set_xlabel(r"$(P_t + P_{t'})\,/\,2$", fontsize=11)
    ax_hi.set_ylabel("Pair count", fontsize=10)
    ax_hi.set_title(f"Pair-average P(yes)  ({label})", fontsize=10)
    ax_hi.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax_hi, top=True, right=True)

    plt.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(_RUN1_DIR)}")

    # Console table
    _hdr = f"  {'Task t':<44}  {'Mirror t prime':<44}  {'P(t)':>6}  {'P(t prime)':>10}  {'avg':>6}"
    print(f"\n  Mirror pairs ({label})  —  {n_pair} pairs")
    print(_hdr)
    print("  " + "─" * 114)
    for (t, tm) in sorted(valid, key=lambda p: (_task_arity(p[0]), p[0])):
        avg = (task_mean[t] + task_mean[tm]) / 2
        print(f"  {t:<44}  {tm:<44}  "
              f"{task_mean[t]:.4f}  {task_mean[tm]:.4f}  {avg:.4f}")

    # Tasks without a mirror in the eligible set
    all_task_set = set(all_tasks)
    cond_to_task = {frozenset(t.split("_and_")): t for t in all_tasks}
    unpaired = []
    for t in sorted(all_tasks):
        flipped  = frozenset(_FLIP[c] for c in t.split("_and_"))
        t_mirror = cond_to_task.get(flipped)
        if t_mirror is None or t_mirror not in task_mean:
            unpaired.append(t)
    if unpaired:
        print(f"\n  Tasks without an eligible mirror ({len(unpaired)}):")
        for t in unpaired:
            flipped  = frozenset(_FLIP[c] for c in t.split("_and_"))
            t_mirror = cond_to_task.get(flipped, "(not in registry)")
            print(f"    {t:<44}  →  mirror: {t_mirror}")


print("\n" + "=" * 80)
print("MIRROR / COMPLEMENT SYMMETRY ANALYSIS")
print("=" * 80)

_mirror_figure(freqs_raw,      all_tasks, "before filtering", "mirror_before.png")
_mirror_figure(freqs_filtered, all_tasks, "after filtering",  "mirror_after.png")
