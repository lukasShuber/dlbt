"""
performance.py  —  behavior run1
---------------------------------
Per-task learning curves for the combined run0 + run1 behavioural dataset
(80 tasks).  Produces three figures with increasing levels of filtering:

  1. unfiltered   — all assignments, only scientific trials (main + probe)
  2. catch_only   — catch-trial attention check only (perf == 1.0)
  3. full         — catch + main-trial performance filter (95th pctile)

Usage:
    cd <repo root>
    python experiments/behavior/run1/performance.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent / "run0"))
from preprocess import filter_assignments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def task_arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _display(task_name: str) -> str:
    return task_name.replace("_and_", " & ").replace("_", "/")


ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

N_COLS   = 10
ROLL     = 8
PANEL_W  = 1.30
PANEL_H  = 1.20


def _make_figure(df: pd.DataFrame, per_task: pd.DataFrame,
                 task_order: list, title: str, out_path: Path) -> None:
    """Draw the 80-panel learning-curve figure and save it."""
    n_tasks = len(task_order)
    n_rows  = int(np.ceil(n_tasks / N_COLS))

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * PANEL_W, n_rows * PANEL_H),
        gridspec_kw={"hspace": 0.70, "wspace": 0.20},
    )
    axes_flat = np.atleast_2d(axes).flatten()

    for idx, task_name in enumerate(task_order):
        ax    = axes_flat[idx]
        sub   = df[df["task_name"] == task_name]
        arity = int(sub["arity"].iloc[0])
        color = ARITY_COLOR.get(arity, "#555")
        n_a   = sub["assignment_id"].nunique()

        perf = sub.groupby("trial")["perf"].agg(["mean", "sem"]).reset_index()
        perf["mean"] = perf["mean"].rolling(ROLL, center=True, min_periods=1).mean()
        perf["sem"]  = (perf["sem"].rolling(ROLL, center=True, min_periods=1).mean()
                        / np.sqrt(ROLL))

        x = perf["trial"].values + 1
        m = perf["mean"].values
        s = perf["sem"].values * 1.96

        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, m, color=color, lw=1.8, alpha=0.95)
        ax.axhline(0.5, ls=":", color="gray", lw=0.7, alpha=0.7)

        ax.set_ylim(0.25, 1.05)
        ax.set_xlim(x[0], x[-1])
        ax.set_yticks([0.5, 1.0])
        ax.set_xticks([int(x[0]), int(x[-1])])
        ax.tick_params(labelsize=4.5, pad=1, length=2)

        overall = float(per_task.loc[task_name, "accuracy"])
        ax.set_title(f"{_display(task_name)}\n{overall:.2f}  n={n_a}",
                     fontsize=4.8, pad=2, color=color, fontweight="bold")
        sns.despine(ax=ax, trim=True)

    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=c, lw=2, label=f"{a}-way")
               for a, c in ARITY_COLOR.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=7,
               frameon=False, ncol=4, bbox_to_anchor=(0.98, 0.005))
    fig.suptitle(title, fontsize=9, y=1.002)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _make_trial_count_figure(per_task: pd.DataFrame, task_order: list,
                              title: str, out_path: Path) -> None:
    """Horizontal bar chart: main-image trial count per task (probe excluded)."""
    n_tasks = len(task_order)
    fig, ax = plt.subplots(figsize=(6.5, n_tasks * 0.22 + 0.8))

    # Draw bottom-to-top so arity-1 tasks are at the top of the chart
    for i, task_name in enumerate(reversed(task_order)):
        arity = int(per_task.loc[task_name, "arity"])
        color = ARITY_COLOR.get(arity, "#555")
        n     = int(per_task.loc[task_name, "n_main_trials"])
        ax.barh(i, n, color=color, alpha=0.80, height=0.72)
        ax.text(n + max(per_task["n_main_trials"].max() * 0.01, 5),
                i, str(n), va="center", fontsize=4.5, color=color)

    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([_display(t) for t in reversed(task_order)], fontsize=4.8)
    ax.set_xlabel("Main-image trials (probe excluded)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, per_task["n_main_trials"].max() * 1.12)

    handles = [Line2D([0], [0], color=c, lw=4, label=f"{a}-way")
               for a, c in ARITY_COLOR.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              loc="lower right", ncol=2)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _prepare_df(df_raw: pd.DataFrame,
                keep_assignments=None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Restrict to scientific trials + mapped tasks, re-rank trial index,
    compute per-task accuracy.  If keep_assignments is None, use all.
    """
    df = df_raw.copy()
    if keep_assignments is not None:
        df = df[df["assignment_id"].isin(keep_assignments)]

    df = df[df["trial_kind"].isin(cfg.USE_TRIAL_KINDS)]
    df = df[df["task_id"].isin(cfg.BEH_ID_TO_TASK)]
    df["task_name"] = df["task_id"].map(cfg.BEH_ID_TO_TASK)
    df["arity"]     = df["task_name"].map(task_arity)

    df = df.sort_values(["assignment_id", "trial"]).reset_index(drop=True)
    df["trial"] = df.groupby("assignment_id").cumcount()

    per_task = (
        df.groupby("task_name")
          .agg(accuracy = ("perf", "mean"),
               n_trials = ("perf", "size"),
               n_assign = ("assignment_id", "nunique"),
               arity    = ("arity", "first"))
          .sort_values(["arity", "accuracy"])
    )

    # Main-trial count per task (probe image trials excluded)
    n_main = (
        df[df["trial_kind"] == "main"]
          .groupby("task_name")
          .size()
          .rename("n_main_trials")
    )
    per_task = per_task.join(n_main, how="left")
    per_task["n_main_trials"] = per_task["n_main_trials"].fillna(0).astype(int)

    task_order = list(per_task.index)
    return df, per_task, task_order


# ---------------------------------------------------------------------------
# Load + concatenate raw data
# ---------------------------------------------------------------------------
print(f"Reading {cfg.BEHAVIOR_CSV_RUN0.name} + {cfg.BEHAVIOR_CSV_RUN1.name}")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
print(f"  Combined raw trials: {len(df_raw):,}  "
      f"({df_raw['assignment_id'].nunique()} assignments)")

all_assignments = set(df_raw["assignment_id"].unique())

# ---------------------------------------------------------------------------
# Filter sets
# ---------------------------------------------------------------------------
# Catch only
_, diag_catch = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = 0.0,   # no main filter
    seed               = cfg.SEED,
)
catch_perf = (
    df_raw[df_raw["trial_kind"] == "catch"]
    .groupby("assignment_id")["perf"].mean()
)
catch_only_assignments = set(
    catch_perf[catch_perf >= cfg.MIN_CATCH_PERF].index
)

# Full filter
_, diag_full = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
full_assignments = set(
    filter_assignments(
        df_raw,
        min_catch_perf     = cfg.MIN_CATCH_PERF,
        main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
        seed               = cfg.SEED,
    )[0]["assignment_id"].unique()
    if False else []
)
# Recompute cleanly
df_filt_full, diag_full = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
full_assignments = set(df_filt_full["assignment_id"].unique())

n_all   = len(all_assignments)
n_catch = len(catch_only_assignments)
n_full  = len(full_assignments)
print(f"\n  Unfiltered:   {n_all} assignments")
print(f"  Catch only:   {n_catch} assignments  ({n_all - n_catch} dropped)")
print(f"  Full filter:  {n_full} assignments  ({n_all - n_full} dropped)")

# ---------------------------------------------------------------------------
# Build and plot each condition
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

conditions = [
    ("unfiltered",
     None,
     f"Per-task learning curves — UNFILTERED  "
     f"({n_all} assignments)",
     cfg.RESULTS_DIR / "performance_unfiltered.png"),

    ("catch_only",
     catch_only_assignments,
     f"Per-task learning curves — CATCH FILTER ONLY  "
     f"({n_catch}/{n_all} assignments passed)",
     cfg.RESULTS_DIR / "performance_catch_only.png"),

    ("full",
     full_assignments,
     f"Per-task learning curves — FULLY FILTERED  "
     f"({n_full}/{n_all} assignments passed)",
     cfg.RESULTS_DIR / "performance_full_filter.png"),
]

for label, keep, title, out_path in conditions:
    print(f"\n--- {label} ---")
    df_c, per_task_c, task_order_c = _prepare_df(df_raw, keep_assignments=keep)
    print(f"  {len(df_c):,} trials  |  {df_c['task_name'].nunique()} tasks  "
          f"|  {df_c['assignment_id'].nunique()} assignments")
    _make_figure(df_c, per_task_c, task_order_c, title, out_path)

    count_path = out_path.with_name(out_path.stem + "_trial_counts.png")
    _make_trial_count_figure(
        per_task_c, task_order_c,
        title=f"Main trials per task  [{label}]",
        out_path=count_path,
    )

print("\nDone.")
