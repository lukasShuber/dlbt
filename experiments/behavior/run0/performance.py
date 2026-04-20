"""
performance.py
--------------
Per-task learning curves from the filtered behavioural data.

Mirrors the learning-curve analysis in
experiments/behavior/run0/notebooks/view-behavior.ipynb, but:
  - applies the same two filtering criteria as the main pipeline
    (catch == 1.0, main >= 95th pctile of Binom(100, 0.5))
  - restricts to scientific trials (main + probe) — same as training
  - plots one learning curve (perf vs. trial index) per task in a grid,
    with mean ± SEM across assignments + a task-level overall accuracy
    printed in each panel

This lets us see, for each of the 22 tasks:
  - does performance climb over trials (learning)?
  - what asymptote does each task reach?
  - which tasks are near ceiling vs. near chance?
    (directly addresses the "signal asymmetry" hypothesis raised in the
     α-peakedness discussion)

Usage:
    cd <repo root>
    python experiments/behavior/run0/performance.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from preprocess import filter_assignments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def task_arity(task_name: str) -> int:
    """1-way / 2-way / 3-way based on the number of '_and_' tokens."""
    return task_name.count("_and_") + 1


# ---------------------------------------------------------------------------
# Load + filter
# ---------------------------------------------------------------------------
print(f"Reading {cfg.BEHAVIOR_CSV}")
df_raw = pd.read_csv(cfg.BEHAVIOR_CSV)

df_filtered, diag = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
print(f"Filtering diagnostics:")
for k, v in diag.items():
    print(f"  {k:<28s} {v}")

# Use scientific trials only (drop warmup, catch) — same as training pipeline
df = df_filtered[df_filtered["trial_kind"].isin(cfg.USE_TRIAL_KINDS)].copy()

# Map the behavioural task_id -> DLBT task name (drops unmapped rows)
df = df[df["task_id"].isin(cfg.BEH_ID_TO_TASK)]
df["task_name"] = df["task_id"].map(cfg.BEH_ID_TO_TASK)
df["arity"]     = df["task_name"].map(task_arity)

print(f"\n{len(df)} scientific trials across {df['task_name'].nunique()} tasks, "
      f"{df['assignment_id'].nunique()} assignments.")

# ---------------------------------------------------------------------------
# Per-task overall accuracy (headline numbers)
# ---------------------------------------------------------------------------
per_task = (
    df.groupby("task_name")
      .agg(accuracy  = ("perf", "mean"),
           n_trials  = ("perf", "size"),
           n_assign  = ("assignment_id", "nunique"),
           arity     = ("arity", "first"))
      .sort_values(["arity", "accuracy"])
)
print("\n========== per-task accuracy ==========")
print(per_task.to_string(float_format="%.3f"))

# ---------------------------------------------------------------------------
# Order tasks for plotting: by arity then by accuracy (asc within arity)
# ---------------------------------------------------------------------------
task_order = list(per_task.index)
n_tasks    = len(task_order)
n_cols     = 5
n_rows     = int(np.ceil(n_tasks / n_cols))

# ---------------------------------------------------------------------------
# Plot: one learning-curve panel per task (perf on left y, RT on right y)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.3 * n_rows),
                         sharex=True, sharey=False)
axes = np.atleast_2d(axes)

COLOR_PERF = "#2a6fb5"
COLOR_RT   = "#e07a1f"

for idx, task_name in enumerate(task_order):
    r, c  = divmod(idx, n_cols)
    ax    = axes[r, c]
    sub   = df[df["task_name"] == task_name]
    arity = int(sub["arity"].iloc[0])
    n_a   = sub["assignment_id"].nunique()

    # Aggregate across assignments at each trial index
    perf = sub.groupby("trial")["perf"].agg(["mean", "sem"]).reset_index()
    rt   = sub.groupby("trial")["reaction_time_msec"].agg(["mean", "sem"]).reset_index()

    # Performance — line + 95% CI shaded band
    x_p = perf["trial"].values + 1
    m_p = perf["mean"].values
    s_p = perf["sem"].values * 1.96     # ~95% CI
    ax.plot(x_p, m_p, color=COLOR_PERF, lw=1.2, alpha=0.95)
    ax.fill_between(x_p, m_p - s_p, m_p + s_p,
                    color=COLOR_PERF, alpha=0.20, linewidth=0)
    ax.axhline(0.5, ls=":", color="gray", lw=0.6)
    ax.set_ylim(0.2, 1.02)
    ax.tick_params(axis="y", labelcolor=COLOR_PERF, labelsize=8)

    # RT on twin axis — line + 95% CI shaded band
    ax2 = ax.twinx()
    x_r = rt["trial"].values + 1
    m_r = rt["mean"].values
    s_r = rt["sem"].values * 1.96
    ax2.plot(x_r, m_r, color=COLOR_RT, lw=1.0, alpha=0.70)
    ax2.fill_between(x_r, m_r - s_r, m_r + s_r,
                     color=COLOR_RT, alpha=0.15, linewidth=0)
    ax2.tick_params(axis="y", labelcolor=COLOR_RT, labelsize=7)
    ax2.set_ylim(300, 1800)

    # Overall-accuracy annotation
    overall = float(per_task.loc[task_name, "accuracy"])
    ax.set_title(f"{task_name}  ({arity}w)\n"
                 f"acc={overall:.3f}  n_ass={n_a}",
                 fontsize=9)

    if c == 0:
        ax.set_ylabel("Performance", color=COLOR_PERF, fontsize=9)
    if c == n_cols - 1:
        ax2.set_ylabel("RT (ms)", color=COLOR_RT, fontsize=9)
    if r == n_rows - 1:
        ax.set_xlabel("Trial (within sequence)", fontsize=9)

# Hide empty subplots
for idx in range(n_tasks, n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r, c].set_visible(False)

fig.suptitle(
    f"Per-task learning curves — {cfg.BEHAVIOR_CSV.name}"
    f"  (filtered: {diag['n_pass_both']}/{diag['n_total_assignments']} assignments)",
    y=1.00, fontsize=11,
)
fig.tight_layout()

out_path = cfg.RESULTS_DIR / "performance_per_task.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")

# ---------------------------------------------------------------------------
# Summary: accuracy grouped by arity (quick visual of signal-strength axis)
# ---------------------------------------------------------------------------
print("\n========== accuracy summary by arity ==========")
by_arity = (
    per_task.reset_index()
            .groupby("arity")
            .agg(mean_acc = ("accuracy", "mean"),
                 min_acc  = ("accuracy", "min"),
                 max_acc  = ("accuracy", "max"),
                 n_tasks  = ("task_name", "size"))
)
print(by_arity.to_string(float_format="%.3f"))

# ---------------------------------------------------------------------------
# Per-DIMENSION accuracy (aggregated across all tasks that involve that
# dimension on the positive side).  This is the sharpest test of the
# "signal asymmetry" hypothesis:
#   - right / left:   lr axis
#   - transparent / opaque: tr axis
#   - glossy / matte: gl axis
#   - large / small:  sl axis
# ---------------------------------------------------------------------------
DIM_TOKENS = {
    "lr": ("right",  "left"),
    "tr": ("transparent", "opaque"),
    "gl": ("glossy", "matte"),
    "sl": ("large",  "small"),
}

print("\n========== per-dimension accuracy (1-way tasks only) ==========")
for dim, toks in DIM_TOKENS.items():
    rows = []
    for tok in toks:
        if tok in per_task.index:
            rows.append((tok, per_task.loc[tok, "accuracy"], per_task.loc[tok, "n_trials"]))
    if rows:
        accs  = np.array([r[1] for r in rows])
        ns    = np.array([r[2] for r in rows])
        w_acc = float((accs * ns).sum() / ns.sum())
        print(f"  {dim}  ({'/'.join(toks)}):  "
              f"{'  '.join(f'{t}={a:.3f}' for t,a,_ in rows)}   "
              f"→ weighted mean = {w_acc:.3f}")
