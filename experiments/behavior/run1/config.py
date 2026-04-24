"""
Configuration for behavior run1.

The full dataset concatenates run0 (22 tasks) and run1 (58 new tasks,
including 4-way conjunctions and negative-dimension tasks) for a combined
80-task corpus.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA     = "stimuli/imgs/metadata.jsonl"
CACHE_PATH   = "stimuli/imgs/clip_rn50_features_v2.pt"
RESOURCES    = Path(__file__).parent / "resources"

# Both CSVs are concatenated at load time (no overlapping assignments)
BEHAVIOR_CSV_RUN0 = RESOURCES / "dlbt-behavior.csv"       # same as run0
BEHAVIOR_CSV_RUN1 = RESOURCES / "dlbt-behavior-run1.csv"  # new data

RESULTS_DIR  = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED               = 42
MIN_CATCH_PERF     = 1.0
MAIN_PERF_QUANTILE = 0.95
USE_TRIAL_KINDS    = ("main", "probe")

# Minimum number of filtered assignments a task must have to be included.
# Tasks below this threshold are dropped from both TRAIN and VAL.
# Set to 0 to include all tasks.
MIN_TASK_ASSIGNMENTS = 0

# ---------------------------------------------------------------------------
# Full BEH_ID_TO_TASK mapping — all 80 tasks across run0 + run1
# Dimension priority in DLBT task names: lr > sl > tr > gl
# ---------------------------------------------------------------------------
BEH_ID_TO_TASK = {
    # ---- 1-way (8) ----
    "glossy":                           "glossy",
    "large":                            "large",
    "left":                             "left",
    "matte":                            "matte",
    "opaque":                           "opaque",
    "right":                            "right",
    "small":                            "small",
    "transparent":                      "transparent",

    # ---- 2-way positive pairs (run0) ----
    "glossy,large":                     "large_and_glossy",
    "glossy,left":                      "left_and_glossy",
    "glossy,right":                     "right_and_glossy",
    "large,left":                       "left_and_large",
    "large,right":                      "right_and_large",
    "transparent,glossy":               "transparent_and_glossy",
    "transparent,large":                "large_and_transparent",
    "transparent,left":                 "left_and_transparent",
    "transparent,right":                "right_and_transparent",

    # ---- 2-way mixed/negative pairs (run1) ----
    "glossy,small":                     "small_and_glossy",
    "matte,large":                      "large_and_matte",
    "matte,left":                       "left_and_matte",
    "matte,right":                      "right_and_matte",
    "matte,small":                      "small_and_matte",
    "opaque,glossy":                    "opaque_and_glossy",
    "opaque,large":                     "large_and_opaque",
    "opaque,left":                      "left_and_opaque",
    "opaque,matte":                     "opaque_and_matte",
    "opaque,right":                     "right_and_opaque",
    "opaque,small":                     "small_and_opaque",
    "small,left":                       "left_and_small",
    "small,right":                      "right_and_small",
    "transparent,matte":                "transparent_and_matte",
    "transparent,small":                "small_and_transparent",

    # ---- 3-way positive (run0) ----
    "glossy,large,right":               "right_and_large_and_glossy",
    "transparent,glossy,large":         "large_and_transparent_and_glossy",
    "transparent,glossy,left":          "left_and_transparent_and_glossy",
    "transparent,glossy,right":         "right_and_transparent_and_glossy",
    "transparent,large,right":          "right_and_large_and_transparent",

    # ---- 3-way mixed/negative (run1) ----
    "glossy,large,left":                "left_and_large_and_glossy",
    "glossy,small,left":                "left_and_small_and_glossy",
    "glossy,small,right":               "right_and_small_and_glossy",
    "matte,large,left":                 "left_and_large_and_matte",
    "matte,large,right":                "right_and_large_and_matte",
    "matte,small,left":                 "left_and_small_and_matte",
    "matte,small,right":                "right_and_small_and_matte",
    "opaque,glossy,large":              "large_and_opaque_and_glossy",
    "opaque,glossy,left":               "left_and_opaque_and_glossy",
    "opaque,glossy,right":              "right_and_opaque_and_glossy",
    "opaque,glossy,small":              "small_and_opaque_and_glossy",
    "opaque,large,left":                "left_and_large_and_opaque",
    "opaque,large,right":               "right_and_large_and_opaque",
    "opaque,matte,large":               "large_and_opaque_and_matte",
    "opaque,matte,left":                "left_and_opaque_and_matte",
    "opaque,matte,right":               "right_and_opaque_and_matte",
    "opaque,matte,small":               "small_and_opaque_and_matte",
    "opaque,small,left":                "left_and_small_and_opaque",
    "opaque,small,right":               "right_and_small_and_opaque",
    "transparent,glossy,small":         "small_and_transparent_and_glossy",
    "transparent,large,left":           "left_and_large_and_transparent",
    "transparent,matte,large":          "large_and_transparent_and_matte",
    "transparent,matte,left":           "left_and_transparent_and_matte",
    "transparent,matte,right":          "right_and_transparent_and_matte",
    "transparent,matte,small":          "small_and_transparent_and_matte",
    "transparent,small,left":           "left_and_small_and_transparent",
    "transparent,small,right":          "right_and_small_and_transparent",

    # ---- 4-way (run1) ----
    "opaque,glossy,large,left":         "left_and_large_and_opaque_and_glossy",
    "opaque,glossy,large,right":        "right_and_large_and_opaque_and_glossy",
    "opaque,glossy,small,left":         "left_and_small_and_opaque_and_glossy",
    "opaque,glossy,small,right":        "right_and_small_and_opaque_and_glossy",
    "opaque,matte,large,left":          "left_and_large_and_opaque_and_matte",
    "opaque,matte,large,right":         "right_and_large_and_opaque_and_matte",
    "opaque,matte,small,left":          "left_and_small_and_opaque_and_matte",
    "opaque,matte,small,right":         "right_and_small_and_opaque_and_matte",
    "transparent,glossy,large,left":    "left_and_large_and_transparent_and_glossy",
    "transparent,glossy,large,right":   "right_and_large_and_transparent_and_glossy",
    "transparent,glossy,small,left":    "left_and_small_and_transparent_and_glossy",
    "transparent,glossy,small,right":   "right_and_small_and_transparent_and_glossy",
    "transparent,matte,large,left":     "left_and_large_and_transparent_and_matte",
    "transparent,matte,large,right":    "right_and_large_and_transparent_and_matte",
    "transparent,matte,small,left":     "left_and_small_and_transparent_and_matte",
    "transparent,matte,small,right":    "right_and_small_and_transparent_and_matte",
}

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_PERF = "#2a6fb5"
C_RT   = "#e07a1f"


# ---------------------------------------------------------------------------
# Task filtering helper
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered, min_assignments: int = MIN_TASK_ASSIGNMENTS):
    """
    Return the set of DLBT task names that have at least `min_assignments`
    filtered assignments.  Pass the already-filtered DataFrame (output of
    filter_assignments).  Set min_assignments=0 to keep all tasks.
    """
    import pandas as _pd
    df = df_filtered[df_filtered["trial_kind"].isin(USE_TRIAL_KINDS)].copy()
    df = df[df["task_id"].isin(BEH_ID_TO_TASK)]
    df["task_name"] = df["task_id"].map(BEH_ID_TO_TASK)
    counts = df.groupby("task_name")["assignment_id"].nunique()
    return set(counts[counts >= min_assignments].index)
