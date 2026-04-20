"""
Preprocess the raw behavioural CSV into aggregated (uid, task_name) counts.

Pipeline (mirrors experiments/behavior/run0/notebooks/view-behavior.ipynb):

  1. Filter assignments by catch-trial performance (4/4 correct).
  2. Filter assignments by main-trial performance (>= 95th pctile of
     Binomial(100, 0.5)).
  3. Restrict to `main` + `probe` trials (drop warmup, catch).
  4. Map `stimulus` filenames to UIDs (first 6 chars).
  5. Map `task_id` strings to DLBT task names via BEH_ID_TO_TASK.
  6. Convert each trial's `choice` ("yes"/"no") to count_1/count_0.
  7. Aggregate across all filtered trials per (uid, task_name) cell.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dlbt.data.dataset import BehavioralDataset, Observation


def _choice_to_action(choice: str, task_name: str, dlbt_tasks) -> int:
    """
    Convert the human yes/no answer to DLBT's action index (0 = left, 1 = right).

    In DLBT, action 1 (`count_1`) = 'press right' = task-condition satisfied,
    action 0 (`count_0`) = 'press left' = task-condition not satisfied.

    A "yes" answer to a task prompt therefore corresponds to action 1.
    """
    return 1 if choice == "yes" else 0


def filter_assignments(
    df: pd.DataFrame,
    min_catch_perf: float = 1.0,
    main_perf_quantile: float = 0.95,
    seed: int = 0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply the two quality-filtering criteria from the notebook.

    Returns:
        filtered DataFrame, diagnostics dict.
    """
    # Catch-trial criterion ------------------------------------------------
    catch_perf = (
        df.loc[df["trial_kind"] == "catch"]
          .groupby("assignment_id")["perf"].mean()
    )
    pass_catch = set(catch_perf[catch_perf >= min_catch_perf].index)

    # Main-trial criterion (threshold drawn from Binom(100, 0.5)) ----------
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n=100, p=0.5, size=100_000)
    threshold = np.percentile(samples, main_perf_quantile * 100) / 100.0

    main_perf = (
        df.loc[df["trial_kind"] == "main"]
          .groupby("assignment_id")["perf"].mean()
    )
    pass_main = set(main_perf[main_perf >= threshold].index)

    keep = pass_catch & pass_main
    filtered = df[df["assignment_id"].isin(keep)].copy()

    diag = dict(
        n_total_assignments  = df["assignment_id"].nunique(),
        n_pass_catch         = len(pass_catch),
        n_pass_main          = len(pass_main),
        n_pass_both          = len(keep),
        main_perf_threshold  = float(threshold),
    )
    return filtered, diag


def aggregate_counts(
    df: pd.DataFrame,
    beh_id_to_task: Dict[str, str],
    use_trial_kinds: Tuple[str, ...] = ("main", "probe"),
) -> Tuple[BehavioralDataset, set, set]:
    """
    Aggregate filtered trials into (uid, task_name) count cells.

    Rows with unmapped task_ids are dropped; we only keep the 22 tasks in
    the BEH_ID_TO_TASK mapping.

    Returns:
        (dataset, probe_uids, main_uids)

    `probe_uids` = UIDs that appeared on `probe` trials (high-rep eval set).
    `main_uids`  = UIDs that appeared on `main` trials (noisy bulk training set).
    The two sets are disjoint by experimental design.
    """
    df = df[df["trial_kind"].isin(use_trial_kinds)].copy()
    df = df[df["task_id"].isin(beh_id_to_task)]
    df["task_name"] = df["task_id"].map(beh_id_to_task)

    # UID = first 6 chars of the stimulus filename
    df["uid"]    = df["stimulus"].str.slice(0, 6)
    df["action"] = (df["choice"] == "yes").astype(int)

    # Identify which UIDs come from probe vs main trials BEFORE aggregating
    probe_uids = set(df.loc[df["trial_kind"] == "probe", "uid"].unique())
    main_uids  = set(df.loc[df["trial_kind"] == "main",  "uid"].unique())

    # One row per (uid, task_name) cell with both counts
    grp = (
        df.groupby(["uid", "task_name"])["action"]
          .agg(count_1="sum", n="count")
          .reset_index()
    )
    grp["count_0"] = grp["n"] - grp["count_1"]
    grp = grp.drop(columns="n")

    records = [
        Observation(uid=row.uid, task_name=row.task_name,
                    count_0=int(row.count_0), count_1=int(row.count_1))
        for row in grp.itertuples(index=False)
    ]
    return BehavioralDataset.from_records(records), probe_uids, main_uids


def load_and_preprocess(
    csv_path,
    beh_id_to_task: Dict[str, str],
    min_catch_perf: float = 1.0,
    main_perf_quantile: float = 0.95,
    use_trial_kinds: Tuple[str, ...] = ("main", "probe"),
    seed: int = 0,
) -> Tuple[BehavioralDataset, set, set, dict]:
    """
    Top-level helper: load CSV, filter, aggregate.
    Returns (dataset, probe_uids, main_uids, diagnostics).
    """
    df = pd.read_csv(csv_path)
    filtered, diag = filter_assignments(
        df, min_catch_perf=min_catch_perf,
        main_perf_quantile=main_perf_quantile, seed=seed,
    )
    ds, probe_uids, main_uids = aggregate_counts(
        filtered, beh_id_to_task, use_trial_kinds=use_trial_kinds
    )

    diag["n_raw_trials"]          = len(df)
    diag["n_filtered_trials"]     = len(filtered)
    diag["n_cells"]               = len(ds)
    diag["n_unique_images"]       = ds.df["uid"].nunique()
    diag["n_unique_tasks"]        = ds.df["task_name"].nunique()
    diag["n_probe_uids"]          = len(probe_uids)
    diag["n_main_uids"]           = len(main_uids)
    diag["trials_per_cell_mean"]  = float(
        (ds.df["count_0"] + ds.df["count_1"]).mean()
    )
    # Split mean-trials/cell by probe vs main for reporting
    in_probe = ds.df["uid"].isin(probe_uids)
    if in_probe.any():
        sub = ds.df[in_probe]
        diag["trials_per_cell_probe"] = float((sub["count_0"] + sub["count_1"]).mean())
    if (~in_probe).any():
        sub = ds.df[~in_probe]
        diag["trials_per_cell_main"]  = float((sub["count_0"] + sub["count_1"]).mean())
    return ds, probe_uids, main_uids, diag
