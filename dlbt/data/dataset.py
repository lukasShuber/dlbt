"""
BehavioralDataset: a thin wrapper around a pandas DataFrame holding
aggregated binary-choice counts per (image, task) observation.

Schema (one row per observation):
    uid        str   -- image UID (matches ImageRef.uid)
    task_name  str   -- task name (matches Task.name)
    count_0    int   -- number of trials where action 0 (left) was chosen
    count_1    int   -- number of trials where action 1 (right) was chosen

Usage:
    ds = BehavioralDataset.from_records([...])
    ds = BehavioralDataset.from_csv("path/to/data.csv")

    # iterate grouped by task (convenient for training)
    for task_name, group in ds.iter_tasks():
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch

from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


_REQUIRED_COLS = {"uid", "task_name", "count_0", "count_1"}


@dataclass
class Observation:
    """Single aggregated observation: one (image, task) pair with choice counts."""
    uid: str
    task_name: str
    count_0: int   # left-button choices
    count_1: int   # right-button choices

    @property
    def total(self) -> int:
        return self.count_0 + self.count_1

    @property
    def freq_1(self) -> float:
        """Empirical frequency of action 1."""
        return self.count_1 / self.total if self.total > 0 else 0.5


class BehavioralDataset:
    """
    Aggregated binary-choice dataset.

    Internally stored as a pandas DataFrame for easy filtering / grouping.
    """

    def __init__(self, df: pd.DataFrame):
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        self.df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_records(cls, records: List[Observation]) -> "BehavioralDataset":
        rows = [
            {"uid": r.uid, "task_name": r.task_name,
             "count_0": r.count_0, "count_1": r.count_1}
            for r in records
        ]
        return cls(pd.DataFrame(rows))

    @classmethod
    def from_csv(cls, path: str) -> "BehavioralDataset":
        df = pd.read_csv(path, dtype={"uid": str})
        return cls(df)

    def to_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_tasks(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Yield (task_name, sub-dataframe) groups, one per task."""
        for task_name, group in self.df.groupby("task_name"):
            yield task_name, group

    def get_task_data(
        self,
        task_name: str,
        image_refs: Dict[str, ImageRef],
        task: Task,
    ) -> Tuple[List[ImageRef], torch.Tensor]:
        """
        Return (image_refs_list, counts_tensor) for one task.

        counts_tensor: float32 Tensor of shape [N, 2].
        """
        sub = self.df[self.df["task_name"] == task_name]
        refs = [image_refs[uid] for uid in sub["uid"]]
        counts = torch.tensor(
            sub[["count_0", "count_1"]].values, dtype=torch.float32
        )
        return refs, counts

    # ------------------------------------------------------------------
    # Noise floor
    # ------------------------------------------------------------------

    def noise_floor(self) -> float:
        """
        Mean binomial-sampling variance across all observations.
        Useful as a lower bound on achievable MSE.
        """
        totals = (self.df["count_0"] + self.df["count_1"]).values.astype(float)
        freq1  = (self.df["count_1"] / totals.clip(min=1)).values
        mask   = totals > 1
        var    = freq1[mask] * (1 - freq1[mask]) / (totals[mask] - 1)
        return float(var.mean()) if mask.any() else 0.0

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def filter_tasks(self, task_names: List[str]) -> "BehavioralDataset":
        return BehavioralDataset(
            self.df[self.df["task_name"].isin(task_names)].copy()
        )

    def filter_uids(self, uids: List[str]) -> "BehavioralDataset":
        return BehavioralDataset(
            self.df[self.df["uid"].isin(uids)].copy()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        n_tasks = self.df["task_name"].nunique()
        n_images = self.df["uid"].nunique()
        return (f"BehavioralDataset({len(self)} obs, "
                f"{n_tasks} tasks, {n_images} images)")
