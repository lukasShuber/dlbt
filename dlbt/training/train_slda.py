"""
Fitting procedure for SldaAgent.

SLDA training is not gradient-based: for each task, we solve a
least-squares problem (closed-form) and then run a line search on the
softmax temperature using the validation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import TASKS
from dlbt.training.metrics import multinomial_nll, corrected_mse


@dataclass
class SldaResult:
    agent:        SldaAgent
    temperatures: Dict[str, float] = field(default_factory=dict)  # task -> fitted temp
    val_nlls:     Dict[str, float] = field(default_factory=dict)   # task -> val NLL
    val_mses:     Dict[str, float] = field(default_factory=dict)


def fit_slda(
    agent: SldaAgent,
    train_dataset: BehavioralDataset,
    val_dataset:   BehavioralDataset,
    image_refs:    Dict[str, ImageRef],
    temperatures:  Optional[List[float]] = None,
) -> SldaResult:
    """
    Fit SldaAgent on all tasks present in train_dataset, then tune each
    task's softmax temperature on val_dataset.

    Args:
        agent:         the SldaAgent (modified in-place).
        train_dataset: training observations; used to determine optimal-action
                       labels for each task.
        val_dataset:   validation observations; used for temperature line search.
        image_refs:    uid -> ImageRef lookup.
        temperatures:  candidate temperature values for the line search.
                       If None, uses 50 log-spaced values in [0.1, 10].

    Returns:
        SldaResult with fitted temperatures and val metrics.
    """
    result = SldaResult(agent=agent)

    # Pre-cache all CLIP features (single pass over all images)
    agent.precompute_features(list(image_refs.values()))

    train_task_names = train_dataset.df["task_name"].unique().tolist()

    for task_name in train_task_names:
        task = TASKS[task_name]

        # ---- Fit decoder on training images -------------------------------
        train_group = train_dataset.df[train_dataset.df["task_name"] == task_name]
        train_refs  = [image_refs[uid] for uid in train_group["uid"]]
        agent.fit(task, train_refs)

        # ---- Tune temperature on val set ----------------------------------
        val_group = val_dataset.df[val_dataset.df["task_name"] == task_name]
        if len(val_group) == 0:
            result.temperatures[task_name] = 1.0
            continue

        val_refs   = [image_refs[uid] for uid in val_group["uid"]]
        val_counts = torch.tensor(
            val_group[["count_0", "count_1"]].values,
            dtype=torch.float32,
            device=agent.device,
        )
        best_temp = agent.fit_temperature(
            task, val_refs, val_counts, temperatures=temperatures
        )
        result.temperatures[task_name] = best_temp

        # Record val metrics at best temperature
        with torch.no_grad():
            probs = agent.choice_probs(val_refs, task)
        result.val_nlls[task_name] = multinomial_nll(probs, val_counts).item()
        result.val_mses[task_name] = corrected_mse(
            probs, val_counts, n_mc_samples=1  # SLDA is deterministic
        )

    return result
