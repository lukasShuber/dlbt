"""DummyAgent: always predicts 50/50."""

from __future__ import annotations

from typing import List

import torch

from dlbt.agents.base import Agent
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class DummyAgent(Agent):
    """Predicts uniform 50/50 choice probabilities regardless of image or task."""

    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        B = len(image_refs)
        return torch.full((B, 2), 0.5)
