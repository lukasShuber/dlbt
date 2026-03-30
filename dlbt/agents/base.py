"""
Agent base class.

All models in this codebase share the same interface:
  choice_probs(image_refs, task) -> Tensor[B, 2]

where index 0 = left button, index 1 = right button.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch

from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class Agent(ABC):
    """
    Abstract base for all decision agents.

    Input:  image_refs (List[ImageRef] of length B), task (Task)
    Output: Tensor of shape [B, 2] — choice probabilities summing to 1.
             Column 0 = P(left), column 1 = P(right).
    """

    @abstractmethod
    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Return choice probability tensor of shape [B, 2]."""
        ...
