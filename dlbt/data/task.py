"""
Task definition for DLBT.

A task is characterised by a utility-difference vector delta_u in R^K:
  delta_u[k] > 0  =>  action 1 (right button) is optimal for latent state k
  delta_u[k] < 0  =>  action 0 (left  button) is optimal for latent state k

For deterministic binary tasks, delta_u[k] in {+1, -1} for all k.

The SEU decision rule reduces to:
  choose action 1  iff  b̃ · delta_u > 0
where b̃ ~ Dirichlet(alpha(x)) is the agent's belief over latent states.

This module defines the 10 tasks used in the paper (4 simple + 6 composite)
and provides factory helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dlbt.constants import K, DIM_FRONT_BACK, DIM_SHAPE, DIM_TRANSP, DIM_GLOSS


@dataclass(frozen=True)
class Task:
    """
    A binary perceptual task.

    Attributes:
        name:    human-readable identifier
        delta_u: array of shape [K] with values in {+1, -1}.
                 Sign indicates which action is optimal per latent state.
    """
    name: str
    delta_u: np.ndarray  # [K], dtype float32

    def __post_init__(self):
        if self.delta_u.shape != (K,):
            raise ValueError(f"delta_u must have shape ({K},), got {self.delta_u.shape}")
        if not np.all(np.isin(self.delta_u, [-1.0, 1.0])):
            raise ValueError("delta_u entries must all be +1 or -1")

    def optimal_action(self, latent_state: int) -> int:
        """Return the optimal action (0 or 1) for a given latent state index."""
        return int(self.delta_u[latent_state] > 0)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Task) and self.name == other.name


# ---------------------------------------------------------------------------
# Task factory helpers
# ---------------------------------------------------------------------------

def _make_delta_u(condition) -> np.ndarray:
    """
    Build a delta_u vector from a condition function.

    condition(front_back, shape, transp, gloss) -> bool
      True  => action 1 is optimal
      False => action 0 is optimal
    """
    delta_u = np.empty(K, dtype=np.float32)
    for k in range(K):
        fb   = (k >> DIM_FRONT_BACK) & 1
        sh   = (k >> DIM_SHAPE)      & 1
        tr   = (k >> DIM_TRANSP)     & 1
        gl   = (k >> DIM_GLOSS)      & 1
        delta_u[k] = 1.0 if condition(fb, sh, tr, gl) else -1.0
    return delta_u


def _task(name: str, condition) -> Task:
    return Task(name=name, delta_u=_make_delta_u(condition))


# ---------------------------------------------------------------------------
# The 10 tasks (4 simple + 6 composite)
# Bit convention:
#   front_back: 0=front, 1=back
#   shape:      0=triangular, 1=non-triangular
#   transp:     0=not transparent, 1=transparent
#   gloss:      0=not glossy, 1=glossy
#
# Action-1 (right button) conventions are set here and fixed throughout.
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    # ---- simple tasks -------------------------------------------------------
    "front_back": _task(
        "front_back",
        lambda fb, sh, tr, gl: fb == 1,           # right = back
    ),
    "triangular": _task(
        "triangular",
        lambda fb, sh, tr, gl: sh == 0,           # right = triangular-faced
    ),
    "transparent": _task(
        "transparent",
        lambda fb, sh, tr, gl: tr == 1,           # right = transparent
    ),
    "glossy": _task(
        "glossy",
        lambda fb, sh, tr, gl: gl == 1,           # right = glossy
    ),

    # ---- composite: Location x Material -------------------------------------
    "back_and_glossy": _task(
        "back_and_glossy",
        lambda fb, sh, tr, gl: fb == 1 and gl == 1,
    ),
    "front_and_transparent": _task(
        "front_and_transparent",
        lambda fb, sh, tr, gl: fb == 0 and tr == 1,
    ),

    # ---- composite: Categorisation x Material --------------------------------
    "triangular_and_transparent": _task(
        "triangular_and_transparent",
        lambda fb, sh, tr, gl: sh == 0 and tr == 1,
    ),
    "nontriangular_and_glossy": _task(
        "nontriangular_and_glossy",
        lambda fb, sh, tr, gl: sh == 1 and gl == 1,
    ),

    # ---- composite: Categorisation x Location --------------------------------
    "triangular_and_front": _task(
        "triangular_and_front",
        lambda fb, sh, tr, gl: sh == 0 and fb == 0,
    ),
    "nontriangular_and_front": _task(
        "nontriangular_and_front",
        lambda fb, sh, tr, gl: sh == 1 and fb == 0,
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS)}")
    return TASKS[name]
