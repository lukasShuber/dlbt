"""
Task definition for DLBT.

A task is characterised by a utility-difference vector delta_u in R^K:
  delta_u[k] > 0  =>  action 1 (right button) is optimal for latent state k
  delta_u[k] < 0  =>  action 0 (left  button) is optimal for latent state k

Four binary latent dimensions (K=16):
  lr: left(0)  / right(1)      — x-position
  tr: opaque(0)/ transparent(1)
  gl: matte(0) / glossy(1)
  sl: small(0) / large(1)      — object scale

Front/back (depth) is excluded: it confounds with apparent size under perspective.

Tasks:
  - 4 simple  + 4 simple-flipped
  - 7 2-way AND composites (all pairs except lr×sl, which is held for val)
  - 3 3-way AND composites (training)
  - 4 val tasks (all involve the held-out lr × sl conjunction)
  Total: 22 tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dlbt.constants import K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE


@dataclass(frozen=True)
class Task:
    """
    A binary perceptual task.

    Attributes:
        name:    human-readable identifier
        delta_u: array of shape [K] with values in {+1, -1}.
    """
    name: str
    delta_u: np.ndarray  # [K], dtype float32

    def __post_init__(self):
        if self.delta_u.shape != (K,):
            raise ValueError(f"delta_u must have shape ({K},), got {self.delta_u.shape}")
        if not np.all(np.isin(self.delta_u, [-1.0, 1.0])):
            raise ValueError("delta_u entries must all be +1 or -1")

    def optimal_action(self, latent_state: int) -> int:
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

    condition(lr, tr, gl, sl) -> bool
      True  => action 1 (right button) is optimal
      False => action 0 (left  button) is optimal
    """
    delta_u = np.empty(K, dtype=np.float32)
    for k in range(K):
        lr = (k >> DIM_LEFT_RIGHT)  & 1
        tr = (k >> DIM_TRANSP)      & 1
        gl = (k >> DIM_GLOSS)       & 1
        sl = (k >> DIM_SMALL_LARGE) & 1
        delta_u[k] = 1.0 if condition(lr, tr, gl, sl) else -1.0
    return delta_u


def _task(name: str, condition) -> Task:
    return Task(name=name, delta_u=_make_delta_u(condition))


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {

    # ---- simple (right = property present) ----------------------------------
    "left_right": _task(
        "left_right",
        lambda lr, tr, gl, sl: lr == 1,        # right = rightward
    ),
    "transparent": _task(
        "transparent",
        lambda lr, tr, gl, sl: tr == 1,
    ),
    "glossy": _task(
        "glossy",
        lambda lr, tr, gl, sl: gl == 1,
    ),
    "large": _task(
        "large",
        lambda lr, tr, gl, sl: sl == 1,
    ),

    # ---- simple-flipped (right = property absent) ---------------------------
    "left": _task(
        "left",
        lambda lr, tr, gl, sl: lr == 0,        # right = leftward
    ),
    "opaque": _task(
        "opaque",
        lambda lr, tr, gl, sl: tr == 0,
    ),
    "matte": _task(
        "matte",
        lambda lr, tr, gl, sl: gl == 0,
    ),
    "small": _task(
        "small",
        lambda lr, tr, gl, sl: sl == 0,
    ),

    # ---- 2-way AND (training — no lr×sl) ------------------------------------
    "right_and_transparent": _task(
        "right_and_transparent",
        lambda lr, tr, gl, sl: lr == 1 and tr == 1,
    ),
    "left_and_transparent": _task(
        "left_and_transparent",
        lambda lr, tr, gl, sl: lr == 0 and tr == 1,
    ),
    "right_and_glossy": _task(
        "right_and_glossy",
        lambda lr, tr, gl, sl: lr == 1 and gl == 1,
    ),
    "left_and_glossy": _task(
        "left_and_glossy",
        lambda lr, tr, gl, sl: lr == 0 and gl == 1,
    ),
    "transparent_and_glossy": _task(
        "transparent_and_glossy",
        lambda lr, tr, gl, sl: tr == 1 and gl == 1,
    ),
    "large_and_transparent": _task(
        "large_and_transparent",
        lambda lr, tr, gl, sl: sl == 1 and tr == 1,
    ),
    "large_and_glossy": _task(
        "large_and_glossy",
        lambda lr, tr, gl, sl: sl == 1 and gl == 1,
    ),

    # ---- 3-way AND (training — no lr×sl) ------------------------------------
    "right_and_transparent_and_glossy": _task(
        "right_and_transparent_and_glossy",
        lambda lr, tr, gl, sl: lr == 1 and tr == 1 and gl == 1,
    ),
    "left_and_transparent_and_glossy": _task(
        "left_and_transparent_and_glossy",
        lambda lr, tr, gl, sl: lr == 0 and tr == 1 and gl == 1,
    ),
    "large_and_transparent_and_glossy": _task(
        "large_and_transparent_and_glossy",
        lambda lr, tr, gl, sl: sl == 1 and tr == 1 and gl == 1,
    ),

    # ---- val tasks: all involve the held-out lr × sl conjunction ------------
    "right_and_large": _task(
        "right_and_large",
        lambda lr, tr, gl, sl: lr == 1 and sl == 1,
    ),
    "left_and_large": _task(
        "left_and_large",
        lambda lr, tr, gl, sl: lr == 0 and sl == 1,
    ),
    "right_and_large_and_glossy": _task(
        "right_and_large_and_glossy",
        lambda lr, tr, gl, sl: lr == 1 and sl == 1 and gl == 1,
    ),
    "right_and_large_and_transparent": _task(
        "right_and_large_and_transparent",
        lambda lr, tr, gl, sl: lr == 1 and sl == 1 and tr == 1,
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS)}")
    return TASKS[name]
