"""
Task definition for DLBT.

A task is characterised by a utility-difference vector delta_u in R^K:
  delta_u[k] > 0  =>  action 1 (right button) is optimal for latent state k
  delta_u[k] < 0  =>  action 0 (left  button) is optimal for latent state k

For deterministic binary tasks, delta_u[k] in {+1, -1} for all k.

The SEU decision rule reduces to:
  choose action 1  iff  b̃ · delta_u > 0
where b̃ ~ Dirichlet(alpha(x)) is the agent's belief over latent states.

Five binary latent dimensions (K=32):
  fb: front(0) / back(1)       — y-position
  lr: left(0)  / right(1)      — x-position
  tr: opaque(0)/ transparent(1)
  gl: matte(0) / glossy(1)
  sl: small(0) / large(1)      — object scale

Tasks are organised as:
  - 5 simple  + 5 simple-flipped (one dimension each, both polarities)
  - 14 2-way AND composites (all dimension pairs; lr×sl held for val)
  -  4 3-way  AND composites (training)
  -  4 val    tasks  (all involve the held-out lr × sl conjunction)
  Total: 32 tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dlbt.constants import (
    K, DIM_FRONT_BACK, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
)


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

    condition(fb, lr, tr, gl, sl) -> bool
      True  => action 1 (right button) is optimal
      False => action 0 (left  button) is optimal
    """
    delta_u = np.empty(K, dtype=np.float32)
    for k in range(K):
        fb = (k >> DIM_FRONT_BACK)  & 1
        lr = (k >> DIM_LEFT_RIGHT)  & 1
        tr = (k >> DIM_TRANSP)      & 1
        gl = (k >> DIM_GLOSS)       & 1
        sl = (k >> DIM_SMALL_LARGE) & 1
        delta_u[k] = 1.0 if condition(fb, lr, tr, gl, sl) else -1.0
    return delta_u


def _task(name: str, condition) -> Task:
    return Task(name=name, delta_u=_make_delta_u(condition))


# ---------------------------------------------------------------------------
# Task registry
# Bit convention:
#   front_back:  0=front,  1=back
#   left_right:  0=left,   1=right
#   transp:      0=opaque, 1=transparent
#   gloss:       0=matte,  1=glossy
#   small_large: 0=small,  1=large
#
# Action-1 (right button) conventions are set here and fixed throughout.
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {

    # ---- simple (right = property present) ----------------------------------
    "front_back": _task(
        "front_back",
        lambda fb, lr, tr, gl, sl: fb == 1,        # right = back
    ),
    "left_right": _task(
        "left_right",
        lambda fb, lr, tr, gl, sl: lr == 1,        # right = rightward
    ),
    "transparent": _task(
        "transparent",
        lambda fb, lr, tr, gl, sl: tr == 1,        # right = transparent
    ),
    "glossy": _task(
        "glossy",
        lambda fb, lr, tr, gl, sl: gl == 1,        # right = glossy
    ),
    "large": _task(
        "large",
        lambda fb, lr, tr, gl, sl: sl == 1,        # right = large
    ),

    # ---- simple-flipped (right = property absent) ---------------------------
    "front": _task(
        "front",
        lambda fb, lr, tr, gl, sl: fb == 0,        # right = front
    ),
    "left": _task(
        "left",
        lambda fb, lr, tr, gl, sl: lr == 0,        # right = leftward
    ),
    "opaque": _task(
        "opaque",
        lambda fb, lr, tr, gl, sl: tr == 0,        # right = opaque
    ),
    "matte": _task(
        "matte",
        lambda fb, lr, tr, gl, sl: gl == 0,        # right = matte
    ),
    "small": _task(
        "small",
        lambda fb, lr, tr, gl, sl: sl == 0,        # right = small
    ),

    # ---- 2-way: Location × Location (fb × lr) -------------------------------
    "back_and_right": _task(
        "back_and_right",
        lambda fb, lr, tr, gl, sl: fb == 1 and lr == 1,
    ),
    "back_and_left": _task(
        "back_and_left",
        lambda fb, lr, tr, gl, sl: fb == 1 and lr == 0,
    ),
    "front_and_left": _task(
        "front_and_left",
        lambda fb, lr, tr, gl, sl: fb == 0 and lr == 0,
    ),

    # ---- 2-way: fb × material -----------------------------------------------
    "back_and_transparent": _task(
        "back_and_transparent",
        lambda fb, lr, tr, gl, sl: fb == 1 and tr == 1,
    ),
    "front_and_transparent": _task(
        "front_and_transparent",
        lambda fb, lr, tr, gl, sl: fb == 0 and tr == 1,
    ),
    "back_and_glossy": _task(
        "back_and_glossy",
        lambda fb, lr, tr, gl, sl: fb == 1 and gl == 1,
    ),
    "front_and_glossy": _task(
        "front_and_glossy",
        lambda fb, lr, tr, gl, sl: fb == 0 and gl == 1,
    ),

    # ---- 2-way: lr × material -----------------------------------------------
    "right_and_transparent": _task(
        "right_and_transparent",
        lambda fb, lr, tr, gl, sl: lr == 1 and tr == 1,
    ),
    "right_and_glossy": _task(
        "right_and_glossy",
        lambda fb, lr, tr, gl, sl: lr == 1 and gl == 1,
    ),
    "left_and_glossy": _task(
        "left_and_glossy",
        lambda fb, lr, tr, gl, sl: lr == 0 and gl == 1,
    ),

    # ---- 2-way: material × material -----------------------------------------
    "transparent_and_glossy": _task(
        "transparent_and_glossy",
        lambda fb, lr, tr, gl, sl: tr == 1 and gl == 1,
    ),

    # ---- 2-way: sl × location -----------------------------------------------
    "large_and_back": _task(
        "large_and_back",
        lambda fb, lr, tr, gl, sl: sl == 1 and fb == 1,
    ),

    # ---- 2-way: sl × material -----------------------------------------------
    "large_and_transparent": _task(
        "large_and_transparent",
        lambda fb, lr, tr, gl, sl: sl == 1 and tr == 1,
    ),
    "large_and_glossy": _task(
        "large_and_glossy",
        lambda fb, lr, tr, gl, sl: sl == 1 and gl == 1,
    ),

    # ---- 3-way composites (training) ----------------------------------------
    "front_and_transparent_and_glossy": _task(
        "front_and_transparent_and_glossy",
        lambda fb, lr, tr, gl, sl: fb == 0 and tr == 1 and gl == 1,
    ),
    "back_and_transparent_and_glossy": _task(
        "back_and_transparent_and_glossy",
        lambda fb, lr, tr, gl, sl: fb == 1 and tr == 1 and gl == 1,
    ),
    "back_and_right_and_glossy": _task(
        "back_and_right_and_glossy",
        lambda fb, lr, tr, gl, sl: fb == 1 and lr == 1 and gl == 1,
    ),
    "large_and_front_and_transparent": _task(
        "large_and_front_and_transparent",
        lambda fb, lr, tr, gl, sl: sl == 1 and fb == 0 and tr == 1,
    ),

    # ---- val tasks: all involve the held-out lr × sl conjunction ------------
    "right_and_large": _task(
        "right_and_large",
        lambda fb, lr, tr, gl, sl: lr == 1 and sl == 1,
    ),
    "left_and_large": _task(
        "left_and_large",
        lambda fb, lr, tr, gl, sl: lr == 0 and sl == 1,
    ),
    "back_and_right_and_large": _task(
        "back_and_right_and_large",
        lambda fb, lr, tr, gl, sl: fb == 1 and lr == 1 and sl == 1,
    ),
    "right_and_large_and_glossy": _task(
        "right_and_large_and_glossy",
        lambda fb, lr, tr, gl, sl: lr == 1 and sl == 1 and gl == 1,
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS)}")
    return TASKS[name]
