"""
Verify the 10 task definitions.

Checks:
  - Simple tasks partition latent states 8/8 (half right, half left).
  - Composite tasks partition 4/12.
  - Prints the delta_u pattern as a 4-dim binary table.

Run from repo root:
    python examples/01_check_tasks.py
"""

import numpy as np
from dlbt.data.task import TASKS
from dlbt.constants import K

SIMPLE    = {"front_back", "triangular", "transparent", "glossy"}
COMPOSITE = set(TASKS) - SIMPLE


def fmt_state(k: int) -> str:
    """Return a short label for latent state k, e.g. 'B/N/T/G'."""
    fb = "B" if (k >> 3) & 1 else "F"   # Back / Front
    sh = "N" if (k >> 2) & 1 else "T"   # Non-tri / Triangular
    tr = "T" if (k >> 1) & 1 else "-"   # Transparent / opaque
    gl = "G" if (k >> 0) & 1 else "-"   # Glossy / matte
    return f"{fb}/{sh}/{tr}/{gl}"


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print(f"{'task':<35}  n_right  n_left  type")
print("-" * 65)
for name, task in TASKS.items():
    n_right = int((task.delta_u > 0).sum())
    n_left  = K - n_right
    kind    = "simple" if name in SIMPLE else "composite"
    print(f"{name:<35}  {n_right:>7}  {n_left:>6}  {kind}")

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
for name in SIMPLE:
    n_right = int((TASKS[name].delta_u > 0).sum())
    assert n_right == K // 2, f"{name}: expected {K//2} right-states, got {n_right}"

for name in COMPOSITE:
    n_right = int((TASKS[name].delta_u > 0).sum())
    assert n_right == K // 4, f"{name}: expected {K//4} right-states, got {n_right}"

print("\nAll assertions passed.")

# ---------------------------------------------------------------------------
# Print full delta_u table for visual inspection
# ---------------------------------------------------------------------------
SHORT = {
    "front_back":                   "FB",
    "triangular":                   "CAT",
    "transparent":                  "TP",
    "glossy":                       "GL",
    "back_and_glossy":              "B+GL",
    "front_and_transparent":        "F+TP",
    "triangular_and_transparent":   "CAT+TP",
    "nontriangular_and_glossy":     "NT+GL",
    "triangular_and_front":         "CAT+F",
    "nontriangular_and_front":      "NT+F",
}

print(f"\n{'state':<16} " + "  ".join(f"{SHORT[n]:>6}" for n in TASKS))
print("-" * (16 + 8 * len(TASKS)))
for k in range(K):
    row = "  ".join(f"{'R' if TASKS[n].delta_u[k] > 0 else 'L':>6}" for n in TASKS)
    print(f"{fmt_state(k):<16} {row}")
