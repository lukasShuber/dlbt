"""
Verify the 22 task definitions.

Checks:
  - Simple tasks (8) partition latent states 8/8 (half right, half left).
  - Composite tasks (14) partition 4/12.
  - Prints the delta_u pattern as a 4-dim binary table.

Run from repo root:
    python examples/01_check_tasks.py
"""

import numpy as np
from dlbt.data.task import TASKS
from dlbt.constants import K

# 4 positive + 4 negative single-dimension tasks
SIMPLE = {
    "right", "transparent", "glossy", "large",
    "left",  "opaque",      "matte",  "small",
}
COMPOSITE = set(TASKS) - SIMPLE


def fmt_state(k: int) -> str:
    """Return a short label for latent state k, e.g. 'R/T/G/L'."""
    lr = "R" if (k >> 3) & 1 else "L"   # Right / Left
    tr = "T" if (k >> 2) & 1 else "-"   # Transparent / opaque
    gl = "G" if (k >> 1) & 1 else "-"   # Glossy / matte
    sl = "B" if (k >> 0) & 1 else "s"   # Big / small
    return f"{lr}/{tr}/{gl}/{sl}"


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print(f"{'task':<40}  n_right  n_left  type")
print("-" * 70)
for name, task in TASKS.items():
    n_right = int((task.delta_u > 0).sum())
    n_left  = K - n_right
    kind    = "simple" if name in SIMPLE else "composite"
    print(f"{name:<40}  {n_right:>7}  {n_left:>6}  {kind}")

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
for name in SIMPLE:
    n_right = int((TASKS[name].delta_u > 0).sum())
    assert n_right == K // 2, f"{name}: expected {K//2} right-states, got {n_right}"

for name in COMPOSITE:
    n_right = int((TASKS[name].delta_u > 0).sum())
    # 2-way AND → K/4 right states; 3-way AND → K/8
    assert n_right in (K // 4, K // 8), \
        f"{name}: expected {K//4} or {K//8} right-states, got {n_right}"

print("\nAll assertions passed.")

# ---------------------------------------------------------------------------
# Print full delta_u table for visual inspection
# ---------------------------------------------------------------------------
SHORT = {t: t[:8] for t in TASKS}   # truncate names to 8 chars for display

print(f"\n{'state':<16} " + "  ".join(f"{SHORT[n]:>8}" for n in TASKS))
print("-" * (16 + 10 * len(TASKS)))
for k in range(K):
    row = "  ".join(f"{'R' if TASKS[n].delta_u[k] > 0 else 'L':>8}" for n in TASKS)
    print(f"{fmt_state(k):<16} {row}")
