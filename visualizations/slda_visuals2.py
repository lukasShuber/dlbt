"""
visualizations/slda_visuals2.py — 3-D point cloud + hyperplanes for SLDA schematic.

Run from repo root:
    python visualizations/slda_visuals2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

os.makedirs("visualizations", exist_ok=True)
OUT = "visualizations/slda_3d_panel.png"

RNG    = np.random.default_rng(42)
COLORS = ["#C0392B", "#2471A3", "#1E8449"]   # Task 1, 2, T
N_PTS  = 500

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
pts = RNG.standard_normal((N_PTS, 3)) * 0.85

normals = np.array(
    [[ 0.80,  0.50,  0.30],
     [ 0.20,  0.85, -0.40],
     [-0.55,  0.25,  0.80]],
    dtype=float,
)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
offsets = np.array([0.08, -0.10, 0.06])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection="3d")

ax.set_axis_off()
ax.view_init(elev=20, azim=-55)
LIM = 2.6
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_zlim(-LIM, LIM)
ax.set_box_aspect([1, 1, 1])

# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------
ax.scatter(
    *pts.T, s=12, c="#888888", alpha=0.35,
    linewidths=0, zorder=1, depthshade=True,
)

# ---------------------------------------------------------------------------
# Hyperplanes
# ---------------------------------------------------------------------------
g = np.linspace(-2.5, 2.5, 35)
XX, YY = np.meshgrid(g, g)

for k, color in enumerate(COLORS):
    n, d = normals[k], offsets[k]
    if abs(n[2]) < 0.05:
        continue

    ZZ = (d - n[0] * XX - n[1] * YY) / n[2]
    ZZ = np.where(np.abs(ZZ) < 2.55, ZZ, np.nan)

    # Filled transparent surface
    ax.plot_surface(
        XX, YY, ZZ,
        alpha=0.22, color=color, linewidth=0,
        antialiased=True, zorder=2,
    )

    # Boundary edges of the plane rectangle
    for xe in [g[0], g[-1]]:
        yv = g
        zv = (d - n[0] * xe - n[1] * yv) / n[2]
        ok = np.abs(zv) < 2.55
        if ok.any():
            ax.plot([xe] * ok.sum(), yv[ok], zv[ok],
                    color=color, lw=1.4, alpha=0.75, zorder=4)
    for ye in [g[0], g[-1]]:
        xv = g
        zv = (d - n[0] * xv - n[1] * ye) / n[2]
        ok = np.abs(zv) < 2.55
        if ok.any():
            ax.plot(xv[ok], [ye] * ok.sum(), zv[ok],
                    color=color, lw=1.4, alpha=0.75, zorder=4)

# ---------------------------------------------------------------------------
# Dimension arrows  (inspired by multi-axis "population representation" style)
#
# 3 solid labelled axes suggest a 3-D subspace;
# faded dashed arrows spread in other directions suggest ℝ^1024.
# ---------------------------------------------------------------------------
L     = 2.3
BLACK = "#111111"
GRAY  = "#999999"

# Helper: draw one arrow + label from origin
def _dim_arrow(direction, name, color, lw, ls, alpha, fontsize, fontstyle="normal"):
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d) * L
    ax.plot(
        [0, d[0]], [0, d[1]], [0, d[2]],
        color=color, lw=lw, ls=ls, alpha=alpha, zorder=7,
    )
    # Arrow tip: small triangle drawn as a short thick segment
    tip_frac = 0.88
    ax.plot(
        [d[0] * tip_frac, d[0]], [d[1] * tip_frac, d[1]], [d[2] * tip_frac, d[2]],
        color=color, lw=lw * 2.2, alpha=alpha, zorder=8, solid_capstyle="round",
    )
    ax.text(
        *(d * 1.22), name,
        fontsize=fontsize, color=color, ha="center", va="center",
        style=fontstyle, fontweight="bold" if color == BLACK else "normal",
    )

# Solid axes  (3 main dimensions)
_dim_arrow([ 1,  0,  0], "Dim 1", BLACK, 1.6, "-", 0.9, 9)
_dim_arrow([ 0,  1,  0], "Dim 2", BLACK, 1.6, "-", 0.9, 9)
_dim_arrow([ 0,  0,  1], "Dim 3", BLACK, 1.6, "-", 0.9, 9)

# Faded dashed axes  (extra dimensions → high-D cue)
_dim_arrow([ 0.62,  0.38, -0.69], "Dim 4",    GRAY, 1.1, "--", 0.55, 8, "italic")
_dim_arrow([-0.48, -0.32,  0.82], "Dim 5",    GRAY, 1.1, "--", 0.55, 8, "italic")
_dim_arrow([ 0.55, -0.80,  0.24], "···",      GRAY, 1.0, "--", 0.45, 9, "italic")
_dim_arrow([-0.72,  0.58, -0.38], "Dim 1024", GRAY, 1.1, "--", 0.55, 8, "italic")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.set_title(
    r"CLIP feature space  $\mathbf{f}(\mathbf{x})\in\mathbb{R}^{1024}$",
    fontsize=11, pad=8,
)

# ---------------------------------------------------------------------------
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT}")
