"""
visualizations/slda/slda_task_panels.py — per-task decision-boundary panels.

For each task produces a small figure with:
  • top    : styled box with 3D-style point cloud + one translucent hyperplane
  • bottom : logistic sigmoid curve

Colors match the three planes in slda_visuals.py.

Run from repo root:
    python visualizations/slda/slda_task_panels.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon, FancyBboxPatch

OUT_DIR = "visualizations/slda"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared point cloud — same seed and distribution as slda_visuals.py
# ---------------------------------------------------------------------------
_rng_cloud = np.random.default_rng(4)
_CLOUD_X = _rng_cloud.multivariate_normal(
    mean=np.array([0.72, 0.95]),
    cov=np.array([[0.55, 0.16], [0.16, 0.40]]),
    size=1200,
)

# ---------------------------------------------------------------------------
# Axis arrows — same geometry as slda_visuals.py
# ---------------------------------------------------------------------------
_ORIGIN = np.array([0.0, 0.0])
_AXES = [
    (np.array([0.0,  3.15]), r"$d_1$",     ( 0.00,  0.28)),
    (np.array([3.35, 0.0 ]), r"$d_2$",     ( 0.28,  0.02)),
    (np.array([2.45,-0.55]), r"$d_3$",     ( 0.28, -0.03)),
    (np.array([1.75,-0.95]), r"$d_4$",     ( 0.22, -0.14)),
    (np.array([-0.90,-1.30]),r"$d_{1024}$",(-0.38, -0.18)),
]

# ---------------------------------------------------------------------------
# Plane geometries — same as slda_visuals.py
# ---------------------------------------------------------------------------
def _make_plane(center, u, v):
    c, u, v = np.array(center), np.array(u), np.array(v)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])

_PLANES = {
    "Task 1": _make_plane((0.55, 0.65), (2.35,  0.32), ( 0.22, 1.10)),
    "Task 2": _make_plane((0.85, 0.95), (1.90, -0.82), ( 0.96, 0.28)),
    "Task T": _make_plane((0.40, 0.80), (1.30,  1.48), (-1.18, 0.32)),
}

# ---------------------------------------------------------------------------
# Task definitions — colors match the planes in slda_visuals.py
# ---------------------------------------------------------------------------
TASKS = [
    dict(name="Task 1", color="#5ab85a", edge="#3a9a3a", k=0.8,  seed=42),
    dict(name="Task 2", color="#9b7fbf", edge="#7a5eaa", k=2.5,  seed=43),
    dict(name="Task T", color="#f5a050", edge="#d07830", k=1.4,  seed=44),
]


def _draw_arrow(ax, start, end, color="black", lw=1.1, mutation_scale=7):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="->", lw=lw, color=color,
            shrinkA=0, shrinkB=0, mutation_scale=mutation_scale,
        ),
        zorder=10,
    )


def make_panel(name, color, edge, k, seed):
    fig = plt.figure(figsize=(2.0, 3.8))

    ax_s = fig.add_axes([0.08, 0.50, 0.86, 0.42])
    ax_g = fig.add_axes([0.18, 0.07, 0.70, 0.32])

    # ── 3D-style point cloud + hyperplane ────────────────────────────
    ax_s.set_aspect("equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])

    # Point cloud (half of 1200 = 600 points, larger markers)
    ax_s.scatter(
        _CLOUD_X[:600, 0], _CLOUD_X[:600, 1],
        s=26, color="0.35", alpha=0.17, linewidths=0, zorder=3,
    )

    # Translucent hyperplane for this task only
    patch = Polygon(
        _PLANES[name], closed=True,
        facecolor=color, edgecolor=edge,
        alpha=0.30, linewidth=1.1,
        joinstyle="round", zorder=2,
    )
    ax_s.add_patch(patch)

    # Axis arrows + labels
    for end, label, offset in _AXES:
        _draw_arrow(ax_s, _ORIGIN, end)
        ax_s.text(
            end[0] + offset[0], end[1] + offset[1], label,
            fontsize=7, ha="center", va="center", zorder=11,
        )

    # Ellipsis dots
    ax_s.scatter([0.18, 0.38, 0.58], [-1.42, -1.42, -1.42],
                 s=3, color="black", zorder=11)

    ax_s.set_xlim(-2.55, 4.05)
    ax_s.set_ylim(-1.85, 3.65)

    # Styled border box with rounded corners
    for sp in ax_s.spines.values():
        sp.set_visible(False)
    ax_s.set_facecolor("none")
    ax_s.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0,rounding_size=0.07",
        transform=ax_s.transAxes,
        facecolor=color + "22",
        edgecolor=edge,
        linewidth=1.8,
        clip_on=False,
        zorder=0,
    ))
    ax_s.set_title(name, color=edge, fontsize=11, fontweight="bold", pad=6)

    # ── Sigmoid — steepness k varies per task ────────────────────────
    z = np.linspace(-4.5, 4.5, 400)
    p = 1.0 / (1.0 + np.exp(-k * z))

    ax_g.plot(z, p, color=color, lw=2.0)
    ax_g.axvline(0, color="0.55", lw=0.8, ls="--")

    ax_g.set_xlim(-4.5, 4.5)
    ax_g.set_ylim(-0.05, 1.05)
    ax_g.set_xticks([])
    ax_g.set_yticks([])

    # Schematic y-axis arrow with just 0/1 labels
    ax_g.annotate("", xy=(0, 1.08), xytext=(0, -0.08),
                  xycoords=("axes fraction", "data"),
                  textcoords=("axes fraction", "data"),
                  arrowprops=dict(arrowstyle="-|>", color="0.30",
                                  lw=0.9, mutation_scale=8),
                  annotation_clip=False)
    ax_g.text(-0.18, 0.0, "0", transform=ax_g.get_yaxis_transform(),
              fontsize=8, ha="right", va="center", color="0.35")
    ax_g.text(-0.18, 1.0, "1", transform=ax_g.get_yaxis_transform(),
              fontsize=8, ha="right", va="center", color="0.35")

    # Schematic x-axis arrow (no labels)
    ax_g.annotate("", xy=(1.08, 0), xytext=(-0.08, 0),
                  xycoords=("axes fraction", "data"),
                  textcoords=("axes fraction", "data"),
                  arrowprops=dict(arrowstyle="-|>", color="0.30",
                                  lw=0.9, mutation_scale=8),
                  annotation_clip=False)

    sns.despine(ax=ax_g, top=True, right=True, bottom=True, left=True)

    # ── Save ─────────────────────────────────────────────────────────
    tag = name.replace(" ", "_").lower()
    out = os.path.join(OUT_DIR, f"task_panel_{tag}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    for task in TASKS:
        make_panel(**task)
