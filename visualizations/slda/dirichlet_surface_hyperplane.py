"""
visualizations/slda/dirichlet_surface_hyperplane.py

Dirichlet PDF surface with one *or more* translucent decision-cut hyperplanes.

Each hyperplane is a constant-z2 cut at position ``z2c`` drawn as a vertical
parallelogram. Multiple planes are supported via the ``planes`` argument; their
default colours match the SLDA task panels (see slda_task_panels.py).

Correct front/back occlusion with an arbitrary number of planes is handled by
keying the matplotlib *zorder* of every surface strip and every plane to its
depth (the z2 coordinate). Strips nearer the viewer (small z2) get a higher
zorder than strips/planes farther back, so they paint on top — this interleaves
surface and planes correctly no matter how many cuts are added.

Run from repo root:
    python visualizations/slda/dirichlet_surface_hyperplane.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from scipy.special import gammaln

OUT_DIR = "visualizations/slda"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Default hyperplane styling — colours match the SLDA task panels
# (slda_task_panels.py: Task 1 green, Task 2 purple, Task T orange)
# ---------------------------------------------------------------------------
TASK_PLANE_COLORS = [
    dict(color="#5ab85a", edge="#3a9a3a"),   # Task 1
    dict(color="#9b7fbf", edge="#7a5eaa"),   # Task 2
    dict(color="#f5a050", edge="#d07830"),   # Task T
]

# A plane is a dict: {"z2c": float, "color": str, "edge": str|None,
#                     "alpha": float|None, "edge_alpha": float|None}
# Only "z2c" and "color" are required; the rest fall back to globals.
DEFAULT_PLANES = [
    dict(z2c=0.62, **TASK_PLANE_COLORS[0]),
    dict(z2c=0.44, **TASK_PLANE_COLORS[1]),
    dict(z2c=0.26, **TASK_PLANE_COLORS[2]),
]


def _logpdf(bary: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    log_B = np.sum(gammaln(alpha)) - gammaln(alpha.sum())
    return np.sum((alpha - 1.0) * np.log(np.clip(bary, 1e-300, None)), axis=1) - log_B


def draw_arrow(ax, start, end, color="black", lw=1.4, mutation_scale=12, zorder=10):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=mutation_scale,
        ),
        zorder=zorder,
    )


def soften_rgba(rgba, mix_with=(1, 1, 1), amount=0.0):
    rgba = np.array(rgba)
    rgb = rgba[:3]
    a = rgba[3]
    mixed = (1 - amount) * rgb + amount * np.array(mix_with)
    return (*mixed, a)


# ---------------------------------------------------------------------------
# zorder bookkeeping — everything is keyed on depth so N planes interleave.
# ---------------------------------------------------------------------------
_Z_BASE = 2.0          # surface strips live in [_Z_BASE, _Z_BASE + 1]
_Z_AXES = 1.0          # axis arrows / labels (behind the surface)
_Z_DOTS = 12.0         # ellipsis dots (on top of everything)


def _strip_zorder(z2v: float) -> float:
    """Nearer strips (small z2) paint later → higher zorder."""
    return _Z_BASE + (1.0 - z2v)


def _plane_zorder(z2c: float) -> float:
    """A plane sits just *behind* the surface at its own depth (embedded look),
    but in front of everything deeper than it."""
    return _Z_BASE + (1.0 - z2c) - 0.05


def plot_dirichlet_surface_hyperplane(
    save_path: str = "visualizations/slda/dirichlet_surface_hyperplane.png",
    alpha: np.ndarray | None = None,
    planes: list[dict] | None = None,
    n_strips: int = 220,
    n_sub: int = 140,
    d_scale: float = 0.95,
    cmap_back: str = "YlOrRd",
    cmap_front: str = "YlOrRd",
    plane_alpha: float = 0.24,
    plane_edge_alpha: float = 0.55,
    shade_split: float | None = None,
    front_alpha: float = 0.18,
    back_alpha: float = 0.085,
):
    """
    Parameters
    ----------
    planes : list of dict, optional
        Each dict describes one hyperplane:
            z2c        : position of the constant-z2 cut (required)
            color      : fill colour (required)
            edge       : outline colour (defaults to ``color``)
            alpha      : fill alpha (defaults to ``plane_alpha``)
            edge_alpha : outline alpha (defaults to ``plane_edge_alpha``)
        Defaults to three task-coloured cuts (see DEFAULT_PLANES).
    shade_split : float, optional
        z2 boundary used purely for the front/back colour muting of the
        surface. Defaults to the frontmost plane position so the single-plane
        appearance is preserved.
    """
    if alpha is None:
        alpha = np.array([4.0, 4.0, 3.0])
    if planes is None:
        planes = DEFAULT_PLANES

    # Fill in per-plane style defaults
    planes = [
        dict(
            z2c=p["z2c"],
            color=p["color"],
            edge=p.get("edge", p["color"]),
            alpha=p.get("alpha", plane_alpha),
            edge_alpha=p.get("edge_alpha", plane_edge_alpha),
        )
        for p in planes
    ]

    if shade_split is None:
        shade_split = min(p["z2c"] for p in planes) if planes else 0.38

    e_dens = np.array([0.00, 3.15])
    e_z1 = np.array([3.35, 0.00])
    e_z2 = np.array([-1.80, -1.20])
    e_z3 = np.array([1.75, -0.95])

    t_g = np.linspace(0.005, 0.995, 100)
    g1, g2 = np.meshgrid(t_g, t_g)
    ok = (g1 + g2) <= 0.995
    b_all = np.column_stack([g1[ok], g2[ok], 1 - g1[ok] - g2[ok]])
    lpdf_max = _logpdf(b_all, alpha).max()

    cmap_back_fn = plt.get_cmap(cmap_back)
    cmap_front_fn = plt.get_cmap(cmap_front)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")
    origin = np.array([0.0, 0.0])

    axes_spec = [
        (e_dens, "density", (0.00, 0.30), 13, True),
        (e_z1, r"$z_1$", (0.28, 0.02), 18, False),
        (e_z3, r"$z_2$", (0.28, -0.12), 18, False),
        (np.array([0.55, -1.25]), r"$z_3$", (0.25, -0.18), 18, False),
        (np.array([-1.30, -0.80]), r"$z_K$", (-0.38, -0.10), 18, False),
    ]

    for end, label, offset, fs, italic in axes_spec:
        draw_arrow(ax, origin, end, color="black", lw=1.45, mutation_scale=13, zorder=_Z_AXES)
        ax.text(
            end[0] + offset[0],
            end[1] + offset[1],
            label,
            fontsize=fs,
            ha="center",
            va="center",
            zorder=_Z_AXES,
            style="italic" if italic else "normal",
        )

    # ------------------------------------------------------------------
    # Build surface strips (one per z2 value)
    # ------------------------------------------------------------------
    z2_vals = np.linspace(0.005, 0.975, n_strips)
    strips = []

    for z2v in sorted(z2_vals, reverse=True):
        z1_max = 1.0 - z2v - 0.005
        if z1_max < 0.02:
            continue

        z1 = np.linspace(0.005, z1_max, n_sub + 1)
        z3 = 1.0 - z1 - z2v
        bary = np.column_stack([z1, np.full_like(z1, z2v), z3])

        pdf = np.exp(_logpdf(bary, alpha) - lpdf_max)
        dv = pdf * d_scale

        sx_f = z1 * e_z1[0] + z2v * e_z2[0] + z3 * e_z3[0]
        sy_f = z1 * e_z1[1] + z2v * e_z2[1] + z3 * e_z3[1]

        sx = sx_f + dv * e_dens[0]
        sy = sy_f + dv * e_dens[1]

        strips.append((z2v, z1, z3, pdf, sx_f, sy_f, sx, sy))

    # ------------------------------------------------------------------
    # Draw surface strips — zorder keyed on depth so planes interleave.
    # ------------------------------------------------------------------
    for z2v, z1, z3, pdf, sx_f, sy_f, sx, sy in strips:
        is_front = z2v < shade_split
        cmap_fn = cmap_front_fn if is_front else cmap_back_fn
        zq = _strip_zorder(z2v)

        for k in range(n_sub):
            avg_pdf = 0.5 * (pdf[k] + pdf[k + 1])
            color = cmap_fn(avg_pdf)

            if is_front:
                color = soften_rgba(color, mix_with=(1.0, 0.40, 0.40), amount=0.11)
                alpha_quad = front_alpha
            else:
                color = soften_rgba(color, mix_with=(1.0, 1.0, 1.0), amount=0.18)
                alpha_quad = back_alpha

            px = [sx[k], sx[k + 1], sx_f[k + 1], sx_f[k]]
            py = [sy[k], sy[k + 1], sy_f[k + 1], sy_f[k]]

            ax.add_patch(
                MplPoly(
                    list(zip(px, py)),
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                    alpha=alpha_quad,
                    zorder=zq,
                )
            )

    # ------------------------------------------------------------------
    # Draw each hyperplane as a full constant-z2 parallelogram.
    # zorder places it at its own depth between the surrounding strips.
    # ------------------------------------------------------------------
    t_lo, t_hi = -0.45, 1.15
    h = d_scale * e_dens * 1.15

    def _floor_pt(z1v, z2c):
        z3v = 1.0 - z1v - z2c
        return np.array(
            [
                z1v * e_z1[0] + z2c * e_z2[0] + z3v * e_z3[0],
                z1v * e_z1[1] + z2c * e_z2[1] + z3v * e_z3[1],
            ]
        )

    for p in planes:
        z2c = p["z2c"]
        p_bl = _floor_pt(t_lo, z2c)
        p_br = _floor_pt(t_hi, z2c)
        plane_pts = [p_bl, p_br, p_br + h, p_bl + h]
        zp = _plane_zorder(z2c)

        ax.add_patch(
            MplPoly(
                plane_pts,
                facecolor=p["color"],
                edgecolor="none",
                alpha=p["alpha"],
                zorder=zp,
            )
        )
        ax.add_patch(
            MplPoly(
                plane_pts,
                facecolor="none",
                edgecolor=p["edge"],
                linewidth=1.0,
                alpha=p["edge_alpha"],
                zorder=zp + 0.01,
            )
        )

    # Ellipsis dots — on top of everything
    ax.scatter(
        [-0.05, -0.20, -0.35],
        [-1.38, -1.38, -1.38],
        s=15,
        color="black",
        zorder=_Z_DOTS,
    )

    ax.set_xlim(-2.55, 4.05)
    ax.set_ylim(-1.85, 3.65)

    fig.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)

    print(f"Saved → {save_path}  ({len(planes)} plane(s))")


if __name__ == "__main__":
    plot_dirichlet_surface_hyperplane()
