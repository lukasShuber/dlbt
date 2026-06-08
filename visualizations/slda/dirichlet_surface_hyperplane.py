"""
visualizations/slda/dirichlet_surface_hyperplane.py

Dirichlet PDF surface with one *or more* translucent decision-cut hyperplanes,
rendered with a rotatable pseudo-3D camera.

Geometry
--------
The simplex lives on a horizontal floor (an equilateral triangle); the density
is the vertical axis. A vertical rotation axis is placed at the *mean* of the
distribution (Dirichlet mean = alpha / alpha.sum()). The ``azimuth`` knob spins
the whole scene about that axis — because we project with an oblique camera that
maps the vertical (density) axis straight up on screen, rotating only swings the
floor around while density stays vertical.

Occlusion (front/back) is handled by keying every drawn quad's matplotlib
*zorder* to its camera depth, so it stays correct at any azimuth and for any
number of planes.

Colours of the default planes match the SLDA task panels (slda_task_panels.py).

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
# (Task 1 green, Task 2 purple, Task T orange)
# ---------------------------------------------------------------------------
TASK_PLANE_COLORS = [
    dict(color="#5ab85a", edge="#3a9a3a"),   # Task 1
    dict(color="#9b7fbf", edge="#7a5eaa"),   # Task 2
    dict(color="#f5a050", edge="#d07830"),   # Task T
]

# A plane is a dict: {"z2c": float, "color": str, "edge": str|None,
#                     "alpha": float|None, "edge_alpha": float|None}
DEFAULT_PLANES = [
    dict(z2c=0.62, **TASK_PLANE_COLORS[0]),
    dict(z2c=0.44, **TASK_PLANE_COLORS[1]),
    dict(z2c=0.26, **TASK_PLANE_COLORS[2]),
]

# zorder bookkeeping
_Z_AXES = 1.0       # axis arrows / labels (behind the surface)
_Z_BASE = 50.0      # surface/plane quads: zorder = _Z_BASE - depth
_Z_DOTS = 200.0     # ellipsis dots (always on top)


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
# Camera: rotate the simplex floor about a vertical axis at the mean, then
# project with an oblique camera (density → straight up on screen).
# ---------------------------------------------------------------------------
def _make_camera(alpha, azimuth_deg=0.0, tilt=0.50, dens_scale=3.0,
                 r_floor=2.2, base_angle_deg=90.0):
    angs = np.radians(base_angle_deg + np.array([0.0, 120.0, 240.0]))
    V = np.stack([r_floor * np.cos(angs), r_floor * np.sin(angs)], axis=1)  # (3,2)

    mean_b = alpha / alpha.sum()
    pivot = mean_b @ V                       # rotation axis foot (world floor xy)

    th = np.radians(azimuth_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])

    def project(bary, dens):
        """bary: (M,3) barycentric, dens: (M,) density height.
        Returns sx, sy (screen) and depth (camera depth, smaller = nearer)."""
        w = np.asarray(bary) @ V                       # (M,2) world floor
        wr = (w - pivot) @ R.T + pivot                 # rotate about pivot
        sx = wr[:, 0]
        depth = wr[:, 1]
        sy = depth * tilt + np.asarray(dens) * dens_scale
        return sx, sy, depth

    return dict(project=project, V=V, pivot=pivot, R=R, tilt=tilt,
                dens_scale=dens_scale)


def plot_dirichlet_surface_hyperplane(
    save_path: str = "visualizations/slda/dirichlet_surface_hyperplane.png",
    alpha: np.ndarray | None = None,
    planes: list[dict] | None = None,
    # ── camera knobs ──────────────────────────────────────────────
    azimuth_deg: float = 35.0,
    tilt: float = 0.50,
    dens_scale: float = 3.0,
    r_floor: float = 2.2,
    base_angle_deg: float = 90.0,
    # ── surface resolution ────────────────────────────────────────
    n_strips: int = 220,
    n_sub: int = 140,
    d_scale: float = 1.0,
    cmap_back: str = "YlOrRd",
    cmap_front: str = "YlOrRd",
    # ── plane styling ─────────────────────────────────────────────
    plane_alpha: float = 0.24,
    plane_edge_alpha: float = 0.55,
    plane_height: float = 1.15,
    n_plane_seg: int = 60,
    front_alpha: float = 0.18,
    back_alpha: float = 0.085,
    show_axes: bool = True,
):
    """
    Parameters
    ----------
    azimuth_deg : float
        Rotation about the vertical axis at the distribution mean (degrees).
    tilt : float
        Camera elevation foreshortening of the floor (0 = edge-on, 1 = top-down).
    planes : list of dict, optional
        Each {z2c, color, edge?, alpha?, edge_alpha?}. Defaults to three
        task-coloured cuts (DEFAULT_PLANES).
    """
    if alpha is None:
        alpha = np.array([4.0, 4.0, 3.0])
    alpha = np.asarray(alpha, dtype=float)
    if planes is None:
        planes = DEFAULT_PLANES

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

    cam = _make_camera(alpha, azimuth_deg, tilt, dens_scale, r_floor, base_angle_deg)
    project = cam["project"]
    pivot = cam["pivot"]
    pivot_depth = pivot[1]   # rotation pivot is fixed under rotation → depth = pivot_y

    # density normalisation
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

    bounds = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf}

    def _track(xs, ys):
        bounds["xmin"] = min(bounds["xmin"], np.min(xs))
        bounds["xmax"] = max(bounds["xmax"], np.max(xs))
        bounds["ymin"] = min(bounds["ymin"], np.min(ys))
        bounds["ymax"] = max(bounds["ymax"], np.max(ys))

    # ------------------------------------------------------------------
    # Surface — one quad ribbon per (z2 strip, z1 sub-interval).
    # zorder per quad from its camera depth → correct at any azimuth.
    # ------------------------------------------------------------------
    z2_vals = np.linspace(0.005, 0.975, n_strips)
    for z2v in z2_vals:
        z1_max = 1.0 - z2v - 0.005
        if z1_max < 0.02:
            continue
        z1 = np.linspace(0.005, z1_max, n_sub + 1)
        z3 = 1.0 - z1 - z2v
        bary = np.column_stack([z1, np.full_like(z1, z2v), z3])
        pdf = np.exp(_logpdf(bary, alpha) - lpdf_max) * d_scale

        sx_t, sy_t, dep_t = project(bary, pdf)
        sx_f, sy_f, dep_f = project(bary, np.zeros_like(pdf))

        for k in range(n_sub):
            avg_pdf = 0.5 * (pdf[k] + pdf[k + 1])
            depth = 0.25 * (dep_t[k] + dep_t[k + 1] + dep_f[k] + dep_f[k + 1])
            is_front = depth < pivot_depth

            cmap_fn = cmap_front_fn if is_front else cmap_back_fn
            color = cmap_fn(min(avg_pdf, 1.0))
            if is_front:
                color = soften_rgba(color, mix_with=(1.0, 0.40, 0.40), amount=0.11)
                a_quad = front_alpha
            else:
                color = soften_rgba(color, mix_with=(1.0, 1.0, 1.0), amount=0.18)
                a_quad = back_alpha

            px = [sx_t[k], sx_t[k + 1], sx_f[k + 1], sx_f[k]]
            py = [sy_t[k], sy_t[k + 1], sy_f[k + 1], sy_f[k]]
            _track(px, py)
            ax.add_patch(MplPoly(list(zip(px, py)), facecolor=color,
                                 edgecolor="none", linewidth=0, alpha=a_quad,
                                 zorder=_Z_BASE - depth))

    # ------------------------------------------------------------------
    # Hyperplanes — each a constant-z2 vertical sheet, segmented along z1
    # so each segment z-orders independently against the surface.
    # ------------------------------------------------------------------
    t_lo, t_hi = -0.45, 1.15
    for p in planes:
        z2c = p["z2c"]
        t = np.linspace(t_lo, t_hi, n_plane_seg + 1)
        z3 = 1.0 - t - z2c
        bary = np.column_stack([t, np.full_like(t, z2c), z3])
        sx_t, sy_t, dep_t = project(bary, np.full_like(t, plane_height))
        sx_b, sy_b, dep_b = project(bary, np.zeros_like(t))

        for k in range(n_plane_seg):
            depth = 0.25 * (dep_t[k] + dep_t[k + 1] + dep_b[k] + dep_b[k + 1])
            px = [sx_b[k], sx_b[k + 1], sx_t[k + 1], sx_t[k]]
            py = [sy_b[k], sy_b[k + 1], sy_t[k + 1], sy_t[k]]
            _track(px, py)
            ax.add_patch(MplPoly(list(zip(px, py)), facecolor=p["color"],
                                 edgecolor="none", alpha=p["alpha"],
                                 zorder=_Z_BASE - depth - 0.02))

        # outline (single border at the sheet's mean depth)
        depth_mean = 0.5 * (dep_t.mean() + dep_b.mean())
        outline = [(sx_b[0], sy_b[0]), (sx_b[-1], sy_b[-1]),
                   (sx_t[-1], sy_t[-1]), (sx_t[0], sy_t[0])]
        ax.add_patch(MplPoly(outline, facecolor="none", edgecolor=p["edge"],
                             linewidth=1.0, alpha=p["edge_alpha"],
                             zorder=_Z_BASE - depth_mean + 0.01))

    # ------------------------------------------------------------------
    # Axis frame — z1/z2/z3 to the simplex vertices + vertical density axis,
    # all emanating from the back-most simplex vertex (so density rises behind
    # the surface, as in the original look). Rotates with the scene.
    # ------------------------------------------------------------------
    if show_axes:
        V = cam["V"]
        verts_b = np.eye(3)                       # z1, z2, z3 unit corners
        vx, vy, vdep = project(verts_b, np.zeros(3))
        back = int(np.argmax(vdep))               # farthest vertex = origin
        o = np.array([vx[back], vy[back]])

        labels = {0: r"$z_1$", 1: r"$z_2$", 2: r"$z_3$"}
        for j in range(3):
            if j == back:
                continue
            end = np.array([vx[j], vy[j]])
            d = end - o
            end = o + d * 1.12                     # slight overshoot for arrowhead
            draw_arrow(ax, o, end, color="black", lw=1.45, mutation_scale=13,
                       zorder=_Z_AXES)
            lab = o + d * 1.22
            ax.text(lab[0], lab[1], labels[j], fontsize=18, ha="center",
                    va="center", zorder=_Z_AXES)
            _track([end[0], lab[0]], [end[1], lab[1]])

        # vertical density axis from the back vertex
        dens_top = o + np.array([0.0, cam["dens_scale"] * (plane_height + 0.15)])
        draw_arrow(ax, o, dens_top, color="black", lw=1.45, mutation_scale=13,
                   zorder=_Z_AXES)
        ax.text(dens_top[0], dens_top[1] + 0.18, "density", fontsize=13,
                ha="center", va="center", style="italic", zorder=_Z_AXES)
        _track([dens_top[0]], [dens_top[1] + 0.3])

        # decorative "...  z_K" hint near the back vertex
        ax.scatter(o[0] + np.array([-0.18, -0.33, -0.48]),
                   o[1] + np.array([-0.30, -0.30, -0.30]),
                   s=15, color="black", zorder=_Z_DOTS)
        ax.text(o[0] - 0.70, o[1] - 0.30, r"$z_K$", fontsize=18,
                ha="center", va="center", zorder=_Z_AXES)

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------
    mx = 0.35
    ax.set_xlim(bounds["xmin"] - mx, bounds["xmax"] + mx)
    ax.set_ylim(bounds["ymin"] - mx, bounds["ymax"] + mx)

    fig.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved → {save_path}  ({len(planes)} plane(s), azimuth={azimuth_deg}°)")


if __name__ == "__main__":
    plot_dirichlet_surface_hyperplane()
