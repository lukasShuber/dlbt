"""
visualizations/slda/dirichlet_surface_hyperplane.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from scipy.special import gammaln

OUT_DIR = "visualizations/slda"
os.makedirs(OUT_DIR, exist_ok=True)


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


def plot_dirichlet_surface_hyperplane(
    save_path: str = "visualizations/slda/dirichlet_surface_hyperplane.png",
    alpha: np.ndarray | None = None,
    n_strips: int = 220,
    n_sub: int = 140,
    d_scale: float = 0.95,
    cmap_back: str = "YlOrRd",
    cmap_front: str = "YlOrRd",
    plane_color: str = "#4CAF50",
    plane_alpha: float = 0.24,
    plane_edge_alpha: float = 0.55,
    z2c: float = 0.38,
):
    if alpha is None:
        alpha = np.array([4.0, 4.0, 3.0])

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
        draw_arrow(ax, origin, end, color="black", lw=1.45, mutation_scale=13, zorder=1)
        ax.text(
            end[0] + offset[0],
            end[1] + offset[1],
            label,
            fontsize=fs,
            ha="center",
            va="center",
            zorder=1,
            style="italic" if italic else "normal",
        )

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

    # Pass 1: all surface strips, with slightly muted far side
    for z2v, z1, z3, pdf, sx_f, sy_f, sx, sy in strips:
        is_front = z2v < z2c
        cmap_fn = cmap_front_fn if is_front else cmap_back_fn

        for k in range(n_sub):
            avg_pdf = 0.5 * (pdf[k] + pdf[k + 1])
            color = cmap_fn(avg_pdf)

            if is_front:
                color = soften_rgba(color, mix_with=(1.0, 0.45, 0.45), amount=0.12)
                alpha_quad = 0.12
            else:
                color = soften_rgba(color, mix_with=(1.0, 1.0, 1.0), amount=0.18)
                alpha_quad = 0.085

            px = [sx[k], sx[k + 1], sx_f[k + 1], sx_f[k]]
            py = [sy[k], sy[k + 1], sy_f[k + 1], sy_f[k]]

            ax.add_patch(
                MplPoly(
                    list(zip(px, py)),
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                    alpha=alpha_quad,
                    zorder=2,
                )
            )

    # Pass 2: constant-z2 hyperplane as full parallelogram
    t_lo, t_hi = -0.45, 1.15

    def _floor_pt_z2c(z1v):
        z3v = 1.0 - z1v - z2c
        return np.array(
            [
                z1v * e_z1[0] + z2c * e_z2[0] + z3v * e_z3[0],
                z1v * e_z1[1] + z2c * e_z2[1] + z3v * e_z3[1],
            ]
        )

    p_bl = _floor_pt_z2c(t_lo)
    p_br = _floor_pt_z2c(t_hi)
    h = d_scale * e_dens * 1.15
    p_tl = p_bl + h
    p_tr = p_br + h

    plane_pts = [p_bl, p_br, p_tr, p_tl]

    ax.add_patch(
        MplPoly(
            plane_pts,
            facecolor=plane_color,
            edgecolor="none",
            alpha=plane_alpha,
            zorder=3,
        )
    )

    ax.add_patch(
        MplPoly(
            plane_pts,
            facecolor="none",
            edgecolor=plane_color,
            linewidth=1.0,
            alpha=plane_edge_alpha,
            zorder=3.1,
        )
    )

    # Pass 3: redraw front side so it occludes the plane and outline
    for z2v, z1, z3, pdf, sx_f, sy_f, sx, sy in strips:
        if z2v >= z2c:
            continue

        for k in range(n_sub):
            avg_pdf = 0.5 * (pdf[k] + pdf[k + 1])
            color = cmap_front_fn(avg_pdf)
            color = soften_rgba(color, mix_with=(1.0, 0.35, 0.35), amount=0.10)

            px = [sx[k], sx[k + 1], sx_f[k + 1], sx_f[k]]
            py = [sy[k], sy[k + 1], sy_f[k + 1], sy_f[k]]

            ax.add_patch(
                MplPoly(
                    list(zip(px, py)),
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                    alpha=0.14,
                    zorder=4,
                )
            )

    # Ellipsis dots — on top of everything
    ax.scatter(
        [-0.05, -0.20, -0.35],
        [-1.38, -1.38, -1.38],
        s=15,
        color="black",
        zorder=5,
    )

    ax.set_xlim(-2.55, 4.05)
    ax.set_ylim(-1.85, 3.65)

    fig.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)

    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_dirichlet_surface_hyperplane()
