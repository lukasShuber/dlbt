"""
visualizations/slda/dirichlet_surface.py

Dirichlet PDF surface, pseudo-3D style matching slda_visuals.py:
  • z1 → right,  z2 → left-down  (isometric simplex floor visible as triangle)
  • density → up
  • Rendered as gradient-coloured strips (painter's algorithm) for smooth look

Run from repo root:
    python visualizations/slda/dirichlet_surface.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from scipy.special import gammaln

OUT_DIR = "visualizations/slda"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _logpdf(bary: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    log_B = np.sum(gammaln(alpha)) - gammaln(alpha.sum())
    return np.sum((alpha - 1.0) * np.log(np.clip(bary, 1e-300, None)), axis=1) - log_B


def draw_arrow(ax, start, end, color="black", lw=1.4, mutation_scale=12, zorder=10):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="->", lw=lw, color=color,
            shrinkA=0, shrinkB=0, mutation_scale=mutation_scale,
        ),
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_dirichlet_surface(
    save_path: str = "visualizations/slda/dirichlet_surface.png",
    alpha: np.ndarray | None = None,
    n_strips: int = 180,      # z2 slices
    n_sub:    int = 120,      # z1 sub-intervals per strip (for gradient colour)
    d_scale:  float = 0.95,  # visual height of peak (in figure units)
    cmap:     str   = "YlOrRd",
):
    if alpha is None:
        alpha = np.array([4.0, 4.0, 3.0])    # symmetric interior peak

    # ── Projection vectors ─────────────────────────────────────────────────
    e_dens = np.array([ 0.00,  3.15])   # up        — density
    e_z1   = np.array([ 3.35,  0.00])   # right     — z_1
    e_z2   = np.array([-1.80, -1.20])   # left-down — z_2
    e_z3   = np.array([ 1.75, -0.95])   # right-down — z_3 (explicit simplex vertex)

    # ── Global max for colour normalisation ───────────────────────────────
    t = np.linspace(0.005, 0.995, 80)
    g1, g2 = np.meshgrid(t, t)
    ok = (g1 + g2) <= 0.995
    b_all = np.column_stack([g1[ok], g2[ok], 1 - g1[ok] - g2[ok]])
    lpdf_max = _logpdf(b_all, alpha).max()

    cmap_fn = plt.get_cmap(cmap)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")
    origin = np.array([0.0, 0.0])

    # ── Axis arrows — drawn FIRST so surface renders on top ───────────────
    # Labels fan out z1 → z2 → z3 → … → zK (e_z2 used for projection only, not drawn)
    axes_spec = [
        (e_dens,                    "density",  ( 0.00,  0.30), 13, True ),
        (e_z1,                      r"$z_1$",   ( 0.28,  0.02), 18, False),
        (e_z3,                      r"$z_2$",   ( 0.28, -0.12), 18, False),
        (np.array([0.55, -1.25]),  r"$z_3$",   ( 0.25, -0.18), 18, False),
        (np.array([-1.30, -0.80]), r"$z_K$",   (-0.38, -0.10), 18, False),
    ]

    for end, label, offset, fs, italic in axes_spec:
        draw_arrow(ax, origin, end, color="black", lw=1.45, mutation_scale=13, zorder=1)
        ax.text(
            end[0] + offset[0], end[1] + offset[1], label,
            fontsize=fs, ha="center", va="center", zorder=1,
            style="italic" if italic else "normal",
        )

    # Ellipsis dots — between z3 and zK
    ax.scatter([-0.05, -0.20, -0.35], [-1.38, -1.38, -1.38],
               s=15, color="black", zorder=1)

    # ── Surface strips (gradient-coloured, painter order: far z2 first) ──
    z2_vals = np.linspace(0.005, 0.975, n_strips)

    for z2v in sorted(z2_vals, reverse=True):
        z1_max = 1.0 - z2v - 0.005
        if z1_max < 0.02:
            continue

        z1 = np.linspace(0.005, z1_max, n_sub + 1)
        z3 = 1.0 - z1 - z2v

        bary = np.column_stack([z1, np.full_like(z1, z2v), z3])
        pdf  = np.exp(_logpdf(bary, alpha) - lpdf_max)

        dv   = pdf * d_scale

        # Floor uses all three simplex vertices — surface extends in z3 direction
        sx_f = z1 * e_z1[0] + z2v * e_z2[0] + z3 * e_z3[0]
        sy_f = z1 * e_z1[1] + z2v * e_z2[1] + z3 * e_z3[1]

        sx   = sx_f + dv * e_dens[0]
        sy   = sy_f + dv * e_dens[1]

        # Render each sub-interval as its own coloured quad
        for k in range(n_sub):
            avg_pdf = 0.5 * (pdf[k] + pdf[k + 1])
            color   = cmap_fn(avg_pdf)
            px = [sx[k],   sx[k+1],   sx_f[k+1], sx_f[k]  ]
            py = [sy[k],   sy[k+1],   sy_f[k+1], sy_f[k]  ]
            ax.add_patch(MplPoly(
                list(zip(px, py)),
                facecolor=color, edgecolor="none",
                linewidth=0, alpha=0.1, zorder=2,
            ))

    # ── Limits ────────────────────────────────────────────────────────────
    ax.set_xlim(-2.55, 4.05)
    ax.set_ylim(-1.85, 3.65)

    fig.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    plot_dirichlet_surface()
