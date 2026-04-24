"""
simplex_threshold_intuition.py
--------------------------------
Visualises why the decision threshold h matters in a K=3 toy DLBT model.

Setup
-----
  K = 3 latent states:  A (yes), B (no), C (no)
  Utility:  Δu = (+1, −1, −1)   →   b·Δu = 2·b_A − 1
  Choice:   "yes"  iff  b·Δu > h   ↔   b_A > (h+1)/2

  Dirichlet prior b ~ Dir(α):  marginal  b_A ~ Beta(α_A, α_B + α_C)

The probability simplex is a 2-D triangle — perfect for visualisation.

Panels
------
  1. Uniform Dir(1,1,1) + h = 0          → P(yes) = 0.25  (biased)
  2. Uniform Dir(1,1,1) + h = h_n        → P(yes) = 0.50  (corrected)
  3. Peaked Dir(6,1,1)  + h = h_n        → P(yes) ≫ 0.5   (model sees "A")
  4. Peaked Dir(1,5,5)  + h = h_n        → P(yes) ≪ 0.5   (model sees "B/C")

The dividing line  b_A = threshold  is a horizontal line through the triangle.
The "yes" region (above) is shaded blue; "no" region (below) shaded red.

Run from repo root:
    python experiments/figures/simplex_threshold_intuition.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.stats import beta as scipy_beta, dirichlet

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Triangle geometry
# A (yes) at top, B (no 1) at bottom-left, C (no 2) at bottom-right
# ---------------------------------------------------------------------------
VA = np.array([0.5, np.sqrt(3) / 2])   # "yes" state vertex
VB = np.array([0.0, 0.0])              # "no" state 1 vertex
VC = np.array([1.0, 0.0])              # "no" state 2 vertex


def bary_to_cart(ba, bb, bc):
    x = ba * VA[0] + bb * VB[0] + bc * VC[0]
    y = ba * VA[1] + bb * VB[1] + bc * VC[1]
    return x, y


def threshold_line(t):
    """
    Cartesian endpoints of the horizontal line b_A = t inside the triangle.
    At height y = t * √3/2, the line runs from x = t/2  to  x = 1 − t/2.
    """
    y  = t * VA[1]
    x0 = t * VA[0]          # left  endpoint (b_B = 1−t, b_C = 0)
    x1 = t * VA[0] + (1-t)  # right endpoint (b_B = 0,   b_C = 1−t)
    return np.array([x0, x1]), np.array([y, y])


def yes_triangle(t):
    """Vertices of the 'yes' sub-triangle above b_A = t."""
    return np.array([VA,
                     [t * VA[0],           t * VA[1]],
                     [t * VA[0] + (1 - t), t * VA[1]]])


def no_trapezoid(t):
    """Vertices of the 'no' trapezoid below b_A = t."""
    return np.array([[t * VA[0],           t * VA[1]],
                     [t * VA[0] + (1 - t), t * VA[1]],
                     VC, VB])


# ---------------------------------------------------------------------------
# Dirichlet density on a dense simplex grid
# ---------------------------------------------------------------------------
N_GRID = 320

def simplex_density(alpha):
    """
    Returns (x, y, pdf) arrays for plotting the Dirichlet density inside
    the triangle using tricontourf.
    """
    eps  = 1e-3
    ba_v = np.linspace(eps, 1 - eps, N_GRID)
    bb_v = np.linspace(eps, 1 - eps, N_GRID)
    BA, BB = np.meshgrid(ba_v, bb_v)
    BC     = 1.0 - BA - BB
    valid  = BC > eps

    ba = BA[valid]; bb = BB[valid]; bc = BC[valid]
    x, y = bary_to_cart(ba, bb, bc)

    pts     = np.stack([ba, bb, bc], axis=0)   # [3, N]
    pdf_val = dirichlet.pdf(pts, alpha)

    return x, y, pdf_val


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
h_n   = 2.0 * float(scipy_beta.median(1, 2)) - 1.0   # ≈ −0.414
th_0  = 0.5                                            # b_A threshold for h=0
th_n  = (h_n + 1) / 2                                 # b_A threshold for h_n  ≈ 0.293

panels = [
    dict(alpha=[1, 1, 1], threshold=th_0,
         title=f"Uniform  |  h = 0",
         note=f"P(yes) = 0.25\n(biased ↓)",
         note_color="#c0392b"),
    dict(alpha=[1, 1, 1], threshold=th_n,
         title=f"Uniform  |  hₙ ≈ {h_n:.2f}",
         note=f"P(yes) = 0.50\n(corrected)",
         note_color="#27ae60"),
    dict(alpha=[6, 1, 1], threshold=th_n,
         title=f"Peaked on A  |  hₙ",
         note=None, note_color=None),
    dict(alpha=[1, 5, 5], threshold=th_n,
         title=f"Peaked on B/C  |  hₙ",
         note=None, note_color=None),
]

C_YES = "#2a6fb5"   # blue
C_NO  = "#e74c3c"   # red
ALPHA_SHADE = 0.18

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
fig.subplots_adjust(wspace=0.08)

for ax, panel in zip(axes, panels):
    alpha     = panel["alpha"]
    threshold = panel["threshold"]

    # -- Dirichlet density --------------------------------------------------
    x, y, pdf = simplex_density(alpha)
    triang     = mtri.Triangulation(x, y)
    lev        = np.linspace(0, pdf.max(), 28)[1:]
    ax.tricontourf(triang, pdf, levels=lev, cmap="YlOrRd", zorder=1)

    # -- Shaded yes / no regions --------------------------------------------
    yes_verts = yes_triangle(threshold)
    no_verts  = no_trapezoid(threshold)
    ax.fill(yes_verts[:, 0], yes_verts[:, 1],
            color=C_YES, alpha=ALPHA_SHADE, zorder=2, lw=0)
    ax.fill(no_verts[:, 0],  no_verts[:, 1],
            color=C_NO,  alpha=ALPHA_SHADE, zorder=2, lw=0)

    # -- Triangle border ----------------------------------------------------
    tri_patch = mpatches.Polygon(
        np.array([VA, VB, VC]), closed=True,
        fill=False, edgecolor="black", lw=1.4, zorder=5,
    )
    ax.add_patch(tri_patch)

    # -- Threshold line -----------------------------------------------------
    lx, ly = threshold_line(threshold)
    ax.plot(lx, ly, color="white", lw=2.2, zorder=6)
    ax.plot(lx, ly, color="black", lw=1.2, ls="--", zorder=7)

    # -- P(yes) annotation --------------------------------------------------
    a_yes = alpha[0]
    a_no  = alpha[1] + alpha[2]
    p_yes = 1.0 - float(scipy_beta.cdf(threshold, a_yes, a_no))

    # place p_yes inside yes region (above line)
    y_yes_mid = (threshold + 1.0) / 2.0 * VA[1]
    ax.text(0.5, y_yes_mid, f"P(yes)\n= {p_yes:.2f}",
            ha="center", va="center", fontsize=9,
            color=C_YES, fontweight="bold", zorder=8)

    # place p_no inside no region (below line)
    y_no_mid = threshold / 2.0 * VA[1]
    ax.text(0.5, y_no_mid, f"P(no)\n= {1-p_yes:.2f}",
            ha="center", va="center", fontsize=9,
            color=C_NO, fontweight="bold", zorder=8)

    # -- Vertex labels ------------------------------------------------------
    offset = 0.07
    ax.text(VA[0], VA[1] + offset, "A\n(yes)", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C_YES)
    ax.text(VB[0] - offset, VB[1] - offset * 0.5, "B\n(no)", ha="center",
            va="top", fontsize=9, color=C_NO)
    ax.text(VC[0] + offset, VC[1] - offset * 0.5, "C\n(no)", ha="center",
            va="top", fontsize=9, color=C_NO)

    # -- α annotation -------------------------------------------------------
    ax.text(0.5, -0.12, f"α = ({', '.join(str(a) for a in alpha)})",
            ha="center", va="top", fontsize=8, color="gray",
            transform=ax.transData)

    # -- Optional note (bias / correction) ----------------------------------
    if panel["note"]:
        ax.text(0.5, 0.02, panel["note"],
                ha="center", va="bottom", fontsize=8.5,
                color=panel["note_color"], fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=panel["note_color"], alpha=0.85, lw=1))

    ax.set_title(panel["title"], fontsize=9.5, pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.22, 1.08)
    ax.axis("off")

# -- Column annotations: bias vs corrected ----------------------------------
axes[0].annotate("", xy=(1.01, 0.5), xytext=(0.99, 0.5),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

fig.text(0.26, 0.97, "← same uniform belief, different threshold →",
         ha="center", va="top", fontsize=8, color="gray", style="italic")
fig.text(0.74, 0.97, "← same threshold hₙ, different learned α →",
         ha="center", va="top", fontsize=8, color="gray", style="italic")

fig.suptitle(
    "K = 3 simplex: why the decision threshold matters\n"
    "Δu = (+1, −1, −1)   →   P(yes) = P(b·Δu > h) = P(b_A > (h+1)/2)",
    fontsize=10, y=1.06,
)

sns.despine(fig=fig, left=True, bottom=True)
plt.tight_layout()

out = OUT_DIR / "simplex_threshold_intuition.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
