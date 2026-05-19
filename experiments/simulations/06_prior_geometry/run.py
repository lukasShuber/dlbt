"""
experiments/simulations/06_prior_geometry/run.py

Visualises how the geometry of a K=3 Dirichlet belief prior shapes binary
choice predictions under three SEU correction levels:

  Column 1 — Standard      : SEU = b · Δu         (threshold b₀ = 0.5)
  Column 2 — Prior-norm    : SEU = b · Δu_norm     (threshold b₀ = 1/K = 1/3)
  Column 2 — Prior-norm    : SEU = b · Δu_norm     (threshold b₀ = 1/K = 1/3)

Three Dirichlet belief distributions (rows):
  Row 1 — Neutral      Dir(1, 1, 1)   — flat prior, no information
  Row 2 — Concentrated Dir(5, 5, 5)   — same centre, less uncertainty
  Row 3 — Peaked       Dir(5, 1, 1)   — shifted toward state 0 (positive)

Each cell contains:
  Top    : K=3 simplex heatmap (viridis Dirichlet density + red/blue region overlay)
  Bottom : SEU density (KDE) with mean (green ▲) and median (orange ●) on x-axis

Run from repo root:
    python experiments/simulations/06_prior_geometry/run.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm as _cm
from scipy.special import gammaln
from scipy.stats import beta as _beta, gaussian_kde
import seaborn as sns

# ── repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# K=3 binary-choice task
# ─────────────────────────────────────────────────────────────────────────────
# State 0 = positive (action=right, delta_u = +1)
# State 1, 2 = negative (action=left, delta_u = -1)
#
# Raw delta_u:         [+1, -1, -1]       (arity=1; m=1 positive, n=2 negative)
# Normalised delta_u:  [+1/m, -1/n, -1/n] = [+1, -0.5, -0.5]

_DU_RAW  = np.array([+1.,  -1.,   -1.  ])
_DU_NORM = np.array([+1.,  -0.5,  -0.5 ])   # /n_pos for pos, /n_neg for neg

# ─────────────────────────────────────────────────────────────────────────────
# Correction stages  (label, delta_u, offset)
# ─────────────────────────────────────────────────────────────────────────────

CORRECTIONS = [
    ("Standard",   _DU_RAW,  0.0),
    ("Prior-norm", _DU_NORM, 0.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet distributions  (label, alpha)
# ─────────────────────────────────────────────────────────────────────────────
_POINT_MASS = None   # sentinel: point mass at centroid b = [1/3, 1/3, 1/3]
_B_POINT    = np.array([1./3, 1./3, 1./3])

DISTS = [
    (r"Dir(⅓,⅓,⅓)",            np.array([1./10, 1./10, 1./10])),
    (r"Dir(1,1,1)",            np.array([1.,   1.,   1.  ])),
    (r"Dir(5,5,5)",            np.array([5.,   5.,   5.  ])),
    (r"Dir(10,10,10)",         np.array([10.,  10.,  10. ])),
    (r"Dir(100,100,100)",      np.array([100., 100., 100.])),
    (r"Point mass $b=\frac{1}{3}\mathbf{1}$", _POINT_MASS),
]

# ─────────────────────────────────────────────────────────────────────────────
# Equilateral triangle vertices (K=3 simplex)
# ─────────────────────────────────────────────────────────────────────────────
# V0 = apex       (b₀ large → state 0, positive)
# V1 = bottom-left (b₁ large → state 1, negative)
# V2 = bottom-right(b₂ large → state 2, negative)
_s3 = np.sqrt(3) / 2           # triangle height  ≈ 0.866
_V  = np.array([
    [0.5,  _s3],   # V0  top
    [0.0,  0.0 ],  # V1  bottom-left
    [1.0,  0.0 ],  # V2  bottom-right
])

# ─────────────────────────────────────────────────────────────────────────────
# Pixel grid over the simplex
# ─────────────────────────────────────────────────────────────────────────────
_RES = 350          # pixels per axis
_gx  = np.linspace(0.0, 1.0, _RES)
_gy  = np.linspace(0.0, _s3,  _RES)
_GX, _GY = np.meshgrid(_gx, _gy)    # [H, W]

# Barycentric coordinates: P = b0·V0 + b1·V1 + b2·V2
#   x = 0.5·b0 + b2    y = s3·b0
#   ⇒  b0 = y/s3,  b2 = x − 0.5·b0,  b1 = 1 − b0 − b2
_b0_pix = _GY / _s3
_b2_pix = _GX - 0.5 * _b0_pix
_b1_pix = 1.0 - _b0_pix - _b2_pix

_EPS    = 1e-9
_INSIDE = (_b0_pix > _EPS) & (_b1_pix > _EPS) & (_b2_pix > _EPS)
_B_pix  = np.stack([_b0_pix, _b1_pix, _b2_pix], axis=-1)  # [H, W, 3]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dir_logpdf(b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Dirichlet log-pdf at grid points b[..., 3] with concentration alpha[3]."""
    log_norm = gammaln(alpha.sum()) - gammaln(alpha).sum()
    return log_norm + ((alpha - 1.0) * np.log(np.clip(b, 1e-30, None))).sum(-1)


def _seu_pix(du: np.ndarray, offset: float) -> np.ndarray:
    """SEU at every pixel: [H, W]."""
    return (_B_pix * du).sum(-1) - offset


def _boundary_xy(du: np.ndarray, offset: float):
    """
    Return ((x_left, y), (x_right, y)) of the SEU=0 line inside the triangle.

    For symmetric du (du[1] == du[2]):
      SEU = (du[0] − du[1])·b₀ + du[1] = offset
      ⇒   b₀ = (offset − du[1]) / (du[0] − du[1])

    At b₀=thresh the triangle spans x ∈ [0.5·thresh, 1 − 0.5·thresh].
    """
    # Works only when du[1] == du[2] (symmetric task)
    denom = du[0] - du[1]
    if abs(denom) < 1e-12:
        return None, None
    thresh = (offset - du[1]) / denom
    if not (0.0 <= thresh <= 1.0):
        return None, None
    y        = thresh * _s3
    x_left   = 0.5 * thresh             # b₂ = 0, b₁ = 1 − thresh
    x_right  = 1.0 - 0.5 * thresh       # b₁ = 0, b₂ = 1 − thresh
    return (x_left, y), (x_right, y)



def _p_yes_analytic(alpha: np.ndarray, du: np.ndarray, offset: float) -> float:
    """
    P(choose right) = P(SEU > 0) = P(b₀ > threshold).

    For symmetric du (du[1]==du[2]):
      threshold = (offset − du[1]) / (du[0] − du[1])
    b₀ ~ Beta(alpha[0], alpha[1:].sum())
    """
    denom  = du[0] - du[1]
    thresh = (offset - du[1]) / denom if abs(denom) > 1e-12 else 0.5
    a0     = float(alpha[0])
    a_rest = float(alpha[1:].sum())
    return float(_beta.sf(thresh, a0, a_rest))


def _log_pdf_centroid(alpha_dir: np.ndarray) -> float:
    """Log Dirichlet pdf at the centroid b = (1/K, …, 1/K)."""
    K = len(alpha_dir)
    return (float(gammaln(alpha_dir.sum()) - gammaln(alpha_dir).sum())
            + float(((alpha_dir - 1.0) * np.log(1.0 / K)).sum()))


_DENSITY_GAMMA = 0.30   # < 1 stretches low-density values so concentrated
                         # distributions show a visibly smaller bright region

def _density_rgba(alpha_dir: np.ndarray) -> np.ndarray:
    """
    Viridis-coloured Dirichlet density, shared scale across rows.

    Each pixel's display value = (pdf / pdf_centroid)^GAMMA, clipped to [0,1].

    - pdf / pdf_centroid = exp(log_pdf - log_pdf_centroid) ∈ [0, 1]
    - The gamma < 1 stretches the low-density tail so that different
      concentrations produce visibly different bright-region sizes:
        Dir(1,1,1)       → uniform 1.0 everywhere (all pixels at centroid level)
        Dir(5,5,5)       → broad gradient, bright region covers most of triangle
        Dir(10,10,10)    → narrower bright region
        Dir(100,100,100) → tiny bright dot, almost all pixels near 0
    """
    log_pdf = _dir_logpdf(_B_pix, alpha_dir)           # [H, W]
    lpc     = _log_pdf_centroid(alpha_dir)              # scalar

    # log(pdf/pdf_centroid) ≤ 0 everywhere; clip underflow
    log_ratio = np.clip(log_pdf - lpc, -700.0, 0.0)    # [H, W]
    ratio     = np.where(_INSIDE, np.exp(log_ratio), 0.0)          # ∈ [0,1]
    display   = np.power(np.clip(ratio, 0.0, 1.0), _DENSITY_GAMMA) # gamma lift

    rgba = _cm.viridis(display)
    rgba[..., 3] = np.where(_INSIDE, 1.0, 0.0)
    return rgba


def _density_rgba_sub(alpha_dir: np.ndarray) -> np.ndarray:
    """
    Density image for sub-uniform Dirichlet (all α < 1).

    Here mass concentrates near the *vertices*, so the density is highest at
    the corners and lowest at the centroid — the opposite of the α > 1 case.
    We use standard per-distribution min/max normalisation so the corner peaks
    appear bright (yellow) and the centre appears dark (purple).
    """
    log_pdf = _dir_logpdf(_B_pix, alpha_dir)
    lpdf_in = log_pdf[_INSIDE]
    lmin, lmax = float(lpdf_in.min()), float(lpdf_in.max())
    if lmax > lmin:
        dens = np.where(_INSIDE, (log_pdf - lmin) / (lmax - lmin), 0.0)
    else:
        dens = np.where(_INSIDE, 0.5, 0.0)
    rgba = _cm.viridis(np.clip(dens, 0.0, 1.0))
    rgba[..., 3] = np.where(_INSIDE, 1.0, 0.0)
    return rgba


def _density_rgba_point() -> np.ndarray:
    """
    Simplex image for a point mass at the centroid b = (1/3, 1/3, 1/3).

    Centroid in Cartesian: x = 0.5·b0 + b2 = 0.5,  y = s3·b0 = s3/3.
    Rendered as a hard yellow dot (viridis max) — no smooth falloff.
    """
    # Centroid position in data coords
    x_c = 0.5 * _B_POINT[0] + _B_POINT[2]   # = 0.5
    y_c = _s3  * _B_POINT[0]                 # = _s3/3

    radius  = 0.028                           # hard circle radius (data units)
    dist    = np.sqrt((_GX - x_c)**2 + (_GY - y_c)**2)
    display = np.where(_INSIDE & (dist <= radius), 1.0, 0.0)

    rgba = _cm.viridis(display)
    rgba[..., 3] = np.where(_INSIDE, 1.0, 0.0)
    return rgba


def _region_rgba(du: np.ndarray, offset: float) -> np.ndarray:
    """
    Semi-transparent red/blue overlay indicating the yes/no decision region.
    Alpha = 0.35 inside the relevant region, 0 elsewhere.
    """
    seu   = _seu_pix(du, offset)        # [H, W]
    H, W  = _GX.shape
    rgba  = np.zeros((H, W, 4), dtype=float)

    m_yes = _INSIDE & (seu > 0)
    rgba[m_yes] = [0.75, 0.12, 0.12, 0.38]   # red, semi-transparent

    m_no  = _INSIDE & (seu <= 0)
    rgba[m_no]  = [0.12, 0.28, 0.78, 0.38]   # blue, semi-transparent

    return rgba


# Monte-Carlo SEU samples
_N_SAMP = 300_000
_RNG    = np.random.default_rng(42)


def _seu_samples(alpha_dir: np.ndarray, du: np.ndarray, offset: float
                 ) -> np.ndarray:
    """Draw N_SAMP SEU values from Dir(alpha_dir)."""
    b = _RNG.dirichlet(alpha_dir, size=_N_SAMP)  # [N, 3]
    return (b * du).sum(1) - offset


# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
_C_YES    = "#C0392B"    # red
_C_NO     = "#2a6fb5"    # blue
_C_MEAN   = "#27AE60"    # green
_C_MEDIAN = "#E67E22"    # orange


# ─────────────────────────────────────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────────────────────────────────────
n_d = len(DISTS)
n_c = len(CORRECTIONS)

# 2 grid-rows per distribution (simplex + CDF), 3 grid-cols
fig = plt.figure(figsize=(11.5, 22.5))
hr  = [2.6, 0.95] * n_d          # height-ratios: tall simplex, short CDF
gs  = gridspec.GridSpec(
    2 * n_d, n_c,
    figure       = fig,
    height_ratios= hr,
    hspace       = 0.14,          # a little breathing room between simplex and density
    wspace       = 0.22,
)

# ── Column headers ────────────────────────────────────────────────────────────
for j, lbl in enumerate(["Standard", "Prior-normalised"]):
    # Place text above the top of the top grid row in column j
    # We use a temporary invisible axis to anchor the coordinate
    _ax_hdr = fig.add_subplot(gs[0, j])
    _ax_hdr.set_visible(False)
    pos = _ax_hdr.get_position()
    fig.text(
        pos.x0 + pos.width / 2, pos.y1 + 0.008,
        lbl, ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        transform=fig.transFigure,
    )
del _ax_hdr, pos

# ── Main grid ─────────────────────────────────────────────────────────────────
for row_i, (dist_name, alpha_dir) in enumerate(DISTS):
    for col_j, (corr_lbl, du, offset_raw) in enumerate(CORRECTIONS):

        offset = float(offset_raw)

        ax_tri = fig.add_subplot(gs[2 * row_i,     col_j])
        ax_cdf = fig.add_subplot(gs[2 * row_i + 1, col_j])

        _point_mass = (alpha_dir is _POINT_MASS)

        # ── Simplex heatmap ───────────────────────────────────────────────────
        # Layer 1: viridis Dirichlet density (or point-mass blob)
        if _point_mass:
            _img = _density_rgba_point()
        elif float(alpha_dir[0]) < 1.0:
            _img = _density_rgba_sub(alpha_dir)
        else:
            _img = _density_rgba(alpha_dir)
        ax_tri.imshow(_img, origin="lower", extent=[0, 1, 0, _s3],
                      aspect="equal", interpolation="bilinear")
        # Triangle border
        tri_patch = plt.Polygon(_V, fill=False, edgecolor="black", lw=1.2)
        ax_tri.add_patch(tri_patch)

        # Decision boundary line
        pt_l, pt_r = _boundary_xy(du, offset)
        if pt_l is not None:
            ax_tri.plot(
                [pt_l[0], pt_r[0]], [pt_l[1], pt_r[1]],
                color="black", lw=1.4, zorder=5,
            )

        # Vertex labels
        _off = 0.055
        ax_tri.text(_V[0, 0],       _V[0, 1] + _off, r"$Z_0^+$",
                    ha="center", va="bottom", fontsize=8.5)
        ax_tri.text(_V[1, 0] - _off, _V[1, 1],       r"$Z_1^-$",
                    ha="right",  va="center", fontsize=8.5)
        ax_tri.text(_V[2, 0] + _off, _V[2, 1],       r"$Z_2^-$",
                    ha="left",   va="center", fontsize=8.5)

        # P(yes) analytic
        if _point_mass:
            seu_val = float((_B_POINT * du).sum()) - offset
            py = 1.0 if seu_val > 1e-9 else (0.5 if abs(seu_val) < 1e-9 else 0.0)
        else:
            py = _p_yes_analytic(alpha_dir, du, offset)

        ax_tri.text(
            0.5, -0.04,
            f"P(yes) = {py:.3f}   P(no) = {1 - py:.3f}",
            ha="center", va="top", fontsize=7.5,
            transform=ax_tri.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
        )

        ax_tri.set_xlim(-0.16, 1.16)
        ax_tri.set_ylim(-0.08, _s3 + 0.18)
        ax_tri.axis("off")

        # Row label on leftmost column
        if col_j == 0:
            ax_tri.text(
                -0.14, 0.5, dist_name,
                ha="right", va="center", fontsize=9.5,
                transform=ax_tri.transAxes, style="italic",
            )

        # ── SEU density (KDE or point-mass spike) ────────────────────────────
        ax_cdf.axvline(0.0, color="black", lw=0.9, ls="--", zorder=4)

        if _point_mass:
            # Degenerate distribution: single spike at seu_val
            spike_color = _C_YES if seu_val > 1e-9 else (
                          _C_NO  if seu_val < -1e-9 else "#666666")
            ax_cdf.annotate(
                "", xy=(seu_val, 0.85), xytext=(seu_val, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=spike_color, lw=2.0),
                zorder=5,
            )
            ax_cdf.plot(seu_val, 0.0, "^", color=_C_MEAN, ms=7,
                        clip_on=False, zorder=6)
            ax_cdf.set_xlim(-1.1, 1.1)
            ax_cdf.set_ylim(0, 1)
        else:
            seu     = _seu_samples(alpha_dir, du, offset)
            kde     = gaussian_kde(seu, bw_method="silverman")
            x_lo    = float(seu.min()); x_hi = float(seu.max())
            x_pad   = (x_hi - x_lo) * 0.05
            xs      = np.linspace(x_lo - x_pad, x_hi + x_pad, 600)
            ys      = kde(xs)

            ax_cdf.fill_between(xs, 0, ys, where=(xs > 0),
                                color=_C_YES, alpha=0.40, zorder=1)
            ax_cdf.fill_between(xs, 0, ys, where=(xs <= 0),
                                color=_C_NO,  alpha=0.40, zorder=1)
            ax_cdf.plot(xs, ys, color="#222222", lw=1.4, zorder=3)

            s_mean   = float(seu.mean())
            s_median = float(np.median(seu))
            ax_cdf.plot(s_mean,   0.0, "^", color=_C_MEAN,   ms=7,
                        clip_on=False, zorder=6)
            ax_cdf.plot(s_median, 0.0, "o", color=_C_MEDIAN, ms=6,
                        clip_on=False, zorder=6)
            py_mc = float((seu > 0).mean())
            ax_cdf.text(0.97, 0.92, f"P(yes)={py_mc:.3f}",
                        ha="right", va="top", fontsize=7.5,
                        color=_C_YES, transform=ax_cdf.transAxes)
            ax_cdf.text(0.03, 0.92, f"P(no)={1 - py_mc:.3f}",
                        ha="left",  va="top", fontsize=7.5,
                        color=_C_NO,  transform=ax_cdf.transAxes)
            ax_cdf.set_ylim(bottom=0)

        ax_cdf.set_xlabel("SEU", fontsize=8)
        if col_j == 0:
            ax_cdf.set_ylabel("Density", fontsize=8)
        ax_cdf.tick_params(labelsize=7)
        ax_cdf.set_yticks([])
        sns.despine(ax=ax_cdf, top=True, right=True, left=True)

# ── Global title ──────────────────────────────────────────────────────────────
fig.suptitle(
    r"Prior geometry: $K=3$ simplex  ×  SEU correction stage",
    fontsize=13, y=1.002,
)

out = PLOTS_DIR / "fig1_prior_geometry.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print(f"{'Distribution':<28}  {'Correction':<16}  {'P(yes) analytic':>16}  "
      f"{'P(yes) MC':>10}  {'mean SEU':>9}  {'median SEU':>10}")
print("─" * 98)

_RNG2 = np.random.default_rng(0)
for dist_name, alpha_dir in DISTS:
    _pt = (alpha_dir is _POINT_MASS)
    for corr_lbl, du, offset_raw in CORRECTIONS:
        offset = float(offset_raw)
        lbl    = corr_lbl
        if _pt:
            sv    = float((_B_POINT * du).sum()) - offset
            py_an = 1.0 if sv > 1e-9 else (0.5 if abs(sv) < 1e-9 else 0.0)
            py_mc = py_an
        else:
            py_an = _p_yes_analytic(alpha_dir, du, offset)
            seu   = (_RNG2.dirichlet(alpha_dir, size=200_000) * du).sum(1) - offset
            py_mc = float((seu > 0).mean())
        dname = dist_name.replace("\n", " ").replace("$", "").replace("\\", "")
        if _pt:
            mean_seu = median_seu = float((_B_POINT * du).sum()) - offset
        else:
            mean_seu   = float(seu.mean())
            median_seu = float(np.median(seu))
        print(f"  {dname:<36}  {lbl:<22}  {py_an:>16.4f}  "
              f"{py_mc:>10.4f}  {mean_seu:>9.4f}  {median_seu:>10.4f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — P(Yes) vs concentration for symmetric Dir(α,α,α)
# ─────────────────────────────────────────────────────────────────────────────

def _make_concentration_figure() -> None:
    """
    Plot P(Yes) as a function of concentration α for symmetric Dir(α,α,α),
    for each of the three SEU correction types.

    For symmetric Dir(α,α,α) the marginal of b0 is Beta(α, 2α):
      Standard  : threshold b0 = 0.5  → P(Yes) = Beta(α,2α).sf(0.5)
      Prior-norm: threshold b0 = 1/3  → P(Yes) = Beta(α,2α).sf(1/3)
    """
    # Continuous sweep: α from 0.1 to 10 000 on log scale
    alphas = np.logspace(-1, 4, 800)

    p_std   = np.array([_beta.sf(0.5,   a, 2.0 * a) for a in alphas])
    p_pnorm = np.array([_beta.sf(1./3., a, 2.0 * a) for a in alphas])

    # Discrete marks the user asked about
    marks        = [1, 10, 100, 1_000, 10_000]
    m_std   = np.array([_beta.sf(0.5,   a, 2.0 * a) for a in marks])
    m_pnorm = np.array([_beta.sf(1./3., a, 2.0 * a) for a in marks])

    # Colors: one per correction (same palette order as column headers)
    C_STD   = "#2a6fb5"   # blue   — Standard
    C_PNO   = "#E67E22"   # orange — Prior trick

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Reference line
    ax.axhline(0.5, color="#bbbbbb", lw=1.0, ls=":", zorder=1)

    # Traces
    ax.plot(alphas, p_std,   color=C_STD, lw=2.0, label="Standard",   zorder=3)
    ax.plot(alphas, p_pnorm, color=C_PNO, lw=2.0, label="Prior trick", zorder=3)

    # Discrete markers (open circles)
    ax.plot(marks, m_std,   "o", color=C_STD, ms=6, mfc="none", mew=1.5, zorder=4)
    ax.plot(marks, m_pnorm, "o", color=C_PNO, ms=6, mfc="none", mew=1.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.15, 15_000)
    ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.set_xticklabels([r"$1$", r"$10$", r"$10^2$", r"$10^3$", r"$10^4$"])
    ax.set_xlabel(r"Concentration $\alpha$  (symmetric $\mathrm{Dir}(\alpha,\alpha,\alpha)$)",
                  fontsize=11)
    ax.set_ylabel(r"$P(\mathrm{Yes})$", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.legend(fontsize=9, frameon=False, loc="center right")
    sns.despine(top=True, right=True)
    plt.tight_layout()

    out = PLOTS_DIR / "p_yes_vs_concentration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


_make_concentration_figure()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — P(Yes) vs concentration, K=16 ontology, one panel per task arity
# ─────────────────────────────────────────────────────────────────────────────

def _make_concentration_k16_figure() -> None:
    """
    Single panel: P(Yes) vs concentration α for K=16 symmetric Dir(α,...,α),
    all four task arities overlaid.

    Color  = arity  (1→blue, 2→teal, 3→orange, 4→purple)
    Solid  = Standard   (threshold b_yes > 0.5)
    Dashed = Prior trick (threshold b_yes > n_pos/K = mean of Beta)
    """
    K16    = 16
    marks  = [1, 10, 100, 1_000, 10_000]
    alphas = np.logspace(-1, 4, 800)

    ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
    arities     = [1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    ax.axhline(0.5, color="#bbbbbb", lw=1.0, ls=":", zorder=1)

    for a in arities:
        n_pos = K16 // (2 ** a)   # 8, 4, 2, 1
        n_neg = K16 - n_pos
        c     = ARITY_COLOR[a]

        thresh_std   = 0.5
        thresh_pnorm = n_pos / K16   # = mean of Beta(n_pos·α, n_neg·α)

        p_std   = np.array([_beta.sf(thresh_std,   a_ * n_pos, a_ * n_neg) for a_ in alphas])
        p_pnorm = np.array([_beta.sf(thresh_pnorm, a_ * n_pos, a_ * n_neg) for a_ in alphas])

        m_std   = np.array([_beta.sf(thresh_std,   m * n_pos, m * n_neg) for m in marks])
        m_pnorm = np.array([_beta.sf(thresh_pnorm, m * n_pos, m * n_neg) for m in marks])

        # Traces
        ax.plot(alphas, p_std,   color=c, lw=2.0, ls="--", zorder=3)
        ax.plot(alphas, p_pnorm, color=c, lw=2.0, ls="-",  zorder=3)

        # Discrete markers: circle = Prior trick, square = Standard
        ax.plot(marks, m_std,   "s", color=c, ms=5, mfc="none", mew=1.4, zorder=4)
        ax.plot(marks, m_pnorm, "o", color=c, ms=5, mfc="none", mew=1.4, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.15, 15_000)
    ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.set_xticklabels([r"$1$", r"$10$", r"$10^2$", r"$10^3$", r"$10^4$"])
    ax.set_xlabel(r"Concentration $\alpha$  (symmetric $\mathrm{Dir}(\alpha,\ldots,\alpha)$, $K=16$)",
                  fontsize=11)
    ax.set_ylabel(r"$P(\mathrm{Yes})$", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # ── Legend: two sections ──
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    arity_handles = [
        mpatches.Patch(color=ARITY_COLOR[a], label=f"Arity {a}")
        for a in arities
    ]
    style_handles = [
        mlines.Line2D([], [], color="black", lw=2.0, ls="--", label="Standard"),
        mlines.Line2D([], [], color="black", lw=2.0, ls="-",  label="Prior trick"),
    ]
    ax.legend(handles=arity_handles + style_handles,
              fontsize=8, frameon=False, loc="upper right",
              ncol=1)

    sns.despine(top=True, right=True)
    plt.tight_layout()

    out = PLOTS_DIR / "p_yes_vs_concentration_k16.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


_make_concentration_k16_figure()
