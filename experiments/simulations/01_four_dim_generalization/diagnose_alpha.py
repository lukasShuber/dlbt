"""
diagnose_alpha.py
-----------------
Inspect the peakedness of learned Dirichlet α on training images, to test
the hypothesis that the mapper never escapes the near-uniform regime.

Metrics (per image, K=16 Dirichlet over latent states):
    S        = Σ α_k            (total concentration / "sharpness")
    max_α    = max_k α_k        (largest component)
    ratio    = max_α / mean(α)  (relative peak vs. uniform baseline)
    N_eff    = exp(H(α/S))      (effective support size; K=uniform, 1=delta)

Uniform baseline reference:
    α=(1.4,...,1.4) → S=22.4, max_α=1.4, ratio=1.0, N_eff=16 (fully diffuse)
A well-peaked posterior:
    one α_k ≈ 50, others ≈ 1 → S≈65, ratio≈12.3, N_eff≈2-3

Usage:
    cd <repo root>
    python experiments/behavior/run0/diagnose_alpha.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import load_image_refs, image_refs_as_list

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent_path  = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt"
cache_path  = Path(cfg.CACHE_PATH)

print(f"Loading agent from {agent_path}")
print(f"Device: {device}")

# Instantiate agent with the SAME config as training
agent = DlbtAgent(
    freeze_encoder = cfg.FREEZE_ENCODER,
    n_mc_samples   = cfg.N_MC,
    device         = device,
    mapper_hidden  = cfg.MAPPER_HIDDEN,
    feature_dim    = 1024,
)
state = torch.load(agent_path, map_location=device)
agent.load_state_dict(state)
agent.eval()

# Load CLIP cache so get_alpha can hit precomputed features
if cache_path.exists():
    agent.load_cache(str(cache_path))
    print(f"Loaded CLIP feature cache ({len(agent._cache)} entries)")
else:
    raise FileNotFoundError(f"No feature cache at {cache_path} — run run.py first.")

# ---------------------------------------------------------------------------
# Pick a sample of images (training + probe) to inspect
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs_list = image_refs_as_list(refs_dict)

# Use only images that are in the CLIP cache
refs_cached = [r for r in refs_list if r.uid in agent._cache]
rng         = np.random.default_rng(cfg.SEED)
sample      = rng.choice(len(refs_cached), size=min(100, len(refs_cached)), replace=False)
sample_refs = [refs_cached[i] for i in sample]

print(f"Sampled {len(sample_refs)} images for α inspection")

# ---------------------------------------------------------------------------
# Forward pass → α
# ---------------------------------------------------------------------------
with torch.no_grad():
    alpha = agent.get_alpha(sample_refs).detach().cpu().numpy()  # [N, K]
print(f"α shape: {alpha.shape},  min={alpha.min():.3f}  max={alpha.max():.3f}")

# ---------------------------------------------------------------------------
# Peakedness metrics
# ---------------------------------------------------------------------------
K         = alpha.shape[1]
S         = alpha.sum(axis=1)                                    # [N]
max_alpha = alpha.max(axis=1)                                    # [N]
mean_a    = alpha.mean(axis=1)                                   # [N]
ratio     = max_alpha / mean_a                                   # [N]  (1.0 = uniform, K = delta)
p         = alpha / S[:, None]                                   # [N, K]
entropy   = -(p * np.log(p + 1e-12)).sum(axis=1)                 # [N]
n_eff     = np.exp(entropy)                                      # [N]  (K = uniform, 1 = delta)

print("\n========== α peakedness summary ==========")
def stat(label, x):
    print(f"  {label:<12s}  mean={x.mean():7.3f}  median={np.median(x):7.3f}  "
          f"min={x.min():7.3f}  max={x.max():7.3f}")
stat("S   (sum α)", S)
stat("max α",      max_alpha)
stat("ratio",      ratio)
stat("N_eff",      n_eff)

print("\nReference points:")
print(f"  uniform α=(1.4,...,1.4):  S=22.4  max=1.4  ratio=1.0   N_eff=16")
print(f"  peaked  α=(50, 1*K-1):    S≈65    max=50   ratio≈12.3  N_eff≈2.0")

# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].hist(max_alpha, bins=30, color="#457B9D", alpha=0.85)
axes[0, 0].axvline(1.4, color="gray", ls="--", label="uniform baseline")
axes[0, 0].set_xlabel("max α_k");  axes[0, 0].set_ylabel("#images")
axes[0, 0].set_title("Peak α per image");  axes[0, 0].legend()

axes[0, 1].hist(S, bins=30, color="#E76F51", alpha=0.85)
axes[0, 1].axvline(22.4, color="gray", ls="--", label="uniform baseline")
axes[0, 1].set_xlabel("Σ α  (concentration)");  axes[0, 1].set_ylabel("#images")
axes[0, 1].set_title("Dirichlet concentration per image");  axes[0, 1].legend()

axes[1, 0].hist(ratio, bins=30, color="#9B5DE5", alpha=0.85)
axes[1, 0].axvline(1.0, color="gray", ls="--", label="uniform")
axes[1, 0].axvline(K,   color="green", ls="--", label="delta (K=16)")
axes[1, 0].set_xlabel("max α / mean α");  axes[1, 0].set_ylabel("#images")
axes[1, 0].set_title("Peak-to-mean ratio");  axes[1, 0].legend()

axes[1, 1].hist(n_eff, bins=30, color="#43AA8B", alpha=0.85)
axes[1, 1].axvline(K,   color="gray",  ls="--", label="uniform (K=16)")
axes[1, 1].axvline(1.0, color="green", ls="--", label="delta (=1)")
axes[1, 1].set_xlabel("effective support size  exp(H(α/S))")
axes[1, 1].set_ylabel("#images")
axes[1, 1].set_title("Effective support size");  axes[1, 1].legend()

fig.suptitle(f"Learned α peakedness — {cfg.RUN_TAG}  (N={len(sample_refs)} images)",
             y=1.00, fontsize=12)
fig.tight_layout()

out_path = cfg.RESULTS_DIR / f"diagnose_alpha_{cfg.RUN_TAG}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
THUMB = 64   # thumbnail edge length in pixels
ZOOM  = 0.3  # OffsetImage zoom — maintains 1:1 ratio regardless of axes size


def load_thumb(ref):
    try:
        img = PILImage.open(ref.path).convert("RGB").resize((THUMB, THUMB))
        return np.array(img)
    except Exception:
        return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)


def add_thumbs(ax, refs, n_rows, zoom=ZOOM):
    """Place square thumbnails down the y-axis of ax (data coords 0..n_rows-1)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    for i, ref in enumerate(refs):
        thumb = load_thumb(ref)
        oi = OffsetImage(thumb, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(oi, (0.5, i), xycoords="data",
                            frameon=False, pad=0,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


# Verbal labels for each K=16 state.
# Bit encoding: bit3=lr (R=1), bit2=tr (Tr=1), bit1=gl (Gl=1), bit0=sl (Lg=1)
def _state_label(k):
    return (
        ("R"  if (k >> 3) & 1 else "L")  + "\n" +
        ("Tr" if (k >> 2) & 1 else "Op") + "\n" +
        ("Gl" if (k >> 1) & 1 else "Mt") + "\n" +
        ("Lg" if  k & 1       else "Sm")
    )

STATE_LABELS = [_state_label(k) for k in range(K)]


# ---------------------------------------------------------------------------
# Per-image α heatmap — top-N most peaked images
# ---------------------------------------------------------------------------
n_show   = 16
idx      = np.argsort(-ratio)[:n_show]   # most peaked images first
top_refs = [sample_refs[i] for i in idx]

fig_h2 = max(5, n_show * 0.35)
fig2   = plt.figure(figsize=(13, fig_h2))
gs2    = gridspec.GridSpec(1, 3, figure=fig2,
                           width_ratios=[1.2, 10, 0.4], wspace=0.02)
ax2_thumb = fig2.add_subplot(gs2[0])
ax2_heat  = fig2.add_subplot(gs2[1])
ax2_cbar  = fig2.add_subplot(gs2[2])

add_thumbs(ax2_thumb, top_refs, n_show, zoom=0.8)
ax2_thumb.set_title("image", fontsize=8)
ax2_thumb.set_ylabel("image (top = most peaked)")

im2 = ax2_heat.imshow(alpha[idx], aspect="auto", cmap="viridis",
                      extent=[-0.5, K - 0.5, n_show - 0.5, -0.5])
ax2_heat.set_xticks(range(K))
ax2_heat.set_xticklabels(STATE_LABELS, rotation=90, fontsize=7, va="top")
ax2_heat.set_yticks([])
ax2_heat.set_ylim(n_show - 0.5, -0.5)
ax2_heat.set_title(f"α per image — top-{n_show} by peak-to-mean ratio  [{cfg.RUN_TAG}]")
ax2_heat.set_xlabel("latent state", labelpad=4)
fig2.colorbar(im2, cax=ax2_cbar, label="α_k")

fig2.tight_layout()
out_path2 = cfg.RESULTS_DIR / f"diagnose_alpha_heatmap_{cfg.RUN_TAG}.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path2}")

# ---------------------------------------------------------------------------
# Full-sample heatmap: every image, sorted by peak state then by ratio
# ---------------------------------------------------------------------------
argmax_state = alpha.argmax(axis=1)                              # [N]
sort_key     = argmax_state * 1e6 - ratio                        # secondary: high ratio first
sort_idx     = np.argsort(sort_key)
N            = len(sort_idx)

sorted_argmax = argmax_state[sort_idx]
boundaries    = np.where(np.diff(sorted_argmax) != 0)[0] + 1

# Count images per argmax state for a quick textual summary
counts_per_state = np.bincount(argmax_state, minlength=K)
print("\nImages per argmax state:")
for k in range(K):
    bar = "#" * int(40 * counts_per_state[k] / max(counts_per_state.max(), 1))
    print(f"  state {k:2d}: {counts_per_state[k]:4d}  {bar}")

sorted_refs = [sample_refs[i] for i in sort_idx]

fig_h = max(9, N * 0.12)
fig3  = plt.figure(figsize=(14, fig_h))
gs    = gridspec.GridSpec(1, 3, figure=fig3,
                          width_ratios=[1.2, 10, 0.4], wspace=0.02)
ax_thumb = fig3.add_subplot(gs[0])
ax_heat  = fig3.add_subplot(gs[1])
ax_cbar  = fig3.add_subplot(gs[2])

add_thumbs(ax_thumb, sorted_refs, N)
ax_thumb.set_title("image", fontsize=8)
ax_thumb.set_ylabel(f"image  (N={N}, sorted by argmax state → ratio)")

im3 = ax_heat.imshow(
    alpha[sort_idx], aspect="auto", cmap="viridis",
    extent=[-0.5, K - 0.5, N - 0.5, -0.5],
)
ax_heat.set_xticks(range(K))
ax_heat.set_xticklabels(STATE_LABELS, rotation=90, fontsize=7, va="top")
ax_heat.set_yticks([])
ax_heat.set_ylim(N - 0.5, -0.5)
ax_heat.set_title(f"α per image — all {N} sampled images  [{cfg.RUN_TAG}]")
ax_heat.set_xlabel("latent state", labelpad=4)

for b in boundaries:
    ax_heat.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)

fig3.colorbar(im3, cax=ax_cbar, label="α_k")
fig3.tight_layout()
out_path3 = cfg.RESULTS_DIR / f"diagnose_alpha_heatmap_all_{cfg.RUN_TAG}.png"
fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path3}")

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
print("\n========== interpretation ==========")
median_ratio = float(np.median(ratio))
median_neff  = float(np.median(n_eff))

if median_ratio < 2.0:
    print(f"  ★ VERY DIFFUSE  (median ratio = {median_ratio:.2f})")
    print("    The mapper is still near the uniform-α regime.")
    print("    Predictions will be compressed toward ∑[δu=+1]/K for any task.")
elif median_ratio < 5.0:
    print(f"  ★ MODERATELY PEAKED  (median ratio = {median_ratio:.2f})")
    print("    α is distinguishing some dimensions but not sharply.")
    print("    Predictions reach [0.1, 0.8] ish; extremes rare.")
else:
    print(f"  ★ SHARPLY PEAKED  (median ratio = {median_ratio:.2f})")
    print("    α concentrates well on specific latent states.")
    print("    Predictions should span [0, 1] for any task.")

print(f"  median effective support size: {median_neff:.2f} / {K}")
print()
