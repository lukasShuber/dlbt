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
sample      = rng.choice(len(refs_cached), size=min(256, len(refs_cached)), replace=False)
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
# Per-image α heatmap for a small subset (sanity check)
# ---------------------------------------------------------------------------
n_show = 16
idx    = np.argsort(-ratio)[:n_show]      # most peaked images first
fig2, ax2 = plt.subplots(figsize=(10, 5))
im = ax2.imshow(alpha[idx], aspect="auto", cmap="viridis")
ax2.set_xlabel("latent state k (0..15)")
ax2.set_ylabel("image (top = most peaked)")
ax2.set_title(f"α per image — top-{n_show} by peak-to-mean ratio")
fig2.colorbar(im, ax=ax2, label="α_k")
fig2.tight_layout()

out_path2 = cfg.RESULTS_DIR / f"diagnose_alpha_heatmap_{cfg.RUN_TAG}.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path2}")

# ---------------------------------------------------------------------------
# Full-sample heatmap: every image, sorted by peak state then by ratio
# ---------------------------------------------------------------------------
# Sort images first by their argmax latent state, then by peak-to-mean ratio
# within each group.  This groups images by which state they "route to" and
# makes any mode collapse visually obvious.
argmax_state = alpha.argmax(axis=1)                              # [N]
sort_key     = argmax_state * 1e6 - ratio                        # secondary: high ratio first
sort_idx     = np.argsort(sort_key)

fig3, ax3 = plt.subplots(figsize=(10, 9))
im3 = ax3.imshow(alpha[sort_idx], aspect="auto", cmap="viridis")
ax3.set_xlabel("latent state k (0..15)")
ax3.set_ylabel(f"image (sorted by argmax state, N={len(sort_idx)})")
ax3.set_title(f"α per image — all {len(sort_idx)} sampled images")
fig3.colorbar(im3, ax=ax3, label="α_k")

# Mark the group boundaries (where argmax changes) with horizontal lines
sorted_argmax = argmax_state[sort_idx]
boundaries    = np.where(np.diff(sorted_argmax) != 0)[0] + 1
for b in boundaries:
    ax3.axhline(b - 0.5, color="white", lw=0.5, alpha=0.6)

# Count images per argmax state for a quick textual summary
counts_per_state = np.bincount(argmax_state, minlength=K)
print("\nImages per argmax state:")
for k in range(K):
    bar = "#" * int(40 * counts_per_state[k] / max(counts_per_state.max(), 1))
    print(f"  state {k:2d}: {counts_per_state[k]:4d}  {bar}")

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
