"""
diagnose_alpha.py
-----------------
Inspect the peakedness of learned Dirichlet α on training images, to test
the hypothesis that the mapper never escapes the near-uniform regime.

Runs diagnostics for both the best-val agent (agent_<tag>.pt) and
the end-of-training agent (agent_<tag>_end.pt).

Metrics (per image, K=16 Dirichlet over latent states):
    S        = Σ α_k            (total concentration / "sharpness")
    max_α    = max_k α_k        (largest component)
    ratio    = max_α / mean(α)  (relative peak vs. uniform baseline)
    N_eff    = exp(H(α/S))      (effective support size; K=uniform, 1=delta)

Usage:
    cd <repo root>
    python experiments/simulations/011_emp_sim/diagnose_alpha.py
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
# Setup — shared across both agents
# ---------------------------------------------------------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_path = Path(cfg.CACHE_PATH)
print(f"Device: {device}")

refs_dict   = load_image_refs(cfg.METADATA)
refs_list   = image_refs_as_list(refs_dict)

THUMB = 64
ZOOM  = 0.3

def load_thumb(ref):
    try:
        img = PILImage.open(ref.path).convert("RGB").resize((THUMB, THUMB))
        return np.array(img)
    except Exception:
        return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)

def add_thumbs(ax, refs, n_rows, zoom=ZOOM):
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
                            frameon=False, pad=0, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

def _state_label(k):
    K_bits = int(np.log2(16))
    return (
        ("R"  if (k >> 3) & 1 else "L")  + "\n" +
        ("Tr" if (k >> 2) & 1 else "Op") + "\n" +
        ("Gl" if (k >> 1) & 1 else "Mt") + "\n" +
        ("Lg" if  k & 1       else "Sm")
    )

# ---------------------------------------------------------------------------
# Main diagnostic function — runs on one agent checkpoint
# ---------------------------------------------------------------------------
def run_diagnostics(agent_path: Path, out_tag: str):
    if not agent_path.exists():
        print(f"\n[SKIP] {agent_path.name} not found — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Loading agent from {agent_path}")

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

    if cache_path.exists():
        agent.load_cache(str(cache_path))
        print(f"Loaded CLIP feature cache ({len(agent._cache)} entries)")
    else:
        raise FileNotFoundError(f"No feature cache at {cache_path} — run run.py first.")

    refs_cached = [r for r in refs_list if r.uid in agent._cache]
    rng         = np.random.default_rng(cfg.SEED)
    sample      = rng.choice(len(refs_cached), size=min(100, len(refs_cached)), replace=False)
    sample_refs = [refs_cached[i] for i in sample]
    print(f"Sampled {len(sample_refs)} images for α inspection")

    with torch.no_grad():
        alpha = agent.get_alpha(sample_refs).detach().cpu().numpy()
    print(f"α shape: {alpha.shape},  min={alpha.min():.3f}  max={alpha.max():.3f}")

    K         = alpha.shape[1]
    S         = alpha.sum(axis=1)
    max_alpha = alpha.max(axis=1)
    mean_a    = alpha.mean(axis=1)
    ratio     = max_alpha / mean_a
    p         = alpha / S[:, None]
    entropy   = -(p * np.log(p + 1e-12)).sum(axis=1)
    n_eff     = np.exp(entropy)

    STATE_LABELS = [_state_label(k) for k in range(K)]

    print(f"\n========== α peakedness summary  [{out_tag}] ==========")
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

    # -- Distribution histograms ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(max_alpha, bins=30, color="#457B9D", alpha=0.85)
    axes[0, 0].axvline(1.4, color="gray", ls="--", label="uniform baseline")
    axes[0, 0].set_xlabel("max α_k"); axes[0, 0].set_ylabel("#images")
    axes[0, 0].set_title("Peak α per image"); axes[0, 0].legend()

    axes[0, 1].hist(S, bins=30, color="#E76F51", alpha=0.85)
    axes[0, 1].axvline(22.4, color="gray", ls="--", label="uniform baseline")
    axes[0, 1].set_xlabel("Σ α  (concentration)"); axes[0, 1].set_ylabel("#images")
    axes[0, 1].set_title("Dirichlet concentration per image"); axes[0, 1].legend()

    axes[1, 0].hist(ratio, bins=30, color="#9B5DE5", alpha=0.85)
    axes[1, 0].axvline(1.0, color="gray", ls="--", label="uniform")
    axes[1, 0].axvline(K,   color="green", ls="--", label="delta (K=16)")
    axes[1, 0].set_xlabel("max α / mean α"); axes[1, 0].set_ylabel("#images")
    axes[1, 0].set_title("Peak-to-mean ratio"); axes[1, 0].legend()

    axes[1, 1].hist(n_eff, bins=30, color="#43AA8B", alpha=0.85)
    axes[1, 1].axvline(K,   color="gray",  ls="--", label="uniform (K=16)")
    axes[1, 1].axvline(1.0, color="green", ls="--", label="delta (=1)")
    axes[1, 1].set_xlabel("effective support size  exp(H(α/S))")
    axes[1, 1].set_ylabel("#images")
    axes[1, 1].set_title("Effective support size"); axes[1, 1].legend()

    fig.suptitle(f"Learned α peakedness — {out_tag}  (N={len(sample_refs)} images)",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    out1 = cfg.RESULTS_DIR / f"diagnose_alpha_{out_tag}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out1}")

    # -- Top-N peaked heatmap ------------------------------------------------
    n_show   = 16
    idx      = np.argsort(-ratio)[:n_show]
    top_refs = [sample_refs[i] for i in idx]

    fig2  = plt.figure(figsize=(13, max(5, n_show * 0.35)))
    gs2   = gridspec.GridSpec(1, 3, figure=fig2, width_ratios=[1.2, 10, 0.4], wspace=0.02)
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
    ax2_heat.set_title(f"α per image — top-{n_show} by peak-to-mean ratio  [{out_tag}]")
    ax2_heat.set_xlabel("latent state", labelpad=4)
    fig2.colorbar(im2, cax=ax2_cbar, label="α_k")
    fig2.tight_layout()
    out2 = cfg.RESULTS_DIR / f"diagnose_alpha_heatmap_{out_tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")

    # -- Full-sample heatmap (all images, sorted by argmax state) ------------
    argmax_state     = alpha.argmax(axis=1)
    sort_key         = argmax_state * 1e6 - ratio
    sort_idx         = np.argsort(sort_key)
    N                = len(sort_idx)
    sorted_argmax    = argmax_state[sort_idx]
    boundaries       = np.where(np.diff(sorted_argmax) != 0)[0] + 1
    counts_per_state = np.bincount(argmax_state, minlength=K)
    sorted_refs      = [sample_refs[i] for i in sort_idx]

    print("\nImages per argmax state:")
    for k in range(K):
        bar = "#" * int(40 * counts_per_state[k] / max(counts_per_state.max(), 1))
        print(f"  state {k:2d}: {counts_per_state[k]:4d}  {bar}")

    fig3 = plt.figure(figsize=(14, max(9, N * 0.12)))
    gs3  = gridspec.GridSpec(1, 3, figure=fig3, width_ratios=[1.2, 10, 0.4], wspace=0.02)
    ax_thumb = fig3.add_subplot(gs3[0])
    ax_heat  = fig3.add_subplot(gs3[1])
    ax_cbar  = fig3.add_subplot(gs3[2])

    add_thumbs(ax_thumb, sorted_refs, N)
    ax_thumb.set_title("image", fontsize=8)
    ax_thumb.set_ylabel(f"image  (N={N}, sorted by argmax state → ratio)")

    im3 = ax_heat.imshow(alpha[sort_idx], aspect="auto", cmap="viridis",
                         extent=[-0.5, K - 0.5, N - 0.5, -0.5])
    ax_heat.set_xticks(range(K))
    ax_heat.set_xticklabels(STATE_LABELS, rotation=90, fontsize=7, va="top")
    ax_heat.set_yticks([])
    ax_heat.set_ylim(N - 0.5, -0.5)
    ax_heat.set_title(f"α per image — all {N} sampled images  [{out_tag}]")
    ax_heat.set_xlabel("latent state", labelpad=4)
    for b in boundaries:
        ax_heat.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)
    fig3.colorbar(im3, cax=ax_cbar, label="α_k")
    fig3.tight_layout()
    out3 = cfg.RESULTS_DIR / f"diagnose_alpha_heatmap_all_{out_tag}.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved → {out3}")

    # -- Interpretation ------------------------------------------------------
    print(f"\n========== interpretation  [{out_tag}] ==========")
    median_ratio = float(np.median(ratio))
    median_neff  = float(np.median(n_eff))
    if median_ratio < 2.0:
        print(f"  ★ VERY DIFFUSE  (median ratio = {median_ratio:.2f})")
        print("    The mapper is still near the uniform-α regime.")
    elif median_ratio < 5.0:
        print(f"  ★ MODERATELY PEAKED  (median ratio = {median_ratio:.2f})")
        print("    α is distinguishing some dimensions but not sharply.")
        print("    Predictions reach [0.1, 0.8] ish; extremes rare.")
    else:
        print(f"  ★ SHARPLY PEAKED  (median ratio = {median_ratio:.2f})")
        print("    α concentrates well on specific latent states.")
        print("    Predictions should span [0, 1] for any task.")
    print(f"  median effective support size: {median_neff:.2f} / {K}")


# ---------------------------------------------------------------------------
# Run for best agent and end agent
# ---------------------------------------------------------------------------
run_diagnostics(
    agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}.pt",
    out_tag    = cfg.RUN_TAG,
)

run_diagnostics(
    agent_path = cfg.RESULTS_DIR / f"agent_{cfg.RUN_TAG}_end.pt",
    out_tag    = f"{cfg.RUN_TAG}_end",
)
