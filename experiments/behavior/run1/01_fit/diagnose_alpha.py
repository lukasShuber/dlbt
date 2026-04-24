"""
run1/01_fit/diagnose_alpha.py
------------------------------
Inspect the peakedness of learned Dirichlet α on probe images, to test
whether the mapper has learned to concentrate probability mass on the
correct latent states.

Metrics (per image, K=16 Dirichlet over latent states):
    S        = Σ α_k            (total concentration / "sharpness")
    max_α    = max_k α_k        (largest component)
    ratio    = max_α / mean(α)  (relative peak vs. uniform baseline)
    N_eff    = exp(H(α/S))      (effective support size; K=uniform, 1=delta)

Auto-discovers all agent_*.pt files in results/ (both best-val and end-of-
training) so it runs for every available RUN_TAG (arity, random, etc.).

Run from repo root:
    python experiments/behavior/run1/01_fit/diagnose_alpha.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.constants import X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_path = Path(cfg.CACHE_PATH)
print(f"Device: {device}")

refs_dict = load_image_refs(cfg.METADATA)
refs_list = image_refs_as_list(refs_dict)

# ---------------------------------------------------------------------------
# Continuous metadata → ground-truth latent state per image
# ---------------------------------------------------------------------------
_cont_meta: dict = {}
with open(cfg.METADATA) as _f:
    for _line in _f:
        _line = _line.strip()
        if not _line:
            continue
        _rec = json.loads(_line)
        _cont_meta[_rec["id"]] = _rec["z"]


def gt_state_idx(uid: str) -> int:
    z  = _cont_meta.get(uid, {})
    lr = int(z.get("pos_xy",      [0])[0] > X_THRESHOLD)
    tr = int(z.get("transparency", 0)     > TRANSP_THRESH)
    gl = int(z.get("glossiness",   0)     > GLOSS_THRESH)
    sl = int(z.get("scale",        0)     > SCALE_THRESH)
    return lr * 8 + tr * 4 + gl * 2 + sl


def gt_state_label(uid: str) -> str:
    z  = _cont_meta.get(uid, {})
    lr = "R"  if z.get("pos_xy",      [0])[0] > X_THRESHOLD  else "L"
    tr = "Tr" if z.get("transparency", 0)     > TRANSP_THRESH else "Op"
    gl = "Gl" if z.get("glossiness",   0)     > GLOSS_THRESH  else "Mt"
    sl = "Lg" if z.get("scale",        0)     > SCALE_THRESH  else "Sm"
    return f"{lr} {tr} {gl} {sl}"


def _state_label(k: int) -> str:
    return (
        ("R"  if (k >> 3) & 1 else "L")  + "\n" +
        ("Tr" if (k >> 2) & 1 else "Op") + "\n" +
        ("Gl" if (k >> 1) & 1 else "Mt") + "\n" +
        ("Lg" if  k & 1       else "Sm")
    )


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------
THUMB = 128


def load_thumb(ref):
    try:
        img = PILImage.open(ref.path).convert("RGB").resize(
            (THUMB, THUMB), PILImage.LANCZOS
        )
        return np.array(img)
    except Exception:
        return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)


def add_thumbs(ax, refs, n_rows, zoom=0.18):
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    for i, ref in enumerate(refs):
        thumb = load_thumb(ref)
        oi    = OffsetImage(thumb, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(oi, (0.5, i), xycoords="data",
                            frameon=False, pad=0, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Probe UID set — from combined run0 + run1 data
# ---------------------------------------------------------------------------
print("\nLoading behavioural data to identify probe images...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
df_filtered, _ = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)

_eligible_names  = set(cfg.TRAIN_TASKS + cfg.VAL_TASKS)
_eligible_beh_id = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in _eligible_names}

_, probe_uids, _ = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _eligible_beh_id,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)

probe_uid_set  = set(probe_uids)
probe_refs_all = [r for r in refs_list if r.uid in probe_uid_set]
print(f"Probe images: {len(probe_refs_all)}")

# ---------------------------------------------------------------------------
# CLIP feature cache (frozen backbone)
# ---------------------------------------------------------------------------
print("Loading CLIP feature cache...")
_cache_agent = DlbtAgent(freeze_encoder=True, n_mc_samples=1, device=device,
                         mapper_hidden=cfg.MAPPER_HIDDEN)
if cache_path.exists():
    _cache_agent.load_cache(str(cache_path))
else:
    raise FileNotFoundError(f"No feature cache at {cache_path} — run run.py first.")
frozen_clip = {uid: feat.clone() for uid, feat in _cache_agent._cache.items()}
del _cache_agent

# ---------------------------------------------------------------------------
# Diagnostic function
# ---------------------------------------------------------------------------
def run_diagnostics(agent_path: Path, out_tag: str):
    if not agent_path.exists():
        print(f"\n[SKIP] {agent_path.name} not found.")
        return

    print(f"\n{'='*60}")
    print(f"Loading agent: {agent_path.name}")

    agent = DlbtAgent(
        freeze_encoder = cfg.FREEZE_ENCODER,
        n_mc_samples   = cfg.N_MC,
        device         = device,
        mapper_hidden  = cfg.MAPPER_HIDDEN,
    )
    state = torch.load(agent_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()

    # Populate feature cache
    if cfg.FREEZE_ENCODER:
        agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    else:
        feat_cache_path = cfg.RESULTS_DIR / f"features_{out_tag}.pt"
        if feat_cache_path.exists():
            print(f"  Loading cached attnpool features from {feat_cache_path.name}...")
            agent.load_cache(str(feat_cache_path))
        else:
            from tqdm import tqdm as _tqdm
            all_refs_list = list(refs_dict.values())
            agent.precompute_backbone_features(all_refs_list)
            with torch.no_grad():
                for i in _tqdm(range(0, len(all_refs_list), 16),
                               desc="  caching attnpool feats", unit="batch"):
                    batch   = all_refs_list[i : i + 16]
                    spatial = torch.stack(
                        [agent._backbone_cache[r.uid] for r in batch]
                    ).to(agent.device)
                    feats = agent.encoder.attnpool(spatial).float()
                    for ref, feat in zip(batch, feats):
                        agent._cache[ref.uid] = feat.cpu()
            agent.save_cache(str(feat_cache_path))
            print(f"  Saved attnpool feature cache -> {feat_cache_path.name}")

    # Probe images available in cache, sorted by ground-truth state
    probe_refs = sorted(
        [r for r in probe_refs_all if r.uid in agent._cache],
        key=lambda r: gt_state_idx(r.uid),
    )
    if not probe_refs:
        print("  No probe images found in cache — skipping.")
        return
    print(f"  Probe images in cache: {len(probe_refs)}")

    with torch.no_grad():
        alpha = agent.get_alpha(probe_refs).detach().cpu().numpy()

    K            = alpha.shape[1]
    STATE_LABELS = [_state_label(k) for k in range(K)]
    gt_states    = [gt_state_idx(r.uid)  for r in probe_refs]
    row_labels   = [gt_state_label(r.uid) for r in probe_refs]

    S         = alpha.sum(axis=1)
    max_alpha = alpha.max(axis=1)
    ratio     = max_alpha / alpha.mean(axis=1)
    p_norm    = alpha / S[:, None]
    n_eff     = np.exp(-(p_norm * np.log(p_norm + 1e-12)).sum(axis=1))

    print(f"\n  α peakedness summary [{out_tag}]")
    for label, x in [("S   (sum α)", S), ("max α", max_alpha),
                     ("ratio", ratio), ("N_eff", n_eff)]:
        print(f"    {label:<12s}  mean={x.mean():7.3f}  median={np.median(x):7.3f}  "
              f"min={x.min():7.3f}  max={x.max():7.3f}")

    # -- Heatmap -------------------------------------------------------------
    n_rows = len(probe_refs)
    fig  = plt.figure(figsize=(14, max(5, n_rows * 0.55)))
    gs   = gridspec.GridSpec(1, 3, figure=fig,
                             width_ratios=[1.2, 10, 0.4], wspace=0.18)
    ax_thumb = fig.add_subplot(gs[0])
    ax_heat  = fig.add_subplot(gs[1])
    ax_cbar  = fig.add_subplot(gs[2])

    add_thumbs(ax_thumb, probe_refs, n_rows, zoom=0.18)
    ax_thumb.set_title("image", fontsize=8)
    ax_thumb.set_ylabel("probe image  (sorted by ground-truth state)")

    im = ax_heat.imshow(alpha, aspect="auto", cmap="YlOrRd",
                        extent=[-0.5, K - 0.5, n_rows - 0.5, -0.5])
    ax_heat.set_xticks(range(K))
    ax_heat.set_xticklabels(STATE_LABELS, rotation=90, fontsize=7, va="top")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=7)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)
    ax_heat.set_xlabel("latent state", labelpad=4)
    ax_heat.set_title(
        f"Learned α — {n_rows} probe images  [{out_tag}]", fontsize=9
    )

    # Blue border on ground-truth state column per row
    for row_i, k in enumerate(gt_states):
        ax_heat.add_patch(plt.Rectangle(
            (k - 0.5, row_i - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=2, zorder=5,
        ))

    fig.colorbar(im, cax=ax_cbar, label="α_k")
    fig.tight_layout()

    out_path = cfg.RESULTS_DIR / f"diagnose_alpha_probe_{out_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Auto-discover all agent checkpoints in RESULTS_DIR
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Collect best-val agents first, then _end variants
best_agents = sorted(cfg.RESULTS_DIR.glob("agent_*.pt"))
best_agents = [p for p in best_agents if not p.stem.endswith("_end")]

for agent_path in best_agents:
    tag = agent_path.stem[len("agent_"):]   # e.g. "attnpool_arity"
    run_diagnostics(agent_path, out_tag=tag)
    # Also run end-of-training variant if it exists
    end_path = agent_path.parent / f"agent_{tag}_end.pt"
    run_diagnostics(end_path, out_tag=f"{tag}_end")

print("\nDone.")
