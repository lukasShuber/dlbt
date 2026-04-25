"""
run1/01_fit/alpha_correction_check.py
--------------------------------------
Sanity check: does correcting α instead of the threshold give the same result?

Two approaches that should agree for uninformative α and be close for learned α:

  hn           — original α, threshold h_n  (MC)
  alpha_scaled — yes-state α scaled by s = K₋/K₊ per task, threshold h = 0  (MC)

For uninformative (uniform) α both approaches restore P(right) = 0.5 exactly.
For the learned α they need not be identical in general — this script tests how
close they are and whether conclusions about the bias correction hold under both.

Run from repo root:
    python experiments/behavior/run1/01_fit/alpha_correction_check.py [--tag TAG]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import beta as scipy_beta, spearmanr
import torch
from torch.distributions import Dirichlet

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import load_image_refs
from dlbt.data.task import get_task
from dlbt.constants import K

# ---------------------------------------------------------------------------
N_MC    = 2000
REGION  = "joint"

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _label(task_name: str) -> str:
    return task_name.replace("_and_", " & ").replace("_", "/")


def _h_n(task_name: str, k: int = K) -> float:
    n       = _arity(task_name)
    k_plus  = k // (2 ** n)
    k_minus = k - k_plus
    return 2.0 * scipy_beta.median(k_plus, k_minus) - 1.0


def _scale(task_name: str, k: int = K) -> float:
    """Multiplicative factor for yes-state α so uninformative α → P=0.5 at h=0."""
    n       = _arity(task_name)
    k_plus  = k // (2 ** n)
    k_minus = k - k_plus
    return k_minus / k_plus     # s = K₋/K₊


def _noise_floor(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1))) \
        if mask.any() else 0.0


def _cmse_nf(pred, true, totals):
    raw = float(np.mean((pred - true) ** 2))
    raw -= float(np.mean(pred * (1 - pred))) / (N_MC - 1)   # MC variance correction
    return raw - _noise_floor(true, totals)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None)
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("results_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
if not candidates:
    raise FileNotFoundError(f"No results_*.pkl in {cfg.RESULTS_DIR}.")

# ---------------------------------------------------------------------------
# Shared feature cache (frozen CLIP, used for feature lookup only)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

refs_dict = load_image_refs(cfg.METADATA)

print("Loading CLIP feature cache...")
_tmp = DlbtAgent(freeze_encoder=True, n_mc_samples=1, device=device,
                 mapper_hidden=cfg.MAPPER_HIDDEN)
cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    _tmp.load_cache(str(cache_path))
else:
    _tmp.precompute_features(list(refs_dict.values()))
frozen_clip = {uid: f.clone() for uid, f in _tmp._cache.items()}
del _tmp

# ---------------------------------------------------------------------------
# Process each results file
# ---------------------------------------------------------------------------
for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    val_tasks  = sorted(res.get("val_tasks", cfg.VAL_TASKS),
                        key=lambda t: (_arity(t), t))
    test_uids  = res.get("test_uids", set())

    # -----------------------------------------------------------------------
    # Load agent
    # -----------------------------------------------------------------------
    ckpt_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found — skipping.")
        continue

    agent = DlbtAgent(
        freeze_encoder = cfg.FREEZE_ENCODER,
        n_mc_samples   = N_MC,
        device         = device,
        mapper_hidden  = cfg.MAPPER_HIDDEN,
    )
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()

    if cfg.FREEZE_ENCODER:
        agent._cache = {uid: f.clone() for uid, f in frozen_clip.items()}
    else:
        feat_cache_path = cfg.RESULTS_DIR / f"features_{run_tag}.pt"
        if feat_cache_path.exists():
            print(f"  Loading cached attnpool features from {feat_cache_path.name}...")
            agent.load_cache(str(feat_cache_path))
        else:
            from tqdm import tqdm as _tqdm
            all_refs = list(refs_dict.values())
            print("  Precomputing backbone + attnpool features...")
            agent.precompute_backbone_features(all_refs)
            with torch.no_grad():
                for i in _tqdm(range(0, len(all_refs), 16), desc="  caching"):
                    batch   = all_refs[i : i + 16]
                    spatial = torch.stack(
                        [agent._backbone_cache[r.uid] for r in batch]
                    ).to(device)
                    feats = agent.encoder.attnpool(spatial).float()
                    for r, feat in zip(batch, feats):
                        agent._cache[r.uid] = feat.cpu()
            agent.save_cache(str(feat_cache_path))
        agent.freeze_encoder = True   # use _cache for fast lookup

    # -----------------------------------------------------------------------
    # Collect predictions for both approaches
    # -----------------------------------------------------------------------
    dlbt_orig = res.get("dlbt", {})

    preds_hn    = []   # (pred, true, totals, task_name) tuples
    preds_ascl  = []

    for task_name in val_tasks:
        h_val  = _h_n(task_name)
        s      = _scale(task_name)
        task   = get_task(task_name)
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=device)
        yes_mask = np.array(task.delta_u) > 0   # [K] bool

        # Ground-truth from pickle
        d = dlbt_orig.get(REGION, {}).get(task_name)
        if d is None:
            continue
        orig_uids = d.get("uids", [])
        mask      = np.array([u in test_uids for u in orig_uids])
        if not mask.any():
            continue

        batch_refs = [refs_dict[orig_uids[i]] for i in np.where(mask)[0]]
        true_p     = d["true"][mask]
        totals     = d["totals"][mask]

        with torch.no_grad():
            alpha = agent.get_alpha(batch_refs).clamp(min=0.1)   # [B, K]

            # --- h_n approach: original α, shifted threshold ---
            b_hn   = Dirichlet(alpha).sample((N_MC,))             # [N, B, K]
            logit  = torch.einsum("nbk,k->nb", b_hn, delta_u)    # [N, B]
            p_hn   = (logit > h_val).float().mean(0).cpu().numpy()

            # --- α-scaling approach: scale yes-states by K₋/K₊, h=0 ---
            alpha_scaled = alpha.clone()
            alpha_scaled[:, yes_mask] *= s
            alpha_scaled = alpha_scaled.clamp(min=0.1)
            b_scl  = Dirichlet(alpha_scaled).sample((N_MC,))
            logit2 = torch.einsum("nbk,k->nb", b_scl, delta_u)
            p_scl  = (logit2 > 0.0).float().mean(0).cpu().numpy()

        valid = totals > 0
        preds_hn.append((p_hn[valid], true_p[valid], totals[valid], task_name))
        preds_ascl.append((p_scl[valid], true_p[valid], totals[valid], task_name))

    if not preds_hn:
        print("  No data — skipping.")
        continue

    # -----------------------------------------------------------------------
    # Pooled metrics
    # -----------------------------------------------------------------------
    def _pool(preds_list):
        pred   = np.concatenate([x[0] for x in preds_list])
        true   = np.concatenate([x[1] for x in preds_list])
        totals = np.concatenate([x[2] for x in preds_list])
        return pred, true, totals

    pred_hn,  true_all, tots_all = _pool(preds_hn)
    pred_scl, _,        _        = _pool(preds_ascl)

    rho_hn,  _ = spearmanr(pred_hn,  true_all)
    rho_scl, _ = spearmanr(pred_scl, true_all)
    mse_hn     = _cmse_nf(pred_hn,  true_all, tots_all)
    mse_scl    = _cmse_nf(pred_scl, true_all, tots_all)

    print(f"\n  Pooled [{REGION}]:")
    print(f"    hₙ (threshold)      ρ={rho_hn:.3f}  cMSE-NF={mse_hn:+.4f}")
    print(f"    α-scaled (h=0)      ρ={rho_scl:.3f}  cMSE-NF={mse_scl:+.4f}")

    # Agreement between the two approaches
    rho_agree, _ = spearmanr(pred_hn, pred_scl)
    mae_agree    = float(np.mean(np.abs(pred_hn - pred_scl)))
    print(f"\n  Agreement between approaches:")
    print(f"    ρ(hₙ, α-scaled) = {rho_agree:.4f}   MAE = {mae_agree:.4f}")

    # -----------------------------------------------------------------------
    # Plot: scatter of hₙ predictions vs α-scaled predictions
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Left: both vs human
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, zorder=0)
    ax.scatter(pred_hn,  true_all, s=12, alpha=0.5, color="#E76F51",
               label=f"hₙ  ρ={rho_hn:.3f}", zorder=2)
    ax.scatter(pred_scl, true_all, s=12, alpha=0.5, color="#9B5DE5",
               label=f"α-scaled  ρ={rho_scl:.3f}", zorder=2)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted P(right)", fontsize=9)
    ax.set_ylabel("Human P(right)", fontsize=9)
    ax.set_title("Both approaches vs human", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax, trim=True)

    # Right: hₙ vs α-scaled (agreement plot)
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, zorder=0)
    ax2.scatter(pred_hn, pred_scl, s=12, alpha=0.5, color="#457B9D", zorder=2)
    ax2.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("hₙ prediction", fontsize=9)
    ax2.set_ylabel("α-scaled prediction", fontsize=9)
    ax2.set_title(f"Agreement  ρ={rho_agree:.4f}  MAE={mae_agree:.4f}", fontsize=9)
    ax2.text(0.05, 0.95, f"s = K₋/K₊ per arity",
             transform=ax2.transAxes, fontsize=7, color="gray", va="top")
    sns.despine(ax=ax2, trim=True)

    fig.suptitle(f"α-correction vs threshold correction  [{run_tag}]", fontsize=10)
    plt.tight_layout()
    out = plots_dir / f"plot_alpha_correction_check_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")
    plt.close()

print("\nDone.")
