"""
run1/01_fit/direction_accuracy.py
----------------------------------
Computes direction accuracy for threshold-corrected (h_n) predictions
on joint generalisation (val tasks × probe images).

"Direction correct" means sign(pred − 0.5) == sign(emp − 0.5):
  e.g. emp=0.60, pred=0.74  → both > 0.5  → correct
       emp=0.38, pred=0.44  → both < 0.5  → correct
       emp=0.60, pred=0.41  → mismatch    → incorrect

Results are broken down by arity.

Run from repo root:
    python experiments/behavior/run1/01_fit/direction_accuracy.py [--tag TAG]
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import beta as scipy_beta
import torch
from torch.distributions import Dirichlet

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import load_image_refs
from dlbt.data.task import get_task
from dlbt.constants import K

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Filter results pkl by tag substring.")
args = parser.parse_args()

N_MC = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _h(task_name: str, k: int = K) -> float:
    n       = _arity(task_name)
    k_plus  = k // (2 ** n)
    k_minus = k - k_plus
    return 2.0 * scipy_beta.median(k_plus, k_minus) - 1.0


def _direction_acc(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of pairs where sign(pred-0.5) == sign(true-0.5)."""
    return float(np.mean(np.sign(pred - 0.5) == np.sign(true - 0.5)))


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
candidates = sorted(cfg.RESULTS_DIR.glob("results_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
if not candidates:
    raise FileNotFoundError(f"No results_*.pkl found in {cfg.RESULTS_DIR}.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
refs_dict = load_image_refs(cfg.METADATA)

# Pre-load frozen CLIP features
print("Loading CLIP feature cache...")
_cache_agent = DlbtAgent(freeze_encoder=True, n_mc_samples=1, device=device,
                         mapper_hidden=cfg.MAPPER_HIDDEN)
cache_path = Path(cfg.CACHE_PATH)
if cache_path.exists():
    _cache_agent.load_cache(str(cache_path))
else:
    _cache_agent.precompute_features(list(refs_dict.values()))
frozen_clip = {uid: feat.clone() for uid, feat in _cache_agent._cache.items()}
del _cache_agent

# ---------------------------------------------------------------------------
# Process each results file
# ---------------------------------------------------------------------------
for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    val_tasks  = sorted(res.get("val_tasks", cfg.VAL_TASKS), key=lambda t: (_arity(t), t))
    test_uids  = res.get("test_uids", set())
    dlbt_orig  = res.get("dlbt", {})
    print(f"  val tasks : {len(val_tasks)}")
    print(f"  test images: {len(test_uids)}")

    # Load agent
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
        agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    else:
        feat_cache_path = cfg.RESULTS_DIR / f"features_{run_tag}.pt"
        if feat_cache_path.exists():
            print(f"  Loading cached attnpool features from {feat_cache_path.name}...")
            agent.load_cache(str(feat_cache_path))
        else:
            from tqdm import tqdm as _tqdm
            all_refs_list = list(refs_dict.values())
            print("  Precomputing backbone + attnpool features...")
            agent.precompute_backbone_features(all_refs_list)
            with torch.no_grad():
                for i in _tqdm(range(0, len(all_refs_list), 16), desc="  caching", unit="batch"):
                    batch   = all_refs_list[i : i + 16]
                    spatial = torch.stack(
                        [agent._backbone_cache[r.uid] for r in batch]
                    ).to(agent.device)
                    feats = agent.encoder.attnpool(spatial).float()
                    for ref, feat in zip(batch, feats):
                        agent._cache[ref.uid] = feat.cpu()
            agent.save_cache(str(feat_cache_path))
        agent.freeze_encoder = True

    # -----------------------------------------------------------------------
    # Inference — h=0 and h_n for all val tasks × test images
    # -----------------------------------------------------------------------
    # Collect per-arity results
    results_h0 = defaultdict(lambda: {"pred": [], "true": []})
    results_hn = defaultdict(lambda: {"pred": [], "true": []})

    for task_name in val_tasks:
        h_val  = _h(task_name)
        n_way  = _arity(task_name)
        task   = get_task(task_name)
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=device)

        orig = dlbt_orig.get("joint", {}).get(task_name)
        if orig is None:
            continue

        orig_uids  = orig.get("uids", [])
        orig_true  = orig["true"]
        orig_tots  = orig["totals"]

        mask = np.array([u in test_uids for u in orig_uids])
        if not mask.any():
            continue

        batch_refs = [refs_dict[orig_uids[i]] for i in np.where(mask)[0]]
        true_p     = orig_true[mask]
        totals     = orig_tots[mask]

        valid = totals > 0
        if not valid.any():
            continue

        batch_refs = [b for b, v in zip(batch_refs, valid) if v]
        true_p     = true_p[valid]

        with torch.no_grad():
            alpha = agent.get_alpha(batch_refs).clamp(min=0.1)   # [B, K]
            b     = Dirichlet(alpha).sample((N_MC,))              # [N, B, K]
            logit = torch.einsum("nbk,k->nb", b, delta_u)        # [N, B]

            p_h0 = (logit > 0.0  ).float().mean(dim=0).cpu().numpy()
            p_hn = (logit > h_val).float().mean(dim=0).cpu().numpy()

        results_h0[n_way]["pred"].append(p_h0)
        results_h0[n_way]["true"].append(true_p)
        results_hn[n_way]["pred"].append(p_hn)
        results_hn[n_way]["true"].append(true_p)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n  Direction accuracy (sign agreement with 0.5 as chance):")
    print(f"  {'arity':<8}  {'n pairs':>8}  {'h=0':>8}  {'h_n':>8}")
    print(f"  {'-'*40}")

    all_pred_h0, all_true_h0 = [], []
    all_pred_hn, all_true_hn = [], []

    for n_way in sorted(set(list(results_h0.keys()) + list(results_hn.keys()))):
        if not results_h0[n_way]["pred"]:
            continue
        pred_h0 = np.concatenate(results_h0[n_way]["pred"])
        true_h0 = np.concatenate(results_h0[n_way]["true"])
        pred_hn = np.concatenate(results_hn[n_way]["pred"])
        true_hn = np.concatenate(results_hn[n_way]["true"])

        acc_h0 = _direction_acc(pred_h0, true_h0)
        acc_hn = _direction_acc(pred_hn, true_hn)
        n_pairs = len(pred_h0)

        print(f"  {n_way}-way    {n_pairs:>8}  {acc_h0:>7.1%}  {acc_hn:>7.1%}")

        all_pred_h0.append(pred_h0); all_true_h0.append(true_h0)
        all_pred_hn.append(pred_hn); all_true_hn.append(true_hn)

    if all_pred_h0:
        pred_h0_all = np.concatenate(all_pred_h0)
        true_h0_all = np.concatenate(all_true_h0)
        pred_hn_all = np.concatenate(all_pred_hn)
        true_hn_all = np.concatenate(all_true_hn)
        print(f"  {'-'*40}")
        print(f"  {'overall':<8}  {len(pred_h0_all):>8}  "
              f"{_direction_acc(pred_h0_all, true_h0_all):>7.1%}  "
              f"{_direction_acc(pred_hn_all, true_hn_all):>7.1%}")

print("\nDone.")
