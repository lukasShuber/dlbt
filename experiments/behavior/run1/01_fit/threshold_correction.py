"""
run1/01_fit/threshold_correction.py
-------------------------------------
Re-runs MC inference on the saved agent with an arity-adjusted decision
threshold

    h_n = 2 · median( Beta(K₊, K₋) ) − 1

where  K₊ = K / 2ⁿ  and  K₋ = K − K₊  for an n-way task.

For 1-way tasks h_1 = 0 (no correction needed).
For conjunctions h_n < 0, which corrects the systematic bias that causes
a model trained only on 1-way tasks to under-predict P(yes) on 2/3/4-way
tasks.

Regions evaluated (val tasks only):
  task_gen  — val tasks × main (train) images
  joint_gen — val tasks × probe images

Produces four plots saved to results/plots/:
  plot_tau_curves_{tag}.png     — cMSE-NF: original vs corrected (task & joint gen)
  plot_tau_summary_{tag}.png    — pooled scatter: original vs corrected (joint gen)
  plot_tau_pertask_{tag}.png    — per-task scatters, corrected (joint gen)
  plot_tau_pertask_task_{tag}.png — per-task scatters, corrected (task gen)

Run from repo root:
    python experiments/behavior/run1/01_fit/threshold_correction.py [--tag TAG]
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
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
# Helpers
# ---------------------------------------------------------------------------

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

N_MC = 2000   # more samples for cleaner corrected inference

C_ORIG = cfg.C_JOINT     # original predictions colour
C_CORR = "#E76F51"       # threshold-corrected colour
C_TASK = cfg.C_TASK

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}


def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _label(task_name: str) -> str:
    return task_name.replace("_and_", " & ").replace("_", "/")


def _h(task_name: str, k: int = K) -> float:
    """
    Arity-adjusted threshold in b·Δu space.

        h_n = 2 · median( Beta(K₊, K₋) ) − 1

    For 1-way tasks this is 0.  For conjunctions it is negative, correcting
    the downward bias in P(yes) that arises from training only on 1-way tasks.
    """
    n       = _arity(task_name)
    k_plus  = k // (2 ** n)
    k_minus = k - k_plus
    med     = scipy_beta.median(k_plus, k_minus)
    return 2.0 * med - 1.0


def _noise_floor(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    if not mask.any():
        return 0.0
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1)))


def _true_sem(true_vals: np.ndarray, totals: np.ndarray) -> np.ndarray:
    safe = np.clip(totals, 1, None)
    sem  = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / safe)
    sem[totals <= 0] = 0
    return sem


def _cmse_nf(pred: np.ndarray, true: np.ndarray, totals: np.ndarray, mc_n: int):
    raw = float(np.mean((pred - true) ** 2))
    if mc_n and mc_n > 1:
        raw -= float(np.mean(pred * (1 - pred))) / (mc_n - 1)
    nf = _noise_floor(true, totals)
    return raw - nf, nf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Filter results pkl by tag substring (default: use all).")
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("results_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
if not candidates:
    raise FileNotFoundError(f"No results_*.pkl found in {cfg.RESULTS_DIR}. Run run.py first.")

# ---------------------------------------------------------------------------
# Print threshold table
# ---------------------------------------------------------------------------
print("Arity-adjusted thresholds (K=16):")
for n in range(1, 5):
    k_plus  = K // (2 ** n)
    k_minus = K - k_plus
    med     = scipy_beta.median(k_plus, k_minus)
    h       = 2.0 * med - 1.0
    print(f"  {n}-way:  K₊={k_plus:2d}  K₋={k_minus:2d}  "
          f"median(Beta)={med:.4f}  h={h:+.4f}")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

refs_dict = load_image_refs(cfg.METADATA)

# CLIP feature cache (frozen backbone)
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

    # Task / image lists from pickle.
    # Intersect with cfg.VAL_TASKS so that tasks excluded by the current
    # MIN_TASK_ASSIGNMENTS setting are dropped even without re-running run.py.
    _pkl_val  = set(res.get("val_tasks", cfg.VAL_TASKS))
    val_tasks = sorted(_pkl_val & set(cfg.VAL_TASKS))
    train_uids  = res.get("train_uids",  set())
    test_uids   = res.get("test_uids",   set())
    print(f"  val tasks: {len(_pkl_val)} in pickle, "
          f"{len(val_tasks)} after MIN_TASK_ASSIGNMENTS filter")

    # Original (h=0) predictions already in the pickle
    dlbt_orig = res.get("dlbt", {})

    # -----------------------------------------------------------------------
    # Load agent
    # -----------------------------------------------------------------------
    ckpt_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path} — skipping.")
        continue

    print(f"  Loading agent from {ckpt_path.name}...")
    agent = DlbtAgent(
        freeze_encoder = cfg.FREEZE_ENCODER,
        n_mc_samples   = N_MC,
        device         = device,
        mapper_hidden  = cfg.MAPPER_HIDDEN,
    )
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()

    # Populate feature cache
    if cfg.FREEZE_ENCODER:
        agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    else:
        # Non-frozen: precompute backbone spatial maps then apply attnpool
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

    # -----------------------------------------------------------------------
    # Empirical lookup (from pickle's dlbt predictions — ground truth is
    # stored alongside predictions in each condition dict)
    # -----------------------------------------------------------------------
    def _get_orig(cond: str, task_name: str):
        """Return (pred [B], true [B], totals [B]) from the original pickle."""
        d = dlbt_orig.get(cond, {}).get(task_name)
        if d is None:
            return None, None, None
        pred = d["pred"]
        # pred may be [n_seeds, B] or [B] — take mean across seeds
        if pred.ndim == 2:
            pred = pred.mean(axis=0)
        return pred, d["true"], d["totals"]

    # -----------------------------------------------------------------------
    # Threshold-corrected MC inference
    # -----------------------------------------------------------------------
    # Evaluate two regions:
    #   "task"  — val tasks × main (train) images
    #   "joint" — val tasks × probe images
    regions = {
        "task":  train_uids,
        "joint": test_uids,
    }

    corr_preds = {r: {} for r in regions}   # region -> task -> {pred, true, totals, h, n_way}

    for region, uids_set in regions.items():
        print(f"\n  Region: {region}  ({len(uids_set)} images)")
        orig_cond = "task" if region == "task" else "joint"

        for task_name in val_tasks:
            h_val = _h(task_name)
            task  = get_task(task_name)
            delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=device)

            # Images present in this region for this task
            orig_pred, orig_true, orig_totals = _get_orig(orig_cond, task_name)
            if orig_pred is None:
                continue
            orig_uids = dlbt_orig[orig_cond][task_name].get("uids", [])
            # Filter to the correct region's UIDs
            mask    = np.array([u in uids_set for u in orig_uids])
            if not mask.any():
                continue

            batch_refs = [refs_dict[orig_uids[i]] for i in np.where(mask)[0]]
            true_p     = orig_true[mask]
            totals     = orig_totals[mask]

            with torch.no_grad():
                alpha  = agent.get_alpha(batch_refs).clamp(min=0.1)   # [B, K]
                b      = Dirichlet(alpha).sample((N_MC,))              # [N, B, K]
                logit  = torch.einsum("nbk,k->nb", b, delta_u)        # [N, B]
                hard   = (logit > h_val).float()
                p_corr = hard.mean(dim=0).cpu().numpy()                # [B]

            corr_preds[region][task_name] = {
                "pred":   p_corr,
                "orig":   orig_pred[mask],
                "true":   true_p,
                "totals": totals,
                "h":      h_val,
                "n_way":  _arity(task_name),
            }

    # -----------------------------------------------------------------------
    # Pool metrics for joint_gen
    # -----------------------------------------------------------------------
    def _pool(region: str):
        tasks_with_data = [t for t in val_tasks if t in corr_preds[region]]
        if not tasks_with_data:
            return None
        pred_c  = np.concatenate([corr_preds[region][t]["pred"]   for t in tasks_with_data])
        pred_o  = np.concatenate([corr_preds[region][t]["orig"]   for t in tasks_with_data])
        true    = np.concatenate([corr_preds[region][t]["true"]   for t in tasks_with_data])
        totals  = np.concatenate([corr_preds[region][t]["totals"] for t in tasks_with_data])
        valid   = totals > 0
        return pred_c[valid], pred_o[valid], true[valid], totals[valid]

    for region in regions:
        pooled = _pool(region)
        if pooled is None:
            print(f"\n  [{region}] No data — skipping metrics.")
            continue
        pred_c, pred_o, true, totals = pooled
        mse_o, nf = _cmse_nf(pred_o, true, totals, N_MC)
        mse_c, _  = _cmse_nf(pred_c, true, totals, N_MC)
        rho_o, _  = spearmanr(pred_o, true)
        rho_c, _  = spearmanr(pred_c, true)
        print(f"\n  [{region}]  NF={nf:.4f}")
        print(f"    Original:   ρ={rho_o:.3f}  cMSE-NF={mse_o:+.4f}")
        print(f"    Corrected:  ρ={rho_c:.3f}  cMSE-NF={mse_c:+.4f}")

    # -----------------------------------------------------------------------
    # Plot 1 — cMSE-NF bar: original vs corrected for task_gen & joint_gen
    # -----------------------------------------------------------------------
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3.5))

    bar_data = []
    for region, label in [("task", "task gen"), ("joint", "joint gen")]:
        pooled = _pool(region)
        if pooled is None:
            continue
        pred_c, pred_o, true, totals = pooled
        mse_o, nf = _cmse_nf(pred_o, true, totals, N_MC)
        mse_c, _  = _cmse_nf(pred_c, true, totals, N_MC)
        bar_data.append((label, mse_o, mse_c))

    x      = np.arange(len(bar_data))
    width  = 0.35
    labels = [d[0] for d in bar_data]
    orig_v = [d[1] for d in bar_data]
    corr_v = [d[2] for d in bar_data]

    ax_bar.bar(x - width / 2, orig_v, width, color=C_ORIG, alpha=0.85, label="original (h=0)")
    ax_bar.bar(x + width / 2, corr_v, width, color=C_CORR, alpha=0.85, label="corrected (hₙ)")
    ax_bar.axhline(0, color="gray", lw=0.8, ls=":")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_ylabel("cMSE − noise floor", fontsize=10)
    ax_bar.set_title(f"Threshold correction  [{run_tag}]", fontsize=10)
    ax_bar.legend(fontsize=8, frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_tau_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2a & 2b — separate pooled scatter for original and corrected
    # -----------------------------------------------------------------------
    pooled_j = _pool("joint")
    if pooled_j is not None:
        pred_c, pred_o, true, totals = pooled_j
        ts        = _true_sem(true, totals)
        mse_o, nf = _cmse_nf(pred_o, true, totals, N_MC)
        mse_c, _  = _cmse_nf(pred_c, true, totals, N_MC)
        rho_o, _  = spearmanr(pred_o, true)
        rho_c, _  = spearmanr(pred_c, true)

        for suffix, pred, color, sublabel, rho, mse in [
            ("orig", pred_o, C_ORIG,
             f"original (h=0)\nρ={rho_o:.3f}  cMSE-NF={mse_o:+.4f}", rho_o, mse_o),
            ("corr", pred_c, C_CORR,
             f"corrected (hₙ)\nρ={rho_c:.3f}  cMSE-NF={mse_c:+.4f}", rho_c, mse_c),
        ]:
            fig_s, ax_s = plt.subplots(figsize=(4.5, 4.2))
            ax_s.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
            ax_s.errorbar(pred, true, yerr=ts,
                          fmt="o", ms=4, alpha=0.5, color=color,
                          elinewidth=0.4, capsize=0, linewidth=0)
            ax_s.set_title(sublabel, fontsize=9, pad=4)
            ax_s.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            ax_s.set_xticks([0, 0.5, 1]); ax_s.set_yticks([0, 0.5, 1])
            ax_s.set_aspect("equal", adjustable="box")
            ax_s.set_xlabel("Predicted P(yes)", fontsize=9)
            ax_s.set_ylabel("Human P(yes)", fontsize=9)
            ax_s.text(0.97, 0.03, f"NF={nf:.4f}",
                      transform=ax_s.transAxes, fontsize=7,
                      ha="right", va="bottom", color="gray")
            fig_s.suptitle(f"Joint gen — pooled  [{run_tag}]", fontsize=10)
            sns.despine(fig=fig_s, trim=True)
            plt.tight_layout()
            out = plots_dir / f"plot_tau_summary_{suffix}_{run_tag}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
            plt.close(fig_s)

    # -----------------------------------------------------------------------
    # Plot 3 & 4 — per-task scatters (corrected), one figure per region
    # -----------------------------------------------------------------------
    for region, region_label in [("joint", "Joint gen"), ("task", "Task gen")]:
        present = [t for t in val_tasks if t in corr_preds[region]]
        if not present:
            continue

        # Pool for suptitle
        pooled_r = _pool(region)
        if pooled_r is not None:
            pred_c_r, _, true_r, tots_r = pooled_r
            mse_r, _ = _cmse_nf(pred_c_r, true_r, tots_r, N_MC)
            rho_r, _ = spearmanr(pred_c_r, true_r)
        else:
            mse_r = rho_r = float("nan")

        N_COLS  = 8
        n_tasks = len(present)
        n_rows  = math.ceil(n_tasks / N_COLS)
        color_suffix = "pertask" if region == "joint" else "pertask_task"

        fig_t, axes_t = plt.subplots(
            n_rows, N_COLS,
            figsize=(N_COLS * 2.1, n_rows * 2.1),
            gridspec_kw={"hspace": 0.60, "wspace": 0.15},
        )
        axes_flat = np.atleast_2d(axes_t).flatten()
        for ax in axes_flat[n_tasks:]:
            ax.set_visible(False)

        for i, (ax, task_name) in enumerate(zip(axes_flat, present)):
            d     = corr_preds[region][task_name]
            valid = d["totals"] > 0
            pm    = d["pred"][valid]
            tv    = d["true"][valid]
            tot   = d["totals"][valid]
            ts    = _true_sem(tv, tot)
            color = ARITY_COLOR.get(d["n_way"], "#555")

            ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
            ax.errorbar(pm, tv, yerr=ts,
                        fmt="o", ms=4, alpha=0.85, color=color,
                        elinewidth=0.5, capsize=0, linewidth=0, zorder=2)

            y_top = 0.97
            ax.text(0.05, y_top, f"h={d['h']:+.3f}  ({d['n_way']}-way)",
                    transform=ax.transAxes, fontsize=5.5, color="gray", va="top")
            y_top -= 0.16
            if valid.sum() >= 2:
                rc, _ = spearmanr(pm, tv)
                mc_v, _ = _cmse_nf(pm, tv, tot, N_MC)
                ax.text(0.05, y_top, f"ρ={rc:.2f}  m={mc_v:.3f}",
                        transform=ax.transAxes, fontsize=5.5, color=color, va="top")

            ax.set_title(_label(task_name), fontsize=6.5, pad=2, color=color)
            ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
            ax.tick_params(labelsize=4.5)

            row_i, col_i = divmod(i, N_COLS)
            if row_i == n_rows - 1 or i >= n_tasks - N_COLS:
                ax.set_xlabel("Pred", fontsize=6)
            if col_i == 0:
                ax.set_ylabel("Human", fontsize=6)

        # Arity legend
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=5, label=f"{a}-way")
                   for a, c in ARITY_COLOR.items() if a > 1]
        fig_t.legend(handles=handles, loc="lower right",
                     bbox_to_anchor=(1.0, 0.0), fontsize=7,
                     frameon=False, ncol=len(handles))

        fig_t.suptitle(
            f"{region_label} — per task  (hₙ corrected)  [{run_tag}]\n"
            f"ρ={rho_r:.3f}   cMSE-NF={mse_r:+.4f}",
            fontsize=9, y=1.01,
        )
        sns.despine(fig=fig_t, trim=True)
        plt.tight_layout()
        out = plots_dir / f"plot_tau_{color_suffix}_{run_tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close()

print("\nDone.")
