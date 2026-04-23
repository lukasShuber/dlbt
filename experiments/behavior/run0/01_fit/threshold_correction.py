"""
01_fit/threshold_correction.py
-------------------------------
Re-runs MC inference on the saved checkpoint with an arity-adjusted decision
threshold instead of the fixed threshold of 0.

Standard eval path in DlbtAgent._choice_probs_eval:
    logit = b · Δu          [N, B]
    hard  = (logit > 0)     ← threshold hardcoded at 0

The fix: replace 0 with τₙ = 2 · median(Beta(K₊, K₋)) − 1, chosen so
that under the uninformative prior a maximally uncertain image gives P = 0.5
regardless of task arity.

    K₊ = K / 2ⁿ    (number of +1 states for an n-way conjunction task)
    K₋ = K − K₊

No retraining needed — only the evaluation threshold changes.

Produces:
    plot_08_threshold_correction_<tag>.png

Run from repo root:
    python experiments/behavior/run0/01_fit/threshold_correction.py
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy.stats import beta as scipy_beta, spearmanr
from torch.distributions import Dirichlet

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.constants import K

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

C_ORIG = cfg.C_JOINT
C_CORR = "#E76F51"

N_MC = 2000   # more samples for cleaner eval


def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _tau(task_name: str, K: int = 16) -> float:
    """
    Threshold in b·Δu space such that P(yes) = 0.5 under uninformative prior.

        τₙ_b+  = median( Beta(K₊, K₋) )
        τₙ     = 2·τₙ_b+ − 1            (convert to b·Δu space)
    """
    n      = _arity(task_name)
    k_plus = K // (2 ** n)
    k_minus = K - k_plus
    median_b_plus = scipy_beta.median(k_plus, k_minus)
    return 2.0 * median_b_plus - 1.0


def _noise_floor_local(true_vals, totals):
    mask = totals > 1
    if not mask.any():
        return 0.0
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1)))


def _true_sem(true_vals, totals):
    totals_safe = np.clip(totals, 1, None)
    sem = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / totals_safe)
    sem[totals <= 0] = 0
    return sem


def _cmse_nf(pred_mean, true_vals, totals, mc_n):
    raw = float(np.mean((pred_mean - true_vals) ** 2))
    if mc_n and mc_n > 1:
        raw -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
    nf = _noise_floor_local(true_vals, totals)
    return raw - nf, nf


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
    raise FileNotFoundError(f"No results files found in {cfg.RESULTS_DIR}.")

# ---------------------------------------------------------------------------
# Print threshold table
# ---------------------------------------------------------------------------
print("Arity-adjusted thresholds (K=16):")
for n in range(1, 5):
    k_plus  = K // (2 ** n)
    k_minus = K - k_plus
    med     = scipy_beta.median(k_plus, k_minus)
    tau     = 2.0 * med - 1.0
    print(f"  {n}-way:  K₊={k_plus:2d}  K₋={k_minus:2d}  "
          f"median(Beta)={med:.4f}  τ={tau:+.4f}")

# ---------------------------------------------------------------------------
# Setup: device, data, agent
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

refs_dict = load_image_refs(cfg.METADATA)

print("Loading behavioural data...")
full_ds, probe_uids, main_uids, _ = load_and_preprocess(
    cfg.BEHAVIOR_CSV,
    beh_id_to_task     = cfg.BEH_ID_TO_TASK,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    use_trial_kinds    = cfg.USE_TRIAL_KINDS,
    seed               = cfg.SEED,
)

_emp_lookup = {}
for row in full_ds.df.itertuples(index=False):
    total = row.count_0 + row.count_1
    p = row.count_1 / total if total > 0 else np.nan
    _emp_lookup[(row.uid, row.task_name)] = (p, total)

def emp_p(uid, task_name):
    v = _emp_lookup.get((uid, task_name))
    return v[0] if v is not None else np.nan

def emp_n(uid, task_name):
    v = _emp_lookup.get((uid, task_name))
    return v[1] if v is not None else 0

# joint_gen: probe images × val tasks
joint_gen_df = full_ds.df[
    full_ds.df["uid"].isin(probe_uids) &
    full_ds.df["task_name"].isin(cfg.VAL_TASKS)
].copy()

# ---------------------------------------------------------------------------
# Process each results file
# ---------------------------------------------------------------------------
for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    # Load original predictions for comparison
    with open(results_path, "rb") as f:
        res = pickle.load(f)
    joint_orig = res["dlbt"].get("joint", {})

    # Load agent checkpoint
    ckpt_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path} — skipping.")
        continue

    print(f"  Loading checkpoint: {ckpt_path}")
    agent = DlbtAgent(
        freeze_encoder = cfg.FREEZE_ENCODER,
        n_mc_samples   = N_MC,
        device         = device,
        mapper_hidden  = cfg.MAPPER_HIDDEN,
    )
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()

    # Populate feature cache
    cache_path = Path(cfg.CACHE_PATH)
    if cache_path.exists() and cfg.FREEZE_ENCODER:
        print("  Loading CLIP feature cache...")
        agent.load_cache(str(cache_path))
    else:
        print("  Precomputing features...")
        agent.precompute_features(list(refs_dict.values()))

    # -----------------------------------------------------------------------
    # Re-run MC inference with arity-adjusted threshold
    # -----------------------------------------------------------------------
    val_tasks = [t for t in cfg.VAL_TASKS if t in joint_orig]
    corr_preds = {}

    print("  Running threshold-corrected inference...")
    for task_name in val_tasks:
        tau     = _tau(task_name)
        n_way   = _arity(task_name)
        task    = TASKS[task_name]
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=device)

        group   = joint_gen_df[joint_gen_df["task_name"] == task_name]
        uids    = group["uid"].tolist()
        if not uids:
            continue

        batch_refs = [refs_dict[uid] for uid in uids]
        true_p  = np.array([emp_p(uid, task_name) for uid in uids])
        totals  = np.array([emp_n(uid, task_name) for uid in uids])

        with torch.no_grad():
            alpha = agent.get_alpha(batch_refs).clamp(min=0.1)   # [B, K]
            b     = Dirichlet(alpha).sample((N_MC,))              # [N, B, K]
            logit = torch.einsum("nbk,k->nb", b, delta_u)        # [N, B]
            hard  = (logit > tau).float()                         # threshold τₙ
            p_right = hard.mean(dim=0).cpu().numpy()              # [B]

        corr_preds[task_name] = {
            "pred":   p_right,
            "true":   true_p,
            "totals": totals,
            "tau":    tau,
            "n_way":  n_way,
        }
        print(f"    {task_name:40s}  τ={tau:+.4f}  ({n_way}-way)")

    # -----------------------------------------------------------------------
    # Pool: original vs corrected
    # -----------------------------------------------------------------------
    orig_pool, corr_pool, trues_pool, tots_pool = [], [], [], []

    for t in val_tasks:
        if t not in corr_preds or t not in joint_orig:
            continue
        d_orig = joint_orig[t]
        d_corr = corr_preds[t]
        valid  = d_orig["totals"] > 0

        p_o = d_orig["pred"]
        p_o = p_o[..., valid] if p_o.ndim == 2 else p_o[valid]
        pm_o = p_o.mean(axis=0) if p_o.ndim == 2 else p_o

        pm_c = d_corr["pred"][valid]
        tv   = d_orig["true"][valid]
        tot  = d_orig["totals"][valid]

        orig_pool.append(pm_o)
        corr_pool.append(pm_c)
        trues_pool.append(tv)
        tots_pool.append(tot)

    orig_pool  = np.concatenate(orig_pool)
    corr_pool  = np.concatenate(corr_pool)
    trues_pool = np.concatenate(trues_pool)
    tots_pool  = np.concatenate(tots_pool)

    rho_orig, _ = spearmanr(orig_pool, trues_pool)
    rho_corr, _ = spearmanr(corr_pool, trues_pool)
    mse_orig, nf = _cmse_nf(orig_pool, trues_pool, tots_pool, N_MC)
    mse_corr, _  = _cmse_nf(corr_pool, trues_pool, tots_pool, N_MC)

    print(f"\n  Pooled ORIGINAL:   ρ={rho_orig:.3f}  cMSE-NF={mse_orig:+.4f}")
    print(f"  Pooled CORRECTED:  ρ={rho_corr:.3f}  cMSE-NF={mse_corr:+.4f}  (NF={nf:.4f})")

    # -----------------------------------------------------------------------
    # Plot A — summary (pooled scatter only)
    # -----------------------------------------------------------------------
    fig_s, ax_p = plt.subplots(figsize=(4.5, 4.5))
    ts = _true_sem(trues_pool, tots_pool)
    ax_p.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax_p.errorbar(corr_pool, trues_pool, yerr=ts,
                  fmt="o", ms=4, alpha=0.6, color=C_CORR,
                  elinewidth=0.4, capsize=0, linewidth=0)
    ax_p.set_title(
        f"Joint gen — pooled  (threshold corrected)\n"
        f"ρ={rho_corr:.3f}   cMSE-NF={mse_corr:+.4f}",
        fontsize=9, pad=4)
    ax_p.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax_p.set_xticks([0, 0.5, 1]); ax_p.set_yticks([0, 0.5, 1])
    ax_p.set_xlabel("Predicted P(yes)", fontsize=10)
    ax_p.set_ylabel("Human P(yes)",     fontsize=10)
    ax_p.text(0.97, 0.03, f"NF={nf:.4f}",
              transform=ax_p.transAxes, fontsize=7, ha="right", va="bottom", color="gray")
    sns.despine(ax=ax_p, trim=True)
    plt.tight_layout()
    out_s = plots_dir / f"plot_08a_threshold_summary_{run_tag}.png"
    plt.savefig(out_s, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_s}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot B — per-task scatters (corrected only)
    # -----------------------------------------------------------------------
    N_TASK_COLS = 6
    n_tasks     = len(val_tasks)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)

    fig_t, axes = plt.subplots(
        n_task_rows, N_TASK_COLS,
        figsize=(N_TASK_COLS * 2.2, n_task_rows * 2.2),
        gridspec_kw={"hspace": 0.55, "wspace": 0.15},
    )
    axes_flat = np.atleast_2d(axes).flatten()

    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for i, (ax, task_name) in enumerate(zip(axes_flat, val_tasks)):
        if task_name not in corr_preds or task_name not in joint_orig:
            ax.set_visible(False)
            continue

        d_orig = joint_orig[task_name]
        d_corr = corr_preds[task_name]
        valid  = d_orig["totals"] > 0

        pm_c = d_corr["pred"][valid]
        tv   = d_orig["true"][valid]
        tot  = d_orig["totals"][valid]
        ts   = _true_sem(tv, tot)

        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        ax.errorbar(pm_c, tv, yerr=ts, fmt="o", ms=4, alpha=0.85, color=C_CORR,
                    elinewidth=0.5, capsize=0, linewidth=0)

        tau   = d_corr["tau"]
        n_way = d_corr["n_way"]
        y_top = 0.97
        ax.text(0.05, y_top, f"τ={tau:+.3f}  ({n_way}-way)",
                transform=ax.transAxes, fontsize=6, color="gray", va="top")
        y_top -= 0.15

        if valid.sum() >= 2:
            rc, _ = spearmanr(pm_c, tv)
            mc_v, _ = _cmse_nf(pm_c, tv, tot, N_MC)
            ax.text(0.05, y_top, f"ρ={rc:.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_CORR, va="top")
            y_top -= 0.13
            ax.text(0.05, y_top, f"mse={mc_v:.3f}",
                    transform=ax.transAxes, fontsize=6, color=C_CORR, va="top")

        label = task_name.replace("_and_", " & ").replace("_", "/")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

        if i // N_TASK_COLS == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred", fontsize=7)
        if i % N_TASK_COLS == 0:
            ax.set_ylabel("Human", fontsize=7)

    fig_t.suptitle(
        f"Joint gen — per task  (threshold corrected)   ρ={rho_corr:.3f}   cMSE-NF={mse_corr:+.4f}",
        fontsize=9, y=1.01,
    )
    sns.despine(fig=fig_t, trim=True)
    plt.tight_layout()
    out_t = plots_dir / f"plot_08b_threshold_pertask_{run_tag}.png"
    plt.savefig(out_t, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_t}")
    plt.close()

print("\nDone.")
