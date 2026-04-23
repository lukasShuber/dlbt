"""
02_data_efficiency/threshold_correction.py
-------------------------------------------
Re-runs MC inference on every per-budget checkpoint with an arity-adjusted
decision threshold τₙ = 2·median(Beta(K₊, K₋)) − 1.

Same threshold logic as 01_fit/threshold_correction.py, applied across all
training-trial budgets so we can compare corrected vs original joint-gen
cMSE-NF as a function of data.

Produces three plots:
    plot_de_tau_curve_<tag>.png
        cMSE-NF vs budget: original vs threshold-corrected (line plot).
    plot_de_tau_summary_<tag>.png
        Pooled scatter (corrected) for the "full" budget.
    plot_de_tau_pertask_<tag>.png
        Per-task scatters (corrected) for the "full" budget.

Run from repo root:
    python experiments/behavior/run0/02_data_efficiency/threshold_correction.py
"""

import argparse
import math
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


def _tau(task_name: str, k: int = 16) -> float:
    """
    Threshold in b·Δu space such that P(yes) = 0.5 under uninformative prior.

        τₙ_b+ = median( Beta(K₊, K₋) )
        τₙ    = 2·τₙ_b+ − 1          (convert to b·Δu space)
    """
    n       = _arity(task_name)
    k_plus  = k // (2 ** n)
    k_minus = k - k_plus
    med     = scipy_beta.median(k_plus, k_minus)
    return 2.0 * med - 1.0


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


def _load_agent(ckpt_path, device):
    """Load a per-budget lightweight checkpoint into a fresh DlbtAgent."""
    agent = DlbtAgent(
        freeze_encoder=cfg.FREEZE_ENCODER,
        n_mc_samples=N_MC,
        device=device,
        mapper_hidden=cfg.MAPPER_HIDDEN,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.mapper.load_state_dict(ckpt["mapper"])
    if "attnpool" in ckpt:
        agent.encoder.attnpool.load_state_dict(ckpt["attnpool"])
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None)
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("data_efficiency_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
if not candidates:
    raise FileNotFoundError(f"No data_efficiency_*.pkl found in {cfg.RESULTS_DIR}.")

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
# Setup: device, data
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
# Feature cache (frozen CLIP backbone)
# ---------------------------------------------------------------------------
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
    run_tag = results_path.stem[len("data_efficiency_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        summary = pickle.load(f)

    results_per_budget = summary["results"]
    val_tasks          = [t for t in cfg.VAL_TASKS if t in
                          next(iter(results_per_budget.values()))["preds"].get("joint_gen", {})]

    if not val_tasks:
        print("  No joint_gen predictions found — skipping.")
        continue

    # Ordered budget labels (same ordering logic as run.py)
    budget_labels = list(results_per_budget.keys())

    # ---------------------------------------------------------------------------
    # Per-budget: run corrected inference
    # ---------------------------------------------------------------------------
    corrected_per_budget = {}   # budget_label -> {task -> {pred, true, totals}}
    orig_cmse_per_budget = {}
    corr_cmse_per_budget = {}

    for budget_label in budget_labels:
        ckpt_path = cfg.RESULTS_DIR / f"agent_{run_tag}_budget_{budget_label}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path} — skipping budget {budget_label}.")
            continue

        print(f"\n  Budget: {budget_label}  (checkpoint: {ckpt_path.name})")
        agent = _load_agent(ckpt_path, device)

        # Populate feature cache for frozen backbone
        if cfg.FREEZE_ENCODER:
            agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
        else:
            # attnpool was fine-tuned — need to recompute features through updated attnpool
            from tqdm import tqdm as _tqdm
            all_refs_list = list(refs_dict.values())
            with torch.no_grad():
                for i in _tqdm(range(0, len(all_refs_list), 16), desc="  caching", unit="batch"):
                    batch   = all_refs_list[i: i + 16]
                    spatial = torch.stack(
                        [agent._backbone_cache[r.uid] for r in batch]
                    ).to(agent.device)
                    feats = agent.encoder.attnpool(spatial).float()
                    for ref, feat in zip(batch, feats):
                        agent._cache[ref.uid] = feat.cpu()

        # Threshold-corrected MC inference for each val task
        corr_preds = {}
        for task_name in val_tasks:
            tau     = _tau(task_name)
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
                alpha   = agent.get_alpha(batch_refs).clamp(min=0.1)   # [B, K]
                b       = Dirichlet(alpha).sample((N_MC,))              # [N, B, K]
                logit   = torch.einsum("nbk,k->nb", b, delta_u)        # [N, B]
                hard    = (logit > tau).float()
                p_right = hard.mean(dim=0).cpu().numpy()                # [B]

            corr_preds[task_name] = {
                "pred":   p_right,
                "true":   true_p,
                "totals": totals,
                "tau":    tau,
                "n_way":  _arity(task_name),
            }

        corrected_per_budget[budget_label] = corr_preds

        # --- Pool for summary metrics ---
        orig_preds_b, corr_preds_b, trues_b, tots_b = [], [], [], []
        orig_jg = results_per_budget[budget_label]["preds"]["joint_gen"]

        for t in val_tasks:
            if t not in corr_preds or t not in orig_jg:
                continue
            d_orig = orig_jg[t]
            d_corr = corr_preds[t]
            valid  = d_orig["totals"] > 0

            p_o  = d_orig["pred"]
            pm_o = p_o[valid]
            pm_c = d_corr["pred"][valid]
            tv   = d_orig["true"][valid]
            tot  = d_orig["totals"][valid]

            orig_preds_b.append(pm_o)
            corr_preds_b.append(pm_c)
            trues_b.append(tv)
            tots_b.append(tot)

        if not orig_preds_b:
            continue

        orig_pool_b  = np.concatenate(orig_preds_b)
        corr_pool_b  = np.concatenate(corr_preds_b)
        trues_pool_b = np.concatenate(trues_b)
        tots_pool_b  = np.concatenate(tots_b)

        orig_cmse, nf = _cmse_nf(orig_pool_b, trues_pool_b, tots_pool_b, cfg.N_MC)
        corr_cmse, _  = _cmse_nf(corr_pool_b, trues_pool_b, tots_pool_b, N_MC)
        orig_cmse_per_budget[budget_label] = orig_cmse
        corr_cmse_per_budget[budget_label] = corr_cmse
        rho_o, _ = spearmanr(orig_pool_b, trues_pool_b)
        rho_c, _ = spearmanr(corr_pool_b, trues_pool_b)
        print(f"    ORIGINAL:   ρ={rho_o:.3f}  cMSE-NF={orig_cmse:+.4f}")
        print(f"    CORRECTED:  ρ={rho_c:.3f}  cMSE-NF={corr_cmse:+.4f}  (NF={nf:.4f})")

    # ---------------------------------------------------------------------------
    # Plot 1 — cMSE-NF vs budget curve
    # ---------------------------------------------------------------------------
    def _budget_x(label):
        if label == "full":
            return summary.get("n_pool", np.nan)
        return int(label)

    common_labels = [b for b in budget_labels
                     if b in orig_cmse_per_budget and b in corr_cmse_per_budget]
    xs   = np.array([_budget_x(b) for b in common_labels], dtype=float)
    y_o  = np.array([orig_cmse_per_budget[b] for b in common_labels])
    y_c  = np.array([corr_cmse_per_budget[b] for b in common_labels])

    fig_c, ax_c = plt.subplots(figsize=(5, 3.5))
    ax_c.axhline(0, ls="--", color="gray", lw=0.8, zorder=0)
    ax_c.plot(xs, y_o, "o-", color=C_ORIG, lw=1.5, ms=5, label="original (τ=0)")
    ax_c.plot(xs, y_c, "o-", color=C_CORR, lw=1.5, ms=5, label="corrected (τₙ)")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Training trials (budget)", fontsize=10)
    ax_c.set_ylabel("cMSE − NF  (joint gen)", fontsize=10)
    ax_c.set_title(f"Data efficiency — threshold correction\n{run_tag}", fontsize=9)
    ax_c.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax_c, trim=True)
    plt.tight_layout()
    out_c = plots_dir / f"plot_de_tau_curve_{run_tag}.png"
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_c}")
    plt.close()

    # ---------------------------------------------------------------------------
    # "Full" budget: summary + per-task scatters
    # ---------------------------------------------------------------------------
    full_label = "full" if "full" in corrected_per_budget else (
        common_labels[-1] if common_labels else None
    )
    if full_label is None or full_label not in corrected_per_budget:
        print("  No full-budget corrected predictions — skipping summary plots.")
        continue

    corr_full = corrected_per_budget[full_label]
    orig_full = results_per_budget[full_label]["preds"]["joint_gen"]

    # Pool for summary metrics
    orig_p_all, corr_p_all, trues_all, tots_all = [], [], [], []
    for t in val_tasks:
        if t not in corr_full or t not in orig_full:
            continue
        d_o   = orig_full[t]
        d_c   = corr_full[t]
        valid = d_o["totals"] > 0
        orig_p_all.append(d_o["pred"][valid])
        corr_p_all.append(d_c["pred"][valid])
        trues_all.append(d_o["true"][valid])
        tots_all.append(d_o["totals"][valid])

    orig_pool  = np.concatenate(orig_p_all)
    corr_pool  = np.concatenate(corr_p_all)
    trues_pool = np.concatenate(trues_all)
    tots_pool  = np.concatenate(tots_all)

    rho_corr, _ = spearmanr(corr_pool, trues_pool)
    mse_corr, nf = _cmse_nf(corr_pool, trues_pool, tots_pool, N_MC)

    # --- Plot 2: pooled summary scatter ---
    fig_s, ax_p = plt.subplots(figsize=(4.5, 4.5))
    ts = _true_sem(trues_pool, tots_pool)
    ax_p.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax_p.errorbar(corr_pool, trues_pool, yerr=ts,
                  fmt="o", ms=4, alpha=0.6, color=C_CORR,
                  elinewidth=0.4, capsize=0, linewidth=0)
    ax_p.set_title(
        f"Joint gen — pooled  (threshold corrected, budget={full_label})\n"
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
    out_s = plots_dir / f"plot_de_tau_summary_{run_tag}.png"
    plt.savefig(out_s, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_s}")
    plt.close()

    # --- Plot 3: per-task scatters ---
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
        if task_name not in corr_full or task_name not in orig_full:
            ax.set_visible(False)
            continue

        d_o   = orig_full[task_name]
        d_c   = corr_full[task_name]
        valid = d_o["totals"] > 0

        pm_c = d_c["pred"][valid]
        tv   = d_o["true"][valid]
        tot  = d_o["totals"][valid]
        ts   = _true_sem(tv, tot)

        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        ax.errorbar(pm_c, tv, yerr=ts, fmt="o", ms=4, alpha=0.85, color=C_CORR,
                    elinewidth=0.5, capsize=0, linewidth=0)

        tau   = d_c["tau"]
        n_way = d_c["n_way"]
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
        f"Joint gen — per task  (threshold corrected, budget={full_label})   "
        f"ρ={rho_corr:.3f}   cMSE-NF={mse_corr:+.4f}",
        fontsize=9, y=1.01,
    )
    sns.despine(fig=fig_t, trim=True)
    plt.tight_layout()
    out_t = plots_dir / f"plot_de_tau_pertask_{run_tag}.png"
    plt.savefig(out_t, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_t}")
    plt.close()

print("\nDone.")
