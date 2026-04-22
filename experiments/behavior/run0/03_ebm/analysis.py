"""
03_ebm/analysis.py — plots for the EBM run.

Generated figures:
  plot_01_curves_<tag>.png         — learning curves + ESS trace
  plot_02_train_<tag>.png          — train region: pooled + per-task
  plot_03_stim_gen_<tag>.png       — stim gen region
  plot_04_task_gen_<tag>.png       — task gen region
  plot_05_joint_gen_<tag>.png      — joint gen region
  plot_06_density_<tag>.png        — learned density marginals for probe images

Run from repo root:
    python experiments/behavior/run0/03_ebm/analysis.py
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers (same as 01_fit/analysis.py)
# ---------------------------------------------------------------------------

def _true_sem(true_vals, totals):
    totals_safe = np.clip(totals, 1, None)
    sem = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / totals_safe)
    sem[totals <= 0] = 0
    return sem


def _compute_metrics(pt, task_names, mc_n=None, n_seeds=1):
    all_preds  = np.concatenate([pt[t]["pred"]   for t in task_names if t in pt])
    all_trues  = np.concatenate([pt[t]["true"]   for t in task_names if t in pt])
    all_totals = np.concatenate([pt[t]["totals"] for t in task_names if t in pt])
    valid = all_totals > 0
    p = all_preds[valid] if all_preds.ndim == 1 else all_preds[..., valid]
    t = all_trues[valid]
    n = all_totals[valid]
    pred_mean = p
    pred_sem  = np.zeros_like(pred_mean)
    raw_mse   = float(np.mean((pred_mean - t) ** 2))
    if mc_n and mc_n > 1:
        raw_mse -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
    nf = float(np.mean(t * (1 - t) / np.clip(n - 1, 1, None)) if (n > 1).any() else 0.0)
    net_mse = raw_mse - nf
    rho, _  = spearmanr(pred_mean, t)
    return pred_mean, pred_sem, t, n, rho, raw_mse, net_mse


def _draw_pooled(ax, pred_mean, pred_sem, trues, totals, rho, raw_mse, net_mse,
                 color, title=""):
    true_sem = _true_sem(trues, totals)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.errorbar(pred_mean, trues, xerr=pred_sem, yerr=true_sem,
                fmt="o", ms=4, alpha=0.15, color=color,
                elinewidth=0.5, capsize=0, linewidth=0)
    ax.set_title(
        f"{title}\nMSE={raw_mse:.4f}  (−NF)={net_mse:+.4f}   ρ={rho:.3f}",
        fontsize=8, pad=4,
    )
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=9)


def _draw_task_panel(ax, pt, task_name, cond_colors, mc_n=None):
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    y_top = 0.93
    for cond, color in cond_colors:
        if task_name not in pt.get(cond, {}):
            continue
        d     = pt[cond][task_name]
        valid = d["totals"] > 0
        pred  = d["pred"][valid] if d["pred"].ndim == 1 else d["pred"][..., valid]
        true_ = d["true"][valid]
        tsem  = _true_sem(true_, d["totals"][valid])
        ax.errorbar(pred, true_, yerr=tsem,
                    fmt="o", ms=3, alpha=0.2, color=color,
                    elinewidth=0.4, capsize=0, linewidth=0)
        if valid.sum() >= 2:
            rho, _ = spearmanr(pred, true_)
            ax.text(0.05, y_top, f"ρ={rho:.2f}",
                    transform=ax.transAxes, fontsize=6, color=color, va="top")
            y_top -= 0.15
    label = task_name.replace("_and_", " & ").replace("_", "/")
    ax.set_title(label, fontsize=7, pad=2)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=5)


def _region_figure(pt, region_name, task_list, cond_colors, color,
                   run_tag, mc_n, noise_floor_val):
    N_TASK_COLS = 6
    n_tasks     = len(task_list)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)
    total_rows  = max(n_task_rows, 2)
    total_cols  = 2 + N_TASK_COLS

    merged_pt = {}
    for cond, _ in cond_colors:
        for t in task_list:
            if t in pt.get(cond, {}) and t not in merged_pt:
                merged_pt[t] = pt[cond][t]

    pred_mean, pred_sem, trues, totals, rho, raw_mse, net_mse = \
        _compute_metrics(merged_pt, task_list, mc_n=mc_n)

    fig_w = total_cols * 1.8 + 0.5
    fig_h = total_rows * 2.0 + 0.6
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(total_rows, total_cols, hspace=0.6, wspace=0.25)

    ax_pooled = fig.add_subplot(gs[:total_rows, :2])
    _draw_pooled(ax_pooled, pred_mean, pred_sem, trues, totals,
                 rho, raw_mse, net_mse, color=color,
                 title=f"{region_name.replace('_', ' ').title()} — pooled")
    ax_pooled.set_xlabel("Predicted P(yes)", fontsize=9)
    ax_pooled.set_ylabel("Human P(yes)",     fontsize=9)
    if noise_floor_val is not None:
        ax_pooled.text(0.97, 0.03, f"NF={noise_floor_val:.4f}",
                       transform=ax_pooled.transAxes, fontsize=7,
                       ha="right", va="bottom", color="gray")

    axes_flat = [fig.add_subplot(gs[r, 2 + c])
                 for r in range(n_task_rows) for c in range(N_TASK_COLS)]
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)
    for i, (ax, task_name) in enumerate(zip(axes_flat, task_list)):
        _draw_task_panel(ax, pt, task_name, cond_colors, mc_n=mc_n)
        r, c = divmod(i, N_TASK_COLS)
        if r == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred", fontsize=7)
        if c == 0:
            ax.set_ylabel("Human", fontsize=7)

    handles = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=c, markersize=5, label=cond)
               for cond, c in cond_colors]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
               fontsize=8, frameon=False, ncol=len(cond_colors))

    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_{region_name}_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Density visualization helpers
# ---------------------------------------------------------------------------

def _marginal_weights(agent, refs, dim_mask):
    """
    Compute the marginal EBM weight mass for a given latent dimension.

    For dimension d, the marginal belief = Σ_{k: bit_d(k)=1} p̃_i_k,
    i.e. the MC sample's probability mass on the "1" side of dimension d.

    Returns:
        marginals: [N] array of marginal values in [0,1] for each MC sample
        weights:   [B, N] importance weights (softmax over scores)
    """
    from dlbt.agents.ebm import EBMAgent
    with torch.no_grad():
        feats   = agent._encode(refs)
        scores  = agent._scores(feats)
        weights = torch.softmax(scores, dim=1).cpu().numpy()   # [B, N]
    mc = agent.mc_samples.cpu().numpy()                         # [N, K]
    marginals = mc[:, dim_mask].sum(axis=1)                     # [N]
    return marginals, weights


def _plot_density_probe(agent, probe_refs, run_tag):
    """
    For each probe image: plot the EBM's marginal density over the 4 latent
    dimensions (lr, tr, gl, sl) as weighted histograms.
    """
    from dlbt.constants import K

    # Bit masks for each latent dimension (bit positions in K=16 states)
    # k ∈ {0,...,15} = 4-bit: bit3=lr, bit2=tr, bit1=gl, bit0=sl
    def _bit_mask(bit):
        return np.array([(k >> bit) & 1 for k in range(K)], dtype=bool)

    dim_info = [
        (3, "Left ← | → Right",   cfg.C_TRAIN),
        (2, "Opaque ← | → Transp",cfg.C_STIM),
        (1, "Matte ← | → Glossy", cfg.C_TASK),
        (0, "Small ← | → Large",  cfg.C_JOINT),
    ]

    n_probes = min(len(probe_refs), 8)
    fig, axes = plt.subplots(n_probes, 4,
                             figsize=(10, n_probes * 1.8),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    if n_probes == 1:
        axes = axes[np.newaxis, :]

    agent.eval()
    for row_i, ref in enumerate(probe_refs[:n_probes]):
        for col_i, (bit, label, color) in enumerate(dim_info):
            ax = axes[row_i, col_i]
            mask = _bit_mask(bit)
            marginals, weights = _marginal_weights(agent, [ref], mask)
            w = weights[0]   # [N]

            # Weighted histogram
            bins = np.linspace(0, 1, 21)
            hist = np.zeros(len(bins) - 1)
            for i, (m, wi) in enumerate(zip(marginals, w)):
                idx = min(int(m * 20), 19)
                hist[idx] += wi
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax.bar(bin_centers, hist, width=0.05, color=color, alpha=0.7)
            ax.axvline(0.5, ls="--", color="gray", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=6)
            if row_i == 0:
                ax.set_title(label, fontsize=7)
            if col_i == 0:
                ax.set_ylabel(ref.uid[:6], fontsize=7, rotation=0, labelpad=28)

    fig.suptitle(f"EBM learned density — marginals over latent dimensions\n"
                 f"(weighted histogram of {agent.n_mc_samples} MC samples)",
                 fontsize=10, y=1.01)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_06_density_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


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
    raise FileNotFoundError(
        f"No results files found in {cfg.RESULTS_DIR}. Run run.py first."
    )

for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name} (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    model_label  = res["model_label"]
    best_epoch   = res["best_epoch"]
    noise_floors = res.get("noise_floors", {})
    curves       = res["curves"]
    preds        = res["dlbt"]

    # -----------------------------------------------------------------------
    # Plot 01 — learning curves + ESS
    # -----------------------------------------------------------------------
    epochs = range(len(curves["train_mses"]))
    has_ess = "train_ess" in curves and curves["train_ess"]

    n_rows = 3 if has_ess else 2
    fig, axes = plt.subplots(1, n_rows, figsize=(5 * n_rows, 3.8))
    if n_rows == 2:
        ax_nll, ax_mse = axes
        ax_ess = None
    else:
        ax_nll, ax_mse, ax_ess = axes

    ax_nll.plot(epochs, curves["train_nlls"], color=cfg.C_TRAIN, label="train", lw=1.2)
    ax_nll.plot(epochs, curves["eval_nlls"],  color=cfg.C_EVAL,  label="eval",  lw=1.2)
    ax_nll.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    ax_nll.set(ylabel="NLL", xlabel="epoch", title=f"{model_label} — NLL")
    ax_nll.legend(fontsize=8); ax_nll.set_ylim(bottom=0)

    ax_mse.plot(epochs, curves["train_mses"], color=cfg.C_TRAIN,  label="train",     lw=1.2)
    ax_mse.plot(epochs, curves["eval_mses"],  color=cfg.C_EVAL,   label="eval",      lw=1.2)
    for key, trace, color in [
        ("stim_gen",  "stim_mses",  cfg.C_STIM),
        ("task_gen",  "task_mses",  cfg.C_TASK),
        ("joint_gen", "joint_mses", cfg.C_JOINT),
    ]:
        if curves.get(trace):
            ax_mse.plot(epochs, curves[trace], color=color, lw=1.0, alpha=0.7,
                        label=key.replace("_", " "))
    ax_mse.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    for key, color in [("train", cfg.C_TRAIN), ("eval", cfg.C_EVAL),
                       ("stim_gen", cfg.C_STIM), ("task_gen", cfg.C_TASK),
                       ("joint_gen", cfg.C_JOINT)]:
        if key in noise_floors:
            ax_mse.axhline(noise_floors[key], ls="--", color=color, alpha=0.35, lw=1)
    ax_mse.set(ylabel="cMSE", xlabel="epoch", title="cMSE")
    ax_mse.legend(fontsize=8); ax_mse.set_ylim(bottom=0)

    if ax_ess is not None:
        ax_ess.plot(epochs, curves["train_ess"], color=cfg.C_EBM, lw=1.2)
        ax_ess.axhline(1.0, ls="--", color="gray", lw=0.8, alpha=0.5)
        ax_ess.axhline(1.0 / cfg.N_MC_SAMPLES, ls=":", color="red",
                       lw=0.8, alpha=0.5)
        ax_ess.axvline(best_epoch, ls=":", color="gray", lw=0.8)
        ax_ess.set(ylabel="ESS / N", xlabel="epoch",
                   title="Effective sample size\n(1=uniform, 1/N=collapsed)",
                   ylim=(0, 1.05))

    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plots 02–05 — one figure per region
    # -----------------------------------------------------------------------
    _region_figure(preds, "02_train",     cfg.TRAIN_TASKS,
                   [("train", cfg.C_TRAIN)], cfg.C_TRAIN, run_tag,
                   cfg.N_MC_SAMPLES, noise_floors.get("train"))

    _region_figure(preds, "03_stim_gen",  cfg.TRAIN_TASKS,
                   [("stim",  cfg.C_STIM)],  cfg.C_STIM,  run_tag,
                   cfg.N_MC_SAMPLES, noise_floors.get("stim_gen"))

    _region_figure(preds, "04_task_gen",  cfg.VAL_TASKS,
                   [("task",  cfg.C_TASK)],  cfg.C_TASK,  run_tag,
                   cfg.N_MC_SAMPLES, noise_floors.get("task_gen"))

    _region_figure(preds, "05_joint_gen", cfg.VAL_TASKS,
                   [("joint", cfg.C_JOINT)], cfg.C_JOINT, run_tag,
                   cfg.N_MC_SAMPLES, noise_floors.get("joint_gen"))

    # -----------------------------------------------------------------------
    # Plot 06 — density marginals for probe images (needs trained agent)
    # -----------------------------------------------------------------------
    agent_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if not agent_path.exists():
        print(f"Agent weights not found at {agent_path} — skipping density plot.")
        continue

    from dlbt.agents.ebm import EBMAgent
    from dlbt.data.image_ref import load_image_refs, image_refs_as_list

    ebm_cfg = res.get("ebm_config", {})
    _agent  = EBMAgent(
        freeze_encoder = True,
        n_mc_samples   = ebm_cfg.get("n_mc_samples", cfg.N_MC_SAMPLES),
        device         = torch.device("cpu"),
        compress_dim   = ebm_cfg.get("compress_dim",  cfg.COMPRESS_DIM),
        hidden_dim     = ebm_cfg.get("hidden_dim",    cfg.HIDDEN_DIM),
        mc_seed        = ebm_cfg.get("mc_seed",       cfg.MC_SEED),
    )
    _agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
    _agent.eval()

    clip_cache = Path(cfg.CACHE_PATH)
    if clip_cache.exists():
        _agent.load_cache(str(clip_cache))

    _refs_dict  = load_image_refs(cfg.METADATA)
    probe_uids  = list(res.get("probe_uids", []))[:8]
    probe_refs  = [_refs_dict[uid] for uid in probe_uids if uid in _refs_dict]

    if probe_refs:
        # Ensure features are available
        _agent.precompute_features(probe_refs)
        _plot_density_probe(_agent, probe_refs, run_tag)
    else:
        print("No probe refs found — skipping density plot.")

print("\nAll plots saved to", plots_dir)
