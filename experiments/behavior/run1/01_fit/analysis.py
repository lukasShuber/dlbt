"""
run1/01_fit/analysis.py — plots for the run1 01_fit training run.

Figures produced per results pkl:
  plot_01_curves_<tag>.png              — learning curves: NLL + cMSE
  plot_02_train_scatter_<tag>.png       — train: pooled scatter
  plot_02_train_grid_<tag>.png          — train: per-task grid
  plot_02b_eval_scatter_<tag>.png       — eval: pooled scatter
  plot_02b_eval_grid_<tag>.png          — eval: per-task grid
  plot_03_stim_gen_scatter_<tag>.png    — stim gen: pooled scatter
  plot_03_stim_gen_grid_<tag>.png       — stim gen: per-task grid
  plot_04_task_gen_scatter_<tag>.png    — task gen: pooled scatter
  plot_04_task_gen_grid_<tag>.png       — task gen: per-task grid
  plot_05_joint_gen_scatter_<tag>.png   — joint gen: pooled scatter
  plot_05_joint_gen_grid_<tag>.png      — joint gen: per-task grid
  plot_06_slda_train_scatter_<tag>.png  — SLDA train: pooled scatter
  plot_06_slda_train_grid_<tag>.png     — SLDA train: per-task grid
  plot_06_slda_stim_scatter_<tag>.png   — SLDA stim gen: pooled scatter
  plot_06_slda_stim_grid_<tag>.png      — SLDA stim gen: per-task grid
  plot_07_alpha_<tag>.png               — learned Dirichlet α heatmap (probe × K)
  plot_08a_probe_matrix_true_<tag>.png  — empirical P(yes): probe images × all tasks
  plot_08b_probe_matrix_pred_<tag>.png  — DLBT predicted P(yes): probe images × all tasks

DLBT and SLDA use the same plotting functions: arity-coloured dots,
ρ + cMSE-NF stats per panel, standalone scatter + grid figures.

Run from repo root:
    python experiments/behavior/run1/01_fit/analysis.py
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from dlbt.agents.dlbt import DlbtAgent
from dlbt.constants import K as _K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.image_ref import load_image_refs, image_refs_as_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_TRAIN = cfg.C_TRAIN
C_EVAL  = cfg.C_EVAL
C_STIM  = cfg.C_STIM
C_TASK  = cfg.C_TASK
C_JOINT = cfg.C_JOINT
C_SLDA  = "#7D6EAE"

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

N_TASK_COLS = 8


def _state_label(k: int) -> str:
    lr = (k >> DIM_LEFT_RIGHT)  & 1
    tr = (k >> DIM_TRANSP)      & 1
    gl = (k >> DIM_GLOSS)       & 1
    sl = (k >> DIM_SMALL_LARGE) & 1
    return (f"{'R' if lr else 'L'} "
            f"{'Tr' if tr else 'Op'} "
            f"{'Gl' if gl else 'Mt'} "
            f"{'Lg' if sl else 'Sm'}")

STATE_LABELS = [_state_label(k) for k in range(_K)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _label(task_name: str) -> str:
    return task_name.replace("_and_", " & ").replace("_", "/")


def _true_sem(true_vals: np.ndarray, totals: np.ndarray) -> np.ndarray:
    totals_safe = np.clip(totals, 1, None)
    sem = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / totals_safe)
    sem[totals <= 0] = 0
    return sem


def _noise_floor_local(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    if not mask.any():
        return 0.0
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1)))


def _compute_metrics(pt: dict, task_names: list, mc_n=None, n_seeds=1):
    """Pool predictions across tasks.

    Returns (pred_mean, pred_sem, true, totals, rho, raw_mse, net_mse)
    or None if no data is available.
    """
    present = [t for t in task_names if t in pt]
    if not present:
        return None

    all_preds  = np.concatenate([pt[t]["pred"]   for t in present], axis=-1)
    all_trues  = np.concatenate([pt[t]["true"]   for t in present])
    all_totals = np.concatenate([pt[t]["totals"] for t in present])

    valid      = all_totals > 0
    all_preds  = all_preds[..., valid] if all_preds.ndim == 2 else all_preds[valid]
    all_trues  = all_trues[valid]
    all_totals = all_totals[valid]

    if all_preds.ndim == 2:
        pred_mean = all_preds.mean(axis=0)
        pred_sem  = all_preds.std(axis=0) / np.sqrt(max(n_seeds, 1))
    else:
        pred_mean = all_preds
        pred_sem  = np.zeros_like(pred_mean)

    raw_mse = float(np.mean((pred_mean - all_trues) ** 2))
    if mc_n and mc_n > 1:
        raw_mse -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
    nf      = _noise_floor_local(all_trues, all_totals)
    net_mse = raw_mse - nf
    rho, _  = spearmanr(pred_mean, all_trues)
    return pred_mean, pred_sem, all_trues, all_totals, rho, raw_mse, net_mse


def _draw_pooled(ax, pt_cond, task_names, rho, raw_mse, net_mse,
                 title="", n_seeds=1, mc_n=None):
    """Pooled scatter with arity-coloured dots (one colour per task arity)."""
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    for task_name in [t for t in task_names if t in pt_cond]:
        color = ARITY_COLOR.get(_arity(task_name), "#555")
        d     = pt_cond[task_name]
        p     = d["pred"]
        valid = d["totals"] > 0
        pv    = p[..., valid] if p.ndim == 2 else p[valid]
        pm    = pv.mean(axis=0) if pv.ndim == 2 else pv
        ps    = pv.std(axis=0) / np.sqrt(max(n_seeds, 1)) if pv.ndim == 2 else np.zeros_like(pm)
        tv    = d["true"][valid]
        tot   = d["totals"][valid]
        ts    = _true_sem(tv, tot)
        ax.errorbar(pm, tv, xerr=ps, yerr=ts,
                    fmt="o", ms=4, alpha=0.45, color=color,
                    elinewidth=0.5, capsize=0, linewidth=0)
    ax.set_title(
        f"{title}\nMSE={raw_mse:.4f}  (−NF)={net_mse:+.4f}   ρ={rho:.3f}",
        fontsize=8, pad=4,
    )
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=9)


def _draw_task_panel(ax, pt: dict, cond: str, task_name: str, n_seeds=1, mc_n=None):
    """Single per-task scatter panel, dot colour = arity colour."""
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    color = ARITY_COLOR.get(_arity(task_name), "#555")

    if task_name not in pt.get(cond, {}):
        ax.set_visible(False)
        return

    d     = pt[cond][task_name]
    p     = d["pred"]
    valid = d["totals"] > 0
    pv    = p[..., valid] if p.ndim == 2 else p[valid]
    pm    = pv.mean(axis=0) if pv.ndim == 2 else pv
    tv    = d["true"][valid]
    tot   = d["totals"][valid]
    ts    = _true_sem(tv, tot)
    ps    = pv.std(axis=0) / np.sqrt(max(n_seeds, 1)) if pv.ndim == 2 else np.zeros_like(pm)

    ax.errorbar(pm, tv, xerr=ps, yerr=ts,
                fmt="o", ms=3, alpha=0.7, color=color,
                elinewidth=0.4, capsize=0, linewidth=0)

    y_top = 0.95
    if valid.sum() >= 2:
        rho, _ = spearmanr(pm, tv)
        raw    = float(np.mean((pm - tv) ** 2))
        if mc_n and mc_n > 1:
            raw -= float(np.mean(pm * (1 - pm))) / (mc_n - 1)
        net = raw - _noise_floor_local(tv, tot)
        ax.text(0.05, y_top,        f"ρ={rho:.2f}", transform=ax.transAxes,
                fontsize=5.5, color=color, va="top")
        ax.text(0.05, y_top - 0.15, f"m={net:.3f}", transform=ax.transAxes,
                fontsize=5.5, color=color, va="top")

    ax.set_title(_label(task_name), fontsize=6, pad=2, color=color)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=4.5)


def _plot_summary(pt, cond, task_names, run_tag, region_name,
                  n_seeds, mc_n, noise_floor_val, title):
    """Save a standalone pooled scatter figure (arity-coloured dots)."""
    metrics = _compute_metrics(pt.get(cond, {}), task_names, mc_n=mc_n, n_seeds=n_seeds)
    if metrics is None:
        print(f"  Skipping summary {region_name}: no data.")
        return
    _, _, _, _, rho, raw_mse, net_mse = metrics

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    _draw_pooled(ax, pt.get(cond, {}), task_names, rho, raw_mse, net_mse,
                 title=title, n_seeds=n_seeds, mc_n=mc_n)
    ax.set_xlabel("Predicted P(yes)", fontsize=9)
    ax.set_ylabel("Human P(yes)",     fontsize=9)
    if noise_floor_val is not None:
        ax.text(0.97, 0.03, f"NF={noise_floor_val:.4f}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="gray")
    ax.set_aspect("equal", adjustable="box")
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=5, label=f"{a}-way")
               for a, c in ARITY_COLOR.items()
               if any(_arity(t) == a for t in task_names)]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_{region_name}_scatter_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def _plot_task_grid(pt, cond, task_list, run_tag, region_name, n_seeds, mc_n):
    """Save a standalone per-task grid figure."""
    task_list = sorted(task_list, key=lambda t: (_arity(t), t))
    present   = [t for t in task_list if t in pt.get(cond, {})]
    if not present:
        print(f"  Skipping grid {region_name}: no data.")
        return

    n_tasks     = len(task_list)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)

    fig, axes = plt.subplots(
        n_task_rows, N_TASK_COLS,
        figsize=(N_TASK_COLS * 1.8, n_task_rows * 2.0),
        gridspec_kw={"hspace": 0.65, "wspace": 0.10},
    )
    axes_flat = np.atleast_2d(axes).flatten()
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for i, (ax, task_name) in enumerate(zip(axes_flat, task_list)):
        _draw_task_panel(ax, pt, cond, task_name, n_seeds=n_seeds, mc_n=mc_n)
        row_i, col_i = divmod(i, N_TASK_COLS)
        if row_i == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred",  fontsize=6)
        if col_i == 0:
            ax.set_ylabel("Human", fontsize=6)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=5, label=f"{a}-way")
               for a, c in ARITY_COLOR.items()]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(1.0, 0.0), fontsize=7,
               frameon=False, ncol=4)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_{region_name}_grid_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _plot_alpha_heatmap(alpha_mat: np.ndarray, row_labels: list,
                        title: str, fname: str) -> None:
    """[n_probe × K] Dirichlet α heatmap, rows=probe images, cols=latent states."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(alpha_mat, ax=ax,
                xticklabels=STATE_LABELS, yticklabels=row_labels,
                cmap="YlOrRd", cbar_kws={"label": "αₖ"},
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Latent state", fontsize=10)
    ax.set_ylabel("Probe image",  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _build_probe_matrix(stim_preds: dict, joint_preds: dict,
                        probe_uids_ordered: list, all_tasks_ordered: list,
                        key: str = "pred") -> np.ndarray:
    """Build [n_probe × n_tasks] matrix; key='pred' or 'true'.

    stim_preds  — task → {pred, true, uids, ...}  (train tasks, probe images)
    joint_preds — task → {pred, true, uids, ...}  (val   tasks, probe images)
    """
    uid_to_row = {uid: i for i, uid in enumerate(probe_uids_ordered)}
    mat = np.full((len(probe_uids_ordered), len(all_tasks_ordered)), np.nan)
    for j, task_name in enumerate(all_tasks_ordered):
        src = stim_preds.get(task_name) or joint_preds.get(task_name)
        if src is None:
            continue
        vals = src[key]
        if key == "pred" and np.ndim(vals) == 2:
            vals = vals.mean(axis=0)      # average over seeds
        for uid, v in zip(src["uids"], vals):
            if uid in uid_to_row:
                mat[uid_to_row[uid], j] = float(v)
    return mat


def _plot_probe_matrix(mat: np.ndarray, row_labels: list, col_labels: list,
                       title: str, fname: str) -> None:
    """[n_probe × n_tasks] probe matrix heatmap."""
    n_tasks = len(col_labels)
    fig_w   = max(9, n_tasks * 0.20)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    sns.heatmap(mat, ax=ax,
                xticklabels=col_labels, yticklabels=row_labels,
                cmap="RdYlBu_r", vmin=0, vmax=1,
                cbar_kws={"label": "P(yes)"},
                linewidths=0.1, linecolor="white")
    ax.set_xlabel("Task", fontsize=10)
    ax.set_ylabel("Probe image", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=5, rotation=60)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# One-time setup: load image refs + CLIP feature cache (for α heatmap)
# ---------------------------------------------------------------------------
_refs_dict   = load_image_refs(cfg.METADATA)
_refs_by_uid = {r.uid: r for r in image_refs_as_list(_refs_dict)}

_frozen_clip: dict = {}
_cache_path = Path(cfg.CACHE_PATH)
if _cache_path.exists():
    _tmp = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                     device=torch.device("cpu"), mapper_hidden=cfg.MAPPER_HIDDEN)
    _tmp.load_cache(str(_cache_path))
    _frozen_clip = {uid: feat.clone() for uid, feat in _tmp._cache.items()}
    del _tmp
    print(f"Loaded CLIP cache ({len(_frozen_clip)} images) for α heatmaps.")
else:
    print(f"[warn] CLIP cache not found at {_cache_path} — α heatmaps will be skipped.")


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
        f"No results_*.pkl found in {cfg.RESULTS_DIR}. Run run.py first."
    )

for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    model_label    = res["model_label"]
    phase_boundary = res["phase_boundary"]
    best_epoch     = res["best_epoch"]
    noise_floors   = res.get("noise_floors", {})
    curves         = res["curves"]
    dlbt           = res["dlbt"]
    slda           = res.get("slda", {})
    n_seeds        = res.get("n_seeds", 1)

    # Task lists read from pickle — robust to config changes between runs.
    train_tasks = res.get("train_tasks", cfg.TRAIN_TASKS)
    val_tasks   = res.get("val_tasks",   cfg.VAL_TASKS)

    has_phase2 = phase_boundary < len(curves["train_nlls"]) - 1

    # -------------------------------------------------------------------
    # Plot 01 — learning curves
    # -------------------------------------------------------------------
    epochs = range(len(curves["train_nlls"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax_nll, ax_mse = axes

    ax_nll.plot(epochs, curves["train_nlls"], color=C_TRAIN, label="train", lw=1.2)
    ax_nll.plot(epochs, curves["eval_nlls"],  color=C_EVAL,  label="eval",  lw=1.2)
    ax_nll.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    if has_phase2:
        ax_nll.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
    ax_nll.set(ylabel="NLL", xlabel="epoch",
               title=f"{model_label} — NLL (train / eval)")
    ax_nll.legend(fontsize=8)
    ax_nll.set_ylim(bottom=0)

    ax_mse.plot(epochs, curves["train_mses"], color=C_TRAIN, label="train", lw=1.2)
    ax_mse.plot(epochs, curves["eval_mses"],  color=C_EVAL,  label="eval",  lw=1.2)
    for key, color, label in [
        ("stim_mses",  C_STIM,  "stim gen"),
        ("task_mses",  C_TASK,  "task gen"),
        ("joint_mses", C_JOINT, "joint gen"),
    ]:
        if key in curves:
            ax_mse.plot(epochs, curves[key], color=color, label=label, lw=1.0, alpha=0.7)
    ax_mse.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    if has_phase2:
        ax_mse.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
    for key, color in [("train", C_TRAIN), ("eval", C_EVAL),
                       ("stim_gen", C_STIM), ("task_gen", C_TASK), ("joint_gen", C_JOINT)]:
        if key in noise_floors:
            ax_mse.axhline(noise_floors[key], ls="--", color=color, alpha=0.3, lw=1)
    ax_mse.set(ylabel="cMSE", xlabel="epoch",
               title=f"{model_label} — cMSE (all regions)")
    ax_mse.legend(fontsize=8)
    ax_mse.set_ylim(bottom=0)

    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -------------------------------------------------------------------
    # Plots 02–05 — DLBT: separate summary scatter + per-task grid
    # -------------------------------------------------------------------
    for region_name, cond, task_list, nf_key, title in [
        ("02_train",     "train", train_tasks, "train",    "DLBT — Train"),
        ("02b_eval",     "eval",  train_tasks, "eval",     "DLBT — Eval"),
        ("03_stim_gen",  "stim",  train_tasks, "stim_gen", "DLBT — Stim Gen"),
        ("04_task_gen",  "task",  val_tasks,   "task_gen", "DLBT — Task Gen"),
        ("05_joint_gen", "joint", val_tasks,   "joint_gen","DLBT — Joint Gen"),
    ]:
        present = [t for t in task_list if t in dlbt.get(cond, {})]
        _plot_summary(
            dlbt, cond, present, run_tag, region_name,
            n_seeds=n_seeds, mc_n=cfg.N_MC,
            noise_floor_val=noise_floors.get(nf_key),
            title=title,
        )
        _plot_task_grid(dlbt, cond, task_list, run_tag, region_name,
                        n_seeds=n_seeds, mc_n=cfg.N_MC)

    # -------------------------------------------------------------------
    # Plot 06 — SLDA: same functions as DLBT (scatter + grid per condition)
    # -------------------------------------------------------------------
    if not slda or not any(slda.get(c) for c in ("train", "stim")):
        print("  Skipping SLDA plots (no predictions).")
        continue

    for slda_cond, slda_region, nf_key, title in [
        ("train", "06_slda_train", "train",    "SLDA — Train"),
        ("stim",  "06_slda_stim",  "stim_gen", "SLDA — Stim Gen"),
    ]:
        present = [t for t in train_tasks if t in slda.get(slda_cond, {})]
        _plot_summary(
            slda, slda_cond, present, run_tag, slda_region,
            n_seeds=1, mc_n=None,
            noise_floor_val=noise_floors.get(nf_key),
            title=title,
        )
        _plot_task_grid(slda, slda_cond, train_tasks, run_tag, slda_region,
                        n_seeds=1, mc_n=None)

    # -------------------------------------------------------------------
    # Plots 07–09 — α heatmap + probe matrices (true + predicted)
    # -------------------------------------------------------------------

    # Probe image ordering: sort by latent_state (same as 03_lbt)
    probe_uids_set     = set(res.get("probe_uids", res.get("test_uids", [])))
    probe_refs_ordered = sorted(
        [_refs_by_uid[uid] for uid in probe_uids_set if uid in _refs_by_uid],
        key=lambda r: r.latent_state,
    )
    probe_uids_ordered = [r.uid for r in probe_refs_ordered]
    image_labels       = [_state_label(r.latent_state) for r in probe_refs_ordered]

    all_tasks_ordered = sorted(
        list(set(train_tasks) | set(val_tasks)),
        key=lambda t: (_arity(t), t),
    )
    task_col_labels = [_label(t) for t in all_tasks_ordered]

    # -- Plot 07: α heatmap (requires agent checkpoint + CLIP cache) --
    agent_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if agent_path.exists() and _frozen_clip:
        _agent = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                           device=torch.device("cpu"), mapper_hidden=cfg.MAPPER_HIDDEN)
        _agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
        _agent._cache = {uid: feat.clone() for uid, feat in _frozen_clip.items()}
        _agent.eval()
        with torch.no_grad():
            alpha_mat = _agent.get_alpha(probe_refs_ordered).cpu().numpy()
        del _agent
        _plot_alpha_heatmap(
            alpha_mat, image_labels,
            title=f"Learned α  [{run_tag}]",
            fname=f"plot_07_alpha_{run_tag}.png",
        )
    else:
        print(f"  Skipping α heatmap: checkpoint or CLIP cache not available.")

    # -- Plots 08a/b: probe matrix (true + predicted) --
    stim_preds  = dlbt.get("stim",  {})
    joint_preds = dlbt.get("joint", {})

    if stim_preds or joint_preds:
        true_mat = _build_probe_matrix(
            stim_preds, joint_preds, probe_uids_ordered, all_tasks_ordered, key="true"
        )
        _plot_probe_matrix(
            true_mat, image_labels, task_col_labels,
            title="Probe matrix — empirical P(yes)",
            fname=f"plot_08a_probe_matrix_true_{run_tag}.png",
        )

        pred_mat = _build_probe_matrix(
            stim_preds, joint_preds, probe_uids_ordered, all_tasks_ordered, key="pred"
        )
        _plot_probe_matrix(
            pred_mat, image_labels, task_col_labels,
            title=f"Probe matrix — DLBT predicted P(yes)  [{run_tag}]",
            fname=f"plot_08b_probe_matrix_pred_{run_tag}.png",
        )
    else:
        print("  Skipping probe matrices: no stim/joint predictions in pkl.")

print(f"\nAll plots saved to {plots_dir}")
