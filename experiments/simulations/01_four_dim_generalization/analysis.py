"""
Simulation 01 — analysis and plots.

Loads results saved by run.py and generates five figures:
  plot_02_curves.png        — DLBT learning curves (NLL + cMSE)
  plot_03_summary.png       — 6-panel pred-vs-true scatter
  plot_04_per_task_dlbt.png — per-task scatter grid (DLBT)
  plot_05_per_task_slda.png — per-task scatter grid (SLDA)
  plot_06_latent_pca.png    — PCA of mapper outputs coloured by each latent dim
  plot_07_latent_tsne.png   — t-SNE of mapper outputs

Run from repo root:
    python experiments/simulations/01_four_dim_generalization/analysis.py
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config as cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
C_TRAIN, C_STIM, C_TASK, C_JOINT = cfg.C_TRAIN, cfg.C_STIM, cfg.C_TASK, cfg.C_JOINT

C_DLBT = "#C44F52"
C_SLDA = "#7D6EAE"
MARKERS = {"train": "o", "stim": "s", "task": "^", "joint": "D"}

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


def _summary_scatter(ax, pt: dict, task_names: list, color: str, marker: str,
                     title: str, mc_n=None, n_seeds=1, n_trials=100):
    """6-panel summary scatter with error bars.

    pt values have shape [n_seeds, n_pts] (DLBT) or [n_pts] (SLDA).
    """
    all_preds = np.concatenate(
        [pt[t]["pred"] for t in task_names if t in pt], axis=-1
    )  # [n_seeds, n_pts] or [n_pts]
    all_trues = np.concatenate(
        [pt[t]["true"] for t in task_names if t in pt]
    )  # [n_pts]

    # Handle both [n_seeds, n_pts] (DLBT) and [n_pts] (SLDA)
    if all_preds.ndim == 2:
        pred_mean = all_preds.mean(axis=0)
        pred_sem  = all_preds.std(axis=0) / np.sqrt(n_seeds)
    else:
        pred_mean = all_preds
        pred_sem  = np.zeros_like(pred_mean)

    true_sem = np.sqrt(np.clip(all_trues * (1 - all_trues), 0, None) / n_trials)

    # cMSE / rho on mean predictions
    raw  = float(np.mean((pred_mean - all_trues) ** 2))
    cmse = raw - float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1) if mc_n else raw
    rho, _ = spearmanr(pred_mean, all_trues)

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.errorbar(pred_mean, all_trues,
                xerr=pred_sem, yerr=true_sem,
                fmt=marker, ms=4, alpha=0.1, color=color,
                elinewidth=0.5, capsize=0, linewidth=0)
    ax.set_title(f"{title}\ncMSE={cmse:.4f}   ρ={rho:.3f}", fontsize=10, pad=4)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# CLI: optional --tag to restrict which result file to process
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None, choices=["frozen", "attnpool"],
                    help="Process only this tag (default: all available)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Auto-detect available result files (new format only — must have "dlbt" key)
# ---------------------------------------------------------------------------
candidates = sorted([
    p for p in [
        cfg.RESULTS_DIR / "results_frozen.pkl",
        cfg.RESULTS_DIR / "results_attnpool.pkl",
    ] if p.exists()
])
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]

available = []
for p in candidates:
    with open(p, "rb") as _f:
        _r = pickle.load(_f)
    if "dlbt" in _r:
        available.append(p)
    else:
        print(f"Skipping {p.name} — old format (missing 'dlbt' key), re-run run.py first.")

if not available:
    raise FileNotFoundError(
        f"No compatible results files found in {cfg.RESULTS_DIR}. Run run.py first."
    )

for results_path in available:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== Processing: {results_path.name} (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    model_label    = res["model_label"]
    phase_boundary = res["phase_boundary"]
    best_epoch     = res["best_epoch"]
    noise_floor    = res["noise_floor"]
    curves         = res["curves"]
    dlbt           = res["dlbt"]    # {cond: {task: {pred: [n_seeds, n_pts], true, uids}}}
    slda           = res["slda"]    # {cond: {task: {pred: [n_pts], true, uids}}}

    # Backward compat: old pkls stored the simple lateral task as "left_right";
    # current code uses "right".
    for _preds in (dlbt, slda):
        for _cond_dict in _preds.values():
            if "left_right" in _cond_dict and "right" not in _cond_dict:
                _cond_dict["right"] = _cond_dict.pop("left_right")
    n_seeds        = res.get("n_seeds", 1)
    n_trials       = res.get("n_trials", cfg.N_TRIALS)

    has_phase2 = phase_boundary < len(curves["train_nlls"]) - 1

    # -----------------------------------------------------------------------
    # Plot 2 — learning curves
    # -----------------------------------------------------------------------
    epochs = range(len(curves["train_nlls"]))

    fig, (ax_nll, ax_mse) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, tr, vl, tg, jg, ylabel in [
        (ax_nll,
         curves["train_nlls"], curves["val_nlls"],
         curves["task_nlls"],  curves["joint_nlls"], "NLL"),
        (ax_mse,
         curves["train_mses"], curves["val_mses"],
         curves["task_mses"],  curves["joint_mses"], "cMSE"),
    ]:
        ax.plot(epochs, tr, color=C_TRAIN, label="train",    lw=1.2)
        ax.plot(epochs, vl, color=C_STIM,  label="stim gen", lw=1.2)
        ax.plot(epochs, tg, color=C_TASK,  label="task gen", lw=1.2)
        ax.plot(epochs, jg, color=C_JOINT, label="joint gen",lw=1.2)
        ax.axvline(best_epoch, ls=":", color="gray", lw=0.8)
        if has_phase2:
            ax.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
            ax.text(phase_boundary + 1, 0.98, "phase 2", fontsize=7,
                    va="top", transform=ax.get_xaxis_transform(), color="black", alpha=0.6)
        ax.set(ylabel=ylabel, xlabel="epoch", title=f"{model_label} — {ylabel}")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    ax_mse.axhline(noise_floor, ls="--", color=C_TRAIN, alpha=0.4, lw=1)
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_02_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3 — 6-panel summary scatter
    # -----------------------------------------------------------------------
    # (pred_dict, task_names, color, marker, title, mc_n, row, col)
    panels = [
        (slda["train"], cfg.TRAIN_TASKS, C_SLDA, MARKERS["train"], "SLDA — Train",              None,     0, 0),
        (slda["stim"],  cfg.TRAIN_TASKS, C_SLDA, MARKERS["stim"],  "SLDA — Stim gen",           None,     1, 0),
        (dlbt["train"], cfg.TRAIN_TASKS, C_DLBT, MARKERS["train"], f"{model_label} — Train",    cfg.N_MC, 0, 1),
        (dlbt["stim"],  cfg.TRAIN_TASKS, C_DLBT, MARKERS["stim"],  f"{model_label} — Stim gen", cfg.N_MC, 1, 1),
        (dlbt["task"],  cfg.VAL_TASKS,   C_DLBT, MARKERS["task"],  f"{model_label} — Task gen", cfg.N_MC, 0, 2),
        (dlbt["joint"], cfg.VAL_TASKS,   C_DLBT, MARKERS["joint"], f"{model_label} — Joint gen",cfg.N_MC, 1, 2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.52, "wspace": 0.10})
    for pt, task_names, color, marker, title, mc_n, row, col in panels:
        ax = axes[row, col]
        _summary_scatter(ax, pt, task_names, color, marker, title, mc_n,
                         n_seeds=n_seeds, n_trials=n_trials)

    # Single shared axis labels
    fig.supxlabel("Predicted P(right)", fontsize=12, y=0.01)
    fig.supylabel("True P(right)", fontsize=12, x=0.01)

    sns.despine(fig=fig, trim=False)
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    out = plots_dir / f"plot_03_summary_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 4 — per-task DLBT grid
    # -----------------------------------------------------------------------
    ALL_TASKS = cfg.TRAIN_TASKS + cfg.VAL_TASKS
    N_COLS    = 8
    N_ROWS    = math.ceil(len(ALL_TASKS) / N_COLS)

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.0, N_ROWS * 2.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for ax in axes.flat[len(ALL_TASKS):]:
        ax.set_visible(False)

    for idx, (ax, task_name) in enumerate(zip(axes.flat, ALL_TASKS)):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        is_val = task_name in cfg.VAL_TASKS
        if not is_val:
            for cond, color in [("train", C_TRAIN), ("stim", C_STIM)]:
                if task_name in dlbt[cond]:
                    d         = dlbt[cond][task_name]
                    pred_mean = d["pred"].mean(axis=0)
                    pred_sem  = d["pred"].std(axis=0) / np.sqrt(n_seeds)
                    true_vals = d["true"]
                    true_sem  = np.sqrt(true_vals * (1 - true_vals) / n_trials)
                    ax.errorbar(pred_mean, true_vals,
                                xerr=pred_sem, yerr=true_sem,
                                fmt='o', ms=3, alpha=0.2, color=color,
                                elinewidth=0.4, capsize=0, linewidth=0)
            # rho on mean predictions
            def _rho_mean(cond, tn):
                if tn not in dlbt[cond]:
                    return float("nan")
                d = dlbt[cond][tn]
                pm = d["pred"].mean(axis=0)
                r, _ = spearmanr(pm, d["true"])
                return r
            ax.text(0.05, 0.93, f"ρ={_rho_mean('train', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
            ax.text(0.05, 0.78, f"ρ={_rho_mean('stim', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")
        else:
            for cond, color in [("task", C_TASK), ("joint", C_JOINT)]:
                if task_name in dlbt[cond]:
                    d         = dlbt[cond][task_name]
                    pred_mean = d["pred"].mean(axis=0)
                    pred_sem  = d["pred"].std(axis=0) / np.sqrt(n_seeds)
                    true_vals = d["true"]
                    true_sem  = np.sqrt(true_vals * (1 - true_vals) / n_trials)
                    ax.errorbar(pred_mean, true_vals,
                                xerr=pred_sem, yerr=true_sem,
                                fmt='o', ms=3, alpha=0.2, color=color,
                                elinewidth=0.4, capsize=0, linewidth=0)
            def _rho_mean_val(cond, tn):
                if tn not in dlbt[cond]:
                    return float("nan")
                d = dlbt[cond][tn]
                pm = d["pred"].mean(axis=0)
                r, _ = spearmanr(pm, d["true"])
                return r
            ax.text(0.05, 0.93, f"ρ={_rho_mean_val('task', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_TASK, va="top")
            ax.text(0.05, 0.78, f"ρ={_rho_mean_val('joint', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_JOINT, va="top")

        ax.set_title(task_name.replace("_and_", " & ").replace("_", "/"), fontsize=7, pad=2)
        row, col = divmod(idx, N_COLS)
        if row == N_ROWS - 1:
            ax.set_xlabel("Pred", fontsize=7)
        if col == 0:
            ax.set_ylabel("True", fontsize=7)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    fig.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c, markersize=5, label=l)
        for c, l in [(C_TRAIN,"train"),(C_STIM,"stim gen"),(C_TASK,"task gen"),(C_JOINT,"joint gen")]
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, frameon=False, ncol=2)
    fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_04_per_task_dlbt_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 5 — per-task SLDA grid
    # -----------------------------------------------------------------------
    N_COLS_S = 8
    N_ROWS_S = math.ceil(len(cfg.TRAIN_TASKS) / N_COLS_S)

    fig, axes = plt.subplots(N_ROWS_S, N_COLS_S, figsize=(N_COLS_S * 2.0, N_ROWS_S * 2.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for ax in axes.flat[len(cfg.TRAIN_TASKS):]:
        ax.set_visible(False)

    for idx, (ax, task_name) in enumerate(zip(axes.flat, cfg.TRAIN_TASKS)):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        for cond, color in [("train", C_TRAIN), ("stim", C_STIM)]:
            if task_name in slda[cond]:
                d = slda[cond][task_name]
                # pred is [n_pts] (no seed dimension) — no pred_sem
                true_sem = np.sqrt(d["true"] * (1 - d["true"]) / n_trials)
                ax.errorbar(d["pred"], d["true"],
                            yerr=true_sem,
                            fmt="s", ms=3, alpha=0.2, color=color,
                            elinewidth=0.4, capsize=0, linewidth=0)
        def _rho_slda(cond, tn):
            if tn not in slda[cond]:
                return float("nan")
            d = slda[cond][tn]
            r, _ = spearmanr(d["pred"], d["true"])
            return r
        ax.text(0.05, 0.93, f"ρ={_rho_slda('train', task_name):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
        ax.text(0.05, 0.78, f"ρ={_rho_slda('stim', task_name):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")
        ax.set_title(task_name.replace("_and_", " & ").replace("_", "/"), fontsize=7, pad=2)
        row, col = divmod(idx, N_COLS_S)
        if row == N_ROWS_S - 1:
            ax.set_xlabel("Pred", fontsize=7)
        if col == 0:
            ax.set_ylabel("True", fontsize=7)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    fig.legend(handles=[
        Line2D([0],[0], marker="s", color="w", markerfacecolor=c, markersize=5, label=l)
        for c, l in [(C_TRAIN,"train"),(C_STIM,"stim gen")]
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, frameon=False)
    fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_05_per_task_slda_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 6 — latent space PCA
    # -----------------------------------------------------------------------
    agent_path = cfg.RESULTS_DIR / f"agent_{run_tag}.pt"
    if not agent_path.exists():
        print(f"Agent weights not found at {agent_path} — skipping latent PCA plot.")
    else:
        import torch
        from dlbt.agents.dlbt import DlbtAgent
        from dlbt.data.image_ref import load_image_refs, image_refs_as_list

        # Load agent
        _device = torch.device("cpu")
        _agent  = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                            device=_device, mapper_hidden=cfg.MAPPER_HIDDEN)
        _agent.load_state_dict(torch.load(agent_path, map_location="cpu"))

        _cache_path = Path(cfg.CACHE_PATH)
        if _cache_path.exists():
            _agent.load_cache(str(_cache_path))
        else:
            _refs_all = image_refs_as_list(load_image_refs(cfg.METADATA))
            _agent.precompute_features(_refs_all)

        _agent.eval()

        # All image refs + continuous metadata
        _refs_dict = load_image_refs(cfg.METADATA)
        _all_refs  = image_refs_as_list(_refs_dict)

        _cont: dict = {}
        with open(cfg.METADATA) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _rec = json.loads(_line)
                _z   = _rec["z"]
                _cont[_rec["id"]] = dict(
                    x            = _z["pos_xy"][0],
                    transparency = _z["transparency"],
                    glossiness   = _z["glossiness"],
                    scale        = _z["scale"],
                )

        # Mapper outputs → Dirichlet means
        with torch.no_grad():
            _alpha = _agent.get_alpha(_all_refs).cpu().numpy()   # [N, K]
        _q = _alpha / _alpha.sum(axis=1, keepdims=True)          # Dirichlet mean

        # PCA into 4D — PC1/PC2 capture easy dims (position, scale);
        # PC3/PC4 may encode harder material dims (transparency, gloss).
        _pca4    = PCA(n_components=4)
        _coords4 = _pca4.fit_transform(_q)          # [N, 4]
        _var4    = _pca4.explained_variance_ratio_

        # Layout: 2 rows × 2 cols
        #   Row 0: PC1 vs PC2, coloured by x (left/right) and scale
        #   Row 1: PC3 vs PC4, coloured by transparency and glossiness
        _pca_panels = [
            # (pc_x_idx, pc_y_idx, key,            title,            cmap,       vmin, vmax)
            (0, 1, "x",            "Left / Right\n(PC1 vs PC2)",  "coolwarm", None, None),
            (0, 1, "scale",        "Large / Small\n(PC1 vs PC2)", "cividis",  0.0,  1.0),
            (2, 3, "transparency", "Transparent\n(PC3 vs PC4)",   "viridis",  0.0,  1.0),
            (2, 3, "glossiness",   "Glossy\n(PC3 vs PC4)",        "plasma",   0.0,  1.0),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(8, 7),
                                 gridspec_kw={"wspace": 0.35, "hspace": 0.45})
        for ax, (xi, yi, key, title, cmap, vmin, vmax) in zip(axes.flat, _pca_panels):
            _vals = np.array([_cont[r.uid][key] for r in _all_refs])
            sc = ax.scatter(_coords4[:, xi], _coords4[:, yi],
                            c=_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=10, alpha=0.7, linewidths=0)
            plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f"PC{xi+1} ({_var4[xi]:.1%})", fontsize=8)
            ax.set_ylabel(f"PC{yi+1} ({_var4[yi]:.1%})", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Mapper latent space — PCA of Dirichlet means  ({model_label})",
                     fontsize=11)
        sns.despine(fig=fig, trim=True)
        plt.tight_layout()
        out = plots_dir / f"plot_06_latent_pca_{run_tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

        # -------------------------------------------------------------------
        # Plot 7 — t-SNE of first 8 PCA components
        # PCA first to denoise (captures most variance in fewer dims),
        # then t-SNE reveals nonlinear cluster structure.
        # -------------------------------------------------------------------
        _n_pca8   = min(8, _q.shape[1] - 1)
        _pca8     = PCA(n_components=_n_pca8)
        _q_pca8   = _pca8.fit_transform(_q)                     # [N, n_pca8]
        _cum_var8 = _pca8.explained_variance_ratio_.cumsum()

        # n_iter was renamed max_iter in sklearn >= 1.5
        import sklearn
        _tsne_kwargs = dict(
            n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=42,
        )
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
            _tsne_kwargs["max_iter"] = 1000
        else:
            _tsne_kwargs["n_iter"] = 1000
        _tsne        = TSNE(**_tsne_kwargs)
        _tsne_coords = _tsne.fit_transform(_q_pca8)             # [N, 2]

        # Same 4-panel layout as PCA, same colormaps
        _tsne_panels = [
            ("x",            "Left / Right",  "coolwarm", None, None),
            ("scale",        "Large / Small", "cividis",  0.0,  1.0),
            ("transparency", "Transparent",   "viridis",  0.0,  1.0),
            ("glossiness",   "Glossy",        "plasma",   0.0,  1.0),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6),
                                 gridspec_kw={"wspace": 0.35})
        for ax, (key, title, cmap, vmin, vmax) in zip(axes, _tsne_panels):
            _vals = np.array([_cont[r.uid][key] for r in _all_refs])
            sc = ax.scatter(_tsne_coords[:, 0], _tsne_coords[:, 1],
                            c=_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=10, alpha=0.7, linewidths=0)
            plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"Mapper latent space — t-SNE  "
            f"(PCA {_n_pca8}D -> 2D, {_cum_var8[-1]:.0%} var.)  ({model_label})",
            fontsize=11, y=1.02,
        )
        sns.despine(fig=fig, left=True, bottom=True)
        plt.tight_layout()
        out = plots_dir / f"plot_07_latent_tsne_{run_tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

print("\nAll plots saved to", plots_dir)
