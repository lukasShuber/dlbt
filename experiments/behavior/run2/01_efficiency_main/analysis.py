"""
run2/01_efficiency_main/analysis.py — trials-per-task budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trials per task) for every
efficiency_main_01*.pkl found in results/.

  Traces (mean ± SEM across seeds):
    • DLBT         — red solid
    • SLDA         — purple solid
    • Anti-human   — gray solid

  Reference lines:
    • chance (P=0.5) — gray dashed, annotated at right edge (cMSE only)
    • Noise ceiling  — dark gray dotted, annotated at left edge (ρ only)

  Markers:
    • Budget grid points — open circular markers, connected by lines
    • All-data point     — filled marker, plotted separately (not connected)

Run from repo root:
    python experiments/behavior/run2/01_efficiency_main/analysis.py
    python experiments/behavior/run2/01_efficiency_main/analysis.py --pkl PATH
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr as _spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to a specific pkl. Default: all efficiency_main_01*.pkl.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis on cMSE plot (also set via cfg.LOG_Y).")
args = parser.parse_args()

_log_y = cfg.LOG_Y or args.log_y

# ---------------------------------------------------------------------------
# Collect pkl paths
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_paths = [Path(args.pkl)]
else:
    pkl_paths = sorted(cfg.RESULTS_DIR.glob("efficiency_main_01*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No efficiency_main_01*.pkl found in {cfg.RESULTS_DIR}")

print(f"Processing {len(pkl_paths)} pkl(s):")
for p in pkl_paths:
    print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    mu  = np.nanmean(arr, axis=0)
    n   = np.sum(~np.isnan(arr), axis=0).astype(float)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mu, sem


def _mean_sem_scalar(arr: np.ndarray):
    n   = float(np.sum(~np.isnan(arr)))
    mu  = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(max(n, 1)))
    return mu, sem


def _plot_trace(ax, tpt_grid, mu, sem, color, ls="-", zorder=3):
    ax.fill_between(tpt_grid, mu - sem, mu + sem,
                    color=color, alpha=0.15, zorder=zorder - 1)
    ax.plot(tpt_grid, mu, color=color, lw=2.0, ls=ls, zorder=zorder)
    ax.plot(tpt_grid, mu, "o", color=color, ms=5, mfc="none",
            mew=1.4, zorder=zorder + 1)


def _plot_all_data_marker(ax, x, mu, sem, color, zorder=5):
    ax.errorbar(x, mu, yerr=sem, fmt="o", color=color,
                ms=7, mfc=color, mew=1.4, capsize=3,
                elinewidth=1.2, zorder=zorder)


def _xaxis_setup(ax, tpt_grid, avg_pool_per_task):
    ax.set_xscale("log")
    x_upper = max(avg_pool_per_task * 2.5, float(tpt_grid[-1]) * 3.0)
    ax.set_xlim(7, x_upper)
    ax.set_xticks([10, 100, 1_000])
    ax.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$"])
    ax.set_xlabel("Trials per task", fontsize=11, fontweight="bold")


def _rho_nc_from_counts(true_mat, count_mat, n_splits=200, seed=0):
    mask    = count_mat > 1
    totals  = count_mat[mask].astype(int)
    count1s = np.round(true_mat[mask] * totals).astype(int)
    n1s     = totals // 2
    n2s     = totals - n1s
    if len(totals) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        k1 = np.array([rng.hypergeometric(c1, t - c1, n1)
                       for c1, t, n1 in zip(count1s, totals, n1s)], dtype=float)
        p1 = k1 / n1s
        p2 = (count1s - k1) / n2s
        rh, _ = _spearmanr(p1, p2)
        if not np.isnan(rh) and rh > -1:
            vals.append((2 * rh) / (1 + rh))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Scatter-plot helpers: predicted vs. empirical, from saved checkpoints
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[4]   # …/run2/01_efficiency_main → repo root


def _task_arity(name: str) -> int:
    return name.count("_and_") + 1


def _try_load_pred_matrices(pkl_path: Path, d: dict):
    """
    Load saved all-data DLBT and SLDA checkpoints, reconstruct probe-matrix
    predictions, and return their seed-averaged means.

    Returns (dlbt_mean, slda_mean) — each [n_probe × n_tasks] or None.
    """
    models_dir = cfg.RESULTS_DIR / "models" / pkl_path.stem
    if not models_dir.exists():
        print(f"  [scatter] no models dir '{models_dir.name}' — skipping")
        return None, None

    seeds              = d["seeds"]
    all_tasks_ordered  = d["all_tasks_ordered"]
    probe_uids_ordered = d["probe_uids_ordered"]
    uid_to_row         = {uid: i for i, uid in enumerate(probe_uids_ordered)}
    n_probe            = len(probe_uids_ordered)
    n_tasks            = len(all_tasks_ordered)

    first_dlbt = models_dir / f"seed{seeds[0]}_all_dlbt.pt"
    first_slda = models_dir / f"seed{seeds[0]}_all_slda.pkl"
    if not first_dlbt.exists():
        print(f"  [scatter] {first_dlbt.name} not found — skipping")
        return None, None

    # Peek to detect whether attnpool was fine-tuned
    ck0 = torch.load(first_dlbt, map_location="cpu", weights_only=False)
    has_attnpool_dlbt = ("attnpool_state_dict" in ck0
                         and ck0["attnpool_state_dict"] is not None)
    del ck0
    has_attnpool_slda = False
    if first_slda.exists():
        with open(first_slda, "rb") as fh:
            art0 = pickle.load(fh)
        has_attnpool_slda = art0.get("attnpool_state_dict") is not None
        del art0

    try:
        from dlbt.agents.dlbt import DlbtAgent as _DLBT
        from dlbt.agents.slda import SldaAgent as _SLDA
        from dlbt.data.image_ref import load_image_refs as _lir, image_refs_as_list as _iral
        from dlbt.data.task import get_task as _get_task
        from dlbt.training.train_slda import slda_probe_matrix as _slda_pm
    except ImportError as exc:
        print(f"  [scatter] dlbt import failed ({exc}) — skipping")
        return None, None

    _dev    = torch.device("cpu")
    by_uid  = {r.uid: r for r in _iral(_lir(_REPO_ROOT / cfg.METADATA))}
    p_refs  = [by_uid[uid] for uid in probe_uids_ordered if uid in by_uid]

    # ── Shared backbone features (frozen backbone — same for both models) ──
    backbone_cache: dict = {}
    frozen_clip:    dict = {}
    needs_backbone = has_attnpool_dlbt or has_attnpool_slda

    if needs_backbone:
        print(f"  [scatter] computing backbone features for {len(p_refs)} probe images…")
        _tmp = _DLBT(freeze_encoder=False, n_mc_samples=1, device=_dev,
                     normalize_utility=cfg.NORMALIZED_UTILITY)
        _tmp.precompute_backbone_features(p_refs)
        backbone_cache = dict(_tmp._backbone_cache)
        del _tmp

    if not has_attnpool_dlbt or not has_attnpool_slda:
        cache_p = _REPO_ROOT / cfg.CACHE_PATH
        if cache_p.exists():
            frozen_clip = torch.load(str(cache_p), map_location="cpu")

    def _attnpool_feats(module, refs):
        """Apply an attnpool module to cached backbone maps → {uid: tensor}."""
        spatial = torch.stack([backbone_cache[r.uid] for r in refs]).to(_dev)
        with torch.no_grad():
            out = module(spatial).float()
        return {r.uid: out[i].cpu() for i, r in enumerate(refs)}

    # ── DLBT: one prediction matrix per seed ─────────────────────────────
    dlbt_preds = []
    for seed in seeds:
        ckpt_p = models_dir / f"seed{seed}_all_dlbt.pt"
        if not ckpt_p.exists():
            continue
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)

        if ckpt["used_base"]:
            dlbt_preds.append(np.full((n_probe, n_tasks), 0.5))
            continue

        agent = _DLBT(freeze_encoder=True, n_mc_samples=cfg.N_MC, device=_dev,
                      normalize_utility=cfg.NORMALIZED_UTILITY)
        agent.mapper.load_state_dict(ckpt["mapper_state_dict"])

        if has_attnpool_dlbt:
            agent.encoder.attnpool.load_state_dict(ckpt["attnpool_state_dict"])
            agent._cache = _attnpool_feats(agent.encoder.attnpool, p_refs)
        else:
            agent._cache = {uid: frozen_clip[uid].clone()
                            for uid in probe_uids_ordered if uid in frozen_clip}

        agent.eval()
        pred = np.full((n_probe, n_tasks), np.nan)
        with torch.no_grad():
            for j, tn in enumerate(all_tasks_ordered):
                pred[:, j] = agent.choice_probs(p_refs, _get_task(tn))[:, 1].cpu().numpy()
        dlbt_preds.append(pred)
        del agent

    # ── SLDA: one prediction matrix per seed ─────────────────────────────
    slda_preds = []
    for seed in seeds:
        ckpt_p = models_dir / f"seed{seed}_all_slda.pkl"
        if not ckpt_p.exists():
            continue
        with open(ckpt_p, "rb") as fh:
            art = pickle.load(fh)

        ap_sd = art.get("attnpool_state_dict")
        if has_attnpool_slda and ap_sd is not None:
            slda_tmp = _SLDA(freeze_encoder=False, device=_dev)
            slda_tmp.encoder.attnpool.load_state_dict(ap_sd)
            probe_feats = {uid: t.numpy()
                           for uid, t in _attnpool_feats(
                               slda_tmp.encoder.attnpool, p_refs).items()}
            del slda_tmp
        else:
            probe_feats = {uid: frozen_clip[uid].numpy()
                           for uid in probe_uids_ordered if uid in frozen_clip}

        pred = _slda_pm(art["scalers"], art["models"], art["use_base"],
                        probe_feats, all_tasks_ordered, uid_to_row, n_probe)
        slda_preds.append(pred)

    print(f"  [scatter] loaded {len(dlbt_preds)} DLBT seeds, {len(slda_preds)} SLDA seeds")

    dlbt_mean = np.nanmean(np.array(dlbt_preds), axis=0) if dlbt_preds else None
    slda_mean = np.nanmean(np.array(slda_preds), axis=0) if slda_preds else None
    return dlbt_mean, slda_mean


def _make_scatter(d: dict, pred_dlbt, pred_slda, plots_dir: Path):
    """
    Two-panel scatter (DLBT | SLDA): predicted vs. empirical P(right) for
    every valid probe × task pair, coloured by task arity.
    """
    true_matrix       = d["true_matrix"]
    all_tasks         = d["all_tasks_ordered"]
    probe_noise_floor = d["probe_noise_floor"]

    arities   = np.array([_task_arity(t) for t in all_tasks])
    ar_vals   = sorted(set(arities))
    AR_COLOR  = getattr(cfg, "ARITY_COLOR",
                        {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"})

    model_list = [(nm, pr, col) for nm, pr, col in [
        ("DLBT", pred_dlbt, cfg.C_DLBT),
        ("SLDA", pred_slda, cfg.C_SLDA),
    ] if pr is not None]
    if not model_list:
        return

    ncols      = len(model_list)
    fig, axes  = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2))
    if ncols == 1:
        axes = [axes]

    for ax, (name, pred, line_col) in zip(axes, model_list):
        row_i, col_i = np.where(~np.isnan(pred) & ~np.isnan(true_matrix))
        x = pred[row_i, col_i]
        y = true_matrix[row_i, col_i]
        task_ar = arities[col_i]

        for ar in ar_vals:
            m = task_ar == ar
            ax.scatter(x[m], y[m], s=3, alpha=0.30, linewidths=0,
                       color=AR_COLOR.get(ar, "#888"),
                       rasterized=True, zorder=3)

        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls="--", zorder=2)

        rho, _ = _spearmanr(x, y)
        cmse   = float(np.mean((x - y) ** 2)) - probe_noise_floor

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(right)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Empirical P(right)", fontsize=10, fontweight="bold")
        ax.set_title(name, fontsize=11, fontweight="bold", color=line_col)
        ax.text(0.04, 0.96,
                f"ρ = {rho:.3f}\ncMSE−NF = {cmse:.4f}",
                transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", color="#333333")
        sns.despine(ax=ax, top=True, right=True)

    handles = [mpatches.Patch(color=AR_COLOR[a], label=f"{a}-way") for a in ar_vals]
    axes[0].legend(handles=handles, fontsize=7, frameon=False,
                   loc="lower right", title="arity", title_fontsize=7)

    plt.tight_layout()
    out = plots_dir / "plot_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(cfg.RESULTS_DIR)}")


# ---------------------------------------------------------------------------
# Per-pkl processing
# ---------------------------------------------------------------------------

def process_pkl(pkl_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading: {pkl_path.name}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    tpt_grid          = np.array(d["trials_per_task"])
    avg_pool_per_task = d["avg_pool_per_task"]
    seeds             = d["seeds"]
    n_seeds           = len(seeds)

    dlbt_cmse = d["dlbt_cmse"]
    dlbt_rho  = d["dlbt_rho"]
    slda_cmse = d["slda_cmse"]
    slda_rho  = d["slda_rho"]
    anti_cmse = d["anti_cmse"]
    anti_rho  = d["anti_rho"]

    dlbt_all_cmse = d["dlbt_all_cmse"]
    dlbt_all_rho  = d["dlbt_all_rho"]
    slda_all_cmse = d["slda_all_cmse"]
    slda_all_rho  = d["slda_all_rho"]
    anti_all_cmse = d["anti_all_cmse"]
    anti_all_rho  = d["anti_all_rho"]

    random_cmse_nf    = d["random_cmse_nf"]
    rho_noise_ceiling = d.get("rho_noise_ceiling", float("nan"))

    if np.isnan(rho_noise_ceiling) and "count_matrix" in d:
        print("  Computing ρ noise ceiling from count_matrix...")
        rho_noise_ceiling = _rho_nc_from_counts(d["true_matrix"], d["count_matrix"])
        print(f"  ρ noise ceiling: {rho_noise_ceiling:.4f}")

    print(f"  Seeds: {n_seeds}  TPT grid: {list(tpt_grid)}  "
          f"avg pool/task: {avg_pool_per_task:.0f}")

    plots_dir = cfg.RESULTS_DIR / "plots" / pkl_path.stem
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _make_figure(metric: str):
        is_cmse = metric == "cmse"

        dlbt_mu, dlbt_sem = _mean_sem(dlbt_cmse if is_cmse else dlbt_rho)
        slda_mu, slda_sem = _mean_sem(slda_cmse if is_cmse else slda_rho)
        anti_mu, anti_sem = _mean_sem(anti_cmse if is_cmse else anti_rho)

        dlbt_all_mu, dlbt_all_sem = _mean_sem_scalar(
            dlbt_all_cmse if is_cmse else dlbt_all_rho)
        slda_all_mu, slda_all_sem = _mean_sem_scalar(
            slda_all_cmse if is_cmse else slda_all_rho)
        anti_all_mu, anti_all_sem = _mean_sem_scalar(
            anti_all_cmse if is_cmse else anti_all_rho)

        fig, ax = plt.subplots(figsize=(5.0, 4.5))

        # ── Reference lines ──────────────────────────────────────────────────
        if is_cmse:
            ax.axhline(random_cmse_nf, color=cfg.C_RNDINI, lw=1.5,
                       ls=(0, (4, 3)), zorder=1)
            ax.annotate("chance (P=0.5)",
                        xy=(1.0, random_cmse_nf),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 5), textcoords="offset points",
                        color=cfg.C_RNDINI, fontsize=8, style="italic",
                        va="bottom", ha="right", zorder=6)
        else:
            if not np.isnan(rho_noise_ceiling):
                ax.axhline(rho_noise_ceiling, color="#555555", lw=1.5,
                           ls=(0, (2, 2)), zorder=2)
                ax.annotate("noise ceiling",
                            xy=(0.0, rho_noise_ceiling),
                            xycoords=("axes fraction", "data"),
                            xytext=(4, 5), textcoords="offset points",
                            color="#555555", fontsize=8, style="italic",
                            va="bottom", ha="left", zorder=6)

        # ── Traces ───────────────────────────────────────────────────────────
        _plot_trace(ax, tpt_grid, anti_mu, anti_sem, cfg.C_ANTI, zorder=2)
        _plot_trace(ax, tpt_grid, slda_mu, slda_sem, cfg.C_SLDA, zorder=3)
        _plot_trace(ax, tpt_grid, dlbt_mu, dlbt_sem, cfg.C_DLBT, zorder=4)

        # ── All-data markers ─────────────────────────────────────────────────
        _plot_all_data_marker(ax, avg_pool_per_task,
                              anti_all_mu, anti_all_sem, cfg.C_ANTI, zorder=4)
        _plot_all_data_marker(ax, avg_pool_per_task,
                              slda_all_mu, slda_all_sem, cfg.C_SLDA, zorder=5)
        _plot_all_data_marker(ax, avg_pool_per_task,
                              dlbt_all_mu, dlbt_all_sem, cfg.C_DLBT, zorder=6)

        _xaxis_setup(ax, tpt_grid, avg_pool_per_task)

        # ── Y axis ───────────────────────────────────────────────────────────
        if is_cmse:
            ax.set_ylabel("cMSE − noise floor", fontsize=11, fontweight="bold")
            if _log_y:
                ax.set_yscale("log")
                ax.set_yticks([0.01, 0.1, 1])
                ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
                ax.yaxis.set_minor_locator(plt.NullLocator())
                ax.set_ylim(0.008, 1.0)
            else:
                ax.set_ylim(0, 0.34)
        else:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=11, fontweight="bold")
            ax.set_ylim(-1, 1)

        # ── Stacked annotations bottom-left (cMSE only) ─────────────────────
        if is_cmse:
            for k, (lbl, col) in enumerate([
                ("anti-human DLBT", cfg.C_ANTI),
                ("SLDA",            cfg.C_SLDA),
                ("DLBT",            cfg.C_DLBT),
            ]):
                ax.text(0.03, 0.03 + k * 0.045, lbl,
                        transform=ax.transAxes,
                        color=col, fontsize=8, fontweight="bold", style="italic",
                        va="bottom", ha="left", zorder=6)

        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.tight_layout()

        tag = "cmse" if is_cmse else "rho"
        out = plots_dir / f"plot_{tag}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.relative_to(cfg.RESULTS_DIR)}")

    _make_figure("cmse")
    _make_figure("rho")

    # ── Scatter: predicted vs. empirical (all-data / full model) ─────────────
    pred_dlbt, pred_slda = _try_load_pred_matrices(pkl_path, d)
    _make_scatter(d, pred_dlbt, pred_slda, plots_dir)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print(f"  {'Model':<20}  {'tpt':>8}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}")
    print("  " + "-" * 70)

    for label, cmse_arr, rho_arr, cmse_all, rho_all in [
        ("DLBT",       dlbt_cmse, dlbt_rho, dlbt_all_cmse, dlbt_all_rho),
        ("SLDA",       slda_cmse, slda_rho, slda_all_cmse, slda_all_rho),
        ("Anti-human", anti_cmse, anti_rho, anti_all_cmse, anti_all_rho),
    ]:
        for b_i, tpt in enumerate(tpt_grid):
            mu_c, sem_c = _mean_sem_scalar(cmse_arr[:, b_i])
            mu_r, sem_r = _mean_sem_scalar(rho_arr[:, b_i])
            print(f"  {label:<20}  {int(tpt):>8,}  "
                  f"{mu_c:+.5f} ± {sem_c:.5f}  "
                  f"{mu_r:+.4f} ± {sem_r:.4f}")
        mu_c, sem_c = _mean_sem_scalar(cmse_all)
        mu_r, sem_r = _mean_sem_scalar(rho_all)
        print(f"  {label:<20}  {'all data':>8}  "
              f"{mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}")
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for pkl_path in pkl_paths:
    process_pkl(pkl_path)
