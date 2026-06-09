"""
run1/021_efficiency_main/analysis.py — trials-per-task budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trials per task) for every
efficiency_main_021*.pkl found in results/.

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
    python experiments/behavior/run1/021_efficiency_main/analysis.py
    python experiments/behavior/run1/021_efficiency_main/analysis.py --pkl PATH
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
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
                    help="Path to a specific pkl. Default: all efficiency_main_021*.pkl.")
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
    pkl_paths = sorted(cfg.RESULTS_DIR.glob("efficiency_main_021*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No efficiency_main_021*.pkl found in {cfg.RESULTS_DIR}")

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

_REPO_ROOT = Path(__file__).parents[4]   # …/run1/021_efficiency_main → repo root


def _task_arity(name: str) -> int:
    return name.count("_and_") + 1


def _try_load_pred_matrices(pkl_path: Path, d: dict):
    """
    Load saved all-data DLBT and SLDA checkpoints, reconstruct per-seed
    probe-matrix predictions.

    Because each seed may have a different fine-tuned attnpool, backbone spatial
    maps are computed once (shared — the ResNet trunk is frozen) and each seed's
    attnpool weights are applied individually, giving genuinely per-seed features.

    Returns (dlbt_preds, slda_preds) — each a list of [n_probe × n_tasks]
    arrays (one per seed), or None if checkpoints are unavailable.
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

    _dev   = torch.device("cpu")
    by_uid = {r.uid: r for r in _iral(_lir(_REPO_ROOT / cfg.METADATA))}
    p_refs = [by_uid[uid] for uid in probe_uids_ordered if uid in by_uid]

    # ── Pre-attnpool backbone maps (ResNet trunk is frozen → identical across
    #    seeds; each seed's attnpool transforms them differently).
    #
    #    DLBT backbone: computed in EVAL mode — matches run.py which calls
    #      agent.eval() at line 492 BEFORE precompute_backbone_features().
    #    SLDA backbone: computed in TRAINING mode — run.py has no explicit
    #      eval() before the SLDA precompute_backbone_features() call (line 569),
    #      so training-mode BatchNorm statistics are used in both training and here.
    # ─────────────────────────────────────────────────────────────────────────
    dlbt_backbone_cache: dict = {}
    slda_backbone_cache: dict = {}
    frozen_clip:         dict = {}

    if has_attnpool_dlbt:
        print(f"  [scatter] computing DLBT backbone maps (eval mode) "
              f"for {len(p_refs)} probe images…")
        _tmp_dlbt = _DLBT(freeze_encoder=False, n_mc_samples=1, device=_dev,
                          normalize_utility=cfg.NORMALIZED_UTILITY)
        _tmp_dlbt.eval()   # CRITICAL: matches run.py agent.eval() before backbone
        _tmp_dlbt.precompute_backbone_features(p_refs)
        dlbt_backbone_cache = dict(_tmp_dlbt._backbone_cache)
        del _tmp_dlbt

    if has_attnpool_slda:
        print(f"  [scatter] computing SLDA backbone maps (training mode) "
              f"for {len(p_refs)} probe images…")
        _tmp_slda = _DLBT(freeze_encoder=False, n_mc_samples=1, device=_dev,
                          normalize_utility=cfg.NORMALIZED_UTILITY)
        # No .eval() — training mode matches run.py SLDA backbone precomputation
        _tmp_slda.precompute_backbone_features(p_refs)
        slda_backbone_cache = dict(_tmp_slda._backbone_cache)
        del _tmp_slda

    if not has_attnpool_dlbt or not has_attnpool_slda:
        cache_p = _REPO_ROOT / cfg.CACHE_PATH
        if cache_p.exists():
            frozen_clip = torch.load(str(cache_p), map_location="cpu")

    def _seed_clip_feats_dlbt(attnpool_module, refs):
        """Apply this seed's attnpool to eval-mode backbone maps → {uid: tensor}."""
        spatial = torch.stack([dlbt_backbone_cache[r.uid] for r in refs]).to(_dev)
        with torch.no_grad():
            out = attnpool_module(spatial).float()
        return {r.uid: out[i].cpu() for i, r in enumerate(refs)}

    def _seed_clip_feats_slda(attnpool_module, refs):
        """Apply this seed's attnpool to training-mode backbone maps → {uid: np}."""
        spatial = torch.stack([slda_backbone_cache[r.uid] for r in refs]).to(_dev)
        with torch.no_grad():
            out = attnpool_module(spatial).float()
        return {r.uid: out[i].cpu().numpy() for i, r in enumerate(refs)}

    # ── DLBT: one prediction matrix per seed ─────────────────────────────────
    dlbt_preds: list = []
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
            # Apply this seed's fine-tuned attnpool to eval-mode backbone maps
            agent.encoder.attnpool.load_state_dict(ckpt["attnpool_state_dict"])
            agent._cache = _seed_clip_feats_dlbt(agent.encoder.attnpool, p_refs)
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

    # ── SLDA: one prediction matrix per seed ─────────────────────────────────
    slda_preds: list = []
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
            # Apply this seed's fine-tuned attnpool to training-mode backbone maps
            probe_feats = _seed_clip_feats_slda(slda_tmp.encoder.attnpool, p_refs)
            del slda_tmp
        else:
            probe_feats = {uid: frozen_clip[uid].numpy()
                           for uid in probe_uids_ordered if uid in frozen_clip}

        pred = _slda_pm(art["scalers"], art["models"], art["use_base"],
                        probe_feats, all_tasks_ordered, uid_to_row, n_probe)
        slda_preds.append(pred)

    print(f"  [scatter] loaded {len(dlbt_preds)} DLBT seeds, "
          f"{len(slda_preds)} SLDA seeds")
    return (dlbt_preds if dlbt_preds else None,
            slda_preds if slda_preds else None)


def _make_scatter(d: dict,
                  dlbt_preds: list | None,
                  slda_preds: list | None,
                  plots_dir: Path):
    """
    One file per model (DLBT, SLDA): predicted vs. empirical P(right).
    Points in gray; transparent error bars (SEM over seeds / binomial SE).
    Saved as both PNG and SVG.
    """
    true_matrix       = d["true_matrix"]
    count_matrix      = d["count_matrix"]          # [n_probe × n_tasks] int
    all_tasks         = d["all_tasks_ordered"]
    probe_noise_floor = d["probe_noise_floor"]

    model_list = []
    for nm, preds, col in [("DLBT", dlbt_preds, cfg.C_DLBT),
                            ("SLDA", slda_preds, cfg.C_SLDA)]:
        if preds is not None and len(preds) > 0:
            arr      = np.array(preds)             # [n_seeds, n_probe, n_tasks]
            mean_mat = np.nanmean(arr, axis=0)
            n_s      = np.sum(~np.isnan(arr), axis=0).astype(float)
            sem_mat  = (np.nanstd(arr, axis=0, ddof=1)
                        / np.sqrt(np.maximum(n_s, 1)))
            model_list.append((nm, mean_mat, sem_mat, col))

    if not model_list:
        return

    # Empirical binomial SE:  sqrt(p*(1-p)/n),  NaN where n==0
    with np.errstate(invalid="ignore", divide="ignore"):
        binom_se = np.where(
            count_matrix > 0,
            np.sqrt(true_matrix * (1 - true_matrix)
                    / np.maximum(count_matrix, 1)),
            np.nan,
        )

    for name, mean_mat, sem_mat, line_col in model_list:
        valid = ~np.isnan(mean_mat) & ~np.isnan(true_matrix)
        ri, ci = np.where(valid)

        x     = mean_mat[ri, ci]
        x_err = sem_mat[ri, ci]
        y     = true_matrix[ri, ci]
        y_err = binom_se[ri, ci]

        fig, ax = plt.subplots(figsize=(3.0, 3.0))

        # Transparent error bars (no caps, thin lines)
        ax.errorbar(
            x, y,
            xerr=x_err, yerr=np.where(np.isnan(y_err), 0, y_err),
            fmt="none",
            ecolor="#888888", elinewidth=0.5, capsize=0,
            alpha=0.18, rasterized=True, zorder=2,
        )
        # Solid scatter points on top
        ax.scatter(
            x, y,
            s=3, color="#888888", alpha=0.40, linewidths=0,
            rasterized=True, zorder=3,
        )

        # Identity line
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls="--", zorder=1)

        # Annotation: ρ and cMSE-NF
        rho, _ = _spearmanr(x, y)
        cmse   = float(np.mean((x - y) ** 2)) - probe_noise_floor
        ax.text(0.04, 0.96,
                f"ρ = {rho:.3f}\ncMSE−NF = {cmse:.4f}",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="left", color="#333333")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(right)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Empirical P(right)",  fontsize=9, fontweight="bold")
        ax.set_title(name, fontsize=10, fontweight="bold", color=line_col)
        sns.despine(ax=ax, top=True, right=True)

        plt.tight_layout()
        stem = f"plot_scatter_{name.lower()}"
        for ext in ("png", "svg"):
            out = plots_dir / f"{stem}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  Saved: {out.relative_to(cfg.RESULTS_DIR)}")
        plt.close(fig)


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
    dlbt_preds, slda_preds = _try_load_pred_matrices(pkl_path, d)
    _make_scatter(d, dlbt_preds, slda_preds, plots_dir)

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
