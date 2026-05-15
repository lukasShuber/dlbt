"""
run1/05_determ_beliefs/analysis.py — plots for the deterministic-belief ablation.

Figures produced
----------------
  Coverage sweep  (from coverage_sweep_<tag>.pkl):
    plot_01_coverage_sweep_<tag>.png    cMSE−NF vs budget, traces per coverage frac
    plot_01b_coverage_rho_<tag>.png     Spearman ρ vs budget

  Arity sweep  (from arity_sweep_<tag>.pkl):
    plot_02_arity_sweep_<tag>.png       cMSE−NF vs budget, traces per arity
    plot_02b_arity_rho_<tag>.png        Spearman ρ vs budget

Style mirrors 02_data_efficiency and 022_data_efficiency_arity exactly.

Run from repo root:
    python experiments/behavior/run1/05_determ_beliefs/analysis.py [--tag TAG] [--log-y]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parents[2] / "run0"))
from preprocess import filter_assignments, aggregate_counts

from dlbt.agents.detbt import DetBTAgent
from dlbt.constants import K as _K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[4]
plots_dir  = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Filter results pkl by tag substring (default: cfg.RUN_TAG).")
parser.add_argument("--log-y", action="store_true", default=False,
                    help="Use log scale on the y-axis of MSE plots.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Latent-state label helpers
# ---------------------------------------------------------------------------
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

def _label(t: str) -> str:
    return t.replace("_and_", " & ").replace("_", "/")

# ---------------------------------------------------------------------------
# One-time setup: image refs + CLIP cache
# ---------------------------------------------------------------------------
_refs_dict   = load_image_refs(_REPO_ROOT / cfg.METADATA)
_refs_by_uid = {r.uid: r for r in image_refs_as_list(_refs_dict)}

_frozen_clip: dict = {}
_cache_path = _REPO_ROOT / cfg.CACHE_PATH
if _cache_path.exists():
    from dlbt.agents.dlbt import DlbtAgent as _DlbtTmp
    _tmp = _DlbtTmp(freeze_encoder=True, n_mc_samples=1,
                    device=torch.device("cpu"), mapper_hidden=cfg.MAPPER_HIDDEN)
    _tmp.load_cache(str(_cache_path))
    _frozen_clip = {uid: feat.clone() for uid, feat in _tmp._cache.items()}
    del _tmp, _DlbtTmp
    print(f"Loaded CLIP cache ({len(_frozen_clip)} images) for α heatmaps.")
else:
    print(f"[warn] CLIP cache not found at {_cache_path} — α heatmaps skipped.")

# ---------------------------------------------------------------------------
# Canonical baselines — computed once from the full behavioral dataset
# ---------------------------------------------------------------------------
_df_raw_can = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
_df_filt_can, _ = filter_assignments(
    _df_raw_can,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
_tasks_can  = cfg.eligible_tasks(_df_filt_can)
_beh_id_can = {k: v for k, v in cfg.BEH_ID_TO_TASK.items() if v in set(_tasks_can)}
_ds_can, _probe_uids_can, _ = aggregate_counts(
    _df_filt_can, _beh_id_can, use_trial_kinds=cfg.USE_TRIAL_KINDS,
)

_probe_refs_can = sorted(
    [_refs_by_uid[uid] for uid in _probe_uids_can if uid in _refs_by_uid],
    key=lambda r: r.latent_state,
)
_probe_uids_ord_can = [r.uid for r in _probe_refs_can]
_uid_row_can  = {uid: i for i, uid in enumerate(_probe_uids_ord_can)}
_task_col_can = {t:   j for j, t   in enumerate(_tasks_can)}
_probe_cells_can = _ds_can.df[_ds_can.df["uid"].isin(_probe_uids_can)].copy()

_n_p_can = len(_probe_uids_ord_can)
_n_t_can = len(_tasks_can)
_true_mat_can = np.full((_n_p_can, _n_t_can), np.nan)
_cnt_mat_can  = np.zeros((_n_p_can, _n_t_can), dtype=np.int32)
for _rc in _probe_cells_can.itertuples(index=False):
    _i = _uid_row_can.get(_rc.uid)
    _j = _task_col_can.get(_rc.task_name)
    _tot = _rc.count_0 + _rc.count_1
    if _i is not None and _j is not None and _tot > 0:
        _true_mat_can[_i, _j] = _rc.count_1 / _tot
        _cnt_mat_can[_i, _j]  = _tot

_nf_mask_can = _cnt_mat_can > 1
_probe_nf_can = (
    float(np.mean(
        _true_mat_can[_nf_mask_can] * (1 - _true_mat_can[_nf_mask_can])
        / (_cnt_mat_can[_nf_mask_can].astype(float) - 1)
    )) if _nf_mask_can.any() else 0.0
)
_valid_rg_can = ~np.isnan(_true_mat_can)
CANONICAL_RANDOM_CMSE_NET = (
    float(np.mean((0.5 - _true_mat_can[_valid_rg_can]) ** 2)) - _probe_nf_can
)

# Random-init DetBT baseline (canonical)
CANONICAL_RAND_INIT_CMSE_NET = float("nan")
CANONICAL_RAND_INIT_RHO      = float("nan")
if _frozen_clip:
    torch.manual_seed(cfg.SEEDS[0])
    _ri_can = DetBTAgent(
        freeze_encoder    = True,
        device            = torch.device("cpu"),
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    _ri_can._cache = {uid: feat.clone() for uid, feat in _frozen_clip.items()}
    _lin_can = _ri_can.mapper[0] if cfg.MAPPER_HIDDEN is None else _ri_can.mapper[2]
    _rng_can = np.random.default_rng(cfg.INIT_SEED)
    _a_can   = _rng_can.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                size=(_lin_can.bias.shape[0],)).astype(np.float32)
    with torch.no_grad():
        _lin_can.bias.copy_(torch.from_numpy(np.log(np.exp(_a_can) - 1.0)))
    _ri_can.eval()
    _pred_ri_can = np.full((_n_p_can, _n_t_can), np.nan)
    with torch.no_grad():
        for _j_can, _t_can in enumerate(_tasks_can):
            _pred_ri_can[:, _j_can] = _ri_can.choice_probs(
                _probe_refs_can, get_task(_t_can)
            )[:, 1].cpu().numpy()
    _valid_ri_can = ~np.isnan(_pred_ri_can) & ~np.isnan(_true_mat_can)
    CANONICAL_RAND_INIT_CMSE_NET = (
        float(np.mean((_pred_ri_can[_valid_ri_can] - _true_mat_can[_valid_ri_can]) ** 2))
        - _probe_nf_can
    )
    _r, _ = spearmanr(_pred_ri_can[_valid_ri_can], _true_mat_can[_valid_ri_can])
    CANONICAL_RAND_INIT_RHO = float(_r)
    del _ri_can

print(f"Canonical baselines — NF={_probe_nf_can:.5f}  "
      f"P(0.5)={CANONICAL_RANDOM_CMSE_NET:.5f}  "
      f"rand-init DetBT={CANONICAL_RAND_INIT_CMSE_NET:.5f}  "
      f"rand-init ρ={CANONICAL_RAND_INIT_RHO:.3f}")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _rho(pred_mat: np.ndarray, true_mat: np.ndarray) -> float:
    """Spearman ρ between predicted and true P(yes) over all valid cells."""
    valid = ~np.isnan(pred_mat) & ~np.isnan(true_mat)
    if valid.sum() < 2:
        return float("nan")
    r, _ = spearmanr(pred_mat[valid], true_mat[valid])
    return float(r)


def _add_delta_inset(ax, keys, dlbt_traces, slda_dict, colors,
                     inset_bounds=(0.13, 0.42, 0.30, 0.22)):
    """
    Small inset: grouped bars of Δ cMSE-NF = DetBT − SLDA (mean over seeds).
    Negative = DetBT better; positive = SLDA better.
    Only powers-of-10 budgets shown.
    """
    all_dlbt_budgets: set[int] = set()
    for k in keys:
        all_dlbt_budgets |= set(dlbt_traces[k].keys())
    bar_budgets = sorted(
        b for b in slda_dict
        if not np.isnan(slda_dict.get(b, np.nan)) and b in all_dlbt_budgets
        and b >= 10 and np.log10(b) % 1 == 0
    )
    if not bar_budgets:
        return

    n_groups = len(bar_budgets)
    n_bars   = len(keys)
    group_w  = 0.80
    bar_w    = group_w / n_bars

    axins = ax.inset_axes(inset_bounds)
    axins.set_facecolor("none")

    for bar_i, key in enumerate(keys):
        trace = dlbt_traces[key]
        xs, ys = [], []
        for g_i, b in enumerate(bar_budgets):
            if b not in trace:
                continue
            dlbt_mean = float(np.nanmean(trace[b]))
            slda_val  = slda_dict[b]
            if np.isnan(dlbt_mean) or np.isnan(slda_val):
                continue
            xs.append(g_i - group_w / 2 + (bar_i + 0.5) * bar_w)
            ys.append(dlbt_mean - slda_val)
        if xs:
            axins.bar(xs, ys, width=bar_w * 0.85, color=colors[key], alpha=0.85)

    axins.axhline(0, color="black", lw=0.7, zorder=5)
    axins.set_xticks(range(n_groups))
    axins.set_xticklabels(
        [r"$10^{" + str(int(np.log10(b))) + r"}$" for b in bar_budgets],
        fontsize=7,
    )
    axins.set_xlim(-0.5, n_groups - 0.5)
    axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, symmetric=True))
    axins.tick_params(axis="y", labelsize=7, labelrotation=90)
    axins.set_ylabel(r"$\Delta$ cMSE$-$NF", fontsize=8, labelpad=2)
    sns.despine(ax=axins, top=True, right=True)


def _load_agent(ckpt_path: str) -> DetBTAgent | None:
    p = Path(ckpt_path)
    if not p.exists():
        print(f"  [warn] Checkpoint not found: {p.name}")
        return None
    agent = DetBTAgent(
        freeze_encoder = True,
        device         = torch.device("cpu"),
        mapper_hidden  = cfg.MAPPER_HIDDEN,
    )
    agent._cache = {uid: feat.clone() for uid, feat in _frozen_clip.items()}
    ckpt = torch.load(p, map_location="cpu")
    agent.mapper.load_state_dict(ckpt["mapper"])
    agent.eval()
    return agent


def _probe_refs_from_uids(probe_uids_ordered: list):
    return [_refs_by_uid[uid] for uid in probe_uids_ordered if uid in _refs_by_uid]


def _probe_matrix_heatmap(mat, row_labels, col_labels, title, fname):
    n_tasks = len(col_labels)
    fig_w   = max(9, n_tasks * 0.20)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    sns.heatmap(mat, ax=ax,
                xticklabels=col_labels, yticklabels=row_labels,
                cmap="RdYlBu_r", vmin=0, vmax=1,
                cbar_kws={"label": "P(yes)"},
                linewidths=0.1, linecolor="white")
    ax.set_xlabel("Task",        fontsize=10)
    ax.set_ylabel("Probe image", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=5, rotation=60)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _alpha_heatmap(alpha_mat, row_labels, title, fname):
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


def _set_log_y(ax):
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1.0)
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])


def _set_x_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1, 1.0e5)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels(["0", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Total trial budget", fontsize=11)


# ===========================================================================
# Coverage sweep plots
# ===========================================================================
_cov_candidates = sorted(cfg.RESULTS_DIR.glob("coverage_sweep_*.pkl"))
_tag_filter = args.tag or cfg.RUN_TAG
_cov_candidates = [p for p in _cov_candidates if _tag_filter in p.stem]

for cov_path in _cov_candidates:
    run_tag = cov_path.stem[len("coverage_sweep_"):]
    print(f"\n=== {cov_path.name}  (coverage, run_tag={run_tag}) ===")

    with open(cov_path, "rb") as f:
        summary = pickle.load(f)

    all_tasks   = summary["all_tasks_ordered"]
    probe_uids  = summary["probe_uids_ordered"]
    true_matrix = summary["true_matrix"]
    slda_res    = summary["slda"]
    dlbt_res    = summary["dlbt"]
    cov_fracs   = summary["coverage_fracs"]

    random_cmse_net      = CANONICAL_RANDOM_CMSE_NET
    random_init_cmse_net = CANONICAL_RAND_INIT_CMSE_NET

    probe_refs    = _probe_refs_from_uids(probe_uids)
    image_labels  = [_state_label(r.latent_state) for r in probe_refs]
    task_labels   = [_label(t) for t in all_tasks]

    # Build traces {frac: {budget: [val_per_seed]}}
    dlbt_traces:     dict[float, dict[int, list]] = {f: {} for f in cov_fracs}
    dlbt_rho_traces: dict[float, dict[int, list]] = {f: {} for f in cov_fracs}

    for seed_data in dlbt_res.values():
        for frac in cov_fracs:
            frac_key = f"{frac:.2f}"
            if frac_key not in seed_data["coverage"]:
                continue
            for bstr, bdata in seed_data["coverage"][frac_key]["budgets"].items():
                b = int(bstr)
                dlbt_traces[frac].setdefault(b, []).append(
                    bdata.get("probe_cmse_net", float("nan")))
                pm = bdata.get("pred_matrix")
                dlbt_rho_traces[frac].setdefault(b, []).append(
                    _rho(pm, true_matrix) if pm is not None else float("nan"))

    slda_budgets = sorted(int(b) for b in slda_res["budgets"])
    slda_y       = [slda_res["budgets"][str(b)].get("probe_cmse_net", float("nan"))
                    for b in slda_budgets]
    slda_dict    = {b: slda_y[i] for i, b in enumerate(slda_budgets)}

    # -----------------------------------------------------------------------
    # Plot 01 — cMSE−NF vs budget, coverage traces
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for frac in cov_fracs:
        trace = dlbt_traces[frac]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.cov_color(frac)
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DetBT {frac:.0%} coverage", zorder=4)
        if any(s > 0 for s in sems):
            lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]
            hi = [m + s            for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    ax.plot(slda_budgets, slda_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    _inset_bounds = [0.13, 0.09, 0.30, 0.22] if args.log_y else [0.13, 0.42, 0.30, 0.22]
    _add_delta_inset(
        ax, cov_fracs, dlbt_traces, slda_dict,
        colors={f: cfg.cov_color(f) for f in cov_fracs},
        inset_bounds=_inset_bounds,
    )

    if not np.isnan(random_cmse_net):
        ax.axhline(random_cmse_net, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)
    if not np.isnan(random_init_cmse_net):
        ax.axhline(random_init_cmse_net, color="#999999", lw=1.5,
                   ls=":", label="Random-init DetBT", zorder=1)

    _set_x_axis(ax)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if args.log_y:
        _set_log_y(ax)
    else:
        ax.set_ylim(0, 0.34)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01_coverage_sweep_{run_tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 01b — Spearman ρ vs budget, coverage traces
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for frac in cov_fracs:
        trace = dlbt_rho_traces[frac]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.cov_color(frac)
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DetBT {frac:.0%} coverage", zorder=4)
        if any(s > 0 for s in sems):
            lo = [m - s for m, s in zip(means, sems)]
            hi = [m + s for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    slda_rho_y = [_rho(slda_res["budgets"][str(b)].get("pred_matrix"), true_matrix)
                  for b in slda_budgets]
    ax.plot(slda_budgets, slda_rho_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    ax.axhline(0, color="#999999", lw=1.5, ls=(0, (4, 3)),
               label="Random (P=0.5)", zorder=1)
    if not np.isnan(CANONICAL_RAND_INIT_RHO):
        ax.axhline(CANONICAL_RAND_INIT_RHO, color="#999999", lw=1.5,
                   ls=":", label="Random-init DetBT", zorder=1)

    _set_x_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
    ax.set_ylim(0, 1)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01b_coverage_rho_{run_tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot: ground truth probe matrix (once)
    # -----------------------------------------------------------------------
    if image_labels:
        _probe_matrix_heatmap(
            true_matrix, image_labels, task_labels,
            title="Probe matrix — empirical P(yes)",
            fname=f"plot_00_probe_matrix_true_{run_tag}.png",
        )

    # -----------------------------------------------------------------------
    # Plot: DetBT predicted probe matrix + α heatmap (final budget, first seed)
    # -----------------------------------------------------------------------
    first_seed = next(iter(dlbt_res.values()))
    for frac in cov_fracs:
        frac_key = f"{frac:.2f}"
        if frac_key not in first_seed["coverage"]:
            continue
        cov_data = first_seed["coverage"][frac_key]
        if not cov_data["budgets"]:
            continue
        max_bstr = str(cov_data["max_budget"])
        bdata    = cov_data["budgets"].get(max_bstr) or cov_data["budgets"][
            str(max(int(k) for k in cov_data["budgets"]))]

        pm = bdata.get("pred_matrix")
        if pm is not None and image_labels:
            _probe_matrix_heatmap(
                pm, image_labels, task_labels,
                title=f"Probe matrix — DetBT  [cov={frac:.0%}  B={max_bstr}]",
                fname=f"plot_03_probe_matrix_detbt_{run_tag}_cov{frac:.2f}_budget{max_bstr}.png",
            )

        if _frozen_clip and image_labels:
            agent = _load_agent(bdata.get("ckpt_path", ""))
            if agent is not None:
                with torch.no_grad():
                    alpha_mat = agent.get_alpha(_probe_refs_from_uids(probe_uids)).cpu().numpy()
                del agent
                _alpha_heatmap(
                    alpha_mat, image_labels,
                    title=f"Learned α — DetBT  [cov={frac:.0%}  B={max_bstr}]",
                    fname=f"plot_04_alpha_{run_tag}_cov{frac:.2f}_budget{max_bstr}.png",
                )

# ===========================================================================
# Arity sweep plots
# ===========================================================================
_arity_candidates = sorted(cfg.RESULTS_DIR.glob("arity_sweep_*.pkl"))
_arity_candidates = [p for p in _arity_candidates if _tag_filter in p.stem]

for arity_path in _arity_candidates:
    run_tag = arity_path.stem[len("arity_sweep_"):]
    print(f"\n=== {arity_path.name}  (arity, run_tag={run_tag}) ===")

    with open(arity_path, "rb") as f:
        summary = pickle.load(f)

    all_tasks   = summary["all_tasks_ordered"]
    probe_uids  = summary["probe_uids_ordered"]
    true_matrix = summary["true_matrix"]
    slda_res    = summary["slda"]
    dlbt_res    = summary["dlbt"]
    arities     = summary["arities"]

    random_cmse_net      = CANONICAL_RANDOM_CMSE_NET
    random_init_cmse_net = CANONICAL_RAND_INIT_CMSE_NET

    probe_refs   = _probe_refs_from_uids(probe_uids)
    image_labels = [_state_label(r.latent_state) for r in probe_refs]
    task_labels  = [_label(t) for t in all_tasks]

    # Build traces {arity: {budget: [val_per_seed]}}
    dlbt_traces:     dict[int, dict[int, list]] = {a: {} for a in arities}
    dlbt_rho_traces: dict[int, dict[int, list]] = {a: {} for a in arities}

    for seed_data in dlbt_res.values():
        for arity in arities:
            arity_key = str(arity)
            if arity_key not in seed_data["arity"]:
                continue
            for bstr, bdata in seed_data["arity"][arity_key]["budgets"].items():
                b = int(bstr)
                dlbt_traces[arity].setdefault(b, []).append(
                    bdata.get("probe_cmse_net", float("nan")))
                pm = bdata.get("pred_matrix")
                dlbt_rho_traces[arity].setdefault(b, []).append(
                    _rho(pm, true_matrix) if pm is not None else float("nan"))

    slda_budgets = sorted(int(b) for b in slda_res["budgets"])
    slda_y       = [slda_res["budgets"][str(b)].get("probe_cmse_net", float("nan"))
                    for b in slda_budgets]
    slda_dict    = {b: slda_y[i] for i, b in enumerate(slda_budgets)}

    # -----------------------------------------------------------------------
    # Plot 02 — cMSE−NF vs budget, arity traces
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for arity in arities:
        trace = dlbt_traces[arity]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.ARITY_COLOR[arity]
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DetBT {arity}-way tasks", zorder=4)
        if any(s > 0 for s in sems):
            lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]
            hi = [m + s            for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    ax.plot(slda_budgets, slda_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    _inset_bounds = [0.13, 0.09, 0.30, 0.22] if args.log_y else [0.13, 0.42, 0.30, 0.22]
    _add_delta_inset(
        ax, arities, dlbt_traces, slda_dict,
        colors=cfg.ARITY_COLOR,
        inset_bounds=_inset_bounds,
    )

    if not np.isnan(random_cmse_net):
        ax.axhline(random_cmse_net, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)
    if not np.isnan(random_init_cmse_net):
        ax.axhline(random_init_cmse_net, color="#999999", lw=1.5,
                   ls=":", label="Random-init DetBT", zorder=1)

    _set_x_axis(ax)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if args.log_y:
        _set_log_y(ax)
    else:
        ax.set_ylim(0, 0.34)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    out = plots_dir / f"plot_02_arity_sweep_{run_tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 02b — Spearman ρ vs budget, arity traces
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for arity in arities:
        trace = dlbt_rho_traces[arity]
        if not trace:
            continue
        budgets_s = sorted(trace.keys())
        means = [float(np.nanmean(trace[b])) for b in budgets_s]
        sems  = [float(np.nanstd(trace[b]) / np.sqrt(max(len(trace[b]), 1)))
                 for b in budgets_s]
        color = cfg.ARITY_COLOR[arity]
        ax.plot(budgets_s, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DetBT {arity}-way tasks", zorder=4)
        if any(s > 0 for s in sems):
            lo = [m - s for m, s in zip(means, sems)]
            hi = [m + s for m, s in zip(means, sems)]
            ax.fill_between(budgets_s, lo, hi, color=color, alpha=0.40, linewidth=0)

    slda_rho_y = [_rho(slda_res["budgets"][str(b)].get("pred_matrix"), true_matrix)
                  for b in slda_budgets]
    ax.plot(slda_budgets, slda_rho_y, "o--", color=cfg.C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    ax.axhline(0, color="#999999", lw=1.5, ls=(0, (4, 3)),
               label="Random (P=0.5)", zorder=1)
    if not np.isnan(CANONICAL_RAND_INIT_RHO):
        ax.axhline(CANONICAL_RAND_INIT_RHO, color="#999999", lw=1.5,
                   ls=":", label="Random-init DetBT", zorder=1)

    _set_x_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
    ax.set_ylim(0, 1)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    out = plots_dir / f"plot_02b_arity_rho_{run_tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

if not _cov_candidates and not _arity_candidates:
    print(f"\nNo result pkls found in {cfg.RESULTS_DIR} matching tag '{_tag_filter}'.")
    print("Run run.py first.")
else:
    print(f"\nAll plots saved to {plots_dir}")
