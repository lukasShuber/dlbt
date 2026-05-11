"""
run1/02_data_efficiency/analysis.py — plots for the coverage sweep.

Figures produced per results pkl
---------------------------------
  plot_01_coverage_sweep_<tag>.png
      Probe-matrix MSE vs trial budget.
      Traces: DLBT at each coverage fraction (light→dark blue),
              SLDA reference (dashed, all tasks).
      Multiple seeds → mean ± SEM shading.

  plot_02_probe_matrix_true_<tag>.png
      Ground truth [n_probe × n_tasks] P(yes) heatmap (once).

  plot_03_probe_matrix_slda_<tag>_budgetB.png
      SLDA predicted probe matrix at each budget.

  plot_04_probe_matrix_dlbt_<tag>_covFRAC_budgetB.png
      DLBT predicted probe matrix per (coverage, budget).
      Only produced for the final (max) budget per coverage trace.

  plot_05_alpha_<tag>_covFRAC_budgetB.png
      Learned Dirichlet α heatmap per (coverage, final budget).
      Requires saved agent checkpoints.

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/analysis.py [--tag TAG]
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parents[2] / "run0"))
from preprocess import filter_assignments, aggregate_counts

from dlbt.agents.dlbt import DlbtAgent
from dlbt.constants import K as _K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

# Coverage fraction → shade of blue (light to dark)
_BLUES = plt.get_cmap("Blues")
_CMAP_OFFSETS = {
    0.10: 0.30,
    0.25: 0.44,
    0.50: 0.58,
    0.75: 0.72,
    1.00: 0.88,
}
def _cov_color(frac: float):
    return _BLUES(_CMAP_OFFSETS.get(frac, 0.6))

C_SLDA = "#7D3C98"   # purple for SLDA reference

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


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


def _arity(t: str) -> int:
    return t.count("_and_") + 1


def _label(t: str) -> str:
    return t.replace("_and_", " & ").replace("_", "/")


# ---------------------------------------------------------------------------
# One-time setup: image refs + CLIP cache (for α heatmaps)
# ---------------------------------------------------------------------------
_REPO_ROOT   = Path(__file__).parents[4]
_refs_dict   = load_image_refs(_REPO_ROOT / cfg.METADATA)
_refs_by_uid = {r.uid: r for r in image_refs_as_list(_refs_dict)}

_frozen_clip: dict = {}
_cache_path = _REPO_ROOT / cfg.CACHE_PATH
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
# Canonical baselines — computed once from the full behavioral dataset.
# These are stable across all result pkls regardless of when they were run.
# ---------------------------------------------------------------------------
_df_raw_can = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
_df_filt_can, _ = filter_assignments(
    _df_raw_can,
    min_catch_perf=cfg.MIN_CATCH_PERF,
    main_perf_quantile=cfg.MAIN_PERF_QUANTILE,
    seed=cfg.SEED,
)
_tasks_can    = cfg.eligible_tasks(_df_filt_can)
_beh_id_can   = {k: v for k, v in cfg.BEH_ID_TO_TASK.items() if v in set(_tasks_can)}
_ds_can, _probe_uids_can, _ = aggregate_counts(
    _df_filt_can, _beh_id_can, use_trial_kinds=cfg.USE_TRIAL_KINDS,
)

# Canonical probe matrix [n_probe_can × n_tasks_can]
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

# Random-init DLBT baseline (canonical)
CANONICAL_RAND_INIT_CMSE_NET = float("nan")
if _frozen_clip:
    torch.manual_seed(cfg.SEEDS[0])
    _ri_can = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                        device=torch.device("cpu"), mapper_hidden=cfg.MAPPER_HIDDEN,
                        normalize_utility=cfg.NORMALIZED_UTILITY)
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
    del _ri_can

print(f"Canonical baselines — NF={_probe_nf_can:.5f}  "
      f"P(0.5)={CANONICAL_RANDOM_CMSE_NET:.5f}  "
      f"rand-init DLBT={CANONICAL_RAND_INIT_CMSE_NET:.5f}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

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


def _plot_alpha_heatmap(alpha_mat: np.ndarray, row_labels: list,
                        title: str, fname: str) -> None:
    """[n_probe × K] Dirichlet α heatmap."""
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


def _load_agent(ckpt_path: str) -> DlbtAgent | None:
    p = Path(ckpt_path)
    if not p.exists():
        print(f"  [warn] Checkpoint not found: {p.name}")
        return None
    agent = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                      device=torch.device("cpu"), mapper_hidden=cfg.MAPPER_HIDDEN)
    agent._cache = {uid: feat.clone() for uid, feat in _frozen_clip.items()}
    ckpt = torch.load(p, map_location="cpu")
    agent.mapper.load_state_dict(ckpt["mapper"])
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Helper: ordered probe ImageRef list from UIDs
# ---------------------------------------------------------------------------
def _probe_refs_from_uids(probe_uids_ordered: list):
    """Return ImageRef list in the given UID order (skipping missing UIDs)."""
    return [_refs_by_uid[uid] for uid in probe_uids_ordered if uid in _refs_by_uid]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Filter results pkl by tag substring "
                         "(default: cfg.RUN_TAG).")
parser.add_argument("--log-y", action="store_true", default=False,
                    help="Use log scale on the y-axis of the coverage sweep plot.")
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("coverage_sweep_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
else:
    candidates = [p for p in candidates if cfg.RUN_TAG in p.stem]

if not candidates:
    raise FileNotFoundError(
        f"No coverage_sweep_*.pkl found in {cfg.RESULTS_DIR}. "
        f"Run run.py first."
    )

# ---------------------------------------------------------------------------
# Main loop over result files
# ---------------------------------------------------------------------------
for results_path in candidates:
    run_tag = results_path.stem[len("coverage_sweep_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        summary = pickle.load(f)

    all_tasks         = summary["all_tasks_ordered"]
    probe_uids        = summary["probe_uids_ordered"]
    true_matrix       = summary["true_matrix"]    # [n_probe × n_tasks]
    slda_res          = summary["slda"]
    dlbt_res          = summary["dlbt"]
    cov_fracs         = summary["coverage_fracs"]

    # Canonical baselines — stable across all pkls (computed from full data above)
    random_cmse_net      = CANONICAL_RANDOM_CMSE_NET
    random_init_cmse_net = CANONICAL_RAND_INIT_CMSE_NET

    n_probe  = len(probe_uids)
    n_tasks  = len(all_tasks)

    # Row/column labels for heatmaps
    probe_refs = _probe_refs_from_uids(probe_uids)
    image_labels    = [_state_label(r.latent_state) for r in probe_refs]
    task_col_labels = [_label(t) for t in all_tasks]

    # -----------------------------------------------------------------------
    # Plot 01 — coverage sweep: probe MSE vs trial budget
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Collect DLBT traces across seeds for each coverage fraction
    # Structure: {frac: {budget_int: [mse_seed0, mse_seed1, ...]}}
    dlbt_traces: dict[float, dict[int, list[float]]] = {f: {} for f in cov_fracs}

    for seed_key, seed_data in dlbt_res.items():
        for frac in cov_fracs:
            frac_key = f"{frac:.2f}"
            if frac_key not in seed_data["coverage"]:
                continue
            cov_data = seed_data["coverage"][frac_key]
            for bstr, bdata in cov_data["budgets"].items():
                b = int(bstr)
                dlbt_traces[frac].setdefault(b, []).append(
                    bdata.get("probe_cmse_net", bdata.get("probe_mse", float("nan")))
                )

    for frac in cov_fracs:
        trace = dlbt_traces[frac]
        if not trace:
            continue
        budgets_sorted = sorted(trace.keys())
        n_seeds_per_b  = [len(trace[b]) for b in budgets_sorted]
        means = [float(np.mean(trace[b])) for b in budgets_sorted]
        sems  = [float(np.std(trace[b]) / np.sqrt(len(trace[b])))
                 if len(trace[b]) > 1 else 0.0
                 for b in budgets_sorted]
        max_sem = max(sems)
        print(f"  cov={frac:.0%}  n_seeds={n_seeds_per_b}  "
              f"max_SEM={max_sem:.5f}  "
              f"{'SEM shading ON' if max_sem > 0 else 'SEM=0 — single seed or identical values'}")
        color = _cov_color(frac)
        ax.plot(budgets_sorted, means, "o-", color=color, lw=2.0, ms=5,
                label=f"DLBT {frac:.0%} coverage", zorder=4)
        if any(s > 0 for s in sems):
            lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]  # clip for log axis
            hi = [m + s            for m, s in zip(means, sems)]
            ax.fill_between(budgets_sorted, lo, hi,
                            color=color, alpha=0.40, linewidth=0)

    # SLDA reference trace
    slda_budgets = sorted(int(b) for b in slda_res["budgets"])
    slda_y       = [slda_res["budgets"][str(b)].get(
                        "probe_cmse_net", slda_res["budgets"][str(b)].get("probe_mse", float("nan")))
                    for b in slda_budgets]
    ax.plot(slda_budgets, slda_y, "o--", color=C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda_res['n_tasks']} tasks)", zorder=3)

    # Random guesser reference
    if not np.isnan(random_cmse_net):
        ax.axhline(random_cmse_net, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)

    # Random-init DLBT reference
    if not np.isnan(random_init_cmse_net):
        ax.axhline(random_init_cmse_net, color="#E76F51", lw=1.5,
                   ls=(0, (4, 3)), label="Random-init DLBT", zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(1, 1.0e5)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels(["0", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Total trial budget", fontsize=11)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if args.log_y:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1.0)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
    else:
        ax.set_ylim(0, 0.30)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    out = plots_dir / f"plot_01_coverage_sweep_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 02 — ground truth probe matrix (once)
    # -----------------------------------------------------------------------
    if image_labels:
        _plot_probe_matrix(
            true_matrix, image_labels, task_col_labels,
            title="Probe matrix — empirical P(yes)",
            fname=f"plot_02_probe_matrix_true_{run_tag}.png",
        )

    # -----------------------------------------------------------------------
    # Plot 03 — SLDA predicted probe matrix at each budget
    # -----------------------------------------------------------------------
    for bstr, bdata in slda_res["budgets"].items():
        pred_mat = bdata.get("pred_matrix")
        if pred_mat is None or not image_labels:
            continue
        _plot_probe_matrix(
            pred_mat, image_labels, task_col_labels,
            title=f"Probe matrix — SLDA predicted P(yes)  [budget={bstr}]",
            fname=f"plot_03_probe_matrix_slda_{run_tag}_budget{bstr}.png",
        )

    # -----------------------------------------------------------------------
    # Plots 04 + 05 — DLBT predicted probe matrix + α heatmap
    #   Produced only for the final (max) budget of each coverage trace,
    #   using the first seed.
    # -----------------------------------------------------------------------
    first_seed_key = next(iter(dlbt_res))
    first_seed     = dlbt_res[first_seed_key]

    for frac in cov_fracs:
        frac_key = f"{frac:.2f}"
        if frac_key not in first_seed["coverage"]:
            continue
        cov_data = first_seed["coverage"][frac_key]
        if not cov_data["budgets"]:
            continue

        # Only use the final (max) budget
        max_bstr  = str(cov_data["max_budget"])
        bdata     = cov_data["budgets"].get(max_bstr)
        if bdata is None:
            # Fallback: use the last available budget
            max_bstr = str(max(int(k) for k in cov_data["budgets"]))
            bdata    = cov_data["budgets"][max_bstr]

        frac_pct = f"{frac:.0%}"

        # -- Plot 04: predicted probe matrix --
        pred_mat = bdata.get("pred_matrix")
        if pred_mat is not None and image_labels:
            _plot_probe_matrix(
                pred_mat, image_labels, task_col_labels,
                title=(f"Probe matrix — DLBT predicted P(yes)  "
                       f"[cov={frac_pct}  budget={max_bstr}]"),
                fname=(f"plot_04_probe_matrix_dlbt_{run_tag}"
                       f"_cov{frac:.2f}_budget{max_bstr}.png"),
            )

        # -- Plot 05: α heatmap --
        if not _frozen_clip or not image_labels:
            continue
        ckpt_path = bdata.get("ckpt_path", "")
        agent = _load_agent(ckpt_path)
        if agent is None:
            continue
        with torch.no_grad():
            alpha_mat = agent.get_alpha(_probe_refs_from_uids(probe_uids)).cpu().numpy()
        del agent
        _plot_alpha_heatmap(
            alpha_mat, image_labels,
            title=f"Learned α  [cov={frac_pct}  budget={max_bstr}]",
            fname=(f"plot_05_alpha_{run_tag}"
                   f"_cov{frac:.2f}_budget{max_bstr}.png"),
        )

print(f"\nAll plots saved to {plots_dir}")
