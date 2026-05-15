"""
run1/compare_02_022.py — overlay 100%-coverage DLBT (02) vs 1-way DLBT (022).

For each run_tag that appears in both
  02_data_efficiency/results/coverage_sweep_<tag>.pkl
  022_data_efficiency_arity/results/arity_sweep_<tag>.pkl
produces two plots saved to run1/results/compare_plots/:

  compare_cmse_<tag>.png   — cMSE−NF vs trial budget
  compare_rho_<tag>.png    — Spearman ρ  vs trial budget

Run from repo root:
    python experiments/behavior/run1/compare_02_022.py [--tag TAG]
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RUN1_DIR  = Path(__file__).parent
_REPO_ROOT = Path(__file__).parents[3]

import importlib.util as _ilu

def _load_config(path, name):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg02  = _load_config(_RUN1_DIR / "02_data_efficiency"       / "config.py", "cfg02")
cfg022 = _load_config(_RUN1_DIR / "022_data_efficiency_arity" / "config.py", "cfg022")

sys.path.insert(0, str(_RUN1_DIR.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

from dlbt.agents.dlbt import DlbtAgent
from dlbt.constants import K as _K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

plots_dir = _RUN1_DIR / "results" / "compare_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BLUES        = plt.get_cmap("Blues")
C_COV100      = _BLUES(0.88)          # 100% coverage trace (matches 02 style)
C_ARITY1      = cfg022.ARITY_COLOR[1] # 1-way arity trace  (matches 022 style)
C_SLDA        = "#7D3C98"

# ---------------------------------------------------------------------------
# Canonical baselines (computed once from full behavioural data)
# ---------------------------------------------------------------------------
_refs_dict   = load_image_refs(_REPO_ROOT / cfg02.METADATA)
_refs_by_uid = {r.uid: r for r in image_refs_as_list(_refs_dict)}

_frozen_clip: dict = {}
_cache_path = _REPO_ROOT / cfg02.CACHE_PATH
if _cache_path.exists():
    _tmp = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                     device=torch.device("cpu"), mapper_hidden=cfg02.MAPPER_HIDDEN)
    _tmp.load_cache(str(_cache_path))
    _frozen_clip = {uid: feat.clone() for uid, feat in _tmp._cache.items()}
    del _tmp

_df_raw = pd.concat(
    [pd.read_csv(cfg02.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg02.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
_df_filt, _ = filter_assignments(
    _df_raw,
    min_catch_perf=cfg02.MIN_CATCH_PERF,
    main_perf_quantile=cfg02.MAIN_PERF_QUANTILE,
    seed=cfg02.SEED,
)
_tasks_can  = cfg02.eligible_tasks(_df_filt)
_beh_id_can = {k: v for k, v in cfg02.BEH_ID_TO_TASK.items() if v in set(_tasks_can)}
_ds_can, _probe_uids_can, _ = aggregate_counts(
    _df_filt, _beh_id_can, use_trial_kinds=cfg02.USE_TRIAL_KINDS,
)
_probe_refs_can = sorted(
    [_refs_by_uid[uid] for uid in _probe_uids_can if uid in _refs_by_uid],
    key=lambda r: r.latent_state,
)
_uid_row_can  = {r.uid: i for i, r in enumerate(_probe_refs_can)}
_task_col_can = {t: j for j, t in enumerate(_tasks_can)}
_probe_cells  = _ds_can.df[_ds_can.df["uid"].isin(_probe_uids_can)].copy()

_n_p = len(_probe_refs_can)
_n_t = len(_tasks_can)
_true_can  = np.full((_n_p, _n_t), np.nan)
_cnt_can   = np.zeros((_n_p, _n_t), dtype=np.int32)
for _rc in _probe_cells.itertuples(index=False):
    _i = _uid_row_can.get(_rc.uid)
    _j = _task_col_can.get(_rc.task_name)
    _tot = _rc.count_0 + _rc.count_1
    if _i is not None and _j is not None and _tot > 0:
        _true_can[_i, _j] = _rc.count_1 / _tot
        _cnt_can[_i, _j]  = _tot

_nf_mask = _cnt_can > 1
_probe_nf = float(np.mean(
    _true_can[_nf_mask] * (1 - _true_can[_nf_mask])
    / (_cnt_can[_nf_mask].astype(float) - 1)
)) if _nf_mask.any() else 0.0

_valid_rg = ~np.isnan(_true_can)
CANONICAL_RANDOM_CMSE = float(np.mean((0.5 - _true_can[_valid_rg]) ** 2)) - _probe_nf

CANONICAL_RAND_INIT_CMSE = float("nan")
CANONICAL_RAND_INIT_RHO  = float("nan")
if _frozen_clip:
    torch.manual_seed(cfg02.SEEDS[0])
    _ri = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg02.N_MC,
                    device=torch.device("cpu"), mapper_hidden=cfg02.MAPPER_HIDDEN,
                    normalize_utility=cfg02.NORMALIZED_UTILITY)
    _ri._cache = {uid: f.clone() for uid, f in _frozen_clip.items()}
    _lin = _ri.mapper[0] if cfg02.MAPPER_HIDDEN is None else _ri.mapper[2]
    _rng_i = np.random.default_rng(cfg02.INIT_SEED)
    _a = _rng_i.uniform(cfg02.INIT_ALPHA_LOW, cfg02.INIT_ALPHA_HIGH,
                        size=(_lin.bias.shape[0],)).astype(np.float32)
    with torch.no_grad():
        _lin.bias.copy_(torch.from_numpy(np.log(np.exp(_a) - 1.0)))
    _ri.eval()
    _pred_ri = np.full((_n_p, _n_t), np.nan)
    with torch.no_grad():
        for _j, _t in enumerate(_tasks_can):
            _pred_ri[:, _j] = _ri.choice_probs(
                _probe_refs_can, get_task(_t))[:, 1].cpu().numpy()
    _v = ~np.isnan(_pred_ri) & ~np.isnan(_true_can)
    CANONICAL_RAND_INIT_CMSE = float(np.mean((_pred_ri[_v] - _true_can[_v]) ** 2)) - _probe_nf
    _r, _ = spearmanr(_pred_ri[_v], _true_can[_v])
    CANONICAL_RAND_INIT_RHO = float(_r)
    del _ri

print(f"Canonical baselines — NF={_probe_nf:.5f}  "
      f"P(0.5)={CANONICAL_RANDOM_CMSE:.5f}  "
      f"rand-init={CANONICAL_RAND_INIT_CMSE:.5f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rho(pred: np.ndarray, true: np.ndarray) -> float:
    v = ~np.isnan(pred) & ~np.isnan(true)
    if v.sum() < 2:
        return float("nan")
    r, _ = spearmanr(pred[v], true[v])
    return float(r)


def _mean_sem(trace: dict[int, list]) -> tuple[list, list, list]:
    budgets = sorted(trace)
    means = [float(np.nanmean(trace[b])) for b in budgets]
    sems  = [float(np.nanstd(trace[b]) / np.sqrt(len(trace[b])))
             if len(trace[b]) > 1 else 0.0 for b in budgets]
    return budgets, means, sems


def _plot_trace(ax, budgets, means, sems, color, label, ls="-"):
    ax.plot(budgets, means, f"o{ls}", color=color, lw=2.0, ms=5,
            label=label, zorder=4)
    if any(s > 0 for s in sems):
        lo = [max(m - s, 1e-4) for m, s in zip(means, sems)]
        hi = [m + s             for m, s in zip(means, sems)]
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.30, linewidth=0)


def _axis_style(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels(["0", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Total trial budget", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    sns.despine(top=True, right=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Only process pkls whose tag contains this string.")
parser.add_argument("--log-y", action="store_true", default=False,
                    help="Use log scale on the y-axis of the cMSE plot.")
args = parser.parse_args()

dir02  = _RUN1_DIR / "02_data_efficiency"  / "results"
dir022 = _RUN1_DIR / "022_data_efficiency_arity" / "results"

# Strip the experiment-specific part ("_coverage" / "_arity") to get a common base tag
pkls02  = {p.stem[len("coverage_sweep_"):].replace("_coverage", ""): p
           for p in sorted(dir02.glob("coverage_sweep_*.pkl"))}
pkls022 = {p.stem[len("arity_sweep_"):].replace("_arity", ""): p
           for p in sorted(dir022.glob("arity_sweep_*.pkl"))}

matching_tags = sorted(set(pkls02) & set(pkls022))
if args.tag:
    matching_tags = [t for t in matching_tags if args.tag in t]

if not matching_tags:
    raise FileNotFoundError(
        f"No matching base tag found between\n  {dir02}\n  {dir022}\n"
        f"  02 tags:  {sorted(pkls02)}\n"
        f"  022 tags: {sorted(pkls022)}"
    )
print(f"Matched base tags: {matching_tags}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for tag in matching_tags:
    print(f"\n=== tag={tag} ===")

    with open(pkls02[tag],  "rb") as f: res02  = pickle.load(f)
    with open(pkls022[tag], "rb") as f: res022 = pickle.load(f)

    true_matrix = res02["true_matrix"]   # should be same in both

    # -- 100% coverage trace from 02 --
    cov_traces_cmse: dict[int, list] = {}
    cov_traces_rho:  dict[int, list] = {}
    for seed_key, seed_data in res02["dlbt"].items():
        cov_data = seed_data["coverage"].get("1.00", {})
        for bstr, bdata in cov_data.get("budgets", {}).items():
            b = int(bstr)
            cov_traces_cmse.setdefault(b, []).append(
                bdata.get("probe_cmse_net", float("nan"))
            )
            pm = bdata.get("pred_matrix")
            cov_traces_rho.setdefault(b, []).append(
                _rho(pm, true_matrix) if pm is not None else float("nan")
            )

    # -- 1-way arity trace from 022 --
    arity_traces_cmse: dict[int, list] = {}
    arity_traces_rho:  dict[int, list] = {}
    for seed_key, seed_data in res022["dlbt"].items():
        arity_data = seed_data["arity"].get("1", {})
        for bstr, bdata in arity_data.get("budgets", {}).items():
            b = int(bstr)
            arity_traces_cmse.setdefault(b, []).append(
                bdata.get("probe_cmse_net", float("nan"))
            )
            pm = bdata.get("pred_matrix")
            arity_traces_rho.setdefault(b, []).append(
                _rho(pm, true_matrix) if pm is not None else float("nan")
            )

    # -- SLDA trace (from 02; identical in 022) --
    slda = res02["slda"]
    slda_budgets = sorted(int(b) for b in slda["budgets"])
    slda_cmse    = [slda["budgets"][str(b)].get("probe_cmse_net", float("nan"))
                    for b in slda_budgets]
    slda_rho     = [_rho(slda["budgets"][str(b)].get("pred_matrix"), true_matrix)
                    for b in slda_budgets]

    # -----------------------------------------------------------------------
    # Plot A — cMSE−NF
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    b, m, s = _mean_sem(cov_traces_cmse)
    _plot_trace(ax, b, m, s, C_COV100, "DLBT 100% coverage", ls="-")

    b, m, s = _mean_sem(arity_traces_cmse)
    _plot_trace(ax, b, m, s, C_ARITY1, "DLBT 1-way tasks (~10% coverage)", ls="--")

    ax.plot(slda_budgets, slda_cmse, "o-", color=C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda['n_tasks']} tasks)", zorder=3)

    if not np.isnan(CANONICAL_RANDOM_CMSE):
        ax.axhline(CANONICAL_RANDOM_CMSE, color="#999999", lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)
    if not np.isnan(CANONICAL_RAND_INIT_CMSE):
        ax.axhline(CANONICAL_RAND_INIT_CMSE, color="#999999", lw=1.5,
                   ls=":", label="Random-init DLBT", zorder=1)

    if args.log_y:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1.0)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
    else:
        ax.set_ylim(0, 0.34)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _axis_style(ax, "cMSE − noise floor")
    plt.tight_layout()
    out = plots_dir / f"compare_cmse_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # -----------------------------------------------------------------------
    # Plot B — Spearman ρ
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    b, m, s = _mean_sem(cov_traces_rho)
    _plot_trace(ax, b, m, s, C_COV100, "DLBT 100% coverage", ls="-")

    b, m, s = _mean_sem(arity_traces_rho)
    _plot_trace(ax, b, m, s, C_ARITY1, "DLBT 1-way tasks (~10% coverage)", ls="--")

    ax.plot(slda_budgets, slda_rho, "o-", color=C_SLDA, lw=2.0, ms=5,
            label=f"SLDA (all {slda['n_tasks']} tasks)", zorder=3)

    ax.axhline(0, color="#999999", lw=1.5, ls=(0, (4, 3)),
               label="Random (P=0.5)", zorder=1)
    if not np.isnan(CANONICAL_RAND_INIT_RHO):
        ax.axhline(CANONICAL_RAND_INIT_RHO, color="#999999", lw=1.5,
                   ls=":", label="Random-init DLBT", zorder=1)

    ax.set_ylim(0, 1)
    _axis_style(ax, r"Spearman $\rho$")
    plt.tight_layout()
    out = plots_dir / f"compare_rho_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print(f"\nAll comparison plots saved to {plots_dir}")
