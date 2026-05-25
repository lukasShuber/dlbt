"""
run1/05_ablations/run.py — belief-representation ablation budget sweep.

Models compared at each trial budget:
  DLBT        full model: MC Dirichlet integration, mapper trained on behaviour
  DetBT       perceptual stochasticity: Dirichlet mean at train AND eval (no MC)
  OneHotBT    perceptual uncertainty: train=mean (DetBT), eval=argmax one-hot
                (learned mapper, but beliefs certain at argmax latent state)

Reference lines (no training — evaluated once):
  Oracle      fixed soft beliefs from metadata latent state
  BehavSuperv ground-truth hard P=1/0 from task structure + true latent state

Protocol mirrors 021_efficiency_main (bootstrap sampling, same budget grid,
all-data point, per-seed weight + data variation for genuine SEM).

Run from repo root:
    python experiments/behavior/run1/05_ablations/run.py
"""

import gc
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.detbt import DetBTAgent
from dlbt.agents.onehot_bt import OneHotBTAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run0"))
from preprocess import filter_assignments, aggregate_counts

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[4]
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict   = load_image_refs(_REPO_ROOT / cfg.METADATA)
all_refs    = image_refs_as_list(refs_dict)
refs_by_uid = {r.uid: r for r in all_refs}
print(f"Loaded {len(refs_dict)} image refs.")

# ---------------------------------------------------------------------------
# Load + filter behavioural data
# ---------------------------------------------------------------------------
print("\nLoading behavioural data...")
df_raw = pd.concat(
    [pd.read_csv(cfg.BEHAVIOR_CSV_RUN0),
     pd.read_csv(cfg.BEHAVIOR_CSV_RUN1)],
    ignore_index=True,
)
df_filtered, _ = filter_assignments(
    df_raw,
    min_catch_perf     = cfg.MIN_CATCH_PERF,
    main_perf_quantile = cfg.MAIN_PERF_QUANTILE,
    seed               = cfg.SEED,
)
print(f"  Filtered: {df_filtered['assignment_id'].nunique()} assignments remain.")

all_tasks_ordered = cfg.eligible_tasks(df_filtered)
n_all_tasks       = len(all_tasks_ordered)
print(f"  Eligible tasks: {n_all_tasks}")

_beh_id_eligible = {k: v for k, v in cfg.BEH_ID_TO_TASK.items()
                    if v in set(all_tasks_ordered)}
full_ds, probe_uids, main_uids = aggregate_counts(
    df_filtered,
    beh_id_to_task  = _beh_id_eligible,
    use_trial_kinds = cfg.USE_TRIAL_KINDS,
)

# ---------------------------------------------------------------------------
# Probe matrix
# ---------------------------------------------------------------------------
probe_refs_ordered = sorted(
    [refs_by_uid[uid] for uid in probe_uids if uid in refs_by_uid],
    key=lambda r: r.latent_state,
)
probe_uids_ordered = [r.uid for r in probe_refs_ordered]
n_probe            = len(probe_uids_ordered)
uid_to_row         = {uid: i for i, uid in enumerate(probe_uids_ordered)}
task_to_col        = {t: j for j, t in enumerate(all_tasks_ordered)}

probe_cells_df = full_ds.df[full_ds.df["uid"].isin(probe_uids)].copy()
true_matrix    = np.full((n_probe, n_all_tasks), np.nan)
count_matrix   = np.zeros((n_probe, n_all_tasks), dtype=np.int32)
for row in probe_cells_df.itertuples(index=False):
    i     = uid_to_row.get(row.uid)
    j     = task_to_col.get(row.task_name)
    total = row.count_0 + row.count_1
    if i is not None and j is not None and total > 0:
        true_matrix[i, j]  = row.count_1 / total
        count_matrix[i, j] = total

_nf_mask = count_matrix > 1
probe_noise_floor = (
    float(np.mean(true_matrix[_nf_mask] * (1 - true_matrix[_nf_mask])
                  / (count_matrix[_nf_mask].astype(float) - 1)))
    if _nf_mask.any() else 0.0
)
_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# Spearman ρ noise ceiling  (split-half + Spearman-Brown)
# ---------------------------------------------------------------------------
def _rho_noise_ceiling(cells_df: pd.DataFrame,
                       n_splits: int = 200, seed: int = 0) -> float:
    df = cells_df.copy()
    df["total"] = df["count_0"] + df["count_1"]
    df = df[df["total"] >= 2].reset_index(drop=True)
    if len(df) < 2:
        return float("nan")
    totals  = df["total"].values.astype(int)
    count1s = df["count_1"].values.astype(int)
    n1s     = totals // 2
    n2s     = totals - n1s
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        k1 = np.array([rng.hypergeometric(c1, t - c1, n1)
                       for c1, t, n1 in zip(count1s, totals, n1s)], dtype=float)
        p1 = k1 / n1s
        p2 = (count1s - k1) / n2s
        rh, _ = spearmanr(p1, p2)
        if not np.isnan(rh) and rh > -1:
            vals.append((2 * rh) / (1 + rh))
    return float(np.mean(vals)) if vals else float("nan")

rho_noise_ceiling = _rho_noise_ceiling(probe_cells_df)
print(f"  Spearman ρ noise ceiling: {rho_noise_ceiling:.4f}")

# ---------------------------------------------------------------------------
# Fixed cell-level 10 % eval split  (for all-data model selection)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split    = np.random.default_rng(cfg.SEED)
n_eval_cells = max(1, int(len(main_cells_df) * 0.10))
eval_idx     = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask    = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df        = main_cells_df[eval_mask].reset_index(drop=True)
pool_df        = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds_global = BehavioralDataset(eval_df)
print(f"\n  Eval cells (all-data model sel): {len(eval_df)}")
print(f"  Train pool cells (90 %%):        {len(pool_df)}")

# ---------------------------------------------------------------------------
# Per-task trial pools
#   task_trial_pools     — full main pool  (used for budget-grid sampling)
#   task_trial_pools_all — 90 % pool only  (used for all-data point)
# ---------------------------------------------------------------------------
task_trial_pools: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in main_cells_df.itertuples(index=False):
    tn = row.task_name
    if tn not in task_trial_pools:
        continue
    pool = task_trial_pools[tn]
    for _ in range(int(row.count_0)):
        pool.append((row.uid, 0))
    for _ in range(int(row.count_1)):
        pool.append((row.uid, 1))

task_trial_pools_all: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in pool_df.itertuples(index=False):
    tn = row.task_name
    if tn not in task_trial_pools_all:
        continue
    p = task_trial_pools_all[tn]
    for _ in range(int(row.count_0)):
        p.append((row.uid, 0))
    for _ in range(int(row.count_1)):
        p.append((row.uid, 1))

pool_sizes       = {t: len(task_trial_pools[t]) for t in all_tasks_ordered}
total_pool_size  = sum(pool_sizes.values())
pool_all_size    = sum(len(v) for v in task_trial_pools_all.values())
avg_pool_per_task = pool_all_size / n_all_tasks
print(f"\n  Trial pool (full) — min: {min(pool_sizes.values())}  "
      f"max: {max(pool_sizes.values())}  total: {total_pool_size:,}")
print(f"  Trial pool (90 %) — total: {pool_all_size:,}  "
      f"avg/task: {avg_pool_per_task:.1f}")

trials_per_task = list(cfg.TRIALS_PER_TASK)
if cfg.FAST_PASS:
    trials_per_task = [trials_per_task[0]]
    print("  FAST_PASS=True → min tpt only")
print(f"  Trials-per-task grid ({len(trials_per_task)} points): {trials_per_task}")

# ---------------------------------------------------------------------------
# CLIP feature cache
# ---------------------------------------------------------------------------
_agent_tmp  = DlbtAgent(freeze_encoder=True, n_mc_samples=1,
                        device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
_cache_path = _REPO_ROOT / cfg.CACHE_PATH
if _cache_path.exists():
    _agent_tmp.load_cache(str(_cache_path))
else:
    _agent_tmp.precompute_features(all_refs)
    _agent_tmp.save_cache(str(_cache_path))
frozen_clip = {uid: feat.clone() for uid, feat in _agent_tmp._cache.items()}
del _agent_tmp
print(f"CLIP cache ready ({len(frozen_clip)} images).")

# ---------------------------------------------------------------------------
# DLBT base agent  (symmetric Dirichlet α = BASE_CONCENTRATION → P ≈ 0.5)
# ---------------------------------------------------------------------------
base_agent = DlbtAgent(
    freeze_encoder    = True,
    n_mc_samples      = cfg.N_MC,
    device            = device,
    mapper_hidden     = cfg.MAPPER_HIDDEN,
    normalize_utility = cfg.NORMALIZED_UTILITY,
)
base_agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
with torch.no_grad():
    _lin = base_agent.mapper[0]
    _lin.weight.zero_()
    _lin.bias.fill_(cfg.BASE_CONCENTRATION)
base_agent.eval()
print(f"Base agent ready (α = {cfg.BASE_CONCENTRATION}).")

# ---------------------------------------------------------------------------
# Helpers — data
# ---------------------------------------------------------------------------

def _sample_and_split(
    tasks: list,
    tpt: int,
    rng: np.random.Generator,
) -> tuple[BehavioralDataset, BehavioralDataset]:
    """
    Sample `tpt` trials per task from task_trial_pools (bootstrap if pool < tpt).
    Aggregate to (uid, task) cells, split 90/10 at cell level.
    Returns (train_ds, val_ds).
    """
    all_trials = []
    for task_name in tasks:
        pool = task_trial_pools[task_name]
        if tpt == 0 or len(pool) == 0:
            continue
        replace = len(pool) < tpt
        chosen  = rng.choice(len(pool), size=tpt, replace=replace)
        for idx in chosen:
            uid, outcome = pool[int(idx)]
            all_trials.append((uid, task_name, outcome))

    if not all_trials:
        empty = BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
        return empty, empty

    rows    = [{"uid": uid, "task_name": tn, "count_0": 1 - out, "count_1": out}
               for uid, tn, out in all_trials]
    cell_df = (pd.DataFrame(rows)
               .groupby(["uid", "task_name"])[["count_0", "count_1"]]
               .sum().reset_index())

    n_cells  = len(cell_df)
    n_val    = max(1, int(n_cells * 0.10))
    perm     = rng.permutation(n_cells)
    val_mask = np.zeros(n_cells, dtype=bool)
    val_mask[perm[:n_val]] = True

    return (BehavioralDataset(cell_df[~val_mask].reset_index(drop=True)),
            BehavioralDataset(cell_df[ val_mask].reset_index(drop=True)))


def _all_data_ds(tasks: list) -> BehavioralDataset:
    """Build training dataset from the fixed 90 % pool (task_trial_pools_all)."""
    rows = []
    for task_name in tasks:
        for uid, outcome in task_trial_pools_all[task_name]:
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _set_mapper_bias(agent, seed: int):
    """Shared mapper bias initialisation for DLBT, DetBT and OneHotBT."""
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    rng_init  = np.random.default_rng(seed)
    alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                 size=(_linear.bias.shape[0],)).astype(np.float32)
    with torch.no_grad():
        _linear.bias.copy_(
            torch.from_numpy(np.log(np.exp(alpha_rnd) - 1.0)).to(device))


def _init_dlbt(seed: int) -> DlbtAgent:
    torch.manual_seed(seed)
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _set_mapper_bias(agent, seed)
    return agent


def _init_detbt(seed: int) -> DetBTAgent:
    torch.manual_seed(seed)
    agent = DetBTAgent(
        freeze_encoder    = True,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _set_mapper_bias(agent, seed)
    return agent


def _init_onehot(seed: int) -> OneHotBTAgent:
    torch.manual_seed(seed)
    agent = OneHotBTAgent(
        freeze_encoder    = True,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _set_mapper_bias(agent, seed)
    return agent


def _base_mse_on_ds(val_ds: BehavioralDataset) -> float:
    """MSE of base model (P=0.5 everywhere) on val_ds — analytical."""
    if val_ds.df.empty:
        return float("nan")
    totals = (val_ds.df["count_0"] + val_ds.df["count_1"]).values.astype(float)
    p_obs  = val_ds.df["count_1"].values / np.clip(totals, 1, None)
    return float(np.mean((0.5 - p_obs) ** 2))


@torch.no_grad()
def _ds_mse_agent(agent, val_ds: BehavioralDataset) -> float:
    """MSE of agent predictions on val_ds."""
    if val_ds.df.empty:
        return float("nan")
    agent.eval()
    pred_list, true_list = [], []
    for task_name, group in val_ds.df.groupby("task_name"):
        task = get_task(task_name)
        uids = [uid for uid in group["uid"].tolist() if uid in refs_by_uid]
        if not uids:
            continue
        refs  = [refs_by_uid[uid] for uid in uids]
        probs = agent.choice_probs(refs, task)[:, 1].cpu().numpy()
        g_sub = group[group["uid"].isin(uids)]
        tot   = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_obs = g_sub["count_1"].values / np.clip(tot, 1, None)
        pred_list.extend(probs.tolist())
        true_list.extend(p_obs.tolist())
    if not pred_list:
        return float("nan")
    return float(np.mean((np.array(pred_list) - np.array(true_list)) ** 2))


def _run_dlbt(agent: DlbtAgent,
              train_ds: BehavioralDataset,
              val_ds: BehavioralDataset):
    """Phase 1 + optional Phase 2 (attnpool) + model selection vs base."""
    train_dlbt(
        agent, train_ds, val_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )
    if not cfg.FREEZE_ENCODER:
        gc.collect(); torch.cuda.empty_cache()
        for p in agent.mapper.parameters():
            p.requires_grad_(False)
        for p in agent.encoder.attnpool.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = False
        agent._cache.clear()
        opt2 = torch.optim.Adam(agent.encoder.attnpool.parameters(),
                                lr=cfg.LR_ATTNPOOL)
        train_dlbt(
            agent, train_ds, val_ds, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            optimizer = opt2,
        )
        agent.eval()
        agent.precompute_backbone_features(all_refs)
        with torch.no_grad():
            for i in range(0, len(all_refs), 16):
                batch   = all_refs[i : i + 16]
                spatial = torch.stack(
                    [agent._backbone_cache[r.uid] for r in batch]
                ).to(device)
                feats = agent.encoder.attnpool(spatial).float()
                for ref, feat in zip(batch, feats):
                    agent._cache[ref.uid] = feat.cpu()
        for p in agent.mapper.parameters():
            p.requires_grad_(True)
        agent.freeze_encoder = True

    if not val_ds.df.empty:
        if _base_mse_on_ds(val_ds) < _ds_mse_agent(agent, val_ds):
            return base_agent
    return agent


def _run_frozen(agent, train_ds: BehavioralDataset,
                val_ds: BehavioralDataset):
    """Phase 1 only (frozen encoder) + model selection vs base."""
    train_dlbt(
        agent, train_ds, val_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )
    if not val_ds.df.empty:
        if _base_mse_on_ds(val_ds) < _ds_mse_agent(agent, val_ds):
            return base_agent
    return agent


@torch.no_grad()
def _probe_matrix(agent) -> np.ndarray:
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    for j, task_name in enumerate(all_tasks_ordered):
        task  = get_task(task_name)
        probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
        pred[:, j] = probs
    return pred


def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    valid   = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _  = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


# ---------------------------------------------------------------------------
# Behavioral supervision probe  (no training — ground-truth P=0/1)
# ---------------------------------------------------------------------------

def _behav_superv_probe_matrix() -> np.ndarray:
    """
    P(right | image, task) = 1.0 if delta_u[image.latent_state] > 0, else 0.0.
    Uses the true rendering-parameter latent state, not the learned mapper.
    No training — evaluated once.
    """
    pred = np.full((n_probe, n_all_tasks), np.nan)
    for j, task_name in enumerate(all_tasks_ordered):
        task = get_task(task_name)
        du   = task.delta_u  # [K], entries ±1
        for i, ref in enumerate(probe_refs_ordered):
            pred[i, j] = 1.0 if float(du[ref.latent_state]) > 0 else 0.0
    return pred


# ===========================================================================
# Reference baseline: no behavioral supervision (ground-truth P=0/1)
# ===========================================================================

print("\nEvaluating no-behavioral-supervision baseline (ground-truth P=0/1)...")
pred_bsup              = _behav_superv_probe_matrix()
bsup_cmse, bsup_rho   = _probe_stats(pred_bsup)
print(f"  BehavSuperv cMSE−NF={bsup_cmse:+.5f}  ρ={bsup_rho:.4f}")

# ===========================================================================
# Main sweep
# ===========================================================================
n_tpt   = len(trials_per_task)
n_seeds = len(cfg.SEEDS)

dlbt_cmse   = np.full((n_seeds, n_tpt), np.nan)
dlbt_rho    = np.full((n_seeds, n_tpt), np.nan)
detbt_cmse  = np.full((n_seeds, n_tpt), np.nan)
detbt_rho   = np.full((n_seeds, n_tpt), np.nan)
onehot_cmse = np.full((n_seeds, n_tpt), np.nan)
onehot_rho  = np.full((n_seeds, n_tpt), np.nan)

dlbt_all_cmse   = np.full(n_seeds, np.nan)
dlbt_all_rho    = np.full(n_seeds, np.nan)
detbt_all_cmse  = np.full(n_seeds, np.nan)
detbt_all_rho   = np.full(n_seeds, np.nan)
onehot_all_cmse = np.full(n_seeds, np.nan)
onehot_all_rho  = np.full(n_seeds, np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    rng_dlbt   = np.random.default_rng(seed_val)
    rng_detbt  = np.random.default_rng(seed_val + 50_000)
    rng_onehot = np.random.default_rng(seed_val + 150_000)

    # ---- Budget grid -------------------------------------------------------
    for b_i, tpt in enumerate(trials_per_task):
        print(f"\n  tpt={tpt:>5,}  (total≈{tpt*n_all_tasks:,})  [{b_i+1}/{n_tpt}]")

        # DLBT
        train_ds, val_ds = _sample_and_split(all_tasks_ordered, tpt, rng_dlbt)
        agent  = _init_dlbt(seed_val)
        chosen = _run_dlbt(agent, train_ds, val_ds)
        pred   = _probe_matrix(chosen)
        dlbt_cmse[s_i, b_i], dlbt_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    DLBT     cMSE−NF={dlbt_cmse[s_i,b_i]:+.5f}  ρ={dlbt_rho[s_i,b_i]:.3f}"
              f"  (base={'yes' if chosen is base_agent else 'no'})")
        del agent, chosen, train_ds, val_ds, pred
        gc.collect(); torch.cuda.empty_cache()

        # DetBT  (perceptual stochasticity)
        train_ds, val_ds = _sample_and_split(all_tasks_ordered, tpt, rng_detbt)
        agent  = _init_detbt(seed_val)
        chosen = _run_frozen(agent, train_ds, val_ds)
        pred   = _probe_matrix(chosen)
        detbt_cmse[s_i, b_i], detbt_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    DetBT    cMSE−NF={detbt_cmse[s_i,b_i]:+.5f}  ρ={detbt_rho[s_i,b_i]:.3f}"
              f"  (base={'yes' if chosen is base_agent else 'no'})")
        del agent, chosen, train_ds, val_ds, pred
        gc.collect(); torch.cuda.empty_cache()

        # OneHotBT  (perceptual uncertainty)
        train_ds, val_ds = _sample_and_split(all_tasks_ordered, tpt, rng_onehot)
        agent  = _init_onehot(seed_val)
        chosen = _run_frozen(agent, train_ds, val_ds)
        pred   = _probe_matrix(chosen)
        onehot_cmse[s_i, b_i], onehot_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    OneHotBT cMSE−NF={onehot_cmse[s_i,b_i]:+.5f}  ρ={onehot_rho[s_i,b_i]:.3f}"
              f"  (base={'yes' if chosen is base_agent else 'no'})")
        del agent, chosen, train_ds, val_ds, pred
        gc.collect(); torch.cuda.empty_cache()

    # ---- All-data point  (fixed 90 % pool, fixed global eval split) --------
    print(f"\n  [All data — {pool_all_size:,} train trials, avg {avg_pool_per_task:.0f}/task]")
    all_ds = _all_data_ds(all_tasks_ordered)

    agent  = _init_dlbt(seed_val)
    chosen = _run_dlbt(agent, all_ds, eval_ds_global)
    pred   = _probe_matrix(chosen)
    dlbt_all_cmse[s_i], dlbt_all_rho[s_i] = _probe_stats(pred)
    print(f"    DLBT all     cMSE−NF={dlbt_all_cmse[s_i]:+.5f}  ρ={dlbt_all_rho[s_i]:.3f}"
          f"  (base={'yes' if chosen is base_agent else 'no'})")
    del agent, chosen, pred
    gc.collect(); torch.cuda.empty_cache()

    agent  = _init_detbt(seed_val)
    chosen = _run_frozen(agent, all_ds, eval_ds_global)
    pred   = _probe_matrix(chosen)
    detbt_all_cmse[s_i], detbt_all_rho[s_i] = _probe_stats(pred)
    print(f"    DetBT all    cMSE−NF={detbt_all_cmse[s_i]:+.5f}  ρ={detbt_all_rho[s_i]:.3f}"
          f"  (base={'yes' if chosen is base_agent else 'no'})")
    del agent, chosen, pred
    gc.collect(); torch.cuda.empty_cache()

    agent  = _init_onehot(seed_val)
    chosen = _run_frozen(agent, all_ds, eval_ds_global)
    pred   = _probe_matrix(chosen)
    onehot_all_cmse[s_i], onehot_all_rho[s_i] = _probe_stats(pred)
    print(f"    OneHotBT all cMSE−NF={onehot_all_cmse[s_i]:+.5f}  ρ={onehot_all_rho[s_i]:.3f}"
          f"  (base={'yes' if chosen is base_agent else 'no'})")
    del agent, chosen, pred
    gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
    "trials_per_task":     trials_per_task,
    "avg_pool_per_task":   avg_pool_per_task,
    "total_pool_size":     total_pool_size,
    "seeds":               cfg.SEEDS,
    "all_tasks_ordered":   all_tasks_ordered,
    "probe_uids_ordered":  probe_uids_ordered,
    "true_matrix":         true_matrix,
    "count_matrix":        count_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_net":     random_cmse_net,
    "rho_noise_ceiling":   rho_noise_ceiling,
    # Reference baseline: no behavioral supervision (ground-truth P=0/1)
    "behav_superv_cmse":       bsup_cmse,
    "behav_superv_rho":        bsup_rho,
    # Budget sweep [n_seeds × n_budgets]
    "dlbt_cmse":           dlbt_cmse,
    "dlbt_rho":            dlbt_rho,
    "detbt_cmse":          detbt_cmse,
    "detbt_rho":           detbt_rho,
    "onehot_cmse":         onehot_cmse,
    "onehot_rho":          onehot_rho,
    # All-data point [n_seeds]
    "dlbt_all_cmse":       dlbt_all_cmse,
    "dlbt_all_rho":        dlbt_all_rho,
    "detbt_all_cmse":      detbt_all_cmse,
    "detbt_all_rho":       detbt_all_rho,
    "onehot_all_cmse":     onehot_all_cmse,
    "onehot_all_rho":      onehot_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
