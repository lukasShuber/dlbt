"""
run1/023_efficiency_main/run.py — full-coverage budget sweep with bootstrap
sampling and anti-human reference.

Protocol
--------
1.  Load + filter run0+run1 data; identify all eligible tasks.
2.  Separate probe images (held-out evaluation) from main images (training).
3.  10 % of main cells held out for DLBT early stopping.
4.  Build per-task individual trial pools from the remaining 90 %.
5.  Build ground-truth probe matrix; compute noise floor.
6.  Pre-compute frozen CLIP features.

7.  For each seed (N_SEEDS total):
      - Both DLBT init weights AND trial sampling change per seed, giving
        genuine variance for SEM error bands.
      a. Budget grid points (TRIAL_BUDGETS):
           For each budget B:
             i.  Sample training data: per task t, allocate B//n_tasks trials.
                 If the pool for t is large enough → sample without replacement.
                 If not → bootstrap (sample with replacement) up to the target.
             ii. Train DLBT on sampled data.          → probe cMSE-NF, ρ
            iii. Train SLDA on sampled data.          → probe cMSE-NF, ρ
             iv. Train anti-human DLBT on label-
                 flipped version of the same sample.  → probe cMSE-NF, ρ
      b. All-data point (every trial in the pool, no sampling):
           Train DLBT, SLDA, anti-human DLBT on the full pool.

8.  Compute random-guesser and random-init DLBT baselines (no seeds).

9.  Save summary dict as results/efficiency_main.pkl.

Run from repo root:
    python experiments/behavior/run1/023_efficiency_main/run.py
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
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.train_slda_attnpool import finetune_slda_attnpool

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
print(f"  Raw trials: {len(df_raw):,}  "
      f"({df_raw['assignment_id'].nunique()} assignments)")

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
print(f"  Aggregated: {len(full_ds):,} cells  "
      f"({full_ds.df['task_name'].nunique()} tasks, "
      f"{full_ds.df['uid'].nunique()} images)")

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
print(f"  Probe images: {n_probe}")

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
if _nf_mask.any():
    _p = true_matrix[_nf_mask]
    _n = count_matrix[_nf_mask].astype(float)
    probe_noise_floor = float(np.mean(_p * (1 - _p) / (_n - 1)))
else:
    probe_noise_floor = 0.0

_valid_rg       = ~np.isnan(true_matrix)
random_cmse_net = float(np.mean((0.5 - true_matrix[_valid_rg]) ** 2)) - probe_noise_floor
print(f"  Probe NF: {probe_noise_floor:.5f}  random-guesser cMSE−NF: {random_cmse_net:.5f}")

# ---------------------------------------------------------------------------
# 10 % eval split of main cells (DLBT early stopping)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split     = np.random.default_rng(cfg.SEED)
n_eval_cells  = max(1, int(len(main_cells_df) * 0.10))
eval_idx      = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask     = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df  = main_cells_df[eval_mask].reset_index(drop=True)
pool_df  = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds  = BehavioralDataset(eval_df)

# Flipped eval set for anti-human early stopping:
# the anti-human model learns p → 1-p, so its validation loss must also be
# measured on flipped labels — otherwise early stopping fires immediately.
eval_df_anti          = eval_df.copy()
eval_df_anti["count_0"], eval_df_anti["count_1"] = (
    eval_df["count_1"].copy(), eval_df["count_0"].copy()
)
eval_ds_anti = BehavioralDataset(eval_df_anti)

print(f"\n  Eval cells (early stopping): {len(eval_df)}")
print(f"  Train pool cells (90 %%):    {len(pool_df)}")

# ---------------------------------------------------------------------------
# Per-task individual trial pools  task_name -> [(uid, outcome), ...]
# ---------------------------------------------------------------------------
task_trial_pools: dict[str, list] = {t: [] for t in all_tasks_ordered}
for row in pool_df.itertuples(index=False):
    tn = row.task_name
    if tn not in task_trial_pools:
        continue
    pool = task_trial_pools[tn]
    for _ in range(int(row.count_0)):
        pool.append((row.uid, 0))
    for _ in range(int(row.count_1)):
        pool.append((row.uid, 1))

pool_sizes      = {t: len(task_trial_pools[t]) for t in all_tasks_ordered}
total_pool_size = sum(pool_sizes.values())
print(f"\n  Trial pool — min: {min(pool_sizes.values())}  "
      f"max: {max(pool_sizes.values())}  "
      f"total: {total_pool_size:,}")

# Cap budget grid at the total pool size — budgets exceeding the pool would
# require bootstrapping every task, making them meaningless upper-budget points.
trial_budgets = [b for b in cfg.TRIAL_BUDGETS if b <= total_pool_size]
if not trial_budgets:
    trial_budgets = cfg.TRIAL_BUDGETS[:1]   # always keep at least one point
if cfg.FAST_PASS and len(trial_budgets) > 2:
    trial_budgets = [trial_budgets[0], trial_budgets[-1]]
    print("  FAST_PASS=True → min + max budget only")
print(f"  Budget grid ({len(trial_budgets)} points, capped at pool): {trial_budgets}")

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
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_sample(tasks: list, budget: int,
                       rng: np.random.Generator,
                       flip: bool = False) -> BehavioralDataset:
    """
    Allocate `budget` trials uniformly across `tasks`.

    Per-task target k = floor(budget / n_tasks), with remainder distributed
    randomly.  Sampling is:
      - Without replacement if pool_size >= k  (standard)
      - With replacement    if pool_size <  k  (bootstrap fallback)

    If flip=True, all trial outcomes are inverted (0↔1) — anti-human condition.
    """
    n     = len(tasks)
    q     = budget // n
    r     = budget % n
    extra = set(int(i) for i in rng.choice(n, size=r, replace=False))
    rows  = []
    for i, task_name in enumerate(tasks):
        k    = q + (1 if i in extra else 0)
        pool = task_trial_pools[task_name]
        if k == 0 or len(pool) == 0:
            continue
        replace = len(pool) < k
        chosen  = rng.choice(len(pool), size=k, replace=replace)
        for idx in chosen:
            uid, outcome = pool[int(idx)]
            if flip:
                outcome = 1 - outcome
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _all_data_ds(tasks: list, flip: bool = False) -> BehavioralDataset:
    """Use every trial in the pool — no sampling, no budget cap."""
    rows = []
    for task_name in tasks:
        for uid, outcome in task_trial_pools[task_name]:
            if flip:
                outcome = 1 - outcome
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _init_agent(seed: int) -> DlbtAgent:
    """
    Fresh DlbtAgent with frozen CLIP cache and initialised mapper bias.
    Uses `seed` for both torch weight init and bias sampling, so different
    seeds give genuinely different starting points — producing meaningful
    variance across the seed loop for SEM computation.
    """
    torch.manual_seed(seed)
    agent = DlbtAgent(
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
        median_correction = cfg.MEDIAN_CORRECTION,
        neutral_alpha     = cfg.NEUTRAL_ALPHA,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    if cfg.INIT_MODE == "uniform":
        bv = float(np.log(np.exp(cfg.INIT_ALPHA) - 1.0))
        with torch.no_grad():
            _linear.bias.fill_(bv)
    elif cfg.INIT_MODE == "random":
        rng_init  = np.random.default_rng(seed)   # tied to outer seed
        alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                     size=(_linear.bias.shape[0],)).astype(np.float32)
        b_init    = np.log(np.exp(alpha_rnd) - 1.0)
        with torch.no_grad():
            _linear.bias.copy_(torch.from_numpy(b_init).to(device))
    else:
        raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")
    return agent


def _train_one(agent: DlbtAgent, train_ds: BehavioralDataset,
               val_ds: BehavioralDataset | None = None):
    """Train agent.  val_ds defaults to the standard eval set; pass
    eval_ds_anti for anti-human runs so early stopping uses flipped labels."""
    return train_dlbt(
        agent, train_ds, val_ds if val_ds is not None else eval_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )


def _dlbt_probe_matrix(agent: DlbtAgent) -> np.ndarray:
    pred = np.full((n_probe, n_all_tasks), np.nan)
    agent.eval()
    with torch.no_grad():
        for j, task_name in enumerate(all_tasks_ordered):
            task  = get_task(task_name)
            probs = agent.choice_probs(probe_refs_ordered, task)[:, 1].cpu().numpy()
            pred[:, j] = probs
    return pred


def _fit_slda(tasks: list, train_ds: BehavioralDataset):
    scalers, models, temps = {}, {}, {}
    for task_name in tasks:
        group = train_ds.df[train_ds.df["task_name"] == task_name]
        uids  = [uid for uid in group["uid"].tolist() if uid in frozen_clip]
        if len(uids) < 1:
            continue
        X       = np.array([frozen_clip[uid].cpu().numpy() for uid in uids])
        g_sub   = group[group["uid"].isin(uids)]
        totals  = (g_sub["count_0"] + g_sub["count_1"]).values.astype(float)
        p_right = g_sub["count_1"].values / np.clip(totals, 1, None)

        scaler = StandardScaler(with_mean=(len(uids) >= 5),
                                with_std=(len(uids) >= 5))
        X_sc   = scaler.fit_transform(X)
        model  = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
        model.fit(X_sc, p_right)

        p_pred = np.clip(model.predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))

        def _nll(log_tau, logits=logits, y=p_right):
            p = 1 / (1 + np.exp(-logits / np.exp(log_tau)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        opt = minimize_scalar(_nll, bounds=(-3.0, 3.0), method="bounded")
        scalers[task_name] = scaler
        models[task_name]  = model
        temps[task_name]   = float(np.exp(opt.x))
    return scalers, models, temps


def _slda_probe_matrix(scalers, models, temps,
                       probe_features: dict | None = None) -> np.ndarray:
    """
    probe_features: uid → np.array override (used after SLDA Stage 2 so the
    fresh attnpool features are used instead of the frozen_clip cache).
    """
    pred            = np.full((n_probe, n_all_tasks), np.nan)
    feat_src        = probe_features if probe_features is not None else {
        uid: frozen_clip[uid].cpu().numpy() for uid in frozen_clip
    }
    probe_uids_clip = [uid for uid in probe_uids_ordered if uid in feat_src]
    probe_X         = np.array([feat_src[uid] for uid in probe_uids_clip])
    for j, task_name in enumerate(all_tasks_ordered):
        if task_name not in models:
            continue
        X_sc   = scalers[task_name].transform(probe_X)
        p_pred = np.clip(models[task_name].predict(X_sc), 1e-6, 1 - 1e-6)
        logits = np.log(p_pred / (1 - p_pred))
        p_cal  = 1 / (1 + np.exp(-logits / temps[task_name]))
        for i_clip, uid in enumerate(probe_uids_clip):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_cal[i_clip])
    return pred


def _init_slda_attnpool_agent() -> DlbtAgent:
    """Fresh DlbtAgent used only for SLDA Stage 2 attnpool fine-tuning.
    Backbone cache is pre-populated so each training step only runs attnpool."""
    ag = DlbtAgent(freeze_encoder=False, n_mc_samples=1,
                   device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
    ag.precompute_backbone_features(all_refs)
    return ag


@torch.no_grad()
def _slda_probe_features(agent_slda: DlbtAgent) -> dict:
    """Extract probe features from a (fine-tuned) attnpool agent."""
    agent_slda.eval()
    return {
        uid: agent_slda._encode([refs_by_uid[uid]])[0].cpu().numpy()
        for uid in probe_uids_ordered if uid in refs_by_uid
    }


def _run_slda(tasks: list, train_ds: BehavioralDataset,
              eval_ds_slda: BehavioralDataset):
    """
    Full SLDA pipeline:
      Stage 1 (always)        — fit per-task ridge decoders on frozen CLIP.
      Stage 2 (if not frozen) — fine-tune attnpool through fixed decoders.

    Returns (pred_matrix, scalers, models, temps).
    """
    scalers, models, temps = _fit_slda(tasks, train_ds)

    if not cfg.FREEZE_ENCODER:
        agent_slda = _init_slda_attnpool_agent()
        finetune_slda_attnpool(
            agent_slda, scalers, models, temps,
            train_ds, eval_ds_slda, refs_dict,
            n_epochs  = cfg.N_EPOCHS_PHASE2,
            patience  = cfg.PATIENCE_PHASE2,
            lr        = cfg.LR_ATTNPOOL,
        )
        pred = _slda_probe_matrix(scalers, models, temps,
                                  probe_features=_slda_probe_features(agent_slda))
        del agent_slda
        gc.collect(); torch.cuda.empty_cache()
    else:
        pred = _slda_probe_matrix(scalers, models, temps)

    return pred, scalers, models, temps


def _probe_stats(pred_mat: np.ndarray) -> tuple[float, float]:
    """Return (cMSE−NF, Spearman ρ) for a predicted probe matrix."""
    valid    = ~np.isnan(pred_mat) & ~np.isnan(true_matrix)
    cmse_nf  = float(np.mean((pred_mat[valid] - true_matrix[valid]) ** 2)) - probe_noise_floor
    rho, _   = spearmanr(pred_mat[valid], true_matrix[valid])
    return cmse_nf, float(rho)


# ---------------------------------------------------------------------------
# Random-guesser and random-init DLBT baselines
# ---------------------------------------------------------------------------
print("\nComputing baselines...")
pred_random          = np.full((n_probe, n_all_tasks), 0.5)
random_cmse_nf, _    = _probe_stats(pred_random)

# Random-init DLBT: average over all seeds for stability
_rnd_cmse_list = []
for _sv in cfg.SEEDS:
    _ag = _init_agent(_sv)
    _pm = _dlbt_probe_matrix(_ag)
    _c, _ = _probe_stats(_pm)
    _rnd_cmse_list.append(_c)
    del _ag
random_init_cmse_nf = float(np.mean(_rnd_cmse_list))
print(f"  Random-guesser cMSE−NF : {random_cmse_nf:.5f}")
print(f"  Random-init DLBT cMSE−NF : {random_init_cmse_nf:.5f}")

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
n_budgets = len(trial_budgets)
n_seeds   = len(cfg.SEEDS)

# [n_seeds × n_budgets] result matrices
dlbt_cmse = np.full((n_seeds, n_budgets), np.nan)
dlbt_rho  = np.full((n_seeds, n_budgets), np.nan)
slda_cmse = np.full((n_seeds, n_budgets), np.nan)
slda_rho  = np.full((n_seeds, n_budgets), np.nan)
anti_cmse = np.full((n_seeds, n_budgets), np.nan)
anti_rho  = np.full((n_seeds, n_budgets), np.nan)

# [n_seeds] all-data results
dlbt_all_cmse = np.full(n_seeds, np.nan)
dlbt_all_rho  = np.full(n_seeds, np.nan)
slda_all_cmse = np.full(n_seeds, np.nan)
slda_all_rho  = np.full(n_seeds, np.nan)
anti_all_cmse = np.full(n_seeds, np.nan)
anti_all_rho  = np.full(n_seeds, np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    # Separate rngs per model type so DLBT, SLDA, and anti each draw
    # independently — but all are reproducible given seed_val.
    rng_dlbt = np.random.default_rng(seed_val)
    rng_slda = np.random.default_rng(seed_val + 100_000)
    rng_anti = np.random.default_rng(seed_val + 200_000)

    # ---- Budget grid -------------------------------------------------------
    for b_i, budget in enumerate(trial_budgets):
        print(f"\n  Budget {budget:>7,}  [{b_i+1}/{n_budgets}]")

        # DLBT
        train_ds = _bootstrap_sample(all_tasks_ordered, budget, rng_dlbt)
        agent    = _init_agent(seed_val)
        _train_one(agent, train_ds)
        pred     = _dlbt_probe_matrix(agent)
        dlbt_cmse[s_i, b_i], dlbt_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    DLBT   cMSE−NF={dlbt_cmse[s_i,b_i]:+.5f}  ρ={dlbt_rho[s_i,b_i]:.3f}")
        del agent, train_ds, pred
        gc.collect(); torch.cuda.empty_cache()

        # SLDA
        train_ds_s = _bootstrap_sample(all_tasks_ordered, budget, rng_slda)
        pred_s, sca, mod, tmp = _run_slda(all_tasks_ordered, train_ds_s, eval_ds)
        slda_cmse[s_i, b_i], slda_rho[s_i, b_i] = _probe_stats(pred_s)
        print(f"    SLDA   cMSE−NF={slda_cmse[s_i,b_i]:+.5f}  ρ={slda_rho[s_i,b_i]:.3f}")
        del train_ds_s, sca, mod, tmp, pred_s

        # Anti-human DLBT (label-flipped, early stopping on flipped val set)
        train_ds_a = _bootstrap_sample(all_tasks_ordered, budget, rng_anti, flip=True)
        agent_a    = _init_agent(seed_val)
        _train_one(agent_a, train_ds_a, val_ds=eval_ds_anti)
        pred_a     = _dlbt_probe_matrix(agent_a)
        anti_cmse[s_i, b_i], anti_rho[s_i, b_i] = _probe_stats(pred_a)
        print(f"    Anti   cMSE−NF={anti_cmse[s_i,b_i]:+.5f}  ρ={anti_rho[s_i,b_i]:.3f}")
        del agent_a, train_ds_a, pred_a
        gc.collect(); torch.cuda.empty_cache()

    # ---- All-data point ----------------------------------------------------
    print(f"\n  [All data — {total_pool_size:,} trials]")

    all_ds = _all_data_ds(all_tasks_ordered)

    agent_all = _init_agent(seed_val)
    _train_one(agent_all, all_ds)
    pred_all = _dlbt_probe_matrix(agent_all)
    dlbt_all_cmse[s_i], dlbt_all_rho[s_i] = _probe_stats(pred_all)
    print(f"    DLBT all  cMSE−NF={dlbt_all_cmse[s_i]:+.5f}  ρ={dlbt_all_rho[s_i]:.3f}")
    del agent_all, pred_all
    gc.collect(); torch.cuda.empty_cache()

    pred_sa, sca, mod, tmp = _run_slda(all_tasks_ordered, all_ds, eval_ds)
    slda_all_cmse[s_i], slda_all_rho[s_i] = _probe_stats(pred_sa)
    print(f"    SLDA all  cMSE−NF={slda_all_cmse[s_i]:+.5f}  ρ={slda_all_rho[s_i]:.3f}")
    del sca, mod, tmp, pred_sa

    anti_all_ds = _all_data_ds(all_tasks_ordered, flip=True)
    agent_anti_all = _init_agent(seed_val)
    _train_one(agent_anti_all, anti_all_ds, val_ds=eval_ds_anti)
    pred_aa = _dlbt_probe_matrix(agent_anti_all)
    anti_all_cmse[s_i], anti_all_rho[s_i] = _probe_stats(pred_aa)
    print(f"    Anti all  cMSE−NF={anti_all_cmse[s_i]:+.5f}  ρ={anti_all_rho[s_i]:.3f}")
    del agent_anti_all, pred_aa, anti_all_ds
    gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "run_tag":             cfg.RUN_TAG,
    "trial_budgets":       trial_budgets,
    "total_pool_size":     total_pool_size,
    "seeds":               cfg.SEEDS,
    "all_tasks_ordered":   all_tasks_ordered,
    "probe_uids_ordered":  probe_uids_ordered,
    "true_matrix":         true_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_nf":      random_cmse_nf,
    "random_init_cmse_nf": random_init_cmse_nf,
    # Budget sweep [n_seeds × n_budgets]
    "dlbt_cmse":           dlbt_cmse,
    "dlbt_rho":            dlbt_rho,
    "slda_cmse":           slda_cmse,
    "slda_rho":            slda_rho,
    "anti_cmse":           anti_cmse,
    "anti_rho":            anti_rho,
    # All-data point [n_seeds]
    "dlbt_all_cmse":       dlbt_all_cmse,
    "dlbt_all_rho":        dlbt_all_rho,
    "slda_all_cmse":       slda_all_cmse,
    "slda_all_rho":        slda_all_rho,
    "anti_all_cmse":       anti_all_cmse,
    "anti_all_rho":        anti_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
