"""
run1/05_ablations/run.py — belief-representation ablation budget sweep.

Models compared at each trial budget:
  DLBT    full model: MC Dirichlet integration, mapper trained on behaviour
  DetBT   deterministic ablation: Dirichlet mean instead of MC samples
  SLDA    ridge decoder baseline (all tasks, frozen CLIP)
  Oracle  fixed beliefs from metadata latent state (no training, evaluated once)

Protocol mirrors 023_efficiency_main (bootstrap sampling, same budget grid,
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
from scipy.optimize import minimize_scalar
from scipy.special import expit as _sigmoid
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.detbt import DetBTAgent
from dlbt.agents.oracle_bt import OracleBTAgent
from dlbt.agents.rand_ont_bt import RandOntBTAgent, make_rand_ontology
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
# 10 % eval split of main cells (early stopping)
# ---------------------------------------------------------------------------
main_cells_df = (full_ds.df[full_ds.df["uid"].isin(main_uids)]
                 .copy().reset_index(drop=True))
rng_split    = np.random.default_rng(cfg.SEED)
n_eval_cells = max(1, int(len(main_cells_df) * 0.10))
eval_idx     = rng_split.choice(len(main_cells_df), size=n_eval_cells, replace=False)
eval_mask    = np.zeros(len(main_cells_df), dtype=bool)
eval_mask[eval_idx] = True

eval_df = main_cells_df[eval_mask].reset_index(drop=True)
pool_df = main_cells_df[~eval_mask].reset_index(drop=True)
eval_ds = BehavioralDataset(eval_df)
print(f"\n  Eval cells (early stopping): {len(eval_df)}")
print(f"  Train pool cells (90 %%):    {len(pool_df)}")

# ---------------------------------------------------------------------------
# Per-task trial pools
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
      f"max: {max(pool_sizes.values())}  total: {total_pool_size:,}")

trial_budgets = [b for b in cfg.TRIAL_BUDGETS if b <= total_pool_size]
if not trial_budgets:
    trial_budgets = cfg.TRIAL_BUDGETS[:1]
if cfg.FAST_PASS:
    trial_budgets = [trial_budgets[0]]
    print("  FAST_PASS=True → min budget only (all-data point covers the max)")
print(f"  Budget grid ({len(trial_budgets)} points): {trial_budgets}")

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
                       rng: np.random.Generator) -> BehavioralDataset:
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
            rows.append({"uid": uid, "task_name": task_name,
                         "count_0": 1 - outcome, "count_1": outcome})
    if not rows:
        return BehavioralDataset(pd.DataFrame(
            columns=["uid", "task_name", "count_0", "count_1"]))
    df  = pd.DataFrame(rows)
    agg = (df.groupby(["uid", "task_name"])[["count_0", "count_1"]]
              .sum().reset_index())
    return BehavioralDataset(agg)


def _all_data_ds(tasks: list) -> BehavioralDataset:
    rows = []
    for task_name in tasks:
        for uid, outcome in task_trial_pools[task_name]:
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
    """Shared mapper bias initialisation for DLBT and DetBT."""
    _linear = agent.mapper[0] if cfg.MAPPER_HIDDEN is None else agent.mapper[2]
    if cfg.INIT_MODE == "random":
        rng_init  = np.random.default_rng(seed)
        alpha_rnd = rng_init.uniform(cfg.INIT_ALPHA_LOW, cfg.INIT_ALPHA_HIGH,
                                     size=(_linear.bias.shape[0],)).astype(np.float32)
        with torch.no_grad():
            _linear.bias.copy_(
                torch.from_numpy(np.log(np.exp(alpha_rnd) - 1.0)).to(device))
    elif cfg.INIT_MODE == "uniform":
        bv = float(np.log(np.exp(cfg.INIT_ALPHA) - 1.0))
        with torch.no_grad():
            _linear.bias.fill_(bv)
    else:
        raise ValueError(f"Unknown INIT_MODE {cfg.INIT_MODE!r}")


def _init_dlbt(seed: int) -> DlbtAgent:
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


def _train_dlbt(agent: DlbtAgent, train_ds: BehavioralDataset) -> None:
    """Phase 1 + optional Phase 2 for DLBT."""
    phase1 = train_dlbt(
        agent, train_ds, eval_ds, refs_dict,
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
            agent, train_ds, eval_ds, refs_dict,
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
                ).to(agent.device)
                feats = agent.encoder.attnpool(spatial).float()
                for ref, feat in zip(batch, feats):
                    agent._cache[ref.uid] = feat.cpu()


def _train_detbt(agent: DetBTAgent, train_ds: BehavioralDataset) -> None:
    """Phase 1 only — DetBT always uses frozen encoder."""
    train_dlbt(
        agent, train_ds, eval_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )


def _init_randont(seed: int, rand_du: dict) -> RandOntBTAgent:
    """Initialise a RandOntBTAgent with the given pre-computed random utility map."""
    torch.manual_seed(seed)
    agent = RandOntBTAgent(
        rand_delta_u      = rand_du,
        freeze_encoder    = True,
        n_mc_samples      = cfg.N_MC,
        device            = device,
        mapper_hidden     = cfg.MAPPER_HIDDEN,
        normalize_utility = cfg.NORMALIZED_UTILITY,
        median_correction = cfg.MEDIAN_CORRECTION,
        neutral_alpha     = cfg.NEUTRAL_ALPHA,
    )
    agent._cache = {uid: feat.clone() for uid, feat in frozen_clip.items()}
    _set_mapper_bias(agent, seed)
    return agent


def _train_randont(agent: RandOntBTAgent, train_ds: BehavioralDataset) -> None:
    """Phase 1 only — same as DLBT frozen encoder."""
    train_dlbt(
        agent, train_ds, eval_ds, refs_dict,
        n_epochs = cfg.N_EPOCHS,
        lr       = cfg.LR,
        patience = cfg.PATIENCE,
    )


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
# SLDA helpers
# ---------------------------------------------------------------------------

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
        scaler  = StandardScaler(with_mean=(len(uids) >= 5),
                                 with_std=(len(uids) >= 5))
        X_sc    = scaler.fit_transform(X)
        model   = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
        model.fit(X_sc, p_right)
        p_pred  = np.clip(model.predict(X_sc), 1e-6, 1 - 1e-6)
        logits  = np.log(p_pred / (1 - p_pred))

        def _nll(log_tau, logits=logits, y=p_right):
            p = _sigmoid(logits / np.exp(log_tau))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        opt = minimize_scalar(_nll, bounds=(-3.0, 3.0), method="bounded")
        scalers[task_name] = scaler
        models[task_name]  = model
        temps[task_name]   = float(np.exp(opt.x))
    return scalers, models, temps


def _slda_probe_matrix(scalers, models, temps,
                       probe_features: dict | None = None) -> np.ndarray:
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
        p_cal  = _sigmoid(logits / temps[task_name])
        for i_clip, uid in enumerate(probe_uids_clip):
            row_i = uid_to_row.get(uid)
            if row_i is not None:
                pred[row_i, j] = float(p_cal[i_clip])
    return pred


def _init_slda_attnpool_agent() -> DlbtAgent:
    ag = DlbtAgent(freeze_encoder=False, n_mc_samples=1,
                   device=device, mapper_hidden=cfg.MAPPER_HIDDEN)
    ag.precompute_backbone_features(all_refs)
    return ag


@torch.no_grad()
def _slda_probe_features(agent_slda: DlbtAgent) -> dict:
    agent_slda.eval()
    return {uid: agent_slda._encode([refs_by_uid[uid]])[0].cpu().numpy()
            for uid in probe_uids_ordered if uid in refs_by_uid}


def _run_slda(tasks: list, train_ds: BehavioralDataset):
    scalers, models, temps = _fit_slda(tasks, train_ds)
    if not cfg.FREEZE_ENCODER_SLDA:
        agent_slda = _init_slda_attnpool_agent()
        finetune_slda_attnpool(
            agent_slda, scalers, models, temps,
            train_ds, eval_ds, refs_dict,
            n_epochs = cfg.N_EPOCHS_PHASE2,
            patience = cfg.PATIENCE_PHASE2,
            lr       = cfg.LR_ATTNPOOL,
        )
        pred = _slda_probe_matrix(scalers, models, temps,
                                  probe_features=_slda_probe_features(agent_slda))
        del agent_slda
        gc.collect(); torch.cuda.empty_cache()
    else:
        pred = _slda_probe_matrix(scalers, models, temps)
    return pred


# ===========================================================================
# Oracle agent  (no training — evaluated once)
# ===========================================================================
print("\nEvaluating Oracle agent...")
oracle_agent = OracleBTAgent(
    concentration     = cfg.ORACLE_CONCENTRATION,
    background        = cfg.ORACLE_BACKGROUND,
    device            = device,
    normalize_utility = cfg.NORMALIZED_UTILITY,
)
oracle_agent.eval()
pred_oracle        = _probe_matrix(oracle_agent)
oracle_cmse, oracle_rho = _probe_stats(pred_oracle)
del oracle_agent
print(f"  Oracle cMSE−NF={oracle_cmse:+.5f}  ρ={oracle_rho:.4f}")

# ===========================================================================
# Main sweep
# ===========================================================================
n_budgets = len(trial_budgets)
n_seeds   = len(cfg.SEEDS)

dlbt_cmse    = np.full((n_seeds, n_budgets), np.nan)
dlbt_rho     = np.full((n_seeds, n_budgets), np.nan)
detbt_cmse   = np.full((n_seeds, n_budgets), np.nan)
detbt_rho    = np.full((n_seeds, n_budgets), np.nan)
slda_cmse    = np.full((n_seeds, n_budgets), np.nan)
slda_rho     = np.full((n_seeds, n_budgets), np.nan)
randont_cmse = np.full((n_seeds, n_budgets), np.nan)
randont_rho  = np.full((n_seeds, n_budgets), np.nan)

dlbt_all_cmse    = np.full(n_seeds, np.nan)
dlbt_all_rho     = np.full(n_seeds, np.nan)
detbt_all_cmse   = np.full(n_seeds, np.nan)
detbt_all_rho    = np.full(n_seeds, np.nan)
slda_all_cmse    = np.full(n_seeds, np.nan)
slda_all_rho     = np.full(n_seeds, np.nan)
randont_all_cmse = np.full(n_seeds, np.nan)
randont_all_rho  = np.full(n_seeds, np.nan)

for s_i, seed_val in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_i+1}/{n_seeds}  (seed_val={seed_val})")

    rng_dlbt  = np.random.default_rng(seed_val)
    rng_detbt = np.random.default_rng(seed_val + 50_000)
    rng_slda  = np.random.default_rng(seed_val + 100_000)
    rng_rando = np.random.default_rng(seed_val + 200_000)

    # Sample a new random ontology for each seed (different random partitions
    # per seed averages over both training randomness and ontology randomness).
    rand_du = make_rand_ontology(all_tasks_ordered, seed=seed_val + 200_000)

    # ---- Budget grid -------------------------------------------------------
    for b_i, budget in enumerate(trial_budgets):
        print(f"\n  Budget {budget:>7,}  [{b_i+1}/{n_budgets}]")

        # DLBT
        train_ds = _bootstrap_sample(all_tasks_ordered, budget, rng_dlbt)
        agent    = _init_dlbt(seed_val)
        _train_dlbt(agent, train_ds)
        pred     = _probe_matrix(agent)
        dlbt_cmse[s_i, b_i], dlbt_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    DLBT   cMSE−NF={dlbt_cmse[s_i,b_i]:+.5f}  ρ={dlbt_rho[s_i,b_i]:.3f}")
        del agent, train_ds, pred
        gc.collect(); torch.cuda.empty_cache()

        # DetBT
        train_ds = _bootstrap_sample(all_tasks_ordered, budget, rng_detbt)
        agent    = _init_detbt(seed_val)
        _train_detbt(agent, train_ds)
        pred     = _probe_matrix(agent)
        detbt_cmse[s_i, b_i], detbt_rho[s_i, b_i] = _probe_stats(pred)
        print(f"    DetBT  cMSE−NF={detbt_cmse[s_i,b_i]:+.5f}  ρ={detbt_rho[s_i,b_i]:.3f}")
        del agent, train_ds, pred
        gc.collect(); torch.cuda.empty_cache()

        # SLDA
        train_ds_s = _bootstrap_sample(all_tasks_ordered, budget, rng_slda)
        pred_s     = _run_slda(all_tasks_ordered, train_ds_s)
        slda_cmse[s_i, b_i], slda_rho[s_i, b_i] = _probe_stats(pred_s)
        print(f"    SLDA   cMSE−NF={slda_cmse[s_i,b_i]:+.5f}  ρ={slda_rho[s_i,b_i]:.3f}")
        del train_ds_s, pred_s

        # RandOnt
        train_ds_r = _bootstrap_sample(all_tasks_ordered, budget, rng_rando)
        agent      = _init_randont(seed_val, rand_du)
        _train_randont(agent, train_ds_r)
        pred_r     = _probe_matrix(agent)
        randont_cmse[s_i, b_i], randont_rho[s_i, b_i] = _probe_stats(pred_r)
        print(f"    RandOnt cMSE−NF={randont_cmse[s_i,b_i]:+.5f}  ρ={randont_rho[s_i,b_i]:.3f}")
        del agent, train_ds_r, pred_r
        gc.collect(); torch.cuda.empty_cache()

    # ---- All-data point ----------------------------------------------------
    print(f"\n  [All data — {total_pool_size:,} trials]")
    all_ds = _all_data_ds(all_tasks_ordered)

    agent = _init_dlbt(seed_val)
    _train_dlbt(agent, all_ds)
    pred  = _probe_matrix(agent)
    dlbt_all_cmse[s_i], dlbt_all_rho[s_i] = _probe_stats(pred)
    print(f"    DLBT all  cMSE−NF={dlbt_all_cmse[s_i]:+.5f}  ρ={dlbt_all_rho[s_i]:.3f}")
    del agent, pred
    gc.collect(); torch.cuda.empty_cache()

    agent = _init_detbt(seed_val)
    _train_detbt(agent, all_ds)
    pred  = _probe_matrix(agent)
    detbt_all_cmse[s_i], detbt_all_rho[s_i] = _probe_stats(pred)
    print(f"    DetBT all cMSE−NF={detbt_all_cmse[s_i]:+.5f}  ρ={detbt_all_rho[s_i]:.3f}")
    del agent, pred
    gc.collect(); torch.cuda.empty_cache()

    pred_sa = _run_slda(all_tasks_ordered, all_ds)
    slda_all_cmse[s_i], slda_all_rho[s_i] = _probe_stats(pred_sa)
    print(f"    SLDA all  cMSE−NF={slda_all_cmse[s_i]:+.5f}  ρ={slda_all_rho[s_i]:.3f}")
    del pred_sa

    agent = _init_randont(seed_val, rand_du)
    _train_randont(agent, all_ds)
    pred  = _probe_matrix(agent)
    randont_all_cmse[s_i], randont_all_rho[s_i] = _probe_stats(pred)
    print(f"    RandOnt all cMSE−NF={randont_all_cmse[s_i]:+.5f}  ρ={randont_all_rho[s_i]:.3f}")
    del agent, pred
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
    "count_matrix":        count_matrix,
    "probe_noise_floor":   probe_noise_floor,
    "random_cmse_net":     random_cmse_net,
    "rho_noise_ceiling":   rho_noise_ceiling,
    # Oracle (scalar — no training)
    "oracle_cmse":         oracle_cmse,
    "oracle_rho":          oracle_rho,
    "oracle_concentration": cfg.ORACLE_CONCENTRATION,
    "oracle_background":   cfg.ORACLE_BACKGROUND,
    # Budget sweep [n_seeds × n_budgets]
    "dlbt_cmse":           dlbt_cmse,
    "dlbt_rho":            dlbt_rho,
    "detbt_cmse":          detbt_cmse,
    "detbt_rho":           detbt_rho,
    "slda_cmse":           slda_cmse,
    "slda_rho":            slda_rho,
    "randont_cmse":        randont_cmse,
    "randont_rho":         randont_rho,
    # All-data point [n_seeds]
    "dlbt_all_cmse":       dlbt_all_cmse,
    "dlbt_all_rho":        dlbt_all_rho,
    "detbt_all_cmse":      detbt_all_cmse,
    "detbt_all_rho":       detbt_all_rho,
    "slda_all_cmse":       slda_all_cmse,
    "slda_all_rho":        slda_all_rho,
    "randont_all_cmse":    randont_all_cmse,
    "randont_all_rho":     randont_all_rho,
}

out_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSaved → {out_path}")
