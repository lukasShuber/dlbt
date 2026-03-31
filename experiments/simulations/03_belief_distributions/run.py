"""
Simulation 03 — belief distribution robustness.

For each seed × ground-truth distribution, synthetic behavioral data is
generated and DLBT (Dirichlet assumption) + SLDA are trained and evaluated.
Ground-truth choice probabilities are computed from the *same* distribution
that generated the training data, so we test:
  "Does DLBT recover accurate choice probabilities even when the true observer
   is not Dirichlet?"

Run from repo root:
    python experiments/simulations/03_belief_distributions/run.py
"""

import gc
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from dlbt.constants import (
    K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

import config as cfg

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# ---------------------------------------------------------------------------
# Load stimuli + continuous metadata
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(cfg.METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

cont_meta: dict = {}
with open(cfg.METADATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        z   = rec["z"]
        cont_meta[rec["id"]] = dict(
            x            = z["pos_xy"][0],
            transparency = z["transparency"],
            glossiness   = z["glossiness"],
            scale        = z["scale"],
        )

# ---------------------------------------------------------------------------
# Shared: Dirichlet alpha (used by all distributions as the common signal)
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    z = cont_meta[uid]
    p_right  = _sigmoid(cfg.BETA       * (z["x"]            - X_THRESHOLD))
    p_transp = _sigmoid(cfg.BETA       * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(cfg.BETA       * (z["glossiness"]   - GLOSS_THRESH))
    p_large  = _sigmoid(cfg.SCALE_BETA * (z["scale"]        - SCALE_THRESH))

    q = np.empty(K, dtype=np.float64)
    for k in range(K):
        k_right  = (k >> DIM_LEFT_RIGHT)  & 1
        k_transp = (k >> DIM_TRANSP)      & 1
        k_glossy = (k >> DIM_GLOSS)       & 1
        k_large  = (k >> DIM_SMALL_LARGE) & 1
        q[k] = (
            (p_right  if k_right  else 1.0 - p_right)  *
            (p_transp if k_transp else 1.0 - p_transp) *
            (p_glossy if k_glossy else 1.0 - p_glossy) *
            (p_large  if k_large  else 1.0 - p_large)
        )

    clarity = (abs(p_right  - 0.5) * 2.0 *
               abs(p_transp - 0.5) * 2.0 *
               abs(p_glossy - 0.5) * 2.0 *
               abs(p_large  - 0.5) * 2.0)
    lam = cfg.BASE_CONCENTRATION + cfg.PEAK * clarity
    return 1e-6 + lam * q


def _dim_logits(uid: str) -> np.ndarray:
    """Raw logit for each of the 4 binary dimensions (used by threshold model)."""
    z = cont_meta[uid]
    return np.array([
        cfg.BETA       * (z["x"]            - X_THRESHOLD),
        cfg.BETA       * (z["transparency"] - TRANSP_THRESH),
        cfg.BETA       * (z["glossiness"]   - GLOSS_THRESH),
        cfg.SCALE_BETA * (z["scale"]        - SCALE_THRESH),
    ])


# ---------------------------------------------------------------------------
# Ground-truth belief samplers — one per distribution
# ---------------------------------------------------------------------------
def _sample_dirichlet(uid: str, task, n_trials: int, rng) -> tuple[int, int]:
    alpha   = gt_alpha(uid)
    beliefs = rng.dirichlet(alpha, size=n_trials)
    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return n_trials - count_1, count_1


def _sample_logistic_normal(uid: str, task, n_trials: int, rng) -> tuple[int, int]:
    """
    Beliefs = softmax(mu + eps), eps ~ N(0, sigma^2 I).
    Mean direction matches the Dirichlet mean; sigma is variance-matched:
      sigma = 1 / sqrt(sum(alpha) + 1)
    so that the logistic-normal spread equals the Dirichlet spread at every
    concentration level.
    """
    alpha = gt_alpha(uid)
    lam   = float(alpha.sum())
    q     = alpha / lam
    # centre-log-ratio mean
    log_q = np.log(q + 1e-10)
    mu    = log_q - log_q.mean()
    sigma = 1.0 / np.sqrt(lam + 1.0)

    eps     = rng.normal(0.0, sigma, size=(n_trials, K))
    logits  = mu[None, :] + eps
    # numerically stable softmax
    logits -= logits.max(axis=1, keepdims=True)
    beliefs = np.exp(logits)
    beliefs /= beliefs.sum(axis=1, keepdims=True)

    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return n_trials - count_1, count_1


def _sample_lapse(uid: str, task, n_trials: int, rng,
                  lapse_rate: float = cfg.LAPSE_RATE) -> tuple[int, int]:
    """
    Each trial: with probability (1 - lapse_rate) draw from Dirichlet;
    with probability lapse_rate choose randomly (p=0.5).
    """
    alpha    = gt_alpha(uid)
    n_normal = int(rng.binomial(n_trials, 1.0 - lapse_rate))
    n_lapse  = n_trials - n_normal

    count_1 = 0
    if n_normal > 0:
        beliefs = rng.dirichlet(alpha, size=n_normal)
        count_1 += int((beliefs @ task.delta_u > 0).sum())
    if n_lapse > 0:
        count_1 += int(rng.binomial(n_lapse, 0.5))

    return n_trials - count_1, count_1


def _sample_threshold(uid: str, task, n_trials: int, rng,
                      sigma: float = cfg.THRESHOLD_SIGMA) -> tuple[int, int]:
    """
    SDT-style observer: for each trial add Gaussian noise to each dimension's
    logit, classify each dimension as 0/1 by sign, then construct the discrete
    belief state and read off the task utility.
    """
    logits = _dim_logits(uid)                                    # [4]
    noisy  = logits[None, :] + rng.normal(0.0, sigma,
                                          size=(n_trials, 4))   # [N, 4]
    c      = (noisy > 0).astype(int)                             # [N, 4]
    states = (  (c[:, 0] << DIM_LEFT_RIGHT)
              | (c[:, 1] << DIM_TRANSP)
              | (c[:, 2] << DIM_GLOSS)
              | (c[:, 3] << DIM_SMALL_LARGE) )                  # [N]
    count_1 = int((task.delta_u[states] > 0).sum())
    return n_trials - count_1, count_1


SAMPLERS = {
    "dirichlet":       _sample_dirichlet,
    "logistic_normal": _sample_logistic_normal,
    "lapse":           _sample_lapse,
    "threshold":       _sample_threshold,
}

# ---------------------------------------------------------------------------
# True choice probability — Monte Carlo under each distribution
# ---------------------------------------------------------------------------
_gt_cache: dict = {}   # keyed by (dist_name, uid, task_name)
_rng_gt = np.random.default_rng(0)
N_MC_GT = 2000


def get_true_p(dist_name: str, uid: str, task_name: str) -> float:
    key = (dist_name, uid, task_name)
    if key in _gt_cache:
        return _gt_cache[key]

    task  = TASKS[task_name]
    alpha = gt_alpha(uid)

    if dist_name == "dirichlet":
        beliefs = _rng_gt.dirichlet(alpha, size=N_MC_GT)
        p = float((beliefs @ task.delta_u > 0).mean())

    elif dist_name == "logistic_normal":
        lam   = float(alpha.sum())
        q     = alpha / lam
        log_q = np.log(q + 1e-10)
        mu    = log_q - log_q.mean()
        sigma = 1.0 / np.sqrt(lam + 1.0)
        eps   = _rng_gt.normal(0.0, sigma, size=(N_MC_GT, K))
        logits = mu[None, :] + eps
        logits -= logits.max(axis=1, keepdims=True)
        beliefs = np.exp(logits)
        beliefs /= beliefs.sum(axis=1, keepdims=True)
        p = float((beliefs @ task.delta_u > 0).mean())

    elif dist_name == "lapse":
        beliefs = _rng_gt.dirichlet(alpha, size=N_MC_GT)
        p_dir   = float((beliefs @ task.delta_u > 0).mean())
        p = (1.0 - cfg.LAPSE_RATE) * p_dir + cfg.LAPSE_RATE * 0.5

    elif dist_name == "threshold":
        logits = _dim_logits(uid)
        noisy  = logits[None, :] + _rng_gt.normal(0.0, cfg.THRESHOLD_SIGMA,
                                                   size=(N_MC_GT, 4))
        c      = (noisy > 0).astype(int)
        states = (  (c[:, 0] << DIM_LEFT_RIGHT)
                  | (c[:, 1] << DIM_TRANSP)
                  | (c[:, 2] << DIM_GLOSS)
                  | (c[:, 3] << DIM_SMALL_LARGE) )
        p = float((task.delta_u[states] > 0).mean())

    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

    _gt_cache[key] = p
    return p


# ---------------------------------------------------------------------------
# Dataset / split helpers
# ---------------------------------------------------------------------------
def make_split(rng_split) -> tuple[set, set]:
    state_to_uids: dict = defaultdict(list)
    for uid in sorted(refs_dict.keys()):
        state_to_uids[refs_dict[uid].latent_state].append(uid)

    train_uids: set = set()
    test_uids:  set = set()
    for state_uids in state_to_uids.values():
        arr    = np.array(state_uids)
        rng_split.shuffle(arr)
        n_test = max(1, round(len(arr) * cfg.IMG_TEST_FRAC))
        test_uids.update(arr[:n_test].tolist())
        train_uids.update(arr[n_test:].tolist())
    return train_uids, test_uids


def make_dataset(dist_name: str, task_names: list,
                 allowed_uids: set, rng) -> BehavioralDataset:
    sampler = SAMPLERS[dist_name]
    avail   = [r for r in refs if r.uid in allowed_uids]
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, avail, rng=rng):
            c0, c1 = sampler(ref.uid, task, cfg.N_TRIALS, rng)
            records.append(Observation(
                uid=ref.uid, task_name=task_name, count_0=c0, count_1=c1,
            ))
    return BehavioralDataset.from_records(records)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def agg_metrics(pred_dict: dict, dist_name: str,
                task_names: list, n_mc: int | None = None) -> tuple[float, float]:
    preds = np.concatenate([pred_dict[t]["pred"] for t in task_names if t in pred_dict])
    trues = np.concatenate([
        np.array([get_true_p(dist_name, uid, t)
                  for uid in pred_dict[t]["uids"]])
        for t in task_names if t in pred_dict
    ])
    raw  = float(np.mean((preds - trues) ** 2))
    cmse = raw - float(np.mean(preds * (1 - preds))) / (n_mc - 1) if n_mc else raw
    rho, _ = spearmanr(preds, trues)
    return float(cmse), float(rho)


def collect_dlbt(agent, ds: BehavioralDataset,
                 dist_name: str, task_names: list) -> dict:
    out = {}
    agent.eval()
    for task_name in task_names:
        group = ds.df[ds.df["task_name"] == task_name]
        if len(group) == 0:
            continue
        task       = TASKS[task_name]
        uids       = group["uid"].tolist()
        batch_refs = [refs_dict[uid] for uid in uids]
        with torch.no_grad():
            pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
        out[task_name] = dict(pred=pred, uids=uids)
    return out


def collect_slda(slda_scalers, slda_models, slda_temps,
                 ds: BehavioralDataset, task_names: list,
                 clip_feat_fn) -> dict:
    out = {}
    for task_name in task_names:
        if task_name not in slda_models:
            continue
        group = ds.df[ds.df["task_name"] == task_name]
        if len(group) == 0:
            continue
        uids     = group["uid"].tolist()
        X        = clip_feat_fn(uids)
        X_scaled = slda_scalers[task_name].transform(X)
        p_pred   = np.clip(slda_models[task_name].predict(X_scaled), 1e-6, 1 - 1e-6)
        logits   = np.log(p_pred / (1 - p_pred))
        tau      = slda_temps[task_name]
        pred     = 1.0 / (1.0 + np.exp(-logits / tau))
        out[task_name] = dict(pred=pred, uids=uids)
    return out


# ---------------------------------------------------------------------------
# Result arrays  [n_seeds × n_distributions]
# ---------------------------------------------------------------------------
S, D = cfg.N_SEEDS, len(cfg.DISTRIBUTIONS)
_nan = lambda: np.full((S, D), np.nan)

res_dlbt = {cond: {"cmse": _nan(), "rho": _nan()}
            for cond in ["train", "stim", "task", "joint"]}
res_slda = {cond: {"cmse": _nan(), "rho": _nan()}
            for cond in ["train", "stim"]}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for s_idx, seed in enumerate(cfg.SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {s_idx + 1}/{cfg.N_SEEDS}  (seed={seed})")
    print(f"{'='*60}")

    rng_split          = np.random.default_rng(seed)
    train_uids, test_uids = make_split(rng_split)
    print(f"  split: {len(train_uids)} train / {len(test_uids)} test images")

    for d_idx, dist_name in enumerate(cfg.DISTRIBUTIONS):
        print(f"\n  Distribution: {cfg.DIST_LABELS[dist_name]}")

        # Unique rng per (seed, distribution) for data generation
        rng_data = np.random.default_rng(seed * 100 + d_idx)

        train_ds     = make_dataset(dist_name, cfg.TRAIN_TASKS, train_uids, rng_data)
        stim_gen_ds  = make_dataset(dist_name, cfg.TRAIN_TASKS, test_uids,  rng_data)
        task_gen_ds  = make_dataset(dist_name, cfg.VAL_TASKS,   train_uids, rng_data)
        joint_gen_ds = make_dataset(dist_name, cfg.VAL_TASKS,   test_uids,  rng_data)

        # -------------------------------------------------------------------
        # Train DLBT  (always with Dirichlet assumption internally)
        # -------------------------------------------------------------------
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = DlbtAgent(freeze_encoder=True, n_mc_samples=cfg.N_MC,
                          device=device, mapper_hidden=cfg.MAPPER_HIDDEN)

        cache_path = Path(cfg.CACHE_PATH)
        if cache_path.exists():
            agent.load_cache(str(cache_path))
        else:
            print("    Precomputing CLIP features...")
            agent.precompute_features(list(refs_dict.values()))
            agent.save_cache(str(cache_path))

        frozen_clip: dict = {uid: feat.clone() for uid, feat in agent._cache.items()}

        phase1 = train_dlbt(
            agent, train_ds, stim_gen_ds, refs_dict,
            n_epochs=cfg.N_EPOCHS_PHASE1, lr=cfg.LR,
            patience=cfg.PATIENCE_PHASE1,
            extra_val_datasets={"task": task_gen_ds, "joint": joint_gen_ds},
        )
        print(f"    DLBT  best epoch {phase1.best_epoch:4d}  "
              f"stim_mse={phase1.best_val_mse:.4f}")

        # -------------------------------------------------------------------
        # Fit SLDA  (always on frozen CLIP features)
        # -------------------------------------------------------------------
        def clip_features(uids: list) -> np.ndarray:
            return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])

        slda_scalers, slda_models, slda_temps = {}, {}, {}

        for task_name in cfg.TRAIN_TASKS:
            group = train_ds.df[train_ds.df["task_name"] == task_name]
            if len(group) == 0:
                continue
            uids    = group["uid"].tolist()
            X       = clip_features(uids)
            p_right = (group["count_1"] / (group["count_0"] + group["count_1"])).values

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model    = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
            model.fit(X_scaled, p_right)

            p_pred = np.clip(model.predict(X_scaled), 1e-6, 1 - 1e-6)
            logits = np.log(p_pred / (1 - p_pred))

            def _nll_tau(log_tau, logits=logits, targets=p_right):
                p = 1.0 / (1.0 + np.exp(-logits / np.exp(log_tau)))
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

            opt = minimize_scalar(_nll_tau, bounds=(-3.0, 3.0), method="bounded")
            slda_scalers[task_name] = scaler
            slda_models[task_name]  = model
            slda_temps[task_name]   = float(np.exp(opt.x))

        # -------------------------------------------------------------------
        # Collect predictions
        # -------------------------------------------------------------------
        agent.eval()
        dlbt_preds = {
            "train": collect_dlbt(agent, train_ds,     dist_name, cfg.TRAIN_TASKS),
            "stim":  collect_dlbt(agent, stim_gen_ds,  dist_name, cfg.TRAIN_TASKS),
            "task":  collect_dlbt(agent, task_gen_ds,  dist_name, cfg.VAL_TASKS),
            "joint": collect_dlbt(agent, joint_gen_ds, dist_name, cfg.VAL_TASKS),
        }
        slda_preds = {
            "train": collect_slda(slda_scalers, slda_models, slda_temps,
                                   train_ds,    cfg.TRAIN_TASKS, clip_features),
            "stim":  collect_slda(slda_scalers, slda_models, slda_temps,
                                   stim_gen_ds, cfg.TRAIN_TASKS, clip_features),
        }

        # -------------------------------------------------------------------
        # Store metrics
        # -------------------------------------------------------------------
        for cond, task_names in [("train", cfg.TRAIN_TASKS), ("stim",  cfg.TRAIN_TASKS),
                                  ("task",  cfg.VAL_TASKS),   ("joint", cfg.VAL_TASKS)]:
            cmse, rho = agg_metrics(dlbt_preds[cond], dist_name, task_names, cfg.N_MC)
            res_dlbt[cond]["cmse"][s_idx, d_idx] = cmse
            res_dlbt[cond]["rho"][s_idx, d_idx]  = rho
            print(f"    DLBT  {cond:6s}  cMSE={cmse:.4f}  ρ={rho:.3f}")

        for cond, task_names in [("train", cfg.TRAIN_TASKS), ("stim", cfg.TRAIN_TASKS)]:
            cmse, rho = agg_metrics(slda_preds[cond], dist_name, task_names, None)
            res_slda[cond]["cmse"][s_idx, d_idx] = cmse
            res_slda[cond]["rho"][s_idx, d_idx]  = rho
            print(f"    SLDA  {cond:6s}  cMSE={cmse:.4f}  ρ={rho:.3f}")

        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = dict(
    distributions = cfg.DISTRIBUTIONS,
    dist_labels   = cfg.DIST_LABELS,
    dist_colors   = cfg.DIST_COLORS,
    seeds         = cfg.SEEDS,
    dlbt          = res_dlbt,
    slda          = res_slda,
)

out_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results → {out_path}")
