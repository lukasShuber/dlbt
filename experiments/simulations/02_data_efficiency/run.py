"""
Simulation 02 — data efficiency.

For each budget b, exactly b behavioral trials are drawn uniformly at random
(with replacement) from the pool of training (image, task) pairs.  At low
budgets most pairs are unobserved; at high budgets each pair accumulates many
trials.  DLBT and SLDA are trained on the resulting sparse dataset and
evaluated on fixed test sets (N_FULL_PER_PAIR trials each) for clean targets.

Run from repo root:
    python experiments/simulations/02_data_efficiency/run.py
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

run_tag = "frozen" if cfg.FREEZE_ENCODER else "attnpool"

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
# Ground-truth Dirichlet observer
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


def gt_p_right(uid: str, task, n_mc: int = 2000, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(uid)
    beliefs = rng.dirichlet(alpha, size=n_mc)
    return float((beliefs @ task.delta_u > 0).mean())


# GT probabilities are deterministic given uid+task — cache globally.
_rng_gt   = np.random.default_rng(0)
_gt_cache: dict = {}


def get_true_p(uid: str, task_name: str) -> float:
    key = (uid, task_name)
    if key not in _gt_cache:
        _gt_cache[key] = gt_p_right(uid, TASKS[task_name], n_mc=2000, rng=_rng_gt)
    return _gt_cache[key]


# ---------------------------------------------------------------------------
# Split helper
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def make_test_dataset(task_names: list, allowed_uids: set, rng) -> BehavioralDataset:
    """Fixed-size test dataset: N_FULL_PER_PAIR trials per (image, task) pair."""
    avail   = [r for r in refs if r.uid in allowed_uids]
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, avail, rng=rng):
            alpha   = gt_alpha(ref.uid)
            beliefs = rng.dirichlet(alpha, size=cfg.N_FULL_PER_PAIR)
            count_1 = int((beliefs @ task.delta_u > 0).sum())
            records.append(Observation(
                uid=ref.uid, task_name=task_name,
                count_0=cfg.N_FULL_PER_PAIR - count_1, count_1=count_1,
            ))
    return BehavioralDataset.from_records(records)


def make_pair_pool(task_names: list, allowed_uids: set, rng) -> list:
    """
    Return the ordered list of (ref, task_name) pairs that would be sampled
    from.  Uses balanced_refs to match the standard data generation procedure.
    Rng is consumed here; call once per seed and reuse the list across budgets.
    """
    avail = [r for r in refs if r.uid in allowed_uids]
    pairs = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, avail, rng=rng):
            pairs.append((ref, task_name))
    return pairs


def make_budget_dataset(pairs: list, total_budget: int, rng) -> BehavioralDataset:
    """
    Sample `total_budget` trials uniformly at random (with replacement) across
    all (image, task) pairs.  Pairs that receive 0 trials are excluded.
    """
    n_pairs = len(pairs)
    # Draw which pair each trial belongs to
    counts = np.bincount(
        rng.integers(0, n_pairs, size=total_budget),
        minlength=n_pairs,
    )

    records = []
    for pair_idx, n_trials in enumerate(counts):
        if n_trials == 0:
            continue
        ref, task_name = pairs[pair_idx]
        task    = TASKS[task_name]
        alpha   = gt_alpha(ref.uid)
        beliefs = rng.dirichlet(alpha, size=int(n_trials))
        count_1 = int((beliefs @ task.delta_u > 0).sum())
        records.append(Observation(
            uid=ref.uid, task_name=task_name,
            count_0=int(n_trials) - count_1, count_1=count_1,
        ))
    return BehavioralDataset.from_records(records)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def agg_metrics(pred_dict: dict, task_names: list,
                n_mc: int | None = None) -> tuple[float, float]:
    """Aggregate cMSE and ρ; use true GT probabilities as targets."""
    preds_list, trues_list = [], []
    for t in task_names:
        if t not in pred_dict:
            continue
        d = pred_dict[t]
        preds_list.append(d["pred"])
        trues_list.append(np.array([get_true_p(uid, t) for uid in d["uids"]]))

    if not preds_list:
        return float("nan"), float("nan")

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)
    raw   = float(np.mean((preds - trues) ** 2))
    cmse  = raw - float(np.mean(preds * (1 - preds))) / (n_mc - 1) if n_mc else raw
    rho, _ = spearmanr(preds, trues)
    return float(cmse), float(rho)


def collect_dlbt(agent, ds: BehavioralDataset, task_names: list) -> dict:
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


def collect_slda(slda_temps, W_slda,
                 ds: BehavioralDataset, task_names: list,
                 clip_feat_fn) -> dict:
    out = {}
    for task_name in task_names:
        if task_name not in slda_temps:
            continue
        group = ds.df[ds.df["task_name"] == task_name]
        if len(group) == 0:
            continue
        uids = group["uid"].tolist()
        X    = clip_feat_fn(uids)
        if W_slda is not None:
            delta_u = TASKS[task_name].delta_u.astype(np.float64)
            logits  = (X @ W_slda) @ delta_u
        else:
            X_scaled = slda_scalers[task_name].transform(X)
            p_pred   = np.clip(slda_models[task_name].predict(X_scaled), 1e-6, 1 - 1e-6)
            logits   = np.log(p_pred / (1 - p_pred))
        tau  = slda_temps[task_name]
        pred = 1.0 / (1.0 + np.exp(-logits / tau))
        out[task_name] = dict(pred=pred, uids=uids)
    return out


# ---------------------------------------------------------------------------
# Result arrays  [n_seeds × n_budgets]
# ---------------------------------------------------------------------------
S, B = cfg.N_SEEDS, len(cfg.BUDGETS)
_nan = lambda: np.full((S, B), np.nan)

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

    rng_split              = np.random.default_rng(seed)
    train_uids, test_uids  = make_split(rng_split)
    print(f"  split: {len(train_uids)} train / {len(test_uids)} test images")

    # Test datasets — N_FULL_PER_PAIR trials, fixed for this seed
    rng_test  = np.random.default_rng(seed + 99_999)
    test_train  = make_test_dataset(cfg.TRAIN_TASKS, train_uids, rng_test)
    test_stim   = make_test_dataset(cfg.TRAIN_TASKS, test_uids,  rng_test)
    test_task   = make_test_dataset(cfg.VAL_TASKS,   train_uids, rng_test)
    test_joint  = make_test_dataset(cfg.VAL_TASKS,   test_uids,  rng_test)

    # Pair pool — generated once per seed, reused across all budgets
    rng_pairs  = np.random.default_rng(seed + 77_777)
    train_pairs = make_pair_pool(cfg.TRAIN_TASKS, train_uids, rng_pairs)
    print(f"  pair pool: {len(train_pairs)} training (image, task) pairs")

    for b_idx, budget in enumerate(cfg.BUDGETS):
        print(f"\n  Budget {budget:>9,} total trials  ({b_idx + 1}/{len(cfg.BUDGETS)})")

        # Training dataset — budget trials, unique rng per (seed, budget)
        rng_train = np.random.default_rng(seed * 10_000 + b_idx)
        train_ds  = make_budget_dataset(train_pairs, budget, rng_train)

        n_obs   = len(train_ds.df)
        n_tasks = train_ds.df["task_name"].nunique()
        print(f"    {n_obs} observations across {n_tasks} tasks")

        # ---------------------------------------------------------------
        # Train DLBT
        # ---------------------------------------------------------------
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = DlbtAgent(freeze_encoder=cfg.FREEZE_ENCODER, n_mc_samples=cfg.N_MC,
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
            agent, train_ds, test_stim, refs_dict,
            n_epochs=cfg.N_EPOCHS_PHASE1, lr=cfg.LR,
            patience=cfg.PATIENCE_PHASE1,
            extra_val_datasets={"task": test_task, "joint": test_joint},
        )
        print(f"    DLBT phase1  best epoch {phase1.best_epoch:4d}  "
              f"stim_mse={phase1.best_val_mse:.4f}")

        # Phase 2 — attnpool fine-tuning (only when FREEZE_ENCODER=False)
        if not cfg.FREEZE_ENCODER:
            for p in agent.mapper.parameters():
                p.requires_grad_(False)
            for p in agent.encoder.attnpool.parameters():
                p.requires_grad_(True)
            agent.freeze_encoder = False
            agent._cache.clear()

            optimizer2 = torch.optim.Adam(
                agent.encoder.attnpool.parameters(), lr=cfg.LR_ATTNPOOL
            )
            phase2 = train_dlbt(
                agent, train_ds, test_stim, refs_dict,
                n_epochs=cfg.N_EPOCHS_PHASE2, patience=cfg.PATIENCE_PHASE2,
                optimizer=optimizer2,
                extra_val_datasets={"task": test_task, "joint": test_joint},
            )
            print(f"    DLBT phase2  best epoch {phase2.best_epoch:4d}  "
                  f"stim_mse={phase2.best_val_mse:.4f}")

            # Repopulate _cache with fine-tuned attnpool features for DLBT predictions
            agent.eval()
            all_refs_list = list(refs_dict.values())
            with torch.no_grad():
                for i in range(0, len(all_refs_list), 16):
                    batch   = all_refs_list[i : i + 16]
                    spatial = torch.stack(
                        [agent._backbone_cache[r.uid] for r in batch]
                    ).to(agent.device)
                    feats = agent.encoder.attnpool(spatial).float()
                    for ref, feat in zip(batch, feats):
                        agent._cache[ref.uid] = feat.cpu()

        # ---------------------------------------------------------------
        # Fit SLDA
        # ---------------------------------------------------------------
        def clip_features(uids: list) -> np.ndarray:
            return np.array([frozen_clip[uid].cpu().numpy() for uid in uids])

        if cfg.SLDA_GT:
            _all_refs = list(refs_dict.values())
            _X_all    = np.stack([frozen_clip[r.uid].cpu().numpy() for r in _all_refs])
            _Y_oh     = np.zeros((len(_all_refs), K), dtype=np.float32)
            for _i, _r in enumerate(_all_refs):
                _Y_oh[_i, _r.latent_state] = 1.0
            W_slda, _, _, _ = np.linalg.lstsq(_X_all, _Y_oh, rcond=None)

            slda_temps = {}
            for task_name in cfg.TRAIN_TASKS:
                group = train_ds.df[train_ds.df["task_name"] == task_name]
                if len(group) == 0:
                    slda_temps[task_name] = 1.0
                    continue
                uids    = group["uid"].tolist()
                X       = np.stack([frozen_clip[uid].cpu().numpy() for uid in uids])
                p_right = (group["count_1"] / (group["count_0"] + group["count_1"])).values
                delta_u = TASKS[task_name].delta_u.astype(np.float64)
                logits  = (X @ W_slda) @ delta_u

                def _nll_tau(log_tau, logits=logits, targets=p_right):
                    p = 1.0 / (1.0 + np.exp(-logits / np.exp(log_tau)))
                    p = np.clip(p, 1e-7, 1 - 1e-7)
                    return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

                opt = minimize_scalar(_nll_tau, bounds=(-3.0, 3.0), method="bounded")
                slda_temps[task_name] = float(np.exp(opt.x))

        else:
            slda_scalers, slda_models, slda_temps = {}, {}, {}
            W_slda = None
            for task_name in cfg.TRAIN_TASKS:
                group = train_ds.df[train_ds.df["task_name"] == task_name]
                if len(group) < 3:
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

        print(f"    SLDA  fitted {len(slda_temps)}/{len(cfg.TRAIN_TASKS)} tasks (GT={cfg.SLDA_GT})")

        # ---------------------------------------------------------------
        # Collect predictions
        # ---------------------------------------------------------------
        agent.eval()
        dlbt_preds = {
            "train": collect_dlbt(agent, test_train, cfg.TRAIN_TASKS),
            "stim":  collect_dlbt(agent, test_stim,  cfg.TRAIN_TASKS),
            "task":  collect_dlbt(agent, test_task,  cfg.VAL_TASKS),
            "joint": collect_dlbt(agent, test_joint, cfg.VAL_TASKS),
        }
        slda_preds = {
            "train": collect_slda(slda_temps, W_slda,
                                   test_train, cfg.TRAIN_TASKS, clip_features),
            "stim":  collect_slda(slda_temps, W_slda,
                                   test_stim,  cfg.TRAIN_TASKS, clip_features),
        }

        # ---------------------------------------------------------------
        # Store metrics
        # ---------------------------------------------------------------
        for cond, task_names in [("train", cfg.TRAIN_TASKS), ("stim",  cfg.TRAIN_TASKS),
                                  ("task",  cfg.VAL_TASKS),   ("joint", cfg.VAL_TASKS)]:
            cmse, rho = agg_metrics(dlbt_preds[cond], task_names, cfg.N_MC)
            res_dlbt[cond]["cmse"][s_idx, b_idx] = cmse
            res_dlbt[cond]["rho"][s_idx, b_idx]  = rho
            print(f"    DLBT  {cond:6s}  cMSE={cmse:.4f}  ρ={rho:.3f}")

        for cond, task_names in [("train", cfg.TRAIN_TASKS), ("stim", cfg.TRAIN_TASKS)]:
            cmse, rho = agg_metrics(slda_preds[cond], task_names, None)
            res_slda[cond]["cmse"][s_idx, b_idx] = cmse
            res_slda[cond]["rho"][s_idx, b_idx]  = rho
            print(f"    SLDA  {cond:6s}  cMSE={cmse:.4f}  ρ={rho:.3f}")

        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = dict(
    budgets  = cfg.BUDGETS,
    seeds    = cfg.SEEDS,
    n_full   = cfg.N_FULL_PER_PAIR,
    run_tag  = run_tag,
    dlbt     = res_dlbt,
    slda     = res_slda,
)

out_path = cfg.RESULTS_DIR / f"results_{run_tag}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results → {out_path}")
