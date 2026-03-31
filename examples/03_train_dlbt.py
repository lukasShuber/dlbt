"""
Train DlbtAgent on a synthetic behavioral dataset with a 2×2 generalization design.

Four evaluation regions arise from a joint split on stimuli (X_test) and tasks (T_test):

    ┌──────────────────┬────────────────────┬──────────────────────┐
    │                  │ Seen tasks         │ Unseen tasks         │
    ├──────────────────┼────────────────────┼──────────────────────┤
    │ Seen images      │ Training region    │ Task generalization  │
    │ Unseen images    │ Stim generalization│ Joint generalization │
    └──────────────────┴────────────────────┴──────────────────────┘

Four latent dimensions (K=16): left/right (x), transparent, glossy,
small/large (scale).

Val tasks hold out all lr × sl conjunctions (left_right × small_large):
the model trains on each dimension separately but never sees their combination.

DLBT is evaluated on all four regions.
SLDA (per-task RidgeCV) is evaluated on training + stimulus-generalization only
(it has no weights for unseen tasks).

Run from repo root:
    python examples/03_train_dlbt.py
"""

import gc
import math
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

from dlbt.constants import (
    K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE,
    X_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, SCALE_THRESH,
)
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt
from dlbt.training.metrics import corrected_mse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METADATA   = "stimuli/imgs/metadata.jsonl"
CACHE_PATH = "stimuli/imgs/clip_rn50_features_v2.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Device: {DEVICE} (no GPU — training will be slow)")

SEED               = 42
N_TRIALS           = 100    # SEU decisions per (image, task)
PEAK               = 15.0   # peak concentration added to matching latent states
BASE_CONCENTRATION = 1.0    # base concentration on all latent states
BETA               = 5.0    # sigmoid sharpness for lr, tr, gl dimensions
SCALE_BETA         = 10.0   # sharpness for scale sigmoid (new stimuli [0.2, 0.8], thresh 0.5)
N_EPOCHS_PHASE1    = 500    # mapper warmup (encoder always frozen)
PATIENCE_PHASE1    = 50     # early-stopping patience for phase 1
N_EPOCHS_PHASE2    = 3000   # attnpool fine-tuning (only if FREEZE_ENCODER=False)
PATIENCE_PHASE2    = 100    # early-stopping patience for phase 2
LR                 = 1e-2   # mapper LR
LR_ATTNPOOL        = 1e-4   # attnpool LR (phase 2 only)
N_MC               = 200    # MC samples for choice_probs during training
FREEZE_ENCODER     = False   # True → frozen only; False → phase 1 then attnpool fine-tune
MAPPER_HIDDEN      = None   # None → linear mapper

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"  # used in all output paths

IMG_TEST_FRAC      = 0.20   # fraction of images held out for stimulus/joint gen

# Holdout (Option B): lr × sl conjunctions held out as T_test.
# Training sees left_right and small_large independently but never combined.
TRAIN_TASKS = [
    # simple
    "left_right", "transparent", "glossy", "large",
    # simple-flipped
    "left", "opaque", "matte", "small",
    # 2-way AND: lr × material
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy", "left_and_glossy",
    # 2-way AND: material × material
    "transparent_and_glossy",
    # 2-way AND: sl × material (no lr × sl)
    "large_and_transparent", "large_and_glossy",
    # 3-way AND
    "right_and_transparent_and_glossy", "left_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
]
VAL_TASKS = [
    # lr × sl conjunctions — never seen during training
    "right_and_large",
    "left_and_large",
    # 3-way extensions
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

# Regime colors (used consistently across all plots)
C_TRAIN    = "#d95f02"   # orange  — training region
C_STIM     = "#1f78b4"   # blue    — stimulus generalization
C_TASK     = "#7570b3"   # purple  — task generalization
C_JOINT    = "#33a02c"   # green   — joint generalization

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load stimuli
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Continuous metadata (used for soft belief generation)
# ---------------------------------------------------------------------------
def _load_continuous_metadata(metadata_path: str) -> dict:
    import json
    result = {}
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            z   = rec["z"]
            result[rec["id"]] = dict(
                x            = z["pos_xy"][0],
                transparency = z["transparency"],
                glossiness   = z["glossiness"],
                scale        = z["scale"],
            )
    return result

cont_meta = _load_continuous_metadata(METADATA)

# ---------------------------------------------------------------------------
# Ground truth Dirichlet — soft beliefs
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    """Structured Dirichlet: soft mean q_k + clarity-scaled concentration λ.

    Four continuous dimensions (front/back excluded — confounds with scale).
    SCALE_BETA is calibrated for the render range [0.2, 0.8] with threshold 0.5.
    """
    z = cont_meta[uid]
    p_right  = _sigmoid(BETA       * (z["x"]            - X_THRESHOLD))
    p_transp = _sigmoid(BETA       * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(BETA       * (z["glossiness"]   - GLOSS_THRESH))
    p_large  = _sigmoid(SCALE_BETA * (z["scale"]        - SCALE_THRESH))

    q = np.empty(K, dtype=np.float64)
    for k in range(K):
        k_right  = (k >> DIM_LEFT_RIGHT)  & 1
        k_transp = (k >> DIM_TRANSP)      & 1
        k_glossy = (k >> DIM_GLOSS)       & 1
        k_large  = (k >> DIM_SMALL_LARGE) & 1
        q[k] = (
            (p_right  if k_right  else (1.0 - p_right))  *
            (p_transp if k_transp else (1.0 - p_transp)) *
            (p_glossy if k_glossy else (1.0 - p_glossy)) *
            (p_large  if k_large  else (1.0 - p_large))
        )

    clarity = (abs(p_right  - 0.5) * 2.0 *
               abs(p_transp - 0.5) * 2.0 *
               abs(p_glossy - 0.5) * 2.0 *
               abs(p_large  - 0.5) * 2.0)
    lam = BASE_CONCENTRATION + PEAK * clarity
    return 1e-6 + lam * q


def gt_p_right(uid: str, task, n_mc: int = 2000, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    alpha   = gt_alpha(uid)
    beliefs = rng.dirichlet(alpha, size=n_mc)
    logits  = beliefs @ task.delta_u
    return float((logits > 0).mean())


def sample_behavior(ref, task, n_trials: int, rng) -> tuple:
    alpha   = gt_alpha(ref.uid)
    beliefs = rng.dirichlet(alpha, size=n_trials)
    logits  = beliefs @ task.delta_u
    count_1 = int((logits > 0).sum())
    return n_trials - count_1, count_1


# ---------------------------------------------------------------------------
# Image split — stratified by latent state  (X_train / X_test)
#
# With K=32 states and ~1000 images, a uniform random split can leave some
# latent states with 0–1 training examples.  Stratified sampling ensures
# each state contributes ~IMG_TEST_FRAC images to the test set.
# ---------------------------------------------------------------------------
from collections import defaultdict

all_uids  = sorted(refs_dict.keys())
rng_split = np.random.default_rng(SEED)

# Group UIDs by latent state
state_to_uids: dict = defaultdict(list)
for uid in all_uids:
    state_to_uids[refs_dict[uid].latent_state].append(uid)

train_uids: set = set()
test_uids:  set = set()
for state_uids in state_to_uids.values():
    arr    = np.array(state_uids)
    rng_split.shuffle(arr)
    n_test = max(1, round(len(arr) * IMG_TEST_FRAC))
    test_uids.update(arr[:n_test].tolist())
    train_uids.update(arr[n_test:].tolist())

print(f"Image split (stratified): {len(train_uids)} train / {len(test_uids)} test "
      f"({len(test_uids)/len(all_uids)*100:.1f}% held out)")

# ---------------------------------------------------------------------------
# Plot 0 — latent-state balance: random vs stratified split
# ---------------------------------------------------------------------------
random_rng   = np.random.default_rng(SEED)
n_test_rand  = max(1, int(len(all_uids) * IMG_TEST_FRAC))
rand_test    = set(random_rng.choice(all_uids, size=n_test_rand, replace=False).tolist())
rand_train   = set(all_uids) - rand_test

states_sorted = sorted(state_to_uids.keys())
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharey=False,
                         gridspec_kw={"wspace": 0.35})

for ax, (train_s, test_s), title in [
    (axes[0], (rand_train, rand_test),   "Random split"),
    (axes[1], (train_uids, test_uids),   "Stratified split"),
]:
    tr_counts = [sum(1 for u in state_to_uids[s] if u in train_s) for s in states_sorted]
    te_counts = [sum(1 for u in state_to_uids[s] if u in test_s)  for s in states_sorted]
    x = np.arange(len(states_sorted))
    ax.bar(x, tr_counts, label="train", color=C_TRAIN, alpha=0.8)
    ax.bar(x, te_counts, bottom=tr_counts, label="test",  color=C_STIM,  alpha=0.8)
    ax.set(xlabel="Latent state (0–31)", ylabel="# images",
           title=f"{title}  (train={len(train_s)}, test={len(test_s)})")
    ax.axhline(np.mean([c + t for c, t in zip(tr_counts, te_counts)]),
               ls=":", color="gray", lw=0.8, label="mean/state")
    ax.legend(fontsize=8)
    # annotate min train count
    min_tr = min(tr_counts)
    ax.text(0.98, 0.97, f"min train/state = {min_tr}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="red" if min_tr == 0 else "black")

sns.despine(trim=True)
plt.savefig("examples/plots/03_latent_state_balance.png", dpi=150, bbox_inches="tight")
print("Saved: examples/plots/03_latent_state_balance.png")
plt.close()

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def make_dataset(task_names: list, allowed_uids: set) -> BehavioralDataset:
    """Generate behavioral data for task_names using only images in allowed_uids."""
    avail   = [r for r in refs if r.uid in allowed_uids]
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, avail, rng=rng):
            c0, c1 = sample_behavior(ref, task, N_TRIALS, rng)
            records.append(Observation(
                uid=ref.uid, task_name=task_name, count_0=c0, count_1=c1,
            ))
    return BehavioralDataset.from_records(records)


# Four evaluation regions
train_ds     = make_dataset(TRAIN_TASKS, train_uids)   # seen images × seen tasks
stim_gen_ds  = make_dataset(TRAIN_TASKS, test_uids)    # unseen images × seen tasks
task_gen_ds  = make_dataset(VAL_TASKS,   train_uids)   # seen images × unseen tasks
joint_gen_ds = make_dataset(VAL_TASKS,   test_uids)    # unseen images × unseen tasks

for name, ds in [("train", train_ds), ("stim_gen", stim_gen_ds),
                 ("task_gen", task_gen_ds), ("joint_gen", joint_gen_ds)]:
    print(f"  {name:12s}: {ds}  noise_floor={ds.noise_floor():.4f}")

# ---------------------------------------------------------------------------
# Train — DLBT  (two-phase when FREEZE_ENCODER=False)
#
# Phase 1 (always): train mapper with frozen encoder until convergence.
#   Ensures the mapper has a meaningful signal before any gradients flow
#   into the encoder.
# Phase 2 (FREEZE_ENCODER=False only): unfreeze attnpool and jointly
#   fine-tune at a much lower LR.
# ---------------------------------------------------------------------------
model_label = "DLBT (frozen)" if FREEZE_ENCODER else "DLBT (attnpool)"

# Always construct the agent in attnpool mode so phase 2 is possible,
# but start with the encoder frozen for phase 1.
agent = DlbtAgent(freeze_encoder=True, n_mc_samples=N_MC, device=DEVICE,
                  mapper_hidden=MAPPER_HIDDEN)

if Path(CACHE_PATH).exists():
    print(f"Loading cached CLIP features from {CACHE_PATH}")
    agent.load_cache(CACHE_PATH)
else:
    print(f"Precomputing CLIP features → {CACHE_PATH}")
    agent.precompute_features(list(refs_dict.values()))
    agent.save_cache(CACHE_PATH)
    print("Saved.")

# ---- Phase 1: mapper warmup (frozen encoder) ------------------------------
print(f"\nPhase 1 — mapper warmup (frozen encoder)...")
phase1 = train_dlbt(
    agent, train_ds, stim_gen_ds, refs_dict,
    n_epochs=N_EPOCHS_PHASE1, lr=LR, patience=PATIENCE_PHASE1,
    extra_val_datasets={"task_gen": task_gen_ds, "joint_gen": joint_gen_ds},
)
print(f"  best epoch: {phase1.best_epoch}  stim_gen_mse: {phase1.best_val_mse:.4f}")

# ---- Phase 2: attnpool fine-tuning (only when requested) ------------------
phase2 = None
if not FREEZE_ENCODER:
    print(f"\nPhase 2 — attnpool fine-tuning...")
    # Free phase-1 optimizer state and any other lingering GPU allocations
    # before switching to the more memory-intensive attnpool training.
    gc.collect()
    torch.cuda.empty_cache()
    # Unfreeze attnpool only; mapper stays frozen (phase 1 brought it to
    # convergence — letting it move in phase 2 causes oscillation).
    # Clear full-feature cache so train_dlbt switches to backbone-feature
    # caching (pre-attnpool spatial maps).
    for p in agent.mapper.parameters():
        p.requires_grad_(False)
    for p in agent.encoder.attnpool.parameters():
        p.requires_grad_(True)
    agent.freeze_encoder = False
    agent._cache.clear()

    optimizer2 = torch.optim.Adam(
        agent.encoder.attnpool.parameters(), lr=LR_ATTNPOOL
    )
    phase2 = train_dlbt(
        agent, train_ds, stim_gen_ds, refs_dict,
        n_epochs=N_EPOCHS_PHASE2, lr=LR, patience=PATIENCE_PHASE2,
        optimizer=optimizer2,
        extra_val_datasets={"task_gen": task_gen_ds, "joint_gen": joint_gen_ds},
    )
    print(f"  best epoch: {phase2.best_epoch}  stim_gen_mse: {phase2.best_val_mse:.4f}")

result = phase2 if phase2 is not None else phase1
print(f"\nFinal best stim_gen_mse: {result.best_val_mse:.4f}")

# Phase 2 clears _cache before training; repopulate it now with the final
# attnpool features so downstream SLDA code can read agent._cache[uid].
if not FREEZE_ENCODER:
    print("Repopulating feature cache with final attnpool features...")
    all_refs_list = list(refs_dict.values())
    agent.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_refs_list), 16),
                      desc="caching features", unit="batch"):
            batch   = all_refs_list[i : i + 16]
            spatial = torch.stack(
                [agent._backbone_cache[r.uid] for r in batch]
            ).to(agent.device)
            feats = agent.encoder.attnpool(spatial).float()
            for ref, feat in zip(batch, feats):
                agent._cache[ref.uid] = feat.cpu()

# ---------------------------------------------------------------------------
# Fit per-task SLDA  (Ridge regression: CLIP features → empirical P(right))
# ---------------------------------------------------------------------------
# Features are already cached in agent._cache after DLBT training.
def clip_features(uids: list) -> np.ndarray:
    return np.array([agent._cache[uid].cpu().numpy() for uid in uids])


print("\nFitting per-task SLDA (RidgeCV + temperature calibration)...")
slda_scalers: dict = {}   # task_name -> StandardScaler (fitted on train images)
slda_models:  dict = {}   # task_name -> RidgeCV
slda_temps:   dict = {}   # task_name -> float  (Platt scaling temperature τ)

for task_name in TRAIN_TASKS:
    group = train_ds.df[train_ds.df["task_name"] == task_name]
    if len(group) == 0:
        continue
    uids    = group["uid"].tolist()
    X       = clip_features(uids)
    p_right = (group["count_1"] / (group["count_0"] + group["count_1"])).values

    # RidgeCV: cross-validates over candidate alphas — avoids manually tuning
    # regularisation strength (critical when features=1024 >> samples~200).
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = RidgeCV(alphas=[1e1, 1e2, 1e3, 1e4, 1e5])
    model.fit(X_scaled, p_right)

    # Temperature calibration (Platt scaling) on training data.
    # Ridge predicts raw probabilities p̂; we fit τ > 0 to minimise NLL:
    #   P = σ( logit(p̂) / τ )
    # This re-calibrates confidence without changing the ranking.
    p_pred = np.clip(model.predict(X_scaled), 1e-6, 1.0 - 1e-6)
    logits  = np.log(p_pred / (1.0 - p_pred))

    def _nll_tau(log_tau, logits=logits, targets=p_right):
        p = 1.0 / (1.0 + np.exp(-logits / np.exp(log_tau)))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))

    opt = minimize_scalar(_nll_tau, bounds=(-3.0, 3.0), method="bounded")
    tau = float(np.exp(opt.x))

    slda_scalers[task_name] = scaler
    slda_models[task_name]  = model
    slda_temps[task_name]   = tau

print(f"  Fitted {len(slda_models)} per-task RidgeCV models  "
      f"(τ range: {min(slda_temps.values()):.3f}–{max(slda_temps.values()):.3f}).")


def slda_predict(task_name: str, uids: list) -> np.ndarray:
    X        = clip_features(uids)
    X_scaled = slda_scalers[task_name].transform(X)
    p_pred   = np.clip(slda_models[task_name].predict(X_scaled), 1e-6, 1.0 - 1e-6)
    logits   = np.log(p_pred / (1.0 - p_pred))
    tau      = slda_temps[task_name]
    return 1.0 / (1.0 + np.exp(-logits / tau))


# ---------------------------------------------------------------------------
# Collect predictions — DLBT (all 4 regions) and SLDA (train + stim gen)
# ---------------------------------------------------------------------------
agent.eval()
rng_gt = np.random.default_rng(SEED + 1)
_gt_cache: dict = {}   # (uid, task_name) -> float


def get_true_p(uid: str, task_name: str) -> float:
    key = (uid, task_name)
    if key not in _gt_cache:
        _gt_cache[key] = gt_p_right(uid, TASKS[task_name], n_mc=1000, rng=rng_gt)
    return _gt_cache[key]


def collect_dlbt(ds: BehavioralDataset) -> dict:
    """Per-task predictions for DLBT on a dataset."""
    out = {}
    for task_name, group in ds.iter_tasks():
        task       = TASKS[task_name]
        batch_refs = [refs_dict[uid] for uid in group["uid"]]
        true_p     = np.array([get_true_p(r.uid, task_name) for r in batch_refs])
        with torch.no_grad():
            pred = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
        raw_mse = float(np.mean((pred - true_p) ** 2))
        mc_corr = float(np.mean(pred * (1 - pred))) / (N_MC - 1)
        rho, _  = spearmanr(pred, true_p)
        out[task_name] = dict(pred=pred, true=true_p,
                              cmse=raw_mse - mc_corr, rho=rho)
    return out


def collect_slda(ds: BehavioralDataset) -> dict:
    """Per-task predictions for SLDA on a dataset (training tasks only)."""
    out = {}
    for task_name, group in ds.iter_tasks():
        if task_name not in slda_models:
            continue
        uids   = group["uid"].tolist()
        true_p = np.array([get_true_p(uid, task_name) for uid in uids])
        pred   = slda_predict(task_name, uids)
        raw_mse = float(np.mean((pred - true_p) ** 2))
        rho, _  = spearmanr(pred, true_p)
        out[task_name] = dict(pred=pred, true=true_p, cmse=raw_mse, rho=rho)
    return out


print("\nCollecting predictions (this may take a few minutes for gt_p_right)...")
dlbt_train     = collect_dlbt(train_ds)
dlbt_stim      = collect_dlbt(stim_gen_ds)
dlbt_task      = collect_dlbt(task_gen_ds)
dlbt_joint     = collect_dlbt(joint_gen_ds)

slda_train     = collect_slda(train_ds)
slda_stim      = collect_slda(stim_gen_ds)

# ---------------------------------------------------------------------------
# Plot 1 — DLBT learning curves  (4 regimes, both phases if applicable)
# ---------------------------------------------------------------------------
# Concatenate phase metrics into a single continuous x-axis.
# Phase 2 epoch 0 repeats the restored best state from phase 1, so we drop it.
def _concat(r1, r2, key):
    v1 = getattr(r1, key) if hasattr(r1, key) else r1.extra_val_nlls.get(key) or r1.extra_val_mses.get(key)
    if r2 is None:
        return list(v1)
    v2 = getattr(r2, key) if hasattr(r2, key) else r2.extra_val_nlls.get(key) or r2.extra_val_mses.get(key)
    return list(v1) + list(v2)[1:]   # drop epoch-0 of phase 2 (duplicate)

n_phase1  = len(phase1.train_nlls)   # number of points in phase 1 (epoch 0 … E1)
phase_boundary = n_phase1 - 1        # x-axis position where phase 2 starts

train_nlls_cat = _concat(phase1, phase2, "train_nlls")
val_nlls_cat   = _concat(phase1, phase2, "val_nlls")
train_mses_cat = _concat(phase1, phase2, "train_mses")
val_mses_cat   = _concat(phase1, phase2, "val_mses")
tg_nlls_cat    = (_concat(phase1, phase2, "extra_val_nlls")
                  if False else   # extra_val stored as dicts, handled below
                  list(phase1.extra_val_nlls["task_gen"]) +
                  (list(phase2.extra_val_nlls["task_gen"])[1:] if phase2 else []))
jg_nlls_cat    = (list(phase1.extra_val_nlls["joint_gen"]) +
                  (list(phase2.extra_val_nlls["joint_gen"])[1:] if phase2 else []))
tg_mses_cat    = (list(phase1.extra_val_mses["task_gen"]) +
                  (list(phase2.extra_val_mses["task_gen"])[1:] if phase2 else []))
jg_mses_cat    = (list(phase1.extra_val_mses["joint_gen"]) +
                  (list(phase2.extra_val_mses["joint_gen"])[1:] if phase2 else []))

epochs = range(len(train_nlls_cat))

fig, (ax_nll, ax_mse) = plt.subplots(1, 2, figsize=(11, 3.8))

for ax, tr, vl, tg, jg, ylabel in [
    (ax_nll, train_nlls_cat, val_nlls_cat, tg_nlls_cat, jg_nlls_cat, "NLL"),
    (ax_mse, train_mses_cat, val_mses_cat, tg_mses_cat, jg_mses_cat, "cMSE"),
]:
    ax.plot(epochs, tr, color=C_TRAIN, label="train",    lw=1.2)
    ax.plot(epochs, vl, color=C_STIM,  label="stim gen", lw=1.2)
    ax.plot(epochs, tg, color=C_TASK,  label="task gen", lw=1.2)
    ax.plot(epochs, jg, color=C_JOINT, label="joint gen",lw=1.2)
    # best epoch within the final phase (offset for phase 2)
    best_x = result.best_epoch + (phase_boundary if phase2 else 0)
    ax.axvline(best_x, ls=":", color="gray", lw=0.8)
    # phase boundary
    if phase2 is not None:
        ax.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.text(phase_boundary + 1, 0.98, "phase 2", fontsize=7,
                va="top", transform=ax.get_xaxis_transform(), color="black", alpha=0.6)
    ax.set(ylabel=ylabel, xlabel="epoch", title=f"{model_label} — {ylabel}")
    ax.legend(fontsize=8)

ax_mse.axhline(train_ds.noise_floor(), ls="--", color=C_TRAIN, alpha=0.4, lw=1,
               label=f"train floor ({train_ds.noise_floor():.4f})")

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(f"examples/plots/03_learning_curves_{RUN_TAG}.png", dpi=150, bbox_inches="tight")
print(f"Saved: examples/plots/03_learning_curves_{RUN_TAG}.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2 — 6-panel summary scatter  (2 rows × 3 cols)
# ---------------------------------------------------------------------------
#  Row 0: DLBT train | DLBT stim gen | DLBT task gen
#  Row 1: SLDA train | SLDA stim gen | DLBT joint gen

def _summary_scatter(ax, pt: dict, task_names: list, color: str, title: str,
                     mc_n=None):
    pred_all = np.concatenate([pt[t]["pred"] for t in task_names if t in pt])
    true_all = np.concatenate([pt[t]["true"] for t in task_names if t in pt])
    raw_mse  = float(np.mean((pred_all - true_all) ** 2))
    if mc_n is not None:
        mc_corr = float(np.mean(pred_all * (1 - pred_all))) / (mc_n - 1)
        cmse    = raw_mse - mc_corr
    else:
        cmse = raw_mse
    rho, _ = spearmanr(pred_all, true_all)
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8, zorder=0)
    ax.scatter(pred_all, true_all, alpha=0.3, s=8, color=color, linewidths=0)
    ax.set(title=f"{title}\ncMSE={cmse:.4f}   ρ={rho:.3f}",
           xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=8)


panels = [
    # (per_task_dict, task_names, color, title, mc_n, row, col)
    (dlbt_train, TRAIN_TASKS, C_TRAIN, f"{model_label} — Train",    N_MC,  0, 0),
    (dlbt_stim,  TRAIN_TASKS, C_STIM,  f"{model_label} — Stim gen", N_MC,  0, 1),
    (dlbt_task,  VAL_TASKS,   C_TASK,  f"{model_label} — Task gen", N_MC,  0, 2),
    (slda_train, TRAIN_TASKS, C_TRAIN, "SLDA — Train",              None,  1, 0),
    (slda_stim,  TRAIN_TASKS, C_STIM,  "SLDA — Stim gen",           None,  1, 1),
    (dlbt_joint, VAL_TASKS,   C_JOINT, f"{model_label} — Joint gen",N_MC,  1, 2),
]

fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.45, "wspace": 0.12})
for pt, task_names, color, title, mc_n, row, col in panels:
    ax = axes[row, col]
    _summary_scatter(ax, pt, task_names, color, title, mc_n)
    if col == 0:
        ax.set_ylabel("True P(right)", fontsize=9)
    if row == 1:
        ax.set_xlabel("Predicted P(right)", fontsize=9)

sns.despine(fig=fig, trim=True)
plt.savefig(f"examples/plots/03_pred_vs_true_{RUN_TAG}.png", dpi=150, bbox_inches="tight")
print(f"Saved: examples/plots/03_pred_vs_true_{RUN_TAG}.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3 — DLBT per-task grid  (all 4 regions, small panels)
# ---------------------------------------------------------------------------
# Each panel shows all images for that task, colored by regime:
#   training tasks: train images (orange) + stim gen images (blue)
#   val tasks:      task gen images (purple) + joint gen images (green)

ALL_TASKS = TRAIN_TASKS + VAL_TASKS
N_COLS    = 8
N_ROWS    = math.ceil(len(ALL_TASKS) / N_COLS)

fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * 2.0, N_ROWS * 2.2),
                         sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.08})
for ax in axes.flat[len(ALL_TASKS):]:
    ax.set_visible(False)

for idx, (ax, task_name) in enumerate(zip(axes.flat, ALL_TASKS)):
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    is_val = task_name in VAL_TASKS
    if not is_val:
        # training tasks: train (orange) + stim gen (blue)
        for pt, color in [(dlbt_train, C_TRAIN), (dlbt_stim, C_STIM)]:
            if task_name in pt:
                d = pt[task_name]
                ax.scatter(d["pred"], d["true"], alpha=0.5, s=5,
                           color=color, linewidths=0)
        rho_tr = dlbt_train.get(task_name, {}).get("rho", float("nan"))
        rho_sg = dlbt_stim.get(task_name,  {}).get("rho", float("nan"))
        ax.text(0.05, 0.93, f"ρ={rho_tr:.2f}", transform=ax.transAxes,
                fontsize=6, color=C_TRAIN, va="top")
        ax.text(0.05, 0.78, f"ρ={rho_sg:.2f}", transform=ax.transAxes,
                fontsize=6, color=C_STIM,  va="top")
    else:
        # val tasks: task gen (purple) + joint gen (green)
        for pt, color in [(dlbt_task, C_TASK), (dlbt_joint, C_JOINT)]:
            if task_name in pt:
                d = pt[task_name]
                ax.scatter(d["pred"], d["true"], alpha=0.5, s=5,
                           color=color, linewidths=0)
        rho_tg = dlbt_task.get(task_name,  {}).get("rho", float("nan"))
        rho_jg = dlbt_joint.get(task_name, {}).get("rho", float("nan"))
        ax.text(0.05, 0.93, f"ρ={rho_tg:.2f}", transform=ax.transAxes,
                fontsize=6, color=C_TASK,  va="top")
        ax.text(0.05, 0.78, f"ρ={rho_jg:.2f}", transform=ax.transAxes,
                fontsize=6, color=C_JOINT, va="top")

    nice = task_name.replace("_and_", " & ").replace("_", "/")
    ax.set_title(nice, fontsize=7, pad=2)
    row, col = divmod(idx, N_COLS)
    if row == N_ROWS - 1:
        ax.set_xlabel("Pred", fontsize=7)
    if col == 0:
        ax.set_ylabel("True", fontsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=5)

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TRAIN,
           markersize=5, label="train"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_STIM,
           markersize=5, label="stim gen"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TASK,
           markersize=5, label="task gen"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_JOINT,
           markersize=5, label="joint gen"),
], loc="lower right", bbox_to_anchor=(1.0, 0.0),
   fontsize=7, frameon=False, ncol=2)
fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
sns.despine(fig=fig, trim=True)
plt.savefig(f"examples/plots/03_per_task_dlbt_{RUN_TAG}.png", dpi=150, bbox_inches="tight")
print(f"Saved: examples/plots/03_per_task_dlbt_{RUN_TAG}.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 4 — SLDA per-task grid  (train + stim gen only)
# ---------------------------------------------------------------------------
N_COLS_S = 8
N_ROWS_S = math.ceil(len(TRAIN_TASKS) / N_COLS_S)

fig, axes = plt.subplots(N_ROWS_S, N_COLS_S,
                         figsize=(N_COLS_S * 2.0, N_ROWS_S * 2.2),
                         sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.08})
for ax in axes.flat[len(TRAIN_TASKS):]:
    ax.set_visible(False)

for idx, (ax, task_name) in enumerate(zip(axes.flat, TRAIN_TASKS)):
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    for pt, color in [(slda_train, C_TRAIN), (slda_stim, C_STIM)]:
        if task_name in pt:
            d = pt[task_name]
            ax.scatter(d["pred"], d["true"], alpha=0.5, s=5,
                       color=color, linewidths=0, marker="s")
    rho_tr = slda_train.get(task_name, {}).get("rho", float("nan"))
    rho_sg = slda_stim.get(task_name,  {}).get("rho", float("nan"))
    ax.text(0.05, 0.93, f"ρ={rho_tr:.2f}", transform=ax.transAxes,
            fontsize=6, color=C_TRAIN, va="top")
    ax.text(0.05, 0.78, f"ρ={rho_sg:.2f}", transform=ax.transAxes,
            fontsize=6, color=C_STIM,  va="top")
    nice = task_name.replace("_and_", " & ").replace("_", "/")
    ax.set_title(nice, fontsize=7, pad=2)
    row, col = divmod(idx, N_COLS_S)
    if row == N_ROWS_S - 1:
        ax.set_xlabel("Pred", fontsize=7)
    if col == 0:
        ax.set_ylabel("True", fontsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=5)

fig.legend(handles=[
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_TRAIN,
           markersize=5, label="train"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_STIM,
           markersize=5, label="stim gen"),
], loc="lower right", bbox_to_anchor=(1.0, 0.0),
   fontsize=7, frameon=False, ncol=1)
fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
sns.despine(fig=fig, trim=True)
plt.savefig(f"examples/plots/03_per_task_slda_{RUN_TAG}.png", dpi=150, bbox_inches="tight")
print(f"Saved: examples/plots/03_per_task_slda_{RUN_TAG}.png")
plt.close()
