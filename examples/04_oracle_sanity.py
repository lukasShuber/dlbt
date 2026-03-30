"""
Oracle sanity check for DLBT.

Replaces the frozen CLIP encoder with oracle features derived directly from
the continuous metadata used to generate ground truth α*(x). Everything
downstream (mapper, Dirichlet, SEU, training loop) is identical to the
standard DLBT agent.

Oracle hierarchy:

  A. Head-only ceiling
     oracle_prealpha — K=16D softplus_inv(gt_alpha(uid)).
     Mapper just inverts Softplus; tests Dirichlet + SEU + loss + optimiser.

  B. Sufficient statistics
     oracle_interaction — K=16D: clarity at the true-state slot, zeros elsewhere.
                          A Linear(K, K) can represent gt_alpha perfectly:
                          α = BASE·1 + PEAK·v.  True linear sufficient stat.
     oracle_soft4-MLP   — 4D soft marginals + 256-GELU MLP (nonlinear).
     oracle_soft4-linear — 4D soft marginals + linear mapper (can't do products).
     oracle_onehot-linear — 16D one-hot + linear (no clarity info).

  C. Bottleneck (example 03) — frozen CLIP features.

Interpretation:
  A should hit the noise floor. B-interaction and B-soft4-MLP should also
  converge near the floor. Gaps in B-weak tiers and C isolate what is lost
  at each level (mapper expressivity vs representation).

Run from repo root:
    python examples/04_oracle_sanity.py
"""

import random
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from torch.distributions import Dirichlet

from dlbt.constants import (
    K, DIM_FRONT_BACK, DIM_SHAPE, DIM_TRANSP, DIM_GLOSS,
    Y_THRESHOLD, TRANSP_THRESH, GLOSS_THRESH, NON_TRIANGULAR_SHAPES,
)
from dlbt.data.image_ref import load_image_refs, image_refs_as_list, balanced_refs
from dlbt.data.task import TASKS
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.agents.base import Agent
from dlbt.training.train_dlbt import train_dlbt

# ---------------------------------------------------------------------------
# Config — must match example 03 for fair comparison
# ---------------------------------------------------------------------------
METADATA = "stimuli/imgs/metadata.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Device: {DEVICE} (no GPU)")

SEED               = 42
N_TRIALS           = 100
PEAK               = 15.0
BASE_CONCENTRATION = 1.0
BETA               = 5.0
N_EPOCHS           = 1000
LR                 = 1e-2
N_MC               = 200

TRAIN_TASKS = [
    # simple — one per dimension
    "front_back", "glossy",
    # composites
    "front_and_transparent",
    "nontriangular_and_glossy",
    "triangular_and_front",
    "nontriangular_and_front",
    "back_and_glossy",
    "triangular_and_transparent",
]
VAL_TASKS = [
    "triangular",
    "transparent"
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load stimuli + continuous metadata
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")


def _load_continuous_metadata(path: str) -> dict:
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            z   = rec["z"]
            result[rec["id"]] = dict(
                y            = z["pos_xy"][1],
                transparency = z["transparency"],
                glossiness   = z["glossiness"],
                is_nontri    = z["shape_name"] in NON_TRIANGULAR_SHAPES,
            )
    return result


cont_meta = _load_continuous_metadata(METADATA)

# ---------------------------------------------------------------------------
# Ground truth — identical to example 03
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gt_alpha(uid: str) -> np.ndarray:
    """
    Structured Dirichlet with soft mean and clarity-scaled concentration.
    Identical to example 03 — keep both in sync.

        q_k(x) = product of per-dimension soft marginals
        λ(x)   = BASE_CONCENTRATION + PEAK · clarity(x)
        α(x)   = ε + λ(x) · q(x)
    """
    z = cont_meta[uid]

    p_back   = _sigmoid(BETA * (z["y"]            - Y_THRESHOLD))
    p_nontri = float(z["is_nontri"])
    p_transp = _sigmoid(BETA * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(BETA * (z["glossiness"]   - GLOSS_THRESH))

    q = np.empty(K, dtype=np.float64)
    for k in range(K):
        k_back   = (k >> DIM_FRONT_BACK) & 1
        k_nontri = (k >> DIM_SHAPE)      & 1
        k_transp = (k >> DIM_TRANSP)     & 1
        k_glossy = (k >> DIM_GLOSS)      & 1
        q[k] = (
            (p_back   if k_back   else (1.0 - p_back))   *
            (p_nontri if k_nontri else (1.0 - p_nontri)) *
            (p_transp if k_transp else (1.0 - p_transp)) *
            (p_glossy if k_glossy else (1.0 - p_glossy))
        )

    clarity = (abs(p_back   - 0.5) * 2.0 *
               abs(p_transp - 0.5) * 2.0 *
               abs(p_glossy - 0.5) * 2.0)

    lam = BASE_CONCENTRATION + PEAK * clarity
    return 1e-6 + lam * q


def gt_p_right(uid: str, task, n_mc: int = 1000, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    beliefs = rng.dirichlet(gt_alpha(uid), size=n_mc)
    return float((beliefs @ task.delta_u > 0).mean())


def sample_behavior(ref, task, n_trials: int, rng) -> tuple[int, int]:
    beliefs = rng.dirichlet(gt_alpha(ref.uid), size=n_trials)
    count_1 = int((beliefs @ task.delta_u > 0).sum())
    return n_trials - count_1, count_1


# ---------------------------------------------------------------------------
# Oracle feature constructors
# ---------------------------------------------------------------------------

def softplus_inv(y: np.ndarray) -> np.ndarray:
    """
    Numerically stable inverse of Softplus: log(exp(y) - 1).
    For y > 20 the function is nearly linear, so we use y directly to
    avoid overflow in exp(y).
    """
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 20, y, np.log(np.expm1(y)))


def oracle_prealpha(uid: str) -> np.ndarray:
    """
    [A — head-only ceiling] softplus_inv(gt_alpha(uid)), shape [K].

    Feeding this to a Linear + Softplus mapper means the mapper only needs
    to learn the identity map (Softplus ∘ softplus_inv = identity). Any
    remaining loss is irreducible noise from the Dirichlet sampling + SEU.
    """
    return softplus_inv(gt_alpha(uid)).astype(np.float32)


def oracle_soft4(uid: str) -> np.ndarray:
    """
    4D soft marginals [p_back, p_nontri, p_transp, p_glossy].
    Exactly the quantities that drive gt_alpha, but the mapper still needs
    to learn the nonlinear product-of-marginals combination rule.
    """
    z = cont_meta[uid]
    return np.array([
        _sigmoid(BETA * (z["y"]            - Y_THRESHOLD)),
        float(z["is_nontri"]),
        _sigmoid(BETA * (z["transparency"] - TRANSP_THRESH)),
        _sigmoid(BETA * (z["glossiness"]   - GLOSS_THRESH)),
    ], dtype=np.float32)


def oracle_onehot(uid: str) -> np.ndarray:
    """
    16D one-hot of the true discrete latent state.
    Missing clarity → can't represent per-image concentration variation.
    """
    z = cont_meta[uid]
    k = (
        (int(z["y"]            > Y_THRESHOLD) << DIM_FRONT_BACK) |
        (int(z["is_nontri"])                  << DIM_SHAPE)      |
        (int(z["transparency"] > TRANSP_THRESH) << DIM_TRANSP)   |
        (int(z["glossiness"]   > GLOSS_THRESH)  << DIM_GLOSS)
    )
    v    = np.zeros(K, dtype=np.float32)
    v[k] = 1.0
    return v


def oracle_interaction(uid: str) -> np.ndarray:
    """
    [B — linear sufficient stat] K=16D vector: clarity at the true-state slot,
    zeros everywhere else.

    Why this is a true linear sufficient stat:
        Let v = oracle_interaction(uid).
        A Linear(K, K) with weight W and bias b can produce:
            α_k = Softplus(b_k + W_k · v)
                = Softplus(b_k + W_{k, true_k} * clarity)
        Setting b_k = softplus_inv(BASE) and W_{k,k} = PEAK / Softplus'(...)
        recovers gt_alpha exactly (up to the soft-q approximation).
        This is linear in v — no products of input features are required.
    """
    z = cont_meta[uid]
    p_back   = _sigmoid(BETA * (z["y"]            - Y_THRESHOLD))
    p_transp = _sigmoid(BETA * (z["transparency"] - TRANSP_THRESH))
    p_glossy = _sigmoid(BETA * (z["glossiness"]   - GLOSS_THRESH))
    clarity  = (abs(p_back   - 0.5) * 2.0 *
                abs(p_transp - 0.5) * 2.0 *
                abs(p_glossy - 0.5) * 2.0)
    k = (
        (int(z["y"]            > Y_THRESHOLD)  << DIM_FRONT_BACK) |
        (int(z["is_nontri"])                   << DIM_SHAPE)      |
        (int(z["transparency"] > TRANSP_THRESH) << DIM_TRANSP)    |
        (int(z["glossiness"]   > GLOSS_THRESH)  << DIM_GLOSS)
    )
    v    = np.zeros(K, dtype=np.float32)
    v[k] = float(clarity)
    return v


# ---------------------------------------------------------------------------
# OracleDlbtAgent — same as DlbtAgent but with a lookup-table encoder
# ---------------------------------------------------------------------------

class OracleDlbtAgent(nn.Module, Agent):
    """
    DLBT with oracle image features instead of CLIP.

    Replaces the encoder with a simple dict lookup. The mapper, Dirichlet
    sampler, and SEU decision rule are identical to DlbtAgent, making this
    a clean ablation of the representation component.

    Compatible with train_dlbt (same interface as DlbtAgent).
    """

    freeze_encoder = True   # tells train_dlbt to call precompute_features

    def __init__(
        self,
        feature_dict:   dict,               # uid -> np.ndarray oracle features
        feature_dim:    int,
        n_mc_samples:   int          = 200,
        device:         torch.device = torch.device("cpu"),
        mapper_hidden:  int | None   = None, # None = linear, int = MLP hidden dim
    ):
        super().__init__()
        self.device       = device
        self.n_mc_samples = n_mc_samples

        # Convert oracle features to device tensors once
        self._cache = {
            uid: torch.tensor(feat, dtype=torch.float32, device=device)
            for uid, feat in feature_dict.items()
        }

        # Linear or MLP mapper
        if mapper_hidden is None:
            linear = nn.Linear(feature_dim, K)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 1.1)
            self.mapper = nn.Sequential(linear, nn.Softplus()).to(device)
        else:
            h1 = nn.Linear(feature_dim, mapper_hidden)
            h2 = nn.Linear(mapper_hidden, K)
            nn.init.xavier_uniform_(h1.weight);  nn.init.zeros_(h1.bias)
            nn.init.xavier_uniform_(h2.weight);  nn.init.constant_(h2.bias, 1.1)
            self.mapper = nn.Sequential(h1, nn.GELU(), h2, nn.Softplus()).to(device)

    # train_dlbt calls this; oracle features are already loaded so it's a no-op
    def precompute_features(self, image_refs, batch_size: int = 16) -> None:
        pass

    def get_alpha(self, image_refs) -> torch.Tensor:
        feats = torch.stack([self._cache[r.uid] for r in image_refs])
        return self.mapper(feats).clamp(min=1e-6)   # [B, K]

    def choice_probs(self, image_refs, task) -> torch.Tensor:
        return (self._choice_probs_train(image_refs, task)
                if self.training
                else self._choice_probs_eval(image_refs, task))

    def _choice_probs_train(self, image_refs, task) -> torch.Tensor:
        N       = self.n_mc_samples
        alpha   = self.get_alpha(image_refs)
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=self.device)
        b       = Dirichlet(alpha).rsample((N,))                    # [N, B, K]
        logit   = torch.einsum("nbk,k->nb", b, delta_u)            # [N, B]
        l2      = torch.stack([-logit, logit], dim=-1)
        soft    = F.softmax(l2, dim=-1)
        hard    = F.one_hot(l2.argmax(-1), 2).float()
        return ((hard - soft).detach() + soft).mean(dim=0)         # [B, 2]

    @torch.no_grad()
    def _choice_probs_eval(self, image_refs, task) -> torch.Tensor:
        N       = self.n_mc_samples
        alpha   = self.get_alpha(image_refs)
        delta_u = torch.tensor(task.delta_u, dtype=torch.float32, device=self.device)
        b       = Dirichlet(alpha).sample((N,))                     # [N, B, K]
        logit   = torch.einsum("nbk,k->nb", b, delta_u)            # [N, B]
        p_right = (logit > 0).float().mean(dim=0)                  # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)         # [B, 2]

    def trainable_parameters(self):
        return list(self.mapper.parameters())


# ---------------------------------------------------------------------------
# Synthetic datasets — same seed as example 03
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def make_synthetic_dataset(task_names: list) -> BehavioralDataset:
    records = []
    for task_name in task_names:
        task = TASKS[task_name]
        for ref in balanced_refs(task, refs, rng=rng):
            c0, c1 = sample_behavior(ref, task, N_TRIALS, rng)
            records.append(Observation(uid=ref.uid, task_name=task_name,
                                        count_0=c0, count_1=c1))
    return BehavioralDataset.from_records(records)


train_ds = make_synthetic_dataset(TRAIN_TASKS)
val_ds   = make_synthetic_dataset(VAL_TASKS)
print(f"Train: {train_ds}  |  Val: {val_ds}")
print(f"Noise floor — train: {train_ds.noise_floor():.4f}  "
      f"val: {val_ds.noise_floor():.4f}")

# ---------------------------------------------------------------------------
# Train both oracle variants
# ---------------------------------------------------------------------------
prealpha_dict    = {uid: oracle_prealpha(uid)    for uid in refs_dict}
interaction_dict = {uid: oracle_interaction(uid) for uid in refs_dict}
soft4_dict       = {uid: oracle_soft4(uid)       for uid in refs_dict}
onehot_dict      = {uid: oracle_onehot(uid)      for uid in refs_dict}

VARIANTS = [
    # label                  feat_dict         feat_dim  mapper_hidden
    ("A-prealpha",           prealpha_dict,    K,        None),  # A: head-only ceiling
    ("B-interaction",        interaction_dict, K,        None),  # B: linear sufficient stat
    ("B-soft4-MLP",          soft4_dict,       4,        256),   # B: sufficient stats, MLP
    ("B-soft4-linear",       soft4_dict,       4,        None),  # B-weak: linear (can't do products)
    ("B-onehot-linear",      onehot_dict,      K,        None),  # B-weak: no clarity
]

results    = {}
trained_agents = {}

for label, feat_dict, feat_dim, mapper_hidden in VARIANTS:
    torch.manual_seed(SEED)   # same init for fair comparison
    print(f"\nTraining {label}  (feat_dim={feat_dim}, "
          f"mapper={'MLP-' + str(mapper_hidden) if mapper_hidden else 'linear'})...")
    agent = OracleDlbtAgent(feat_dict, feat_dim, n_mc_samples=N_MC,
                             device=DEVICE, mapper_hidden=mapper_hidden)
    res = train_dlbt(
        agent, train_ds, val_ds, refs_dict,
        n_epochs=N_EPOCHS, lr=LR, patience=N_EPOCHS,
    )
    print(f"  best_epoch={res.best_epoch}  best_val_mse={res.best_val_mse:.4f}")
    results[label]        = res
    trained_agents[label] = agent

# ---------------------------------------------------------------------------
# Plot 1: learning curves
# ---------------------------------------------------------------------------
noise_train = train_ds.noise_floor()
noise_val   = val_ds.noise_floor()

COLORS = {
    "A-prealpha":        ("#000000", "#888888"),   # black  — head-only ceiling
    "B-interaction":     ("#762a83", "#c2a5cf"),   # purple — linear sufficient stat
    "B-soft4-MLP":       ("#1a9641", "#a6d96a"),   # green  — sufficient stats, MLP
    "B-soft4-linear":    ("#2166ac", "#92c5de"),   # blue   — soft4, linear (weak)
    "B-onehot-linear":   ("#d6604d", "#f4a582"),   # red    — onehot, no clarity (weak)
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax_nll, ax_mse = axes

for label, res in results.items():
    c_train, c_val = COLORS[label]
    ep = range(len(res.train_nlls))
    ax_nll.plot(ep, res.train_nlls, color=c_train, label=f"{label} train")
    ax_nll.plot(ep, res.val_nlls,   color=c_val,   label=f"{label} val", ls="--")
    ax_mse.plot(ep, res.train_mses, color=c_train, label=f"{label} train")
    ax_mse.plot(ep, res.val_mses,   color=c_val,   label=f"{label} val", ls="--")

ax_mse.axhline(noise_train, ls=":", color="gray", lw=1,
               label=f"train floor ({noise_train:.4f})")
ax_mse.axhline(noise_val, ls=":", color="gray", lw=1, alpha=0.5,
               label=f"val floor ({noise_val:.4f})")

ax_nll.set(xlabel="epoch", ylabel="NLL",  title="Oracle — NLL")
ax_mse.set(xlabel="epoch", ylabel="cMSE", title="Oracle — cMSE")
for ax in axes:
    ax.legend(fontsize=7)
    ax.axvline(0, ls=":", color="gray", lw=0.5)

sns.despine(trim=True)
plt.tight_layout()
Path("examples/plots").mkdir(exist_ok=True)
plt.savefig("examples/plots/04_oracle_curves.png", dpi=150)
print("\nSaved: examples/plots/04_oracle_curves.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: scatter — predicted vs true P(right)
# ---------------------------------------------------------------------------
rng_gt = np.random.default_rng(SEED + 1)

# Pre-build group lookup per dataset
def task_groups(ds: BehavioralDataset) -> dict:
    return {tn: g for tn, g in ds.iter_tasks()}

train_groups = task_groups(train_ds)
val_groups   = task_groups(val_ds)

fig, axes = plt.subplots(5, 2, figsize=(9, 17), sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.45, "wspace": 0.12})

for row, (label, agent) in enumerate(trained_agents.items()):
    agent.eval()
    c_train, c_val = COLORS[label]

    for col, (task_names, groups, split, color) in enumerate([
        (TRAIN_TASKS, train_groups, "train", c_train),
        (VAL_TASKS,   val_groups,   "val",   c_val),
    ]):
        pred_all, true_all = [], []
        for task_name in task_names:
            task       = TASKS[task_name]
            batch_refs = [refs_dict[uid] for uid in groups[task_name]["uid"]]
            true_p = np.array([gt_p_right(r.uid, task, n_mc=500, rng=rng_gt)
                                for r in batch_refs])
            with torch.no_grad():
                pred_p = agent.choice_probs(batch_refs, task)[:, 1].cpu().numpy()
            pred_all.append(pred_p)
            true_all.append(true_p)

        pred_all = np.concatenate(pred_all)
        true_all = np.concatenate(true_all)
        raw_mse  = float(np.mean((pred_all - true_all) ** 2))
        mc_corr  = float(np.mean(pred_all * (1 - pred_all))) / (N_MC - 1)
        cmse     = raw_mse - mc_corr
        rho, _   = spearmanr(pred_all, true_all)

        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8, zorder=0)
        ax.scatter(pred_all, true_all, alpha=0.35, s=8, color=color, linewidths=0)
        ax.set_title(f"{label} — {split}\ncMSE={cmse:.4f}   ρ={rho:.3f}", fontsize=8)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("True P(right)", fontsize=9)
        if row == len(trained_agents) - 1:
            ax.set_xlabel("Predicted P(right)", fontsize=9)

sns.despine(fig=fig, trim=True)
plt.savefig("examples/plots/04_oracle_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: examples/plots/04_oracle_scatter.png")
plt.close()
