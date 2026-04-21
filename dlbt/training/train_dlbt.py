"""
Training loop for DlbtAgent.

Design principles:
  - Typed to DlbtAgent (not a generic Agent) — keeps the loop simple and auditable.
  - Receives train + val datasets; test set is never seen here.
  - Core loop is minimal; logging/checkpointing are handled via callbacks.
  - Early stopping on val MSE.

Usage:
    result = train_dlbt(agent, train_ds, val_ds, image_refs)
    print(result.best_val_mse)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import TASKS, Task
from dlbt.training.metrics import multinomial_nll, mse, corrected_mse


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    agent:        DlbtAgent
    train_nlls:   List[float] = field(default_factory=list)
    val_nlls:     List[float] = field(default_factory=list)
    train_mses:   List[float] = field(default_factory=list)
    val_mses:     List[float] = field(default_factory=list)
    train_kls:    List[float] = field(default_factory=list)  # mean KL per epoch (0 if kl_weight==0)
    best_epoch:   int = 0
    best_val_mse: float = float("inf")
    extra_val_nlls: Dict[str, List[float]] = field(default_factory=dict)
    extra_val_mses: Dict[str, List[float]] = field(default_factory=dict)
    end_state:    dict = field(default_factory=dict)  # weights at end of training


# ---------------------------------------------------------------------------
# KL regularisation
# ---------------------------------------------------------------------------

def dirichlet_kl_uniform(alpha: torch.Tensor, alpha0: float = 1.0) -> torch.Tensor:
    """
    Mean KL( Dir(α_i) || Dir(α0 · 1) ) over a batch of Dirichlet parameters.

    KL between two Dirichlet distributions Dir(α) and Dir(β) is:
        log Γ(Σα_k) - Σ log Γ(α_k) - log Γ(Σβ_k) + Σ log Γ(β_k)
        + Σ (α_k - β_k)(ψ(α_k) - ψ(Σα_k))

    For the uniform symmetric prior β = α0 · 1 this simplifies to a single
    scalar reference per batch row.

    Args:
        alpha:  [B, K] concentration parameters, all strictly positive.
        alpha0: scalar prior concentration (default 1.0 → flat Dirichlet).
                • alpha0 = 1.0  uniform prior, penalises any peaking.
                • alpha0 < 1.0  sparse prior (promotes peaked distributions).
                • alpha0 > 1.0  smooth prior (allows moderate spreading).
    Returns:
        Scalar mean KL (differentiable w.r.t. alpha).
    """
    K         = alpha.shape[1]
    alpha_sum = alpha.sum(dim=1)                           # [B]
    a0        = alpha.new_tensor(alpha0)

    kl = (
        torch.lgamma(alpha_sum)                            # log Γ(Σα_k)
        - alpha.lgamma().sum(dim=1)                        # −Σ log Γ(α_k)
        - torch.lgamma(a0 * K)                             # −log Γ(K·α0)
        + K * torch.lgamma(a0)                             # +K log Γ(α0)
        + (                                                # Σ (α_k − α0)(ψ(α_k) − ψ(Σα_k))
            (alpha - alpha0)
            * (torch.digamma(alpha)
               - torch.digamma(alpha_sum.unsqueeze(1)))
          ).sum(dim=1)
    )
    return kl.mean()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    agent: DlbtAgent,
    dataset: BehavioralDataset,
    image_refs: Dict[str, ImageRef],
) -> tuple[float, float]:
    """
    Evaluate agent on a dataset. Returns (nll, mse).
    """
    all_probs: List[torch.Tensor]  = []
    all_counts: List[torch.Tensor] = []

    for task_name, group in dataset.iter_tasks():
        task    = TASKS[task_name]
        refs    = [image_refs[uid] for uid in group["uid"]]
        counts  = torch.tensor(
            group[["count_0", "count_1"]].values,
            dtype=torch.float32,
            device=agent.device,
        )
        probs = agent.choice_probs(refs, task)   # [B, 2]
        all_probs.append(probs)
        all_counts.append(counts)

    probs_cat  = torch.cat(all_probs,  dim=0)    # [N_total, 2]
    counts_cat = torch.cat(all_counts, dim=0)

    nll_val = multinomial_nll(probs_cat, counts_cat).item()
    mse_val = corrected_mse(probs_cat, counts_cat,
                            n_mc_samples=agent.n_mc_samples)
    return nll_val, mse_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dlbt(
    agent: DlbtAgent,
    train_dataset: BehavioralDataset,
    val_dataset:   BehavioralDataset,
    image_refs:    Dict[str, ImageRef],
    n_epochs:      int   = 300,
    lr:            float = 1e-3,
    patience:      int   = 30,
    callbacks:     List[Callable[[int, float, float], None]] = (),
    optimizer:     Optional[torch.optim.Optimizer] = None,
    grad_clip:     float = 1.0,
    extra_val_datasets: Optional[Dict[str, BehavioralDataset]] = None,
    kl_weight:     float = 0.0,
    prior_alpha:   float = 1.0,
) -> TrainResult:
    """
    Train a DlbtAgent on behavioural choice data.

    Args:
        agent:          the DlbtAgent to train (modified in-place).
        train_dataset:  training observations.
        val_dataset:    validation observations (used for early stopping).
        image_refs:     uid -> ImageRef lookup for all images.
        n_epochs:       maximum number of full-dataset passes.
        lr:             Adam learning rate (ignored if optimizer is provided).
        patience:       early-stopping patience (epochs without val improvement).
        callbacks:      list of callables called each epoch as
                        callback(epoch, val_nll, val_mse).
                        Use for logging, TensorBoard, etc.
        optimizer:      optional pre-built optimizer. Use this to set per-
                        parameter-group learning rates (e.g. different LRs for
                        the mapper and attnpool). If None, a default Adam with
                        lr is constructed from agent.trainable_parameters().
        grad_clip:      max gradient norm (torch.nn.utils.clip_grad_norm_).
                        Prevents early-epoch NLL spikes when using high LR.
        kl_weight:      weight λ for the Dirichlet KL regulariser.
                        loss = NLL + λ · mean_i KL(Dir(α_i) || Dir(α0·1)).
                        Set to 0.0 (default) for pure NLL — fully backward-
                        compatible with the unregularised training loop.
        prior_alpha:    concentration α0 of the symmetric Dirichlet prior.
                        1.0 (default) = uniform prior, penalises peaking.

    Returns:
        TrainResult with metrics and best-weight agent.
    """
    result    = TrainResult(agent=agent)
    extra_val_datasets = extra_val_datasets or {}
    result.extra_val_nlls = {k: [] for k in extra_val_datasets}
    result.extra_val_mses = {k: [] for k in extra_val_datasets}
    if optimizer is None:
        optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=lr)

    # Pre-build the list of unique training image refs (used by KL term).
    # Done once here so the inner loop doesn't rebuild it every epoch.
    train_uids     = list(train_dataset.df["uid"].unique())
    train_refs_all = [image_refs[uid] for uid in train_uids]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    # Pre-cache features before the training loop.
    # frozen:  cache full CLIP features [1024] — forward passes become lookups.
    # attnpool: cache pre-attnpool spatial maps — each epoch only runs attnpool + mapper.
    all_refs = list(image_refs.values())
    if agent.freeze_encoder:
        agent.precompute_features(all_refs)
    else:
        agent.precompute_backbone_features(all_refs)

    # Baseline evaluation (epoch 0)
    train_nll0, train_mse0 = evaluate(agent, train_dataset, image_refs)
    val_nll0,   val_mse0   = evaluate(agent, val_dataset,   image_refs)
    result.train_nlls.append(train_nll0)
    result.train_mses.append(train_mse0)
    result.val_nlls.append(val_nll0)
    result.val_mses.append(val_mse0)
    result.train_kls.append(0.0)
    result.best_val_mse = val_mse0

    for name, ds in extra_val_datasets.items():
        nll, mse_ = evaluate(agent, ds, image_refs)
        result.extra_val_nlls[name].append(nll)
        result.extra_val_mses[name].append(mse_)

    # Save initial weights
    best_state = copy.deepcopy(agent.state_dict())
    no_improve = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="training", unit="epoch")
    for epoch in pbar:

        # ---- Forward + backward pass over all tasks -----------------------
        # Gradient accumulation: backward() after each task so only one
        # computation graph lives in memory at a time (critical for phase 2
        # where attnpool is trainable and graphs are much larger).
        agent.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_total    = len(train_dataset)

        for task_name, group in train_dataset.iter_tasks():
            task   = TASKS[task_name]
            refs   = [image_refs[uid] for uid in group["uid"]]
            counts = torch.tensor(
                group[["count_0", "count_1"]].values,
                dtype=torch.float32,
                device=agent.device,
            )
            probs     = agent.choice_probs(refs, task)                    # [B, 2]
            task_loss = multinomial_nll(probs, counts) * len(refs) / n_total
            task_loss.backward()           # free graph immediately
            total_loss += task_loss.item()

        # ---- KL regularisation (optional) ---------------------------------
        # Computed over all unique training images (not per-task, to avoid
        # double-counting images that appear in multiple tasks).
        # kl_weight=0.0 skips this entirely → pure NLL, backward-compatible.
        epoch_kl = 0.0
        if kl_weight > 0.0:
            alpha     = agent.get_alpha(train_refs_all)                    # [N, K]
            kl_loss   = kl_weight * dirichlet_kl_uniform(alpha, prior_alpha)
            kl_loss.backward()
            epoch_kl  = kl_loss.item()
            total_loss += epoch_kl

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), grad_clip)
        optimizer.step()

        # ---- Evaluation ---------------------------------------------------
        agent.eval()
        train_nll, train_mse_val = evaluate(agent, train_dataset, image_refs)
        val_nll,   val_mse_val   = evaluate(agent, val_dataset,   image_refs)

        result.train_nlls.append(train_nll)
        result.train_mses.append(train_mse_val)
        result.val_nlls.append(val_nll)
        result.val_mses.append(val_mse_val)
        result.train_kls.append(epoch_kl)

        for name, ds in extra_val_datasets.items():
            nll, mse_ = evaluate(agent, ds, image_refs)
            result.extra_val_nlls[name].append(nll)
            result.extra_val_mses[name].append(mse_)

        # ---- LR schedule --------------------------------------------------
        scheduler.step()

        # ---- Progress bar -------------------------------------------------
        postfix = dict(
            train_nll=f"{train_nll:.3f}",
            val_nll=f"{val_nll:.3f}",
            val_mse=f"{val_mse_val:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if kl_weight > 0.0:
            postfix["kl"] = f"{epoch_kl:.4f}"
        pbar.set_postfix(**postfix)

        # ---- Callbacks ----------------------------------------------------
        for cb in callbacks:
            cb(epoch, val_nll, val_mse_val)

        # ---- Early stopping -----------------------------------------------
        if val_mse_val < result.best_val_mse:
            result.best_val_mse = val_mse_val
            result.best_epoch   = epoch
            best_state          = copy.deepcopy(agent.state_dict())
            no_improve          = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}. Best epoch: {result.best_epoch}.")
                break

    # Save end-of-training weights before restoring best
    result.end_state = copy.deepcopy(agent.state_dict())
    # Restore best weights
    agent.load_state_dict(best_state)
    return result
