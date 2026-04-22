"""
Training loop for FlexAgent (logistic-normal SEU).

Mirrors train_dlbt.py exactly, with two differences:
  1. Typed to FlexAgent instead of DlbtAgent.
  2. KL regulariser is the closed-form Gaussian KL
       KL( N(μ,Σ) || N(0, prior_std²·I) )
     instead of the Dirichlet KL.  This penalises beliefs that deviate from a
     diffuse uniform-like prior on the simplex.

Usage:
    from dlbt.agents.flex import FlexAgent
    from dlbt.training.train_flex import train_flex

    agent  = FlexAgent(cov_type="diag", ...)
    result = train_flex(agent, train_ds, val_ds, refs_dict,
                        kl_weight=0.01, prior_std=1.0)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from dlbt.agents.flex import FlexAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import TASKS
from dlbt.training.metrics import multinomial_nll, corrected_mse


# ---------------------------------------------------------------------------
# Result container (same schema as TrainResult in train_dlbt.py)
# ---------------------------------------------------------------------------

@dataclass
class FlexTrainResult:
    agent:        FlexAgent
    train_nlls:   List[float] = field(default_factory=list)
    val_nlls:     List[float] = field(default_factory=list)
    train_mses:   List[float] = field(default_factory=list)
    val_mses:     List[float] = field(default_factory=list)
    train_kls:    List[float] = field(default_factory=list)
    best_epoch:   int = 0
    best_val_mse: float = float("inf")
    extra_val_nlls: Dict[str, List[float]] = field(default_factory=dict)
    extra_val_mses: Dict[str, List[float]] = field(default_factory=dict)
    end_state:    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation helper (identical to train_dlbt.evaluate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    agent: FlexAgent,
    dataset: BehavioralDataset,
    image_refs: Dict[str, ImageRef],
) -> tuple[float, float]:
    all_probs:  List[torch.Tensor] = []
    all_counts: List[torch.Tensor] = []

    for task_name, group in dataset.iter_tasks():
        task   = TASKS[task_name]
        refs   = [image_refs[uid] for uid in group["uid"]]
        counts = torch.tensor(
            group[["count_0", "count_1"]].values,
            dtype=torch.float32,
            device=agent.device,
        )
        probs = agent.choice_probs(refs, task)
        all_probs.append(probs)
        all_counts.append(counts)

    probs_cat  = torch.cat(all_probs,  dim=0)
    counts_cat = torch.cat(all_counts, dim=0)
    nll_val = multinomial_nll(probs_cat, counts_cat).item()
    mse_val = corrected_mse(probs_cat, counts_cat, n_mc_samples=agent.n_mc_samples)
    return nll_val, mse_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_flex(
    agent: FlexAgent,
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
    prior_std:     float = 1.0,
) -> FlexTrainResult:
    """
    Train a FlexAgent on behavioural choice data.

    Args:
        agent:          FlexAgent to train (modified in-place).
        train_dataset:  training observations.
        val_dataset:    validation observations (early stopping).
        image_refs:     uid -> ImageRef lookup.
        n_epochs:       maximum epochs.
        lr:             Adam LR (ignored if optimizer is provided).
        patience:       early-stopping patience.
        callbacks:      epoch callbacks(epoch, val_nll, val_mse).
        optimizer:      optional pre-built optimiser.
        grad_clip:      gradient norm clip (0 = disabled).
        extra_val_datasets: additional datasets to evaluate each epoch.
        kl_weight:      λ for Gaussian KL regulariser.
                        loss = NLL + λ · mean_i KL( N(μ_i, Σ_i) || N(0, prior_std²·I) ).
                        Set to 0.0 for pure NLL.
        prior_std:      prior standard deviation for the Gaussian KL.
                        1.0 → standard normal prior (moderate regularisation).
                        Larger → weaker regularisation.

    Returns:
        FlexTrainResult with metrics and best-weight agent.
    """
    result = FlexTrainResult(agent=agent)
    extra_val_datasets = extra_val_datasets or {}
    result.extra_val_nlls = {k: [] for k in extra_val_datasets}
    result.extra_val_mses = {k: [] for k in extra_val_datasets}

    if optimizer is None:
        optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=lr)

    # Pre-build unique training image refs (used by KL term).
    train_uids     = list(train_dataset.df["uid"].unique())
    train_refs_all = [image_refs[uid] for uid in train_uids]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    # Pre-cache features
    all_refs = list(image_refs.values())
    if agent.freeze_encoder:
        agent.precompute_features(all_refs)
    else:
        agent.precompute_backbone_features(all_refs)

    # Baseline (epoch 0)
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

    best_state = copy.deepcopy(agent.state_dict())
    no_improve = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="training (flex)", unit="epoch")
    for epoch in pbar:

        # ---- Forward + backward (task by task for memory efficiency) --------
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
            probs     = agent.choice_probs(refs, task)
            task_loss = multinomial_nll(probs, counts) * len(refs) / n_total
            task_loss.backward()
            total_loss += task_loss.item()

        # ---- Gaussian KL regularisation (optional) -------------------------
        # Computed once over unique training images to avoid double-counting.
        epoch_kl = 0.0
        if kl_weight > 0.0:
            kl_loss  = kl_weight * agent.kl_loss(train_refs_all, prior_std=prior_std)
            kl_loss.backward()
            epoch_kl = kl_loss.item()
            total_loss += epoch_kl

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), grad_clip)
        optimizer.step()

        # ---- Evaluation -----------------------------------------------------
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

        scheduler.step()

        # ---- Progress bar ---------------------------------------------------
        postfix = dict(
            train_nll=f"{train_nll:.3f}",
            val_nll=f"{val_nll:.3f}",
            val_mse=f"{val_mse_val:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if kl_weight > 0.0:
            postfix["kl"] = f"{epoch_kl:.4f}"
        pbar.set_postfix(**postfix)

        for cb in callbacks:
            cb(epoch, val_nll, val_mse_val)

        # ---- Early stopping -------------------------------------------------
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

    result.end_state = copy.deepcopy(agent.state_dict())
    agent.load_state_dict(best_state)
    return result
