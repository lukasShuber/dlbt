"""
Training loop for LbtAgent.

Mirrors train_dlbt exactly — same NLL loss, same straight-through
forward pass, same early stopping on val cMSE — but skips the
feature-precomputation step (LbtAgent has no encoder).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from dlbt.agents.lbt import LbtAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import get_task
from dlbt.training.metrics import multinomial_nll, corrected_mse


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    agent:        LbtAgent
    train_nlls:   List[float] = field(default_factory=list)
    val_nlls:     List[float] = field(default_factory=list)
    train_mses:   List[float] = field(default_factory=list)
    val_mses:     List[float] = field(default_factory=list)
    best_epoch:   int = 0
    best_val_mse: float = float("inf")
    extra_val_nlls: Dict[str, List[float]] = field(default_factory=dict)
    extra_val_mses: Dict[str, List[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    agent:      LbtAgent,
    dataset:    BehavioralDataset,
    image_refs: Dict[str, ImageRef],
) -> tuple[float, float]:
    """Return (nll, cMSE) for the given dataset."""
    all_probs:  List[torch.Tensor] = []
    all_counts: List[torch.Tensor] = []

    for task_name, group in dataset.iter_tasks():
        task   = get_task(task_name)
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
    mse_val = corrected_mse(probs_cat, counts_cat,
                            n_mc_samples=agent.n_mc_samples)
    return nll_val, mse_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lbt(
    agent:          LbtAgent,
    train_dataset:  BehavioralDataset,
    val_dataset:    BehavioralDataset,
    image_refs:     Dict[str, ImageRef],
    n_epochs:       int   = 2000,
    lr:             float = 1e-2,
    patience:       int   = 200,
    grad_clip:      float = 1.0,
    optimizer:      Optional[torch.optim.Optimizer] = None,
    extra_val_datasets: Optional[Dict[str, BehavioralDataset]] = None,
) -> TrainResult:
    """
    Train an LbtAgent on behavioural choice data (pure NLL).

    Identical to train_dlbt except there is no encoder to precompute.

    Args:
        agent:          LbtAgent to train (modified in-place).
        train_dataset:  training observations.
        val_dataset:    held-out cells used for early stopping.
        image_refs:     uid -> ImageRef lookup.
        n_epochs:       maximum epochs.
        lr:             Adam learning rate (ignored if optimizer provided).
        patience:       early-stopping patience in epochs.
        grad_clip:      max gradient norm.
        optimizer:      optional pre-built optimizer.
        extra_val_datasets: additional datasets evaluated each epoch.

    Returns:
        TrainResult — agent is restored to best-val-MSE weights.
    """
    result             = TrainResult(agent=agent)
    extra_val_datasets = extra_val_datasets or {}
    result.extra_val_nlls = {k: [] for k in extra_val_datasets}
    result.extra_val_mses = {k: [] for k in extra_val_datasets}

    if optimizer is None:
        optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    # Baseline (epoch 0)
    agent.eval()
    tn0, tm0 = evaluate(agent, train_dataset, image_refs)
    vn0, vm0 = evaluate(agent, val_dataset,   image_refs)
    result.train_nlls.append(tn0); result.train_mses.append(tm0)
    result.val_nlls.append(vn0);   result.val_mses.append(vm0)
    result.best_val_mse = vm0

    for name, ds in extra_val_datasets.items():
        nll, mse_ = evaluate(agent, ds, image_refs)
        result.extra_val_nlls[name].append(nll)
        result.extra_val_mses[name].append(mse_)

    best_state = copy.deepcopy(agent.state_dict())
    no_improve = 0
    n_total    = len(train_dataset)

    pbar = tqdm(range(1, n_epochs + 1), desc="train_lbt", unit="epoch")
    for epoch in pbar:

        agent.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for task_name, group in train_dataset.iter_tasks():
            task   = get_task(task_name)
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

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        agent.eval()
        tn, tm = evaluate(agent, train_dataset, image_refs)
        vn, vm = evaluate(agent, val_dataset,   image_refs)
        result.train_nlls.append(tn); result.train_mses.append(tm)
        result.val_nlls.append(vn);   result.val_mses.append(vm)

        for name, ds in extra_val_datasets.items():
            nll, mse_ = evaluate(agent, ds, image_refs)
            result.extra_val_nlls[name].append(nll)
            result.extra_val_mses[name].append(mse_)

        pbar.set_postfix(
            train_nll=f"{tn:.3f}",
            val_nll  =f"{vn:.3f}",
            val_mse  =f"{vm:.4f}",
            lr       =f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        if vm < result.best_val_mse:
            result.best_val_mse = vm
            result.best_epoch   = epoch
            best_state          = copy.deepcopy(agent.state_dict())
            no_improve          = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}. Best: {result.best_epoch}.")
                break

    agent.load_state_dict(best_state)
    return result
