"""
Training loop for EBMAgent.

Mirrors train_dlbt.py in structure with one key difference: images within
each task are processed in inner mini-batches of size `inner_batch_size`.
This keeps the [B_inner, N, C+K] activation tensor manageable in memory
(default: [32, 1000, 144] ≈ 18 MB), regardless of how many images a task
has (~980 main images per task).

Gradients are accumulated across inner batches before the optimiser step,
so the effective batch is still the full task (equivalent to processing all
images at once, just split for memory).

An additional diagnostic — mean effective sample size (ESS / N) — is
computed and logged each epoch. ESS close to 1 means the EBM has collapsed
to a near-delta distribution; ESS close to 1.0 means the weights are
nearly uniform (the model hasn't learned to concentrate mass yet).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from dlbt.agents.ebm import EBMAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import TASKS
from dlbt.training.metrics import multinomial_nll, corrected_mse


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EBMTrainResult:
    agent:        EBMAgent
    train_nlls:   List[float] = field(default_factory=list)
    val_nlls:     List[float] = field(default_factory=list)
    train_mses:   List[float] = field(default_factory=list)
    val_mses:     List[float] = field(default_factory=list)
    train_ess:    List[float] = field(default_factory=list)   # mean ESS/N per epoch
    best_epoch:   int   = 0
    best_val_mse: float = float("inf")
    extra_val_nlls: Dict[str, List[float]] = field(default_factory=dict)
    extra_val_mses: Dict[str, List[float]] = field(default_factory=dict)
    end_state:    dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    agent: EBMAgent,
    dataset: BehavioralDataset,
    image_refs: Dict[str, ImageRef],
    inner_batch_size: int = 64,
) -> tuple[float, float]:
    """Evaluate agent on a dataset. Returns (nll, cMSE)."""
    all_probs:  List[torch.Tensor] = []
    all_counts: List[torch.Tensor] = []

    for task_name, group in dataset.iter_tasks():
        task   = TASKS[task_name]
        refs   = [image_refs[uid] for uid in group["uid"]]
        counts = torch.tensor(
            group[["count_0", "count_1"]].values,
            dtype=torch.float32, device=agent.device,
        )
        # Inner batching for memory efficiency during eval too
        for b0 in range(0, len(refs), inner_batch_size):
            refs_b   = refs[b0 : b0 + inner_batch_size]
            counts_b = counts[b0 : b0 + inner_batch_size]
            probs_b  = agent.choice_probs(refs_b, task)
            all_probs.append(probs_b)
            all_counts.append(counts_b)

    probs_cat  = torch.cat(all_probs,  dim=0)
    counts_cat = torch.cat(all_counts, dim=0)
    nll_val = multinomial_nll(probs_cat, counts_cat).item()
    mse_val = corrected_mse(probs_cat, counts_cat, n_mc_samples=agent.n_mc_samples)
    return nll_val, mse_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ebm(
    agent: EBMAgent,
    train_dataset: BehavioralDataset,
    val_dataset:   BehavioralDataset,
    image_refs:    Dict[str, ImageRef],
    n_epochs:      int   = 500,
    lr:            float = 1e-3,
    patience:      int   = 50,
    inner_batch_size: int = 32,
    callbacks:     List[Callable] = (),
    optimizer:     Optional[torch.optim.Optimizer] = None,
    grad_clip:     float = 1.0,
    extra_val_datasets: Optional[Dict[str, BehavioralDataset]] = None,
) -> EBMTrainResult:
    """
    Train an EBMAgent on behavioural choice data.

    Args:
        agent:            EBMAgent to train (modified in-place).
        train_dataset:    training observations.
        val_dataset:      validation observations (early stopping on cMSE).
        image_refs:       uid → ImageRef lookup.
        n_epochs:         maximum training epochs.
        lr:               Adam learning rate.
        patience:         early-stopping patience (epochs without improvement).
        inner_batch_size: images per inner mini-batch within each task.
                          Reduce if GPU memory is tight; 32 is safe for most GPUs.
        callbacks:        list of (epoch, val_nll, val_mse) → None callables.
        optimizer:        optional pre-built optimiser.
        grad_clip:        max gradient norm (0 = disabled).
        extra_val_datasets: additional datasets evaluated each epoch.

    Returns:
        EBMTrainResult with metrics, best-weight agent, and ESS trace.
    """
    result = EBMTrainResult(agent=agent)
    extra_val_datasets = extra_val_datasets or {}
    result.extra_val_nlls = {k: [] for k in extra_val_datasets}
    result.extra_val_mses = {k: [] for k in extra_val_datasets}

    if optimizer is None:
        optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01,
    )

    # Pre-cache CLIP features for all images (frozen encoder only)
    all_refs = list(image_refs.values())
    agent.precompute_features(all_refs)

    # Unique training refs for ESS diagnostic
    train_uids     = list(train_dataset.df["uid"].unique())
    train_refs_all = [image_refs[uid] for uid in train_uids]

    # Baseline (epoch 0)
    agent.eval()
    train_nll0, train_mse0 = evaluate(agent, train_dataset, image_refs, inner_batch_size)
    val_nll0,   val_mse0   = evaluate(agent, val_dataset,   image_refs, inner_batch_size)
    result.train_nlls.append(train_nll0)
    result.train_mses.append(train_mse0)
    result.val_nlls.append(val_nll0)
    result.val_mses.append(val_mse0)
    result.train_ess.append(agent.ess(train_refs_all[:64]))   # quick ESS estimate
    result.best_val_mse = val_mse0

    for name, ds in extra_val_datasets.items():
        nll, mse_ = evaluate(agent, ds, image_refs, inner_batch_size)
        result.extra_val_nlls[name].append(nll)
        result.extra_val_mses[name].append(mse_)

    best_state = copy.deepcopy(agent.state_dict())
    no_improve = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="training (EBM)", unit="epoch")
    for epoch in pbar:

        # ---- Forward + backward --------------------------------------------
        agent.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_total    = len(train_dataset)

        for task_name, group in train_dataset.iter_tasks():
            task   = TASKS[task_name]
            refs   = [image_refs[uid] for uid in group["uid"]]
            counts = torch.tensor(
                group[["count_0", "count_1"]].values,
                dtype=torch.float32, device=agent.device,
            )
            n_task = len(refs)

            # Inner mini-batch loop: accumulate gradients across sub-batches.
            # Each sub-batch contributes (Bb / n_total) of the epoch loss,
            # so the sum across all sub-batches equals the full-task NLL
            # scaled to the dataset size (same normalisation as train_dlbt).
            for b0 in range(0, n_task, inner_batch_size):
                b1       = min(b0 + inner_batch_size, n_task)
                refs_b   = refs[b0:b1]
                counts_b = counts[b0:b1]
                Bb       = b1 - b0

                probs_b   = agent.choice_probs(refs_b, task)              # [Bb, 2]
                loss_b    = multinomial_nll(probs_b, counts_b) * Bb / n_total
                loss_b.backward()
                total_loss += loss_b.item()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), grad_clip)
        optimizer.step()

        # ---- Evaluation ----------------------------------------------------
        agent.eval()
        train_nll, train_mse_val = evaluate(agent, train_dataset, image_refs, inner_batch_size)
        val_nll,   val_mse_val   = evaluate(agent, val_dataset,   image_refs, inner_batch_size)

        # ESS: sample 64 training images for speed
        ess_val = agent.ess(train_refs_all[:64])

        result.train_nlls.append(train_nll)
        result.train_mses.append(train_mse_val)
        result.val_nlls.append(val_nll)
        result.val_mses.append(val_mse_val)
        result.train_ess.append(ess_val)

        for name, ds in extra_val_datasets.items():
            nll, mse_ = evaluate(agent, ds, image_refs, inner_batch_size)
            result.extra_val_nlls[name].append(nll)
            result.extra_val_mses[name].append(mse_)

        scheduler.step()

        # ---- Progress bar --------------------------------------------------
        pbar.set_postfix(
            train_nll=f"{train_nll:.3f}",
            val_nll  =f"{val_nll:.3f}",
            val_mse  =f"{val_mse_val:.4f}",
            ess      =f"{ess_val:.3f}",
            lr       =f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        for cb in callbacks:
            cb(epoch, val_nll, val_mse_val)

        # ---- Early stopping ------------------------------------------------
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
