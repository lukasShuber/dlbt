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
    best_epoch:   int = 0
    best_val_mse: float = float("inf")


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
) -> TrainResult:
    """
    Train a DlbtAgent on behavioural choice data.

    Args:
        agent:          the DlbtAgent to train (modified in-place).
        train_dataset:  training observations.
        val_dataset:    validation observations (used for early stopping).
        image_refs:     uid -> ImageRef lookup for all images.
        n_epochs:       maximum number of full-dataset passes.
        lr:             Adam learning rate.
        patience:       early-stopping patience (epochs without val improvement).
        callbacks:      list of callables called each epoch as
                        callback(epoch, val_nll, val_mse).
                        Use for logging, TensorBoard, etc.

    Returns:
        TrainResult with metrics and best-weight agent.
    """
    result    = TrainResult(agent=agent)
    optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    # Pre-cache CLIP features for frozen encoder (no-op if finetuned)
    if agent.freeze_encoder:
        all_refs = list(image_refs.values())
        agent.precompute_features(all_refs)

    # Baseline evaluation (epoch 0)
    train_nll0, train_mse0 = evaluate(agent, train_dataset, image_refs)
    val_nll0,   val_mse0   = evaluate(agent, val_dataset,   image_refs)
    result.train_nlls.append(train_nll0)
    result.train_mses.append(train_mse0)
    result.val_nlls.append(val_nll0)
    result.val_mses.append(val_mse0)
    result.best_val_mse = val_mse0

    # Save initial weights
    best_state = copy.deepcopy(agent.state_dict())
    no_improve = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="training", unit="epoch")
    for epoch in pbar:

        # ---- Forward + backward pass over all tasks -----------------------
        agent.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=agent.device)

        for task_name, group in train_dataset.iter_tasks():
            task   = TASKS[task_name]
            refs   = [image_refs[uid] for uid in group["uid"]]
            counts = torch.tensor(
                group[["count_0", "count_1"]].values,
                dtype=torch.float32,
                device=agent.device,
            )
            probs = agent.choice_probs(refs, task)   # [B, 2]
            total_loss = total_loss + multinomial_nll(probs, counts) * len(refs)

        # Normalise by total number of observations across tasks
        total_loss = total_loss / len(train_dataset)
        total_loss.backward()
        optimizer.step()

        # ---- Evaluation ---------------------------------------------------
        agent.eval()
        train_nll, train_mse_val = evaluate(agent, train_dataset, image_refs)
        val_nll,   val_mse_val   = evaluate(agent, val_dataset,   image_refs)

        result.train_nlls.append(train_nll)
        result.train_mses.append(train_mse_val)
        result.val_nlls.append(val_nll)
        result.val_mses.append(val_mse_val)

        # ---- LR schedule --------------------------------------------------
        scheduler.step()

        # ---- Progress bar -------------------------------------------------
        pbar.set_postfix(
            train_nll=f"{train_nll:.3f}",
            val_nll=f"{val_nll:.3f}",
            val_mse=f"{val_mse_val:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

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

    # Restore best weights
    agent.load_state_dict(best_state)
    return result
