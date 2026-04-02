"""
Fitting procedure for SldaAgent.

SldaAgent is a standard nn.Module (mapper Linear(1024, K) + log_temperature)
and is trained with the same gradient-based loop as DlbtAgent.  This module
provides a thin wrapper around train_dlbt() with SLDA-appropriate defaults.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from dlbt.agents.slda import SldaAgent
from dlbt.data.dataset import BehavioralDataset
from dlbt.data.image_ref import ImageRef
from dlbt.training.train_dlbt import train_dlbt, TrainResult


def fit_slda(
    agent: SldaAgent,
    train_dataset: BehavioralDataset,
    val_dataset:   BehavioralDataset,
    image_refs:    Dict[str, ImageRef],
    n_epochs:      int   = 300,
    lr:            float = 1e-3,
    patience:      int   = 30,
) -> TrainResult:
    """
    Train SldaAgent on behavioural choice data.

    Delegates to train_dlbt(), which is fully compatible with SldaAgent:
      - freeze_encoder = True  → precompute_features() is called once
      - n_mc_samples   = None  → corrected_mse() skips MC variance correction
      - trainable_parameters() → mapper weights + log_temperature

    Args:
        agent:         the SldaAgent (modified in-place).
        train_dataset: training observations.
        val_dataset:   validation observations (used for early stopping).
        image_refs:    uid -> ImageRef lookup for all images.
        n_epochs:      maximum training epochs.
        lr:            Adam learning rate.
        patience:      early-stopping patience (epochs without val improvement).

    Returns:
        TrainResult with loss curves and best-weight agent.
    """
    return train_dlbt(
        agent,
        train_dataset,
        val_dataset,
        image_refs,
        n_epochs=n_epochs,
        lr=lr,
        patience=patience,
    )
