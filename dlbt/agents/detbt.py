"""
DetBTAgent: Deterministic Belief Tomography agent (DLBT ablation).

Replaces the Monte Carlo Dirichlet integration of DlbtAgent with a
deterministic forward pass through the Dirichlet mean:

    μ_x     = α_x / Σ_k α_xk          (point mass at Dirichlet mean)
    logit_t = μ_x · ΔU_t
    p̃_xt   = σ(logit_t)

Identical architecture (same CLIP backbone, same mapper, same training
objective). Only the belief integration step changes. Because
softmax → dot-product → sigmoid is fully differentiable, no
straight-through estimator is needed during training.

This tests whether modelling uncertainty over beliefs is necessary, or
whether a deterministic belief representation at the Dirichlet mean is
sufficient to capture graded human choice frequencies.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from dlbt.agents.dlbt import DlbtAgent
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class DetBTAgent(DlbtAgent):
    """
    Deterministic BT agent: belief collapses to the Dirichlet mean μ = α / Σα.

    Inherits all infrastructure from DlbtAgent (encoder, mapper, feature
    caching, get_alpha, _delta_u). Only the choice-probability computation
    is overridden.

    n_mc_samples is fixed at 1 so that corrected_mse() in the training loop
    skips the MC-variance correction (correct behaviour for a deterministic
    predictor).
    """

    def __init__(
        self,
        freeze_encoder: bool = True,
        device: torch.device = torch.device("cpu"),
        mapper_hidden: Optional[int] = None,
        feature_dim: int = 1024,
        normalize_utility: bool = False,
    ):
        super().__init__(
            freeze_encoder=freeze_encoder,
            n_mc_samples=1,          # no MC → correction skipped in corrected_mse
            device=device,
            mapper_hidden=mapper_hidden,
            feature_dim=feature_dim,
            normalize_utility=normalize_utility,
        )

    # -----------------------------------------------------------------------
    # Core deterministic forward pass
    # -----------------------------------------------------------------------

    def _forward_det(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        Deterministic choice-probability computation.

          1. α = mapper(encoder(x))          [B, K]
          2. μ = α / Σ_k α_k                 [B, K]   Dirichlet mean
          3. logit = μ · ΔU_t                [B]
          4. p̃ = σ(logit)                   [B]

        Returns:
            [B, 2] tensor of (P(left), P(right)).
        """
        alpha   = self.get_alpha(image_refs)                     # [B, K]
        mu      = alpha / alpha.sum(dim=-1, keepdim=True)        # [B, K]
        delta_u = self._delta_u(task)                            # [K]
        logit   = torch.einsum("bk,k->b", mu, delta_u)          # [B]
        p_right = torch.sigmoid(logit)                           # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)      # [B, 2]

    def _choice_probs_train(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Training path: fully differentiable, no ST estimator needed."""
        return self._forward_det(image_refs, task)

    def _choice_probs_eval(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Eval path: same computation, gradients disabled by caller."""
        return self._forward_det(image_refs, task)
