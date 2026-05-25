"""
OneHotBTAgent: Certain-belief ablation of DLBT.

Training is identical to DetBTAgent (NLL through the Dirichlet mean —
fully differentiable).  At *evaluation* time the belief collapses to a hard
one-hot at the argmax of the mapper output α:

    k* = argmax_k α_k        (most-likely latent state)
    μ  = one_hot(k*)         [K]   — certain belief
    logit_t = μ · ΔU_t
    p̃_xt   = σ(logit_t)

Scientific role (run1/05_ablations):
    "Does modelling graded perceptual uncertainty over the latent state (as in
     DLBT / DetBT) add value beyond hard certain beliefs at the argmax?"

Because training is identical to DetBT, any difference in probe performance
is attributable solely to the belief representation at evaluation time
(mean vs. one-hot), not to a difference in what the mapper has learned.
"""

from __future__ import annotations

from typing import List

import torch

from dlbt.agents.detbt import DetBTAgent
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class OneHotBTAgent(DetBTAgent):
    """
    Certain-belief agent.  Training: Dirichlet mean (same as DetBT).
    Evaluation: hard one-hot at argmax(α).

    No additional parameters beyond DetBTAgent.
    """

    # -----------------------------------------------------------------------
    # Eval override: argmax one-hot instead of Dirichlet mean
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _choice_probs_eval(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """
        Certain-belief choice-probability computation (evaluation only).

          1. α  = mapper(encoder(x))              [B, K]
          2. k* = argmax_k α_k                    [B]   (same as argmax of mean)
          3. μ  = one_hot(k*)                     [B, K]
          4. logit = μ · ΔU_t                     [B]
          5. p̃ = σ(logit)                        [B]

        Returns:
            [B, 2] tensor of (P(left), P(right)).
        """
        alpha   = self.get_alpha(image_refs)                      # [B, K]
        k_star  = alpha.argmax(dim=-1)                            # [B]
        one_hot = torch.zeros_like(alpha)                         # [B, K]
        one_hot.scatter_(1, k_star.unsqueeze(1), 1.0)             # one-hot at k*
        delta_u = self._delta_u(task)                             # [K]
        logit   = torch.einsum("bk,k->b", one_hot, delta_u)      # [B]
        p_right = torch.sigmoid(logit)                            # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)       # [B, 2]

    # _choice_probs_train is inherited from DetBTAgent:
    #   → uses _forward_det (Dirichlet mean, fully differentiable)
    #   → identical training signal to DetBT
