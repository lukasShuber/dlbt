"""
LbtAgent: Latent Belief Tomography agent (no deep encoder).

A learnable lookup table of Dirichlet concentration parameters α,
one row per image (indexed by UID).  No encoder, no mapper.

The choice-probability computation is *identical* to DlbtAgent:
  - Training path: straight-through argmax over MC Dirichlet samples.
  - Eval path:     clean hard MC average.

Use cases:
  - Sanity-check simulations where the true α is known.
  - Oracle fitting: directly optimise α from behavioral counts.
  - Any setting where image features are not available or not needed.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from dlbt.constants import K
from dlbt.data.image_ref import ImageRef
from dlbt.data.task import Task


class LbtAgent(nn.Module):
    """
    Bare LBT agent with a learnable per-image α table.

    Args:
        uid_list:     ordered list of image UIDs to be modelled.
        n_mc_samples: Monte Carlo samples for choice_probs().
        device:       torch device.
        init_alpha:   initial value for all α entries (default 1.0).
    """

    def __init__(
        self,
        uid_list:           List[str],
        n_mc_samples:       int = 1000,
        device:             torch.device = torch.device("cpu"),
        init_alpha:         float = 1.0,
        normalize_utility:  bool = False,
    ):
        super().__init__()
        self.uid_list          = list(uid_list)
        self.uid_to_idx        = {uid: i for i, uid in enumerate(self.uid_list)}
        self.n_mc_samples      = n_mc_samples
        self.device            = device
        self.normalize_utility = normalize_utility

        n = len(self.uid_list)
        # Parameterise as log_alpha so softplus always gives positive α.
        # softplus(x0) = init_alpha  =>  x0 = log(exp(init_alpha) - 1)
        import math
        x0 = math.log(math.exp(init_alpha) - 1.0) if init_alpha > 0 else 0.0
        self.log_alpha = nn.Parameter(
            torch.full((n, K), x0, device=device)
        )

    # -----------------------------------------------------------------------
    # Core: α lookup
    # -----------------------------------------------------------------------

    def get_alpha(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """
        Return Dirichlet concentration parameters α for a batch of images.
        Shape: [B, K], all entries strictly positive.
        """
        idxs = torch.tensor(
            [self.uid_to_idx[r.uid] for r in image_refs],
            device=self.device,
        )
        return F.softplus(self.log_alpha[idxs]).clamp(min=1e-6)   # [B, K]

    # -----------------------------------------------------------------------
    # Utility vector
    # -----------------------------------------------------------------------

    def _delta_u(self, task: Task) -> torch.Tensor:
        """
        Return the utility vector for `task`.

        normalize_utility=False (default):
            ΔU[k] = +1  if k ∈ Z+,  −1  if k ∈ Z−   (original SEU rule)

        normalize_utility=True:
            ΔU[k] = +1/|Z+|  if k ∈ Z+,  −1/|Z−|  if k ∈ Z−
            Equivalent to comparing posterior/prior ratios — unbiased under
            a uniform prior regardless of task arity.
        """
        du = torch.tensor(task.delta_u, dtype=torch.float32, device=self.device)
        if self.normalize_utility:
            n_pos = float((du > 0).sum())
            n_neg = float((du < 0).sum())
            du = torch.where(du > 0, du / n_pos, du / n_neg)
        return du

    # -----------------------------------------------------------------------
    # Choice probabilities — mirrors DlbtAgent exactly
    # -----------------------------------------------------------------------

    def choice_probs(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        if self.training:
            return self._choice_probs_train(image_refs, task)
        return self._choice_probs_eval(image_refs, task)

    def _choice_probs_train(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Straight-through argmax — identical to DlbtAgent._choice_probs_train."""
        N       = self.n_mc_samples
        alpha   = self.get_alpha(image_refs).clamp(min=0.1)           # [B, K]
        delta_u = self._delta_u(task)                                  # [K]

        b      = Dirichlet(alpha).rsample((N,))                        # [N, B, K]
        logit  = torch.einsum("nbk,k->nb", b, delta_u)                # [N, B]

        logits_2d  = torch.stack([-logit, logit], dim=-1)             # [N, B, 2]
        probs_soft = F.softmax(logits_2d, dim=-1)
        hard       = F.one_hot(logits_2d.argmax(-1), 2).float()
        st         = (hard - probs_soft).detach() + probs_soft
        return st.mean(dim=0)                                          # [B, 2]

    @torch.no_grad()
    def _choice_probs_eval(
        self,
        image_refs: List[ImageRef],
        task: Task,
    ) -> torch.Tensor:
        """Clean hard MC average — identical to DlbtAgent._choice_probs_eval."""
        N       = self.n_mc_samples
        alpha   = self.get_alpha(image_refs).clamp(min=0.1)           # [B, K]
        delta_u = self._delta_u(task)                                  # [K]

        b       = Dirichlet(alpha).sample((N,))                        # [N, B, K]
        logit   = torch.einsum("nbk,k->nb", b, delta_u)               # [N, B]
        hard    = (logit > 0).float()                                  # [N, B]
        p_right = hard.mean(dim=0)                                     # [B]
        return torch.stack([1 - p_right, p_right], dim=-1)            # [B, 2]

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        return list(self.parameters())
