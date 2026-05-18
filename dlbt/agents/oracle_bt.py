"""
OracleBTAgent: Fixed-belief agent using the true latent state from image metadata.

Belief representation
---------------------
    α_x[k]  = concentration   if k == image_ref.latent_state
    α_x[k'] = background      otherwise

This is a Dirichlet prior peaked at the true latent state bin (derived from
the continuous rendering parameters via threshold binarisation).  With
moderate concentration (e.g. 5.0) the beliefs are soft — not a hard one-hot,
which would collapse to the deterministic 05_determ_beliefs ablation.

Forward pass
------------
Inherits DetBTAgent's deterministic mean:

    μ_x     = α_x / Σ_k α_xk           (Dirichlet mean)
    logit_t = μ_x · ΔU_t
    p̃_xt   = σ(logit_t)

No CLIP encoder or mapper is invoked — `get_alpha()` is fully determined by
`image_ref.latent_state` (an integer in [0, K-1]) and the two scalar
hyperparameters `concentration` and `background`.

No training is needed.  This agent is used only in evaluation mode.

Scientific role
---------------
Compared against trained DLBT in run1/06_fixed_beliefs:
  "Does learning from behavioural data beat a soft prior based solely on
   knowledge of each image's latent state bin?"
"""

from __future__ import annotations

from typing import List

import torch

from dlbt.agents.detbt import DetBTAgent
from dlbt.constants import K
from dlbt.data.image_ref import ImageRef


class OracleBTAgent(DetBTAgent):
    """
    Fixed beliefs peaked at the true latent state (from image_ref.latent_state).

    Parameters
    ----------
    concentration : float
        Dirichlet mass placed on the true latent state dimension.
        Default 5.0 — moderate peak; tune via config.CONCENTRATION.
    background : float
        Dirichlet mass on all other K-1 dimensions.  Default 0.1.
    """

    def __init__(
        self,
        concentration: float = 5.0,
        background: float = 0.1,
        device: torch.device = torch.device("cpu"),
        normalize_utility: bool = False,
    ):
        # n_mc_samples=1 (inherited from DetBTAgent) → corrected_mse returns raw MSE
        super().__init__(
            freeze_encoder    = True,
            device            = device,
            normalize_utility = normalize_utility,
        )
        self.concentration = concentration
        self.background    = background

    # ------------------------------------------------------------------
    # Core override: beliefs from metadata, not from CLIP features
    # ------------------------------------------------------------------

    def get_alpha(self, image_refs: List[ImageRef]) -> torch.Tensor:
        """
        Return peaked Dirichlet α for each image.

        Does NOT call _encode() — the CLIP cache is never accessed.
        The mapper (Linear + Softplus) is also bypassed entirely.

        Returns:
            [B, K] tensor — α_x[latent_state(x)] = concentration,
                            α_x[k']              = background  (k' ≠ latent_state(x))
        """
        B      = len(image_refs)
        states = torch.tensor(
            [r.latent_state for r in image_refs],
            dtype  = torch.long,
            device = self.device,
        )                                                           # [B]
        alpha  = torch.full((B, K), self.background, device=self.device)
        alpha.scatter_(1, states.unsqueeze(1), self.concentration)  # peak at true state
        return alpha                                                # [B, K]
