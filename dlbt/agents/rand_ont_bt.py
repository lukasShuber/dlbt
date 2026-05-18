"""
RandOntBTAgent: DLBT with a randomly permuted task ontology.

Scientific role
---------------
Tests whether DLBT's performance depends on the *intended semantic*
task-to-latent-state mapping, or whether any arity-matched partition works.

Procedure
---------
For each task t with arity a_t (number of _and_ conjuncts + 1):

  1. Positive-set size  |Z_t^+| = K / 2^{a_t}   (mirrors the true ontology)
  2. Sample Z_t^{+,rand} — a uniformly random subset of {0, …, K-1} of that size.
  3. Set Z_t^{-,rand} = Z \\ Z_t^{+,rand}.
  4. Build random utility vector:
        ΔU_rand(z) = +1   if z ∈ Z_t^{+,rand}
                     -1   otherwise

Partitions are sampled **independently per task** (no global permutation of
latent states — a single permutation could be re-learned).

The resulting agent is architecturally identical to DlbtAgent; only
`_delta_u()` is overridden to look up the pre-computed random vector.

Usage
-----
    rand_du = make_rand_ontology(task_names, seed=42)
    agent   = RandOntBTAgent(rand_delta_u=rand_du, ...)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from dlbt.agents.dlbt import DlbtAgent
from dlbt.constants import K
from dlbt.data.task import Task


# ---------------------------------------------------------------------------
# Ontology builder
# ---------------------------------------------------------------------------

def make_rand_ontology(
    task_names: List[str],
    seed: int,
    k: int = K,
) -> Dict[str, np.ndarray]:
    """
    Build a random arity-matched utility vector for every task.

    Parameters
    ----------
    task_names : list of task name strings (e.g. ["left", "left_and_glossy", ...])
    seed       : RNG seed — use different seeds for different training runs
                 so that the SEM across seeds reflects both training and
                 ontology randomness.
    k          : number of latent states (default: K=16)

    Returns
    -------
    dict mapping each task_name -> ndarray of shape [k] with values in {-1., +1.}
    """
    rng     = np.random.default_rng(seed)
    all_idx = np.arange(k)
    result  = {}
    for name in task_names:
        arity  = name.count("_and_") + 1
        n_pos  = k // (2 ** arity)
        if n_pos < 1:
            n_pos = 1
        pos_idx  = rng.choice(all_idx, size=n_pos, replace=False)
        delta_u  = np.full(k, -1.0, dtype=np.float32)
        delta_u[pos_idx] = 1.0
        result[name] = delta_u
    return result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RandOntBTAgent(DlbtAgent):
    """
    DLBT with pre-computed random utility vectors.

    Architecturally identical to DlbtAgent — same CLIP encoder, same mapper,
    same MC Dirichlet integration, same training objective.  Only the ΔU
    vectors differ: each task gets an arity-matched but semantically random
    partition of the latent state space.

    Parameters
    ----------
    rand_delta_u : dict[str, np.ndarray]
        Mapping from task_name -> delta_u array of shape [K] with {-1, +1}.
        Build with :func:`make_rand_ontology`.
    **kwargs :
        All other arguments forwarded to DlbtAgent.__init__.
    """

    def __init__(
        self,
        rand_delta_u: Dict[str, np.ndarray],
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store as plain tensors (not nn.Parameters — not trained).
        # Moved to device lazily in _delta_u().
        self._rand_du: Dict[str, torch.Tensor] = {
            name: torch.tensor(du, dtype=torch.float32)
            for name, du in rand_delta_u.items()
        }

    # -----------------------------------------------------------------------
    # Override: random utility lookup
    # -----------------------------------------------------------------------

    def _delta_u(self, task: Task) -> torch.Tensor:
        """
        Return the random ΔU vector for *task*, with optional normalisation.

        Falls back silently to the semantic delta_u for any task not in the
        precomputed map (this should not happen in normal use).
        """
        if task.name in self._rand_du:
            du = self._rand_du[task.name].to(self.device)
        else:
            du = torch.tensor(task.delta_u, dtype=torch.float32, device=self.device)

        if self.normalize_utility:
            n_pos = float((du > 0).sum())
            n_neg = float((du < 0).sum())
            du = torch.where(du > 0, du / n_pos, du / n_neg)
        return du
