"""
Evaluation metrics for binary-choice models.

All functions operate on:
    pred_probs: Tensor[N, 2]  — predicted choice probabilities
    counts:     Tensor[N, 2]  — observed choice counts (integers as floats)
"""

from __future__ import annotations

import torch


def multinomial_nll(
    pred_probs: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-trial negative log-likelihood under the multinomial model.

        NLL = -1/T * sum_j sum_a counts[j,a] * log pred_probs[j,a]

    where T = total number of trials across all observations.

    The multinomial normalisation constants (factorials) are omitted because
    they do not affect optimisation.

    Args:
        pred_probs: [N, 2] predicted probabilities, must sum to 1 along dim=1.
        counts:     [N, 2] observed counts (float32).

    Returns:
        scalar Tensor (differentiable).
    """
    log_p  = torch.log(pred_probs.clamp(min=1e-8))       # [N, 2]
    T      = counts.sum()                                  # scalar
    return -(counts * log_p).sum() / T


def mse(
    pred_probs: torch.Tensor,
    counts: torch.Tensor,
) -> float:
    """
    Mean squared error between predicted P(right) and empirical frequency of right.

    Args:
        pred_probs: [N, 2]
        counts:     [N, 2]

    Returns:
        Python float.
    """
    totals    = counts.sum(dim=1, keepdim=True).clamp(min=1)
    emp_freqs = counts / totals                            # [N, 2]
    return torch.mean((pred_probs - emp_freqs) ** 2).item()


def corrected_mse(
    pred_probs: torch.Tensor,
    counts: torch.Tensor,
    n_mc_samples: int = 1000,
) -> float:
    """
    Bias-corrected MSE: raw MSE minus the MC-sampling variance of the predictor.

    Subtracts E[Var_{MC}(p̂)] to remove the noise contribution from finite
    Monte Carlo sampling, giving a fairer comparison between models using
    different numbers of samples.

    Args:
        pred_probs:   [N, 2]
        counts:       [N, 2]
        n_mc_samples: number of MC samples used to produce pred_probs.

    Returns:
        Python float.
    """
    totals    = counts.sum(dim=1, keepdim=True).clamp(min=1)
    emp_freqs = counts / totals

    raw_mse  = torch.mean((pred_probs - emp_freqs) ** 2)
    mc_var   = pred_probs * (1 - pred_probs) / max(n_mc_samples - 1, 1)
    return (raw_mse - mc_var.mean()).item()
