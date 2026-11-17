"""REINFORCE with explicit masking, baselines, and reduction conventions."""

from dataclasses import dataclass
from typing import Literal

import torch

from rl_engine.algorithms.losses import masked_mean

Baseline = Literal["none", "batch_mean"]
Normalization = Literal["token_mean", "sequence_mean"]


@dataclass(frozen=True, slots=True)
class ReinforceOutput:
    loss: torch.Tensor
    mean_return: torch.Tensor
    mean_advantage: torch.Tensor


def compute_reinforce_advantages(
    returns: torch.Tensor,
    baseline: Baseline = "batch_mean",
) -> torch.Tensor:
    """Compute trajectory advantages using an action-independent detached baseline."""

    if returns.ndim != 1:
        raise ValueError(f"returns must have shape [batch], got {tuple(returns.shape)}")
    if not returns.is_floating_point():
        raise TypeError("returns must use a floating-point dtype")
    if not torch.isfinite(returns).all():
        raise ValueError("returns must contain only finite values")
    if baseline == "none":
        return returns
    if baseline == "batch_mean":
        return returns - returns.mean().detach()
    raise ValueError(f"unknown REINFORCE baseline: {baseline}")


def reinforce_loss(
    logprobs: torch.Tensor,
    returns: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    baseline: Baseline = "batch_mean",
    normalization: Normalization = "sequence_mean",
) -> ReinforceOutput:
    """Calculate the negative Monte Carlo policy-gradient objective.

    ``sequence_mean`` sums active token log probabilities for each trajectory and then averages
    trajectories. ``token_mean`` averages all active token contributions across the batch. The
    former gives each completion equal weight; the latter gives each generated token equal weight.
    """

    if logprobs.ndim != 2:
        raise ValueError(f"logprobs must have shape [batch, tokens], got {logprobs.shape}")
    if logprobs.shape != completion_mask.shape:
        raise ValueError("logprobs and completion_mask must have identical shapes")
    if returns.shape != (logprobs.shape[0],):
        raise ValueError(f"returns must have shape ({logprobs.shape[0]},)")
    if not logprobs.is_floating_point():
        raise TypeError("logprobs must use a floating-point dtype")
    if completion_mask.dtype != torch.bool:
        raise TypeError("completion_mask must use torch.bool")
    if not torch.isfinite(logprobs).all():
        raise ValueError("logprobs must contain only finite values")
    if not completion_mask.any(dim=1).all():
        raise ValueError("every trajectory must contain at least one active token")

    advantages = compute_reinforce_advantages(returns, baseline)
    token_objective = logprobs * advantages.unsqueeze(1)

    if normalization == "token_mean":
        loss = -masked_mean(token_objective, completion_mask)
    elif normalization == "sequence_mean":
        sequence_objective = token_objective.masked_fill(~completion_mask, 0.0).sum(dim=1)
        loss = -sequence_objective.mean()
    else:
        raise ValueError(f"unknown REINFORCE loss normalization: {normalization}")

    return ReinforceOutput(
        loss=loss,
        mean_return=returns.mean().detach(),
        mean_advantage=advantages.mean().detach(),
    )

