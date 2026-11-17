"""Group Relative Policy Optimization tensor objective and diagnostics."""

from dataclasses import dataclass
from typing import Literal

import torch

from rl_engine.algorithms.losses import masked_mean

Normalization = Literal["token_mean", "sequence_mean"]


@dataclass(frozen=True, slots=True)
class GRPOOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor
    behavior_kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor


def importance_ratios(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Return token-level policy ratios ``pi_current / pi_behavior``."""

    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current_logprobs and old_logprobs must have identical shapes")
    if not current_logprobs.is_floating_point() or not old_logprobs.is_floating_point():
        raise TypeError("log probabilities must use floating-point dtypes")
    if not torch.isfinite(current_logprobs).all() or not torch.isfinite(old_logprobs).all():
        raise ValueError("log probabilities must contain only finite values")
    return torch.exp(current_logprobs - old_logprobs)


def nonnegative_kl_estimate(
    current_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Return the per-token k3 estimator of ``KL(pi_current || pi_reference)``.

    For ``delta = log(pi_reference) - log(pi_current)``, the estimator is
    ``exp(delta) - delta - 1``. It is nonnegative per sample and is zero when policies match.
    """

    if current_logprobs.shape != reference_logprobs.shape:
        raise ValueError("current and reference log probabilities must have identical shapes")
    delta = reference_logprobs - current_logprobs
    return torch.exp(delta) - delta - 1.0


def _reduce_tokens(
    values: torch.Tensor,
    mask: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    if normalization == "token_mean":
        return masked_mean(values, mask)
    if normalization == "sequence_mean":
        token_counts = mask.sum(dim=1)
        if (token_counts == 0).any():
            raise ValueError("every sequence must contain at least one active token")
        per_sequence = values.masked_fill(~mask, 0.0).sum(dim=1) / token_counts
        return per_sequence.mean()
    raise ValueError(f"unknown GRPO loss normalization: {normalization}")


def grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: torch.Tensor | None = None,
    kl_coefficient: float = 0.0,
    normalization: Normalization = "token_mean",
) -> GRPOOutput:
    """Calculate the clipped GRPO objective over generated tokens.

    ``token_mean`` gives every active token equal weight. ``sequence_mean`` first averages the
    active tokens in each completion and then averages completions, giving each completion equal
    weight regardless of its length.
    """

    if current_logprobs.ndim != 2:
        raise ValueError("log probabilities must have shape [batch, tokens]")
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current and old log probabilities must have identical shapes")
    if completion_mask.shape != current_logprobs.shape:
        raise ValueError("completion_mask and log probabilities must have identical shapes")
    if completion_mask.dtype != torch.bool:
        raise TypeError("completion_mask must use torch.bool")
    if advantages.shape != (current_logprobs.shape[0],):
        raise ValueError(f"advantages must have shape ({current_logprobs.shape[0]},)")
    if not advantages.is_floating_point() or not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite floating-point values")
    if not completion_mask.any(dim=1).all():
        raise ValueError("every sequence must contain at least one active token")
    if not torch.isfinite(current_logprobs).all() or not torch.isfinite(old_logprobs).all():
        raise ValueError("log probabilities must contain only finite values")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be between zero and one")
    if kl_coefficient < 0:
        raise ValueError("kl_coefficient must be nonnegative")
    if kl_coefficient > 0 and reference_logprobs is None:
        raise ValueError("reference_logprobs are required when kl_coefficient is positive")

    # Neutralize padding before exponentials. Masking an already-overflowed value afterward can
    # still produce NaN gradients through expressions such as zero multiplied by infinity.
    safe_current_logprobs = current_logprobs.masked_fill(~completion_mask, 0.0)
    safe_old_logprobs = old_logprobs.masked_fill(~completion_mask, 0.0)
    ratios = importance_ratios(safe_current_logprobs, safe_old_logprobs)
    expanded_advantages = advantages.unsqueeze(1)
    unclipped = ratios * expanded_advantages
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(unclipped, clipped_ratios * expanded_advantages)
    policy_loss = -_reduce_tokens(surrogate, completion_mask, normalization)

    if reference_logprobs is None:
        reference_kl = current_logprobs.new_zeros(())
    else:
        if not torch.isfinite(reference_logprobs).all():
            raise ValueError("reference_logprobs must contain only finite values")
        safe_reference_logprobs = reference_logprobs.masked_fill(~completion_mask, 0.0)
        token_reference_kl = nonnegative_kl_estimate(
            safe_current_logprobs, safe_reference_logprobs
        )
        reference_kl = _reduce_tokens(token_reference_kl, completion_mask, normalization)

    token_behavior_kl = nonnegative_kl_estimate(safe_current_logprobs, safe_old_logprobs)
    behavior_kl = masked_mean(token_behavior_kl, completion_mask)
    clipped = (ratios < 1.0 - clip_epsilon) | (ratios > 1.0 + clip_epsilon)

    return GRPOOutput(
        loss=policy_loss + kl_coefficient * reference_kl,
        policy_loss=policy_loss.detach(),
        reference_kl=reference_kl.detach(),
        behavior_kl=behavior_kl.detach(),
        clip_fraction=masked_mean(clipped.to(ratios.dtype), completion_mask).detach(),
        mean_ratio=masked_mean(ratios, completion_mask).detach(),
    )

