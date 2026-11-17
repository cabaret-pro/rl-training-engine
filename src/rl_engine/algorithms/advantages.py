"""Advantage estimators shared by policy-gradient algorithms."""

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Normalize each reward against completions sampled for the same prompt.

    The standard deviation uses the population convention (``correction=0``), making the
    reduction explicit and well-defined for small groups. Constant-reward groups receive exactly
    zero advantage. Singleton groups are rejected because they violate GRPO sampling semantics.
    """

    if rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("rewards and group_ids must both have shape [batch]")
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have identical shapes")
    if not rewards.is_floating_point():
        raise TypeError("rewards must use a floating-point dtype")
    if group_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_ids must use an integer dtype")
    if rewards.numel() == 0:
        raise ValueError("cannot compute advantages for an empty batch")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must contain only finite values")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    advantages = torch.empty_like(rewards)
    for group_id in torch.unique(group_ids, sorted=True):
        group_mask = group_ids == group_id
        group_rewards = rewards[group_mask]
        if group_rewards.numel() < 2:
            raise ValueError(f"GRPO group {int(group_id)} contains fewer than two completions")
        centered = group_rewards - group_rewards.mean()
        standard_deviation = group_rewards.std(correction=0)
        if standard_deviation == 0:
            advantages[group_mask] = 0.0
        else:
            advantages[group_mask] = centered / (standard_deviation + epsilon)
    return advantages

