"""Validated data carried from rollout generation into policy optimization."""

from dataclasses import dataclass, field

import torch


def _require_vector(name: str, tensor: torch.Tensor, length: int | None = None) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {tensor.shape[0]}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One generated completion and the behavior-policy state needed to train on it.

    ``old_logprobs`` contains one value per generated token, never a sequence-level sum.
    ``completion_mask`` is aligned to those values; inactive positions are padding and must not
    contribute to an objective. ``attention_mask`` covers the concatenated prompt and completion.
    """

    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    policy_version: int
    prompt_group_id: int
    reward_metadata: dict[str, float] = field(default_factory=dict)
    values: torch.Tensor | None = None
    token_rewards: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_vector("prompt_ids", self.prompt_ids)
        _require_vector("completion_ids", self.completion_ids)
        completion_length = self.completion_ids.shape[0]
        _require_vector(
            "attention_mask",
            self.attention_mask,
            self.prompt_ids.shape[0] + completion_length,
        )
        _require_vector("completion_mask", self.completion_mask, completion_length)
        _require_vector("old_logprobs", self.old_logprobs, completion_length)

        if self.prompt_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("prompt_ids must use an integer dtype")
        if self.completion_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("completion_ids must use an integer dtype")
        if self.attention_mask.dtype != torch.bool or self.completion_mask.dtype != torch.bool:
            raise TypeError("attention_mask and completion_mask must use torch.bool")
        if not self.old_logprobs.is_floating_point():
            raise TypeError("old_logprobs must use a floating-point dtype")
        _require_finite("old_logprobs", self.old_logprobs)
        _require_finite("rewards", self.rewards)

        if self.rewards.numel() == 0:
            raise ValueError("rewards must contain at least one value")
        if self.policy_version < 0:
            raise ValueError("policy_version must be nonnegative")
        if self.prompt_group_id < 0:
            raise ValueError("prompt_group_id must be nonnegative")
        if completion_length == 0 or not self.completion_mask.any():
            raise ValueError("a trajectory must contain at least one active completion token")

        for name, tensor in (("values", self.values), ("token_rewards", self.token_rewards)):
            if tensor is not None:
                _require_vector(name, tensor, completion_length)
                _require_finite(name, tensor)

    @property
    def num_completion_tokens(self) -> int:
        """Return the number of non-padding generated tokens."""

        return int(self.completion_mask.sum().item())

