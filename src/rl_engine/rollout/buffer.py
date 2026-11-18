"""Bounded, group-aware rollout storage and training-batch collation."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rl_engine.rollout.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Right-padded tensors derived from a collection of validated trajectories."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    policy_versions: torch.Tensor
    prompt_group_ids: torch.Tensor
    reward_metadata: tuple[dict[str, float], ...]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]


def collate_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    pad_token_id: int,
) -> TrajectoryBatch:
    """Collate trajectories without allowing padding to affect policy objectives."""

    if not trajectories:
        raise ValueError("cannot collate an empty trajectory collection")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be nonnegative")
    if any(trajectory.rewards.numel() != 1 for trajectory in trajectories):
        raise ValueError("training collation requires one scalar reward per trajectory")

    batch_size = len(trajectories)
    max_completion_length = max(item.completion_ids.shape[0] for item in trajectories)
    max_sequence_length = max(
        item.prompt_ids.shape[0] + item.completion_ids.shape[0] for item in trajectories
    )
    device = trajectories[0].prompt_ids.device
    token_dtype = trajectories[0].prompt_ids.dtype
    logprob_dtype = trajectories[0].old_logprobs.dtype

    input_ids = torch.full(
        (batch_size, max_sequence_length),
        pad_token_id,
        dtype=token_dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_sequence_length), dtype=torch.bool, device=device
    )
    completion_ids = torch.full(
        (batch_size, max_completion_length),
        pad_token_id,
        dtype=token_dtype,
        device=device,
    )
    completion_mask = torch.zeros(
        (batch_size, max_completion_length), dtype=torch.bool, device=device
    )
    old_logprobs = torch.zeros(
        (batch_size, max_completion_length), dtype=logprob_dtype, device=device
    )

    for row, trajectory in enumerate(trajectories):
        if trajectory.prompt_ids.device != device or trajectory.completion_ids.device != device:
            raise ValueError("all trajectory token tensors must be on the same device")
        if trajectory.prompt_ids.dtype != token_dtype:
            raise TypeError("all trajectory token tensors must use the same dtype")
        if trajectory.old_logprobs.dtype != logprob_dtype:
            raise TypeError("all old_logprobs tensors must use the same dtype")

        prompt_length = trajectory.prompt_ids.shape[0]
        completion_length = trajectory.completion_ids.shape[0]
        sequence_length = prompt_length + completion_length
        input_ids[row, :prompt_length] = trajectory.prompt_ids
        input_ids[row, prompt_length:sequence_length] = trajectory.completion_ids
        attention_mask[row, :sequence_length] = trajectory.attention_mask
        completion_ids[row, :completion_length] = trajectory.completion_ids
        completion_mask[row, :completion_length] = trajectory.completion_mask
        old_logprobs[row, :completion_length] = trajectory.old_logprobs

    return TrajectoryBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor(
            [item.prompt_ids.shape[0] for item in trajectories],
            dtype=torch.int64,
            device=device,
        ),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_logprobs=old_logprobs,
        rewards=torch.stack([item.rewards.reshape(()) for item in trajectories]),
        policy_versions=torch.tensor(
            [item.policy_version for item in trajectories], dtype=torch.int64, device=device
        ),
        prompt_group_ids=torch.tensor(
            [item.prompt_group_id for item in trajectories], dtype=torch.int64, device=device
        ),
        reward_metadata=tuple(dict(item.reward_metadata) for item in trajectories),
    )


class ExperienceBuffer:
    """A bounded FIFO that preserves complete prompt groups and enforces policy freshness."""

    def __init__(self, *, max_groups: int, max_policy_staleness: int = 0) -> None:
        if max_groups <= 0:
            raise ValueError("max_groups must be positive")
        if max_policy_staleness < 0:
            raise ValueError("max_policy_staleness must be nonnegative")
        self._groups: deque[tuple[Trajectory, ...]] = deque()
        self.max_groups = max_groups
        self.max_policy_staleness = max_policy_staleness

    def __len__(self) -> int:
        return len(self._groups)

    def add_group(self, trajectories: Sequence[Trajectory]) -> None:
        """Add a complete rollout group or apply backpressure when full."""

        if len(self._groups) >= self.max_groups:
            raise BufferError("experience buffer is full")
        if len(trajectories) < 2:
            raise ValueError("a GRPO experience group requires at least two trajectories")
        group_ids = {item.prompt_group_id for item in trajectories}
        if len(group_ids) != 1:
            raise ValueError("all trajectories in a group must share prompt_group_id")
        policy_versions = {item.policy_version for item in trajectories}
        if len(policy_versions) != 1:
            raise ValueError("all trajectories in a group must share policy_version")
        self._groups.append(tuple(trajectories))

    def pop_groups(
        self,
        count: int,
        *,
        current_policy_version: int,
    ) -> tuple[Trajectory, ...]:
        """Remove complete FIFO groups after atomically validating policy staleness."""

        if not 0 < count <= len(self._groups):
            raise ValueError("count must be between one and the number of buffered groups")
        selected = tuple(list(self._groups)[:count])
        for group in selected:
            rollout_version = group[0].policy_version
            staleness = current_policy_version - rollout_version
            if staleness < 0:
                raise ValueError("rollout policy version is newer than the current policy")
            if staleness > self.max_policy_staleness:
                raise ValueError(
                    f"rollout policy staleness {staleness} exceeds maximum "
                    f"{self.max_policy_staleness}"
                )

        popped = [self._groups.popleft() for _ in range(count)]
        return tuple(trajectory for group in popped for trajectory in group)

