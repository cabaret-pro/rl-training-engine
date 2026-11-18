"""Deterministic prompt sampling with checkpointable state and GRPO grouping."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    prompt: PromptRecord
    prompt_group_id: int
    completion_index: int


class PromptSampler:
    """Samples shuffled prompts and preserves enough state for exact recovery."""

    def __init__(self, prompts: Sequence[PromptRecord], *, seed: int) -> None:
        if not prompts:
            raise ValueError("prompt sampler requires at least one prompt")
        prompt_ids = [prompt.prompt_id for prompt in prompts]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("prompt IDs must be unique")
        if seed < 0:
            raise ValueError("seed must be nonnegative")

        self._prompts = tuple(prompts)
        self._generator = torch.Generator().manual_seed(seed)
        self._order = self._new_order()
        self._position = 0
        self._epoch = 0
        self._next_group_id = 0

    def _new_order(self) -> list[int]:
        return torch.randperm(len(self._prompts), generator=self._generator).tolist()

    @property
    def epoch(self) -> int:
        return self._epoch

    def sample(self, count: int) -> list[PromptRecord]:
        """Return distinct prompts, advancing to a new shuffled epoch as needed."""

        if not 0 < count <= len(self._prompts):
            raise ValueError("sample count must be between one and the dataset size")

        selected: list[PromptRecord] = []
        while len(selected) < count:
            remaining = len(self._order) - self._position
            take = min(count - len(selected), remaining)
            indices = self._order[self._position : self._position + take]
            selected.extend(self._prompts[index] for index in indices)
            self._position += take
            if self._position == len(self._order):
                self._order = self._new_order()
                self._position = 0
                self._epoch += 1
        return selected

    def sample_groups(self, num_prompts: int, group_size: int) -> list[RolloutRequest]:
        """Sample prompts and expand each into a contiguous GRPO completion group."""

        if group_size < 2:
            raise ValueError("GRPO group_size must be at least two")
        requests: list[RolloutRequest] = []
        for prompt in self.sample(num_prompts):
            group_id = self._next_group_id
            self._next_group_id += 1
            requests.extend(
                RolloutRequest(prompt, group_id, completion_index)
                for completion_index in range(group_size)
            )
        return requests

    def state_dict(self) -> dict[str, object]:
        """Return an isolated snapshot suitable for checkpoint serialization."""

        return {
            "order": list(self._order),
            "position": self._position,
            "epoch": self._epoch,
            "next_group_id": self._next_group_id,
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a validated sampler snapshot."""

        required = {"order", "position", "epoch", "next_group_id", "generator_state"}
        if set(state) != required:
            raise ValueError(f"sampler state fields must be exactly {sorted(required)}")
        order = state["order"]
        if not isinstance(order, list) or sorted(order) != list(range(len(self._prompts))):
            raise ValueError("sampler order must be a permutation of prompt indices")
        position = state["position"]
        epoch = state["epoch"]
        next_group_id = state["next_group_id"]
        generator_state = state["generator_state"]
        if not isinstance(position, int) or not 0 <= position < len(self._prompts):
            raise ValueError("sampler position is out of range")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be nonnegative")
        if not isinstance(next_group_id, int) or next_group_id < 0:
            raise ValueError("next_group_id must be nonnegative")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("generator_state must be a tensor")

        self._order = list(order)
        self._position = position
        self._epoch = epoch
        self._next_group_id = next_group_id
        self._generator.set_state(generator_state)

