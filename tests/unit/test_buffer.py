import pytest
import torch

from rl_engine.rollout.buffer import ExperienceBuffer, collate_trajectories
from rl_engine.rollout.trajectory import Trajectory


def trajectory(
    group_id: int,
    *,
    version: int = 2,
    prompt: tuple[int, ...] = (10, 11),
    completion: tuple[int, ...] = (20, 21),
    active_tokens: int | None = None,
) -> Trajectory:
    if active_tokens is None:
        active_tokens = len(completion)
    completion_mask = torch.arange(len(completion)) < active_tokens
    return Trajectory(
        prompt_ids=torch.tensor(prompt),
        completion_ids=torch.tensor(completion),
        attention_mask=torch.cat([torch.ones(len(prompt), dtype=torch.bool), completion_mask]),
        completion_mask=completion_mask,
        old_logprobs=torch.arange(len(completion), dtype=torch.float32) * -0.1,
        rewards=torch.tensor(float(group_id)),
        policy_version=version,
        prompt_group_id=group_id,
        reward_metadata={"correctness": float(group_id)},
    )


def test_collation_right_pads_without_activating_padding() -> None:
    short = trajectory(0, prompt=(4,), completion=(5,), active_tokens=1)
    long = trajectory(0, prompt=(6, 7), completion=(8, 9, 0), active_tokens=2)

    batch = collate_trajectories([short, long], pad_token_id=99)

    assert batch.input_ids.tolist() == [[4, 5, 99, 99, 99], [6, 7, 8, 9, 0]]
    assert batch.attention_mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, False],
    ]
    assert batch.completion_mask.tolist() == [[True, False, False], [True, True, False]]
    assert batch.old_logprobs[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert batch.prompt_lengths.tolist() == [1, 2]
    assert batch.reward_metadata == ({"correctness": 0.0}, {"correctness": 0.0})


def test_buffer_preserves_complete_fifo_groups() -> None:
    buffer = ExperienceBuffer(max_groups=3)
    buffer.add_group([trajectory(10), trajectory(10)])
    buffer.add_group([trajectory(11), trajectory(11)])

    popped = buffer.pop_groups(1, current_policy_version=2)

    assert [item.prompt_group_id for item in popped] == [10, 10]
    assert len(buffer) == 1


def test_buffer_applies_backpressure_at_group_boundary() -> None:
    buffer = ExperienceBuffer(max_groups=1)
    buffer.add_group([trajectory(0), trajectory(0)])

    with pytest.raises(BufferError, match="full"):
        buffer.add_group([trajectory(1), trajectory(1)])


def test_buffer_rejects_split_or_mixed_groups() -> None:
    buffer = ExperienceBuffer(max_groups=2)

    with pytest.raises(ValueError, match="at least two"):
        buffer.add_group([trajectory(0)])
    with pytest.raises(ValueError, match="prompt_group_id"):
        buffer.add_group([trajectory(0), trajectory(1)])
    with pytest.raises(ValueError, match="policy_version"):
        buffer.add_group([trajectory(0, version=1), trajectory(0, version=2)])


def test_stale_pop_is_rejected_without_consuming_group() -> None:
    buffer = ExperienceBuffer(max_groups=2, max_policy_staleness=1)
    buffer.add_group([trajectory(0, version=3), trajectory(0, version=3)])

    with pytest.raises(ValueError, match="staleness 2"):
        buffer.pop_groups(1, current_policy_version=5)

    assert len(buffer) == 1


def test_allowed_staleness_is_inclusive() -> None:
    buffer = ExperienceBuffer(max_groups=1, max_policy_staleness=2)
    buffer.add_group([trajectory(0, version=3), trajectory(0, version=3)])

    assert len(buffer.pop_groups(1, current_policy_version=5)) == 2
