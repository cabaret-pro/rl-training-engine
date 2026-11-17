import pytest
import torch

from rl_engine.rollout import Trajectory


def make_trajectory(**overrides: object) -> Trajectory:
    fields: dict[str, object] = {
        "prompt_ids": torch.tensor([10, 11]),
        "completion_ids": torch.tensor([20, 21, 0]),
        "attention_mask": torch.tensor([True, True, True, True, False]),
        "completion_mask": torch.tensor([True, True, False]),
        "old_logprobs": torch.tensor([-0.2, -0.3, 0.0]),
        "rewards": torch.tensor(1.0),
        "policy_version": 3,
        "prompt_group_id": 7,
        "reward_metadata": {"correctness": 1.0},
    }
    fields.update(overrides)
    return Trajectory(**fields)  # type: ignore[arg-type]


def test_valid_trajectory_reports_active_tokens() -> None:
    trajectory = make_trajectory()

    assert trajectory.num_completion_tokens == 2


def test_rejects_sequence_level_logprob() -> None:
    with pytest.raises(ValueError, match="old_logprobs must have length 3"):
        make_trajectory(old_logprobs=torch.tensor([-0.5]))


def test_rejects_nonfinite_logprobs() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_trajectory(old_logprobs=torch.tensor([-0.2, torch.nan, 0.0]))


def test_rejects_misaligned_attention_mask() -> None:
    with pytest.raises(ValueError, match="attention_mask must have length 5"):
        make_trajectory(attention_mask=torch.tensor([True, True, True]))


def test_rejects_non_boolean_masks() -> None:
    with pytest.raises(TypeError, match=r"torch\.bool"):
        make_trajectory(completion_mask=torch.tensor([1, 1, 0]))


def test_optional_ppo_fields_are_token_aligned() -> None:
    with pytest.raises(ValueError, match="values must have length 3"):
        make_trajectory(values=torch.tensor([0.1, 0.2]))

