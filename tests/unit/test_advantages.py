import pytest
import torch

from rl_engine.algorithms.advantages import group_relative_advantages


def test_normalizes_rewards_within_each_prompt_group() -> None:
    rewards = torch.tensor([1.0, 3.0, 10.0, 10.0, 14.0])
    group_ids = torch.tensor([7, 7, 4, 4, 4])

    advantages = group_relative_advantages(rewards, group_ids, epsilon=1e-12)

    for group_id in (4, 7):
        group = advantages[group_ids == group_id]
        assert group.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert group.std(correction=0).item() == pytest.approx(1.0, abs=1e-6)


def test_equal_reward_group_has_exactly_zero_advantages() -> None:
    advantages = group_relative_advantages(
        torch.tensor([2.0, 2.0, 2.0]),
        torch.tensor([42, 42, 42]),
    )

    assert torch.equal(advantages, torch.zeros(3))
    assert torch.isfinite(advantages).all()


def test_preserves_original_batch_order_for_interleaved_groups() -> None:
    advantages = group_relative_advantages(
        torch.tensor([1.0, 20.0, 3.0, 10.0]),
        torch.tensor([0, 1, 0, 1]),
        epsilon=1e-12,
    )

    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 1.0, -1.0]))


def test_rejects_singleton_prompt_group() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        group_relative_advantages(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0, 0, 1]),
        )


def test_rejects_nonfinite_rewards() -> None:
    with pytest.raises(ValueError, match="finite"):
        group_relative_advantages(
            torch.tensor([1.0, torch.nan]),
            torch.tensor([0, 0]),
        )
