import math

import pytest
import torch

from rl_engine.algorithms.grpo import (
    grpo_loss,
    importance_ratios,
    nonnegative_kl_estimate,
)


def test_matching_policies_have_unit_ratios_and_zero_kl() -> None:
    logprobs = torch.tensor([[-0.2, -1.7], [-0.8, -0.4]])

    ratios = importance_ratios(logprobs, logprobs)
    kl = nonnegative_kl_estimate(logprobs, logprobs)

    assert torch.equal(ratios, torch.ones_like(ratios))
    assert torch.equal(kl, torch.zeros_like(kl))


def test_clipped_objective_matches_independent_scalar_calculation() -> None:
    old = torch.zeros((2, 2), dtype=torch.float64)
    current = torch.tensor(
        [[math.log(1.5), math.log(0.9)], [math.log(0.5), math.log(1.1)]],
        dtype=torch.float64,
    )
    advantages = torch.tensor([2.0, -3.0], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, False]])

    output = grpo_loss(current, old, advantages, mask, clip_epsilon=0.2)

    expected_surrogates = torch.tensor([1.2 * 2.0, 0.8 * -3.0], dtype=torch.float64)
    assert output.policy_loss.item() == pytest.approx(-expected_surrogates.mean().item())
    assert output.clip_fraction.item() == 1.0


@pytest.mark.parametrize(
    ("ratio", "advantage"),
    [(1.5, 1.0), (0.5, -1.0)],
)
def test_clipping_eliminates_gradient_in_clipped_direction(
    ratio: float, advantage: float
) -> None:
    current = torch.tensor([[math.log(ratio)]], dtype=torch.float64, requires_grad=True)
    output = grpo_loss(
        current,
        torch.zeros_like(current),
        torch.tensor([advantage], dtype=torch.float64),
        torch.ones_like(current, dtype=torch.bool),
    )

    output.loss.backward()

    assert current.grad is not None
    assert current.grad.item() == pytest.approx(0.0)


def test_padding_has_no_gradient_or_diagnostic_weight() -> None:
    current = torch.tensor([[-0.1, 100.0]], requires_grad=True)
    old = torch.tensor([[-0.1, -100.0]])
    mask = torch.tensor([[True, False]])

    output = grpo_loss(current, old, torch.tensor([1.0]), mask)
    output.loss.backward()

    assert current.grad is not None
    assert current.grad[0, 1].item() == 0.0
    assert output.mean_ratio.item() == pytest.approx(1.0)
    assert output.clip_fraction.item() == 0.0


def test_reference_kl_is_nonnegative_and_contributes_to_total_loss() -> None:
    current = torch.tensor([[-0.2, -0.5]])
    old = current.clone()
    reference = torch.tensor([[-0.7, -0.1]])
    mask = torch.ones_like(current, dtype=torch.bool)

    output = grpo_loss(
        current,
        old,
        torch.tensor([0.0]),
        mask,
        reference_logprobs=reference,
        kl_coefficient=0.3,
    )

    assert output.reference_kl.item() >= 0
    assert output.loss.item() == pytest.approx(0.3 * output.reference_kl.item())


def test_sequence_mean_gives_completions_equal_weight() -> None:
    current = torch.zeros((2, 2))
    old = torch.zeros_like(current)
    mask = torch.tensor([[True, True], [True, False]])
    advantages = torch.tensor([1.0, 3.0])

    token_output = grpo_loss(current, old, advantages, mask, normalization="token_mean")
    sequence_output = grpo_loss(current, old, advantages, mask, normalization="sequence_mean")

    assert token_output.loss.item() == pytest.approx(-5 / 3)
    assert sequence_output.loss.item() == pytest.approx(-2.0)

