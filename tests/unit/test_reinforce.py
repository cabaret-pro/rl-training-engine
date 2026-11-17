import pytest
import torch

from rl_engine.algorithms.reinforce import (
    compute_reinforce_advantages,
    reinforce_loss,
)


def test_batch_mean_advantages_are_centered() -> None:
    returns = torch.tensor([1.0, 2.0, 6.0])

    advantages = compute_reinforce_advantages(returns)

    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-7)


def test_positive_return_increases_selected_action_probability() -> None:
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    selected_logprob = logits.log_softmax(dim=0)[1].reshape(1, 1)

    output = reinforce_loss(
        selected_logprob,
        torch.tensor([1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        baseline="none",
    )
    output.loss.backward()

    assert logits.grad is not None
    assert logits.grad[1] < 0
    assert logits.grad[0] > 0


def test_padding_tokens_have_zero_gradient() -> None:
    logprobs = torch.tensor(
        [[-0.1, -0.2, -100.0], [-0.3, -100.0, -100.0]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    output = reinforce_loss(
        logprobs,
        torch.tensor([1.0, -0.5]),
        mask,
        baseline="none",
    )
    output.loss.backward()

    assert logprobs.grad is not None
    assert torch.equal(logprobs.grad[~mask], torch.zeros(3))
    assert torch.count_nonzero(logprobs.grad[mask]) == 3


def test_rejects_trajectory_without_active_tokens() -> None:
    with pytest.raises(ValueError, match="every trajectory"):
        reinforce_loss(
            torch.zeros((2, 2)),
            torch.ones(2),
            torch.tensor([[True, False], [False, False]]),
        )


def test_reduction_conventions_are_explicit() -> None:
    logprobs = torch.tensor([[-1.0, -1.0], [-1.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    returns = torch.tensor([1.0, 1.0])

    sequence = reinforce_loss(
        logprobs, returns, mask, baseline="none", normalization="sequence_mean"
    )
    token = reinforce_loss(
        logprobs, returns, mask, baseline="none", normalization="token_mean"
    )

    assert sequence.loss.item() == pytest.approx(1.5)
    assert token.loss.item() == pytest.approx(1.0)

