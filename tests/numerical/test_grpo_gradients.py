import torch

from rl_engine.algorithms.grpo import grpo_loss


def test_grpo_matches_finite_difference_away_from_clip_boundaries() -> None:
    current = torch.tensor(
        [[-0.30, -0.70, -1.20], [-0.45, -0.85, -0.20]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old = torch.tensor(
        [[-0.35, -0.75, -1.10], [-0.40, -0.80, -0.25]],
        dtype=torch.float64,
    )
    reference = torch.tensor(
        [[-0.50, -0.60, -1.00], [-0.60, -0.70, -0.40]],
        dtype=torch.float64,
    )
    advantages = torch.tensor([0.8, -1.3], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def objective(values: torch.Tensor) -> torch.Tensor:
        return grpo_loss(
            values,
            old,
            advantages,
            mask,
            reference_logprobs=reference,
            kl_coefficient=0.07,
        ).loss

    analytical = torch.autograd.grad(objective(current), current)[0]
    numerical = torch.zeros_like(current)
    epsilon = 1e-6
    with torch.no_grad():
        for index in range(current.numel()):
            positive = current.detach().clone()
            negative = current.detach().clone()
            positive.view(-1)[index] += epsilon
            negative.view(-1)[index] -= epsilon
            numerical.view(-1)[index] = (objective(positive) - objective(negative)) / (2 * epsilon)

    torch.testing.assert_close(analytical, numerical, rtol=1e-6, atol=1e-8)
