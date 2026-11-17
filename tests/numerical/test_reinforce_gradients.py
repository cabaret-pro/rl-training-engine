import torch

from rl_engine.algorithms.reinforce import reinforce_loss


def test_reinforce_matches_finite_difference_gradient() -> None:
    logprobs = torch.tensor(
        [[-0.2, -0.7, -1.1], [-0.4, -0.9, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    returns = torch.tensor([1.25, -0.75], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    loss = reinforce_loss(logprobs, returns, mask, baseline="none").loss
    analytical = torch.autograd.grad(loss, logprobs)[0]

    epsilon = 1e-6
    numerical = torch.zeros_like(logprobs)
    with torch.no_grad():
        for index in range(logprobs.numel()):
            positive = logprobs.detach().clone()
            negative = logprobs.detach().clone()
            positive.view(-1)[index] += epsilon
            negative.view(-1)[index] -= epsilon
            positive_loss = reinforce_loss(positive, returns, mask, baseline="none").loss
            negative_loss = reinforce_loss(negative, returns, mask, baseline="none").loss
            numerical.view(-1)[index] = (positive_loss - negative_loss) / (2 * epsilon)

    torch.testing.assert_close(analytical, numerical, rtol=1e-7, atol=1e-9)

