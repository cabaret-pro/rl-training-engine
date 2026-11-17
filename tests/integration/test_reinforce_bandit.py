import torch

from rl_engine.algorithms.reinforce import reinforce_loss


def train_contextual_bandit(seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    num_contexts = 4
    logits = torch.zeros((num_contexts, num_contexts), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.08)

    for _ in range(120):
        contexts = torch.randint(num_contexts, (256,), generator=generator)
        distributions = torch.distributions.Categorical(logits=logits[contexts])
        actions = distributions.sample()
        returns = actions.eq(contexts).float()
        selected_logprobs = distributions.log_prob(actions).unsqueeze(1)
        output = reinforce_loss(
            selected_logprobs,
            returns,
            torch.ones_like(selected_logprobs, dtype=torch.bool),
            baseline="batch_mean",
        )

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

    success_probability = logits.softmax(dim=-1).diagonal().mean()
    return float(success_probability.detach())


def test_reinforce_learns_contextual_bandit_across_seeds() -> None:
    success_probabilities = [train_contextual_bandit(seed) for seed in (3, 11, 29)]

    assert min(success_probabilities) > 0.95
