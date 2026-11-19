from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_engine.models.policy import CausalLMPolicy
from rl_engine.rollout.buffer import collate_trajectories
from rl_engine.rollout.trajectory import Trajectory
from rl_engine.training.trainer import GRPOTrainer, GRPOTrainerConfig, TrainerState


class PositionIndependentLM(nn.Module):
    def __init__(self, vocabulary_size: int = 5) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def sample(action: int, reward: float, *, version: int = 0) -> Trajectory:
    old_logprob = torch.tensor([-torch.log(torch.tensor(5.0))])
    return Trajectory(
        prompt_ids=torch.tensor([4]),
        completion_ids=torch.tensor([action]),
        attention_mask=torch.tensor([True, True]),
        completion_mask=torch.tensor([True]),
        old_logprobs=old_logprob,
        rewards=torch.tensor(reward),
        policy_version=version,
        prompt_group_id=0,
    )


def test_grpo_update_improves_rewarded_action_and_advances_state() -> None:
    model = PositionIndependentLM()
    policy = CausalLMPolicy(model)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    trainer = GRPOTrainer(policy, optimizer, GRPOTrainerConfig(max_grad_norm=10.0))
    batch = collate_trajectories([sample(1, 0.0), sample(2, 1.0)], pad_token_id=0)

    metrics = trainer.update(batch)

    assert model.logits[2] > 0
    assert model.logits[1] < 0
    assert trainer.state == TrainerState(global_step=1, policy_version=1)
    assert metrics.reward_mean == pytest.approx(0.5)
    assert metrics.mean_ratio == pytest.approx(1.0)
    assert metrics.grad_norm > 0


def test_multiple_epochs_account_for_each_optimizer_step() -> None:
    model = PositionIndependentLM()
    policy = CausalLMPolicy(model)
    trainer = GRPOTrainer(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.01),
        GRPOTrainerConfig(optimization_epochs=3, max_grad_norm=10.0),
    )
    batch = collate_trajectories([sample(1, 0.0), sample(2, 1.0)], pad_token_id=0)

    metrics = trainer.update(batch)

    assert trainer.state.global_step == 3
    assert trainer.state.policy_version == 3
    assert metrics.optimization_epochs == 3


def test_rejects_stale_rollouts_before_changing_parameters() -> None:
    model = PositionIndependentLM()
    initial = model.logits.detach().clone()
    policy = CausalLMPolicy(model)
    trainer = GRPOTrainer(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        GRPOTrainerConfig(max_policy_staleness=0),
        state=TrainerState(global_step=4, policy_version=2),
    )
    batch = collate_trajectories(
        [sample(1, 0.0, version=1), sample(2, 1.0, version=1)], pad_token_id=0
    )

    with pytest.raises(ValueError, match="staleness"):
        trainer.update(batch)

    torch.testing.assert_close(model.logits, initial)
    assert trainer.state == TrainerState(global_step=4, policy_version=2)


def test_reference_policy_is_frozen() -> None:
    model = PositionIndependentLM()
    reference_model = PositionIndependentLM()
    policy = CausalLMPolicy(model)
    reference = CausalLMPolicy(reference_model)
    trainer = GRPOTrainer(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        GRPOTrainerConfig(kl_coefficient=0.1, max_grad_norm=10.0),
        reference_policy=reference,
    )
    batch = collate_trajectories([sample(1, 0.0), sample(2, 1.0)], pad_token_id=0)

    trainer.update(batch)

    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())
