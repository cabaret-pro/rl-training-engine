from types import SimpleNamespace

import torch
from torch import nn

from rl_engine.models.policy import CausalLMPolicy
from rl_engine.rewards.arithmetic import ArithmeticReward
from rl_engine.rollout.buffer import ExperienceBuffer
from rl_engine.rollout.generator import GeneratedCompletion
from rl_engine.rollout.sampler import PromptRecord, PromptSampler, RolloutRequest
from rl_engine.training.coordinator import CoordinatorConfig, TrainingCoordinator
from rl_engine.training.trainer import GRPOTrainer, GRPOTrainerConfig


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(10))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class DeterministicBackend:
    def generate(
        self, requests: list[RolloutRequest], *, policy_version: int
    ) -> list[GeneratedCompletion]:
        generated = []
        uniform_logprob = -torch.log(torch.tensor(10.0))
        for request in requests:
            correct = request.completion_index == 1
            token = int(request.prompt.metadata["answer"]) if correct else 9
            generated.append(
                GeneratedCompletion(
                    request=request,
                    prompt_ids=torch.tensor([1]),
                    completion_ids=torch.tensor([token]),
                    attention_mask=torch.tensor([True, True]),
                    completion_mask=torch.tensor([True]),
                    behavior_logprobs=uniform_logprob.reshape(1),
                    completion_text=str(token),
                    policy_version=policy_version,
                )
            )
        return generated


def test_coordinator_runs_complete_synchronous_grpo_iteration() -> None:
    prompts = [
        PromptRecord("one", "What is 1 + 1?", {"answer": 2}),
        PromptRecord("two", "What is 2 + 2?", {"answer": 4}),
    ]
    model = TinyLM()
    policy = CausalLMPolicy(model)
    trainer = GRPOTrainer(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        GRPOTrainerConfig(max_grad_norm=10.0),
    )
    coordinator = TrainingCoordinator(
        sampler=PromptSampler(prompts, seed=7),
        rollout_backend=DeterministicBackend(),
        reward=ArithmeticReward(),
        buffer=ExperienceBuffer(max_groups=2),
        trainer=trainer,
        config=CoordinatorConfig(prompts_per_step=2, group_size=2, pad_token_id=0),
    )

    metrics = coordinator.run_step()

    assert trainer.state.global_step == 1
    assert trainer.state.policy_version == 1
    assert metrics.trajectories == 4
    assert metrics.generated_tokens == 4
    assert metrics.update.reward_mean == 0.5
    assert metrics.rollout_latency_ms >= 0
    assert metrics.reward_latency_ms >= 0
    assert metrics.update_latency_ms >= 0
    assert len(coordinator.buffer) == 0
