"""Observable single-process rollout-to-optimization lifecycle."""

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

from rl_engine.rewards.base import RewardFunction
from rl_engine.rollout.buffer import ExperienceBuffer, collate_trajectories
from rl_engine.rollout.generator import RolloutBackend, evaluate_completions
from rl_engine.rollout.sampler import PromptSampler
from rl_engine.rollout.trajectory import Trajectory
from rl_engine.training.trainer import GRPOTrainer, UpdateMetrics


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    prompts_per_step: int
    group_size: int
    pad_token_id: int

    def __post_init__(self) -> None:
        if self.prompts_per_step <= 0:
            raise ValueError("prompts_per_step must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be nonnegative")


@dataclass(frozen=True, slots=True)
class CoordinatorMetrics:
    update: UpdateMetrics
    rollout_latency_ms: float
    reward_latency_ms: float
    update_latency_ms: float
    generated_tokens: int
    trajectories: int
    policy_staleness: int


class TrainingCoordinator:
    """Coordinate synchronous GRPO iterations with complete prompt-group boundaries."""

    def __init__(
        self,
        *,
        sampler: PromptSampler,
        rollout_backend: RolloutBackend,
        reward: RewardFunction,
        buffer: ExperienceBuffer,
        trainer: GRPOTrainer,
        config: CoordinatorConfig,
    ) -> None:
        if buffer.max_groups < config.prompts_per_step:
            raise ValueError("experience buffer cannot hold one configured rollout step")
        self.sampler = sampler
        self.rollout_backend = rollout_backend
        self.reward = reward
        self.buffer = buffer
        self.trainer = trainer
        self.config = config

    def run_step(self) -> CoordinatorMetrics:
        """Run one synchronous rollout and update boundary."""

        requests = self.sampler.sample_groups(
            self.config.prompts_per_step,
            self.config.group_size,
        )
        rollout_version = self.trainer.state.policy_version
        started = perf_counter()
        generated = self.rollout_backend.generate(requests, policy_version=rollout_version)
        rollout_latency_ms = (perf_counter() - started) * 1000
        if len(generated) != len(requests):
            raise ValueError("rollout backend returned a different number of completions")

        started = perf_counter()
        trajectories = evaluate_completions(generated, self.reward)
        reward_latency_ms = (perf_counter() - started) * 1000
        groups: dict[int, list[Trajectory]] = defaultdict(list)
        for trajectory in trajectories:
            groups[trajectory.prompt_group_id].append(trajectory)
        if len(groups) != self.config.prompts_per_step:
            raise ValueError("rollout backend did not preserve the configured prompt groups")
        if any(len(group) != self.config.group_size for group in groups.values()):
            raise ValueError("rollout backend returned an incomplete prompt group")
        for group in groups.values():
            self.buffer.add_group(group)

        buffered = self.buffer.pop_groups(
            self.config.prompts_per_step,
            current_policy_version=self.trainer.state.policy_version,
        )
        batch = collate_trajectories(buffered, pad_token_id=self.config.pad_token_id)
        policy_staleness = self.trainer.state.policy_version - rollout_version
        started = perf_counter()
        update_metrics = self.trainer.update(batch)
        update_latency_ms = (perf_counter() - started) * 1000
        generated_tokens = sum(item.num_completion_tokens for item in buffered)

        return CoordinatorMetrics(
            update=update_metrics,
            rollout_latency_ms=rollout_latency_ms,
            reward_latency_ms=reward_latency_ms,
            update_latency_ms=update_latency_ms,
            generated_tokens=generated_tokens,
            trajectories=len(buffered),
            policy_staleness=policy_staleness,
        )

