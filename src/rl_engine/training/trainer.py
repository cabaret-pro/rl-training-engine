"""Single-process GRPO optimizer with explicit policy-version validation."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from rl_engine.algorithms.advantages import group_relative_advantages
from rl_engine.algorithms.grpo import grpo_loss
from rl_engine.models.policy import CausalLMPolicy
from rl_engine.rollout.buffer import TrajectoryBatch


@dataclass(frozen=True, slots=True)
class GRPOTrainerConfig:
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0
    advantage_epsilon: float = 1e-6
    normalization: Literal["token_mean", "sequence_mean"] = "token_mean"
    optimization_epochs: int = 1
    max_grad_norm: float = 1.0
    max_policy_staleness: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between zero and one")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must be nonnegative")
        if self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive")
        if self.optimization_epochs <= 0:
            raise ValueError("optimization_epochs must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.max_policy_staleness < 0:
            raise ValueError("max_policy_staleness must be nonnegative")


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    policy_version: int = 0


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    loss: float
    policy_loss: float
    reference_kl: float
    behavior_kl: float
    clip_fraction: float
    mean_ratio: float
    grad_norm: float
    reward_mean: float
    reward_std: float
    optimization_epochs: int


class GRPOTrainer:
    """Apply one or more GRPO optimizer steps to a synchronous rollout batch."""

    def __init__(
        self,
        policy: CausalLMPolicy,
        optimizer: torch.optim.Optimizer,
        config: GRPOTrainerConfig,
        *,
        reference_policy: CausalLMPolicy | None = None,
        state: TrainerState | None = None,
    ) -> None:
        if config.kl_coefficient > 0 and reference_policy is None:
            raise ValueError("reference_policy is required when kl_coefficient is positive")
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.reference_policy = reference_policy
        self.state = state if state is not None else TrainerState()
        if self.reference_policy is not None:
            self.reference_policy.eval()
            for parameter in self.reference_policy.parameters():
                parameter.requires_grad_(False)

    def _validate_policy_versions(self, batch: TrajectoryBatch) -> None:
        versions = torch.unique(batch.policy_versions).tolist()
        if len(versions) != 1:
            raise ValueError("a synchronous update requires one rollout policy version")
        rollout_version = int(versions[0])
        staleness = self.state.policy_version - rollout_version
        if staleness < 0:
            raise ValueError("rollout policy version is newer than trainer policy")
        if staleness > self.config.max_policy_staleness:
            raise ValueError(
                f"rollout policy staleness {staleness} exceeds maximum "
                f"{self.config.max_policy_staleness}"
            )

    def update(self, batch: TrajectoryBatch) -> UpdateMetrics:
        """Optimize on a rollout batch and return detached numerical diagnostics."""

        self._validate_policy_versions(batch)
        advantages = group_relative_advantages(
            batch.rewards,
            batch.prompt_group_ids,
            epsilon=self.config.advantage_epsilon,
        )
        reference_logprobs: torch.Tensor | None = None
        if self.reference_policy is not None:
            with torch.no_grad():
                reference_logprobs = self.reference_policy(batch)

        epoch_outputs: list[tuple[float, float, float, float, float, float, float]] = []
        for _ in range(self.config.optimization_epochs):
            current_logprobs = self.policy(batch)
            output = grpo_loss(
                current_logprobs,
                batch.old_logprobs,
                advantages,
                batch.completion_mask,
                clip_epsilon=self.config.clip_epsilon,
                reference_logprobs=reference_logprobs,
                kl_coefficient=self.config.kl_coefficient,
                normalization=self.config.normalization,
            )
            if not torch.isfinite(output.loss):
                raise FloatingPointError("GRPO loss is not finite")

            self.optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            try:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("policy gradient norm is not finite") from error
            self.optimizer.step()
            self.state.global_step += 1
            self.state.policy_version += 1
            epoch_outputs.append(
                (
                    float(output.loss.detach()),
                    float(output.policy_loss),
                    float(output.reference_kl),
                    float(output.behavior_kl),
                    float(output.clip_fraction),
                    float(output.mean_ratio),
                    float(grad_norm),
                )
            )

        means = tuple(sum(values) / len(values) for values in zip(*epoch_outputs, strict=True))
        return UpdateMetrics(
            loss=means[0],
            policy_loss=means[1],
            reference_kl=means[2],
            behavior_kl=means[3],
            clip_fraction=means[4],
            mean_ratio=means[5],
            grad_norm=means[6],
            reward_mean=float(batch.rewards.mean()),
            reward_std=float(batch.rewards.std(correction=0)),
            optimization_epochs=self.config.optimization_epochs,
        )

