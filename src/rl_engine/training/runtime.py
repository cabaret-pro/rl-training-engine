"""Build and execute a reproducible single-process arithmetic GRPO run."""

import copy
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rl_engine.config import EngineConfig, GRPOConfig
from rl_engine.evaluation.arithmetic import ArithmeticTaskConfig, generate_arithmetic_prompts
from rl_engine.evaluation.evaluator import EvaluationMetrics, LanguageModelEvaluator
from rl_engine.models.loading import load_causal_lm
from rl_engine.models.policy import CausalLMPolicy
from rl_engine.rewards.arithmetic import ArithmeticReward
from rl_engine.rollout.buffer import ExperienceBuffer
from rl_engine.rollout.generator import SamplingConfig, TransformersRolloutBackend
from rl_engine.rollout.sampler import PromptSampler
from rl_engine.training.checkpoint import load_checkpoint, save_checkpoint
from rl_engine.training.coordinator import (
    CoordinatorConfig,
    CoordinatorMetrics,
    TrainingCoordinator,
)
from rl_engine.training.trainer import GRPOTrainer, GRPOTrainerConfig, TrainerState

MetricSink = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    experiment: str
    global_step: int
    policy_version: int
    generated_tokens: int
    final_reward_mean: float
    initial_evaluation_reward: float
    final_evaluation_reward: float
    initial_pass_rate: float
    final_pass_rate: float


def seed_everything(seed: int) -> None:
    """Seed local-process randomness used by model initialization and token sampling."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_training_runtime(config: EngineConfig) -> GRPOConfig:
    """Reject unsupported runtime combinations before loading model weights."""

    if not isinstance(config.algorithm, GRPOConfig):
        raise ValueError("the language-model runtime currently supports only GRPO")
    if config.rollout.backend != "transformers":
        raise ValueError("the current runtime supports only the transformers rollout backend")
    if config.reward.name != "arithmetic":
        raise ValueError("the current runtime supports only the arithmetic reward")
    if config.logging.backend != "console":
        raise ValueError("the current runtime supports only console metric logging")
    return config.algorithm


def console_metric_sink(metrics: dict[str, object]) -> None:
    print(json.dumps(metrics, sort_keys=True), flush=True)


def _metric_record(metrics: CoordinatorMetrics, state: TrainerState) -> dict[str, object]:
    return {
        "kind": "training",
        "step": state.global_step,
        "policy_version": state.policy_version,
        "train": asdict(metrics.update),
        "system": {
            "rollout_latency_ms": metrics.rollout_latency_ms,
            "reward_latency_ms": metrics.reward_latency_ms,
            "update_latency_ms": metrics.update_latency_ms,
            "generated_tokens": metrics.generated_tokens,
            "trajectories": metrics.trajectories,
            "policy_staleness": metrics.policy_staleness,
        },
    }


def _evaluation_record(metrics: EvaluationMetrics, state: TrainerState) -> dict[str, object]:
    return {
        "kind": "evaluation",
        "step": state.global_step,
        "policy_version": state.policy_version,
        "evaluation": asdict(metrics),
    }


def train_from_config(
    config: EngineConfig,
    *,
    metric_sink: MetricSink = console_metric_sink,
    checkpoint_directory: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainingRunSummary:
    """Run arithmetic GRPO until ``training.max_steps`` optimizer updates complete."""

    algorithm = validate_training_runtime(config)
    seed_everything(config.experiment.seed)
    loaded = load_causal_lm(config.model)
    reference_model = copy.deepcopy(loaded.model) if algorithm.kl_coefficient > 0 else None
    policy = CausalLMPolicy(loaded.model)
    reference_policy = CausalLMPolicy(reference_model) if reference_model is not None else None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.training.learning_rate)
    trainer = GRPOTrainer(
        policy,
        optimizer,
        GRPOTrainerConfig(
            clip_epsilon=algorithm.clip_epsilon,
            kl_coefficient=algorithm.kl_coefficient,
            advantage_epsilon=algorithm.advantage_epsilon,
            normalization=algorithm.loss_normalization,
            optimization_epochs=algorithm.optimization_epochs,
            max_grad_norm=config.training.max_grad_norm,
            max_policy_staleness=0,
        ),
        reference_policy=reference_policy,
    )
    prompt_count = max(
        config.training.batch_size * 4,
        config.evaluation.num_examples,
    )
    prompts = generate_arithmetic_prompts(
        ArithmeticTaskConfig(
            count=prompt_count,
            seed=config.experiment.seed,
            split="train",
        )
    )
    sampler = PromptSampler(prompts, seed=config.experiment.seed)
    if resume_checkpoint is not None:
        load_checkpoint(
            resume_checkpoint,
            policy=policy,
            optimizer=optimizer,
            sampler=sampler,
            trainer_state=trainer.state,
            expected_config=config,
        )
    backend = TransformersRolloutBackend(
        loaded.model,
        loaded.tokenizer,
        SamplingConfig(
            max_completion_tokens=config.rollout.max_completion_tokens,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            top_k=config.rollout.top_k,
        ),
    )
    coordinator = TrainingCoordinator(
        sampler=sampler,
        rollout_backend=backend,
        reward=ArithmeticReward(
            success_reward=config.reward.success_reward,
            failure_reward=config.reward.failure_reward,
        ),
        buffer=ExperienceBuffer(max_groups=config.training.batch_size),
        trainer=trainer,
        config=CoordinatorConfig(
            prompts_per_step=config.training.batch_size,
            group_size=algorithm.group_size,
            pad_token_id=loaded.tokenizer.pad_token_id,
        ),
    )
    evaluation_prompts = generate_arithmetic_prompts(
        ArithmeticTaskConfig(
            count=config.evaluation.num_examples,
            seed=config.experiment.seed + 1,
            split="evaluation",
        )
    )
    evaluator = LanguageModelEvaluator(
        loaded.model,
        loaded.tokenizer,
        ArithmeticReward(
            success_reward=config.reward.success_reward,
            failure_reward=config.reward.failure_reward,
        ),
        max_completion_tokens=config.rollout.max_completion_tokens,
    )
    initial_evaluation = evaluator.evaluate(evaluation_prompts)
    metric_sink(_evaluation_record(initial_evaluation, trainer.state))

    generated_tokens = 0
    final_reward_mean = 0.0
    final_evaluation = initial_evaluation
    last_evaluation_step = trainer.state.global_step
    while trainer.state.global_step < config.training.max_steps:
        metrics = coordinator.run_step()
        generated_tokens += metrics.generated_tokens
        final_reward_mean = metrics.update.reward_mean
        if trainer.state.global_step % config.logging.interval_steps == 0:
            metric_sink(_metric_record(metrics, trainer.state))
        if trainer.state.global_step % config.evaluation.interval_steps == 0:
            final_evaluation = evaluator.evaluate(evaluation_prompts)
            last_evaluation_step = trainer.state.global_step
            metric_sink(_evaluation_record(final_evaluation, trainer.state))
        if (
            checkpoint_directory is not None
            and trainer.state.global_step % config.checkpoint.interval_steps == 0
        ):
            save_checkpoint(
                Path(checkpoint_directory) / f"checkpoint-{trainer.state.global_step}",
                policy=policy,
                optimizer=optimizer,
                sampler=sampler,
                trainer_state=trainer.state,
                config=config,
                tokenizer=loaded.tokenizer,
            )

    if last_evaluation_step != trainer.state.global_step:
        final_evaluation = evaluator.evaluate(evaluation_prompts)
        metric_sink(_evaluation_record(final_evaluation, trainer.state))

    return TrainingRunSummary(
        experiment=config.experiment.name,
        global_step=trainer.state.global_step,
        policy_version=trainer.state.policy_version,
        generated_tokens=generated_tokens,
        final_reward_mean=final_reward_mean,
        initial_evaluation_reward=initial_evaluation.mean_reward,
        final_evaluation_reward=final_evaluation.mean_reward,
        initial_pass_rate=initial_evaluation.pass_rate,
        final_pass_rate=final_evaluation.pass_rate,
    )

