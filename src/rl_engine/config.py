"""Validated experiment configuration loaded before expensive resources are allocated."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled and obsolete configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    seed: int = Field(default=42, ge=0)


class ModelConfig(StrictModel):
    name: str = Field(min_length=1)
    revision: str | None = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class ReinforceConfig(StrictModel):
    name: Literal["reinforce"]
    baseline: Literal["none", "batch_mean"] = "batch_mean"
    loss_normalization: Literal["token_mean", "sequence_mean"] = "sequence_mean"


class GRPOConfig(StrictModel):
    name: Literal["grpo"]
    group_size: PositiveInt = 8
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    kl_coefficient: float = Field(default=0.0, ge=0)
    advantage_epsilon: PositiveFloat = 1e-6
    loss_normalization: Literal["token_mean", "sequence_mean"] = "token_mean"
    optimization_epochs: PositiveInt = 1


class PPOConfig(StrictModel):
    name: Literal["ppo"]
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    value_coefficient: float = Field(default=0.5, ge=0)
    entropy_coefficient: float = Field(default=0.0, ge=0)
    gamma: float = Field(default=1.0, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)


AlgorithmConfig = Annotated[
    ReinforceConfig | GRPOConfig | PPOConfig,
    Field(discriminator="name"),
]


class TrainingConfig(StrictModel):
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt = 1
    max_steps: PositiveInt
    max_grad_norm: PositiveFloat = 1.0


class RolloutConfig(StrictModel):
    backend: Literal["transformers", "vllm"] = "transformers"
    temperature: PositiveFloat = 1.0
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    max_completion_tokens: PositiveInt


class RewardConfig(StrictModel):
    name: Literal["arithmetic", "scheduling"]
    success_reward: float = 1.0
    failure_reward: float = 0.0


class EvaluationConfig(StrictModel):
    interval_steps: PositiveInt
    num_examples: PositiveInt


class CheckpointConfig(StrictModel):
    interval_steps: PositiveInt
    keep_last: PositiveInt = 3


class LoggingConfig(StrictModel):
    backend: Literal["console", "mlflow", "wandb"] = "console"
    interval_steps: PositiveInt = 10


class EngineConfig(StrictModel):
    experiment: ExperimentConfig
    model: ModelConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig
    rollout: RolloutConfig
    reward: RewardConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def validate_compatibility(self) -> "EngineConfig":
        if self.rollout.backend == "vllm" and self.model.dtype == "float32":
            raise ValueError("vLLM rollout requires model.dtype to be float16 or bfloat16")
        if isinstance(self.algorithm, GRPOConfig) and self.algorithm.group_size < 2:
            raise ValueError("GRPO requires algorithm.group_size >= 2")
        if (
            isinstance(self.algorithm, GRPOConfig)
            and self.training.max_steps % self.algorithm.optimization_epochs != 0
        ):
            raise ValueError("training.max_steps must be divisible by GRPO optimization_epochs")
        return self


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return EngineConfig.model_validate(raw_config)

