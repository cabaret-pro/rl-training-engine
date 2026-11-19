import pytest

from rl_engine.config import EngineConfig
from rl_engine.training.runtime import validate_training_runtime


def raw_config() -> dict[str, object]:
    return {
        "experiment": {"name": "test", "seed": 1},
        "model": {"name": "local", "dtype": "float32", "device": "cpu"},
        "algorithm": {"name": "grpo", "group_size": 2},
        "training": {"learning_rate": 1e-3, "batch_size": 1, "max_steps": 1},
        "rollout": {"backend": "transformers", "max_completion_tokens": 2},
        "reward": {"name": "arithmetic"},
        "evaluation": {"interval_steps": 1, "num_examples": 2},
        "checkpoint": {"interval_steps": 1},
        "logging": {"backend": "console", "interval_steps": 1},
    }


def test_runtime_accepts_supported_grpo_configuration() -> None:
    algorithm = validate_training_runtime(EngineConfig.model_validate(raw_config()))

    assert algorithm.name == "grpo"


def test_runtime_rejects_unimplemented_algorithm_before_loading_model() -> None:
    config = raw_config()
    config["algorithm"] = {"name": "ppo"}

    with pytest.raises(ValueError, match="only GRPO"):
        validate_training_runtime(EngineConfig.model_validate(config))


def test_runtime_rejects_unimplemented_logging_backend() -> None:
    config = raw_config()
    config["logging"] = {"backend": "mlflow", "interval_steps": 1}

    with pytest.raises(ValueError, match="console"):
        validate_training_runtime(EngineConfig.model_validate(config))

