from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_engine.config import EngineConfig, GRPOConfig, load_config

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("filename", ["reinforce.yaml", "grpo.yaml", "ppo.yaml"])
def test_example_configs_are_valid(filename: str) -> None:
    config = load_config(ROOT / "configs" / filename)

    assert config.training.max_steps > 0


def test_grpo_config_uses_discriminated_model() -> None:
    config = load_config(ROOT / "configs" / "grpo.yaml")

    assert isinstance(config.algorithm, GRPOConfig)
    assert config.algorithm.group_size == 8


def test_unknown_fields_are_rejected() -> None:
    raw = load_config(ROOT / "configs" / "grpo.yaml").model_dump()
    raw["training"]["learnnig_rate"] = raw["training"].pop("learning_rate")

    with pytest.raises(ValidationError, match="learning_rate"):
        EngineConfig.model_validate(raw)


def test_grpo_rejects_single_completion_groups() -> None:
    raw = load_config(ROOT / "configs" / "grpo.yaml").model_dump()
    raw["algorithm"]["group_size"] = 1

    with pytest.raises(ValidationError, match="group_size >= 2"):
        EngineConfig.model_validate(raw)


def test_vllm_rejects_float32_models() -> None:
    raw = load_config(ROOT / "configs" / "grpo.yaml").model_dump()
    raw["rollout"]["backend"] = "vllm"
    raw["model"]["dtype"] = "float32"

    with pytest.raises(ValidationError, match="vLLM rollout"):
        EngineConfig.model_validate(raw)
