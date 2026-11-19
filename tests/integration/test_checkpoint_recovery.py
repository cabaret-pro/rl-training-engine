import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rl_engine.config import EngineConfig
from rl_engine.rollout.sampler import PromptRecord, PromptSampler
from rl_engine.training.checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint
from rl_engine.training.trainer import TrainerState


def config() -> EngineConfig:
    return EngineConfig.model_validate(
        {
            "experiment": {"name": "checkpoint-test", "seed": 12},
            "model": {"name": "tiny", "dtype": "float32", "device": "cpu"},
            "algorithm": {"name": "grpo", "group_size": 2},
            "training": {"learning_rate": 0.01, "batch_size": 1, "max_steps": 10},
            "rollout": {"backend": "transformers", "max_completion_tokens": 2},
            "reward": {"name": "arithmetic"},
            "evaluation": {"interval_steps": 1, "num_examples": 2},
            "checkpoint": {"interval_steps": 5},
            "logging": {"backend": "console", "interval_steps": 1},
        }
    )


def components() -> tuple[nn.Linear, torch.optim.Adam, PromptSampler, TrainerState]:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prompts = [PromptRecord(str(index), str(index)) for index in range(4)]
    return model, optimizer, PromptSampler(prompts, seed=22), TrainerState()


def train_steps(
    model: nn.Linear,
    optimizer: torch.optim.Adam,
    sampler: PromptSampler,
    state: TrainerState,
    count: int,
) -> None:
    for _ in range(count):
        prompt_value = float(sampler.sample(1)[0].prompt_id)
        random_value = random.random() + float(np.random.random())
        inputs = torch.randn(3) + prompt_value + random_value
        loss = model(inputs).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state.global_step += 1
        state.policy_version += 1


def assert_optimizer_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    for left_values, right_values in zip(
        left_state["state"].values(), right_state["state"].values(), strict=True
    ):
        assert left_values.keys() == right_values.keys()
        for key in left_values:
            if isinstance(left_values[key], torch.Tensor):
                torch.testing.assert_close(left_values[key], right_values[key], rtol=0, atol=0)
            else:
                assert left_values[key] == right_values[key]


def test_resumed_training_exactly_matches_continuous_training(tmp_path: Path) -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    continuous = components()
    train_steps(*continuous, 10)

    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    interrupted = components()
    train_steps(*interrupted, 5)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint-5",
        policy=interrupted[0],
        optimizer=interrupted[1],
        sampler=interrupted[2],
        trainer_state=interrupted[3],
        config=config(),
    )

    torch.manual_seed(999)
    resumed = components()
    load_checkpoint(
        checkpoint,
        policy=resumed[0],
        optimizer=resumed[1],
        sampler=resumed[2],
        trainer_state=resumed[3],
        expected_config=config(),
    )
    train_steps(*resumed, 5)

    for continuous_parameter, resumed_parameter in zip(
        continuous[0].parameters(), resumed[0].parameters(), strict=True
    ):
        torch.testing.assert_close(continuous_parameter, resumed_parameter, rtol=0, atol=0)
    assert_optimizer_equal(continuous[1], resumed[1])
    assert resumed[3] == continuous[3] == TrainerState(global_step=10, policy_version=10)
    assert resumed[2].state_dict()["position"] == continuous[2].state_dict()["position"]


def test_integrity_validation_detects_modified_checkpoint_file(tmp_path: Path) -> None:
    model, optimizer, sampler, state = components()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        policy=model,
        optimizer=optimizer,
        sampler=sampler,
        trainer_state=state,
        config=config(),
    )
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        validate_checkpoint(checkpoint)


def test_checkpoint_refuses_configuration_drift(tmp_path: Path) -> None:
    model, optimizer, sampler, state = components()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        policy=model,
        optimizer=optimizer,
        sampler=sampler,
        trainer_state=state,
        config=config(),
    )
    changed = config().model_copy(
        update={"experiment": config().experiment.model_copy(update={"seed": 99})}
    )

    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(
            checkpoint,
            policy=model,
            optimizer=optimizer,
            sampler=sampler,
            trainer_state=state,
            expected_config=changed,
        )
