from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from rl_engine.config import EngineConfig
from rl_engine.training.runtime import train_from_config


def create_local_tiny_model(path: Path) -> None:
    vocabulary = {
        "<eos>": 0,
        "<unk>": 1,
        "What": 2,
        "is": 3,
        "?": 4,
        "+": 5,
        "-": 6,
        "*": 7,
        "Give": 8,
        "only": 9,
        "the": 10,
        "final": 11,
        "number": 12,
        ".": 13,
        "0": 14,
        "1": 15,
        "2": 16,
        "3": 17,
        "4": 18,
        "5": 19,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    model.save_pretrained(path)


def test_local_transformers_model_runs_complete_training_step(tmp_path: Path) -> None:
    create_local_tiny_model(tmp_path)
    config = EngineConfig.model_validate(
        {
            "experiment": {"name": "offline-smoke", "seed": 5},
            "model": {"name": str(tmp_path), "dtype": "float32", "device": "cpu"},
            "algorithm": {
                "name": "grpo",
                "group_size": 2,
                "kl_coefficient": 0.0,
            },
            "training": {
                "learning_rate": 1e-3,
                "batch_size": 1,
                "max_steps": 1,
                "max_grad_norm": 1.0,
            },
            "rollout": {
                "backend": "transformers",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_completion_tokens": 2,
            },
            "reward": {"name": "arithmetic"},
            "evaluation": {"interval_steps": 1, "num_examples": 2},
            "checkpoint": {"interval_steps": 1},
            "logging": {"backend": "console", "interval_steps": 1},
        }
    )
    records: list[dict[str, object]] = []

    summary = train_from_config(config, metric_sink=records.append)

    assert summary.experiment == "offline-smoke"
    assert summary.global_step == 1
    assert summary.policy_version == 1
    assert summary.generated_tokens >= 2
    assert len(records) == 1
    assert records[0]["step"] == 1
