from rl_engine.config import EngineConfig
from rl_engine.observability.system_metrics import collect_run_metadata


def test_run_metadata_records_reproducibility_context() -> None:
    config = EngineConfig.model_validate(
        {
            "experiment": {"name": "metadata-test", "seed": 17},
            "model": {
                "name": "local-model",
                "revision": "fixed-revision",
                "dtype": "float32",
                "device": "cpu",
            },
            "algorithm": {"name": "grpo", "group_size": 2},
            "training": {"learning_rate": 1e-3, "batch_size": 1, "max_steps": 1},
            "rollout": {"backend": "transformers", "max_completion_tokens": 2},
            "reward": {"name": "arithmetic"},
            "evaluation": {"interval_steps": 1, "num_examples": 2},
            "checkpoint": {"interval_steps": 1},
            "logging": {"backend": "console", "interval_steps": 1},
        }
    )

    metadata = collect_run_metadata(config)

    assert metadata["kind"] == "run_metadata"
    assert metadata["seed"] == 17
    assert metadata["model"]["revision"] == "fixed-revision"  # type: ignore[index]
    assert metadata["model"]["resolved_device"] == "cpu"  # type: ignore[index]
    assert metadata["software"]["torch"]  # type: ignore[index]
    assert metadata["hardware"]["machine"]  # type: ignore[index]
