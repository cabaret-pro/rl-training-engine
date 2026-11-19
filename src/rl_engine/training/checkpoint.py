"""Atomic, integrity-checked training checkpoints and exact RNG restoration."""

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from rl_engine.config import EngineConfig
from rl_engine.rollout.sampler import PromptSampler
from rl_engine.training.trainer import TrainerState

FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    cuda_state = state["cuda"]
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)


def _manifest(root: Path, config: EngineConfig) -> dict[str, object]:
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "format_version": FORMAT_VERSION,
        "files": files,
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
        },
    }


def save_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer_state: TrainerState,
    sampler: PromptSampler,
    config: EngineConfig,
    scheduler: Any | None = None,
    tokenizer: Any | None = None,
) -> Path:
    """Write a checkpoint atomically at an optimizer-update boundary."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        policy_directory = temporary / "policy"
        tokenizer_directory = temporary / "tokenizer"
        policy_directory.mkdir()
        tokenizer_directory.mkdir()
        torch.save(policy.state_dict(), policy_directory / "state.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(
            scheduler.state_dict() if scheduler is not None else None,
            temporary / "scheduler.pt",
        )
        torch.save(_rng_state(), temporary / "rng_state.pt")
        torch.save(sampler.state_dict(), temporary / "sampler_state.pt")
        (temporary / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": trainer_state.global_step,
                    "policy_version": trainer_state.policy_version,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(tokenizer_directory)
        else:
            (tokenizer_directory / "metadata.json").write_text(
                json.dumps({"saved": False}) + "\n", encoding="utf-8"
            )
        (temporary / "manifest.json").write_text(
            json.dumps(_manifest(temporary, config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def validate_checkpoint(path: str | Path) -> dict[str, object]:
    """Verify format and every recorded file hash before deserializing state."""

    checkpoint = Path(path)
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("checkpoint manifest files field is invalid")
    actual_files = {
        item.relative_to(checkpoint).as_posix()
        for item in checkpoint.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    if actual_files != set(files):
        raise ValueError("checkpoint file set does not match manifest")
    for relative_path, expected_hash in files.items():
        if _sha256(checkpoint / relative_path) != expected_hash:
            raise ValueError(f"checkpoint integrity check failed: {relative_path}")
    return manifest


def load_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer_state: TrainerState,
    sampler: PromptSampler,
    expected_config: EngineConfig,
    scheduler: Any | None = None,
) -> None:
    """Validate and restore all mutable state needed at the next rollout boundary."""

    checkpoint = Path(path)
    validate_checkpoint(checkpoint)
    saved_config = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
    if saved_config != expected_config.model_dump(mode="json"):
        raise ValueError("checkpoint configuration does not match the requested experiment")

    parameter = next(policy.parameters(), None)
    map_location = parameter.device if parameter is not None else torch.device("cpu")
    policy_state = torch.load(
        checkpoint / "policy" / "state.pt",
        map_location=map_location,
        weights_only=True,
    )
    optimizer_state = torch.load(
        checkpoint / "optimizer.pt",
        map_location=map_location,
        weights_only=False,
    )
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt",
        map_location=map_location,
        weights_only=False,
    )
    sampler_state = torch.load(
        checkpoint / "sampler_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    rng_state = torch.load(
        checkpoint / "rng_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    raw_trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )

    policy.load_state_dict(policy_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(scheduler_state)
    elif scheduler_state is not None:
        raise ValueError("checkpoint contains scheduler state but no scheduler was provided")
    sampler.load_state_dict(sampler_state)
    trainer_state.global_step = int(raw_trainer_state["global_step"])
    trainer_state.policy_version = int(raw_trainer_state["policy_version"])
    _restore_rng_state(rng_state)

