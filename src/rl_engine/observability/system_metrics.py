"""Reproducibility metadata collected without allocating model weights."""

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from rl_engine.config import EngineConfig
from rl_engine.models.loading import resolve_device, resolve_dtype


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _source_revision() -> tuple[str | None, bool | None]:
    repository = Path(__file__).parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def collect_run_metadata(config: EngineConfig) -> dict[str, object]:
    """Describe the exact code, model request, software, and hardware for a run."""

    device = resolve_device(config.model.device)
    dtype = resolve_dtype(config.model.dtype, device)
    source_revision, source_dirty = _source_revision()
    return {
        "kind": "run_metadata",
        "experiment": config.experiment.name,
        "seed": config.experiment.seed,
        "source": {
            "revision": source_revision,
            "dirty": source_dirty,
        },
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
            "requested_dtype": config.model.dtype,
            "resolved_dtype": str(dtype).removeprefix("torch."),
            "requested_device": config.model.device,
            "resolved_device": str(device),
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy"),
            "rl_training_engine": _package_version("rl-training-engine"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": torch.backends.mps.is_available(),
        },
    }

