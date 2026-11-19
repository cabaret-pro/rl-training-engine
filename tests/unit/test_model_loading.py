import pytest
import torch

from rl_engine.models.loading import resolve_device, resolve_dtype


def test_auto_device_resolves_to_available_backend() -> None:
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda", "mps"}


def test_float16_cpu_is_rejected_before_model_loading() -> None:
    with pytest.raises(ValueError, match="float16"):
        resolve_dtype("float16", torch.device("cpu"))


def test_float32_cpu_is_supported() -> None:
    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32

