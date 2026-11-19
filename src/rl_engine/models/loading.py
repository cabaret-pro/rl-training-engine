"""Lazy Hugging Face model loading with explicit device and dtype selection."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rl_engine.config import ModelConfig


@dataclass(frozen=True, slots=True)
class LoadedCausalLM:
    model: nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 language-model training is not supported on CPU")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 was requested but this CUDA device does not support it")
    return dtype


def load_causal_lm(config: ModelConfig) -> LoadedCausalLM:
    """Load a causal LM and tokenizer only after runtime compatibility checks pass."""

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision_kwargs = {"revision": config.revision} if config.revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(config.name, **revision_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=dtype,
        **revision_kwargs,
    )
    model.to(device)
    model.config.use_cache = False
    return LoadedCausalLM(model=model, tokenizer=tokenizer, device=device, dtype=dtype)

