"""Shared masked reductions for token-level policy objectives."""

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over active elements, rejecting empty masks instead of returning NaN."""

    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have identical shapes, got {values.shape} and {mask.shape}"
        )
    if mask.dtype != torch.bool:
        raise TypeError("mask must use torch.bool")
    active_count = mask.sum()
    if active_count.item() == 0:
        raise ValueError("masked reduction requires at least one active element")
    return values.masked_fill(~mask, 0.0).sum() / active_count

