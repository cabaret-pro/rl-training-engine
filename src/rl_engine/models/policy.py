"""Causal language-model adapter with explicit generated-token alignment."""

from typing import Protocol

import torch
from torch import nn

from rl_engine.rollout.buffer import TrajectoryBatch


class CausalLMOutput(Protocol):
    logits: torch.Tensor


class CausalLM(Protocol):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CausalLMOutput: ...


def completion_logprobs_from_logits(
    logits: torch.Tensor,
    batch: TrajectoryBatch,
) -> torch.Tensor:
    """Gather one raw-policy log probability for every generated-token slot.

    In a causal LM, logits at sequence position ``k - 1`` predict the token at position ``k``.
    Variable prompt lengths therefore require row-specific gather positions. Inactive completion
    positions are replaced before both gathers and set to exact zero in the result.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("logits batch and sequence dimensions must match input_ids")
    if logits.shape[2] <= 0:
        raise ValueError("logits vocabulary dimension must be positive")
    if (batch.prompt_lengths <= 0).any():
        raise ValueError("every sequence must contain at least one prompt token")

    batch_size, completion_length = batch.completion_ids.shape
    completion_offsets = torch.arange(completion_length, device=logits.device).unsqueeze(0)
    prediction_positions = batch.prompt_lengths.to(logits.device).unsqueeze(1) - 1
    prediction_positions = prediction_positions + completion_offsets
    safe_positions = prediction_positions.masked_fill(~batch.completion_mask, 0)
    if (safe_positions >= logits.shape[1]).any():
        raise ValueError("active completion token has no corresponding prediction position")

    row_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
    selected_logits = logits[row_indices, safe_positions]
    safe_token_ids = batch.completion_ids.to(logits.device).masked_fill(
        ~batch.completion_mask, 0
    )
    if ((safe_token_ids < 0) | (safe_token_ids >= logits.shape[2])).any():
        raise ValueError("active completion token ID is outside the model vocabulary")

    token_logprobs = selected_logits.log_softmax(dim=-1).gather(
        dim=-1, index=safe_token_ids.unsqueeze(-1)
    )
    return token_logprobs.squeeze(-1).masked_fill(~batch.completion_mask, 0.0)


class CausalLMPolicy(nn.Module):
    """Expose generated-token log probabilities from a Hugging Face-style causal LM."""

    def __init__(self, model: CausalLM) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model

    def forward(self, batch: TrajectoryBatch) -> torch.Tensor:
        output = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        return completion_logprobs_from_logits(output.logits, batch)

