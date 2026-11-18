"""Correctness-first Transformers-compatible rollout generation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from rl_engine.rewards.base import RewardFunction
from rl_engine.rollout.sampler import RolloutRequest
from rl_engine.rollout.trajectory import Trajectory


class Tokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> Mapping[str, torch.Tensor]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    max_completion_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != 0:
            raise ValueError(
                "the baseline backend only supports identity sampling "
                "(temperature=1, top_p=1, top_k=0) so stored behavior logprobs remain exact"
            )


@dataclass(frozen=True, slots=True)
class GeneratedCompletion:
    request: RolloutRequest
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    behavior_logprobs: torch.Tensor
    completion_text: str
    policy_version: int


def completion_mask_from_eos(
    completion_ids: torch.Tensor,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Mark generated tokens through the first EOS, excluding subsequent padding."""

    if completion_ids.ndim != 1:
        raise ValueError("completion_ids must be one-dimensional")
    mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    ended = False
    for index, token in enumerate(completion_ids.tolist()):
        if ended:
            continue
        if pad_token_id is not None and token == pad_token_id and token != eos_token_id:
            ended = True
            continue
        mask[index] = True
        if eos_token_id is not None and token == eos_token_id:
            ended = True
    return mask


def evaluate_completions(
    completions: Sequence[GeneratedCompletion],
    reward: RewardFunction,
) -> list[Trajectory]:
    """Evaluate generated text and construct algorithm-ready trajectories."""

    trajectories: list[Trajectory] = []
    for generated in completions:
        result = reward.score(
            generated.request.prompt.text,
            generated.completion_text,
            generated.request.prompt.metadata,
        )
        trajectories.append(
            Trajectory(
                prompt_ids=generated.prompt_ids,
                completion_ids=generated.completion_ids,
                attention_mask=generated.attention_mask,
                completion_mask=generated.completion_mask,
                old_logprobs=generated.behavior_logprobs,
                rewards=generated.behavior_logprobs.new_tensor(result.value),
                policy_version=generated.policy_version,
                prompt_group_id=generated.request.prompt_group_id,
                reward_metadata=dict(result.components),
            )
        )
    return trajectories


class TransformersRolloutBackend:
    """Generate one completion at a time for inspectable baseline semantics."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        config: SamplingConfig,
    ) -> None:
        if tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define an EOS or padding token ID")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _model_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def generate(
        self,
        requests: Sequence[RolloutRequest],
        *,
        policy_version: int,
    ) -> list[GeneratedCompletion]:
        if policy_version < 0:
            raise ValueError("policy_version must be nonnegative")
        was_training = self.model.training
        self.model.eval()
        generated: list[GeneratedCompletion] = []
        try:
            with torch.inference_mode():
                for request in requests:
                    generated.append(self._generate_one(request, policy_version))
        finally:
            self.model.train(was_training)
        return generated

    def _generate_one(
        self,
        request: RolloutRequest,
        policy_version: int,
    ) -> GeneratedCompletion:
        encoded = self.tokenizer(
            request.prompt.text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self._model_device())
        attention_mask = encoded["attention_mask"].to(self._model_device()).to(torch.bool)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tokenizer must return input_ids with shape [1, prompt_tokens]")
        if input_ids.shape != attention_mask.shape or not attention_mask.all():
            raise ValueError("single-request tokenization must return an unpadded attention mask")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_new_tokens=self.config.max_completion_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        sequences = output.sequences if hasattr(output, "sequences") else output
        if sequences.ndim != 2 or sequences.shape[0] != 1:
            raise ValueError("model.generate must return sequences with shape [1, tokens]")

        prompt_length = input_ids.shape[1]
        prompt_ids = sequences[0, :prompt_length].detach().clone()
        completion_ids = sequences[0, prompt_length:].detach().clone()
        if completion_ids.numel() == 0:
            raise ValueError("model.generate returned no completion tokens")
        completion_mask = completion_mask_from_eos(
            completion_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
        if not completion_mask.any():
            raise ValueError("generated completion contains no active tokens")

        sequence_attention_mask = torch.cat(
            [torch.ones_like(prompt_ids, dtype=torch.bool), completion_mask]
        ).unsqueeze(0)
        logits = self.model(
            input_ids=sequences,
            attention_mask=sequence_attention_mask,
        ).logits
        first_prediction = prompt_length - 1
        last_prediction = first_prediction + completion_ids.numel()
        prediction_logits = logits[0, first_prediction:last_prediction]
        safe_completion_ids = completion_ids.masked_fill(~completion_mask, 0)
        behavior_logprobs = prediction_logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=safe_completion_ids.unsqueeze(-1),
        ).squeeze(-1)
        behavior_logprobs = behavior_logprobs.masked_fill(~completion_mask, 0.0).detach()

        return GeneratedCompletion(
            request=request,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            attention_mask=sequence_attention_mask.squeeze(0),
            completion_mask=completion_mask,
            behavior_logprobs=behavior_logprobs,
            completion_text=self.tokenizer.decode(
                completion_ids[completion_mask].tolist(), skip_special_tokens=True
            ),
            policy_version=policy_version,
        )

