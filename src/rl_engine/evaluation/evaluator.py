"""Deterministic held-out evaluation for verifiable language-model tasks."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import nn

from rl_engine.rewards.base import RewardFunction
from rl_engine.rollout.generator import Tokenizer, completion_mask_from_eos
from rl_engine.rollout.sampler import PromptRecord


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mean_reward: float
    reward_std: float
    pass_rate: float
    mean_completion_length: float
    generated_tokens: int
    latency_ms: float
    num_examples: int


class LanguageModelEvaluator:
    """Evaluate greedy completions without constructing training trajectories."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        reward: RewardFunction,
        *,
        max_completion_tokens: int,
    ) -> None:
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.max_completion_tokens = max_completion_tokens

    def _device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def evaluate(self, prompts: Sequence[PromptRecord]) -> EvaluationMetrics:
        if not prompts:
            raise ValueError("evaluation requires at least one prompt")
        rewards: list[float] = []
        passes: list[float] = []
        completion_lengths: list[int] = []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        was_training = self.model.training
        self.model.eval()
        started = perf_counter()
        try:
            with torch.inference_mode():
                for prompt in prompts:
                    encoded = self.tokenizer(
                        prompt.text,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    input_ids = encoded["input_ids"].to(self._device())
                    attention_mask = encoded["attention_mask"].to(self._device())
                    output: Any = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=self.max_completion_tokens,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    sequences = output.sequences if hasattr(output, "sequences") else output
                    completion_ids = sequences[0, input_ids.shape[1] :]
                    completion_mask = completion_mask_from_eos(
                        completion_ids,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    completion_text = self.tokenizer.decode(
                        completion_ids[completion_mask].tolist(),
                        skip_special_tokens=True,
                    )
                    result = self.reward.score(prompt.text, completion_text, prompt.metadata)
                    rewards.append(result.value)
                    passes.append(float(result.components.get("correctness", 0.0) > 0))
                    completion_lengths.append(int(completion_mask.sum()))
        finally:
            self.model.train(was_training)

        reward_tensor = torch.tensor(rewards, dtype=torch.float64)
        latency_ms = (perf_counter() - started) * 1000
        return EvaluationMetrics(
            mean_reward=float(reward_tensor.mean()),
            reward_std=float(reward_tensor.std(correction=0)),
            pass_rate=sum(passes) / len(passes),
            mean_completion_length=sum(completion_lengths) / len(completion_lengths),
            generated_tokens=sum(completion_lengths),
            latency_ms=latency_ms,
            num_examples=len(prompts),
        )

