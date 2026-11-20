from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_engine.evaluation.evaluator import LanguageModelEvaluator
from rl_engine.rewards.arithmetic import ArithmeticReward
from rl_engine.rollout.sampler import PromptRecord


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool
    ) -> dict[str, torch.Tensor]:
        del text, return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([[4]]), "attention_mask": torch.tensor([[1]])}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return {5: "2", 6: "wrong"}.get(token_ids[0], "") if token_ids else ""


class AlternatingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def generate(self, **kwargs: object) -> SimpleNamespace:
        assert kwargs["do_sample"] is False
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        token = 5 if self.calls == 0 else 6
        self.calls += 1
        sequence = torch.cat([input_ids, torch.tensor([[token, 2]])], dim=1)
        return SimpleNamespace(sequences=sequence)


def test_evaluator_reports_reward_pass_rate_lengths_and_restores_mode() -> None:
    model = AlternatingModel()
    model.train()
    evaluator = LanguageModelEvaluator(
        model,
        FakeTokenizer(),
        ArithmeticReward(),
        max_completion_tokens=2,
    )
    prompts = [
        PromptRecord("a", "one", {"answer": 2}),
        PromptRecord("b", "two", {"answer": 4}),
    ]

    metrics = evaluator.evaluate(prompts)

    assert model.training
    assert metrics.mean_reward == pytest.approx(0.5)
    assert metrics.reward_std == pytest.approx(0.5)
    assert metrics.pass_rate == pytest.approx(0.5)
    assert metrics.mean_completion_length == 2
    assert metrics.generated_tokens == 4
    assert metrics.num_examples == 2
    assert metrics.latency_ms >= 0


def test_evaluator_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LanguageModelEvaluator(
            AlternatingModel(),
            FakeTokenizer(),
            ArithmeticReward(),
            max_completion_tokens=2,
        ).evaluate([])
