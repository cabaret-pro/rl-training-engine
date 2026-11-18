from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_engine.rewards.arithmetic import ArithmeticReward
from rl_engine.rollout.generator import (
    SamplingConfig,
    TransformersRolloutBackend,
    completion_mask_from_eos,
    evaluate_completions,
)
from rl_engine.rollout.sampler import PromptRecord, RolloutRequest


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool
    ) -> dict[str, torch.Tensor]:
        assert text == "What is 1 + 1?"
        assert return_tensors == "pt"
        assert add_special_tokens
        return {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "2" if token_ids[0] == 6 else "wrong"


class FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(8))
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        return torch.cat([input_ids, torch.tensor([[6, 2, 0]])], dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        positions = torch.arange(input_ids.shape[1], dtype=self.bias.dtype).view(1, -1, 1)
        logits = self.bias.view(1, 1, -1) + positions
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], -1, -1))


def request() -> RolloutRequest:
    prompt = PromptRecord("add-1", "What is 1 + 1?", {"answer": 2})
    return RolloutRequest(prompt, prompt_group_id=7, completion_index=0)


def test_eos_mask_includes_eos_and_excludes_later_padding() -> None:
    mask = completion_mask_from_eos(
        torch.tensor([9, 2, 0, 0]), eos_token_id=2, pad_token_id=0
    )

    assert mask.tolist() == [True, True, False, False]


def test_baseline_backend_rejects_transformed_sampling() -> None:
    with pytest.raises(ValueError, match="identity sampling"):
        SamplingConfig(max_completion_tokens=8, temperature=0.7)
    with pytest.raises(ValueError, match="identity sampling"):
        SamplingConfig(max_completion_tokens=8, top_p=0.9)


def test_backend_generates_aligned_behavior_logprobs_and_restores_mode() -> None:
    model = FakeCausalLM()
    model.train()
    backend = TransformersRolloutBackend(
        model,
        FakeTokenizer(),
        SamplingConfig(max_completion_tokens=3),
    )

    generated = backend.generate([request()], policy_version=4)[0]

    assert model.training
    assert generated.prompt_ids.tolist() == [4, 5]
    assert generated.completion_ids.tolist() == [6, 2, 0]
    assert generated.completion_mask.tolist() == [True, True, False]
    assert generated.attention_mask.tolist() == [True, True, True, True, False]
    assert generated.policy_version == 4
    assert generated.completion_text == "2"
    expected = torch.log_softmax(torch.zeros(8), dim=0)[torch.tensor([6, 2])]
    torch.testing.assert_close(generated.behavior_logprobs[:2], expected)
    assert generated.behavior_logprobs[2].item() == 0.0
    assert model.generate_kwargs["do_sample"] is True


def test_reward_evaluation_preserves_group_policy_and_components() -> None:
    backend = TransformersRolloutBackend(
        FakeCausalLM(),
        FakeTokenizer(),
        SamplingConfig(max_completion_tokens=3),
    )
    generated = backend.generate([request()], policy_version=9)

    trajectory = evaluate_completions(generated, ArithmeticReward())[0]

    assert trajectory.rewards.item() == 1.0
    assert trajectory.reward_metadata == {"correctness": 1.0}
    assert trajectory.prompt_group_id == 7
    assert trajectory.policy_version == 9
