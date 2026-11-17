from collections.abc import Mapping

import pytest

from rl_engine.rewards import ArithmeticReward, CompositeReward, RewardResult
from rl_engine.rewards.composite import WeightedReward


def test_arithmetic_reward_uses_final_numeric_answer() -> None:
    reward = ArithmeticReward()

    result = reward.score("What is 17 + 28?", "17 + 28 = 45", {"answer": 45})

    assert result.value == 1.0
    assert result.components == {"correctness": 1.0}


def test_arithmetic_reward_handles_commas_and_decimals() -> None:
    reward = ArithmeticReward(success_reward=2.0, failure_reward=-1.0)

    assert reward.score("", "Answer: 1,024.50", {"answer": "1024.5"}).value == 2.0


def test_arithmetic_reward_does_not_evaluate_expressions() -> None:
    reward = ArithmeticReward()

    result = reward.score("", "__import__('os').system('echo unsafe')", {"answer": 0})

    assert result.value == 0.0


def test_arithmetic_reward_requires_expected_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        ArithmeticReward().score("", "42", {})


class ConstantReward:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(
        self, prompt: str, completion: str, metadata: Mapping[str, object]
    ) -> RewardResult:
        return RewardResult(self.value, {"raw": self.value})


def test_composite_reward_applies_weights_and_preserves_components() -> None:
    reward = CompositeReward(
        [
            WeightedReward("correctness", 2.0, ConstantReward(0.5)),
            WeightedReward("format", -0.5, ConstantReward(0.2)),
        ]
    )

    result = reward.score("prompt", "completion", {})

    assert result.value == pytest.approx(0.9)
    assert result.components == {"correctness": 1.0, "format": -0.1}
