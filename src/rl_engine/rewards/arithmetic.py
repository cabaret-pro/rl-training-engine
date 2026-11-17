"""Exact arithmetic answer verification without executing generated text."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from rl_engine.rewards.base import RewardResult

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


@dataclass(frozen=True, slots=True)
class ArithmeticReward:
    success_reward: float = 1.0
    failure_reward: float = 0.0

    def score(
        self,
        prompt: str,
        completion: str,
        metadata: Mapping[str, object],
    ) -> RewardResult:
        """Compare the last numeric completion token with ``metadata['answer']`` exactly."""

        del prompt
        if "answer" not in metadata:
            raise ValueError("arithmetic reward metadata must include 'answer'")

        matches = _NUMBER.findall(completion.replace(",", ""))
        correct = False
        if matches:
            try:
                correct = Decimal(matches[-1]) == Decimal(str(metadata["answer"]))
            except InvalidOperation:
                correct = False

        value = self.success_reward if correct else self.failure_reward
        return RewardResult(value=value, components={"correctness": value})

