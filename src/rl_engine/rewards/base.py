"""Reward interfaces with component-level observability."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RewardResult:
    """A scalar reward plus independently loggable component scores."""

    value: float
    components: Mapping[str, float]


class RewardFunction(Protocol):
    """Scores structured prompt/completion records without depending on an algorithm."""

    def score(
        self,
        prompt: str,
        completion: str,
        metadata: Mapping[str, object],
    ) -> RewardResult: ...

