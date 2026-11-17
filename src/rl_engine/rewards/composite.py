"""Weighted aggregation of independently observable reward functions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rl_engine.rewards.base import RewardFunction, RewardResult


@dataclass(frozen=True, slots=True)
class WeightedReward:
    name: str
    weight: float
    reward: RewardFunction


class CompositeReward:
    def __init__(self, rewards: Sequence[WeightedReward]) -> None:
        if not rewards:
            raise ValueError("composite reward requires at least one component")
        names = [item.name for item in rewards]
        if len(names) != len(set(names)):
            raise ValueError("composite reward component names must be unique")
        self._rewards = tuple(rewards)

    def score(
        self,
        prompt: str,
        completion: str,
        metadata: Mapping[str, object],
    ) -> RewardResult:
        total = 0.0
        components: dict[str, float] = {}
        for item in self._rewards:
            result = item.reward.score(prompt, completion, metadata)
            weighted_value = item.weight * result.value
            components[item.name] = weighted_value
            total += weighted_value
        return RewardResult(value=total, components=components)

