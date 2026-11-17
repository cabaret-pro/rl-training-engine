"""Policy-gradient algorithms implemented from explicit tensor objectives."""

from rl_engine.algorithms.advantages import group_relative_advantages
from rl_engine.algorithms.grpo import GRPOOutput, grpo_loss, importance_ratios
from rl_engine.algorithms.reinforce import (
    ReinforceOutput,
    compute_reinforce_advantages,
    reinforce_loss,
)

__all__ = [
    "GRPOOutput",
    "ReinforceOutput",
    "compute_reinforce_advantages",
    "group_relative_advantages",
    "grpo_loss",
    "importance_ratios",
    "reinforce_loss",
]


