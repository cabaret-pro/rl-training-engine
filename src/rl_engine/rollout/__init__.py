"""Rollout data contracts and generation backends."""

from rl_engine.rollout.buffer import ExperienceBuffer, TrajectoryBatch, collate_trajectories
from rl_engine.rollout.generator import (
    GeneratedCompletion,
    SamplingConfig,
    TransformersRolloutBackend,
    evaluate_completions,
)
from rl_engine.rollout.sampler import PromptRecord, PromptSampler, RolloutRequest
from rl_engine.rollout.trajectory import Trajectory

__all__ = [
    "ExperienceBuffer",
    "GeneratedCompletion",
    "PromptRecord",
    "PromptSampler",
    "RolloutRequest",
    "SamplingConfig",
    "Trajectory",
    "TrajectoryBatch",
    "TransformersRolloutBackend",
    "collate_trajectories",
    "evaluate_completions",
]




