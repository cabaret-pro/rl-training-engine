"""Reproducible, exactly verifiable arithmetic prompt generation."""

from dataclasses import dataclass
from typing import Literal

import torch

from rl_engine.rollout.sampler import PromptRecord

Operation = Literal["add", "subtract", "multiply"]


@dataclass(frozen=True, slots=True)
class ArithmeticTaskConfig:
    count: int
    seed: int
    min_operand: int = 0
    max_operand: int = 100
    operations: tuple[Operation, ...] = ("add", "subtract", "multiply")
    split: str = "train"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.min_operand < 0 or self.max_operand < self.min_operand:
            raise ValueError("operand range must be nonnegative and ordered")
        if not self.operations:
            raise ValueError("at least one arithmetic operation is required")
        if len(self.operations) != len(set(self.operations)):
            raise ValueError("arithmetic operations must be unique")
        if not self.split or not self.split.replace("-", "").replace("_", "").isalnum():
            raise ValueError("split must be a nonempty identifier")


def _problem(operation: Operation, left: int, right: int) -> tuple[str, int]:
    if operation == "add":
        return f"What is {left} + {right}? Give only the final number.", left + right
    if operation == "subtract":
        larger, smaller = max(left, right), min(left, right)
        return f"What is {larger} - {smaller}? Give only the final number.", larger - smaller
    if operation == "multiply":
        return f"What is {left} * {right}? Give only the final number.", left * right
    raise ValueError(f"unsupported arithmetic operation: {operation}")


def generate_arithmetic_prompts(config: ArithmeticTaskConfig) -> list[PromptRecord]:
    """Generate a stable prompt set from a local torch RNG without touching global state."""

    generator = torch.Generator().manual_seed(config.seed)
    operand_count = config.max_operand - config.min_operand + 1
    prompts: list[PromptRecord] = []
    for index in range(config.count):
        left = int(torch.randint(operand_count, (), generator=generator)) + config.min_operand
        right = int(torch.randint(operand_count, (), generator=generator)) + config.min_operand
        operation_index = int(torch.randint(len(config.operations), (), generator=generator))
        operation = config.operations[operation_index]
        text, answer = _problem(operation, left, right)
        prompts.append(
            PromptRecord(
                prompt_id=f"{config.split}-{config.seed}-{index}",
                text=text,
                metadata={
                    "answer": answer,
                    "operation": operation,
                    "left_operand": left,
                    "right_operand": right,
                    "split": config.split,
                },
            )
        )
    return prompts

