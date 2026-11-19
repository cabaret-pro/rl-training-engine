import pytest

from rl_engine.evaluation.arithmetic import ArithmeticTaskConfig, generate_arithmetic_prompts
from rl_engine.rewards.arithmetic import ArithmeticReward


def test_generation_is_reproducible_without_reusing_global_rng() -> None:
    config = ArithmeticTaskConfig(count=8, seed=42, min_operand=1, max_operand=10)

    assert generate_arithmetic_prompts(config) == generate_arithmetic_prompts(config)


def test_different_split_and_seed_produce_disjoint_ids() -> None:
    train = generate_arithmetic_prompts(ArithmeticTaskConfig(count=10, seed=1, split="train"))
    test = generate_arithmetic_prompts(ArithmeticTaskConfig(count=10, seed=2, split="test"))

    assert {item.prompt_id for item in train}.isdisjoint(item.prompt_id for item in test)


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
def test_generated_answers_are_verified_exactly(operation: str) -> None:
    prompts = generate_arithmetic_prompts(
        ArithmeticTaskConfig(
            count=5,
            seed=9,
            min_operand=2,
            max_operand=12,
            operations=(operation,),  # type: ignore[arg-type]
        )
    )
    reward = ArithmeticReward()

    for prompt in prompts:
        result = reward.score(prompt.text, str(prompt.metadata["answer"]), prompt.metadata)
        assert result.value == 1.0


def test_subtraction_tasks_have_nonnegative_answers() -> None:
    prompts = generate_arithmetic_prompts(
        ArithmeticTaskConfig(count=100, seed=3, operations=("subtract",))
    )

    assert all(int(item.metadata["answer"]) >= 0 for item in prompts)


def test_prompt_generation_does_not_execute_or_embed_code() -> None:
    prompt = generate_arithmetic_prompts(ArithmeticTaskConfig(count=1, seed=0))[0]

    assert "import" not in prompt.text
    assert "eval" not in prompt.text
