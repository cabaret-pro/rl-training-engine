import pytest

from rl_engine.rollout.sampler import PromptRecord, PromptSampler


def prompts(count: int = 5) -> list[PromptRecord]:
    return [
        PromptRecord(str(index), f"prompt {index}", {"answer": index})
        for index in range(count)
    ]


def test_sampler_is_deterministic_for_same_seed() -> None:
    first = PromptSampler(prompts(), seed=17)
    second = PromptSampler(prompts(), seed=17)

    assert first.sample(5) == second.sample(5)
    assert first.sample(3) == second.sample(3)


def test_grouped_requests_are_contiguous_and_unique() -> None:
    requests = PromptSampler(prompts(), seed=4).sample_groups(2, 3)

    assert len(requests) == 6
    assert [request.prompt_group_id for request in requests] == [0, 0, 0, 1, 1, 1]
    assert [request.completion_index for request in requests] == [0, 1, 2, 0, 1, 2]
    assert requests[0].prompt == requests[2].prompt
    assert requests[0].prompt != requests[3].prompt


def test_state_restoration_reproduces_future_samples_and_group_ids() -> None:
    original = PromptSampler(prompts(), seed=91)
    original.sample_groups(2, 4)
    state = original.state_dict()

    restored = PromptSampler(prompts(), seed=999)
    restored.load_state_dict(state)

    assert restored.sample_groups(3, 2) == original.sample_groups(3, 2)
    assert restored.state_dict()["epoch"] == original.state_dict()["epoch"]


def test_state_snapshot_is_not_aliased_to_sampler() -> None:
    sampler = PromptSampler(prompts(), seed=2)
    state = sampler.state_dict()
    state["order"][0] = 999  # type: ignore[index]

    assert sampler.state_dict()["order"][0] != 999  # type: ignore[index]


def test_rejects_duplicate_prompt_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        PromptSampler([PromptRecord("same", "a"), PromptRecord("same", "b")], seed=0)


def test_rejects_invalid_group_size() -> None:
    with pytest.raises(ValueError, match="at least two"):
        PromptSampler(prompts(), seed=0).sample_groups(1, 1)
