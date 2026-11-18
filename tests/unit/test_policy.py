from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_engine.models.policy import CausalLMPolicy, completion_logprobs_from_logits
from rl_engine.rollout.buffer import collate_trajectories
from rl_engine.rollout.trajectory import Trajectory


def trajectory(
    prompt: tuple[int, ...],
    completion: tuple[int, ...],
    active: int,
) -> Trajectory:
    completion_mask = torch.arange(len(completion)) < active
    return Trajectory(
        prompt_ids=torch.tensor(prompt),
        completion_ids=torch.tensor(completion),
        attention_mask=torch.cat([torch.ones(len(prompt), dtype=torch.bool), completion_mask]),
        completion_mask=completion_mask,
        old_logprobs=torch.zeros(len(completion)),
        rewards=torch.tensor(1.0),
        policy_version=0,
        prompt_group_id=0,
    )


def test_gathers_logits_preceding_each_completion_token() -> None:
    batch = collate_trajectories(
        [
            trajectory((1, 2), (3, 4), 2),
            trajectory((5,), (2, 1), 2),
        ],
        pad_token_id=0,
    )
    logits = torch.zeros((2, 4, 6), dtype=torch.float64)
    logits[0, 1, 3] = 2.0
    logits[0, 2, 4] = 3.0
    logits[1, 0, 2] = 4.0
    logits[1, 1, 1] = 5.0

    actual = completion_logprobs_from_logits(logits, batch)

    expected = torch.tensor(
        [
            [
                torch.log_softmax(logits[0, 1], dim=-1)[3],
                torch.log_softmax(logits[0, 2], dim=-1)[4],
            ],
            [
                torch.log_softmax(logits[1, 0], dim=-1)[2],
                torch.log_softmax(logits[1, 1], dim=-1)[1],
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_padding_with_out_of_vocabulary_id_is_neutralized_before_gather() -> None:
    batch = collate_trajectories(
        [trajectory((1,), (2,), 1), trajectory((1,), (3, 4), 2)],
        pad_token_id=999,
    )
    logits = torch.zeros((2, 3, 8), requires_grad=True)

    logprobs = completion_logprobs_from_logits(logits, batch)
    logprobs.sum().backward()

    assert logprobs[0, 1].item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


def test_rejects_active_token_outside_vocabulary() -> None:
    batch = collate_trajectories([trajectory((1,), (9,), 1)], pad_token_id=0)

    with pytest.raises(ValueError, match="vocabulary"):
        completion_logprobs_from_logits(torch.zeros((1, 2, 5)), batch)


class FakeCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(vocabulary_size, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = self.bias.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def test_policy_adapter_preserves_model_gradients() -> None:
    batch = collate_trajectories([trajectory((1,), (2,), 1)], pad_token_id=0)
    model = FakeCausalLM(vocabulary_size=5)

    logprobs = CausalLMPolicy(model)(batch)
    (-logprobs.mean()).backward()

    assert model.bias.grad is not None
    assert torch.count_nonzero(model.bias.grad) == 5
