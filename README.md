# RL Training Engine

A correctness-first reinforcement learning training engine for language models. The project
implements REINFORCE, GRPO, and PPO from first principles, with explicit trajectory contracts,
numerical correctness tests, reproducible checkpoints, and observable training systems.

The project deliberately starts with transparent, single-process PyTorch implementations. Faster
rollout backends and distributed execution will be added only after the underlying objectives and
gradients are independently verified.

## Status

Early development. The first milestone covers validated configuration, trajectory data contracts,
reward interfaces, and a numerically tested REINFORCE baseline. GRPO is the first major language
model training milestone.

## Development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Design principles

- Correctness before throughput.
- Explicit, validated data contracts.
- Replaceable algorithms, rollout backends, and rewards.
- Reproducible experiment and checkpoint state.
- Observable numerical and systems behavior.

## Roadmap

1. Foundations and REINFORCE correctness.
2. GRPO objectives, diagnostics, and reference comparisons.
3. End-to-end arithmetic training with a 0.5B parameter model.
4. PPO, deterministic checkpoint recovery, profiling, and distributed execution.

## License

MIT

