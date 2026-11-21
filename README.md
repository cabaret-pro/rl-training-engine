# RL Training Engine

A correctness-first reinforcement learning training engine for language models. The project
implements REINFORCE, GRPO, and PPO from first principles, with explicit trajectory contracts,
numerical correctness tests, reproducible checkpoints, and observable training systems.

The project deliberately starts with transparent, single-process PyTorch implementations. Faster
rollout backends and distributed execution will be added only after the underlying objectives and
gradients are independently verified.

## Status

Active development. The repository now includes numerically tested REINFORCE and GRPO objectives,
a synchronous Transformers training path, deterministic checkpoint recovery, held-out arithmetic
evaluation, and structured experiment metrics. A pinned Qwen2.5-0.5B-Instruct CPU compatibility
run has completed through generation, backpropagation, evaluation, and checkpointing. See the
[smoke-test report](docs/experiments/qwen_0_5b_cpu_smoke.md).

## Development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Run the offline-tested CLI path with a configured model:

```bash
uv sync --extra dev --extra lm
uv run rl-engine validate-config --config configs/experiments/arithmetic_grpo_smoke.yaml
uv run rl-engine train \
  --config configs/experiments/arithmetic_grpo_smoke.yaml \
  --output outputs/arithmetic-grpo-qwen-smoke
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

