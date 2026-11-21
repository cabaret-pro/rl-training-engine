# Qwen2.5-0.5B CPU compatibility smoke test

## Purpose

This experiment verifies that the engine can load a pinned 0.5B parameter language model and run
the complete arithmetic GRPO lifecycle through a real Transformers model. It is a compatibility
test, not evidence of learning effectiveness.

## Reproduction

```bash
uv sync --extra dev --extra lm
uv run rl-engine train \
  --config configs/experiments/arithmetic_grpo_smoke.yaml \
  --output outputs/arithmetic-grpo-qwen-smoke
```

The model revision, seed, device, dtype, sampling configuration, and dependency versions are
recorded in the configuration and result artifact.

## Result

The run completed successfully on an Apple M4 Pro with 24 GB unified memory, using CPU float32:

- The pinned Qwen2.5-0.5B-Instruct model loaded successfully.
- Greedy held-out evaluation completed before and after training.
- Two completions were sampled from one arithmetic prompt group.
- Token-level behavior log probabilities were reconstructed.
- The GRPO objective completed forward and backward execution.
- The optimizer advanced global step and policy version to 1.
- An integrity-checked checkpoint containing model, optimizer, RNG, sampler, and trainer state was
  created.

Both sampled completions received zero reward. Group-relative advantages were therefore exactly
zero, producing loss `0.0` and gradient norm `0.0`. This is expected GRPO behavior for a
zero-variance reward group and demonstrates that the engine does not manufacture a learning signal.

The raw metrics are preserved in
[`benchmarks/results/qwen_0_5b_cpu_smoke.json`](../../benchmarks/results/qwen_0_5b_cpu_smoke.json).

## What this establishes

This run establishes model-scale compatibility, probability reconstruction, full-model
backpropagation, evaluation, and checkpoint creation for the 0.5B target class.

It does not establish reward improvement, sample efficiency, stable multi-step optimization, or
GPU throughput. Those require a larger experiment with sufficient reward variance and multiple
seeds.

