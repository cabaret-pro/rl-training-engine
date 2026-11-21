"""Command-line entry point for the RL training engine."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from rl_engine.config import load_config
from rl_engine.observability.logger import CompositeMetricSink, JsonlMetricSink
from rl_engine.training.runtime import console_metric_sink, train_from_config

app = typer.Typer(no_args_is_help=True, help="Train and validate reinforcement learning policies.")


@app.callback()
def main() -> None:
    """Train and validate reinforcement learning policies."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    from rl_engine import __version__

    typer.echo(__version__)


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Validate an experiment configuration without loading a model."""
    loaded = load_config(config)
    typer.echo(f"valid: {loaded.experiment.name} ({loaded.algorithm.name})")


@app.command()
def train(
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    resume: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    output: Annotated[Path | None, typer.Option(file_okay=False)] = None,
) -> None:
    """Run a single-process language-model training experiment."""

    if config is None and resume is None:
        raise typer.BadParameter("provide --config for a new run or --resume for a checkpoint")
    loaded_config = load_config(config if config is not None else resume / "config.yaml")
    checkpoint_output = resume.parent if resume is not None else None
    default_output = Path("outputs") / loaded_config.experiment.name
    output_directory = output or checkpoint_output or default_output
    output_directory.mkdir(parents=True, exist_ok=True)
    metric_sink = CompositeMetricSink(
        [
            console_metric_sink,
            JsonlMetricSink(output_directory / "metrics.jsonl", append=resume is not None),
        ]
    )
    summary = train_from_config(
        loaded_config,
        metric_sink=metric_sink,
        checkpoint_directory=output_directory,
        resume_checkpoint=resume,
    )
    (output_directory / "run_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"complete: {summary.experiment} at step {summary.global_step}")



if __name__ == "__main__":
    app()

