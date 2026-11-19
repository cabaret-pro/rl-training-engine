"""Command-line entry point for the RL training engine."""

from pathlib import Path
from typing import Annotated

import typer

from rl_engine.config import load_config
from rl_engine.training.runtime import train_from_config

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
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Run a single-process language-model training experiment."""

    summary = train_from_config(load_config(config))
    typer.echo(f"complete: {summary.experiment} at step {summary.global_step}")



if __name__ == "__main__":
    app()

