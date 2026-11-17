"""Command-line entry point for the RL training engine."""

import typer

app = typer.Typer(no_args_is_help=True, help="Train and validate reinforcement learning policies.")


@app.callback()
def main() -> None:
    """Train and validate reinforcement learning policies."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    from rl_engine import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()

