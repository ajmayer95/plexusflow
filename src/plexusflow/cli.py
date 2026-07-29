"""plexusflow command-line interface.

Stage subcommands (`ingest`, `register`, `segment`, `fuse`, `vectorize`,
`edit`, `qc`) will be filled in as each stage lands.  For the walking
skeleton this exposes only `plexusflow version` and `plexusflow demo`.
"""
from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(add_completion=False, help="Whole-organ vascular reconstruction.")


@app.command()
def version() -> None:
    """Print the installed plexusflow version."""
    typer.echo(f"plexusflow {__version__}")


@app.command()
def demo() -> None:
    """Placeholder — runs the end-to-end demo on `examples/4tile_demo/`."""
    typer.echo("demo not yet implemented — see docs/roadmap.md")


if __name__ == "__main__":
    app()
