"""Smoke tests for the walking skeleton."""
import plexusflow


def test_version_string():
    assert isinstance(plexusflow.__version__, str)
    assert plexusflow.__version__.count(".") == 2


def test_cli_importable():
    from plexusflow import cli
    assert cli.app is not None
