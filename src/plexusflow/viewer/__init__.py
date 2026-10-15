"""napari-based viewer for plexusflow stages and releases.

Entry points:
    plexusflow view <STAGE_ROOT>        read-only (also frozen releases)
    plexusflow workbench <STAGE_ROOT>   + fresh OF, tagging, boundary edits (stage folders only)
"""
from __future__ import annotations

from .app import ReadOnlyMosaicViewer

# Backwards-compat alias for the earlier build-up viewer's class name.
ReadOnlyViewer = ReadOnlyMosaicViewer

__all__ = ["ReadOnlyMosaicViewer", "ReadOnlyViewer"]
