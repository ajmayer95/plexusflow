"""napari-based read-only viewer for plexusflow stages.

Entry point:
    plexusflow view <STAGE_ROOT>
"""
from __future__ import annotations

from .app import ReadOnlyMosaicViewer

# Backwards-compat alias for the earlier build-up viewer's class name.
ReadOnlyViewer = ReadOnlyMosaicViewer

__all__ = ["ReadOnlyMosaicViewer", "ReadOnlyViewer"]
