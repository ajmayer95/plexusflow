"""Per-tile flow analysis over a mosaic graph.

Two entry points:
* ``condense`` — turn the raw vectorize graph into a clean vessel graph
  (contract every maximal degree-2 chain into one edge between real
  junction/leaf endpoints).
* ``mosaic_piv`` — run Farneback optical flow per tile, extract Q(t)
  per vessel, write ``measurements_piv[tile_id]`` on each vessel edge.

Shared low-level pieces live in :mod:`plexusflow.analyze.geometry`.
"""
from __future__ import annotations

from .geometry import (
    load_tile_geometry,
    graph_to_tile,
    graph_to_tile_batch,
    tile_video_shape,
)
from .condense import condense_graph, condense_stage
from .mosaic_piv import analyze_stage

__all__ = [
    "load_tile_geometry",
    "graph_to_tile",
    "graph_to_tile_batch",
    "tile_video_shape",
    "condense_graph",
    "condense_stage",
    "analyze_stage",
]
