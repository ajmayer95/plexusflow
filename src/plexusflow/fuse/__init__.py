"""Feather-blend per-tile images into a single stitched image, in world
coordinates from a saved positions.json.

Three flavours:

* :func:`feather_intensity`   — feather StdFilter or MaxMed tiles (uint16/8)
* :func:`feather_probabilities` — feather per-tile ilastik outputs; supports
  either arithmetic-mean or logit-space fusion (the latter is the correct
  Bayesian combination when tile classifiers are approximately independent).

All variants share the same warping + distance-weighted feather kernel, and
correctly propagate a per-pixel coverage mask (uncovered pixels stay at 0).
"""
from .feather import (
    feather_intensity,
    feather_probabilities,
    compute_world_bbox,
    feather_kernel_from_tile,
)

__all__ = [
    "feather_intensity",
    "feather_probabilities",
    "compute_world_bbox",
    "feather_kernel_from_tile",
]
