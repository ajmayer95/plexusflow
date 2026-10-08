"""Pluggable segmentation backends.

Every backend consumes a tile (single 2-D reference image or a
(T, H, W) video) and returns a common `SegmentationOutput` — a per-pixel
vessel probability in [0, 1] plus optional per-pixel reliability weight
and method-specific extras.

Fusion / downstream stages depend only on the common interface, so
new backends (ilastik, SAM, etc.) can be dropped in without changing
anything else.
"""
from .base import Segmenter, SegmentationOutput

__all__ = ["Segmenter", "SegmentationOutput"]
