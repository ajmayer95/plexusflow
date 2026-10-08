"""Common interface for every segmentation backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class SegmentationOutput:
    """Per-tile segmentation result.

    Fields
    ------
    probability
        H x W float32 in [0, 1].  Per-pixel P(vessel | tile, method).
    weight
        Optional H x W float32.  Per-pixel spatial reliability for
        multi-tile fusion (e.g. distance-from-tile-boundary, focus,
        SNR).  When None the fusion stage falls back to a uniform 1.
    method
        Short identifier (e.g. `"optflow.farneback_mag"`).  Used in
        provenance strings on downstream graph edges.
    params
        Full parameter dict — the round-trippable record of what was
        actually computed.  Written into the QC report and edit log.
    aux
        Backend-specific extras (e.g. the raw motion-magnitude image
        before normalisation, or coherence maps).  Not required by the
        fusion stage but useful for QC and human-in-the-loop review.
    """
    probability: np.ndarray
    weight: np.ndarray | None = None
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = self.probability
        if p.ndim != 2:
            raise ValueError(f"probability must be 2-D, got shape {p.shape}")
        if p.dtype.kind != "f":
            self.probability = p.astype(np.float32)
            p = self.probability
        pmin, pmax = float(p.min()), float(p.max())
        if pmin < -1e-6 or pmax > 1.0 + 1e-6:
            raise ValueError(
                f"probability must lie in [0, 1]; got [{pmin:.4f}, {pmax:.4f}]"
            )
        if self.weight is not None:
            if self.weight.shape != p.shape:
                raise ValueError(
                    f"weight shape {self.weight.shape} != probability shape {p.shape}"
                )
            self.weight = self.weight.astype(np.float32)


class Segmenter(Protocol):
    """Protocol implemented by every backend.

    Backends declare what they accept via ``accepts``:

    * ``"video"`` — needs a (T, H, W) frame stack (motion methods).
    * ``"image"`` — needs a single (H, W) reference image.
    * ``"both"`` — accepts either; the backend chooses what to use.
    """

    method_name: str
    accepts: str  # "video" | "image" | "both"

    def predict(self, tile: np.ndarray) -> SegmentationOutput: ...
