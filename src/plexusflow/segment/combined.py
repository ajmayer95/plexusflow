"""Combine multiple SegmentationOutputs into one consensus probability.

For a small ensemble of backends (e.g. temporal-std + flow-coherence),
combine via a weighted mean in log-odds space — the natural home of
probability combination.  Each backend keeps its own preprocessing;
combination is a plain scalar-weighted sum after logit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .base import SegmentationOutput


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x)).astype(np.float32)


@dataclass
class CombinedSegmenter:
    """Weighted log-odds combination of several backend outputs.

    Given N segmenters and N corresponding scalar weights, the fused
    per-pixel logit is::

        logit(p_combined) = Σ_i w_i · logit(p_i) / Σ_i w_i

    which is the standard "opinion pool" for probability aggregation.
    Because the weights are scalar, the combination is uniform across
    pixels; per-pixel weights (spatial reliability) enter later at the
    fusion stage.
    """
    segmenters: list = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    accepts: str = "video"        # inherits whatever the base backends need
    method_name: str = "combined"

    def __post_init__(self) -> None:
        if len(self.segmenters) != len(self.weights):
            raise ValueError(
                f"segmenters ({len(self.segmenters)}) and weights ({len(self.weights)}) length mismatch"
            )
        if not self.segmenters:
            raise ValueError("Combined needs at least one component segmenter")
        acc = {s.accepts for s in self.segmenters}
        if acc & {"video"} and acc & {"image"}:
            self.accepts = "both"  # heterogeneous — caller responsible
        else:
            self.accepts = next(iter(acc))

    def predict(self, tile: np.ndarray) -> SegmentationOutput:
        outs = [s.predict(tile) for s in self.segmenters]
        w = np.array(self.weights, dtype=np.float64)
        w = w / max(w.sum(), 1e-30)
        stacked = np.stack([_logit(o.probability) for o in outs], axis=0)
        combined_logit = (w[:, None, None] * stacked).sum(axis=0)
        combined_p = _sigmoid(combined_logit)
        return SegmentationOutput(
            probability=combined_p,
            method=self.method_name + ":" + "+".join(o.method for o in outs),
            params=dict(
                components=[o.method for o in outs],
                weights=list(self.weights),
            ),
            aux=dict(components=outs),
        )
