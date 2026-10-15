"""No-OF, projection-based motion cues.

Cheap "does this pixel change in time?" statistics.  These are strong
vessel indicators for red-cell-perfused microscopy because vessels
carry moving cells while background stays static.  Free — no dense
optical-flow computation needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import SegmentationOutput


def _clip_normalise(raw: np.ndarray, pct: float) -> np.ndarray:
    """Divide by the given percentile, clip to [0, 1]."""
    hi = float(np.percentile(raw, pct))
    if hi <= 1e-30:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip(raw / hi, 0.0, 1.0).astype(np.float32)


@dataclass
class TemporalStdSegmenter:
    """P(vessel|x) ∝ std of intensity over time at pixel x, clipped
    to [0, 1] by a percentile-normalisation.

    Strong baseline for any acquisition where the moving structure
    (blood cells) is what defines the vessel.  Insensitive to absolute
    intensity, so tolerates illumination gradients.  Cannot separate
    real motion from photon shot noise — if a background pixel has
    high shot noise it will register.
    """
    normalise_pct: float = 99.0
    frame_step: int = 1  # subsample frames for speed
    accepts: str = "video"
    method_name: str = "activity.temporal_std"
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, video: np.ndarray) -> SegmentationOutput:
        if video.ndim != 3:
            raise TypeError(
                f"{self.method_name} needs a (T, H, W) video; got shape {video.shape}"
            )
        v = video[:: self.frame_step].astype(np.float32, copy=False)
        raw = v.std(axis=0)
        p = _clip_normalise(raw, self.normalise_pct)
        return SegmentationOutput(
            probability=p,
            method=self.method_name,
            params=dict(normalise_pct=self.normalise_pct,
                        frame_step=self.frame_step,
                        n_frames_used=int(v.shape[0])),
            aux=dict(raw=raw),
        )


@dataclass
class FrameDiffStdSegmenter:
    """P ∝ std of consecutive frame-differences.  Sharper vessel
    localisation than TemporalStd on datasets with slow illumination
    drift or a low-frequency background trend (differencing removes DC
    per pixel).  Same complexity as TemporalStd."""
    normalise_pct: float = 99.0
    frame_step: int = 1
    accepts: str = "video"
    method_name: str = "activity.framediff_std"
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, video: np.ndarray) -> SegmentationOutput:
        if video.ndim != 3:
            raise TypeError(
                f"{self.method_name} needs a (T, H, W) video; got shape {video.shape}"
            )
        v = video[:: self.frame_step].astype(np.float32, copy=False)
        diffs = np.diff(v, axis=0)
        raw = diffs.std(axis=0)
        p = _clip_normalise(raw, self.normalise_pct)
        return SegmentationOutput(
            probability=p,
            method=self.method_name,
            params=dict(normalise_pct=self.normalise_pct,
                        frame_step=self.frame_step,
                        n_frames_used=int(v.shape[0])),
            aux=dict(raw=raw),
        )
