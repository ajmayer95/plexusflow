"""Dense optical-flow segmentation backends (OpenCV Farneback)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import SegmentationOutput


def _to_uint8_normalised(video: np.ndarray) -> np.ndarray:
    """Rescale a float video to per-frame percentile-clipped uint8 for
    stable dense-OF input."""
    v = video.astype(np.float32, copy=False)
    # Global percentile normalisation — preserves relative brightness.
    lo = float(np.percentile(v, 1.0))
    hi = float(np.percentile(v, 99.0))
    scale = 255.0 / max(hi - lo, 1e-12)
    return np.clip((v - lo) * scale, 0, 255).astype(np.uint8)


def _clip_normalise(raw: np.ndarray, pct: float) -> np.ndarray:
    hi = float(np.percentile(raw, pct))
    if hi <= 1e-30:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip(raw / hi, 0.0, 1.0).astype(np.float32)


def _iter_farneback(video_u8: np.ndarray, frame_step: int, farneback_kw: dict):
    """Yield the dense flow between consecutive kept frames."""
    import cv2  # lazy import: keep OpenCV optional at import time
    idx = list(range(0, video_u8.shape[0], frame_step))
    for a, b in zip(idx[:-1], idx[1:]):
        yield cv2.calcOpticalFlowFarneback(
            video_u8[a], video_u8[b], None, **farneback_kw
        )


_DEFAULT_FARNEBACK_KW = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)


@dataclass
class FarnebackMagnitudeSegmenter:
    """P(vessel|x) ∝ time-averaged |v(x)| from Farneback dense OF
    between consecutive (sub-sampled) frames.

    Better vessel specificity than temporal-std when non-vessel pixels
    have noise that changes intensity but not spatially-coherent
    displacement (e.g. shot noise + slow illumination).  Slower —
    typically ~1s per (512×512) frame pair.
    """
    normalise_pct: float = 99.0
    frame_step: int = 2  # cheap 2x speedup with negligible loss at 250 fps
    farneback_kw: dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_FARNEBACK_KW))
    accepts: str = "video"
    method_name: str = "optflow.farneback_mag"
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, video: np.ndarray) -> SegmentationOutput:
        if video.ndim != 3:
            raise TypeError(
                f"{self.method_name} needs a (T, H, W) video; got shape {video.shape}"
            )
        v_u8 = _to_uint8_normalised(video)
        T, H, W = v_u8.shape
        acc_mag = np.zeros((H, W), dtype=np.float32)
        n_pairs = 0
        for flow in _iter_farneback(v_u8, self.frame_step, self.farneback_kw):
            acc_mag += np.hypot(flow[..., 0], flow[..., 1])
            n_pairs += 1
        raw = acc_mag / max(n_pairs, 1)
        p = _clip_normalise(raw, self.normalise_pct)
        return SegmentationOutput(
            probability=p,
            method=self.method_name,
            params=dict(normalise_pct=self.normalise_pct,
                        frame_step=self.frame_step,
                        farneback_kw=self.farneback_kw,
                        n_pairs=n_pairs),
            aux=dict(raw=raw),
        )


@dataclass
class FarnebackCoherenceSegmenter:
    """P(vessel|x) ∝ directional-coherence of local OF over time.

    For each pixel accumulate the *unit-magnitude* flow direction as a
    complex phasor and take |mean(e^{iθ_t})|.  ∈ [0, 1]:

    * 1 → every frame's flow points the same direction (persistent
      vessel motion).
    * 0 → directions are uniformly random (noise / static regions).

    Directly separates "moves consistently" from "moves noisily" — the
    complement of pure magnitude-based methods.  Multiply with a
    magnitude backend to combine strength and consistency.
    """
    frame_step: int = 2
    min_mag_pct: float = 5.0  # skip flow with |v| below this pct: pure-noise pixels
    farneback_kw: dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_FARNEBACK_KW))
    accepts: str = "video"
    method_name: str = "optflow.farneback_coherence"
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, video: np.ndarray) -> SegmentationOutput:
        if video.ndim != 3:
            raise TypeError(
                f"{self.method_name} needs a (T, H, W) video; got shape {video.shape}"
            )
        v_u8 = _to_uint8_normalised(video)
        T, H, W = v_u8.shape
        z_sum = np.zeros((H, W), dtype=np.complex64)
        w_sum = np.zeros((H, W), dtype=np.float32)
        for flow in _iter_farneback(v_u8, self.frame_step, self.farneback_kw):
            mag = np.hypot(flow[..., 0], flow[..., 1])
            thr = float(np.percentile(mag, self.min_mag_pct))
            mask = (mag > thr).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                unit_re = np.where(mag > 0, flow[..., 0] / mag, 0.0)
                unit_im = np.where(mag > 0, flow[..., 1] / mag, 0.0)
            z_sum += (unit_re + 1j * unit_im) * mask
            w_sum += mask
        with np.errstate(invalid="ignore", divide="ignore"):
            z_mean = np.where(w_sum > 0, z_sum / w_sum, 0.0)
        p = np.abs(z_mean).astype(np.float32)  # already in [0, 1]
        return SegmentationOutput(
            probability=p,
            method=self.method_name,
            params=dict(frame_step=self.frame_step,
                        min_mag_pct=self.min_mag_pct,
                        farneback_kw=self.farneback_kw),
            aux=dict(z_mean=z_mean),
        )
