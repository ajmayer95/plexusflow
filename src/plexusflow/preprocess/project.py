"""Temporal projections from a raw frame stack.

Fade-cut logic (dropping photobleached tail frames) is inlined here rather
than a separate dependency; the algorithm is a simple mean-intensity trace
+ smooth + threshold-crossing detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── fade detection ──────────────────────────────────────────
def cut_before_fade(
    frames: np.ndarray,
    p_drop: float = 0.75,
    p_back: float = 0.90,
    smooth: int = 15,
    backoff: int = 5,
) -> tuple[np.ndarray, int | None]:
    """Drop frames after fluorescence has bleached below ``p_drop`` × baseline.

    Parameters
    ----------
    frames
        ``(T, H, W)`` float stack.
    p_drop
        Fractional threshold of the smoothed mean-intensity trace below
        which frames are considered faded.  Baseline is the median of the
        first quarter of the trace.
    p_back
        (Reserved for a future asymmetric threshold; currently unused.)
    smooth
        Boxcar length for smoothing the mean-intensity trace.
    backoff
        Extra frames to drop before the detected cut, in case the fade
        onset is gradual.

    Returns
    -------
    (kept_frames, cut_at)
        Frames up to but not including ``cut_at``.  ``cut_at`` is ``None``
        if no fade was detected.
    """
    _ = p_back  # kept for future use
    mean_t = frames.reshape(frames.shape[0], -1).mean(axis=1)
    kern = np.ones(smooth) / smooth
    trace = np.convolve(mean_t, kern, mode="same")
    baseline = np.median(trace[: max(len(trace) // 4, 1)])
    below = trace < baseline * p_drop
    if not below.any():
        return frames, None
    cut_at = max(0, int(np.argmax(below)) - backoff)
    if cut_at <= 0:
        return frames, cut_at
    return frames[:cut_at], cut_at


# ── projections ─────────────────────────────────────────────
@dataclass
class ProjectionSet:
    """All temporal projections computed jointly (single fade-cut pass)."""

    median: np.ndarray  # H x W float32
    std: np.ndarray     # H x W float32 — flow / lumen proxy
    max: np.ndarray     # H x W float32 — full lumen extent
    cv: np.ndarray      # H x W float32 — dimensionless flow proxy
    n_kept: int         # # frames after fade-cut
    n_total: int        # frames in
    cut_at: int | None


def temporal_median(
    frames: np.ndarray,
    apply_fade_cut: bool = True,
    **fade_kw,
) -> np.ndarray:
    """Return the temporal median of ``frames`` (fade-cut on by default)."""
    if apply_fade_cut:
        frames, _ = cut_before_fade(frames, **fade_kw)
    return np.median(frames, axis=0).astype(np.float32)


def temporal_projections(
    frames: np.ndarray,
    apply_fade_cut: bool = True,
    **fade_kw,
) -> ProjectionSet:
    """Return median / std / max / CV in one pass after fade-cut."""
    n_total = frames.shape[0]
    cut_at = None
    if apply_fade_cut:
        frames, cut_at = cut_before_fade(frames, **fade_kw)
    med = np.median(frames, axis=0).astype(np.float32)
    std = frames.std(axis=0).astype(np.float32)
    mx = frames.max(axis=0).astype(np.float32)
    cv = (std / np.maximum(med, 1.0)).astype(np.float32)
    return ProjectionSet(
        median=med, std=std, max=mx, cv=cv,
        n_kept=frames.shape[0], n_total=n_total, cut_at=cut_at,
    )
