"""Hybrid videos: exact synthetic tracers on a real tile's static background with that camera's noise.

frame = clip(Poisson((bg + scale · tracers) · g) / g + N(0, r²), 0, 255)
  bg       per-pixel 10th percentile of the real clip (tissue texture and stuck puncta; moving tracers removed)
  tracers  Gaussian spots from the exact P2 Stokes field (tracers.py): width and peak brightness depend on depth z
           (σ(z) = σ0 · clip(√(1 + (z/z_c)²), 1, w_max), peak ∝ clip(1/(1 + (z/z_c)²), 0, 1), lognormal brightness),
           stretched along the in-frame displacement (exposure = frame interval)
  scale    either matches the lumen temporal SD of the real clip / a typical value, or sets tracer contrast relative to noise
  g, r     photon-transfer camera model of that tile (bundle.tile_noise)
Optional whole-frame jitter (rigid shift per frame) is applied before the camera noise.
"""
from __future__ import annotations

import numpy as np

from .tracers import splat

# tracer settings calibrated against real videos (docs/synthetic_validation.md)
TRACERS_JUNCTION = dict(C=0.05, sigma0=0.8, z_c=16.0, w_max=4.0, i_min=0.0, spread=0.8)
TRACERS_VESSEL = dict(C=0.033, sigma0=0.8, z_c=16.0, w_max=4.0, spread=0.8)
FDT = 0.004                       # s per frame (250 fps)
A1, A2, PHI2 = 0.4, 0.12, 0.375 * np.pi


def waveform(f0, fdt=FDT, a1=A1, a2=A2, phi2=PHI2):
    """Parent flow shape s(t) = 1 + a1 cos ωt + a2 cos(2ωt + φ2), t in frames."""
    return lambda t: 1 + a1 * np.cos(2 * np.pi * f0 * t * fdt) + a2 * np.cos(4 * np.pi * f0 * t * fdt + phi2)


def tau_at_frames(s_t, nf):
    """∫ s dt at the frame boundaries 0..nf (frames); flow in frame f is evaluated at its centre f + 0.5."""
    return np.r_[0, np.cumsum(s_t(np.arange(nf) + 0.5))]


class JunctionFrame:
    """Junction geometry (x, y) = (col, −row) in the tile frame → crop pixel coordinates."""

    def __init__(self, r0, c0):
        self.r0, self.c0 = r0, c0

    def pos(self, X):
        return (-X[:, 1] - self.r0).astype(float), (X[:, 0] - self.c0).astype(float)

    def disp(self, A, B):
        return (-(B[:, 1] - A[:, 1])).astype(float), (B[:, 0] - A[:, 0]).astype(float)


class VesselFrame:
    """Single-vessel geometry (x, y) → image (row = y − y0, col = x − x0)."""

    def __init__(self, x0, y0):
        self.x0, self.y0 = x0, y0

    def pos(self, X):
        return (X[:, 1] - self.y0).astype(float), (X[:, 0] - self.x0).astype(float)

    def disp(self, A, B):
        ra, ca = A[:, 1] - self.y0, A[:, 0] - self.x0; rb, cb = B[:, 1] - self.y0, B[:, 0] - self.x0
        return (rb - ra).astype(float), (cb - ca).astype(float)


def tracer_movie(rec, idx, taus, nf, shape, frame, tracer, rng, intens):
    """Pure tracer image sequence (nf, H, W) for one speed.  `intens` (tracer id → brightness) is shared across speeds of
    the same advection so each tracer keeps its brightness; new tracers draw theirs in order of first appearance."""
    img = np.zeros((nf,) + tuple(shape))
    i_min = tracer.get("i_min", 0)
    for f in range(nf):
        ia, pa = rec[idx[taus[f]]]; ib, pb = rec[idx[taus[f + 1]]]
        common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
        A, B = pa[ka], pb[kb]
        for i in common:
            if i not in intens:
                intens[i] = rng.lognormal(-0.5 * tracer["spread"] ** 2, tracer["spread"])
        M = 0.5 * (A + B); u2 = (M[:, 2] / tracer["z_c"]) ** 2
        sig = tracer["sigma0"] * np.clip(np.sqrt(1 + u2), 1, tracer["w_max"])
        amp = np.array([intens[i] for i in common]) * np.clip(1 / (1 + u2), i_min, 1)
        rows, cols = frame.pos(M); dr, dc = frame.disp(A, B)
        splat(img[f], rows, cols, dr, dc, sig, amp)
    return img


def noise_sd(bg, mask, gain, read):
    """Camera noise SD at the median background level inside `mask`."""
    return np.sqrt(np.median(bg[mask]) / gain + read ** 2)


def scale_to_sd(img, bg, gain, read, target_sd, noise_mask, signal_mask):
    """Tracer scale so that (tracer SD² + noise SD²) matches target_sd² in the lumen."""
    nsd = noise_sd(bg, noise_mask, gain, read)
    return np.sqrt(max(target_sd ** 2 - nsd ** 2, 1e-6)) / max(np.median(img.std(0)[signal_mask]), 1e-9)


def scale_to_contrast(img, bg, gain, read, contrast, mask):
    """Tracer scale so that the lumen tracer-only SD = contrast × camera noise SD."""
    return contrast * noise_sd(bg, mask, gain, read) / max(np.median(img.std(0)[mask]), 1e-9)


def camera(img, bg, scale, gain, read, rng, jitter=None):
    """bg + scale · tracers, optional per-frame rigid shift jitter (nf, 2) = (dy, dx) px, then Poisson + read noise."""
    clean = bg[None] + scale * img
    if jitter is not None:
        import cv2
        H, W = bg.shape
        for f in range(len(clean)):
            Mw = np.float32([[1, 0, jitter[f, 1]], [0, 1, jitter[f, 0]]])
            clean[f] = cv2.warpAffine(clean[f].astype(np.float32), Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(rng.poisson(np.maximum(clean, 0) * gain) / gain + rng.normal(0, read, clean.shape), 0, 255).astype(np.float32)
