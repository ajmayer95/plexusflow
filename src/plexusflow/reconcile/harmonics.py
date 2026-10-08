"""Per-trace harmonic summaries of a flux time series Q(t) sampled at FS."""
from __future__ import annotations
import numpy as np

FS = 250.0          # Q_t sampling rate (Hz)


def phasors(q, f0, K=3, fs=FS):
    """LSQ DC + K harmonics of f0.  Returns (a0, H) with q ≈ a0 + Σ_k Re(H_k e^{i 2π k f0 t})."""
    q = np.asarray(q, float); t = np.arange(len(q)) / fs; ok = np.isfinite(q)
    X = np.column_stack([np.ones(ok.sum())] + [fn(2 * np.pi * k * f0 * t[ok]) for k in range(1, K + 1)
                                               for fn in (np.cos, np.sin)])
    b, *_ = np.linalg.lstsq(X, q[ok], rcond=None)
    return b[0], np.array([b[1 + 2 * (k - 1)] - 1j * b[2 + 2 * (k - 1)] for k in range(1, K + 1)])


def coherence(q, f0, K=3, fs=FS):
    """Harmonic coherence ρ = ‖f'‖² / ‖f‖²  (f = demeaned trace, f' = its LSQ projection on K harmonics of f0);
    nan if the trace is unusable."""
    q = np.asarray(q, float); t = np.arange(len(q)) / fs; ok = np.isfinite(q)
    if f0 is None or not np.isfinite(f0) or f0 <= 0 or ok.sum() < 4 * K + 10:
        return np.nan
    q, t = q[ok], t[ok]
    H = np.column_stack([fn(2 * np.pi * k * f0 * t) for k in range(1, K + 1) for fn in (np.cos, np.sin)])
    f = q - q.mean(); Hc = H - H.mean(axis=0)
    beta, *_ = np.linalg.lstsq(Hc, f, rcond=None); fp = Hc @ beta
    return float(fp @ fp) / float(f @ f) if f @ f > 0 else np.nan


def beat_scatter(q, f0, fs=FS):
    """Relative per-beat H1 scatter: std over complete cardiac cycles of |H1_cycle − mean H1|, × √2 / |mean H1|
    (DC + 2 harmonics fitted per cycle).  nan with fewer than 3 complete cycles."""
    q = np.asarray(q, float); t = np.arange(len(q)) / fs; cyc = np.floor(t * f0).astype(int)
    full = [c for c in np.unique(cyc) if np.sum(cyc == c) >= 0.9 * fs / f0]
    if len(full) < 3:
        return np.nan
    H1 = []
    for c in full:
        sel = cyc == c; tt = t[sel]
        X = np.column_stack([np.ones(sel.sum()), np.cos(2 * np.pi * f0 * tt), np.sin(2 * np.pi * f0 * tt),
                             np.cos(4 * np.pi * f0 * tt), np.sin(4 * np.pi * f0 * tt)])
        b, *_ = np.linalg.lstsq(X, q[sel], rcond=None); H1.append(b[1] - 1j * b[2])
    H1 = np.array(H1); Hm = H1.mean()
    return float(np.std(np.abs(H1 - Hm)) / max(abs(Hm), 1e-12) * np.sqrt(2))
