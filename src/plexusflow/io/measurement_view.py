"""
MeasurementView — schema-version-agnostic accessor for measurements_piv.

Downstream code should read measurements through this accessor rather
than reaching into the dict directly, so that a future schema bump
does not require touching every read site.

Two entry points:

    view = MeasurementView(m_dict)         # wrap one measurement
    best = MeasurementView.best(edge_data) # pick canonical measurement on an edge

All accessors normalise across:
- v3 flat schema (pre-2026-07): mean_Q, R_um_tile, snr_ac_fit_db, v_t, ...
- v4 nested schema:              m.harm.mean_Q, m.geom.R_um, m.snr.dc_db, ...

Derived quantities (v_t, amp_Q, PI_rms, phase, eta) are computed
on-demand from stored fields, so v4 can drop them without breaking
callers.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


# ─── v3 → v4 flat-key fallback tables ────────────────────────────────────

_V3_R_KEYS = ('R_um_tile', 'R_med_tile', 'R_um_piv', 'R_med_piv')
_V3_R_HARM4_KEYS = ('R_harm4_tile', 'R_harm4_piv')


def _first_finite(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if v is not None and (isinstance(v, (int, str))
                              or (np.isscalar(v) and np.isfinite(v))):
            return v
    return None


class _AttrDict:
    """Thin wrapper for a dict that supports both m['x'] and m.x access,
    and returns None on missing keys instead of KeyError."""

    __slots__ = ('_d',)

    def __init__(self, d):
        object.__setattr__(self, '_d', d or {})

    def __getattr__(self, k):
        return self._d.get(k)

    def __getitem__(self, k):
        return self._d.get(k)

    def get(self, k, default=None):
        return self._d.get(k, default)

    def __contains__(self, k):
        return k in self._d

    def __repr__(self):
        return f'_AttrDict({self._d!r})'


class MeasurementView:
    """Read-only accessor over a single measurement dict."""

    __slots__ = ('_m', '_version')

    def __init__(self, m: dict):
        self._m = m or {}
        self._version = int(m.get('schema_version', 3)) if m else 3

    # ─── identity ────────────────────────────────────────────────────
    @property
    def tile_id(self) -> Optional[int]:
        return self._m.get('tile_id')

    @property
    def flow_from(self) -> Optional[int]:
        return self._m.get('flow_from')

    @property
    def flow_to(self) -> Optional[int]:
        return self._m.get('flow_to')

    @property
    def schema_version(self) -> int:
        return self._version

    @property
    def fit_success(self) -> bool:
        if self._version >= 4:
            h = self._m.get('harm') or {}
            return bool(h.get('fit_success', False))
        return bool(self._m.get('fit_success', False))

    # ─── geometry ────────────────────────────────────────────────────
    @property
    def geom(self) -> _AttrDict:
        if self._version >= 4:
            return _AttrDict(self._m.get('geom'))
        # v3: synthesize from flat keys
        return _AttrDict({
            'R_um':             _first_finite(self._m, _V3_R_KEYS),
            'R_conductance_um': _first_finite(self._m, _V3_R_HARM4_KEYS),
            'R_fit_free_um':    None,  # not in v3
            'L_um':             self._m.get('L_um_tile') or self._m.get('L_um_piv'),
            'cl_shift_px':      self._m.get('cl_shift'),
        })

    # ─── Q extractors ────────────────────────────────────────────────
    @property
    def Q_nls(self) -> _AttrDict:
        if self._version >= 4:
            return _AttrDict(self._m.get('Q_nls'))
        # v3: only the harmonic-fit-DC of Q_plug is stored as `mean_Q`.
        # Treat it as `best`; the four extractor variants are None.
        return _AttrDict({
            'plug_ridge': None, 'plug_seg': None,
            'pois_ridge': None, 'pois_seg': None,
            'best':       self._m.get('mean_Q'),
            'best_key':   'v3_mean_Q',
        })

    # ─── harmonic fit ────────────────────────────────────────────────
    @property
    def harm(self) -> _AttrDict:
        if self._version >= 4:
            return _AttrDict(self._m.get('harm'))
        # v3: harmonics + f0_hz + mean_Q are flat
        return _AttrDict({
            'f0_hz':     self._m.get('f0_hz'),
            'mean_Q':    self._m.get('mean_Q'),
            'harmonics': self._m.get('harmonics') or [],
            'fit_success': self._m.get('fit_success'),
        })

    def harmonic(self, k: int) -> _AttrDict:
        """Return the k-th harmonic (k=1,2,3) as an AttrDict, or empty."""
        hs = self.harm.harmonics or []
        for h in hs:
            if h.get('k') == k:
                return _AttrDict(h)
        return _AttrDict({})

    # ─── SNR ─────────────────────────────────────────────────────────
    @property
    def snr(self) -> _AttrDict:
        """SNR family. v4 exposes: dc_db, h1_db, h2_db, h3_db.
        Per-harmonic h*_db come from harmonics[k]['snr_db'] (v4) with
        legacy flat-field fallback for older graphs where harmonics may
        not carry per-harmonic SNR."""
        h1 = self.harmonic(1).snr_db
        h2 = self.harmonic(2).snr_db
        h3 = self.harmonic(3).snr_db
        if self._version >= 4:
            dc = (self._m.get('snr') or {}).get('dc_db')
        else:
            dc = self._m.get('snr_dc_fit_db')
        # v3 legacy fallback for h1_db: if the per-harmonic SNR is
        # missing, use the flat 'snr_ac_fit_db' which was the AC-quality
        # gate (H1-dominated for narrow-band signals like cardiac).
        if h1 is None:
            legacy = self._m.get('snr_ac_fit_db')
            if legacy is not None and np.isfinite(legacy):
                h1 = float(legacy)
        return _AttrDict({
            'dc_db': dc,
            'h1_db': h1,
            'h2_db': h2,
            'h3_db': h3,
        })

    # ─── time series ─────────────────────────────────────────────────
    @property
    def Q_t(self) -> Optional[np.ndarray]:
        return self._m.get('Q_t')

    @property
    def v_t(self) -> Optional[np.ndarray]:
        """v_t (µm/s), plug/mean velocity.

        v4: recomputed from Q_t on demand as v = Q · 1e6 / (π · R²).
            Exact because v4 stores Q_t as Q_plug (matches this formula).
        v3: returns stored v_t directly.  Stored v_t is derived from
            Q_plug while stored Q_t is Q_pois, so the recompute would be
            wrong by the r_pois/r_plug ratio (≈1.47 on tile-4).
        """
        if self._version <= 3:
            v = self._m.get('v_t')
            return np.asarray(v) if v is not None else None
        Q = self.Q_t
        R = self.geom.R_um
        if Q is None or R is None or not np.isfinite(R) or R <= 0:
            return None
        return np.asarray(Q) * 1e6 / (np.pi * float(R) ** 2)

    # ─── derived quantities (recomputed on demand) ──────────────────
    @property
    def amp_Q(self) -> Optional[float]:
        """|H1| — amplitude of the fundamental harmonic."""
        return self.harmonic(1).amp

    @property
    def phase_deg(self) -> Optional[float]:
        """Phase of the fundamental harmonic in degrees."""
        phi = self.harmonic(1).phi
        return float(np.degrees(phi)) if phi is not None else None

    @property
    def PI_rms(self) -> Optional[float]:
        """Pulsatility index: sqrt(Σ|H_k|²) / |mean_Q|.
        Parseval-form; matches v3 storage."""
        m = self.harm.mean_Q
        if m is None or not np.isfinite(m) or abs(m) < 1e-9:
            return None
        amp2 = sum(
            (h.get('amp') or 0.0) ** 2
            for h in (self.harm.harmonics or [])
        )
        return float(np.sqrt(amp2) / abs(m))


# ─── edge-level canonical measurement picker ─────────────────────────

def best_measurement(edge_data: dict) -> Optional[MeasurementView]:
    """Return the canonical MeasurementView for an edge.

    v4: uses edge['best_measurement_idx'].
    v3: falls back to highest-SNR tile via snr_harm_fit_db / snr_ac_fit_db.
    Returns None if the edge has no measurements.
    """
    mp = edge_data.get('measurements_piv') or []
    if not mp:
        return None

    idx = edge_data.get('best_measurement_idx')
    if isinstance(idx, (int, np.integer)) and 0 <= idx < len(mp):
        return MeasurementView(mp[idx])

    # v3 fallback: pick by SNR
    def _snr_key(m):
        for k in ('snr_harm_fit_db', 'snr_ac_fit_db'):
            v = m.get(k)
            if v is not None and np.isfinite(v):
                return float(v)
        return -np.inf

    best = max(mp, key=_snr_key)
    return MeasurementView(best)


# Public alias — matches the natural mental model:
# `MeasurementView.best(edge)` reads like "get the best measurement."
MeasurementView.best = staticmethod(best_measurement)
