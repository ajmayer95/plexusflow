"""Read-only mosaic viewer — exploration only, no editing or compute.

Designed for two audiences:

  1. **Intern onboarding** — they can see the mosaic, color edges by any
     measured field, click an edge to inspect its Q(t) and metadata.  No
     buttons that change the graph; no compute that takes minutes.

  2. **Publication / data sharing** — a single self-contained Qt+napari
     app a reviewer can launch on a gpickle to explore the underlying
     measurements without learning the analysis pipeline.

Run from CLI:

    python -m pertile.viewer.mosaic_readonly_app <graph.gpickle> \
        [--tiff <stitched.tif>] [--initial-field mean_Q]

Or from Python:

    from plexusflow.viewer.app import ReadOnlyMosaicViewer
    v = ReadOnlyMosaicViewer(graph_path, tiff_path=None,
                              initial_field='mean_Q')
    v.run()
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# ── PerTileFlow shim ──────────────────────────────────────────────────
# Simulate tab still uses pertile.analysis.transmission_line (~6k lines,
# port deferred per memory/plexusflow_pertile_port_2026-08-21).  Make
# `pertile` importable from any CWD by prepending the PerTileFlow root
# to sys.path here.  Candidate paths cover both the Drive checkout and
# an optional PERTILEFLOW_ROOT env override.  Silently no-ops on machines
# where the checkout isn't present.
_pertile_candidates = [
    os.environ.get('PERTILEFLOW_ROOT'),
    str(Path.home() / 'Library' / 'CloudStorage'
        / 'GoogleDrive-amayer@flatironinstitute.org'
        / 'My Drive' / 'Python' / 'Yolk Sac Code' / 'PerTileFlow'),
    str(Path(__file__).resolve().parents[4] / 'PerTileFlow'),
]
for _cand in _pertile_candidates:
    if _cand and Path(_cand).joinpath('pertile', '__init__.py').exists():
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
from typing import Optional, Tuple, List, Dict, Any

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ── macOS Qt5 QComboBox native-popup segfault workaround ─────────────────
# On recent macOS (Darwin 24/25/26), PyQt5's cocoa platform plugin crashes with
# EXC_BAD_ACCESS inside QComboBox::showPopup() → libqcocoa when a combo opens
# its NATIVE NSMenu popup.  Which popup is used is decided in showPopup() from
# the style hint SH_ComboBox_UseNativePopup (the macOS style returns True) —
# BEFORE the view matters, so swapping the view does nothing.  Override just
# that one hint via a QProxyStyle (delegates everything else, so the look is
# unchanged) and install it on every combo at construction.  Covers the combos
# built lazily per-tab and the ``QComboBox as _QComboBox`` aliases (same class).
# Only QMacStyle returns true for the (Qt5-unexposed) SH_ComboBox_UseNativePopup
# hint; Fusion — via QCommonStyle — returns false, so a Fusion-styled combo uses
# the Qt-drawn popup and never touches libqcocoa.  Give each combo its own Fusion
# QStyle (parented so it stays alive) — surgical: only combos change style, the
# rest of the app keeps the native macOS look.
try:
    from PyQt5.QtWidgets import QComboBox as _QCB, QStyleFactory as _QSF

    if not getattr(_QCB, "_plexus_popup_patched", False):
        _orig_qcb_init = _QCB.__init__

        def _plexus_qcb_init(self, *a, **k):
            _orig_qcb_init(self, *a, **k)
            try:
                _st = _QSF.create("Fusion")
                if _st is not None:
                    _st.setParent(self)      # keep the style alive with the combo
                    self.setStyle(_st)
            except Exception:
                pass

        _QCB.__init__ = _plexus_qcb_init
        _QCB._plexus_popup_patched = True
except Exception:
    pass

# v4 migration: canonical accessor for measurements_piv entries + edges.
# Legacy flat fields (m['mean_Q'], m['R_um_tile'], ...) and edge *_piv
# promotions (ed['mean_Q_piv'], ...) are being removed downstream; go
# through MeasurementView so a schema bump doesn't require re-editing
# every read site.
from ..mosaic.measurement_view import MeasurementView


# ── Defaults / constants ───────────────────────────────────────────────
# Default fields to expose in the dropdown (subset of what PIV stores).
# Order matters — first ones are most useful.
DEFAULT_FIELDS: List[str] = [
    # PIV pipeline is the live truth (kymo pipeline retired).  All
    # measurement-derived fields use the `_piv` aggregation; SNR uses
    # the two fit-residual metrics added in the 2026-06-04 reanalysis.
    'mean_Q_piv', 'amp_Q_piv', 'PI_piv', 'phase_piv',
    # H2 amplitude — useful alongside amp_Q_piv for distortion analyses.
    'amp_Q_h2_piv',
    # SNRs from fit-residual power (dB):
    #   harm-fit = full Wiki SNR (DC + AC as signal)   → mean Q inference
    #   ac-fit   = AC-only SNR (drops DC from signal)  → cardiac waveform
    'snr_harm_fit_db_piv', 'snr_ac_fit_db_piv',
    # Waveform-shape decomposition (added by qt_refit, June 2026):
    #   pulse_frac = AC / (DC + AC) — overall pulsatility
    #   H1_frac    = ½|Q₁|² / P_osc  — fundamental's share of AC content
    #   H2_frac    = ½|Q₂|² / P_osc  — second harmonic's share
    # Higher harmonics (amp_Q_h3_piv, phase_h2/h3_piv, H3_frac_piv) are
    # written to edges by qt_refit but kept out of this default list to
    # avoid bloating the dropdown.  Add them here if a specific analysis
    # needs them on hover/color.
    'pulse_frac_piv', 'H1_frac_piv', 'H2_frac_piv',
    'f0_hz_piv', 'v_max_piv',
    # geometry from edge attrs (pipeline-neutral)
    'radius', 'length',
    # viscous dissipation Φ = r·L·⟨Q²⟩ (W).  Available for both
    # measured (precomputed from PIV Q_t + geometry) and sim (from
    # `result.dissipation`).  Sim path goes via SIM_FIELD_MAP.
    'dissipation',
    # AC-only (pulsatile) dissipation Φ_AC = R_e · ½·Σ|Q_Hk|² (~nW).
    # Uses tile-corrected conductance_piv and best-tile harmonic amps.
    # Different from `dissipation` (which includes DC) — this isolates
    # where pulsatile mechanical work is being absorbed by viscous
    # heating + wall storage in the network.
    'dissipation_AC_piv',
]
# Categorical field handled specially (discrete per-class colors):
CATEGORICAL_FIELDS = ['quality_tier_piv', 'harmonic_class']
# Per-edge simulation outputs (in-memory only, written by
# `_run_simulation`, never persisted to the gpickle).  Stored on edges
# under the keys below; the Source toggle in the View tab decides
# whether the dropdown fields ('mean_Q', 'amp_Q', 'PI', 'phase') resolve
# to measured or sim values.  Q magnitudes are stored as |.| so they
# match the measured `mean_Q ≥ 0` convention (memory/methods.md).
# ── Field composition (4-selector model) ─────────────────────────────
# The View tab composes the colormap field from four selectors:
#   Source    — Measured / Simulated   (existing radio)
#   Quantity  — Q / P                  (P only valid in Simulated)
#   Property  — Magnitude / Phase / Resolution / Mean / PI / Dissipation
#               / Pressure drop / Radius / Length / Harmonic class
#   Harmonic  — DC / H1 / H2 / H3      (only for harmonic-keyed properties)
# Each property is tagged with the list of (source, quantity) tuples
# for which it is valid; `_combo_valid` enforces the matrix.
PROPERTY_DEFS = [
    # (key,           display label,          is_harmonic_keyed, valid_in)
    ('magnitude',     'Magnitude |·|',        True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P'),
         ('sim_carved', 'Q')}),
    # Signed value (real part, NO |·|) — sim only.  The point is DC pressure,
    # where the gauge can dip below 0; |·| would hide the sign.  Rendered on a
    # diverging colormap centred at 0.  (At AC harmonics this is the in-phase
    # component, whose sign depends on the solver phase reference.)
    ('signed',        'Signed (no |·|)',      True,
        {('sim', 'Q'), ('sim', 'P')}),
    ('phase',         'Phase ∠·',             True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P'),
         ('sim_carved', 'Q')}),
    # (Resolution Z = amp/SE removed — same ranking as SNR (dB) for
    # fixed-N, so redundant from the user's perspective.  Z is still
    # computed internally by _harmonic_fit_full and used by the
    # Resolved-harmonic-class categorical view, which is a detection-
    # significance question — the right place for Z.)
    # Per-harmonic SNR from fit_harmonics — populated on canonical
    # graphs as Q_DC_snr_db / Q_H1_snr_db / Q_H2_snr_db / Q_H3_snr_db.
    # Useful for filtering down to vessels with clean cardiac content
    # at a specific harmonic.
    ('snr',           'SNR (dB)',             True,
        {('measured', 'Q')}),
    # `mean` dropped — redundant with `magnitude @ DC` for measured Q
    # and sim Q (both anchored to |Q̄| ≥ 0).  Sim P loses access to the
    # signed gauge DC, which is fine — pressure_drop carries the
    # meaningful absolute differential and gauge polarity isn't a daily
    # diagnostic.
    # PI per harmonic: 2·|Q_Hk| / |Q_DC| (Gosling-style peak-to-peak/mean).
    # PI@DC is undefined (returned None).  PI@H1 is the classic PI;
    # PI@H2 / PI@H3 measure higher-harmonic content as a fraction of mean.
    ('PI',            'Pulsatility index',    True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P'),
         ('sim_carved', 'Q')}),
    # OSI (oscillation index, Q-form).  OSI_Q = ½·(1 − |∫Q|/∫|Q|) over
    # one period — aggregate (NOT harmonic-keyed).  0 = unidirectional;
    # ½ = perfectly bidirectional.  Reconstructed from DC + H1..H3.
    ('OSI',           'OSI (oscillation index)', False,
        {('measured', 'Q'), ('sim', 'Q')}),
    # Heart-rate fundamental.  Measured-Q reads the top-level `f0_hz`
    # attr set during analysis (== best measurement's f0); sim reads
    # the solver f0 it ran with.  Useful for spotting tiles where the
    # fit locked onto an aliased peak or a slightly different rate.
    ('frequency',     'Fundamental f₀ (Hz)',  False,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    # Aggregate canonical SNRs (in dB).  These match the fit_harmonics
    # convention written to per-tile measurements_piv entries and the
    # canonical edge attrs (snr_ac_fit_db, snr_harm_fit_db).
    ('snr_ac',        'SNR_AC (dB) — H1+H2+H3', False,
        {('measured', 'Q')}),
    ('snr_total',     'SNR_total (dB) — DC+AC', False,
        {('measured', 'Q')}),
    # (Var(fit)/Var(resid) total_snr removed — redundant with the dB
    # SNR family above.  The underlying `_h_total_snr` field is still
    # cached for the percentile-filter fallback on legacy graphs.)
    ('dissipation',   'Dissipation Φ',        False,
        {('measured', 'Q'), ('sim', 'Q')}),
    # AC-only dissipation: R_e · ½·Σ|Q_Hk|² (~nW) — where pulsatile
    # mechanical work is being absorbed.  Uses tile-corrected G_e.
    ('dissipation_ac', 'Dissipation Φ_AC (pulsatile only)', False,
        {('measured', 'Q'), ('sim', 'Q')}),
    # Fraction of total Q²-energy (≡ dissipation) that's pulsatile:
    # ½·Σ|Q_Hk|² / (Q_DC² + ½·Σ|Q_Hk|²).  Bounded [0, 1].
    ('pulse_frac',     '% pulsatile (AC / total energy)', False,
        {('measured', 'Q'), ('sim', 'Q')}),
    ('drop',          'Pressure drop',        False,
        {('sim', 'P')}),
    ('radius',        'Radius (px)',          False,
        {('measured', 'Q'), ('measured', 'P'),
         ('sim', 'Q'), ('sim', 'P')}),
    ('length',        'Length (px)',          False,
        {('measured', 'Q'), ('measured', 'P'),
         ('sim', 'Q'), ('sim', 'P')}),
    ('harmonic_class','Resolved-harmonic class', False,
        {('measured', 'Q')}),
    # Fraction of AC power in the selected harmonic:
    #   |H_k|² / (|H1|² + |H2|² + |H3|²).   Harmonic-keyed (DC undefined).
    ('harmonic_fraction', 'Harmonic fraction |Hk|²/ΣAC', True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P')}),
    # Arterial↔venous waveform score (AGGREGATE — same for every harmonic).
    # Cosine-similarity of an edge's AC waveform to the arterial (DA-segment
    # boundary edge) vs venous (V boundary edge) reference: + = arterial-like,
    # − = venous-like.  In sim mode the references come from the simulated
    # boundary-edge waveforms.
    ('waveform_av',   'Waveform A↔V (+art / −ven)', False,
        {('measured', 'Q'), ('sim', 'Q')}),
]
# Every measured-Q property is ALSO valid under 'normalized' source
# (which reads `measurements_piv_normalized` + the `_h_*_norm` cache
# populated by pertile.mosaic.tile_normalize).  Auto-extend `valid_in`
# rather than duplicating tuples in every entry — this keeps a single
# source of truth for "which properties come from PIV measurements".
for _key, _label, _keyed, _valid_in in PROPERTY_DEFS:
    if ('measured', 'Q') in _valid_in:
        _valid_in.add(('normalized', 'Q'))
del _key, _label, _keyed, _valid_in

HARMONIC_KEYS = ['DC', 'H1', 'H2', 'H3']
HARMONIC_LABELS = {'DC': 'DC', 'H1': 'H₁', 'H2': 'H₂', 'H3': 'H₃'}
# Phase at DC is undefined; harmonic-class is already H1-resolved → skip.
def _safe_float(val) -> Optional[float]:
    """Coerce a candidate value to float, returning None for missing /
    non-numeric / non-finite inputs.  Used by the field resolver."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _place_figure_top_right(fig, margin: int = 12) -> None:
    """Resize a matplotlib Qt-backend figure window to its `figsize` in
    pixels and move it to the top-right corner of the primary screen.

    Forces `WindowNoState` first to undo any maximized state inherited
    from the OS or a prior session (macOS in particular likes to restore
    fullscreen on `show()`).  No-op on non-Qt backends.
    """
    mgr = getattr(fig.canvas, 'manager', None)
    if mgr is None:
        return
    win = getattr(mgr, 'window', None)
    if win is None or not hasattr(win, 'move'):
        return
    try:
        from qtpy.QtWidgets import QApplication
        from qtpy.QtCore import Qt as _Qt
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        # Drop any maximized / fullscreen state inherited from elsewhere.
        try:
            win.setWindowState(_Qt.WindowNoState)
        except Exception:
            pass
        # Resize to the figsize-derived pixel footprint.
        try:
            w_in, h_in = fig.get_size_inches()
            dpi = float(fig.dpi)
            w_px = int(round(w_in * dpi))
            h_px = int(round(h_in * dpi))
            win.resize(w_px, h_px)
        except Exception:
            pass
        # frameGeometry is only meaningful after the window has been
        # mapped at least once; QApplication.processEvents() lets the
        # WM finalize geometry so the right-edge alignment is accurate.
        try:
            QApplication.processEvents()
        except Exception:
            pass
        try:
            fw = win.frameGeometry().width()
        except Exception:
            fw = win.width()
        x = avail.x() + avail.width() - fw - margin
        y = avail.y() + margin
        win.move(int(x), int(y))
    except Exception:
        pass


def _osi_from_harmonics(dc, amps_phi, n_samples: int = 200):
    """OSI_Q = ½ · (1 − |∫₀ᵀ Q dt| / ∫₀ᵀ |Q| dt)  reconstructed over one
    period from the stored harmonics.

    `amps_phi`: iterable of (amp_k, phi_k) for k = 1, 2, 3, ... such that
        Q(t) = dc + Σ_k amp_k · cos(k·ω·t + phi_k)

    Returns None if the reconstructed waveform has zero magnitude (no
    information to decide direction).  OSI ranges [0, ½]:
      0   : pure unidirectional flow (|∫Q| = ∫|Q|)
      ½   : perfectly bidirectional flow (∫Q = 0)
    """
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    Q = np.full(n_samples, float(dc), dtype=float)
    for k, (amp, phi) in enumerate(amps_phi, start=1):
        Q = Q + float(amp) * np.cos(2.0 * np.pi * k * t + float(phi))
    mean_Q = float(np.mean(Q))
    mean_abs_Q = float(np.mean(np.abs(Q)))
    if mean_abs_Q < 1e-30 or not np.isfinite(mean_abs_Q):
        return None
    return 0.5 * (1.0 - abs(mean_Q) / mean_abs_Q)


def _combo_valid(source: str, quantity: str, prop_key: str,
                  harmonic: str) -> bool:
    pdef = next((p for p in PROPERTY_DEFS if p[0] == prop_key), None)
    if pdef is None:
        return False
    _, _, is_harmonic, valid_in = pdef
    if (source, quantity) not in valid_in:
        return False
    # Phase at DC is mathematically undefined; PI ("pulsatile / mean")
    # measures non-DC content, so attaching it to the DC slot is
    # conceptually empty.  Grey both at DC regardless of whether they
    # are harmonic-keyed in the property table.
    if harmonic == 'DC' and prop_key in ('phase', 'PI', 'harmonic_fraction'):
        return False
    return True


SIM_FIELD_MAP: Dict[str, str] = {
    'mean_Q':          '_sim_tmp_mean_Q',
    'amp_Q':           '_sim_tmp_amp_Q',
    'PI':              '_sim_tmp_PI',
    'phase':           '_sim_tmp_phase_Q',
    'dissipation':     '_sim_tmp_dissipation', # from result.dissipation
    'pressure_mean':   '_sim_tmp_pressure_mean', # sim-only, P_mid,DC (Pa)
    'pressure_amp':    '_sim_tmp_pressure_amp',  # sim-only, |P_mid,H1| (Pa)
    'pressure_amp_H2': '_sim_tmp_pressure_amp_H2', # sim-only, |P_mid,H2| (Pa)
    'pressure_phase':  '_sim_tmp_phase_P',       # sim-only
    'pressure_drop':   '_sim_tmp_pressure_drop', # sim-only, |P_u_DC − P_v_DC| (Pa)
    # Geometry overrides — populated only when "Uniform conductance" is
    # checked at sim time, so radius/length fields show the override
    # value in sim mode.  Heterogeneous sim → keys absent → fall through
    # to the measured top-level attrs.
    'radius':          '_sim_tmp_radius',
    'length':          '_sim_tmp_length',
}
# Fields the comparison plot should treat as pressure (P-mode).
SIM_PRESSURE_FIELDS = {'pressure_mean', 'pressure_amp', 'pressure_amp_H2',
                       'pressure_phase'}
# Storage-only keys (not in any dropdown).
SIM_INTERNAL_KEYS: List[str] = [
    '_sim_tmp_harmonics',     # complex Q coeffs [DC, H1, H2, H3] per edge
    '_sim_tmp_p_harmonics',   # complex midpoint-P coeffs per edge
    '_sim_tmp_f0_hz',         # solver f0 for that edge
    # `_sim_tmp_radius` / `_sim_tmp_length` are listed in SIM_FIELD_MAP
    # (dropdown override), so they're already cleaned via that path.
]
# Discrete colour map for the resolved-harmonics class.
HARMONIC_CLASS_COLORS = {
    0: '#666666',   # DC only — no resolved pulsatile component
    1: '#1f77b4',   # DC + H1
    2: '#ffbf00',   # DC + H1 + H2
    3: '#d62728',   # DC + H1 + H2 + H3
}

# Raw video frame dimensions used to size tile boundaries on the mosaic.
# This is the imaging-pipeline default; if your videos have different
# native dimensions, pass them in or set them as class attrs.
TILE_RAW_HEIGHT = 704
TILE_RAW_WIDTH = 640

QUALITY_RANK = {'A': 3, 'B': 2, 'C': 1, 'X': 0}
TIER_COLORS = {'A': '#00ff00', 'B': '#ffff00', 'C': '#ff8800', 'X': '#888888'}
# Per-harmonic SNR threshold for "harmonic is real, not noise".
# Per-harmonic detection significance Z = |Q̂|/σ is the underlying
# reliability surface.  We cache four layers (Z_DC, Z_H1, Z_H2, Z_H3)
# per edge and threshold at *display time* — no precomputed tiers.
# Categories are just level sets of these layers and live in the
# colormap slider, never in the cache.
#
# DC and AC have different null distributions, so when you want a
# common threshold axis convert each layer to its p-value before
# comparing:
#   AC (Hk): Â/σ ~ Rayleigh        → P(>z) = exp(-z²/2)
#   DC:      |a₀|·√N/σ ~ folded-normal → P(>z) = erfc(z/√2)
HARMONIC_SNR_THRESHOLD = 3.0  # legacy figure-annotation reference; not used by gates.
# Cache-schema version for the `_h_*` per-edge attributes.  Bump
# whenever the meaning of cached values changes (e.g. σ std→MAD).
# An edge whose `_h_cache_ver` is missing or less than this constant
# is re-computed at startup.
HARMONIC_CACHE_VERSION = 2


def _harmonic_snrs(Q_t: np.ndarray, f0: float,
                    dt: float = 1.0/250) -> Optional[Dict[str, float]]:
    """Per-harmonic Z-statistics: Z = Â/SE.  σ from MAD (robust to
    motion-spike outliers), SE_amp = σ·√(2/N), SE_dc = σ/√N.

    Z is the raw ratio (Â, not the debiased amplitude) so the Rayleigh
    null interpretation applies: under H₀, Z ~ Rayleigh(σ=1) with
    E[Z] ≈ 1.25 and P(Z > z) = exp(−z²/2).  Threshold tiers:
        Z ≥ 4 excellent, ≥ 3 good, ≥ 2 marginal, < 2 unresolved.

    Returns dict {DC, H1, H2, H3, sigma, r2} or None on failure.
    `sigma` is the MAD-based estimate (not std).
    """
    try:
        from .harmonic import fit_harmonics  # type: ignore
    except ImportError:
        try:
            from ..analyze.harmonic import fit_harmonics
        except ImportError:
            return None
    try:
        hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                            loss='huber', include_dc=True)
    except Exception:
        return None
    resid = np.asarray(hr.get('resid', []), dtype=float)
    resid_ok = resid[np.isfinite(resid)] if resid.size else None
    if resid_ok is None or resid_ok.size == 0:
        return None
    med = float(np.median(resid_ok))
    sigma = 1.4826 * float(np.median(np.abs(resid_ok - med)))
    if not (np.isfinite(sigma) and sigma > 0):
        # MAD can collapse to 0 if >50% of residuals are exactly equal —
        # fall back to std so SE remains well-defined.
        sigma = float(np.std(resid_ok))
    N = len(Q_t)
    se_dc = sigma / np.sqrt(N) if N > 0 else np.inf
    se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else np.inf
    z_dc = abs(float(hr['a0'])) / max(se_dc, 1e-30)
    z = {1: 0.0, 2: 0.0, 3: 0.0}
    for h in hr['harmonics']:
        k = h.get('k')
        if k in z:
            z[k] = float(h['amp']) / max(se_amp, 1e-30)
    return {
        'DC': z_dc, 'H1': z[1], 'H2': z[2], 'H3': z[3],
        'sigma': sigma, 'r2': float(hr.get('r2', float('nan'))),
    }


def _harmonic_fit_full(Q_t: np.ndarray, f0: float,
                        dt: float = 1.0/250) -> Optional[Dict[str, float]]:
    """Like _harmonic_snrs but returns amplitudes + phases too.  σ from
    MAD (robust).  Adds `total_snr` = Var(fit)/Var(resid) — the
    per-vessel "how clean overall" scalar that's invariant to whether
    the waveform is H1-only or harmonic-rich.

    Returns dict with keys:
        amp_DC, Z_DC
        amp_H{k}, phase_H{k}, Z_H{k}   for k=1,2,3
        sigma, r2, total_snr
    """
    try:
        from ..analyze.harmonic import fit_harmonics
    except ImportError:
        return None
    try:
        hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                            loss='huber', include_dc=True)
    except Exception:
        return None
    resid = np.asarray(hr.get('resid', []), dtype=float)
    resid_ok = resid[np.isfinite(resid)] if resid.size else None
    if resid_ok is None or resid_ok.size == 0:
        return None
    med = float(np.median(resid_ok))
    sigma = 1.4826 * float(np.median(np.abs(resid_ok - med)))
    if not (np.isfinite(sigma) and sigma > 0):
        sigma = float(np.std(resid_ok))
    # Var(fit)/Var(resid) uses ordinary variance for both — robust σ
    # only enters via the SE denominators below.
    signal_fit = np.asarray(hr.get('signal', []), dtype=float)
    sig_ok = signal_fit[np.isfinite(signal_fit)] if signal_fit.size else None
    total_snr = float('nan')
    if sig_ok is not None and sig_ok.size and resid_ok.size:
        var_r = float(np.var(resid_ok))
        if var_r > 1e-30:
            total_snr = float(np.var(sig_ok) / var_r)
    N = len(Q_t)
    se_dc = sigma / np.sqrt(N) if N > 0 else np.inf
    se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else np.inf
    a0 = float(hr['a0'])
    out = {
        'amp_DC': a0,
        'Z_DC': abs(a0) / max(se_dc, 1e-30),
        'sigma': sigma,
        'r2': float(hr.get('r2', float('nan'))),
        'total_snr': total_snr,
    }
    for k in (1, 2, 3):
        out[f'amp_H{k}'] = 0.0
        out[f'phase_H{k}'] = 0.0
        out[f'Z_H{k}'] = 0.0
    for h in hr['harmonics']:
        k = h.get('k')
        if k in (1, 2, 3):
            out[f'amp_H{k}'] = float(h['amp'])
            out[f'phase_H{k}'] = float(h['phi'])
            out[f'Z_H{k}'] = float(h['amp']) / max(se_amp, 1e-30)
    return out


def _harmonic_class(snrs: Dict[str, float],
                     threshold: float = HARMONIC_SNR_THRESHOLD) -> int:
    """Highest k such that every Hk' for k' ≤ k passes the threshold.
    Returns 0 if even H1 fails (signal is essentially DC + noise)."""
    if snrs is None:
        return 0
    n = 0
    for k in (1, 2, 3):
        if snrs.get(f'H{k}', 0.0) >= threshold:
            n = k
        else:
            break
    return n


def _harmonic_class_label(n: int) -> str:
    """Human-readable resolved-harmonics label."""
    if n == 0: return 'DC only'
    if n == 1: return 'DC + H1'
    if n == 2: return 'DC + H1 + H2'
    return 'DC + H1 + H2 + H3'
# Color for edges with no usable measurement (missing PIV, gated/flagged,
# or a field value that's NaN/inf).  Dim enough to read as "no data,
# present for topology only" but visible against the dark mosaic image.
NO_DATA_COLOR = '#555555'
RADIAL_UNCERTAIN_COLOR = '#ff8800'   # orange overlay for edges tagged
                                       # ``radial_uncertain=True``
FLOW_MISMATCH_COLOR = '#ff00ff'      # magenta overlay for edges whose flow
                                       # direction disagrees across high-SNR tiles


def _measurement_usable(m: Optional[dict]) -> bool:
    """True if a PIV measurement is a 'real' (non-garbage) record.

    Schema-agnostic: works on both legacy (quality_tier present) and
    canonical (quality_tier dropped) graphs.

    On canonical: usable if Q_t exists and Q_H1_snr_db is finite.
    On legacy:    usable if quality_tier ≠ 'X' and fit_success.
    """
    if m is None:
        return False
    mv = MeasurementView(m)
    # v4 migration: was m.get('fit_success', True). MeasurementView.fit_success
    # defaults False when the field is absent, so preserve the "assume ok
    # unless explicitly failed" default here by falling back to the raw dict.
    if 'fit_success' in m and not mv.fit_success:
        return False
    # Tier gate — computed on the fly if not stored.
    try:
        from ..analysis.quality import quality_tier as _qt_of
        if _qt_of(m) == 'X':
            return False
        return True
    except Exception:
        pass
    # Canonical gate: require Q_t + finite H1 SNR
    if mv.Q_t is None:  # v4 migration: was m.get('Q_t') is None
        return False
    snr = mv.snr.h1_db  # v4 migration: was m.get('Q_H1_snr_db', m.get('snr_h1_db'))
    if snr is None or not np.isfinite(float(snr)):
        # Fallback: accept anything that has Q_t.  The precompute will
        # try a refit and reject downstream if it fails.
        return True
    return True


# ── Helpers ────────────────────────────────────────────────────────────
def _best_measurement(piv_list: Optional[list]) -> Optional[dict]:
    """Pick the highest-SNR measurement for one edge.

    Schema-agnostic ranking — prefers canonical Q_H1_snr_db when present,
    falls back to (quality_tier, snr_f0) on legacy graphs.
    """
    if not piv_list:
        return None
    # v4 migration: keep "default True" fit_success semantics via raw dict,
    # since MeasurementView.fit_success defaults False on missing field.
    candidates = [m for m in piv_list
                  if ('fit_success' not in m) or MeasurementView(m).fit_success]
    if not candidates:
        return None
    def _rank(m):
        mv = MeasurementView(m)
        # Canonical: just H1 SNR (highest wins)
        # v4 migration: was m.get('Q_H1_snr_db', m.get('snr_h1_db'))
        snr_h1 = mv.snr.h1_db
        if snr_h1 is not None and np.isfinite(float(snr_h1)):
            return (1, float(snr_h1))   # tier 1 = canonical
        # Legacy: tier (computed on the fly if not stored) + snr_f0
        try:
            from ..analysis.quality import quality_tier as _qt_of
            tier = _qt_of(m)
        except Exception:
            tier = m.get('quality_tier', 'X')
        # v4 migration: snr_f0 → snr.h1_db (same physical quantity in v3)
        return (0,
                QUALITY_RANK.get(tier, 0),
                (mv.snr.h1_db or 0))
    return max(candidates, key=_rank)


def _resolve_edge_field(G, u, v, field: str,
                        source: str = 'measured') -> Optional[float]:
    """Return the scalar value of `field` for edge (u, v), or None.

    `source` selects which underlying data path is used.  For
    `source='sim'` the field name is mapped via SIM_FIELD_MAP to the
    in-memory `_sim_tmp_*` keys; geometry / categorical / non-mappable
    fields fall through to the measured-side logic regardless of source.
    """
    d = G.edges[u, v]
    if source == 'sim' and field in SIM_FIELD_MAP:
        val = d.get(SIM_FIELD_MAP[field])
        if val is not None:
            try:
                return float(val) if np.isfinite(val) else None
            except (TypeError, ValueError):
                return None
        # Fall through to the measured branch if the sim override is
        # absent (e.g. radius/length in heterogeneous sim mode).
    # Geometry, precomputed harmonic_class — top-level edge attrs.
    if field in ('radius', 'length', 'harmonic_class'):
        val = d.get(field)
        if val is None:
            return None
        try:
            return float(val) if np.isfinite(val) else None
        except (TypeError, ValueError):
            return None
    # Measured dissipation — precomputed once at load time and stored
    # in-memory as `_meas_dissipation` so the colormap lookup is O(1).
    if field == 'dissipation':
        val = d.get('_meas_dissipation')
        if val is None:
            return None
        try:
            return float(val) if np.isfinite(val) else None
        except (TypeError, ValueError):
            return None
    if field == 'dissipation_AC_piv':
        # Per-edge AC-only viscous dissipation Φ_AC = R_e · ½·Σ_k |Q_Hk|²
        # (Watts).  R_e = 1/G_e where G_e = conductance_piv = πR⁴/(8μL).
        # Uses TILE-CORRECTED conductance (set by patch_R_um_tile) and
        # the best-tile harmonic amplitudes.  Returns ~nW magnitude on
        # yolk-sac vessels (Pa·s/nL × (nL/s)² = Pa·nL/s = 1e-9 W).
        # v4 migration: was d.get('conductance_piv'). Compute from geometry
        # via MeasurementView (πR⁴/(8μL) with μ = 3.5e-3 Pa·s).
        mv = MeasurementView.best(d)
        if mv is None:
            return None
        R_um = mv.geom.R_um
        L_um = mv.geom.L_um
        if R_um is None or L_um is None:
            return None
        try:
            G_e_f = math.pi * float(R_um) ** 4 / (8.0 * 3.5e-3 * float(L_um))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(G_e_f) or G_e_f <= 0:
            return None
        m = mv._m  # keep raw dict handle for legacy Q_H{k}_amp fallback below
        harms = mv.harm.harmonics or []  # v4 migration: was m.get('harmonics')
        if harms:
            sum_sq = sum(float(h.get('amp', 0.0) or 0.0) ** 2
                          for h in harms[:3])
        else:
            # Legacy canonical: Q_H1_amp + Q_H2_amp + Q_H3_amp at top level
            sum_sq = 0.0
            for k in (1, 2, 3):
                a = m.get(f'Q_H{k}_amp')
                if a is not None and np.isfinite(a):
                    sum_sq += float(a) ** 2
        if sum_sq <= 0:
            return None
        phi_ac = 0.5 * sum_sq / G_e_f
        return float(phi_ac) if np.isfinite(phi_ac) else None
    # All other fields come from the best PIV measurement.  For
    # source='normalized', read from the parallel `_normalized` list
    # (populated by pertile.mosaic.tile_normalize).
    meas_key = ('measurements_piv_normalized' if source == 'normalized'
                else 'measurements_piv')
    m = _best_measurement(d.get(meas_key))
    if m is None:
        return None
    if field in m:
        val = m[field]
        try:
            v_ = float(val)
            return v_ if np.isfinite(v_) else None
        except (TypeError, ValueError):
            return None
    return None


class _TiedVShim:
    """Context manager: temporarily add high-conductance phantom edges
    between all V-node pairs, so `P_V1(ω) ≈ P_V2(ω)` at every harmonic
    in the solve.  Removes shims on exit.

    Physically: the two vitelline veins drain into a shared atrial
    chamber, so their pressures should equal at every ω.  The shim
    enforces this numerically without changing the solver.

    Shim edge has very small effective resistance so its admittance
    dominates.  Chosen R = 1e-6 µm-scale so the shim admittance is
    ~10^6 × any real vessel — pressure difference across the shim is
    driven to zero, without introducing measurable flow.
    """
    def __init__(self, G, sink_nodes: list, r_um: float = 1e-6,
                 l_um: float = 1e-3):
        self.G = G
        self.pairs = []
        for i in range(len(sink_nodes)):
            for j in range(i + 1, len(sink_nodes)):
                self.pairs.append((sink_nodes[i], sink_nodes[j]))
        self.r_um = r_um
        self.l_um = l_um

    def __enter__(self):
        # Add phantom edges with tiny R and small L (huge conductance)
        for u, v in self.pairs:
            if self.G.has_edge(u, v):
                continue  # don't clobber real edges
            self.G.add_edge(u, v,
                            R_um_tile=self.r_um, L_um_tile=self.l_um,
                            R_um_piv=self.r_um, L_um_piv=self.l_um,
                            radius=self.r_um, length=self.l_um,
                            _shim_tied_v=True)
        return self

    def __exit__(self, *args):
        for u, v in self.pairs:
            ed = self.G.get_edge_data(u, v)
            if ed is not None and ed.get('_shim_tied_v'):
                self.G.remove_edge(u, v)


def _actual_boundary_Q_dc(G, node, edge_flows) -> Optional[float]:
    """Sum edge DC flows at a boundary node, with correct sign per
    edge-orientation, to get the true net Q at that node.

    `boundary_Q_harmonics` in the solver result is (bug-adjacent) just
    the INPUT bc_harmonics echoed back — it is NOT the solver's computed
    flow.  For a genuine Pass-1 prediction we have to compute Q at the
    boundary node ourselves from the edge-flow field.

    Convention: `edge_flows[(u, v)][0].real` = DC flow from u to v.
    At a boundary node n, net flow INTO the network is:
        Σ over incident edges (n, nbr):  +Q if edge stored as (n, nbr)
                                          −Q if edge stored as (nbr, n)
    That gives +Q at sources (blood pushed in) and −Q at sinks (drained).
    """
    net = 0.0
    saw_any = False
    for nbr in G.neighbors(node):
        Q = edge_flows.get((node, nbr))
        if Q is not None:
            net += float(np.asarray(Q, dtype=complex)[0].real)
            saw_any = True
        else:
            Q = edge_flows.get((nbr, node))
            if Q is not None:
                net -= float(np.asarray(Q, dtype=complex)[0].real)
                saw_any = True
    return net if saw_any else None


def _print_dc_bc_diagnostic(G, source_nodes, sink_nodes, result_p1) -> None:
    """Log per-port comparison of Pass-1 predicted DC vs measured DC at
    each source/sink node.

    Answers: "does the network geometry, driven by ΔP alone, reproduce
    the measured DC split at boundaries?"  Big systematic difference
    per class (A or V) → the geometry model doesn't fit that side.

    Prints to stdout so it shows up in the console next to the sim log.
    """
    print()
    print('=' * 70)
    print('  DC-BC diagnostic — Pass-1 (geometry) vs measured  (all in nL/s)')
    print('  Both are in flow-into-network convention:')
    print('    + at sources (blood pushed in), − at sinks (blood drawn out).')
    print('=' * 70)
    all_pred_A = 0.0; all_pred_V = 0.0
    for label, nodes in (('SOURCE (A)', source_nodes),
                          ('SINK   (V)', sink_nodes)):
        if not nodes:
            continue
        rows = []
        total_pred = 0.0
        total_meas = 0.0
        for n in nodes:
            bch = G.nodes[n].get('bc_harmonics')
            # NB: `result_p1.boundary_Q_harmonics` is (per solver bug) just
            # the input `bc_harmonics` echoed back, NOT the solved flow.
            # For a real "prediction" we sum edge_flows around the node.
            q_pred_dc = _actual_boundary_Q_dc(G, n, result_p1.edge_flows)
            if bch is None or q_pred_dc is None:
                rows.append((n, None, None))
                continue
            # `bc_harmonics[0]` is stored in flow-into-network convention
            # (positive at sources, negative at sinks) — matches the
            # convention of `_actual_boundary_Q_dc`.
            q_meas_dc = float(np.asarray(bch, dtype=complex)[0].real)
            rows.append((n, q_pred_dc, q_meas_dc))
            total_pred += q_pred_dc
            total_meas += q_meas_dc
        if 'SOURCE' in label:
            all_pred_A = total_pred
        else:
            all_pred_V = total_pred
        print(f'{label}:  Q_pred_pass1   Q_meas   ratio    diff')
        print('          ----------------------------------------')
        for n, qp, qm in rows:
            if qp is None or qm is None:
                print(f'  node {n:>6}:   (no bc_harmonics or pass-1 result)')
                continue
            if abs(qm) > 1e-30:
                ratio = qp / qm
            else:
                ratio = float('inf')
            print(f'  node {n:>6}:  {qp:+10.4f}   {qm:+10.4f}   '
                  f'{ratio:+7.3f}   {qp - qm:+10.4f}')
        if len(rows) > 1:
            if abs(total_meas) > 1e-30:
                ratio_tot = total_pred / total_meas
            else:
                ratio_tot = float('inf')
            print(f'   Σ total:  {total_pred:+10.4f}   {total_meas:+10.4f}   '
                  f'{ratio_tot:+7.3f}   {total_pred - total_meas:+10.4f}')
        print()
    # DC conservation check — with pure DC Dirichlet, KCL should give
    # ΣQ_in = 0 at every interior node, so the boundary flows must sum
    # to 0.  Non-zero total = numerical error or a solver-level leak.
    net_pred = all_pred_A + all_pred_V
    print(f'  DC conservation (Σ_A + Σ_V, should be 0):  {net_pred:+.6f}  '
          f'({100 * abs(net_pred) / max(abs(all_pred_A), 1e-30):.3f}% of Σ_A)')
    print('=' * 70)
    print()


def _pick_existing_dir(val):
    """Resolve a video-dir spec to the FIRST directory that actually exists on
    this machine.  `val` may be a single path/str, a list of them, or a
    comma-separated string — so one launch command works across machines
    (e.g. desktop vs laptop) by listing every candidate location; whichever is
    present locally is used.  Returns (chosen_dir_or_None, [all_candidates])."""
    if val is None:
        return None, []
    raw = val if isinstance(val, (list, tuple)) else [val]
    cands = []
    for it in raw:
        if it is None:
            continue
        parts = str(it).split(',') if isinstance(it, str) else [it]
        for part in parts:
            s = str(part).strip()
            if s:
                cands.append(Path(s).expanduser())
    chosen = next((c for c in cands if c.is_dir()), None)
    return chosen, cands


def _video_glob(pattern, tile_id):
    """Build the fallback glob for a tile's video so it RESPECTS the pattern's
    prefix (e.g. the '10x ' magnification) — otherwise a dir holding both
    '10x …loc{t}_…' and '20x …loc{t}_…' could return the wrong magnification."""
    if pattern and "loc{vid}" in pattern:
        prefix = pattern.split("loc{vid}")[0]          # e.g. "10x 250fps "
        return f"{prefix}loc{tile_id}_*"
    return f"*loc{tile_id}_*"


# ══════════════════════════════════════════════════════════════════════
# Viewer
# ══════════════════════════════════════════════════════════════════════
class ReadOnlyMosaicViewer:
    """napari-based read-only mosaic viewer with field coloring + edge
    inspection.  No buttons that modify state."""

    def __init__(
        self,
        graph_path: Path,
        tiff_path: Optional[Path] = None,
        tile_positions: Optional[Path] = None,
        video_dir: Optional[Path] = None,
        video_pattern: Optional[str] = None,
        of_video_dir: Optional[Path] = None,
        of_video_pattern: Optional[str] = None,
        initial_field: str = 'mean_Q',
        cache_harmonic_class: bool = True,
        force_recompute_harmonic_class: bool = False,
        rotate_cw: bool = False,
        rotate_ccw: bool = False,
    ):
        self.graph_path = Path(graph_path)
        self.tiff_path = Path(tiff_path) if tiff_path else None
        self.tile_positions_path = Path(tile_positions) if tile_positions else None
        # video_dir / of_video_dir may each be a list of CANDIDATE locations
        # (or a comma-separated string); use whichever exists on this machine,
        # so one command works on desktop and laptop.
        self.video_dir, self._video_candidates = _pick_existing_dir(video_dir)
        self.video_pattern = video_pattern
        if self._video_candidates and self.video_dir is None:
            print("  ⚠ Video overlay: none of the candidate dirs exist on this "
                  "machine: " + ", ".join(str(c) for c in self._video_candidates))
        elif self.video_dir is not None and len(self._video_candidates) > 1:
            print(f"  Video dir: {self.video_dir}  "
                  f"(first existing of {len(self._video_candidates)} candidates)")
        if of_video_dir is not None:
            self.of_video_dir, _ = _pick_existing_dir(of_video_dir)
        else:
            self.of_video_dir = self.video_dir
        self.of_video_pattern = of_video_pattern or self.video_pattern
        self.cache_harmonic_class = bool(cache_harmonic_class)
        self.force_recompute_harmonic_class = bool(force_recompute_harmonic_class)
        self.rotate_cw = bool(rotate_cw)
        self.rotate_ccw = bool(rotate_ccw)
        if self.rotate_cw and self.rotate_ccw:
            raise ValueError("Pass at most one of --rotate-cw / --rotate-ccw.")
        self.G = self._load_graph()
        # Normalization now lives on the same graph.  A graph that has
        # been through `pertile.mosaic.tile_normalize` carries side-by-side
        # `measurements_piv_normalized` on each edge plus a `tile_gauges`
        # dict on G.graph — the 'Normalized' data-source radio and the
        # Normalize tab detect this via `_normalization_available()`.
        n_gauges = len(self.G.graph.get('tile_gauges', {}))
        if n_gauges:
            print(f"  Loaded graph has tile_gauges for {n_gauges} tiles "
                  f"(f0_canonical = "
                  f"{self.G.graph.get('f0_canonical_hz', float('nan')):.3f} Hz)"
                  " — 'Normalized' source available")
        self.mosaic = self._load_tiff() if self.tiff_path else None
        self.tiles = self._load_tile_positions() if self.tile_positions_path else {}
        self.edge_list: List[Tuple[int, int]] = list(self.G.edges())
        # Currently-overlaid video tile (napari layer + tile_id) — only
        # one at a time; clicking a new edge replaces the previous overlay.
        self._video_layer = None
        self._video_tile_id: Optional[int] = None
        # Precompute midpoints for nearest-edge lookup.
        self._edge_midpoints = np.array([
            [0.5 * (self.G.nodes[u]['x'] + self.G.nodes[v]['x']),
             0.5 * (self.G.nodes[u]['y'] + self.G.nodes[v]['y'])]
            for (u, v) in self.edge_list
        ])
        # Precompute per-edge `harmonic_class` (0/1/2/3 = number of
        # statistically resolved cardiac harmonics).  Stored on the
        # in-memory graph only — never written back to the gpickle.
        # Cheap enough to do once at startup (~30 s for ~6k edges).
        self._precompute_harmonic_classes()
        # Precompute measured viscous dissipation per edge from PIV
        # Q_t + geometry.  Stored as `_meas_dissipation` (W) on edge
        # attrs, in-memory only.
        self._precompute_measured_dissipation()
        # ── Four-selector field model ──────────────────────────────
        # Source already added below as `self.current_source`.
        # Quantity: 'Q' or 'P'.  P only valid in Simulated source.
        # Property: see PROPERTY_DEFS.  Some are harmonic-keyed.
        # Harmonic: 'DC' / 'H1' / 'H2' / 'H3'.  Ignored for non-keyed
        # properties.
        self.current_quantity: str = 'Q'
        self.current_property: str = 'magnitude'
        self.current_harmonic: str = 'DC'
        # Backwards-compat: `current_field` is now a synthesized string
        # so legacy checks like `'phase' in field` and
        # `field in SIM_PRESSURE_FIELDS` keep working.  See the
        # `current_field` @property below.  `self.fields` is no longer
        # needed (the selectors are static) but is kept as an empty
        # list to avoid breaking anything that iterates it.
        self.fields: List[str] = []
        # initial_field may still set the starting Property if it
        # matches a known key.
        if (initial_field
                and any(p[0] == initial_field for p in PROPERTY_DEFS)):
            self.current_property = initial_field
        self.log_scale = False
        self.show_nodes = False
        self.show_kcl = False
        self._kcl_layer = None
        self._edges_layer = None
        self._nodes_layer = None
        self._click_marker_layer = None
        self._tile_boundaries_layer = None
        self._tile_labels_layer = None
        self.show_tile_boundaries = False
        self.show_tile_labels = False
        # Edge value labels — show the current property value at each
        # edge's midpoint.  Hidden by default and ALSO requires a single
        # tile to be selected (Tile filter ≠ "All tiles") to render —
        # otherwise the labels overlap into clutter at mosaic scale.
        self.show_edge_labels = False
        self._edge_labels_layer = None
        # Network-wide reliability filtering is intentionally out of the
        # UI for now — the cached per-harmonic Z layers (`_h_Z_DC`,
        # `_h_Z_HN`) are the right basis for a future continuous,
        # threshold-at-display-time filter.  See discussion thread for
        # the design (four-layer Z field, no precomputed tiers).
        # edge (u,v) → {'amp_DC', 'Z_DC', 'amp_H1', 'phase_H1', 'Z_H1', ...}
        # Populated for the currently-loaded tile only (either by the
        # "Browse tiles" video-load path, or eagerly by the tile filter
        # combo).  Used both by the time-series inspect panel for
        # cross-edge-comparable phase data, and by the tile filter to
        # restrict measured-Q colormap fields to one tile's values.
        self._tile_harmonic_cache: Dict[Tuple[int, int], dict] = {}
        # Tile filter state — None ⇒ "All tiles", colour by best
        # measurement across tiles (default).  When set to an int,
        # measured-Q fields read per-tile values from
        # `_tile_harmonic_cache`; edges with no measurement on the
        # filtered tile render as grey.  Doesn't affect sim or geometry.
        self.current_tile_filter: Optional[int] = None
        # Percentile threshold on per-edge total_snr (Var(fit)/Var(resid)).
        # Hides the bottom X % of edges by network-wide `_h_total_snr`.
        # The percentile is always computed from the graph-wide cached
        # best-of-edge values — independent of whether a tile filter is
        # active.  0 = no filtering, default.
        # Absolute-dB threshold on Q_H1_snr_db.  Sentinel −20 dB = off.
        self.total_snr_pct_filter: float = -20.0
        # (Legacy cache, kept for compatibility; no longer read.)
        self._cached_total_snr_threshold: Optional[float] = None
        # Absolute-dB threshold on Q_H2_snr_db.  Sentinel −20 dB = off.
        self.total_snr_h2_pct_filter: float = -20.0
        self._cached_total_snr_h2_threshold: Optional[float] = None
        # Percentile threshold on per-edge viscous dissipation Φ (W).
        # Hides the bottom X % of edges by network-wide `_meas_dissipation`.
        # 0 = no filtering, default.  Cached threshold + stats are
        # invalidated when the filter value or graph changes.
        self.dissipation_pct_filter: float = 0.0
        self._cached_dissipation_threshold: Optional[float] = None
        self._cached_dissipation_stats: Optional[Dict[str, float]] = None
        # Edges queued for the side-by-side comparison plot.  Each
        # entry is (u, v); duplicates ignored.  Cleared via the
        # "Clear comparison" button or when the user explicitly resets.
        self._comparison_edges: List[Tuple[int, int]] = []
        self._comparison_marker_layer = None
        self._comparison_edge_overlay_layer = None
        # Boundary nodes — arterial sources ('source') and venous sinks
        # ('sink').  Identified from the `boundary_type` node attribute
        # set by the analysis pipeline; both have a complex
        # `bc_harmonics` array (DC + H1 + H2 + H3).
        self._source_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'source']
        self._sink_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'sink']
        self._boundary_nodes = self._source_nodes + self._sink_nodes
        self._source_layer = None
        self._sink_layer = None
        print(f"  boundary: {len(self._source_nodes)} sources (A), "
              f"{len(self._sink_nodes)} sinks (V)")
        # Simulation state (forward transmission-line solve).  Results
        # live IN-MEMORY only — never persisted to the gpickle.  Keys
        # use the `_sim_tmp_` prefix so they don't collide with the
        # editing-app's stored `_sim` fields.
        self._sim_active: bool = False
        self._sim_last_D: float = float('nan')
        self._sim_last_f0_hz: float = float('nan')
        # If the loaded graph carries persisted sim results (`_sim_tmp_*`
        # on any edge, or a `sim_meta` blob on G.graph), flip _sim_active
        # so the View tab's Simulated radio is enabled from the start.
        # No new solve is triggered — the fields are just already present.
        if self.G.graph.get('sim_meta'):
            self._sim_active = True
        else:
            for _u, _v, _d in self.G.edges(data=True):
                if '_sim_tmp_harmonics' in _d or '_sim_tmp_mean_Q' in _d:
                    self._sim_active = True
                    break
        if self._sim_active:
            _meta = self.G.graph.get('sim_meta') or {}
            self._sim_last_D = float(_meta.get('D_areal', float('nan')))
            self._sim_last_f0_hz = float(_meta.get('f0_hz', float('nan')))
        # Data source for the View tab field colormap + the comparison
        # plot.  'measured' uses PIV results from `measurements_piv`;
        # 'normalized' uses tile-gauge-normalized measurements
        # (`measurements_piv_normalized`) computed by the Normalize tab
        # via `pertile.mosaic.tile_normalize`; 'sim' uses the in-memory
        # `_sim_tmp_*` outputs.  The Source radio in the View tab drives
        # this; 'normalized' is disabled until tile_gauges are populated
        # on the graph, and 'sim' is disabled until `_sim_active`.
        self.current_source: str = 'measured'
        # Mosaic height — used to invert node y so points align with the
        # image layer (mosaic uses display-y = row, graph stores math-y).
        # Falls back to max graph-y if no TIFF is loaded.
        if self.mosaic is not None:
            self.mosaic_height = int(self.mosaic.shape[0])
        else:
            ys = [self.G.nodes[n].get('y', 0) for n in self.G.nodes()]
            self.mosaic_height = int(max(ys) + 1) if ys else 0
        # Per-node "best" measured value for the currently-selected field;
        # populated by `_refresh_edges` (which already iterates edges) so
        # nodes can be colored by the mean of their incident edges.
        self._node_field_values: Dict[int, float] = {}

    # ── data loading ───────────────────────────────────────────────────
    def _meas_key(self, source: Optional[str] = None) -> str:
        """Return the per-edge measurements-list key for a source.

        'normalized' → `measurements_piv_normalized`, else
        `measurements_piv`.  Defaults to `self.current_source`.
        """
        src = source if source is not None else \
            getattr(self, 'current_source', 'measured')
        return ('measurements_piv_normalized' if src == 'normalized'
                else 'measurements_piv')

    # ── Tile-filtered measured-Q resolver ─────────────────────────────
    def _resolve_tile_filtered_value(self, u, v, d: dict,
                                       prop: str) -> Optional[float]:
        """Read the per-tile equivalent of the requested measured-Q
        property from `_tile_harmonic_cache` (populated for whichever
        tile is currently selected as the filter).  Returns None when
        the edge has no measurement on that tile, when the property
        isn't well-defined per-tile (dissipation has no clean per-tile
        equivalent), or when the harmonic fit failed for that
        measurement.
        """
        # Geometry doesn't change per tile.
        if prop == 'radius':
            return _safe_float(d.get('radius'))
        if prop == 'length':
            return _safe_float(d.get('length'))
        # Frequency — pulled directly from the matching tile's
        # measurement (the per-tile harmonic cache reused this same
        # f0 during the fit, but doesn't store it back, so it's
        # cheaper to read the source).
        if prop == 'frequency':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.harm.f0_hz)
            return None

        entry = self._tile_harmonic_cache.get((u, v))
        if entry is None:
            return None  # no measurement on the filtered tile for this edge

        if prop == 'PI':
            # Per-harmonic PI_k = 2·|Q_Hk| / |Q_DC|.  DC slot undefined.
            k_idx_pi = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_pi == 0:
                return None
            a0 = abs(float(entry.get('amp_DC', 0.0) or 0.0))
            amp_k = float(entry.get(f'amp_H{k_idx_pi}', 0.0) or 0.0)
            if a0 < 1e-15:
                return None
            return 2.0 * amp_k / a0
        if prop == 'total_snr':
            return _safe_float(entry.get('total_snr'))
        if prop == 'snr_ac':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.snr.h1_db)
            return None
        if prop == 'snr_total':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.snr.dc_db)
            return None
        if prop == 'harmonic_class':
            snrs = {'DC': float(entry.get('Z_DC', 0.0) or 0.0),
                    'H1': float(entry.get('Z_H1', 0.0) or 0.0),
                    'H2': float(entry.get('Z_H2', 0.0) or 0.0),
                    'H3': float(entry.get('Z_H3', 0.0) or 0.0)}
            return _safe_float(_harmonic_class(snrs))
        if prop == 'dissipation':
            # `_meas_dissipation` is a graph-level aggregate from PIV
            # processing; no clean per-tile equivalent exists.  Return
            # None so the field reads as "not available at this tile".
            return None
        if prop == 'pulse_frac':
            # Per-tile: ½·Σ|H_k|² / (Q_DC² + ½·Σ|H_k|²) using THIS
            # tile's amplitudes from the harmonic cache.  R cancels,
            # so this is also the per-tile dissipation AC fraction.
            try:
                a0 = float(entry.get('amp_DC', 0.0) or 0.0)
                ac_sq = sum(float(entry.get(f'amp_H{k}', 0.0) or 0.0) ** 2
                             for k in (1, 2, 3))
                ac_e = 0.5 * ac_sq
                tot = a0 * a0 + ac_e
                return float(ac_e / tot) if tot > 0 else None
            except (TypeError, ValueError):
                return None
        if prop == 'dissipation_ac':
            # Per-tile: R_e · ½·Σ|H_k|² where R_e uses the per-tile
            # tile-corrected geometry (R_um_tile, L_um_tile) on the
            # tile's measurement entry.  Units: Pa·nL/s = nW.
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id != tile_id:
                    continue
                R_um = mv_mm.geom.R_um
                L_um = mv_mm.geom.L_um
                if R_um is None or L_um is None:
                    return None
                try:
                    R_um_f = float(R_um); L_um_f = float(L_um)
                except (TypeError, ValueError):
                    return None
                if not (np.isfinite(R_um_f) and np.isfinite(L_um_f)
                        and R_um_f > 0 and L_um_f > 0):
                    return None
                # G_e in nL/(Pa·s):  π R⁴ / (8 μ L)   with μ = 3.5e-3 Pa·s,
                # R, L in µm; convert µm⁴/µm = µm³ → 1e-9 nL·m³/m³ = 1e-9 nL/...
                # Actually: π R^4 (µm⁴) / (8μ L µm) = π·R⁴/(8μL) (µm³ / (Pa·s))
                # = 1e-9 nL/(Pa·s).  So G_nL_per_Pa_s = G_µm3_per_Pa_s × 1e-9.
                G_per_Pa_s_um3 = (np.pi * R_um_f ** 4 /
                                    (8.0 * 3.5e-3 * L_um_f))
                G_nL = G_per_Pa_s_um3 * 1e-9
                if G_nL <= 0: return None
                # Σ amp_H_k² from this tile's harmonics list.
                # v4 migration: was mm.get('harmonics')
                harms = mv_mm.harm.harmonics or []
                if harms:
                    sum_sq = sum(float(h.get('amp', 0.0) or 0.0) ** 2
                                  for h in harms[:3])
                else:
                    sum_sq = 0.0
                    for k in (1, 2, 3):
                        a = mm.get(f'Q_H{k}_amp')
                        if a is not None and np.isfinite(a):
                            sum_sq += float(a) ** 2
                if sum_sq <= 0:
                    return None
                phi_ac = 0.5 * sum_sq / G_nL    # nW
                return float(phi_ac)
            return None

        # Harmonic-keyed properties — keys in the per-tile cache match
        # the layout produced by `_harmonic_fit_full`.
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        if prop == 'magnitude':
            if k_idx == 0:
                v_ = entry.get('amp_DC')
                return abs(_safe_float(v_)) if v_ is not None else None
            return _safe_float(entry.get(f'amp_H{k_idx}'))
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            return _safe_float(entry.get(f'phase_H{k_idx}'))
        if prop == 'resolution':
            if k_idx == 0:
                return _safe_float(entry.get('Z_DC'))
            return _safe_float(entry.get(f'Z_H{k_idx}'))
        if prop == 'snr':
            # Per-harmonic SNR from THIS tile's measurement (canonical
            # fields Q_DC_snr_db / Q_Hk_snr_db on the per-tile entry).
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    if k_idx == 0:
                        return _safe_float(mm.get('Q_DC_snr_db'))
                    return _safe_float(mm.get(f'Q_H{k_idx}_snr_db'))
            return None
        return None

    # ── AC-harmonic helpers (harmonic_fraction / waveform_av) ─────────
    def _edge_ac_harmonics(self, d, src, qty):
        """Complex AC harmonics [H1, H2, H3] for an edge under the given
        source/quantity, or None.  Measured-Q reads measurements_piv;
        normalized-Q reads measurements_piv_normalized; sim-Q reads
        `_sim_tmp_harmonics`; sim-P reads `_sim_tmp_p_harmonics`
        (each stored as complex [DC, H1, H2, H3])."""
        if src in ('measured', 'normalized') and qty == 'Q':
            meas_key = ('measurements_piv_normalized' if src == 'normalized'
                        else 'measurements_piv')
            m = _best_measurement(d.get(meas_key))
            if m is None:
                return None
            # Canonical fields first; fall back to v2's harmonics list.
            # v4 migration: was m.get('harmonics')
            _harms = MeasurementView(m).harm.harmonics or []
            out = []
            for k in (1, 2, 3):
                a = m.get(f'Q_H{k}_amp')
                ph = m.get(f'Q_H{k}_phi')
                if (a is None or not np.isfinite(a)) and len(_harms) >= k:
                    a = _harms[k - 1].get('amp')
                    ph = _harms[k - 1].get('phi')
                if (a is None or ph is None
                        or not (np.isfinite(a) and np.isfinite(ph))):
                    return None
                out.append(float(a) * np.exp(1j * float(ph)))
            return np.array(out, dtype=complex)
        if src == 'sim' and qty == 'Q':
            h = d.get('_sim_tmp_harmonics')
            if h is None or len(h) < 4:
                return None
            return np.asarray(h[1:4], dtype=complex)
        if src == 'sim' and qty == 'P':
            p = d.get('_sim_tmp_p_harmonics')
            if p is None or len(p) < 4:
                return None
            return np.asarray(p[1:4], dtype=complex)
        return None

    def _boundary_edge_ac(self, nodes, src):
        """AC harmonics of the first boundary edge incident to one of
        `nodes` with a usable waveform — the 'DA segment' (source side) or
        the 'V boundary edge' (sink side)."""
        if not nodes:
            return None
        for n in nodes:
            if n not in self.G:
                continue
            for nbr in self.G.neighbors(n):
                ac = self._edge_ac_harmonics(self.G.edges[n, nbr], src, 'Q')
                if ac is not None and float(np.linalg.norm(ac)) > 1e-30:
                    return ac
        return None

    def _av_reference_harmonics(self, src):
        """(A_ref, V_ref) AC-harmonic reference vectors for the waveform_av
        field: A from an arterial (source) boundary edge, V from a venous
        (sink) boundary edge.  Returns None if either is unavailable.
        Cached per source; the cache is busted on sim re-run / clear."""
        key = (src,)
        cache = getattr(self, '_av_ref_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        A = self._boundary_edge_ac(getattr(self, '_source_nodes', None), src)
        V = self._boundary_edge_ac(getattr(self, '_sink_nodes', None), src)
        refs = (A, V) if (A is not None and V is not None) else None
        self._av_ref_cache = (key, refs)
        return refs

    # ── Field resolver (4-selector model) ─────────────────────────────
    def _resolve_field_value(self, u, v) -> Optional[float]:
        """Return the per-edge scalar for the current selector state,
        or None.  Centralised lookup used by both the colormap and the
        node-averaging pass.

        Routing matrix:
          aggregate properties  → top-level edge attr lookup
          harmonic-keyed (measured) → cached `_h_amp/phase/Z_H{k}` etc.
          harmonic-keyed (sim, Q)   → derived from `_sim_tmp_harmonics`
          harmonic-keyed (sim, P)   → derived from `_sim_tmp_p_harmonics`
        Invalid combos return None (rendered grey).
        """
        if not self.G.has_edge(u, v):
            return None
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property,
                             self.current_harmonic):
            return None
        d = self.G.edges[u, v]
        prop = self.current_property
        src = self.current_source
        qty = self.current_quantity

        # ── Carved-sim source: read `_h_*_sim_carved` fields directly.
        # These are populated by the Carved Sim tab's Run button.  Only
        # Q-flavour fields are supported (sim solves for edge fluxes).
        if src == 'sim_carved' and qty == 'Q':
            k_idx = HARMONIC_KEYS.index(self.current_harmonic)
            if prop == 'magnitude':
                key = ('_h_amp_DC_sim_carved' if k_idx == 0
                       else f'_h_amp_H{k_idx}_sim_carved')
                return _safe_float(d.get(key))
            if prop == 'phase':
                if k_idx == 0: return None
                return _safe_float(d.get(f'_h_phase_H{k_idx}_sim_carved'))
            if prop == 'PI':
                if k_idx == 0: return None
                amp = _safe_float(d.get(f'_h_amp_H{k_idx}_sim_carved'))
                dc  = _safe_float(d.get('_h_amp_DC_sim_carved'))
                if amp is None or dc is None or dc == 0: return None
                val = 2.0 * abs(amp) / abs(dc)
                return val if np.isfinite(val) else None
            # Fallthrough: unsupported prop for sim_carved → None
            return None

        # Tile-filter override: when a tile is selected, every measured-Q
        # value is taken from THAT tile's harmonic fit instead of the
        # best-across-tiles graph attrs.  Edges with no measurement on
        # the filtered tile fall through to None (rendered grey).  Same
        # for normalized-Q (uses measurements_piv_normalized instead).
        if (src in ('measured', 'normalized') and qty == 'Q'
                and self.current_tile_filter is not None):
            return self._resolve_tile_filtered_value(u, v, d, prop)

        # BC-anchor phase override: for edges incident to a boundary
        # anchor node, use the tile-14-only measurement (raw φ +
        # tile-14 gauge) rather than the multi-tile average.  The BC
        # anchors live on tile 14, so its measurement is the trustworthy
        # one; the multi-tile average is polluted by low-SNR outside-
        # tile measurements that can be π-flipped, which then makes
        # sink stubs look out-of-sync with each other on the map.
        if (prop == 'phase' and src in ('measured', 'normalized')
                and qty == 'Q'):
            k_idx_bc = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_bc >= 1:
                bt_u = self.G.nodes[u].get('boundary_type')
                bt_v = self.G.nodes[v].get('boundary_type')
                if bt_u in ('source', 'sink') or bt_v in ('source', 'sink'):
                    tg = self.G.graph.get('tile_gauges') or {}
                    g14 = tg.get(14) or {}
                    gk_re = f'z_H{k_idx_bc}_re'
                    gk_im = f'z_H{k_idx_bc}_im'
                    gauge_arg = 0.0
                    if (g14.get(gk_re) is not None
                            and g14.get(gk_im) is not None):
                        gauge_arg = float(np.arctan2(float(g14[gk_im]),
                                                     float(g14[gk_re])))
                    for m in (d.get('measurements_piv') or []):
                        if int(m.get('tile_id', -1)) != 14:
                            continue
                        for h in (m.get('harm', {}).get('harmonics') or []):
                            if int(h.get('k', 0)) != k_idx_bc:
                                continue
                            if h.get('phi') is not None:
                                return float(h['phi']) + gauge_arg
                        break

        # ── Harmonic fraction: |H_k|² / (|H1|²+|H2|²+|H3|²) ──
        if prop == 'harmonic_fraction':
            k_idx_hf = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_hf == 0:
                return None  # DC is not part of the AC fraction
            ac = self._edge_ac_harmonics(d, src, qty)
            if ac is None:
                return None
            mags2 = np.abs(ac) ** 2
            tot = float(mags2.sum())
            if tot < 1e-30:
                return None
            return float(mags2[k_idx_hf - 1] / tot)

        # ── Waveform A↔V (aggregate; identical for every harmonic) ──
        if prop == 'waveform_av':
            if qty != 'Q':
                return None
            refs = self._av_reference_harmonics(src)
            if refs is None:
                return None
            e_ac = self._edge_ac_harmonics(d, src, qty)
            if e_ac is None:
                return None

            def _cossim(a, b):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-30 or nb < 1e-30:
                    return 0.0
                return float(np.real(np.vdot(b, a)) / (na * nb))

            return _cossim(e_ac, refs[0]) - _cossim(e_ac, refs[1])

        # ── Aggregate (non-harmonic-keyed) properties ──
        # 'mean' was dropped — use Magnitude @ DC for the |Q̄| / |P̄|
        # equivalent.
        if prop == 'PI':
            # Per-harmonic Gosling-style PI: PI_k = 2·|Q_Hk| / |Q_DC|.
            # k_idx = 0 (DC) is undefined; H1 is classic PI.
            k_idx_pi = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_pi == 0:
                return None  # PI at DC is undefined
            if src in ('measured', 'normalized') and qty == 'Q':
                meas_key = ('measurements_piv_normalized'
                            if src == 'normalized' else 'measurements_piv')
                m = _best_measurement(d.get(meas_key))
                if m is None:
                    return None
                try:
                    # Compute from harmonics list (works for both raw and
                    # normalized measurements — normalized doesn't populate
                    # the legacy top-level Q_H{k}_amp/Q_DC fields).
                    harms = (MeasurementView(m).harm.harmonics or [])
                    if k_idx_pi - 1 < len(harms):
                        amp_k = harms[k_idx_pi - 1].get('amp')
                    else:
                        amp_k = m.get(f'Q_H{k_idx_pi}_amp')
                    dc = (m.get('harm') or {}).get('mean_Q')
                    if dc is None:
                        dc = m.get('Q_DC')
                    if (amp_k is None or dc is None or float(dc) == 0):
                        return None
                    val = 2.0 * abs(float(amp_k)) / abs(float(dc))
                    return val if np.isfinite(val) else None
                except (TypeError, ValueError):
                    return None
            if src == 'sim' and qty == 'Q':
                # Sim path: derive PI_k from cached harmonics if available,
                # otherwise fall back to the cached H1-only PI.
                h = d.get('_sim_tmp_harmonics')
                if h is not None and k_idx_pi < len(h):
                    dc = abs(float(h[0].real))
                    if dc < 1e-30:
                        return None
                    return 2.0 * float(abs(h[k_idx_pi])) / dc
                if k_idx_pi == 1:
                    return _safe_float(d.get('_sim_tmp_PI'))
                return None
            if src == 'sim' and qty == 'P':
                p = d.get('_sim_tmp_p_harmonics')
                if p is None or k_idx_pi >= len(p):
                    return None
                p_dc = abs(float(p[0].real))
                if not np.isfinite(p_dc) or p_dc < 1e-12:
                    return None
                amp_k = float(abs(p[k_idx_pi]))
                if not np.isfinite(amp_k):
                    return None
                return 2.0 * amp_k / p_dc
            return None
        if prop == 'dissipation':
            if src == 'measured':
                return _safe_float(d.get('_meas_dissipation'))
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_dissipation'))
            return None
        if prop == 'dissipation_ac':
            if src == 'measured':
                # Persisted on v2 canonicals as dissipation_AC_piv (nW).
                return _safe_float(d.get('dissipation_AC_piv'))
            if src == 'sim':
                # Φ_AC = R_e · ½·Σ|H_k|² from sim Q harmonics.
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 4:
                    return None
                sum_sq = float(sum(abs(complex(hk)) ** 2 for hk in h[1:4]))
                if sum_sq <= 0: return None
                # Get G_e from edge geometry (same as measured path uses).
                # v4 migration: was d.get('conductance_piv'); compute from
                # geometry via MeasurementView (πR⁴/(8μL), μ=3.5e-3 Pa·s).
                mv_ce = MeasurementView.best(d)
                if mv_ce is None:
                    return None
                R_um = mv_ce.geom.R_um; L_um = mv_ce.geom.L_um
                if R_um is None or L_um is None:
                    return None
                try:
                    G_e = math.pi * float(R_um) ** 4 / (8.0 * 3.5e-3 * float(L_um))
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(G_e) or G_e <= 0:
                    return None
                return float(0.5 * sum_sq / float(G_e))
            return None
        if prop == 'pulse_frac':
            if src == 'measured':
                # Persisted on v2 canonicals as pulse_frac_piv ∈ [0, 1].
                # Same number whether interpreted as fraction of Q²-energy
                # or fraction of viscous dissipation — R_e cancels.
                return _safe_float(d.get('pulse_frac_piv'))
            if src == 'sim':
                # AC fraction from sim Q harmonics.
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 4:
                    return None
                dc = abs(complex(h[0]))
                ac_sq = sum(abs(complex(hk)) ** 2 for hk in h[1:4])
                ac_e = 0.5 * ac_sq
                tot = dc * dc + ac_e
                return float(ac_e / tot) if tot > 0 else None
            return None
        if prop == 'OSI':
            # Reconstruct Q(t) from DC + harmonics over one period and
            # compute OSI_Q = ½·(1 − |∫Q|/∫|Q|).  Aggregate field — the
            # harmonic selector is ignored.
            if src in ('measured', 'normalized') and qty == 'Q':
                meas_key = ('measurements_piv_normalized'
                            if src == 'normalized' else 'measurements_piv')
                m = _best_measurement(d.get(meas_key))
                if m is None:
                    return None
                try:
                    dc_q = m.get('Q_DC')
                    if dc_q is None:
                        dc_q = (m.get('harm') or {}).get('mean_Q')
                    if dc_q is None or not np.isfinite(dc_q):
                        dc_q = 0.0
                    amps_phi = []
                    harms = (MeasurementView(m).harm.harmonics or [])
                    for k in (1, 2, 3):
                        a = m.get(f'Q_H{k}_amp')
                        p_ = m.get(f'Q_H{k}_phi')
                        if ((a is None or p_ is None) and len(harms) >= k):
                            a = harms[k - 1].get('amp')
                            p_ = harms[k - 1].get('phi')
                        if a is None or p_ is None: continue
                        if not (np.isfinite(a) and np.isfinite(p_)): continue
                        amps_phi.append((float(a), float(p_)))
                    if not amps_phi:
                        return None
                    return _osi_from_harmonics(float(dc_q), amps_phi)
                except (TypeError, ValueError):
                    return None
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 2:
                    return None
                try:
                    dc_q = float(np.real(h[0]))
                    amps_phi = []
                    for k in range(1, len(h)):
                        a = float(abs(h[k]))
                        p_ = float(np.angle(h[k]))
                        if np.isfinite(a) and np.isfinite(p_):
                            amps_phi.append((a, p_))
                    if not amps_phi:
                        return None
                    return _osi_from_harmonics(dc_q, amps_phi)
                except (TypeError, ValueError, AttributeError):
                    return None
            return None
        if prop == 'drop':
            return _safe_float(d.get('_sim_tmp_pressure_drop'))
        if prop == 'radius':
            # sim-mode override takes precedence (uniform-R / uniform-C).
            if src == 'sim':
                v_ = d.get('_sim_tmp_radius')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('radius'))
        if prop == 'length':
            if src == 'sim':
                v_ = d.get('_sim_tmp_length')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('length'))
        if prop == 'harmonic_class':
            return _safe_float(d.get('harmonic_class'))
        if prop == 'total_snr':
            # Cached during _precompute_harmonic_classes; only present
            # for measured Q edges that had a usable Q_t to fit.
            return _safe_float(d.get('_h_total_snr'))
        if prop == 'snr_ac':
            # AC-total SNR in dB: P(H1+H2+H3) / MS_BL.  Canonical edge attr.
            return _safe_float(d.get('snr_ac_fit_db'))
        if prop == 'snr_total':
            # DC+AC SNR in dB: P(DC+H1+H2+H3) / MS_BL.  Canonical edge attr.
            return _safe_float(d.get('snr_harm_fit_db'))
        if prop == 'frequency':
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_f0_hz'))
            if src == 'normalized':
                # Normalized graphs are on a common canonical f0 (stored
                # at the graph level by tile_normalize).
                return _safe_float(self.G.graph.get('f0_canonical_hz'))
            # Measured: top-level edge attr is set during analysis from
            # the best measurement; fall back to scanning the per-tile
            # measurement list if it isn't there.
            v_ = d.get('f0_hz')
            if v_ is None:
                m = _best_measurement(d.get('measurements_piv'))
                if m is not None:
                    v_ = MeasurementView(m).harm.f0_hz
            return _safe_float(v_)

        # ── Harmonic-keyed (magnitude / phase / resolution) ──
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        # Cache-field suffix routes 'normalized' to `_h_..._norm` variants
        # populated by pertile.mosaic.tile_normalize.
        cache_suffix = '_norm' if src == 'normalized' else ''
        if prop == 'magnitude':
            if src in ('measured', 'normalized') and qty == 'Q':
                if k_idx == 0:
                    v_ = d.get(f'_h_amp_DC{cache_suffix}')
                    if v_ is None:
                        meas_key = ('measurements_piv_normalized'
                                    if src == 'normalized'
                                    else 'measurements_piv')
                        m = _best_measurement(d.get(meas_key))
                        if m is None:
                            return None
                        dc = m.get('Q_DC')
                        if dc is None:
                            dc = MeasurementView(m).harm.mean_Q
                        if dc is None:
                            dc = float('nan')
                        try:
                            return abs(float(dc))
                        except (TypeError, ValueError):
                            return None
                    try:
                        return abs(float(v_))
                    except (TypeError, ValueError):
                        return None
                return _safe_float(d.get(f'_h_amp_H{k_idx}{cache_suffix}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            return None
        if prop == 'signed':
            # Signed real value (no |·|) — sim P/Q.  DC: the signed gauge
            # (can be < 0).  AC: in-phase component (sign is phase-ref-dependent).
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
            elif src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
            else:
                return None
            if h is None or k_idx >= len(h):
                return None
            return float(np.real(h[k_idx]))
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            if src in ('measured', 'normalized') and qty == 'Q':
                return _safe_float(d.get(f'_h_phase_H{k_idx}{cache_suffix}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            return None
        if prop == 'resolution':
            if src in ('measured', 'normalized') and qty == 'Q':
                if k_idx == 0:
                    return _safe_float(d.get(f'_h_Z_DC{cache_suffix}'))
                return _safe_float(d.get(f'_h_Z_H{k_idx}{cache_suffix}'))
            return None
        if prop == 'snr':
            # Per-harmonic SNR — canonical (top-level) edge attrs.  Same
            # SNR under normalized source: gauge scaling is deterministic
            # and doesn't affect the underlying noise-relative amplitude.
            if src in ('measured', 'normalized') and qty == 'Q':
                if k_idx == 0:
                    return _safe_float(d.get('Q_DC_snr_db'))
                return _safe_float(d.get(f'Q_H{k_idx}_snr_db'))
            return None
        return None

    # ── Synthesized "current_field" name ──────────────────────────────
    @property
    def current_field(self) -> str:
        """Synthesized canonical key from the 4 selectors.

        Format: ``<quantity>.<property>[@<harmonic>]`` (e.g.
        ``Q.magnitude@H2`` / ``P.mean`` / ``Q.PI``).  Legacy code paths
        check substrings like ``'phase' in field`` — these stay valid
        because the property key appears verbatim in the string.
        """
        parts = [self.current_quantity, self.current_property]
        prop = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        if prop is not None and prop[2]:
            return '.'.join(parts) + '@' + self.current_harmonic
        return '.'.join(parts)

    @current_field.setter
    def current_field(self, _val):
        """Setter kept as a no-op for legacy compatibility.  The actual
        selection is owned by the four selector state variables."""
        return

    def _current_field_label(self) -> str:
        """Human-readable label for the colorbar.

        Appends a SI unit suffix so colormap values are unambiguous
        when switching between Quantity (Q vs P) and Source (sim vs
        measured).  Properties whose unit is already in the display
        label (`SNR (dB)`, `Fundamental f₀ (Hz)`, `Radius (px)`,
        `Length (px)`) intentionally get no extra suffix to avoid
        duplication; categorical / dimensionless properties (PI, OSI,
        harmonic class) get nothing.
        """
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        prop_lbl = pdef[1] if pdef else self.current_property
        qsym = self.current_quantity
        # (property, quantity) → unit string.  None / '' means "no unit
        # suffix needed".  Lookup order: (prop, qsym) → (prop, '*').
        UNITS = {
            ('magnitude', 'Q'):    'nL/s',
            ('magnitude', 'P'):    'Pa',
            ('signed',    'Q'):    'nL/s',
            ('signed',    'P'):    'Pa',
            ('phase',     '*'):    'rad',
            ('dissipation', '*'):  'W',
            ('drop',       '*'):   'Pa',
            # PI, OSI: dimensionless; SNR/SNR_AC/SNR_total: already (dB);
            # frequency: already (Hz); radius/length: already (px);
            # harmonic_class: categorical.  All omitted on purpose.
        }
        prop_key = self.current_property
        unit = UNITS.get((prop_key, qsym),
                          UNITS.get((prop_key, '*'), ''))
        # Compose source/harmonic suffix as before.
        if pdef is not None and pdef[2]:
            qual = f"  ({qsym}, {HARMONIC_LABELS[self.current_harmonic]})"
        else:
            qual = f"  ({qsym})"
        unit_str = f"  [{unit}]" if unit else ''
        return prop_lbl + qual + unit_str

    def _load_graph(self):
        print(f"Loading graph: {self.graph_path}")
        with open(self.graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _load_tiff(self):
        try:
            import tifffile
            print(f"Loading TIFF: {self.tiff_path}")
            img = tifffile.imread(self.tiff_path)
            print(f"  shape={img.shape}, dtype={img.dtype}")
            return img
        except Exception as e:
            print(f"  Could not load TIFF ({e}); proceeding without it.")
            return None

    def _load_tile_positions(self) -> Dict[int, dict]:
        """Parse tile_positions_manual.json → {tile_id: {translate_x,
        translate_y, scale_x, scale_y}}.  Also computes the global tile
        offset (min translate_x/y across all tiles) so video tiles can
        be placed in image-row coordinates: row = translate_y − offset_y.
        """
        try:
            with open(self.tile_positions_path) as f:
                tp = json.load(f)
        except Exception as e:
            print(f"  Could not load tile positions ({e}); video overlay disabled.")
            return {}
        tiles: Dict[int, dict] = {}
        for vid_str, entry in tp.get('tiles', {}).items():
            try:
                vid = int(vid_str)
            except ValueError:
                continue
            # Full 3x3 affine (last row [0,0,1]).  plexusflow's
            # positions.json stores this under 'affine_matrix' and tiles
            # can be rotated; the scale_x/y + translate_x/y fields alone
            # are just the diagonal of the affine and don't describe a
            # rotated tile's actual footprint on the mosaic.
            A = entry.get('affine_matrix')
            if A is not None:
                A = np.asarray(A, dtype=float)
            else:
                A = np.array([
                    [float(entry.get('scale_y', 1.0)), 0.0,
                     float(entry.get('translate_y', 0.0))],
                    [0.0, float(entry.get('scale_x', 1.0)),
                     float(entry.get('translate_x', 0.0))],
                    [0.0, 0.0, 1.0],
                ])
            tiles[vid] = {
                'translate_x': float(entry.get('translate_x', A[1, 2])),
                'translate_y': float(entry.get('translate_y', A[0, 2])),
                'scale_x':     float(entry.get('scale_x', A[1, 1])),
                'scale_y':     float(entry.get('scale_y', A[0, 0])),
                'affine':      A,
            }
        # Global tile offset — must match the fuse's `compute_world_bbox`
        # (plexusflow/fuse/feather.py): floor(min(corner_y)) - pad,
        # floor(min(corner_x)) - pad, using the four warped corners of
        # every tile (not just the translate).  `min(translate)` alone
        # misses the top rows of rotated tiles + the pad and shifts every
        # overlay by that amount.
        FUSE_PAD = 20
        if tiles:
            corners_yx = []
            for t in tiles.values():
                A = t.get('affine')
                if A is None:
                    continue
                raw = np.array([[0.0, 0.0],
                                [0.0, TILE_RAW_WIDTH],
                                [TILE_RAW_HEIGHT, TILE_RAW_WIDTH],
                                [TILE_RAW_HEIGHT, 0.0]])
                corners_yx.append(raw @ A[:2, :2].T + A[:2, 2])
            if corners_yx:
                corners_yx = np.concatenate(corners_yx, axis=0)
                self._tile_offset_y = float(np.floor(corners_yx[:, 0].min())
                                             - FUSE_PAD)
                self._tile_offset_x = float(np.floor(corners_yx[:, 1].min())
                                             - FUSE_PAD)
            else:
                self._tile_offset_x = min(t['translate_x']
                                          for t in tiles.values())
                self._tile_offset_y = min(t['translate_y']
                                          for t in tiles.values())
        else:
            self._tile_offset_x = 0.0
            self._tile_offset_y = 0.0
        print(f"  Loaded positions for {len(tiles)} tiles "
              f"from {self.tile_positions_path.name} "
              f"(offset={self._tile_offset_x:.0f}, {self._tile_offset_y:.0f})")
        return tiles

    def _get_video_path_for(self, tile_id: int) -> Optional[Path]:
        """Resolve the display video for ``tile_id``.

        Tries the primary ``video_dir`` first (typically the realtime
        downsampled bundle), then falls back to ``of_video_dir`` (the
        raw folder-of-frames) when the realtime file isn't present for
        this tile.  Lets you build the realtime bundle incrementally
        without leaving un-downsampled tiles without a display video.
        """
        if self.video_pattern is None:
            return None
        for base in (self.video_dir, self.of_video_dir):
            if base is None:
                continue
            name = self.video_pattern.format(vid=tile_id)
            candidate = base / name
            if candidate.exists():
                return candidate
            matches = list(base.glob(_video_glob(self.video_pattern, tile_id)))
            if matches:
                return matches[0]
        return None

    def _get_of_video_path_for(self, tile_id: int) -> Optional[Path]:
        """Resolve the video source used by fresh optical-flow/PIV compute.

        This is intentionally separate from `_get_video_path_for`: the
        display overlay can use lightweight realtime TIFF stacks while OF
        recomputation reads the raw frame folders.
        """
        if self.of_video_dir is None or self.of_video_pattern is None:
            return None
        name = self.of_video_pattern.format(vid=tile_id)
        candidate = self.of_video_dir / name
        if candidate.exists():
            return candidate
        matches = list(self.of_video_dir.glob(_video_glob(self.of_video_pattern, tile_id)))
        if matches:
            return matches[0]
        return None

    def _remove_video_overlay(self):
        # Stop autoplay before destroying anything that the
        # AnimationThread is connected to.
        try:
            self.viewer.window._qt_viewer.dims.stop()
        except Exception:
            pass
        if self._video_layer is not None:
            try:
                self.viewer.layers.remove(self._video_layer)
            except Exception:
                pass
            self._video_layer = None
            self._video_tile_id = None
        # Per-tile harmonic cache is tile-local; without a tile, drop it.
        self._tile_harmonic_cache.clear()

    def _populate_tile_harmonic_cache(self, tile_id: int):
        """Compute per-tile harmonic fits for every edge that has a
        measurement on `tile_id`.  Uses THAT tile's measurement (not
        the best-quality one) so the f0 is consistent everywhere in
        the view.

        Respects `self.current_source`:
          - 'measured' / 'sim': read raw `measurements_piv`, native f0
          - 'normalized':      read `measurements_piv_normalized`,
                               canonical f0, and apply per-harmonic
                               gauge corrections for H2/H3 (Q_t alone
                               only carries H1 time-shift).
        """
        self._tile_harmonic_cache.clear()
        src = getattr(self, 'current_source', 'measured')
        meas_key = ('measurements_piv_normalized' if src == 'normalized'
                    else 'measurements_piv')
        F0_canonical = self.G.graph.get('f0_canonical_hz')
        tile_gauges = self.G.graph.get('tile_gauges') or {}
        n = 0
        for u, v, d in self.G.edges(data=True):
            meas = d.get(meas_key) or []
            m_tile = next((mm for mm in meas
                           if MeasurementView(mm).tile_id == tile_id), None)
            if m_tile is None:
                continue
            mv_tile = MeasurementView(m_tile)
            if mv_tile.Q_t is None:
                continue
            if not _measurement_usable(m_tile):
                continue
            Q_t = np.asarray(mv_tile.Q_t, dtype=float)
            if Q_t.size < 30:
                continue
            f0_for_fit = (float(F0_canonical) if src == 'normalized'
                          and F0_canonical else
                          float(mv_tile.harm.f0_hz or 2.5))
            full = _harmonic_fit_full(Q_t, f0=f0_for_fit, dt=1.0/250)
            if full is None:
                continue
            # Per-harmonic H2/H3 gauge correction — only applies for
            # normalized source (raw Q_t carries only the H1 time-shift,
            # so H2/H3 phases from the refit have `k·arg(z_H1)` residual
            # shift and z_H1_mag scaling that must be corrected to their
            # own per-harmonic gauge).
            if src == 'normalized':
                g_entry = tile_gauges.get(int(tile_id)) or {}
                z_H1_arg_deg = g_entry.get('z_arg_deg')
                z_H1_mag = g_entry.get('z_mag')
                # Only apply the amp `× z_H1_mag/z_Hk_mag` correction if
                # the pipeline actually divided AC by |z_H1| (i.e. the
                # `level_ac_amplitude` flag was on).  Otherwise Q_t has
                # raw AC amplitudes and refitted H2/H3 amps are already
                # correct — only the H2/H3 PHASE needs undoing/redoing
                # (since the H1 time-shift always applies at k·ω).
                did_amp_leveling = bool((self.G.graph.get(
                    'normalization_params') or {}).get('level_ac_amplitude'))
                if z_H1_arg_deg is not None and z_H1_mag not in (None, 0):
                    for k_h in (2, 3):
                        z_Hk_arg = g_entry.get(f'z_H{k_h}_arg_deg')
                        z_Hk_mag = g_entry.get(f'z_H{k_h}_mag')
                        if z_Hk_arg is None or z_Hk_mag in (None, 0):
                            continue   # H1 fallback for missing gauge
                        dphi = np.radians(k_h * z_H1_arg_deg - z_Hk_arg)
                        p0 = full.get(f'phase_H{k_h}')
                        if p0 is not None and np.isfinite(p0):
                            pnew = float(p0) + dphi
                            full[f'phase_H{k_h}'] = float(
                                ((pnew + np.pi) % (2 * np.pi)) - np.pi)
                        if did_amp_leveling:
                            scale = float(z_H1_mag) / float(z_Hk_mag)
                            a0 = full.get(f'amp_H{k_h}')
                            if a0 is not None and np.isfinite(a0):
                                full[f'amp_H{k_h}'] = float(a0) * scale
            self._tile_harmonic_cache[(u, v)] = full
            n += 1
        print(f"  Computed per-tile harmonics for {n} edges in tile "
              f"{tile_id} (f0 anchored at tile-{tile_id}'s value).")

    def _load_video_overlay(self, tile_id: int):
        """Load the per-tile video and add it as a napari 3D Image layer
        translated/scaled onto the mosaic so the embryo lines up with
        the graph.  Replaces any prior overlay (only one at a time).

        Tile switches reuse the same Image layer (in-place data update).
        Destroying and re-adding the layer leaves napari's
        AnimationThread holding a dangling reference to the destroyed
        previous QtDimSliderWidget, which raises TypeError on the next
        .play() (`prev_slider.fps_changed.disconnect(...)` fails).
        Keeping the layer alive avoids that path entirely.
        """
        if tile_id == self._video_tile_id:
            return  # already showing
        # Stop autoplay before mutating layer data — keeps the
        # AnimationThread from sampling .data during the swap.
        try:
            self.viewer.window._qt_viewer.dims.stop()
        except Exception:
            pass
        if tile_id not in self.tiles:
            print(f"  No tile_positions entry for tile {tile_id}; "
                  f"can't place the video on the mosaic.")
            return
        video_path = self._get_video_path_for(tile_id)
        if video_path is None:
            print(f"  No video found for tile {tile_id} "
                  f"(looked under {self.video_dir} with pattern "
                  f"{self.video_pattern!r}).")
            return
        from ..io.tiff import load_tiff_stack, cut_before_fade
        print(f"  Loading video for tile {tile_id}: {video_path}")
        stack = load_tiff_stack(video_path, max_frames=600)
        stack, cut_frame = cut_before_fade(stack, verbose=False)
        # Temporally decimate for ~real-time playback.  The 30-fps play rate
        # (below) assumes the realtime preview is ~8× block-averaged (~75-90
        # frames ≈ real recording duration).  This stage's realtime videos are
        # only ~4× downsampled (~200 frames), so 30 fps ran ~2× too slow.
        # Decimate to the ~90-frame target so playback ≈ real time regardless
        # of how much the preview file was pre-downsampled.
        _pt = 90
        if stack.shape[0] > int(_pt * 1.4):
            _stride = max(1, int(round(stack.shape[0] / _pt)))
            stack = stack[::_stride]
            print(f"  temporally decimated video ×{_stride} → "
                  f"{stack.shape[0]} frames (≈ real-time at 30 fps)")
        # Normalise to uint8 with 1%/99% contrast — keeps GPU upload cheap
        # and gives a sensible default contrast.
        if stack.dtype != np.uint8:
            sample = stack[::max(1, stack.shape[0] // 20)]
            lo, hi = np.percentile(sample, (1.0, 99.0))
            if hi <= lo:
                hi = lo + 1.0
            stack = np.clip(
                (stack.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                0, 255).astype(np.uint8)
        # Compute the effective scale factor between the loaded video's
        # frame size and the canonical raw frame size.  Full-res
        # acquisitions are TILE_RAW_HEIGHT × TILE_RAW_WIDTH; bundles
        # that ship pre-downsampled videos will arrive smaller.  This
        # lets a single overlay path handle both the full-res and the
        # 2×-downsampled bundles correctly — without this detection
        # the bundle videos sit in the top-left quarter of their
        # intended footprint because the viewer's internal `[::2,::2]`
        # below would compound the bundle's downsample.
        H_in, W_in = stack.shape[1], stack.shape[2]
        ds_factor_y = TILE_RAW_HEIGHT / float(H_in) if H_in else 1.0
        ds_factor_x = TILE_RAW_WIDTH / float(W_in) if W_in else 1.0
        # Only run the internal GPU-memory downsample if the input is
        # close to full resolution (within ~30%).  Below that we trust
        # the bundle's downsampling and skip the slicing — the loaded
        # video is already small enough.
        display_downsampled = False
        if ds_factor_x < 1.5 and ds_factor_y < 1.5:
            stack = np.ascontiguousarray(stack[:, ::2, ::2])
            ds_factor_y *= 2.0
            ds_factor_x *= 2.0
            display_downsampled = True
        T, H, W = stack.shape
        # tile_width/height come from the mosaic_graph (we don't have a
        # mosaic-image reference if the user didn't pass --tiff).  Use
        # scale_x/scale_y as a multiplicative correction relative to the
        # raw video pixels.  The placement convention matches
        # `_tile_mgmt._load_tile_video`.
        tile_entry = self.tiles[tile_id]
        A_tile = tile_entry.get('affine')
        # Build a 4×4 spatial affine for the (T, Y, X) image.  Time axis
        # is identity; spatial axes carry the tile's full 2×3 affine
        # (rotation + scale + translate), pre-multiplied by a diagonal
        # scale that maps the *loaded* video's pixel size back to the
        # canonical raw-frame pixel size.  Global tile offset shifts
        # negative translates into image-row coords.
        S_pixel = np.array([[ds_factor_y, 0.0], [0.0, ds_factor_x]])
        if A_tile is not None:
            M = A_tile[:2, :2] @ S_pixel
            t_yx = A_tile[:2, 2] - np.array([self._tile_offset_y,
                                             self._tile_offset_x])
        else:
            M = np.diag([tile_entry['scale_y'] * ds_factor_y,
                         tile_entry['scale_x'] * ds_factor_x])
            t_yx = np.array([tile_entry['translate_y'] - self._tile_offset_y,
                             tile_entry['translate_x'] - self._tile_offset_x])
        affine44 = np.eye(4)
        affine44[1:3, 1:3] = M
        affine44[1:3, 3] = t_yx
        print(f"  video source {H_in}×{W_in}"
              f"{' (display downsampled 2x)' if display_downsampled else ''}"
              f" → pixel-scale ({ds_factor_y:.2f}, {ds_factor_x:.2f})"
              f"  tile_translate=({t_yx[0]:.0f}, {t_yx[1]:.0f})")
        new_name = (f"Video tile {tile_id} "
                    f"({H_in}x{W_in}"
                    f"{' -> display 2x' if display_downsampled else ''})")
        # Try in-place layer reuse first.  If the existing layer is
        # still attached to the viewer we mutate its data + transform
        # without destroying the QtDimSliderWidget that the
        # AnimationThread is connected to.
        replaced_in_place = False
        if (self._video_layer is not None
                and self._video_layer in self.viewer.layers):
            try:
                self._video_layer.data = stack
                self._video_layer.scale = (1.0, 1.0, 1.0)
                self._video_layer.translate = (0.0, 0.0, 0.0)
                self._video_layer.affine = affine44
                self._video_layer.name = new_name
                self._video_layer.contrast_limits = (0, 255)
                replaced_in_place = True
            except Exception as exc:
                print(f"  in-place video swap failed ({exc!r}); "
                      f"falling back to remove+add.")
                self._remove_video_overlay()
        if not replaced_in_place:
            self._video_layer = self.viewer.add_image(
                stack,
                name=new_name,
                colormap='gray',
                contrast_limits=(0, 255),
                interpolation2d='nearest',
                affine=affine44,
                opacity=0.85,
                blending='translucent',
            )
        self._video_tile_id = tile_id
        # Populate per-tile harmonic cache (used by the time-series
        # inspect panel for cross-edge-comparable phase at the loaded
        # tile's f0).  Fast: typically 100–500 edges × ~5 ms each.
        self._populate_tile_harmonic_cache(tile_id)
        # Reset the time slider to frame 0.  Use the modern `current_step`
        # API — `set_point(0, 0)` is deprecated in napari ≥ 0.5 and on a
        # second layer swap can leave the slider at a stale index that
        # points past the end of a shorter new stack → blank display.
        try:
            cs = list(self.viewer.dims.current_step)
            if len(cs) >= 1:
                cs[0] = 0
                self.viewer.dims.current_step = tuple(cs)
        except Exception:
            try:
                self.viewer.dims.set_point(0, 0)  # legacy fallback
            except Exception:
                pass
        # Defer play() to the next event-loop tick so the new layer's
        # data/transform settles before the AnimationThread attaches.
        # Wrap in a defensive scrub: if napari's AnimationThread is
        # still holding a stale prev_slider reference from a prior
        # session (would raise TypeError on fps_changed.disconnect),
        # clear it so play() starts cleanly.
        def _start_play():
            qt_dims = None
            try:
                qt_dims = self.viewer.window._qt_viewer.dims
            except Exception:
                return
            try:
                at = getattr(qt_dims, '_animation_thread', None)
                # napari's set_slider reads prev_slider = self.slider on
                # the AnimationThread and calls
                # prev_slider.fps_changed.disconnect(...) — that raises
                # TypeError if prev_slider was destroyed by an earlier
                # layer removal.  Clearing at.slider to None skips the
                # disconnect path on the first attach to the new widget.
                if at is not None and hasattr(at, 'slider'):
                    at.slider = None
            except Exception:
                pass
            try:
                # 30 fps display rate.  Pair with the realtime preview
                # videos (8× temporally block-averaged) for ~real-time
                # playback: 75 frames / 30 fps ≈ 2.5 s, matching the
                # 2.4 s real recording duration.
                qt_dims.play(axis=0, fps=30)
            except TypeError:
                # Last-resort scrub: napari's internal disconnect chose
                # to raise — swallow it; the new slider will still play
                # on the user's next manual interaction.
                pass
            except Exception:
                pass
        try:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, _start_play)
        except Exception:
            _start_play()
        cut_note = f" (truncated at frame {cut_frame})" if cut_frame else ""
        print(f"  → {T} frames placed via affine "
              f"translate=({t_yx[0]:.0f}, {t_yx[1]:.0f}){cut_note}")

    def _precompute_measured_dissipation(self):
        """Compute viscous dissipation per edge from PIV Q_t + geometry.

        Φ = r · L · ⟨Q²⟩  (Watts)
        with r = 8μ / (πR⁴), L from edge attr `length`, R from `radius`,
        and ⟨Q²⟩ = mean(Q_t²) using the best PIV measurement.  Result
        stored as `_meas_dissipation` on each edge (in-memory only).
        Edges without a usable measurement / geometry get nan.
        """
        from ..analyze.constants import PX_SIZE_UM
        from ..analyze.constants import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        nl_to_m3 = 1e-12  # 1 nL = 1e-12 m³  → Q [nL/s] × 1e-12 = Q [m³/s]
        n = 0
        for u, v, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            mv = MeasurementView(m) if m is not None else None
            R_px = d.get('radius')
            L_px = d.get('length')
            # v4 migration: was m.get('Q_t') is None
            if (mv is None or mv.Q_t is None
                    or R_px is None or L_px is None):
                d['_meas_dissipation'] = float('nan')
                continue
            try:
                R_px_f = float(R_px); L_px_f = float(L_px)
            except (TypeError, ValueError):
                d['_meas_dissipation'] = float('nan')
                continue
            if not (np.isfinite(R_px_f) and R_px_f > 0
                    and np.isfinite(L_px_f) and L_px_f > 0):
                d['_meas_dissipation'] = float('nan')
                continue
            R_m = R_px_f * px_to_m
            L_m = L_px_f * px_to_m
            r = 8.0 * MU_DEFAULT / (np.pi * R_m ** 4)
            Q_t = np.asarray(mv.Q_t, dtype=float) * nl_to_m3  # v4: was m['Q_t']
            Q2 = float(np.nanmean(Q_t ** 2))
            d['_meas_dissipation'] = r * L_m * Q2
            n += 1
        print(f"  Precomputed dissipation for {n} edges "
              f"(Φ in W; range typical 1e-15 to 1e-12)")

    def _precompute_harmonic_classes(self):
        """Run one harmonic fit per edge (DC + H1 + H2 + H3) and cache
        everything needed for the per-harmonic field views:

          • `harmonic_class`   — int 0/1/2/3 or None
          • `_h_amp_DC`, `_h_Z_DC`
          • `_h_amp_H1`, `_h_phase_H1`, `_h_Z_H1`
          • `_h_amp_H2`, `_h_phase_H2`, `_h_Z_H2`
          • `_h_amp_H3`, `_h_phase_H3`, `_h_Z_H3`
          • `_h_r2`, `_h_sigma`

        The `_h_` prefix marks them as viewer-cached derived attributes.
        Cached back to the gpickle if `cache_harmonic_class` is True so
        relaunches are instant.

        Skip rules:
          - If `harmonic_class` is already present on every edge and
            the user didn't pass `--force-recompute-harmonic-class`,
            do nothing.
          - If `--force-recompute-harmonic-class`, wipe existing cache
            first and recompute from scratch.
        """
        import time
        CACHED_KEYS = (
            'harmonic_class',
            '_h_amp_DC', '_h_Z_DC',
            '_h_amp_H1', '_h_phase_H1', '_h_Z_H1',
            '_h_amp_H2', '_h_phase_H2', '_h_Z_H2',
            '_h_amp_H3', '_h_phase_H3', '_h_Z_H3',
            '_h_r2', '_h_sigma',
            '_h_total_snr', '_h_cache_ver',
        )
        if self.force_recompute_harmonic_class:
            for _, _, d in self.G.edges(data=True):
                for k in CACHED_KEYS:
                    d.pop(k, None)
        # Skip the recompute only if EVERY edge has the full `_h_*` cache
        # AT THE CURRENT SCHEMA VERSION.  Bumping HARMONIC_CACHE_VERSION
        # invalidates everything stored under the old convention (e.g.
        # σ definition changed std → MAD) without needing the user to
        # pass --force-recompute-harmonic-class.
        n_with_class = 0
        n_with_h1_amp = 0
        n_with_cur_ver = 0
        for _, _, d in self.G.edges(data=True):
            if 'harmonic_class' in d:
                n_with_class += 1
            if '_h_amp_H1' in d:
                n_with_h1_amp += 1
            if d.get('_h_cache_ver', 0) >= HARMONIC_CACHE_VERSION:
                n_with_cur_ver += 1
        n_edges = self.G.number_of_edges()
        if (n_with_class == n_edges
                and n_with_h1_amp == n_edges
                and n_with_cur_ver == n_edges):
            print(f"  harmonic_class + `_h_*` cache present on all "
                  f"{n_with_class} edges at version "
                  f"{HARMONIC_CACHE_VERSION}; skipping precompute.")
            return
        if n_with_class == n_edges and n_with_cur_ver < n_edges:
            print(f"  harmonic cache present but at older schema "
                  f"version (have {n_with_cur_ver}/{n_edges} at v"
                  f"{HARMONIC_CACHE_VERSION}); recomputing with new "
                  f"MAD-σ + total_snr convention.")
            # Wipe stale entries so the loop below actually re-fits
            # them (otherwise the `_h_amp_H1 in d` skip would fire).
            for _, _, d in self.G.edges(data=True):
                if d.get('_h_cache_ver', 0) < HARMONIC_CACHE_VERSION:
                    for k in CACHED_KEYS:
                        d.pop(k, None)
        elif n_with_class == n_edges:
            print(f"  harmonic_class present on all {n_with_class} edges "
                  f"but `_h_*` cache missing on {n_edges - n_with_h1_amp}; "
                  f"running full precompute …")
        n_total = self.G.number_of_edges()
        n_resolved = 0
        n_failed = 0
        t0 = time.time()
        print(f"  Precomputing harmonic_class for {n_total} edges ...")
        for i, (u, v, d) in enumerate(self.G.edges(data=True)):
            # Skip only if BOTH the class label AND the extended
            # `_h_*` cache are present — otherwise we re-fit so the
            # 4-selector resolver can read amp/phase/Z at H₁/H₂/H₃.
            if 'harmonic_class' in d and '_h_amp_H1' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            mv = MeasurementView(m) if m is not None else None
            # v4 migration: was m.get('Q_t') / m.get('f0_hz')
            if m is None or not _measurement_usable(m) \
                    or mv.Q_t is None:
                d['harmonic_class'] = None
                continue
            Q_t = np.asarray(mv.Q_t, dtype=float)
            if Q_t.size < 30:
                d['harmonic_class'] = None
                continue
            full = _harmonic_fit_full(
                Q_t, f0=float(mv.harm.f0_hz or 2.5), dt=1.0/250)
            if full is None:
                d['harmonic_class'] = None
                n_failed += 1
                continue
            # Pack Z dict for the classification helper.
            snrs = {'DC': full['Z_DC'], 'H1': full['Z_H1'],
                    'H2': full['Z_H2'], 'H3': full['Z_H3']}
            d['harmonic_class'] = int(_harmonic_class(snrs))
            d['_h_amp_DC'] = full['amp_DC']
            d['_h_Z_DC'] = full['Z_DC']
            for k in (1, 2, 3):
                d[f'_h_amp_H{k}'] = full[f'amp_H{k}']
                d[f'_h_phase_H{k}'] = full[f'phase_H{k}']
                d[f'_h_Z_H{k}'] = full[f'Z_H{k}']
            d['_h_r2'] = full['r2']
            d['_h_sigma'] = full['sigma']
            d['_h_total_snr'] = full.get('total_snr', float('nan'))
            d['_h_cache_ver'] = HARMONIC_CACHE_VERSION
            n_resolved += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (n_total - i - 1) / max(rate, 1e-3)
                print(f"    {i + 1}/{n_total} edges  "
                      f"({rate:.0f}/s, ~{eta:.0f}s remaining)")
        elapsed = time.time() - t0
        # Tally distribution
        counts = {0: 0, 1: 0, 2: 0, 3: 0, None: 0}
        for _, _, d in self.G.edges(data=True):
            counts[d.get('harmonic_class')] = (
                counts.get(d.get('harmonic_class'), 0) + 1)
        print(f"  Done in {elapsed:.1f} s.  "
              f"Class distribution: "
              f"DC-only={counts[0]}, +H1={counts[1]}, "
              f"+H1+H2={counts[2]}, +H1+H2+H3={counts[3]}, "
              f"no-data={counts[None]}")
        # Persist back to the gpickle so the next launch skips compute.
        if self.cache_harmonic_class:
            self._save_graph_with_harmonic_class()

    def _save_graph_with_harmonic_class(self):
        """Atomically write self.G back to self.graph_path so the
        harmonic_class attributes survive across viewer launches.

        Atomic = pickle to a temp file in the same directory, then
        os.replace() onto the target.  This guarantees a partial write
        can never leave the gpickle in a corrupt state.

        Only `harmonic_class` is ever added by this viewer; every other
        attribute on G is left as it was loaded.  No edges are added or
        removed.
        """
        import os, pickle, tempfile
        target = Path(self.graph_path).resolve()
        if not target.parent.is_dir():
            print(f"  Save skipped: parent dir {target.parent} not found.")
            return
        try:
            fd, tmp = tempfile.mkstemp(
                suffix='.harm_save.tmp', dir=str(target.parent),
                prefix='.' + target.stem + '.')
            try:
                with os.fdopen(fd, 'wb') as f:
                    pickle.dump(self.G, f,
                                 protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, target)
                size_mb = target.stat().st_size / (1024 ** 2)
                print(f"  Cached harmonic_class to {target.name} "
                      f"({size_mb:.0f} MB).  "
                      f"Disable with --no-cache-harmonic-class.")
            except Exception:
                try: os.remove(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  Warning: could not write cache: {e}  "
                  f"(compute will repeat next launch)")

    # ─────────────────────────────────────────────────────────────────
    # =================================================================
    # Normalize tab: per-tile complex gauges + amp gauges, computed
    # in-memory via `pertile.mosaic.tile_normalize`.  Writes normalized
    # measurements side-by-side with raw and unlocks the 'Normalized'
    # data source in the View tab.
    # =================================================================
    def _build_carved_sim_tab(self, layout):
        """Panel controls for the porous-carved transmission-line sim.

        Runs the recipe:
          1. Recompute simple-normalization v2 fields (with optional
             tile exclusion).
          2. Fit per-outlet α_k on the DC field via constrained NNLS.
          3. Carve the graph (drop tagged edges + A anchors, promote
             outlets to sources), extract DA reference + per-outlet
             local phase.
          4. Run the LOCKED two-pass TL sim (P=0 at V pass 1, Q-rebuild
             at V pass 2) with the outlets as sources.
          5. Write `_h_*_sim_carved` fields the View tab reads.
        """
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QLineEdit, QDoubleSpinBox, QPlainTextEdit, QCheckBox)
        # ── Detect status (updated on each refresh) ─────────────────
        _stat_group = QGroupBox("Detected carve")
        _stat_lay = QVBoxLayout(_stat_group)
        self._carved_stat_lbl = QLabel(self._carved_sim_status_text())
        self._carved_stat_lbl.setWordWrap(True)
        self._carved_stat_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _stat_lay.addWidget(self._carved_stat_lbl)
        _refresh_btn = QPushButton("Refresh status")
        _refresh_btn.clicked.connect(self._carved_sim_refresh_status)
        _stat_lay.addWidget(_refresh_btn)
        layout.addWidget(_stat_group)

        # ── Inputs ──────────────────────────────────────────────────
        _in_group = QGroupBox("Sim inputs")
        _in_lay = QVBoxLayout(_in_group)
        _row = QHBoxLayout()
        _row.addWidget(QLabel("Exclude tiles:"))
        self._carved_excl_edit = QLineEdit("")
        self._carved_excl_edit.setPlaceholderText(
            "comma-separated (e.g. 1,2,3,4,5,6 for the low-f0 ramp-up tiles)")
        self._carved_excl_edit.setToolTip(
            "Which per-tile measurements to skip when rebuilding the v2 "
            "fields, computing per-outlet phase shifts, and fitting α.  "
            "Tiles 1-6 are the developmental HR ramp-up recordings.")
        _row.addWidget(self._carved_excl_edit)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("D_areal [1/Pa]:"))
        self._carved_D_spin = QDoubleSpinBox()
        self._carved_D_spin.setDecimals(6)
        self._carved_D_spin.setRange(1e-7, 1e-2)
        self._carved_D_spin.setSingleStep(1e-5)
        self._carved_D_spin.setValue(3.5e-4)
        self._carved_D_spin.setToolTip(
            "Areal distensibility used by the transmission-line solver.")
        _row.addWidget(self._carved_D_spin)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("η (loss tan):"))
        self._carved_eta_spin = QDoubleSpinBox()
        self._carved_eta_spin.setDecimals(2)
        self._carved_eta_spin.setRange(0.0, 2.0)
        self._carved_eta_spin.setSingleStep(0.05)
        self._carved_eta_spin.setValue(0.0)
        self._carved_eta_spin.setToolTip(
            "Wall loss tangent η in D_eff(ω) = D·(1 − iη).\n"
            "0 = elastic (default). ~0.4 matches the current fit sweet spot.")
        _row.addWidget(self._carved_eta_spin)
        _row.addStretch(1)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("V handling:"))
        _lbl = QLabel("LOCKED two-pass (P=0 pass 1, Q-rebuild pass 2)")
        _lbl.setStyleSheet("color: #666; font-style: italic;")
        _row.addWidget(_lbl); _row.addStretch(1)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("α source:"))
        _lbl = QLabel("Constrained NNLS on measured DC field (Σα_zone = equal)")
        _lbl.setStyleSheet("color: #666; font-style: italic;")
        _row.addWidget(_lbl); _row.addStretch(1)
        _in_lay.addLayout(_row)
        layout.addWidget(_in_group)

        # ── Run ─────────────────────────────────────────────────────
        _run_group = QGroupBox("Run")
        _run_lay = QVBoxLayout(_run_group)
        self._carved_run_btn = QPushButton("Run Carved Sim")
        self._carved_run_btn.setToolTip(
            "Rebuilds v2 fields, fits α_k, runs the LOCKED two-pass TL sim, "
            "and writes _h_*_sim_carved fields.  Takes ~30-60 s.")
        self._carved_run_btn.clicked.connect(self._carved_sim_run)
        _run_lay.addWidget(self._carved_run_btn)
        self._carved_status = QPlainTextEdit()
        self._carved_status.setReadOnly(True)
        self._carved_status.setMaximumHeight(220)
        self._carved_status.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        _run_lay.addWidget(self._carved_status)
        layout.addWidget(_run_group)

        # ── View overlay ────────────────────────────────────────────
        _view_group = QGroupBox("Display")
        _view_lay = QVBoxLayout(_view_group)
        self._carved_use_chk = QCheckBox(
            "Show the carved sim on the View tab (_h_*_sim_carved fields)")
        self._carved_use_chk.setToolTip(
            "When on, the View tab reads _h_*_sim_carved fields.  When off, "
            "the normal `_h_*` / `_h_*_norm` fields are used.")
        self._carved_use_chk.toggled.connect(
            self._carved_sim_view_toggled)
        _view_lay.addWidget(self._carved_use_chk)
        self._carve_markers_chk = QCheckBox(
            "Show carve boundary markers  (★ = flow OUT of porous, "
            "□ = flow IN to porous)")
        self._carve_markers_chk.setToolTip(
            "Overlays a marker at each outlet node, coloured by the "
            "current field (matches edge colouring).  Star = net flow "
            "OUT of the porous zone into the trusted plexus; Square = "
            "net flow INTO the porous zone from the plexus.")
        self._carve_markers_chk.toggled.connect(
            self._on_carve_markers_toggled)
        _view_lay.addWidget(self._carve_markers_chk)
        layout.addWidget(_view_group)

    def _carved_sim_status_text(self):
        """One-line status describing the carve on the current graph."""
        try:
            tagged = sum(1 for _, _, d in self.G.edges(data=True)
                          if d.get('radial_uncertain'))
            if tagged == 0:
                return ("No `radial_uncertain` tags on the graph — nothing "
                        "to carve.  Tag some vessels via the View tab first.")
            from .carved_sim_backend import find_porous_and_outlets
            U, outlets, outlet_comp, A_a, V_a, comps = find_porous_and_outlets(
                self.G)
            comp_sizes = [len(c) for c in comps]
            return (f"tagged edges: {tagged}   porous nodes: {len(U)}   "
                    f"outlets: {len(outlets)}   "
                    f"zones: {len(comps)} sizes {comp_sizes}\n"
                    f"A anchors: {sorted(A_a)}   V anchors: {sorted(V_a)}")
        except Exception as e:
            return f"(status error: {e})"

    def _carved_sim_refresh_status(self):
        self._carved_stat_lbl.setText(self._carved_sim_status_text())

    def _carved_sim_view_toggled(self, on: bool):
        # Cache the flag; `_refresh_edges` reads it to decide which field
        # to use for edge colouring.
        self._carved_sim_show = bool(on)
        if hasattr(self, '_refresh_edges'):
            self._refresh_edges()

    def _on_carve_markers_toggled(self, on: bool):
        self._show_carve_markers = bool(on)
        self._update_carve_boundary_markers()

    def _classify_outlet_direction(self, o, U, A_a, V_a):
        """Return ('in', 1.0) if `o` is the porous-side node that a
        boundary vessel (A-anchor stub) feeds INTO the porous zone via
        — the physical inflow port for the region.  Otherwise return
        ('out', 1.0) because by construction of our sim, every other
        outlet is a source (α_k ≥ 0) delivering flow OUT of the porous
        zone into the trusted plexus.

        Detection: if `o` is a direct neighbor of any A anchor via an
        UNtagged edge (the A stub), it's the inflow port.  All other
        outlets are outflow."""
        for a in A_a:
            if not self.G.has_edge(o, a): continue
            d = self.G.edges[o, a]
            if d.get('radial_uncertain'): continue
            # o is on an A stub → inflow-from-A boundary
            return 'in', 1.0
        # V-anchor-adjacent porous nodes (unlikely in our current
        # geometry but handled for generality) — those would also be
        # inflow ports for the porous zone from the V side.
        for vn in V_a:
            if not self.G.has_edge(o, vn): continue
            d = self.G.edges[o, vn]
            if d.get('radial_uncertain'): continue
            return 'in', 1.0
        return 'out', 1.0

    def _update_carve_boundary_markers(self):
        """Add/update the 'CarveMarkers' napari points layer.  Each
        outlet gets a ★ (net flow OUT of porous), and every porous-side
        node of an A- or V-anchor boundary vessel gets a □ (inflow port
        from the anatomical anchor).  Colours match the current edge
        colormap.  No-op if the checkbox is off or nothing to show."""
        if not getattr(self, '_show_carve_markers', False):
            if 'CarveMarkers' in self.viewer.layers:
                self.viewer.layers['CarveMarkers'].visible = False
            return
        try:
            from .carved_sim_backend import find_porous_and_outlets
            U, outlets, _, A_a, V_a, _ = find_porous_and_outlets(self.G)
        except Exception:
            return
        # Also find porous-side nodes of A/V anchor stubs — the inflow
        # ports for the region.  These may or may not qualify as
        # "outlets" (they don't if their only untagged edge is TO the
        # anchor).  We mark them as ▢ regardless.
        inflow_nodes = set()
        for anc in list(A_a) + list(V_a):
            if not self.G.has_node(anc): continue
            for m in self.G.neighbors(anc):
                if m not in U: continue
                d = self.G.edges[m, anc]
                if d.get('radial_uncertain'): continue
                inflow_nodes.add(m)
        # Marker list = outlets ∪ inflow_nodes (inflow_nodes win on
        # symbol tie-break).
        marker_nodes = list(dict.fromkeys(list(outlets) + list(inflow_nodes)))
        if not marker_nodes:
            return
        pts = []; symbols = []; face = []
        # Get the current edge LC's cmap + norm so markers use the same
        # scaling.
        cmap_name = 'viridis'; norm = None
        if hasattr(self, '_edge_layer') and self._edge_layer is not None:
            try:
                cmap_name = self._edge_layer.face.colormap.name
            except Exception:
                pass
        # Build a colormap-lookup function from matplotlib.
        import matplotlib.cm as _mcm
        cmap_obj = _mcm.get_cmap(cmap_name)
        # Field values for markers — sampled from the "outward" edge
        # (outlet: untagged edge to trusted plexus; inflow node:
        # untagged edge to the anchor, i.e. the A/V stub itself).
        def _sample_val(n):
            # For an inflow port, pick the A/V-stub edge; for a normal
            # outlet, pick an untagged plexus-facing edge.
            if n in inflow_nodes:
                for anc in list(A_a) + list(V_a):
                    if self.G.has_edge(n, anc) and \
                            not self.G.edges[n, anc].get('radial_uncertain'):
                        if hasattr(self, '_resolve_field_value'):
                            v = self._resolve_field_value(n, anc)
                            if v is not None: return float(v)
                            v = self._resolve_field_value(anc, n)
                            if v is not None: return float(v)
            for m in self.G.neighbors(n):
                if m in U or m in A_a or m in V_a: continue
                if self.G.edges[n, m].get('radial_uncertain'): continue
                if hasattr(self, '_resolve_field_value'):
                    v = self._resolve_field_value(n, m)
                    if v is not None: return float(v)
                    v = self._resolve_field_value(m, n)
                    if v is not None: return float(v)
            return np.nan

        vals = np.asarray([_sample_val(n) for n in marker_nodes], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            v_lo, v_hi = float(np.nanpercentile(finite, 2)), \
                         float(np.nanpercentile(finite, 98))
        else:
            v_lo, v_hi = 0.0, 1.0
        vspan = max(v_hi - v_lo, 1e-9)

        H_mos = getattr(self, 'mosaic_height', None)
        for o, val in zip(marker_nodes, vals):
            if o in inflow_nodes:
                sym = 'square'
            else:
                sym = 'star'
            if np.isfinite(val):
                t = float(np.clip((val - v_lo) / vspan, 0.0, 1.0))
                rgba = list(cmap_obj(t))
            else:
                rgba = [0.5, 0.5, 0.5, 0.8]
            x = float(self.G.nodes[o]['x'])
            y = float(self.G.nodes[o]['y'])
            # napari points want (row, col) — same convention as the
            # existing Edges layer uses (row = H_mos - y).
            row = (float(H_mos) - y) if H_mos is not None else y
            pts.append([row, x]); symbols.append(sym); face.append(rgba)
        if not pts:
            return
        pts_np = np.asarray(pts, dtype=float)
        sym_np = np.asarray(symbols, dtype=object)
        face_np = np.asarray(face, dtype=float)
        if 'CarveMarkers' in self.viewer.layers:
            lyr = self.viewer.layers['CarveMarkers']
            lyr.data = pts_np
            try:
                lyr.symbol = sym_np
            except Exception:
                pass
            try:
                lyr.face_color = face_np
            except Exception:
                pass
            lyr.visible = True
        else:
            self.viewer.add_points(
                pts_np, name='CarveMarkers',
                symbol=sym_np,
                face_color=face_np,
                size=40,
                border_color='black', border_width=0.15,
            )

        # ── Boundary vessels (A/V stubs) as their own coloured shapes ─
        # These are the untagged edges from an A/V anchor to a porous
        # node.  Coloured the same way (current colormap) using the
        # `_resolve_field_value` on that edge.
        bv_lines = []; bv_cols = []
        for anc in list(A_a) + list(V_a):
            if not self.G.has_node(anc): continue
            for m in self.G.neighbors(anc):
                if m not in U: continue
                d = self.G.edges[m, anc]
                if d.get('radial_uncertain'): continue
                x1 = float(self.G.nodes[anc]['x']); y1 = float(self.G.nodes[anc]['y'])
                x2 = float(self.G.nodes[m]['x']);   y2 = float(self.G.nodes[m]['y'])
                row1 = (float(H_mos) - y1) if H_mos is not None else y1
                row2 = (float(H_mos) - y2) if H_mos is not None else y2
                bv_lines.append([[row1, x1], [row2, x2]])
                val = None
                if hasattr(self, '_resolve_field_value'):
                    val = self._resolve_field_value(anc, m)
                    if val is None:
                        val = self._resolve_field_value(m, anc)
                if val is not None and np.isfinite(val):
                    t = float(np.clip((float(val) - v_lo) / vspan, 0.0, 1.0))
                    bv_cols.append(list(cmap_obj(t)))
                else:
                    bv_cols.append([0.9, 0.6, 0.1, 0.9])   # fallback amber
        if bv_lines:
            bv_arr = np.asarray(bv_lines, dtype=float)
            bv_col_arr = np.asarray(bv_cols, dtype=float)
            if 'CarveBoundaryVessels' in self.viewer.layers:
                lyr = self.viewer.layers['CarveBoundaryVessels']
                lyr.data = list(bv_arr)
                try: lyr.edge_color = bv_col_arr
                except Exception:
                    try: lyr.border_color = bv_col_arr
                    except Exception: pass
                lyr.visible = True
            else:
                self.viewer.add_shapes(
                    list(bv_arr), shape_type='line',
                    name='CarveBoundaryVessels',
                    edge_color=bv_col_arr,
                    edge_width=8.0)
        else:
            if 'CarveBoundaryVessels' in self.viewer.layers:
                self.viewer.layers['CarveBoundaryVessels'].visible = False

    def _carved_sim_run(self):
        from qtpy.QtWidgets import QApplication
        import time
        self._carved_run_btn.setEnabled(False)
        self._carved_status.setPlainText("Starting ...")
        QApplication.processEvents()
        # Parse exclude-tiles text
        excl = set()
        txt = self._carved_excl_edit.text().strip()
        if txt:
            for t in txt.split(','):
                t = t.strip()
                if not t: continue
                try:
                    excl.add(int(t))
                except ValueError:
                    pass
        D = float(self._carved_D_spin.value())
        eta = float(getattr(self, '_carved_eta_spin', None).value()
                     if getattr(self, '_carved_eta_spin', None) is not None else 0.0)
        # Install η monkey-patch (constant-loss-tangent form D_eff = D·(1 − iη))
        from ..analyze import transmission_line as _tl_c
        if not hasattr(type(self), '_pristine_vessel_admittance'):
            type(self)._pristine_vessel_admittance = _tl_c._vessel_admittance
        _orig_va_c = type(self)._pristine_vessel_admittance
        _eta_c = eta
        def _patched_va_c(R_m, L_m, omega, mu=_tl_c.MU_DEFAULT,
                           rho=_tl_c.RHO_BLOOD, D=None):
            D_eff = (D if D is not None else 1e-6) * (1.0 - 1j * _eta_c)
            return _orig_va_c(R_m, L_m, omega, mu, rho, D_eff)
        _tl_c._vessel_admittance = _patched_va_c
        try:
            from .carved_sim_backend import run_carved_sim_full
            t0 = time.time()
            self._carved_status.appendPlainText(
                f"Excluding tiles: {sorted(excl) if excl else 'none'}\n"
                f"D_areal = {D:.2e}   η = {eta:.2f}\n"
                f"Running (see terminal for solver progress) ...")
            QApplication.processEvents()
            stats = run_carved_sim_full(
                self.G, exclude_tiles=excl if excl else None,
                D=D, verbose=True)
            # stamp η on the graph for downstream consumers
            self.G.graph['carved_sim_eta'] = float(eta)
            self.G.graph['carved_sim_D_areal'] = float(D)
            self.G.graph['carved_sim_wall_model'] = 'constant_loss_tangent'
            # Apply per-tile π-flip to carved sim fields (matches sync_v2
            # convention on measured; see _apply_pi_flip_and_persist_sim).
            self._apply_pi_flip_and_persist_sim(mode='carved')
            dt = time.time() - t0
            self._carved_status.appendPlainText(
                f"Done in {dt:.0f} s.\n"
                f"  v2 fields recomputed: {stats['n_v2']}\n"
                f"  outlets: {stats['n_outlets']}\n"
                f"  Σα (total): {stats['alpha_sum']:.3f} nL/s\n"
                f"  sim fields written: {stats['n_sim_fields']}")
            # Unlock the 'Simulated (carved)' radio in the View tab.
            self._refresh_carved_sim_radio_enable()
            self._carved_stat_lbl.setText(self._carved_sim_status_text())
        except Exception as e:
            import traceback
            self._carved_status.appendPlainText(
                f"FAILED: {e}\n{traceback.format_exc()}")
        finally:
            self._carved_run_btn.setEnabled(True)

    def _build_normalize_tab(self, layout):
        from qtpy.QtWidgets import (
            QDoubleSpinBox, QSpinBox, QCheckBox, QGridLayout, QLabel,
            QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
            QSizePolicy,
        )

        # ── Status readout ───────────────────────────────────────────
        status_group = QGroupBox("Normalization status")
        status_lay = QVBoxLayout(status_group)
        self._norm_status_label = QLabel()
        self._norm_status_label.setWordWrap(True)
        status_lay.addWidget(self._norm_status_label)
        layout.addWidget(status_group)

        # ── Parameters ───────────────────────────────────────────────
        params_group = QGroupBox("Parameters")
        grid = QGridLayout(params_group)
        row = 0

        def _spin(default, mn, mx, step=1.0, decimals=2, suffix=""):
            sp = QDoubleSpinBox()
            sp.setRange(float(mn), float(mx))
            sp.setSingleStep(float(step))
            sp.setDecimals(int(decimals))
            if suffix:
                sp.setSuffix(suffix)
            sp.setValue(float(default))
            return sp

        def _int_spin(default, mn, mx):
            sp = QSpinBox(); sp.setRange(int(mn), int(mx))
            sp.setValue(int(default))
            return sp

        # H1 primary min-SNR
        grid.addWidget(QLabel("H1 primary min-SNR (dB)"), row, 0)
        self._norm_h1_snr = _spin(3.0, -20.0, 30.0, step=0.5, decimals=1, suffix=" dB")
        self._norm_h1_snr.setToolTip(
            "Minimum SNR for a shared-vessel observation to enter the "
            "primary H1 sync.  Lower → more tiles in the primary set, "
            "noisier gauges.  3 dB is a robust default.")
        grid.addWidget(self._norm_h1_snr, row, 1); row += 1

        # H2 primary min-SNR
        grid.addWidget(QLabel("H2 primary min-SNR (dB)"), row, 0)
        self._norm_h2_snr = _spin(0.0, -20.0, 30.0, step=0.5, decimals=1, suffix=" dB")
        self._norm_h2_snr.setToolTip(
            "H2 has 6–12 dB less amplitude than H1, so a lower gate makes "
            "sense.  0 dB is the default.")
        grid.addWidget(self._norm_h2_snr, row, 1); row += 1

        # Fill-in gate
        grid.addWidget(QLabel("Fill-in min-SNR (dB)"), row, 0)
        self._norm_fillin_snr = _spin(-6.0, -20.0, 20.0, step=0.5, decimals=1, suffix=" dB")
        self._norm_fillin_snr.setToolTip(
            "Per-observation gate for the multi-hop fill-in stage.  Below "
            "-6 dB, phasor noise dominates and estimates become unreliable.")
        grid.addWidget(self._norm_fillin_snr, row, 1); row += 1

        # Min R̄
        grid.addWidget(QLabel("Fill-in min R̄ (phase concentration)"), row, 0)
        self._norm_min_rbar = _spin(0.40, 0.0, 1.0, step=0.05, decimals=2)
        self._norm_min_rbar.setToolTip(
            "Reject fill-ins whose per-observation z_est estimates disagree "
            "too much (Rayleigh concentration below this threshold — random "
            "phase gives R̄→0, perfect agreement R̄=1).")
        grid.addWidget(self._norm_min_rbar, row, 1); row += 1

        # Max hops
        grid.addWidget(QLabel("Max fill-in hops"), row, 0)
        self._norm_max_hops = _int_spin(5, 1, 10)
        self._norm_max_hops.setToolTip(
            "Number of BFS-like hops the fill-in loop will try before falling "
            "through to the nearest-neighbor-borrow stage.")
        grid.addWidget(self._norm_max_hops, row, 1); row += 1

        # Checkboxes
        self._norm_compute_h2 = QCheckBox("Compute H2 gauges")
        self._norm_compute_h2.setChecked(True)
        self._norm_compute_h2.setToolTip(
            "Run a parallel sync at 2·f0.  Costs ~30-60 s more but lets the "
            "viewer show H2 amplitude/phase fields without falling back to "
            "the H1 gauge.")
        grid.addWidget(self._norm_compute_h2, row, 0, 1, 2); row += 1

        self._norm_signflip = QCheckBox("Run sign-flip rescue")
        self._norm_signflip.setChecked(True)
        self._norm_signflip.setToolTip(
            "Post-sync pass that flips any tile whose per-neighbor phase "
            "residual drops below the threshold below after inversion — "
            "fixes residual 180° discontinuities in low-SNR regions.")
        grid.addWidget(self._norm_signflip, row, 0, 1, 2); row += 1

        grid.addWidget(QLabel("Sign-flip max post-flip |resid| (°)"), row, 0)
        self._norm_signflip_max = _spin(60.0, 20.0, 90.0, step=5.0,
                                          decimals=0, suffix=" °")
        self._norm_signflip_max.setToolTip(
            "A tile flip is COMMITTED only if the post-flip per-neighbor "
            "phase |resid| lands below this.  Loosening it (60–75°) helps "
            "for low-pulsatility tiles whose H1 phase is intrinsically "
            "noisy — flipping to 50° is still a big win over 130°.  "
            "Tightening it (30°) is stricter and flips fewer tiles.")
        grid.addWidget(self._norm_signflip_max, row, 1); row += 1

        self._norm_level_ac = QCheckBox(
            "Level AC amplitude (divide by |z_H1|, |z_H2|)")
        self._norm_level_ac.setChecked(False)
        self._norm_level_ac.setToolTip(
            "OFF (default, recommended for real biology): normalize DC "
            "and align AC phase, but leave |H1|/|H2| unchanged so per-tile "
            "pulsatility is preserved as a biological signal.\n\n"
            "ON: also divide AC by |z_H_k|.  Unified gauge model — forces "
            "normalized PI ≈ constant across tiles.  Useful for cross-tile "
            "waveform-shape checks, but destroys regional PI variation "
            "(e.g. heart-adjacent tiles are truly more pulsatile).")
        grid.addWidget(self._norm_level_ac, row, 0, 1, 2); row += 1

        # Amp-gauge min SNR
        grid.addWidget(QLabel("Amp-gauge min-SNR (dB)"), row, 0)
        self._norm_amp_snr = _spin(6.0, 0.0, 30.0, step=0.5, decimals=1, suffix=" dB")
        self._norm_amp_snr.setToolTip(
            "Minimum H1 SNR for a measurement to contribute to the PI_rel "
            "and CO_rel amplitude leveling.  Independent from the phase "
            "sync gate; usually stricter since amplitude leveling amplifies "
            "any bias.")
        grid.addWidget(self._norm_amp_snr, row, 1); row += 1

        layout.addWidget(params_group)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._norm_compute_btn = QPushButton("Compute normalization")
        self._norm_compute_btn.setToolTip(
            "Run the full pipeline (H1 sync + fill-in + H2 sync + fill-in + "
            "sign-flip rescue + amp gauges + apply).  Populates "
            "measurements_piv_normalized on every edge and enables the "
            "'Normalized' data source.")
        self._norm_compute_btn.clicked.connect(self._on_normalize_clicked)
        btn_row.addWidget(self._norm_compute_btn)

        self._norm_show_gauges_btn = QPushButton("Show gauges…")
        self._norm_show_gauges_btn.setToolTip(
            "Pop a matplotlib figure summarizing every per-tile gauge "
            "(G_DC, G_H1, G_H2, f0) with bars color-coded by source "
            "(primary / fillin / borrowed).  Quick QC of what the pipeline "
            "recovered — spot outliers, thin-coverage tiles, and sign-flip "
            "candidates at a glance.")
        self._norm_show_gauges_btn.setEnabled(self._normalization_available())
        self._norm_show_gauges_btn.clicked.connect(self._on_show_gauges_clicked)
        btn_row.addWidget(self._norm_show_gauges_btn)

        self._norm_diag_btn = QPushButton("Show diagnostics…")
        self._norm_diag_btn.setToolTip(
            "Pop a diagnostic figure showing per-pair residuals and "
            "per-tile weighted-mean residuals for each gauge "
            "(G_DC, G_H1 amp+phase, G_H2 amp+phase).  Residuals are the "
            "mismatch between observed shared-vessel ratios and what the "
            "recovered gauges predict — large residuals for a tile mean "
            "its gauge doesn't explain its data well (unreliable fill-in, "
            "sign-flip candidate, or genuinely inconsistent).")
        self._norm_diag_btn.setEnabled(self._normalization_available())
        self._norm_diag_btn.clicked.connect(self._on_show_diagnostics_clicked)
        btn_row.addWidget(self._norm_diag_btn)

        self._norm_save_btn = QPushButton("Save normalized graph…")
        self._norm_save_btn.setToolTip(
            "Write the graph (raw + normalized fields) to disk as a new "
            "*_normalized.gpickle so a later viewer launch has the "
            "normalization ready without re-computing.")
        self._norm_save_btn.setEnabled(self._normalization_available())
        self._norm_save_btn.clicked.connect(self._on_save_normalized_clicked)
        btn_row.addWidget(self._norm_save_btn)
        layout.addLayout(btn_row)

        # ── Progress log ──────────────────────────────────────────────
        self._norm_log = QPlainTextEdit()
        self._norm_log.setReadOnly(True)
        self._norm_log.setPlaceholderText(
            "Progress and per-stage results will appear here.")
        self._norm_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._norm_log.setMinimumHeight(180)
        layout.addWidget(self._norm_log)

        # Fill in the status label now
        self._refresh_normalize_status()

    def _refresh_normalize_status(self):
        """Update the Normalize-tab status label based on
        `G.graph['tile_gauges']` and any stored normalization params."""
        if not hasattr(self, '_norm_status_label'):
            return
        tg = self.G.graph.get('tile_gauges') or {}
        if not tg:
            self._norm_status_label.setText(
                "<b>Not computed.</b>  Set parameters below and click "
                "<i>Compute normalization</i> to run the pipeline.  The "
                "'Normalized' Data-source radio will unlock afterward.")
            return
        f0 = self.G.graph.get('f0_canonical_hz')
        stats = self.G.graph.get('tile_normalize_stats') or {}
        n_edges = stats.get('n_edges_touched', '?')
        n_meas = stats.get('n_measurements', '?')
        n_fb = stats.get('n_h2_fallback', '?')
        self._norm_status_label.setText(
            f"<b>Normalized.</b>  {len(tg)} tiles have gauges; "
            f"{n_edges} edges normalized ({n_meas} measurements, "
            f"{n_fb} H2 measurements fell back to H1 gauge). "
            f"Canonical f0 = {f0:.3f} Hz." if f0 else
            f"<b>Normalized.</b>  {len(tg)} tiles have gauges.")

    def _log_norm(self, msg: str):
        """Append a line to the Normalize-tab progress log."""
        if hasattr(self, '_norm_log'):
            self._norm_log.appendPlainText(msg)
            # Process events so the log paints even while a long compute
            # is running on the UI thread (poor person's threading).
            try:
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass

    def _on_normalize_clicked(self):
        """Run the tile_normalize pipeline in-place on self.G."""
        from ..mosaic.tile_normalize import (
            NormalizeParams, normalize_graph_in_place)
        params = NormalizeParams(
            min_snr_db_h1=float(self._norm_h1_snr.value()),
            min_snr_db_h2=float(self._norm_h2_snr.value()),
            fillin_min_snr_db=float(self._norm_fillin_snr.value()),
            fillin_min_rbar=float(self._norm_min_rbar.value()),
            max_hops=int(self._norm_max_hops.value()),
            compute_h2=bool(self._norm_compute_h2.isChecked()),
            run_sign_flip_rescue=bool(self._norm_signflip.isChecked()),
            rescue_final_max_deg=float(self._norm_signflip_max.value()),
            amp_min_snr_db=float(self._norm_amp_snr.value()),
            level_ac_amplitude=bool(self._norm_level_ac.isChecked()),
        )
        self._norm_log.clear()
        self._log_norm("Starting normalization with:")
        for f, v in [
                ('  H1 min-SNR',    f'{params.min_snr_db_h1} dB'),
                ('  H2 min-SNR',    f'{params.min_snr_db_h2} dB' if params.compute_h2 else '(H2 disabled)'),
                ('  Fill-in gate',  f'{params.fillin_min_snr_db} dB'),
                ('  Min R̄',        f'{params.fillin_min_rbar}'),
                ('  Max hops',      f'{params.max_hops}'),
                ('  Sign-flip',     'on' if params.run_sign_flip_rescue else 'off'),
                ('  Amp min-SNR',   f'{params.amp_min_snr_db} dB'),
                ('  Level AC amp',  'on (unified gauge)' if params.level_ac_amplitude
                                     else 'off (preserve PI)'),
        ]:
            self._log_norm(f'  {f:<20}{v}')
        self._log_norm('')
        self._norm_compute_btn.setEnabled(False)
        try:
            def _progress(stage, pct):
                self._log_norm(f'  [{pct*100:>3.0f}%] {stage}')
            stats = normalize_graph_in_place(
                self.G, params, progress=_progress)
            self._log_norm('')
            self._log_norm('=== DONE ===')
            for k, val in stats.items():
                if isinstance(val, list) and len(val) > 12:
                    val = f'({len(val)} items)'
                self._log_norm(f'  {k}: {val}')
        except Exception as e:
            import traceback
            self._log_norm(f'ERROR: {e}')
            self._log_norm(traceback.format_exc())
        finally:
            self._norm_compute_btn.setEnabled(True)
            # Refresh status, unlock the Normalized radio, and rebuild
            # anything that depends on the fresh cache.
            self._refresh_normalize_status()
            self._refresh_source_radio_enable()
            self._norm_save_btn.setEnabled(self._normalization_available())
            self._norm_show_gauges_btn.setEnabled(self._normalization_available())
            self._norm_diag_btn.setEnabled(self._normalization_available())
            if hasattr(self, '_tile_harmonic_cache'):
                self._tile_harmonic_cache.clear()
                if (getattr(self, 'current_tile_filter', None) is not None
                        and self.current_tile_filter != 'All'):
                    self._populate_tile_harmonic_cache(
                        self.current_tile_filter)
            self._refresh_edges()
            self._refresh_nodes()
            self._refresh_cbar()

    def _on_show_gauges_clicked(self):
        """Pop a matplotlib figure summarizing all per-tile gauges under
        the unified `G_DC / G_H1 / G_H2` model.

        Layout (bar plots, one row per gauge quantity):
          G_DC = CO_rel      (source=amp)
          |G_H1| = |z_H1|    (source=h1)
          arg(G_H1)          (source=h1)
          |G_H2| = |z_H2|    (source=h2)
          arg(G_H2)          (source=h2)
          f0_hz              (source=amp)
        Bars are color-coded by source (primary / fillin / borrowed) so
        the reliability of each gauge is visible at a glance.
        """
        tg = self.G.graph.get('tile_gauges') or {}
        if not tg:
            return
        import matplotlib.pyplot as plt
        import numpy as np
        tiles = sorted(tg.keys())
        n = len(tiles)
        xs = np.arange(n)

        def _colors(field: str):
            palette = {
                'primary': '#2b8cbe',
                'fillin':  '#fdae6b',
                'borrowed': '#de2d26',
                None:      '#bdbdbd',
            }
            return [palette.get(tg[t].get(field), '#bdbdbd') for t in tiles]

        def _bar_ax(ax, values, colors, ylabel, title,
                    baseline: Optional[float] = None):
            arr = np.array([np.nan if v is None else float(v)
                            for v in values])
            ax.bar(xs, arr, color=colors, edgecolor='#333', linewidth=0.4)
            if baseline is not None:
                ax.axhline(baseline, color='k', lw=0.7, ls='--', alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlim(-0.5, n - 0.5)
            ax.grid(axis='y', alpha=0.3)

        # 6-row layout: G_DC, |G_H1|, arg(G_H1), |G_H2|, arg(G_H2), f0
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(6, 1, hspace=0.55)

        # G_DC
        ax = fig.add_subplot(gs[0, 0])
        _bar_ax(ax, [tg[t].get('CO_rel') for t in tiles],
                _colors('source_amp'),
                'G_DC  (= CO_rel)',
                'G_DC — per-tile mean_Q leveling factor',
                baseline=1.0)

        # |G_H1|
        ax = fig.add_subplot(gs[1, 0])
        _bar_ax(ax, [tg[t].get('z_mag') for t in tiles],
                _colors('source_h1'),
                '|G_H1|',
                '|G_H1| — H1 amplitude gauge',
                baseline=1.0)

        # arg(G_H1)
        ax = fig.add_subplot(gs[2, 0])
        _bar_ax(ax, [tg[t].get('z_arg_deg') for t in tiles],
                _colors('source_h1'),
                'arg(G_H1)  (°)',
                'arg(G_H1) — H1 phase gauge',
                baseline=0.0)

        # |G_H2|
        ax = fig.add_subplot(gs[3, 0])
        _bar_ax(ax, [tg[t].get('z_H2_mag') for t in tiles],
                _colors('source_h2'),
                '|G_H2|',
                '|G_H2| — H2 amplitude gauge (grey bars = no H2 gauge; H1 fallback used)',
                baseline=1.0)

        # arg(G_H2)
        ax = fig.add_subplot(gs[4, 0])
        _bar_ax(ax, [tg[t].get('z_H2_arg_deg') for t in tiles],
                _colors('source_h2'),
                'arg(G_H2)  (°)',
                'arg(G_H2) — H2 phase gauge (grey bars = no H2 gauge; H1 fallback used)',
                baseline=0.0)

        # f0
        ax = fig.add_subplot(gs[5, 0])
        _bar_ax(ax, [tg[t].get('f0_hz') for t in tiles],
                _colors('source_amp'),
                'f0 (Hz)',
                'f0 — per-tile median fundamental (measured)',
                baseline=(self.G.graph.get('f0_canonical_hz') or None))

        # X-axis labels only on the last row
        for i, ax in enumerate(fig.axes):
            if i < len(fig.axes) - 1:
                ax.set_xticks(xs)
                ax.set_xticklabels([])
            else:
                ax.set_xticks(xs)
                ax.set_xticklabels([str(t) for t in tiles],
                                    rotation=90, fontsize=7)
                ax.set_xlabel('tile ID', fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor='#2b8cbe', edgecolor='#333', label='primary'),
            Patch(facecolor='#fdae6b', edgecolor='#333', label='fillin'),
            Patch(facecolor='#de2d26', edgecolor='#333', label='borrowed'),
            Patch(facecolor='#bdbdbd', edgecolor='#333', label='missing'),
        ]
        fig.legend(handles=handles, loc='upper right', ncol=4,
                    fontsize=9, bbox_to_anchor=(0.98, 0.995))
        f0c = self.G.graph.get('f0_canonical_hz')
        fig.suptitle(
            f"Per-tile normalization gauges  ({n} tiles, "
            f"f0_canonical = {f0c:.3f} Hz)" if f0c else
            f"Per-tile normalization gauges ({n} tiles)",
            fontsize=13, y=0.995)
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

    def _on_show_diagnostics_clicked(self):
        """Compute residuals against the stored gauges and plot per-tile /
        per-pair diagnostics.  For each gauge, `residual` = mismatch
        between observed shared-vessel ratios and what the recovered
        gauge predicts.  A large per-tile residual means that tile's
        gauge does NOT explain its own shared-vessel data well — either
        the gauge is wrong (bad fill-in / wrong sign) OR the tile's data
        is inconsistent with a single-scalar-gauge model.
        """
        if not self._normalization_available():
            return
        from ..mosaic.tile_normalize import (
            compute_gauge_diagnostics, NormalizeParams)
        params = NormalizeParams(
            fillin_min_snr_db=float(self._norm_fillin_snr.value()),
            amp_min_snr_db=float(self._norm_amp_snr.value()),
        )
        self._log_norm('Computing gauge diagnostics …')
        diag = compute_gauge_diagnostics(self.G, params)
        self._log_norm(
            f'  H1: {len(diag["pair_phase_H1"])} phase pairs, '
            f'{len(diag["pair_amp_H1"])} amp pairs, '
            f'{len(diag["per_tile_phase_H1"])} tiles covered')
        if diag['pair_phase_H2']:
            self._log_norm(
                f'  H2: {len(diag["pair_phase_H2"])} phase pairs, '
                f'{len(diag["per_tile_phase_H2"])} tiles covered')
        self._log_norm(
            f'  DC: {len(diag["pair_amp_DC"])} pairs, '
            f'{len(diag["per_tile_amp_DC"])} tiles covered')

        import matplotlib.pyplot as plt
        import numpy as np

        # Two figures: one for per-pair histograms, one for per-tile bars.
        # Per-pair histograms are compact; per-tile bars are wide.

        # ── Figure 1: per-pair residual histograms (3 rows × 2 cols) ──
        fig1, axes = plt.subplots(3, 2, figsize=(13, 9))
        row_labels = [
            ('DC log-amp',  diag['pair_amp_DC'],   'log(mean_Q ratio)'),
            ('H1 log-amp',  diag['pair_amp_H1'],   'log(|Q_H1| ratio)'),
            ('H2 log-amp',  diag['pair_amp_H2'],   'log(|Q_H2| ratio)'),
        ]
        for row, (label, recs, xlab) in enumerate(row_labels):
            ax = axes[row, 0]
            if not recs:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            else:
                vals = np.array([r[2] for r in recs])
                ws = np.array([r[3] for r in recs])
                ax.hist(vals, bins=50, weights=ws, color='#2b8cbe',
                        edgecolor='#333', linewidth=0.3)
                ax.axvline(0, color='k', ls='--', lw=0.7, alpha=0.5)
                med = float(np.average(np.abs(vals), weights=ws))
                ax.set_title(f'{label} pair residuals — '
                             f'weighted mean |resid| = {med:.3f}',
                             fontsize=9)
            ax.set_xlabel(xlab, fontsize=8)
            ax.set_ylabel('weight-sum', fontsize=8)
            ax.grid(alpha=0.3)

        phase_rows = [
            ('H1 phase', diag['pair_phase_H1']),
            ('H2 phase', diag['pair_phase_H2']),
        ]
        for i, (label, recs) in enumerate(phase_rows):
            ax = axes[i, 1]
            if not recs:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            else:
                vals = np.array([r[2] for r in recs])
                ws = np.array([r[3] for r in recs])
                ax.hist(vals, bins=50, weights=ws, color='#fdae6b',
                        edgecolor='#333', linewidth=0.3, range=(-180, 180))
                ax.axvline(0, color='k', ls='--', lw=0.7, alpha=0.5)
                med = float(np.average(np.abs(vals), weights=ws))
                ax.set_title(f'{label} pair residuals — '
                             f'weighted mean |resid| = {med:.2f}°',
                             fontsize=9)
                ax.set_xlim(-180, 180)
            ax.set_xlabel('resid (°)', fontsize=8)
            ax.set_ylabel('weight-sum', fontsize=8)
            ax.grid(alpha=0.3)

        # Empty bottom-right — put a summary table there
        ax = axes[2, 1]
        ax.axis('off')
        tg = diag['gauges']
        n_tiles = len(tg)
        n_prim_amp = sum(1 for g in tg.values() if g.get('source_amp') == 'primary')
        n_fill_amp = sum(1 for g in tg.values() if g.get('source_amp') == 'fillin')
        n_brw_amp  = sum(1 for g in tg.values() if g.get('source_amp') == 'borrowed')
        n_prim_h1 = sum(1 for g in tg.values() if g.get('source_h1') == 'primary')
        n_fill_h1 = sum(1 for g in tg.values() if g.get('source_h1') == 'fillin')
        n_brw_h1  = sum(1 for g in tg.values() if g.get('source_h1') == 'borrowed')
        n_prim_h2 = sum(1 for g in tg.values() if g.get('source_h2') == 'primary')
        n_fill_h2 = sum(1 for g in tg.values() if g.get('source_h2') == 'fillin')
        n_brw_h2  = sum(1 for g in tg.values() if g.get('source_h2') == 'borrowed')
        f0c = self.G.graph.get('f0_canonical_hz')
        txt = (
            f'Gauge source counts ({n_tiles} tiles total)\n\n'
            f'  G_DC :  primary {n_prim_amp:2d}   fillin {n_fill_amp:2d}   borrowed {n_brw_amp:2d}\n'
            f'  G_H1 :  primary {n_prim_h1:2d}   fillin {n_fill_h1:2d}   borrowed {n_brw_h1:2d}\n'
            f'  G_H2 :  primary {n_prim_h2:2d}   fillin {n_fill_h2:2d}   borrowed {n_brw_h2:2d}\n\n'
            f'  f0_canonical = {f0c:.3f} Hz' if f0c else ''
        )
        ax.text(0.05, 0.95, txt, family='monospace', fontsize=10,
                va='top', transform=ax.transAxes)
        fig1.suptitle('Gauge diagnostics — per-pair residuals '
                      '(observed offset vs recovered-gauge prediction)',
                      fontsize=12)
        fig1.tight_layout()
        plt.show(block=False)
        try:
            _place_figure_top_right(fig1)
        except Exception:
            pass

        # ── Figure 2: per-tile weighted-mean residuals as bar plot ──
        tiles = sorted(tg.keys())
        n = len(tiles)
        xs = np.arange(n)

        def _colors(field: str):
            palette = {
                'primary': '#2b8cbe', 'fillin': '#fdae6b',
                'borrowed': '#de2d26', None: '#bdbdbd'}
            return [palette.get(tg[t].get(field), '#bdbdbd') for t in tiles]

        def _bar_ax(ax, values, colors, ylabel, title, obs_counts=None):
            arr = np.array([np.nan if v is None else float(v) for v in values])
            ax.bar(xs, arr, color=colors, edgecolor='#333', linewidth=0.4)
            ax.axhline(0, color='k', lw=0.5, alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlim(-0.5, n - 0.5)
            ax.grid(axis='y', alpha=0.3)
            if obs_counts is not None:
                # Annotate obs count below each bar as small numbers
                ax2 = ax.twinx()
                ax2.set_ylim(0, 1)
                ax2.set_yticks([])
                for i, t in enumerate(tiles):
                    n_obs = obs_counts.get(t, 0)
                    if n_obs > 0:
                        ax2.text(i, 0.02, str(n_obs), ha='center',
                                  va='bottom', fontsize=6, color='#555')

        fig2 = plt.figure(figsize=(14, 12))
        gs = fig2.add_gridspec(5, 1, hspace=0.55)

        # DC log-amp per-tile residual
        pt = diag['per_tile_amp_DC']
        ax = fig2.add_subplot(gs[0, 0])
        _bar_ax(ax,
                [pt.get(t, (None,))[0] for t in tiles],
                _colors('source_amp'),
                '|resid|',
                'G_DC per-tile: weighted mean |log(mean_Q ratio) - log(CO ratio)|. '
                'Small = gauge explains data; large = inconsistent.',
                obs_counts=diag['obs_count_DC'])

        # H1 amp
        pt = diag['per_tile_amp_H1']
        ax = fig2.add_subplot(gs[1, 0])
        _bar_ax(ax,
                [pt.get(t, (None,))[0] for t in tiles],
                _colors('source_h1'),
                '|resid|',
                '|G_H1| per-tile: weighted mean |log(|Q_H1| ratio) - log(|z_H1| ratio)|.',
                obs_counts=diag['obs_count_H1'])

        # H1 phase
        pt = diag['per_tile_phase_H1']
        ax = fig2.add_subplot(gs[2, 0])
        _bar_ax(ax,
                [pt.get(t, (None,))[0] for t in tiles],
                _colors('source_h1'),
                '|resid| (°)',
                'arg(G_H1) per-tile: weighted mean |phase resid (°)|. '
                'Values >45° flag sign-flip / bad-fill candidates.',
                obs_counts=diag['obs_count_H1'])

        # H2 amp
        pt = diag['per_tile_amp_H2']
        ax = fig2.add_subplot(gs[3, 0])
        _bar_ax(ax,
                [pt.get(t, (None,))[0] for t in tiles],
                _colors('source_h2'),
                '|resid|',
                '|G_H2| per-tile: weighted mean |log ratio| resid.',
                obs_counts=diag['obs_count_H2'])

        # H2 phase
        pt = diag['per_tile_phase_H2']
        ax = fig2.add_subplot(gs[4, 0])
        _bar_ax(ax,
                [pt.get(t, (None,))[0] for t in tiles],
                _colors('source_h2'),
                '|resid| (°)',
                'arg(G_H2) per-tile: weighted mean |phase resid (°)|.',
                obs_counts=diag['obs_count_H2'])

        for i, ax in enumerate(fig2.axes[::2]):   # only "real" axes, skip twins
            if i < 4:
                ax.set_xticks(xs); ax.set_xticklabels([])
            else:
                ax.set_xticks(xs)
                ax.set_xticklabels([str(t) for t in tiles],
                                    rotation=90, fontsize=7)
                ax.set_xlabel('tile ID (small grey number = obs count above bar 0)',
                               fontsize=9)

        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor='#2b8cbe', edgecolor='#333', label='primary'),
            Patch(facecolor='#fdae6b', edgecolor='#333', label='fillin'),
            Patch(facecolor='#de2d26', edgecolor='#333', label='borrowed'),
            Patch(facecolor='#bdbdbd', edgecolor='#333', label='missing'),
        ]
        fig2.legend(handles=handles, loc='upper right', ncol=4,
                     fontsize=9, bbox_to_anchor=(0.98, 0.995))
        fig2.suptitle(
            'Per-tile weighted-mean residuals — how well each tile\'s '
            'gauge explains its shared-vessel observations.  Tall bars '
            '= gauge unreliable.', fontsize=12, y=0.995)
        plt.show(block=False)
        try:
            _place_figure_top_right(fig2)
        except Exception:
            pass

    def _on_save_normalized_clicked(self):
        """Write the current graph (raw + normalized fields) to a
        `*_normalized.gpickle` file next to the source graph."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        default_name = self.graph_path.with_name(
            self.graph_path.stem + '_normalized.gpickle')
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save normalized graph",
            str(default_name),
            "Pickled graph (*.gpickle);;All files (*)")
        if not path:
            return
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.G, f)
            self._log_norm(f'Saved normalized graph → {path}')
            QMessageBox.information(
                self.viewer.window._qt_window,
                "Saved",
                f"Wrote {path}")
        except Exception as exc:
            self._log_norm(f'Save failed: {exc}')
            QMessageBox.critical(
                self.viewer.window._qt_window,
                "Save failed",
                f"Could not save graph:\n{exc}")

    # =================================================================
    # Edit tab: boundary marking + recursive-KCL bc_harmonics extraction.
    # This is the ONE mutation path in the "read-only" viewer — opt-in via
    # the Edit tab's mode selector, otherwise every other tab stays purely
    # visual.  See docs/graph_data_guide.md §2 for the bc_* schema.
    # =================================================================
    def _build_edit_panel(self):
        """Boundary-editing tab.  See class docstring / graph_data_guide §2.

        Mode selector (radio group):
          - Off (default)              — clicks route to the normal
                                          edge-inspection path.
          - Mark source (Shift-click)  — click a node → boundary_type='source'
                                          + recursive junction-KCL bc_harmonics.
          - Mark sink (Ctrl-click)     — click a node → boundary_type='sink'
                                          + recursive junction-KCL bc_harmonics.
          - Mark edge                  — click a node → boundary_type='edge'
                                          (toggle; no bc_harmonics).
          - Clear boundary             — click a node → remove boundary_type
                                          and all bc_* attributes.

        Recursive-KCL parameters:
          - max_depth (default 5)
          - snr_threshold_db (default 3.0)

        Save button writes G back to `self.graph_path` with a rolling
        `.pre_edit_<timestamp>.gpickle` backup.  Auto-save is intentionally
        NOT enabled — you commit when you're ready.
        """
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
            QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton, QLabel,
            QTextEdit, QCheckBox)
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # Mode selector
        mode_group = QGroupBox("Boundary marking mode")
        mode_lay = QVBoxLayout(mode_group)
        self._edit_mode_btns = QButtonGroup(panel)
        for label, key, tip in [
            ("Off",              None,     "Read-only mode (default)."),
            ("Mark source",     'source', "Click a node → tag as arterial 'source' + recursive junction-KCL."),
            ("Mark sink",       'sink',   "Click a node → tag as venous 'sink' + recursive junction-KCL."),
            ("Mark edge",       'edge',   "Click a node → toggle 'edge' tag (no bc_harmonics)."),
            ("Clear at click",  'clear',  "Click a boundary node → remove ITS boundary_type + all its bc_* attrs.  Only the clicked node is affected; other boundaries are untouched."),
            ("Move junction",   'move_junction', "TWO-CLICK: (1) click a graph node that's adjacent to a boundary node (a 'junction'), (2) click a new node.  Re-routes the boundary edge from (old-junction ↔ boundary) to (new-junction ↔ boundary), then re-runs junction-KCL at the boundary node with the new incident vessel.  Use when the current junction is too deep in a low-signal vein — pick a better-signal node closer to the plexus."),
        ]:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setChecked(key is None)
            rb.toggled.connect(lambda checked, k=key: self._set_edit_mode(k) if checked else None)
            mode_lay.addWidget(rb)
            self._edit_mode_btns.addButton(rb)
        lay.addWidget(mode_group)

        # Recursive-KCL parameters
        kcl_group = QGroupBox("Recursive junction-KCL parameters")
        kcl_lay = QGridLayout(kcl_group)
        kcl_lay.addWidget(QLabel("Max depth:"), 0, 0)
        self._edit_max_depth = QSpinBox(); self._edit_max_depth.setRange(1, 10)
        self._edit_max_depth.setValue(5)
        self._edit_max_depth.setToolTip("How deep to walk into the network to characterize unmeasured branches (each hop invokes one more mass-conservation constraint).")
        kcl_lay.addWidget(self._edit_max_depth, 0, 1)
        kcl_lay.addWidget(QLabel("SNR threshold (dB):"), 1, 0)
        self._edit_snr_thresh = QDoubleSpinBox()
        self._edit_snr_thresh.setRange(0, 30); self._edit_snr_thresh.setSingleStep(1.0)
        self._edit_snr_thresh.setValue(3.0)
        self._edit_snr_thresh.setToolTip("Edges below this SNR trigger recursion into the far end.")
        kcl_lay.addWidget(self._edit_snr_thresh, 1, 1)
        lay.addWidget(kcl_group)

        # Badge-label option: show measured DC flux next to A/V
        badge_group = QGroupBox("Boundary badge labels")
        badge_lay = QVBoxLayout(badge_group)
        self._edit_show_flux_chk = QCheckBox("Show DC flux on A/V badges")
        self._edit_show_flux_chk.setToolTip(
            "When checked, boundary badges show 'A +1.42' / 'V -0.97' "
            "instead of just 'A' / 'V'.  Sign convention: positive = "
            "injected into network (source), negative = removed (sink)."
            "  A sink whose DC came out positive → flow direction "
            "unresolved or mis-tagged.")
        self._edit_show_flux_chk.stateChanged.connect(
            lambda _st: self._refresh_boundary_layers())
        badge_lay.addWidget(self._edit_show_flux_chk)
        lay.addWidget(badge_group)

        # Save
        save_btn = QPushButton("Save graph to disk")
        save_btn.setToolTip(
            "Overwrite the loaded gpickle with the current in-memory graph.  "
            "Writes a .pre_edit_<timestamp>.gpickle backup first.  "
            "Atomic write (temp file + os.replace) so a crash mid-save "
            "can't corrupt the file.")
        save_btn.clicked.connect(self._save_graph_edits)
        lay.addWidget(save_btn)

        # Status log
        lay.addWidget(QLabel("Recent edits:"))
        self._edit_status = QTextEdit(); self._edit_status.setReadOnly(True)
        self._edit_status.setMaximumHeight(240)
        self._edit_status.setPlaceholderText("Recent boundary edits will appear here.")
        lay.addWidget(self._edit_status)

        lay.addStretch()
        return panel

    def _set_edit_mode(self, mode):
        self._edit_mode = mode
        if mode:
            msg = f"[Edit] mode = {mode!r} — click on the mosaic to apply."
        else:
            msg = "[Edit] off — clicks route to normal inspection."
        print(msg)
        if hasattr(self, '_edit_status'):
            self._edit_status.append(msg)

    def _on_edit_click(self, x_graph: float, y_graph: float,
                        max_px: float = 20.0):
        """Handle a click when an Edit-tab mode is active.

        `x_graph`, `y_graph` are in graph coordinates
        (x = G.nodes[n]['x'], y = mosaic_height - row).  Finds the
        nearest node globally (not just existing boundary nodes) and
        applies the current edit mode.
        """
        best_n, best_d = None, float('inf')
        for n in self.G.nodes():
            dx = self.G.nodes[n]['x'] - x_graph
            dy = self.G.nodes[n]['y'] - y_graph
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_n, best_d = n, d
        if best_n is None or best_d > max_px:
            msg = (f"click ({x_graph:.0f}, {y_graph:.0f}): "
                   f"no node within {max_px:.0f}px")
            self._edit_status.append(msg)
            return
        self._apply_edit_mode(int(best_n), best_d)

    def _apply_edit_mode(self, node: int, click_dist_px: float):
        """Execute the current edit mode's action on `node`."""
        mode = getattr(self, '_edit_mode', None)
        if mode is None:
            return
        pos = f"(x={self.G.nodes[node]['x']:.0f}, y={self.G.nodes[node]['y']:.0f})"
        if mode == 'clear':
            cur_bt = self.G.nodes[node].get('boundary_type')
            if cur_bt is None:
                self._edit_status.append(
                    f"node {node} {pos}: not a boundary — nothing to clear")
                return
            removed = []
            for k in ('boundary_type', 'bc_harmonics', 'bc_Q_t',
                       'bc_snr_db', 'bc_source', 'bc_n_edges',
                       'bc_junction', 'true_tip', 'max_depth_reached',
                       'n_recursions', 'edges_used'):
                if k in self.G.nodes[node]:
                    del self.G.nodes[node][k]
                    removed.append(k)
            self._edit_status.append(
                f"node {node} {pos}: was {cur_bt!r}, cleared "
                f"{len(removed)} attrs (this node only)")
        elif mode == 'edge':
            cur = self.G.nodes[node].get('boundary_type')
            if cur == 'edge':
                del self.G.nodes[node]['boundary_type']
                self._edit_status.append(f"node {node} {pos}: 'edge' removed")
            else:
                self.G.nodes[node]['boundary_type'] = 'edge'
                self._edit_status.append(f"node {node} {pos}: → edge")
        elif mode in ('source', 'sink'):
            self.G.nodes[node]['boundary_type'] = mode
            self.G.nodes[node]['true_tip'] = True
            try:
                from ..mosaic.kcl_reconstruct import (
                    compute_bc_harmonics_recursive)
                result = compute_bc_harmonics_recursive(
                    self.G, node, mode,
                    max_depth=int(self._edit_max_depth.value()),
                    snr_threshold_db=float(self._edit_snr_thresh.value()))
            except Exception as e:
                import traceback; traceback.print_exc()
                self._edit_status.append(
                    f"node {node} {pos}: tagged as {mode}, KCL failed: "
                    f"{type(e).__name__}: {e}")
                return
            if result is None:
                self._edit_status.append(
                    f"node {node} {pos}: {mode.upper()} but no bc_harmonics "
                    f"computable (no measured edges reachable)")
            else:
                for k in ('bc_harmonics', 'bc_Q_t', 'bc_snr_db',
                           'bc_source', 'edges_used', 'max_depth_reached',
                           'n_edges', 'n_recursions'):
                    v = result.get(k)
                    if v is not None:
                        self.G.nodes[node][k] = v
                bch = result['bc_harmonics']
                self._edit_status.append(
                    f"node {node} {pos}: {mode.upper()}  "
                    f"DC={bch[0].real:+.3f} nL/s  "
                    f"|H1|={abs(bch[1]):.3f}  "
                    f"snr={result['bc_snr_db']:.1f}dB  "
                    f"src={result['bc_source']}  "
                    f"n_edges={result['n_edges']}  "
                    f"depth={result['max_depth_reached']}")
        # Sync boundary lists + refresh the A/V marker layers so the newly
        # tagged node shows up (or the cleared one disappears) immediately.
        self._refresh_boundary_layers()

    def _refresh_boundary_layers(self):
        """Recompute _source_nodes / _sink_nodes / _boundary_nodes from
        current graph state, then update (or create) the "Arterial
        boundary (A)" and "Venous boundary (V)" Points layers so newly
        tagged nodes are visually badged.

        Handles the S27 case (0 boundary nodes at launch → layers were
        never created) by creating the layers on first mark.  Removes
        layers when a stage drops to zero of that type.
        """
        self._source_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'source']
        self._sink_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'sink']
        self._boundary_nodes = self._source_nodes + self._sink_nodes

        def _to_display(n):
            return [self.mosaic_height - self.G.nodes[n]['y'],
                    self.G.nodes[n]['x']]

        show_flux = bool(getattr(self, '_edit_show_flux_chk', None)
                          and self._edit_show_flux_chk.isChecked())

        def _label(n, letter):
            if not show_flux:
                return letter
            bch = self.G.nodes[n].get('bc_harmonics')
            if bch is None or not len(bch):
                return f'{letter} (—)'
            dc = float(np.real(bch[0]))
            return f'{letter} {dc:+.2f}'

        def _sync(existing_attr, node_list, letter, face_color, name):
            layer = getattr(self, existing_attr, None)
            if node_list:
                pts = np.array([_to_display(n) for n in node_list])
                strs = [_label(n, letter) for n in node_list]
                if layer is not None:
                    try:
                        layer.data = pts
                        # napari 0.5+ text is a TextManager; set .string
                        try:
                            layer.text.string = strs
                        except Exception:
                            # older napari: .text is a dict
                            layer.text = {'string': strs, 'size': 12,
                                          'color': 'white', 'anchor': 'center'}
                        layer.visible = True
                    except Exception:
                        import traceback; traceback.print_exc()
                else:
                    try:
                        new_layer = self.viewer.add_points(
                            pts, size=18, face_color=face_color,
                            border_color='white', name=name,
                            text={'string': strs, 'size': 12,
                                  'color': 'white', 'anchor': 'center'})
                    except TypeError:
                        new_layer = self.viewer.add_points(
                            pts, size=18, face_color=face_color,
                            edge_color='white', name=name,
                            text={'string': strs, 'size': 12,
                                  'color': 'white', 'anchor': 'center'})
                    setattr(self, existing_attr, new_layer)
            else:
                # Zero nodes of this type — hide the layer if it exists.
                if layer is not None:
                    try:
                        layer.data = np.empty((0, 2))
                        layer.visible = False
                    except Exception:
                        pass

        _sync('_source_layer', self._source_nodes, 'A',
              '#d62728', 'Arterial boundary (A)')
        _sync('_sink_layer', self._sink_nodes, 'V',
              '#1f77b4', 'Venous boundary (V)')

    def _save_graph_edits(self):
        """Persist Edit-tab changes to disk (atomic, with backup)."""
        from datetime import datetime
        import os, pickle, shutil, tempfile
        target = Path(self.graph_path).resolve()
        if not target.parent.is_dir():
            self._edit_status.append(
                f"save FAILED: parent dir {target.parent} not found")
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = target.with_suffix(f'.pre_edit_{stamp}.gpickle')
        try:
            shutil.copy(str(target), str(backup))
        except Exception as e:
            self._edit_status.append(
                f"save FAILED: could not copy backup: {e}")
            return
        try:
            fd, tmp = tempfile.mkstemp(
                suffix='.edit_save.tmp', dir=str(target.parent),
                prefix='.' + target.stem + '.')
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, target)
            size_mb = target.stat().st_size / (1024 ** 2)
            self._edit_status.append(
                f"SAVED {target.name} ({size_mb:.0f} MB) "
                f"[backup: {backup.name}]")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._edit_status.append(f"save FAILED: {e}")

    def _discover_fields(self) -> List[str]:
        """Keep only the fields actually present in this graph's data."""
        found: List[str] = []
        # Top-level edge attributes (geometry + precomputed harmonic_class).
        for f in ('radius', 'length', 'harmonic_class'):
            for _, _, d in self.G.edges(data=True):
                if f in d:
                    found.append(f); break
        # Dissipation lives under `_meas_dissipation` (precomputed at
        # load time); surface it as the user-facing `dissipation` field.
        for _, _, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                found.append('dissipation'); break
        # PIV-side fields
        for f in DEFAULT_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # Categorical PIV fields (quality_tier).  harmonic_class is also
        # categorical but lives top-level, so it's already in `found`.
        for f in CATEGORICAL_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # When a sim has been run, expose the sim-only fields — these
        # have no measured equivalent.  Other sim outputs (mean_Q,
        # amp_Q, PI, phase) share names with measured and route via
        # the Source toggle, not the dropdown.
        if getattr(self, '_sim_active', False):
            sim_only_checks = [
                ('pressure_mean',   '_sim_tmp_pressure_mean'),
                ('pressure_amp',    '_sim_tmp_pressure_amp'),
                ('pressure_amp_H2', '_sim_tmp_pressure_amp_H2'),
                ('pressure_phase',  '_sim_tmp_phase_P'),
                ('pressure_drop',   '_sim_tmp_pressure_drop'),
            ]
            for display_name, edge_key in sim_only_checks:
                for _, _, d in self.G.edges(data=True):
                    if edge_key in d:
                        found.append(display_name); break
        return found

    # ── napari + Qt UI ─────────────────────────────────────────────────
    def _setup_viewer(self):
        import napari
        title = f"Mosaic (read-only) — {self.graph_path.name}"
        self.viewer = napari.Viewer(title=title)

        # Background image (if available)
        if self.mosaic is not None:
            self.viewer.add_image(self.mosaic, name='Mosaic', opacity=0.75)
            # Optional vessel-mask overlay from the stage's mask/mask.tif
            # (same fused-pixel grid as the mosaic).  Hidden by default —
            # toggle its visibility in napari's layer list.
            try:
                if self.tiff_path is not None:
                    _mask_p = self.tiff_path.parent.parent / 'mask' / 'mask.tif'
                    if _mask_p.exists():
                        import tifffile as _tf
                        _m = _tf.imread(str(_mask_p))
                        if _m.ndim == 2 and _m.shape == self.mosaic.shape[:2]:
                            self.viewer.add_labels(
                                (_m > 0).astype('uint8'),
                                name='Vessel mask', opacity=0.4, visible=False)
                            print(f"  mask overlay: {_mask_p.name} "
                                  f"({int((_m > 0).sum())} px, toggle in layer list)")
            except Exception as _e:
                print(f"  (mask overlay skipped: {_e})")

        # Edges layer is built lazily by _refresh_edges (which removes the
        # previous one and re-adds a fresh one).  This sidesteps a napari
        # quirk where Shapes layers don't re-render colour after a bulk
        # .data + .edge_color update on an initially-empty layer.
        self._edges_layer = None
        # Nodes layer (hidden by default). napari ≥ 0.5 renamed edge_* →
        # border_*; keep an older-napari fallback for old envs.
        try:
            self._nodes_layer = self.viewer.add_points(
                np.empty((0, 2)), size=4, face_color='white',
                border_color='black', name='Nodes', visible=False,
            )
        except TypeError:
            self._nodes_layer = self.viewer.add_points(
                np.empty((0, 2)), size=4, face_color='white',
                edge_color='black', name='Nodes', visible=False,
            )
        # Comparison-set marker layer — large halo rings, per-point
        # coloured to match the comparison plot.  Sized larger than the
        # cyan click marker so both stay visible when the user clicks an
        # edge that's already in the comparison set.
        try:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                border_color='#1f77b4', border_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )
        except TypeError:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                edge_color='#1f77b4', edge_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )

        # Click-highlight marker.  napari ≥ 0.5 renamed edge_* → border_*
        # AND made border_width relative to point size by default (0-1).
        # Pass a relative width (~20% of size) which works on both old
        # and new napari API names; if your napari is too old to accept
        # `border_width`, swap it for `edge_width=0.2`.
        try:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                border_color='cyan', border_width=0.2,
                name='Selected', symbol='ring',
            )
        except TypeError:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                edge_color='cyan', edge_width=0.2,
                name='Selected', symbol='ring',
            )

        # Boundary-node badges — red dot + "A" at arterial sources, blue
        # dot + "V" at venous sinks.  Always visible above the edges.
        def _to_display(n):
            return [self.mosaic_height - self.G.nodes[n]['y'],
                    self.G.nodes[n]['x']]
        if self._source_nodes:
            src_pts = np.array([_to_display(n) for n in self._source_nodes])
            self._source_layer = self.viewer.add_points(
                src_pts, size=18, face_color='#d62728',
                border_color='white',
                name='Arterial boundary (A)',
                text={'string': ['A'] * len(self._source_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )
        if self._sink_nodes:
            snk_pts = np.array([_to_display(n) for n in self._sink_nodes])
            self._sink_layer = self.viewer.add_points(
                snk_pts, size=18, face_color='#1f77b4',
                border_color='white',
                name='Venous boundary (V)',
                text={'string': ['V'] * len(self._sink_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )

        # Click handler attached after every layer rebuild via
        # `_attach_edge_click`.
        # Use the viewer-wide mouse-drag callback so it survives layer
        # rebuilds: it looks up the click position and asks
        # _inspect_nearest_edge to do hit testing against our
        # precomputed midpoints.  Boundary-node hits are intercepted
        # first so the user can inspect BC harmonics.
        @self.viewer.mouse_drag_callbacks.append
        def _on_click(viewer, event):
            if event.type != 'mouse_press' or len(event.position) < 2:
                return
            # Region tag/untag armed: clicks belong to the shape being
            # drawn.  Inspecting here would rebuild the highlight layer,
            # which napari makes the active layer — stealing the clicks
            # from the region layer after a couple of presses.
            _rl = getattr(self, '_region_tag_layer', None)
            if (_rl is not None and _rl in self.viewer.layers
                    and str(_rl.mode).startswith('add_')):
                return
            row, col = self._world_to_mosaic_data(event.position)
            # Edit-tab boundary marking takes priority — when a mode is
            # active, every click is a node-tag action.
            if getattr(self, '_edit_mode', None):
                self._on_edit_click(col, self.mosaic_height - row)
                return
            if self._inspect_nearest_boundary_node(col, row):
                return
            # Click-to-tag: bypass the info-inspect flow and toggle the
            # nearest edge's radial_uncertain flag directly.  Same
            # hit-testing radius as the normal inspect click.
            if getattr(self, '_click_to_tag_mode', False):
                self._tag_nearest_edge(col, row)
                return
            self._inspect_nearest_edge(col, row)

    def _world_to_mosaic_data(self, world_pos):
        """Convert a napari world position (event.position) to mosaic
        data coordinates (row, col).

        With no rotation active, world == data for the Mosaic layer
        (identity affine), so this just returns the last two elements
        of world_pos.  With --rotate-cw/--rotate-ccw, the Mosaic layer
        carries an affine whose inverse maps the clicked world position
        back to the unrotated mosaic row/col — that's where the graph
        node coords live, so hit-testing against `_edge_midpoints`
        works correctly.
        """
        ref_layer = None
        for name in ('Mosaic', 'Nodes', 'Edges'):
            if name in self.viewer.layers:
                ref_layer = self.viewer.layers[name]
                break
        if ref_layer is None:
            return float(world_pos[-2]), float(world_pos[-1])
        try:
            data_pos = ref_layer.world_to_data(world_pos)
        except Exception:
            return float(world_pos[-2]), float(world_pos[-1])
        return float(data_pos[-2]), float(data_pos[-1])

    def _setup_panel(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
            QCheckBox, QPushButton, QGroupBox, QTextEdit, QTabWidget,
            QDoubleSpinBox, QScrollArea, QLineEdit,
        )
        # Radial-uncertainty state defaults — must exist before the View
        # tab's uncertainty checkbox reads them.
        if not hasattr(self, '_show_radial_uncertain'):
            self._show_radial_uncertain = False   # off by default
        if not hasattr(self, '_radial_tags_dirty'):
            self._radial_tags_dirty = False
        if not hasattr(self, '_click_to_tag_mode'):
            self._click_to_tag_mode = False
        if not hasattr(self, '_region_tag_layer'):
            self._region_tag_layer = None
        if not hasattr(self, '_region_action_pending'):
            self._region_action_pending = 'tag'
        if not hasattr(self, '_region_shape_pending'):
            self._region_shape_pending = 'add_rectangle'
        tabs = QTabWidget()
        view_panel = QWidget(); view_layout = QVBoxLayout(view_panel)
        ts_panel = QWidget();   ts_layout = QVBoxLayout(ts_panel)
        sim_panel = QWidget();  sim_layout = QVBoxLayout(sim_panel)
        norm_panel = QWidget(); norm_layout = QVBoxLayout(norm_panel)

        # ══════ Tab 1: VIEW ═══════════════════════════════════════════
        layout = view_layout    # alias for the existing widget code

        # ── Data source: Measured (PIV) / Normalized (tile-gauged) / Simulated ──
        from qtpy.QtWidgets import QRadioButton, QButtonGroup
        src_group = QGroupBox("Data source")
        src_lay = QHBoxLayout(src_group)
        self._src_radio_measured = QRadioButton("Measured")
        self._src_radio_measured.setChecked(self.current_source == 'measured')
        self._src_radio_norm = QRadioButton("Normalized")
        self._src_radio_norm.setChecked(self.current_source == 'normalized')
        self._src_radio_norm.setEnabled(self._normalization_available())
        if not self._normalization_available():
            self._src_radio_norm.setToolTip(
                "Compute or load a normalization in the Normalize tab to enable.")
        self._src_radio_sim = QRadioButton("Sim (intact)")
        self._src_radio_sim.setChecked(self.current_source == 'sim')
        self._src_radio_sim.setEnabled(self._sim_active)
        if not self._sim_active:
            self._src_radio_sim.setToolTip(
                "Run a simulation in the Simulate tab to enable.")
        # Carved sim option — enabled once any `_h_*_sim_carved` fields exist
        # on the graph (populated by the Carved Sim tab's Run button).
        self._src_radio_sim_carved = QRadioButton("Sim (carved)")
        self._src_radio_sim_carved.setChecked(
            self.current_source == 'sim_carved')
        self._src_radio_sim_carved.setEnabled(
            self._carved_sim_data_available())
        if not self._src_radio_sim_carved.isEnabled():
            self._src_radio_sim_carved.setToolTip(
                "Run the porous-carved sim in the Carved Sim tab to enable.")
        self._src_button_group = QButtonGroup(src_group)
        self._src_button_group.addButton(self._src_radio_measured)
        self._src_button_group.addButton(self._src_radio_norm)
        self._src_button_group.addButton(self._src_radio_sim)
        self._src_button_group.addButton(self._src_radio_sim_carved)
        self._src_radio_measured.toggled.connect(self._on_source_change)
        self._src_radio_norm.toggled.connect(self._on_source_change)
        self._src_radio_sim.toggled.connect(self._on_source_change)
        self._src_radio_sim_carved.toggled.connect(self._on_source_change)
        src_lay.addWidget(self._src_radio_measured)
        src_lay.addWidget(self._src_radio_norm)
        src_lay.addWidget(self._src_radio_sim)
        src_lay.addWidget(self._src_radio_sim_carved)
        src_lay.addStretch(1)
        layout.addWidget(src_group)

        # Field selector — four small combos compose the colormap.
        fld_group = QGroupBox("Color edges by")
        fld_lay = QVBoxLayout(fld_group)

        # Quantity row (Q / P; P disabled in Measured source)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel('Quantity:'))
        self._quantity_combo = QComboBox()
        self._quantity_combo.addItem('Q (flow)',     userData='Q')
        self._quantity_combo.addItem('P (pressure)', userData='P')
        self._quantity_combo.setCurrentIndex(
            0 if self.current_quantity == 'Q' else 1)
        self._quantity_combo.currentIndexChanged.connect(
            self._on_quantity_change)
        q_row.addWidget(self._quantity_combo); q_row.addStretch(1)
        fld_lay.addLayout(q_row)

        # Property row
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel('Property:'))
        self._property_combo = QComboBox()
        for key, label, _is_harm, _valid_in in PROPERTY_DEFS:
            self._property_combo.addItem(label, userData=key)
            if key == self.current_property:
                self._property_combo.setCurrentIndex(
                    self._property_combo.count() - 1)
        self._property_combo.currentIndexChanged.connect(
            self._on_property_change)
        p_row.addWidget(self._property_combo); p_row.addStretch(1)
        fld_lay.addLayout(p_row)

        # Harmonic row (only enabled for harmonic-keyed properties)
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel('Harmonic:'))
        self._harmonic_combo = QComboBox()
        for hk in HARMONIC_KEYS:
            self._harmonic_combo.addItem(HARMONIC_LABELS[hk], userData=hk)
            if hk == self.current_harmonic:
                self._harmonic_combo.setCurrentIndex(
                    self._harmonic_combo.count() - 1)
        self._harmonic_combo.currentIndexChanged.connect(
            self._on_harmonic_change)
        h_row.addWidget(self._harmonic_combo); h_row.addStretch(1)
        fld_lay.addLayout(h_row)

        # Sync enable-state of the combos (Quantity, Harmonic) and the
        # item flags inside Property to the current Source + selection.
        self._sync_field_combos()

        self._log_chk = QCheckBox("Log scale")
        self._log_chk.setToolTip("Useful for mean_Q, amp_Q (orders of magnitude).")
        self._log_chk.toggled.connect(self._on_log_toggle)
        fld_lay.addWidget(self._log_chk)

        # ── Radial-uncertainty tagging ──────────────────────────────
        # Small panel in the View tab so the user can visually mark
        # vessels whose radius they don't trust, without touching the
        # solver.  The flag persists as ``edge['radial_uncertain']=True``
        # on the graph — downstream analysis can filter on it.
        _ru_row = QWidget()
        _ru_lay = QVBoxLayout(_ru_row)
        _ru_lay.setContentsMargins(0, 4, 0, 0)
        self._radial_uncertain_chk = QCheckBox(
            "Highlight uncertain vessels (orange)")
        self._radial_uncertain_chk.setChecked(self._show_radial_uncertain)
        self._radial_uncertain_chk.setToolTip(
            "When on, edges flagged ``radial_uncertain=True`` render in "
            "orange regardless of the current colormap.  Toggle 'u' on a "
            "selected edge to add/remove the flag; press Save to persist.")
        self._radial_uncertain_chk.toggled.connect(
            self._on_radial_uncertain_toggle)
        _ru_lay.addWidget(self._radial_uncertain_chk)

        # Click-to-tag mode: while ON, every click on a vessel toggles
        # its tag instead of showing info.  Much faster for bulk tagging.
        self._click_to_tag_chk = QCheckBox("Click-to-tag mode")
        self._click_to_tag_chk.setChecked(False)
        self._click_to_tag_chk.setToolTip(
            "When on, clicking a vessel directly toggles its "
            "``radial_uncertain`` flag instead of opening the info box.  "
            "Turn off to return to normal inspect-on-click behaviour.")
        self._click_to_tag_chk.toggled.connect(
            self._on_click_to_tag_toggle)
        _ru_lay.addWidget(self._click_to_tag_chk)

        _btn_row = QWidget()
        _btn_lay = QHBoxLayout(_btn_row)
        _btn_lay.setContentsMargins(0, 0, 0, 0)
        self._radial_toggle_btn = QPushButton("Toggle on selected  (u)")
        self._radial_toggle_btn.setToolTip(
            "Flip the ``radial_uncertain`` flag on the currently selected "
            "edge.  Same as pressing 'u' in the napari canvas.")
        self._radial_toggle_btn.clicked.connect(
            self._toggle_radial_uncertain_on_selected)
        _btn_lay.addWidget(self._radial_toggle_btn)
        self._radial_save_btn = QPushButton("Save tags")
        self._radial_save_btn.setToolTip(
            "Write the current radial_uncertain flags back to the graph "
            "file on disk.  No solver rerun.")
        self._radial_save_btn.clicked.connect(self._save_radial_uncertain_tags)
        _btn_lay.addWidget(self._radial_save_btn)
        _ru_lay.addWidget(_btn_row)

        # Region-based tag/untag: pick a shape (Rectangle or Lasso), then
        # press Tag or Untag — every edge midpoint inside the drawn region
        # gets ``radial_uncertain`` flipped in that direction.  ONE scratch
        # Shapes layer is reused across sessions so we never hit the
        # napari layer-lifecycle race that made the old "Rectangle tag…"
        # button stop working after ~two uses.
        _rg_row = QWidget()
        _rg_lay = QHBoxLayout(_rg_row)
        _rg_lay.setContentsMargins(0, 0, 0, 0)
        _rg_lay.addWidget(QLabel("Region:"))
        self._region_shape_combo = QComboBox()
        self._region_shape_combo.addItem("Rectangle", userData="add_rectangle")
        self._region_shape_combo.addItem("Ellipse", userData="add_ellipse")
        self._region_shape_combo.addItem("Lasso", userData="add_polygon_lasso")
        self._region_shape_combo.addItem("Polygon", userData="add_polygon")
        self._region_shape_combo.setToolTip(
            "Rectangle / Ellipse: drag.  Lasso: click, trace, click to "
            "close.  Polygon: click corners, double-click to close.")
        _rg_lay.addWidget(self._region_shape_combo)
        self._region_tag_btn = QPushButton("Tag")
        self._region_tag_btn.setToolTip(
            "Draw the selected region on the canvas — every edge whose "
            "midpoint falls inside gets ``radial_uncertain=True`` on release.")
        self._region_tag_btn.clicked.connect(
            lambda: self._start_region_action("tag"))
        _rg_lay.addWidget(self._region_tag_btn)
        self._region_untag_btn = QPushButton("Untag")
        self._region_untag_btn.setToolTip(
            "Draw a region — every edge midpoint inside gets its "
            "``radial_uncertain`` flag CLEARED on release.")
        self._region_untag_btn.clicked.connect(
            lambda: self._start_region_action("untag"))
        _rg_lay.addWidget(self._region_untag_btn)
        self._region_stop_btn = QPushButton("Stop")
        self._region_stop_btn.setToolTip(
            "Leave region drawing and return to normal click behaviour.")
        self._region_stop_btn.clicked.connect(self._reset_region_layer)
        _rg_lay.addWidget(self._region_stop_btn)
        _rg_lay.addStretch(1)
        _ru_lay.addWidget(_rg_row)

        self._radial_status_lbl = QLabel("")
        self._radial_status_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _ru_lay.addWidget(self._radial_status_lbl)

        fld_lay.addWidget(_ru_row)
        # Initial status update.
        self._refresh_radial_status()

        # ── Flow-direction mismatch highlight ───────────────────────
        # Edges measured in ≥2 tiles whose flow DIRECTION (flow_from→
        # flow_to) disagrees between tiles, counting only measurements
        # above the SNR threshold so noise-floor sign flips (where the
        # direction is unidentifiable) don't register.  Read-only —
        # computed from measurements, nothing written to the graph.
        self._show_flow_mismatch = getattr(self, '_show_flow_mismatch', False)
        _fm_row = QWidget()
        _fm_lay = QVBoxLayout(_fm_row)
        _fm_lay.setContentsMargins(0, 4, 0, 0)
        self._flow_mismatch_chk = QCheckBox(
            "Highlight flow-direction mismatches (magenta)")
        self._flow_mismatch_chk.setChecked(self._show_flow_mismatch)
        self._flow_mismatch_chk.setToolTip(
            "Edges measured in ≥2 tiles whose flow direction "
            "(flow_from→flow_to) disagrees between tiles, counting only "
            "measurements with SNR_H1 ≥ the threshold below.  Renders in "
            "magenta regardless of the current colormap.")
        self._flow_mismatch_chk.toggled.connect(self._on_flow_mismatch_toggle)
        _fm_lay.addWidget(self._flow_mismatch_chk)

        _fm_snr_row = QWidget()
        _fm_snr_lay = QHBoxLayout(_fm_snr_row)
        _fm_snr_lay.setContentsMargins(0, 0, 0, 0)
        _fm_snr_lay.addWidget(QLabel("min SNR_H1 (dB):"))
        self._flow_mismatch_snr_spin = QDoubleSpinBox()
        self._flow_mismatch_snr_spin.setRange(-20.0, 40.0)
        self._flow_mismatch_snr_spin.setDecimals(1)
        self._flow_mismatch_snr_spin.setSingleStep(1.0)
        self._flow_mismatch_snr_spin.setValue(3.0)
        self._flow_mismatch_snr_spin.setToolTip(
            "Only measurements with snr_harm_fit_db ≥ this value count "
            "toward a mismatch.  3 dB ≈ 'reliable'; 6 dB ≈ 'strong'.")
        self._flow_mismatch_snr_spin.valueChanged.connect(
            self._on_flow_mismatch_snr_change)
        _fm_snr_lay.addWidget(self._flow_mismatch_snr_spin)
        _fm_snr_lay.addStretch(1)
        _fm_lay.addWidget(_fm_snr_row)

        self._flow_mismatch_status_lbl = QLabel("")
        self._flow_mismatch_status_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _fm_lay.addWidget(self._flow_mismatch_status_lbl)
        fld_lay.addWidget(_fm_row)
        self._compute_flow_mismatch_edges()   # initial status + cache

        # Colour-bound controls — two modes selectable by the combo:
        #   'Percentile' (default): clip to the [lo, hi] percentiles of
        #       the finite per-edge distribution.  Loose clip = 2/98.
        #   'Manual':    use the user-typed [vmin, vmax] literally.
        # Phase fields ignore both and stay fixed at [-pi, pi]; diverging
        # fields under Percentile become symmetric, under Manual respect
        # the user's exact limits.
        bounds_row = QWidget()
        b_lay = QHBoxLayout(bounds_row); b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.addWidget(QLabel('bounds:'))
        self._bounds_mode_combo = QComboBox()
        self._bounds_mode_combo.addItems(['Percentile', 'Manual'])
        self._bounds_mode_combo.setToolTip(
            "Percentile: colormap clipped to [lo pct, hi pct] of the\n"
            "  finite per-edge distribution (auto-adaptive).\n"
            "Manual: use the exact vmin / vmax typed to the right.\n"
            "Phase fields always stay at [-pi, pi] regardless.")
        self._bounds_mode_combo.currentIndexChanged.connect(
            self._on_bounds_mode_change)
        b_lay.addWidget(self._bounds_mode_combo)
        # Percentile spinboxes (visible when mode == 'Percentile')
        self._pct_lo_lbl = QLabel(' lo pct:')
        b_lay.addWidget(self._pct_lo_lbl)
        self._pct_lo_spin = QDoubleSpinBox()
        self._pct_lo_spin.setRange(0.0, 100.0)
        self._pct_lo_spin.setDecimals(1)
        self._pct_lo_spin.setSingleStep(0.5)
        self._pct_lo_spin.setValue(2.0)
        self._pct_lo_spin.setToolTip(
            "Lower percentile of finite per-edge values, mapped to the\n"
            "low end of the colormap.  Values below get clipped.")
        self._pct_lo_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_lo_spin)
        self._pct_hi_lbl = QLabel(' hi pct:')
        b_lay.addWidget(self._pct_hi_lbl)
        self._pct_hi_spin = QDoubleSpinBox()
        self._pct_hi_spin.setRange(0.0, 100.0)
        self._pct_hi_spin.setDecimals(1)
        self._pct_hi_spin.setSingleStep(0.5)
        self._pct_hi_spin.setValue(98.0)
        self._pct_hi_spin.setToolTip(
            "Upper percentile of finite per-edge values, mapped to the\n"
            "high end of the colormap.  Values above get clipped.")
        self._pct_hi_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_hi_spin)
        # Manual vmin / vmax spinboxes (visible when mode == 'Manual')
        self._vmin_lbl = QLabel(' vmin:')
        b_lay.addWidget(self._vmin_lbl)
        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e12, 1e12)
        self._vmin_spin.setDecimals(4)
        self._vmin_spin.setSingleStep(0.1)
        self._vmin_spin.setValue(0.0)
        self._vmin_spin.setToolTip(
            "Manual lower colormap bound (in the field's own units).\n"
            "Under 'Manual' mode this is used exactly, including for\n"
            "diverging fields (limits are NOT auto-symmetrised).")
        self._vmin_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._vmin_spin)
        self._vmax_lbl = QLabel(' vmax:')
        b_lay.addWidget(self._vmax_lbl)
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e12, 1e12)
        self._vmax_spin.setDecimals(4)
        self._vmax_spin.setSingleStep(0.1)
        self._vmax_spin.setValue(1.0)
        self._vmax_spin.setToolTip("Manual upper colormap bound.")
        self._vmax_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._vmax_spin)
        b_lay.addStretch(1)
        fld_lay.addWidget(bounds_row)
        # Start on Percentile mode: hide the manual widgets inline
        # (can't call _on_bounds_mode_change here — the cbar overlay
        # and edge/node layers don't exist yet).
        for _w in (self._vmin_lbl, self._vmin_spin,
                    self._vmax_lbl, self._vmax_spin):
            _w.setVisible(False)

        # Phase-only offset — add a constant to every displayed phase
        # value (wraps into [-180, 180] deg) so you can visually align
        # simulated vs normalized/measured phases.  Ignored for non-
        # phase fields.
        phase_row = QWidget()
        pr_lay = QHBoxLayout(phase_row); pr_lay.setContentsMargins(0, 0, 0, 0)
        pr_lay.addWidget(QLabel('Phase offset (deg):'))
        self._phase_offset_spin = QDoubleSpinBox()
        self._phase_offset_spin.setRange(-360.0, 360.0)
        self._phase_offset_spin.setDecimals(1)
        self._phase_offset_spin.setSingleStep(5.0)
        self._phase_offset_spin.setValue(0.0)
        self._phase_offset_spin.setToolTip(
            "Add a constant offset (in degrees) to every displayed\n"
            "phase value before colour-mapping.  Wraps into [-180, 180].\n"
            "Only active when Property='phase'; ignored for other fields.\n"
            "Use to visually align simulated vs measured/normalized phases.")
        self._phase_offset_spin.valueChanged.connect(self._on_bounds_change)
        pr_lay.addWidget(self._phase_offset_spin)
        _reset_btn = QPushButton('reset')
        _reset_btn.setFixedWidth(60)
        _reset_btn.setToolTip('Reset the phase offset to 0.')
        _reset_btn.clicked.connect(lambda: self._phase_offset_spin.setValue(0.0))
        pr_lay.addWidget(_reset_btn)
        self._phase_zero_boundary_chk = QCheckBox('Zero at boundary mean')
        self._phase_zero_boundary_chk.setChecked(True)
        self._phase_zero_boundary_chk.setToolTip(
            "Subtract the mean H_k phase of the four boundary-arm edges\n"
            "from every displayed phase value.  Applies to whatever\n"
            "field is currently shown (sim, normalized, measured), so\n"
            "phase maps from different sources become directly\n"
            "comparable — 0° at the boundaries by construction.\n"
            "Additive with the Phase offset (deg) spinbox above.")
        self._phase_zero_boundary_chk.toggled.connect(self._on_bounds_change)
        pr_lay.addWidget(self._phase_zero_boundary_chk)
        pr_lay.addStretch(1)
        fld_lay.addWidget(phase_row)

        # Colour bar toggle — overlays a floating colorbar on top of the
        # napari canvas (right side, vertically centred).  No embedded
        # canvas in the side panel.
        self._cbar_chk = QCheckBox("Show colour bar (overlaid on canvas)")
        self._cbar_chk.setChecked(True)
        self._cbar_chk.toggled.connect(self._on_cbar_toggle)
        fld_lay.addWidget(self._cbar_chk)
        # The overlay QLabel lives in `self._cbar_overlay`; created /
        # replaced by `_refresh_cbar()`.
        self._cbar_overlay = None
        self._cbar_orig_resize = None
        # Tile filter — restricts measured-Q fields to one tile's
        # measurement instead of best-across-tiles.  Only meaningful
        # when tile positions are loaded; hidden otherwise.
        self._tile_filter_combo = None
        if self.tiles:
            tilef_row = QWidget()
            tilef_lay = QHBoxLayout(tilef_row)
            tilef_lay.setContentsMargins(0, 0, 0, 0)
            tilef_lay.addWidget(QLabel("Tile filter:"))
            self._tile_filter_combo = QComboBox()
            self._tile_filter_combo.addItem(
                "All tiles (best measurement)", userData=None)
            for tid in sorted(self.tiles.keys()):
                self._tile_filter_combo.addItem(
                    f"tile {tid}", userData=int(tid))
            self._tile_filter_combo.setToolTip(
                "When set to a specific tile, every measured-Q field\n"
                "shows that tile's measurement only (not the best\n"
                "across tiles).  Edges with no measurement on the\n"
                "selected tile render as 'no data' grey.\n"
                "Doesn't affect Sim fields or geometry.")
            self._tile_filter_combo.currentIndexChanged.connect(
                self._on_tile_filter_change)
            tilef_lay.addWidget(self._tile_filter_combo)
            tilef_lay.addStretch(1)
            fld_lay.addWidget(tilef_row)
        layout.addWidget(fld_group)

        # Percentile filter on total_snr — gates measured-Q fields by
        # network-wide percentile of Var(fit)/Var(resid).  Percentile is
        # computed against the GRAPH-WIDE cached values (best-of-edge),
        # so the threshold is stable regardless of whether a tile
        # filter is on.  0 = no filtering, default.
        from qtpy.QtWidgets import QDoubleSpinBox
        tspct_row = QWidget()
        tspct_lay = QHBoxLayout(tspct_row)
        tspct_lay.setContentsMargins(0, 0, 0, 0)
        tspct_lay.addWidget(QLabel('Min SNR_H1 (dB):'))
        self._tspct_filter_spin = QDoubleSpinBox()
        self._tspct_filter_spin.setRange(-20.0, 60.0)
        self._tspct_filter_spin.setDecimals(1)
        self._tspct_filter_spin.setSingleStep(1.0)
        self._tspct_filter_spin.setValue(-20.0)
        self._tspct_filter_spin.setToolTip(
            "Hide edges whose SNR_H1 (dB) — the canonical per-edge\n"
            "cardiac-fundamental SNR (Q_H1_snr_db) — is below this\n"
            "absolute threshold.  Only affects measured-Q.  Set to the\n"
            "spin's minimum (−20 dB) to disable filtering.")
        self._tspct_filter_spin.valueChanged.connect(
            self._on_total_snr_pct_change)
        tspct_lay.addWidget(self._tspct_filter_spin)
        tspct_lay.addStretch(1)
        fld_lay.addWidget(tspct_row)

        # H2 fundamental SNR percentile filter — same shape as H1 above.
        tspct_h2_row = QWidget()
        tspct_h2_lay = QHBoxLayout(tspct_h2_row)
        tspct_h2_lay.setContentsMargins(0, 0, 0, 0)
        tspct_h2_lay.addWidget(QLabel('Min SNR_H2 (dB):'))
        self._tspct_h2_filter_spin = QDoubleSpinBox()
        self._tspct_h2_filter_spin.setRange(-20.0, 60.0)
        self._tspct_h2_filter_spin.setDecimals(1)
        self._tspct_h2_filter_spin.setSingleStep(1.0)
        self._tspct_h2_filter_spin.setValue(-20.0)
        self._tspct_h2_filter_spin.setToolTip(
            "Hide edges whose SNR_H2 (dB) — the per-edge second-harmonic\n"
            "SNR (Q_H2_snr_db) — is below this absolute threshold.\n"
            "Combines with the H1 gate above via AND.  Only affects\n"
            "measured-Q.  Set to −20 dB to disable.")
        self._tspct_h2_filter_spin.valueChanged.connect(
            self._on_total_snr_h2_pct_change)
        tspct_h2_lay.addWidget(self._tspct_h2_filter_spin)
        tspct_h2_lay.addStretch(1)
        fld_lay.addWidget(tspct_h2_row)

        # Percentile filter on viscous dissipation Φ — gates measured-Q
        # fields by network-wide percentile of `_meas_dissipation` (W).
        # Tooltip shows the actual distribution so the user knows what
        # cut they're applying in absolute terms.  0 = no filtering.
        diss_row = QWidget()
        diss_lay = QHBoxLayout(diss_row)
        diss_lay.setContentsMargins(0, 0, 0, 0)
        diss_lay.addWidget(QLabel('Hide bottom % by dissipation Φ:'))
        self._dissipation_filter_spin = QDoubleSpinBox()
        self._dissipation_filter_spin.setRange(0.0, 99.0)
        self._dissipation_filter_spin.setDecimals(1)
        self._dissipation_filter_spin.setSingleStep(1.0)
        self._dissipation_filter_spin.setValue(0.0)
        # Compute the network-wide dissipation distribution to populate
        # the tooltip range.  Updated once at startup; not re-derived
        # because the field is precomputed and doesn't change post-load.
        _diss_vals = [
            float(dd.get('_meas_dissipation'))
            for _, _, dd in self.G.edges(data=True)
            if dd.get('_meas_dissipation') is not None
                and np.isfinite(dd.get('_meas_dissipation'))
                and float(dd.get('_meas_dissipation')) > 0
        ]
        if _diss_vals:
            arr = np.asarray(_diss_vals)
            self._cached_dissipation_stats = dict(
                n=int(arr.size), min=float(arr.min()),
                p10=float(np.percentile(arr, 10)),
                p50=float(np.percentile(arr, 50)),
                p90=float(np.percentile(arr, 90)),
                max=float(arr.max()),
            )
            _s = self._cached_dissipation_stats
            tooltip = (
                "Hide edges in the bottom X % of viscous dissipation Φ\n"
                "(W; computed as Φ = r·L·⟨Q²⟩ where r = 8μ/(πR⁴) and\n"
                "⟨Q²⟩ = mean(Q_t²) — see methods.md).  Threshold is\n"
                "computed from the GRAPH-WIDE distribution of measured\n"
                "values, so it's stable regardless of Tile filter.\n"
                "Only affects measured-Q fields.  0 = no filtering.\n\n"
                f"Network-wide Φ distribution ({_s['n']} edges):\n"
                f"  min  : {_s['min']:.2e} W\n"
                f"  10%  : {_s['p10']:.2e} W\n"
                f"  median: {_s['p50']:.2e} W\n"
                f"  90%  : {_s['p90']:.2e} W\n"
                f"  max  : {_s['max']:.2e} W")
        else:
            tooltip = ("Hide bottom X % by dissipation Φ.\n"
                       "(No edges have computed dissipation yet.)")
        self._dissipation_filter_spin.setToolTip(tooltip)
        self._dissipation_filter_spin.valueChanged.connect(
            self._on_dissipation_pct_change)
        diss_lay.addWidget(self._dissipation_filter_spin)
        diss_lay.addStretch(1)
        fld_lay.addWidget(diss_row)

        # Tile browser — load any tile's video by picking from a dropdown,
        # independent of edge selection.  Only shown if positions + video
        # paths are configured.
        if bool(self.tiles) and self.video_dir is not None:
            tile_group = QGroupBox("Browse tiles")
            tile_lay = QVBoxLayout(tile_group)
            self._tile_combo = QComboBox()
            # Tile entries: each item carries the int tile ID as
            # userData so the handler can use currentData() directly
            # instead of parsing the label text.  Tiles whose video
            # file isn't present in `video_dir` (typical for
            # partial-coverage bundles like the intern packages) get
            # " (no video)" appended and are dim-disabled.
            self._tile_combo.addItem("(pick a tile)", userData=None)
            n_missing = 0
            for tid in sorted(self.tiles.keys()):
                vid_path = self._get_video_path_for(int(tid))
                label = (f"{tid}"
                          if vid_path is not None
                          else f"{tid}  (no video)")
                self._tile_combo.addItem(label, userData=int(tid))
                if vid_path is None:
                    n_missing += 1
                    idx = self._tile_combo.count() - 1
                    item = self._tile_combo.model().item(idx)
                    if item is not None:
                        from qtpy.QtCore import Qt as _Qt
                        item.setFlags(item.flags()
                                       & ~_Qt.ItemIsEnabled
                                       & ~_Qt.ItemIsSelectable)
            tip = ("Pick a tile ID then click 'Load this tile' to "
                    "overlay\nits raw 250-fps video on the mosaic.")
            if n_missing:
                tip += (f"\n\n{n_missing} tile(s) are greyed out "
                         f"because their video files\nare not present "
                         f"in this bundle.")
            self._tile_combo.setToolTip(tip)
            tile_lay.addWidget(self._tile_combo)
            self._tile_load_btn = QPushButton('Load this tile video')
            self._tile_load_btn.clicked.connect(self._load_video_from_dropdown)
            tile_lay.addWidget(self._tile_load_btn)
            layout.addWidget(tile_group)

        # Display toggles
        d_group = QGroupBox("Display")
        d_lay = QVBoxLayout(d_group)
        self._nodes_chk = QCheckBox("Show nodes")
        self._nodes_chk.toggled.connect(self._on_nodes_toggle)
        d_lay.addWidget(self._nodes_chk)
        if bool(self.tiles):
            self._tile_boundaries_chk = QCheckBox("Show tile boundaries")
            self._tile_boundaries_chk.toggled.connect(
                self._on_tile_boundaries_toggle)
            d_lay.addWidget(self._tile_boundaries_chk)
            self._tile_labels_chk = QCheckBox("Show tile numbers")
            self._tile_labels_chk.toggled.connect(
                self._on_tile_labels_toggle)
            d_lay.addWidget(self._tile_labels_chk)
        # Edge value labels (only render when Tile filter selects a
        # single tile — at All-tiles scale the labels overlap badly).
        self._edge_labels_chk = QCheckBox("Show edge value labels (single tile)")
        self._edge_labels_chk.setToolTip(
            "Show the current Property value at each edge's midpoint.\n"
            "Only renders when Tile filter is set to a specific tile —\n"
            "in All-tiles mode this would clutter the mosaic.\n"
            "Labels update when you change Property / Harmonic / tile.")
        self._edge_labels_chk.toggled.connect(self._on_edge_labels_toggle)
        d_lay.addWidget(self._edge_labels_chk)
        # Kirchhoff KCL residuals: per-node signed sum of incident Q
        # for the selected tile's measurements.  Internal nodes should
        # net to ~0 (mass conservation); large residuals flag measurement
        # / geometry inconsistencies.
        self._kcl_chk = QCheckBox("Show Kirchhoff residuals (tile mode)")
        self._kcl_chk.setToolTip(
            "Per-node signed sum of incident Q on this tile's measurements.\n"
            "For interior nodes, KCL says the sum should be ≈ 0 (mass\n"
            "conservation).  Non-zero residuals at interior nodes are\n"
            "the joint measurement-noise + geometry-misfit + topology\n"
            "error.  Boundary nodes carry the net flow into/out of the\n"
            "carve at the boundary, expected to be non-zero.\n"
            "Only renders when Tile filter selects a specific tile.\n"
            "Diverging colourmap: red = net outflow, blue = net inflow.")
        self._kcl_chk.toggled.connect(self._on_kcl_toggle)
        d_lay.addWidget(self._kcl_chk)
        layout.addWidget(d_group)

        layout.addStretch()

        # ══════ Tab 2: TIME SERIES ════════════════════════════════════
        layout = ts_layout
        # Edge inspection panel (now lives in the time-series tab).
        inspect_group = QGroupBox("Selected edge")
        inspect_lay = QVBoxLayout(inspect_group)
        self._info_box = QTextEdit()
        self._info_box.setReadOnly(True)
        self._info_box.setMaximumHeight(220)
        self._info_box.setPlaceholderText(
            "Click an edge in the viewer to inspect it.")
        inspect_lay.addWidget(self._info_box)
        # ── Go to edge (u, v) — jump-to-vessel by node ids ──────────────
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Go to (u,v):"))
        self._goto_edge_input = QLineEdit()
        self._goto_edge_input.setPlaceholderText(
            "e.g. 89617, 90142  or  (89617, 90142)")
        self._goto_edge_input.returnPressed.connect(self._goto_edge_from_input)
        goto_row.addWidget(self._goto_edge_input, 1)
        self._goto_edge_btn = QPushButton("Go")
        self._goto_edge_btn.setToolTip(
            "Select the edge with the given node ids (u,v).  Reuses the "
            "normal click-selection path — populates info box, highlights "
            "the vessel, centers the viewport, loads video if auto-load is "
            "on.  Then click 'Compute fresh PIV/OF' + 'Show OF pipeline "
            "walkthrough' to see the diagnostic figure.")
        self._goto_edge_btn.clicked.connect(self._goto_edge_from_input)
        goto_row.addWidget(self._goto_edge_btn)
        inspect_lay.addLayout(goto_row)
        self._plot_qt_btn = QPushButton("Plot Q(t) for selected")
        self._plot_qt_btn.setEnabled(False)
        self._plot_qt_btn.clicked.connect(self._plot_selected_qt)
        inspect_lay.addWidget(self._plot_qt_btn)
        # ── Overlay: compare current source with a second source ────────
        _cmp_row = QHBoxLayout()
        _cmp_row.addWidget(QLabel("Compare current source with:"))
        from qtpy.QtWidgets import QComboBox
        self._overlay_source_combo = QComboBox()
        self._overlay_source_combo.addItem("(none)", None)
        self._overlay_source_combo.addItem("measured", "measured")
        self._overlay_source_combo.addItem("normalized", "normalized")
        self._overlay_source_combo.addItem("sim (intact)", "sim")
        self._overlay_source_combo.addItem("sim (carved)", "sim_carved")
        self._overlay_source_combo.setToolTip(
            "If set (≠ (none)), 'Plot Q(t)' overlays this second source's "
            "reconstructed Q(t) on the same axes as the primary source "
            "(the one selected by the View-tab Source radios).\n"
            "Uses stored harmonic phasors: DC + H1/H2/H3.")
        _cmp_row.addWidget(self._overlay_source_combo)
        _cmp_row.addStretch(1)
        self._plot_overlay_btn = QPushButton("Plot overlay")
        self._plot_overlay_btn.setEnabled(False)
        self._plot_overlay_btn.setToolTip(
            "Plot Q(t) for BOTH the current source (View-tab Source radio) "
            "and the 'Compare with' source above, overlaid on one axis.\n"
            "Both waveforms are reconstructed from stored harmonics (DC+H1..H3).")
        self._plot_overlay_btn.clicked.connect(self._plot_selected_qt_overlay)
        _cmp_row.addWidget(self._plot_overlay_btn)
        inspect_lay.addLayout(_cmp_row)
        self._plot_bnd_kcl_btn = QPushButton(
            "Plot KCL sum at boundary (selected edge's boundary node)")
        self._plot_bnd_kcl_btn.setEnabled(False)
        self._plot_bnd_kcl_btn.setToolTip(
            "For a selected boundary EDGE (arm from V/A tip to first "
            "junction J): pop a figure showing the KCL sum of J's OTHER "
            "incident vessels (tile 14 preferred), which by KCL equals "
            "the flow through the arm.  This is the 'true' boundary Q(t) "
            "your colleague should use as a BC — not the noisy arm PIV.\n\n"
            "Enabled only when the selected edge touches a boundary node.")
        self._plot_bnd_kcl_btn.clicked.connect(self._on_plot_boundary_kcl)
        inspect_lay.addWidget(self._plot_bnd_kcl_btn)
        # Raw vs normalized comparison — 4-panel figure per edge.
        # Enabled once the graph has been normalized (via the Normalize
        # tab, or loaded with normalization already applied).  Note the
        # main "Plot Q(t)" button auto-invokes this same handler when
        # the active data source is 'Normalized'.
        self._compare_norm_btn = QPushButton(
            "Diagnose normalization (raw + norm + phasors)")
        self._compare_norm_btn.setToolTip(
            "Pop a 6-panel diagnostic for the selected edge showing how "
            "the per-tile normalization gauge (CO_rel, z_arg_H1, f0/f0_can) "
            "brings the tiles into a common frame.\n\n"
            "Panels:\n"
            " (top-L) RAW Q(t) per tile, native f0, harmonic fit dashed\n"
            " (top-M) NORMALIZED Q(t) per tile, canonical f0, plus the\n"
            "         SNR-weighted mean phasor waveform (purple bold)\n"
            "         and a CONSISTENCY check from _h_amp/phase_H{k}_norm\n"
            " (top-R) |amplitude| bars per harmonic  (raw vs normalized)\n"
            " (bot-L) Per-tile gauges: |z_t|, CO_rel, f0/f0_can with ψ\n"
            " (bot-M) Cycle-folded overlay — one canonical cycle per tile\n"
            " (bot-R) Normalized phasors in complex plane, per harmonic\n\n"
            "If normalization worked, the norm Q(t) traces should overlay\n"
            "tightly and the phasor markers should cluster on a single\n"
            "point per harmonic (= no residual per-tile drift).")
        self._compare_norm_btn.setEnabled(False)
        self._compare_norm_btn.clicked.connect(self._plot_raw_vs_norm_qt)
        inspect_lay.addWidget(self._compare_norm_btn)
        # Run Farneback optical flow/PIV on the selected edge from the
        # tile video.  Read-only: no graph mutation, no measurement
        # persistence — just shows Q(t), spectrum, and a curved ROI
        # overlay.  The implementation detects spatially-downsampled
        # demo videos and adjusts the effective pixel size accordingly.
        self._run_of_btn = QPushButton("Compute fresh PIV/OF for selected")
        self._run_of_btn.setEnabled(False)
        self._run_of_btn.setToolTip(
            "Reload the tile video, run Farneback OF/PIV on the\n"
            "selected vessel, and pop up Q(t) + spectrum.  No data\n"
            "is written back to the graph.  Requires --tile-positions\n"
            "and --video-dir; ~5-10s per call.")
        self._run_of_btn.clicked.connect(self._on_run_optical_flow_clicked)
        inspect_lay.addWidget(self._run_of_btn)
        # Merge the whole degree-2 vessel chain before running OF (default on).
        # Degree-2 nodes are pass-through points where KCL is trivial, so the
        # per-segment subdivision only costs signal; merging runs OF on the
        # full vessel.  Toggle off to compare against the single clicked segment.
        from qtpy.QtWidgets import QCheckBox as _QCheckBox
        self._of_merge_chk = _QCheckBox("Merge degree-2 vessel chain (whole vessel)")
        self._of_merge_chk.setChecked(True)
        self._of_merge_chk.setToolTip(
            "Run OF on the whole vessel (all degree-2 segments merged) instead\n"
            "of just the clicked segment.  More cross-sections, higher SNR, one\n"
            "consistent flow direction.  Off = single-segment (legacy) behaviour.")
        inspect_lay.addWidget(self._of_merge_chk)
        # Toggle OF-driven centerline refinement.  When OFF, the raw graph
        # centerline is used as-is — useful when refinement mis-tracks (e.g.
        # pushes the CL off-axis on curved / bimodal vessels, giving a
        # non-parabolic profile).  Default ON (with the do-no-harm revert guard).
        self._of_refine_chk = _QCheckBox("Refine centerline (OF ridge-snap)")
        self._of_refine_chk.setChecked(True)
        self._of_refine_chk.setToolTip(
            "Snap the centerline onto the OF velocity ridge before analysis.\n"
            "OFF = use the raw graph centerline as-is.  Turn OFF if the refined\n"
            "CL drifts off the vessel (bimodal / non-parabolic radial profile).")
        inspect_lay.addWidget(self._of_refine_chk)
        # Clear the curved OF-region overlay (created by Run OF; persists
        # across mosaic interactions until cleared).  Disabled when no
        # overlay is active.
        self._clear_of_region_btn = QPushButton("Clear OF region overlay")
        self._clear_of_region_btn.setEnabled(False)
        self._clear_of_region_btn.setToolTip(
            "Remove the curved orange ribbon that highlights the last\n"
            "OF integration region on the mosaic.")
        self._clear_of_region_btn.clicked.connect(
            self._tear_down_of_region_overlay)
        inspect_lay.addWidget(self._clear_of_region_btn)
        # Comprehensive stage-by-stage OF pipeline walkthrough for the last
        # fresh OF run.  ('Compute fresh PIV/OF' already pops it automatically;
        # this button re-opens it.)  Folds in what used to be three separate
        # popups — the 4-panel diagnostic, the Kloosterman per-slice figure,
        # and the vessel-quality verdict — into one single figure driven by
        # one data path.
        self._of_pipeline_btn = QPushButton("Show OF pipeline walkthrough")
        self._of_pipeline_btn.setEnabled(False)
        self._of_pipeline_btn.setToolTip(
            "Re-open the full stage-by-stage OF pipeline figure for the last "
            "fresh OF run: ROI selection, background subtraction, mean flow "
            "field, centerline refinement (coarse shift → per-slice snap → "
            "offset smoothing), vessel mask + tangent, radial-profile binning, "
            "the three radii + how each is chosen, the Kloosterman per-slice "
            "gate with reject reasons, Q-estimator comparison, and Q(t) — plus "
            "an overall PASS / CROP / RAILING verdict.")
        self._of_pipeline_btn.clicked.connect(self._show_of_pipeline_figures)
        inspect_lay.addWidget(self._of_pipeline_btn)

        # Independent OpenPIV estimate on the same edge, for OF-vs-PIV comparison.
        self._piv_compare_btn = QPushButton("Compute PIV vs OF (OpenPIV)")
        self._piv_compare_btn.setEnabled(False)
        self._piv_compare_btn.setToolTip(
            "After 'Compute fresh PIV/OF', run OpenPIV ensemble (16-px windows, "
            "50% overlap, phase-sorted correlation averaging) on the SAME frames + "
            "centerline + tangent + R.  dt is auto-tuned so peak displacement ≈ "
            "10 px (Poelma optimum).  Pops a comparison figure — Q(t) overlay, "
            "profile shapes, sig2noise field, harmonic summary.")
        self._piv_compare_btn.clicked.connect(self._on_run_piv_comparison_clicked)
        inspect_lay.addWidget(self._piv_compare_btn)

        # Winsize sweep — runs PIV at 4 window sizes on the same edge and
        # plots recovered Q vs winsize.  Answers "should we use smaller
        # windows on this vessel?" empirically for the current data.
        self._piv_sweep_btn = QPushButton("PIV winsize sweep (8, 12, 16, 24)")
        self._piv_sweep_btn.setEnabled(False)
        self._piv_sweep_btn.setToolTip(
            "After 'Compute PIV vs OF', run OpenPIV at winsize ∈ {8, 12, "
            "16, 24} on the same edge and plot Q vs winsize.  Small windows "
            "have better spatial resolution but lower correlation SNR; "
            "large windows the reverse.  This sweep shows the tradeoff "
            "empirically for THIS vessel's speckle.  ~40 s.")
        self._piv_sweep_btn.clicked.connect(self._on_run_piv_sweep_clicked)
        inspect_lay.addWidget(self._piv_sweep_btn)

        # Comparison set — accumulate edges across multiple clicks and
        # overlay their Q(t)s on a single figure.  Useful for comparing
        # nearby vessels' waveforms.
        self._add_to_comp_btn = QPushButton("Add to comparison set")
        self._add_to_comp_btn.setEnabled(False)
        self._add_to_comp_btn.setToolTip(
            "Add the currently-selected edge to a comparison set.\n"
            "Use 'Plot comparison' to see all of them overlaid.")
        self._add_to_comp_btn.clicked.connect(self._add_to_comparison)
        inspect_lay.addWidget(self._add_to_comp_btn)
        self._plot_comp_btn = QPushButton("Plot comparison (0)")
        self._plot_comp_btn.setEnabled(False)
        self._plot_comp_btn.clicked.connect(self._plot_comparison)
        inspect_lay.addWidget(self._plot_comp_btn)
        self._clear_comp_btn = QPushButton("Clear comparison")
        self._clear_comp_btn.setEnabled(False)
        self._clear_comp_btn.clicked.connect(self._clear_comparison)
        inspect_lay.addWidget(self._clear_comp_btn)
        # Video overlay controls — only shown if positions + video paths
        # were given on the CLI.  Otherwise hide them rather than confuse
        # the user.
        video_enabled = bool(self.tiles) and self.video_dir is not None
        self._auto_video_chk = QCheckBox("Auto-load video on edge click")
        self._auto_video_chk.setEnabled(video_enabled)
        self._auto_video_chk.setToolTip(
            "When ON, clicking an edge automatically loads + overlays\n"
            "the raw video for that tile.  Loading takes ~5-10s per\n"
            "tile; the previous overlay is removed each click.")
        inspect_lay.addWidget(self._auto_video_chk)
        self._video_btn = QPushButton("Load video for selected tile")
        self._video_btn.setEnabled(False)
        self._video_btn.clicked.connect(self._load_video_for_selected)
        inspect_lay.addWidget(self._video_btn)
        self._unload_video_btn = QPushButton("Remove video overlay")
        self._unload_video_btn.setEnabled(False)
        self._unload_video_btn.clicked.connect(
            lambda: (self._remove_video_overlay(),
                     self._unload_video_btn.setEnabled(False)))
        inspect_lay.addWidget(self._unload_video_btn)
        if not video_enabled:
            note = QLabel(
                "<small><i>Video overlay disabled — pass <code>"
                "--tile-positions</code> + <code>--video-dir</code> + "
                "<code>--video-pattern</code> (or <code>--config</code>) "
                "to enable.</i></small>")
            note.setWordWrap(True)
            inspect_lay.addWidget(note)
        layout.addWidget(inspect_group)
        layout.addStretch()

        # ══════ Tab 3: SIMULATE (unified — intact + carved recipes) ═══
        # One tab, source-recipe radio at the top toggles between:
        #   - Intact: 2 sources at DA anchors, α auto-fit on DC
        #   - Carved: N sources at porous outlets, per-zone Σα constrained fit
        # Both share the LOCKED two-pass V recipe and the same D_areal knob.
        from qtpy.QtWidgets import (QRadioButton as _QRB,
                                     QButtonGroup as _QBG,
                                     QGroupBox as _QGB,
                                     QVBoxLayout as _QVL,
                                     QWidget as _QW)
        layout = sim_layout
        _recipe_group = _QGB("Source recipe")
        _recipe_lay = _QVL(_recipe_group)
        self._sim_recipe_intact_radio = _QRB(
            "Intact — 2 sources at DA anchors, α auto-fit on DC")
        self._sim_recipe_carved_radio = _QRB(
            "Carved — N sources at porous outlets "
            "(requires tagged edges in Uncertain tab)")
        self._sim_recipe_intact_radio.setChecked(True)
        _bg = _QBG(_recipe_group)
        _bg.addButton(self._sim_recipe_intact_radio)
        _bg.addButton(self._sim_recipe_carved_radio)
        _recipe_lay.addWidget(self._sim_recipe_intact_radio)
        _recipe_lay.addWidget(self._sim_recipe_carved_radio)
        layout.addWidget(_recipe_group)

        # Intact recipe container — holds the existing simulate-tab widgets.
        self._sim_intact_panel = _QW()
        _intact_container_lay = _QVL(self._sim_intact_panel)
        _intact_container_lay.setContentsMargins(0, 0, 0, 0)
        self._build_simulate_tab(_intact_container_lay)
        _intact_container_lay.addStretch()
        layout.addWidget(self._sim_intact_panel)

        # Carved recipe container — holds the carved-sim widgets.
        self._sim_carved_panel = _QW()
        _carved_container_lay = _QVL(self._sim_carved_panel)
        _carved_container_lay.setContentsMargins(0, 0, 0, 0)
        self._build_carved_sim_tab(_carved_container_lay)
        self._sim_carved_panel.setVisible(False)
        layout.addWidget(self._sim_carved_panel)

        # Gate the carved-recipe radio: only enabled if the graph has
        # tagged edges to define a porous zone.
        _n_tagged = sum(1 for _, _, d in self.G.edges(data=True)
                        if d.get('radial_uncertain'))
        if _n_tagged == 0:
            self._sim_recipe_carved_radio.setEnabled(False)
            self._sim_recipe_carved_radio.setToolTip(
                "No edges tagged `radial_uncertain` — tag uncertain "
                "edges in the Uncertain tab first.")

        def _on_recipe_toggle(_checked=None):
            is_carved = self._sim_recipe_carved_radio.isChecked()
            self._sim_intact_panel.setVisible(not is_carved)
            self._sim_carved_panel.setVisible(is_carved)
        self._sim_recipe_intact_radio.toggled.connect(_on_recipe_toggle)
        self._sim_recipe_carved_radio.toggled.connect(_on_recipe_toggle)
        layout.addStretch()

        # Info label (View tab footer)
        info = QLabel(
            f"<small>{self.G.number_of_edges()} edges, "
            f"{self.G.number_of_nodes()} nodes<br>"
            f"<b>Read-only viewer</b> — no graph edits, no PIV re-runs."
            f"</small>")
        info.setWordWrap(True)
        view_layout.addWidget(info)

        # Assemble the tabbed panel
        tabs.addTab(view_panel, "View")
        tabs.addTab(ts_panel, "Time series")
        # Wrap the Simulate tab in a scroll area so the Run button is always
        # reachable even when optional sections are expanded.
        _sim_scroll = QScrollArea()
        _sim_scroll.setWidgetResizable(True)
        _sim_scroll.setWidget(sim_panel)
        tabs.addTab(_sim_scroll, "Simulate")
        # (The Carved Sim tab is now folded into the Simulate tab as the
        # "Carved" source-recipe option — one tab, radio-toggled.)
        self._build_normalize_tab(norm_layout)
        _norm_scroll = QScrollArea()
        _norm_scroll.setWidgetResizable(True)
        _norm_scroll.setWidget(norm_panel)
        tabs.addTab(_norm_scroll, "Normalize")
        # Edit tab: boundary-node marking (source/sink/edge/clear) + save.
        # The rest of the viewer stays read-only.
        edit_panel = self._build_edit_panel()
        _edit_scroll = QScrollArea()
        _edit_scroll.setWidgetResizable(True)
        _edit_scroll.setWidget(edit_panel)
        tabs.addTab(_edit_scroll, "Edit")
        self._edit_mode = None
        self.viewer.window.add_dock_widget(tabs, name="Controls", area='right')
        self._selected_edge: Optional[Tuple[int, int]] = None
        # Radial-uncertainty tagging: viewer-editable per-edge flag
        # (``radial_uncertain=True``) that overrides the edge colour with
        # RADIAL_UNCERTAIN_COLOR when highlighting is on.  Solver
        # behaviour is unchanged — this is a downstream-analysis marker.
        self._show_radial_uncertain = False   # off by default (opt in via checkbox)
        self._radial_tags_dirty = False       # unsaved changes tracker
        # napari keybinding: ``u`` toggles the flag on the currently
        # selected edge.  Registered here (after viewer + selection state
        # exist) so it survives layer rebuilds.
        try:
            @self.viewer.bind_key('u', overwrite=True)
            def _toggle_radial_uncertain_kb(_v):
                self._toggle_radial_uncertain_on_selected()
        except Exception:
            pass
        # Build the initial colour bar to reflect current field.
        self._refresh_cbar()

    # ── core: refresh edge rendering ───────────────────────────────────
    def _refresh_edges(self):
        # Always draw every edge in the network.  Edges with a usable
        # PIV measurement get coloured by the selected field; edges with
        # no measurement or with measurement-quality flagged garbage
        # (gated, flag-on-delta, fit failed) get a single dim-grey
        # colour so the topology stays visible without pretending we
        # have data we don't.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        # total_snr percentile filter — applies only to measured-Q
        # fields.  Threshold computed ONCE from the network-wide
        # `_h_total_snr` distribution; per-edge gate below.  Re-derive
        # if the percentile changed (handler clears the cache).
        ts_filter_active = (
            float(self.total_snr_pct_filter) > -20.0
            and self.current_source in ('measured', 'normalized')
            and self.current_quantity == 'Q'
        )
        # Dissipation percentile filter — applies only to measured-Q.
        ds_filter_active = (
            float(self.dissipation_pct_filter) > 0.0
            and self.current_source in ('measured', 'normalized')
            and self.current_quantity == 'Q'
        )
        ds_threshold: Optional[float] = None
        if ds_filter_active:
            if self._cached_dissipation_threshold is None:
                _ds_vals = [
                    float(dd.get('_meas_dissipation'))
                    for _, _, dd in self.G.edges(data=True)
                    if dd.get('_meas_dissipation') is not None
                        and np.isfinite(dd.get('_meas_dissipation'))
                        and float(dd.get('_meas_dissipation')) > 0
                ]
                if _ds_vals:
                    self._cached_dissipation_threshold = float(
                        np.percentile(np.asarray(_ds_vals),
                                       self.dissipation_pct_filter))
                else:
                    self._cached_dissipation_threshold = -np.inf
            ds_threshold = self._cached_dissipation_threshold

        ts_threshold: Optional[float] = None
        if ts_filter_active:
            ts_threshold = float(self.total_snr_pct_filter)

        # H2 SNR absolute-dB filter (mirrors H1).
        ts_h2_filter_active = (
            float(self.total_snr_h2_pct_filter) > -20.0
            and self.current_source in ('measured', 'normalized')
            and self.current_quantity == 'Q'
        )
        ts_h2_threshold: Optional[float] = None
        if ts_h2_filter_active:
            ts_h2_threshold = float(self.total_snr_h2_pct_filter)

        def _best_per_tile_snr(edge_d: dict, k_target: int) -> float:
            """Max across measuring tiles of the per-tile harmonic-fit SNR
            (dB) for harmonic k_target.  Returns −inf if none.  Reads
            from measurements_piv because Q_H<k>_snr_db is not always
            precomputed as a top-level edge attr."""
            best = -np.inf
            for m in (edge_d.get('measurements_piv') or []):
                for h in (m.get('harm', {}).get('harmonics') or []):
                    if int(h.get('k', 0)) != k_target:
                        continue
                    s = h.get('snr_db')
                    if s is not None and np.isfinite(s) and float(s) > best:
                        best = float(s)
                    break
            return best

        EDGE_WIDTH_MIN = 0.5
        EDGE_WIDTH_MAX = 22.0
        lines: List[np.ndarray] = []
        vals: List[Optional[float]] = []   # None ⇒ no usable measurement
        tiers: List[Optional[str]] = []
        widths: List[float] = []
        H_mosaic = self.mosaic_height
        for (u, v) in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            # napari shapes layer wants (row, col) = (display_y, x);
            # display_y = mosaic_height − graph_y.  Straight node-to-node line
            # for every edge (robust: all edges render).  The true curved
            # centerline is drawn only for the selected edge (_highlight_edge_path).
            lines.append(np.array([[H_mosaic - y1, x1],
                                    [H_mosaic - y2, x2]]))
            # Edge width ≈ vessel radius (NOT diameter — keeps neighbour
            # edges from overlapping at large radii while still being
            # visually proportional).
            r_px = float(d.get('radius') or 1.0)
            widths.append(float(np.clip(r_px,
                                         EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)))
            usable = _measurement_usable(m)
            if is_categorical:
                # Two categorical fields, different sources:
                #   quality_tier   — per-measurement (tier letter)
                #   harmonic_class — top-level edge attr (int 0-3)
                # Apply the same total_snr percentile gate so the
                # filter is consistent across continuous + categorical
                # views of measured-Q data.
                if (ts_filter_active and ts_threshold is not None):
                    edge_tsnr = _best_per_tile_snr(d, 1)
                    gate_passes = (np.isfinite(edge_tsnr)
                                    and edge_tsnr >= ts_threshold)
                else:
                    gate_passes = True
                # H2 SNR gate combines with H1/dissipation via AND.
                if (gate_passes and ts_h2_filter_active
                        and ts_h2_threshold is not None):
                    edge_h2 = _best_per_tile_snr(d, 2)
                    if (not np.isfinite(edge_h2)
                            or edge_h2 < ts_h2_threshold):
                        gate_passes = False
                # Dissipation gate combines with SNR gate via AND.
                if (gate_passes and ds_filter_active
                        and ds_threshold is not None):
                    edge_diss = d.get('_meas_dissipation')
                    if (edge_diss is None
                            or not np.isfinite(edge_diss)
                            or float(edge_diss) < ds_threshold):
                        gate_passes = False
                if not gate_passes:
                    tiers.append(None)
                elif self.current_property == 'harmonic_class':
                    tiers.append(d.get('harmonic_class'))
                elif 'quality_tier' in field:
                    tiers.append(m.get('quality_tier') if usable else None)
                else:
                    tiers.append(None)
                vals.append(None)
            else:
                # 4-selector resolver — already handles measured/sim
                # routing and harmonic indexing.  Returns None for
                # invalid combos or missing data.
                val = self._resolve_field_value(u, v)
                # total_snr percentile gate.  Always checks the
                # graph-wide cached `_h_total_snr` (best-of-edge), so
                # the filter is stable across tile-filter changes.
                if (ts_filter_active and val is not None
                        and ts_threshold is not None):
                    edge_tsnr = _best_per_tile_snr(d, 1)
                    if (not np.isfinite(edge_tsnr)
                            or edge_tsnr < ts_threshold):
                        val = None
                # H2 SNR gate combines with H1 via AND.
                if (ts_h2_filter_active and val is not None
                        and ts_h2_threshold is not None):
                    edge_h2 = _best_per_tile_snr(d, 2)
                    if (not np.isfinite(edge_h2)
                            or edge_h2 < ts_h2_threshold):
                        val = None
                # Dissipation gate combines with SNR gate via AND.
                if (ds_filter_active and val is not None
                        and ds_threshold is not None):
                    edge_diss = d.get('_meas_dissipation')
                    if (edge_diss is None
                            or not np.isfinite(edge_diss)
                            or float(edge_diss) < ds_threshold):
                        val = None
                vals.append(val if (val is not None
                                    and np.isfinite(val)) else None)
                tiers.append(None)

        # Compute colors.  Edges with `vals[i] is None` (no usable
        # measurement) get NO_DATA_COLOR regardless of the chosen field.
        if is_categorical:
            cat_map = (HARMONIC_CLASS_COLORS
                       if field == 'harmonic_class' else TIER_COLORS)
            colors = [
                cat_map.get(t, NO_DATA_COLOR) if t is not None
                else NO_DATA_COLOR
                for t in tiers]
        else:
            finite_arr = np.array(
                [v for v in vals if v is not None], dtype=float)
            if len(finite_arr) == 0:
                colors = [NO_DATA_COLOR] * len(lines)
            else:
                # Signed / cyclic fields render on a fixed linear scale —
                # log10 is meaningless for them.
                diverging = ('waveform_av' in field) or ('signed' in field)
                if self.log_scale and not diverging and 'phase' not in field:
                    pos = finite_arr[finite_arr > 0]
                    if len(pos) == 0:
                        finite_eff = finite_arr
                    else:
                        finite_eff = np.log10(pos)
                else:
                    finite_eff = finite_arr
                # Bounds: 'Percentile' clips to the [lo, hi] pct of the
                # finite per-edge distribution (auto-adaptive; defaults
                # 2/98).  'Manual' uses the user's exact vmin/vmax.
                bounds_mode = 'Percentile'
                _bm_combo = getattr(self, '_bounds_mode_combo', None)
                if _bm_combo is not None:
                    bounds_mode = _bm_combo.currentText()
                if bounds_mode == 'Manual':
                    vlo = float(getattr(self, '_vmin_spin', None).value()
                                 if getattr(self, '_vmin_spin', None)
                                 else 0.0)
                    vhi = float(getattr(self, '_vmax_spin', None).value()
                                 if getattr(self, '_vmax_spin', None)
                                 else 1.0)
                    if vlo > vhi:
                        vlo, vhi = vhi, vlo
                else:
                    p_lo = float(getattr(self, '_pct_lo_spin', None).value()
                                 if getattr(self, '_pct_lo_spin', None)
                                 else 2.0)
                    p_hi = float(getattr(self, '_pct_hi_spin', None).value()
                                 if getattr(self, '_pct_hi_spin', None)
                                 else 98.0)
                    if p_lo > p_hi:
                        p_lo, p_hi = p_hi, p_lo
                    vlo, vhi = np.percentile(finite_eff, [p_lo, p_hi])
                if vlo == vhi:
                    vhi = vlo + 1e-9
                if 'phase' in field:
                    cmap = plt.get_cmap('hsv')
                    # π-unified phase display (phase_gradient_recipe.md):
                    # phase = 0 (red) at the boundary reference, hue
                    # rotates through yellow / green / cyan / blue /
                    # magenta with increasing lag, back to red at π.
                    # Values are folded onto [0, π] before mapping so a
                    # π-flipped tile-gauge branch renders the same as
                    # its unflipped counterpart — the (u,v)/gauge sign
                    # is treated as gauge-freedom, not physics.
                    _vlo, _vhi = 0.0, float(np.pi)
                    norm = mcolors.Normalize(vmin=_vlo, vmax=_vhi)
                    self._last_vmin, self._last_vmax = _vlo, _vhi
                    # Optional constant offset (degrees -> radians).
                    self._phase_offset_rad = float(np.radians(
                        getattr(self, '_phase_offset_spin', None).value()
                        if getattr(self, '_phase_offset_spin', None)
                        else 0.0))
                    # Auto-zero at boundary mean: subtract the CIRCULAR
                    # mean of the current field over the four boundary-
                    # arm edges so every source (sim / normalized /
                    # measured) shows 0° at the boundaries → directly
                    # comparable phase maps across sources.
                    self._phase_boundary_offset_rad = 0.0
                    if getattr(self, '_phase_zero_boundary_chk', None) is not None \
                            and self._phase_zero_boundary_chk.isChecked():
                        _b_ph = []
                        for _i, (_u, _v) in enumerate(self.edge_list):
                            _u_bt = self.G.nodes[_u].get('boundary_type')
                            _v_bt = self.G.nodes[_v].get('boundary_type')
                            if _u_bt in ('source', 'sink') or _v_bt in ('source', 'sink'):
                                _vv = vals[_i]
                                if _vv is not None and np.isfinite(_vv):
                                    _b_ph.append(float(_vv))
                        if _b_ph:
                            _cm = np.mean(np.exp(1j * np.asarray(_b_ph)))
                            self._phase_boundary_offset_rad = float(np.angle(_cm))
                elif diverging:
                    # Diverging map centred at 0: blue = venous-like,
                    # red = arterial-like.  Under Percentile mode the
                    # limits are auto-symmetrised; under Manual mode
                    # we respect the user's exact vmin/vmax (so you can
                    # deliberately show an asymmetric divergent map).
                    cmap = plt.get_cmap('coolwarm')
                    if bounds_mode == 'Manual':
                        norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                        self._last_vmin, self._last_vmax = float(vlo), float(vhi)
                    else:
                        vmax_sym = float(max(abs(vlo), abs(vhi), 1e-9))
                        norm = mcolors.Normalize(vmin=-vmax_sym, vmax=vmax_sym)
                        self._last_vmin, self._last_vmax = -vmax_sym, vmax_sym
                else:
                    cmap = plt.get_cmap('viridis')
                    norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                    self._last_vmin, self._last_vmax = float(vlo), float(vhi)
                colors = []
                _ph_offset = getattr(self, '_phase_offset_rad', 0.0)
                _ph_bnd_offset = getattr(self, '_phase_boundary_offset_rad', 0.0)
                _is_phase = 'phase' in field
                for v_ in vals:
                    if v_ is None:
                        colors.append(NO_DATA_COLOR)
                    else:
                        if _is_phase:
                            # π-unified LAG convention: 0 at boundary
                            # anchors, increases outward with delay.
                            #   lag = (φ_ref − φ_edge) mod 2π
                            #   if lag < π: lag += π   (unify)
                            #   lag -= π               → [0, π)
                            # Sign is opposite of raw φ_edge because
                            # arctan2(-B, A) gives φ that DECREASES with
                            # propagation delay, so lag = ref − φ_edge
                            # is the physically-intuitive direction.
                            rel = (-(float(v_) + _ph_offset)
                                    + _ph_bnd_offset) % (2 * np.pi)
                            if rel < np.pi:
                                rel = rel + np.pi
                            v_use = rel - np.pi
                        elif (self.log_scale and not diverging and v_ > 0):
                            v_use = np.log10(max(v_, 1e-30))
                        else:
                            v_use = v_
                        colors.append(mcolors.to_hex(cmap(norm(v_use))))

        # Radial-uncertainty highlight override.  Runs LAST so it wins
        # over the field-driven colormap.  Only applies when the user
        # has the highlight toggle on.
        if getattr(self, '_show_radial_uncertain', False):
            for i, (u, v) in enumerate(self.edge_list):
                if self.G.edges[u, v].get('radial_uncertain'):
                    colors[i] = RADIAL_UNCERTAIN_COLOR

        # Flow-direction mismatch highlight override (also runs LAST; wins
        # over the radial-uncertain colour when both apply to an edge).
        if getattr(self, '_show_flow_mismatch', False):
            mm = getattr(self, '_flow_mismatch_edges', None)
            if mm is None:
                mm = self._compute_flow_mismatch_edges()
            for i, (u, v) in enumerate(self.edge_list):
                if frozenset((u, v)) in mm:
                    colors[i] = FLOW_MISMATCH_COLOR

        # Uncertain-fit override: colour the near-arterial splitter edges by
        # their fitted conductance correction θ (red = ↑, blue = ↓ vs baseline).
        _unc_chk = getattr(self, '_unc_show_theta', None)
        _unc_theta = getattr(self, '_unc_theta', None)
        if _unc_theta and _unc_chk is not None and _unc_chk.isChecked():
            from matplotlib import cm as _cm
            _tn = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            for i, (u, v) in enumerate(self.edge_list):
                th = _unc_theta.get(frozenset((u, v)))
                if th is not None and th > 0:
                    colors[i] = mcolors.to_hex(_cm.coolwarm(_tn(np.log10(th))))

        # Rebuild the edges layer from scratch.  Replacing it cleanly
        # avoids a napari Shapes-layer rendering bug where the .data +
        # .edge_color assignments on an existing layer don't always
        # repaint correctly.
        if self._edges_layer is not None:
            try:
                self.viewer.layers.remove(self._edges_layer)
            except (KeyError, ValueError):
                pass
            self._edges_layer = None
        if lines:
            self._edges_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=colors,
                edge_width=widths,
                name='Edges', opacity=0.7,
            )
        n_usable = sum(1 for v in vals if v is not None)
        print(f"  Drew {len(lines)} edges  "
              f"({n_usable} coloured by {field}, "
              f"{len(lines) - n_usable} grey: no usable measurement)")
        # Status message via window title
        title = (f"Mosaic (read-only) — {self.graph_path.name}  |  "
                 f"field={field}  |  n_drawn={len(lines)}")
        self.viewer.title = title
        # Refresh the colour-bar overlay to match the new bounds /
        # colormap.  (Old `_cbar_canvas` attribute is gone; the live
        # state is in `_cbar_chk` + `_cbar_overlay`.)
        if getattr(self, '_cbar_chk', None) is not None:
            self._refresh_cbar()
        # Edge value labels (no-op unless toggle on + single tile)
        self._refresh_edge_labels()

    def _refresh_nodes(self):
        if self._nodes_layer is None:
            return
        if not self.show_nodes:
            self._nodes_layer.visible = False
            return
        # Collect per-node positions (with y-flip to align with the mosaic
        # image) and per-node field values (mean of incident edges with
        # usable measurements).  Nodes with no incident usable edge get
        # NO_DATA_COLOR — same convention as the edges layer.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        node_field: Dict[int, list] = {n: [] for n in self.G.nodes()}
        for u, v in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            if is_categorical:
                if not _measurement_usable(m):
                    continue
                if self.current_property == 'harmonic_class':
                    hc = d.get('harmonic_class')
                    if hc is not None:
                        node_field[u].append(int(hc))
                        node_field[v].append(int(hc))
                else:   # quality_tier
                    tier = m.get('quality_tier', 'X')
                    node_field[u].append(QUALITY_RANK.get(tier, 0))
                    node_field[v].append(QUALITY_RANK.get(tier, 0))
            else:
                val = self._resolve_field_value(u, v)
                if val is not None and np.isfinite(val):
                    node_field[u].append(val)
                    node_field[v].append(val)
        # Build points + colors in node-order
        nodes = list(self.G.nodes())
        pts = np.array([[self.mosaic_height - self.G.nodes[n]['y'],
                         self.G.nodes[n]['x']] for n in nodes])
        vals = np.array([np.mean(node_field[n]) if node_field[n] else np.nan
                         for n in nodes], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0 or is_categorical:
            face = [NO_DATA_COLOR] * len(nodes)
        else:
            arr = vals.copy()
            if self.log_scale:
                pos = finite[finite > 0]
                if len(pos):
                    finite_eff = np.log10(pos)
                    arr = np.where((arr > 0) & np.isfinite(arr),
                                   np.log10(np.maximum(arr, 1e-30)),
                                   np.nan)
                else:
                    finite_eff = finite
            else:
                finite_eff = finite
            vlo, vhi = np.percentile(finite_eff, [2, 98])
            if vlo == vhi:
                vhi = vlo + 1e-9
            cmap = (plt.get_cmap('hsv') if 'phase' in field
                    else plt.get_cmap('viridis'))
            if 'phase' in field:
                # π-unified: fold onto [0, π] (see edge phase pipeline)
                norm = mcolors.Normalize(vmin=0.0, vmax=float(np.pi))
                _ph_offset_node = float(np.radians(
                    getattr(self, '_phase_offset_spin', None).value()
                    if getattr(self, '_phase_offset_spin', None)
                    else 0.0))
                # Auto-zero at boundary mean (per-source; matches edges)
                _ph_bnd_offset_node = 0.0
                if getattr(self, '_phase_zero_boundary_chk', None) is not None \
                        and self._phase_zero_boundary_chk.isChecked():
                    _b_ph = []
                    for _n, _av in zip(nodes, arr):
                        if not np.isfinite(_av): continue
                        if self.G.nodes[_n].get('boundary_type') in ('source','sink'):
                            _b_ph.append(float(_av))
                    if _b_ph:
                        _cm = np.mean(np.exp(1j * np.asarray(_b_ph)))
                        _ph_bnd_offset_node = float(np.angle(_cm))
            else:
                norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                _ph_offset_node = 0.0
                _ph_bnd_offset_node = 0.0
            face = []
            _is_phase = 'phase' in field
            for v_ in arr:
                if not np.isfinite(v_):
                    face.append(NO_DATA_COLOR)
                else:
                    if _is_phase:
                        # π-unified LAG convention (matches edge pipeline):
                        # lag = (ref − v) mod 2π, folded to [0, π).
                        rel = (-(float(v_) + _ph_offset_node)
                                + _ph_bnd_offset_node) % (2 * np.pi)
                        if rel < np.pi:
                            rel = rel + np.pi
                        v_use = rel - np.pi
                    else:
                        v_use = v_
                    face.append(mcolors.to_hex(cmap(norm(v_use))))
        self._nodes_layer.data = pts
        if len(pts):
            self._nodes_layer.face_color = face
        self._nodes_layer.visible = True

    # ── Simulate tab + colour-bar UI ───────────────────────────────────
    def _build_uncertain_tab(self, layout):
        """Fit the near-arterial uncertain-edge conductances (flow splitters)
        + ΔP so the network's downstream DC flows better match measured
        mean_Q, with a ΔP-only control.  Colours splitters by the fitted
        conductance correction and reports the improvement + predicted ΔP."""
        from qtpy.QtWidgets import (QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
                                    QComboBox, QHBoxLayout, QCheckBox, QWidget)
        self._unc_theta = {}          # frozenset(u,v) -> fitted θ multiplier
        self._unc_result = None
        intro = QLabel(
            "<b>Uncertain-vessel flow routing.</b>  Uncertain edges near the "
            "arterial inlet act as flow splitters with uncertain conductance. "
            "This fits their relative conductances (+ a free ΔP) so the "
            "network's <i>downstream</i> DC flows better match measured mean_Q — "
            "with a ΔP-only control isolating the splitters' effect.")
        intro.setWordWrap(True); layout.addWidget(intro)

        def _row(lbl, w):
            r = QWidget(); rl = QHBoxLayout(r); rl.setContentsMargins(0, 2, 0, 2)
            rl.addWidget(QLabel(lbl)); rl.addWidget(w); rl.addStretch(1)
            layout.addWidget(r); return w

        self._unc_tile = QComboBox()
        self._unc_tile.addItem("Whole plexus (best-tile mean_Q)", None)
        for t in sorted((self.G.graph.get('tile_f0_piv', {}) or {})):
            self._unc_tile.addItem(f"Tile {t}", int(t))
        _row("Match:", self._unc_tile)
        self._unc_dp = QDoubleSpinBox(); self._unc_dp.setRange(0.1, 500.); self._unc_dp.setValue(10.)
        self._unc_dp.setToolTip("Baseline ΔP (Pa); the fit scales it (free ΔP).")
        _row("ΔP baseline (Pa):", self._unc_dp)
        self._unc_unc_hops = QSpinBox(); self._unc_unc_hops.setRange(1, 60); self._unc_unc_hops.setValue(15)
        _row("Splitter reach (hops from source):", self._unc_unc_hops)
        self._unc_down_hops = QSpinBox(); self._unc_down_hops.setRange(1, 60); self._unc_down_hops.setValue(15)
        _row("Downstream targets beyond (hops):", self._unc_down_hops)
        self._unc_min_snr = QDoubleSpinBox(); self._unc_min_snr.setRange(-20., 40.); self._unc_min_snr.setValue(3.)
        _row("Min DC SNR (dB):", self._unc_min_snr)
        self._unc_reg = QDoubleSpinBox(); self._unc_reg.setRange(0., 5.); self._unc_reg.setDecimals(2)
        self._unc_reg.setValue(0.3); self._unc_reg.setToolTip("Keeps θ near 1; raise if θ rail.")
        _row("Regularisation:", self._unc_reg)

        self._unc_run_btn = QPushButton("Run fit")
        self._unc_run_btn.clicked.connect(self._run_uncertain_fit)
        layout.addWidget(self._unc_run_btn)
        self._unc_status = QLabel(""); self._unc_status.setWordWrap(True)
        layout.addWidget(self._unc_status)
        self._unc_result_lbl = QLabel(""); self._unc_result_lbl.setWordWrap(True)
        self._unc_result_lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._unc_result_lbl)
        self._unc_show_theta = QCheckBox("Colour splitters by θ (red ↑ / blue ↓ vs baseline)")
        self._unc_show_theta.setChecked(True)
        self._unc_show_theta.toggled.connect(lambda _v: self._refresh_edges())
        layout.addWidget(self._unc_show_theta)
        self._unc_plot_btn = QPushButton("Show plots (scatter / θ histogram)")
        self._unc_plot_btn.clicked.connect(self._show_uncertain_plots)
        self._unc_plot_btn.setEnabled(False)
        layout.addWidget(self._unc_plot_btn)
        self._unc_map_btn = QPushButton("Show residual map (naive vs corrected, spatial)")
        self._unc_map_btn.clicked.connect(self._show_uncertain_residual_map)
        self._unc_map_btn.setEnabled(False)
        layout.addWidget(self._unc_map_btn)

    def _run_uncertain_fit(self):
        from qtpy.QtWidgets import QApplication
        import time
        try:
            from scripts.fit_uncertain_conductances import fit_downstream
        except Exception:
            import importlib.util
            from pathlib import Path
            sp = Path(__file__).resolve().parents[2] / "scripts" / "fit_uncertain_conductances.py"
            spec = importlib.util.spec_from_file_location("fit_unc_mod", sp)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            fit_downstream = mod.fit_downstream
        self._unc_run_btn.setEnabled(False)
        self._unc_status.setText("<small>Running forward solve + fit … (~30–60 s)</small>")
        QApplication.processEvents()
        t0 = time.time()
        try:
            res = fit_downstream(
                self.G,
                unc_hops=int(self._unc_unc_hops.value()),
                down_hops=int(self._unc_down_hops.value()),
                min_snr_dc=float(self._unc_min_snr.value()),
                reg=float(self._unc_reg.value()),
                dp=float(self._unc_dp.value()),
                tile=self._unc_tile.currentData())
        except Exception as e:
            self._unc_status.setText(f"<small><font color='#c33'>Fit failed: {e}</font></small>")
            self._unc_run_btn.setEnabled(True); return
        self._unc_result = res
        self._unc_theta = {frozenset(uv): float(th)
                           for uv, th in zip(res['unc_edge_uv'], res['theta'])}
        if not res['n_targets'] or not len(res['theta']):
            self._unc_status.setText("<small><font color='#c33'>No splitters/targets — "
                                     "loosen the hop / SNR thresholds.</font></small>")
            self._unc_run_btn.setEnabled(True); return
        imp = 100*(res['r_control']-res['r_full'])/res['r_control']
        th = res['theta']
        self._unc_result_lbl.setText(
            f"splitters: {res['n_splitters']}   downstream targets: {res['n_targets']}\n"
            f"UNCORRECTED (ΔP only)  residual={res['r_control']:.3f}  ΔP={res['dp_control']:.1f} Pa\n"
            f"CORRECTED   (ΔP + θ)   residual={res['r_full']:.3f}  ΔP={res['dp_full']:.1f} Pa\n"
            f"→ downstream matching improved {imp:+.1f}%\n"
            f"θ: median {np.median(th):.2f}, range [{th.min():.2f}, {th.max():.2f}], "
            f"{int(np.sum(np.abs(np.log(th))>0.3))} moved >2×")
        self._unc_status.setText(f"<small>Done in {time.time()-t0:.0f} s.  "
                                 "Splitters coloured on the mosaic by θ.</small>")
        self._unc_plot_btn.setEnabled(True)
        self._unc_map_btn.setEnabled(True)
        self._unc_run_btn.setEnabled(True)
        self._refresh_edges()

    def _show_uncertain_plots(self):
        res = getattr(self, '_unc_result', None)
        if res is None or not len(res.get('theta', [])):
            return
        import matplotlib.pyplot as plt
        theta = res['theta']; tq = np.asarray(res['tgt_q'])
        sc, sf = np.asarray(res['sim_control']), np.asarray(res['sim_full'])
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        lim = np.percentile(np.abs(tq), 99) if len(tq) else 1.0
        for a, (qs, ttl, rr) in zip(ax[:2],
                [(sc, 'Uncorrected (ΔP only)', res['r_control']),
                 (sf, 'Corrected (ΔP + splitters)', res['r_full'])]):
            a.scatter(tq, qs, s=7, alpha=.25, edgecolors='none')
            a.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
            a.set_xlabel('measured mean_Q (nL/s)'); a.set_title(f'{ttl}\nresidual = {rr:.3f}')
        ax[0].set_ylabel('simulated DC flow (nL/s)')
        ax[2].hist(np.log10(theta), bins=35, color='steelblue'); ax[2].axvline(0, color='k', ls='--')
        ax[2].set_xlabel('log10 θ'); ax[2].set_title('splitter conductance corrections')
        fig.suptitle(f"Uncertain-vessel fit — downstream matching "
                     f"{100*(res['r_control']-res['r_full'])/res['r_control']:+.1f}%,  "
                     f"predicted ΔP = {res['dp_full']:.1f} Pa")
        fig.tight_layout(); plt.show()

    def _show_uncertain_residual_map(self):
        """Spatial map of the per-edge downstream residual |sim − meas|:
        naive (ΔP only) | corrected (ΔP + splitters) | per-edge improvement.
        Shows WHERE the conductance fit helps (or hurts), on the mosaic."""
        res = getattr(self, '_unc_result', None)
        if res is None or not len(res.get('theta', [])) or not res.get('n_targets'):
            return
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        tq = np.asarray(res['tgt_q'])
        r_naive = np.abs(np.asarray(res['sim_control']) - tq)
        r_corr  = np.abs(np.asarray(res['sim_full'])    - tq)
        improve = r_naive - r_corr                      # + = better after fit
        pos = {n: (self.G.nodes[n]['x'], self.G.nodes[n]['y']) for n in self.G.nodes()}
        segs = np.array([[pos[u], pos[v]] for u, v in res['tgt_edge_uv']])
        usegs = np.array([[pos[u], pos[v]] for u, v in res['unc_edge_uv']])
        src = [n for n, a in self.G.nodes(data=True) if a.get('boundary_type') == 'source']
        snk = [n for n, a in self.G.nodes(data=True) if a.get('boundary_type') == 'sink']

        fig, ax = plt.subplots(1, 3, figsize=(19, 7), sharex=True, sharey=True)
        vmax = np.percentile(r_naive, 95) or 1.0
        for a, (rr, ttl) in zip(ax[:2],
                [(r_naive, f"Naive (ΔP only)   ‖r‖={res['r_control']:.3f}"),
                 (r_corr,  f"Corrected (ΔP+θ)  ‖r‖={res['r_full']:.3f}")]):
            lc = LineCollection(segs, array=rr, cmap='viridis',
                                norm=mcolors.Normalize(0, vmax), linewidths=1.4)
            a.add_collection(lc); plt.colorbar(lc, ax=a, shrink=.6, label='|sim − meas| (nL/s)')
            a.set_title(ttl)
        vlim = np.percentile(np.abs(improve), 95) or 1.0
        lci = LineCollection(segs, array=improve, cmap='RdBu',
                             norm=mcolors.TwoSlopeNorm(vcenter=0., vmin=-vlim, vmax=vlim),
                             linewidths=1.6)
        ax[2].add_collection(lci); plt.colorbar(lci, ax=ax[2], shrink=.6,
                                                label='|r_naive| − |r_corr|  (red=better)')
        ax[2].set_title('Per-edge improvement')
        for a in ax:                                     # splitters (grey) + boundary
            a.add_collection(LineCollection(usegs, colors='0.6', linewidths=0.8, zorder=1))
            for n in src: a.scatter(*pos[n], c='red', marker='^', s=70, zorder=5)
            for n in snk: a.scatter(*pos[n], c='blue', marker='v', s=70, zorder=5)
            a.autoscale(); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"Downstream residual map — overall "
                     f"{100*(res['r_control']-res['r_full'])/res['r_control']:+.1f}% "
                     f"({res['n_targets']} target edges; grey = splitters)")
        fig.tight_layout(); plt.show()

    def _build_simulate_tab(self, layout):
        """Forward transmission-line solver controls.

        MVP scope: BCs come from the stored `bc_harmonics` on boundary
        nodes (no custom override yet).  D and f₀ are user-adjustable.
        Results are written to the in-memory graph under the
        `_sim_tmp_*` namespace and added to the View-tab field dropdown
        when present.  Never persisted to the gpickle.
        """
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QRadioButton, QButtonGroup, QDoubleSpinBox, QCheckBox,
            QComboBox as _QComboBox)

        # ── Detect default f₀ from the graph ──
        self._sim_detected_f0 = self._detect_sim_f0_hz()

        # ── BC source ──
        bc_group = QGroupBox("Boundary conditions")
        bc_lay = QVBoxLayout(bc_group)
        # Sole BC mode (2026-07-02): two-pass Dirichlet-tie ΔP with
        # per-boundary normalized-Q-spectrum matching.
        #   Pass 1: Dirichlet ΔP at DC only, no AC drive.  Extract Q_0
        #           at every boundary from the DC solve.
        #   Pass 2: fully Neumann Q at every boundary; each Q at
        #           harmonic k is Q_0 · (bc_harmonics[k] / bc_harmonics[0]),
        #           i.e. Q_0 scaled by that node's measured normalized
        #           spectrum.
        # Result: Q_V and Q_A both reproduce the measured normalized
        # Q spectrum (|H_k/H_0| and phases) at every boundary while DC
        # is driven by ΔP.
        self._sim_bc_radio_dirichlet = QRadioButton(
            "Dirichlet-tie ΔP + normalized-Q match")
        # Not offered as a UI option in plexusflow — see note below where
        # the simplified BC scheme is described.  Always False so
        # _run_simulation takes the equal-flux branch.
        self._sim_bc_radio_dirichlet.setChecked(False)
        self._sim_bc_radio_dirichlet.hide()
        self._sim_bc_radio_dirichlet.setToolTip(
            "Sole simulation BC mode.\n\n"
            "Two-pass solve:\n"
            "  1) Pure DC Dirichlet: P_A − P_V = ΔP (below), no AC drive.\n"
            "     Extract Q̂_DC(n) at every boundary from the DC solve\n"
            "     (summed from edge_flows, respects conservation exactly).\n"
            "  2) Fully Neumann Q at every boundary:\n"
            "       target[0] = Q̂_DC(n)                (predicted DC)\n"
            "       target[k] = Q̂_DC(n) · bch[k]/bch[0]   (measured PI ratio)\n"
            "     The INJECTED PI at each boundary equals the MEASURED PI\n"
            "     bch[k]/bch[0], by construction — that's the boundary\n"
            "     invariant we trust.  AC magnitude scales with predicted\n"
            "     DC (so an AC-magnitude test is coupled to DC accuracy).\n\n"
            "Result: measured PI ratio preserved at every boundary; DC\n"
            "magnitude at every boundary is the geometry prediction.")
        # Legacy radios removed 2026-07-02 (were: classwave, measured,
        # custom).  Kept as attributes for back-compat with other code
        # paths that check .isChecked() — they're always False now.
        self._sim_bc_radio_classwave = QRadioButton()
        self._sim_bc_radio_classwave.setChecked(False)
        self._sim_bc_radio_classwave.hide()
        self._sim_bc_radio_measured = QRadioButton()
        self._sim_bc_radio_measured.setChecked(False)
        self._sim_bc_radio_measured.hide()
        self._sim_bc_radio_custom = QRadioButton()
        self._sim_bc_radio_custom.setChecked(False)
        self._sim_bc_radio_custom.hide()
        # NEW BC mode: equal-flux at A (Q(A1)=Q(A2)=Q_A_total/2), V grounded.
        # Trust the anatomical claim that paired A ports have equal DC flow;
        # let the total flux be a knob (default = measured A DC sum, but
        # user can scan it).  The A pressures P(A1), P(A2) become model
        # OUTPUTS — if geometry is self-consistent with A1=A2 they should
        # be near-equal; if not, the split gives a diagnostic of network
        # asymmetry.  V pressures pinned to 0 at all harmonics.  AC at A
        # uses measured PI ratio × prescribed Q_A/2 per port; V AC is
        # a full model output.
        self._sim_bc_radio_equalflux = QRadioButton(
            "Equal-flux at A + grounded V")
        # Simplified plexusflow BC: this is the SOLE mode.  Always True so
        # _run_simulation takes the two-pass equal-flux branch.
        self._sim_bc_radio_equalflux.setChecked(True)
        self._sim_bc_radio_equalflux.hide()
        self._sim_bc_radio_equalflux.setToolTip(
            "Q(A1) = Q(A2) = Q_A_total / 2, prescribed as Q-BC at both A\n"
            "nodes.  V grounded at P=0 (all harmonics).  AC at A uses\n"
            "measured PI × Q_A/2 per port (measured PI preserved).\n\n"
            "Q_A_total (spinbox below) defaults to the sum of measured A\n"
            "DC — sweep it to explore how V split responds to total flux.\n\n"
            "Diagnostic prints P(A1), P(A2), P(V1), P(V2) as solver outputs\n"
            "— non-equal A pressures indicate geometric asymmetry between\n"
            "the two arterial paths.")
        bc_group_buttons = QButtonGroup(bc_group)
        bc_group_buttons.addButton(self._sim_bc_radio_dirichlet)
        bc_group_buttons.addButton(self._sim_bc_radio_equalflux)
        # BC-mode radios hidden — plexusflow ships with a single BC scheme.
        # A static label documents what runs:
        _bc_scheme_lbl = QLabel(
            "<b>Sole BC scheme</b> — two-pass, mixed BC by harmonic:<br>"
            "&nbsp;&nbsp;<b>DC</b>: <tt>Q(A_r) = Q_A_total / N_A</tt> at every A (Neumann); "
            "<tt>P(V_r) = 0</tt> at every V (Dirichlet, common reservoir).<br>"
            "&nbsp;&nbsp;→ Pass 1 solves for the network-determined venous split "
            "<tt>Q&#770;_DC(V_r)</tt>.<br>"
            "&nbsp;&nbsp;<b>AC (n≥1)</b>: fully Neumann at every boundary — "
            "<tt>Q^(n)(A_r) = (Q_A_total/N_A) · bch[n]/bch[0]</tt>, "
            "<tt>Q^(n)(V_r) = Q&#770;_DC(V_r) · bch[n]/bch[0]</tt>.")
        _bc_scheme_lbl.setWordWrap(True)
        bc_lay.addWidget(_bc_scheme_lbl)

        # Q_A_total spinbox (used only in equal-flux mode)
        qa_row = QHBoxLayout()
        qa_row.addWidget(QLabel("Q_A_total (nL/s, prescribed arterial DC):"))
        self._sim_qa_spin = QDoubleSpinBox()
        self._sim_qa_spin.setDecimals(3)
        self._sim_qa_spin.setRange(0.001, 100.0)
        self._sim_qa_spin.setSingleStep(0.1)
        # Default = Σ measured A DC (bc_harmonics[0].real over source nodes).
        # Falls back to 1.0 nL/s if no A bc_harmonics available.
        qa_default = 0.0
        for _an in self._source_nodes:
            _bch = self.G.nodes[_an].get('bc_harmonics')
            if _bch is not None:
                try:
                    qa_default += float(np.asarray(_bch, dtype=complex)[0].real)
                except Exception:
                    pass
        if not (np.isfinite(qa_default) and qa_default > 0):
            qa_default = 1.0
        self._sim_qa_spin.setValue(qa_default)
        self._sim_qa_spin.setToolTip(
            "Total arterial DC flux (Σ over A ports).  In equal-flux mode,\n"
            "each A port is prescribed Q(Ai) = Q_A_total / N_A.\n\n"
            "Default = Σ measured A DC (from bc_harmonics on A nodes) so\n"
            "the sim's absolute magnitudes match the measurement.  Scan\n"
            "to see how the venous split and interior flows respond.")
        qa_row.addWidget(self._sim_qa_spin)
        qa_row.addStretch(1)
        bc_lay.addLayout(qa_row)

        # ── V-closure sub-selector ────────────────────────────────────
        # Only relevant in the Dirichlet-tie path.  Pass 1 always uses
        # common P_V = 0.  In Pass 2:
        #   flux    — Q-BC at every boundary including V (current default;
        #             matches historical behaviour but effectively pins
        #             each V node's DC to its Pass-1 prediction).
        #   pressure — Q-BC at A only; V held at Dirichlet P=0 across all
        #             harmonics (grounded venous).  V flows are then
        #             MODEL OUTPUTS, not BCs.  Cleaner if you want the
        #             venous DC split to remain a network prediction.
        v_closure_group = QGroupBox("V closure (Pass 2)")
        v_closure_lay = QHBoxLayout(v_closure_group)
        self._sim_v_closure_flux = QRadioButton("Per-node flux at V")
        # Sole V-closure in plexusflow — Pass 2 injects
        # Q^(n)(V_r) = Q̂_DC(V_r) × bch[n]/bch[0] (paper-outline scheme).
        self._sim_v_closure_flux.setChecked(True)
        self._sim_v_closure_flux.setToolTip(
            "Pass 2 applies a Q-BC at every boundary (A and V).\n"
            "V DC values equal Pass-1 predictions; V AC values are\n"
            "Q_pred × measured (H_k/H_0).  Effectively fully-Neumann.\n"
            "Uses measured V waveform as an input.")
        self._sim_v_closure_pressure = QRadioButton("Common pressure at V (grounded, P_V_AC=0)")
        self._sim_v_closure_pressure.setToolTip(
            "V nodes held at Dirichlet P=0 across ALL harmonics.\n"
            "Assumes venous reservoir has NO AC pressure fluctuation\n"
            "— a strong simplification (real embryonic atrium DOES\n"
            "oscillate). V flows fully predicted; measured V waveform\n"
            "becomes a test target, not an input.")
        # "Tied common P at V (nonzero AC)" — uses the solver's new
        # `tied_pressure_nodes` parameter, which adds a penalty
        # admittance between listed nodes (forces equal P at every
        # harmonic) and excludes them from Q-BC / conservation
        # rescaling.  V AC pressure becomes a solve output constrained
        # to be equal at both V ports (shared-atrium physics).
        self._sim_v_closure_meas_ac = QRadioButton(
            "Predicted DC + measured AC at V (raw phasor)")
        # Not offered in plexusflow — kept as attribute for _run_simulation
        # branch check; forced False.
        self._sim_v_closure_meas_ac.setChecked(False)
        self._sim_v_closure_meas_ac.setToolTip(
            "V DC = Q̂_DC(V) predicted from Pass 1 (untrusted V DC "
            "magnitude replaced by geometry).\n"
            "V AC target[k] = bch[k] MEASURED PHASOR DIRECTLY at every "
            "harmonic (k=1,2,3) — no PI rescaling to predicted DC.\n\n"
            "Use when: V DC magnitudes are untrustworthy but V AC "
            "amplitude AND phase are trustworthy (atrial pressure "
            "oscillation is heart-driven, doesn't scale with vessel "
            "cross-section bias).")
        self._sim_v_closure_tied = QRadioButton("Tied common P at V (nonzero AC)")
        self._sim_v_closure_tied.setToolTip(
            "Adds a penalty admittance between V nodes in the solver so\n"
            "P_V1(ω) ≈ P_V2(ω) at EVERY harmonic — including AC.  V nodes\n"
            "get no Q-BC (they're network outputs, not inputs).  V DC\n"
            "gauge is set by the interior pin.\n\n"
            "Physically: shared atrial chamber with oscillating pressure.\n"
            "V AC pressure develops from network response to A drive,\n"
            "but constrained to be equal at both V ports.\n"
            "V AC flow is a genuine model output — can be tested against\n"
            "measured V waveform.")
        _v_closure_bg = QButtonGroup(v_closure_group)
        _v_closure_bg.addButton(self._sim_v_closure_flux)
        _v_closure_bg.addButton(self._sim_v_closure_pressure)
        _v_closure_bg.addButton(self._sim_v_closure_meas_ac)
        _v_closure_bg.addButton(self._sim_v_closure_tied)
        # V-closure sub-selector hidden — plexusflow uses the "Per-node
        # flux at V" scheme (documented in the top BC-scheme label above).
        # The QRadioButton objects still exist so downstream .isChecked()
        # calls don't crash.  Layout add-widgets omitted below.
        v_closure_group.hide()
        v_closure_lay.addStretch(1)
        bc_lay.addWidget(v_closure_group)

        # DC-consistency diagnostic checkbox
        self._sim_dc_diag_chk = QCheckBox("Print Pass-1 vs measured DC per port")
        self._sim_dc_diag_chk.setChecked(True)
        self._sim_dc_diag_chk.setToolTip(
            "After Pass 1, log a per-port table comparing:\n"
            "  Q_pred (from ΔP + geometry + D)  vs\n"
            "  Q_meas = Re(bc_harmonics[0])     — the measured DC at that node.\n\n"
            "Their ratio and difference tell you how well the network\n"
            "geometry alone explains the measured DC split — separately\n"
            "for arterial and venous ports.  If either class has a big\n"
            "systematic mismatch, that's a geometry-model failure.")
        bc_lay.addWidget(self._sim_dc_diag_chk)

        # Quick summary of available boundary nodes.
        bc_info_lines = [
            f"Sources (A): {len(self._source_nodes)}",
            f"Sinks (V):   {len(self._sink_nodes)}",
        ]
        bc_lay.addWidget(QLabel("<small><tt>"
                                 + "<br>".join(bc_info_lines)
                                 + "</tt></small>"))

        # ── Custom waveform sub-panel ──
        # Twelve spinboxes: amp fraction (|H_k|/|Q̄|) + phase (degrees)
        # for H1, H2, H3 on EACH side (A and V) independently.  DC
        # magnitude is set automatically from the Inlet/outlet flux +
        # N_src / N_snk and the sign convention (+ on A, − on V into
        # the network).  Equal-split mode is the default — every A
        # boundary gets the same A waveform and every V boundary gets
        # the same V waveform.  Live preview reflects both at once.
        self._sim_custom_group = QGroupBox(
            "Custom waveform shape  (independent A and V)")
        cust_lay = QVBoxLayout(self._sim_custom_group)
        _cust_help_lbl = QLabel(
            "<small>Specify the A and V waveforms separately.  Each "
            "row gives the amp fraction (|H<sub>k</sub>|/|Q̄|) and "
            "phase (degrees) per harmonic for that side.  The chosen "
            "shape is applied to ALL boundaries of that type — every "
            "A gets the A spec, every V gets the V spec.  DC magnitudes "
            "come from flux / N.  Preview shows both with the sign "
            "chosen so DC reads positive on each side.</small>")
        _cust_help_lbl.setWordWrap(True)
        cust_lay.addWidget(_cust_help_lbl)
        # Twelve spinboxes — (side, harmonic, kind) → spinbox.
        # Header row labels the columns.
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(QLabel("<b></b>"))
        hdr_row.addWidget(QLabel("<b>A amp</b>"))
        hdr_row.addWidget(QLabel("<b>A phase°</b>"))
        hdr_row.addWidget(QLabel("<b>V amp</b>"))
        hdr_row.addWidget(QLabel("<b>V phase°</b>"))
        hdr_row.addStretch(1)
        cust_lay.addLayout(hdr_row)
        self._sim_custom_spins = {}   # (side, harmonic, kind) → spinbox
        # Sensible defaults: H1 ≈ 0.5 (PI ~ 1), H2 ≈ 0.15, H3 ≈ 0.05.
        # Same defaults for both sides; the user changes V to taste.
        _default_amps = {1: 0.5, 2: 0.15, 3: 0.05}
        for k in (1, 2, 3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"H{k}:"))
            for side in ('A', 'V'):
                amp = QDoubleSpinBox()
                amp.setDecimals(3); amp.setRange(0.0, 5.0)
                amp.setSingleStep(0.05)
                amp.setValue(_default_amps[k])
                amp.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(amp)
                phase = QDoubleSpinBox()
                phase.setDecimals(1); phase.setRange(-360.0, 360.0)
                phase.setSingleStep(5.0); phase.setValue(0.0)
                phase.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(phase)
                self._sim_custom_spins[(side, k, 'amp')] = amp
                self._sim_custom_spins[(side, k, 'phase')] = phase
            row.addStretch(1)
            cust_lay.addLayout(row)
        # Preview canvas (small matplotlib figure)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigureCanvas)
        self._sim_custom_fig = Figure(figsize=(4.0, 1.7), dpi=90,
                                       tight_layout=True)
        self._sim_custom_canvas = _FigureCanvas(self._sim_custom_fig)
        self._sim_custom_canvas.setMinimumHeight(150)
        cust_lay.addWidget(self._sim_custom_canvas)
        # Initial enable state — spinboxes disabled (measured is the
        # default BC source), preview canvas always visible.
        for sp in self._sim_custom_spins.values():
            sp.setEnabled(False)
        # Hidden unless "Custom BCs" is selected — the class-waveform default
        # and measured mode don't need the spinbox grid + preview (saves a
        # large amount of vertical space).
        self._sim_custom_group.setVisible(False)
        bc_lay.addWidget(self._sim_custom_group)
        # Legacy custom-waveform editor — only relevant for the removed
        # Custom BC mode.  Hidden in the simplified plexusflow tab.
        self._sim_custom_group.hide()
        layout.addWidget(bc_group)
        # Draw the initial preview after widgets exist (will reflect
        # measured BCs since that's the default selection).  Connect
        # the flux + equal-split widgets later — those signals are
        # wired below once the corresponding controls are built.

        # ── Show BC waveforms ──
        # Pop a dialog with the measured Q(t) at every source and sink,
        # read straight from `bc_Q_t` on each boundary node.  Two panels:
        # all A's overlaid on the left, all V's overlaid on the right.
        bc_plot_btn = QPushButton("Show boundary Q(t) waveforms")
        bc_plot_btn.setToolTip(
            "Plot the measured Q(t) at every A source and every V sink,\n"
            "as stored in each boundary node's bc_Q_t.  Straight from\n"
            "the graph — no reconstruction, no simulation.")
        bc_plot_btn.clicked.connect(self._show_bc_waveforms)
        layout.addWidget(bc_plot_btn)

        # ── Parameters ──
        param_group = QGroupBox("Parameters")
        param_lay = QVBoxLayout(param_group)
        # D (distensibility) — areal convention.  Split into mantissa
        # and exponent so the user can pick orders of magnitude
        # explicitly: D = mantissa × 10^exponent.  Default 1.0e-3.
        d_row = QHBoxLayout()
        d_row.addWidget(QLabel("D (1/Pa, areal):"))
        self._sim_D_mantissa = QDoubleSpinBox()
        self._sim_D_mantissa.setDecimals(2)
        self._sim_D_mantissa.setRange(1.0, 9.99)
        self._sim_D_mantissa.setSingleStep(0.1)
        self._sim_D_mantissa.setValue(1.0)
        self._sim_D_mantissa.setToolTip(
            "Mantissa for D in scientific notation (1.00 to 9.99).")
        d_row.addWidget(self._sim_D_mantissa)
        d_row.addWidget(QLabel("× 10^"))
        from qtpy.QtWidgets import QSpinBox
        self._sim_D_exponent = QSpinBox()
        self._sim_D_exponent.setRange(-7, -1)
        self._sim_D_exponent.setSingleStep(1)
        self._sim_D_exponent.setValue(-3)
        self._sim_D_exponent.setToolTip(
            "Exponent for D in scientific notation (−7 to −1).\n"
            "Areal convention: typical HH-stage yolk-sac D ≈ 1.3e-3.")
        d_row.addWidget(self._sim_D_exponent)
        d_row.addStretch(1)
        param_lay.addLayout(d_row)
        # η — wall loss angle (constant-loss / structural-damping form).
        # D_eff(ω) = D · (1 − iη) applied per harmonic via a monkey-patch
        # of _vessel_admittance during the solve. η=0 → elastic.
        eta_row = QHBoxLayout()
        eta_row.addWidget(QLabel("η (loss tan, wall):"))
        self._sim_eta_spin = QDoubleSpinBox()
        self._sim_eta_spin.setDecimals(2)
        self._sim_eta_spin.setRange(0.0, 2.0)
        self._sim_eta_spin.setSingleStep(0.05)
        self._sim_eta_spin.setValue(0.0)
        self._sim_eta_spin.setToolTip(
            "Wall loss tangent η in D_eff(ω) = D·(1 − iη).\n"
            "0 = elastic (default). ~0.4 matches the current fit sweet spot.")
        eta_row.addWidget(self._sim_eta_spin)
        eta_row.addStretch(1)
        param_lay.addLayout(eta_row)
        # f0 (read-only label, auto-detected).
        f0_row = QHBoxLayout()
        f0_row.addWidget(QLabel("f₀ (Hz):"))
        f0_val = (f"{self._sim_detected_f0:.3f}"
                  if np.isfinite(self._sim_detected_f0) else "n/a")
        self._sim_f0_lbl = QLabel(f"<tt>{f0_val}</tt>  "
                                   "<small>(auto from graph)</small>")
        f0_row.addWidget(self._sim_f0_lbl)
        f0_row.addStretch(1)
        param_lay.addLayout(f0_row)
        # Inlet/outlet flux: scale BCs so total source DC = this value.
        flux_row = QHBoxLayout()
        flux_row.addWidget(QLabel("Inlet/outlet flux (nL/s):"))
        self._sim_flux_spin = QDoubleSpinBox()
        self._sim_flux_spin.setDecimals(3)
        self._sim_flux_spin.setRange(0.001, 1000.0)
        self._sim_flux_spin.setSingleStep(0.1)
        self._sim_flux_spin.setValue(1.0)
        self._sim_flux_spin.setToolTip(
            "Total inflow (Σ source DC) is scaled to this value.  All\n"
            "boundary harmonics are scaled by the same factor, so the\n"
            "AC content's relative shape is preserved.  The solver's\n"
            "conservation step then enforces Σ sinks = Σ sources.")
        flux_row.addWidget(self._sim_flux_spin)
        flux_row.addStretch(1)
        # Legacy — only used by the removed Dirichlet-tie BC mode.
        # Hidden but kept so downstream code can still read .value().
        for _i in range(flux_row.count()):
            w = flux_row.itemAt(_i).widget()
            if w is not None: w.hide()
        # Mean pressure difference ΔP = P_A − P_V (Pa).  Active only
        # when the Dirichlet-tie BC mode is selected.  Set to 0 to fall
        # back to flux-driven mode (V is Dirichlet-tied, A is Q-driven).
        # Non-zero → pure Dirichlet: P at A and V both prescribed,
        # arterial Q becomes a solve output.
        dp_row = QHBoxLayout()
        dp_row.addWidget(QLabel("Mean ΔP = P_A − P_V  (Pa):"))
        self._sim_dp_spin = QDoubleSpinBox()
        self._sim_dp_spin.setDecimals(2)
        self._sim_dp_spin.setRange(0.0, 1000.0)
        self._sim_dp_spin.setSingleStep(1.0)
        self._sim_dp_spin.setValue(10.0)          # default 10 Pa (2026-07-02)
        self._sim_dp_spin.setToolTip(
            "ΔP = P_A − P_V (Pa), applied at DC in Pass 1.\n"
            "Default 10 Pa.  Q_0 at every boundary comes out of that\n"
            "DC solve; Pass 2 then scales the measured normalized Q\n"
            "spectrum at each boundary by Q_0 to build Neumann Q BCs.")
        dp_row.addWidget(self._sim_dp_spin)
        dp_row.addStretch(1)
        # Legacy — only used by the removed Dirichlet-tie BC mode.  Hidden.
        for _i in range(dp_row.count()):
            w = dp_row.itemAt(_i).widget()
            if w is not None: w.hide()
        # V-side BC handling: lets the user switch from Q-BC at V
        # (default, measured Q_V drives the network) to pressure
        # matching at V, with two flavours.  Both P-match modes turn
        # the measured Q_V into a validation target rather than an
        # input — what the user trusts physically is constrained, what
        # they trust empirically can be checked independently.

        # DC distribution + per-type AC averaging:
        # - Unchecked: preserve each boundary's measured ratios.
        # - Checked: every A gets the SAME waveform (DC = flux/N_src
        #   plus the complex mean of measured AC across all A's); same
        #   for V (with sign flip).  The A↔V phase difference (e.g.
        #   cardiac antiphase) is preserved across types.
        self._sim_equal_split_chk = QCheckBox(
            "Uniform per-type BCs  "
            "(one waveform per A's, one per V's)")
        self._sim_equal_split_chk.setChecked(True)
        self._sim_equal_split_chk.setToolTip(
            "Checked (default):  every A gets the SAME waveform and\n"
            "every V gets the SAME waveform.  In Custom mode that's\n"
            "the A / V spec from the spinboxes; in Measured mode it's\n"
            "the per-type complex average of stored bc_harmonics.\n"
            "DC is exactly ±flux/N per side, so mass balance is exact\n"
            "by construction and the solver rebalance is a no-op.\n\n"
            "Unchecked:  each boundary keeps its measured per-node\n"
            "ratios (Measured mode only; meaningless for Custom\n"
            "because the Custom spec is already per-type).  The\n"
            "solver's DC-conservation step may rescale slightly so\n"
            "Σ = 0; applied total flux can drift from the target.")
        # Legacy — Dirichlet-tie-mode toggle, always effectively "per-type"
        # in the equal-flux BC scheme.  Hidden.
        self._sim_equal_split_chk.hide()
        # Live BC preview reacts to flux + equal-split changes too, so
        # the user always sees the actual scaled waveform.
        self._sim_flux_spin.valueChanged.connect(
            self._refresh_custom_preview)
        self._sim_equal_split_chk.toggled.connect(
            self._refresh_custom_preview)
        # Draw the initial preview now that all the inputs exist.
        self._refresh_custom_preview()
        # Equalize the inflow stem conductance per A node.  For each
        # arterial source, walk from A through degree-2 nodes to the
        # first branch point — that's the "inflow stem".  Override
        # every stem edge's radius so each A's stem has the same total
        # Poiseuille resistance (median across A's).  Lengths are kept;
        # only radii change.  Useful when stem lengths differ
        # (e.g. one A has a single edge to first branch, another has
        # two edges) and you want symmetric inflow pressure drops.
        # Equalize A-stem conductance — UI removed for now.  The checkbox
        # object stays parentless (so `.isChecked()` still works in the
        # solver path and returns False), but isn't added to any layout.
        # To restore: re-add `param_lay.addWidget(self._sim_equalize_A_chk)`.
        self._sim_equalize_A_chk = QCheckBox(
            "Equalize A-stem conductance  (A → first-branch point)")
        self._sim_equalize_A_chk.setChecked(False)
        # Geometry override mode: measured, uniform radius only, or
        # uniform conductance (R AND L constant).  All three keep the
        # original graph intact; only the solve-time copy is altered.
        geom_row = QHBoxLayout()
        geom_row.addWidget(QLabel("Geometry:"))
        self._sim_geom_combo = _QComboBox()
        self._sim_geom_combo.addItems([
            "Measured (default)",
            "Uniform radius  (R const, L from measured)",
            "Uniform conductance  (R, L both constant)",
        ])
        self._sim_geom_combo.setToolTip(
            "Measured: use the graph's stored radius/length per edge.\n"
            "Uniform radius: override every R to 5 px; conductance is\n"
            "  then determined by per-edge length alone.\n"
            "Uniform conductance: override both R (5 px) and L (50 px),\n"
            "  so flow distribution is purely topological.\n"
            "Original graph is unchanged; only the solve sees overrides.")
        geom_row.addWidget(self._sim_geom_combo)
        geom_row.addStretch(1)
        param_lay.addLayout(geom_row)
        layout.addWidget(param_group)

        # ── Actions ──
        actions_row = QHBoxLayout()
        self._sim_run_btn = QPushButton("Run simulation")
        self._sim_run_btn.clicked.connect(self._run_simulation)
        actions_row.addWidget(self._sim_run_btn)
        self._sim_clear_btn = QPushButton("Clear simulation")
        self._sim_clear_btn.setEnabled(False)
        self._sim_clear_btn.clicked.connect(self._clear_simulation)
        actions_row.addWidget(self._sim_clear_btn)
        self._sim_save_btn = QPushButton("Save graph state…")
        self._sim_save_btn.setToolTip(
            "Pickle the current in-memory graph (raw + normalized +\n"
            "sim + tile_gauges + all metadata) to a *.gpickle.  Reload\n"
            "the viewer against that file and everything — normalized\n"
            "fields, sim results, and the UI parameter selections — is\n"
            "restored automatically.")
        self._sim_save_btn.clicked.connect(self._on_save_state_clicked)
        actions_row.addWidget(self._sim_save_btn)
        layout.addLayout(actions_row)

        # ── Status ──
        self._sim_status_lbl = QLabel(
            "<small><i>No simulation has been run yet.  Results live "
            "in memory under `_sim_tmp_*` fields and disappear when "
            "you close the viewer.</i></small>")
        self._sim_status_lbl.setWordWrap(True)
        layout.addWidget(self._sim_status_lbl)

        # ── Loaded-sim indicator (populated from G.graph['sim_meta']
        #    if the graph was saved after a previous run) ──
        self._sim_loaded_lbl = QLabel("")
        self._sim_loaded_lbl.setWordWrap(True)
        self._sim_loaded_lbl.setStyleSheet(
            "QLabel { color: #446; background: #eef4ff; "
            "padding: 4px; border: 1px solid #99b; }")
        self._sim_loaded_lbl.hide()
        layout.addWidget(self._sim_loaded_lbl)
        layout.addStretch(1)

        # ── No-boundary guard ──
        # If the graph carries no source / sink designations (e.g. a
        # generic mosaic without A/V assignment), every control on
        # this tab is meaningless — the solver has nothing to drive
        # the flow with.  Disable the whole tab's widgets and show
        # a single explanatory label instead of half-working spinboxes.
        if not self._source_nodes and not self._sink_nodes:
            for w in (bc_group, param_group, actions_row,
                      self._sim_status_lbl):
                _w = (w if hasattr(w, 'setEnabled')
                       else None)  # actions_row is a layout
                if _w is not None:
                    _w.setEnabled(False)
            # Disable layout-only children too (actions_row).
            for i in range(actions_row.count()):
                item = actions_row.itemAt(i).widget()
                if item is not None:
                    item.setEnabled(False)
            # Replace the bottom status with a clear message.
            self._sim_status_lbl.setText(
                "<small><b>Simulation disabled.</b>  This graph has no "
                "boundary nodes designated as sources (A) or sinks (V), "
                "so there's nothing to drive a flow simulation.  Add "
                "<code>boundary_type</code> annotations (\"source\" / "
                "\"sink\") to the relevant nodes in the gpickle to "
                "enable this tab.</small>")
            self._sim_status_lbl.setEnabled(True)  # leave the note readable

        # If the graph carries a saved sim_meta from a prior session,
        # restore the UI controls and show the "loaded" banner.
        try:
            self._restore_sim_ui_from_graph()
        except Exception as _exc:
            print(f"[sim] restore from graph failed: {_exc}")

    # ══════════════════════════════════════════════════════════════════
    # Kinematic pump tab — reduced load-coupled tubular-heart model
    # (reuses scripts/sim_kinematic_load_pump.py)
    # ══════════════════════════════════════════════════════════════════

    def _build_pump_tab(self, layout):
        """Kinematic tubular-heart pump: prescribed heart motion, arterial
        (Q_out) / venous (Q_in) waveforms respond to vascular load (ligation).
        Defaults are the Somites21 DA/SV-calibrated pump.  Visualises the
        flow + pressure waveforms; writes per-port pump flux to `_pump_*`
        node fields."""
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QDoubleSpinBox, QSpinBox, QComboBox as _QComboBox)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigureCanvas)

        intro = QLabel(
            "<b>Kinematic tubular-heart pump</b> — the heart MOTION is "
            "prescribed; arterial ejection (Q_out) and venous filling (Q_in) "
            "are solved and RESPOND to vascular load.  Defaults = the "
            "Somites21 DA/SV-calibrated pump.  Ligate / partially occlude a "
            "boundary port to see the load response.")
        intro.setWordWrap(True); layout.addWidget(intro)

        self._pump_spins = {}
        pg = QGroupBox("Pump parameters"); pl = QVBoxLayout(pg)

        def add_param(key, label, lo, hi, step, val):
            row = QHBoxLayout(); row.addWidget(QLabel(label))
            spn = QDoubleSpinBox(); spn.setDecimals(2); spn.setRange(lo, hi)
            spn.setSingleStep(step); spn.setValue(val)
            row.addWidget(spn); row.addStretch(1); pl.addLayout(row)
            self._pump_spins[key] = spn

        add_param("closure_ratio", "Closure ratio (stroke):", 0.2, 0.95, 0.05, 0.95)
        add_param("duty", "Duty cycle (active frac):", 0.2, 0.9, 0.05, 0.35)
        add_param("tau_A", "τ_A  arterial Windkessel:", 0.0, 1.8, 0.1, 1.26)
        add_param("tau_V", "τ_V  venous Windkessel:", 0.0, 1.0, 0.05, 0.34)
        add_param("C_H", "C_H  chamber compliance:", 0.05, 0.6, 0.02, 0.13)
        add_param("C_AV", "C_A=C_V  compartment compl.:", 0.05, 0.6, 0.02, 0.18)
        add_param("leak_eps", "Leak ε (backflow frac):", 0.0, 0.35, 0.02, 0.08)
        add_param("G_net0", "G_net0  (load scale):", 0.1, 1.0, 0.05, 0.25)
        layout.addWidget(pg)

        lg = QGroupBox("Ligation / occlusion"); ll = QVBoxLayout(lg)
        lr = QHBoxLayout(); lr.addWidget(QLabel("Ligate port:"))
        self._pump_lig_combo = _QComboBox()
        self._pump_lig_combo.addItem("None", userData=None)
        for n in (self._source_nodes + self._sink_nodes):
            bt = self.G.nodes[n].get("boundary_type", "?")
            self._pump_lig_combo.addItem(f"Node {n} ({bt})", userData=int(n))
        lr.addWidget(self._pump_lig_combo); lr.addStretch(1); ll.addLayout(lr)
        orow = QHBoxLayout(); orow.addWidget(QLabel("Occlusion %:"))
        self._pump_occl = QSpinBox(); self._pump_occl.setRange(0, 100)
        self._pump_occl.setSingleStep(10); self._pump_occl.setValue(100)
        self._pump_occl.setToolTip("100% = full ligation; <100% = graded stenosis")
        orow.addWidget(self._pump_occl); orow.addStretch(1); ll.addLayout(orow)
        layout.addWidget(lg)

        self._pump_run_btn = QPushButton("Run pump")
        self._pump_run_btn.clicked.connect(self._run_pump)
        layout.addWidget(self._pump_run_btn)
        self._pump_status = QLabel(
            "<small><i>Not run yet.  The first run builds the DC load model "
            "(~20–40 s); later runs are fast.</i></small>")
        self._pump_status.setWordWrap(True); layout.addWidget(self._pump_status)

        self._pump_fig = Figure(figsize=(5.2, 4.6), dpi=90, tight_layout=True)
        self._pump_canvas = _FigureCanvas(self._pump_fig)
        self._pump_canvas.setMinimumHeight(380)
        layout.addWidget(self._pump_canvas)

        self._pump_state = None
        if not self._source_nodes or not self._sink_nodes:
            self._pump_run_btn.setEnabled(False)
            self._pump_status.setText(
                "<small><font color='#c33'>This graph has no A/V boundary "
                "nodes with bc_harmonics — the pump needs measured boundary "
                "waveforms to calibrate against.</font></small>")

    def _run_pump(self):
        from qtpy.QtWidgets import QApplication
        import sys as _sys
        from pathlib import Path as _Path
        _scripts = str(_Path(__file__).resolve().parents[2] / "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        try:
            import sim_kinematic_load_pump as klp
            import sim_compartment as sc
        except Exception as e:
            self._pump_status.setText(
                f"<small><font color='#c33'>import failed: {e}</font></small>")
            return
        self._pump_run_btn.setEnabled(False)
        try:
            if self._pump_state is None:
                self._pump_status.setText("<small>Building DC load model "
                                          "(first run, ~20–40 s)…</small>")
                QApplication.processEvents()
                self._pump_state = sc.setup(str(self.graph_path), n_harmonics=3)
            state = self._pump_state
            p = klp.base_fixed_params(state["F0_HZ"])
            for k in ("closure_ratio", "duty", "tau_A", "tau_V", "C_H",
                      "C_AV", "leak_eps", "G_net0"):
                klp._set_param(p, k, float(self._pump_spins[k].value()))
            kin = klp.precompute_kinematics(p)
            bA = [int(b) for b in state["bidx_A"]]
            bV = [int(b) for b in state["bidx_V"]]
            Q0 = state["Qmeas_by_n"][0]
            A_DC = sum(abs(Q0[b]) for b in bA) * 1e12
            base = klp.graph_load(state, None); R0 = base["R_eff"]
            rec0 = klp.solve_pump(kin, p, p["G_net0"], n_cycles=60)
            scale = A_DC / max(rec0["Q_out"].mean(), klp.EPS)

            lp = self._pump_lig_combo.currentData()
            occl = self._pump_occl.value() / 100.0
            rec_lig = None; ktag = ""
            if lp is not None and occl > 0:
                if occl >= 1.0:
                    Rl = klp.graph_load(state, lp)["R_eff"]
                else:
                    Rl = klp.graded_occlusion_keff(state, lp, 1.0 - occl)
                kk = R0 / Rl
                rec_lig = klp.solve_pump(kin, p, p["G_net0"] * kk, n_cycles=60)
                ktag = f"  (ligation k={kk:.2f})"

            # ── Feed the pump's boundary waveforms into the LINEAR network
            # forward solve so ALL per-edge fields get recorded (DC, H1, H2, H3,
            # phase, pressure, PI).  Valid by harmonic balance: the
            # nonlinearity lives in the pump; the plexus is linear, so
            # per-harmonic propagation is exact.  (Interior AC is a forward
            # prediction — the pump's load model is DC-only — but there is no
            # interior measurement to contradict it.)
            # Feed the WINDKESSEL-FILTERED (DA/SV-equivalent) boundary waveforms
            # — the plexus boundary is the dorsal aorta, downstream of the
            # arterial Windkessel — not the raw ventricular pulse.
            r_use = rec_lig if rec_lig is not None else rec0
            Hout = np.asarray(klp._arch_filter(klp.to_harmonics(r_use["Q_out"]),
                                               p["tau_A"]), dtype=complex)
            Hin = np.asarray(klp._arch_filter(klp.to_harmonics(r_use["Q_in"]),
                                              p["tau_V"]), dtype=complex)
            bc_override = {}
            for b in bA:                         # sources: + into-network
                nid = state["nodes"][b]
                bc_override[nid] = Hout * (base["w_A"].get(b, 0.0) * scale)
            for b in bV:                         # sinks: − into-network
                nid = state["nodes"][b]
                bc_override[nid] = -Hin * (base["w_V"].get(b, 0.0) * scale)
            if lp is not None and occl >= 1.0 and lp in bc_override:
                bc_override[lp] = np.zeros(4, dtype=complex)   # full ligation: Q=0
            D_phys = 1.5e-3
            from ..analyze.transmission_line import solve_transmission_line  # ported 2026-09-30
            result = solve_transmission_line(
                self.G, D=D_phys, n_harmonics=3, f0_hz=float(state["F0_HZ"]),
                bc_harmonics_override=bc_override, verbose=False)
            self._store_sim_result(result, float(state["F0_HZ"]), D_phys)

            self._plot_pump(rec0, rec_lig, scale, state["F0_HZ"], ktag)

            import numpy as _np
            def met(rec):
                Qo = rec["Q_out"]; Ho = klp.to_harmonics(Qo)
                return (Qo.mean() * scale, rec["P_A"].max(),
                        abs(Ho[1]) / max(abs(Ho[0]), klp.EPS))
            mo, af, pu = met(rec0)
            txt = (f"<small>baseline: output <b>{mo:.2f}</b> nL/s, afterload "
                   f"{af:.2f}, pulsatility {pu:.2f}{ktag}")
            if rec_lig is not None:
                ml, afl, pul = met(rec_lig)
                txt += (f"<br>→ ligated: output <b>{ml:.2f}</b> "
                        f"({(ml-mo)/max(mo,1e-9)*100:+.0f}%), afterload {afl:.2f}, "
                        f"pulsatility {pul:.2f}")
            txt += ("<br><i>All per-edge fields recorded — switch the View-tab "
                    "Source to <b>Simulated</b> to color the mosaic.</i></small>")
            self._pump_status.setText(txt)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._pump_status.setText(
                f"<small><font color='#c33'>pump run failed: {e}</font></small>")
        finally:
            self._pump_run_btn.setEnabled(True)

    def _plot_pump(self, rec0, rec_lig, scale, F0, ktag):
        T_ms = 1000.0 / F0; tau = rec0["tau"]
        fig = self._pump_fig; fig.clear()
        ax1 = fig.add_subplot(2, 1, 1); ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(tau*T_ms, rec0["Q_out"]*scale, "r", lw=2, label="Q_out (arterial)")
        ax1.plot(tau*T_ms, rec0["Q_in"]*scale, "b", lw=2, label="Q_in (venous)")
        if rec_lig is not None:
            ax1.plot(tau*T_ms, rec_lig["Q_out"]*scale, "r--", lw=1.3, label="Q_out lig")
            ax1.plot(tau*T_ms, rec_lig["Q_in"]*scale, "b--", lw=1.3, label="Q_in lig")
        ax1.axhline(0, color="grey", lw=0.5); ax1.legend(fontsize=7)
        ax1.set_ylabel("flow [nL/s]")
        ax1.set_title("Pump flows" + ktag, fontsize=9)
        ax2.plot(tau*T_ms, rec0["P_A"], "r", lw=1.3, label="P_A (art.)")
        ax2.plot(tau*T_ms, rec0["P_V"], "b", lw=1.3, label="P_V (ven.)")
        ax2.plot(tau*T_ms, rec0["P_H"], "k", lw=1.0, label="P_H (chamber)")
        ax2.axhline(0, color="grey", lw=0.5); ax2.legend(fontsize=7)
        ax2.set_xlabel("t [ms]"); ax2.set_ylabel("pressure (a.u.)")
        ax2.set_title("Compartment pressures", fontsize=9)
        self._pump_canvas.draw()

    # ══════════════════════════════════════════════════════════════════
    # Inference tab — self-contained adapter region
    # ──────────────────────────────────────────────────────────────────
    # All inference plumbing (tile carve, inversion call, diagnostic
    # render) lives in the methods below.  To swap out the underlying
    # inversion algorithm, edit only:
    #   • `_inference_get_carve_edges`  (which edges form the carve)
    #   • `_inference_run`              (the per-tile call returning a
    #                                    dict of summary scalars)
    #   • `_inference_render_diagnostic` (the matplotlib popup figure)
    # The UI builder + dispatch glue below depend only on those
    # adapter methods, so the View / Time-series / Simulate tabs
    # don't need to change when the inversion model changes.
    # ══════════════════════════════════════════════════════════════════

    def _build_inference_tab(self, layout):
        """UI for the additive empirical-σ Bayesian inference.

        Per-edge measurement variance from the canonical Q_H1_snr_db
        field (r_j = 10^(-SNR/10)).  Total Σ_j = |Q_H1,j|²·(r_j + s²)
        where s² is the squared fractional model error, marginalized
        analytically over a 21-pt log grid.  D₀ has a log-normal prior
        centred at 1.5×10⁻³ /Pa, τ = ln 10.

        Per-edge D scaling:  D_e = D₀ · (R_e / R_ref)^α.  α = 0
        is the default (uniform D₀); α = +1 lets bigger vessels be
        more compliant (Murray-inverse, common for embryonic walls).
        D̂ is reported AT R_ref (median measured-edge radius).

        After running on a tile, click an edge in the viewer (the
        Time-series tab updates "Selected edge") then hit the Compare
        button to see measured vs predicted Q(t) for that edge."""
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout,
            QComboBox, QCheckBox, QTextEdit, QDoubleSpinBox,
        )
        # ── Tile picker ──
        pick_group = QGroupBox("Tile")
        pick_lay = QVBoxLayout(pick_group)
        self._inference_tile_ids = self._inference_available_tiles()
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Tile ID:"))
        self._inf_tile_combo = QComboBox()
        for tid in self._inference_tile_ids:
            self._inf_tile_combo.addItem(str(tid), userData=int(tid))
        if not self._inference_tile_ids:
            self._inf_tile_combo.addItem("(none — no PIV tiles found)")
            self._inf_tile_combo.setEnabled(False)
        self._inf_tile_combo.currentIndexChanged.connect(
            self._on_inference_tile_change)
        pick_row.addWidget(self._inf_tile_combo)
        pick_row.addStretch(1)
        pick_lay.addLayout(pick_row)
        self._inf_show_carve_chk = QCheckBox("Highlight carve on mosaic")
        self._inf_show_carve_chk.setChecked(False)
        self._inf_show_carve_chk.toggled.connect(
            self._on_inference_show_carve_toggle)
        pick_lay.addWidget(self._inf_show_carve_chk)
        layout.addWidget(pick_group)

        # ── Parameters ──
        param_group = QGroupBox("Priors and model")
        param_lay = QVBoxLayout(param_group)
        sb_row = QHBoxLayout()
        sb_row.addWidget(QLabel("σ_b (H₁ boundary forcing, Pa):"))
        self._inf_sigma_b_spin = QDoubleSpinBox()
        self._inf_sigma_b_spin.setRange(0.1, 1000.0)
        self._inf_sigma_b_spin.setDecimals(1)
        self._inf_sigma_b_spin.setSingleStep(1.0)
        self._inf_sigma_b_spin.setValue(7.0)
        self._inf_sigma_b_spin.setToolTip(
            "Absolute b prior σ (independent of s²).  Sensitivity\n"
            "sweep showed D̂ is invariant across [1, 100] Pa on\n"
            "identified tiles.  Check 'flat' to use σ_b = 10⁴ Pa,\n"
            "the wide-prior limit (formally improper but numerically\n"
            "stable — sees the b̂ that the data alone can support).")
        sb_row.addWidget(self._inf_sigma_b_spin)
        self._inf_flat_b_prior_cb = QCheckBox("flat (σ_b → ∞)")
        self._inf_flat_b_prior_cb.setToolTip(
            "When checked, override σ_b with 10⁴ Pa — far above the\n"
            "BC confound threshold (~50 Pa) — to approximate an\n"
            "improper flat prior on b_H₁.  Use this to confirm that\n"
            "the data alone (without a prior on b magnitude) push\n"
            "D̂ toward the resistive limit, exposing the D·|b̂|\n"
            "structural identifiability ridge.")
        self._inf_flat_b_prior_cb.toggled.connect(
            lambda checked: self._inf_sigma_b_spin.setEnabled(not checked))
        sb_row.addWidget(self._inf_flat_b_prior_cb)
        sb_row.addStretch(1)
        param_lay.addLayout(sb_row)
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("α  (D_e = D₀·(R/R_ref)^α):"))
        self._inf_alpha_spin = QDoubleSpinBox()
        self._inf_alpha_spin.setRange(-3.0, 3.0)
        self._inf_alpha_spin.setDecimals(2)
        self._inf_alpha_spin.setSingleStep(0.25)
        self._inf_alpha_spin.setValue(0.0)
        self._inf_alpha_spin.setToolTip(
            "Radius-scaling of per-edge D.\n"
            "  α = 0    → uniform D₀  (default)\n"
            "  α = +1   → D linear in R (bigger vessels more compliant)\n"
            "  α = −1   → D inverse in R (Murray-like, stiffer big vessels)\n"
            "R_ref = median measured-edge radius for the tile.\n"
            "D̂ is reported at R = R_ref so it's interpretable as the\n"
            "typical-vessel distensibility independent of α.")
        alpha_row.addWidget(self._inf_alpha_spin)
        alpha_row.addStretch(1)
        param_lay.addLayout(alpha_row)
        # ── BC correlation length ℓ (spatial smoothness on b̂_H₁) ──
        bcL_row = QHBoxLayout()
        bcL_row.addWidget(QLabel("BC correlation length ℓ (px):"))
        self._inf_bc_length_spin = QDoubleSpinBox()
        self._inf_bc_length_spin.setRange(0.0, 5000.0)
        self._inf_bc_length_spin.setDecimals(0)
        self._inf_bc_length_spin.setSingleStep(50.0)
        self._inf_bc_length_spin.setValue(0.0)
        self._inf_bc_length_spin.setToolTip(
            "Spatial smoothness prior on b_H₁ across boundary nodes.\n"
            "    C_ij = σ_b² · exp(−d_ij² / 2ℓ²)\n"
            "  ℓ = 0    → independent BCs (default; no smoothing)\n"
            "  ℓ ≈ tile size → adjacent BCs correlated; less BC freedom\n"
            "                  → harder for boundary forcing to absorb\n"
            "                    D-dependence (constrains b̂_H₁ → tighter D̂)\n"
            "Use this to test whether a tile's D̂ is driven by BC\n"
            "absorption flexibility — if D̂ shifts substantially as ℓ\n"
            "grows, the original tile was boundary-absorption-dominated.")
        bcL_row.addWidget(self._inf_bc_length_spin)
        bcL_row.addStretch(1)
        param_lay.addLayout(bcL_row)
        # ── Harmonic-mode selector ──
        # Use H₁ only (default), H₂ only, or both jointly.  In joint mode
        # b_H₁ and b_H₂ are independent (each gets its own Gaussian prior)
        # but D is shared, so H₂ adds an independent constraint on D.
        from PyQt5.QtWidgets import QComboBox
        harm_row = QHBoxLayout()
        harm_row.addWidget(QLabel("Harmonic:"))
        self._inf_harmonic_combo = QComboBox()
        self._inf_harmonic_combo.addItems(["H₁ only", "H₂ only", "H₁ + H₂ joint"])
        self._inf_harmonic_combo.setCurrentIndex(0)
        self._inf_harmonic_combo.setToolTip(
            "Which harmonic(s) the inference uses.\n"
            "  H₁ only      — default, single channel at f₀.\n"
            "  H₂ only      — single channel at 2f₀ (SNR_H₂ usually lower).\n"
            "  H₁ + H₂ joint — both channels, b̂_H₁ and b̂_H₂ each get\n"
            "                   their own prior, D shared.  Tightens CIs\n"
            "                   ~30-40% on tiles with adequate H₂ SNR;\n"
            "                   does NOT break the structural b–D ridge.")
        harm_row.addWidget(self._inf_harmonic_combo)
        harm_row.addStretch(1)
        param_lay.addLayout(harm_row)
        # ── SNR-based σ floor (opt-in hybrid noise model) ──
        from PyQt5.QtWidgets import QCheckBox
        snr_row = QHBoxLayout()
        self._inf_snr_floor_cb = QCheckBox("Use SNR-based σ (replace empirical)")
        self._inf_snr_floor_cb.setChecked(False)
        self._inf_snr_floor_cb.setToolTip(
            "Off (default): noise model is empirical "
            "σ²_j = â² + ŝ²·|Q_j|², with (â, ŝ) marginalised.\n"
            "On: REPLACE the empirical fit with the SNR-implied "
            "σ²_j = |Q_j|²·10^(−SNR_j/10) per edge, using the stored "
            "SNR (Q_H1_snr_db on canonical, harmonics[k].snr_db on v2).\n"
            "Trusts the stored SNR completely.  Tightens D̂ when "
            "the empirical fit is over-inflating ŝ to absorb model "
            "misfit (RMS|z| ≪ 1), but vulnerable to misleading "
            "stored SNR (e.g. embryo-motion-contaminated edges).")
        snr_row.addWidget(self._inf_snr_floor_cb)
        snr_row.addStretch(1)
        param_lay.addLayout(snr_row)
        info = QLabel(
            "<small><i>σ²_j = â² + ŝ²·|Q_H,j|² "
            "(additive + proportional, both marginalised).<br>"
            "γ (SNR weighting) → 0 from inferred-shape test; "
            "SNR field NOT used.<br>"
            "D₀ ~ LogNormal(ln 1.5×10⁻³, ln 10).  "
            "Anatomical A/V → BC by default.</i></small>")
        info.setWordWrap(True)
        param_lay.addWidget(info)
        layout.addWidget(param_group)

        # ── Actions ──
        act_row = QHBoxLayout()
        self._inf_run_btn = QPushButton("Run inference")
        self._inf_run_btn.clicked.connect(self._on_inference_run_clicked)
        act_row.addWidget(self._inf_run_btn)
        self._inf_diag_btn = QPushButton("D₀ posterior")
        self._inf_diag_btn.setEnabled(False)
        self._inf_diag_btn.clicked.connect(
            self._on_inference_diagnostic_clicked)
        act_row.addWidget(self._inf_diag_btn)
        layout.addLayout(act_row)
        compare_row = QHBoxLayout()
        self._inf_compare_btn = QPushButton(
            "Compare measured vs predicted Q(t) (selected edge)")
        self._inf_compare_btn.setEnabled(False)
        self._inf_compare_btn.setToolTip(
            "After running inference, click an edge in the viewer\n"
            "(Time-series tab shows the selected edge), then click\n"
            "this button.  Pops up a comparison of measured Q(t) and\n"
            "the inference's predicted Q(t) at MAP D₀ and MAP b̂.")
        self._inf_compare_btn.clicked.connect(
            self._on_inference_compare_clicked)
        compare_row.addWidget(self._inf_compare_btn)
        layout.addLayout(compare_row)

        # ── Results ──
        self._inf_results_box = QTextEdit()
        self._inf_results_box.setReadOnly(True)
        self._inf_results_box.setMaximumHeight(220)
        self._inf_results_box.setPlaceholderText(
            "Pick a tile and click 'Run inversion'.")
        layout.addWidget(self._inf_results_box)
        # Status
        self._inf_status_lbl = QLabel(
            "<small><i>Inference results live in memory; nothing is "
            "written back to the gpickle.</i></small>")
        self._inf_status_lbl.setWordWrap(True)
        layout.addWidget(self._inf_status_lbl)

        # Carve overlay layer slots (created lazily on first toggle).
        self._inference_carve_layer = None        # edge Shapes layer
        self._inference_carve_node_layer = None   # boundary-node Points layer
        self._inference_last_result = None
        # OF region overlay on the mosaic — curved orange ribbon showing
        # the actual integration region for the last OF run.  Persists
        # until the next OF run replaces it or "Clear OF region" is hit.
        self._of_region_layer = None
        # No initial carve render — the checkbox defaults to off so
        # the launch view stays uncluttered.  The first toggle-on
        # triggers `_refresh_inference_carve_overlay` lazily.

    # ── Adapter ──────────────────────────────────────────────────────
    # These three methods are the ONLY entry points into the inversion
    # algorithm.  Swap them out (or replace the whole region) to change
    # what "Run inversion" means.
    def _inference_available_tiles(self) -> List[int]:
        """Return a sorted list of tile IDs the inversion can run on.
        Default rule: any tile that appears as `tile_id` on at least
        one usable PIV measurement."""
        seen = set()
        for _, _, d in self.G.edges(data=True):
            for m in (d.get('measurements_piv') or []):
                # v4 migration: was m.get('tile_id')
                tid = MeasurementView(m).tile_id
                if tid is None:
                    continue
                try:
                    seen.add(int(tid))
                except (TypeError, ValueError):
                    pass
        return sorted(seen)

    def _inference_get_carve_topology(self, tile_id: int) -> dict:
        """Return carve geometry for the given tile as a dict with keys
        `edges` (list of (u,v)) and `boundary_nodes` (list of node ids).
        Wraps `scripts.inspect_tile.build_tile_problem`."""
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _scripts = _Path(self.graph_path).resolve().parent
            # Walk up to find the project root with a `scripts/` folder.
            for _ancestor in [_scripts] + list(_scripts.parents):
                if (_ancestor / 'scripts' / 'inspect_tile.py').exists():
                    _root = _ancestor
                    break
            else:
                from pertile.viewer import mosaic_readonly_app as _self_mod
                _root = _Path(_self_mod.__file__).resolve().parents[2]
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))
            if str(_root / 'scripts') not in _sys.path:
                _sys.path.insert(0, str(_root / 'scripts'))
            from inspect_tile import build_tile_problem
            prob = build_tile_problem(self.G, int(tile_id))
        except Exception as e:
            print(f"  inference: carve extraction failed ({type(e).__name__}: {e})")
            return {'edges': [], 'boundary_nodes': []}
        return {
            'edges': [tuple(e) for e in prob.get('edges_in', [])],
            'boundary_nodes': list(prob.get('boundary_nodes', [])),
        }

    def _inference_run(self, tile_id: int, sigma_b_Pa: float = 7.0,
                        alpha: float = 0.0, bc_length_px: float = 0.0,
                        harmonic_mode: str = 'H1',
                        cap_omega_mult: float = None,
                        use_snr_floor: bool = False):
        """Run the additive empirical-σ Bayesian inference on one tile.

        Model:
          Σ_j = |Q_H1,j|² · (r_j + s²)        r_j = 10^(−SNR_H1,j/10)
          D_e = D₀ · (R_e / R_ref)^α          R_ref = median measured-edge R
          b_H1 ~ N(0, σ_b² · I)
          D₀ ~ LogNormal(ln 1.5×10⁻³, ln 10)

        s² is marginalised analytically over a 21-pt log grid in
        [10⁻⁴, 10⁰].  b_H1 is marginalised analytically (Gaussian
        conjugate).  D₀ posterior is on a 41-pt log grid in
        [10⁻⁵, 10⁻¹].

        Returns a dict with:
          tile_id, alpha, sigma_b
          D_hat, lo, hi, width, tier
          s2_at_map, frac_err_pct
          log_p_D (1D D₀ posterior, normalised)
          D_GRID
          n_meas, snr_median, R_ref_um
          b_hat (MAP b at MAP D₀, MAP s²) — real-imag stacked, length 2·n_B
          Q_H1_pred (per-edge predicted H1 phasor at MAP, all edges)
          prob (the carve)
          R_e (per-edge radius array, in metres)
        """
        import math
        import networkx as nx
        from scipy.special import logsumexp
        from pertile.analysis.local_pressure_inference import (
            _edge_geometry, extract_tile_subgraph_spatial,
        )
        # Fixed physical constants — match the standalone sweep scripts.
        # Viscosity (Pa·s), cardiac f₀ (Hz, tile-anchored), pixel size (m/px).
        MU = 3.5e-3
        PX_SIZE_M = 1.7e-6
        # Per-tile f₀: prefer the tile-anchored value, fall back to the
        # graph-level default, then 2.5 Hz.
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        F0_HZ = float(tile_f0_map.get(int(tile_id)) or
                       self.G.graph.get('f0_hz') or 2.5)
        # ── Carve the tile's spatial neighbourhood ──
        try:
            edges_in, _, boundary_nodes, interior_nodes = (
                extract_tile_subgraph_spatial(
                    self.G, int(tile_id),
                    inset_frac=0.0,
                    restrict_to_tile_piv_nodes=False,
                    drop_dangling_boundaries=True))
        except Exception as e:
            return {'error': f'spatial carve failed: {e}'}
        edges_in = [tuple(e) for e in edges_in]
        boundary_nodes = list(boundary_nodes)
        interior_nodes = list(interior_nodes)
        # ── Promote anatomical A/V nodes to boundary ──
        # The spatial carve defines "boundary" by perimeter geometry, but
        # nodes with boundary_type == 'source' (A) or 'sink' (V) are
        # anatomical pressure anchors (heart-side inflow / venous return).
        # They MUST be treated as open boundaries; left as interior, the
        # inference would force the model to equilibrate at them, which
        # is physically wrong.  Promote any A/V node currently in the
        # carve interior to boundary so its pressure is a free parameter
        # of the inference, not a quantity the network has to balance.
        try:
            _av_in_interior = [
                n for n in interior_nodes
                if self.G.nodes[n].get('boundary_type') in ('source', 'sink')]
            if _av_in_interior:
                _av_set = set(_av_in_interior)
                boundary_nodes = boundary_nodes + _av_in_interior
                interior_nodes = [n for n in interior_nodes
                                    if n not in _av_set]
            _n_av_promoted = len(_av_in_interior)
        except Exception:
            _n_av_promoted = 0
        # ── Build the subgraph (carries radius/length attrs needed below) ──
        sub = nx.Graph()
        for n in set(boundary_nodes) | set(interior_nodes):
            nd = self.G.nodes[n]
            sub.add_node(n, x=float(nd.get('x', 0.0)),
                          y=float(nd.get('y', 0.0)))
        for (u, v) in edges_in:
            sub.add_edge(u, v, **dict(self.G.edges[u, v]))
        # ── Build per-edge Q observations from canonical TOP-LEVEL fields ──
        # Matches Steven's build_tile_problem convention: each edge gets
        # its canonical-best Q_DC / Q_H1 (promoted from the best-tile
        # measurement during graph build), regardless of whether the
        # source tile equals `tile_id`.  This is what /tmp/sweep_additive.py
        # used and is the basis of the headline D̂ numbers.  Per-tile
        # measurements (m['tile_id'] == tile_id specifically) are used
        # only by the comparison plot.
        # ── Per-edge observations: TILE-LOCAL matched pairs ──
        # Read Q_DC, Q_H1, Q_DC_SNR, Q_H1_SNR all from the SAME per-tile
        # entry in measurements_piv (matched on tile_id).  An edge that
        # was *not* measured on this tile drops out of the likelihood
        # entirely — no silent borrowing from the canonical top-level
        # field (which may have been promoted from a different tile and
        # would carry that tile's amplitude / phase calibration).
        #
        # This is the per-tile inference's defining commitment: every
        # (Q, σ) pair in y comes from THIS tile's own acquisition.
        # Bridges and edges only measured on other tiles still appear in
        # the forward-model topology (so the carve geometry is unchanged)
        # but contribute nothing to y.
        nL_PER_M3 = 1e12
        n_edges_in = len(edges_in)
        Q_DC_obs = np.full(n_edges_in, np.nan, dtype=float)
        Q_H1_obs = np.full(n_edges_in, np.nan, dtype=complex)
        Q_H2_obs = np.full(n_edges_in, np.nan, dtype=complex)
        valid_dc = np.zeros(n_edges_in, dtype=bool)
        valid_h1 = np.zeros(n_edges_in, dtype=bool)
        valid_h2 = np.zeros(n_edges_in, dtype=bool)
        snr_per_edge = np.full(n_edges_in, np.nan)
        snr_H2_per_edge = np.full(n_edges_in, np.nan)
        snr_DC_per_edge = np.full(n_edges_in, np.nan)
        # Per-edge time-series length N — needed to convert the
        # canonical SNR_H1_db (a time-series scale derived from the
        # band-limited residual MS_BL) into a phasor estimation variance.
        # See σ²_meas below for the relationship.
        N_per_edge = np.full(n_edges_in, np.nan)
        for i, (u, v) in enumerate(edges_in):
            ed = self.G.edges[u, v]
            # Find THIS tile's measurement on this edge.  None → edge
            # is unmeasured on this tile (a bridge or only-measured-
            # elsewhere); skip from the likelihood.
            m_tile = next((mm for mm in (ed.get('measurements_piv') or [])
                            if MeasurementView(mm).tile_id == int(tile_id)), None)
            if m_tile is None:
                continue
            mv_tile = MeasurementView(m_tile)
            # Sign convention: prefer the per-tile measurement's own
            # flow direction.  v2 edges have flow_from/flow_to set ONLY
            # on per-tile entries (top-level is None until promotion
            # writes flow_from_piv / flow_to_piv).  Canonical edges
            # historically had them at the top level too — fall back
            # there if the per-tile dict doesn't carry them.
            # v4 migration: was m_tile.get('flow_from') / .get('flow_to')
            ff = mv_tile.flow_from
            ft = mv_tile.flow_to
            if ff is None or ft is None:
                # v4 migration: ed.get('flow_from_piv') is dropped in v4.
                # The v4-clean fallback source for flow_from/_to on the edge
                # is MeasurementView.best(ed).flow_from/_to; keep the legacy
                # ed.get('flow_from_piv') tail as a v3-graph safety net.
                mv_best = MeasurementView.best(ed)
                if mv_best is not None:
                    ff = ed.get('flow_from', mv_best.flow_from)
                    ft = ed.get('flow_to',   mv_best.flow_to)
                else:
                    ff = ed.get('flow_from', ed.get('flow_from_piv'))
                    ft = ed.get('flow_to',   ed.get('flow_to_piv'))
            if ff is None or ft is None:
                continue
            sign = 1.0 if (ff == u and ft == v) else -1.0
            # DC channel.  Canonical: 'Q_DC'.  v2: 'mean_Q'.
            mq = m_tile.get('Q_DC')
            if mq is None or not np.isfinite(mq):
                # v4 migration: was m_tile.get('mean_Q')
                mq = mv_tile.harm.mean_Q
            if mq is not None and np.isfinite(mq):
                Q_DC_obs[i] = float(mq) * sign / nL_PER_M3
                valid_dc[i] = True
            # H1 / H2 channels.  Canonical: 'Q_H1_amp','Q_H1_phi','Q_H2_amp',...
            #                    v2:        harmonics[0]['amp']/['phi'], harmonics[1]...
            amp = m_tile.get('Q_H1_amp'); phase = m_tile.get('Q_H1_phi')
            amp2 = m_tile.get('Q_H2_amp'); phase2 = m_tile.get('Q_H2_phi')
            # v4 migration: was m_tile.get('harmonics')
            _harms = mv_tile.harm.harmonics or []
            if (amp is None or not np.isfinite(amp)) and len(_harms) >= 1:
                amp = _harms[0].get('amp')
                phase = _harms[0].get('phi')
            if (amp2 is None or not np.isfinite(amp2)) and len(_harms) >= 2:
                amp2 = _harms[1].get('amp')
                phase2 = _harms[1].get('phi')
            if (amp is not None and phase is not None
                    and np.isfinite(amp) and np.isfinite(phase)):
                Q_H1_obs[i] = (float(amp) * np.exp(1j * float(phase))
                                * sign / nL_PER_M3)
                valid_h1[i] = True
            if (amp2 is not None and phase2 is not None
                    and np.isfinite(amp2) and np.isfinite(phase2)):
                Q_H2_obs[i] = (float(amp2) * np.exp(1j * float(phase2))
                                * sign / nL_PER_M3)
                valid_h2[i] = True
            # SNRs (per-edge measurement noise via the additive model)
            # Try canonical schema first; fall back to v2 schema (harmonics[k].snr_db).
            snr_H1 = m_tile.get('Q_H1_snr_db', np.nan)
            snr_H2 = m_tile.get('Q_H2_snr_db', np.nan)
            snr_DC = m_tile.get('Q_DC_snr_db', np.nan)
            # v4 migration: was m_tile.get('harmonics')
            _harms = mv_tile.harm.harmonics or []
            if (snr_H1 is None or not np.isfinite(snr_H1)) and len(_harms) >= 1:
                snr_H1 = _harms[0].get('snr_db', np.nan)
            if (snr_H2 is None or not np.isfinite(snr_H2)) and len(_harms) >= 2:
                snr_H2 = _harms[1].get('snr_db', np.nan)
            snr_per_edge[i] = snr_H1
            snr_H2_per_edge[i] = snr_H2
            snr_DC_per_edge[i] = snr_DC
            Q_t = mv_tile.Q_t  # v4 migration: was m_tile.get('Q_t')
            if Q_t is not None:
                try:
                    N_per_edge[i] = len(Q_t)
                except TypeError:
                    pass
        prob = dict(
            sub=sub, edges_in=edges_in,
            boundary_nodes=boundary_nodes, interior_nodes=interior_nodes,
            Q_DC_obs=Q_DC_obs, Q_H1_obs=Q_H1_obs, Q_H2_obs=Q_H2_obs,
            valid_dc=valid_dc, valid_h1=valid_h1, valid_h2=valid_h2,
        )
        # Choose validity mask based on the harmonic mode.
        # H1-only and H1+H2 both require valid_h1; H2-only requires valid_h2.
        if harmonic_mode == 'H2':
            valid = prob['valid_dc'] & prob['valid_h2']
        elif harmonic_mode == 'H1+H2':
            # Joint mode requires both channels measured on the same edges.
            valid = prob['valid_dc'] & prob['valid_h1'] & prob['valid_h2']
        else:
            valid = prob['valid_dc'] & prob['valid_h1']
        n_v = int(valid.sum())
        if n_v < 10:
            return {'error': f'Tile {tile_id} has only {n_v} valid edges (< 10) under harmonic mode {harmonic_mode}'}
        # Per-edge R, L in meters
        edges_in = prob['edges_in']
        R_e = np.zeros(len(edges_in)); L_e = np.zeros(len(edges_in))
        for k, (u, v) in enumerate(edges_in):
            R_m, L_m = _edge_geometry(self.G.edges[u, v], PX_SIZE_M)
            R_e[k] = max(R_m, 1e-9); L_e[k] = max(L_m, 1e-9)
        R_ref = float(np.median(R_e[valid]))
        R_factor = (R_e / R_ref) ** float(alpha)
        # ── Per-edge measurement noise (phasor variance, white-residual) ──
        # Honest additive-plus-proportional noise model:
        #
        #     σ²_j  =  a²  +  s² · |Q_j|²
        #
        # a (m³/s)         — measurement-noise floor, signal-independent.
        #                    Keeps low-|Q| edges from running away under
        #                    pure proportional weighting (the σ²→0 problem
        #                    with σ²∝|Q|² alone).
        # s² (dimensionless) — proportional misfit (ridge + KCL deficit,
        #                    both scale with |Q|).
        #
        # Both inferred / marginalised — we no longer assert SNR-based
        # weighting (r_j) or the (2/N) phasor factor; the previous test
        # (learned_noise_model_full.py) showed γ → 0 on the SNR axis with
        # posterior mass 1.000 on every tile.  The canonical SNR field
        # carries no per-edge precision information, so dropping r_j is
        # the inferred preference, not an assumption.
        #
        # The crossover  |Q|* = a/s  is the flow magnitude below which
        # measurement noise dominates and above which model misfit
        # dominates — a real, interpretable, defensible number.
        # ── Choose data channel(s) for the likelihood ──
        # primary_channel = the channel we marginalise b for and use to
        # build the "main" T(D) etc.  In joint mode, H₁ is primary and
        # H₂ is added on top.
        primary_channel = 'H2' if harmonic_mode == 'H2' else 'H1'
        if primary_channel == 'H1':
            Q_h_primary = prob['Q_H1_obs'][valid]
        else:
            Q_h_primary = prob['Q_H2_obs'][valid]
        Q_abs2 = np.abs(Q_h_primary) ** 2
        Q_abs = np.sqrt(Q_abs2)
        y_H1 = np.concatenate([Q_h_primary.real, Q_h_primary.imag])
        valid_idx = np.where(valid)[0]
        Q_abs2_stack = np.concatenate([0.5 * Q_abs2, 0.5 * Q_abs2])
        # H₂ data on the same edges, used only when harmonic_mode == 'H1+H2'.
        if harmonic_mode == 'H1+H2':
            Q_h2_data = prob['Q_H2_obs'][valid]
            Q_abs2_H2 = np.abs(Q_h2_data) ** 2
            y_H2 = np.concatenate([Q_h2_data.real, Q_h2_data.imag])
        else:
            Q_h2_data = None; Q_abs2_H2 = None; y_H2 = None
        # ── Boundary forcing prior ──
        # Default (bc_length_px = 0): b_H₁,n ~ N(0, σ_b²) independently.
        #                              P₀ = σ_b² · I  (diagonal).
        # Spatial-smoothness mode (bc_length_px > 0): Gaussian-process
        # prior — adjacent BC nodes are correlated, so the inference
        # cannot place arbitrarily different b̂ values at nearby BCs to
        # absorb D-dependence.
        #
        #   C_ij = σ_b² · exp(−d_ij² / 2ℓ²)         (squared-exponential)
        #
        # with d_ij = Euclidean distance between BC nodes i, j (mosaic
        # px), and ℓ = bc_length_px the user-set correlation length.
        # A small ridge is added to guarantee numerical positive
        # definiteness when many BC nodes are spatially close.
        n_B = len(prob['boundary_nodes'])
        if bc_length_px and bc_length_px > 0:
            bc_xy = np.zeros((n_B, 2))
            for i_b, n in enumerate(prob['boundary_nodes']):
                nd = self.G.nodes[n]
                bc_xy[i_b] = (float(nd.get('x', 0.0)),
                               float(nd.get('y', 0.0)))
            d_sq = ((bc_xy[:, None, :] - bc_xy[None, :, :]) ** 2).sum(axis=2)
            C_bc = np.exp(-d_sq / (2.0 * float(bc_length_px) ** 2))
            C_bc = sigma_b_Pa ** 2 * C_bc
            C_bc += 1e-6 * sigma_b_Pa ** 2 * np.eye(n_B)
            # Real-form block-diagonal: Re and Im components are
            # independent under the complex circularly-symmetric Gaussian.
            zero_blk = np.zeros((n_B, n_B))
            P0 = 0.5 * np.block([[C_bc, zero_blk],
                                   [zero_blk, C_bc]])
        else:
            P0 = np.diag(np.concatenate([
                np.full(n_B, 0.5 * sigma_b_Pa ** 2),
                np.full(n_B, 0.5 * sigma_b_Pa ** 2),
            ]))
        # ── D₀ grid + physical-validity cap ──
        # The linear-RC forward model is only valid where the per-edge
        # compliance number  C_e = 8μω·L²·D / R²  ≪ 1.  Beyond C_e ~ 1,
        # the wall-storage current is no longer a perturbation, the
        # distributed/lumped agreement breaks down at the O(1) level,
        # and (more importantly) the linear wall law ΔA/A = D·ΔP is
        # itself questionable.
        #
        # Cap the D₀ grid at the value where  max_e(C_e) = C_E_CAP.
        # This is a tile-specific bound determined by edge geometry.
        # Without the cap, degenerate tiles can extrapolate their
        # broad posterior into the high-D tail (C_e ≳ 1) and
        # manufacture a "finite-compliance preference" from a regime
        # where the model itself is not trustworthy.  With the cap,
        # the inference is restricted to the physically defensible
        # window and degenerate tiles correctly read as unidentified
        # rather than borderline-strong.
        C_E_CAP = 0.3                              # physically-defensible upper bound
        # Per-edge geometry factor K_e = 8μω·L²/R² (so C_e = D·K_e).
        # Use ω at the channel that drives the largest κ²:
        #   H₁-only  → ω = 2π·f₀
        #   H₂-only  → ω = 4π·f₀                (κ² doubled per edge)
        #   joint    → use H₂'s ω to keep both channels safely inside C_e cap
        # Override: cap_omega_mult lets the caller force a *shared* D-grid
        # across H1-only and H2-only runs (set to 2.0 in both calls →
        # both grids anchored on the tighter H2 cap).  Without it, falls
        # back to the existing per-mode behavior.
        if cap_omega_mult is not None:
            omega_for_cap = float(cap_omega_mult) * 2.0 * np.pi * F0_HZ
        else:
            omega_for_cap = (4.0 * np.pi * F0_HZ
                              if harmonic_mode in ('H2', 'H1+H2')
                              else 2.0 * np.pi * F0_HZ)
        K_e = np.zeros(len(edges_in))
        valid_geom = (R_e > 0) & (L_e > 0)
        K_e[valid_geom] = (8.0 * MU * omega_for_cap
                            * L_e[valid_geom] ** 2
                            / R_e[valid_geom] ** 2)
        # Cap based on measured edges (α-scaling tweaks per-edge D, but
        # for the cap we use D₀ at R_ref → D_e = D₀; the α correction
        # is at most a factor R_max/R_ref which is order 1).
        K_meas_max = float(K_e[valid].max() if valid.any() else 0.0)
        D_cap = (C_E_CAP / K_meas_max) if K_meas_max > 0 else 1.0
        LOG_D_CAP = np.log10(max(D_cap, 1e-5))
        LOG_D_GRID = np.arange(-5.0, min(-1.0, LOG_D_CAP) + 0.001, 0.10)
        D_GRID = 10.0 ** LOG_D_GRID
        # ── Noise-model grids (additive a + proportional s²) ──
        # a (m³/s) is the measurement-noise floor; ŝ² is the proportional
        # misfit (dimensionless).  Grids trimmed to the support actually
        # populated by the 53/84-tile sweep posteriors:
        #   â ∈ [10⁻⁵, 10⁻² nL/s]  →  log₁₀(a/[m³/s]) ∈ [-17, -14]
        #   ŝ² ∈ [0.04, 4]          →  log₁₀(s²) ∈ [-1.4, 0.6]
        # Wide enough to not rail; small enough to keep per-tile inference
        # well under 10 s (was ~30 s when the grids spanned [-18, -13] ×
        # [-4, 0.4]).  Extend if you find a tile railing the boundary.
        LOG_A_GRID = np.arange(-17.0, -14.0 + 0.001, 0.30)   # 11 pts
        A_GRID = 10.0 ** LOG_A_GRID
        LOG_S2_GRID = np.arange(-1.4, 0.6 + 0.001, 0.30)     # 7 pts
        S2_GRID = 10.0 ** LOG_S2_GRID
        MU_D, TAU_D = np.log(1.5e-3), np.log(10.0)
        log_prior_D = -0.5 * ((np.log(D_GRID) - MU_D) / TAU_D) ** 2
        # ── O(C_e) distributed two-port forward model ──
        # The full distributed two-port is
        #   Y_self  = G · κ · coth(κ)
        #   Y_trans = G · κ · csch(κ)
        # with κ² = i·ω·D·8μ·L²/R² = i·C_e (Desktop's dimensionless number).
        # In the small-C_e regime (C_e ≲ 0.3, enforced by the D-grid cap
        # above) the hyperbolics are indistinguishable from their two-term
        # polynomial expansion to O(C_e²) ~ 0.003%:
        #
        #   ξ·coth(ξ) = 1 + ξ²/3 − ξ⁴/45 + …    →   Y_self  ≈ G · (1 + κ²/3)
        #   ξ·csch(ξ) = 1 − ξ²/6 + 7ξ⁴/360 − … →   Y_trans ≈ G · (1 − κ²/6)
        #
        # We use the polynomial form because (a) the regime of validity is
        # explicit in the code (anything past O(C_e²) is dropped, and the
        # D_cap keeps us there), (b) no small-|κ| special case is needed,
        # (c) it is the "right" linearization — the lumped-π misdistributes
        # the same total shunt as Y_self = G(1 + κ²/2), Y_trans = G, which
        # has the wrong O(C_e) coefficients per edge (a 0.6% error growing
        # linearly with D — exactly the bug that inflated high-D tails).
        #
        # Nodal admittance:    Y[u,u] += Y_self,e  ;  Y[u,v] -= Y_trans,e
        # Edge flow readout:   Q_e = Y_self,e · p_u − Y_trans,e · p_v
        # (manuscript convention — flow at u's end, includes wall-storage current)
        interior_nodes = prob['interior_nodes']
        boundary_nodes = prob['boundary_nodes']
        n_int = len(interior_nodes)
        interior_idx = {n: i for i, n in enumerate(interior_nodes)}
        boundary_idx = {n: i for i, n in enumerate(boundary_nodes)}
        edge_G = np.zeros(len(edges_in))
        edge_kappa_factor = np.zeros(len(edges_in), dtype=complex)
        # κ²_e / D_e = i · ω · 8μ · L² / R²       (stored once per edge)
        for k, (u, v) in enumerate(edges_in):
            if R_e[k] <= 0 or L_e[k] <= 0: continue
            edge_G[k] = float(np.pi * R_e[k] ** 4 / (8.0 * MU * L_e[k]))
            edge_kappa_factor[k] = (1j * 2.0 * np.pi * F0_HZ
                                     * 8.0 * MU
                                     * (L_e[k] ** 2) / (R_e[k] ** 2))
        # ω at the "primary" channel (drives the main T(D) build and the
        # κ²/C_e cap).  H₂-only mode runs at ω = 2·ω_H₁ ; joint mode uses
        # ω_H₁ as primary and adds a 2·ω_H₁ branch below.
        omega_h1 = 2.0 * np.pi * F0_HZ
        omega_h2 = 2.0 * omega_h1
        omega_primary = omega_h2 if primary_channel == 'H2' else omega_h1
        # κ_factor at the primary channel (used in the D-cap and in
        # _edge_admittances).  H₂ has 2× the storage current per edge.
        kappa_factor_primary = edge_kappa_factor * (
            2.0 if primary_channel == 'H2' else 1.0)

        def _edge_admittances(D_e_array, kappa_factor=None):
            """O(C_e) linearization of the distributed two-port — the
            polynomial form valid for C_e ≪ 1 (enforced by the D-grid
            cap above).  Accurate to O(C_e²) ~ 0.003% in our regime.

                Y_self,e  = G_e · (1 + κ²_e / 3)
                Y_trans,e = G_e · (1 − κ²_e / 6)

            with κ² = i·ω·D·8μ·L²/R² (= i·C_e) supplied per edge.
            Pass a different `kappa_factor` to evaluate at a different ω
            (e.g. 2·edge_kappa_factor for the H₂ channel)."""
            kf = kappa_factor if kappa_factor is not None else kappa_factor_primary
            kappa_sq = kf * D_e_array
            Y_self = edge_G * (1.0 + kappa_sq / 3.0)
            Y_trans = edge_G * (1.0 - kappa_sq / 6.0)
            # Force the resistive limit on degenerate / unused edges
            Y_self = np.where(edge_G > 0, Y_self, 0.0)
            Y_trans = np.where(edge_G > 0, Y_trans, 0.0)
            return Y_self, Y_trans

        def _T_H1_at_D(D0):
            """Build the distributed-two-port H1 admittance system at
            scalar D₀, with per-edge D_e = D₀·R_factor.  Returns the
            full (n_edges, n_B) complex transfer matrix where
              T[e, jb] = Y_self,e · p_u^{(jb)} − Y_trans,e · p_v^{(jb)}
            and p_u^{(jb)}, p_v^{(jb)} are the interior pressures driven
            by the jb-th unit boundary pressure."""
            D_e_arr = float(D0) * R_factor
            Y_self_arr, Y_trans_arr = _edge_admittances(D_e_arr)
            L_mat = np.zeros((n_int, n_int), dtype=complex)
            B_mat = np.zeros((n_int, n_B), dtype=complex)
            for k, (u, v) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                Y_s = Y_self_arr[k]; Y_t = Y_trans_arr[k]
                u_int = u in interior_idx; v_int = v in interior_idx
                if u_int and v_int:
                    iu, iv = interior_idx[u], interior_idx[v]
                    L_mat[iu, iu] += Y_s; L_mat[iv, iv] += Y_s
                    L_mat[iu, iv] -= Y_t; L_mat[iv, iu] -= Y_t
                elif u_int and not v_int:
                    iu, jb = interior_idx[u], boundary_idx[v]
                    L_mat[iu, iu] += Y_s; B_mat[iu, jb] -= Y_t
                elif v_int and not u_int:
                    iv, jb = interior_idx[v], boundary_idx[u]
                    L_mat[iv, iv] += Y_s; B_mat[iv, jb] -= Y_t
            # Solve interior Laplacian; lstsq for rank-deficient safety
            diag_med = np.median(np.abs(np.diag(L_mat))) if n_int else 1.0
            scale = 1.0 / diag_med if diag_med > 0 else 1.0
            P_int = np.linalg.lstsq(L_mat * scale, -B_mat * scale,
                                     rcond=None)[0]
            # Edge flow readout: Q_e = Y_self·p_u − Y_trans·p_v
            T = np.zeros((len(edges_in), n_B), dtype=complex)
            for k, (u, v) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                Y_s = Y_self_arr[k]; Y_t = Y_trans_arr[k]
                for jb in range(n_B):
                    pu = (P_int[interior_idx[u], jb] if u in interior_idx
                          else (1.0 if boundary_idx[u] == jb else 0.0))
                    pv = (P_int[interior_idx[v], jb] if v in interior_idx
                          else (1.0 if boundary_idx[v] == jb else 0.0))
                    T[k, jb] = Y_s * pu - Y_t * pv
            return T

        # ── Grid evaluation:  σ²_j = a² + s²·|Q_j|²  ──
        # Joint (a, s²) marginalisation on each D grid point.
        # In H₁+H₂ joint mode, the H₂ likelihood (with its own b̂_H₂
        # marginalised under the same Gaussian prior) is ADDED to the
        # H₁ log-likelihood at each (D, a, s²) grid point.  This is
        # valid because b̂_H₁ and b̂_H₂ are independent in the prior.
        #
        # ── Optional SNR floor (use_snr_floor=True) ──
        # The empirical (a, s²) fit can absorb model misfit as "noise"
        # and confidently down-weight edges that are actually fine.
        # The SNR floor prevents that: σ²_j ≥ |Q_j|² · 10^(-SNR_j/10),
        # which is the variance implied by the stored per-edge SNR.
        # We apply σ²_j = max(σ²_empirical, σ²_SNR) inside the loop.
        # Edges with no usable SNR get σ²_SNR = 0 → floor inactive.
        if use_snr_floor:
            snr_use_H1 = snr_per_edge[valid]
            sigma2_snr_H1 = np.where(np.isfinite(snr_use_H1),
                                       Q_abs2 * 10.0 ** (-snr_use_H1 / 10.0),
                                       0.0)
            if harmonic_mode == 'H1+H2' and Q_abs2_H2 is not None:
                snr_use_H2 = snr_H2_per_edge[valid]
                sigma2_snr_H2 = np.where(np.isfinite(snr_use_H2),
                                           Q_abs2_H2 * 10.0 ** (-snr_use_H2 / 10.0),
                                           0.0)
            elif primary_channel == 'H2':
                snr_use_H2 = snr_H2_per_edge[valid]
                sigma2_snr_H1 = np.where(np.isfinite(snr_use_H2),
                                           Q_abs2 * 10.0 ** (-snr_use_H2 / 10.0),
                                           0.0)
                sigma2_snr_H2 = None
            else:
                sigma2_snr_H2 = None
        else:
            sigma2_snr_H1 = None
            sigma2_snr_H2 = None

        kappa_factor_H2 = edge_kappa_factor * 2.0
        log_L_H1 = np.full((D_GRID.size, A_GRID.size, S2_GRID.size), -np.inf)
        log_L_H2 = np.full((D_GRID.size, A_GRID.size, S2_GRID.size), -np.inf)
        T_cache = {}
        T_H2_cache = {}
        for i, D0 in enumerate(D_GRID):
            try:
                T_full = _T_H1_at_D(float(D0))
                T_cache[i] = T_full
                T_sel = T_full[valid]
                n_r = T_sel.shape[0]
                T_R = np.zeros((2 * n_r, 2 * n_B))
                T_R[:n_r, :n_B] = T_sel.real; T_R[:n_r, n_B:] = -T_sel.imag
                T_R[n_r:, :n_B] = T_sel.imag; T_R[n_r:, n_B:] = T_sel.real
                # H₂ branch (joint mode only): build a separate T_R_H2 by
                # temporarily swapping the κ factor inside the closure.
                T_R_H2 = None
                if harmonic_mode == 'H1+H2':
                    # Re-evaluate the same forward-model build but at 2ω.
                    D_e_arr_H2 = float(D0) * R_factor
                    ks_H2 = kappa_factor_H2 * D_e_arr_H2
                    Y_self_H2 = edge_G * (1.0 + ks_H2 / 3.0)
                    Y_trans_H2 = edge_G * (1.0 - ks_H2 / 6.0)
                    Y_self_H2 = np.where(edge_G > 0, Y_self_H2, 0.0)
                    Y_trans_H2 = np.where(edge_G > 0, Y_trans_H2, 0.0)
                    L_mat_H2 = np.zeros((n_int, n_int), dtype=complex)
                    B_mat_H2 = np.zeros((n_int, n_B), dtype=complex)
                    for k, (u_, v_) in enumerate(edges_in):
                        if edge_G[k] == 0: continue
                        Y_s = Y_self_H2[k]; Y_t = Y_trans_H2[k]
                        u_int = u_ in interior_idx; v_int = v_ in interior_idx
                        if u_int and v_int:
                            iu, iv = interior_idx[u_], interior_idx[v_]
                            L_mat_H2[iu, iu] += Y_s; L_mat_H2[iv, iv] += Y_s
                            L_mat_H2[iu, iv] -= Y_t; L_mat_H2[iv, iu] -= Y_t
                        elif u_int and not v_int:
                            iu, jb = interior_idx[u_], boundary_idx[v_]
                            L_mat_H2[iu, iu] += Y_s; B_mat_H2[iu, jb] -= Y_t
                        elif v_int and not u_int:
                            iv, jb = interior_idx[v_], boundary_idx[u_]
                            L_mat_H2[iv, iv] += Y_s; B_mat_H2[iv, jb] -= Y_t
                    diag_med = (np.median(np.abs(np.diag(L_mat_H2)))
                                  if n_int else 1.0)
                    scale = 1.0 / diag_med if diag_med > 0 else 1.0
                    P_int_H2 = np.linalg.lstsq(L_mat_H2 * scale,
                                                 -B_mat_H2 * scale, rcond=None)[0]
                    T_H2 = np.zeros((len(edges_in), n_B), dtype=complex)
                    for k, (u_, v_) in enumerate(edges_in):
                        if edge_G[k] == 0: continue
                        Y_s = Y_self_H2[k]; Y_t = Y_trans_H2[k]
                        for jb in range(n_B):
                            pu = (P_int_H2[interior_idx[u_], jb] if u_ in interior_idx
                                  else (1.0 if boundary_idx[u_] == jb else 0.0))
                            pv = (P_int_H2[interior_idx[v_], jb] if v_ in interior_idx
                                  else (1.0 if boundary_idx[v_] == jb else 0.0))
                            T_H2[k, jb] = Y_s * pu - Y_t * pv
                    T_H2_cache[i] = T_H2
                    T_H2_sel = T_H2[valid]
                    T_R_H2 = np.zeros((2 * n_r, 2 * n_B))
                    T_R_H2[:n_r, :n_B] = T_H2_sel.real
                    T_R_H2[:n_r, n_B:] = -T_H2_sel.imag
                    T_R_H2[n_r:, :n_B] = T_H2_sel.imag
                    T_R_H2[n_r:, n_B:] = T_H2_sel.real
                # Compute the H₁ likelihood (always needed) and the H₂
                # likelihood (only in joint mode) on the SAME (a, s²) grid.
                # The two channels are INDEPENDENT in (a, s²) — each
                # gets its own per-channel noise model.  The factorisation
                # over independent priors lets us logsumexp each channel's
                # likelihood separately and then add the two log evidences.
                for ja, a_ in enumerate(A_GRID):
                    a2 = float(a_) ** 2
                    for js, s2 in enumerate(S2_GRID):
                        # H₁ branch.  With SNR-noise mode on, replace
                        # the empirical (â, ŝ) variance entirely with
                        # the SNR-implied σ²_j = |Q_j|²·10^(−SNR_j/10).
                        # The (a, s²) loop still runs but log L now
                        # doesn't depend on (a, s²) — they become
                        # marginalised over a constant likelihood, so
                        # they fall out of the posterior on D.
                        if sigma2_snr_H1 is not None:
                            sigma2 = sigma2_snr_H1.copy()
                            # Replace any zero / non-finite σ² (edges
                            # with missing SNR) with the empirical
                            # value so they still contribute.
                            bad = (sigma2 <= 0) | ~np.isfinite(sigma2)
                            if bad.any():
                                sigma2_emp = a2 + float(s2) * Q_abs2
                                sigma2[bad] = sigma2_emp[bad]
                        else:
                            sigma2 = a2 + float(s2) * Q_abs2
                        Sigma_diag = np.concatenate(
                            [0.5 * sigma2, 0.5 * sigma2])
                        C = np.diag(Sigma_diag) + T_R @ P0 @ T_R.T
                        C += 1e-10 * np.mean(np.diag(C)) * np.eye(C.shape[0])
                        try:
                            L = np.linalg.cholesky(0.5 * (C + C.T))
                            z = np.linalg.solve(L, y_H1)
                            log_L_H1[i, ja, js] = (-np.sum(np.log(np.diag(L)))
                                                    - 0.5 * (z @ z))
                        except np.linalg.LinAlgError:
                            pass
                        # H₂ branch — only in joint mode.  Independent
                        # noise parameters: each channel sees the same
                        # (a, s²) grid here, but they MARGINALISE
                        # independently below.
                        if harmonic_mode == 'H1+H2' and T_R_H2 is not None:
                            if sigma2_snr_H2 is not None:
                                sigma2_H2 = sigma2_snr_H2.copy()
                                bad = (sigma2_H2 <= 0) | ~np.isfinite(sigma2_H2)
                                if bad.any():
                                    s2e = a2 + float(s2) * Q_abs2_H2
                                    sigma2_H2[bad] = s2e[bad]
                            else:
                                sigma2_H2 = a2 + float(s2) * Q_abs2_H2
                            Sigma_diag_H2 = np.concatenate(
                                [0.5 * sigma2_H2, 0.5 * sigma2_H2])
                            C_H2 = np.diag(Sigma_diag_H2) + T_R_H2 @ P0 @ T_R_H2.T
                            C_H2 += 1e-10 * np.mean(np.diag(C_H2)) * np.eye(C_H2.shape[0])
                            try:
                                L_H2_ch = np.linalg.cholesky(0.5 * (C_H2 + C_H2.T))
                                z_H2 = np.linalg.solve(L_H2_ch, y_H2)
                                log_L_H2[i, ja, js] = (-np.sum(np.log(np.diag(L_H2_ch)))
                                                       - 0.5 * (z_H2 @ z_H2))
                            except np.linalg.LinAlgError:
                                pass
            except Exception:
                pass
        if not np.isfinite(log_L_H1).any():
            return {'error': 'Likelihood grid is all -inf — inference failed'}
        # Independent (a, s²) marginalisation per channel.  log p(D) is
        # the sum of per-channel log marginal likelihoods plus the D prior.
        log_p_D = logsumexp(log_L_H1, axis=(1, 2))
        if harmonic_mode == 'H1+H2' and np.isfinite(log_L_H2).any():
            log_p_D = log_p_D + logsumexp(log_L_H2, axis=(1, 2))
        log_p_D = log_p_D + log_prior_D
        log_p_D = log_p_D - log_p_D.max()  # normalise for stability
        # `log_L` is used by downstream MAP-(a, s²) code; in joint mode
        # we summarise as H1 + H2 so the heatmap still corresponds to the
        # full inference, but per-channel marginalisation has already
        # happened in log_p_D above.
        if harmonic_mode == 'H1+H2':
            log_L = log_L_H1 + log_L_H2
        else:
            log_L = log_L_H1
        j_map = int(np.argmax(log_p_D))
        D_map = float(D_GRID[j_map])
        # ── Two interval reports ──
        # (a) 3-nat posterior SUPPORT region: log p ≥ log p_max − 3.
        #     NOT a credible interval — its enclosed posterior mass is
        #     not 0.95 in general (Desktop critique).  Kept as a visual
        #     diagnostic because it cleanly separates SHARP / MOD / LOOSE
        #     tiers across the existing sweep results.
        band = log_p_D >= log_p_D[j_map] - 3.0
        lo = float(D_GRID[band][0]) if band.any() else float('nan')
        hi = float(D_GRID[band][-1]) if band.any() else float('nan')
        width = math.log10(hi / lo) if lo and lo > 0 else float('inf')
        # (b) Proper 90% (and 95%) credible intervals from the actual
        #     posterior mass on the log-D grid.  log_p_D values are
        #     evaluated as log p(x_D | y) on a uniformly-spaced log grid,
        #     so the bin widths are constant and posterior mass per bin
        #     is proportional to exp(log_p_D − max).
        post = np.exp(log_p_D - log_p_D.max())
        post = post / post.sum()
        # Highest-Posterior-Density intervals: sort bins by descending
        # mass, accumulate until the target probability is met, then
        # report the lo/hi grid endpoints of the included set.  For a
        # unimodal posterior this matches the standard HPD construction.
        def _hpd(p_target):
            order = np.argsort(-post)          # descending
            cum = np.cumsum(post[order])
            n_inc = int(np.searchsorted(cum, p_target) + 1)
            n_inc = min(n_inc, post.size)
            keep = np.sort(order[:n_inc])
            return float(D_GRID[keep[0]]), float(D_GRID[keep[-1]])
        lo_90, hi_90 = _hpd(0.90)
        lo_95, hi_95 = _hpd(0.95)
        width_90 = math.log10(hi_90 / lo_90) if lo_90 > 0 else float('inf')
        width_95 = math.log10(hi_95 / lo_95) if lo_95 > 0 else float('inf')
        interior = (j_map > 0 and j_map < D_GRID.size - 1)
        tier = ('SHARP' if (interior and width <= 0.5)
                else 'MODERATE' if (interior and width <= 1.0)
                else 'LOOSE' if (interior and width <= 2.0)
                else 'UN-LOC')
        # ── MAP (a, s²) per channel at the MAP D ──
        # The H₁ MAP is from log_L_H1, the H₂ MAP from log_L_H2 (joint
        # mode only).  Each channel has its OWN measurement-noise floor
        # â and proportional misfit ŝ — they don't share parameters.
        LL_H1_atD = log_L_H1[j_map]
        j_a_H1, j_s_H1 = np.unravel_index(int(np.argmax(LL_H1_atD)),
                                            LL_H1_atD.shape)
        a_map_H1 = float(A_GRID[j_a_H1])
        s2_map_H1 = float(S2_GRID[j_s_H1])
        s_map_H1 = float(np.sqrt(s2_map_H1))
        Q_crossover_H1 = a_map_H1 / max(s_map_H1, 1e-30)
        # Keep the legacy fields wired to the H₁ MAP for back-compat with
        # the diagnostic popup, compare popup, and downstream consumers.
        a_map = a_map_H1
        s2_map = s2_map_H1
        s_map = s_map_H1
        Q_crossover = Q_crossover_H1
        # H₂ MAP (joint mode)
        if harmonic_mode == 'H1+H2' and np.isfinite(log_L_H2).any():
            LL_H2_atD = log_L_H2[j_map]
            j_a_H2, j_s_H2 = np.unravel_index(int(np.argmax(LL_H2_atD)),
                                                LL_H2_atD.shape)
            a_map_H2 = float(A_GRID[j_a_H2])
            s2_map_H2 = float(S2_GRID[j_s_H2])
            s_map_H2 = float(np.sqrt(s2_map_H2))
            Q_crossover_H2 = a_map_H2 / max(s_map_H2, 1e-30)
        else:
            a_map_H2 = float('nan'); s2_map_H2 = float('nan')
            s_map_H2 = float('nan'); Q_crossover_H2 = float('nan')
        # ── MAP b̂_H₁ at (D_map, a_map_H1, s2_map_H1) ──
        T_at_map = T_cache[j_map]
        T_sel = T_at_map[valid]
        n_r = T_sel.shape[0]
        T_R = np.zeros((2 * n_r, 2 * n_B))
        T_R[:n_r, :n_B] = T_sel.real; T_R[:n_r, n_B:] = -T_sel.imag
        T_R[n_r:, :n_B] = T_sel.imag; T_R[n_r:, n_B:] = T_sel.real
        sigma2 = a_map_H1**2 + s2_map_H1 * Q_abs2
        Sigma_diag = np.concatenate([0.5 * sigma2, 0.5 * sigma2])
        try:
            A = T_R.T @ np.diag(1.0 / Sigma_diag) @ T_R + np.linalg.inv(P0)
            b_hat = np.linalg.solve(A, T_R.T @ (y_H1 / Sigma_diag))
        except np.linalg.LinAlgError:
            b_hat = np.zeros(2 * n_B)
        b_complex = b_hat[:n_B] + 1j * b_hat[n_B:]
        # Predicted H1 phasor at every edge in the carve (including bridges)
        Q_H1_pred = T_at_map @ b_complex
        # ── MAP b̂_H₂ at (D_map, a_map_H2, s2_map_H2) — joint mode only ──
        b_complex_H2 = None
        Q_H2_pred = None
        if harmonic_mode == 'H1+H2' and j_map in T_H2_cache:
            T_at_map_H2 = T_H2_cache[j_map]
            T_sel_H2 = T_at_map_H2[valid]
            T_R_H2 = np.zeros((2 * n_r, 2 * n_B))
            T_R_H2[:n_r, :n_B] = T_sel_H2.real
            T_R_H2[:n_r, n_B:] = -T_sel_H2.imag
            T_R_H2[n_r:, :n_B] = T_sel_H2.imag
            T_R_H2[n_r:, n_B:] = T_sel_H2.real
            sigma2_H2 = a_map_H2**2 + s2_map_H2 * Q_abs2_H2
            Sigma_diag_H2 = np.concatenate([0.5 * sigma2_H2, 0.5 * sigma2_H2])
            try:
                A_H2 = (T_R_H2.T @ np.diag(1.0 / Sigma_diag_H2) @ T_R_H2
                        + np.linalg.inv(P0))
                b_hat_H2 = np.linalg.solve(
                    A_H2, T_R_H2.T @ (y_H2 / Sigma_diag_H2))
                b_complex_H2 = b_hat_H2[:n_B] + 1j * b_hat_H2[n_B:]
                Q_H2_pred = T_at_map_H2 @ b_complex_H2
            except np.linalg.LinAlgError:
                b_complex_H2 = np.zeros(n_B, dtype=complex)
                Q_H2_pred = np.zeros(len(edges_in), dtype=complex)
        # ── Per-edge Fisher information about D₀ ──
        # I_j(D₀) = |∂Q_pred,j / ∂D₀|² / σ²_total,j        (at MAP b̂, MAP s²)
        # Numerically: central finite difference on T(D) and apply b̂.
        # Edges with high I_j drive D̂; edges with low I_j are passengers.
        # The sum across edges (Σ I_j) is the conditional Fisher info on
        # D₀ given b̂; CI width scales as 1/√Σ I_j.
        try:
            eps_D = 0.05 * D_map
            T_plus = _T_H1_at_D(D_map + eps_D)
            T_minus = _T_H1_at_D(max(D_map - eps_D, D_GRID[0]))
            dT_dD = (T_plus - T_minus) / (2.0 * eps_D)
            dQ_dD = dT_dD @ b_complex                      # complex, per-edge
            # σ²_total at MAP s² for the measured edges only.  For unmeasured
            # edges in the carve we report Fisher info = 0 (they don't appear
            # in y, so they have no Fisher contribution about D₀).
            fisher_per_edge = np.zeros(len(edges_in))
            for k_e in range(len(edges_in)):
                if not (prob['valid_dc'][k_e] and prob['valid_h1'][k_e]):
                    continue
                Q_obs_k = prob['Q_H1_obs'][k_e]
                Q_abs2_k = float(np.abs(Q_obs_k) ** 2)
                # σ²_total at MAP = a² + s²·|Q|²  (matches the marginal LL)
                sigma2_k = a_map**2 + s2_map * Q_abs2_k
                if sigma2_k <= 0: continue
                fisher_per_edge[k_e] = float(
                    (dQ_dD[k_e].real ** 2 + dQ_dD[k_e].imag ** 2)
                    / sigma2_k)
        except Exception:
            fisher_per_edge = np.zeros(len(edges_in))
        # ── DC pressure solve at boundary nodes ──
        # At DC (ω = 0) the distributed two-port reduces exactly to the
        # purely resistive limit:  κ → 0,  Y_self → G,  Y_trans → G.
        # Build the DC admittance from scratch using the same
        # distributed framework with D₀ = 0 → κ = 0 → resistive limit.
        # This guarantees the DC and H₁ solves use the SAME forward map
        # (manuscript form), not a separate lumped variant.
        b_DC = np.zeros(n_B)
        # Diagnostic-only DC residuals (kept None if the DC solve below
        # is skipped or fails — the H₁ inference is unaffected either way).
        Q_DC_meas_diag = None
        Q_DC_pred_diag = None
        Q_DC_pred_full = None
        sigma_DC_diag = None
        valid_dc_idx_diag = None
        try:
            L_DC = np.zeros((n_int, n_int), dtype=float)
            B_DC_mat = np.zeros((n_int, n_B), dtype=float)
            for k, (uu, vv) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                G = edge_G[k]      # κ → 0 limit: Y_self = Y_trans = G
                u_int = uu in interior_idx; v_int = vv in interior_idx
                if u_int and v_int:
                    iu, iv = interior_idx[uu], interior_idx[vv]
                    L_DC[iu, iu] += G; L_DC[iv, iv] += G
                    L_DC[iu, iv] -= G; L_DC[iv, iu] -= G
                elif u_int and not v_int:
                    iu, jb = interior_idx[uu], boundary_idx[vv]
                    L_DC[iu, iu] += G; B_DC_mat[iu, jb] -= G
                elif v_int and not u_int:
                    iv, jb = interior_idx[vv], boundary_idx[uu]
                    L_DC[iv, iv] += G; B_DC_mat[iv, jb] -= G
            diag_med_DC = (np.median(np.abs(np.diag(L_DC)))
                            if n_int else 1.0)
            scale_DC = 1.0 / diag_med_DC if diag_med_DC > 0 else 1.0
            P_int_DC = np.linalg.lstsq(
                L_DC * scale_DC, -B_DC_mat * scale_DC, rcond=None)[0]
            T_DC = np.zeros((len(edges_in), n_B))
            for k, (uu, vv) in enumerate(edges_in):
                G = edge_G[k]
                if G == 0: continue
                for jb in range(n_B):
                    pu = (P_int_DC[interior_idx[uu], jb] if uu in interior_idx
                          else (1.0 if boundary_idx[uu] == jb else 0.0))
                    pv = (P_int_DC[interior_idx[vv], jb] if vv in interior_idx
                          else (1.0 if boundary_idx[vv] == jb else 0.0))
                    # DC limit of Q_e = Y_self·p_u − Y_trans·p_v → G·(p_u − p_v)
                    T_DC[k, jb] = G * (pu - pv)
            valid_dc_idx = np.where(prob['valid_dc'])[0]
            if valid_dc_idx.size >= n_B:
                T_DC_sel = T_DC[valid_dc_idx]
                Q_DC_meas = prob['Q_DC_obs'][valid_dc_idx]
                # Per-edge DC measurement variance from canonical SNR_DC
                # (same convention as H₁):  σ²_DC,j = |Q_DC,j|²·10^(−SNR_DC/10).
                # Fall back to a 30% relative floor for edges where SNR_DC
                # isn't populated.  This is just the LS weighting for the
                # post-hoc b̂_DC solve — it doesn't enter the H₁ inference.
                snr_DC_sel = snr_DC_per_edge[valid_dc_idx]
                snr_DC_sel = np.where(np.isfinite(snr_DC_sel),
                                        snr_DC_sel, np.nan)
                r_DC_sel = 10.0 ** (-snr_DC_sel / 10.0)
                sigma_DC_emp = np.abs(Q_DC_meas) * np.sqrt(r_DC_sel)
                # Where SNR is missing → fall back to 30 % relative
                fallback = 0.3 * np.abs(Q_DC_meas)
                sigma_DC = np.where(np.isfinite(sigma_DC_emp),
                                      sigma_DC_emp, fallback)
                # Hard floor against zero-magnitude edges
                sigma_DC = np.maximum(sigma_DC, 1e-18)
                W_DC = 1.0 / sigma_DC ** 2
                ATA = T_DC_sel.T @ (W_DC[:, None] * T_DC_sel)
                ATb = T_DC_sel.T @ (W_DC * Q_DC_meas)
                diag_scale = (np.median(np.abs(np.diag(ATA)))
                                if n_B else 1.0) or 1.0
                ATA += 1e-3 * diag_scale * np.eye(n_B)
                b_DC = np.linalg.solve(ATA, ATb)
                # DC gauge fix: the absolute pressure at boundary nodes
                # is defined only up to a constant.  The ridge term
                # selected an arbitrary minimum-norm gauge.  For
                # visualisation we subtract the mean so that displayed
                # |b̂_DC,n| reflects spatial pressure VARIATION rather
                # than the gauge offset.  Differences between BC nodes
                # are gauge-invariant; absolute values are not.
                b_DC = b_DC - float(np.mean(b_DC))
                # Stash DC residual ingredients for the diagnostic panel.
                # Q_DC prediction is gauge-invariant (T_DC rows are p_u−p_v
                # differences), so the mean-subtraction above doesn't matter.
                Q_DC_pred_diag = T_DC_sel @ b_DC
                Q_DC_meas_diag = Q_DC_meas
                sigma_DC_diag = sigma_DC
                valid_dc_idx_diag = valid_dc_idx
                # Also store full-edges-in Q_DC prediction (parallel
                # to Q_H1_pred) so the Compare popup can show the
                # *inferred* DC mean alongside the measured one.
                Q_DC_pred_full = T_DC @ b_DC
        except Exception:
            pass
        delta_at_zero = float(
            logsumexp(log_L[j_map]) - logsumexp(log_L[0]))
        return dict(
            tile_id=int(tile_id),
            alpha=float(alpha),
            sigma_b=float(sigma_b_Pa),
            bc_length_px=float(bc_length_px),
            n_av_promoted=int(_n_av_promoted),
            # Physical-validity D cap based on max_e(C_e) ≤ C_E_CAP
            C_e_cap=float(C_E_CAP),
            D_cap=float(D_cap),
            cap_omega_mult=(None if cap_omega_mult is None
                              else float(cap_omega_mult)),
            K_meas_max=float(K_meas_max),
            # C_e at MAP — sanity check on whether the chosen D is well
            # inside the linear-RC regime.
            C_e_at_map=float(D_map * K_meas_max),
            D_hat=D_map,
            lo=lo, hi=hi, width=width,             # 3-nat support region
            lo_90=lo_90, hi_90=hi_90, width_90=width_90,   # HPD 90% CI
            lo_95=lo_95, hi_95=hi_95, width_95=width_95,   # HPD 95% CI
            tier=tier,
            s2_at_map=s2_map,
            a_at_map=a_map,
            s_at_map=s_map,
            Q_crossover=Q_crossover,
            frac_err_pct=float(np.sqrt(s2_map) * 100),
            log_p_D=log_p_D,
            D_GRID=D_GRID,
            log_L=log_L,
            S2_GRID=S2_GRID,
            A_GRID=A_GRID,
            n_meas=n_v,
            snr_median=float(np.nanmedian(snr_per_edge[valid_idx])),
            R_ref_um=float(R_ref * 1e6),
            R_min_um=float(R_e[valid].min() * 1e6),
            R_max_um=float(R_e[valid].max() * 1e6),
            b_hat=b_hat,
            b_complex=b_complex,
            b_DC=b_DC,
            # DC residual ingredients for the diagnostic (None if DC solve
            # was skipped — Panel D will just plot H₁ in that case).
            Q_DC_meas_diag=Q_DC_meas_diag,
            Q_DC_pred_diag=Q_DC_pred_diag,
            Q_DC_pred_full=Q_DC_pred_full,
            harmonic_mode=str(harmonic_mode),
            # Per-channel noise-model MAP values (joint-mode only for H2)
            a_at_map_H1=a_map_H1,
            s_at_map_H1=s_map_H1,
            Q_crossover_H1=Q_crossover_H1,
            a_at_map_H2=a_map_H2,
            s_at_map_H2=s_map_H2,
            Q_crossover_H2=Q_crossover_H2,
            # H₂ MAP forcing + prediction (joint mode only)
            b_complex_H2=b_complex_H2,
            Q_H2_pred=Q_H2_pred,
            sigma_DC_diag=sigma_DC_diag,
            valid_dc_idx_diag=valid_dc_idx_diag,
            Q_H1_pred=Q_H1_pred,
            # Fisher info is stored in BOTH units:
            #   fisher_per_edge       : I_j(D)        in 1/Pa²   (raw)
            #   fisher_per_edge_logD  : I_j(log D)    dimensionless,
            #     converted via chain rule  ∂Q/∂(log D) = D · ∂Q/∂D
            #     so I_log_j = I_D_j · D̂²
            #     Directly comparable to the log-normal prior precision
            #     1/τ_D² = 1/(ln 10)² ≈ 0.189.
            fisher_per_edge=fisher_per_edge,
            fisher_per_edge_logD=fisher_per_edge * (D_map ** 2),
            R_e=R_e,
            prob=prob,
            delta_at_zero=delta_at_zero,
        )

    def _inference_render_diagnostic(self, tile_id, result):
        """Six-panel diagnostic for the additive-noise Bayesian fit:
          (1) D₀ posterior with the log-normal prior overlaid,
          (2) priors + physical assumptions as a text panel,
          (3) 2D log-likelihood heatmap over (D₀, s²),
          (4) normalised residual histogram (H₁ Re∪Im + DC overlay)
              against N(0,1) reference,
          (5) per-edge fit scatter Re(H₁),
          (6) per-edge fit scatter Im(H₁).
        """
        import matplotlib.pyplot as plt
        import tempfile
        D_GRID = result['D_GRID']
        log_p_D = result['log_p_D']
        log_L = result['log_L']
        S2_GRID = result['S2_GRID']
        D_hat = result['D_hat']
        lo, hi = result['lo'], result['hi']
        s2_map = result['s2_at_map']
        frac_pct = result['frac_err_pct']
        prob = result['prob']
        b_complex = result['b_complex']
        # Physical constants (mirror values used in _inference_run)
        MU_PHYS = 3.5e-3
        RHO_PHYS = 1.06e3
        PX_SIZE_M_PHYS = 1.7e-6
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        f0_used = float(tile_f0_map.get(int(tile_id))
                         or self.G.graph.get('f0_hz') or 2.5)
        R_ref_m = result['R_ref_um'] * 1e-6
        # Womersley at R_ref: Wo = R√(ωρ/μ)
        omega = 2.0 * np.pi * f0_used
        Wo_ref = R_ref_m * np.sqrt(omega * RHO_PHYS / MU_PHYS)
        # RC attenuation length (Desktop-corrected).  For a distributed RC
        # vessel with r' = 8μ/(πR⁴), c' = πR²D, the propagation constant
        # γ = √(iωr'c') = √(i·8μωD/R²).  Re(γ) = √(4μωD/R²), so
        # the amplitude attenuation / phase accumulation length is
        #   ℓ_RC = 1/Re(γ) = R / (2·√(μ·ω·D)).
        # MORE compliance → SHORTER ℓ_RC (more storage, stronger damping).
        # Earlier display formula had D in the numerator; that was a
        # dimensional / physical error in the label only, not in the
        # inference itself.
        ell_ref_m = R_ref_m / max(
            2.0 * np.sqrt(MU_PHYS * omega * D_hat), 1e-30)

        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

        # ── Panel A: D₀ posterior with prior ──
        ax = fig.add_subplot(gs[0, 0])
        p_D = np.exp(log_p_D - log_p_D.max())
        MU_D, TAU_D = np.log(1.5e-3), np.log(10.0)
        log_prior_D = -0.5 * ((np.log(D_GRID) - MU_D) / TAU_D) ** 2
        prior = np.exp(log_prior_D - log_prior_D.max())
        ax.plot(D_GRID, prior, color='C7', lw=1.4, ls=':',
                 label=('prior: LogN(ln 1.5×10⁻³, τ=ln 10)'))
        # 95% prior support band
        D_prior_lo = np.exp(MU_D - 2 * TAU_D)
        D_prior_hi = np.exp(MU_D + 2 * TAU_D)
        ax.axvspan(D_prior_lo, D_prior_hi, color='C7', alpha=0.06,
                    label=f'95% prior band [{D_prior_lo:.1e}, {D_prior_hi:.1e}]')
        ax.fill_between(D_GRID, 0, p_D, alpha=0.25, color='C0')
        ax.plot(D_GRID, p_D, color='C0', lw=2.0,
                 label='posterior p(D₀ | y)')
        ax.axvline(D_hat, color='C3', lw=1.4, ls='--',
                    label=f'MAP D̂ = {D_hat:.2e}')
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color='C3', alpha=0.10,
                        label=f'3-nat support [{lo:.1e}, {hi:.1e}]')
        # Proper 90% HPD credible interval (Desktop critique fix: the
        # 3-nat band is not a probability-calibrated CI).
        lo_90 = result.get('lo_90'); hi_90 = result.get('hi_90')
        if lo_90 is not None and hi_90 is not None and np.isfinite(lo_90):
            ax.axvspan(lo_90, hi_90, color='C0', alpha=0.18,
                        label=f'90% HPD CI [{lo_90:.1e}, {hi_90:.1e}]')
        ax.set_xscale('log')
        ax.set_xlabel('D₀ (/Pa)')
        ax.set_ylabel('density (peak-normalised)')
        ax.set_title(f'D₀ posterior  ·  tile {tile_id}  ·  α = {result["alpha"]:+.2f}  ·  {result["tier"]}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        # (Old Panel B text block dropped — clutter without much value.
        # Headline numbers are in the suptitle / panel titles instead.)
        ax_info = None  # unused, prior code referenced this name
        _info_lines_unused = [
            '── Forward model — distributed two-port (manuscript form) ──',
            '   Y_self,e  = G_e · (1 + κ_e²/3)',
            '   Y_trans,e = G_e · (1 − κ_e²/6)',
            '   κ_e² = i·ω·D_e·8μ L_e²/R_e²   ( = i·C_e ;  C_e = compliance number )',
            '   Edge flow:  Q_e = Y_self,e · p_u  −  Y_trans,e · p_v',
            f'   D-grid capped at  max_e C_e ≤ {result["C_e_cap"]:.2f}  '
            f'(D_cap = {result["D_cap"]:.2e}/Pa).',
            f'   At MAP D̂, max_e C_e ≈ {result["C_e_at_map"]:.3f}.',
            '',
            '── Priors ──',
            '   D₀  ~  LogNormal( μ = ln 1.5×10⁻³,  τ = ln 10 )',
            f'        ≈95% prior support: [{D_prior_lo:.1e}, {D_prior_hi:.1e}] /Pa',
            f'   b_H₁ ~  Complex N(0, σ_b² · I_{{n_B}})  with σ_b = {result["sigma_b"]:.1f} Pa',
            '        Var(Re b_n) = Var(Im b_n) = σ_b²/2;  σ_b = RMS complex amplitude.',
            f'   a   log-uniform on [{result["A_GRID"][0]:.0e}, {result["A_GRID"][-1]:.0e}]  '
            f'(m³/s, {result["A_GRID"].size} pts)',
            f'   s²  log-uniform on [{S2_GRID[0]:.0e}, {S2_GRID[-1]:.0e}]  '
            f'(dimensionless, {S2_GRID.size} pts)',
            '',
            '── Noise model (additive + proportional, both inferred) ──',
            '   σ²_j  =  a²  +  s² · |Q_H₁,j|²',
            '   a   (m³/s)        — measurement-noise FLOOR (absolute, signal-independent)',
            '   s²  (dimensionless) — proportional MISFIT (ridge + KCL deficit ∝ |Q|)',
            '   |Q|* = a / s       — crossover: |Q| < |Q|* → noise-dominated edge',
            '                                    |Q| ≥ |Q|* → misfit-dominated edge',
            f'   â = {result["a_at_map"]*1e12:.4f} nL/s   ·   '
            f'ŝ = {result["s_at_map"]:.3f}   ·   '
            f'|Q|* = {result["Q_crossover"]*1e12:.4f} nL/s',
            f'   median SNR_H₁ = {result["snr_median"]:+.1f} dB '
            f'(reported, NOT used in weighting; γ → 0 from inferred-shape test)',
            '',
            '── Identifiability note ──',
            '   Under C_e ≪ 1 with approximately-uniform b̂_H₁, the data identify',
            '   the product D·|b̂_H₁| and the shape of b̂ across BC nodes; the',
            '   absolute split between D and |b̂_H₁| depends on σ_b.',
            f'   D̂ above is the value at σ_b = {result["sigma_b"]:.1f} Pa.  Treat as one',
            '   point on a structural b–D ridge, not a unique measurement.',
            '',
            '── Per-edge D ──',
            f'   D_e = D₀ · (R_e / R_ref)^α   with α = {result["alpha"]:+.2f}',
            f'   R_ref (median) = {result["R_ref_um"]:.1f} µm  ·  '
            f'range [{result["R_min_um"]:.1f}, {result["R_max_um"]:.1f}] µm',
            '',
            '── Physical constants ──',
            f'   μ  (viscosity)  = {MU_PHYS*1e3:.1f} mPa·s',
            f'   ρ  (density)   = {RHO_PHYS:.0f} kg/m³',
            f'   px size        = {PX_SIZE_M_PHYS*1e6:.2f} µm/px',
            f'   f₀  (this tile) = {f0_used:.3f} Hz   (per-tile from canonical graph)',
            '',
            '── Derived ──',
            f'   ω = 2π·f₀          = {omega:.2f}  rad/s',
            f'   Wo at R_ref        = R·√(ωρ/μ)  = {Wo_ref:.3f}   '
            f'(quasi-steady Poiseuille valid if Wo small)',
            f'   RC propagation depth  ℓ_RC = R/(2·√(μωD))  '
            f'at R_ref = {ell_ref_m*1e6:.1f} µm',
            '',
            '── Carve ──',
            f'   {len(prob["edges_in"])} edges in carve  '
            f'({result["n_meas"]} measured H₁)',
            f'   {len(prob["boundary_nodes"])} boundary, '
            f'{len(prob["interior_nodes"])} interior nodes',
            (f'   ({result.get("n_av_promoted", 0)} anatomical A/V '
             f'anchors promoted from interior → boundary)'
             if result.get("n_av_promoted", 0) else
             '   (no anatomical A/V anchors in this carve)'),
        ]
        # (text panel removed)

        # ── Panel B: 2D (a, s²) log-likelihood at MAP D ──
        # Slice the 3D log_L at MAP D, plot the (a, s²) joint posterior.
        # Use an adaptive display range (tighter than the old hard-coded
        # -20) and overlay 1/2/3-σ contours so the actual constrained
        # region is visible even when the posterior is very peaked.
        ax = fig.add_subplot(gs[0, 1])
        A_GRID = result["A_GRID"]
        LL_atD = log_L[int(np.argmax(result["log_p_D"]))]
        log_L_disp = LL_atD - LL_atD.max()
        A_disp = A_GRID * 1e12          # nL/s
        # Adaptive vmin: cover the 99th-percentile of log p mass on grid
        # (so most of the colourmap range falls on the constrained region,
        # not the -∞ floor that dominated the old display).
        finite_ll = log_L_disp[np.isfinite(log_L_disp)]
        vmin = float(np.percentile(finite_ll, 1)) if finite_ll.size else -10.0
        vmin = max(vmin, -10.0)         # never go below -10 nats for clarity
        im = ax.pcolormesh(A_disp, S2_GRID, log_L_disp.T,
                            cmap='viridis', shading='auto',
                            vmin=vmin, vmax=0)
        plt.colorbar(im, ax=ax, label='log p(a, s² | y) − max  (nats)')
        # 1/2/3-σ contour boundaries: Δlog p ≈ -0.5 / -2 / -4.5 in 2D.
        try:
            ax.contour(A_disp, S2_GRID, log_L_disp.T,
                        levels=[-4.5, -2.0, -0.5],
                        colors='white', linewidths=[0.5, 0.8, 1.1],
                        linestyles=['dotted', 'dashed', 'solid'])
        except Exception:
            pass
        a_disp = result["a_at_map"] * 1e12
        ax.plot(a_disp, s2_map, 'o', color='red', mfc='none', ms=9,
                 mew=2, label=f'MAP  â = {a_disp:.3f} nL/s,  ŝ = {result["s_at_map"]:.2g}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('a  (nL/s, measurement noise floor)')
        ax.set_ylabel('s²  (proportional misfit)')
        ax.set_title('(a, s²) joint posterior at MAP D')
        ax.legend(loc='upper right', fontsize=9)

        # ── Per-edge fit data + per-edge SNR (used in panels D, E, F) ──
        valid = prob['valid_dc'] & prob['valid_h1']
        Q_meas = prob['Q_H1_obs'][valid]
        Q_pred = result['Q_H1_pred'][valid]
        n_v = int(valid.sum())
        snrs = np.zeros(n_v)
        valid_idx = np.where(valid)[0]
        for k, i in enumerate(valid_idx):
            u, v = prob['edges_in'][i]
            for m in (self.G.edges[u, v].get('measurements_piv') or []):
                # v4 migration: was m.get('tile_id')
                if MeasurementView(m).tile_id == int(tile_id):
                    # Q_H1_snr_db is a canonical top-level field; keep.
                    snrs[k] = m.get('Q_H1_snr_db', 0.0) or 0.0; break

        # ── Panel C: Normalised residual histogram (H₁ + DC overlay) ──
        ax_hist = fig.add_subplot(gs[1, 0])
        Q_abs2 = np.abs(Q_meas) ** 2
        # σ²_total per edge = a² + s²·|Q|²;  per-component σ²/2
        a_map = float(result["a_at_map"])
        sigma_per_comp = np.sqrt(0.5 * (a_map**2 + s2_map * Q_abs2))
        sigma_per_comp = np.where(sigma_per_comp > 0, sigma_per_comp, 1e-30)
        z_re = (Q_meas.real - Q_pred.real) / sigma_per_comp
        z_im = (Q_meas.imag - Q_pred.imag) / sigma_per_comp
        z_H1 = np.concatenate([z_re, z_im])

        # DC residuals from the same Bayesian fit's b̂_DC solve
        # (post-hoc LS at the H₁ MAP).  σ_DC was built from canonical
        # SNR_DC per edge, same convention as H₁: σ²_DC,j = |Q_DC,j|²·10^(−SNR_DC/10).
        Q_DC_meas_d = result.get('Q_DC_meas_diag')
        Q_DC_pred_d = result.get('Q_DC_pred_diag')
        sigma_DC_d  = result.get('sigma_DC_diag')
        z_DC = None
        if (Q_DC_meas_d is not None and Q_DC_pred_d is not None
                and sigma_DC_d is not None and len(Q_DC_meas_d) > 0):
            sd = np.where(sigma_DC_d > 0, sigma_DC_d, 1e-30)
            z_DC = (Q_DC_meas_d - Q_DC_pred_d) / sd
            z_DC = np.asarray(z_DC, float)

        # Symmetric x limit covers whichever channel has fatter tails
        _z_pool = z_H1 if z_DC is None else np.concatenate([z_H1, z_DC])
        z_lim = max(4.0, np.percentile(np.abs(_z_pool), 95) * 1.2)
        bins = np.linspace(-z_lim, z_lim, 60)

        ax_hist.hist(z_H1, bins=bins, density=True,
                      histtype='stepfilled', color='C0', alpha=0.55,
                      edgecolor='C0', linewidth=1.3,
                      label=f'H₁ (Re ∪ Im, n = {z_H1.size})')
        if z_DC is not None:
            ax_hist.hist(z_DC, bins=bins, density=True,
                          histtype='stepfilled', color='C2', alpha=0.45,
                          edgecolor='C2', linewidth=1.3,
                          label=f'DC (n = {z_DC.size})')
        x_ref = np.linspace(-z_lim, z_lim, 200)
        gauss_ref = np.exp(-0.5 * x_ref ** 2) / np.sqrt(2 * np.pi)
        ax_hist.plot(x_ref, gauss_ref, color='C3', lw=1.6, ls='--',
                      label='N(0, 1)')
        ax_hist.axvline(0, color='#444', lw=0.4, ls=':')
        ax_hist.set_xlim(-z_lim, z_lim)
        ax_hist.set_xlabel('normalised residual  z = (Q_meas − Q_pred) / σ')
        ax_hist.set_ylabel('density')
        rms_H1 = float(np.sqrt(np.mean(z_H1 ** 2)))
        title_str = f'Residuals  ·  RMS|z|_H₁ = {rms_H1:.2f} (target 1)'
        if z_DC is not None and z_DC.size > 0:
            rms_DC = float(np.sqrt(np.mean(z_DC ** 2)))
            title_str += f'  ·  RMS|z|_DC = {rms_DC:.2f}'
        ax_hist.set_title(title_str, fontsize=11)
        ax_hist.legend(fontsize=9, loc='upper right')
        ax_hist.grid(True, alpha=0.25)

        # ── Panel D: per-edge fit Q_meas vs Q_pred (Re ∪ Im stacked) ──
        ax_fit = fig.add_subplot(gs[1, 1])
        scale = 1e12   # m³/s → nL/s
        meas_stack = np.concatenate([Q_meas.real, Q_meas.imag]) * scale
        pred_stack = np.concatenate([Q_pred.real, Q_pred.imag]) * scale
        snr_stack = np.concatenate([snrs, snrs])
        amp_max = float(max(np.max(np.abs(meas_stack)),
                             np.max(np.abs(pred_stack)), 0.01))
        sc = ax_fit.scatter(meas_stack, pred_stack,
                             c=snr_stack, cmap='viridis', s=14,
                             edgecolor='black', linewidth=0.25)
        plt.colorbar(sc, ax=ax_fit, label='SNR_H₁ (dB)')
        ax_fit.plot([-amp_max, amp_max], [-amp_max, amp_max],
                     'k--', lw=0.6, alpha=0.6)
        ax_fit.set_xlim(-amp_max, amp_max); ax_fit.set_ylim(-amp_max, amp_max)
        ax_fit.set_xlabel('measured  Q_H₁  (nL/s)')
        ax_fit.set_ylabel('predicted Q_H₁  (nL/s)')
        ax_fit.set_title('Per-edge fit: Re ∪ Im (H₁)')
        ax_fit.set_aspect('equal'); ax_fit.grid(True, alpha=0.25)

        fig.suptitle(
            f"Distributed-RC Bayesian D₀ inference  ·  tile {tile_id}  ·  "
            f"α = {result['alpha']:+.2f}  ·  σ_b = {result['sigma_b']:.1f} Pa  ·  "
            f"D̂ = {D_hat:.2e} /Pa  ·  3-nat width = {result['width']:.2f} dec  ·  "
            f"{result['tier']}",
            fontsize=11, fontweight='bold')

        # Save to PNG then display via Qt dialog (napari event loop)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as _tmp:
            _tmp_path = _tmp.name
        fig.savefig(_tmp_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        self._show_diagnostic_png(_tmp_path, int(tile_id), result)

    def _show_diagnostic_png(self, png_path: str, tile_id: int, result: dict):
        """Pop up a Qt dialog displaying a saved diagnostic PNG.

        The dialog is non-modal, has a fixed window title with the
        tile_id and headline D̂, and is retained on `self` so Python's
        GC doesn't close it the instant this method returns."""
        from qtpy.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QScrollArea)
        from qtpy.QtGui import QPixmap
        from qtpy.QtCore import Qt
        dlg = QDialog(self.viewer.window._qt_window)
        # Window title summarises the result; width-in-decades replaces
        # the old σ_D/D̂ relative-error display since the new Bayesian
        # posterior reports a 3-nat CI band, not a σ.
        width_str = (f"width {result['width']:.2f} dec"
                     if 'width' in result
                     else f"σ_D/D̂ = {result.get('rel_sigma_D', float('nan')):.1%}")
        dlg.setWindowTitle(
            f"Inference diagnostic — tile {tile_id}  "
            f"D̂ = {result['D_hat']:.3e},  {width_str}")
        # Layout: scroll area → label → pixmap
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        lbl = QLabel()
        pixmap = QPixmap(png_path)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        # Match the rendered figure size (default render dpi=180 →
        # 15×9 inches → ~2700×1620 px) but cap to 90% of screen.
        try:
            screen = self.viewer.window._qt_window.screen().availableGeometry()
            target_w = min(int(pixmap.width()),
                           int(0.9 * screen.width()))
            target_h = min(int(pixmap.height() + 50),
                           int(0.9 * screen.height()))
        except Exception:
            target_w, target_h = 1200, 800
        dlg.resize(target_w, target_h)
        # Keep a reference so the dialog stays alive after this method
        # returns.  Reuse a single list slot so old dialogs are GC'd
        # naturally if the user closes them.
        if not hasattr(self, '_inf_diag_dialogs'):
            self._inf_diag_dialogs = []
        # Prune any that have already been closed.
        self._inf_diag_dialogs = [
            d for d in self._inf_diag_dialogs if d.isVisible()]
        self._inf_diag_dialogs.append(dlg)
        dlg.show()

    # ── UI glue ──────────────────────────────────────────────────────
    def _on_inference_tile_change(self, _idx=None):
        # Stale: previous result was for a different tile.
        self._inf_diag_btn.setEnabled(False)
        if hasattr(self, '_inf_compare_btn'):
            self._inf_compare_btn.setEnabled(False)
        self._inf_results_box.setPlainText("")
        self._inference_last_result = None
        # Drop the BC forcing + Fisher-info overlays from previous tile.
        self._tear_down_inference_bc_overlay()
        self._tear_down_inference_fisher_overlay()
        if self._inf_show_carve_chk.isChecked():
            self._refresh_inference_carve_overlay()

    def _on_inference_show_carve_toggle(self, checked: bool):
        if not checked:
            self._tear_down_inference_carve_overlay()
        else:
            self._refresh_inference_carve_overlay()

    def _refresh_inference_carve_overlay(self):
        """Update the carve highlight on the mosaic — both the carved
        edges (thick yellow lines) and the carve boundary nodes (filled
        orange dots) — to reflect the currently-selected tile."""
        self._tear_down_inference_carve_overlay()
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        topo = self._inference_get_carve_topology(int(tile_id))
        edges = topo['edges']
        bnodes = topo['boundary_nodes']
        H = self.mosaic_height
        # ── Edge overlay (thick yellow lines) ──
        lines = []
        for (u, v) in edges:
            if not self.G.has_edge(u, v):
                continue
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
        if lines:
            self._inference_carve_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color='#ffd400',  # bright yellow
                edge_width=8.0,
                opacity=0.85,
                name=f'Inference carve (tile {tile_id})',
            )
        # ── Boundary-node overlay (filled dots) ──
        pts = []
        for n in bnodes:
            try:
                x = self.G.nodes[n]['x']
                y = self.G.nodes[n]['y']
            except KeyError:
                continue
            pts.append([H - y, x])
        if pts:
            self._inference_carve_node_layer = self.viewer.add_points(
                np.asarray(pts), size=18,
                face_color='#ff6600',  # orange
                border_color='white',
                opacity=0.95,
                name=f'Carve boundary nodes (tile {tile_id})',
            )

    def _tear_down_inference_carve_overlay(self):
        for attr in ('_inference_carve_layer',
                     '_inference_carve_node_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)

    def _refresh_inference_bc_overlay(self, result):
        """After an inference run, draw the inferred boundary forcing
        on the mosaic as two napari layers:

          - Points layer at each BC node, with
              face_color = |b̂_DC|   (DC pressure magnitude, viridis)
              size      = |b̂_H₁|    (H₁ pressure amplitude)
          - Vectors layer of arrows, one per BC node:
              origin    = BC node
              direction = arg(b̂_H₁)  (H₁ phase, math convention)
              length    ∝ |b̂_H₁|     so amplitude redundant-encoded

        Removes any previous BC layers first."""
        from matplotlib import cm
        self._tear_down_inference_bc_overlay()
        boundary_nodes = result['prob']['boundary_nodes']
        if not boundary_nodes:
            return
        b_complex = result['b_complex']
        b_DC = result.get('b_DC')
        if b_DC is None:
            b_DC = np.zeros(len(boundary_nodes))
        H = self.mosaic_height
        b_amp_H1 = np.abs(b_complex)
        b_amp_DC = np.abs(b_DC)
        H1_max = max(b_amp_H1.max(), 1e-12)
        DC_max = max(b_amp_DC.max(), 1e-12)
        # ── Points layer ──
        pts_yx = []
        for n in boundary_nodes:
            x = float(self.G.nodes[n]['x'])
            y = float(self.G.nodes[n]['y'])
            pts_yx.append([H - y, x])
        pts_yx = np.asarray(pts_yx)
        # Marker size scales linearly with |b̂_H1|; clamped to a min
        # visible size so zero-forcing nodes are still locatable.
        sizes = 12.0 + 60.0 * (b_amp_H1 / H1_max)
        face_colors = cm.viridis(b_amp_DC / DC_max)
        self._inference_bc_layer = self.viewer.add_points(
            pts_yx,
            size=sizes,
            face_color=face_colors,
            border_color='black',
            border_width=0.15,
            opacity=0.92,
            name=(f"Inference BC tile {result['tile_id']}  "
                  f"(size=|b̂_H₁|≤{H1_max:.2f}Pa, "
                  f"color=|b̂_DC − mean|≤{DC_max:.2f}Pa)"),
        )
        # ── Vectors layer for phase arrows ──
        n_b = len(boundary_nodes)
        # Pixel scale for the arrows: pick a fraction of the mosaic span
        # so they're always visible at the default zoom.
        try:
            span = max(self.mosaic_height, self.mosaic_width)
        except Exception:
            span = 1000
        arrow_max_len = 0.025 * float(span)
        vectors = np.zeros((n_b, 2, 2))
        for k, n in enumerate(boundary_nodes):
            x = float(self.G.nodes[n]['x'])
            y = float(self.G.nodes[n]['y'])
            ang = float(np.angle(b_complex[k]))
            scale = float(b_amp_H1[k] / H1_max)
            L_arrow = arrow_max_len * scale
            # napari uses (row, col) with row = down.  Mosaic flips with
            # row = H − y_graph, so increasing y_graph maps to decreasing
            # row.  Math direction (cos ang, sin ang) in (x, y) maps to
            # napari direction (−sin ang, cos ang) in (row, col).
            vectors[k, 0] = [H - y, x]
            vectors[k, 1] = [-L_arrow * np.sin(ang),
                              L_arrow * np.cos(ang)]
        try:
            self._inference_bc_arrow_layer = self.viewer.add_vectors(
                vectors,
                edge_color='red',
                edge_width=2.4,
                opacity=0.95,
                length=1.0,
                vector_style='arrow',  # napari ≥ 0.5
                name=f"Inference BC phase tile {result['tile_id']} (arrow = arg b̂_H₁)",
            )
        except TypeError:
            # napari < 0.5: vector_style not supported; fall back to plain lines.
            self._inference_bc_arrow_layer = self.viewer.add_vectors(
                vectors,
                edge_color='red', edge_width=2.4,
                opacity=0.95, length=1.0,
                name=f"Inference BC phase tile {result['tile_id']} (line = arg b̂_H₁)",
            )

    def _tear_down_inference_bc_overlay(self):
        for attr in ('_inference_bc_layer',
                     '_inference_bc_arrow_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)

    def _refresh_inference_fisher_overlay(self, result):
        """Per-edge Fisher information about D₀, rendered as a napari
        Shapes (lines) layer where each carve edge's color encodes
        I_j (in log-D units, dimensionless), and a separate Shapes
        layer with the top-K most-informative edges drawn in bold red.

        Reading: bright edges (top of cmap) are doing the work — they're
        the ones whose data is constraining D̂.  Dim edges contribute
        little to the Bayesian update from prior to posterior.

        The TOP-K layer (default K = 10) overlays a thicker red highlight
        on the dominant edges so they pop out at any zoom."""
        from matplotlib import cm, colors as mcolors
        self._tear_down_inference_fisher_overlay()
        prob = result['prob']
        edges_in = prob['edges_in']
        fpe = result.get('fisher_per_edge_logD')
        if fpe is None or len(edges_in) == 0:
            return
        fpe = np.asarray(fpe)
        I_total = max(float(fpe.sum()), 1e-30)
        # ── Edge segments in napari (row, col) coords ──
        H = self.mosaic_height
        lines = []; values = []
        for k, (u, v) in enumerate(edges_in):
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
            values.append(fpe[k])
        if not lines:
            return
        values = np.asarray(values, dtype=float)
        # Color by log10(I_j) so the wide dynamic range is legible.
        # Floor at the smallest positive value so log10 doesn't explode.
        v_pos = values[values > 0]
        if v_pos.size == 0:
            return
        v_min_pos = max(np.percentile(v_pos, 5), 1e-30)
        v_eff = np.where(values > 0, values, v_min_pos)
        log_v = np.log10(v_eff)
        log_v_clip = np.clip(log_v, np.percentile(log_v, 5),
                              np.percentile(log_v, 95))
        norm = mcolors.Normalize(vmin=log_v_clip.min(),
                                  vmax=log_v_clip.max())
        colors_rgba = cm.viridis(norm(log_v_clip))
        max_frac = float(values.max() / I_total)
        self._inference_fisher_layer = self.viewer.add_shapes(
            lines, shape_type='line',
            edge_color=colors_rgba,
            edge_width=2.4,
            opacity=0.85,
            name=(f"Inference Fisher info tile {result['tile_id']}  "
                  f"(viridis = log₁₀ I_j;  top edge = "
                  f"{max_frac*100:.1f}% of I_total)"),
        )
        # ── Top-K highlight (red, thicker) ──
        K = min(10, len(lines))
        order = np.argsort(-values)
        top_lines = [lines[i] for i in order[:K]]
        top_frac = float(values[order[:K]].sum() / I_total)
        self._inference_fisher_top_layer = self.viewer.add_shapes(
            top_lines, shape_type='line',
            edge_color='red', edge_width=4.5,
            opacity=0.85,
            name=(f"Inference Fisher TOP-{K} tile {result['tile_id']}  "
                  f"(carries {top_frac*100:.0f}% of I_total)"),
        )

    def _tear_down_inference_fisher_overlay(self):
        for attr in ('_inference_fisher_layer',
                     '_inference_fisher_top_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)

    def _on_inference_run_clicked(self):
        from qtpy.QtWidgets import QApplication
        import time
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        flat_b_prior = bool(self._inf_flat_b_prior_cb.isChecked())
        # σ_b = 10⁴ Pa is far above the BC confound regime (~50 Pa) and
        # far above the σ-sweep range that's been validated (3–100 Pa).
        # In this limit P₀ ≫ Σ_obs everywhere, so the b prior contribution
        # to C = Σ_obs + T·P₀·Tᵀ is dominated by the data via T·Tᵀ —
        # i.e. an effective improper-flat-prior result with no
        # normalisation pathology, since the marginal log p(y|D, s²)
        # uses the proper Gaussian formula throughout.
        if flat_b_prior:
            sigma_b = 1.0e4
        else:
            sigma_b = float(self._inf_sigma_b_spin.value())
        alpha = float(self._inf_alpha_spin.value())
        bc_length = float(self._inf_bc_length_spin.value())
        # Harmonic mode selector
        harm_idx = self._inf_harmonic_combo.currentIndex()
        harmonic_mode = ['H1', 'H2', 'H1+H2'][harm_idx]
        use_snr_floor = bool(getattr(self, '_inf_snr_floor_cb', None)
                              and self._inf_snr_floor_cb.isChecked())
        self._inf_run_btn.setEnabled(False)
        self._inf_diag_btn.setEnabled(False)
        self._inf_compare_btn.setEnabled(False)
        bc_tag = (f",  ℓ = {bc_length:.0f} px (smooth BC prior)"
                  if bc_length > 0 else "")
        sb_tag = ("σ_b = flat (10⁴ Pa)" if flat_b_prior
                    else f"σ_b = {sigma_b:.1f} Pa")
        self._inf_status_lbl.setText(
            f"<small>Running additive empirical-σ inference on tile "
            f"{tile_id}  ({sb_tag},  α = {alpha:+.2f},  "
            f"harmonic = {harmonic_mode}{bc_tag}) …</small>")
        QApplication.processEvents()
        t0 = time.time()
        try:
            res = self._inference_run(int(tile_id),
                                         sigma_b_Pa=sigma_b,
                                         alpha=alpha,
                                         bc_length_px=bc_length,
                                         harmonic_mode=harmonic_mode,
                                         use_snr_floor=use_snr_floor)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Inference failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        elapsed = time.time() - t0
        if res.get('error'):
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>{res['error']}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        # Format result summary
        prob = res['prob']
        n_edges = len(prob['edges_in'])
        n_valid_h1 = int((prob['valid_dc'] & prob['valid_h1']).sum())
        n_B = len(prob['boundary_nodes'])
        bc_len = res.get('bc_length_px', 0.0)
        bc_str = (f"  ·  ℓ = {bc_len:.0f} px (smooth BC)"
                  if bc_len > 0 else "  ·  ℓ = 0  (independent BCs)")
        harm_str = f"  ·  harmonic = {res.get('harmonic_mode', 'H1')}"
        lines = [
            f"Tile {res['tile_id']}  ·  α = {res['alpha']:+.2f}  ·  "
            f"σ_b = {res['sigma_b']:.1f} Pa{bc_str}{harm_str}",
            "",
            f"  D̂           = {res['D_hat']:.3e}  /Pa "
            f"(at R_ref = {res['R_ref_um']:.1f} µm)",
            f"  3-nat support region = [{res['lo']:.2e}, {res['hi']:.2e}]",
            f"  90% HPD CI            = [{res.get('lo_90', float('nan')):.2e}, "
            f"{res.get('hi_90', float('nan')):.2e}]  "
            f"(width {res.get('width_90', float('nan')):.2f} dec)",
            f"  CI width    = {res['width']:.2f} decades  ·  {res['tier']}",
            "",
            # Final IDENTIFICATION verdict: simple, single threshold on Δ@0
            f"  Δ@0          = {res['delta_at_zero']:+.1f} nats  "
            f"({'IDENTIFIED' if res['delta_at_zero'] >= 10 else 'not identified'})",
            "",
            "  ── Inferred noise model:  σ²_j = â² + ŝ²·|Q_j|² ──",
            f"  â  (measurement floor)     = {res['a_at_map']*1e12:.4f} nL/s",
            f"  ŝ  (proportional misfit)   = {res['s_at_map']:.3f}  "
            f"(dimensionless)",
            f"  |Q|* = â / ŝ (crossover)   = {res['Q_crossover']*1e12:.4f} nL/s",
            "           below |Q|* = measurement-noise-dominated edges",
            "",
            f"  n_meas       = {n_valid_h1}  (of {n_edges} carve edges)",
            (f"  n_boundary   = {n_B}  "
             f"({res.get('n_av_promoted', 0)} A/V anchors promoted from interior)"
             if res.get('n_av_promoted', 0)
             else f"  n_boundary   = {n_B}"),
            f"  median SNR  = {res['snr_median']:+.1f} dB",
            f"  R range     = [{res['R_min_um']:.1f}, "
            f"{res['R_max_um']:.1f}] µm (median {res['R_ref_um']:.1f})",
        ]
        # ── Bayesian information summary (per-edge Fisher info) ──
        fpe_logD = res.get('fisher_per_edge_logD')
        if fpe_logD is not None:
            fpe_logD = np.asarray(fpe_logD)
            I_total = float(fpe_logD.sum())
            prior_prec = 1.0 / (np.log(10.0) ** 2)
            ratio = I_total / max(prior_prec, 1e-30)
            n_eff = (fpe_logD.sum() ** 2) / max((fpe_logD ** 2).sum(), 1e-30)
            sigma_logD = 1.0 / np.sqrt(max(prior_prec + I_total, 1e-30))
            D_lap_lo = res['D_hat'] * np.exp(-1.96 * sigma_logD)
            D_lap_hi = res['D_hat'] * np.exp(+1.96 * sigma_logD)
            lap_w = (np.log10(D_lap_hi / D_lap_lo)
                     if D_lap_lo > 0 else float('inf'))
            # Top-K contributions
            top_k = min(5, fpe_logD.size)
            order = np.argsort(-fpe_logD)
            top_frac = (fpe_logD[order[:top_k]].sum()
                        / max(I_total, 1e-30))
            lines.append("")
            lines.append(
                "  ── Bayesian information (Fisher / Laplace) ──")
            lines.append(
                f"  I_total (log-D)            = {I_total:.2f}")
            lines.append(
                f"  prior precision (1/τ_D²)  = {prior_prec:.3f}")
            lines.append(
                f"  data/prior ratio           = {ratio:.1f}× "
                f"({'data-dominated' if ratio >= 5 else 'mixed' if ratio >= 2 else 'PRIOR-DRIVEN'})")
            lines.append(
                f"  n_effective informative   = {n_eff:.1f}  "
                f"(of {n_valid_h1} measured edges)")
            lines.append(
                f"  top-{top_k} edges contribute {top_frac*100:.0f}% of total Fisher info")
            lines.append(
                f"  Laplace 95% CI              = [{D_lap_lo:.2e}, "
                f"{D_lap_hi:.2e}]  (width {lap_w:.2f} dec)")
        self._inf_results_box.setPlainText("\n".join(lines))
        self._inf_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · D̂ = {res['D_hat']:.2e} /Pa.  "
            f"Click <b>D₀ posterior</b> for the evidence plot, or pick an "
            f"edge in the viewer then <b>Compare</b> for predicted vs "
            f"measured Q(t).</small>")
        self._inference_last_result = (int(tile_id), res)
        self._inf_run_btn.setEnabled(True)
        self._inf_diag_btn.setEnabled(True)
        self._inf_compare_btn.setEnabled(True)
        if not self._inf_show_carve_chk.isChecked():
            self._inf_show_carve_chk.setChecked(True)
        # Draw the inferred boundary forcing on the mosaic as napari
        # layers — Points (size = |b̂_H₁|, color = |b̂_DC|) + Vectors
        # (arrows showing arg(b̂_H₁)).  Persistent on the canvas until
        # the next inference run or a tile change.
        try:
            self._refresh_inference_bc_overlay(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                self._inf_status_lbl.text() +
                f"  <font color='#c33'>(BC overlay failed: "
                f"{type(e).__name__}: {e})</font>")
        # Per-edge Fisher-info / sensitivity overlay
        try:
            self._refresh_inference_fisher_overlay(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                self._inf_status_lbl.text() +
                f"  <font color='#c33'>(Fisher overlay failed: "
                f"{type(e).__name__}: {e})</font>")

    def _on_inference_diagnostic_clicked(self):
        from qtpy.QtWidgets import QApplication
        if self._inference_last_result is None:
            return
        tile_id, res = self._inference_last_result
        self._inf_status_lbl.setText(
            "<small>Rendering D₀ posterior …</small>")
        QApplication.processEvents()
        try:
            self._inference_render_diagnostic(tile_id, res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Diagnostic render failed: "
                f"{type(e).__name__}: {e}</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>D₀ posterior for tile {tile_id} rendered.</small>")

    def _on_inference_compare_clicked(self):
        """Compare measured vs predicted Q(t) for the currently-selected
        edge, using the MAP D̂ + b̂ from the last inference run."""
        from qtpy.QtWidgets import QApplication
        if self._inference_last_result is None:
            self._inf_status_lbl.setText(
                "<small><font color='#c33'>No inference result — run "
                "the inference first.</font></small>")
            return
        if self._selected_edge is None:
            self._inf_status_lbl.setText(
                "<small><font color='#c33'>No edge selected — click an "
                "edge in the viewer first (the Time-series tab updates "
                "the selection).</font></small>")
            return
        tile_id, res = self._inference_last_result
        u, v = self._selected_edge
        # Look up edge in the carve's edges_in
        edges_in = res['prob']['edges_in']
        edge_idx = None
        for i, (a, b) in enumerate(edges_in):
            if (a, b) == (u, v) or (b, a) == (u, v):
                edge_idx = i; break
        if edge_idx is None:
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Selected edge ({u}, {v}) "
                f"is not in tile {tile_id}'s carve.</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>Rendering compare plot for edge ({u}, {v}) "
            f"in tile {tile_id} …</small>")
        QApplication.processEvents()
        try:
            self._inference_render_compare(tile_id, res, edge_idx)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Compare render failed: "
                f"{type(e).__name__}: {e}</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>Compare plot for edge ({u}, {v}) rendered.</small>")

    def _inference_render_compare(self, tile_id, result, edge_idx):
        """Three-panel comparison for one edge:
          (1) Phasor — measured H₁ vs predicted H₁ (complex plane),
              with the per-edge σ_meas as a circle around the measurement.
          (2) Single-cycle overlay — measured Q(t) cycle-folded (binned)
              vs reconstructed Q_DC + |H₁|·cos(ωt + φ) at MAP.
          (3) Full raw Q(t) — measured vs reconstructed (DC + H₁).
        """
        import matplotlib.pyplot as plt
        import tempfile
        prob = result['prob']
        edges_in = prob['edges_in']
        u, v = edges_in[edge_idx]
        ed = self.G.edges[u, v]
        # Pick the per-tile measurement for this edge
        m = next((mm for mm in (ed.get('measurements_piv') or [])
                   if MeasurementView(mm).tile_id == int(tile_id)), None)
        mv = MeasurementView(m) if m is not None else None
        # v4 migration: was m.get('Q_t')
        if m is None or mv.Q_t is None:
            raise RuntimeError(
                f"Edge ({u}, {v}) has no Q_t measurement on tile {tile_id}")
        # Canonical convention: m['Q_t'], m['Q_DC'], m['Q_H1_amp'] are
        # stored in the FLOW_FROM → FLOW_TO direction.  The model
        # works in the (u, v) graph-storage order.  When flow_from→
        # flow_to is opposite (u, v), apply `sign = −1` to bring the
        # measurement into the model's convention so the comparison
        # is apples-to-apples.
        # ed.get('flow_from')/('flow_to') is an edge-level attr, not the
        # _piv-promoted one; kept.
        ff, ft = ed.get('flow_from'), ed.get('flow_to')
        sign = 1.0 if (ff == u and ft == v) else -1.0
        Q_t = np.asarray(mv.Q_t, dtype=float) * sign  # v4: was m['Q_t']
        # f₀: prefer m's harmonic-fit frequency (matches the canonical
        # Q_H1_amp / Q_H1_phi); fall back to tile-level f₀ then 2.5.
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        # v4 migration: was m.get('f0_hz')
        f0 = float(mv.harm.f0_hz
                   or tile_f0_map.get(int(tile_id))
                   or self.G.graph.get('f0_hz')
                   or 2.5)
        dt = 1.0 / 250.0
        N = len(Q_t)
        t_axis = np.arange(N) * dt
        amp = float(m.get('Q_H1_amp', 0.0))
        phi = float(m.get('Q_H1_phi', 0.0))
        Q_DC_meas_nL = float(m.get('Q_DC', 0.0)) * sign
        Q_H1_meas_nL = amp * np.exp(1j * phi) * sign
        # H₂ (if available on this edge)
        amp_H2 = m.get('Q_H2_amp', None)
        phi_H2 = m.get('Q_H2_phi', None)
        if (amp_H2 is not None and phi_H2 is not None
                and np.isfinite(amp_H2) and np.isfinite(phi_H2)):
            amp_H2 = float(amp_H2); phi_H2 = float(phi_H2)
            Q_H2_meas_nL = amp_H2 * np.exp(1j * phi_H2) * sign
            has_H2 = True
        else:
            amp_H2 = float('nan'); phi_H2 = float('nan')
            Q_H2_meas_nL = complex('nan')
            has_H2 = False
        SCALE = 1e12   # m³/s → nL/s
        # Inferred Q̄ from the DC solve (b̂_DC).  Q_DC_pred_full is in m³/s
        # in the (u,v) edge-orientation convention used by the inference;
        # convert to nL/s and back into the canonical (flow_from→flow_to)
        # convention so it lines up with Q_DC_meas_nL above.
        Q_DC_pred_full = result.get('Q_DC_pred_full')
        if Q_DC_pred_full is not None and edge_idx < len(Q_DC_pred_full):
            Q_DC_pred_nL = float(Q_DC_pred_full[edge_idx]) * SCALE * sign
        else:
            Q_DC_pred_nL = float('nan')
        Q_H1_pred_nL = result['Q_H1_pred'][edge_idx] * SCALE
        # Predicted H₂ phasor — only populated in H₁+H₂ joint mode
        Q_H2_pred_arr = result.get('Q_H2_pred')
        if (Q_H2_pred_arr is not None
                and edge_idx < len(Q_H2_pred_arr)
                and np.all(np.isfinite([Q_H2_pred_arr[edge_idx].real,
                                          Q_H2_pred_arr[edge_idx].imag]))):
            Q_H2_pred_nL = complex(Q_H2_pred_arr[edge_idx]) * SCALE
            has_H2_pred = True
        else:
            Q_H2_pred_nL = complex('nan')
            has_H2_pred = False
        # Per-edge σ for H₁ under the inferred additive+proportional model:
        #   σ²_total = â² + ŝ²·|Q|²
        # â is the absolute measurement floor (in m³/s); ŝ² is the
        # proportional misfit.  In display units (nL/s):
        a_nL = result['a_at_map'] * SCALE         # m³/s → nL/s
        s_map = result['s_at_map']
        sigma_H1_total_nL = np.sqrt(
            a_nL**2 + (s_map * abs(Q_H1_meas_nL))**2)
        # For reference: the original SNR-derived measurement σ.  This is
        # no longer used in the inference (γ → 0 from the inferred-shape
        # test), but display it alongside for diagnostic.
        snr_h1_db = m.get('Q_H1_snr_db', np.nan)
        if np.isfinite(snr_h1_db):
            sigma_H1_meas_nL = abs(Q_H1_meas_nL) * 10.0 ** (-snr_h1_db / 20.0)
        else:
            sigma_H1_meas_nL = np.nan
        sigma_H1_nL = sigma_H1_meas_nL   # kept for downstream lim computation
        # ── Figure ──
        fig = plt.figure(figsize=(14, 14), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])
        # Phasor panel
        ax = fig.add_subplot(gs[0, 0])
        # Measurement with σ circle
        ax.plot([0, Q_H1_meas_nL.real], [0, Q_H1_meas_nL.imag],
                 color='C0', lw=2.0, label=f'measured |H₁| = {abs(Q_H1_meas_nL):.3f} nL/s')
        ax.plot(Q_H1_meas_nL.real, Q_H1_meas_nL.imag, 'o',
                 color='C0', ms=10)
        theta = np.linspace(0, 2*np.pi, 80)
        if np.isfinite(sigma_H1_meas_nL) and sigma_H1_meas_nL > 0:
            ax.plot(Q_H1_meas_nL.real + sigma_H1_meas_nL * np.cos(theta),
                     Q_H1_meas_nL.imag + sigma_H1_meas_nL * np.sin(theta),
                     color='C0', lw=0.8, ls=':', alpha=0.7,
                     label=f'σ_meas (PIV noise) = {sigma_H1_meas_nL:.3f} nL/s '
                            f'(SNR {snr_h1_db:.1f} dB)')
        if np.isfinite(sigma_H1_total_nL) and sigma_H1_total_nL > 0:
            ax.plot(Q_H1_meas_nL.real + sigma_H1_total_nL * np.cos(theta),
                     Q_H1_meas_nL.imag + sigma_H1_total_nL * np.sin(theta),
                     color='C0', lw=1.2, ls='--', alpha=0.6,
                     label=f'σ_total (meas + ŝ²·|Q|) = '
                            f'{sigma_H1_total_nL:.3f} nL/s')
        # Prediction
        ax.plot([0, Q_H1_pred_nL.real], [0, Q_H1_pred_nL.imag],
                 color='C3', lw=2.0,
                 label=f'predicted |H₁| = {abs(Q_H1_pred_nL):.3f} nL/s')
        ax.plot(Q_H1_pred_nL.real, Q_H1_pred_nL.imag, '^',
                 color='C3', ms=10)
        # H₂ phasors (when available) — drawn fainter / dotted to keep
        # the H₁ phasor as the dominant visual.
        if has_H2:
            ax.plot([0, Q_H2_meas_nL.real], [0, Q_H2_meas_nL.imag],
                     color='C0', lw=1.0, ls=':', alpha=0.7,
                     label=f'measured |H₂| = {abs(Q_H2_meas_nL):.3f} nL/s')
            ax.plot(Q_H2_meas_nL.real, Q_H2_meas_nL.imag, 's',
                     color='C0', ms=7, mfc='none', mew=1.5)
        if has_H2_pred:
            ax.plot([0, Q_H2_pred_nL.real], [0, Q_H2_pred_nL.imag],
                     color='C3', lw=1.0, ls=':', alpha=0.7,
                     label=f'predicted |H₂| = {abs(Q_H2_pred_nL):.3f} nL/s')
            ax.plot(Q_H2_pred_nL.real, Q_H2_pred_nL.imag, 's',
                     color='C3', ms=7, mfc='none', mew=1.5)
        # Origin marker
        ax.plot(0, 0, 'k+', ms=8)
        lim_candidates = [abs(Q_H1_meas_nL), abs(Q_H1_pred_nL),
                            sigma_H1_total_nL
                            if np.isfinite(sigma_H1_total_nL) else 0,
                            sigma_H1_meas_nL
                            if np.isfinite(sigma_H1_meas_nL) else 0]
        if has_H2:    lim_candidates.append(abs(Q_H2_meas_nL))
        if has_H2_pred: lim_candidates.append(abs(Q_H2_pred_nL))
        lim = 1.4 * max(lim_candidates)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('Re Q_H₁  (nL/s)')
        ax.set_ylabel('Im Q_H₁  (nL/s)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.axhline(0, color='#888', lw=0.5)
        ax.axvline(0, color='#888', lw=0.5)
        dphi_deg = np.degrees(np.angle(Q_H1_pred_nL) -
                                np.angle(Q_H1_meas_nL))
        ax.set_title(
            f"H₁ phasor — measured vs predicted\n"
            f"|pred|/|meas| = {abs(Q_H1_pred_nL)/max(abs(Q_H1_meas_nL),1e-12):.2f}  ·  "
            f"Δφ = {((dphi_deg + 180) % 360) - 180:+.1f}°",
            fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

        # Cycle-folded overlay
        # ── Carve + boundary-node H₁ forcing ──
        # Spatial map of the inference carve.  Edges in light grey;
        # selected edge in red.  At each boundary node:
        #   color  → |Q_DC| throughput  (sum of measured |Q_DC| on
        #            carve edges adjacent to the BC node, nL/s).
        #            Tells you which boundary nodes are "DC-busy" —
        #            i.e. carry meaningful flow.
        #   size   → |b̂_n|  amplitude of the inferred H₁ pressure
        #            forcing at that BC node, Pa.
        #   arrow  → arg(b̂_n)  the H₁ phase, drawn as an unambiguous
        #            arrow (one direction only — no 180° flip).
        ax = fig.add_subplot(gs[0, 1])
        prob = result['prob']
        boundary_nodes = prob['boundary_nodes']
        edges_in = prob['edges_in']
        b_complex = result['b_complex']
        Q_DC_obs = prob['Q_DC_obs']           # SI m³/s (signed)
        valid_dc = prob['valid_dc']
        # Gather node positions for all nodes touched by the carve
        all_nodes_xy = {}
        for (uu, vv) in edges_in:
            for nn in (uu, vv):
                if nn not in all_nodes_xy:
                    nd = self.G.nodes[nn]
                    all_nodes_xy[nn] = (float(nd.get('x', 0)),
                                          float(nd.get('y', 0)))
        # Draw all carve edges (light grey, selected edge highlighted)
        sel_drawn = False
        for (uu, vv) in edges_in:
            x1, y1 = all_nodes_xy[uu]; x2, y2 = all_nodes_xy[vv]
            is_selected = ((uu, vv) == (u, v) or (vv, uu) == (u, v))
            if is_selected:
                ax.plot([x1, x2], [y1, y2], color='red',
                         lw=3.5, zorder=4, alpha=0.95,
                         label='selected edge' if not sel_drawn else None)
                sel_drawn = True
            else:
                ax.plot([x1, x2], [y1, y2], color='#bbb',
                         lw=0.6, alpha=0.6, zorder=1)
        # DC throughput per BC node: sum of |Q_DC| over carve edges
        # adjacent to the BC node, converted to nL/s for display.
        SCALE_nL = 1e12
        boundary_set = set(boundary_nodes)
        node_DC_through = {n: 0.0 for n in boundary_nodes}
        for i, (uu, vv) in enumerate(edges_in):
            if not valid_dc[i]: continue
            q = abs(Q_DC_obs[i]) * SCALE_nL
            if uu in boundary_set: node_DC_through[uu] += q
            if vv in boundary_set: node_DC_through[vv] += q
        DC_values = np.array([node_DC_through[n] for n in boundary_nodes])
        # b̂ amplitude + phase
        b_amp = np.abs(b_complex)
        amp_max = max(b_amp.max(), 1e-12)
        bx = np.array([all_nodes_xy[n][0] for n in boundary_nodes])
        by = np.array([all_nodes_xy[n][1] for n in boundary_nodes])
        # Marker size ∝ |b̂|.  Note: marker AREA scales with the 's'
        # arg in scatter, so to make perceived diameter ∝ |b̂| we want
        # s ∝ |b̂|² .  We compromise with s ∝ |b̂| so even small forcings
        # remain visible.
        sizes = 40 + 350.0 * (b_amp / amp_max)
        # Color from DC throughput on a sequential cmap
        sc = ax.scatter(bx, by, s=sizes, c=DC_values, cmap='viridis',
                         edgecolor='black', linewidth=0.7, zorder=5,
                         alpha=0.95)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                             label='Σ |Q_DC| adjacent (nL/s)')
        # Unambiguous phase arrows.  One direction only — origin at BC
        # node, head at (cos φ, sin φ) — no 180° flip risk.  Length
        # also scaled by |b̂| so small forcings get short arrows.
        ax_span = max(bx.max() - bx.min(), by.max() - by.min(),
                        1.0) if len(bx) else 1.0
        arrow_max_len = 0.06 * ax_span
        for k_node, n in enumerate(boundary_nodes):
            if b_amp[k_node] == 0: continue
            x0, y0 = all_nodes_xy[n]
            ang = np.angle(b_complex[k_node])
            scale = b_amp[k_node] / amp_max
            dx_arrow = arrow_max_len * scale * np.cos(ang)
            dy_arrow = arrow_max_len * scale * np.sin(ang)
            ax.annotate('', xy=(x0 + dx_arrow, y0 + dy_arrow),
                         xytext=(x0, y0),
                         arrowprops=dict(arrowstyle='-|>', color='black',
                                          lw=1.4, alpha=0.9,
                                          mutation_scale=10),
                         zorder=6)
        ax.set_aspect('equal')
        ax.set_xlabel('x (graph px)'); ax.set_ylabel('y (graph px)')
        ax.set_title(
            f"Carve + BC forcing  ({len(boundary_nodes)} BC nodes)\n"
            f"size ∝ |b̂_H₁|  (max = {amp_max:.2f} Pa)  ·  "
            f"arrow ∝ phase  ·  color = ΣDC throughput",
            fontsize=9)
        if sel_drawn:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ── Single-cycle overlay (was gs[0,1] → now gs[1,:]) ──
        ax = fig.add_subplot(gs[1, :])
        T_period = 1.0 / f0
        # Measured cycle-fold: bin Q_t by cycle phase
        cycle_phase = (t_axis * f0) % 1.0
        N_bins = 32
        bins = np.linspace(0, 1, N_bins + 1)
        binned = np.full(N_bins, np.nan)
        for k in range(N_bins):
            mask = (cycle_phase >= bins[k]) & (cycle_phase < bins[k+1])
            if mask.any():
                binned[k] = np.mean(Q_t[mask])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(bin_centers, binned, 'o-', color='C0',
                 lw=1.5, ms=5, label='measured (cycle-folded)')
        # Reconstructed: Q_DC + |H1|·cos(2πf₀t + φ)  [ + H₂ if available ]
        t_one = np.linspace(0, T_period, 200)
        # Measured reconstruction: H₁ alone (and H₁+H₂ if measured)
        Q_recon_meas = Q_DC_meas_nL + amp * np.cos(2*np.pi*f0*t_one + phi) * sign
        ax.plot(t_one / T_period, Q_recon_meas, color='C0',
                 lw=1.0, ls='--', alpha=0.7,
                 label='measured Q̄ + H₁')
        if has_H2:
            Q_recon_meas_H12 = Q_recon_meas + amp_H2 * np.cos(
                2 * np.pi * (2 * f0) * t_one + phi_H2) * sign
            ax.plot(t_one / T_period, Q_recon_meas_H12, color='C0',
                     lw=1.2, alpha=0.85,
                     label='measured Q̄ + H₁ + H₂')
        # Predicted reconstruction: Q̄_pred + H₁_pred (+ H₂_pred if joint)
        Q_pred_dc = Q_DC_pred_nL if np.isfinite(Q_DC_pred_nL) else Q_DC_meas_nL
        Q_recon_pred = Q_pred_dc + abs(Q_H1_pred_nL) * np.cos(
            2*np.pi*f0*t_one + np.angle(Q_H1_pred_nL))
        ax.plot(t_one / T_period, Q_recon_pred, color='C3',
                 lw=1.6, ls='--', alpha=0.85,
                 label='predicted Q̄ + H₁ at MAP')
        if has_H2_pred:
            Q_recon_pred_H12 = Q_recon_pred + abs(Q_H2_pred_nL) * np.cos(
                2 * np.pi * (2 * f0) * t_one + np.angle(Q_H2_pred_nL))
            ax.plot(t_one / T_period, Q_recon_pred_H12, color='C3',
                     lw=2.0,
                     label='predicted Q̄ + H₁ + H₂ at MAP')
        ax.set_xlabel('cycle fraction')
        ax.set_ylabel('Q (nL/s)')
        ax.set_title('Single-cycle: measured vs predicted', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        # ── Full Q(t) raw ──
        # Show three traces:
        #   - measured Q(t) raw
        #   - measured reconstruction = Q̄_meas + |H₁|_meas·cos(...)
        #   - PREDICTED reconstruction = Q̄_pred + |H₁|_pred·cos(...)
        # where Q̄_pred = T_DC[e]·b̂_DC is the INFERRED DC mean (not the
        # measured one).  Both Q̄_pred and |H₁|_pred come from the same
        # boundary-pressure fit, so the red curve is what the model
        # actually predicts at this edge.  The horizontal dashed lines
        # mark the two DC means side-by-side for visual comparison.
        ax = fig.add_subplot(gs[2, :])
        ax.plot(t_axis, Q_t, color='C0', lw=0.6, alpha=0.85,
                 label='measured Q(t) raw')
        # Measured H1 reconstruction (uses measured Q̄)
        Q_meas_recon = Q_DC_meas_nL + amp * np.cos(
            2*np.pi*f0*t_axis + phi) * sign
        ax.plot(t_axis, Q_meas_recon, color='C0', lw=1.2, ls=':',
                 label=f'measured Q̄ + H₁ fit  (Q̄_meas = {Q_DC_meas_nL:+.4f} nL/s)')
        # Measured H₁+H₂ reconstruction — visible when H₂ is measured
        if has_H2:
            Q_meas_recon_H12 = Q_meas_recon + amp_H2 * np.cos(
                2 * np.pi * (2 * f0) * t_axis + phi_H2) * sign
            ax.plot(t_axis, Q_meas_recon_H12, color='C0', lw=1.3, ls='-',
                     alpha=0.85,
                     label=f'measured Q̄ + H₁ + H₂  (|H₂| = {amp_H2:.4f} nL/s)')
        # PREDICTED reconstruction: inferred Q̄ + inferred H1 (+ H₂ in joint)
        Q_pred_full = Q_DC_pred_nL + abs(Q_H1_pred_nL) * np.cos(
            2*np.pi*f0*t_axis + np.angle(Q_H1_pred_nL))
        ax.plot(t_axis, Q_pred_full, color='C3', lw=1.4, ls='--', alpha=0.85,
                 label=f'predicted Q̄ + H₁ at MAP  (Q̄_pred = {Q_DC_pred_nL:+.4f} nL/s)')
        if has_H2_pred:
            Q_pred_full_H12 = Q_pred_full + abs(Q_H2_pred_nL) * np.cos(
                2 * np.pi * (2 * f0) * t_axis + np.angle(Q_H2_pred_nL))
            ax.plot(t_axis, Q_pred_full_H12, color='C3', lw=1.8,
                     label=f'predicted Q̄ + H₁ + H₂ at MAP  '
                            f'(|H₂_pred| = {abs(Q_H2_pred_nL):.4f} nL/s)')
        # DC reference lines
        ax.axhline(Q_DC_meas_nL, color='C0', ls='-', lw=0.6, alpha=0.4)
        if np.isfinite(Q_DC_pred_nL):
            ax.axhline(Q_DC_pred_nL, color='C3', ls='-', lw=0.6, alpha=0.4)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Q (nL/s)')
        # Title shows the DC residual at this edge
        if np.isfinite(Q_DC_pred_nL):
            dc_resid = Q_DC_meas_nL - Q_DC_pred_nL
            ax.set_title(f'Full Q(t)  ·  ΔQ̄ (meas − pred) = '
                         f'{dc_resid:+.4f} nL/s', fontsize=10)
        else:
            ax.set_title('Full Q(t)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Tile {tile_id}, edge ({u}, {v})  —  "
            f"D̂ = {result['D_hat']:.2e} /Pa  ·  "
            f"α = {result['alpha']:+.2f}  ·  f₀ = {f0:.2f} Hz",
            fontsize=11, fontweight='bold')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as _tmp:
            _tmp_path = _tmp.name
        fig.savefig(_tmp_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        self._show_diagnostic_png(_tmp_path, int(tile_id), result)

    # ══════════════════════════════════════════════════════════════════
    # End of inference adapter region.
    # ══════════════════════════════════════════════════════════════════

    def _show_bc_waveforms(self):
        """Pop a matplotlib dialog with two panels: all A (source) bc_Q_t
        overlaid on the left, all V (sink) bc_Q_t overlaid on the right.
        Reads straight from each boundary node's `bc_Q_t` — no phasor
        reconstruction, no simulation."""
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigCanvas,
            NavigationToolbar2QT as _NavToolbar)
        FS = 250.0
        src_nodes = [n for n in self._source_nodes
                     if self.G.nodes[n].get('bc_Q_t') is not None]
        snk_nodes = [n for n in self._sink_nodes
                     if self.G.nodes[n].get('bc_Q_t') is not None]
        if not src_nodes and not snk_nodes:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self.viewer.window._qt_window if self.viewer else None,
                "No BC waveforms",
                "No boundary nodes carry bc_Q_t.  Run a batch analyze "
                "that populates the recursive-KCL BC harmonics first.")
            return
        fig = Figure(figsize=(11, 4.4), tight_layout=True)
        for i, (nodes, title, colour) in enumerate([
                (src_nodes, "Sources (A) — bc_Q_t", "C0"),
                (snk_nodes, "Sinks (V) — bc_Q_t",   "C3")]):
            ax = fig.add_subplot(1, 2, i + 1)
            for n in nodes:
                qt = np.asarray(self.G.nodes[n]['bc_Q_t'], dtype=float)
                t = np.arange(len(qt)) / FS
                mean = float(np.mean(qt))
                ax.plot(t, qt, colour + '-', lw=1.5, alpha=0.85,
                        label=f"node {n}  ⟨Q⟩ = {mean:+.3f} nL/s")
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("t (s)"); ax.set_ylabel("Q (nL/s)")
            ax.grid(alpha=0.3)
            if nodes:
                ax.legend(fontsize=8, loc='upper right')
        dlg = QDialog()
        dlg.setWindowTitle("Boundary Q(t) waveforms  (from bc_Q_t)")
        dlg.resize(1050, 480)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            "<i>Measured Q(t) at every boundary node — straight from "
            "<tt>bc_Q_t</tt>.  Not a sim output.</i>"))
        canvas = _FigCanvas(fig); v.addWidget(canvas)
        v.addWidget(_NavToolbar(canvas, dlg))
        dlg.exec_()

    def _detect_sim_f0_hz(self) -> float:
        """Pick the solver's default heart rate.  Priority:
        1. `G.graph['tile_f0_piv'][14]` — tile 14 is the BC / phase gauge
           (every bc_harmonics phasor is expressed at this f0).
        2. `G.graph['f0_canonical_hz']` — the normalization-plateau f0.
        3. `G.graph['sim_f0_hz']` if set by the pipeline.
        4. Median of `G.graph['tile_f0_piv']` (fallback).
        5. Median of `G.graph['tile_f0s']` (legacy kymograph).
        6. Constant 2.5 Hz.
        """
        tile_f0 = self.G.graph.get('tile_f0_piv') or {}
        for k in (14, '14'):
            v = tile_f0.get(k)
            if v is not None:
                try:
                    v = float(v)
                    if np.isfinite(v) and v > 0:
                        return v
                except (TypeError, ValueError):
                    pass
        f0_can = self.G.graph.get('f0_canonical_hz')
        if f0_can is not None:
            try:
                f0_can = float(f0_can)
                if np.isfinite(f0_can) and f0_can > 0:
                    return f0_can
            except (TypeError, ValueError):
                pass
        f0 = self.G.graph.get('sim_f0_hz')
        if f0 is not None:
            try:
                f0 = float(f0)
                if np.isfinite(f0) and f0 > 0:
                    return f0
            except (TypeError, ValueError):
                pass
        for key in ('tile_f0_piv', 'tile_f0s'):
            d = self.G.graph.get(key) or {}
            vals = []
            for v in d.values():
                try:
                    vv = float(v)
                    if np.isfinite(vv) and vv > 1.0:
                        vals.append(vv)
                except (TypeError, ValueError):
                    pass
            if vals:
                return float(np.median(vals))
        return 2.5

    def _find_inflow_stem(self, a_node) -> List[Tuple[int, int]]:
        """Walk from a boundary node through degree-2 nodes to the
        first node whose degree ≠ 2 (branch point or dead end).
        Returns the ordered list of edges traversed."""
        edges: List[Tuple[int, int]] = []
        prev = None
        curr = a_node
        while True:
            if curr != a_node and self.G.degree(curr) > 2:
                break  # branch point
            nbrs = [n for n in self.G.neighbors(curr) if n != prev]
            if not nbrs:
                break  # dead end
            nxt = nbrs[0]
            edges.append((curr, nxt))
            prev = curr
            curr = nxt
            # Cap to avoid pathological infinite loops on self-loops.
            if len(edges) > 64:
                break
        return edges

    def _stem_poiseuille_resistance(self, stem_edges) -> float:
        """Sum Σ 8μ L_e / (π R_e⁴) along a stem (Pa·s/m³)."""
        from ..analyze.constants import PX_SIZE_UM
        from ..analyze.constants import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        total = 0.0
        for (u, v) in stem_edges:
            d = self.G.edges[u, v]
            try:
                R_px = float(d.get('radius'))
                L_px = float(d.get('length'))
            except (TypeError, ValueError):
                continue
            if not (R_px > 0 and L_px > 0):
                continue
            R_m = R_px * px_to_m
            L_m = L_px * px_to_m
            total += 8.0 * MU_DEFAULT * L_m / (np.pi * R_m ** 4)
        return total

    def _classwave_bc_override(self, bc_signed):
        """Phenomenological class-waveform boundary condition.

        For each class (sources=A, sinks=V), build per-port BC phasors
            Q_i^(0) = measured DC mean   (defines the share w_i)
            Q_i^(n) = w_i^(0) · A_c^(n)   for n ≥ 1
        where w_i = |Q_i^(0)| / Σ_class |Q_j^(0)| and A_c^(n) is the
        weighted least-squares class amplitude
            A_c^(n) = Σ_i w_i Q_i^(n) / Σ_i w_i² .

        `bc_signed`: node → complex [DC, H1, H2, H3] in into-network
        convention (sinks already sign-flipped).  Returns the same shape.
        This is the boundary-only model adopted June 2026 (see
        scripts/sim_compartment.py); the network solves the interior but
        interior AC is treated as indicative only.
        """
        out: Dict[int, np.ndarray] = {}
        for members_all in (self._source_nodes, self._sink_nodes):
            members = [n for n in members_all if n in bc_signed]
            if not members:
                continue
            dc = np.array([abs(complex(bc_signed[n][0])) for n in members])
            tot = float(dc.sum())
            w = dc / tot if tot > 0 else np.ones(len(members)) / len(members)
            for n in members:
                out[n] = np.asarray(bc_signed[n], dtype=complex).copy()
            n_harm = min(len(bc_signed[n]) for n in members)
            for h in range(1, n_harm):
                q = np.array([complex(bc_signed[n][h]) for n in members])
                A_c = np.sum(w * q) / max(float(np.sum(w ** 2)), 1e-30)
                for k, n in enumerate(members):
                    out[n][h] = w[k] * A_c
        return out

    def _run_simulation(self):
        """Forward transmission-line solve with a prescribed-Q boundary.

        Boundary phasors come from the selected BC source:
          • Class-waveform (default): Q_i^(n) = w_i^(0)·A_c^(n) — each port's
            DC-flow share times its class waveform, fit from bc_harmonics.
          • Measured: per-node bc_harmonics as-is.
          • Custom: user-specified per-class waveform.
        Results are written to the in-memory `_sim_tmp_*` fields.
        """
        if not self._source_nodes and not self._sink_nodes:
            self._sim_status_lbl.setText(
                "<small><font color='#c33'>No boundary nodes on this "
                "graph (need `boundary_type` set).  Cannot run.</font>"
                "</small>")
            return
        # Boundary-condition mode.
        dirichlet_tie = bool(getattr(self, '_sim_bc_radio_dirichlet', None)
                              is not None
                              and self._sim_bc_radio_dirichlet.isChecked())
        equal_flux_A = bool(getattr(self, '_sim_bc_radio_equalflux', None)
                             is not None
                             and self._sim_bc_radio_equalflux.isChecked())
        classwave = bool(getattr(self, '_sim_bc_radio_classwave', None)
                         is not None
                         and self._sim_bc_radio_classwave.isChecked())
        # Else: legacy prescribed-Q path (unchanged) ----------------------
        import time
        from qtpy.QtWidgets import QApplication
        D = (float(self._sim_D_mantissa.value())
             * (10.0 ** int(self._sim_D_exponent.value())))
        eta = float(getattr(self, '_sim_eta_spin', None).value()
                     if getattr(self, '_sim_eta_spin', None) is not None else 0.0)
        # Install a monkey-patch on _vessel_admittance so every solver call
        # made in this run sees D_eff(ω) = D · (1 − iη). Uses a class-level
        # pristine reference so repeated runs don't double-wrap.
        # Matches lock_D6p5e4_eta0p4.py's convention.
        from ..analyze import transmission_line as _tl
        if not hasattr(type(self), '_pristine_vessel_admittance'):
            type(self)._pristine_vessel_admittance = _tl._vessel_admittance
        _orig_va = type(self)._pristine_vessel_admittance
        _eta_val = eta
        def _patched_vessel_admittance(R_m, L_m, omega, mu=_tl.MU_DEFAULT,
                                         rho=_tl.RHO_BLOOD, D=None):
            D_eff = (D if D is not None else 1e-6) * (1.0 - 1j * _eta_val)
            return _orig_va(R_m, L_m, omega, mu, rho, D_eff)
        _tl._vessel_admittance = _patched_vessel_admittance
        self._sim_last_eta = _eta_val
        # Re-detect f0 on every run so mid-session changes to
        # G.graph['f0_canonical_hz'] (e.g. after a fresh Normalize) are
        # picked up.  Also refresh the cached value used by the label.
        f0 = self._detect_sim_f0_hz()
        self._sim_detected_f0 = f0
        if not np.isfinite(f0):
            f0 = 2.5
        target_flux = float(self._sim_flux_spin.value())
        # Geometry override mode — combo index ⇒ behaviour:
        #   0 → measured            (no override)
        #   1 → uniform radius      (R const, L preserved)
        #   2 → uniform conductance (R AND L const)
        geom_mode_idx = int(self._sim_geom_combo.currentIndex())
        override_R = geom_mode_idx in (1, 2)
        override_L = geom_mode_idx == 2
        geom_tag = (
            "" if geom_mode_idx == 0
            else ", uniform R" if geom_mode_idx == 1
            else ", uniform R+L")
        split_tag = (", equal-split" if self._sim_equal_split_chk.isChecked()
                     else "")
        eqA_tag = (", A-stem eq" if self._sim_equalize_A_chk.isChecked()
                   else "")
        split_tag += eqA_tag
        v_bc_tag = ""
        if classwave:
            bc_src_tag = ", class-waveform BCs"
        elif self._sim_bc_radio_custom.isChecked():
            bc_src_tag = ", CUSTOM BCs"
        else:
            bc_src_tag = ""
        self._sim_run_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Running solve at D = {D:.3g}, f₀ = {f0:.3f} Hz "
            f"flux = {target_flux:.3g} nL/s"
            f"{bc_src_tag}{geom_tag}{split_tag}{v_bc_tag} …</small>")
        QApplication.processEvents()

        # Build bc_harmonics_override.  Stored `bc_harmonics` on this
        # graph carries +outflow at sinks; the solver's "Q into network"
        # convention is +inflow everywhere, so we flip sinks by −1.
        # Two distribution modes:
        #   measured-ratio (default): scale all by `target_flux / |Σ src
        #     DC|` so the relative per-node imbalance is preserved.  The
        #     solver's DC-conservation step may then rescale slightly.
        #   equal-split: set DC = ±target_flux / N per type, keep AC
        #     harmonics measured.  Mass balance exact → no rebalance.
        equal_split = bool(self._sim_equal_split_chk.isChecked())
        custom_bc = bool(self._sim_bc_radio_custom.isChecked())
        bc_signed: Dict[int, np.ndarray] = {}
        sum_src_dc = 0.0
        for n in (self._source_nodes + self._sink_nodes):
            bch = self.G.nodes[n].get('bc_harmonics')
            if bch is None:
                continue
            s = -1.0 if n in self._sink_nodes else 1.0
            arr = s * np.asarray(bch, dtype=complex)
            bc_signed[n] = arr
            if n in self._source_nodes and np.isfinite(arr[0].real):
                sum_src_dc += float(arr[0].real)

        bc_override: Dict[int, np.ndarray] = {}
        if classwave:
            # Phenomenological boundary model: replace each port's measured
            # AC phasors with w_i·A_c (its DC-flow share × the class
            # waveform).  Measured DC is preserved (it defines w_i).
            bc_override = self._classwave_bc_override(bc_signed)
        elif custom_bc:
            # Build BCs from the user-specified waveforms.  A and V
            # specs are now independent — each side has its own amps +
            # phases.  Every A boundary gets the A spec, every V
            # boundary gets the V spec (with sign convention applied so
            # V's Q-into-network is negative-DC).
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            n_src = max(1, sum(1 for n in self._source_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            n_snk = max(1, sum(1 for n in self._sink_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk  # canonical sign flip
            ac_A = [amps_A[i] * Qbar_A * np.exp(1j * phases_A[i])
                    for i in range(3)]
            ac_V = [amps_V[i] * Qbar_V * np.exp(1j * phases_V[i])
                    for i in range(3)]
            A_arr = np.array([Qbar_A] + ac_A, dtype=complex)
            V_arr = np.array([Qbar_V] + ac_V, dtype=complex)
            for n in self._source_nodes:
                bc_override[n] = A_arr.copy()
            for n in self._sink_nodes:
                bc_override[n] = V_arr.copy()
        elif equal_split:
            # Each boundary inside a type (A or V) gets THE SAME
            # waveform.  AC harmonics are the per-type complex average
            # of the measured BCs, so the A↔V phase difference (e.g.
            # ventricular ejection vs atrial suction) is preserved
            # across types but per-node spread within a type is gone.
            src_arrs = [bc_signed[n] for n in self._source_nodes
                        if n in bc_signed]
            snk_arrs = [bc_signed[n] for n in self._sink_nodes
                        if n in bc_signed]
            n_src = max(1, len(src_arrs))
            n_snk = max(1, len(snk_arrs))
            src_avg = (np.mean(np.stack(src_arrs, axis=0), axis=0)
                       if src_arrs else None)
            snk_avg = (np.mean(np.stack(snk_arrs, axis=0), axis=0)
                       if snk_arrs else None)
            if src_avg is not None:
                src_avg = src_avg.copy()
                src_avg[0] = +target_flux / n_src
            if snk_avg is not None:
                snk_avg = snk_avg.copy()
                snk_avg[0] = -target_flux / n_snk
            for n in bc_signed:
                if n in self._source_nodes and src_avg is not None:
                    bc_override[n] = src_avg.copy()
                elif n in self._sink_nodes and snk_avg is not None:
                    bc_override[n] = snk_avg.copy()
        else:
            if abs(sum_src_dc) > 1e-12:
                flux_scale = target_flux / abs(sum_src_dc)
            else:
                flux_scale = 1.0
            for n, arr in bc_signed.items():
                bc_override[n] = flux_scale * arr

        # Geometry overrides: save the keys we touch, override for the
        # duration of the solve, restore in `finally` so the mutation
        # is invisible to the rest of the viewer state.
        R_UNIFORM_PX, L_UNIFORM_PX = 5.0, 50.0
        radius_keys = ('radius', 'radius_px', 'radius_px_true')
        length_keys = ('length', 'length_true', 'path_length_px')
        touched_keys = []
        if override_R:
            touched_keys.extend(radius_keys)
        if override_L:
            touched_keys.extend(length_keys)
        saved_geom: Dict[Tuple[int, int], dict] = {}
        if touched_keys:
            for u, v, d in self.G.edges(data=True):
                saved_geom[(u, v)] = {k: d.get(k) for k in touched_keys}
                if override_R:
                    for k in radius_keys:
                        d[k] = R_UNIFORM_PX
                if override_L:
                    for k in length_keys:
                        d[k] = L_UNIFORM_PX

        # A-stem conductance equalization.  For each A node, walk to
        # the first branch and set every edge in that stem to a
        # per-stem-uniform radius such that Σ 8μL/(πR⁴) = target_R,
        # where target_R is the median across all A stems.  Lengths
        # preserved.  Applied AFTER any uniform-R/L override so the
        # stems get the correct per-stem radius rather than the
        # uniform value.
        equalize_A = bool(self._sim_equalize_A_chk.isChecked())
        if equalize_A and self._source_nodes:
            from ..analyze.constants import PX_SIZE_UM
            from ..analyze.constants import MU_DEFAULT
            px_to_m = PX_SIZE_UM * 1e-6
            stems = {a: self._find_inflow_stem(a)
                     for a in self._source_nodes}
            resistances = {a: self._stem_poiseuille_resistance(e)
                           for a, e in stems.items()}
            finite_R = [r for r in resistances.values()
                        if r > 0 and np.isfinite(r)]
            if finite_R:
                target_R = float(np.median(finite_R))
                for a, edges in stems.items():
                    if not edges:
                        continue
                    L_total_m = 0.0
                    for (u, v) in edges:
                        L_px = self.G.edges[u, v].get('length')
                        if L_px is None:
                            continue
                        try:
                            L_total_m += float(L_px) * px_to_m
                        except (TypeError, ValueError):
                            continue
                    if L_total_m <= 0 or target_R <= 0:
                        continue
                    R_new_m = (8.0 * MU_DEFAULT * L_total_m
                                / (np.pi * target_R)) ** 0.25
                    R_new_px = R_new_m / px_to_m
                    for (u, v) in edges:
                        key = (u, v) if self.G.has_edge(u, v) else (v, u)
                        if key not in saved_geom:
                            saved_geom[key] = {}
                        ed = self.G.edges[u, v]
                        for rk in radius_keys:
                            if rk not in saved_geom[key]:
                                saved_geom[key][rk] = ed.get(rk)
                            ed[rk] = R_new_px

        # V-side BC mode (combo index → behaviour):
        #   0 → Q at V (measured)            — standard Q-BC everywhere
        #   1 → P-match V at DC + AC rescaled — Option B.  Pass 1 sets
        #        P=0 at every V to predict Q̄_V from physics.  Pass 2
        #        prescribes per-V AC = (|measured_n|/|measured_dc|)
        #        × Q̄_V_predicted, with measured phases — i.e. the
        #        mean-flow-normalised measured amplitudes used as
        #        constraints, scaled by the model's DC prediction.
        v_bc_mode = 0   # V-side P-match removed with Norton closure

        def _q_into_node(edge_flows_dict, node):
            """Sum Q on incident edges, signed so result = Q flowing
            INTO `node`.  edge_flows is keyed (u, v) with +Q along
            u→v."""
            n_harmonics = 3
            acc = np.zeros(n_harmonics + 1, dtype=complex)
            for nbr in self.G.neighbors(node):
                if (nbr, node) in edge_flows_dict:
                    acc += np.asarray(edge_flows_dict[(nbr, node)],
                                       dtype=complex)
                elif (node, nbr) in edge_flows_dict:
                    acc -= np.asarray(edge_flows_dict[(node, nbr)],
                                       dtype=complex)
            return acc

        try:
            # Lazy import — keeps viewer startup cheap and avoids
            # pulling in the heavy adaptation module circular chain.
            from ..analyze.transmission_line import solve_transmission_line  # ported 2026-09-30
            t0 = time.time()
            if equal_flux_A:
                # Two-pass equal-flux:
                #   Pass 1: Q(A_i) = Q_A_total / N_A prescribed, P(V) = 0
                #           at every harmonic.  Extract Q̂_DC(V) at each
                #           V node via edge_flows sum.  AC at V from this
                #           pass is a model output (network response to
                #           the AC at A alone).
                #   Pass 2: Q(A_i) unchanged; Q(V_j) = Q̂_DC(V_j) ·
                #           measured PI ratio at V_j (all harmonics).
                #           V AC waveform is now the trusted measurement,
                #           scaled to the DC the geometry predicts.
                Q_A_total = float(getattr(self, '_sim_qa_spin', None).value()
                                    if getattr(self, '_sim_qa_spin', None)
                                    else 0.0)
                N_A = max(1, len(self._source_nodes))
                Q_A_per_port = Q_A_total / N_A

                # A-side BC (same in both passes): PI-preserving from measurement
                def _build_A_bc():
                    out: Dict[int, np.ndarray] = {}
                    for n in self._source_nodes:
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(Q_A_per_port)
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is not None:
                            bch = np.asarray(bch, dtype=complex)
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(Q_A_per_port) * (bch[k] / DC_meas)
                        out[n] = target
                    return out

                # --- Pass 1: Q at A only, V grounded ---
                # Pass 1 always grounds V at P=0 to give a clean geometry
                # prediction of Q_DC at V.  The V-closure choice only
                # affects Pass 2's V handling.
                bc_override_p1 = _build_A_bc()
                result_p1 = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=bc_override_p1,
                    sink_pressure_bc=np.zeros(4, dtype=complex),
                    verbose=False,
                )
                # Same V-closure sub-modes as Dirichlet-tie:
                v_closure_pressure_eq = bool(
                    getattr(self, '_sim_v_closure_pressure', None)
                    is not None
                    and self._sim_v_closure_pressure.isChecked())
                v_closure_tied_eq = bool(
                    getattr(self, '_sim_v_closure_tied', None)
                    is not None
                    and self._sim_v_closure_tied.isChecked())

                # DC-BC diagnostic — Pass-1 predicted V DC vs measured
                if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                        and self._sim_dc_diag_chk.isChecked()):
                    _print_dc_bc_diagnostic(
                        self.G, self._source_nodes, self._sink_nodes,
                        result_p1)

                # --- Build Pass 2 Q-BC ---
                # 'flux': add Q-BC at V (measured PI × predicted DC).
                # 'pressure' / 'tied': keep Q-BC only at A; V handling is
                #                       done via sink_pressure_bc or shim.
                # 'meas_ac': Q-BC at V with target[k] = measured bch[k].
                v_closure_meas_ac_eq = bool(
                    getattr(self, '_sim_v_closure_meas_ac', None) is not None
                    and self._sim_v_closure_meas_ac.isChecked())
                bc_override_p2: Dict[int, np.ndarray] = dict(bc_override_p1)  # A unchanged
                if not (v_closure_pressure_eq or v_closure_tied_eq):
                    for n in self._sink_nodes:
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is None:
                            continue
                        bch = np.asarray(bch, dtype=complex)
                        q_pred = _actual_boundary_Q_dc(
                            self.G, n, result_p1.edge_flows)
                        if q_pred is None:
                            continue
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(q_pred)
                        if v_closure_meas_ac_eq:
                            # V AC = measured phasor DIRECTLY.
                            for k in range(1, 4):
                                if k < len(bch):
                                    target[k] = bch[k]
                        else:
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(q_pred) * (bch[k] / DC_meas)
                        bc_override_p2[n] = target

                # --- Pass 2 solve ---
                if v_closure_tied_eq and len(self._sink_nodes) >= 2:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        tied_pressure_nodes=list(self._sink_nodes),
                        verbose=False,
                    )
                elif v_closure_pressure_eq:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        sink_pressure_bc=np.zeros(4, dtype=complex),
                        verbose=False,
                    )
                else:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        verbose=False,
                    )

                # A-pressure diagnostic: what pressures did geometry need
                # at each A to deliver equal flux?  Read from Pass 2's
                # node_pressures (Pass 1 has the same A pressures).
                if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                        and self._sim_dc_diag_chk.isChecked()):
                    print()
                    print('=' * 70)
                    print(f'  Equal-flux BC: Q(A_i) = {Q_A_per_port:+.4f} '
                          f'nL/s at each of {N_A} A ports  '
                          f'(Q_A_total = {Q_A_total:+.4f})')
                    print('  A pressures (solver output — should be near-')
                    print('  equal if geometry is symmetric between A paths):')
                    node_P = getattr(result_p1, 'node_pressures', None)
                    for n in self._source_nodes:
                        P_dc = None
                        if isinstance(node_P, dict):
                            pv = node_P.get(n)
                            if pv is not None:
                                P_dc = float(np.asarray(pv, dtype=complex)[0].real)
                        if P_dc is None:
                            print(f'    node {n:>6}:  (pressure not in result)')
                        else:
                            print(f'    node {n:>6}:  P_DC = {P_dc:+.4f} Pa')
                    print('=' * 70)
                    print()

            elif dirichlet_tie:
                # New default: V nodes Dirichlet-tied to common P_SV.
                # Two-pass approach:
                #   Pass 1: pure DC Dirichlet at ΔP (no AC pressure BC).
                #           Extract Q_0 at every boundary from the DC solve.
                #   Pass 2: fully Neumann Q at every boundary.  Each Q at
                #           harmonic k is the per-node measured normalized
                #           spectrum (H_k / H_0) rescaled by that node's
                #           Q_0 from pass 1.  This guarantees Q at every
                #           boundary has the same H_k/H_0 ratio as measured
                #           while DC is driven by ΔP.
                dP = float(getattr(self, '_sim_dp_spin', None).value()
                            if getattr(self, '_sim_dp_spin', None) else 0.0)

                if dP > 0.0:
                    # --- Pass 1: DC Dirichlet, no AC drive ---
                    p_sink_p1 = np.zeros(4, dtype=complex)
                    p_source_p1 = np.zeros(4, dtype=complex)
                    p_source_p1[0] = complex(dP)
                    result_p1 = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        sink_pressure_bc=p_sink_p1,
                        source_pressure_bc=p_source_p1,
                        verbose=False,
                    )

                    # V-closure sub-mode: 'flux' pins V (current behaviour),
                    # 'pressure' leaves V at Dirichlet P=0 so V flows stay
                    # a network prediction rather than a BC.
                    # 'tied' ties P_V1(ω) = P_V2(ω) at every harmonic via
                    # a phantom shim edge; V AC pressure becomes a model
                    # output with the shared-atrium constraint.
                    v_closure_pressure = bool(
                        getattr(self, '_sim_v_closure_pressure', None)
                        is not None
                        and self._sim_v_closure_pressure.isChecked())
                    v_closure_tied = bool(
                        getattr(self, '_sim_v_closure_tied', None)
                        is not None
                        and self._sim_v_closure_tied.isChecked())
                    v_closure_meas_ac = bool(
                        getattr(self, '_sim_v_closure_meas_ac', None)
                        is not None
                        and self._sim_v_closure_meas_ac.isChecked())

                    # --- Diagnostic: Pass-1 predicted vs measured DC per port ---
                    if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                            and self._sim_dc_diag_chk.isChecked()):
                        _print_dc_bc_diagnostic(
                            self.G, self._source_nodes, self._sink_nodes,
                            result_p1)

                    # --- Build boundary Q for Pass 2 ---
                    # Boundary invariant we trust: the MEASURED PI ratio
                    # bch[k]/bch[0] at each port.  To make sure the
                    # INJECTED PI (target[k]/target[0]) equals the measured
                    # PI, we scale the AC target by target[0] itself:
                    #   target[0] = Q̂_DC(n)                          (predicted DC)
                    #   target[k] = target[0] · bch[k]/bch[0]         (k ≥ 1)
                    # Then injected PI = target[k]/target[0] = bch[k]/bch[0]
                    # = measured PI, by construction.  If we injected bch[k]
                    # DIRECTLY (bypassing the scale), the injected PI would
                    # equal bch[k]/target[0] = measured_AC/predicted_DC,
                    # which is NOT the measured PI when Q̂_DC ≠ Re(bch[0]).
                    # 'pressure': Q-BC at A only, V held Dirichlet P=0.
                    # 'tied':     Q-BC at A only, V AC free but tied via penalty.
                    # 'flux':     Q-BC at both A and V (V AC = Q̂_DC · bch[k]/bch[0]).
                    # 'meas_ac':  Q-BC at both A and V, V AC target = bch[k]
                    #             DIRECTLY (measured phasor, no PI rescaling).
                    v_only_A_qbc = v_closure_pressure or v_closure_tied
                    which_nodes = list(self._source_nodes)
                    if not v_only_A_qbc:
                        which_nodes += list(self._sink_nodes)
                    bc_override_p2: Dict[int, np.ndarray] = {}
                    for n in which_nodes:
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is None:
                            continue
                        bch = np.asarray(bch, dtype=complex)
                        # Use edge_flows sum for actual pass-1 predicted DC
                        # (not the echoed boundary_Q_harmonics field).
                        q_pred = _actual_boundary_Q_dc(
                            self.G, n, result_p1.edge_flows)
                        if q_pred is None:
                            continue
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(q_pred)           # predicted DC
                        is_v = n in self._sink_nodes
                        if is_v and v_closure_meas_ac:
                            # V AC = measured phasor DIRECTLY (untrusted DC
                            # replaced with prediction, trusted AC kept as-is).
                            for k in range(1, 4):
                                if k < len(bch):
                                    target[k] = bch[k]
                        else:
                            # Default (A nodes; V under 'flux' closure): AC =
                            # predicted DC × measured PI ratio.
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(q_pred) * (bch[k] / DC_meas)
                        bc_override_p2[n] = target

                    # --- Pass 2 solve ---
                    if v_closure_tied and len(self._sink_nodes) >= 2:
                        # Tied V: penalty admittance in solver ties V
                        # pressures at every harmonic; V nodes get no
                        # Q-BC (V flows are model outputs).
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            tied_pressure_nodes=list(self._sink_nodes),
                            verbose=False,
                        )
                    elif v_closure_pressure:
                        # Grounded venous: keep P_V = 0 at all harmonics
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            sink_pressure_bc=np.zeros(4, dtype=complex),
                            verbose=False,
                        )
                    else:
                        # Fully Neumann Q at every boundary (default)
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            verbose=False,
                        )
                else:
                    # Flux-driven fallback: Q at A from measured/rescaled
                    # bc_override, V held at Dirichlet common P=0 (AC=0
                    # too — pure zero pressure at V; no AC pressure drive).
                    a_only = {n: arr for n, arr in bc_override.items()
                              if n in self._source_nodes}
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=a_only,
                        sink_pressure_bc=np.zeros(4, dtype=complex),
                        verbose=False,
                    )
            elif v_bc_mode == 0:
                # Standard: prescribe Q at every boundary.
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=bc_override,
                    verbose=False,
                )
            else:
                # Option B: P-match V at DC, rescale per-V AC by the
                # predicted Q̄_V × measured per-V PI/phase.
                # Pass 1: predict Q̄_V per V.
                a_only_override = {
                    n: arr for n, arr in bc_override.items()
                    if n in self._source_nodes
                }
                result_pass1 = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=a_only_override,
                    sink_pressure_bc=0.0,
                    verbose=False,
                )
                # Build pass-2 override: A's unchanged, V's get DC =
                # predicted (signed into-network) and AC = ratio ×
                # predicted DC, with the ratio taken from each V's
                # MEASURED bc_harmonics so the per-V PI and phase
                # are the data-side constraint.
                pass2_override: Dict[int, np.ndarray] = dict(a_only_override)
                for sn in self._sink_nodes:
                    measured = self.G.nodes[sn].get('bc_harmonics')
                    if measured is None:
                        continue
                    measured = np.asarray(measured, dtype=complex)
                    q_into_v = _q_into_node(result_pass1.edge_flows, sn)
                    # Solver's "Q at V" convention is +inflow into the
                    # network at V = −outflow_from_network_at_V.
                    # `q_into_v` is the predicted Q flowing INTO V
                    # from the network (= outflow_from_network), so
                    # the solver-convention Q_V is −q_into_v.
                    Q_v_dc_new = -q_into_v[0].real
                    new_arr = np.zeros_like(measured)
                    new_arr[0] = Q_v_dc_new
                    # Ratio Q_n / Q_dc is convention-independent (any
                    # uniform sign flip cancels).  Multiplying by the
                    # new DC rescales the AC magnitude while keeping
                    # the measured per-V phase intact.
                    if abs(measured[0]) > 1e-12 and abs(Q_v_dc_new) > 1e-30:
                        for k in range(1, len(measured)):
                            new_arr[k] = (measured[k] / measured[0]
                                           ) * Q_v_dc_new
                    pass2_override[sn] = new_arr
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=pass2_override,
                    verbose=False,
                )
            elapsed = time.time() - t0
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._sim_status_lbl.setText(
                f"<small><font color='#c33'>Solver failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._sim_run_btn.setEnabled(True)
            return
        finally:
            # Always restore the graph's original geometry so the
            # mutation is invisible to the rest of the viewer.
            if saved_geom:
                for (u, v), saved in saved_geom.items():
                    for k, original in saved.items():
                        if original is None:
                            self.G.edges[u, v].pop(k, None)
                        else:
                            self.G.edges[u, v][k] = original

        n = self._store_sim_result(result, f0, D, override_R,
                                   override_L, R_UNIFORM_PX, L_UNIFORM_PX)
        # Stamp the graph with the run parameters so the state can be
        # saved + reloaded with the UI restored.  Any subsequent 'Save
        # graph state…' click will pickle this alongside the sim data.
        self._stamp_sim_meta(D=D, f0=f0, n_edges=n, elapsed_s=elapsed)
        self._sim_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · {n} edges<br>"
            f"D = {D:.3g}, f₀ = {f0:.3f} Hz<br>"
            f"Switch the View-tab Source to <b>Simulated</b> to view "
            f"sim fields; P quantity becomes available.</small>")
        self._sim_run_btn.setEnabled(True)
        self._sim_clear_btn.setEnabled(True)

    # ── Sim persistence: stamp / save / restore ─────────────────────
    def _stamp_sim_meta(self, D, f0, n_edges, elapsed_s):
        """Write G.graph['sim_meta'] snapshotting the UI parameters
        used for this run.  Read back by _restore_sim_ui_from_graph()
        on next viewer launch to repopulate the controls."""
        from datetime import datetime, timezone
        dp = float(self._sim_dp_spin.value()
                    if getattr(self, '_sim_dp_spin', None) else 0.0)
        qat = float(self._sim_qa_spin.value()
                     if getattr(self, '_sim_qa_spin', None) else 1.0)
        bc_mode = ('dirichlet_tie'
                    if (getattr(self, '_sim_bc_radio_dirichlet', None)
                        and self._sim_bc_radio_dirichlet.isChecked())
                    else 'equalflux')
        if (getattr(self, '_sim_v_closure_meas_ac', None)
                and self._sim_v_closure_meas_ac.isChecked()):
            v_closure = 'meas_ac'
        elif (getattr(self, '_sim_v_closure_pressure', None)
                and self._sim_v_closure_pressure.isChecked()):
            v_closure = 'pressure'
        elif (getattr(self, '_sim_v_closure_tied', None)
                and self._sim_v_closure_tied.isChecked()):
            v_closure = 'tied'
        else:
            v_closure = 'flux'
        self.G.graph['sim_meta'] = dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            D_areal=float(D),
            eta=float(getattr(self, '_sim_last_eta', 0.0)),
            wall_model='constant_loss_tangent',
            f0_hz=float(f0),
            K_harmonics=3,
            bc_mode=bc_mode,
            v_closure=v_closure,
            dp_target_Pa=dp,
            q_a_total_nL_s=qat,
            n_edges_written=int(n_edges),
            elapsed_s=float(elapsed_s),
        )

    def _on_save_state_clicked(self):
        """Pickle the whole in-memory graph to <stem>_state.gpickle.
        Bundles normalization fields, sim results, tile_gauges, and
        the sim_meta snapshot in one file."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        default_name = self.graph_path.with_name(
            self.graph_path.stem + '_state.gpickle')
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save graph state (normalized + sim + metadata)",
            str(default_name),
            "Pickled graph (*.gpickle);;All files (*)")
        if not path:
            return
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.G, f)
            has_sim = bool(self.G.graph.get('sim_meta'))
            has_norm = bool(self.G.graph.get('tile_gauges'))
            QMessageBox.information(
                self.viewer.window._qt_window, "Saved",
                f"Wrote {path}\n"
                f"  normalized fields: {'yes' if has_norm else 'no'}\n"
                f"  sim fields:        {'yes' if has_sim else 'no'}")
        except Exception as exc:
            QMessageBox.critical(
                self.viewer.window._qt_window, "Save failed",
                f"Could not save graph:\n{exc}")

    def _restore_sim_ui_from_graph(self):
        """If G.graph carries a sim_meta blob (i.e. the graph was
        saved after a previous simulation run), set the Simulate tab
        controls to match and populate _sim_loaded_lbl so the user can
        see what was run."""
        meta = self.G.graph.get('sim_meta')
        lbl = getattr(self, '_sim_loaded_lbl', None)
        if not meta or lbl is None:
            return
        # Radios
        bc_mode = meta.get('bc_mode', 'dirichlet_tie')
        if bc_mode == 'equalflux' and getattr(self, '_sim_bc_radio_equalflux', None):
            self._sim_bc_radio_equalflux.setChecked(True)
        elif getattr(self, '_sim_bc_radio_dirichlet', None):
            self._sim_bc_radio_dirichlet.setChecked(True)
        vc = meta.get('v_closure', 'meas_ac')
        vc_radio = {
            'flux':     getattr(self, '_sim_v_closure_flux', None),
            'pressure': getattr(self, '_sim_v_closure_pressure', None),
            'tied':     getattr(self, '_sim_v_closure_tied', None),
            'meas_ac':  getattr(self, '_sim_v_closure_meas_ac', None),
        }.get(vc)
        if vc_radio is not None:
            vc_radio.setChecked(True)
        # Spinboxes
        if getattr(self, '_sim_dp_spin', None) and 'dp_target_Pa' in meta:
            self._sim_dp_spin.setValue(float(meta['dp_target_Pa']))
        if getattr(self, '_sim_qa_spin', None) and 'q_a_total_nL_s' in meta:
            self._sim_qa_spin.setValue(float(meta['q_a_total_nL_s']))
        # Label
        ts = str(meta.get('timestamp', '?'))[:19].replace('T', ' ')
        # ΔP is only used in dirichlet_tie mode.  In equalflux (the only
        # remaining active mode) it's ignored by the solve, so omit it
        # from the label to avoid the misleading "10 Pa" that came from
        # the hidden default spinbox value.
        _line2 = (f"D = {meta.get('D_areal', float('nan')):.3g},  "
                  f"f₀ = {meta.get('f0_hz', float('nan')):.3f} Hz,  "
                  f"Q_A_total = {meta.get('q_a_total_nL_s', float('nan')):.3f} nL/s")
        if meta.get('bc_mode') == 'dirichlet_tie':
            _line2 += f",  ΔP = {meta.get('dp_target_Pa', float('nan')):.2f} Pa"
        lbl.setText(
            f"<b>Loaded saved simulation</b>  ({ts})<br>"
            f"{_line2}<br>"
            f"BC = <tt>{meta.get('bc_mode','?')}</tt>,  "
            f"V closure = <tt>{meta.get('v_closure','?')}</tt><br>"
            f"<i>Sim fields already populated on the graph.  Switch View "
            f"tab Source to Simulated to view.  Click <b>Run</b> to "
            f"re-execute with any parameter change.</i>")
        lbl.show()
        # Turn on the Clear button since sim data is present
        if getattr(self, '_sim_clear_btn', None) is not None:
            self._sim_clear_btn.setEnabled(True)

    def _apply_pi_flip_and_persist_sim(self, mode='intact'):
        """Per-tile π-flip resolution on sim H1/H2/H3 (each independent),
        matching what sync_v2 did to measured. Then write persistent
        sim fields (`_h_Z_H{k}_sim`, `_h_phase_H{k}_sim`, `mean_Q_sim`,
        `amp_Q_sim`, `phase_sim`) so displays / overlays / save-state
        all see the flipped values.

        mode='intact': reads `_sim_tmp_harmonics` (complex [DC,H1,H2,H3])
        mode='carved': reads `_h_amp_H{k}_sim_carved` + `_h_phase_H{k}_sim_carved`
                       + `_h_amp_DC_sim_carved`
        Reference in both modes: `_h_amp_H{k}_norm` * exp(i·`_h_phase_H{k}_norm`).
        """
        # 1) Gather per-edge (sim, meas) H1/H2/H3 phasors + tile ids.
        def _tile_of(d):
            tms = d.get('measurements_piv') or []
            if isinstance(tms, list) and tms and isinstance(tms[0], dict):
                for f in ('tile_id', 'tid', 'vid'):
                    if f in tms[0]: return tms[0][f]
            return None
        def _meas_Z(d, k):
            a = d.get(f'_h_amp_H{k}_norm')
            p = d.get(f'_h_phase_H{k}_norm')
            if a is None or p is None or not np.isfinite(a) or not np.isfinite(p):
                return None
            return complex(float(a) * np.exp(1j * float(p)))
        def _sim_Z(d, k):
            if mode == 'intact':
                arr = d.get('_sim_tmp_harmonics')
                if arr is None: return None
                arr = np.asarray(arr, dtype=complex)
                return complex(arr[k]) if k < len(arr) else None
            # carved
            a = d.get(f'_h_amp_H{k}_sim_carved')
            p = d.get(f'_h_phase_H{k}_sim_carved')
            if a is None or p is None or not np.isfinite(a) or not np.isfinite(p):
                return None
            return complex(float(a) * np.exp(1j * float(p)))
        # SNR weight (edge-level)
        def _snr_w(d):
            for tm in (d.get('measurements_piv') or []):
                if isinstance(tm, dict):
                    s = tm.get('SNR_H1_dB')
                    if s is not None and np.isfinite(s):
                        return 10 ** (float(s) / 10.0)
            return 1.0
        # 2) Build per-tile flip maps for k=1,2,3 (each independent)
        flip = {1: {}, 2: {}, 3: {}}
        by_tile = {}
        for u, v, d in self.G.edges(data=True):
            t = _tile_of(d)
            if t is None: continue
            by_tile.setdefault(t, []).append((u, v, d))
        for t, edges in by_tile.items():
            if len(edges) < 2: continue
            for k in (1, 2, 3):
                r0 = 0.0; r1 = 0.0; n_ok = 0
                for u, v, d in edges:
                    zs = _sim_Z(d, k); zm = _meas_Z(d, k)
                    if zs is None or zm is None: continue
                    w = _snr_w(d)
                    r0 += w * abs(zs - zm)**2
                    r1 += w * abs(-zs - zm)**2
                    n_ok += 1
                if n_ok < 2: continue
                flip[k][t] = 0.0 if r0 <= r1 else np.pi
        # 3) Apply flips and write persistent sim fields
        n_flipped_H1 = sum(1 for f in flip[1].values() if f > 0)
        for u, v, d in self.G.edges(data=True):
            t = _tile_of(d)
            # DC magnitude (no flip)
            if mode == 'intact':
                arr = d.get('_sim_tmp_harmonics')
                if arr is None: continue
                arr = np.asarray(arr, dtype=complex).copy()
                # apply per-harmonic flips
                if len(arr) > 1 and t in flip[1]:
                    arr[1] = arr[1] * np.exp(1j * flip[1][t])
                if len(arr) > 2 and t in flip[2]:
                    arr[2] = arr[2] * np.exp(1j * flip[2][t])
                if len(arr) > 3 and t in flip[3]:
                    arr[3] = arr[3] * np.exp(1j * flip[3][t])
                d['_sim_tmp_harmonics'] = arr
                # Persistent write (matches lock_D6p5e4_eta0p4.py convention)
                d['_h_Z_DC_sim'] = complex(arr[0])
                d['mean_Q_sim'] = float(np.real(arr[0]))
                for k in range(1, min(4, len(arr))):
                    d[f'_h_Z_H{k}_sim'] = complex(arr[k])
                    d[f'_h_phase_H{k}_sim'] = float(np.angle(arr[k]))
                    if k == 1:
                        d['amp_Q_sim'] = float(abs(arr[1]))
                        d['phase_sim'] = float(np.angle(arr[1]))
                        # Update _sim_tmp_phase_Q to reflect the flip
                        d['_sim_tmp_phase_Q'] = float(np.angle(arr[1]))
            else:  # carved
                # Apply flip to the carved amp/phase pair (amp unchanged; phase += flip)
                for k in (1, 2, 3):
                    if t not in flip[k]: continue
                    ph_key = f'_h_phase_H{k}_sim_carved'
                    if ph_key in d and np.isfinite(d[ph_key]):
                        d[ph_key] = float(((d[ph_key] + flip[k][t] + np.pi) % (2*np.pi)) - np.pi)
                    zk = d.get(f'_h_Z_H{k}_sim_carved')
                    if zk is not None:
                        d[f'_h_Z_H{k}_sim_carved'] = complex(zk) * np.exp(1j * flip[k][t])
        n_tiles_total = len({t for t in by_tile.keys()})
        self.G.graph[f'pi_flip_stats_{mode}'] = dict(
            n_flipped_H1=int(n_flipped_H1),
            n_tiles=int(n_tiles_total),
        )
        try:
            print(f"[π-flip {mode}] H1: {n_flipped_H1}/{n_tiles_total} tiles flipped")
        except Exception:
            pass

    def _store_sim_result(self, result, f0, D, override_R=False,
                          override_L=False, R_UNIFORM_PX=5.0,
                          L_UNIFORM_PX=50.0):
        """Write a solve_transmission_line `result` into the per-edge
        `_sim_tmp_*` fields (mean_Q, amp_Q, PI, phase, pressure, harmonics…),
        activate the Simulated source, and refresh the View-tab combos.
        Shared by the Simulate tab and the Pump tab.  Returns #edges written."""
        # New sim data invalidates any cached A/V waveform references.
        self._av_ref_cache = None
        # Write per-edge sim results to the in-memory graph.  Use the
        # `_sim_tmp_` namespace so we don't collide with editing-app
        # outputs (`mean_Q_sim`, `PI_sim`, …) already stored on the
        # graph.
        #
        # Q fields are stored as magnitudes:
        #   mean_Q  →  |mean_Q|       (matches measured: mean_Q ≥ 0
        #                              by convention — see methods.md)
        #   amp_Q   →  |H₁(Q)|        (already a magnitude in the result)
        #
        # Two phases are exposed so the user can see both flow and
        # pressure timing per harmonic.  `result.phase` is the Q-phase
        # with the sign canonicalisation already applied by the solver
        # (flips when flow direction would otherwise be negative);
        # `result.pressure_phase` is the midpoint-pressure H₁ phase.
        n = 0
        for (u, v) in result.edge_flows.keys():
            if not self.G.has_edge(u, v):
                continue
            d = self.G.edges[u, v]
            mq = result.mean_Q.get((u, v), float('nan'))
            aq = result.amp_Q.get((u, v), float('nan'))
            d['_sim_tmp_mean_Q'] = (float(abs(mq)) if np.isfinite(mq)
                                    else float('nan'))
            d['_sim_tmp_amp_Q'] = (float(abs(aq)) if np.isfinite(aq)
                                    else float('nan'))
            d['_sim_tmp_PI'] = float(
                result.PI.get((u, v), float('nan')))
            d['_sim_tmp_phase_Q'] = float(
                result.phase.get((u, v), float('nan')))
            d['_sim_tmp_phase_P'] = float(
                result.pressure_phase.get((u, v), float('nan')))
            d['_sim_tmp_dissipation'] = float(
                result.dissipation.get((u, v), float('nan')))
            # Geometry overrides — written only when the corresponding
            # override mode was active, so the sim-mode View tab can
            # show the constant value.  Heterogeneous sim → keys
            # absent → fall through to the measured top-level attrs.
            if override_R:
                d['_sim_tmp_radius'] = R_UNIFORM_PX
            if override_L:
                d['_sim_tmp_length'] = L_UNIFORM_PX
            # Persist the full complex harmonic array (DC + H₁ + H₂ + H₃)
            # so `_plot_comparison` can reconstruct a synthetic Q(t)
            # when the user is viewing a sim field.  Sign-canonicalise
            # so mean_Q ≥ 0 (matches measured `mean_Q` convention from
            # methods.md), shifting all harmonic phases by π if needed.
            harm = result.edge_flows.get((u, v))
            if harm is not None and len(harm) >= 1:
                arr = np.asarray(harm, dtype=complex).copy()
                if np.isfinite(arr[0].real) and arr[0].real < 0:
                    arr = -arr
                d['_sim_tmp_harmonics'] = arr
                d['_sim_tmp_f0_hz'] = float(f0)
            # Midpoint pressure harmonics + summary scalars
            # (sim-only fields, available in the View tab when the
            # Simulated source is active).  Pressure PI is undefined
            # near the gauge node where |P_DC| → 0; we record nan
            # there rather than picking an arbitrary reference.
            p_u = result.node_pressures.get(u)
            p_v = result.node_pressures.get(v)
            if p_u is not None and p_v is not None:
                p_u_arr = np.asarray(p_u, dtype=complex)
                p_v_arr = np.asarray(p_v, dtype=complex)
                p_mid = 0.5 * (p_u_arr + p_v_arr)
                d['_sim_tmp_p_harmonics'] = p_mid
                p_dc = float(p_mid[0].real) if len(p_mid) > 0 else float('nan')
                amp_p = (float(abs(p_mid[1]))
                         if len(p_mid) > 1 else float('nan'))
                # Signed mean midpoint pressure (Pa).  Kept signed so
                # the gauge polarity is visible in the colormap; users
                # who only want magnitude can toggle log scale off and
                # read the sign directly.
                d['_sim_tmp_pressure_mean'] = p_dc
                d['_sim_tmp_pressure_amp'] = amp_p
                # H2 midpoint pressure amplitude (Pa) — analogous to
                # pressure_amp but at the second harmonic.  NaN if the
                # solver didn't compute H2.
                amp_p_h2 = (float(abs(p_mid[2]))
                             if len(p_mid) > 2 else float('nan'))
                d['_sim_tmp_pressure_amp_H2'] = amp_p_h2
                # Pressure drop across the edge: |P_u_DC − P_v_DC| (Pa).
                # Magnitude so the colormap reads unambiguously; sign
                # would depend on edge orientation (undirected graph).
                if len(p_u_arr) > 0 and len(p_v_arr) > 0:
                    d['_sim_tmp_pressure_drop'] = float(
                        abs(p_u_arr[0].real - p_v_arr[0].real))
            n += 1
        # ── π-flip resolution + persistent field write ─────────────────
        # After the raw sim is stored in `_sim_tmp_*`, apply the per-tile
        # π-flip on H1/H2/H3 (independently) using measured `_h_*_norm`
        # phasors as reference. This puts sim into the same sync_v2
        # amplitude-sign frame the measured field lives in, so the View
        # tab's phase and |H1|/|DC| displays are directly comparable to
        # measured. Also write persistent `_h_Z_H*_sim`, `mean_Q_sim`,
        # `amp_Q_sim`, `phase_sim` fields (matching lock_D6p5e4_eta0p4.py's
        # convention) so downstream scripts / overlays read from a common
        # namespace and the sim state survives save-and-reload.
        self._apply_pi_flip_and_persist_sim(mode='intact')
        self._sim_active = True
        self._sim_last_D = D
        self._sim_last_f0_hz = f0
        # Enable the Simulated source radio in the View tab.
        if getattr(self, '_src_radio_sim', None) is not None:
            self._src_radio_sim.setEnabled(True)
            self._src_radio_sim.setToolTip("")
        # Re-sync the field combos so the P quantity and sim-only
        # properties become enabled (or stay enabled).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        return n

    # ── Ligation experiment helpers ───────────────────────────────────
    def _clear_simulation(self):
        """Strip `_sim_tmp_*` keys from every edge, fall back to the
        measured source, and refresh the View-tab UI."""
        sim_keys = list(SIM_FIELD_MAP.values()) + SIM_INTERNAL_KEYS
        removed = 0
        for _, _, d in self.G.edges(data=True):
            for k in sim_keys:
                if k in d:
                    del d[k]
                    removed += 1
        self._sim_active = False
        self._av_ref_cache = None
        # Force the View tab back to the measured source.
        self.current_source = 'measured'
        if getattr(self, '_src_radio_measured', None) is not None:
            self._src_radio_measured.blockSignals(True)
            self._src_radio_sim.blockSignals(True)
            try:
                self._src_radio_measured.setChecked(True)
                self._src_radio_sim.setEnabled(False)
            finally:
                self._src_radio_measured.blockSignals(False)
                self._src_radio_sim.blockSignals(False)
        # Resync the field combos (P quantity + sim-only properties
        # disappear / get greyed out) and re-render.
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        self._sim_clear_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Cleared {removed} sim values from "
            f"{self.G.number_of_edges()} edges.</small>")

    def _on_bounds_change(self, _val=None):
        # Any bounds spinbox changed → re-render with the new clip range.
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_bounds_mode_change(self, _idx=None):
        # Toggle visibility of Percentile vs Manual widgets to reduce
        # clutter, and re-render with the newly active bounds source.
        combo = getattr(self, '_bounds_mode_combo', None)
        if combo is None:
            return
        is_manual = (combo.currentText() == 'Manual')
        for w in (getattr(self, '_pct_lo_lbl', None),
                   getattr(self, '_pct_lo_spin', None),
                   getattr(self, '_pct_hi_lbl', None),
                   getattr(self, '_pct_hi_spin', None)):
            if w is not None:
                w.setVisible(not is_manual)
        for w in (getattr(self, '_vmin_lbl', None),
                   getattr(self, '_vmin_spin', None),
                   getattr(self, '_vmax_lbl', None),
                   getattr(self, '_vmax_spin', None)):
            if w is not None:
                w.setVisible(is_manual)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_total_snr_pct_change(self, val: float):
        """Update the network-wide total_snr percentile filter.  The
        threshold itself is re-derived from cached `_h_total_snr`
        values on every refresh — set independently of any tile filter
        so it stays stable when you swap tiles."""
        self.total_snr_pct_filter = float(val)
        # Force re-derivation of the cached threshold next refresh.
        self._cached_total_snr_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_total_snr_h2_pct_change(self, val: float):
        """Same as `_on_total_snr_pct_change` but for Q_H2_snr_db."""
        self.total_snr_h2_pct_filter = float(val)
        self._cached_total_snr_h2_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_dissipation_pct_change(self, val: float):
        """Update the network-wide dissipation percentile filter.
        Threshold is re-derived from `_meas_dissipation` (W) values
        across the whole graph — stable regardless of Tile filter."""
        self.dissipation_pct_filter = float(val)
        self._cached_dissipation_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_tile_filter_change(self, _idx=None):
        """Switch between best-measurement coloring and per-tile-only
        coloring.  When a tile is selected, repopulates
        `_tile_harmonic_cache` for that tile (~5 ms × edges with a
        measurement on this tile) so the resolver reads per-tile values
        immediately.  Always re-populating keeps the cache consistent
        even if the video-load path wrote a different tile's data
        since the last filter change.
        """
        if self._tile_filter_combo is None:
            return
        new_filter = self._tile_filter_combo.currentData()
        new_filter = int(new_filter) if new_filter is not None else None
        if new_filter == self.current_tile_filter:
            return
        self.current_tile_filter = new_filter
        if new_filter is not None:
            self._populate_tile_harmonic_cache(new_filter)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        if self.show_kcl:
            self._refresh_kcl_overlay()

    def _on_cbar_toggle(self, checked: bool):
        if not bool(checked) and self._cbar_overlay is not None:
            self._cbar_overlay.setParent(None)
            self._cbar_overlay.deleteLater()
            self._cbar_overlay = None
        else:
            self._refresh_cbar()

    def _refresh_cbar(self):
        """Render the colormap legend as a floating QLabel/QPixmap on the
        napari canvas (right side, vertically centred).  Same approach
        as the editing viewer's `_update_colorbar_layer`."""
        # Always tear down the previous overlay (if any) before drawing.
        old = self._cbar_overlay
        if old is not None:
            old.setParent(None)
            old.deleteLater()
            self._cbar_overlay = None
        if not getattr(self, '_cbar_chk', None) \
                or not self._cbar_chk.isChecked():
            return
        from matplotlib.colors import Normalize, LogNorm
        from qtpy.QtWidgets import QLabel
        from qtpy.QtGui import QPixmap, QImage
        from qtpy.QtCore import Qt
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        vmin = getattr(self, '_last_vmin', None)
        vmax = getattr(self, '_last_vmax', None)
        if is_categorical or vmin is None or vmax is None:
            return  # nothing meaningful to draw
        is_phase = (self.current_property == 'phase')
        use_log = bool(self.log_scale) and not is_phase
        # Compose colorbar label from the selectors so it always reads
        # cleanly: "Magnitude |·|  (Q, H₁)", "Pulsatility index  (P)", …
        label = self._current_field_label()
        # Build the matplotlib figure (Agg, transparent bg, white text).
        dpi = 100
        if is_phase:
            # Half-circle color wheel for π-unified phase (matches
            # the [0, π] fold applied in the display pipeline).
            fig_sz = 2.8
            fig = plt.figure(figsize=(fig_sz, fig_sz), dpi=dpi)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='polar')
            n = 512
            theta = np.linspace(0, np.pi, n + 1)
            r_in, r_out = 0.45, 1.0
            C = np.tile(np.linspace(0, np.pi, n), (1, 1))
            ax.pcolormesh(theta, [r_in, r_out], C,
                           cmap=plt.get_cmap('hsv'),
                           norm=Normalize(vmin=0, vmax=np.pi),
                           shading='auto')
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_thetamin(0); ax.set_thetamax(180)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.text(0, 0, label, ha='center', va='center',
                    color='white', fontsize=11, transform=ax.transData)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
        else:
            fig_h_in, fig_w_in = 5.0, 1.8
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            cmap = plt.get_cmap('viridis')
            if use_log and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax)
            cbar.set_label(label, color='white', fontsize=14, labelpad=10)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                           labelsize=11, width=2, length=5)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
            fig.subplots_adjust(left=0.05, right=0.35, top=0.97, bottom=0.03)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        # Parent to the vispy canvas widget so it floats over the image.
        qt_viewer = self.viewer.window._qt_viewer
        canvas_widget = qt_viewer.canvas.native
        overlay = QLabel(canvas_widget)
        overlay.setPixmap(pixmap)
        overlay.setFixedSize(w, h)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.setStyleSheet("background: transparent;")
        overlay.raise_()
        # Position: right edge, vertically centred.
        def _reposition():
            pw = canvas_widget.width()
            ph = canvas_widget.height()
            x = max(0, pw - w - 20)
            y = max(0, (ph - h) // 2)
            overlay.move(x, y)
        _reposition()
        overlay.show()
        # Re-position on canvas resize (preserve any prior handler).
        if self._cbar_orig_resize is None:
            self._cbar_orig_resize = canvas_widget.resizeEvent
        def _on_resize(event, _orig=self._cbar_orig_resize):
            _orig(event)
            cur = self._cbar_overlay
            if cur is not None and cur.isVisible():
                _reposition()
        canvas_widget.resizeEvent = _on_resize
        self._cbar_overlay = overlay

    # ── tile boundaries + labels (optional overlay) ────────────────────
    def _refresh_tile_boundaries(self):
        # Remove existing layer (None on first call) and re-add if toggled on.
        if self._tile_boundaries_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_boundaries_layer)
            except (KeyError, ValueError):
                pass
            self._tile_boundaries_layer = None
        if not self.show_tile_boundaries or not self.tiles:
            return
        rects: List[np.ndarray] = []
        # Raw tile corners in (y, x) = (row, col) order.
        raw_corners = np.array([
            [0.0,                 0.0],
            [0.0,                 TILE_RAW_WIDTH],
            [TILE_RAW_HEIGHT,     TILE_RAW_WIDTH],
            [TILE_RAW_HEIGHT,     0.0],
        ])
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            A = t.get('affine')
            if A is None:
                r0 = t['translate_y'] - self._tile_offset_y
                c0 = t['translate_x'] - self._tile_offset_x
                r1 = r0 + TILE_RAW_HEIGHT * t['scale_y']
                c1 = c0 + TILE_RAW_WIDTH * t['scale_x']
                rects.append(np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]))
                continue
            M = A[:2, :2]
            tvec = A[:2, 2] - np.array([self._tile_offset_y, self._tile_offset_x])
            # (4,2) corners after rotation+scale, plus offset-shifted translate.
            mapped = raw_corners @ M.T + tvec
            rects.append(mapped)
        self._tile_boundaries_layer = self.viewer.add_shapes(
            rects, shape_type='polygon',
            edge_color='yellow', edge_width=2.0,
            face_color='transparent',
            name='Tile boundaries', opacity=0.7,
        )

    def _refresh_tile_labels(self):
        if self._tile_labels_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_labels_layer)
            except (KeyError, ValueError):
                pass
            self._tile_labels_layer = None
        if not self.show_tile_labels or not self.tiles:
            return
        pts: List[List[float]] = []
        labels: List[str] = []
        centre_raw = np.array([TILE_RAW_HEIGHT * 0.5, TILE_RAW_WIDTH * 0.5])
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            A = t.get('affine')
            if A is None:
                r0 = t['translate_y'] - self._tile_offset_y
                c0 = t['translate_x'] - self._tile_offset_x
                r_mid = r0 + 0.5 * TILE_RAW_HEIGHT * t['scale_y']
                c_mid = c0 + 0.5 * TILE_RAW_WIDTH * t['scale_x']
            else:
                M = A[:2, :2]
                tvec = (A[:2, 2]
                        - np.array([self._tile_offset_y, self._tile_offset_x]))
                r_mid, c_mid = M @ centre_raw + tvec
            pts.append([r_mid, c_mid])
            labels.append(str(tid))
        self._tile_labels_layer = self.viewer.add_points(
            np.array(pts), size=0, name='Tile numbers',
            text={'string': labels, 'size': 14,
                   'color': 'yellow', 'anchor': 'center'},
        )

    # ── click handler ──────────────────────────────────────────────────
    def _inspect_nearest_boundary_node(
            self, x: float, y: float, max_px: float = 18.0) -> bool:
        """If the click is within `max_px` of a boundary node, display
        its BC harmonics in the info panel and return True.  Otherwise
        return False so the caller can fall through to edge hit-test.

        `x`, `y` are in graph coordinates (x = G.nodes[n]['x'], y is
        the click-row converted to graph y via mosaic_height − row).
        """
        if not self._boundary_nodes:
            return False
        y_graph = self.mosaic_height - y
        best_n, best_d = None, np.inf
        for n in self._boundary_nodes:
            dx = self.G.nodes[n]['x'] - x
            dy = self.G.nodes[n]['y'] - y_graph
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_n, best_d = n, d
        if best_n is None or best_d > max_px:
            return False
        # Disable edge-only buttons (BC click clears edge selection).
        self._selected_edge = None
        try:
            self._plot_qt_btn.setEnabled(False)
            self._add_to_comp_btn.setEnabled(False)
            self._video_btn.setEnabled(False)
            self._run_of_btn.setEnabled(False)
        except AttributeError:
            pass
        d = self.G.nodes[best_n]
        btype = d.get('boundary_type', '?')
        letter = 'A' if btype == 'source' else 'V' if btype == 'sink' else '?'
        side = ('arterial source' if btype == 'source'
                else 'venous sink' if btype == 'sink' else btype)
        info: List[str] = []
        info.append(f"Boundary node {best_n}  [{letter}]  ({side})")
        info.append(f"  position: ({d.get('x', float('nan')):.0f}, "
                    f"{d.get('y', float('nan')):.0f})")
        bch = d.get('bc_harmonics')
        if bch is None:
            info.append("\n  No bc_harmonics stored on this node.")
        else:
            bch = np.asarray(bch)
            # Apply canonical sign convention so phases at sources and
            # sinks report the same impedance phase, not differ by π:
            #   stored Q at sources = +inflow
            #   stored Q at sinks   = +outflow (network sees −Q)
            # → multiply by −1 at sinks to get Q_signed (always "into
            #   the network"), then read magnitude / phase off that.
            sign = -1.0 if btype == 'sink' else 1.0
            bch_signed = bch * sign
            sign_note = " (Q_signed = −Q_outflow)" if sign < 0 else ""
            info.append(f"\n  bc_harmonics (length {len(bch)}): "
                        f"DC + H₁ + H₂ + H₃{sign_note}")
            labels = ['DC', 'H₁', 'H₂', 'H₃']
            for i, c in enumerate(bch_signed[:4]):
                lab = labels[i] if i < len(labels) else f'H{i}'
                if i == 0:
                    info.append(f"    {lab}:  Q̄ = {float(np.real(c)):+.4g}  "
                                f"(imag = {float(np.imag(c)):+.3g})")
                else:
                    mag = float(np.abs(c))
                    ph = float(np.angle(c))
                    info.append(f"    {lab}:  |a| = {mag:.4g},  "
                                f"∠ = {np.degrees(ph):+.1f}°")
        self._info_box.setPlainText("\n".join(info))
        # Move the click marker to the boundary node so the user sees
        # which one they hit.
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - d['y'], d['x']]])
        return True

    def _highlight_edge_path(self, u, v):
        """Overlay the selected edge's true curved centerline (its `path`), or a
        straight node-to-node line if the edge has no path.  The main Edges
        layer is drawn straight for rendering robustness; this shows the real
        vessel geometry only for the clicked edge."""
        H = self.mosaic_height
        d = self.G.edges[u, v]
        p = d.get('path')
        if p is not None and len(p) >= 2:
            p = np.asarray(p, dtype=float)
            coords = np.column_stack([H - p[:, 1], p[:, 0]])
        else:
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            coords = np.array([[H - y1, x1], [H - y2, x2]])
        lyr = getattr(self, '_edge_path_layer', None)
        if lyr is not None:
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            self._edge_path_layer = None
        self._edge_path_layer = self.viewer.add_shapes(
            [coords], shape_type='path', edge_color='#00e5ff',
            edge_width=2.5, name='Selected vessel path', opacity=1.0)

    def _inspect_nearest_edge(self, x: float, y: float, max_px: float = 25.0):
        # `y` here is the display row from napari's click event.  The
        # graph stores y in math-convention (image displayed with
        # mosaic_height − y).  Convert click-row back to graph-y so we
        # can compare against precomputed midpoints.
        if len(self._edge_midpoints) == 0:
            return
        y_graph = self.mosaic_height - y
        d2 = ((self._edge_midpoints[:, 0] - x) ** 2 +
              (self._edge_midpoints[:, 1] - y_graph) ** 2)
        i = int(np.argmin(d2))
        dist = float(np.sqrt(d2[i]))
        if dist > max_px:
            self._info_box.setPlainText(
                f"No edge within {max_px} px of click "
                f"(closest was {dist:.1f} px away).")
            self._plot_qt_btn.setEnabled(False)
            self._click_marker_layer.data = np.empty((0, 2))
            return
        u, v = self.edge_list[i]
        self._selected_edge = (u, v)
        self._plot_qt_btn.setEnabled(True)
        if hasattr(self, '_plot_overlay_btn'):
            self._plot_overlay_btn.setEnabled(True)
        # Enable boundary-KCL button only when this edge touches a boundary
        if hasattr(self, '_plot_bnd_kcl_btn'):
            bt_u = self.G.nodes[u].get('boundary_type')
            bt_v = self.G.nodes[v].get('boundary_type')
            self._plot_bnd_kcl_btn.setEnabled(
                bt_u in ('source', 'sink') or bt_v in ('source', 'sink'))
        # Enable raw-vs-normalized compare only if this edge has both
        # measurements_piv (raw) AND measurements_piv_normalized populated
        # (side-by-side after tile_normalize runs).
        if hasattr(self, '_compare_norm_btn'):
            ed = self.G.edges[u, v]
            has_norm = (self._normalization_available()
                        and bool(ed.get('measurements_piv_normalized')))
            self._compare_norm_btn.setEnabled(has_norm)
        self._add_to_comp_btn.setEnabled(True)
        # OF button requires a tile + video; gate it on having both.
        _of_ok = bool(self.tiles) and self.of_video_dir is not None
        self._run_of_btn.setEnabled(_of_ok)
        # Place click marker at (mosaic_height − midpoint_y, midpoint_x).
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - self._edge_midpoints[i, 1],
              self._edge_midpoints[i, 0]]])
        # Overlay the selected edge's TRUE curved centerline (main edges are
        # drawn straight for robustness; the real path shows only on click).
        self._highlight_edge_path(u, v)
        # Enable / drive the video buttons
        m_best = _best_measurement(
            self.G.edges[u, v].get('measurements_piv'))
        # v4 migration: was m_best.get('tile_id')
        self._selected_tile_id = (MeasurementView(m_best).tile_id
                                  if m_best is not None else None)
        can_video = (self._selected_tile_id is not None
                     and bool(self.tiles)
                     and self.video_dir is not None)
        self._video_btn.setEnabled(can_video)
        if can_video and getattr(self, '_auto_video_chk', None) \
                and self._auto_video_chk.isChecked():
            self._load_video_overlay(int(self._selected_tile_id))
            self._unload_video_btn.setEnabled(True)
        # Build readable summary — show ALL measurements for this edge,
        # ordered best-quality-first.  Each measurement has a small
        # marker indicating whether PIV's internal gates accepted it
        # ("✓" usable / "✗" rejected) but we don't surface the tier
        # vocabulary anywhere visible.
        d = self.G.edges[u, v]
        meas = d.get('measurements_piv') or []
        meas_ranked = sorted(
            meas,
            # v4 migration: mm.get('snr_f0') → MeasurementView(mm).snr.h1_db.
            # quality_tier and snr_pulse are not in the mapping table; keep.
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            (MeasurementView(mm).snr.h1_db or 0),
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        info: List[str] = []
        info.append(f"Edge ({u}, {v})")
        info.append(f"  endpoints: ({self.G.nodes[u]['x']:.0f}, "
                    f"{self.G.nodes[u]['y']:.0f}) → "
                    f"({self.G.nodes[v]['x']:.0f}, "
                    f"{self.G.nodes[v]['y']:.0f})")
        info.append(f"  radius: {d.get('radius', float('nan')):.1f} px")
        info.append(f"  length: {d.get('length', float('nan')):.1f} px")
        info.append(f"  measurements: {len(meas)} "
                    f"(ranked by quality, best first)")
        display_keys = ('mean_Q', 'amp_Q', 'PI', 'phase', 'snr_pulse',
                        'snr_f0', 'f0_hz', 'v_max')
        # TODO(v4-migration): display_keys still uses legacy flat names
        # ('mean_Q', 'snr_f0', 'f0_hz', ...).  On v4 these won't be present
        # as flat keys on `mm` — need to route each key through MeasurementView
        # (mean_Q → harm.mean_Q, snr_f0 → snr.h1_db, f0_hz → harm.f0_hz,
        #  amp_Q → amp_Q property, PI → PI_rms property, phase → phase_deg,
        #  snr_pulse / v_max not exposed).
        for idx, mm in enumerate(meas_ranked):
            mark = '✓' if _measurement_usable(mm) else '✗'
            # v4 migration: was mm.get('tile_id')
            tile_id = MeasurementView(mm).tile_id
            info.append(f"\n  {mark} measurement {idx + 1} "
                        f"(tile {tile_id}):")
            for k in display_keys:
                if k in mm:
                    val = mm[k]
                    try:
                        info.append(f"    {k}: {float(val):.4g}")
                    except (TypeError, ValueError):
                        info.append(f"    {k}: {val}")
        # Per-harmonic Z-statistics + classification for the best
        # measurement.  Computed on demand (one harmonic fit per click,
        # cheap).  Shows the user which harmonics rise above the noise.
        best = meas_ranked[0] if meas_ranked else None
        best_mv = MeasurementView(best) if best is not None else None
        # v4 migration: was best.get('Q_t') / best.get('f0_hz', 2.5)
        if best is not None and best_mv.Q_t is not None:
            snrs = _harmonic_snrs(
                np.asarray(best_mv.Q_t, dtype=float),
                f0=float(best_mv.harm.f0_hz or 2.5),
                dt=1.0/250)
            if snrs is not None:
                klass = _harmonic_class(snrs)
                info.append("\n  Per-harmonic SNR (Z = amp / SE_amp):")
                for k in ('DC', 'H1', 'H2', 'H3'):
                    pass_mark = '✓' if snrs[k] >= HARMONIC_SNR_THRESHOLD else '·'
                    info.append(f"    {pass_mark} Z({k}) = {snrs[k]:.1f}")
                info.append(f"  Resolved: {_harmonic_class_label(klass)} "
                            f"(threshold Z > {HARMONIC_SNR_THRESHOLD:.0f})")
        self._info_box.setPlainText("\n".join(info))

    def _goto_edge_from_input(self):
        """Jump to an edge by its (u, v) node ids typed in the goto box.

        Reuses `_inspect_nearest_edge` at the edge's midpoint so all the
        normal click-selection state gets populated identically (info
        box, marker, path highlight, tile pick, video auto-load).  Also
        centers the napari viewport on the vessel so it is visible.
        """
        text = (self._goto_edge_input.text() or "").strip()
        if not text:
            return
        cleaned = (text.replace('(', ' ').replace(')', ' ')
                       .replace(',', ' ').replace(';', ' '))
        parts = cleaned.split()
        if len(parts) != 2:
            self._info_box.setPlainText(
                f"Bad Go-to input {text!r}. Expected two node ids like "
                f"'89617, 90142'.")
            return
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            self._info_box.setPlainText(
                f"Bad Go-to input {text!r}. Node ids must be integers.")
            return
        # Locate in edge_list (either direction).
        i = None
        for j, (a, b) in enumerate(self.edge_list):
            if {a, b} == {u, v}:
                i = j
                break
        if i is None:
            self._info_box.setPlainText(
                f"Edge ({u}, {v}) not in current graph.")
            return
        xm = float(self._edge_midpoints[i, 0])
        ym_graph = float(self._edge_midpoints[i, 1])
        y_display = float(self.mosaic_height - ym_graph)
        # Route through the click-selection path (max_px=1 forces the same
        # index we already located; distance is zero at the midpoint).
        self._inspect_nearest_edge(xm, y_display, max_px=1.0)
        # Center the napari camera on the vessel without changing zoom.
        try:
            cur_zoom = self.viewer.camera.zoom
            self.viewer.camera.center = (0.0, y_display, xm)
            self.viewer.camera.zoom = cur_zoom
        except Exception:
            pass

    # ── comparison set management ──────────────────────────────────────
    def _add_to_comparison(self):
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        if (u, v) in self._comparison_edges:
            return  # already in set
        self._comparison_edges.append((u, v))
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()

    def _clear_comparison(self):
        self._comparison_edges.clear()
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()

    def _refresh_comparison_markers(self):
        # Always tear down the previous edge overlay — napari Shapes
        # layers don't always repaint cleanly on .data/.edge_color reassign
        # (same bug worked around in the main edges layer rebuild).
        if self._comparison_edge_overlay_layer is not None:
            try:
                self.viewer.layers.remove(
                    self._comparison_edge_overlay_layer)
            except (KeyError, ValueError):
                pass
            self._comparison_edge_overlay_layer = None

        if self._comparison_marker_layer is None:
            return
        if not self._comparison_edges:
            self._comparison_marker_layer.data = np.empty((0, 2))
            return
        # Match the comparison plot's per-edge colors:
        #   index 0 → matplotlib C0 (blue, Edge A)
        #   index 1 → matplotlib C1 (orange, Edge B)
        # Any extras beyond the two used by the plot fade to grey.
        comp_colors = ['#1f77b4', '#ff7f0e']
        H = self.mosaic_height
        pts, marker_colors = [], []
        lines, line_colors, line_widths = [], [], []
        for k, (u, v) in enumerate(self._comparison_edges):
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            color = (comp_colors[k] if k < len(comp_colors)
                     else '#888888')
            # Mid-point dot for the small marker layer.
            pts.append([H - 0.5 * (y1 + y2), 0.5 * (x1 + x2)])
            marker_colors.append(color)
            # Thick bright line overlaying the actual edge — the
            # primary "this edge is in the comparison set" signal,
            # visible on top of any field colouring.
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
            line_colors.append(color)
            # Width is generous; the underlying edge_width caps at
            # EDGE_WIDTH_MAX=22 so 16 px sits well above any vessel
            # while not swamping the image.
            line_widths.append(16.0)

        # Small filled dot at each edge midpoint — secondary cue, easy
        # to see when the edge itself is short or off-screen.
        self._comparison_marker_layer.data = np.asarray(pts)
        self._comparison_marker_layer.size = 14
        try:
            self._comparison_marker_layer.border_color = marker_colors
            self._comparison_marker_layer.face_color = marker_colors
        except (AttributeError, TypeError):
            try:
                self._comparison_marker_layer.edge_color = marker_colors
                self._comparison_marker_layer.face_color = marker_colors
            except (AttributeError, TypeError):
                pass

        # Bold edge overlay — the headline marker.
        if lines:
            self._comparison_edge_overlay_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=line_colors,
                edge_width=line_widths,
                name='Comparison edges',
                opacity=0.85,
            )

    def _refresh_comparison_buttons(self):
        n = len(self._comparison_edges)
        self._plot_comp_btn.setText(f"Plot comparison ({n})")
        self._plot_comp_btn.setEnabled(n >= 2)
        self._clear_comp_btn.setEnabled(n > 0)

    def _build_sim_measurement(self, u, v, n_frames: int = 750,
                                quantity: str = 'Q'):
        """Reconstruct a measurement-shaped dict from the stored
        per-edge sim harmonics for edge (u, v).

        `quantity = 'Q'` pulls from `_sim_tmp_harmonics` (flow); `'P'`
        pulls from `_sim_tmp_p_harmonics` (midpoint pressure).  The
        returned dict has the same keys the comparison plot reads from
        `measurements_piv` (Q_t, f0_hz, mean_Q, amp_Q, PI, tile_id),
        so it can flow through the existing pipeline unchanged; the
        Q_t slot carries P(t) values in pressure mode and the caller
        relabels axes accordingly.

        Returns None when the edge has no sim data or f0 is missing.
        """
        if not self.G.has_edge(u, v):
            return None
        d = self.G.edges[u, v]
        harm_key = ('_sim_tmp_p_harmonics' if quantity == 'P'
                    else '_sim_tmp_harmonics')
        harm = d.get(harm_key)
        if harm is None:
            return None
        f0 = d.get('_sim_tmp_f0_hz')
        if f0 is None:
            f0 = self._sim_last_f0_hz
        try:
            f0 = float(f0)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(f0) and f0 > 0):
            return None
        arr = np.asarray(harm, dtype=complex)
        dt = 1.0 / 250.0
        t = np.arange(n_frames) * dt
        omega = 2 * np.pi * f0
        # Real-time reconstruction: f(t) = Σ_k Re{ c_k · exp(i k ω t) }
        #                                = Σ_k Re(c_k) cos(kωt) − Im(c_k) sin(kωt)
        sig = np.full(n_frames, arr[0].real, dtype=float)
        for k in range(1, len(arr)):
            sig += (arr[k].real * np.cos(k * omega * t)
                    - arr[k].imag * np.sin(k * omega * t))
        mean_val = float(arr[0].real)
        amp_val = float(abs(arr[1])) if len(arr) > 1 else float('nan')
        PI = (2.0 * amp_val / abs(mean_val)
              if abs(mean_val) > 1e-12 else float('nan'))
        return {
            'Q_t': sig,        # carries P(t) when quantity='P'
            'f0_hz': f0,
            'mean_Q': mean_val,
            'amp_Q': amp_val,
            'PI': PI,
            'tile_id': None,
            'quality_tier': 'A',
            'snr_pulse': 99.0,
            'snr_f0': 99.0,
            'fit_success': True,
        }

    def _plot_comparison(self):
        """Two-edge comparison figure focused on harmonic attenuation.

        Layout (three rows):
          Row 1 — two raw Q(t) panels (one per edge), shared y-axis;
                  amplitude attenuation reads visually.
          Row 2 — single-period overlay of bold harmonic fits (no
                  individual cycles); tiny corner note with per-edge Z
                  values for H₁, H₂.
          Row 3 — attenuation–phase scatter, one marker per harmonic.
                  x = Δφ (degrees, B − A); y = pulse-relative
                  attenuation. Fixed axes (−90°…+90°, 0…1.2) for
                  cross-call consistency. Reference lines at y=1, x=0.

        Pulse-relative attenuation (per harmonic k):

            atten_k = (|H_k^B| / Q̄_B) / (|H_k^A| / Q̄_A)

        strips out the trivial mean-flow rescaling and isolates the
        per-harmonic damping of the pulsatile component.  Q̄ is taken
        from the harmonic-fit DC term (a0).

        H₂ marker is filled if Z_{H₂} ≥ 3 on BOTH edges (resolved); else
        drawn as an open circle at 50 % alpha with the failing Z value
        tagged next to it.
        """
        if len(self._comparison_edges) < 2:
            return
        from ..analyze.harmonic import fit_harmonics
        dt = 1.0 / 250.0
        Z_THRESH_H2 = 3.0
        # Source selector: 'sim' uses reconstructed Q(t) (or P(t) for
        # pressure-mode fields) from the stored per-edge harmonics;
        # 'measured' uses PIV Q_t.
        src = getattr(self, 'current_source', 'measured')
        # Pressure mode: when the Quantity selector is P (sim only),
        # the comparison plot reconstructs midpoint P(t) instead of
        # Q(t) and relabels axes accordingly.
        quantity = ('P' if (src == 'sim'
                            and self.current_quantity == 'P')
                    else 'Q')
        # Pick measurement per edge (prefer the loaded-tile measurement
        # in measured mode; sim mode has one synthetic measurement per
        # edge).
        per_edge_data = []
        for (u, v) in self._comparison_edges:
            if not self.G.has_edge(u, v):
                continue
            if src == 'sim':
                m = self._build_sim_measurement(u, v, quantity=quantity)
                if m is None:
                    continue
                per_edge_data.append((u, v, m))
                continue
            meas = self.G.edges[u, v].get('measurements_piv') or []
            if not meas:
                continue
            # Focus-tile rule (same as the colormap + Plot Q(t)):
            # filter > loaded video > best-of-edge fallback.
            tid = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else self._video_tile_id)
            m = None
            if tid is not None:
                m = next((mm for mm in meas
                          if MeasurementView(mm).tile_id == tid), None)
            if m is None:
                m = _best_measurement(meas)
            # v4 migration: was m.get('Q_t')
            if m is None or MeasurementView(m).Q_t is None:
                continue
            per_edge_data.append((u, v, m))
        if len(per_edge_data) < 2:
            return
        per_edge_data = per_edge_data[:2]

        # ── Regime classification + A/B ordering ───────────────────────
        # Per-edge topo_av_score (0 = arterial side, 1 = venous side,
        # 0.5 = interior).  Pulse-source proximity ordering rule:
        #   arterial-dominated regime (both score < 0.5):
        #     Edge A = smaller graph_dist_art (closer to arterial source)
        #     pulse propagates A → B (same as DC flow)
        #   venous-dominated regime (both score > 0.5):
        #     Edge A = smaller graph_dist_ven (closer to venous sink,
        #              which is the *local* pulse source)
        #     pulse propagates A → B (opposite to DC flow)
        #   cross-regime: ordering is arbitrary; flag in the suptitle
        #              and render row-3 markers as open circles.
        def _edge_topo(u, v):
            d = self.G.edges[u, v]
            return (float(d.get('topo_av_score', np.nan)),
                    float(d.get('graph_dist_art', np.nan)),
                    float(d.get('graph_dist_ven', np.nan)))
        topo = [_edge_topo(u, v) for (u, v, _) in per_edge_data]
        scores = [t[0] for t in topo]
        if all(np.isfinite(scores)) and max(scores) < 0.5:
            regime = 'arterial'
        elif all(np.isfinite(scores)) and min(scores) > 0.5:
            regime = 'venous'
        elif any(not np.isfinite(s) for s in scores):
            regime = 'unknown'
        else:
            regime = 'cross'
        swapped = False
        if regime == 'arterial':
            # smaller graph_dist_art = closer to A source = upstream in pulse-prop
            if topo[1][1] < topo[0][1]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        elif regime == 'venous':
            # smaller graph_dist_ven = closer to V sink (= local pulse source)
            if topo[1][2] < topo[0][2]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        # cross / unknown: leave the user-supplied order
        if swapped:
            # Keep the in-canvas overlays in sync with the final A/B
            # assignment: the user clicked in one order, the regime
            # ordering may have swapped them, so the rings/edge highlight
            # colors must follow suit (A → blue, B → orange).
            new_order = [(u, v) for (u, v, _) in per_edge_data]
            remaining = [e for e in self._comparison_edges
                          if e not in new_order]
            self._comparison_edges = new_order + remaining
            self._refresh_comparison_markers()

        def _wrap(p):
            return (float(p) + np.pi) % (2 * np.pi) - np.pi

        edge_fits = []
        edge_colors = ['C0', 'C1']
        edge_letters = ['A', 'B']
        for (u, v, m) in per_edge_data:
            mv_m = MeasurementView(m)
            # v4 migration: was m['Q_t'] / m.get('f0_hz', 2.5)
            Q_t = np.asarray(mv_m.Q_t, dtype=float)
            T = len(Q_t)
            f0 = float(mv_m.harm.f0_hz or 2.5)
            try:
                hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                                    loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                h1, h2, h3 = harms.get(1), harms.get(2), harms.get(3)
                phi1 = (float(np.arctan2(-h1['B'], h1['A']))
                        if h1 is not None else float('nan'))
                phi2 = (float(np.arctan2(-h2['B'], h2['A']))
                        if h2 is not None else float('nan'))
                phi3 = (float(np.arctan2(-h3['B'], h3['A']))
                        if h3 is not None else float('nan'))
                amp1 = float(h1['amp']) if h1 is not None else float('nan')
                amp2 = float(h2['amp']) if h2 is not None else float('nan')
                amp3 = float(h3['amp']) if h3 is not None else float('nan')
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                se_amp = (sigma * np.sqrt(2.0 / T)
                          if T > 0 else float('inf'))
                z1 = amp1 / max(se_amp, 1e-30) if np.isfinite(amp1) else np.nan
                z2 = amp2 / max(se_amp, 1e-30) if np.isfinite(amp2) else np.nan
                z3 = amp3 / max(se_amp, 1e-30) if np.isfinite(amp3) else np.nan
            except Exception:
                a0 = 0.0
                amp1 = amp2 = amp3 = phi1 = phi2 = phi3 = np.nan
                z1 = z2 = z3 = np.nan
            edge_fits.append(dict(
                u=u, v=v, m=m, Q_t=Q_t, T=T, f0=f0, a0=a0,
                amp1=amp1, amp2=amp2, amp3=amp3,
                phi1=phi1, phi2=phi2, phi3=phi3,
                z1=z1, z2=z2, z3=z3))

        # Quantity-aware axis / title labels for Q vs P modes.
        unit = 'Pa' if quantity == 'P' else 'nL/s'
        sym = quantity  # 'Q' or 'P'
        sym_bar = f'{sym}̄'  # combining macron → Q̄ / P̄
        mean_label = f'mean_{sym}'

        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.55, wspace=0.30,
            left=0.09, right=0.97, top=0.91, bottom=0.07)

        # ── Top: raw signal, shared y across both panels ──
        ax_raw_A = fig.add_subplot(gs[0, 0])
        ax_raw_B = fig.add_subplot(gs[0, 1], sharey=ax_raw_A)
        for k, ed in enumerate(edge_fits):
            ax = (ax_raw_A, ax_raw_B)[k]
            t_axis = np.arange(ed['T']) * dt
            ax.plot(t_axis, ed['Q_t'], color=edge_colors[k],
                    lw=0.7, alpha=0.9)
            # mean_Q / PI from the fresh harmonic fit (a0, amp1) — canonical
            # graphs don't carry legacy mean_Q/PI keys, so fall back from m.
            a0_k = ed['a0']
            amp1_k = ed['amp1']
            if np.isfinite(a0_k):
                mean_disp = abs(a0_k)
            else:
                # v4 migration: was ed['m'].get('mean_Q')
                _mq = MeasurementView(ed['m']).harm.mean_Q
                mean_disp = float(_mq) if _mq is not None else float('nan')
            if (np.isfinite(amp1_k) and np.isfinite(a0_k)
                    and abs(a0_k) > 0):
                pi_disp = 2.0 * amp1_k / abs(a0_k)
            else:
                # 'PI' is a legacy H1-only Gosling PI; MeasurementView.PI_rms
                # is Parseval-form (all harmonics).  Not exactly the same
                # quantity; keep raw dict-read as best-effort v3 fallback.
                # TODO(v4-migration): PI legacy flat key dropped in v4.
                pi_disp = float(ed['m'].get('PI', float('nan')))
            # SNR_AC = 10·log10(Z1² + Z2² + Z3²) — total cardiac-harmonic
            # power SNR from this edge's fresh fit (DC-free).
            _z_sq_k = sum(z*z for z in (ed['z1'], ed['z2'], ed['z3'])
                           if np.isfinite(z))
            snr_ac_db_k = (10.0 * np.log10(_z_sq_k)
                            if _z_sq_k > 0 else float('nan'))
            ax.set_title(
                f"Edge {edge_letters[k]}: ({ed['u']}, {ed['v']}) "
                # v4 migration: was ed['m'].get('tile_id')
                f"tile {MeasurementView(ed['m']).tile_id}\n"
                f"{mean_label}={mean_disp:.3f}  "
                f"PI={pi_disp:.2f}  "
                f"SNR_AC={snr_ac_db_k:.1f} dB  "
                f"f₀={ed['f0']:.2f} Hz",
                fontsize=9, color=edge_colors[k])
            ax.grid(alpha=0.3)
            ax.set_xlabel('time (s)')
            if k == 0:
                ax.set_ylabel(f'{sym} ({unit})')
        ymins = [np.nanmin(ed['Q_t']) for ed in edge_fits]
        ymaxs = [np.nanmax(ed['Q_t']) for ed in edge_fits]
        ax_raw_A.set_ylim(min(ymins), max(ymaxs))

        # ── Middle: per-edge single-period fold with harmonic decomposition ──
        # Two panels (A on left, B on right), each showing (Q − Q̄) / Q̄
        # over one cycle, decomposed into individual H1, H2, H3 components
        # plus the combined fit.  Mean-normalised so the comparison is of
        # pulsatile fraction shape, not absolute amplitude.
        ax_fold_A = fig.add_subplot(gs[1, 0])
        ax_fold_B = fig.add_subplot(gs[1, 1], sharey=ax_fold_A)
        ax_folds = [ax_fold_A, ax_fold_B]
        N_fold = 200
        phase_grid = np.linspace(0.0, 1.0, N_fold, endpoint=False)
        for k, ed in enumerate(edge_fits):
            ax = ax_folds[k]
            f0 = ed['f0']
            qbar = ed['a0']
            if not (np.isfinite(f0) and f0 > 0
                    and np.isfinite(qbar) and abs(qbar) > 0):
                ax.text(0.5, 0.5, '(no fit)', transform=ax.transAxes,
                         ha='center', va='center', color='#888')
                continue
            color = edge_colors[k]
            T_period = 1.0 / f0
            omega = 2 * np.pi * f0
            t_one = phase_grid * T_period
            qbar_abs = abs(qbar)
            # Individual harmonic components — lighter, dashed/dotted
            harmonic_styles = [
                (1, ed['amp1'], ed['phi1'], '--', 0.7,
                 f"H₁ (|H₁|/{sym_bar}={ed['amp1']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp1']) else None),
                (2, ed['amp2'], ed['phi2'], ':',  0.7,
                 f"H₂ (|H₂|/{sym_bar}={ed['amp2']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp2']) else None),
                (3, ed['amp3'], ed['phi3'], (0, (1, 3)), 0.6,
                 f"H₃ (|H₃|/{sym_bar}={ed['amp3']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp3']) else None),
            ]
            y_total = np.zeros(N_fold)
            for kh, amp_k, phi_k, ls, lw_alpha, lab in harmonic_styles:
                if not (np.isfinite(amp_k) and np.isfinite(phi_k)):
                    continue
                y_k = amp_k * np.cos(kh * omega * t_one + phi_k)
                y_total += y_k
                ax.plot(phase_grid, y_k / qbar_abs,
                         color=color, lw=1.0, alpha=lw_alpha,
                         linestyle=ls, label=lab)
            # Combined (bold)
            ax.plot(phase_grid, y_total / qbar_abs,
                     color=color, lw=2.4, alpha=0.95,
                     label="sum (H₁+H₂+H₃)")
            ax.axhline(0.0, color='#888', lw=0.5, ls=':')
            ax.set_xlim(0, 1)
            ax.set_xlabel('cycle fraction')
            if k == 0:
                ax.set_ylabel(f'({sym} − {sym_bar}) / {sym_bar}')
            # Z-resolution tags
            zs_disp = ' '.join([
                f"Z_H{kh}={ed[f'z{kh}']:.1f}" if np.isfinite(ed[f'z{kh}']) else f"Z_H{kh}=n/a"
                for kh in (1, 2, 3)
            ])
            ax.set_title(
                f"Edge {edge_letters[k]} harmonic decomposition\n"
                f"{zs_disp}",
                fontsize=9, color=edge_colors[k])
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
            ax.grid(alpha=0.3)

        # ── Bottom row: per-harmonic transfer function (H₁, H₂) ──
        # Left  — amplitude pulsatile fraction |H_k|/Q̄ for each edge.
        # Right — phase shift Δφ_k between A and B.
        # Both panels overlay a faint √k diffusive prediction so
        # observed vs predicted is a one-glance read.  H₃ is dropped
        # since (a) it's the noisiest and (b) two harmonics already
        # over-determine the diffusive scaling.
        edA, edB = edge_fits[0], edge_fits[1]
        # v4 migration: was edA['m'].get('mean_Q') / edB['m'].get('mean_Q')
        def _fallback_mean_q(_m):
            _mq = MeasurementView(_m).harm.mean_Q
            return float(_mq) if _mq is not None else float('nan')
        qbarA = (edA['a0'] if np.isfinite(edA['a0']) and abs(edA['a0']) > 0
                 else _fallback_mean_q(edA['m']))
        qbarB = (edB['a0'] if np.isfinite(edB['a0']) and abs(edB['a0']) > 0
                 else _fallback_mean_q(edB['m']))
        cross_regime = (regime == 'cross')

        def _pulse_frac(amp, q):
            if not (np.isfinite(amp) and np.isfinite(q)) or q == 0:
                return np.nan
            return amp / abs(q)

        ks = [1, 2]
        amps_A = [edA['amp1'], edA['amp2']]
        amps_B = [edB['amp1'], edB['amp2']]
        phis_A = [edA['phi1'], edA['phi2']]
        phis_B = [edB['phi1'], edB['phi2']]
        zs_A   = [edA.get('z1', np.nan), edA.get('z2', np.nan)]
        zs_B   = [edB.get('z1', np.nan), edB.get('z2', np.nan)]

        pA = [_pulse_frac(amps_A[i], qbarA) for i in range(2)]
        pB = [_pulse_frac(amps_B[i], qbarB) for i in range(2)]

        ax_atten = fig.add_subplot(gs[2, 0])
        ax_phase = fig.add_subplot(gs[2, 1])

        # ── Attenuation panel ───────────────────────────────────────
        # Stems show |H_k|/Q̄ per edge.  Atten annotation in the gap.
        x_off = 0.16
        STEM_LW = 6
        all_p = [p for p in pA + pB if np.isfinite(p) and p > 0]
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            xA, xB = k - x_off, k + x_off
            if np.isfinite(pA[i]) and pA[i] > 0:
                ax_atten.vlines(xA, 1e-4, pA[i], colors='C0',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xA, pA[i], 'o', ms=12,
                              mfc='C0' if solid else 'none',
                              mec='C0', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            if np.isfinite(pB[i]) and pB[i] > 0:
                ax_atten.vlines(xB, 1e-4, pB[i], colors='C1',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xB, pB[i], 'o', ms=12,
                              mfc='C1' if solid else 'none',
                              mec='C1', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            # Atten label between the stems.
            if (np.isfinite(pA[i]) and pA[i] > 0
                    and np.isfinite(pB[i]) and pB[i] > 0):
                atten = pB[i] / pA[i]
                ax_atten.annotate(
                    '', xy=(xB, pB[i]), xytext=(xA, pA[i]),
                    arrowprops=dict(arrowstyle='->', color='#444',
                                    lw=1.2,
                                    connectionstyle='arc3,rad=-0.2'))
                ax_atten.text(
                    k, np.sqrt(pA[i] * pB[i]),
                    f"×{atten:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='#222',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#bbb', alpha=0.9))

        ax_atten.plot([], [], 'o', color='C0', ms=10, label='Edge A')
        ax_atten.plot([], [], 'o', color='C1', ms=10, label='Edge B')
        ax_atten.legend(fontsize=7, loc='upper right')

        ax_atten.set_yscale('log')
        if all_p:
            ax_atten.set_ylim(max(1e-3, 0.3 * min(all_p)),
                              max(all_p) * 1.6)
        else:
            ax_atten.set_ylim(1e-3, 1.0)
        ax_atten.set_xlim(0.5, 2.5)
        ax_atten.set_xticks([1, 2])
        ax_atten.set_xticklabels(['H₁', 'H₂'])
        ax_atten.set_xlabel('harmonic')
        ax_atten.set_ylabel(
            f'|H$_k$| / {sym_bar}  (pulsatile fraction)')
        atten_title = 'Amplitude per harmonic'
        if cross_regime:
            atten_title += ' [cross-regime: hollow]'
        ax_atten.set_title(atten_title, fontsize=10)
        ax_atten.grid(alpha=0.3, which='both', axis='y')

        # ── Phase panel ─────────────────────────────────────────────
        dphis = []
        for i, k in enumerate(ks):
            if np.isfinite(phis_A[i]) and np.isfinite(phis_B[i]):
                dphis.append(np.degrees(_wrap(phis_B[i] - phis_A[i])))
            else:
                dphis.append(np.nan)
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            if np.isfinite(dphis[i]):
                ax_phase.vlines(k, 0, dphis[i],
                                colors='#444', lw=STEM_LW * 0.6,
                                alpha=0.9 if solid else 0.4)
                ax_phase.plot(k, dphis[i], 'o', ms=13,
                              mfc='#444' if solid else 'none',
                              mec='#444', mew=2.0,
                              alpha=0.95 if solid else 0.7)
                ax_phase.text(k + 0.08, dphis[i],
                              f"{dphis[i]:+.0f}°",
                              ha='left', va='center', fontsize=9,
                              color='#222')
        ax_phase.axhline(0.0, color='#888', lw=0.7, ls='--', alpha=0.7)
        # Symmetric y-range around 0 so positive and negative shifts read.
        if any(np.isfinite(d) for d in dphis):
            absmax = max(abs(d) for d in dphis if np.isfinite(d))
            ymax = max(30.0, absmax * 1.25)
            ax_phase.set_ylim(-ymax, ymax)
        else:
            ax_phase.set_ylim(-90, 90)
        ax_phase.set_xlim(0.5, 2.5)
        ax_phase.set_xticks([1, 2])
        ax_phase.set_xticklabels(['H₁', 'H₂'])
        ax_phase.set_xlabel('harmonic')
        ax_phase.set_ylabel('Δφ  (degrees,  B − A)')
        ax_phase.set_title('Phase shift per harmonic', fontsize=10)
        ax_phase.grid(alpha=0.3, axis='y')

        # Title (bold) + subtitle (smaller, italic) for regime context.
        if regime == 'arterial':
            subtitle = ("arterial-dominated pair; pulse propagates "
                        "A → B (DC flow A → B)")
        elif regime == 'venous':
            subtitle = ("venous-dominated pair; pulse propagates A → B "
                        "(DC flow is reversed: B → A)")
        elif regime == 'cross':
            subtitle = ("CROSS-REGIME pair — attenuation does not "
                        "directly apply (markers hollow)")
        else:
            subtitle = ("regime unknown — topo_av_score missing on one "
                        "or both edges")
        main_title = (
            "Edge A vs B — pressure attenuation"
            if quantity == 'P'
            else "Edge A vs B — harmonic attenuation")
        if self._video_tile_id is not None:
            main_title += f"  (tile {self._video_tile_id})"
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.985)
        fig.text(0.5, 0.953, subtitle,
                 ha='center', va='top', fontsize=10, style='italic',
                 color='#444')
        plt.show()

    # ── Optical-flow re-analysis (read-only) ──────────────────────────
    def _of_tile_geometry(self):
        """Parse `tile_positions` into TileInfo + reference affine, applying
        the SAME re-referencing the batch does (batch_analyze_mosaic_piv lines
        912-926), cached.  Returns (tiles_by_vid, affine_sx, affine_sy) and
        stores the batch tile offsets on self.  This lets fresh OF reuse the
        batch's `mosaic_to_tile_coords` / `mosaic_radius_to_tile` so a
        recompute reproduces the stored Q(t) exactly."""
        if getattr(self, '_of_tinfo', None) is not None:
            return self._of_tinfo, self._of_affine_sx, self._of_affine_sy
        from ..cli.batch_analyze_mosaic import (
            parse_tile_positions, _get_reference_affine)
        import json as _json
        tp = str(self.tile_positions_path)
        tiles = parse_tile_positions(tp)
        affine_sy, affine_sx = _get_reference_affine(tp)
        jd = _json.load(open(tp)); ref = int(jd.get('reference_vid', 1))
        if abs(affine_sy - 1) > 1e-6 or abs(affine_sx - 1) > 1e-6:
            rtd = jd['tiles'][str(ref)]
            rty = rtd.get('translate_y', 0.0); rtx = rtd.get('translate_x', 0.0)
            for t in tiles.values():
                oty = t._translate_y or 0.0; otx = t._translate_x or 0.0
                t._translate_y = (oty - rty) / affine_sy
                t._translate_x = (otx - rtx) / affine_sx
                t.tile_width /= affine_sx; t.tile_height /= affine_sy
        self._of_tile_offset_x = min(t.top_left_x for t in tiles.values())
        self._of_tile_offset_y = min(t.top_left_y for t in tiles.values())
        self._of_tinfo = tiles
        self._of_affine_sx = affine_sx; self._of_affine_sy = affine_sy
        return tiles, affine_sx, affine_sy

    def _show_kloosterman_slices(self):
        """Per-slice Kloosterman diagnostic — R, V_CL, Q along the vessel
        arc, with median lines and comparison to Q_direct / Q_pois_free.
        Shows how the median-of-slices Q_kl is derived and which slices
        got gated out (R constraint / low-signal)."""
        import matplotlib.pyplot as plt
        from .mosaic import show_figure
        data = getattr(self, '_last_of_pipeline', None)
        if not data:
            self._info_box.setPlainText(
                "Run 'Compute fresh PIV/OF' on an edge first.")
            return
        res = data['res']
        v_conv = data['v_conv']; q_conv = data['q_conv']
        r_tile_px = float(data.get('r_tile', np.nan))
        px_um = q_conv * (0.0 if q_conv == 0 else 1)  # will overwrite below
        # Prefer analyze constants for the µm scale (px→µm on R axis)
        try:
            from ..analyze.constants import PX_SIZE_UM
            _px_um = float(PX_SIZE_UM)
        except Exception:
            _px_um = 1.7
        Q_slice_px = np.asarray(res.get('Q_kloosterman_per_slice', []),
                                 dtype=float)
        R_slice_px = np.asarray(res.get('R_kloosterman_per_slice', []),
                                 dtype=float)
        if len(Q_slice_px) == 0:
            self._info_box.setPlainText(
                "No Kloosterman per-slice data in the OF result.\n"
                "(Batch stores medians; re-run 'Compute fresh PIV/OF' "
                "which populates the per-slice arrays.)")
            return
        # Physical units.  Our stored `Q_kloosterman_per_slice` uses
        # `use_r_prior_area=True` — Q_slice = ½·V_CL·π·R_prior² — i.e.
        # V_CL from the Poiseuille fit but AREA fixed to the mask radius.
        # The Kloosterman 2014 paper convention uses the *fitted* R_slice
        # for the area too (Q_slice = ½·V_CL·π·R_fit²), which is what
        # ``Q_slice_paper_nLs`` reproduces here — rescale by (R_fit/R_prior)².
        Q_slice_ours_nLs = Q_slice_px * q_conv        # our default (R_prior area)
        R_slice_um       = R_slice_px * _px_um        # µm per slice (fit)
        with np.errstate(invalid='ignore', divide='ignore'):
            area_ratio = np.where(
                np.isfinite(R_slice_px) & (r_tile_px > 0),
                (R_slice_px / r_tile_px) ** 2, np.nan)
        Q_slice_paper_nLs = Q_slice_ours_nLs * area_ratio
        # V_CL is independent of the area convention — same for both.
        with np.errstate(invalid='ignore', divide='ignore'):
            V_slice_ums = np.where(
                np.isfinite(R_slice_um) & (R_slice_um > 0),
                2.0 * (Q_slice_ours_nLs * 1e6) / (np.pi * (r_tile_px * _px_um) ** 2),
                np.nan)
        arc = np.arange(len(Q_slice_px), dtype=float)
        valid = np.isfinite(Q_slice_px) & (Q_slice_px != 0)
        Q_med_ours  = float(np.nanmedian(Q_slice_ours_nLs[valid]))  if valid.any() else np.nan
        Q_med_paper = float(np.nanmedian(Q_slice_paper_nLs[valid])) if valid.any() else np.nan
        R_med = float(np.nanmedian(R_slice_um[valid])) if valid.any() else np.nan
        V_med = float(np.nanmedian(V_slice_ums[valid])) if valid.any() else np.nan
        # Prior R (from graph) + free-fit R for context lines
        R_prior_um = r_tile_px * _px_um
        R_free_um  = float(res.get('R_fit_free', np.nan)) * _px_um
        V_free_ums = float(res.get('v0_free', np.nan)) * v_conv
        V_fix_ums  = float(res.get('v0_fixed', np.nan)) * v_conv
        Q_direct_nLs = float(V_fix_ums / 1e6) * np.pi * R_prior_um ** 2 / 2.0
        Q_pois_free_nLs = (float(V_free_ums / 1e6)
                            * np.pi * R_free_um ** 2 / 2.0
                            if np.isfinite(R_free_um) else np.nan)

        # ── (0,0) Vessel with per-slice R half-widths ───────────────────
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(15, 11), dpi=110)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28,
                       height_ratios=[1.0, 0.85])
        fig.suptitle(f"Kloosterman per-slice — {data.get('title','')}",
                      fontsize=11)

        ax = fig.add_subplot(gs[0, :])
        vfield = res.get('vfield_mean')
        cl_used = res.get('cl_used')     # ROI-local (row, col)
        if vfield is not None and cl_used is not None and len(cl_used) >= 2:
            roi_h, roi_w = vfield.shape[:2]
            # Grayscale background = mean intensity over the ROI (from the
            # frames the OF ran on).  Gives full vessel context so parts
            # outside the tight vessel mask are still visible instead of
            # showing black.
            frames = data.get('frames')
            r0_off, c0_off = res.get('roi_offset', (0, 0))
            bg = None
            if frames is not None:
                fr = np.asarray(frames)
                if fr.ndim == 3 and fr.shape[0] >= 1:
                    mean_frame = fr.mean(axis=0)
                    r1_off = min(mean_frame.shape[0], r0_off + roi_h)
                    c1_off = min(mean_frame.shape[1], c0_off + roi_w)
                    bg = mean_frame[r0_off:r1_off, c0_off:c1_off]
                    lo, hi = np.percentile(bg, (1, 99))
                    if hi > lo:
                        ax.imshow(bg, cmap='gray',
                                   vmin=lo, vmax=hi, alpha=1.0)
            speed_ums = np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2) * v_conv
            # Overlay the OF magnitude with alpha where sampled.  Cells
            # outside the vessel mask are transparent so the grayscale
            # intensity below shows through.
            masked_speed = np.ma.masked_where(speed_ums <= 0, speed_ums)
            im = ax.imshow(masked_speed, cmap='hot',
                            vmin=0, vmax=float(np.percentile(speed_ums, 99)),
                            alpha=0.85 if bg is not None else 1.0)
            plt.colorbar(im, ax=ax, label='|v| (µm/s)', shrink=0.75)
            cl = np.asarray(cl_used, dtype=float)
            # Plot the EXACT centerline the analysis used (cl_used).  The
            # refinement now smooths the per-slice offsets
            # (ridge_offset_smooth_sigma), so no display-only smoothing is
            # needed and the drawn curve == the curve Q was computed on.
            cl_smooth = cl.copy()
            ax.plot(cl_smooth[:, 1], cl_smooth[:, 0], color='cyan', lw=1.5,
                     alpha=0.9,
                     label='CL (analysis centerline); segment half-widths = R_fit')
            # Local tangents from finite differences on smoothed CL.
            t = np.zeros_like(cl_smooth)
            t[1:-1] = cl_smooth[2:] - cl_smooth[:-2]
            t[0]    = cl_smooth[1] - cl_smooth[0]
            t[-1]   = cl_smooth[-1] - cl_smooth[-2]
            norm_len = np.hypot(t[:, 0], t[:, 1])
            norm_len[norm_len < 1e-9] = 1.0
            t /= norm_len[:, None]
            n = np.column_stack([t[:, 1], -t[:, 0]])   # perpendicular
            # Decimate slices for display so adjacent perpendiculars don't
            # visually merge into a mess.  Target spacing = R_prior along
            # the arc → visible neighbours don't overlap by more than a
            # slice-width.  Slices in between are shown as small ticks so
            # you can still see coverage.
            N = min(len(cl_smooth), len(R_slice_px))
            arc_len = np.concatenate([[0.0], np.cumsum(np.hypot(
                np.diff(cl_smooth[:, 0]), np.diff(cl_smooth[:, 1])))])
            step_target = max(r_tile_px * 1.0, 4.0)
            show_full = np.zeros(N, dtype=bool)
            last_arc = -np.inf
            for i in range(N):
                if arc_len[i] - last_arc >= step_target:
                    show_full[i] = True
                    last_arc = arc_len[i]
            # ticks: everything not shown-full
            for i in range(N):
                r_use = R_slice_px[i] if np.isfinite(R_slice_px[i]) \
                        else r_tile_px
                if not np.isfinite(r_use) or r_use <= 0:
                    continue
                colour = 'lime' if valid[i] else '#888888'
                if show_full[i]:
                    p1 = cl_smooth[i] - r_use * n[i]
                    p2 = cl_smooth[i] + r_use * n[i]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=colour,
                             lw=1.6, alpha=0.9)
                else:
                    p1 = cl_smooth[i] - 2.0 * n[i]
                    p2 = cl_smooth[i] + 2.0 * n[i]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=colour,
                             lw=0.6, alpha=0.55)
            n_shown = int(show_full.sum())
            hint = (f'  ·  showing {n_shown}/{N} slice half-widths '
                    f'(spacing ≈ R_prior); the rest as tick marks')
            ax.set_title(
                f'vessel with per-slice fit half-widths — '
                f'{int(valid.sum())}/{N} slices accepted '
                f'(lime = accepted, gray = rejected)  '
                f'R_prior = {R_prior_um:.1f} µm{hint}')
            ax.legend(fontsize=8, loc='upper right',
                       facecolor='black', edgecolor='none',
                       labelcolor='white', framealpha=0.55)
            # Rotate into the mosaic frame (same as the OF-pipeline panel)
            # so this matches the main viewer's orientation.
            try:
                _tid = int(self._selected_tile_id)
                _L = self._of_geom.tiles[_tid].A_full[:2, :2]
                import matplotlib.transforms as _mtx
                _aff = _mtx.Affine2D.from_values(
                    _L[1, 1], _L[0, 1], _L[1, 0], _L[0, 0], 0.0, 0.0)
                for _art in list(ax.images) + list(ax.lines) + list(ax.collections):
                    _art.set_transform(_aff + ax.transData)
                _cor = np.array([[0, 0], [roi_w, 0], [0, roi_h], [roi_w, roi_h]], float)
                _xt = _L[1, 1] * _cor[:, 0] + _L[1, 0] * _cor[:, 1]
                _yt = _L[0, 1] * _cor[:, 0] + _L[0, 0] * _cor[:, 1]
                _pad = 0.05 * max(np.ptp(_xt), np.ptp(_yt), 1.0)
                ax.set_xlim(_xt.min() - _pad, _xt.max() + _pad)
                ax.set_ylim(_yt.max() + _pad, _yt.min() - _pad)
                ax.set_aspect('equal')
            except Exception as _e:
                print(f"  (KL panel orient skipped: {_e})")
            ax.set_xlabel('col (ROI px)'); ax.set_ylabel('row (ROI px)')
        else:
            ax.text(0.5, 0.5,
                     'no velocity field / centerline available',
                     transform=ax.transAxes, ha='center', va='center',
                     color='gray')
            ax.set_xticks([]); ax.set_yticks([])

        # ── (1,0) R per slice ─────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(arc[valid], R_slice_um[valid], 'ko', ms=4,
                 label=f'per-slice R  (n={int(valid.sum())})')
        ax.axhline(R_med, color='green', ls='--', lw=1.4,
                    label=f'median R = {R_med:.1f} µm')
        ax.axhline(R_prior_um, color='blue', ls=':', lw=1.2,
                    label=f'R_prior (graph) = {R_prior_um:.1f} µm')
        if np.isfinite(R_free_um):
            ax.axhline(R_free_um, color='red', ls=':', lw=1.2,
                        label=f'R_free = {R_free_um:.1f} µm')
        ax.axhspan(0.2 * R_prior_um, 3.0 * R_prior_um,
                    color='yellow', alpha=0.06,
                    label='gate ×[0.2, 3.0] R_prior')
        ax.set_xlabel('slice index'); ax.set_ylabel('R (µm)')
        ax.set_title('per-slice radius'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

        # ── (1,1) V_CL per slice ─────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(arc[valid], V_slice_ums[valid], 'ko', ms=4,
                 label='per-slice V_CL')
        if np.isfinite(V_med):
            ax.axhline(V_med, color='green', ls='--', lw=1.4,
                        label=f'median V_CL = {V_med:.0f} µm/s')
        if np.isfinite(V_fix_ums):
            ax.axhline(V_fix_ums, color='blue', ls=':', lw=1.2,
                        label=f'V0 (fixed R fit) = {V_fix_ums:.0f}')
        if np.isfinite(V_free_ums):
            ax.axhline(V_free_ums, color='red', ls=':', lw=1.2,
                        label=f'V0 (free R fit) = {V_free_ums:.0f}')
        ax.set_xlabel('slice index'); ax.set_ylabel('V_CL (µm/s)')
        ax.set_title('per-slice centerline velocity'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

        # ── (1,2) Q per slice — ours (R_prior area) vs paper (R_fit area) ─
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(arc[valid], Q_slice_ours_nLs[valid], 'o',
                 color='#1f77b4', ms=4,
                 label=f'ours: Q = ½·V_CL·π·R_prior²  (n={int(valid.sum())})')
        ax.plot(arc[valid], Q_slice_paper_nLs[valid], 's',
                 color='#d62728', ms=4, alpha=0.85,
                 label='paper: Q = ½·V_CL·π·R_fit²')
        if np.isfinite(Q_med_ours):
            ax.axhline(Q_med_ours, color='#1f77b4', ls='--', lw=1.6,
                        label=f'median (ours, current default) = {Q_med_ours:.3f} nL/s')
        if np.isfinite(Q_med_paper):
            ax.axhline(Q_med_paper, color='#d62728', ls='--', lw=1.6,
                        label=f'median (Kloosterman 2014) = {Q_med_paper:.3f} nL/s')
        if np.isfinite(Q_direct_nLs):
            ax.axhline(Q_direct_nLs, color='gray', ls=':', lw=1.0,
                        label=f'Q_direct = {Q_direct_nLs:.3f}')
        if np.isfinite(Q_pois_free_nLs):
            ax.axhline(Q_pois_free_nLs, color='black', ls=':', lw=1.0,
                        label=f'Q_pois_free = {Q_pois_free_nLs:.3f}')
        ax.set_xlabel('slice index'); ax.set_ylabel('Q (nL/s)')
        ax.set_title('per-slice Q — ours vs paper'); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        show_figure(fig, position='top-right')

    def _render_of_pipeline(self, data):
        """Browsable SEQUENCE of figures walking the entire OF pipeline, so
        every processing step is visible on its own page:

          Fig 1  Input & preprocessing  — ROI selection, background
                 subtraction, mean flow field.
          Fig 2  Centerline refinement  — coarse rigid shift, per-slice
                 parabola snap, offset smoothing, final CL.
          Fig 3  Radii & cross-sections — vessel mask + tangent, and the
                 SEGMENTED (mask) radius vs the INFERRED (fit) radius drawn
                 directly on the video crop AND the speed heatmap, with a
                 perpendicular velocity-arrow fan at several cross-sections
                 (arrow length = the fitted Poiseuille profile).  Plus the
                 radial profile with all three fits.
          Fig 4  Kloosterman deep-dive  — per-slice R / V_CL / Q, the
                 median-band gate, the reject-reason breakdown, the vessel
                 with per-slice half-widths, and how (Q_char, V_CL_char,
                 R_char) are derived.
          Fig 5  Q estimators & temporal — estimator comparison, cycle Q(t),
                 FFT/SNR, and the overall verdict.

        Driven entirely by the ONE `unified` result dict (+ its `diag`
        block) — what you see here is exactly what produced the reported Q.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        from scipy.ndimage import uniform_filter1d
        from .mosaic import show_figure
        try:
            from ..analyze.centerline import centerline_geometry
        except Exception:
            centerline_geometry = None

        res = data['res']
        diag = res.get('diag') or {}
        v_conv = float(data['v_conv']); q_conv = float(data['q_conv'])
        r_tile = float(data.get('r_tile', np.nan))            # R_prior (px)
        frame_dt_s = float(data.get('frame_dt_s', 1.0 / 250))
        f0_hz = float(data.get('f0_hz') or 2.5)
        best_key = data.get('best_key', 'kl')
        u, v = data.get('edge', (0, 0))
        tile_id = data.get('tile_id')
        px_um = v_conv * frame_dt_s                            # effective µm/px
        R_prior_um = r_tile * px_um

        r0, c0 = diag.get('roi_offset', res.get('roi_offset', (0, 0)))
        H, W = diag.get('roi_shape', (0, 0))
        vfield = res.get('vfield_mean')
        if (H == 0 or W == 0) and vfield is not None:
            H, W = vfield.shape[:2]
        speed = (np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2) * v_conv
                 if vfield is not None else None)
        _spd_hi = (float(np.percentile(speed[speed > 0], 99))
                   if speed is not None and np.any(speed > 0) else 1.0)

        def _arr(x):
            return np.asarray(x, dtype=float) if x is not None else None
        cl_init = _arr(diag.get('cl_initial'))
        cl_coarse = _arr(diag.get('cl_coarse'))
        cl_ref = _arr(res.get('cl_used'))
        offs_raw = diag.get('offsets_raw')
        offs_sm = diag.get('offsets_smooth')

        # radii (px) + fit velocities (px/frame)
        R_free_px = float(res.get('R_fit_free', np.nan))
        R_kl_px = float(res.get('R_kloosterman', np.nan))
        v0_fix = float(res.get('v0_fixed', np.nan))
        v0_free = float(res.get('v0_free', np.nan))
        R_free_ratio = (R_free_px / r_tile
                        if (np.isfinite(R_free_px) and r_tile > 0) else np.nan)

        # ── quality verdict (suptitle across all pages) ─────────────────
        bc = np.asarray(res.get('bin_centers', []), float)
        pv = np.abs(np.asarray(res.get('profile_v', []), float)) * v_conv
        bc_um = bc * px_um
        fin = np.isfinite(pv)
        in_win = np.abs(bc) <= max(r_tile, 1e-6)
        cov = float((fin & in_win).sum()) / float(max(in_win.sum(), 1))
        # Measure the wall shell at the FITTED radius when the free-R fit found a
        # wider-but-bounded vessel (segmentation under-measured R).  Otherwise an
        # under-segmented parabola reads as "blunt" only because the R_seg shell
        # sits on the vessel CORE, not its wall (e.g. R_seg=13µm on a real 25µm
        # vessel).  Keep R_seg when the free fit is absent or actually railing the
        # gate (ratio ≥ 1.95), where the fitted R is itself untrustworthy.
        R_wall_um = R_prior_um
        if (np.isfinite(R_free_ratio) and 1.15 < R_free_ratio < 1.95
                and np.isfinite(R_free_px) and R_free_px > 0):
            R_wall_um = R_free_px * px_um
        core = fin & (np.abs(bc_um) <= 0.30 * R_wall_um)
        shell = fin & (np.abs(bc_um) >= 0.80 * R_wall_um) & (np.abs(bc_um) <= 1.05 * R_wall_um)
        v_peak = (float(np.nanmax(pv[core])) if core.any()
                  else (float(np.nanmax(pv[fin])) if fin.any() else np.nan))
        v_edge = float(np.nanmean(pv[shell])) if shell.any() else np.nan
        taper = (1.0 - v_edge / v_peak) if (np.isfinite(v_peak) and v_peak > 1e-9
                                            and np.isfinite(v_edge)) else np.nan
        cropped = cov < 0.60
        # RAILING = the free-R fit actually rails the display gate ([0.5,2.0]·R);
        # a moderate 1.3–1.9× is a legitimately wider-than-segmented vessel, not
        # a runaway fit (neighbour bleed is already cut upstream).  (taper above is
        # kept as an informational descriptor only — no BLUNT flag: it false-failed
        # under-segmented but otherwise clean vessels.)
        railing = np.isfinite(R_free_ratio) and R_free_ratio >= 1.95
        n_valid_kl = int(res.get('Q_kloosterman_n_valid', 0) or 0)
        # Split the verdict by WHAT each flag invalidates.  CROP starves every
        # estimator of data → always a hard failure.  RAILING describes the R-FIT
        # geometry: it invalidates Q only when the canonical estimator IS an R-fit
        # (Pois free-/mask-R).  When the reported Q is Kloosterman — per-slice,
        # independent of the free-R blow-up — with enough valid slices, RAILING is
        # an advisory NOTE, not a failure.  (E.g. a clean central peak with low
        # side-shoulders that don't trip neighbour rejection but still drag free-R
        # to >2×: the Kloosterman Q is fine.)
        hard, soft = [], []
        if cropped:
            hard.append('CROP')
        _geom = ['RAILING'] if railing else []
        if best_key == 'kl' and n_valid_kl >= 3:
            soft += _geom
        else:
            hard += _geom
        flags = hard + soft
        if not flags:
            verdict = 'PASS'
        elif hard:
            verdict = ' + '.join(hard) + (
                f"  ·  note: {' + '.join(soft)}" if soft else '')
        else:
            verdict = 'PASS  ·  note: ' + ' + '.join(soft)
        vcol = '#2ca02c' if not hard else '#d62728'

        # ── Q estimators (nL/s) ─────────────────────────────────────────
        Q_kl = float(res.get('Q_kloosterman', np.nan)) * q_conv
        Q_plug = float(res.get('Q_plug_mean', np.nan)) * q_conv
        Q_pois_mask = float(res.get('Q_pois_mean', np.nan)) * q_conv
        Q_pois_free = ((abs(v0_free) * np.pi * R_free_px ** 2 / 2.0) * q_conv
                       if (np.isfinite(v0_free) and np.isfinite(R_free_px) and R_free_px > 0)
                       else np.nan)
        Q_canon = {'kl': Q_kl, 'pois_free': Q_pois_free,
                   'pois_seg': Q_pois_mask}.get(best_key, Q_kl)
        canon_name = {'kl': 'Kloosterman', 'pois_free': 'Pois free-R',
                      'pois_seg': 'Pois mask-R'}.get(best_key, 'Kloosterman')

        fig = plt.figure(figsize=(15, 19), dpi=100)
        fig.suptitle(
            f"OF pipeline — edge ({u}, {v}) · tile {tile_id} · reported Q = "
            f"{abs(Q_canon):.3f} nL/s ({canon_name}) · "
            f"R_seg = {R_prior_um:.1f} µm · {verdict}",
            fontsize=13, color=vcol, fontweight='bold', y=0.992)

        def _draw_speed(ax):
            if speed is not None:
                ax.imshow(np.ma.masked_where(speed <= 0, speed), cmap='hot',
                          vmin=0, vmax=_spd_hi, origin='upper')
            ax.set_aspect('equal')

        def _orient(ax, h, w):
            """Rotate a spatial panel from the raw tile-camera frame into the
            mosaic frame the main viewer shows (the stitcher's per-tile affine
            is a rotation) so the crops match the viewer's orientation.  Applies
            one affine to every artist (image, lines, quiver collections,
            cross-section arrow patches) and re-limits from the rotated corners
            (y inverted = row-down)."""
            try:
                import matplotlib.transforms as _mtx
                _L = self._of_geom.tiles[int(tile_id)].A_full[:2, :2]
                _aff = _mtx.Affine2D.from_values(
                    _L[1, 1], _L[0, 1], _L[1, 0], _L[0, 0], 0.0, 0.0)
                for _a in (list(ax.images) + list(ax.lines)
                           + list(ax.collections) + list(ax.patches)):
                    _a.set_transform(_aff + ax.transData)
                    # Quiver (and other offset collections) position their
                    # glyphs via a SEPARATE offset transform — rotate that too,
                    # or the arrows detach from the rotated image.
                    if hasattr(_a, 'set_offset_transform'):
                        try:
                            _a.set_offset_transform(_aff + ax.transData)
                        except Exception:
                            pass
                _cor = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
                _xt = _L[1, 1] * _cor[:, 0] + _L[1, 0] * _cor[:, 1]
                _yt = _L[0, 1] * _cor[:, 0] + _L[0, 0] * _cor[:, 1]
                _pad = 0.05 * max(np.ptp(_xt), np.ptp(_yt), 1.0)
                ax.set_xlim(_xt.min() - _pad, _xt.max() + _pad)
                ax.set_ylim(_yt.max() + _pad, _yt.min() - _pad)
                ax.set_aspect('equal')
                return True
            except Exception:
                return False

        # One tall SCROLLABLE figure, 3×3, consolidated to the non-redundant
        # panels.  Dropped (redundant / broken): the mean-OF-field quiver panel
        # (broken auto-scaled vectors + duplicate of the speed heatmap), the
        # raw→smoothed pointwise scatter (re-draws the binned profile), the
        # mask+tangent panel, the CL-offset plot, per-slice R, and the second
        # (video-crop) radii panel — one vessel-image panel is enough.
        gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.26,
                              height_ratios=[1, 1, 1, 1.3],
                              top=0.962, bottom=0.03, left=0.06, right=0.975)

        # analysed (kept) vs full (untrimmed) centerline arc lengths (µm).
        def _arclen(a):
            a = _arr(a)
            return (float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))
                    if (a is not None and len(a) >= 2) else np.nan)
        cl_kept = _arr(data.get('centerline_rc'))
        cl_all = _arr(data.get('centerline_full_rc'))
        _kept_um = _arclen(cl_kept) * px_um
        _full_um = _arclen(cl_all) * px_um
        _pct = (100.0 * _kept_um / _full_um
                if (np.isfinite(_kept_um) and np.isfinite(_full_um)
                    and _full_um > 0) else np.nan)

        # (0,0) ROI LOCATOR — zoomed to the vessel, trim shown
        ax = fig.add_subplot(gs[0, 0])
        frames = data.get('frames'); Hf = Wf = None
        if frames is not None:
            fr = np.asarray(frames)
            if fr.ndim == 3 and fr.shape[0]:
                mf = fr.mean(axis=0); Hf, Wf = mf.shape[:2]
                loi, hii = np.percentile(mf, (1, 99))
                ax.imshow(mf, cmap='gray', vmin=loi, vmax=hii, origin='upper')
                ax.add_patch(Rectangle((c0, r0), W, H, fill=False,
                                       edgecolor='lime', lw=2.0))
                if cl_all is not None and len(cl_all) >= 2:
                    ax.plot(cl_all[:, 1], cl_all[:, 0], color='white', lw=2.4,
                            ls=(0, (4, 2)), alpha=0.9, label='full vessel CL')
                if cl_kept is not None and len(cl_kept) >= 2:
                    ax.plot(cl_kept[:, 1], cl_kept[:, 0], color='cyan', lw=1.6,
                            label='analysed CL')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6.5, loc='upper right', facecolor='black',
                              labelcolor='white', framealpha=0.5)
        if Hf:
            _orient(ax, Hf, Wf)
            try:
                _Lz = self._of_geom.tiles[int(tile_id)].A_full[:2, :2]
                _bc = np.array([[c0, r0], [c0 + W, r0],
                                [c0, r0 + H], [c0 + W, r0 + H]], float)
                _bx = _Lz[1, 1] * _bc[:, 0] + _Lz[1, 0] * _bc[:, 1]
                _by = _Lz[0, 1] * _bc[:, 0] + _Lz[0, 0] * _bc[:, 1]
                _pz = 0.35 * max(np.ptp(_bx), np.ptp(_by), 1.0)
                ax.set_xlim(_bx.min() - _pz, _bx.max() + _pz)
                ax.set_ylim(_by.max() + _pz, _by.min() - _pz)
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, 'no frames', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        _crop_str = (f"CL kept {_kept_um:.0f}/{_full_um:.0f} µm ({_pct:.0f}%)"
                     if np.isfinite(_pct) else f"CL {_kept_um:.0f} µm")
        ax.set_title(f"ROI locator — green box = CROP (mosaic orient.)\n"
                     f"{_crop_str} · ROI {W}×{H} px", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # (0,1:) radial profile + 3 fits + gate — widened across the two right
        # cells (the old (0,1) "vessel flow + cyan fan" panel was redundant with
        # the 3D per-slice-parabola panel below, which shows the fitted profile on
        # the ACTUAL curved vessel; dropped it and gave the space to the profile).
        ax = fig.add_subplot(gs[0, 1:])
        _neigh = np.asarray(diag.get('neighbour_mask', []), dtype=bool)
        _cent = np.asarray(diag.get('central_mask', []), dtype=bool)
        if _neigh.shape == bc_um.shape and _neigh.any():
            # Split measured points: central vessel (kept for free-R fit) vs
            # bleed-in neighbour (rejected past the first velocity valley).
            _kept = fin & ~_neigh
            _rej = fin & _neigh
            ax.plot(bc_um[_kept], pv[_kept], 'o', ms=4, color='#1f77b4',
                    label='measured (central)')
            ax.plot(bc_um[_rej], pv[_rej], 'x', ms=5, color='#999999', mew=1.4,
                    label='rejected (neighbour bleed)')
            if _cent.shape == bc_um.shape and _cent.any():
                _ci = np.where(_cent)[0]
                _lo_um = float(bc_um[_ci.min()]); _hi_um = float(bc_um[_ci.max()])
                _xmin = float(bc_um.min()) - 1.0; _xmax = float(bc_um.max()) + 1.0
                ax.axvspan(_xmin, _lo_um, color='gray', alpha=0.12)
                ax.axvspan(_hi_um, _xmax, color='gray', alpha=0.12,
                           label='rejected region')
        else:
            ax.plot(bc_um[fin], pv[fin], 'o', ms=4, color='#1f77b4',
                    label='measured')
        rr = np.linspace(-R_prior_um, R_prior_um, 100)
        if np.isfinite(v0_fix):
            ax.plot(rr, np.clip(abs(v0_fix) * v_conv * (1 - (rr / R_prior_um) ** 2), 0, None),
                    '-', color='#2ca02c', lw=1.8, label=f'mask-R fit ({R_prior_um:.0f} µm)')
        if np.isfinite(R_free_px) and R_free_px > 0 and np.isfinite(v0_free):
            R_free_um = R_free_px * px_um
            rr2 = np.linspace(-R_free_um, R_free_um, 100)
            ax.plot(rr2, np.clip(abs(v0_free) * v_conv * (1 - (rr2 / R_free_um) ** 2), 0, None),
                    '--', color='#d62728', lw=1.8, label=f'free-R fit ({R_free_um:.0f} µm)')
        lo = float(diag.get('kl_r_slice_lo', 0.5)); hi = float(diag.get('kl_r_slice_hi', 2.0))
        ax.axvspan(lo * R_prior_um, hi * R_prior_um, color='orange', alpha=0.06)
        ax.axvspan(-hi * R_prior_um, -lo * R_prior_um, color='orange', alpha=0.06,
                   label=f'fitted-R gate [{lo:.1f},{hi:.1f}]·R')
        for xv in (-R_prior_um, R_prior_um):
            ax.axvline(xv, color='magenta', ls=':', lw=1.0)
        _pk = [np.nanpercentile(pv[fin], 97)] if fin.any() else []
        if np.isfinite(v0_fix):
            _pk.append(abs(v0_fix) * v_conv)
        if np.isfinite(v0_free):
            _pk.append(abs(v0_free) * v_conv)
        if _pk:
            ax.set_ylim(0, 1.3 * max(_pk))
        ax.set_xlabel('signed ρ (µm)'); ax.set_ylabel('|v| (µm/s)')
        ax.set_title(f"radial profile + fits — taper {taper:.2f}, "
                     f"R_free/R_seg {R_free_ratio:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

        # ── Kloosterman per-slice prep ──
        kl = diag.get('kl') or {}
        psQ = np.asarray(kl.get('per_slice_Q', []), float)
        psR = np.asarray(kl.get('per_slice_R', []), float)
        vmask = np.asarray(kl.get('valid_mask', []), bool)
        reasons = list(kl.get('per_slice_reason', []))
        Nkl = len(psQ)
        good = vmask if vmask.size == Nkl else np.isfinite(psQ)
        xi = np.arange(Nkl)
        psQ_nl = psQ * q_conv; psR_um = psR * px_um
        Q_char = float(res.get('Q_kloosterman', np.nan)) * q_conv
        V_char = float(kl.get('V_CL_char', np.nan)) * v_conv
        R_char_um = float(kl.get('R_char', np.nan)) * px_um
        Qmed = float(np.nanmedian(psQ_nl[np.isfinite(psQ_nl)])) if np.isfinite(psQ_nl).any() else np.nan

        # (1,0) CL refinement — graph CL vs OF ridge-tracked CL
        ax = fig.add_subplot(gs[1, 0]); _draw_speed(ax)
        if cl_init is not None:
            ax.plot(cl_init[:, 1], cl_init[:, 0], color='cyan', lw=1.4, ls=':',
                    label='graph CL (initial)')
        if cl_ref is not None:
            ax.plot(cl_ref[:, 1], cl_ref[:, 0], color='lime', lw=1.8,
                    label='ridge-tracked CL')
        ax.legend(fontsize=7, loc='upper right', facecolor='black',
                  labelcolor='white', framealpha=0.5)
        _sh = res.get('cl_shift')
        ax.set_title("centerline refinement — OF ridge-tracking (mosaic orient.)\n"
                     + (f"mean shift {_sh:+.1f} px onto the flow ridge"
                        if _sh is not None else ""), fontsize=9)
        _orient(ax, H, W); ax.set_xticks([]); ax.set_yticks([])

        # (1,1) per-slice Q + median-band gate  (the spatial per-slice accept/
        # reject + fitted half-widths now live in the 3D panel below — this is
        # the quantitative gate view, no longer duplicated as a 2D half-widths
        # panel).
        ax = fig.add_subplot(gs[1, 1])
        # Q is a flow-rate magnitude → orient the whole panel positive (same sign
        # convention as the Q(t) panel: flip the trace by the median sign so the
        # points, median line, band and Q_char all read positive together).
        _sgn = -1.0 if (np.isfinite(Qmed) and Qmed < 0) else 1.0
        psQ_p = psQ_nl * _sgn
        Qmed_p = Qmed * _sgn
        Q_char_p = Q_char * _sgn
        if Nkl:
            ax.plot(xi[good], psQ_p[good], 'o', ms=5, color='#2ca02c', label='accepted')
            bad = (~good) & np.isfinite(psQ_p)
            ax.plot(xi[bad], psQ_p[bad], 'x', ms=6, color='#999', label='rejected')
            for bi in list(xi[bad])[:5]:
                if bi < len(reasons):
                    ax.annotate(str(reasons[bi]).split('(')[0], (bi, psQ_p[bi]),
                                fontsize=6, color='#666', ha='center')
        if np.isfinite(Qmed_p):
            ax.axhline(Qmed_p, color='k', lw=1.0, label=f'median {Qmed_p:.3f}')
            ax.axhspan(0.5 * Qmed_p, 1.5 * Qmed_p, color='green', alpha=0.07,
                       label='band [0.5,1.5]·med')
        if np.isfinite(Q_char_p):
            ax.axhline(Q_char_p, color='#1f77b4', ls='--', lw=1.6, label=f'Q_char {Q_char_p:.3f}')
        ax.set_xlabel('slice'); ax.set_ylabel('Q_slice (nL/s)')
        ax.set_title("per-slice Q + median-band gate", fontsize=9)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

        # (1,2) Q estimator bar
        ax = fig.add_subplot(gs[1, 2])
        bars = [('plug', Q_plug, '#7f7f7f'), ('Pois mask-R', Q_pois_mask, '#2ca02c'),
                ('Pois free-R', Q_pois_free, '#d62728'), ('Kloosterman', Q_kl, '#1f77b4')]
        names = [b[0] for b in bars]
        vals = [abs(b[1]) if np.isfinite(b[1]) else 0.0 for b in bars]
        cols = [b[2] for b in bars]
        edgs = ['black' if n == canon_name else 'none' for n in names]
        lws = [2.5 if n == canon_name else 0 for n in names]
        ax.bar(range(len(names)), vals, color=cols, edgecolor=edgs, linewidth=lws)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('|Q| (nL/s)')
        ax.set_title(f"Q estimators — canonical = {canon_name}", fontsize=9)
        ax.grid(alpha=0.25, axis='y')

        # (2,0) FULL Q(t) time series + harmonic fit  (not the cycle-average).
        # CONSISTENT with the reported canonical Q: the per-frame Q_pois_t is the
        # mask-R Poiseuille SHAPE (fit v0(t) with R fixed to R_seg); its absolute
        # scale (R_seg) generally differs from the canonical estimator's radius
        # (Kloosterman uses per-slice fitted R_char).  Those are INDEPENDENT paths
        # — Q_pois_t never touches Kloosterman — so raw they disagree by ~(R/ R_seg)².
        # We therefore rescale the waveform by pois_seg_factor (= Q_canon / mean
        # Q_pois) so its mean lands on the reported line: the temporal modulation is
        # the real per-frame OF, the magnitude is the canonical value.  (The raw
        # per-estimator magnitudes are compared side-by-side in the estimator bar.)
        ax = fig.add_subplot(gs[2, 0])
        _psf = float(data.get('pois_seg_factor', 1.0) or 1.0)
        Qraw = np.asarray(res.get('Q_pois_t', []), float) * q_conv * _psf
        if Qraw.size and np.isfinite(np.nanmean(Qraw)) and np.nanmean(Qraw) < 0:
            Qraw = -Qraw                    # orient positive (Q is a magnitude)
        t_s = np.arange(Qraw.size) * frame_dt_s
        mean_q = float(np.nanmean(Qraw)) if Qraw.size else np.nan
        a0 = amp1 = np.nan
        fitsig = np.array([])
        if Qraw.size > 8 and np.isfinite(f0_hz) and f0_hz > 0:
            try:
                from ..analyze.harmonic import fit_harmonics
                hr = fit_harmonics(Qraw, frame_dt=frame_dt_s, f0=f0_hz,
                                   K=3, include_dc=True)
                a0 = float(hr.get('a0', np.nan))
                amp1 = float(hr.get('amp1', np.nan))
                fitsig = np.asarray(hr.get('signal', []), float)
            except Exception:
                pass
            ax.plot(t_s, Qraw, color='#c2c2c2', lw=0.7,
                    label='Q(t) (canonical scale)')
            if fitsig.size == Qraw.size:
                ax.plot(t_s, fitsig, color='#1f77b4', lw=1.6,
                        label='harmonic fit (DC+3H)')
            if np.isfinite(Q_canon):
                # mean now equals the reported value by construction
                ax.axhline(abs(Q_canon), color='#d62728', ls='-.', lw=1.3,
                           label=f'reported = mean {abs(Q_canon):.3f} ({canon_name})')
            ax.legend(fontsize=6.5, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'no Q(t)', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_xlabel('time (s)'); ax.set_ylabel('Q (nL/s)')
        ax.set_title(f"Q(t) + harmonic fit (canonical scale) — mean {mean_q:.3f}, "
                     f"H1 {amp1:.3f} nL/s · f₀={f0_hz:.2f} Hz", fontsize=8.5)
        ax.grid(alpha=0.3)

        # (2,1:) verdict + compact Kloosterman derivation (spans two cells)
        ax = fig.add_subplot(gs[2, 1:]); ax.axis('off')

        def _f(x, fmt='{:.2f}'):
            return fmt.format(x) if (x is not None and np.isfinite(x)) else '—'
        cats = {}
        for i, rs in enumerate(reasons):
            if i < len(good) and good[i]:
                cats['accepted'] = cats.get('accepted', 0) + 1
            else:
                key = str(rs).split('(')[0]
                cats[key] = cats.get(key, 0) + 1
        cat_str = ", ".join(f"{k} {n}" for k, n in
                            sorted(cats.items(), key=lambda kv: -kv[1])[:4])
        # When Kloosterman was NOT used, say so loudly + why it failed.
        _kl_failed = (canon_name != 'Kloosterman')
        _kl_reason = str(kl.get('reason', '?'))
        _fail_line = (f"  ⚠ KLOOSTERMAN FAILED ({int(good.sum())}/{Nkl} slices, "
                      f"{_kl_reason}) → fell back to {canon_name}\n"
                      f"    why: {cat_str}\n"
                      if _kl_failed else "")
        txt = (
            f"VERDICT:  {verdict}\n"
            + _fail_line +
            f"  coverage {cov*100:.0f}%   taper {_f(taper)}   "
            f"R_free/R_seg {_f(R_free_ratio)}\n"
            f"  Kloosterman: {int(good.sum())}/{Nkl} slices kept\n"
            f"  reported Q = {abs(Q_canon):.4f} nL/s ({canon_name})\n"
            f"  R_seg {R_prior_um:.1f} µm    R_inferred {_f(R_free_px*px_um,'{:.1f}')} µm\n"
            + ("  → mask-R and fitted-R agree — both reliable\n"
               if not (set(flags) & {'CROP', 'RAILING'})
               else "  → trust mask-R; fitted-R unreliable here\n")
            + "\nKloosterman derivation\n"
            "──────────────────────\n"
            f"  incl: |v| > {diag.get('kl_v_incl_frac', 0.25):.2f}·v_peak"
            f" (contiguous extent → wall)\n"
            f"  R gate [{lo:.1f},{hi:.1f}]·R_seg (loose; median-Q rejects)\n"
            f"  outlier: |Q| in [0.5,1.5]·median, sign-matched\n"
            f"  outcomes: {cat_str}\n"
            f"  Q_char {_f(Q_char,'{:.4f}')}   V_CL {_f(V_char,'{:.0f}')}"
            f"   R_char {_f(R_char_um,'{:.1f}')} µm"
        )
        ax.text(0.0, 1.0, txt, transform=ax.transAxes, ha='left', va='top',
                family='monospace', fontsize=8,
                bbox=dict(boxstyle='round',
                          facecolor=('#e8f5e9' if not hard else '#fdecea'),
                          edgecolor=vcol, lw=1.5))

        # (3,:) 3D per-slice parabolic fits on the ACTUAL curved vessel
        # (Kloosterman-thesis Fig 4.11): each cross-section's fitted parabola is
        # drawn at its real spatial position along the perpendicular, so the
        # vessel's real curvature shows in the x–y (col,row) plane.  The vessel
        # speed image sits on the z=0 floor, the cyan centerline curves through
        # it, and the white ±R_fit walls trace the full vessel extent implied by
        # the parabola radii.  Accepted = coloured, rejected = gray.
        ax = fig.add_subplot(gs[3, :], projection='3d')
        psV = np.asarray(kl.get('per_slice_v_CL', []), float)
        profs = np.asarray(kl.get('per_slice_profiles', []), float)
        rho_g = np.asarray(kl.get('rho', []), float)
        drew3d = False
        if (cl_ref is not None and centerline_geometry is not None
                and profs.ndim == 2 and profs.shape[0] == Nkl and rho_g.size
                and len(cl_ref) >= 2 and Nkl >= 2 and vfield is not None):
            import matplotlib.cm as _cm
            clr = np.asarray(cl_ref, float)           # (N, 2) = (row, col)
            _s3, _tg3, nm3 = centerline_geometry(clr)  # normals in (row, col)
            # z-scale from the measured cross-section velocities
            _pf = np.abs(profs[np.isfinite(profs)]) * v_conv
            _zmax = float(np.nanpercentile(_pf, 99)) if _pf.size else 1.0
            # floor: grayscale ROI intensity image — extends BEYOND the vessel
            # into the tissue, so the inferred radius (parabolas + ±R_fit walls)
            # can be compared against the vessel's actual edges in the image.
            try:
                import matplotlib.pyplot as _plt
                _fm = diag.get('frame_mean')
                _fm = (np.asarray(_fm, float) if _fm is not None
                       else np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2))
                _st = max(1, min(_fm.shape[:2]) // 80)   # ≤~80px for a fast surface
                _fs = _fm[::_st, ::_st]
                _lo, _hi = np.percentile(_fs, (1, 99))
                _g = np.clip((_fs - _lo) / max(_hi - _lo, 1e-6), 0, 1)
                _Yg, _Xg = np.mgrid[0:_fs.shape[0], 0:_fs.shape[1]]
                ax.plot_surface(_Xg * _st * px_um, _Yg * _st * px_um,
                                np.zeros_like(_g), facecolors=_plt.cm.gray(_g),
                                rstride=1, cstride=1, shade=False, linewidth=0,
                                antialiased=False, alpha=0.4, zorder=0)
            except Exception:
                pass
            n_acc = max(int(good.sum()), 1); _ai = 0
            wp, wm = [], []
            for i in range(Nkl):
                acc = bool(i < len(good) and good[i])
                col = (_cm.turbo(_ai / n_acc) if acc else '#b0b0b0')
                if acc:
                    _ai += 1
                pi = clr[min(i, len(clr) - 1)]; ni = nm3[min(i, len(nm3) - 1)]
                # measured cross-section points at their real (col,row)
                pr = profs[i]; fm = np.isfinite(pr)
                if fm.any():
                    pts = pi[None, :] + rho_g[fm][:, None] * ni[None, :]
                    ax.scatter(pts[:, 1] * px_um, pts[:, 0] * px_um,
                               np.abs(pr[fm]) * v_conv, color=col,
                               s=(9 if acc else 5), alpha=(0.95 if acc else 0.3),
                               edgecolors='none', zorder=(5 if acc else 1))
                # fitted parabola along the real perpendicular
                if (acc and i < len(psV) and i < len(psR)
                        and np.isfinite(psV[i]) and np.isfinite(psR[i]) and psR[i] > 0):
                    rr = np.linspace(-psR[i], psR[i], 40)
                    pos = pi[None, :] + rr[:, None] * ni[None, :]
                    zf = np.clip(abs(psV[i]) * (1 - (rr / psR[i]) ** 2), 0, None) * v_conv
                    ax.plot(pos[:, 1] * px_um, pos[:, 0] * px_um, zf,
                            color=col, lw=1.6, alpha=0.95, zorder=5)
                    wp.append(pi + psR[i] * ni); wm.append(pi - psR[i] * ni)
            # centerline + ±R_fit extent walls, lifted just above the floor
            # heatmap so they aren't buried under it.
            _zoff = 0.04 * max(_zmax, 1.0)
            ax.plot(clr[:, 1] * px_um, clr[:, 0] * px_um, np.full(len(clr), _zoff),
                    color='lime', lw=2.6, alpha=1.0, zorder=6, label='centerline')
            if len(wp) >= 2:
                wp = np.asarray(wp); wm = np.asarray(wm)
                ax.plot(wp[:, 1] * px_um, wp[:, 0] * px_um, np.full(len(wp), _zoff),
                        color='magenta', lw=2.2, alpha=1.0, zorder=6)
                ax.plot(wm[:, 1] * px_um, wm[:, 0] * px_um, np.full(len(wm), _zoff),
                        color='magenta', lw=2.2, alpha=1.0, zorder=6,
                        label='vessel extent (±R_fit)')
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlabel('col (µm)'); ax.set_ylabel('row (µm)')
            ax.set_zlabel('|v| (µm/s)')
            ax.set_zlim(0, 1.05 * max(_zmax, 1.0))
            ax.view_init(elev=26, azim=-72)
            drew3d = True
        if not drew3d:
            ax.text2D(0.5, 0.5, 'no per-slice profile data',
                      transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title(f"per-slice parabolic fits on the real curved vessel — "
                     f"coloured = accepted ({int(good.sum())}), gray = rejected · "
                     f"magenta = ±R_fit extent  (reject if R² < "
                     f"{diag.get('kl_min_r2', 0.5):.2f})", fontsize=9)

        # ── save PNG + show in a scrollable window (with the mpl toolbar) ──
        try:
            from pathlib import Path as _Path
            _out = _Path.home() / 'Desktop' / 'of_diagnostics'
            _out.mkdir(parents=True, exist_ok=True)
            fig.savefig(_out / f'OF_pipeline_tile{tile_id}_edge{u}-{v}.png',
                        dpi=130, facecolor='white')
        except Exception as _e:
            print(f"  (pipeline PNG save skipped: {_e})")
        self._show_scrollable_figure(fig, position='top-right')


    def _show_scrollable_figure(self, fig, position=None):
        """Show one tall Figure in a scrollable Qt dialog with the navigation
        toolbar (zoom / pan / home / save).  The canvas keeps its full rendered
        size and a QScrollArea provides vertical scrolling, so the stacked
        pipeline sections read top→bottom like a document.  Falls back to
        show_figure if the Qt path is unavailable."""
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
            from qtpy.QtWidgets import (QDialog, QVBoxLayout, QScrollArea,
                                        QApplication, QSizePolicy)
            from qtpy.QtCore import Qt, QSize
            canvas = FigureCanvasQTAgg(fig)
            w = int(fig.get_figwidth() * fig.dpi)
            h = int(fig.get_figheight() * fig.dpi)
            # setFixedSize (not setMinimumSize) — matplotlib's canvas
            # overrides sizeHint() after draw() and would shrink the
            # canvas back to the dialog viewport, which collapses the
            # scroll area and makes the bars disappear a moment after
            # the dialog opens.  Fixed size prevents that race.
            canvas.setFixedSize(w, h)
            dialog = QDialog()
            dialog.setWindowTitle(
                fig._suptitle.get_text().split(chr(10))[0]
                if fig._suptitle else 'OF pipeline')
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(dialog); lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(NavigationToolbar2QT(canvas, dialog))
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            # Force scrollbars always visible so it's obviously scrollable.
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            # Kill the scroll-area's own sizeHint so it doesn't demand the
            # canvas's full size and drag the dialog to fig_h — which is
            # what macOS was doing (dialog grew to figure, no scroll needed
            # → bars flashed then vanished).  Small size hint = the outer
            # dialog.resize() actually wins.
            scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            scroll.setMinimumSize(QSize(400, 300))
            scroll.setWidget(canvas); lay.addWidget(scroll)
            screen = QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else None
            if avail is not None:
                vw = min(w + 26, int(avail.width() * 0.62))
                vh = min(h + 60, int(avail.height() * 0.85))
            else:
                vw, vh = min(w + 26, 1200), 900
            # setFixedSize (not resize) — locks the dialog against Qt's
            # sizeHint-based auto-grow so it can't expand to the canvas.
            dialog.setFixedSize(vw, vh)
            canvas.draw(); dialog.show()
            # After show, restore resizability so the user can drag corners
            # if they want a bigger view.
            dialog.setMinimumSize(QSize(400, 300))
            dialog.setMaximumSize(QSize(16777215, 16777215))
            if position == 'top-right' and avail is not None:
                try:
                    QApplication.processEvents()
                    fw = dialog.frameGeometry().width()
                    dialog.move(int(avail.x() + avail.width() - fw - 12),
                                int(avail.y() + 12))
                except Exception:
                    pass
            _d = [d for d in getattr(self, '_pipeline_dialogs', []) if d.isVisible()]
            _d.append(dialog); self._pipeline_dialogs = _d
        except Exception as _e:
            from .mosaic import show_figure
            print(f"  (scroll dialog failed, using plain canvas: {_e})")
            show_figure(fig, position=position)

    def _show_fig_with_toolbar(self, fig, position=None):
        """Show a matplotlib Figure in a Qt dialog that includes the standard
        navigation toolbar (zoom / pan / home / save) — which the bare
        ``show_figure`` embed lacks.  Keeps the dialog alive on the viewer and
        falls back to ``show_figure`` if the Qt toolbar path is unavailable."""
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
            from qtpy.QtWidgets import QDialog, QVBoxLayout, QApplication
            from qtpy.QtCore import Qt
            canvas = FigureCanvasQTAgg(fig)
            dialog = QDialog()
            _t = (fig._suptitle.get_text().split(chr(10))[0]
                  if fig._suptitle else 'Figure')
            dialog.setWindowTitle(_t)
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(dialog); lay.setContentsMargins(0, 0, 0, 0)
            toolbar = NavigationToolbar2QT(canvas, dialog)
            lay.addWidget(toolbar); lay.addWidget(canvas)
            dialog.resize(int(fig.get_figwidth() * fig.dpi),
                          int(fig.get_figheight() * fig.dpi) + 40)
            canvas.draw(); dialog.show()
            if position == 'top-right':
                try:
                    QApplication.processEvents()
                    screen = QApplication.primaryScreen()
                    if screen is not None:
                        avail = screen.availableGeometry()
                        fw = dialog.frameGeometry().width()
                        n = len(getattr(self, '_pipeline_dialogs', []))
                        dialog.move(int(avail.x() + avail.width() - fw - 12 - 26 * n),
                                    int(avail.y() + 12 + 26 * n))
                except Exception:
                    pass
            _d = [d for d in getattr(self, '_pipeline_dialogs', []) if d.isVisible()]
            _d.append(dialog)
            self._pipeline_dialogs = _d
        except Exception as _e:
            from .mosaic import show_figure
            print(f"  (toolbar dialog failed, using plain canvas: {_e})")
            show_figure(fig, position=position)

    def _show_of_pipeline_figures(self):
        """Re-open the full OF pipeline walkthrough for the last fresh OF run."""
        data = getattr(self, '_last_of_pipeline', None)
        if not data:
            self._info_box.setPlainText(
                "Run 'Compute fresh PIV/OF' on an edge first.")
            return
        try:
            self._render_of_pipeline(data)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"OF pipeline figure failed: {type(e).__name__}: {e}")

    def _on_run_piv_comparison_clicked(self):
        """Run OpenPIV ensemble on the same edge as the last fresh OF run
        and pop the OF-vs-PIV comparison figure.  Uses the OF result to
        auto-tune dt (target 10 px peak displacement, Poelma optimum) and
        to supply the shared tangent field so integration is apples-to-apples.
        """
        data = getattr(self, '_last_of_pipeline', None)
        if not data or 'frames' not in data:
            self._info_box.setPlainText(
                "Run 'Compute fresh PIV/OF' on an edge first — the PIV "
                "comparison reuses that run's frames and geometry.")
            return
        of_res = data['res']
        try:
            from ..analysis.piv_openpiv import compute_piv_openpiv_Q
            from ..analysis.piv_of_compare_figures import plot_piv_vs_of
            # Peak axial velocity in px/frame (from OF mean field) for the
            # dt auto-tune.  Fall back to abs(v0_fixed) if vfield missing.
            vf = of_res.get('vfield_mean')
            if vf is not None:
                peak_v = float(np.nanpercentile(
                    np.hypot(vf[..., 0], vf[..., 1]), 95))
            else:
                peak_v = abs(float(of_res.get('v0_fixed', 0.0) or 0.0))
            # Tangent + mask from OF so only the estimator differs.
            # CRITICAL: OF stores rr_in/cc_in/cl_used in ROI-relative
            # coordinates (offset by roi_offset).  My PIV samples the full
            # frame, so shift back to full-frame coords before passing.
            r0, c0 = of_res.get('roi_offset', (0, 0))
            tan = dict(
                cl_used=of_res['cl_used'] + np.array([r0, c0]),
                rr_in=of_res['rr_in'] + int(r0),
                cc_in=of_res['cc_in'] + int(c0),
                rho_in=of_res['rho_in'],
                t_row_in=of_res['t_row_in'], t_col_in=of_res['t_col_in'])
            self._info_box.setPlainText(
                "Running OpenPIV ensemble … (~10-20 s)")
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            piv_res = compute_piv_openpiv_Q(
                data['frames'], data['centerline_rc'], data['r_tile'],
                tangent_from_of=tan,
                peak_v_hint_pxpf=peak_v,
                dt=None,
                windowsizes=(32, 16), overlap=(16, 8), num_iterations=2,
                n_phase_bins=40, f0_hz=data.get('f0_hz') or 2.5,
                frame_dt_s=data['frame_dt_s'],
                max_pairs_per_bin=40,
                n_bins=30, fit_fraction=0.75,
            )
            self._last_piv_compare = dict(of=of_res, piv=piv_res, tan=tan,
                                           peak_v_hint=peak_v)
            if hasattr(self, '_piv_sweep_btn'):
                self._piv_sweep_btn.setEnabled(True)
            plot_piv_vs_of(
                of_res, piv_res,
                frame_dt_s=data['frame_dt_s'],
                v_conv=data['v_conv'], q_conv=data['q_conv'],
                f0_hz=data.get('f0_hz') or 2.5,
                title=data.get('title', ''),
                stored_Qt=data.get('stored_Qt'))
            of_qp = of_res.get('Q_pois_mean', np.nan) * data['q_conv']
            piv_qp = piv_res.get('Q_pois_mean', np.nan) * data['q_conv']
            of_qP = of_res.get('Q_plug_mean', np.nan) * data['q_conv']
            piv_qP = piv_res.get('Q_plug_mean', np.nan) * data['q_conv']
            self._info_box.setPlainText(
                f"PIV vs OF (nL/s):\n"
                f"  Q_pois : OF={of_qp:+.3f}  PIV={piv_qp:+.3f}\n"
                f"  Q_plug : OF={of_qP:+.3f}  PIV={piv_qP:+.3f}   (bulk-flow "
                f"check, resolution-independent)\n"
                f"  dt={piv_res.get('dt_used')} frames  peak_disp="
                f"{piv_res.get('peak_disp_seen', 0):.1f} px  "
                f"n_win={piv_res.get('n_windows', 0)}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"PIV comparison failed: {type(e).__name__}: {e}")

    def _on_run_piv_sweep_clicked(self):
        """Run OpenPIV at four different finest-window sizes and plot the
        recovered Q vs winsize.  Answers whether smaller windows help on
        THIS vessel empirically.  Reuses the last PIV-comparison inputs."""
        data = getattr(self, '_last_of_pipeline', None)
        cmp = getattr(self, '_last_piv_compare', None)
        if not data or not cmp or 'frames' not in data:
            self._info_box.setPlainText(
                "Run 'Compute PIV vs OF' first — sweep reuses those inputs.")
            return
        try:
            from ..analysis.piv_openpiv import compute_piv_openpiv_Q
            import matplotlib.pyplot as plt
            from qtpy.QtWidgets import QApplication
            sweep_specs = [
                # (finest_ws, coarser_ws, overlap_tuple, n_iter, label)
                (8,  16, (8, 4),  2, '8'),
                (12, 24, (12, 6), 2, '12'),
                (16, 32, (16, 8), 2, '16'),
                (24, 48, (24, 12), 2, '24'),
            ]
            results = []
            for i, (fw, cw, ov, ni, lbl) in enumerate(sweep_specs):
                self._info_box.setPlainText(
                    f"PIV sweep {i+1}/4: winsize={lbl} …")
                QApplication.processEvents()
                r = compute_piv_openpiv_Q(
                    data['frames'], data['centerline_rc'], data['r_tile'],
                    tangent_from_of=cmp['tan'],
                    peak_v_hint_pxpf=cmp['peak_v_hint'],
                    dt=None,
                    windowsizes=(cw, fw), overlap=ov, num_iterations=ni,
                    n_phase_bins=40, f0_hz=data.get('f0_hz') or 2.5,
                    frame_dt_s=data['frame_dt_s'], max_pairs_per_bin=40,
                )
                results.append((lbl, r))
            # Plot: Q_pois and Q_plug vs winsize, plus OF reference.
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            of_res = cmp['of']; q_conv = data['q_conv']
            of_qp = of_res.get('Q_pois_mean', np.nan) * q_conv
            of_qP = of_res.get('Q_plug_mean', np.nan) * q_conv
            ws = [int(lbl) for lbl, _ in results]
            q_pois = [r.get('Q_pois_mean', np.nan) * q_conv for _, r in results]
            q_plug = [r.get('Q_plug_mean', np.nan) * q_conv for _, r in results]
            peak_disp = [r.get('peak_disp_seen', np.nan) for _, r in results]
            ax[0].plot(ws, q_pois, 'o-', color='C3', label='PIV Q_pois', lw=2, ms=8)
            ax[0].plot(ws, q_plug, 's--', color='C3', label='PIV Q_plug', alpha=0.7, ms=8)
            ax[0].axhline(of_qp, color='C0', ls='-', label=f'OF Q_pois ({of_qp:+.3f})')
            ax[0].axhline(of_qP, color='C0', ls='--', alpha=0.7, label=f'OF Q_plug ({of_qP:+.3f})')
            ax[0].set_xlabel('finest PIV window size (px)')
            ax[0].set_ylabel('⟨Q⟩ (nL/s)')
            ax[0].set_title('PIV Q vs winsize — small = better resolution, worse SNR')
            ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
            ax[1].plot(ws, peak_disp, 'o-', color='k', ms=8)
            ax[1].axhspan(10, 12, color='green', alpha=0.15, label='Poelma optimum')
            ax[1].set_xlabel('finest PIV window size (px)')
            ax[1].set_ylabel('peak displacement observed (px)')
            ax[1].set_title('Peak displacement per winsize (should stay in optimum band)')
            ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
            fig.suptitle(f'PIV winsize sweep — {data.get("title", "")}',
                         fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            plt.show()
            summary = ', '.join(
                f'ws{lbl}: Q_pois={r.get("Q_pois_mean", np.nan) * q_conv:+.2f}'
                for lbl, r in results)
            self._info_box.setPlainText(
                f"PIV sweep done. {summary}. OF Q_pois={of_qp:+.2f} nL/s.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"PIV sweep failed: {type(e).__name__}: {e}")

    def _degree2_chain_edges(self, u, v):
        """Maximal degree-2 chain of edges containing (u, v), ordered and
        connected.  A vessel is subdivided into segments by trace nodes; the
        interior nodes of a single vessel are degree-2 (pass-through, where KCL
        is trivially satisfied).  Merging the chain lets fresh OF run on the
        whole vessel instead of one short segment — more cross-sections, higher
        SNR, one consistent flow direction.  We walk outward from u and from v
        through degree-2 nodes, stopping at the first junction (deg>=3) or leaf.

        Returns an ordered edge list [(n0,n1),(n1,n2),...] with (u,v) inside.
        The tile-FOV trim in the handler clips whatever runs off-screen, so a
        chain that crosses a tile boundary is naturally limited to the visible
        run — no explicit tile logic needed here."""
        UG = self.G.to_undirected() if self.G.is_directed() else self.G

        def walk(prev, cur):
            # extend from `cur`, arriving from `prev`, through degree-2 nodes
            out = []
            while UG.degree(cur) == 2:
                nxt = next((n for n in UG.neighbors(cur) if n != prev), None)
                if nxt is None or nxt == prev:
                    break
                out.append((cur, nxt))
                if nxt in (u, v):            # closed loop guard
                    break
                prev, cur = cur, nxt
            return out

        left = walk(v, u)                    # [(u,x1),(x1,x2),...] away from v
        right = walk(u, v)                   # [(v,y1),(y1,y2),...] away from u
        left_ordered = [(b, a) for (a, b) in reversed(left)]   # ...->x1->u
        return left_ordered + [(u, v)] + right

    def _on_run_optical_flow_clicked(self):
        """Run the unified Farneback OF (`compute_of_profile_and_Q`) on
        the currently-selected edge and pop the editing viewer's full
        6-panel diagnostic — exactly the same figure as Analyze Mode in
        `mosaic_app`, with no graph writes and no napari arrow overlay.

        Strategy: build the `of_result` dict in the shape
        `_analyze_of_display` expects, alias `self.mosaic_graph` to
        `self.G` so the editing-viewer method can read AV refs without
        modification, then call the method as an unbound function.

        This deliberately replaces the earlier `compute_of_qt_local`
        path — that one only returned `Q_t` + `v_mean`, missing the
        velocity field, vessel mask, radial profile, and ROI bounds
        the figure needs."""
        from qtpy.QtWidgets import QApplication
        import time as _time
        if self._selected_edge is None:
            self._info_box.setPlainText(
                "Click an edge first, then press 'Run optical flow'.")
            return
        u, v = self._selected_edge
        # Focus-tile rule.  Priority order:
        #   1. View-tab tile filter (explicit user choice).
        #   2. Currently-loaded video tile (the video that's actually
        #      on screen — fresh OF should match what the user can see).
        #   3. Click-derived "best-quality" tile from the selected edge
        #      (fallback for edges that have stored measurements but no
        #      currently-loaded video).
        # Old code skipped #2 and jumped straight to #3, which made
        # fresh OF run on a different tile's video than the loaded one.
        tile_id = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else (self._video_tile_id
                           if self._video_tile_id is not None
                           else self._selected_tile_id))
        if tile_id is None:
            self._info_box.setPlainText(
                f"Edge ({u}, {v}) has no PIV measurement on file, so "
                f"there's no recorded tile to use.\n"
                f"Load a tile video first ('Browse tiles' or the "
                f"per-edge button), or set the Tile filter in the View "
                f"tab to choose a tile explicitly.")
            return
        if not self.tiles or self.of_video_dir is None:
            self._info_box.setPlainText(
                "Optical flow needs --tile-positions and --of-video-dir "
                "(or --video-dir as a fallback) "
                "on the CLI (or the corresponding entries in the "
                "config bundle).")
            return
        video_path = self._get_of_video_path_for(int(tile_id))
        if video_path is None:
            self._info_box.setPlainText(
                f"No video found for tile {tile_id} under "
                f"{self.of_video_dir}.")
            return

        # Lazy imports so launch stays light.  Rewired to plexusflow's
        # own coord helpers (graph_to_tile_batch + graph_radius_to_tile_px);
        # pertile's mosaic_to_tile_coords / get_chain_coords aren't shipped
        # in the standalone plexusflow port.
        try:
            from ..analyze.profile import compute_of_profile_and_Q
            from ..analyze.constants import PX_SIZE_UM, FRAME_DT_S
            from ..io.tiff import load_tiff_stack, cut_before_fade
            from ..analyze.mosaic_piv import (
                graph_to_tile_batch, graph_radius_to_tile_px,
                load_tile_geometry, tile_video_shape)
            from ..config import StageConfig
        except Exception as e:
            self._info_box.setPlainText(
                f"Optical-flow imports failed: {type(e).__name__}: {e}")
            return

        # Build a small chain-coord helper — concatenate edge paths for a
        # chain of (u, v) hops, dropping duplicate junctions.  Returns
        # (coords_xy, mean_radius, was_reversed_first) so downstream tile-
        # coord math sees a single ordered centerline.
        def _get_chain_coords_local(G, chain, margin_px=None):
            if not chain:
                return None, float('nan'), False
            all_pts = []
            radii = []
            for i, (uu, vv) in enumerate(chain):
                ed = G.edges[uu, vv]
                pts = ed.get('pts', ed.get('path'))
                if pts is None:
                    continue
                p = np.asarray(pts, dtype=float)
                # Order pts to run u → v (path may be stored either way).
                x0, y0 = float(G.nodes[uu]['x']), float(G.nodes[uu]['y'])
                x1, y1 = float(G.nodes[vv]['x']), float(G.nodes[vv]['y'])
                if (np.hypot(p[0, 0] - x1, p[0, 1] - y1)
                        < np.hypot(p[0, 0] - x0, p[0, 1] - y0)):
                    p = p[::-1]
                if all_pts and len(all_pts[-1]):
                    if np.allclose(all_pts[-1][-1], p[0]):
                        p = p[1:]
                all_pts.append(p)
                # Junction-edge radii excluded (matches pertile convention)
                if len(chain) > 2 and (i == 0 or i == len(chain) - 1):
                    continue
                r = ed.get('radius')
                if r is not None:
                    radii.append(float(r))
            if not all_pts:
                return None, float('nan'), False
            coords_xy = np.concatenate(all_pts, axis=0)
            mean_r = float(np.median(radii)) if radii else \
                     float(np.median([G.edges[uu, vv].get('radius', 5.0)
                                       for uu, vv in chain]))
            # Minimal junction trim.  The old fixed 8px discarded a lot of good
            # vessel length; instead trim just enough to step off the exact
            # junction node (where the CL direction is undefined / kinked) and
            # leave ALL residual end-effect slices — junction bulges, merges,
            # abrupt direction changes — to be rejected downstream by the
            # Kloosterman median-flow-rate band and per-slice validity gates.
            # Flat ~3px (NOT radius-scaled — thick vessels don't need a bigger
            # trim), but fractionally capped so a SHORT vessel (incl. short LARGE
            # ones) is never gutted: never remove more than 15% of the arc per
            # end, and the ≥5-point guard keeps very short vessels whole.
            # ``margin_px=None`` (default) requests this; a number overrides it.
            #
            # Resample to ~1px UNIFORM spacing BEFORE trimming.  The raw graph
            # vessel path can be wildly nonuniform — e.g. two long terminal
            # segments plus a cluster of mid points.  Trimming by dropping whole
            # end POINTS then removes far more than the intended margin: one real
            # vessel had end segments of 9.8 and 15.6px, so a 3px margin dropped
            # the two endpoints and cut 65% of the arc (62→21µm).  Uniform spacing
            # makes the margin remove exactly ~m px per end.
            if len(coords_xy) >= 2:
                _seg = np.diff(coords_xy, axis=0)
                _s = np.concatenate(
                    [[0.0], np.cumsum(np.hypot(_seg[:, 0], _seg[:, 1]))])
                _tot = float(_s[-1])
                if _tot >= 2.0:
                    _n = int(np.clip(round(_tot) + 1,
                                     max(5, len(coords_xy)), 400))
                    _sn = np.linspace(0.0, _tot, _n)
                    coords_xy = np.column_stack([
                        np.interp(_sn, _s, coords_xy[:, 0]),
                        np.interp(_sn, _s, coords_xy[:, 1])])
            if len(coords_xy) > 4:
                seg = np.diff(coords_xy, axis=0)
                arc = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
                total = arc[-1]
                m = margin_px if margin_px is not None else min(3.0, 0.15 * total)
                if m > 0:
                    keep = (arc >= m) & (arc <= total - m)
                    if keep.sum() >= 5:
                        coords_xy = coords_xy[keep]
            return coords_xy, mean_r, False

        self._run_of_btn.setEnabled(False)
        self._info_box.setPlainText(
            f"Running optical flow on edge ({u}, {v}), tile {tile_id} "
            f"…  (~5–10s)")
        QApplication.processEvents()

        try:
            t0 = _time.time()
            # 1. Centerline + radius from the graph (mosaic Cartesian).
            #    Optionally merge the whole degree-2 vessel chain so OF runs on
            #    the full vessel, not just the clicked segment.
            if getattr(self, '_of_merge_chk', None) is not None and self._of_merge_chk.isChecked():
                of_chain = self._degree2_chain_edges(u, v)
            else:
                of_chain = [(u, v)]
            coords_xy, mean_radius_px, _was_rev = _get_chain_coords_local(
                self.G, of_chain, margin_px=None)   # minimal junction trim
            # Also grab the UNtrimmed chain so the figure can show exactly how
            # much centerline was cropped off the ends (vs. kept for analysis).
            coords_full_xy, _, _ = _get_chain_coords_local(
                self.G, of_chain, margin_px=0.0)
            if len(of_chain) > 1:
                print(f"  merged degree-2 chain: {len(of_chain)} segments "
                      f"({of_chain[0][0]} … {of_chain[-1][1]})")
            if coords_xy is None or len(coords_xy) < 5:
                self._info_box.setPlainText(
                    f"Vessel ({u}, {v}) too short for OF "
                    f"(N={0 if coords_xy is None else len(coords_xy)}).")
                self._run_of_btn.setEnabled(True)
                return

            # 2. Load full-res video, percentile-normalise to uint8.
            print("\n  ── Optical-flow re-analysis (read-only) ──")
            print(f"  edge ({u}, {v}), tile {tile_id}, "
                  f"video {video_path}")
            # No frame cap → matches batch's default (max_frames=None loads
            # ALL raw frames, then cut_before_fade trims the fade-out).
            # Fresh-OF and batch now see identical stacks.
            stack = load_tiff_stack(video_path, max_frames=None)
            stack, _cut_at = cut_before_fade(stack, verbose=False)
            # Pass the float32 stack straight through — do NOT pre-clip to
            # uint8.  The batch does the same, and compute_of_profile_and_Q
            # then applies its bg_percentile background subtraction
            # (of_profile L29-30).  Pre-converting to uint8 here made the
            # function see uint8 and SKIP the background subtraction (L31) →
            # OF tracked the static background instead of the moving speckle,
            # producing the "awful" fresh-PIV result.  Keep it float.
            stack = np.asarray(stack, dtype=np.float32)
            T, H, W = stack.shape
            print(f"  compute source: {T} frames, {H}×{W}, "
                  f"radius (mosaic) = {mean_radius_px:.1f} px")

            # Downsample factor is used ONLY for the px→µm unit conversion
            # below.  Geometry is handled natively by mosaic_to_tile_coords
            # (via video_w/tile_width), so it is correct at any resolution.
            ds_y = float(H) / float(TILE_RAW_HEIGHT) if H else 1.0
            ds_x = float(W) / float(TILE_RAW_WIDTH) if W else 1.0

            # 3. Convert mosaic Cartesian coords → tile pixel using the SAME
            #    batch transforms (mosaic_to_tile_coords + mosaic_radius_to_tile
            #    with affine re-referencing), so a fresh recompute reproduces
            #    the stored Q(t) exactly.  The earlier inline (coord−translate)/
            #    scale did NOT apply the affine re-reference or the
            #    video_w/tile_width scaling → the centerline was misplaced and
            #    OF integrated the wrong pixels (the "awful" result).
            # Reuse the batch analyzer's geom object — same source of truth
            # so a fresh recompute here matches the stored batch measurement
            # exactly.  Cached on the viewer after first build.
            if getattr(self, '_of_geom', None) is None:
                cfg = StageConfig.load(self.graph_path.parents[1])
                all_shapes = {tid: shp for tid in cfg.discover_raw_tile_ids()
                              for shp in [tile_video_shape(cfg, tid)]
                              if shp is not None}
                self._of_geom = load_tile_geometry(cfg, video_shapes=all_shapes)
            geom = self._of_geom
            centerline_rc = graph_to_tile_batch(coords_xy, geom, int(tile_id))
            # Untrimmed full-vessel CL in the same tile frame (for the figure's
            # "how much was cropped" overlay).  Not in-bounds-trimmed.
            try:
                centerline_full_rc = graph_to_tile_batch(
                    coords_full_xy, geom, int(tile_id))
            except Exception:
                centerline_full_rc = centerline_rc
            # Trim to largest in-bounds contiguous segment
            margin = 2
            ib = ((centerline_rc[:, 0] >= margin) & (centerline_rc[:, 0] <= H - margin) &
                  (centerline_rc[:, 1] >= margin) & (centerline_rc[:, 1] <= W - margin))
            if np.any(ib) and not np.all(ib):
                changes = np.diff(ib.astype(int))
                starts = list(np.where(changes == 1)[0] + 1)
                ends = list(np.where(changes == -1)[0] + 1)
                if ib[0]:  starts = [0] + starts
                if ib[-1]: ends = ends + [len(ib)]
                if starts and ends:
                    n = min(len(starts), len(ends))
                    best = int(np.argmax(np.array(ends[:n]) - np.array(starts[:n])))
                    centerline_rc = centerline_rc[starts[best]:ends[best]]
            r_tile = float(graph_radius_to_tile_px(
                float(mean_radius_px), geom, int(tile_id)))
            vessel_angle = float(np.arctan2(
                centerline_rc[-1, 0] - centerline_rc[0, 0],
                centerline_rc[-1, 1] - centerline_rc[0, 1]))
            print(f"  centerline tile range: rows "
                  f"[{centerline_rc[:, 0].min():.0f}, {centerline_rc[:, 0].max():.0f}], cols "
                  f"[{centerline_rc[:, 1].min():.0f}, {centerline_rc[:, 1].max():.0f}]")
            print(f"  radius (tile)  = {r_tile:.1f} px  "
                  f"(θ = {np.degrees(vessel_angle):.0f}°)")

            # 4. Run the unified OF pipeline — same call as the editing
            # viewer's Analyze Mode.
            edge_data = self.G.edges[u, v]
            # v4: f0 lives per-tile in measurements_piv[tile]['harm']['f0_hz'],
            # not on the edge.  Reading the dead flat 'f0_hz' returned None and
            # silently fell back to 2.5 Hz — which mis-set the cycle-averaging
            # period and the harmonic fit (flat waveform, SNR≈0).  Use the tile.
            f0 = self._tile_f0_hz(edge_data, tile_id)
            # Farneback params match the frozen calibration
            # (memory/of_calibration_frozen.md): fixed winsize=3, dt=1.
            # Batch pipeline uses these; keeping fresh OF aligned so a
            # fresh recompute reproduces stored Q(t) for the same edge.
            fb_ws = 3
            unified = compute_of_profile_and_Q(
                stack, centerline_rc, r_tile,
                bg_percentile=10,
                fb_winsize=fb_ws, fb_levels=1,
                fb_iterations=3,
                fb_poly_n=3, fb_poly_sigma=0.7,
                # dt=1: reverted from dt=3 (2026-08-24-verified aliasing on
                # fast A vessels — v~600-1000 µm/s hits 4-6 px/frame at dt=3,
                # well beyond Farneback winsize=3 capture, flat-tops the peak).
                # dt=1 is unbiased at all speeds; a bit noisier for slow venous.
                dt=1, n_bins=30, fit_fraction=0.75,
                refine_centerline=(getattr(self, '_of_refine_chk', None) is None
                                   or self._of_refine_chk.isChecked()),
                use_flow_tangent=True,
                tangent_smooth_sigma=5.0,
                f0_hz=f0, frame_dt_s=FRAME_DT_S,
                return_diagnostics=True,   # for the "Show OF pipeline" button
            )
            # DEBUG: confirm the σ=4 centerline smoothing is reaching fresh OF.
            _clu = np.asarray(unified.get('cl_used')) if unified.get('cl_used') is not None else None
            _rgh = (float(np.mean(np.abs(_clu[2:] - 2 * _clu[1:-1] + _clu[:-2])))
                    if _clu is not None and len(_clu) > 3 else -1.0)
            print(f"[fresh OF] cl_used roughness = {_rgh:.3f} px  "
                  f"(smoothed σ=4 ≈ <1 ; jagged σ=0 ≈ >1.5)")
            elapsed = _time.time() - t0
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"Optical flow failed: {type(e).__name__}: {e}")
            self._run_of_btn.setEnabled(True)
            return

        # 5. Build the of_result dict in the shape _analyze_of_display
        #    expects — mirrors the editing viewer's adapter at
        #    pertile/viewer/mosaic/_kirchhoff.py:2870-2937.
        #
        # If the bundle ships pre-downsampled videos (e.g. ds_x = 0.5
        # for a 2× spatial downsample) one displayed pixel covers
        # (1 / ds) × PX_SIZE_UM real micrometers, so use the effective
        # pixel size for the unit conversion.  At ds = 1 (full-res)
        # this collapses to the original formula.
        ds_avg = 0.5 * (ds_x + ds_y) if (ds_x and ds_y) else 1.0
        PX_SIZE_EFF = float(PX_SIZE_UM) / ds_avg
        conv = PX_SIZE_EFF ** 3 / FRAME_DT_S * 1e-6  # px²/frame → nL/s
        v_conv = PX_SIZE_EFF / FRAME_DT_S            # px/frame  → µm/s
        R_um = r_tile * PX_SIZE_EFF

        # ── Segmentation-CL re-bin of the SAME OF field (no extra Farneback):
        # this reproduces the batch's Q_pois_seg — the canonical default since
        # the KCL bake-off (2026-07; raw seg conserves mass better than the
        # ridge-refined CL or the r(R)-calibrated value).  We report seg as the
        # headline Q and rescale the (ridge) Poiseuille waveform to the seg
        # mean, so the fresh overlay matches the stored Q(t) — which the batch
        # now builds the same way (repoint block in batch_analyze_mosaic_piv).
        # The ridge value is kept and still shown for reference.
        Q_pois_ridge_px = float(unified.get('Q_pois_mean', np.nan))
        Q_pois_seg_px = Q_pois_ridge_px
        pois_seg_factor = 1.0
        try:
            from ..analyze.profile import _rebin_profile
            _seg = _rebin_profile(unified, centerline_rc, r_tile,
                                  n_bins=30, fit_fraction=0.75)
            _sv = float(_seg.get('Q_pois_mean', np.nan))
            if (np.isfinite(_sv) and np.isfinite(Q_pois_ridge_px)
                    and abs(Q_pois_ridge_px) > 1e-9):
                Q_pois_seg_px = _sv
                pois_seg_factor = abs(_sv) / abs(Q_pois_ridge_px)
        except Exception:
            import traceback; traceback.print_exc()

        # Match the BATCH canonical scale target (mosaic_piv:_analyze_vessel):
        # Kloosterman → Q_pois_free → seg.  Kloosterman is the citable,
        # most-defensible method, so fresh OF's headline Q and rescaled
        # waveform reproduce the stored batch Q(t) instead of the old
        # seg-Poiseuille default.  (Overrides pois_seg_factor computed above.)
        _Q_kl_px = float(unified.get('Q_kloosterman', np.nan))
        _v0_free = float(unified.get('v0_free', np.nan))
        _R_free_px = float(unified.get('R_fit_free', np.nan))
        _Q_pf_px = (abs(_v0_free) * np.pi * _R_free_px ** 2 / 2.0
                    if (np.isfinite(_v0_free) and np.isfinite(_R_free_px)
                        and _R_free_px > 0) else np.nan)
        # Kloosterman → bounded mask-R Poiseuille (seg) → railing-prone free-R.
        if np.isfinite(_Q_kl_px):
            _Q_scale_px = _Q_kl_px; self._of_best_key = 'kl'
        elif np.isfinite(Q_pois_seg_px):
            _Q_scale_px = Q_pois_seg_px; self._of_best_key = 'pois_seg'
        elif np.isfinite(_Q_pf_px):
            _Q_scale_px = _Q_pf_px; self._of_best_key = 'pois_free'
        else:
            _Q_scale_px = Q_pois_seg_px; self._of_best_key = 'pois_seg'
        if np.isfinite(_Q_scale_px) and abs(Q_pois_ridge_px) > 1e-9:
            pois_seg_factor = abs(_Q_scale_px) / abs(Q_pois_ridge_px)

        # Stash the full result for the "Show OF pipeline" button (4-figure
        # diagnostic).  Grab this tile's stored Q(t) for the fresh-vs-stored panel.
        _sQ = None
        for _mm in (self.G.edges[u, v].get('measurements_piv') or []):
            if _mm.get('tile_id') == int(tile_id):
                _sQ = _mm.get('Q_t'); break
        self._last_of_pipeline = dict(
            res=unified, frame_dt_s=FRAME_DT_S, v_conv=v_conv, q_conv=conv,
            stored_Qt=_sQ, f0_hz=f0, title=f"edge ({u}, {v}) tile {tile_id}",
            # Which vessel/tile this result is for — lets the "Diagnose
            # vessel quality" button tell whether the stashed OF matches the
            # currently-selected edge (and re-run fresh OF if it's stale).
            edge=(int(u), int(v)), tile_id=int(tile_id),
            # Canonical-Q scale + key so the pipeline figure can rebuild the
            # reported Q(t) waveform straight from `unified` (one data path).
            pois_seg_factor=float(pois_seg_factor),
            best_key=getattr(self, '_of_best_key', 'kl'),
            # Stash the raw stack + geometry so the PIV comparison button can
            # rerun on identical inputs without reloading video from disk.
            frames=stack, centerline_rc=centerline_rc,
            centerline_full_rc=centerline_full_rc, r_tile=r_tile)
        if hasattr(self, '_of_pipeline_btn'):
            self._of_pipeline_btn.setEnabled(True)
        if hasattr(self, '_piv_compare_btn'):
            self._piv_compare_btn.setEnabled(True)
        Q_plug_nLs = unified['Q_plug_mean'] * conv
        Q_pois_nLs = unified['Q_pois_mean'] * conv       # ridge CL (reference)
        Q_pois_seg_nLs = Q_pois_seg_px * conv            # seg CL = canonical default
        v0_fix_ums = unified['v0_fixed'] * v_conv
        v0_free_ums = unified.get('v0_free', np.nan) * v_conv
        R_fit_free = unified.get('R_fit_free', r_tile)
        Q_pois_free_nLs = (unified.get('v0_free', np.nan)
                            * np.pi * R_fit_free ** 2 / 2.0 * conv)
        v_mean_ums = Q_plug_nLs * 1e6 / (np.pi * R_um ** 2)
        vf = unified.get('vfield_mean')
        roi_offset = unified.get('roi_offset', (0, 0))
        r0_roi, c0_roi = int(roi_offset[0]), int(roi_offset[1])
        if vf is not None:
            roi_h, roi_w = vf.shape[:2]
        else:
            roi_h, roi_w = H, W
            r0_roi = c0_roi = 0
        rr_in = unified['rr_in']
        cc_in = unified['cc_in']
        mask_roi = np.zeros((roi_h, roi_w), dtype=bool)
        if len(rr_in) > 0:
            mask_roi[rr_in, cc_in] = True
        Q_pois_orig_nLs = unified.get(
            'Q_pois_orig_mean', np.nan) * conv
        v0_orig_ums = unified.get('v0_orig', np.nan) * v_conv

        of_result = dict(
            Q_mean=Q_plug_nLs,
            Q_poiseuille=Q_pois_seg_nLs,   # canonical default = seg CL
            Q_poiseuille_ridge=Q_pois_nLs,
            Q_pois_orig=Q_pois_orig_nLs,
            Q_pois_free=Q_pois_free_nLs,
            v_mean=v_mean_ums,
            v0_fit=v0_fix_ums,
            v0_free=v0_free_ums,
            v0_orig=v0_orig_ums,
            R_stored_px=r_tile,
            R_flow_px=R_fit_free,
            R_flow_um=R_fit_free * PX_SIZE_UM,
            delta_px=unified.get('cl_shift', 0.0),
            best_side='center',
            fit_left=None, fit_right=None,
            c_floor=np.nan,
            profile_r=unified['bin_centers'],
            profile_v=np.abs(unified['profile_v']) * v_conv,
            profile_v_orig=np.abs(unified.get('profile_v_orig',
                np.full_like(unified['profile_v'], np.nan))) * v_conv,
            profile_r_local=None, profile_v_local=None,
            Q_t_of_mean=unified['Q_plug_t'] * conv,
            # ridge per-frame waveform rescaled to the seg mean (canonical
            # default), so this matches the stored Q(t) built by the batch.
            Q_t_of_pois=unified['Q_pois_t'] * conv * pois_seg_factor,
            Q_t_of_local=None,
            Q_local=np.nan,
            v_field_mean=(vf * v_conv if vf is not None else None),
            roi_bounds=(r0_roi, r0_roi + roi_h, c0_roi, c0_roi + roi_w),
            vessel_mask_roi=mask_roi,
            wide_mask_roi=mask_roi,
            cl_shift=unified.get('cl_shift', 0.0),
            cl_used=unified.get('cl_used'),
            # Initial (pre-refinement) tile-frame centerline the OF pipeline
            # received.  Kept so the diagnostic can plot both CLs on the
            # mean-field panel and expose how much the refinement moved.
            cl_initial=np.asarray(centerline_rc, dtype=float),
            v0_deconv=unified.get('v0_deconv', np.nan) * v_conv,
            R_deconv=unified.get('R_deconv', r_tile),
            sigma_of_fit=unified.get('sigma_of_fit', np.nan),
        )

        # 6. Curved ROI overlay on the mosaic — DISABLED in the plexusflow
        #    port: it used pertile's simple scale+translate tile model
        #    (`_of_tile_geometry` → `batch_analyze_mosaic.parse_tile_positions`),
        #    which doesn't exist here — this port uses the affine `_of_geom`.
        #    The walkthrough figure's ROI-locator panel (full tile + green crop
        #    box, mosaic-oriented) now serves the same "where is the crop"
        #    purpose, so this napari ribbon is not re-implemented.

        # 7. Info-box summary (matches what shows in the editing viewer's
        #    console, condensed).
        self._info_box.setPlainText(
            f"Optical flow — edge ({u}, {v}), tile {tile_id}\n"
            f"  vessel            = {len(of_chain)} segment(s) merged"
            f"{' [whole degree-2 vessel]' if len(of_chain) > 1 else ' [single segment]'}\n"
            f"  source            = {video_path}\n"
            f"  compute frames    = {T} × {H} × {W}\n"
            f"  Q_pois (seg CL)   = {abs(Q_pois_seg_nLs):.4f} nL/s  [canonical default]\n"
            f"  Q_pois (ridge CL) = {abs(Q_pois_nLs):.4f} nL/s  (×{pois_seg_factor:.3f} → seg)\n"
            f"  Q_plug (ridge CL) = {abs(Q_plug_nLs):.4f} nL/s\n"
            f"  v̄                 = {abs(v_mean_ums):.0f} µm/s\n"
            f"  R (tile)          = {r_tile:.2f} px  "
            f"({R_um:.2f} µm)\n"
            f"  CL shift          = {unified.get('cl_shift', 0.0):+.1f} px\n"
            f"  computed in {elapsed:.1f}s   "
            f"(read-only: graph not modified)")

        # 7. Render the comprehensive stage-by-stage OF pipeline walkthrough
        #    (single figure, one data path off `unified`/`_last_of_pipeline`).
        #    Replaces the old 4-panel `_render_of_diagnostic` popup — that
        #    method is retained but unused, in case a minimal view is wanted.
        try:
            self._render_of_pipeline(self._last_of_pipeline)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                self._info_box.toPlainText()
                + f"\n\nPipeline figure failed: "
                f"{type(e).__name__}: {e}")
        self._run_of_btn.setEnabled(True)

    def _tile_f0_hz(self, edge_data, tile_id, fallback=2.5):
        """Heart-rate f0 (Hz) for a given tile's measurement on this edge.

        v4 stores f0 per measurement in ``measurements_piv[tile]['harm']
        ['f0_hz']`` — NOT as a flat edge field.  Falls back to the tile-f0
        graph metadata, then any other measurement's f0, then a sane yolk-sac
        default, so the cycle-averaging period and harmonic fit are never
        driven by a stale 2.5 Hz."""
        def _ok(x):
            try:
                return x is not None and np.isfinite(float(x)) and float(x) > 0
            except (TypeError, ValueError):
                return False
        meas = edge_data.get('measurements_piv') or []
        # 1. this tile's own measurement
        for mm in meas:
            if mm.get('tile_id') == int(tile_id):
                f0 = (mm.get('harm') or {}).get('f0_hz')
                if _ok(f0):
                    return float(f0)
                break
        # 2. graph-level per-tile f0 metadata — try the canonical key
        # `tile_f0_piv` (what batch writes) first, then the legacy
        # `tile_f0s` for backwards compat.
        for meta_key in ('tile_f0_piv', 'tile_f0s'):
            tf = getattr(self.G, 'graph', {}).get(meta_key) or {}
            for key in (int(tile_id), str(tile_id)):
                if _ok(tf.get(key)):
                    return float(tf[key])
        # 3. any measurement on this edge with a usable f0
        for mm in meas:
            f0 = (mm.get('harm') or {}).get('f0_hz')
            if _ok(f0):
                return float(f0)
        # 4. legacy flat field, then default
        f0 = edge_data.get('f0_hz')
        return float(f0) if _ok(f0) else float(fallback)

    def _show_of_region_overlay(self, of_result, tile_id, u, v,
                                  sx, sy, trans_x, trans_y_disp):
        """Render the OF integration mask as a curved orange ribbon on
        the mosaic.  Replaces any previous OF overlay.

        Pipeline:
          1.  Extract mask boundary contour(s) in ROI-pixel coords
              via skimage.measure.find_contours.
          2.  ROI → tile pixel (add roi_offset).
          3.  Tile pixel → mosaic display (multiply by sx/sy, add tile
              translates).  Inverse of the forward transform used to
              build centerline_rc for compute_of_profile_and_Q.
          4.  Add as a napari Shapes layer (polygon, orange fill +
              outline).  Mosaic display coords are (row, col) with
              row 0 at the top — matches every other Shapes layer in
              this viewer.

        Persists until the next OF run replaces it or
        _tear_down_of_region_overlay clears it explicitly."""
        try:
            from skimage.measure import find_contours
        except ImportError:
            print("  (skimage missing — skipping ROI overlay)")
            return
        mask = of_result.get('vessel_mask_roi')
        if mask is None or not mask.any():
            return
        roi_bounds = of_result.get('roi_bounds')
        if roi_bounds is None:
            return
        r0_roi, _r1_roi, c0_roi, _c1_roi = roi_bounds
        # find_contours returns a list of (N, 2) arrays in (row, col)
        # ROI coords.  Each ring becomes one polygon — keep them all so
        # vessels with internal holes (rare) draw correctly.
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            return
        polys = []
        H_mosaic = self.mosaic_height
        for c in contours:
            if len(c) < 3:
                continue
            i_roi = c[:, 0]
            j_roi = c[:, 1]
            # ROI → tile pixel
            i_tile = i_roi + r0_roi
            j_tile = j_roi + c0_roi
            # Tile pixel → mosaic display (inverse of the j_tile / i_tile
            # formulae used when building centerline_rc).  display_row
            # is the napari row index (row 0 at the top).
            display_row = i_tile * sy + trans_y_disp
            display_col = j_tile * sx + trans_x
            # napari Shapes wants (row, col) — same as display.  No
            # mosaic-height flip needed here because we never went
            # through the Cartesian-y representation.
            poly = np.column_stack([display_row, display_col])
            polys.append(poly)
        # Replace any prior overlay before adding the new one.
        self._tear_down_of_region_overlay(suppress_button=True)
        if not polys:
            return
        _prev_active = self.viewer.layers.selection.active
        self._of_region_layer = self.viewer.add_shapes(
            polys,
            shape_type='polygon',
            edge_color='#ff8800',
            face_color='#ff8800',
            edge_width=2.5,
            opacity=0.32,
            name=f'OF region — edge ({u},{v}), tile {tile_id}',
        )
        # Don't let the overlay capture mouse events — clicks must still
        # reach the edges layer for selection / inspection.
        try:
            self._of_region_layer.mouse_pan = False
            self._of_region_layer.mouse_zoom = False
        except Exception:
            pass
        if _prev_active is not None:
            try:
                self.viewer.layers.selection.active = _prev_active
            except Exception:
                pass
        try:
            self._clear_of_region_btn.setEnabled(True)
        except AttributeError:
            pass

    def _tear_down_of_region_overlay(self, suppress_button: bool = False):
        """Remove the OF region Shapes layer if present.  `suppress_button`
        skips the Clear-button disable (used internally by the show path
        when replacing one overlay with another)."""
        lyr = getattr(self, '_of_region_layer', None)
        if lyr is not None:
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            self._of_region_layer = None
        if not suppress_button:
            try:
                self._clear_of_region_btn.setEnabled(False)
            except AttributeError:
                pass

    def _on_plot_boundary_kcl(self):
        """Handler for the 'Plot KCL sum at boundary' button.  Checks
        that the selected edge touches a boundary node, then calls
        `_plot_boundary_kcl_qt` on that boundary side."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        bt_u = self.G.nodes[u].get('boundary_type')
        bt_v = self.G.nodes[v].get('boundary_type')
        if bt_u in ('source', 'sink'):
            boundary_node, junction_node = u, v
        elif bt_v in ('source', 'sink'):
            boundary_node, junction_node = v, u
        else:
            try:
                self._info_box.setPlainText(
                    "Selected edge is not a boundary arm — "
                    "click an edge that touches an A or V boundary node.")
            except Exception:
                pass
            return
        self._plot_boundary_kcl_qt(boundary_node, junction_node)

    def _plot_boundary_kcl_qt(self, boundary_node: int, junction_node: int):
        """For a boundary edge (arm from `boundary_node` at the tip to
        `junction_node` in the network), pop a figure showing the
        KCL-reconstructed Q(t) at the junction (sum of J's OTHER
        incident vessels, sign-flipped so ``+`` = flow out through the
        arm toward the boundary).  This is the "true" boundary Q(t)
        that a downstream sim should use as a BC — NOT the noisy arm
        measurement.

        Overlays the arm's own raw Q(t) traces (per tile) for
        comparison.  Uses tile 14 (the reference tile) for every
        contributing vessel where available; falls back to best-SNR
        per vessel otherwise.

        Prints numeric summary to the info box.
        """
        import matplotlib.pyplot as plt
        REF_TILE = 14

        def _get_meas_at_tile(u, v, tile_id):
            ed = self.G.edges[u, v]
            for m in (ed.get('measurements_piv') or []):
                if m.get('tile_id') == tile_id:
                    return m
            return None

        def _best_meas(u, v):
            ed = self.G.edges[u, v]
            meas = ed.get('measurements_piv') or []
            if not meas:
                return None
            def key(m):
                h = ((m.get('harm') or {}).get('harmonics') or [{}])[0]
                return h.get('snr_db') if h.get('snr_db') is not None else -np.inf
            return max(meas, key=key)

        # Recurse KCL at J for a subgraph excluding the boundary side.
        # For each measured branch use its tile 14 Q_t (or best-SNR fallback).
        # For each unmeasured branch, recurse one hop into the grey node's
        # own neighbors (same rule).
        _visited = {boundary_node, junction_node}
        contributions = []   # list of (edge_repr, tile_id, Q_t signed, mean_Q)
        for nbr in self.G.neighbors(junction_node):
            if nbr in _visited: continue
            m = _get_meas_at_tile(junction_node, nbr, REF_TILE) or _best_meas(junction_node, nbr)
            if m is not None:
                mv = MeasurementView(m)
                if mv.Q_t is None: continue
                Q_t = np.asarray(mv.Q_t, dtype=float)
                # Sign for KCL at J: measurement's flow_from tells us which way flow
                # is defined.  We want "flow OUT of J toward nbr".
                #   If flow_from == J, measurement is J→nbr → sign = +1.
                #   Else (flow_from == nbr) → measurement is nbr→J → sign = -1.
                sign = +1.0 if mv.flow_from == junction_node else -1.0
                contributions.append((f'({junction_node},{nbr})', m.get('tile_id'),
                                      sign * Q_t, sign * (mv.harm.mean_Q or 0)))
            else:
                # Grey branch — one hop recursion into `nbr`'s neighbors
                _visited_inner = _visited | {nbr}
                for nbr2 in self.G.neighbors(nbr):
                    if nbr2 in _visited_inner: continue
                    m2 = _get_meas_at_tile(nbr, nbr2, REF_TILE) or _best_meas(nbr, nbr2)
                    if m2 is None: continue
                    mv2 = MeasurementView(m2)
                    if mv2.Q_t is None: continue
                    Q_t = np.asarray(mv2.Q_t, dtype=float)
                    # At `nbr`, contribution to "flow OUT of nbr toward nbr2".
                    # By KCL at nbr: this equals flow INTO nbr from J side
                    # (= flow OUT of J toward nbr).  Sign per above rule.
                    sign = +1.0 if mv2.flow_from == nbr else -1.0
                    contributions.append(
                        (f'({nbr}→{nbr2}) [via grey]', m2.get('tile_id'),
                         sign * Q_t, sign * (mv2.harm.mean_Q or 0)))

        if not contributions:
            print(f'[boundary Q(t)] No usable contributions at J={junction_node} — cannot reconstruct.')
            return

        # Align lengths (contributions may have different N; take the min)
        min_N = min(len(c[2]) for c in contributions)
        Q_sum = np.zeros(min_N)
        for _, _, Q_t, _ in contributions:
            Q_sum += Q_t[:min_N]
        # KCL sum at J = "sum of flow OUT of J to all OTHER branches" = flow OUT
        # of J toward the boundary arm (via KCL closure).  For a boundary sink,
        # bc_harmonics convention: +DC = flow injected INTO network at boundary.
        # A physical sink drains network → bc DC negative.  Sum here IS the
        # "flow through the arm toward the boundary", so bc = -Q_sum for sinks
        # and bc = +Q_sum for sources.  But for VIEWING the raw KCL sum vs the
        # arm's own Q_t, we plot Q_sum unsigned (as computed).
        mean_sum = float(Q_sum.mean())

        # Also grab the arm's raw traces for comparison
        arm_traces = []
        arm_meas = self.G.edges[boundary_node, junction_node].get(
            'measurements_piv') or []
        for m in sorted(arm_meas,
                         key=lambda mm: (
                             ((mm.get('harm') or {}).get('harmonics') or [{}])[0]
                             .get('snr_db') or -np.inf), reverse=True):
            mv = MeasurementView(m)
            if mv.Q_t is None: continue
            # Same sign rule at boundary_node
            sign = +1.0 if mv.flow_from == boundary_node else -1.0
            arm_traces.append((m.get('tile_id'), sign * np.asarray(mv.Q_t, dtype=float),
                                 (((m.get('harm') or {}).get('harmonics') or [{}])[0])
                                 .get('snr_db')))

        # Build the figure
        fig, ax = plt.subplots(figsize=(11, 5.5))
        t_sum = np.arange(min_N) / 250.0
        # Individual contributions (thin, gray-ish)
        for label, tile_id, Q_t, mq in contributions:
            t = np.arange(min(len(Q_t), min_N)) / 250.0
            ax.plot(t, Q_t[:len(t)], lw=0.7, alpha=0.55,
                     label=f'contrib {label} tile={tile_id}  mean={mq:+.3f}')
        # The KCL sum (bold blue)
        ax.plot(t_sum, Q_sum, lw=2.4, color='C0',
                 label=f'KCL SUM at J={junction_node}  mean={mean_sum:+.3f}')
        # The arm's raw traces (dashed orange, for comparison)
        for tile_id, Q_t, snr in arm_traces:
            t = np.arange(len(Q_t)) / 250.0
            ax.plot(t, Q_t, lw=1.0, ls='--', color='C1', alpha=0.7,
                     label=(f'arm ({boundary_node}→{junction_node}) '
                             f'tile={tile_id}  mean={Q_t.mean():+.3f}  '
                             f'SNR={snr:.1f}dB'))

        boundary_type = self.G.nodes[boundary_node].get('boundary_type', '?')
        ax.set_xlabel('t (s)')
        ax.set_ylabel('Q (nL/s) — signed to match "flow OUT of J toward arm"')
        ax.set_title(f'Boundary Q(t) reconstruction  —  '
                      f'{boundary_type.upper()} {boundary_node},  J={junction_node}\n'
                      f'Bold blue = KCL SUM (what your colleague should use as BC).  '
                      f'Dashed orange = arm\'s own raw Q(t) (untrusted at boundary).')
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

        # Summary in info box
        lines = [
            f"Boundary Q(t) at {boundary_type.upper()} {boundary_node}",
            f"  J = {junction_node}, contributions = {len(contributions)}",
            f"  KCL SUM  mean_Q = {mean_sum:+.4f} nL/s",
            "  arm raw  mean_Q = " + ', '.join(
                f"{Q_t.mean():+.3f}(tile{tid})" for tid, Q_t, _ in arm_traces),
        ]
        try:
            self._info_box.setPlainText('\n'.join(lines))
        except Exception:
            pass

    def _get_edge_harmonics(self, u, v, source):
        """Return [Z_DC, Z_H1, Z_H2, Z_H3] complex phasors for an edge in
        a given source ∈ {'measured', 'normalized', 'sim', 'sim_carved'}.
        Returns None if the source has no data for this edge.
        Convention matches compare_waveforms.py: Q(t) = Re(Z0) + Σ Re(Zk·e^{iωkt})."""
        d = self.G.edges[u, v]
        Z = [None, None, None, None]
        if source == 'normalized':
            # NOTE: `_h_Z_DC_norm` is a display proxy (|DC|·10, real scale)
            # and `_h_Z_H{k}_norm` is an SNR ratio 10^(snr/20). Only the
            # `_h_amp_*_norm` + `_h_phase_*_norm` scalars are the true
            # per-tile-gauged phasor components.
            a0 = d.get('_h_amp_DC_norm')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0), 0.0)
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}_norm')
                p = d.get(f'_h_phase_H{k}_norm')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
        elif source == 'sim':
            arr = d.get('_sim_tmp_harmonics')
            if arr is not None:
                arr = np.asarray(arr, dtype=complex)
                for k in range(min(4, len(arr))):
                    Z[k] = complex(arr[k])
        elif source == 'sim_carved':
            a0 = d.get('_h_amp_DC_sim_carved')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0))
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}_sim_carved')
                p = d.get(f'_h_phase_H{k}_sim_carved')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
        elif source == 'measured':
            # `_h_Z_DC` / `_h_Z_H{k}` are display-scale proxies (same
            # convention as their _norm counterparts). Use the true
            # amp+phase scalars instead.
            a0 = d.get('_h_amp_DC')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0), 0.0)
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}'); p = d.get(f'_h_phase_H{k}')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
            # Last-resort fallback: fit Q_t of the top-quality tile via FFT
            if all(z is None for z in Z):
                meas = d.get('measurements_piv') or []
                if meas:
                    m = sorted(meas, key=lambda mm: (
                        QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                        MeasurementView(mm).snr.h1_db or 0.0), reverse=True)[0]
                    Q_t = m.get('Q_t')
                    if Q_t is not None and len(Q_t) > 8:
                        Qt = np.asarray(Q_t, dtype=float)
                        N = len(Qt)
                        Z[0] = complex(float(np.mean(Qt)), 0.0)
                        # crude one-sided spectrum: pick nearest bin to f0
                        f0 = self._detect_sim_f0_hz()
                        if not np.isfinite(f0): f0 = 2.5
                        dt = float(m.get('frame_dt_s') or m.get('dt_s') or 1.0/250.0)
                        F = np.fft.rfft(Qt - np.mean(Qt)) / (N / 2.0)
                        freqs = np.fft.rfftfreq(N, d=dt)
                        for k in (1, 2, 3):
                            j = int(np.argmin(np.abs(freqs - k*f0)))
                            if 0 < j < len(F):
                                Z[k] = complex(F[j])
        return Z

    def _reconstruct_qt(self, Z, n_cycles=2, n_samples=200):
        """Q(t) = Re(Z_DC) + Σ_{k=1..3} Re(Z_k · exp(i·k·ω₁·t)).
        Returns (t_cycles, Q_arr)."""
        f0 = self._detect_sim_f0_hz()
        if not np.isfinite(f0): f0 = 2.5
        omega1 = 2*np.pi*f0
        t_grid = np.linspace(0.0, n_cycles/f0, n_samples)
        q = np.zeros(n_samples)
        if Z[0] is not None: q += float(np.real(Z[0]))
        for k in (1, 2, 3):
            if Z[k] is None: continue
            q += np.real(Z[k] * np.exp(1j*k*omega1*t_grid))
        return t_grid * f0, q

    def _plot_selected_qt_overlay(self):
        """Overlay Q(t) from the current View-tab source with the
        'Compare with' source from the combo box (both reconstructed
        from stored harmonics DC + H1..H3). Same convention as
        compare_waveforms.py."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        src_a = getattr(self, 'current_source', 'measured')
        src_b = self._overlay_source_combo.currentData()
        if src_b is None or src_b == src_a:
            # nothing to compare — fall back to the ordinary plot
            self._plot_selected_qt()
            return
        Z_a = self._get_edge_harmonics(u, v, src_a)
        Z_b = self._get_edge_harmonics(u, v, src_b)
        # Canonicalise each source so Q̄ ≥ 0 (undirected-edge convention).
        # If DC is negative, flip the ENTIRE phasor set (multiply all
        # harmonics by −1) — that's the same as reversing (u, v) as the
        # reference direction. Preserves waveform shape.
        def _flip_if_neg(Z):
            if Z[0] is None: return Z
            if float(np.real(Z[0])) < 0:
                return [(None if z is None else -z) for z in Z]
            return Z
        Z_a = _flip_if_neg(Z_a)
        Z_b = _flip_if_neg(Z_b)
        if all(z is None for z in Z_a) or all(z is None for z in Z_b):
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self.viewer.window._qt_window, "No overlay",
                f"One of the sources has no phasor data on edge ({u}, {v}).\n"
                f"  {src_a}: {'ok' if any(z is not None for z in Z_a) else 'missing'}\n"
                f"  {src_b}: {'ok' if any(z is not None for z in Z_b) else 'missing'}")
            return
        t_a, q_a = self._reconstruct_qt(Z_a)
        t_b, q_b = self._reconstruct_qt(Z_b)
        import matplotlib
        matplotlib.use('QtAgg', force=False) if hasattr(matplotlib, 'use') else None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), facecolor='white')
        dc_a = float(np.real(Z_a[0])) if Z_a[0] is not None else 0.0
        dc_b = float(np.real(Z_b[0])) if Z_b[0] is not None else 0.0
        h1_a = abs(Z_a[1]) if Z_a[1] is not None else 0.0
        h1_b = abs(Z_b[1]) if Z_b[1] is not None else 0.0
        ax.plot(t_a, q_a, '-',  color='steelblue', lw=1.8,
                 label=f'{src_a}  (Q̄={dc_a:.3f}, |H1|={h1_a:.3f})')
        ax.plot(t_b, q_b, '--', color='crimson',   lw=1.8,
                 label=f'{src_b}  (Q̄={dc_b:.3f}, |H1|={h1_b:.3f})')
        ax.axhline(dc_a, color='steelblue', ls=':', lw=0.7, alpha=0.5)
        ax.axhline(dc_b, color='crimson',   ls=':', lw=0.7, alpha=0.5)
        ax.set_xlabel('cycles'); ax.set_ylabel('Q (per-source native units)')
        ax.set_title(f'Edge ({u}, {v})   {src_a}  vs  {src_b}   '
                      f'|H1|_ratio = {h1_b/max(h1_a,1e-30):.2f}×')
        ax.legend(loc='best'); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.show()

    def _plot_selected_qt(self):
        if self._selected_edge is None:
            return
        # Source-aware:
        #   'sim'        → plot the reconstructed sim waveform
        #   'normalized' → open the side-by-side raw vs normalized 4-panel
        #                  view (was the "Compare raw vs normalized" button)
        #   'measured'   → the standard 3-row Q(t) + harmonic decomp + FFT
        src = getattr(self, 'current_source', 'measured')
        if src == 'sim':
            self._plot_selected_sim_qt()
            return
        if src == 'normalized':
            self._plot_raw_vs_norm_qt()
            return
        u, v = self._selected_edge
        meas = self.G.edges[u, v].get('measurements_piv') or []
        if not meas:
            return
        # Rank by quality so colour order is deterministic.
        meas_ranked = sorted(
            meas,
            # v4 migration: mm.get('snr_f0') → MeasurementView(mm).snr.h1_db.
            # quality_tier and snr_pulse are not in the mapping table; keep.
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            (MeasurementView(mm).snr.h1_db or 0),
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        # Focus-tile rule: prefer the View-tab tile filter if set
        # (matches what's currently coloured on the network), else the
        # loaded video tile, else fall back to all measurements.
        filter_note = None
        focus_tid = (self.current_tile_filter
                      if self.current_tile_filter is not None
                      else self._video_tile_id)
        if focus_tid is not None:
            filtered = [m for m in meas_ranked
                        if MeasurementView(m).tile_id == focus_tid]
            source = ('tile filter'
                       if self.current_tile_filter is not None
                       else 'currently loaded video')
            if filtered:
                meas_ranked = filtered
                filter_note = (f"showing only tile {focus_tid} "
                                f"({source})")
            else:
                filter_note = (f"this edge has no measurement from "
                                f"tile {focus_tid} ({source}) — "
                                f"showing all instead")

        # Three-row figure:
        #   top = raw Q(t) for every measurement
        #   mid = harmonic decomposition of the best measurement
        #   bot = magnitude spectrum |FFT(Q_t)| (best measurement)
        # The top two share a time axis; the bottom has its own freq axis.
        # 75% of the previous (10×10) inch footprint per user request.
        fig = plt.figure(figsize=(7.5, 7.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85],
                              hspace=0.35)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_spec = fig.add_subplot(gs[2, 0])

        # ── Top: raw Q(t) traces for every measurement ──
        for m in meas_ranked:
            mv_m = MeasurementView(m)
            # v4 migration: was m.get('Q_t')
            Q_t = mv_m.Q_t
            if Q_t is None:
                continue
            Q_t = np.asarray(Q_t, dtype=float)
            # Q is a flow-rate magnitude → display with a positive mean (flip the
            # whole trace if its mean is negative; stored Q_t is already
            # convention-oriented, so this is normally a no-op).
            if np.isfinite(np.nanmean(Q_t)) and np.nanmean(Q_t) < 0:
                Q_t = -Q_t
            t = np.arange(len(Q_t)) / 250.0  # 250 fps
            usable = _measurement_usable(m)
            mark = '✓' if usable else '✗'
            # v4 migration: tile_id / mean_Q via MeasurementView.
            # 'PI' (H1-only Gosling) has no MeasurementView property; use
            # PI_rms (Parseval, all-harmonic) as an approximate substitute.
            # TODO(v4-migration): PI legacy flat key dropped in v4.
            _mq = mv_m.harm.mean_Q
            _mq_disp = float(_mq) if _mq is not None else float('nan')
            _pi_disp = mv_m.PI_rms
            if _pi_disp is None:
                _pi_disp = m.get('PI', float('nan'))
            ax_top.plot(
                t, Q_t,
                lw=0.8 if usable else 0.6,
                ls='-' if usable else '--',
                alpha=0.85 if usable else 0.5,
                label=(f"{mark} tile {mv_m.tile_id}  "
                       f"mean_Q={_mq_disp:.3f}  "
                       f"PI={float(_pi_disp):.2f}"))
        ax_top.set_ylabel('Q (nL/s)')
        title_top = (f'Edge ({u}, {v}) — raw Q(t) across '
                     f'{len(meas_ranked)} of {len(meas)} measurements')
        if filter_note:
            title_top += f'\n({filter_note})'
        ax_top.set_title(title_top)
        ax_top.legend(fontsize=8, loc='best')
        ax_top.grid(alpha=0.3)

        # ── Bottom: harmonic decomposition of the best measurement ──
        # Refit harmonics on the raw Q_t so we get H1 *and* H2 even
        # though the measurement only stores H1 fields directly.
        best = next((m for m in meas_ranked
                     if _measurement_usable(m)
                     and MeasurementView(m).Q_t is not None),
                    None)
        if best is not None:
            from ..analyze.harmonic import fit_harmonics
            best_mv = MeasurementView(best)
            # v4 migration: was best['Q_t'] / best.get('f0_hz', 2.5)
            Q_t = np.asarray(best_mv.Q_t, dtype=float)
            T = len(Q_t)
            t = np.arange(T) / 250.0
            f0 = float(best_mv.harm.f0_hz or 2.5)
            try:
                hr = fit_harmonics(Q_t, frame_dt=1.0/250, f0=f0,
                                    K=3, loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                omega = 2 * np.pi * f0
                # Cumulative reconstructions
                y_dc = np.full(T, a0)
                h1 = harms.get(1)
                y_h1 = (y_dc + h1['A'] * np.cos(omega * t)
                              + h1['B'] * np.sin(omega * t)) if h1 else y_dc
                h2 = harms.get(2)
                y_h2 = (y_h1
                        + h2['A'] * np.cos(2 * omega * t)
                        + h2['B'] * np.sin(2 * omega * t)) if h2 else y_h1
                h3 = harms.get(3)
                y_h3 = (y_h2
                        + h3['A'] * np.cos(3 * omega * t)
                        + h3['B'] * np.sin(3 * omega * t)) if h3 else y_h2
                # Relative phase: phi_k − k·phi_1 (wrapped to [-π, π]).
                # This is what determines waveform shape — when Δφ_rel=0
                # higher harmonics reinforce H1's peak; when ±π they
                # cancel at the peak.
                def _wrap(p):
                    return (float(p) + np.pi) % (2 * np.pi) - np.pi
                # Per-harmonic Z-statistics from the same fit's residuals.
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                N = len(Q_t)
                se_dc = sigma / max(np.sqrt(N), 1e-30)
                se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else float('inf')
                z_dc = abs(a0) / max(se_dc, 1e-30)
                z1 = (h1['amp'] / max(se_amp, 1e-30)) if h1 else 0.0
                z2 = (h2['amp'] / max(se_amp, 1e-30)) if h2 else 0.0
                z3 = (h3['amp'] / max(se_amp, 1e-30)) if h3 else 0.0
                snrs = {'DC': z_dc, 'H1': z1, 'H2': z2, 'H3': z3}
                klass = _harmonic_class(snrs)
                # Raw Q(t) shown faintly for noise context; a Butterworth
                # low-pass filtered version (cutoff = 4·f0, just above
                # H3) is overlaid darker so you can see the underlying
                # cardiac shape without the high-frequency junk.
                ax_mid.plot(t, Q_t, color='#bbb', lw=0.5, alpha=0.45,
                            label='raw Q(t)')
                try:
                    from scipy.signal import butter, filtfilt
                    nyq = 0.5 * 250.0   # 125 Hz at 250 fps (was 0.5 / dt with dt undefined: overlay never drawn)
                    cutoff = min(4.0 * f0, 0.9 * nyq)
                    b, a_lp = butter(N=4, Wn=cutoff / nyq, btype='low')
                    Q_lp = filtfilt(b, a_lp, Q_t)
                    ax_mid.plot(t, Q_lp, color='#444', lw=0.9, alpha=0.85,
                                label=f'Q(t) low-passed @ {cutoff:.1f} Hz')
                except Exception:
                    pass
                ax_mid.plot(t, y_dc, color='C2', lw=1.2, ls=':',
                            label=f'DC = {a0:.3f}  (Z={z_dc:.1f})')
                if h1 is not None:
                    ax_mid.plot(t, y_h1, color='C0', lw=1.5,
                                label=f"+ H1 (amp={h1['amp']:.3f}, "
                                      f"PI={2*h1['amp']/max(abs(a0),1e-12):.2f}, "
                                      f"φ={h1['phi']:+.2f}, Z={z1:.1f})")
                if h2 is not None:
                    dphi2 = _wrap(h2['phi'] - 2 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h2, color='C3', lw=1.4, ls='--',
                                label=f"+ H2 (amp={h2['amp']:.3f}, "
                                      f"H2/H1={h2['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi2:+.2f}, Z={z2:.1f})")
                if h3 is not None:
                    dphi3 = _wrap(h3['phi'] - 3 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h3, color='C1', lw=1.4, ls='-.',
                                label=f"+ H3 (amp={h3['amp']:.3f}, "
                                      f"H3/H1={h3['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi3:+.2f}, Z={z3:.1f})")
                # SNR_AC = 10·log10(Z1² + Z2² + Z3²) — total cardiac-harmonic
                # power SNR, DC-free, on the same dB scale as Q_Hk_snr_db.
                _z_sq = sum(z*z for z in (z1, z2, z3) if np.isfinite(z))
                snr_ac_db = (10.0 * np.log10(_z_sq)
                              if _z_sq > 0 else float('nan'))
                ax_mid.set_title(
                    # v4 migration: was best.get('tile_id')
                    f"Harmonic decomposition — tile {best_mv.tile_id}  "
                    f"(f₀ = {f0:.2f} Hz,  SNR_AC = {snr_ac_db:.1f} dB)")
            except Exception as e:
                ax_mid.text(0.5, 0.5, f'fit_harmonics failed: {e}',
                            ha='center', va='center',
                            transform=ax_mid.transAxes)

            # ── Spectrum panel ──
            # FFT magnitude of the (de-meaned) Q_t to reveal what
            # frequency content is in the signal beyond the fitted
            # harmonics — camera shake, electrical noise, sub-harmonics.
            dt = 1.0 / 250
            Q_dm = Q_t - np.mean(Q_t)
            freqs = np.fft.rfftfreq(T, d=dt)
            mag = (2.0 / T) * np.abs(np.fft.rfft(Q_dm))
            ax_spec.semilogy(freqs, np.maximum(mag, 1e-6),
                              color='#444', lw=0.7)
            # Mark the fitted-harmonic frequencies
            for k in (1, 2, 3):
                ax_spec.axvline(k * f0, color='C0' if k == 1
                                          else 'C3' if k == 2 else 'C1',
                                ls='--', lw=0.9, alpha=0.7,
                                label=f'k·f₀ = {k*f0:.2f} Hz' if k <= 3 else None)
            # Cut off display just past the 4th harmonic — everything
            # higher is noise / aliasing for cardiac-rate signals.
            ax_spec.set_xlim(0, min(4.5 * f0, freqs[-1]))
            ax_spec.set_xlabel('frequency (Hz)')
            ax_spec.set_ylabel('|FFT(Q)|  (nL/s)')
            ax_spec.set_title('Magnitude spectrum  '
                              '(content outside the H1-H3 dashed lines = noise / aliasing)')
            ax_spec.legend(fontsize=8, loc='upper right')
            ax_spec.grid(alpha=0.3, which='both')
        else:
            ax_mid.text(0.5, 0.5,
                        'No usable measurement with Q_t — cannot decompose.',
                        ha='center', va='center',
                        transform=ax_mid.transAxes)
            ax_spec.text(0.5, 0.5, 'No spectrum to show.',
                          ha='center', va='center',
                          transform=ax_spec.transAxes)
        ax_mid.set_xlabel('time (s)'); ax_mid.set_ylabel('Q (nL/s)')
        ax_mid.legend(fontsize=8, loc='best')
        ax_mid.grid(alpha=0.3)
        # Show first (non-blocking) so the window is mapped, THEN
        # resize + move.  Doing it the other way around lets macOS
        # apply its "restore previous window state" rule, which is
        # what was re-maximizing the window despite figsize=(7.5, 7.5).
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

    def _plot_raw_vs_norm_qt(self):
        """4-panel comparison of raw vs normalized Q(t) for the selected edge.

        Reads both `measurements_piv` and `measurements_piv_normalized`
        off the same graph — populated by the Normalize tab (which calls
        `pertile.mosaic.tile_normalize.apply_gauges_to_graph`).
        """
        if self._selected_edge is None:
            return
        if not self._normalization_available():
            print('[compare] No normalization has been computed on this graph. '
                  'Use the Normalize tab first.')
            return
        u, v = self._selected_edge
        ms_raw = self.G.edges[u, v].get('measurements_piv') or []
        ms_norm = self.G.edges[u, v].get('measurements_piv_normalized') or []
        # Match by tile_id — build a dict per side
        raw_by_tile = {}
        for m in ms_raw:
            tid = m.get('tile_id')
            if tid is None: continue
            mv_m = MeasurementView(m)
            Q_t = mv_m.Q_t
            if Q_t is not None:
                raw_by_tile.setdefault(int(tid), []).append(m)
        norm_by_tile = {}
        for m in ms_norm:
            tid = m.get('tile_id')
            if tid is None: continue
            mv_m = MeasurementView(m)
            Q_t = mv_m.Q_t
            if Q_t is not None:
                norm_by_tile.setdefault(int(tid), []).append(m)
        common_tiles = sorted(set(raw_by_tile) & set(norm_by_tile))
        if not common_tiles:
            print(f'[compare] no common tile measurements for edge ({u},{v})')
            return

        import matplotlib.pyplot as plt
        tile_gauges = (self.G.graph.get('tile_gauges') or {})
        f0_can = float(self.G.graph.get('f0_canonical_hz', 2.838))

        # Colour map: assign one colour per tile so raw + norm curves match.
        colours = plt.cm.tab10.colors + plt.cm.tab20.colors
        colour_of = {t: colours[i % len(colours)] for i, t in enumerate(common_tiles)}
        # 2 x 3 layout — third column split into:
        #   top    = |H_k| bar chart (raw vs normalized amplitudes per tile)
        #   bottom = polar-style phasor plot (normalized H_k per tile + mean)
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.8])
        axs = np.array([
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ])
        ax_mag = fig.add_subplot(gs[0, 2])
        ax_phasor = fig.add_subplot(gs[1, 2])
        def _harm_fit(m, f0_use, T_samples):
            """Reconstruct Q_fit(t) = DC + Σ_k amp_k cos(2π k f0 t + φ_k)
            from the stored (amp, phi) on this measurement.  Returns
            (t_array, Q_fit) or (None, None) if data is missing."""
            harm_d = m.get('harm') or {}
            dc = harm_d.get('mean_Q')
            if dc is None or not np.isfinite(dc):
                return None, None
            t_arr = np.arange(T_samples) / 250.0
            y = np.full(T_samples, float(dc), dtype=float)
            for h in (harm_d.get('harmonics') or []):
                a = h.get('amp'); p = h.get('phi'); k = h.get('k')
                if (a is None or p is None or k is None
                    or not np.isfinite(a) or not np.isfinite(p)):
                    continue
                y = y + float(a) * np.cos(
                    2.0 * np.pi * float(k) * f0_use * t_arr + float(p))
            return t_arr, y

        # (0,0): raw Q(t) per tile + per-tile harmonic fits (dashed)
        ax_raw = axs[0, 0]
        for tid in common_tiles:
            m = raw_by_tile[tid][0]
            Q_t = np.asarray(MeasurementView(m).Q_t, dtype=float)
            t = np.arange(len(Q_t)) / 250.0
            ax_raw.plot(t, Q_t, lw=0.7, color=colour_of[tid], alpha=0.45,
                         label=f'tile {tid}')
            # Fit uses this tile's own f0 (native).
            f0_t = float((m.get('harm') or {}).get('f0_hz') or 2.5)
            _t_fit, Q_fit = _harm_fit(m, f0_t, len(Q_t))
            if Q_fit is not None:
                ax_raw.plot(_t_fit, Q_fit, lw=1.6, ls='--',
                             color=colour_of[tid], alpha=0.95,
                             label=f'tile {tid} fit (f0={f0_t:.2f}Hz)')
        ax_raw.set_xlabel('t (s)'); ax_raw.set_ylabel('Q (nL/s)')
        ax_raw.set_title(f'RAW Q(t)  edge ({u},{v})  —  {len(common_tiles)} tiles\n'
                          '(faint = raw, dashed = harmonic fit at tile-native f0)')
        ax_raw.grid(alpha=0.3); ax_raw.legend(fontsize=7, loc='best', ncol=2)

        # (0,1): normalized Q(t) per tile + per-tile fits + SNR-weighted mean
        ax_norm = axs[0, 1]
        for tid in common_tiles:
            m = norm_by_tile[tid][0]
            Q_t = np.asarray(MeasurementView(m).Q_t, dtype=float)
            t = np.arange(len(Q_t)) / 250.0
            ax_norm.plot(t, Q_t, lw=0.7, color=colour_of[tid], alpha=0.4,
                          label=f'tile {tid}')
            # Normalized harmonics live at the canonical f0.
            _t_fit, Q_fit = _harm_fit(m, f0_can, len(Q_t))
            if Q_fit is not None:
                ax_norm.plot(_t_fit, Q_fit, lw=1.4, ls='--',
                              color=colour_of[tid], alpha=0.9,
                              label=f'tile {tid} fit')
        # Reconstruct the mean-phasor waveform (what the coloring uses
        # under Normalize / All-tiles).  Read the cache written by
        # tile_normalize.apply_gauges_to_graph.
        ed = self.G.edges[u, v]
        mean_ph = ed.get('_h_norm_mean_phasors') or {}
        if mean_ph:
            n_meas = int(ed.get('_h_norm_mean_n', 0))
            # Reconstruct on the same time base as the raw traces.
            first_tile = common_tiles[0]
            T_ref = len(np.asarray(MeasurementView(norm_by_tile[first_tile][0]).Q_t))
            t = np.arange(T_ref) / 250.0
            # amp stored on `harmonics[k]['amp']` is the physical
            # cosine amplitude sqrt(A² + B²) (see pertile.analysis.
            # harmonic.fit_harmonics), so reconstruction is
            #    Q(t) = DC + Σ |H_k| cos(k ω t + arg H_k)
            # — no factor of 2.
            dc = complex(mean_ph.get('DC', complex('nan')))
            y = np.full(T_ref, float(dc.real) if np.isfinite(dc.real) else 0.0)
            for k in (1, 2, 3):
                H = complex(mean_ph.get(f'H{k}', complex('nan')))
                if np.isfinite(H.real) and np.isfinite(H.imag):
                    y = y + abs(H) * np.cos(
                        k * 2.0 * np.pi * f0_can * t + np.angle(H))
            ax_norm.plot(t, y, color='#7a1fa2', lw=2.4, alpha=0.75,
                          label=f'SNR-wtd mean phasor  (n={n_meas})')
        # ── Consistency check: reconstruct from the scalar cache used by
        # BOTH the network colormap AND my overlay code — should overlay
        # exactly on the purple phasor-mean line above. If it doesn't,
        # the scalar cache diverges from the mean-phasor cache, and one
        # or both are being read wrong downstream.
        _a_dc = ed.get('_h_amp_DC_norm')
        if _a_dc is not None and np.isfinite(_a_dc):
            first_tile = common_tiles[0]
            T_ref = len(np.asarray(MeasurementView(norm_by_tile[first_tile][0]).Q_t))
            t_sc = np.arange(T_ref) / 250.0
            y_sc = np.full(T_ref, float(_a_dc))
            for k in (1, 2, 3):
                a = ed.get(f'_h_amp_H{k}_norm'); p = ed.get(f'_h_phase_H{k}_norm')
                if a is None or p is None: continue
                if not (np.isfinite(a) and np.isfinite(p)): continue
                y_sc = y_sc + float(a) * np.cos(
                    k * 2.0 * np.pi * f0_can * t_sc + float(p))
            ax_norm.plot(t_sc, y_sc, color='#1a9850', lw=1.5, ls=':',
                          label=r'scalar cache: _h_amp_H{k}_norm + _h_phase_H{k}_norm',
                          alpha=0.95)
        ax_norm.axhline(0, color='0.7', lw=0.5, alpha=0.5)
        ax_norm.set_xlabel('t (s)'); ax_norm.set_ylabel('Q_norm (nL/s)')
        ax_norm.set_title(
            f'NORMALIZED Q(t)  (canonical f0={f0_can:.2f} Hz)\n'
            'NOTE: faint tile traces = raw normalized samples\n'
            '(may sit ≠ DC-shifted); dashed = per-tile fit at f0_can;\n'
            'purple = SNR-wtd mean phasor;  '
            'green dotted = scalar-cache reconstruction (must overlap purple).')
        ax_norm.grid(alpha=0.3); ax_norm.legend(fontsize=7, loc='best')

        # (1,0): per-tile scalars table (bar plot). H1 gauge in solid,
        # H2 gauge as thinner overlay so you can see if H1 and H2 agree
        # or drifted independently under the two spectral_sync passes.
        ax_gauge = axs[1, 0]
        xs = np.arange(len(common_tiles))
        z_mags   = [tile_gauges.get(t, {}).get('z_mag', np.nan) for t in common_tiles]
        z_H2_mags = [tile_gauges.get(t, {}).get('z_H2_mag', np.nan) for t in common_tiles]
        co_rels  = [tile_gauges.get(t, {}).get('CO_rel', np.nan) for t in common_tiles]
        f0s      = [tile_gauges.get(t, {}).get('f0_hz', np.nan) for t in common_tiles]
        ax_gauge.axhline(1, color='gray', lw=0.4)
        w = 0.22
        ax_gauge.bar(xs - 1.5*w, z_mags, w, label='|z_t| (H1)',
                      color=[colour_of[t] for t in common_tiles], alpha=0.6,
                      edgecolor='black')
        ax_gauge.bar(xs - 0.5*w, z_H2_mags, w, label='|z_t| (H2)',
                      color=[colour_of[t] for t in common_tiles], alpha=0.4,
                      edgecolor='black', hatch='xx')
        ax_gauge.bar(xs + 0.5*w, co_rels, w, label='CO_rel (DC)',
                      color=[colour_of[t] for t in common_tiles], alpha=0.9,
                      edgecolor='black', hatch='//')
        ax_gauge.bar(xs + 1.5*w, np.array(f0s) / f0_can, w, label='f0/f0_can',
                      color=[colour_of[t] for t in common_tiles], alpha=0.3,
                      edgecolor='black', hatch='..')
        ax_gauge.set_xticks(xs)
        ax_gauge.set_xticklabels([str(t) for t in common_tiles])
        ax_gauge.set_xlabel('tile'); ax_gauge.set_ylabel('gauge value (× reference)')
        ax_gauge.set_title(
            'Per-tile gauges — magnitude: H1 and H2 use their own |z_t|; '
            'PHASE: only ψ₁ is applied (2·ψ₁ for H2, 3·ψ₁ for H3 — single '
            'time-shift per tile)\n'
            'ψ₂_meas is shown for diagnostic only: how well H2\'s independent '
            'sync agrees with the applied 2·ψ₁ (small |ψ₂_meas − 2·ψ₁| = '
            'sync is self-consistent)')
        ax_gauge.legend(fontsize=7, ncol=2); ax_gauge.grid(axis='y', alpha=0.3)
        def _wrap180(x):
            return ((x + 180) % 360) - 180
        for i, tid in enumerate(common_tiles):
            psi_h1 = tile_gauges.get(tid, {}).get('z_arg_deg', np.nan)
            psi_h2_meas = tile_gauges.get(tid, {}).get('z_H2_arg_deg', np.nan)
            top_y = max([v for v in [z_mags[i], z_H2_mags[i], co_rels[i],
                                       (f0s[i] or f0_can)/f0_can]
                         if v is not None and np.isfinite(v)] or [1.0])
            lbl_parts = []
            if np.isfinite(psi_h1):
                # ψ₁ applied to H1
                lbl_parts.append(f'ψ₁={psi_h1:+.0f}°')
                # 2·ψ₁ applied to H2 (single time-shift, linear in k)
                lbl_parts.append(f'2ψ₁={_wrap180(2*psi_h1):+.0f}°')
            if np.isfinite(psi_h2_meas):
                # ψ₂_meas: what H2's independent sync computed; NOT applied
                lbl_parts.append(f'ψ₂ᵐ={psi_h2_meas:+.0f}°')
                # Δ = ψ₂_meas − 2·ψ₁: self-consistency check
                if np.isfinite(psi_h1):
                    lbl_parts.append(f'Δ={_wrap180(psi_h2_meas - 2*psi_h1):+.0f}°')
            if lbl_parts:
                ax_gauge.text(i, top_y + 0.05, '\n'.join(lbl_parts),
                               fontsize=6.5, ha='center')

        # (1,1): CYCLE-FOLDED overlay — collapse all cycles onto ONE canonical
        # period.  If normalization worked, tiles should overlay tightly on
        # the single-cycle window.
        ax_ov = axs[1, 1]
        cycle_samples = int(round(250 / f0_can))   # samples per canonical cycle
        phase_axis = np.arange(cycle_samples) / cycle_samples   # 0..1 within a cycle
        for tid in common_tiles:
            Q_t = np.asarray(MeasurementView(norm_by_tile[tid][0]).Q_t, dtype=float)
            n_cycles = len(Q_t) // cycle_samples
            if n_cycles < 1: continue
            folded = Q_t[:n_cycles * cycle_samples].reshape(n_cycles, cycle_samples)
            # Faint traces of individual cycles + bold mean
            for cyc in folded:
                ax_ov.plot(phase_axis, cyc, color=colour_of[tid], lw=0.4, alpha=0.25)
            mean_cycle = folded.mean(axis=0)
            ax_ov.plot(phase_axis, mean_cycle, color=colour_of[tid], lw=2.5,
                        alpha=0.95, label=f'tile {tid}  (mean of {n_cycles} cycles)')
        # SNR-weighted mean phasor reconstruction on the same phase axis.
        mean_ph_ov = self.G.edges[u, v].get('_h_norm_mean_phasors') or {}
        if mean_ph_ov:
            t_cycle = phase_axis / f0_can   # seconds within one cycle
            dc = complex(mean_ph_ov.get('DC', complex('nan')))
            y = np.full(cycle_samples,
                         float(dc.real) if np.isfinite(dc.real) else 0.0)
            for k in (1, 2, 3):
                H = complex(mean_ph_ov.get(f'H{k}', complex('nan')))
                if np.isfinite(H.real) and np.isfinite(H.imag):
                    y = y + abs(H) * np.cos(
                        k * 2.0 * np.pi * f0_can * t_cycle + np.angle(H))
            n_meas = int(self.G.edges[u, v].get('_h_norm_mean_n', 0))
            ax_ov.plot(phase_axis, y, color='#7a1fa2', lw=2.5, alpha=0.8,
                        label=f'SNR-wtd mean phasor  (n={n_meas})')
        ax_ov.set_xlabel('cardiac phase  (fraction of one cycle at canonical f0)')
        ax_ov.set_ylabel('Q_norm (nL/s)')
        ax_ov.set_title('Cycle-folded overlay  (bold = mean cycle per tile)')
        ax_ov.grid(alpha=0.3); ax_ov.legend(fontsize=8, loc='best')

        # ── Phasor panel (right column, full height) ─────────────────────
        # Plots each tile's NORMALIZED H_k phasor in complex plane, with
        # H1 (circle), H2 (square), H3 (triangle).  Also draws the
        # SNR-weighted mean phasor (thick black marker + arrow from origin).
        # If normalization aligned the tiles perfectly, all tiles should
        # cluster on a single point per harmonic and the mean should sit
        # exactly there.  Phase mismatch shows up as angular spread; the
        # mean sits at the centroid of the tile points (weighted by SNR).
        harm_markers = {1: 'o', 2: 's', 3: '^'}
        max_r = 0.0

        def _phasors_at(mm):
            """Return {k: complex H_k} from a measurement dict."""
            out = {}
            hs = ((mm.get('harm') or {}).get('harmonics') or [])
            for k in (1, 2, 3):
                ix = k - 1
                if ix >= len(hs) or hs[ix] is None:
                    continue
                a = hs[ix].get('amp'); p = hs[ix].get('phi')
                if a is None or p is None or not np.isfinite(a) or not np.isfinite(p):
                    continue
                out[k] = float(a) * np.exp(1j * float(p))
            return out

        # Pull Z/2 sign-map (which tiles got flipped by per-edge sign
        # resolution before the aggregate mean was computed). Marked with
        # an extra X hatch + a "×(-1)" annotation next to the marker.
        _mp_signs = self.G.edges[u, v].get('_h_norm_mean_phasors') or {}
        signs_by_k = {k: (_mp_signs.get(f'H{k}_signs') or {}) for k in (1, 2, 3)}
        for tid in common_tiles:
            H_norm = _phasors_at(norm_by_tile[tid][0])
            for k in (1, 2, 3):
                if k not in H_norm:
                    continue
                H = H_norm[k]
                max_r = max(max_r, abs(H))
                sign = int(signs_by_k.get(k, {}).get(int(tid), 1))
                ax_phasor.plot([0, H.real], [0, H.imag],
                                color=colour_of[tid], lw=0.6, alpha=0.4)
                # Base marker at the RAW per-tile phasor location
                ax_phasor.plot(H.real, H.imag,
                                marker=harm_markers[k], ms=9,
                                markerfacecolor=colour_of[tid],
                                markeredgecolor='black' if sign > 0 else 'red',
                                mew=(0.5 if sign > 0 else 1.5),
                                linestyle='None')
                # If flipped, also draw the ALIGNED (flipped) phasor as an
                # open marker at -H, and a thin dashed arrow connecting
                # them so it's obvious which raw phasor was inverted.
                if sign < 0:
                    ax_phasor.plot([0, -H.real], [0, -H.imag],
                                    color=colour_of[tid], lw=0.6, ls=':', alpha=0.4)
                    ax_phasor.plot(-H.real, -H.imag,
                                    marker=harm_markers[k], ms=9,
                                    markerfacecolor='none',
                                    markeredgecolor=colour_of[tid],
                                    mew=1.5, linestyle='None')
                    ax_phasor.annotate('', xy=(-H.real, -H.imag),
                                        xytext=(H.real, H.imag),
                                        arrowprops=dict(
                                            arrowstyle='->',
                                            color='red', alpha=0.5,
                                            lw=0.8, ls='dashed'))
        # SNR-weighted mean phasors from the cache
        mean_ph_pan = self.G.edges[u, v].get('_h_norm_mean_phasors') or {}
        for k in (1, 2, 3):
            H_mean = complex(mean_ph_pan.get(f'H{k}', complex('nan')))
            if not (np.isfinite(H_mean.real) and np.isfinite(H_mean.imag)):
                continue
            max_r = max(max_r, abs(H_mean))
            ax_phasor.plot([0, H_mean.real], [0, H_mean.imag],
                            color='#7a1fa2', lw=2.0, alpha=0.85)
            ax_phasor.plot(H_mean.real, H_mean.imag,
                            marker=harm_markers[k], ms=14,
                            markerfacecolor='#7a1fa2', markeredgecolor='white',
                            mew=1.2, linestyle='None')
        # Axes cosmetics
        R = max(max_r * 1.15, 1e-6)
        ax_phasor.set_xlim(-R, R); ax_phasor.set_ylim(-R, R)
        ax_phasor.set_aspect('equal', adjustable='box')
        ax_phasor.axhline(0, color='gray', lw=0.4)
        ax_phasor.axvline(0, color='gray', lw=0.4)
        # Unit-magnitude reference ring at R/2 and R for phase reading
        theta = np.linspace(0, 2*np.pi, 200)
        for r in (R/2, R):
            ax_phasor.plot(r * np.cos(theta), r * np.sin(theta),
                            color='gray', lw=0.3, alpha=0.5)
        ax_phasor.set_xlabel('Re(H)  [nL/s]')
        ax_phasor.set_ylabel('Im(H)  [nL/s]')
        ax_phasor.set_title('Normalized phasors  (H1=○, H2=□, H3=△)\n'
                              'purple = SNR-weighted mean')
        ax_phasor.grid(alpha=0.25)
        # Legend: one entry per tile (H1 marker) + a mean entry
        from matplotlib.lines import Line2D
        legend_handles = []
        for tid in common_tiles:
            legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colour_of[tid],
                                          markeredgecolor='black', mew=0.5,
                                          markersize=9, label=f'tile {tid}'))
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#7a1fa2',
                                      markeredgecolor='white', mew=1.2,
                                      markersize=12, label='SNR-wtd mean'))
        ax_phasor.legend(handles=legend_handles, fontsize=8, loc='best')

        # ── Magnitude bar chart (top-right) ─────────────────────────────
        # Groups: DC, H1, H2, H3.  Within each group: pairs of bars for
        # each tile: (raw, normalized).  Overplot the SNR-weighted mean
        # normalized magnitude as a dashed reference line per harmonic.
        def _amps(mm):
            """Return {DC, H1, H2, H3} magnitudes from a measurement."""
            out = {'DC': float('nan'), 'H1': float('nan'),
                    'H2': float('nan'), 'H3': float('nan')}
            hh = mm.get('harm') or {}
            dc = hh.get('mean_Q')
            if dc is not None and np.isfinite(dc):
                out['DC'] = abs(float(dc))
            for h in (hh.get('harmonics') or []):
                k = int(h.get('k', 0))
                a = h.get('amp')
                if a is not None and np.isfinite(a) and 1 <= k <= 3:
                    out[f'H{k}'] = abs(float(a))
            return out

        keys = ['DC', 'H1', 'H2', 'H3']
        x_base = np.arange(len(keys))
        n_tiles = len(common_tiles)
        # 2 bars per tile per group, tucked into unit-wide slot
        slot_w = 0.8 / max(1, 2 * n_tiles)
        for i, tid in enumerate(common_tiles):
            raw_amps = _amps(raw_by_tile[tid][0])
            norm_amps = _amps(norm_by_tile[tid][0])
            offset_r = -0.4 + (2 * i) * slot_w
            offset_n = offset_r + slot_w
            ax_mag.bar(x_base + offset_r,
                        [raw_amps[k] for k in keys], slot_w,
                        color=colour_of[tid], alpha=0.35,
                        edgecolor=colour_of[tid], linewidth=0.6)
            ax_mag.bar(x_base + offset_n,
                        [norm_amps[k] for k in keys], slot_w,
                        color=colour_of[tid], alpha=0.85,
                        edgecolor='black', linewidth=0.4)
        # Style-only proxy entries in the legend (tile colours are
        # already visible in the other panels).
        from matplotlib.patches import Patch
        ax_mag.add_artist(ax_mag.legend(
            handles=[Patch(facecolor='gray', alpha=0.35, edgecolor='gray',
                            label='raw'),
                      Patch(facecolor='gray', alpha=0.85, edgecolor='black',
                            label='normalized')],
            fontsize=7, loc='upper right'))
        # SNR-weighted mean amplitudes as horizontal reference marks
        mean_ph_bar = self.G.edges[u, v].get('_h_norm_mean_phasors') or {}
        if mean_ph_bar:
            mean_amps = [abs(complex(mean_ph_bar.get('DC', complex('nan'))))]
            for k in (1, 2, 3):
                mean_amps.append(abs(complex(mean_ph_bar.get(f'H{k}', complex('nan')))))
            for xi, ma in zip(x_base, mean_amps):
                if np.isfinite(ma):
                    ax_mag.hlines(ma, xi - 0.42, xi + 0.42,
                                    colors='#7a1fa2', lw=2.2, alpha=0.9,
                                    zorder=5)
        ax_mag.set_xticks(x_base); ax_mag.set_xticklabels(keys)
        ax_mag.set_ylabel('|amplitude|  [nL/s]')
        ax_mag.set_title('Magnitudes: raw (faint) vs normalized (bold)\n'
                          'purple bar = SNR-wtd mean normalized')
        ax_mag.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Raw vs normalized  edge ({u},{v})   '
                      f'[{len(common_tiles)} tiles]', fontsize=12)
        fig.tight_layout()
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

    def _plot_selected_sim_qt(self):
        """Single-edge plot of the SIMULATED waveform (View tab Source =
        Simulated).  Mirrors the measured `_plot_selected_qt` layout but
        reconstructs f(t) from the per-edge complex solver harmonics; plots
        flow Q(t) or midpoint pressure P(t) per the Quantity selector."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        quantity = 'P' if self.current_quantity == 'P' else 'Q'
        m = self._build_sim_measurement(u, v, quantity=quantity)
        if m is None:
            if getattr(self, '_info_box', None) is not None:
                self._info_box.append(
                    "\n[sim] No simulation data on this edge — run the "
                    "Simulate or Pump tab first.")
            return
        harm_key = ('_sim_tmp_p_harmonics' if quantity == 'P'
                    else '_sim_tmp_harmonics')
        arr = np.asarray(self.G.edges[u, v].get(harm_key), dtype=complex)
        unit = 'Pa' if quantity == 'P' else 'nL/s'
        sym = quantity
        sig = np.asarray(m['Q_t'], dtype=float)   # carries P(t) in P-mode
        f0 = float(m['f0_hz'])
        dt = 1.0 / 250.0
        T = len(sig)
        t = np.arange(T) * dt
        omega = 2 * np.pi * f0

        fig = plt.figure(figsize=(7.5, 7.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85],
                              hspace=0.35)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_spec = fig.add_subplot(gs[2, 0])

        # ── Top: reconstructed waveform ──
        ax_top.plot(t, sig, color='C0', lw=1.2,
                    label=(f"sim {sym}(t)   mean={m['mean_Q']:.3g}   "
                           f"PI={m.get('PI', float('nan')):.2f}"))
        ax_top.set_ylabel(f'{sym} ({unit})')
        ax_top.set_title(f'Edge ({u}, {v}) — simulated {sym}(t)   '
                         f'(f₀ = {f0:.2f} Hz)')
        ax_top.legend(fontsize=8, loc='best')
        ax_top.grid(alpha=0.3)

        # ── Mid: exact harmonic decomposition from solver coefficients ──
        def _wrap(p):
            return (float(p) + np.pi) % (2 * np.pi) - np.pi
        colors = {1: 'C0', 2: 'C3', 3: 'C1'}
        styles = {1: '-', 2: '--', 3: '-.'}
        c0 = float(arr[0].real) if len(arr) else 0.0
        amp1 = float(abs(arr[1])) if len(arr) > 1 else 0.0
        phi1 = float(np.angle(arr[1])) if len(arr) > 1 else 0.0
        ax_mid.plot(t, sig, color='#bbb', lw=0.6, alpha=0.5,
                    label='full reconstruction')
        y = np.full(T, c0)
        ax_mid.plot(t, y, color='C2', lw=1.2, ls=':', label=f'DC = {c0:.3g}')
        for k in range(1, min(len(arr), 4)):
            ck = arr[k]
            y = y + (ck.real * np.cos(k * omega * t)
                     - ck.imag * np.sin(k * omega * t))
            ampk = float(abs(ck))
            phik = float(np.angle(ck))
            if k == 1:
                lbl = (f"+ H1 (amp={ampk:.3g}, "
                       f"PI={2*ampk/max(abs(c0),1e-12):.2f}, φ={phik:+.2f})")
            else:
                dphi = _wrap(phik - k * phi1)
                lbl = (f"+ H{k} (amp={ampk:.3g}, "
                       f"H{k}/H1={ampk/max(amp1,1e-12):.2f}, "
                       f"Δφ_rel={dphi:+.2f})")
            ax_mid.plot(t, y, color=colors[k], lw=1.4, ls=styles[k],
                        label=lbl)
        ax_mid.set_xlabel('time (s)')
        ax_mid.set_ylabel(f'{sym} ({unit})')
        ax_mid.set_title('Exact harmonic decomposition '
                         '(solver coefficients — no fit)')
        ax_mid.legend(fontsize=8, loc='best')
        ax_mid.grid(alpha=0.3)

        # ── Bottom: magnitude spectrum (band-limited to H1–H3) ──
        sig_dm = sig - np.mean(sig)
        freqs = np.fft.rfftfreq(T, d=dt)
        mag = (2.0 / T) * np.abs(np.fft.rfft(sig_dm))
        ax_spec.semilogy(freqs, np.maximum(mag, 1e-9), color='#444', lw=0.7)
        for k in (1, 2, 3):
            ax_spec.axvline(k * f0, color=colors[k], ls='--', lw=0.9,
                            alpha=0.7, label=f'k·f₀ = {k*f0:.2f} Hz')
        ax_spec.set_xlim(0, min(4.5 * f0, freqs[-1]))
        ax_spec.set_xlabel('frequency (Hz)')
        ax_spec.set_ylabel(f'|FFT({sym})|  ({unit})')
        ax_spec.set_title('Magnitude spectrum  '
                          '(simulated — content only at k·f₀)')
        ax_spec.legend(fontsize=8, loc='upper right')
        ax_spec.grid(alpha=0.3, which='both')

        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

    # ── callbacks ──────────────────────────────────────────────────────
    def _sync_field_combos(self):
        """Sync enable / disable state of the Quantity, Property, and
        Harmonic combos to the current Source + selection.  Greys out
        invalid combinations (e.g. P in Measured, Pressure drop in
        Q-mode, Phase at DC) so the user can never pick a bad combo.

        If the current selection becomes invalid after a Source change,
        falls back to the first valid Property / Quantity it finds.
        """
        if not hasattr(self, '_quantity_combo'):
            return
        from qtpy.QtCore import Qt
        # Quantity: P only enabled when Sim source is active.
        sim_active = bool(getattr(self, '_sim_active', False))
        qmodel = self._quantity_combo.model()
        q_p_item = qmodel.item(1)  # 'P' entry
        if q_p_item is not None:
            flags = q_p_item.flags()
            if sim_active and self.current_source == 'sim':
                q_p_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)
            else:
                q_p_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_quantity == 'P':
                    self.current_quantity = 'Q'
                    self._quantity_combo.blockSignals(True)
                    self._quantity_combo.setCurrentIndex(0)
                    self._quantity_combo.blockSignals(False)
        # Property: enable rows matching (source, quantity, harmonic)
        # validity — delegate to _combo_valid so DC-incompatible
        # properties (phase, PI) grey out symmetrically.
        pmodel = self._property_combo.model()
        first_valid_idx = -1
        for i, (key, _lbl, _h, _valid_in) in enumerate(PROPERTY_DEFS):
            item = pmodel.item(i)
            if item is None:
                continue
            ok = _combo_valid(self.current_source, self.current_quantity,
                               key, self.current_harmonic)
            flags = item.flags()
            if ok:
                item.setFlags(flags | Qt.ItemIsEnabled
                                | Qt.ItemIsSelectable)
                if first_valid_idx < 0:
                    first_valid_idx = i
            else:
                item.setFlags(flags & ~Qt.ItemIsEnabled
                                & ~Qt.ItemIsSelectable)
        # If the currently-selected Property is invalid, fall back.
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property, self.current_harmonic):
            if first_valid_idx >= 0:
                self.current_property = PROPERTY_DEFS[first_valid_idx][0]
                self._property_combo.blockSignals(True)
                self._property_combo.setCurrentIndex(first_valid_idx)
                self._property_combo.blockSignals(False)
        # Harmonic combo: enabled only for harmonic-keyed properties.
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        harmonic_keyed = bool(pdef and pdef[2])
        self._harmonic_combo.setEnabled(harmonic_keyed)
        # Disable DC entry when Property = phase (undefined).
        hmodel = self._harmonic_combo.model()
        dc_item = hmodel.item(0)
        if dc_item is not None:
            flags = dc_item.flags()
            if self.current_property == 'phase':
                dc_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_harmonic == 'DC':
                    self.current_harmonic = 'H1'
                    self._harmonic_combo.blockSignals(True)
                    self._harmonic_combo.setCurrentIndex(1)
                    self._harmonic_combo.blockSignals(False)
            else:
                dc_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)
        # Reflect Source/Quantity in the time-series "Plot" button label so
        # it's clear whether it shows measured PIV or the simulated waveform.
        if getattr(self, '_plot_qt_btn', None) is not None:
            if self.current_source == 'sim':
                sym = 'P' if self.current_quantity == 'P' else 'Q'
                self._plot_qt_btn.setText(
                    f"Plot simulated {sym}(t) for selected")
            else:
                self._plot_qt_btn.setText("Plot Q(t) for selected")

    def _on_quantity_change(self, _idx=None):
        new_q = self._quantity_combo.currentData()
        if new_q is None or new_q == self.current_quantity:
            return
        self.current_quantity = new_q
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_property_change(self, _idx=None):
        new_p = self._property_combo.currentData()
        if new_p is None or new_p == self.current_property:
            return
        self.current_property = new_p
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_harmonic_change(self, _idx=None):
        new_h = self._harmonic_combo.currentData()
        if new_h is None or new_h == self.current_harmonic:
            return
        self.current_harmonic = new_h
        # Harmonic gates property availability (phase/PI invalid at DC).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _on_log_toggle(self, checked: bool):
        self.log_scale = bool(checked)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    # ── Radial-uncertainty tagging ────────────────────────────────

    def _on_radial_uncertain_toggle(self, checked: bool):
        self._show_radial_uncertain = bool(checked)
        self._refresh_edges()

    def _compute_flow_mismatch_edges(self):
        """Find edges measured in ≥2 tiles whose flow DIRECTION disagrees.

        Only measurements with ``snr_harm_fit_db`` ≥ the SNR threshold count
        (below the threshold the phasor is noise-dominated and its sign is
        unidentifiable, so a flip isn't a real disagreement).  An undirected
        edge has only two possible directions, so ≥2 distinct
        ``(flow_from, flow_to)`` pairs among qualifying measurements means the
        tiles disagree.  Caches the result (a set of ``frozenset({u, v})``)
        and refreshes the status label.  Returns the set.
        """
        thr = (float(self._flow_mismatch_snr_spin.value())
               if hasattr(self, '_flow_mismatch_snr_spin') else 3.0)
        mismatched = set()
        n_shared = 0
        for u, v, d in self.G.edges(data=True):
            dirs = set()
            n_qual = 0
            for m in (d.get('measurements_piv') or []):
                mv_m = MeasurementView(m)
                # v4 migration: was m.get('flow_from') / m.get('flow_to')
                ff, ft = mv_m.flow_from, mv_m.flow_to
                if ff is None or ft is None:
                    continue
                # v4 migration: was m.get('snr_harm_fit_db') → snr.dc_db
                snr = mv_m.snr.dc_db
                try:
                    if snr is None or not (float(snr) >= thr):
                        continue
                except (TypeError, ValueError):
                    continue
                n_qual += 1
                dirs.add((ff, ft))
            if n_qual >= 2:
                n_shared += 1
                if len(dirs) >= 2:
                    mismatched.add(frozenset((u, v)))
        self._flow_mismatch_edges = mismatched
        if hasattr(self, '_flow_mismatch_status_lbl'):
            self._flow_mismatch_status_lbl.setText(
                f"Mismatched: {len(mismatched)} of {n_shared} shared "
                f"(SNR_H1 ≥ {thr:g} dB)")
        return mismatched

    def _on_flow_mismatch_toggle(self, checked: bool):
        self._show_flow_mismatch = bool(checked)
        self._compute_flow_mismatch_edges()
        self._refresh_edges()

    def _on_flow_mismatch_snr_change(self, _v=None):
        self._compute_flow_mismatch_edges()
        if getattr(self, '_show_flow_mismatch', False):
            self._refresh_edges()

    def _toggle_radial_uncertain_on_selected(self):
        """Flip the ``radial_uncertain`` flag on ``self._selected_edge``."""
        if self._selected_edge is None:
            self._info_box.setPlainText(
                "No edge selected — click an edge first, then press "
                "'u' or the Toggle button.")
            return
        u, v = self._selected_edge
        if not self.G.has_edge(u, v):
            return
        d = self.G.edges[u, v]
        was = bool(d.get('radial_uncertain'))
        if was:
            d.pop('radial_uncertain', None)
        else:
            d['radial_uncertain'] = True
        self._radial_tags_dirty = True
        action = "cleared" if was else "flagged"
        self._info_box.setPlainText(
            f"Edge ({u}, {v}) {action} as radial-uncertain.")
        self._refresh_edges()
        self._refresh_radial_status()

    def _refresh_radial_status(self):
        """Update the status label with the count of tagged edges."""
        if not hasattr(self, '_radial_status_lbl'):
            return
        n = sum(1 for _, _, dd in self.G.edges(data=True)
                if dd.get('radial_uncertain'))
        dirty = " (unsaved)" if self._radial_tags_dirty else ""
        self._radial_status_lbl.setText(
            f"Tagged: {n} vessel(s){dirty}")

    def _on_click_to_tag_toggle(self, checked: bool):
        """Enter/exit click-to-tag mode."""
        self._click_to_tag_mode = bool(checked)
        if checked:
            self._info_box.setPlainText(
                "Click-to-tag mode ON. Every click on a vessel toggles its "
                "radial_uncertain flag. Uncheck to return to normal.")
        else:
            self._info_box.setPlainText(
                "Click-to-tag mode OFF.")

    def _tag_nearest_edge(self, x_graph: float, y_graph: float,
                            max_px: float = 20.0):
        """Toggle radial_uncertain on the edge nearest ``(x, y)`` in
        graph coordinates.  Used by click-to-tag mode + rectangle tag."""
        import numpy as np
        if self._edge_midpoints is None or len(self._edge_midpoints) == 0:
            return
        d2 = ((self._edge_midpoints[:, 0] - x_graph) ** 2 +
              (self._edge_midpoints[:, 1] - y_graph) ** 2)
        i = int(np.argmin(d2))
        dist = float(np.sqrt(d2[i]))
        if dist > max_px:
            return
        u, v = self.edge_list[i]
        d = self.G.edges[u, v]
        was = bool(d.get('radial_uncertain'))
        if was:
            d.pop('radial_uncertain', None)
        else:
            d['radial_uncertain'] = True
        self._radial_tags_dirty = True
        self._refresh_edges()
        self._refresh_radial_status()

    def _ensure_region_layer(self):
        """Create the scratch Shapes layer used for region tag/untag if
        it doesn't exist yet, and register the mouse-drag callback ONCE.
        Reusing a single layer across button-presses avoids the napari
        add/remove race that made the old rectangle tag stop working
        after a couple of uses."""
        lyr = getattr(self, '_region_tag_layer', None)
        # `lyr not in self.viewer.layers` covers the case where the user
        # deleted the layer manually via napari's layer panel.
        if lyr is not None and lyr in self.viewer.layers:
            return lyr
        lyr = self.viewer.add_shapes(
            [], name='RegionTag',
            edge_color=RADIAL_UNCERTAIN_COLOR,
            face_color='transparent', edge_width=2.0, opacity=0.9,
        )
        self._region_tag_layer = lyr

        # Act on each shape only once napari has fully committed it
        # (data event, action 'added'), deferred to the next event-loop
        # tick.  The old mouse-drag callback fired on the FIRST release of
        # a click-trace-click lasso, cleared the layer while napari was
        # still building the shape, and crashed napari's lasso handler.
        from qtpy.QtCore import QTimer

        def _on_region_data(event):
            if str(getattr(event, 'action', '')) == 'added':
                QTimer.singleShot(0, self._on_region_shape_added)

        lyr.events.data.connect(_on_region_data)
        return lyr

    def _on_region_shape_added(self):
        """Apply the pending tag/untag to the shape just drawn (rectangle,
        ellipse, lasso or polygon), clear it, and stay armed for the next."""
        import numpy as np
        lyr = getattr(self, '_region_tag_layer', None)
        if lyr is None or lyr not in self.viewer.layers or not len(lyr.data):
            return
        poly_rc = np.asarray(lyr.data[-1], dtype=float)[:, -2:]
        if lyr.shape_type[-1] == 'ellipse' and len(poly_rc) == 4:
            c = poly_rc.mean(axis=0)
            a = 0.5 * (poly_rc[1] + poly_rc[2]) - c
            b = 0.5 * (poly_rc[2] + poly_rc[3]) - c
            t = np.linspace(0, 2 * np.pi, 96, endpoint=False)[:, None]
            poly_rc = c + np.cos(t) * a + np.sin(t) * b
        action = getattr(self, '_region_action_pending', 'tag')
        if len(poly_rc) >= 3:
            n = self._apply_region_action(poly_rc, action)
            if n:
                self._radial_tags_dirty = True
            verb = "Tagged" if action == "tag" else "Untagged"
            self._info_box.setPlainText(
                f"{verb} {n} vessel(s) via region.  Draw another region to "
                f"{action} more, or press Stop.")
            self._refresh_edges()
            self._refresh_radial_status()
        try:
            mode = lyr.mode
            lyr.data = []
            idx = self.viewer.layers.index(lyr)
            if idx != len(self.viewer.layers) - 1:
                self.viewer.layers.move(idx, len(self.viewer.layers) - 1)
            self.viewer.layers.selection.active = lyr
            lyr.mode = mode
        except Exception:
            pass

    def _reset_region_layer(self):
        """Clear scratch shapes and drop out of draw mode so accidental
        clicks don't add more shapes.  Layer itself is kept for reuse."""
        lyr = getattr(self, '_region_tag_layer', None)
        if lyr is None or lyr not in self.viewer.layers:
            return
        try:
            lyr.data = []
            lyr.mode = 'pan_zoom'
        except Exception:
            pass

    def _start_region_action(self, action: str):
        """action ∈ {'tag', 'untag'}.  Sets the scratch shapes layer's
        mode to the currently-selected shape (Rectangle / Lasso) and
        records the pending action so the drag callback knows what to
        apply on release.

        Robustness: the scratch layer is moved to the TOP of the layer
        stack every time we arm, and Qt events are pumped between the
        active-layer change and the mode change.  Without this, after a
        few sessions napari's mouse-event dispatcher was ending up on the
        Edges or Mosaic layer (whichever was frontmost) and reading the
        drag as a pan/zoom instead of a shape-add."""
        from qtpy.QtWidgets import QApplication
        shape_mode = self._region_shape_combo.currentData()   # 'add_rectangle' / 'add_polygon_lasso'
        self._region_action_pending = action
        self._region_shape_pending = shape_mode
        lyr = self._ensure_region_layer()
        try:
            lyr.data = []
        except Exception:
            pass
        # Force the scratch layer to be visible, top of the stack, and
        # the interactive-active layer — belt and suspenders against
        # napari's mouse-dispatch quirks.
        try:
            lyr.visible = True
            idx = self.viewer.layers.index(lyr)
            top = len(self.viewer.layers) - 1
            if idx != top:
                self.viewer.layers.move(idx, top)
        except Exception:
            pass
        self.viewer.layers.selection.active = lyr
        # Let napari process the active-layer change before we set the
        # mode; without this pump the mode change occasionally lands on
        # a stale layer state and the next drag reads as pan/zoom.
        try:
            QApplication.processEvents()
        except Exception:
            pass
        lyr.mode = shape_mode
        shape_label = self._region_shape_combo.currentText()
        self._info_box.setPlainText(
            f"{shape_label} {action}: draw regions on the canvas — each "
            f"is applied when the shape closes.  Every edge whose midpoint "
            f"is inside gets ``radial_uncertain`` "
            f"{'set' if action == 'tag' else 'cleared'}.  Press Stop when done.")

    def _apply_region_action(self, poly_rc, action: str):
        """poly_rc: (N, 2) polygon vertices in display (row, col) coords.
        action ∈ {'tag', 'untag'}.  Returns the count of edges whose flag
        actually changed (i.e. was toggled, not already in target state)."""
        import numpy as np
        from matplotlib.path import Path
        H = self.mosaic_height
        # edge midpoints stored as (x, y) with y in graph frame → display
        # (row, col) = (H − y, x)
        disp = np.column_stack([H - self._edge_midpoints[:, 1],
                                 self._edge_midpoints[:, 0]])
        inside = Path(np.asarray(poly_rc)).contains_points(disp)
        n = 0
        for i in np.flatnonzero(inside):
            u, v = self.edge_list[i]
            d = self.G.edges[u, v]
            was = bool(d.get('radial_uncertain'))
            if action == 'tag' and not was:
                d['radial_uncertain'] = True
                n += 1
            elif action == 'untag' and was:
                d.pop('radial_uncertain', None)
                n += 1
        return n

    def _save_radial_uncertain_tags(self):
        """Write the current graph back to disk with updated flags."""
        import pickle
        from qtpy.QtWidgets import QMessageBox
        try:
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.G, f)
            self._radial_tags_dirty = False
            self._refresh_radial_status()
            n = sum(1 for _, _, dd in self.G.edges(data=True)
                    if dd.get('radial_uncertain'))
            QMessageBox.information(
                self.viewer.window._qt_window,
                "Radial-uncertainty tags saved",
                f"Wrote {n} tagged vessel(s) to:\n{self.graph_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self.viewer.window._qt_window,
                "Save failed",
                f"Could not save graph:\n{exc}",
            )

    def _on_source_change(self, _checked=None):
        # Four-way exclusive group; only act when EXACTLY one is checked
        radios = [self._src_radio_measured,
                  getattr(self, '_src_radio_norm', None),
                  self._src_radio_sim,
                  getattr(self, '_src_radio_sim_carved', None)]
        checked = [r for r in radios if r is not None and r.isChecked()]
        if len(checked) != 1:
            return  # transient state during exclusive-group switch
        if (getattr(self, '_src_radio_sim_carved', None) is not None
                and self._src_radio_sim_carved.isChecked()):
            new_src = 'sim_carved'
        elif self._src_radio_sim.isChecked():
            new_src = 'sim'
        elif (getattr(self, '_src_radio_norm', None) is not None
              and self._src_radio_norm.isChecked()):
            new_src = 'normalized'
        else:
            new_src = 'measured'
        if new_src == self.current_source:
            return
        self.current_source = new_src
        # Re-sync enablement; the selection may need to fall back.
        self._sync_field_combos()
        # Refresh carve boundary markers if they're on (colours follow
        # the source).
        try:
            self._update_carve_boundary_markers()
        except Exception:
            pass
        # Tile-harmonic cache must be rebuilt against the new source's Q_t
        if hasattr(self, '_tile_harmonic_cache'):
            self._tile_harmonic_cache.clear()
            if (getattr(self, 'current_tile_filter', None) is not None
                    and self.current_tile_filter != 'All'):
                self._populate_tile_harmonic_cache(self.current_tile_filter)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()

    def _normalization_available(self) -> bool:
        """True if the loaded graph already has normalized fields."""
        try:
            return bool(self.G.graph.get('tile_gauges'))
        except Exception:
            return False

    def _carved_sim_data_available(self) -> bool:
        """True if any edge carries `_h_*_sim_carved` fields from a
        previous Carved Sim run."""
        try:
            for _, _, d in self.G.edges(data=True):
                if any(k.endswith('_sim_carved') for k in d.keys()):
                    return True
            return False
        except Exception:
            return False

    def _refresh_carved_sim_radio_enable(self):
        """Called after a Carved Sim run to unlock the radio option."""
        try:
            if hasattr(self, '_src_radio_sim_carved'):
                avail = self._carved_sim_data_available()
                self._src_radio_sim_carved.setEnabled(avail)
                if avail:
                    self._src_radio_sim_carved.setToolTip("")
        except Exception:
            pass

    def _refresh_source_radio_enable(self):
        """Call after normalization runs (or is cleared) to update the
        Normalized radio's enabled state + tooltip."""
        if not hasattr(self, '_src_radio_norm'):
            return
        avail = self._normalization_available()
        self._src_radio_norm.setEnabled(avail)
        self._src_radio_norm.setToolTip(
            "" if avail else
            "Compute or load a normalization in the Normalize tab to enable.")
        if not avail and self.current_source == 'normalized':
            # Fall back to measured
            self.current_source = 'measured'
            self._src_radio_measured.setChecked(True)

    # ── Custom-BC helpers ────────────────────────────────────────────
    def _on_bc_source_change(self, _checked=None):
        """Toggle the Custom-waveform spinboxes based on BC source.
        The preview canvas stays live in both modes — it always shows
        whatever waveform would actually be applied at solve time."""
        if not hasattr(self, '_sim_custom_group'):
            return
        is_custom = bool(self._sim_bc_radio_custom.isChecked())
        # Show the custom-waveform grid only in Custom mode (keeps the tab
        # compact under the default class-waveform / measured modes).
        self._sim_custom_group.setVisible(is_custom)
        for sp in getattr(self, '_sim_custom_spins', {}).values():
            sp.setEnabled(is_custom)
        # Re-render the preview for the new mode.
        self._refresh_custom_preview()

    def _read_custom_waveform(self):
        """Return per-side (amps, phases) tuples for the A and V custom
        waveform specs.  Amplitudes are |H_k|/|Q̄| (length 3, k=1..3);
        phases are in radians (length 3).
        Returns: ((amps_A, phases_A), (amps_V, phases_V)).
        """
        out = {}
        for side in ('A', 'V'):
            amps = []
            phases = []
            for k in (1, 2, 3):
                a = float(
                    self._sim_custom_spins[(side, k, 'amp')].value())
                p = np.radians(float(
                    self._sim_custom_spins[(side, k, 'phase')].value()))
                amps.append(a)
                phases.append(p)
            out[side] = (amps, phases)
        return out['A'], out['V']

    def _refresh_custom_preview(self):
        """Refresh the BC waveform preview.

        Shows whichever waveform would actually be applied at solve
        time — custom (from the spinboxes) or measured (per-type
        complex average of stored bc_harmonics, with the canonical
        −1 sign-flip applied to sinks).  Y-axis is absolute Q(t) in
        nL/s so the A↔V magnitude asymmetry reads honestly.
        """
        if not hasattr(self, '_sim_custom_fig'):
            return
        f0 = (float(self._sim_detected_f0)
              if np.isfinite(self._sim_detected_f0) else 2.5)
        T_period = 1.0 / f0
        t = np.linspace(0.0, 2.0 * T_period, 400)
        omega = 2.0 * np.pi * f0
        target_flux = float(self._sim_flux_spin.value())
        is_custom = self._sim_bc_radio_custom.isChecked()
        n_src = max(1, len(self._source_nodes))
        n_snk = max(1, len(self._sink_nodes))

        self._sim_custom_fig.clear()
        ax = self._sim_custom_fig.add_subplot(111)

        if is_custom:
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk
            y_A = np.full_like(t, Qbar_A)
            y_V = np.full_like(t, Qbar_V)
            for k_idx, (a, p) in enumerate(
                    zip(amps_A, phases_A), start=1):
                y_A += a * Qbar_A * np.cos(k_idx * omega * t + p)
            for k_idx, (a, p) in enumerate(
                    zip(amps_V, phases_V), start=1):
                y_V += a * Qbar_V * np.cos(k_idx * omega * t + p)
            PI_A = 2.0 * abs(amps_A[0]) if amps_A[0] else 0.0
            PI_V = 2.0 * abs(amps_V[0]) if amps_V[0] else 0.0
            title = (f"CUSTOM  Q̄_A = {abs(Qbar_A):.3f}  "
                     f"(PI_A ≈ {PI_A:.2f}),  "
                     f"Q̄_V = {abs(Qbar_V):.3f} nL/s  "
                     f"(PI_V ≈ {PI_V:.2f})")
        else:
            # Measured: per-type complex average of stored bc_harmonics,
            # with canonical sign-flip on sinks (Q into network).
            src_arrs = [np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                     dtype=complex)
                        for n in self._source_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            snk_arrs = [-np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                      dtype=complex)
                        for n in self._sink_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            if not src_arrs or not snk_arrs:
                ax.text(0.5, 0.5,
                        'No measured bc_harmonics on this graph.',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=10, color='#888')
                ax.set_xticks([]); ax.set_yticks([])
                self._sim_custom_canvas.draw_idle()
                return
            src_avg = np.mean(np.stack(src_arrs, axis=0), axis=0).copy()
            snk_avg = np.mean(np.stack(snk_arrs, axis=0), axis=0).copy()
            # Apply the same flux scaling the solver path uses.
            if self._sim_equal_split_chk.isChecked():
                src_avg[0] = +target_flux / len(src_arrs)
                snk_avg[0] = -target_flux / len(snk_arrs)
                mode_label = 'equal-split'
            else:
                sum_src = sum(arr[0].real for arr in src_arrs)
                if abs(sum_src) > 1e-12:
                    scale = target_flux / abs(sum_src)
                    src_avg = scale * src_avg
                    snk_avg = scale * snk_avg
                mode_label = 'measured-ratio'
            # Reconstruct Q(t) for both types.
            y_A = np.full_like(t, src_avg[0].real, dtype=float)
            y_V = np.full_like(t, snk_avg[0].real, dtype=float)
            kmax = min(len(src_avg), 4)
            for k in range(1, kmax):
                y_A += (src_avg[k].real * np.cos(k * omega * t)
                        - src_avg[k].imag * np.sin(k * omega * t))
                y_V += (snk_avg[k].real * np.cos(k * omega * t)
                        - snk_avg[k].imag * np.sin(k * omega * t))
            title = (f"MEASURED (per-type avg, {mode_label})   "
                     f"Q̄_A = {abs(src_avg[0].real):.3f},  "
                     f"Q̄_V = {abs(snk_avg[0].real):.3f} nL/s")

        # Display convention: flip each trace so its DC reads positive.
        # "Positive" = the sign that yields DC > 0 for that side, so
        # both A and V sit on the same axis and the magnitudes are
        # directly comparable.  Solver-internal sign convention (sinks
        # store -Q_outflow) is unaffected — this is purely cosmetic.
        s_A = -1.0 if float(np.nanmean(y_A)) < 0 else 1.0
        s_V = -1.0 if float(np.nanmean(y_V)) < 0 else 1.0
        y_A = y_A * s_A
        y_V = y_V * s_V

        ax.plot(t * 1000.0, y_A, color='C0', lw=1.5, label='A')
        ax.plot(t * 1000.0, y_V, color='C1', lw=1.5, label='V')
        ax.axhline(0, color='#888', lw=0.5)
        ax.set_xlabel('t (ms)', fontsize=8)
        ax.set_ylabel('Q(t) (nL/s)', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        self._sim_custom_canvas.draw_idle()

    def _on_nodes_toggle(self, checked: bool):
        self.show_nodes = bool(checked)
        self._refresh_nodes()

    def _on_tile_boundaries_toggle(self, checked: bool):
        self.show_tile_boundaries = bool(checked)
        self._refresh_tile_boundaries()

    def _on_tile_labels_toggle(self, checked: bool):
        self.show_tile_labels = bool(checked)
        self._refresh_tile_labels()

    def _on_edge_labels_toggle(self, checked: bool):
        self.show_edge_labels = bool(checked)
        self._refresh_edge_labels()

    def _on_kcl_toggle(self, checked: bool):
        self.show_kcl = bool(checked)
        self._refresh_kcl_overlay()

    def _refresh_kcl_overlay(self):
        """Show per-node KCL residual (signed Σ Q at the selected tile)
        as a Points layer.  Only renders when show_kcl is True AND a
        single tile is selected (current_tile_filter is not None)."""
        # Tear down any existing overlay first.
        if self._kcl_layer is not None:
            try:
                self.viewer.layers.remove(self._kcl_layer)
            except (KeyError, ValueError):
                pass
            self._kcl_layer = None
        if not self.show_kcl:
            return
        tile_id = self.current_tile_filter
        if tile_id is None:
            self._info_box.setPlainText(
                "Kirchhoff residuals need a specific tile selected "
                "(Tile filter in the View tab).")
            return

        # Compute per-node signed Q sum from THIS tile's measurements.
        residual = {n: 0.0 for n in self.G.nodes()}
        n_contrib = {n: 0 for n in self.G.nodes()}
        for u, v in self.G.edges():
            ed = self.G.edges[u, v]
            m = next((mm for mm in (ed.get('measurements_piv') or [])
                       if isinstance(mm, dict)
                       and MeasurementView(mm).tile_id == int(tile_id)), None)
            if m is None:
                continue
            mv_m = MeasurementView(m)
            mq = m.get('Q_DC')
            if mq is None or not np.isfinite(mq):
                # v4 migration: was m.get('mean_Q')
                mq = mv_m.harm.mean_Q
            if mq is None or not np.isfinite(mq):
                continue
            # v4 migration: was m.get('flow_from') / m.get('flow_to')
            ff = mv_m.flow_from; ft = mv_m.flow_to
            if ff is None or ft is None:
                continue
            # Treat measured |Q| as flowing ff → ft.  Signed residual at
            # node n = (in − out) so positive = net inflow, negative =
            # net outflow.  By KCL on an interior node, residual ≈ 0.
            q = float(abs(mq))
            # Flow leaves ff, arrives at ft.  Subtract at ff, add at ft.
            residual[ff] -= q
            residual[ft] += q
            n_contrib[ff] += 1
            n_contrib[ft] += 1

        # Build node positions + colour.
        nodes_with_data = [n for n, c in n_contrib.items() if c > 0]
        if not nodes_with_data:
            return
        pts = np.array([[self.mosaic_height - self.G.nodes[n]['y'],
                          self.G.nodes[n]['x']]
                         for n in nodes_with_data])
        vals = np.array([residual[n] for n in nodes_with_data])
        max_abs = float(np.percentile(np.abs(vals), 99)) or 1.0
        # Diverging colormap: red = positive (net inflow), blue = negative.
        from matplotlib import cm as _mpl_cm
        norm = mcolors.Normalize(vmin=-max_abs, vmax=+max_abs)
        colors = _mpl_cm.RdBu_r(norm(vals))
        # Size by |residual| relative to max, so the worst residuals pop.
        sizes = 6.0 + 20.0 * np.abs(vals) / max(max_abs, 1e-12)
        self._kcl_layer = self.viewer.add_points(
            pts, name=f"KCL residual (tile {tile_id})  ·  "
                       f"max |Σ Q| = {max_abs:.3f} nL/s",
            face_color=colors, size=sizes,
            edge_color='black', edge_width=0.4,
            opacity=0.85,
        )
        # Diagnostic summary in the info box.
        interior_vals = []
        boundary_vals = []
        for n in nodes_with_data:
            if self.G.nodes[n].get('boundary_type') in ('source', 'sink'):
                boundary_vals.append(residual[n])
            else:
                interior_vals.append(residual[n])
        msg_parts = [f"Tile {tile_id} KCL residual:"]
        if interior_vals:
            interior_vals = np.array(interior_vals)
            msg_parts.append(
                f"  interior (n={len(interior_vals)}): "
                f"RMS = {float(np.sqrt(np.mean(interior_vals**2))):.4f} nL/s, "
                f"max |Σ| = {float(np.abs(interior_vals).max()):.4f}")
        if boundary_vals:
            boundary_vals = np.array(boundary_vals)
            msg_parts.append(
                f"  boundary (n={len(boundary_vals)}): "
                f"max |Σ| = {float(np.abs(boundary_vals).max()):.4f}  "
                f"(expected non-zero — net carve in/outflow)")
        self._info_box.setPlainText("\n".join(msg_parts))

        # Popup the distribution histogram.
        self._render_kcl_distribution(
            tile_id,
            np.asarray(interior_vals, dtype=float),
            np.asarray(boundary_vals, dtype=float),
        )

    def _render_kcl_distribution(self, tile_id, interior_vals, boundary_vals):
        """Save + display a histogram of KCL residuals for the current
        tile.  Splits interior (should be ≈ 0) from boundary (carve net
        flow) so they can be compared on the same axes."""
        import matplotlib.pyplot as plt
        import tempfile
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        # Pool both sets for x-range determination.
        pool = []
        if interior_vals.size: pool.append(interior_vals)
        if boundary_vals.size: pool.append(boundary_vals)
        if not pool:
            plt.close(fig); return
        all_vals = np.concatenate(pool)
        lim = float(max(np.percentile(np.abs(all_vals), 99) * 1.15, 1e-6))
        bins = np.linspace(-lim, lim, 50)
        if interior_vals.size:
            rms_i = float(np.sqrt(np.mean(interior_vals ** 2)))
            ax.hist(interior_vals, bins=bins, density=False,
                    histtype='stepfilled', color='#4477aa', alpha=0.7,
                    edgecolor='#1f3a5a', linewidth=1.2,
                    label=f"interior  (n={len(interior_vals)},  RMS={rms_i:.4f} nL/s)")
        if boundary_vals.size:
            ax.hist(boundary_vals, bins=bins, density=False,
                    histtype='stepfilled', color='#cc6677', alpha=0.7,
                    edgecolor='#7a2a32', linewidth=1.2,
                    label=f"boundary  (n={len(boundary_vals)})")
        ax.axvline(0.0, color='#333', lw=0.6, ls=':')
        ax.set_xlabel("KCL residual  (Σ Q arriving − Σ Q leaving)  [nL/s]")
        ax.set_ylabel("node count")
        ax.set_title(f"KCL residual distribution  ·  tile {tile_id}")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.25)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png = f.name
        fig.savefig(png, dpi=130, bbox_inches='tight')
        plt.close(fig)
        # Display in a small Qt dialog.
        try:
            from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
            from qtpy.QtGui import QPixmap
            from qtpy.QtCore import Qt
            dlg = QDialog(self.viewer.window._qt_window)
            dlg.setWindowTitle(f"KCL residual distribution — tile {tile_id}")
            lay = QVBoxLayout(dlg)
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            lbl = QLabel()
            pm = QPixmap(png); lbl.setPixmap(pm); lbl.setAlignment(Qt.AlignCenter)
            scroll.setWidget(lbl); lay.addWidget(scroll)
            dlg.resize(min(int(pm.width()) + 30, 1200),
                        min(int(pm.height()) + 80, 800))
            self._kcl_dist_dialog = dlg   # retain so GC doesn't kill it
            dlg.show()
        except Exception as e:
            self._info_box.append(f"\n(could not show histogram dialog: {e})\n"
                                    f"figure saved to: {png}")

    def _refresh_edge_labels(self):
        """Render the current Property value as text at each edge's
        midpoint.  Only active when self.show_edge_labels is True AND
        Tile filter is set to a specific tile — in All-tiles mode the
        labels clutter the mosaic.
        """
        # Always start by removing any existing label layer; we then
        # decide whether to add a fresh one based on current state.
        if self._edge_labels_layer is not None:
            try:
                self.viewer.layers.remove(self._edge_labels_layer)
            except (KeyError, ValueError):
                pass
            self._edge_labels_layer = None
        if not self.show_edge_labels:
            return
        if self.current_tile_filter is None:
            return
        tile_id = int(self.current_tile_filter)
        prop = self.current_property
        # Format helpers for the most common property types
        def _fmt(val: float) -> str:
            if prop == 'phase':
                return f"{np.degrees(val):+.0f}°"
            if prop in ('snr', 'snr_ac', 'snr_total'):
                return f"{val:+.1f} dB"
            if prop == 'PI':
                return f"{val:.2f}"
            if prop == 'frequency':
                return f"{val:.2f} Hz"
            if prop == 'dissipation':
                return f"{val:.1e} W"
            if prop in ('radius', 'length'):
                return f"{val:.0f}"
            # default: 3-sig-fig
            return f"{val:.3g}"
        points = []
        texts = []
        H_mosaic = self.mosaic_height
        for u, v in self.edge_list:
            d = self.G.edges[u, v]
            # Only edges with a measurement on the active tile.
            # v4 migration: was m.get('tile_id')
            if not any(MeasurementView(m).tile_id == tile_id
                       for m in (d.get('measurements_piv') or [])):
                continue
            val = self._resolve_field_value(u, v)
            if val is None or not np.isfinite(val):
                continue
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            xm = 0.5 * (float(x1) + float(x2))
            ym = 0.5 * (float(y1) + float(y2))
            points.append([H_mosaic - ym, xm])
            texts.append(_fmt(float(val)))
        if not points:
            return
        try:
            self._edge_labels_layer = self.viewer.add_points(
                np.asarray(points, dtype=float),
                size=0,
                text={'string': texts, 'size': 8, 'color': 'white',
                       'anchor': 'center'},
                name='Edge labels',
                face_color='transparent',
                border_color='transparent',
            )
        except Exception as e:
            print(f"  Could not add edge-label layer: {e}")

    def _load_video_for_selected(self):
        tid = getattr(self, '_selected_tile_id', None)
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)

    def _load_video_from_dropdown(self):
        if not hasattr(self, '_tile_combo'):
            return
        # currentData() returns the int tile ID stored as userData;
        # falls back to None for the "(pick a tile)" placeholder and
        # for greyed-out entries (which can't be selected anyway).
        tid = self._tile_combo.currentData()
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)

    # ── public entry ───────────────────────────────────────────────────
    def run(self):
        self._setup_viewer()
        self._setup_panel()
        self._refresh_edges()
        self._refresh_nodes()
        # Initial colour-bar draw — needs `_last_vmin`/`_last_vmax`
        # populated by the first `_refresh_edges()` above, so do it here
        # rather than at the end of `_setup_panel`.
        self._refresh_cbar()
        if self.rotate_cw or self.rotate_ccw:
            self._apply_view_rotation(ccw=self.rotate_ccw)
        import napari
        napari.run()

    def _apply_view_rotation(self, ccw: bool = False):
        """Rotate every layer by 90° on top of its base affine.

        CW   (default):  (row, col) → (col,         H − row)
        CCW (ccw=True):  (row, col) → (W − col,     row     )

        Useful for datasets stored on their sides (e.g. 15-somite,
        which needs CCW rotation to put the embryo body axis vertical
        in the same orientation as the 21- and 27-somite mosaics).

        Caches the rotation affine and subscribes to the layers'
        `events.inserted` signal so that layers added LATER (tile
        numbers, video overlay, click markers, …) also get rotated.
        """
        H = self.mosaic.shape[0] if self.mosaic is not None else int(
            self.G.graph.get('mosaic_height', 0))
        W = self.mosaic.shape[1] if self.mosaic is not None else int(
            self.G.graph.get('mosaic_width', 0))
        if H <= 0 or W <= 0:
            return
        if ccw:
            rot_affine = np.array([[0, -1, W],
                                     [1,  0, 0],
                                     [0,  0, 1]], dtype=float)
        else:
            rot_affine = np.array([[0,  1, 0],
                                     [-1, 0, H],
                                     [0,  0, 1]], dtype=float)
        self._rotation_affine = rot_affine

        # Only skip the Mosaic image — its tissue is already in the
        # correct/upright orientation.  Everything else (graph shapes,
        # points, markers, per-tile Video overlays) was captured in a
        # 90°-CCW frame relative to the mosaic, so display-rotating them
        # aligns them with the upright tissue.
        _skip_prefixes = ('Mosaic',)

        def _should_skip(layer) -> bool:
            name = str(getattr(layer, 'name', '') or '')
            return any(name.startswith(p) for p in _skip_prefixes)

        def _rotate_layer(layer):
            if _should_skip(layer):
                return
            base = np.array(layer.affine.affine_matrix)
            if base.shape[0] == 3:
                layer.affine = rot_affine @ base
            elif base.shape[0] == 4:
                rot4 = np.eye(4)
                rot4[1:3, 1:3] = rot_affine[:2, :2]
                rot4[1:3, 3] = rot_affine[:2, 2]
                layer.affine = rot4 @ base

        for layer in self.viewer.layers:
            _rotate_layer(layer)

        # Catch layers added later (tile-number overlay, video, markers).
        def _on_layer_inserted(event):
            try:
                _rotate_layer(event.value)
            except Exception as exc:
                print(f"  WARN: failed to rotate new layer "
                      f"{getattr(event.value, 'name', '?')!r}: {exc}")
        self.viewer.layers.events.inserted.connect(_on_layer_inserted)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    # Strip empty-string CLI args — VS Code launch.json inputs pass
    # through an empty "" placeholder for optional flags (e.g. a
    # conditional --rotate-cw); argparse would reject "" otherwise.
    sys.argv = [a for a in sys.argv if a != '']

    parser = argparse.ArgumentParser(
        description="Read-only mosaic viewer (intern + publication use).")
    parser.add_argument('graph', nargs='?', default=None,
                        help="Path to mosaic_graph_analyzed.gpickle. "
                             "Optional if --config is given.")
    parser.add_argument('--config', default=None,
                        help="Path to a batch_analyze_config.json (the "
                             "same one the analysis pipeline uses).  "
                             "Convenience: pulls graph, tiff, tile "
                             "positions, video dir, and video pattern "
                             "from one file.  Individual --tiff / "
                             "--tile-positions / --video-dir / "
                             "--video-pattern args override.")
    parser.add_argument('--tiff', default=None,
                        help="Optional stitched mosaic TIFF for background.")
    parser.add_argument('--tile-positions', default=None,
                        help="Path to tile_positions_manual.json.  "
                             "Required for video overlay.")
    parser.add_argument('--video-dir', default=None, action='append',
                        help="Directory containing per-tile videos. Required "
                             "for video overlay.  May be given MULTIPLE times "
                             "(or comma-separated) to list candidate locations "
                             "for different machines; the first that exists on "
                             "this machine is used (e.g. a desktop path and a "
                             "laptop path in one command).")
    parser.add_argument('--video-pattern', default=None,
                        help="Filename pattern for tile videos, e.g. "
                             "'10x 250fps loc{vid}_C001H001S0001'. "
                             "Required for video overlay.")
    parser.add_argument('--of-video-dir', default=None, action='append',
                        help="Optional directory for fresh OF/PIV compute. "
                             "Use this to display downsampled videos while "
                             "computing on raw frame folders. Falls back to "
                             "--video-dir when omitted.")
    parser.add_argument('--of-video-pattern', default=None,
                        help="Optional filename/folder pattern for fresh "
                             "OF/PIV compute videos. Falls back to "
                             "--video-pattern when omitted.")
    parser.add_argument('--initial-field', default='mean_Q',
                        help="Field to colour edges by on startup. "
                             "Default: mean_Q.")
    parser.add_argument('--no-cache-harmonic-class', action='store_true',
                        help="Don't write the precomputed harmonic_class "
                             "back to the gpickle.  Default is to cache "
                             "(adds an edge attr; touches no other data).")
    parser.add_argument('--force-recompute-harmonic-class',
                        action='store_true',
                        help="Wipe any existing harmonic_class values and "
                             "recompute from scratch.  Useful if PIV was "
                             "re-run and the cached values are stale.")
    parser.add_argument('--rotate-cw', action='store_true',
                        help="Rotate the graph 90° clockwise on startup.")
    parser.add_argument('--rotate-ccw', action='store_true',
                        help="Rotate the graph 90° counter-clockwise on "
                             "startup.  Use for 15-somite (stored on its "
                             "side).")
    parser.add_argument('--auto-rotate', action='store_true',
                        help="Auto-select rotation from graph path: "
                             "'Somites21' → --rotate-cw, 'Somites15' → "
                             "--rotate-ccw, others → no rotation.")
    args = parser.parse_args()

    # Auto-rotate: derive from graph path if no explicit flag was given.
    if args.auto_rotate and not (args.rotate_cw or args.rotate_ccw):
        graph_str = str(args.graph or '')
        if 'Somites21' in graph_str:
            args.rotate_cw = True
        elif 'Somites15' in graph_str:
            args.rotate_ccw = True

    # Resolve paths via config if provided; individual args win.
    # Paths in the config can be RELATIVE to the config file's directory
    # — that makes the data folder portable (intern only changes the
    # config path, not its contents).
    cfg: dict = {}
    cfg_dir: Optional[Path] = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg_dir = cfg_path.parent
        with open(cfg_path) as f:
            cfg = json.load(f)

    def _resolve(cli_val, cfg_key):
        v = cli_val if cli_val is not None else cfg.get(cfg_key)
        if v is None:
            return None
        p = Path(v)
        if p.is_absolute() or cfg_dir is None:
            return p
        return (cfg_dir / p).resolve()

    def _resolve_multi(cli_list, cfg_key):
        """Resolve a video-dir spec to a LIST of candidate paths (CLI args win
        over config; config value may itself be a string or a list).  Each
        relative path is resolved against the config dir."""
        vals = cli_list if cli_list else cfg.get(cfg_key)
        if vals is None:
            return None
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        out = []
        for v in vals:
            parts = str(v).split(',') if isinstance(v, str) else [v]
            for part in parts:
                s = str(part).strip()
                if not s:
                    continue
                p = Path(s)
                out.append(p if (p.is_absolute() or cfg_dir is None)
                           else (cfg_dir / p).resolve())
        return out or None

    # Prefer a viewer-specific graph key so the config's `mosaic_graph`
    # (the BATCH pipeline's input) isn't loaded by the viewer.  Falls back
    # to `mosaic_graph` for old configs.  CLI positional always wins.
    graph = _resolve(args.graph, 'viewer_graph')
    if not graph:
        graph = _resolve(args.graph, 'mosaic_graph')
    if not graph:
        parser.error("graph path required (positional arg or "
                     "config['viewer_graph'] / config['mosaic_graph']).")
    tiff = _resolve(args.tiff, 'mosaic_tiff')
    tile_positions = _resolve(args.tile_positions, 'tile_positions')
    video_dir = _resolve_multi(args.video_dir, 'video_dir')
    of_video_dir = _resolve_multi(args.of_video_dir, 'of_video_dir')
    # video_pattern is a filename pattern (not a path) — pass through.
    video_pattern = args.video_pattern or cfg.get('video_pattern')
    of_video_pattern = (args.of_video_pattern
                        or cfg.get('of_video_pattern')
                        or video_pattern)

    viewer = ReadOnlyMosaicViewer(
        Path(graph),
        tiff_path=Path(tiff) if tiff else None,
        tile_positions=Path(tile_positions) if tile_positions else None,
        video_dir=video_dir,            # list of candidate dirs (or None)
        video_pattern=video_pattern,
        of_video_dir=of_video_dir,      # list of candidate dirs (or None)
        of_video_pattern=of_video_pattern,
        initial_field=args.initial_field,
        cache_harmonic_class=not args.no_cache_harmonic_class,
        force_recompute_harmonic_class=args.force_recompute_harmonic_class,
        rotate_cw=args.rotate_cw,
        rotate_ccw=args.rotate_ccw,
    )
    viewer.run()


if __name__ == '__main__':
    main()
