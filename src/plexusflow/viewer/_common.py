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
# to sys.path here: $PERTILEFLOW_ROOT, else a sibling PerTileFlow checkout.
# Silently no-ops where neither exists (the Simulate tab is then unavailable).
_pertile_candidates = [
    os.environ.get('PERTILEFLOW_ROOT'),
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
from ..io.measurement_view import MeasurementView


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
         ('sim_carved', 'Q'), ('reconciled', 'Q')}),
    # Signed value (real part, NO |·|) — sim only.  The point is DC pressure,
    # where the gauge can dip below 0; |·| would hide the sign.  Rendered on a
    # diverging colormap centred at 0.  (At AC harmonics this is the in-phase
    # component, whose sign depends on the solver phase reference.)
    ('signed',        'Signed (no |·|)',      True,
        {('sim', 'Q'), ('sim', 'P'), ('reconciled', 'Q')}),
    ('phase',         'Phase ∠·',             True,
        {('measured', 'Q'), ('sim', 'Q'), ('sim', 'P'),
         ('sim_carved', 'Q'), ('reconciled', 'Q')}),
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
         ('sim_carved', 'Q'), ('reconciled', 'Q')}),
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
# Reconciliation outputs (results/dc_reconciled_v2.pkl + ac_field_v2.pkl, see `_load_reconciliation`):
# relative DC uncertainty from the IRLS ensemble, and the bootstrap SD of the winner tile's time gauge.
PROPERTY_DEFS += [
    ('rec_dc_sd',     'DC uncertainty (ensemble SD / |Q\u2032|)', False, {('reconciled', 'Q')}),
    ('rec_gauge_sd',  'Tile time-gauge SD (deg)', False, {('reconciled', 'Q')}),
]

HARMONIC_KEYS = ['DC', 'H1', 'H2', 'H3']
HARMONIC_LABELS = {'DC': 'DC', 'H1': 'H₁', 'H2': 'H₂', 'H3': 'H₃'}
# Phase at DC is undefined; harmonic-class is already H1-resolved → skip.
_SUPPRESSED_SEEN: set = set()


def _note_suppressed(where: str, exc: BaseException) -> None:
    """Report (once per site) an exception the viewer deliberately survives on a computation path, instead of
    swallowing it silently: the result shown is then partial, and the user should know why."""
    if where in _SUPPRESSED_SEEN:
        return
    _SUPPRESSED_SEEN.add(where)
    print(f"  ⚠ viewer: {where} failed ({type(exc).__name__}: {exc}); continuing without it", file=sys.stderr)

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
    """Like _harmonic_snrs but returns amplitudes + phases too.

    Display fit: robust (Huber IRLS, analyze.harmonic.fit_harmonics).  The paper's AC field uses the plain least-squares
    phasors of plexusflow.reconcile.harmonics; on coherent traces (ρ ≥ 0.2, 21S) the two agree to |H1| ±3 % and
    H1 phase ≤ 2.4° (95th pct) — they differ only on noisy traces, which the analysis does not use.

    σ from
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

    Usable if fit_success is not explicitly False and Q_t exists.  (The old pertile quality-tier gate was
    never importable here and has been removed; behaviour is unchanged.)
    """
    if m is None:
        return False
    mv = MeasurementView(m)
    # v4 migration: was m.get('fit_success', True). MeasurementView.fit_success
    # defaults False when the field is absent, so preserve the "assume ok
    # unless explicitly failed" default here by falling back to the raw dict.
    if 'fit_success' in m and not mv.fit_success:
        return False
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
    # All other fields come from the best PIV measurement.
    meas_key = 'measurements_piv'
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
