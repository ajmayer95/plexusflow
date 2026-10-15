"""
Transmission line network solver for pulsatile blood flow.

Implements the Fancher & Katifori (2022) frequency-domain transmission line
equations. Each vessel is a distributed RLC element with:
  - Viscous resistance  r = 8μ/(πR⁴)
  - Inertance           ℓ = ρ/(πR²)
  - Compliance          c = πR²D   (areal distensibility, ΔA/A = D·ΔP)

The dispersion relation is:
  k(ω) = sqrt(-iω c (r + iω ℓ))

For each harmonic n of the cardiac cycle (ω_n = 2π n f0), the network
Laplacian L(ω_n) is assembled and the system solved:
  L(ω_n) P_n = Q_n^(bc)

where Q_n^(bc) are harmonic coefficients of flow at boundary nodes.

DC limit (ω=0) recovers Poiseuille: L(0) = diag(G_ij) Kirchhoff Laplacian.

Parameters
----------
ρ : float
    Blood density ≈ 1060 kg/m³
μ : float
    Blood viscosity = 3.5 mPa·s = 3.5e-3 Pa·s
D : float
    Vessel wall AREAL distensibility (1/Pa), defined by
    ΔA/A = D·ΔP.  Free parameter; empirically ≈ 1.3×10⁻³ 1/Pa for
    HH-stage yolk sac.  NOTE: 2026-05-18 switched from the radius
    convention (ΔR/R = D_R·ΔP, c = 2πR²D_R) to the areal convention
    used in vascular-biomechanics literature.  D_areal = 2 · D_radius
    for the same physics.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

# scaffold-aware BC helpers (first-junction convention support).
# Under the ``first_junction`` bc_convention, boundary flux is injected
# at the first downstream junction and boundary-arm edges are marked
# ``scaffold=True`` so this solver skips them.  Under the legacy
# convention, these helpers pass through and read leaf-side attributes.
from .scaffold import (
    effective_boundary_nodes as _effective_boundary_nodes,
    effective_boundary_type as _effective_boundary_type,
    is_active_edge as _is_active_edge,
    is_first_junction_convention as _is_first_junction_convention,
)


# Physical constants
RHO_BLOOD = 1060.0        # kg/m³
MU_DEFAULT = 3.5e-3       # Pa·s (3.5 mPa·s)


@dataclass
class TransmissionLineResult:
    """Results from transmission line network solve."""
    # Per-edge predicted flow (complex, per harmonic)
    edge_flows: Dict[Tuple[int, int], np.ndarray]  # (u,v) -> [Q_dc, Q_1, Q_2, ...]
    # Per-node predicted pressure (complex, per harmonic)
    node_pressures: Dict[int, np.ndarray]           # node -> [P_dc, P_1, P_2, ...]
    # Derived quantities per edge
    mean_Q: Dict[Tuple[int, int], float]            # nL/s
    amp_Q: Dict[Tuple[int, int], float]             # nL/s (fundamental amplitude)
    PI: Dict[Tuple[int, int], float]                # pulsatility index
    phase: Dict[Tuple[int, int], float]             # phase of fundamental forward-direction Q (rad, sign-canonicalised)
    pressure_phase: Dict[Tuple[int, int], float]    # legacy: midpoint-pressure H1 phase (kept for revertibility)
    WSS: Dict[Tuple[int, int], float]               # time-avg wall shear stress (Pa)
    dissipation: Dict[Tuple[int, int], float]       # r*L*<Q²> (W) viscous dissipation
    pulsatile_cost: Dict[Tuple[int, int], float]    # <Q²>/Q̄² (1 = steady, 2 = PI≈2)
    RPSI: Dict[Tuple[int, int], float]              # max(dQ/dt) / Q̄  (rate of pulsatility)
    eta: Dict[Tuple[int, int], float]               # waveform sharpness η = RPSI/(PI·πf0)
    # Compliance storage per edge
    Q_stored: Dict[Tuple[int, int], np.ndarray]     # complex [0, Q̂_1^stored, Q̂_2^stored, ...]
    storage_fraction: Dict[Tuple[int, int], float]  # |Q̂_1^stored| / |Q̂_1^(μ)|  (fundamental)
    # Parameters used
    f0_hz: float
    n_harmonics: int
    D: float                                         # distensibility (1/Pa)
    mu: float                                        # viscosity (Pa·s)
    # Boundary info
    boundary_nodes: List[int]
    boundary_Q_harmonics: Dict[int, np.ndarray]     # node -> [Q_dc, Q_1, ...]
    # Network info
    n_edges: int
    n_nodes: int


def compute_rpsi_from_harmonics(Q_harmonics, f0_hz, n_eval=1000):
    """Compute RPSI = (Q_sys - Q_dia) / Δt_rise / Q̄ from harmonic coefficients.

    Reconstructs Q(t) over one cycle, finds diastolic min → systolic max,
    computes the average slope of the systolic upstroke normalized by mean Q.

    Parameters
    ----------
    Q_harmonics : array of complex [Q_dc, Q_1, Q_2, ...]
    f0_hz : float, fundamental frequency
    n_eval : int, grid points for reconstruction

    Returns
    -------
    float — RPSI value (1/s units)
    """
    q_dc = Q_harmonics[0].real
    if abs(q_dc) < 1e-15:
        return np.nan
    omega0 = 2.0 * np.pi * f0_hz
    T_cycle = 1.0 / f0_hz
    t_grid = np.linspace(0, T_cycle, n_eval, endpoint=False)

    # Reconstruct Q(t) = Q_dc + Re[ Σ_n Q̂_n exp(i n ω₀ t) ]
    qt = np.full(n_eval, q_dc, dtype=float)
    for n in range(1, len(Q_harmonics)):
        qt += np.real(Q_harmonics[n] * np.exp(1j * n * omega0 * t_grid))

    qt_abs = np.abs(qt)

    # Find diastolic min → systolic max (double to avoid wrap)
    qt_double = np.concatenate([qt_abs, qt_abs])
    i_dia = int(np.argmin(qt_double[:n_eval]))
    search_end = min(i_dia + n_eval, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * T_cycle / n_eval

    if dt_rise < 1e-10 or q_sys <= q_dia:
        return np.nan

    return float((q_sys - q_dia) / dt_rise / abs(q_dc))


def compute_rpsi_from_qt(Q_t, f0_hz, frame_dt_s, smooth_frames=3):
    """Compute RPSI = max(dQ/dt) / Q̄ from a measured Q(t) time series.

    Cycle-averages Q(t) first, then differentiates the averaged waveform.

    Parameters
    ----------
    Q_t : (T,) array — per-frame flow
    f0_hz : float — cardiac frequency
    frame_dt_s : float — frame interval
    smooth_frames : int — moving average before cycle folding

    Returns
    -------
    float — RPSI value (1/s units)
    """
    qt = np.asarray(Q_t, dtype=float)
    # Sign-flip so mean is positive (preserves waveform shape)
    if np.nanmean(qt) < 0:
        qt = -qt
    finite = np.isfinite(qt)
    if finite.sum() < 20:
        return np.nan

    q_mean = float(np.nanmean(qt))
    if q_mean < 1e-15:
        return np.nan

    # Light smoothing
    if smooth_frames > 1:
        from scipy.ndimage import uniform_filter1d
        qt_s = uniform_filter1d(qt, smooth_frames)
    else:
        qt_s = qt

    # Fold onto single cardiac cycle
    cycle_samples = int(round(1.0 / (f0_hz * frame_dt_s)))
    if cycle_samples < 10:
        return np.nan
    n_cycles = len(qt_s) // cycle_samples
    if n_cycles < 2:
        return np.nan

    # Trim to whole cycles and reshape
    qt_trim = qt_s[:n_cycles * cycle_samples].reshape(n_cycles, cycle_samples)
    qt_avg = np.nanmean(qt_trim, axis=0)  # averaged single cycle

    # Find diastolic min → next systolic max on the averaged cycle
    # Double the cycle to avoid wrap-around issues
    qt_double = np.concatenate([qt_avg, qt_avg])

    i_min = int(np.argmin(qt_double[:cycle_samples]))  # diastolic trough

    # Search for systolic peak AFTER the diastolic trough (within one cycle)
    search_end = min(i_min + cycle_samples, len(qt_double))
    i_max = i_min + int(np.argmax(qt_double[i_min:search_end]))

    q_dia = qt_double[i_min]
    q_sys = qt_double[i_max]
    dt_rise = (i_max - i_min) * frame_dt_s

    if dt_rise < frame_dt_s or q_sys <= q_dia:
        return np.nan

    return float((q_sys - q_dia) / dt_rise / q_mean)


def compute_eta_from_harmonics(Q_harmonics, f0_hz, n_eval=1000):
    r"""Compute waveform sharpness index η from harmonic coefficients.

    η = (Q_sys - Q_dia) / Δt_rise / Q_rms

    Same upstroke slope as RPSI, but normalized by Q_rms = √⟨Q²⟩
    instead of Q̄.  Units: s⁻¹.  Well-defined even when Q̄ → 0.

    Parameters
    ----------
    Q_harmonics : array of complex [Q_dc, Q_1, Q_2, ...]
    f0_hz : float, fundamental frequency
    n_eval : int, grid points for reconstruction

    Returns
    -------
    float — η (s⁻¹)
    """
    omega0 = 2.0 * np.pi * f0_hz
    T_cycle = 1.0 / f0_hz
    t_grid = np.linspace(0, T_cycle, n_eval, endpoint=False)

    # Reconstruct Q(t)
    qt = np.full(n_eval, Q_harmonics[0].real, dtype=float)
    for n in range(1, len(Q_harmonics)):
        qt += np.real(Q_harmonics[n] * np.exp(1j * n * omega0 * t_grid))

    q_rms = float(np.sqrt(np.mean(qt**2)))
    if q_rms < 1e-30:
        return np.nan

    # Find diastolic min → systolic max (same as RPSI)
    qt_double = np.concatenate([qt, qt])
    i_dia = int(np.argmin(qt_double[:n_eval]))
    search_end = min(i_dia + n_eval, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * T_cycle / n_eval

    if dt_rise < 1e-10 or q_sys <= q_dia:
        return np.nan

    upstroke_slope = (q_sys - q_dia) / dt_rise
    return float(upstroke_slope / q_rms)


def compute_eta_from_qt(Q_t, f0_hz, frame_dt_s, smooth_frames=3):
    r"""Compute waveform sharpness index η from a measured Q(t) time series.

    η = (Q_sys - Q_dia) / Δt_rise / Q_rms

    Same upstroke slope as RPSI, but normalized by Q_rms = √⟨Q²⟩
    instead of Q̄.  Units: s⁻¹.  Well-defined even when Q̄ → 0.

    Parameters
    ----------
    Q_t : (T,) array — per-frame flow
    f0_hz : float — cardiac frequency
    frame_dt_s : float — frame interval
    smooth_frames : int — moving average before cycle folding

    Returns
    -------
    float — η (s⁻¹)
    """
    qt = np.asarray(Q_t, dtype=float)
    if np.nanmean(qt) < 0:
        qt = -qt
    finite = np.isfinite(qt)
    if finite.sum() < 20:
        return np.nan

    # Light smoothing
    if smooth_frames > 1:
        from scipy.ndimage import uniform_filter1d
        qt_s = uniform_filter1d(qt, smooth_frames)
    else:
        qt_s = qt

    # Fold onto single cardiac cycle
    cycle_samples = int(round(1.0 / (f0_hz * frame_dt_s)))
    if cycle_samples < 10:
        return np.nan
    n_cycles = len(qt_s) // cycle_samples
    if n_cycles < 2:
        return np.nan

    qt_trim = qt_s[:n_cycles * cycle_samples].reshape(n_cycles, cycle_samples)
    qt_avg = np.nanmean(qt_trim, axis=0)

    # Smooth the cycle average for stable slope detection
    from scipy.ndimage import uniform_filter1d as uf1d
    qt_smooth = uf1d(qt_avg, max(3, cycle_samples // 30))

    q_rms = float(np.sqrt(np.mean(qt_smooth**2)))
    if q_rms < 1e-30:
        return np.nan

    # Find diastolic min → systolic max (same logic as RPSI)
    qt_double = np.concatenate([qt_smooth, qt_smooth])
    i_dia = int(np.argmin(qt_double[:cycle_samples]))
    search_end = min(i_dia + cycle_samples, len(qt_double))
    i_sys = i_dia + int(np.argmax(qt_double[i_dia:search_end]))

    q_dia = qt_double[i_dia]
    q_sys = qt_double[i_sys]
    dt_rise = (i_sys - i_dia) * frame_dt_s

    if dt_rise < frame_dt_s or q_sys <= q_dia:
        return np.nan

    upstroke_slope = (q_sys - q_dia) / dt_rise
    return float(upstroke_slope / q_rms)


def _get_edge_geometry(G: nx.Graph, u: int, v: int,
                       radii_m: Optional[Dict[Tuple[int, int], float]] = None,
                       ) -> Tuple[float, float]:
    """Get radius (m) and length (m) for an edge.

    If *radii_m* is provided, use the override radius (already in meters)
    instead of reading from the graph.  Length always comes from the graph.
    """
    from plexusflow.analyze.constants import PX_SIZE_UM
    px_to_m = PX_SIZE_UM * 1e-6  # µm/px * m/µm

    data = G.edges[u, v]

    # --- Radius ---
    R_m = None
    if radii_m is not None:
        R_m = radii_m.get((u, v))
        if R_m is None:
            R_m = radii_m.get((v, u))
        if R_m is not None and (R_m <= 0 or not np.isfinite(R_m)):
            return None, None

    if R_m is None:
        # Prefer tile-corrected radius if available
        R_px = data.get('radius_px_true')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('R_fit_px')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('radius_px')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            R_px = data.get('radius')
        if R_px is None or not np.isfinite(R_px) or R_px <= 0:
            return None, None
        R_m = R_px * px_to_m

    # --- Length ---
    # Prefer tile-corrected length if available
    L_px = data.get('length_true')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        L_px = data.get('length')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        L_px = data.get('path_length_px')
    if L_px is None or not np.isfinite(L_px) or L_px <= 0:
        return None, None

    L_m = L_px * px_to_m

    return R_m, L_m


def _per_length_params(R_m: float, mu: float = MU_DEFAULT,
                       rho: float = RHO_BLOOD, D=1e-6):
    """Compute per-unit-length transmission line parameters.

    Parameters
    ----------
    R_m : float
        Vessel radius in meters
    mu : float
        Viscosity in Pa·s
    rho : float
        Density in kg/m³
    D : float or callable
        Distensibility in 1/Pa, or a callable D(R_m) implementing a
        compliance model (e.g. thin-wall: D = 2R / (E*h) → D(R) = 2R/(E*h)).

    Returns
    -------
    r : float
        Resistance per unit length [Pa·s/m⁴]
    ell : float
        Inertance per unit length [kg/m⁵]
    c : float
        Compliance per unit length [m³/(Pa·m)] = [m²/Pa]
    """
    D_eff = D(R_m) if callable(D) else D
    A = np.pi * R_m**2
    r = 8 * mu / (np.pi * R_m**4)      # Poiseuille resistance/length
    ell = rho / A                        # inertance/length
    # Compliance/length = ∂A/∂P.  D is the AREAL distensibility
    # (ΔA/A = D·ΔP), so ∂A/∂P = A·D = πR²·D directly.
    # NOTE: prior convention was radius distensibility (ΔR/R = D·ΔP),
    # which gave c = 2πR²D.  Switched 2026-05-18 to areal convention
    # to match common vascular-biomechanics literature.  D values
    # under the new convention are 2× the old ones for the same
    # physics (D_areal = 2·D_radius).
    c = np.pi * R_m**2 * D_eff           # compliance/length: ∂A/∂P = πR²D [m²/Pa]
    return r, ell, c


def _vessel_admittance(R_m: float, L_m: float, omega: float,
                       mu: float = MU_DEFAULT, rho: float = RHO_BLOOD,
                       D: float = 1e-6) -> complex:
    """Compute the 2-port admittance for a vessel at frequency omega.

    For a transmission line segment of length L with propagation constant
    k(ω) = sqrt(-iω c (r + iω ℓ)), the characteristic impedance is
    Z_c = sqrt((r + iω ℓ) / (iω c)), and the 2-port admittance matrix is:

        Y = (1/Z_c) * [[coth(kL), -csch(kL)],
                        [-csch(kL), coth(kL)]]

    For the Kirchhoff Laplacian assembly, the diagonal contribution is
    Y_diag = coth(kL)/Z_c and off-diagonal is Y_off = -csch(kL)/Z_c.

    At DC (omega=0): Y_diag = Y_off = G = πR⁴/(8μL) (Poiseuille conductance).

    Returns
    -------
    Y_diag, Y_off : complex
        Diagonal and off-diagonal admittance entries.
    """
    r, ell, c = _per_length_params(R_m, mu, rho, D)

    if abs(omega) < 1e-12:
        # DC limit: pure Poiseuille conductance
        G = np.pi * R_m**4 / (8 * mu * L_m)
        return G, -G

    # Propagation constant: k = sqrt(-iω c (r + iω ℓ))
    z = r + 1j * omega * ell       # series impedance per length
    y = 1j * omega * c             # shunt admittance per length
    gamma_sq = z * y               # k² = z*y
    gamma = np.sqrt(gamma_sq)      # propagation constant k

    # Characteristic impedance: Z_c = sqrt(z/y)
    Z_c = np.sqrt(z / y)

    kL = gamma * L_m

    # Numerical safety for small kL (Taylor expand)
    if abs(kL) < 1e-6:
        # coth(x) ≈ 1/x + x/3, csch(x) ≈ 1/x - x/6
        # Y_diag = (1/Z_c)(1/kL + kL/3), Y_off = -(1/Z_c)(1/kL - kL/6)
        Y_diag = (1.0 / Z_c) * (1.0 / kL + kL / 3.0)
        Y_off = -(1.0 / Z_c) * (1.0 / kL - kL / 6.0)
    elif abs(kL.real) > 500:
        # Heavily damped: coth(kL) → 1, csch(kL) → 0
        # Vessel is so lossy that the far end is effectively decoupled.
        Y_diag = 1.0 / Z_c
        Y_off = 0.0
        return Y_diag, Y_off
    else:
        # Use exp form for numerical stability when Re(kL) is large
        e_pos = np.exp(kL)
        e_neg = np.exp(-kL)
        sinh_kL = (e_pos - e_neg) / 2.0
        cosh_kL = (e_pos + e_neg) / 2.0

        if abs(sinh_kL) < 1e-30:
            # Degenerate case
            G = np.pi * R_m**4 / (8 * mu * L_m)
            return G, -G

        coth_kL = cosh_kL / sinh_kL
        csch_kL = 1.0 / sinh_kL

        Y_diag = coth_kL / Z_c
        Y_off = -csch_kL / Z_c

    return Y_diag, Y_off


def _assemble_laplacian(G: nx.Graph, omega: float,
                        edge_list: List[Tuple[int, int]],
                        node_to_idx: Dict[int, int],
                        mu: float, rho: float, D: float,
                        radii_m: Optional[Dict[Tuple[int, int], float]] = None,
                        ) -> csr_matrix:
    """Assemble the frequency-dependent network admittance matrix (Laplacian).

    L(ω)_ii = Σ_j Y_diag(edge_ij, ω)
    L(ω)_ij = Y_off(edge_ij, ω)
    """
    N = len(node_to_idx)
    L = lil_matrix((N, N), dtype=complex)

    for u, v in edge_list:
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is None:
            continue

        Y_diag, Y_off = _vessel_admittance(R_m, L_m, omega, mu, rho, D)

        i = node_to_idx[u]
        j = node_to_idx[v]

        L[i, i] += Y_diag
        L[j, j] += Y_diag
        L[i, j] += Y_off
        L[j, i] += Y_off

    return L.tocsr()


def _extract_boundary_harmonics(
    G: nx.Graph,
    boundary_nodes: List[int],
    f0_hz: float,
    n_harmonics: int,
    frame_dt: float,
) -> Dict[int, np.ndarray]:
    """Extract flow harmonic coefficients at boundary nodes from Q_t data.

    For each boundary node, find all edges connected to it that have Q_t.
    Sum the flows (with correct sign convention) to get net Q(t) at the node.
    Then decompose into harmonics via fit_harmonics.

    Returns
    -------
    boundary_Q : dict
        node -> complex array [Q_dc, Q_1, Q_2, ..., Q_n_harmonics]
        Q_dc is real (mean flow in nL/s)
        Q_k is complex (A_k - i*B_k convention)
    """
    from plexusflow.analyze.harmonic import fit_harmonics

    boundary_Q = {}

    # Determine reference tile (tile 14 by default, or from graph metadata)
    ref_tile = G.graph.get('reference_vid', 14)

    for node in boundary_nodes:
        # If the node already carries a `bc_harmonics` attribute (from
        # the post-M5 bc_harmonics population, KCL reconstruction, or
        # first-junction promotion), use it directly.  Sign convention:
        # `bc_harmonics` stores "injected INTO network" (positive DC at
        # sources, negative DC at sinks).  This matches the solver's
        # convention directly, so no sign flip is needed.
        stored_bc = G.nodes[node].get('bc_harmonics')
        if stored_bc is not None:
            arr = np.asarray(stored_bc, dtype=complex)
            target = n_harmonics + 1
            if len(arr) >= target:
                arr = arr[:target]
            else:
                padded = np.zeros(target, dtype=complex)
                padded[:len(arr)] = arr
                arr = padded
            boundary_Q[node] = arr
            continue

        # For boundary edges, prefer the reference tile's PIV measurement
        # to ensure consistent f0 and Q_t across all boundaries.
        # IMPORTANT: do NOT take abs() — preserve the phase information
        # in the waveform. Q_t from PIV is already positive (magnitude).
        Q_raw = None

        for neighbor in G.neighbors(node):
            data = G.edges[node, neighbor]

            # Try reference tile's PIV Q_t first
            piv_list = data.get('measurements_piv', [])
            ref_meas = [m for m in piv_list if m.get('tile_id') == ref_tile]
            if ref_meas:
                Qt_src = ref_meas[0].get('Q_t')
                if Qt_src is not None and len(Qt_src) > 20:
                    Q_raw = np.asarray(Qt_src, dtype=float)
                    continue

            # Fallback: top-level Q_t
            Q_t = data.get('Q_t')
            if Q_t is None or len(Q_t) == 0:
                continue
            Q_t = np.asarray(Q_t, dtype=float)

            if Q_raw is None:
                Q_raw = Q_t.copy()
            else:
                n_min = min(len(Q_raw), len(Q_t))
                Q_raw = Q_raw[:n_min] + Q_t[:n_min]

        if Q_raw is None or len(Q_raw) < 20:
            boundary_Q[node] = np.zeros(n_harmonics + 1, dtype=complex)
            continue

        # Ensure Q is positive magnitude (PIV stores positive, but
        # kymograph may have signed values)
        if np.nanmean(Q_raw) < 0:
            Q_raw = -Q_raw

        # Solver convention: positive Q = flow injected into network.
        # Sources inject (positive), sinks extract (negative).
        bt = G.nodes[node].get('boundary_type', 'source')
        solver_sign = -1.0 if bt == 'sink' else +1.0
        Q_total = solver_sign * Q_raw

        # Direct harmonic fit to raw Q_t (no cycle-averaging — that inflates
        # harmonic amplitudes by phase-locking noisy cycles)
        hr = fit_harmonics(Q_total, frame_dt, f0_hz, K=n_harmonics,
                           loss='huber', include_dc=True)

        # Pack into complex array: [DC, H1, H2, ..., Hn]
        coeffs = np.zeros(n_harmonics + 1, dtype=complex)
        coeffs[0] = float(np.nanmean(Q_total))  # DC from nanmean

        # fit_harmonics convention: signal = A·cos(kωt) + B·sin(kωt)
        # Complex Fourier convention: Re[Q̂·e^(ikωt)] = A·cos(kωt) - B·sin(kωt)
        # So Q̂ = A - iB (negate the imaginary part)
        for h in hr['harmonics']:
            k = h['k']
            if k <= n_harmonics:
                coeffs[k] = complex(h['A'], -h['B'])

        boundary_Q[node] = coeffs

    return boundary_Q


def _classify_boundary_nodes(
    G: nx.Graph,
    boundary_nodes: List[int],
) -> Tuple[List[int], List[int]]:
    """Classify boundary nodes as sources (arterial) or sinks (venous).

    Sources: flow leaves the boundary node into the network (flow_from == node).
    Sinks:   flow enters the boundary node from the network (flow_to == node).

    Hard-coded override for 21 somites (4 boundary nodes):
      Sources: 59735, 59736  (arterial inlets, lower y / upper in image)
      Sinks:   59734, 59737  (venous outlets)

    Returns
    -------
    sources, sinks : list of int
    """
    # Hard-coded for 21 somites
    _KNOWN_SOURCES = {59735, 59736}
    _KNOWN_SINKS = {59734, 59737}

    # First try: use effective boundary_type node attribute if available.
    # Under first-junction convention this reads boundary_type_effective
    # (from the promoted junctions); under legacy it reads boundary_type
    # (from the anatomical leaves).
    sources = [n for n in boundary_nodes
               if _effective_boundary_type(G, n) == 'source']
    sinks = [n for n in boundary_nodes
             if _effective_boundary_type(G, n) == 'sink']
    if sources and sinks:
        return sources, sinks

    # Hard-coded for 21 somites
    bnd_set = set(boundary_nodes)
    if bnd_set == (_KNOWN_SOURCES | _KNOWN_SINKS):
        sources = [n for n in boundary_nodes if n in _KNOWN_SOURCES]
        sinks = [n for n in boundary_nodes if n in _KNOWN_SINKS]
        return sources, sinks

    # Fallback: classify from measured flow direction
    sources = []
    sinks = []
    for node in boundary_nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            sinks.append(node)
            continue
        # Check majority flow direction across all incident edges
        net_outflow = 0.0
        for nb in neighbors:
            data = G.edges[node, nb]
            flow_from = data.get('flow_from')
            mean_Q = data.get('mean_Q', 0)
            if flow_from == node:
                net_outflow += abs(mean_Q)
            else:
                net_outflow -= abs(mean_Q)
        if net_outflow > 0:
            sources.append(node)
        else:
            sinks.append(node)

    return sources, sinks


def solve_transmission_line(
    G: nx.Graph,
    D=1e-6,
    n_harmonics: int = 3,
    f0_hz: Optional[float] = None,
    mu: float = MU_DEFAULT,
    rho: float = RHO_BLOOD,
    boundary_nodes: Optional[List[int]] = None,
    bc_harmonics_override: Optional[Dict[int, np.ndarray]] = None,
    radii_m: Optional[Dict[Tuple[int, int], float]] = None,
    verbose: bool = True,
    E_wall: Optional[float] = None,
    h_wall: float = 1e-6,
    sink_pressure_bc: Optional[float] = None,
    source_pressure_bc: Optional[float] = None,
    sink_impedance=None,
    merged_boundary: bool = False,
    tied_pressure_nodes: Optional[List[int]] = None,
) -> TransmissionLineResult:
    """Solve the transmission line network at DC + n_harmonics.

    Boundary conditions:
      - Q(t) prescribed at ALL boundary nodes (sources and sinks)
      - The outlier boundary node (largest |Q_dc| deviation) is uniformly
        rescaled so that DC flow is conserved: ΣQ_source = ΣQ_sink

    If *sink_pressure_bc* is not None:
      - Sink (venous) boundary nodes get Dirichlet P = sink_pressure_bc (Pa)
        instead of Q BCs.  Only source (arterial) nodes keep Q BCs.
      - DC conservation adjustment is applied only among source nodes.
      - The P-BC nodes serve as the DC gauge (no separate pinning needed).

    If *sink_impedance* is not None:
      - Sink nodes get impedance BCs: P = Z_ven(ω) · Q at each harmonic.
        Implemented by adding ground admittance Y = 1/Z_ven to the
        Laplacian diagonal at sink nodes (Q_rhs = 0).
      - Can be a scalar (frequency-independent R) or callable Z(omega)
        returning complex impedance.  For a Windkessel:
        Z(ω) = R_ven / (1 + iω R_ven C_ven)
      - Z → 0 approaches P = 0.  Z → ∞ approaches free node.
      - Takes precedence over sink_pressure_bc if both are set.

    If *merged_boundary* is True:
      - All boundary nodes (arterial + venous) are merged into a single
        node with shared pressure P_u.  Total arterial Q is prescribed
        at this merged node.  Venous flows become predictions.
      - DC: P_u pinned to 0 (gauge).  Flow distributes among veins by topology.
      - AC: P_u is solved (shared oscillation).  Venous pulsatility predicted.
      - Takes precedence over all other sink BC modes.
      - AC harmonics: compliance makes L(ω) non-singular, so Q prescribed
        at all boundary nodes works directly (no pressure pinning needed)
      - DC: L(0) is singular (Kirchhoff), so pin P=0 at one interior node

    Algorithm:
    1. Identify and classify boundary nodes (source vs sink)
    2. Extract Q(t) harmonics at ALL boundary nodes
    3. Rescale outlier node's harmonics to enforce DC conservation
    4. For each harmonic n:
       a. Assemble L(ω_n)
       b. DC only: pin P=0 at one interior node
       c. Prescribe Q at all boundary nodes in RHS
       d. Solve for all pressures
    5. Reconstruct edge flows from pressure differences

    Parameters
    ----------
    G : nx.Graph
        Mosaic graph with measurements (Q_t, geometry, boundary_type)
    D : float
        Distensibility in 1/Pa. This is the free parameter.
    n_harmonics : int
        Number of cardiac harmonics to solve (plus DC)
    f0_hz : float, optional
        Heart rate. If None, read from G.graph['tile_f0s'] (median).
    mu : float
        Blood viscosity in Pa·s
    rho : float
        Blood density in kg/m³
    boundary_nodes : list, optional
        Override boundary node detection
    bc_harmonics_override : dict, optional
        Pre-computed boundary harmonics {node_id: complex_array}.
        If provided, skips automatic extraction from Q_t data.
        Each array is [Q_dc, Q_1, ..., Q_n] with n_harmonics+1 entries.
    E_wall : float, optional
        If given, overrides D with the thin-wall compliance model
        D(R) = 2R / (E_wall * h_wall).  E_wall in Pa (e.g. 20e3 = 20 kPa).
    h_wall : float
        Endothelial wall thickness in meters (default 1 µm).
        Only used when E_wall is provided.
    verbose : bool

    Returns
    -------
    TransmissionLineResult
    """
    from plexusflow.analyze.constants import FRAME_DT_S

    # Thin-wall compliance model: D(R) = 2R / (E_wall * h_wall)
    if E_wall is not None:
        D = lambda R, _E=E_wall, _h=h_wall: 2.0 * R / (_E * _h)

    # --- 1. Identify boundary nodes ---
    # Under first-junction convention these are the promoted junctions
    # (nodes with boundary_type_effective); under legacy they are the
    # anatomical leaves (nodes with boundary_type).
    if boundary_nodes is None:
        boundary_nodes = _effective_boundary_nodes(G)

    if len(boundary_nodes) < 2:
        raise ValueError(f"Need >= 2 boundary nodes, found {len(boundary_nodes)}")

    # --- 2. Classify sources vs sinks ---
    source_nodes, sink_nodes = _classify_boundary_nodes(G, boundary_nodes)

    if not source_nodes:
        raise ValueError("No source (arterial) boundary nodes found")
    if not sink_nodes:
        raise ValueError("No sink (venous) boundary nodes found")

    # --- 3. Determine f0 ---
    if f0_hz is None:
        tile_f0s = G.graph.get('tile_f0s', {})
        if tile_f0s:
            f0_hz = float(np.median(list(tile_f0s.values())))
        else:
            raise ValueError("No f0_hz provided and no tile_f0s in graph")

    if verbose:
        D_disp = f"{D(10e-6):.2e} @10µm" if callable(D) else f"{D:.2e}"
        print(f"Transmission line solve: D={D_disp} 1/Pa, f0={f0_hz:.3f} Hz, "
              f"{n_harmonics} harmonics, µ={mu*1e3:.1f} mPa·s")
        print(f"  Sources: {source_nodes}")
        print(f"  Sinks:   {sink_nodes}")

    # --- 4. Build node/edge lists ---
    boundary_set = set(boundary_nodes)
    boundary_edges = set()
    # Keep track of boundary edges for post-hoc assignment, but do NOT
    # exclude them from the solve.  BCs are applied directly at boundary nodes.
    bc_remap = {}  # identity: boundary node -> itself (no remap)
    for bn in boundary_nodes:
        nbrs = list(G.neighbors(bn))
        if len(nbrs) == 1:
            boundary_edges.add((bn, nbrs[0]))
            boundary_edges.add((nbrs[0], bn))
        elif len(nbrs) > 1:
            best_nbr = max(nbrs,
                           key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
            boundary_edges.add((bn, best_nbr))
            boundary_edges.add((best_nbr, bn))

    # All edges with valid geometry, excluding scaffold edges (which
    # carry a fake R from the anatomical arm and shouldn't enter the
    # linear-RC assembly).  Under legacy convention, no edges have
    # ``scaffold=True`` so this is a no-op filter.
    edge_list = []
    n_scaffold_skipped = 0
    for u, v, ed in G.edges(data=True):
        if not _is_active_edge(ed):
            n_scaffold_skipped += 1
            continue
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is not None:
            edge_list.append((u, v))

    # Node set: only nodes that participate in at least one active edge.
    # Scaffold-only nodes (typically anatomical leaves after first-junction
    # promotion) would otherwise contribute a zero row/column to the
    # Laplacian and make it singular.
    active_node_set: set = set()
    for u, v in edge_list:
        active_node_set.add(u)
        active_node_set.add(v)
    # Boundary nodes must also be in the solver's node set even if all
    # their edges got filtered somehow (safety belt).
    active_node_set.update(boundary_nodes)
    all_nodes = [n for n in G.nodes() if n in active_node_set]
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)
    n_dropped_nodes = G.number_of_nodes() - N

    if verbose:
        conv = 'first_junction' if _is_first_junction_convention(G) else 'legacy'
        print(f"  {len(edge_list)}/{G.number_of_edges()} edges in solve "
              f"({len(boundary_nodes)} boundary nodes, "
              f"{len(boundary_edges)//2} boundary edges, "
              f"{n_scaffold_skipped} scaffold skipped, "
              f"{n_dropped_nodes} scaffold-isolated nodes dropped, "
              f"convention={conv})")

    # --- 5. Extract flow harmonics at ALL boundary nodes ---
    if bc_harmonics_override is not None:
        # Use user-supplied harmonics, padding/truncating to n_harmonics+1
        bc_harmonics = {}
        for n in boundary_nodes:
            if n in bc_harmonics_override:
                arr = np.asarray(bc_harmonics_override[n], dtype=complex)
                target_len = n_harmonics + 1
                if len(arr) >= target_len:
                    bc_harmonics[n] = arr[:target_len]
                else:
                    padded = np.zeros(target_len, dtype=complex)
                    padded[:len(arr)] = arr
                    bc_harmonics[n] = padded
            else:
                bc_harmonics[n] = np.zeros(n_harmonics + 1, dtype=complex)
        if verbose:
            print("  Using user-supplied boundary harmonics")
    else:
        bc_harmonics = _extract_boundary_harmonics(
            G, boundary_nodes, f0_hz, n_harmonics, FRAME_DT_S)

    if verbose:
        for node, coeffs in bc_harmonics.items():
            role = 'Source' if node in source_nodes else 'Sink'
            print(f"  {role} {node}: Q_dc={coeffs[0].real:.3f} nL/s, "
                  f"|Q_1|={abs(coeffs[1]):.3f} nL/s")

    # --- 6. Determine sink BC mode ---
    # Priority: merged_boundary > sink_impedance > pressure BCs > Q (default)
    use_merged_bc = merged_boundary
    use_impedance_bc = sink_impedance is not None and not use_merged_bc
    use_pressure_bc = (sink_pressure_bc is not None
                       and not use_impedance_bc and not use_merged_bc)
    # `tied_pressure_nodes` is a passive BC on those nodes (they act as
    # a flow drain via the tie, absorbing whatever the network delivers
    # at whatever shared pressure the solve chooses).  Flag as such so
    # the DC conservation step doesn't rescale the arterial input.
    use_tied_bc = bool(tied_pressure_nodes) and len(tied_pressure_nodes) >= 2
    # Source-side Dirichlet — pressure prescribed at A too.  Used with
    # a matching sink_pressure_bc for the Dirichlet-tie mode where ΔP =
    # P_A − P_V drives the network.  Q at all boundary nodes becomes a
    # solve output.
    use_source_pressure_bc = (source_pressure_bc is not None
                              and use_pressure_bc)
    sink_has_passive_bc = (use_impedance_bc or use_pressure_bc
                            or use_merged_bc or use_tied_bc)

    # Coerce pressure-BC arguments into per-harmonic complex arrays for
    # unified downstream handling.  A scalar means "DC only, all AC = 0".
    # A dict {node: scalar/array} means per-node pressure BC — used for
    # ΔP sensitivity sweeps at the venous side.
    def _coerce_p_bc(x):
        if x is None: return None
        if isinstance(x, dict):
            return {int(k): _coerce_p_bc(v) for k, v in x.items()}
        arr = np.zeros(n_harmonics + 1, dtype=complex)
        if np.ndim(x) == 0:
            arr[0] = complex(x)
        else:
            xa = np.asarray(x, dtype=complex)
            k = min(len(xa), n_harmonics + 1)
            arr[:k] = xa[:k]
        return arr
    p_bc_sink_arr = _coerce_p_bc(sink_pressure_bc)
    p_bc_source_arr = (_coerce_p_bc(source_pressure_bc)
                        if use_source_pressure_bc else None)

    if use_merged_bc:
        q_bc_nodes = list(source_nodes)  # only sources prescribe Q
        p_bc_nodes = []
        z_bc_nodes = []
        if verbose:
            print("  Merged boundary BC: all boundary nodes share P_u")
            print("    Arterial Q prescribed (total), venous Q predicted")
    elif use_impedance_bc:
        q_bc_nodes = list(source_nodes)
        p_bc_nodes = []
        z_bc_nodes = list(sink_nodes)
        z_ven = sink_impedance
        if verbose:
            z_disp = f"{z_ven:.2e}" if not callable(z_ven) else "Z(ω)"
            print(f"  Sink BC: impedance Z={z_disp} at {z_bc_nodes}")
    elif use_pressure_bc:
        # Sinks always get Dirichlet.  Sources: Q by default, unless
        # source_pressure_bc is set (Dirichlet-tie ΔP mode).
        if use_source_pressure_bc:
            q_bc_nodes = []                    # no prescribed Q anywhere
            p_bc_nodes = list(source_nodes) + list(sink_nodes)
        else:
            q_bc_nodes = list(source_nodes)
            p_bc_nodes = list(sink_nodes)
        z_bc_nodes = []
        if isinstance(p_bc_sink_arr, dict):
            # per-node pressure BC — pick any value for the "legacy
            # display" scalar (first sink's DC); verbose shows the dict
            _any_arr = next(iter(p_bc_sink_arr.values()))
            p_bc_value = float(_any_arr[0].real) if _any_arr is not None else 0.0
            if verbose:
                _summary = {int(k): float(v[0].real)
                             for k, v in p_bc_sink_arr.items()}
                print(f"  Sink BC: per-node Dirichlet P_DC = {_summary} Pa")
        else:
            p_bc_value = float(p_bc_sink_arr[0].real)   # legacy display only
            if verbose:
                if use_source_pressure_bc:
                    dp = (float(p_bc_source_arr[0].real)
                           - float(p_bc_sink_arr[0].real))
                    print(f"  Dirichlet-tie BC: ΔP(A−V) = {dp:.2f} Pa "
                          f"(A: {source_nodes}, V: {sink_nodes})")
                else:
                    print(f"  Sink BC: Dirichlet P_DC={p_bc_value:.2f} Pa "
                          f"at {p_bc_nodes} "
                          f"(AC amps = {np.abs(p_bc_sink_arr[1:])})")
    else:
        q_bc_nodes = list(boundary_nodes)
        p_bc_nodes = []
        z_bc_nodes = []

    # Tied-pressure nodes get no Q-BC — they're network outputs,
    # constrained only by the penalty admittance between them.  Remove
    # them from q_bc_nodes so the RHS isn't polluted with Q=0 that
    # would then be rescaled by the DC conservation step.
    if tied_pressure_nodes:
        tied_set = set(tied_pressure_nodes)
        q_bc_nodes = [n for n in q_bc_nodes if n not in tied_set]
        p_bc_nodes = [n for n in p_bc_nodes if n not in tied_set]

    # --- 6a. DC conservation (among Q-BC nodes only) ---
    # When sinks have passive BCs (P or Z), conservation is automatic.
    # With all-Q BCs, distribute imbalance proportionally.
    Q_dc_total = sum(bc_harmonics[n][0].real for n in q_bc_nodes)

    if abs(Q_dc_total) > 1e-10 and not sink_has_passive_bc:
        abs_dc = {n: abs(bc_harmonics[n][0].real) for n in q_bc_nodes}
        total_abs = sum(abs_dc.values())

        if verbose:
            print(f"  DC imbalance: {Q_dc_total:.3f} nL/s")

        if total_abs > 1e-10:
            for n in q_bc_nodes:
                share = abs_dc[n] / total_abs
                correction = share * Q_dc_total
                Q_dc_old = bc_harmonics[n][0].real
                Q_dc_new = Q_dc_old - correction
                if abs(Q_dc_old) > 1e-10:
                    scale = Q_dc_new / Q_dc_old
                    bc_harmonics[n] = bc_harmonics[n] * scale
                    if verbose:
                        print(f"    Node {n}: {Q_dc_old:.3f} -> {Q_dc_new:.3f} nL/s "
                              f"(x{scale:.3f})")
    elif sink_has_passive_bc and verbose:
        bc_type = ('merged' if use_merged_bc else
                   'impedance' if use_impedance_bc else 'pressure')
        print(f"  Source Q_dc total: {Q_dc_total:.3f} nL/s "
              f"(sink flows determined by {bc_type} solve)")

    # --- 6b. Apply BCs directly at boundary nodes (no remap) ---
    bc_harmonics_remapped = {bn: bc_harmonics[bn].copy() for bn in q_bc_nodes}
    solve_bc_nodes = list(q_bc_nodes)
    # Nodes whose boundary edge flow comes from the solve (not overridden)
    solve_flow_bc_nodes = set(p_bc_nodes) | set(z_bc_nodes)
    if use_merged_bc:
        # In merged mode, ALL boundary node flows come from the solve
        solve_flow_bc_nodes = set(boundary_nodes)
    if use_tied_bc:
        # Tied nodes are also "flow from solve" — their Q is not
        # prescribed, it's the network's response to the pressure tie.
        solve_flow_bc_nodes |= set(tied_pressure_nodes)

    # --- 7. Solve at each frequency ---
    # Include ALL graph nodes in node_P for pressure storage (even excluded ones)
    node_P = {n: np.zeros(n_harmonics + 1, dtype=complex) for n in G.nodes()}
    bc_set = set(solve_bc_nodes)

    frequencies = [0.0] + [2 * np.pi * k * f0_hz for k in range(1, n_harmonics + 1)]

    # Pick an interior node to pin P=0 at DC (needed because Kirchhoff L is singular)
    pin_node = None
    for n in all_nodes:
        if n not in bc_set:
            pin_node = n
            break

    # Precompute merged-BC penalty admittance
    if use_merged_bc:
        if verbose:
            print(f"  Merged BC: {len(boundary_nodes)} boundary nodes "
                  f"connected by penalty admittance (shared P_u)")
        B_idxs = [node_to_idx[bn] for bn in boundary_nodes]

    for harm_idx, omega in enumerate(frequencies):
        # Assemble full Laplacian on all N nodes
        L_full = _assemble_laplacian(G, omega, edge_list, node_to_idx,
                                     mu, rho, D, radii_m=radii_m)

        if use_merged_bc:
            # --- Merged boundary: penalty-edge approach ---
            # Add large admittance between all pairs of boundary nodes
            # to force them to share a common pressure P_u.
            # Prescribe Q at arterial nodes, Q=0 at venous nodes.
            L_solve = L_full.tolil()

            # Determine penalty magnitude: ~1000× max diagonal entry
            diag_abs = np.abs(L_full.diagonal())
            Y_big = 1e3 * max(diag_abs.max(), 1e-20)

            # Add penalty edges between all boundary pairs
            for i_b in range(len(B_idxs)):
                for j_b in range(i_b + 1, len(B_idxs)):
                    bi, bj = B_idxs[i_b], B_idxs[j_b]
                    L_solve[bi, bj] -= Y_big   # off-diagonal (negative)
                    L_solve[bj, bi] -= Y_big
                    L_solve[bi, bi] += Y_big   # diagonal correction
                    L_solve[bj, bj] += Y_big

            # RHS: arterial Q at source nodes, 0 at sinks and interior
            Q_solve = np.zeros(N, dtype=complex)
            for sn in source_nodes:
                idx = node_to_idx[sn]
                Q_solve[idx] = bc_harmonics[sn][harm_idx] * 1e-12

            # DC: pin P=0 at one boundary node (gauge reference)
            if harm_idx == 0:
                pin_b = B_idxs[0]
                L_solve[pin_b, :] = 0
                L_solve[pin_b, pin_b] = 1.0
                Q_solve[pin_b] = 0.0

            L_solve = L_solve.tocsr()
            try:
                P = spsolve(L_solve, Q_solve)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: Merged solve failed at harmonic "
                          f"{harm_idx}: {e}")
                continue

            if verbose and harm_idx == 0:
                P_abs = np.abs(P)
                P_boundary = [P[bi] for bi in B_idxs]
                P_spread = max(abs(p) for p in P_boundary) - min(abs(p) for p in P_boundary)
                n_nonzero = np.count_nonzero(P_abs > 1e-20)
                print(f"  Merged DC: {n_nonzero}/{N} nodes |P|>0, "
                      f"max|P|={P_abs.max():.4e}, "
                      f"boundary P spread={P_spread:.2e} "
                      f"(Y_big={Y_big:.2e})")
            elif verbose and harm_idx == 1:
                P_u_mean = np.mean([P[bi] for bi in B_idxs])
                print(f"  Merged AC h=1: |P_u|={abs(P_u_mean):.4e} Pa")
        else:
            # --- Standard per-node solve ---
            # Build RHS: Q at Q-BC nodes, 0 elsewhere
            Q_rhs = np.zeros(N, dtype=complex)
            for node in solve_bc_nodes:
                idx = node_to_idx[node]
                Q_rhs[idx] = bc_harmonics_remapped[node][harm_idx]

            Q_rhs_si = Q_rhs * 1e-12
            L_solve = L_full.tolil()
            Q_solve = Q_rhs_si.copy()

            # Impose Dirichlet P at pressure-BC nodes.
            # Sinks all share `p_bc_sink_arr[harm_idx]`; if sources are
            # also P-BC (Dirichlet-tie ΔP mode), they share
            # `p_bc_source_arr[harm_idx]`.  The distinct-nodes-same-P
            # design keeps V1/V2 (and A1/A2) as separate nodes so
            # individual boundary Q's are still readable post-solve.
            # Guard: when no P-BC is set (all boundaries are Q-BC), the
            # arrays are None — indexing must be skipped.
            _per_node_sink_P = isinstance(p_bc_sink_arr, dict)
            p_sink_val = (p_bc_sink_arr[harm_idx]
                           if (p_bc_sink_arr is not None
                                and not _per_node_sink_P) else None)
            p_src_val = (p_bc_source_arr[harm_idx]
                          if p_bc_source_arr is not None else None)
            for pn in p_bc_nodes:
                idx = node_to_idx[pn]
                L_solve[idx, :] = 0
                L_solve[idx, idx] = 1.0
                if pn in sink_nodes:
                    if _per_node_sink_P:
                        _arr = p_bc_sink_arr.get(int(pn))
                        Q_solve[idx] = (_arr[harm_idx]
                                         if _arr is not None else 0.0)
                    else:
                        Q_solve[idx] = p_sink_val
                elif p_src_val is not None:
                    Q_solve[idx] = p_src_val
                else:
                    Q_solve[idx] = 0.0

            # Impose impedance BC at sink nodes
            for zn in z_bc_nodes:
                idx = node_to_idx[zn]
                z_val = z_ven(omega) if callable(z_ven) else z_ven
                if abs(z_val) > 1e-30:
                    Y_ground = 1.0 / z_val
                    L_solve[idx, idx] += Y_ground

            # DC: pin P=0 at one interior node
            if harm_idx == 0 and pin_node is not None and not sink_has_passive_bc:
                idx = node_to_idx[pin_node]
                L_solve[idx, :] = 0
                L_solve[idx, idx] = 1.0
                Q_solve[idx] = 0.0

            # Tied-pressure penalty: force P(n_i) ≈ P(n_j) for every
            # pair of tied nodes at THIS harmonic, without pinning the
            # value.  Independent of merged_boundary (which ties ALL
            # boundaries).  Used e.g. to model a shared atrial chamber
            # (tie the V nodes) while keeping the AC pressure a solve
            # output rather than a Dirichlet=0.
            if tied_pressure_nodes and len(tied_pressure_nodes) >= 2:
                diag_abs_tp = np.abs(L_full.diagonal())
                Y_tie = 1e3 * max(diag_abs_tp.max(), 1e-20)
                T_idxs = [node_to_idx[t] for t in tied_pressure_nodes
                          if t in node_to_idx]
                for i_t in range(len(T_idxs)):
                    for j_t in range(i_t + 1, len(T_idxs)):
                        ti, tj = T_idxs[i_t], T_idxs[j_t]
                        L_solve[ti, tj] -= Y_tie
                        L_solve[tj, ti] -= Y_tie
                        L_solve[ti, ti] += Y_tie
                        L_solve[tj, tj] += Y_tie
                # DC gauge: pin ONE tied node to P=0 at DC.  The tie
                # makes the others follow.  Without this, the tie
                # relative-constraint only enforces DIFFERENCES between
                # tied nodes; absolute pressure drifts to ~1e10 Pa
                # because the interior pin's constraint on absolute V
                # pressure is weak compared to Y_tie.  Only applied at
                # DC (AC has compliance making L(ω) rank-full).
                if harm_idx == 0 and T_idxs:
                    ti = T_idxs[0]
                    L_solve[ti, :] = 0
                    L_solve[ti, ti] = 1.0
                    Q_solve[ti] = 0.0

            L_solve = L_solve.tocsr()

            try:
                P = spsolve(L_solve, Q_solve)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: Solve failed at harmonic {harm_idx} "
                          f"(ω={omega:.2f}): {e}")
                continue

        # Store pressures
        for node in all_nodes:
            node_P[node][harm_idx] = P[node_to_idx[node]]

        if verbose and harm_idx == 0 and not use_merged_bc:
            P_abs = np.abs(P)
            n_nonzero = np.count_nonzero(P_abs > 1e-20)
            print(f"  DC solve: {n_nonzero}/{len(P)} nodes with |P|>0, "
                  f"max|P|={P_abs.max():.4e}, "
                  f"Q_rhs nonzero={np.count_nonzero(np.abs(Q_solve) > 1e-20)}, "
                  f"max|Q_rhs|={np.abs(Q_solve).max():.4e}, "
                  f"L nnz={L_solve.nnz}, pin={pin_node}")

    # --- 8. Reconstruct edge flows from pressure differences ---
    edge_flows = {}
    mean_Q_dict = {}
    amp_Q_dict = {}
    PI_dict = {}
    RPSI_dict = {}
    eta_dict = {}
    phase_dict = {}
    pressure_phase_dict = {}
    WSS_dict = {}
    dissipation_dict = {}
    pulsatile_cost_dict = {}
    Q_stored_dict = {}
    storage_frac_dict = {}

    for u, v in edge_list:
        R_m, L_m = _get_edge_geometry(G, u, v, radii_m=radii_m)
        if R_m is None:
            continue

        Q_harmonics = np.zeros(n_harmonics + 1, dtype=complex)

        for harm_idx, omega in enumerate(frequencies):
            Y_diag, Y_off = _vessel_admittance(R_m, L_m, omega, mu, rho, D)

            P_u = node_P[u][harm_idx]
            P_v = node_P[v][harm_idx]

            # Flow from u to v (current out of port u)
            Q_uv = Y_diag * P_u + Y_off * P_v  # m³/s
            Q_harmonics[harm_idx] = Q_uv * 1e12  # -> nL/s

        edge_flows[(u, v)] = Q_harmonics

        # Summary statistics
        q_dc = Q_harmonics[0].real
        q1 = Q_harmonics[1] if n_harmonics >= 1 else 0.0

        mean_Q_dict[(u, v)] = q_dc
        amp_Q_dict[(u, v)] = abs(q1)
        if abs(q_dc) > 1e-12:
            PI_dict[(u, v)] = 2.0 * abs(q1) / abs(q_dc)
        else:
            PI_dict[(u, v)] = np.nan

        # RPSI: max(dQ/dt) / Q̄ — analytical from harmonics
        RPSI_dict[(u, v)] = compute_rpsi_from_harmonics(Q_harmonics, f0_hz)
        # η: waveform sharpness = max|dQ/dt| / (πf0 (Qmax-Qmin))
        eta_dict[(u, v)] = compute_eta_from_harmonics(Q_harmonics, f0_hz)

        # Phase = angle of forward-direction Q_H1 (sign-canonicalised so
        # DC ≥ 0).  Previously this was midpoint pressure phase, which
        # is also orientation-independent but bakes in a ~180° structural
        # split between source-side and sink-side edges; that masked the
        # small physiological A→V phase lag.  Sign-canonicalising Q
        # restores orientation-independence without that split.
        # CHANGED 2026-05-27 — to revert, replace below with:
        #   P1_mid = (node_P[u][1] + node_P[v][1]) / 2.0 if n_harmonics >= 1 else 0.0
        #   phase_dict[(u, v)] = float(np.angle(P1_mid))
        if n_harmonics >= 1:
            Q_canon = Q_harmonics if Q_harmonics[0].real >= 0 else -Q_harmonics
            phase_dict[(u, v)] = float(np.angle(Q_canon[1]))
        else:
            phase_dict[(u, v)] = 0.0
        # Keep the legacy pressure-phase under a separate key so anything
        # downstream that wants it can still find it.
        P1_mid_legacy = ((node_P[u][1] + node_P[v][1]) / 2.0
                         if n_harmonics >= 1 else 0.0)
        pressure_phase_dict[(u, v)] = float(np.angle(P1_mid_legacy))

        # Wall shear stress: TAWSS = 4μ/(πR³) · ⟨|Q(t)|⟩
        # Reconstruct Q(t) over one cycle from harmonics, then mean(|Q|)
        _n_pts = 128
        _t_cyc = np.linspace(0, 2 * np.pi, _n_pts, endpoint=False)
        _Q_t = np.full(_n_pts, Q_harmonics[0].real)  # DC
        for _nh in range(1, n_harmonics + 1):
            _Q_t += (Q_harmonics[_nh].real * np.cos(_nh * _t_cyc)
                     - Q_harmonics[_nh].imag * np.sin(_nh * _t_cyc))
        _Q_mean_abs_m3s = float(np.mean(np.abs(_Q_t))) * 1e-12
        WSS_dict[(u, v)] = 4.0 * mu * _Q_mean_abs_m3s / (np.pi * R_m**3)

        # Viscous dissipation: Φ = r·L·⟨Q²⟩  (Watts)
        Q_dc_sq = (q_dc * 1e-12)**2
        Q_ac_sq = 0.5 * sum(abs(Q_harmonics[n] * 1e-12)**2
                            for n in range(1, n_harmonics + 1))
        Q2_avg = Q_dc_sq + Q_ac_sq
        r_per_length = 8.0 * mu / (np.pi * R_m**4)
        dissipation_dict[(u, v)] = r_per_length * L_m * Q2_avg

        # Pulsatile cost: ⟨Q²⟩/Q̄²
        pulsatile_cost_dict[(u, v)] = (Q2_avg / Q_dc_sq
                                       if Q_dc_sq > 1e-30 else np.nan)

        # Compliance storage: Q̂_n^stored for each AC harmonic
        # Q̂_stored = (γ/r) · (P_u + P_v)(cosh(γL) - 1) / sinh(γL)
        Q_stored_harmonics = np.zeros(n_harmonics + 1, dtype=complex)
        for harm_idx in range(1, n_harmonics + 1):
            omega_h = 2.0 * np.pi * harm_idx * f0_hz
            r_pl, ell_pl, c_pl = _per_length_params(R_m, mu, rho, D)
            z_h = r_pl + 1j * omega_h * ell_pl
            y_h = 1j * omega_h * c_pl
            gamma_h = np.sqrt(z_h * y_h)
            kL_h = gamma_h * L_m

            P_u_h = node_P[u][harm_idx]
            P_v_h = node_P[v][harm_idx]

            if abs(kL_h) < 1e-6:
                # Small kL: lumped approx Q_stored ≈ iωcL · (P_u+P_v)/2
                Q_stored_harmonics[harm_idx] = (
                    1j * omega_h * c_pl * L_m * (P_u_h + P_v_h) / 2.0
                ) * 1e12  # m³/s → nL/s
            elif abs(kL_h.real) > 500:
                # Heavily damped: negligible storage
                Q_stored_harmonics[harm_idx] = 0.0
            else:
                e_pos = np.exp(kL_h)
                e_neg = np.exp(-kL_h)
                sinh_kL = (e_pos - e_neg) / 2.0
                cosh_kL = (e_pos + e_neg) / 2.0
                if abs(sinh_kL) > 1e-30:
                    Q_stored_harmonics[harm_idx] = (
                        (gamma_h / r_pl)
                        * (P_u_h + P_v_h) * (cosh_kL - 1.0) / sinh_kL
                    ) * 1e12  # → nL/s

        Q_stored_dict[(u, v)] = Q_stored_harmonics
        # Storage fraction: |Q̂_1^stored| / |Q̂_1^(μ)| at fundamental
        q1_in = abs(Q_harmonics[1]) if n_harmonics >= 1 else 0.0
        q1_stored = abs(Q_stored_harmonics[1]) if n_harmonics >= 1 else 0.0
        storage_frac_dict[(u, v)] = (q1_stored / q1_in
                                     if q1_in > 1e-30 else np.nan)

    # Assign boundary edges:
    #  - Q-BC nodes (sources): override from BC harmonics (as before)
    #  - P-BC nodes (sinks):   Q already computed from pressure solve, no override
    n_be_assigned = 0
    for bn in boundary_nodes:
        if bn in solve_flow_bc_nodes:
            # P-BC node: boundary edge flow comes from the pressure solve.
            # It's already in edge_flows from the reconstruction loop above.
            # Just ensure derived metrics are computed.
            nbrs = list(G.neighbors(bn))
            interior = nbrs[0] if len(nbrs) == 1 else max(
                nbrs, key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
            if G.has_edge(bn, interior):
                be = (bn, interior)
            elif G.has_edge(interior, bn):
                be = (interior, bn)
            else:
                continue
            if be in edge_flows:
                Q_harm_be = edge_flows[be]
                q_dc = Q_harm_be[0].real
                q1 = Q_harm_be[1] if n_harmonics >= 1 else 0.0
                mean_Q_dict[be] = q_dc
                amp_Q_dict[be] = abs(q1)
                PI_dict[be] = 2.0 * abs(q1) / abs(q_dc) if abs(q_dc) > 1e-12 else np.nan
                RPSI_dict[be] = compute_rpsi_from_harmonics(Q_harm_be, f0_hz)
                eta_dict[be] = compute_eta_from_harmonics(Q_harm_be, f0_hz)
                bc_type = 'P-BC' if bn in set(p_bc_nodes) else 'Z-BC'
                if verbose:
                    print(f"  {bc_type} edge {be}: Q_dc={q_dc:.3f} (from solve), "
                          f"|Q_1|={abs(q1):.3f}, PI={PI_dict[be]:.3f}")
            n_be_assigned += 1
            continue

        # Q-BC node (source): override from BC harmonics
        nbrs = list(G.neighbors(bn))
        interior = nbrs[0] if len(nbrs) == 1 else max(
            nbrs, key=lambda n: abs(G.edges[bn, n].get('mean_Q', 0)))
        if interior is None:
            continue
        if G.has_edge(bn, interior):
            be = (bn, interior)
        elif G.has_edge(interior, bn):
            be = (interior, bn)
        else:
            continue
        Q_bc = bc_harmonics[bn]
        edge_flows[be] = Q_bc
        q_dc = Q_bc[0].real
        q1 = Q_bc[1] if n_harmonics >= 1 else 0.0
        mean_Q_dict[be] = q_dc
        amp_Q_dict[be] = abs(q1)
        PI_dict[be] = 2.0 * abs(q1) / abs(q_dc) if abs(q_dc) > 1e-12 else np.nan
        RPSI_dict[be] = compute_rpsi_from_harmonics(Q_bc, f0_hz)
        eta_dict[be] = compute_eta_from_harmonics(Q_bc, f0_hz)
        n_be_assigned += 1
        if verbose:
            print(f"  Q-BC edge {be}: Q_dc={q_dc:.3f}, |Q_1|={abs(q1):.3f}, "
                  f"PI={PI_dict[be]:.3f}")
        # Phase = forward-direction Q_H1 (sign-canonicalised so DC ≥ 0).
        # CHANGED 2026-05-27 (see note at the interior-edge phase block).
        if n_harmonics >= 1 and abs(q1) > 0:
            Q_canon_be = Q_bc if Q_bc[0].real >= 0 else -Q_bc
            phase_dict[be] = float(np.angle(Q_canon_be[1]))
        else:
            phase_dict[be] = np.nan
        # Legacy pressure-phase kept alongside.
        interior_legacy = bc_remap.get(bn, bn)
        if interior_legacy in node_P and n_harmonics >= 1:
            pressure_phase_dict[be] = float(np.angle(node_P[interior_legacy][1]))
        else:
            pressure_phase_dict[be] = np.nan
        # WSS and dissipation for boundary edge
        R_m, L_m = _get_edge_geometry(G, be[0], be[1], radii_m=radii_m)
        if R_m is not None:
            _Q_t_be = np.full(_n_pts, Q_bc[0].real)
            for _nh in range(1, n_harmonics + 1):
                _Q_t_be += (Q_bc[_nh].real * np.cos(_nh * _t_cyc)
                            - Q_bc[_nh].imag * np.sin(_nh * _t_cyc))
            _Q_mean_abs_be = float(np.mean(np.abs(_Q_t_be))) * 1e-12
            WSS_dict[be] = 4.0 * mu * _Q_mean_abs_be / (np.pi * R_m ** 3)
            Q_dc_sq = (q_dc * 1e-12) ** 2
            Q_ac_sq = 0.5 * sum(abs(Q_bc[n] * 1e-12) ** 2
                                for n in range(1, n_harmonics + 1))
            r_pl = 8.0 * mu / (np.pi * R_m ** 4)
            dissipation_dict[be] = r_pl * L_m * (Q_dc_sq + Q_ac_sq)
            pulsatile_cost_dict[be] = ((Q_dc_sq + Q_ac_sq) / Q_dc_sq
                                       if Q_dc_sq > 1e-30 else np.nan)

    if verbose:
        print(f"  Assigned {n_be_assigned} boundary edges from BCs")
        n_finite = sum(1 for v in PI_dict.values() if np.isfinite(v))
        print(f"  Solved: {len(edge_flows)} edges, {n_finite} with finite PI")
        pis = [v for v in PI_dict.values() if np.isfinite(v)]
        if pis:
            print(f"  PI: median={np.median(pis):.3f}, "
                  f"mean={np.mean(pis):.3f}, range=[{min(pis):.3f}, {max(pis):.3f}]")
        qs = [abs(v) for v in mean_Q_dict.values() if abs(v) > 1e-12]
        if qs:
            print(f"  |mean_Q|: median={np.median(qs):.3f} nL/s, "
                  f"range=[{min(qs):.3f}, {max(qs):.3f}]")

        # Flow conservation check at boundary nodes
        for node in boundary_nodes:
            total_q = 0.0
            for nb in G.neighbors(node):
                if (node, nb) in mean_Q_dict:
                    total_q += mean_Q_dict[(node, nb)]
                elif (nb, node) in mean_Q_dict:
                    total_q -= mean_Q_dict[(nb, node)]
            role = 'Source' if node in source_nodes else 'Sink'
            print(f"  {role} {node}: net Q = {total_q:.3f} nL/s")

    if verbose and WSS_dict:
        wss_vals = [v for v in WSS_dict.values() if np.isfinite(v)]
        if wss_vals:
            print(f"  WSS: median={np.median(wss_vals):.3f} Pa, "
                  f"range=[{min(wss_vals):.3f}, {max(wss_vals):.3f}]")
        diss_vals = list(dissipation_dict.values())
        total_diss = sum(diss_vals)
        print(f"  Total dissipation: {total_diss:.3e} W")
        pc_vals = [v for v in pulsatile_cost_dict.values() if np.isfinite(v)]
        if pc_vals:
            print(f"  Pulsatile cost ⟨Q²⟩/Q̄²: median={np.median(pc_vals):.2f}, "
                  f"range=[{min(pc_vals):.2f}, {max(pc_vals):.2f}]")

        # Compliance storage diagnostics
        sf_vals = [v for v in storage_frac_dict.values() if np.isfinite(v)]
        if sf_vals:
            print(f"  Storage fraction |Q̂₁ˢᵗᵒʳᵉᵈ|/|Q̂₁ⁱⁿ|: "
                  f"median={np.median(sf_vals):.4f}, "
                  f"range=[{min(sf_vals):.4f}, {max(sf_vals):.4f}]")
            # Global conservation check: ΣQ̂_n^stored = ΣQ̂_n^ext
            for nh in range(1, n_harmonics + 1):
                total_stored = sum(
                    Q_stored_dict[k][nh] for k in Q_stored_dict) * 1e-12
                total_ext = sum(
                    bc_harmonics[bn][nh] for bn in q_bc_nodes) * 1e-12
                err = abs(total_stored - total_ext)
                rel = err / max(abs(total_ext), 1e-30)
                print(f"  Conservation h={nh}: "
                      f"|ΣQ_stored|={abs(total_stored)*1e12:.4f}, "
                      f"|ΣQ_ext|={abs(total_ext)*1e12:.4f}, "
                      f"rel_err={rel:.2e}")

    # Store representative scalar D for display (evaluate at R=10 µm if callable)
    D_scalar = D(10e-6) if callable(D) else D

    return TransmissionLineResult(
        edge_flows=edge_flows,
        node_pressures=node_P,
        mean_Q=mean_Q_dict,
        amp_Q=amp_Q_dict,
        PI=PI_dict,
        RPSI=RPSI_dict,
        eta=eta_dict,
        phase=phase_dict,
        pressure_phase=pressure_phase_dict,
        WSS=WSS_dict,
        dissipation=dissipation_dict,
        pulsatile_cost=pulsatile_cost_dict,
        Q_stored=Q_stored_dict,
        storage_fraction=storage_frac_dict,
        f0_hz=f0_hz,
        n_harmonics=n_harmonics,
        D=D_scalar,
        mu=mu,
        boundary_nodes=boundary_nodes,
        boundary_Q_harmonics=bc_harmonics,
        n_edges=len(edge_flows),
        n_nodes=N,
    )




# ── PORT NOTE ────────────────────────────────────────────────────────────
# This file is a trimmed port of pertile/analysis/transmission_line.py
# (2026-09-30). Original had 6189 lines; this port keeps 1539 (25%) — the
# core solver plus the harmonic/eta/rpsi utilities. The 4450 lines of
# `optimize_transmission_line`, `optimize_greyzone_*`, and `_build_sheet_admittance`
# variants were dropped because they are superseded by the outlet-template +
# α_j fit recipe in plexusflow/scripts/inference/pipeline_outlet_agg.py.
#
# Downstream code that needs an optimizer should use the outlet-template
# approach; if the historical optimize_greyzone_* variants ARE needed, they
# can be pulled from pertile/analysis/transmission_line.py in git history.
