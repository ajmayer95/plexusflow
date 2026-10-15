"""Observation-level spectral tile synchronization: phase U(1) + log-amp R.

Replaces the three-step pipeline `sync_tile_gauges` +
`fill_missing_gauges` + `sign_flip_rescue` with a single spectral solve
on the observation-level connection matrix. Motivation:

  - The BFS 2-coloring in the old `sync_tile_gauges` cannot see
    frustrated cycles. A connection-Laplacian eigenvector solve sees
    every cycle at once.
  - `fill_missing_gauges` exists only because the old primary sync had
    an n>=3 gate. Observation-level construction has no such gate;
    connectivity is the only requirement.
  - `sign_flip_rescue` is single-site Ising dynamics on a discrete
    variable that is a gauge no-op (tile-level Q -> -Q is absorbed by
    z_t -> -z_t). Genuine per-pair sign errors are localized instead by
    triangle parity, which is Z_2 holonomy on the connection matrix
    and lives outside the U(1) gauge group.

Interface (drop-in replacement for the three deprecated functions):

    gauges = spectral_sync(G, harmonic_idx, params)
    → {tile_id: {z_re, z_im, log_amp, psi_rad, source}}

Sign vs gauge across harmonics
------------------------------
Two distinct symmetries:
  - Gauge rotation (time shift): phi_k -= k * arg(z_H1) — LINEAR in k.
  - Orientation flip (Q -> -Q):  phi_k += pi for EVERY k — CONSTANT in k.

The canonicalization applied in `_extract_edge_measurements_canonical`
implements the orientation flip on the raw Q; per-harmonic gauge
rotation is a separate step and lives on the apply side. Do NOT
combine them.

Two-sided weight
----------------
For a ratio Q_s/Q_t the effective SNR is the harmonic mean of the two
sides (bright meets dim: only as good as the worse one):

    kappa_e = (kappa_s * kappa_t) / (kappa_s + kappa_t)

with kappa = 10 ** (SNR_dB / 10). SNR is already amplitude-dependent,
so DO NOT multiply by amp squared as well — that double-counts.

Giant-component reporting
-------------------------
Tiles outside the tile-pair giant component cannot be gauge-aligned to
the anchor and are dropped. That drop is logged loudly on
`G.graph['tile_sync_diagnostics']['tiles_missing_from_giant']` so a
missing tile in a downstream figure does not go unnoticed.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Canonical observation extraction (single source of truth for orientation)
# =============================================================================
def _canonical_f0_from_measurements(G, ref_tile=14):
    """Canonical f0 for the sync's f0-proximity gate.

    Default uses tile `ref_tile`'s (=14) median f0 to match the sim
    solver's chosen f0 (stored in G.graph['sim_meta']['f0_hz'] for the
    stage-21 canonical, which is tile 14's f0). Keeping the sync's
    canonical the same as the sim's ensures the empirical and sim
    fields are on the same cycle reference.

    Fallback to global median if the ref tile has no measurements.
    """
    ref_f0 = []
    all_f0 = []
    for u, v, d in G.edges(data=True):
        for m in d.get('measurements_piv') or []:
            f0 = (m.get('harm') or {}).get('f0_hz')
            if f0 is None or not np.isfinite(f0): continue
            all_f0.append(f0)
            if ref_tile is not None and m.get('tile_id') == ref_tile:
                ref_f0.append(f0)
    if ref_f0:
        return float(np.median(ref_f0))
    return float(np.median(all_f0)) if all_f0 else float('nan')


def _extract_observations_canonical(G, harmonic_idx: int,
                                     min_snr_db: float = -np.inf,
                                     f0_tolerance_frac: float = 1.5,
                                     min_pulsatility: float = 0.10,
                                     max_tile_id: int = None,
                                     min_tile_id: int = 1,
                                     f0_soft_sigma_frac: float = 1.0):
    """f0_soft_sigma_frac: Gaussian taper on pair weight based on how far
    each tile's f0 is from canonical F0. Per-tile factor:
        exp(-((f0 - F0)^2) / (2 * (sigma_frac * F0)^2))
    Pair weight is the product of the two tiles' factors. This softly
    downweights drift-affected tiles without a hard cliff.

    Defaults widened 2026-08-24 (f0_tolerance_frac 0.50 → 1.5,
    f0_soft_sigma_frac 0.08 → 1.0) so early-acquisition slow-HR tiles
    (e.g. tile 1 at 0.87 Hz vs canonical 2.46 Hz) still contribute to
    the sync. Old defaults dropped their weight to exp(-32) ≈ 0 and
    left the top-left mosaic corner as noise. The wide-tolerance run
    reproduces the smooth phase field of canonical_2 on the Somites21
    plexusflow graph. If a truly-noise tile drifts in, it'll be caught
    by the SNR gate anyway.
    """
    """Extract per-tile (Q, snr, amp) in canonical (u, v) orientation.

    Returns
    -------
    observations : list of (edge_key, tile_s, tile_t, r_unit, weight)
        edge_key = (u, v) as stored in G; s < t; r_unit = Q_s/Q_t
        with |r_unit|=1 (unit-modulus ratio).
    edge_meas    : {edge_key: [(tile, Q_canonical, snr_db, amp), ...]}
        Per-edge canonical-Q measurements, useful for downstream apply.
    """
    # f0-proximity filter — reject measurements whose fitted f0 differs
    # from the population median by more than `f0_tolerance_frac`. In
    # snake-pattern acquisition the heart rate drifts across tiles; a
    # tile whose f0 is far off is fitting a different oscillation (or a
    # spurious spectral peak). Its `phi` refers to the phase of THAT
    # oscillation, not the cardiac cycle other tiles are seeing.
    # Including such measurements corrupts pair overlaps in the sync.
    F0_canonical = _canonical_f0_from_measurements(G)
    # Two-sided gate ±f0_tolerance_frac of canonical F0. Both directions
    # matter: low-f0 tiles fit subharmonics; high-f0 tiles are fitted at
    # frequencies mismatched from canonical. Either kind contaminates
    # gauges of every neighbour they pair with — see docstring.
    f0_lo = F0_canonical * (1 - f0_tolerance_frac)
    f0_hi = F0_canonical * (1 + f0_tolerance_frac)

    n_meas_total = 0; n_meas_dropped_f0 = 0; n_meas_dropped_pi = 0
    edge_meas: Dict[Tuple, List] = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        for m in (ed.get('measurements_piv') or []):
            harms = ((m.get('harm') or {}).get('harmonics') or [])
            if harmonic_idx >= len(harms) or harms[harmonic_idx] is None:
                continue
            h = harms[harmonic_idx]
            phi = h.get('phi'); amp = h.get('amp'); snr = h.get('snr_db')
            f0 = (m.get('harm') or {}).get('f0_hz')
            mean_Q = (m.get('harm') or {}).get('mean_Q')
            if any(x is None for x in (phi, amp, snr, f0)):
                continue
            if not (np.isfinite(phi) and np.isfinite(amp)
                    and np.isfinite(snr) and np.isfinite(f0)):
                continue
            if amp <= 0 or f0 <= 0 or snr < min_snr_db:
                continue
            n_meas_total += 1
            if f0 < f0_lo or f0 > f0_hi:
                n_meas_dropped_f0 += 1
                continue
            # Pulsatility gate: PI = 2·|H1| / |mean_Q|. Measurements with
            # PI far below unity have too little AC energy for H1 phase
            # to be determined — mostly-steady flow with tiny oscillation.
            # Its phi is noise-dominated. High PI (flow near zero mean) is
            # fine; the risk is at low PI with strong DC.
            if mean_Q is not None and np.isfinite(mean_Q) and abs(mean_Q) > 1e-9:
                pi = 2.0 * amp / abs(mean_Q)
                if pi < min_pulsatility:
                    n_meas_dropped_pi += 1
                    continue
            ff = m.get('flow_from'); ft = m.get('flow_to')
            if (ff, ft) == (u, v):
                sign = +1.0
            elif (ff, ft) == (v, u):
                sign = -1.0
            else:
                continue
            Q_canon = sign * amp * np.exp(1j * phi)
            tid = m.get('tile_id')
            if tid is None:
                continue
            if max_tile_id is not None and int(tid) > max_tile_id:
                continue
            if min_tile_id is not None and int(tid) < min_tile_id:
                continue
            edge_meas[(u, v)].append(
                (int(tid), Q_canon, float(snr), float(amp), float(f0)))

    observations = []
    sigma_hz = F0_canonical * f0_soft_sigma_frac
    two_sig2 = 2 * sigma_hz ** 2 if sigma_hz > 0 else float('inf')
    for edge, ms in edge_meas.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                if a[0] == b[0]: continue
                if abs(a[1]) < 1e-12 or abs(b[1]) < 1e-12: continue
                s, t = (a, b) if a[0] < b[0] else (b, a)
                s_id, Q_s, snr_s, amp_s, f0_s = s
                t_id, Q_t, snr_t, amp_t, f0_t = t
                w = _two_sided_kappa(snr_s, snr_t)
                if w <= 0: continue
                # Soft f0 taper — product of per-tile Gaussian weights.
                # Pair with either tile far off canonical gets down-weighted.
                if two_sig2 < float('inf'):
                    taper_s = np.exp(-((f0_s - F0_canonical) ** 2) / two_sig2)
                    taper_t = np.exp(-((f0_t - F0_canonical) ** 2) / two_sig2)
                    w = w * taper_s * taper_t
                if w <= 0: continue
                r = Q_s / Q_t
                r_unit = r / abs(r)
                observations.append((edge, s_id, t_id, r_unit, w))
    if n_meas_total > 0 and n_meas_dropped_f0 > 0:
        print(f'[tile_sync_u1] f0 gate ±{int(f0_tolerance_frac*100)}% of '
              f'{F0_canonical:.3f} Hz dropped {n_meas_dropped_f0}/{n_meas_total} '
              f'measurements ({100*n_meas_dropped_f0/n_meas_total:.1f}%)')
    return observations, dict(edge_meas)


def _two_sided_kappa(snr_s_db: float, snr_t_db: float) -> float:
    """Two-sided effective weight for a ratio Q_s/Q_t.

    Harmonic mean of per-side linear SNRs — the ratio is only as
    informative as the worse of the two sides. Do NOT multiply by
    amplitude squared afterwards; SNR is already amp-dependent, and
    double-counting biases the solve toward high-amplitude vessels.
    """
    kappa_s = 10.0 ** (min(snr_s_db, 30.0) / 10.0)
    kappa_t = 10.0 ** (min(snr_t_db, 30.0) / 10.0)
    return kappa_s * kappa_t / (kappa_s + kappa_t)


# =============================================================================
# Spectral solve
# =============================================================================
def _build_matrices(observations, tiles):
    """Assemble H (Hermitian, complex, for U(1) phase) and L (real,
    weighted Laplacian for log-amp) from the observation list.

    Observation-level construction: every shared-edge ratio contributes
    one term to H and L directly. No pair aggregation, no unit-modulus
    normalization step, no separate concentration statistic.
    """
    T = len(tiles)
    tile_idx = {t: i for i, t in enumerate(tiles)}
    H = sp.lil_matrix((T, T), dtype=complex)
    L_rows, L_cols, L_vals = [], [], []
    b = np.zeros(T)                     # amp RHS after eliminating pin
    D_phase = np.zeros(T)               # phase-degree per tile
    D_amp = np.zeros(T)                 # amp-degree per tile
    log_ratio_sum = defaultdict(complex)  # for optional diagnostics
    for (edge, s, t, r_unit, w) in observations:
        if s not in tile_idx or t not in tile_idx:
            continue
        i, j = tile_idx[s], tile_idx[t]
        # Phase (U(1))
        H[i, j] += w * r_unit
        H[j, i] += w * np.conj(r_unit)
        D_phase[i] += w; D_phase[j] += w
        # Amplitude (R): needs log|Q_s/Q_t| — we do not carry it in
        # r_unit; pull magnitudes from H below. See amplitude solve.
    return H.tocsr(), D_phase


def _build_amp_matrices(observations, tiles, edge_meas):
    """Weighted Laplacian for log-amplitude leveling.

    log(|z_s|) - log(|z_t|) = log|Q_s/Q_t| per observation (pinned tile
    0 for gauge). Same structure as the phase matrix, different scalar
    quantity per observation.
    """
    T = len(tiles)
    tile_idx = {t: i for i, t in enumerate(tiles)}
    rows, cols, vals = [], [], []
    b = np.zeros(T)
    D = np.zeros(T)
    # Re-derive per-obs log-amp from the canonical Q. Cheaper than
    # threading it through the phase observation list.
    for edge, ms in edge_meas.items():
        for i_ in range(len(ms)):
            for j_ in range(i_ + 1, len(ms)):
                a, b_ = ms[i_], ms[j_]
                if a[0] == b_[0]: continue
                if abs(a[1]) < 1e-12 or abs(b_[1]) < 1e-12: continue
                s, t = (a, b_) if a[0] < b_[0] else (b_, a)
                s_id, Q_s, snr_s, amp_s = s[0], s[1], s[2], s[3]
                t_id, Q_t, snr_t, amp_t = t[0], t[1], t[2], t[3]
                if s_id not in tile_idx or t_id not in tile_idx: continue
                i, j = tile_idx[s_id], tile_idx[t_id]
                w = _two_sided_kappa(snr_s, snr_t)
                if w <= 0: continue
                d = float(np.log(abs(Q_s)) - np.log(abs(Q_t)))
                rows += [i, j, i, j]
                cols += [i, j, j, i]
                vals += [w, w, -w, -w]
                b[i] += w * d
                b[j] -= w * d
                D[i] += w; D[j] += w
    L = sp.csr_matrix((vals, (rows, cols)), shape=(T, T))
    return L, b, D


def _solve_phase(H: sp.csr_matrix, tiles: List[int]
                 ) -> Tuple[Dict[int, complex], float, float]:
    """Leading eigenvector of the connection matrix H → per-tile z_t.

    Uses dense `scipy.linalg.eigh` because sparse `eigsh` on this size
    of matrix (T ≤ 100 tiles typical) can return a spurious LOCALIZED
    eigenvector when the top eigenvalues are close — most |v_i| land
    near zero and a handful concentrate the mass, which makes those
    zero-mass tiles' gauges essentially undefined and their residuals
    against the solve read as ~π (masquerading as sign errors).
    Dense eigh is O(T^3) but T=53 costs microseconds and returns the
    true delocalized top eigenvector deterministically.
    """
    import scipy.linalg
    T = len(tiles)
    if T < 2:
        return ({tiles[0]: 1.0 + 0j} if T == 1 else {},
                float('nan'), float('nan'))
    H_dense = H.toarray()
    # The RAW H's top eigvec localizes on high-degree tiles when the
    # weighted-degree distribution is heavy-tailed (a few high-SNR
    # pairs dominate). Angles on low-degree tiles then land near |v|=0
    # and get fudged to arg=0, so their tile-residuals surface as ~π —
    # producing dozens of spurious sign-flip candidates and a bogus
    # per-tile-residual leaderboard.
    #
    # Solve the NORMALIZED matrix D^{-1/2} H D^{-1/2} instead. Its top
    # eigvec is delocalized (same math as the normalized graph
    # Laplacian for U(1) sync — Bandeira/Singer/Spielman). Angle of
    # the recovered z_i is invariant to the D^{-1/2} scaling, so no
    # un-normalization is needed.
    D = np.abs(H_dense).sum(axis=1)
    Dsqrt_inv = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    Hn = H_dense * Dsqrt_inv[:, None] * Dsqrt_inv[None, :]
    ev_n, vecs_n = scipy.linalg.eigh(Hn)      # ascending
    v = vecs_n[:, -1]                          # top eigvec of Hn
    z = np.where(np.abs(v) > 1e-12, v / np.abs(v), 1.0 + 0j)
    # Pin gauge: rotate so the anchor tile (index 0) has zero phase.
    z_anchor = z[0]
    if abs(z_anchor) > 1e-12:
        z = z * np.conj(z_anchor)
    lam_top_norm = float(ev_n[-1])
    frustration_n = 1.0 - lam_top_norm
    # Also return the raw H's top eigenvalue for interpretability
    ev_raw, _ = scipy.linalg.eigh(H_dense)
    lam_top = float(ev_raw[-1])
    z_dict = {tiles[i]: complex(z[i]) for i in range(T)}
    return z_dict, lam_top, frustration_n


def _solve_amp(L: sp.csr_matrix, b: np.ndarray, tiles: List[int]
               ) -> Dict[int, float]:
    """Weighted graph Laplacian → per-tile log-amplitude, pin[0]=0."""
    T = len(tiles)
    if T < 2:
        return {tiles[0]: 0.0} if T == 1 else {}
    pin = 0
    keep = np.array([i for i in range(T) if i != pin])
    L_red = L.tolil()[keep][:, keep].tocsc()
    try:
        log_c_red = spla.spsolve(L_red, b[keep])
    except Exception:
        log_c_red = np.zeros(len(keep))
    log_c = np.zeros(T)
    log_c[keep] = log_c_red
    log_c[pin] = 0.0
    return {tiles[i]: float(log_c[i]) for i in range(T)}


# =============================================================================
# Diagnostics
# =============================================================================
def _per_obs_residuals(observations, z_dict):
    r_all, w_all = [], []
    for (_, s, t, r_unit, w) in observations:
        if s not in z_dict or t not in z_dict:
            continue
        pred = z_dict[s] * np.conj(z_dict[t])
        r_all.append(np.angle(r_unit * np.conj(pred)))
        w_all.append(w)
    return np.array(r_all), np.array(w_all)


def _leave_one_pair_out(observations, tiles):
    """Leave-one-pair-out (LOPO) residuals — masking-safe pair diagnostic.

    A high-leverage pair with many high-SNR observations (43 shared edges
    against a typical 6-to-26) can drag z_s and z_t onto itself and hide
    a genuine sign error — its own residual looks fine while the flip
    surfaces in neighbouring pairs' residuals instead. The single-fit
    triangle-parity report is vulnerable to this: it changes rankings
    with weight choice, and can miss a confirmed sign error entirely.

    LOPO removes each pair's observations, re-solves the eigenproblem on
    the surrounding data, and scores the removed pair's per-observation
    weighted residual against the leverage-free solution. A genuinely
    consistent pair still fits. A pair that was fitting itself lights
    up. Cost is 149 x 53x53 eigenproblems for stage 21 — fractions of
    a second — so no need for SMW-style downdates yet.

    Returns {pair: dict(loo_resid_median_deg, loo_resid_wrms_deg,
                        n_obs, is_signflip_candidate)} per pair.
    """
    obs_by_pair = defaultdict(list)
    for o in observations:
        obs_by_pair[(o[1], o[2])].append(o)
    reports = {}
    for pair in obs_by_pair:
        rest = [o for o in observations if (o[1], o[2]) != pair]
        # Re-check giant component without this pair
        tG_r = nx.Graph()
        for (_, s, t, _, _) in rest: tG_r.add_edge(s, t)
        comps = list(nx.connected_components(tG_r))
        if not comps: continue
        giant = max(comps, key=len)
        if pair[0] not in giant or pair[1] not in giant:
            # Removing this pair disconnected an endpoint — LOO undefined.
            reports[pair] = dict(disconnecting=True,
                                  n_obs=len(obs_by_pair[pair]))
            continue
        rest = [o for o in rest if o[1] in giant and o[2] in giant]
        tiles_r = sorted(giant)
        H_r, _ = _build_matrices(rest, tiles_r)
        z_r, _, _ = _solve_phase(H_r, tiles_r)
        if pair[0] not in z_r or pair[1] not in z_r: continue
        # Score removed pair's observations against leverage-free solve
        rs, ws = [], []
        for (_, s, t, r_unit, w) in obs_by_pair[pair]:
            pred = z_r[s] * np.conj(z_r[t])
            rs.append(np.angle(r_unit * np.conj(pred)))
            ws.append(w)
        rs = np.array(rs); ws = np.array(ws)
        ad = np.degrees(np.abs(rs))
        med = float(np.median(ad))
        wrms = float(np.degrees(np.sqrt(np.average(rs**2, weights=ws))))
        reports[pair] = dict(
            n_obs=len(rs),
            loo_resid_median_deg=med,
            loo_resid_wrms_deg=wrms,
            loo_resid_w_frac_over_90=float(ws[ad > 90].sum() / ws.sum()),
            disconnecting=False,
            # Sign-flip candidate = median LOO residual within 30° of pi.
            # Distinguishes a genuine flip (near-pi) from a merely
            # inconsistent pair (large residual, not near pi).
            is_signflip_candidate=(abs(med - 180.0) < 30.0),
        )
    return reports


def _tile_aggregate_residuals(observations, z_dict, tiles):
    """Per-tile aggregate residual against all its measuring neighbours.

    A single bad tile can masquerade as several bad pairs (tile 31
    appearing in 4 of 6 flagged pairs was the symptom). This rolls
    per-pair residuals up to the tile level so a data-quality problem
    in one tile's PIV pipeline shows as one high tile-residual, not
    N high pair-residuals.
    """
    per_tile_r = defaultdict(list)
    per_tile_w = defaultdict(list)
    for (_, s, t, r_unit, w) in observations:
        if s not in z_dict or t not in z_dict: continue
        pred = z_dict[s] * np.conj(z_dict[t])
        r = np.angle(r_unit * np.conj(pred))
        per_tile_r[s].append(r); per_tile_w[s].append(w)
        per_tile_r[t].append(r); per_tile_w[t].append(w)
    out = {}
    for tid in tiles:
        if tid not in per_tile_r: continue
        rs = np.array(per_tile_r[tid]); ws = np.array(per_tile_w[tid])
        ad = np.degrees(np.abs(rs))
        out[int(tid)] = dict(
            n_obs=int(len(rs)),
            median_deg=float(np.median(ad)),
            wrms_deg=float(np.degrees(np.sqrt(np.average(rs**2, weights=ws)))),
            w_frac_over_90=float(ws[ad > 90].sum() / ws.sum()),
        )
    return out


def _triangle_parity_report(observations, z_dict, tile_pair_graph,
                             min_R_gate_frac: float = 0.1):
    """Localize per-pair sign errors via Z_2 triangle parity.

    A triangle carries an odd number of sign-flipped edges iff
    arg(h_ab * h_bc * h_ca) is near pi. The bad-triangle count is
    weighted by min(|R_ab|, |R_bc|, |R_ca|) — near-zero-|R| pairs
    carry no information and should not vote in the count.

    Returns list of {pair, n_obs, R_mag, h_arg_deg, bad_triangle_frac}
    for pairs that participate in bad triangles with meaningful weight.
    """
    pair_R = defaultdict(complex)
    pair_W = defaultdict(float)
    pair_n = defaultdict(int)
    for (_, s, t, r_unit, w) in observations:
        pair_R[(s, t)] += w * r_unit
        pair_W[(s, t)] += w
        pair_n[(s, t)] += 1
    h_lookup, R_lookup = {}, {}
    for (s, t), R in pair_R.items():
        if abs(R) < 1e-18: continue
        h_lookup[(s, t)] = R / abs(R); h_lookup[(t, s)] = np.conj(R / abs(R))
        R_lookup[(s, t)] = abs(R); R_lookup[(t, s)] = abs(R)

    tris = [tri for tri in nx.enumerate_all_cliques(tile_pair_graph)
            if len(tri) == 3]
    Rmins = []
    for tri in tris:
        a, b, c = sorted(tri)
        try:
            Rmins.append(min(R_lookup[(a, b)], R_lookup[(b, c)],
                             R_lookup[(a, c)]))
        except KeyError:
            Rmins.append(0.0)
    Rmin_arr = np.array(Rmins) if Rmins else np.array([0.0])
    positive = Rmin_arr[Rmin_arr > 0]
    R_median = float(np.median(positive)) if len(positive) > 0 else 0.0
    gate = R_median * min_R_gate_frac

    edge_bad_R = defaultdict(float)
    edge_tri_R = defaultdict(float)
    n_bad = 0; n_bad_gated = 0
    bad_R_sum = 0.0; tot_R_sum = 0.0
    for tri, Rmin in zip(tris, Rmins):
        a, b, c = sorted(tri)
        try:
            prod = h_lookup[(a, b)] * h_lookup[(b, c)] * h_lookup[(c, a)]
        except KeyError:
            continue
        is_bad = abs(np.angle(prod)) > np.pi / 2
        tot_R_sum += Rmin
        if is_bad:
            n_bad += 1
            bad_R_sum += Rmin
            if Rmin >= gate: n_bad_gated += 1
        for e in [(a, b), (b, c), (a, c)]:
            edge_tri_R[e] += Rmin
            if is_bad: edge_bad_R[e] += Rmin

    weight_frac = bad_R_sum / max(tot_R_sum, 1e-18)
    # Report only pairs whose bad-triangle involvement is high AND whose
    # own resultant carries real weight (|R| >= gate). Additionally gate
    # on `n_obs >= 2`: a pair with a single shared edge is unidentifiable
    # between "pair-level sign error" and "single edge mislabelled at
    # the PIV level" — the sign variable at n=1 has zero degrees of
    # freedom against a per-observation mislabel, so promoting such a
    # pair as a "sign error" is over-claiming. n=1 pairs still
    # contribute to the solve; they just cannot be diagnosed as
    # sign-flipped.
    reports = []
    for e in edge_bad_R:
        if edge_bad_R[e] < gate: continue      # trivial weight
        R_mag = R_lookup.get(e, 0.0)
        if R_mag < gate: continue
        n = int(pair_n.get(e, pair_n.get((e[1], e[0]), 0)))
        if n < 2: continue                      # unidentifiable
        frac = edge_bad_R[e] / max(edge_tri_R[e], 1e-18)
        if frac < 0.5: continue                 # not majority-bad
        h = h_lookup.get(e)
        if h is None: continue
        reports.append(dict(
            pair=(int(e[0]), int(e[1])),
            n_obs=n,
            R_mag=float(R_mag),
            h_arg_deg=float(np.degrees(np.angle(h))),
            bad_triangle_frac=float(frac),
        ))
    reports.sort(key=lambda r: -r['bad_triangle_frac'])
    return dict(
        n_triangles=len(tris),
        n_bad=n_bad,
        n_bad_gated=n_bad_gated,
        weight_frac=float(weight_frac),
        gate=gate,
        R_median=R_median,
        pair_sign_errors=reports,
    )


def _cross_validate_one_seed(observations, tiles, hold_frac: float, seed: int):
    """Single CV split — see `_cross_validate` for the averaging wrapper."""
    import random
    rng = random.Random(seed)
    idxs = list(range(len(observations)))
    rng.shuffle(idxs)
    n_hold = int(hold_frac * len(idxs))
    hold = set(idxs[:n_hold])
    train = [obs for i, obs in enumerate(observations) if i not in hold]
    heldo = [obs for i, obs in enumerate(observations) if i in hold]
    tG_tr = nx.Graph()
    for (_, s, t, _, _) in train: tG_tr.add_edge(s, t)
    comps = list(nx.connected_components(tG_tr))
    if not comps: return None
    giant = max(comps, key=len)
    train = [o for o in train if o[1] in giant and o[2] in giant]
    tiles_tr = sorted(giant)
    H_tr, _ = _build_matrices(train, tiles_tr)
    z_tr, _, _ = _solve_phase(H_tr, tiles_tr)
    r_test, w_test = [], []
    for (_, s, t, r_unit, w) in heldo:
        if s not in z_tr or t not in z_tr: continue
        pred = z_tr[s] * np.conj(z_tr[t])
        r_test.append(np.angle(r_unit * np.conj(pred)))
        w_test.append(w)
    if not r_test: return None
    return np.array(r_test), np.array(w_test)


def _cross_validate(observations, tiles, hold_frac: float = 0.20,
                     n_seeds: int = 5):
    """Robust held-out phase-noise estimate — averaged over `n_seeds`
    random splits so a single unlucky split can't masquerade as
    in-sample. Empirically with the kappa-two-sided weighting a single
    20% split may sit within 0.01° of in-sample (well-determined solve
    plus balanced weights); averaging over seeds surfaces the real
    generalization gap. Also emits a `spread_deg` (max–min across
    seeds) as a stability diagnostic.
    """
    per_seed = []
    for seed in range(n_seeds):
        out = _cross_validate_one_seed(observations, tiles, hold_frac, seed)
        if out is None: continue
        r_test, w_test = out
        ad = np.degrees(np.abs(r_test))
        over90 = ad > 90
        per_seed.append(dict(
            median_deg=float(np.median(ad)),
            p95_deg=float(np.percentile(ad, 95)),
            w_frac_over_90=float(w_test[over90].sum() / w_test.sum()),
            n_test=int(len(r_test)),
        ))
    if not per_seed:
        return dict(median_deg=float('nan'), p95_deg=float('nan'),
                    w_frac_over_90=float('nan'),
                    spread_deg=float('nan'), n_test=0, n_seeds=0)
    return dict(
        median_deg=float(np.mean([s['median_deg'] for s in per_seed])),
        p95_deg=float(np.mean([s['p95_deg'] for s in per_seed])),
        w_frac_over_90=float(np.mean([s['w_frac_over_90'] for s in per_seed])),
        spread_deg=float(max(s['median_deg'] for s in per_seed) -
                          min(s['median_deg'] for s in per_seed)),
        n_test=int(per_seed[0]['n_test']),
        n_seeds=len(per_seed),
    )


# =============================================================================
# Public entry point — drop-in for the deprecated three-step pipeline
# =============================================================================
def _tile14_hop_boundary_phase(G, anchor, harmonic_idx, ref_tile=14,
                                 snr_min=6.0, max_hops=3):
    """Return the tile-`ref_tile` phase for a boundary anchor's flow.

    If the direct anchor edge has good SNR on the reference tile, use
    it. Otherwise walk BFS outward from the anchor and pick the
    highest-SNR reference-tile measurement on any incident edge (KCL
    proxy — for a leaf anchor, all its flow comes through the same
    interior vessel(s)). Returns dict or None."""
    from collections import deque
    visited = {anchor}
    queue = deque([(anchor, 0)])
    best = None
    while queue:
        node, hop = queue.popleft()
        if hop > max_hops: continue
        for nb in G.neighbors(node):
            d = G.edges[node, nb]
            for m in d.get('measurements_piv') or []:
                if m.get('tile_id') != ref_tile: continue
                harms = m['harm']['harmonics']
                if len(harms) <= harmonic_idx: continue
                h = harms[harmonic_idx]
                phi = h.get('phi'); amp = h.get('amp'); snr = h.get('snr_db')
                if any(x is None or not np.isfinite(x)
                       for x in (phi, amp, snr)): continue
                cand = dict(edge=(node, nb), hop=hop+1,
                             phi_raw=float(phi),
                             snr=float(snr), amp=float(amp),
                             ff=m.get('flow_from'), ft=m.get('flow_to'))
                if best is None or cand['snr'] > best['snr']:
                    best = cand
            if nb not in visited:
                visited.add(nb); queue.append((nb, hop+1))
    return best


def _boundary_simultaneity_report(G, harmonic_idx, ref_tile=14):
    """Gauge-free boundary-phase measurement — the load-bearing
    independent validation of the whole pipeline.

    Reads each anchor's flow phase through tile `ref_tile` directly.
    Because the SAME tile measures multiple anchors, the per-tile
    gauge factor cancels in inter-anchor comparisons: reporting the
    spread of anchor phases across a fixed tile requires no sync.

    This function separates:
      - `direct`: tile-`ref_tile` measurement on each anchor's own
        edge (may be low-SNR for some anchors)
      - `best`:   tile-`ref_tile` measurement on the highest-SNR
        vessel reachable within `max_hops` of the anchor (KCL proxy
        for flow into/out of the anchor)

    Returns a dict persisted to `G.graph[f'tile_sync_boundary_H{k+1}']`.
    """
    anchors = [n for n, d in G.nodes(data=True)
               if d.get('boundary_type') in ('source', 'sink')]
    if not anchors:
        return {}

    per_anchor = []
    for a in anchors:
        # Direct
        direct = None
        for nb in G.neighbors(a):
            for m in G.edges[a, nb].get('measurements_piv') or []:
                if m.get('tile_id') != ref_tile: continue
                harms = m['harm']['harmonics']
                if len(harms) <= harmonic_idx: continue
                h = harms[harmonic_idx]
                phi = h.get('phi'); amp = h.get('amp'); snr = h.get('snr_db')
                if any(x is None or not np.isfinite(x)
                       for x in (phi, amp, snr)): continue
                direct = dict(edge=(a, nb), phi_raw=float(phi),
                               snr=float(snr), amp=float(amp),
                               ff=m.get('flow_from'), ft=m.get('flow_to'))
                break
            if direct: break
        best = _tile14_hop_boundary_phase(G, a, harmonic_idx,
                                            ref_tile=ref_tile, snr_min=6.0)
        per_anchor.append(dict(
            anchor=int(a),
            boundary_type=G.nodes[a].get('boundary_type'),
            direct=direct, best=best,
        ))

    def circ_std_deg(phases_rad):
        z = np.mean(np.exp(1j * np.asarray(phases_rad)))
        return float(np.degrees(np.sqrt(-2 * np.log(max(abs(z), 1e-12)))))

    # Same-tile inter-anchor comparison → gauge-free simultaneity check
    direct_phis = [p['direct']['phi_raw'] for p in per_anchor
                    if p['direct'] is not None]
    direct_snrs = [p['direct']['snr']    for p in per_anchor
                    if p['direct'] is not None]
    best_phis   = [p['best']['phi_raw']  for p in per_anchor
                    if p['best'] is not None]
    best_snrs   = [p['best']['snr']      for p in per_anchor
                    if p['best'] is not None]

    return dict(
        ref_tile=int(ref_tile),
        n_anchors=len(anchors),
        # Gauge-free direct: same tile, per-anchor's own edge
        direct_phases_deg=[float(np.degrees(p)) for p in direct_phis],
        direct_snrs_db=direct_snrs,
        direct_circular_std_deg=(circ_std_deg(direct_phis)
                                  if len(direct_phis) >= 2 else float('nan')),
        # Best-SNR proxy: same tile, hop to best-SNR vessel per anchor
        best_phases_deg=[float(np.degrees(p)) for p in best_phis],
        best_snrs_db=best_snrs,
        best_circular_std_deg=(circ_std_deg(best_phis)
                                if len(best_phis) >= 2 else float('nan')),
        per_anchor=per_anchor,
        note=('same-tile phases can be compared directly without sync — '
              'the per-tile gauge cancels. Non-trivial circular_std is '
              'physical noise + wave transit + PIV noise, NOT gauge error.'),
    )


def _z2_anchor_signs(G):
    """{tile_id: +1 or −1} for tiles that measure a boundary anchor edge.

    Uses anchor-edge label direction (as originally): +1 if the tile's
    flow_from/flow_to at the anchor edge matches anatomical convention
    (out-of-source or into-sink), −1 if inverted. This has been observed
    to also (accidentally) correct wrong-branch sync assignments that
    a majority-mean_Q-based rule cannot see.
    """
    signs = {}
    for anchor in G.nodes:
        bt = G.nodes[anchor].get('boundary_type')
        if bt not in ('source', 'sink'): continue
        for nb in G.neighbors(anchor):
            for m in G.edges[anchor, nb].get('measurements_piv') or []:
                ff = m.get('flow_from'); ft = m.get('flow_to')
                tid = m.get('tile_id')
                if ff is None or ft is None or tid is None: continue
                if bt == 'source':
                    physio_ff, physio_ft = anchor, nb
                else:
                    physio_ff, physio_ft = nb, anchor
                s = +1 if (ff == physio_ff and ft == physio_ft) else -1
                if tid not in signs:
                    signs[int(tid)] = s
    return signs


def _z2_pair_relations(G):
    """{(t1, t2): +1 or −1} — +1 if tiles t1, t2 agree on
    flow_from/flow_to on majority of their shared edges, −1 if opposite.
    Purely label-based, no phase data used.
    """
    from collections import defaultdict
    shared = defaultdict(list)
    for u, v, d in G.edges(data=True):
        ms = d.get('measurements_piv') or []
        for i in range(len(ms)):
            for j in range(i+1, len(ms)):
                m1, m2 = ms[i], ms[j]
                t1, t2 = m1.get('tile_id'), m2.get('tile_id')
                if t1 is None or t2 is None or t1 == t2: continue
                ff1, ft1 = m1.get('flow_from'), m1.get('flow_to')
                ff2, ft2 = m2.get('flow_from'), m2.get('flow_to')
                if None in (ff1, ft1, ff2, ft2): continue
                same = (ff1 == ff2 and ft1 == ft2)
                shared[(t1, t2)].append(same)
                shared[(t2, t1)].append(same)
    rel = {}
    for (t1, t2), sames in shared.items():
        rel[(t1, t2)] = +1 if sum(sames) >= len(sames) / 2 else -1
    return rel


def _z2_walk(G):
    """BFS-propagate physiological-direction signs from boundary
    anchor tiles to all reachable tiles. Returns {tile_id: ±1}.
    Uses ONLY flow_from/flow_to labels — no phase data, so it can't be
    confused by low-SNR tiles whose mean_Q sign is a coin flip.

    Result: for each tile, whether to flip its Q phasors by π before
    the U(1) eigenvector sync so that the sync sees observations
    already in a single consistent physiological direction convention.
    """
    from collections import deque
    anchor_signs = _z2_anchor_signs(G)
    if not anchor_signs:
        return {}
    relations = _z2_pair_relations(G)
    tile_sign = dict(anchor_signs)
    visited = set(anchor_signs.keys())
    queue = deque(anchor_signs.keys())
    while queue:
        t1 = queue.popleft()
        for (a, b), rel in relations.items():
            if a != t1 or b in visited: continue
            tile_sign[int(b)] = int(tile_sign[t1]) * int(rel)
            visited.add(b); queue.append(b)
    return tile_sign


def _boundary_anchor_observations(G, harmonic_idx, boost_weight=1000.0,
                                    ref_tile=14, max_hops=3):
    """Pseudo-observations that pin tile `ref_tile` as the sole gauge
    baseline for the whole plexus.

    For each boundary anchor node, find tile `ref_tile`'s highest-SNR
    measurement — direct at the anchor edge if SNR is decent, otherwise
    walk BFS outward and use tile `ref_tile`'s best-SNR incident vessel
    as a KCL proxy for the boundary flow. Canonicalize to physiological
    direction (positive-Q = downstream) via sign(mean_Q). Emit a
    synthetic pair-observation linking tile `ref_tile` to a phantom
    "REF" tile whose gauge is fixed at 0. With boost_weight >> real
    pair weights, the eigenvector solve pins arg(z_{ref_tile}) so that
    all four anchor phases (through this one tile) land at 0°. Every
    other tile's gauge is then determined relative to tile `ref_tile`
    via the ordinary pair-overlap constraints.

    Why ref_tile=14 for stage 21: tile 14 measures all four anatomical
    anchors — 2 arterial sources, 2 venous sinks — with SNR 13–15 dB
    on three and −4 dB on the fourth (rescuable via 3-hop to a
    high-SNR interior vessel of tile 14). Using ONE reference tile
    guarantees all four boundary pins are on the same clock. The
    gauge-free simultaneity of those four measurements (~5.8° circular
    std at H1) is an independent observation persisted in
    G.graph['tile_sync_boundary_H1']; the pin uses that measurement,
    doesn't create it.
    """
    obs = []
    for anchor in G.nodes:
        bt = G.nodes[anchor].get('boundary_type')
        if bt not in ('source', 'sink'): continue
        best = _tile14_hop_boundary_phase(G, anchor, harmonic_idx,
                                            ref_tile=ref_tile,
                                            snr_min=-np.inf,
                                            max_hops=max_hops)
        if best is None: continue
        edge = best['edge']
        mean_Q = None
        for m in G.edges[edge].get('measurements_piv') or []:
            if m.get('tile_id') == ref_tile:
                mean_Q = m['harm'].get('mean_Q')
                break
        if mean_Q is None or not np.isfinite(mean_Q): continue
        physio_sign = +1.0 if mean_Q >= 0 else -1.0
        phi_physio = float(best['phi_raw']) + (
            np.pi if physio_sign < 0 else 0.0)
        obs.append((edge, int(ref_tile), "REF",
                    complex(np.cos(phi_physio), np.sin(phi_physio)),
                    float(boost_weight)))
    return obs


def _kcl_pair_observations(G, harmonic_idx, z2_signs=None, min_snr_db=-5.0,
                             kcl_weight_frac=0.3):
    """KCL-derived pair-observations at interior nodes — a NEW information
    source additive to pair-overlap constraints.

    For each interior node with ≥2 measured incident edges, KCL says
    Σ_e sign_e · Q_e = 0. We encode this as extra pair-observations
    between the tiles measuring incident edges (even if those tiles
    don't share the same edge, which pair-overlap couldn't reach).

    At degree-2 nodes: exact pair constraint.
    At degree ≥3: approximate — pair contribution weighted by how much
    of the node's total balance this pair accounts for (dominance).
    """
    from collections import defaultdict
    obs = []
    for node in G.nodes:
        if G.nodes[node].get('boundary_type'): continue
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2: continue
        edges_info = []
        for nb in neighbors:
            d = G.edges[node, nb]
            best = None
            for m in d.get('measurements_piv') or []:
                harms = m['harm']['harmonics']
                if len(harms) <= harmonic_idx: continue
                h = harms[harmonic_idx]
                phi = h.get('phi'); amp = h.get('amp'); snr = h.get('snr_db')
                ff = m.get('flow_from'); ft = m.get('flow_to')
                tid = m.get('tile_id')
                if any(x is None or not np.isfinite(x)
                       for x in (phi, amp, snr)): continue
                if ff is None or ft is None or tid is None: continue
                if snr < min_snr_db: continue
                if ff == node: sign = +1.0
                elif ft == node: sign = -1.0
                else: continue
                if z2_signs is not None:
                    sign *= z2_signs.get(int(tid), 1)
                if best is None or snr > best['snr']:
                    best = dict(tid=int(tid), sign=sign, amp=amp, phi=phi,
                                snr=snr, edge=(node, nb))
            if best is not None:
                edges_info.append(best)
        if len(edges_info) < 2: continue
        total_amp = sum(abs(e['amp']) for e in edges_info)
        for i in range(len(edges_info)):
            for j in range(i + 1, len(edges_info)):
                ei, ej = edges_info[i], edges_info[j]
                if ei['tid'] == ej['tid']: continue
                kappa_i = 10 ** (min(ei['snr'], 30) / 10)
                kappa_j = 10 ** (min(ej['snr'], 30) / 10)
                w_snr = kappa_i * kappa_j / (kappa_i + kappa_j)
                pair_amp = abs(ei['amp']) + abs(ej['amp'])
                dominance = pair_amp / max(total_amp, 1e-12)
                w = w_snr * dominance * kcl_weight_frac
                if w <= 0: continue
                dir_sign = -ei['sign'] * ej['sign']
                r = dir_sign * np.exp(1j * (ei['phi'] - ej['phi']))
                r_unit = r / max(abs(r), 1e-18)
                s, t = (ei['tid'], ej['tid']) if ei['tid'] < ej['tid'] \
                        else (ej['tid'], ei['tid'])
                if s == ei['tid']:
                    obs.append((ei['edge'], s, t, r_unit, w))
                else:
                    obs.append((ej['edge'], s, t, 1 / r_unit, w))
    return obs


def _cluster_pi_flip(G, harmonic_idx: int, z_dict: Dict,
                      ref_tile: int = 14,
                      min_coherence: float = 0.15,
                      max_iters: int = 3):
    """Post-sync cluster-level π-flip anchored on a reference tile.

    The connection-Laplacian eigenvector objective maximizes
    Σ w cos(δ), which is symmetric under δ→δ+π. When the plexus
    contains a subgraph of tiles that share pair observations mostly
    with each other and only weakly with the rest, the sync can land
    the whole cluster on the +π branch. All internal pair-overlaps
    are still satisfied (differences cancel), but the cluster reads
    as π-off from neighbours in the RENDERED phase map.

    Algorithm (per pass):
      1. For each tile, compute the SNR·amp-weighted circular mean of
         (phi_raw - arg(z_tile)) across its measured edges.
      2. Build a spatial-adjacency graph of tiles by bounding-box
         overlap of their measured-edge midpoints.
      3. Build the phase-agreement subgraph: keep adjacency edges
         where the medians differ by less than 90°.
      4. Connected components of the agreement subgraph are phase
         clusters.
      5. Never flip the cluster containing `ref_tile` (the anatomical
         anchor's tile — for stage 21 that is tile 14). For every
         other cluster, examine its adjacency edges to tiles OUTSIDE
         itself; if >50% of those external contacts disagree by >90°,
         flip every tile in the cluster by π.

    Iterate until no cluster flips (usually 1 iter).

    Returns (z_dict_out, flipped_tiles: list[int]).
    """
    def _wrap(x): return ((x + np.pi) % (2*np.pi)) - np.pi

    def _pass(z_now):
        # Per-tile weighted circular median + edge midpoints
        per_tile = defaultdict(list)
        for u, v, d in G.edges(data=True):
            if 'x' not in G.nodes[u] or 'x' not in G.nodes[v]:
                continue
            mx = 0.5 * (G.nodes[u]['x'] + G.nodes[v]['x'])
            my = 0.5 * (G.nodes[u]['y'] + G.nodes[v]['y'])
            for m in (d.get('measurements_piv') or []):
                tid = m.get('tile_id')
                if tid is None:
                    continue
                z_t = z_now.get(int(tid))
                if z_t is None:
                    continue
                harms = ((m.get('harm') or {}).get('harmonics') or [])
                if harmonic_idx >= len(harms) or harms[harmonic_idx] is None:
                    continue
                h = harms[harmonic_idx]
                phi = h.get('phi'); amp = h.get('amp'); snr = h.get('snr_db')
                if any(x is None or not np.isfinite(x)
                        for x in (phi, amp, snr)):
                    continue
                w = (10 ** (min(snr, 30) / 10.0)) * amp
                per_tile[int(tid)].append(
                    (mx, my, _wrap(phi - float(np.angle(z_t))), w))

        med = {}
        bb = {}
        for tid, recs in per_tile.items():
            phis = np.array([r[2] for r in recs])
            ws = np.array([r[3] for r in recs])
            z = np.sum(ws * np.exp(1j * phis)) / max(ws.sum(), 1e-12)
            med[tid] = (float(np.angle(z)), float(abs(z)))
            xs = np.array([r[0] for r in recs])
            ys = np.array([r[1] for r in recs])
            bb[tid] = (xs.min(), xs.max(), ys.min(), ys.max())

        # Spatial adjacency (bbox overlap)
        adj = defaultdict(set)
        tids = list(bb)
        for i, a in enumerate(tids):
            xa0, xa1, ya0, ya1 = bb[a]
            for b in tids[i+1:]:
                xb0, xb1, yb0, yb1 = bb[b]
                ox = max(0, min(xa1, xb1) - max(xa0, xb0))
                oy = max(0, min(ya1, yb1) - max(ya0, yb0))
                if ox > 0 and oy > 0:
                    adj[a].add(b); adj[b].add(a)

        # Phase-agreement subgraph → connected components
        agree = nx.Graph()
        for t in adj:
            agree.add_node(t)
        for t in adj:
            for nb in adj[t]:
                if t >= nb:
                    continue
                if t not in med or nb not in med:
                    continue
                m_t, c_t = med[t]; m_n, c_n = med[nb]
                if c_t < min_coherence or c_n < min_coherence:
                    continue
                if abs(_wrap(m_t - m_n)) < np.radians(90):
                    agree.add_edge(t, nb)
        clusters = [set(c) for c in nx.connected_components(agree)
                     if len(c) >= 1]

        # Locate reference cluster
        ref_cluster = None
        for c in clusters:
            if ref_tile in c:
                ref_cluster = c
                break

        flipped_here = []
        z_out = dict(z_now)
        if ref_cluster is None:
            return z_out, flipped_here

        for cluster in clusters:
            if cluster is ref_cluster:
                continue
            externals = []
            for t in cluster:
                for nb in adj[t]:
                    if nb in cluster:
                        continue
                    if nb not in med:
                        continue
                    m_nb, c_nb = med[nb]
                    if c_nb < min_coherence:
                        continue
                    m_t, c_t = med.get(t, (None, None))
                    if m_t is None or c_t < min_coherence:
                        continue
                    externals.append((m_t, m_nb))
            if len(externals) < 1:
                continue
            n_disc = sum(1 for mt, mn in externals
                          if abs(_wrap(mt - mn)) > np.radians(90))
            # >50% disagree — flips singletons (n=1) when their one
            # contact disagrees, which is common for boundary tiles like
            # tile 1 that only spatially touch one or two neighbours.
            if n_disc / len(externals) > 0.5:
                for t in cluster:
                    z_out[t] = z_out[t] * np.exp(1j * np.pi)
                flipped_here.extend(sorted(int(t) for t in cluster))
        return z_out, flipped_here

    all_flipped: List[int] = []
    for _ in range(max_iters):
        z_dict, flipped = _pass(z_dict)
        if not flipped:
            break
        all_flipped.extend(flipped)
    return z_dict, all_flipped


def spectral_sync(G, harmonic_idx: int,
                  min_snr_db: float = -np.inf,
                  cross_validate: bool = True,
                  boundary_pin: bool = False,
                  boundary_pin_boost: float = 1000.0,
                  z2_precanonicalize: bool = True,
                  kcl_constraints: bool = True,
                  kcl_min_snr_db: float = -5.0,
                  kcl_weight_frac: float = 0.3,
                  cluster_pi_flip: bool = True,
                  cluster_pi_flip_ref_tile: int = 14,
                  manual_pi_flip_tiles=None,
                  progress=None) -> Dict[int, dict]:
    """
    NOTE on `boundary_pin` (default False):
        Boundaries carry a gauge-free measurement — the four anchor
        edges are measured by the same tile (tile 14 in stage 21),
        so their tile-14 phases can be compared without any sync at
        all. That gives us "the four anchors are within ~2° of each
        other" as an observation, and a topology-only sync's ability
        to reproduce it becomes an end-to-end validation of the whole
        pipeline against data it never used. Pinning consumes that
        validation — you force the sync to reproduce the observation
        rather than earning it. So the default is off. Turn it on
        (with a moderate boost, not 1000×) only when you specifically
        want boundaries to constrain the interior at a documented
        strength, and always report the pinned-vs-unpinned backbone
        delta so the reader knows what the pin cost.
    """
    """Solve per-tile phase + log-amplitude gauges spectrally.

    Returns
    -------
    gauges : {tile_id: {z_re, z_im, log_amp, psi_rad, source}}
        Format identical to the deprecated `sync_tile_gauges` +
        `fill_missing_gauges` + `sign_flip_rescue` pipeline output, so
        the downstream `_tile_gauge_summary` and `apply_gauges_to_graph`
        continue to work unchanged.

    Also writes `G.graph['tile_sync_diagnostics_H{harmonic_idx+1}']`.
    """
    observations, edge_meas = _extract_observations_canonical(
        G, harmonic_idx, min_snr_db=min_snr_db)
    if not observations:
        return {}

    # Giant-component gating — LOUD reporting of drop-outs.
    tG = nx.Graph()
    for (_, s, t, _, _) in observations:
        tG.add_edge(s, t)
    tiles_all = sorted(tG.nodes)
    comps = list(nx.connected_components(tG))
    if not comps:
        return {}
    giant = max(comps, key=len)
    tiles = sorted(giant)
    dropped = [{
        'tile_id': int(t),
        'degree': int(tG.degree[t]),
    } for t in tiles_all if t not in giant]
    if dropped:
        # Loud print so operators notice tiles missing from downstream figures.
        print(f'[tile_sync_u1 H{harmonic_idx+1}] dropped {len(dropped)} '
              f'tile(s) outside giant component: {dropped}')

    observations = [o for o in observations if o[1] in giant and o[2] in giant]

    # Z2 pre-canonicalization (anchor-propagated tile-level physio signs).
    # Runs a label-only BFS from boundary anchor tiles (whose physio
    # direction we know from anatomy) outward, propagating sign via
    # flow_from/flow_to agreement across shared edges. Then multiplies
    # each observation's r by sign_s*sign_t so the sync sees pair
    # overlaps in a single physio convention. Resolves the "wrong-branch
    # cluster" the pure eigenvector can't see — the sync's continuous
    # optimization has multiple valid local maxima (branches), and this
    # tells it which branch to pick via anatomical labelling.
    if z2_precanonicalize:
        z2_signs = _z2_walk(G)
        n_flipped = sum(1 for s in z2_signs.values() if s < 0)
        print(f'[tile_sync_u1 H{harmonic_idx+1}] Z2 walk: {len(z2_signs)} tiles '
              f'reached, {n_flipped} flipped to physio direction')
        observations = [
            (edge, s, t, r * (z2_signs.get(s, 1) * z2_signs.get(t, 1)), w)
            for (edge, s, t, r, w) in observations
        ]
    else:
        z2_signs = {}

    # Boundary anchor pins — Dirichlet-like constraints on tiles that
    # measure anchor edges. Adds a phantom "REF" tile pinned to phase 0,
    # linked to each anchor-measuring tile via a high-weight pseudo-obs
    # whose phase equals the anchor measurement's physiological phase.
    # Effect: after the solve, boundary tiles' gauges land on values that
    # make their boundary edges show phase 0, and the interior tiles
    # arrange around those pins. Turn off with boundary_pin=False for a
    # gauge-neutral (topology-only) sync.
    boundary_obs = []
    if boundary_pin:
        boundary_obs = _boundary_anchor_observations(G, harmonic_idx,
                                                     boost_weight=boundary_pin_boost)
        if boundary_obs:
            print(f'[tile_sync_u1 H{harmonic_idx+1}] pinning {len(boundary_obs)} '
                  f'boundary anchor(s): '
                  f'{[(o[1], o[0]) for o in boundary_obs]}')

    # KCL constraints — additive to pair-overlap constraints.
    # For each interior node with ≥2 measured tiles, add pair observations
    # between edges' tiles from KCL mass balance. This bridges tiles that
    # don't share edges but do share nodes.
    kcl_obs = []
    if kcl_constraints:
        kcl_obs_raw = _kcl_pair_observations(
            G, harmonic_idx, z2_signs=z2_signs,
            min_snr_db=kcl_min_snr_db,
            kcl_weight_frac=kcl_weight_frac)
        kcl_obs = [o for o in kcl_obs_raw
                   if o[1] in giant and o[2] in giant]
        if kcl_obs:
            print(f'[tile_sync_u1 H{harmonic_idx+1}] KCL: added '
                  f'{len(kcl_obs)} node-derived pair observations '
                  f'(weight factor {kcl_weight_frac})')

    # Phase (U(1)) — include phantom REF tile in the tile list if pinning
    tiles_with_ref = list(tiles)
    if boundary_obs:
        tiles_with_ref.append("REF")
    obs_for_solve = list(observations) + boundary_obs + kcl_obs
    H, D_phase = _build_matrices(obs_for_solve, tiles_with_ref)
    z_dict_full, lam_top, frust_n = _solve_phase(H, tiles_with_ref)
    # Strip the phantom REF tile from the output. The eigenvector
    # solve pins the global rotation such that arg(z[tiles[0]]) = 0
    # (see _solve_phase); we additionally rotate so arg(z[REF]) = 0
    # since REF is the physical zero-phase reference for the boundaries.
    if boundary_obs and "REF" in z_dict_full:
        z_ref = z_dict_full["REF"]
        if abs(z_ref) > 1e-12:
            rot = np.conj(z_ref)
            z_dict = {t: z_dict_full[t] * rot for t in tiles}
        else:
            z_dict = {t: z_dict_full[t] for t in tiles}
    else:
        z_dict = {t: z_dict_full[t] for t in tiles}

    # Bake Z2 signs into z_dict. The sync was fit to sign-corrected
    # observations, so its z_arg represents the gauge that maps
    # sign[t]*Q_raw[t] to true H. Storing z'[t] = sign[t]*z[t] means
    # downstream Q_raw/z' == physio-aligned H, no separate sign flag.
    if z2_precanonicalize:
        for t in list(z_dict):
            s = z2_signs.get(t, 1)
            if s < 0:
                z_dict[t] = -z_dict[t]

    # Post-sync cluster-level π-flip anchored on the reference tile.
    # Catches the case where the cos-symmetric eigenvector puts a whole
    # weakly-connected sub-cluster on the wrong branch; flips only the
    # non-reference clusters, as a unit, so internal consistency is
    # preserved. See _cluster_pi_flip docstring for the algorithm.
    cluster_flipped: List[int] = []
    if cluster_pi_flip:
        z_dict, cluster_flipped = _cluster_pi_flip(
            G, harmonic_idx, z_dict,
            ref_tile=cluster_pi_flip_ref_tile)
        if cluster_flipped:
            print(f'[tile_sync_u1 H{harmonic_idx+1}] cluster π-flip: '
                  f'{len(cluster_flipped)} tile(s) flipped: '
                  f'{cluster_flipped}')

    # Explicit user-specified pi flip on top of the automated sync.
    # Applied AFTER cluster_pi_flip so it wins when both act on the
    # same tile. Sign of z (complex phasor) negated in place.
    if manual_pi_flip_tiles:
        manual_flipped = []
        for t in manual_pi_flip_tiles:
            if t in z_dict:
                z_dict[t] = -z_dict[t]
                manual_flipped.append(int(t))
        if manual_flipped:
            print(f'[tile_sync_u1 H{harmonic_idx+1}] manual π-flip: '
                  f'{len(manual_flipped)} tile(s) flipped: '
                  f'{sorted(manual_flipped)}')

    # Amplitude (R Laplacian) — same structure, different scalar per obs
    L, b, D_amp = _build_amp_matrices(observations, tiles, edge_meas)
    log_c = _solve_amp(L, b, tiles)

    # Per-observation residuals (in-sample)
    r_in, w_in = _per_obs_residuals(observations, z_dict)
    ad_in = np.degrees(np.abs(r_in)) if len(r_in) else np.array([np.nan])
    in_med = float(np.median(ad_in)) if len(r_in) else float('nan')
    in_wfrac90 = (float(w_in[ad_in > 90].sum() / w_in.sum())
                   if len(r_in) else float('nan'))

    # Pair-level diagnostics
    #   parity: cheap unweighted triangle-parity indicator (retained for
    #     backwards compatibility; use with caution — see LOPO below).
    #   lopo:   leave-one-pair-out — masking-safe. High-leverage pairs
    #     that fit themselves at the expense of neighbours show up here,
    #     which is the failure mode that made a single-fit residual
    #     rank-flip with weight choice.
    #   Sign-flip candidate list is built from LOPO (median resid near pi).
    parity = _triangle_parity_report(observations, z_dict, tG.subgraph(giant))
    lopo = _leave_one_pair_out(observations, tiles)
    pair_signflip_candidates = sorted(
        [dict(pair=(int(k[0]), int(k[1])), **v) for k, v in lopo.items()
         if v.get('is_signflip_candidate')],
        key=lambda r: -r['n_obs'])
    pair_inconsistent = sorted(
        [dict(pair=(int(k[0]), int(k[1])), **v) for k, v in lopo.items()
         if (not v.get('disconnecting') and not v.get('is_signflip_candidate')
             and v.get('loo_resid_median_deg', 0) > 45)],
        key=lambda r: -r['loo_resid_median_deg'])

    # Per-tile aggregate residuals — catches "one bad PIV tile" masquerading
    # as multiple bad pairs (e.g. tile 31 previously appeared in 4 of 6
    # flagged pairs — a symptom of a tile-level data quality problem).
    tile_agg = _tile_aggregate_residuals(observations, z_dict, tiles)
    tile_agg_sorted = sorted(
        [dict(tile=int(t), **v) for t, v in tile_agg.items()],
        key=lambda r: -r['wrms_deg'])[:10]

    # Cross-validation → honest per-observation phase noise, averaged
    # over multiple seeds to defend against single-split luck.
    if cross_validate:
        cv = _cross_validate(observations, tiles)
    else:
        cv = dict(median_deg=float('nan'), p95_deg=float('nan'),
                  w_frac_over_90=float('nan'), spread_deg=float('nan'),
                  n_test=0, n_seeds=0)

    # ------------------------------------------------------------------
    # Gauge-free boundary simultaneity report — independent validation.
    # ------------------------------------------------------------------
    # Direct-read the four anchor phases from tile 14 (same tile →
    # per-tile gauge cancels in inter-anchor comparisons). Then compare
    # to what the sync predicts at those same anchors (each anchor edge's
    # phase under z_dict). If they agree, the sync reproduces a
    # measurement it never used — end-to-end validation.
    boundary_report = _boundary_simultaneity_report(G, harmonic_idx,
                                                     ref_tile=14)
    # Sync-predicted anchor phases: for each anchor's tile-14 measurement
    # on its own edge, compute phi_raw - arg(z_{tile 14}).
    if 14 in z_dict:
        z14_arg_rad = float(np.angle(z_dict[14]))
        sync_phases = []
        for a_info in boundary_report.get('per_anchor', []):
            d = a_info.get('direct')
            if d is None: continue
            p = d['phi_raw'] - z14_arg_rad
            p = ((p + np.pi) % (2*np.pi)) - np.pi
            sync_phases.append(float(np.degrees(p)))
        boundary_report['sync_predicted_phases_deg'] = sync_phases
        if len(sync_phases) >= 2:
            zbar = np.mean(np.exp(1j * np.radians(sync_phases)))
            boundary_report['sync_predicted_circular_std_deg'] = float(
                np.degrees(np.sqrt(-2 * np.log(max(abs(zbar), 1e-12)))))
    G.graph[f'tile_sync_boundary_H{harmonic_idx+1}'] = boundary_report

    # Persist diagnostics
    diag_key = f'tile_sync_diagnostics_H{harmonic_idx+1}'
    G.graph[diag_key] = dict(
        n_observations=int(len(observations)),
        n_tiles=int(len(tiles)),
        n_pairs=int(sum(1 for _ in tG.subgraph(giant).edges())),
        weight_sum=float(w_in.sum()) if len(w_in) else 0.0,
        lambda_top=lam_top,
        normalized_frustration=frust_n,
        # --- centre-of-distribution residual (stable) ---
        in_sample_median_deg=in_med,
        held_out_median_deg=cv['median_deg'],
        held_out_spread_deg=cv.get('spread_deg', float('nan')),
        held_out_n_seeds=cv.get('n_seeds', 0),
        # Per-observation phase noise floor — averaged over seeds so a
        # single lucky split cannot masquerade as in-sample. Downstream
        # sigma_phi consumers should use this. Averaging 19 shared
        # edges brings 8.7° down to about 2° per-edge; noise floor
        # HALVED vs. the 13° from the old (amp²·SNR double-counted)
        # weighting, so downstream error budgets need updating.
        phase_noise_floor_deg=cv['median_deg'],
        # --- tail statistics (independent number, do NOT read as median
        # correction: the tail generalizes much worse than the centre) ---
        in_sample_w_frac_over_90=in_wfrac90,
        held_out_w_frac_over_90=cv['w_frac_over_90'],
        # --- triangle parity (kept for backwards compat) ---
        n_triangles=parity['n_triangles'],
        n_bad_triangles=parity['n_bad'],
        n_bad_triangles_gated=parity['n_bad_gated'],
        bad_triangle_weight_frac=parity['weight_frac'],
        # --- LOPO — the load-bearing pair diagnostic (masking-safe) ---
        pair_sign_errors=pair_signflip_candidates,     # median LOO resid near π
        pair_inconsistent=pair_inconsistent,            # large but NOT near π
        # --- per-tile aggregate (catches bad-PIV-tile-as-many-bad-pairs) ---
        tile_residuals_top10=tile_agg_sorted,
        tiles_missing_from_giant=dropped,
        cluster_pi_flipped_tiles=list(cluster_flipped),
    )

    # Assemble gauges dict in the shape the existing apply code expects.
    out = {}
    for tid in tiles:
        z = z_dict.get(tid, 1.0 + 0j)
        # Fold log-amplitude into z, matching the old convention where
        # z_re/z_im carried both magnitude and phase.
        mag = float(np.exp(log_c.get(tid, 0.0)))
        z_full = mag * z
        out[int(tid)] = dict(
            z_re=float(z_full.real),
            z_im=float(z_full.imag),
            log_amp=float(np.log(max(abs(z_full), 1e-12))),
            psi_rad=float(np.angle(z_full)),
            source='spectral',
        )
    return out
