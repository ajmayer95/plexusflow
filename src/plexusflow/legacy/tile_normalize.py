"""Per-tile mosaic normalization: recover a complex gauge z_t per tile per
harmonic so measurements across tiles can be compared on the same axes.

The pipeline is:

  1. `sync_tile_gauges(G, harmonic_idx=0, ...)`
        U(1) phase-sync + weighted L+ log-amplitude solve on the tile-pair
        graph.  Uses only tile pairs above `min_snr_db` for the primary
        sync.  Returns `{tile_id: {'z_re', 'z_im', 'source': 'primary'}}`.

  2. `fill_missing_gauges(gauges, G, harmonic_idx, ...)`
        Multi-hop fill-in for tiles missing from the primary sync.  Each
        hop widens the reference set to include tiles filled in the prior
        hop.  Rejects fills with phase-concentration below `min_rbar`.
        Adds `source: 'fillin'` (or 'borrowed' for the final
        nearest-neighbor stage) to newly-filled entries.

  3. `sign_flip_rescue(gauges, G, harmonic_idx, ...)`
        Post-hoc greedy pass: any tile whose per-neighbor phase residual
        drops below 45° after a global sign flip gets flipped.  Fixes the
        classic remaining-180° discontinuities in low-SNR regions.

  4. `sync_amp_gauges(G, min_snr_db=6.0, ...)`
        Weighted-L+ leveling of `mean_Q` alone → per-tile `CO_rel`
        (= G_DC in the unified gauge model).  No separate PI_rel sync:
        after normalizing DC by CO_rel and H1 by |z_H1|, the leveled
        pulsatility 2·|H1|/|Q̄| falls out automatically.

  5. `fill_missing_amp_gauges(amp_gauges, G, ...)`
        Multi-hop + neighbor-borrow for tiles missing CO_rel (mirrors
        step 2's pattern).  Ensures every tile with any measurement gets
        a DC gauge; low-SNR tiles inherit from their strongest neighbor.

  6. `apply_gauges_to_graph(G, gauges_h1, gauges_h2, amp_gauges)`
        Writes per-tile normalized values SIDE-BY-SIDE with the raw:
          - `measurements_piv_normalized` on each edge
          - `_h_{amp,phase,Z}_H{k}_norm`, `_h_amp_DC_norm` cache fields
          - `G.graph['tile_gauges']` (per-tile summary)
          - `G.graph['f0_canonical_hz']` (median plateau f0)
        Raw `measurements_piv` is UNTOUCHED.  The viewer can display either
        by switching data source.

  7. `normalize_graph_in_place(G, params, progress=...)`
        One-shot convenience that runs steps 1-6.  `progress` is an
        optional callable `(stage_name: str, pct: float)` for UI
        integration.

Unified gauge model (stored on `G.graph['tile_gauges']`):

    G_DC ≡ CO_rel  ∈ ℝ⁺   — DC amplitude gauge (mean_Q dividend)
    G_H1 ≡ z_H1    ∈ ℂ    — H1 phasor gauge  (|·| = amp, arg = phase)
    G_H2 ≡ z_H2    ∈ ℂ    — H2 phasor gauge
    f0_hz                  — canonical fundamental (median plateau f0)

Field names in tile_gauges (kept for storage stability):
    CO_rel, z_re, z_im, z_mag, z_arg_deg,
    z_H2_re, z_H2_im, z_H2_mag, z_H2_arg_deg

All functions operate in-memory on the passed `G` and return the computed
dicts as well.  No file I/O — call `pickle.dump` yourself if you want
persistence.

Convention: `harmonic_idx=0` means H1 (the fundamental), `idx=1` means H2.

Historical scripts this replaces:
    scratchpad/tile_sync_solve.py
    scratchpad/save_pi_co_gauges_v2.py
    scratchpad/sign_flip_rescue.py
    scratchpad/write_gauges_to_graph.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..io.measurement_view import MeasurementView

# =============================================================================
# Parameters
# =============================================================================


@dataclass
class NormalizeParams:
    """All tunables for the normalization pipeline in one struct."""

    # Phase-gauge anchor: the tile whose psi_rad gets pinned to 0 after
    # the eigenvector sync.  All other tiles' z_arg values (and thus all
    # per-edge phases produced by this normalization) are expressed in
    # that tile's H1 phase frame.  None → use tiles[0] (default).
    anchor_tile: Optional[int] = None

    # Sync primary (per-harmonic; H2 is set separately since it has lower SNR)
    min_snr_db_h1: float = 3.0
    min_snr_db_h2: float = 0.0

    # Fill-in (used for both harmonics)
    fillin_min_snr_db: float = -6.0
    fillin_min_rbar: float = 0.40
    fillin_zmag_cap: Tuple[float, float] = (0.3, 3.0)
    max_hops: int = 5

    # Sign-flip rescue
    rescue_flip_threshold_deg: float = 90.0
    rescue_min_improvement: float = 0.5   # fractional drop in per-tile residual
    rescue_final_max_deg: float = 60.0    # must land below this to commit.
                                          # Was 45° originally but tiles with
                                          # low pulsatility (weak H1) legitimately
                                          # have noisy phase gauges; 60° catches
                                          # the "almost" flips (e.g. tile 17
                                          # going 132° → 48°) without becoming
                                          # over-permissive.
    rescue_max_iters: int = 20

    # Amplitude gauges (PI_rel, CO_rel)
    amp_min_snr_db: float = 6.0
    amp_min_mean_Q: float = 1e-4
    amp_n_min: int = 2                    # min shared vessels per tile pair

    # H2 sync enabled?  If False, H2 falls back to H1 gauge everywhere.
    compute_h2: bool = True

    # Sign-flip rescue enabled?
    run_sign_flip_rescue: bool = True

    # AC-amplitude leveling policy.
    #   False (default) — apply only DC leveling and AC PHASE alignment.
    #     Preserves per-tile pulsatility as a biological signal (e.g.
    #     regions near the heart are truly more pulsatile than periphery).
    #     Recommended when the physical assumption is that PI varies
    #     regionally.
    #   True  — also divide AC amplitudes by |z_H_k| (unified gauge model).
    #     Forces normalized PI ≈ constant across tiles.  Useful for
    #     cross-tile "shape check" (do shared vessels agree on waveform
    #     morphology once amplitude bias is removed?), but destroys
    #     regional PI variation.
    level_ac_amplitude: bool = False

    # Post-sync cluster-level π-flip anchored on `cluster_pi_flip_ref_tile`.
    # Fixes the cos-symmetric eigenvector failure where a weakly-connected
    # sub-cluster of tiles lands on the +π branch relative to the rest of
    # the map. See _cluster_pi_flip in tile_sync_u1.
    cluster_pi_flip: bool = True
    cluster_pi_flip_ref_tile: int = 14

    # Boundary-anchor pinning during the eigenvector sync itself.
    # When True, adds high-weight pseudo-observations that tie the four
    # tile-14 boundary-anchor edges to the phase of the boundary node's
    # existing `bc_harmonics`. This gives the sync a physio-anchored
    # constraint that survives when weak cross-lobe topology otherwise
    # leaves the two anatomical lobes free to land on opposite branches
    # of the cos-symmetric eigenvector solution.
    # `boundary_pin_boost` sets the weight of the pin relative to a
    # typical measured-pair observation (weights are already SNR-scaled).
    boundary_pin: bool = True   # migrated to Recipe B (was False)
    boundary_pin_boost: float = 10.0

    # Explicit per-tile pi-flip applied AFTER the sync + cluster_pi_flip
    # step. Empty by default. Use this when the automated sync leaves
    # specific tiles on the wrong branch and you can identify them from
    # a diagnostic (e.g. per-tile phase relative to the boundary anchor).
    # The set is applied to both H1 and H2 gauges.
    manual_pi_flip_tiles: frozenset = frozenset()

    # Post-normalize per-edge pi-flip against the boundary-anchor
    # convention: all boundaries must be in phase at H1, and every
    # interior edge should sit within `boundary_anchor_edge_flip_thresh_deg`
    # of the arterial-source anchor phase (phases lag from boundary, they
    # do not lead by pi). Any edge whose normalized H1 phase falls
    # further than the threshold gets its H1 and H2 phasors negated.
    # This is the "phase should lag from boundary" convention codified
    # as a normalize-pipeline step so a viewer Normalize click reproduces
    # the corrected phase map.
    boundary_anchor_edge_flip: bool = True
    boundary_anchor_edge_flip_thresh_deg: float = 90.0

    # Post-normalize global H2 branch check.  H1 aggregation is anchored
    # by boundary_anchor_edge_flip above; H2 has no anchor because
    # `_resolve_signs_max_coherence` picks per-harmonic signs
    # independently.  This step compares aggregated H2 phase at DA-stub
    # edges to their tile-14 raw H2 phase; if they systematically
    # disagree by π, every edge's H2 phasor is rotated by π.  Fixes the
    # bumpy-rise pulse-shape artifact observed on 2026-09-30 in outlet
    # templates.  See scratchpad/diag_outlet_template_h2.py.
    h2_global_branch_check: bool = True

    # Fringe-tile cutoff for the cross-tile phasor / DC aggregation that
    # feeds `_h_*_H*_norm` on each edge. Tiles with id <= this value are
    # excluded from `_stat_mean_H` and `_stat_mean_DC`. Consistent with
    # `_canonical_f0`, which already treats tiles 1..6 as the "frequency-
    # anomalous fringe" for the Stage-21 mosaic. Default 5 matches the
    # "tiles 1-5 unreliable" rule of thumb; set to 0 to disable.
    fringe_tile_cutoff: int = 5

    # Radius-consistency gate for the cross-tile aggregation. For each
    # edge, compute the per-tile R_profile_um median across measurements that
    # already pass fringe_tile_cutoff, then drop tiles whose R deviates
    # from that median by more than this fraction. Rationale: Q ~ R²·V, so
    # a per-tile R inference error of X% inflates that tile's mean_Q and
    # |H_k| by ~2X%, which then dominates the SNR-weighted aggregate if
    # that tile happens to have high H1 SNR. Empirically validated on
    # edge (50136,50144) in Somites21: tile 9's R = 33.1 µm vs. tile
    # median 25.2 µm → dropped, aggregate DC 0.154 → 0.042 (much closer
    # to sim). Set to a large value (e.g. 10.0) to effectively disable.
    radius_consistency_rel_thresh: float = 0.30

    def per_harmonic_min_snr(self, harmonic_idx: int) -> float:
        return self.min_snr_db_h1 if harmonic_idx == 0 else self.min_snr_db_h2


ProgressCb = Optional[Callable[[str, float], None]]


def _emit(progress: ProgressCb, stage: str, pct: float) -> None:
    if progress is not None:
        try:
            progress(stage, float(pct))
        except Exception:
            pass


# =============================================================================
# Per-edge measurement extraction (shared by sync + rescue)
# =============================================================================


def _extract_edge_measurements(G, harmonic_idx: int, min_snr_db: float = -np.inf):
    """Rebuild per-edge phasor measurements from the graph's raw
    `measurements_piv`, with canonical flow direction applied.  Returns
    `{(u, v): [dict, ...]}` where each dict has:

        tile   : int tile_id
        Q      : complex phasor, sign-flipped to match (u, v) canonical
        sigma  : per-measurement phase-noise proxy
        f0     : Hz (fundamental)
        snr    : dB
        amp    : |Q| pre-sign (positive)
    """
    edge_meas = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        for m in (ed.get('measurements_piv') or []):
            mv = MeasurementView(m)
            harms = mv.harm.harmonics or []
            if len(harms) <= harmonic_idx:
                continue
            h = harms[harmonic_idx]
            phi = h.get('phi')
            amp = h.get('amp')
            snr = h.get('snr_db')
            f0 = mv.harm.f0_hz
            if any(x is None for x in (phi, amp, snr, f0)):
                continue
            if not (np.isfinite(phi) and np.isfinite(amp)
                    and np.isfinite(snr) and np.isfinite(f0)):
                continue
            if amp <= 0 or f0 <= 0 or snr < min_snr_db:
                continue
            ff, ft = mv.flow_from, mv.flow_to
            if ff is None or ft is None:
                continue
            if (ff, ft) == (u, v):
                sign = +1.0
            elif (ff, ft) == (v, u):
                sign = -1.0
            else:
                continue
            Q_canonical = sign * amp * np.exp(1j * phi)
            sig_A = h.get('sigma_A')
            sig_B = h.get('sigma_B')
            if (sig_A and sig_B and np.isfinite(sig_A) and np.isfinite(sig_B)):
                sigma = float(np.sqrt(0.5 * (sig_A**2 + sig_B**2)))
            else:
                sigma = float(amp / 10**(min(snr, 30) / 20))
            edge_meas[(u, v)].append(dict(
                tile=int(mv.tile_id),
                Q=Q_canonical,
                sigma=sigma,
                f0=float(f0),
                snr=float(snr),
                amp=float(amp),
            ))
    return edge_meas


# =============================================================================
# STEP 1: Primary tile-graph U(1) sync + amplitude L⁺ solve
# =============================================================================


def sync_tile_gauges(G, harmonic_idx: int, params: NormalizeParams,
                     progress: ProgressCb = None) -> Dict[int, dict]:
    """Recover z_t = |z_t| · exp(i·psi_t) per tile for the given harmonic.

    Returns `{tile_id: {'z_re', 'z_im', 'log_amp', 'psi_rad', 'source'}}`
    with `source='primary'` for tiles that passed the primary sync.
    Tiles missing from the return value need to be filled by
    `fill_missing_gauges`.
    """
    _emit(progress, f'sync H{harmonic_idx+1}: extract', 0.05)
    min_snr = params.per_harmonic_min_snr(harmonic_idx)
    edge_meas = _extract_edge_measurements(G, harmonic_idx, min_snr_db=-np.inf)

    # ------------------------------------------------------------------
    # Build tile-pair overlaps (SNR-gated for primary sync)
    # ------------------------------------------------------------------
    overlaps = defaultdict(list)
    for edge, ms in edge_meas.items():
        if len(ms) < 2:
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if ms[i]['tile'] == ms[j]['tile']:
                    continue
                if ms[i]['snr'] < min_snr or ms[j]['snr'] < min_snr:
                    continue
                s_idx, t_idx = ((i, j) if ms[i]['tile'] < ms[j]['tile']
                                else (j, i))
                s = ms[s_idx]['tile']
                t = ms[t_idx]['tile']
                Q_s = ms[s_idx]['Q']
                Q_t = ms[t_idx]['Q']
                log_amp = np.log(abs(Q_s)) - np.log(abs(Q_t))
                phase = float(((np.angle(Q_s) - np.angle(Q_t)) + np.pi)
                              % (2 * np.pi) - np.pi)
                overlaps[(s, t)].append(dict(log_amp=log_amp, phase=phase))

    # Aggregate per pair
    pair_summary = {}
    for (s, t), recs in overlaps.items():
        if len(recs) < 3:
            continue
        phases = np.array([r['phase'] for r in recs])
        log_amps = np.array([r['log_amp'] for r in recs])
        z_bar = np.mean(np.exp(1j * phases))
        R_bar = float(abs(z_bar))
        psi_st = float(np.angle(z_bar))
        med_log = float(np.median(log_amps))
        mad_log = float(np.median(np.abs(log_amps - med_log)))
        n = len(recs)
        pair_summary[(s, t)] = dict(
            n=n, psi=psi_st, log_amp=med_log,
            R_bar=R_bar, mad_log=mad_log,
            w_phase=n * R_bar,
            w_amp=n / max(mad_log, 0.03),
        )

    if not pair_summary:
        return {}

    # ------------------------------------------------------------------
    # Reduce to giant connected component
    # ------------------------------------------------------------------
    _emit(progress, f'sync H{harmonic_idx+1}: solve', 0.20)
    tiles = sorted({t for pair in pair_summary for t in pair})
    tG = nx.Graph()
    tG.add_edges_from(pair_summary.keys())
    tG.add_nodes_from(tiles)
    giant = max(nx.connected_components(tG), key=len)
    if len(giant) < len(tiles):
        tiles = sorted(giant)
        pair_summary = {p: v for p, v in pair_summary.items()
                        if p[0] in giant and p[1] in giant}
    tile_idx = {t: i for i, t in enumerate(tiles)}
    T = len(tiles)

    # ------------------------------------------------------------------
    # Phase sync via leading eigenvector of Hermitian overlap matrix
    # ------------------------------------------------------------------
    H = sp.lil_matrix((T, T), dtype=complex)
    for (s, t), summary in pair_summary.items():
        i, j = tile_idx[s], tile_idx[t]
        w = summary['w_phase']
        phase_offset = summary['psi']
        H[i, j] = w * np.exp(1j * phase_offset)
        H[j, i] = w * np.exp(-1j * phase_offset)
    H = H.tocsr()

    # eigsh needs at least k+1 rows; k=1 works for T>=2
    k = min(3, max(1, T - 1))
    eigvals, eigvecs = spla.eigsh(H, k=k, which='LA')
    v = eigvecs[:, -1]
    psi_recovered = np.angle(v)

    # ------------------------------------------------------------------
    # BFS 2-coloring to resolve the sign ambiguity in each connected
    # component of the pair-residual graph
    # ------------------------------------------------------------------
    initial_resid = {}
    for (s, t), summary in pair_summary.items():
        i, j = tile_idx[s], tile_idx[t]
        pred = psi_recovered[i] - psi_recovered[j]
        r = ((summary['psi'] - pred + np.pi) % (2 * np.pi)) - np.pi
        initial_resid[(s, t)] = np.degrees(r)

    # Phase-gauge anchor: pin either the user-chosen tile, or tiles[0].
    anchor = (int(params.anchor_tile)
                if (params.anchor_tile is not None
                    and int(params.anchor_tile) in tile_idx)
                else tiles[0])
    color = {t: None for t in tiles}
    color[anchor] = 0
    adjacency = defaultdict(list)
    for (s, t), r in initial_resid.items():
        label = 'flip' if abs(r) > 90 else 'same'
        adjacency[s].append((t, label))
        adjacency[t].append((s, label))
    queue = [anchor]
    while queue:
        t = queue.pop(0)
        for (t2, label) in adjacency[t]:
            needed = color[t] if label == 'same' else 1 - color[t]
            if color[t2] is None:
                color[t2] = needed
                queue.append(t2)
    for tile in tiles:
        if color[tile] == 1:
            psi_recovered[tile_idx[tile]] += np.pi
    psi_recovered = ((psi_recovered + np.pi) % (2 * np.pi)) - np.pi
    psi_recovered = ((psi_recovered - psi_recovered[tile_idx[anchor]] + np.pi)
                     % (2 * np.pi)) - np.pi

    # ------------------------------------------------------------------
    # Amplitude L+ solve
    # ------------------------------------------------------------------
    _emit(progress, f'sync H{harmonic_idx+1}: amp solve', 0.35)
    rows, cols, vals_L = [], [], []
    b = np.zeros(T)
    for (s, t), summary in pair_summary.items():
        i, j = tile_idx[s], tile_idx[t]
        w = summary['w_amp']
        d = summary['log_amp']
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        vals_L += [w, w, -w, -w]
        b[i] += w * d
        b[j] -= w * d
    L = sp.csr_matrix((vals_L, (rows, cols)), shape=(T, T))
    # Pin the amplitude gauge at the same anchor tile used for phase.
    pin = tile_idx[anchor]
    keep = np.array([k for k in range(T) if k != pin])
    if len(keep) > 0:
        L_red = L.tolil()[keep][:, keep].tocsc()
        log_amp_red = spla.spsolve(L_red, b[keep])
        log_z = np.zeros(T)
        log_z[keep] = log_amp_red
        log_z[pin] = 0.0
    else:
        log_z = np.zeros(T)

    # ------------------------------------------------------------------
    # Pack into dict
    # ------------------------------------------------------------------
    out = {}
    for i, t in enumerate(tiles):
        z = np.exp(log_z[i] + 1j * psi_recovered[i])
        out[int(t)] = dict(
            log_amp=float(log_z[i]),
            psi_rad=float(psi_recovered[i]),
            z_re=float(z.real),
            z_im=float(z.imag),
            source='primary',
        )
    return out


# =============================================================================
# STEP 2: Multi-hop fill-in + nearest-neighbor borrow
# =============================================================================


def fill_missing_gauges(gauges: Dict[int, dict], G, harmonic_idx: int,
                        params: NormalizeParams,
                        progress: ProgressCb = None) -> Dict[int, dict]:
    """Fill in tiles missing from the primary sync via multi-hop weighted
    circular mean of `z_ref · (Q_me / Q_ref)`, plus a final
    nearest-neighbor borrow fallback for tiles that still lack coverage.

    Mutates and returns `gauges` (in-place).  Newly-added tiles carry
    `source='fillin'` or `source='borrowed'`.
    """
    edge_meas_all = _extract_edge_measurements(
        G, harmonic_idx, min_snr_db=-np.inf)

    all_tiles_seen = set()
    for ms in edge_meas_all.values():
        for m in ms:
            all_tiles_seen.add(m['tile'])
    remaining = set(all_tiles_seen) - set(gauges.keys())

    def _z_of(tid):
        e = gauges.get(int(tid))
        if e is None:
            return None
        return complex(e['z_re'], e['z_im'])

    # --------------------- Multi-hop primary→fillin→fillin ---------------
    for hop in range(1, params.max_hops + 1):
        if not remaining:
            break
        reference_tiles = set(gauges.keys())
        newly_filled_this_hop = {}
        for t_miss in sorted(remaining):
            estimates = []
            z_ref_list = []
            for edge, ms in edge_meas_all.items():
                me = [m for m in ms if m['tile'] == t_miss]
                if not me:
                    continue
                others = [m for m in ms if m['tile'] in reference_tiles]
                if not others:
                    continue
                for m_me in me:
                    for m_ref in others:
                        if m_me['snr'] < params.fillin_min_snr_db:
                            continue
                        if m_ref['snr'] < params.fillin_min_snr_db:
                            continue
                        z_ref = _z_of(m_ref['tile'])
                        if z_ref is None:
                            continue
                        if abs(m_ref['Q']) < 1e-12:
                            continue
                        z_est = z_ref * (m_me['Q'] / m_ref['Q'])
                        snr_min = min(m_me['snr'], m_ref['snr'])
                        if not np.isfinite(snr_min):
                            continue
                        snr_lin = 10 ** (min(snr_min, 30) / 10)
                        amp_min = min(m_me['amp'], m_ref['amp'])
                        hop_weight = 0.5 ** (hop - 1)
                        w = amp_min ** 2 * snr_lin * hop_weight
                        estimates.append((z_est, w))
                        z_ref_list.append(z_ref)
            if not estimates:
                continue
            zs = np.array([z for z, _ in estimates])
            ws = np.array([w for _, w in estimates])
            z_ref_arr = np.array(z_ref_list)
            signs = np.sign(np.real(zs * np.conj(z_ref_arr)))
            signs[signs == 0] = 1
            zs_aligned = zs * signs
            unit_zs = zs_aligned / np.abs(zs_aligned)
            z_phase_bar = np.sum(unit_zs * ws) / np.sum(ws)
            psi_bar = np.angle(z_phase_bar)
            concentration = abs(z_phase_bar)
            if concentration < params.fillin_min_rbar:
                continue   # too diffuse — reject, try next hop or fall through
            log_mags = np.log(np.abs(zs))
            log_mag_bar = np.average(log_mags, weights=ws)
            mag_bar = np.exp(log_mag_bar)
            mag_capped = float(np.clip(mag_bar, *params.fillin_zmag_cap))
            z_est_final = mag_capped * np.exp(1j * psi_bar)
            newly_filled_this_hop[t_miss] = dict(
                log_amp=float(np.log(max(abs(z_est_final), 1e-9))),
                psi_rad=float(psi_bar),
                z_re=float(z_est_final.real),
                z_im=float(z_est_final.imag),
                source='fillin',
                fillin_hop=hop,
                fillin_rbar=float(concentration),
            )
        gauges.update(newly_filled_this_hop)
        remaining -= set(newly_filled_this_hop.keys())
        _emit(progress, f'fill-in hop {hop}', 0.4 + 0.03 * hop)
        if not newly_filled_this_hop:
            break

    # --------------------- Nearest-neighbor borrow (final stage) ---------
    if remaining:
        gauged = set(gauges.keys())
        for t_miss in sorted(remaining):
            neighbor_scores = defaultdict(float)
            for edge, ms in edge_meas_all.items():
                me = [m for m in ms if m['tile'] == t_miss]
                if not me:
                    continue
                for m_me in me:
                    if m_me['snr'] < params.fillin_min_snr_db:
                        continue
                    for m_n in ms:
                        if m_n['tile'] == t_miss:
                            continue
                        if m_n['tile'] not in gauged:
                            continue
                        if m_n['snr'] < params.fillin_min_snr_db:
                            continue
                        snr_min = min(m_me['snr'], m_n['snr'])
                        snr_lin = 10 ** (min(snr_min, 30) / 10)
                        amp_min = min(m_me['amp'], m_n['amp'])
                        neighbor_scores[m_n['tile']] += amp_min ** 2 * snr_lin
            if not neighbor_scores:
                continue
            best = max(neighbor_scores, key=neighbor_scores.get)
            z_b = _z_of(best)
            if z_b is None:
                continue
            gauges[int(t_miss)] = dict(
                log_amp=float(np.log(max(abs(z_b), 1e-9))),
                psi_rad=float(np.angle(z_b)),
                z_re=float(z_b.real),
                z_im=float(z_b.imag),
                source='borrowed',
                borrowed_from=int(best),
            )

    return gauges


# =============================================================================
# STEP 3: Sign-flip rescue
# =============================================================================


def sign_flip_rescue(gauges: Dict[int, dict], G, harmonic_idx: int,
                     params: NormalizeParams,
                     progress: ProgressCb = None) -> Tuple[Dict[int, dict], List[int]]:
    """Greedy pass: for each tile, if flipping (z → -z) drops the tile's
    weighted per-neighbor phase residual by ≥`rescue_min_improvement` and
    the result is below `rescue_final_max_deg`, commit the flip.  Iterate
    until stable or `rescue_max_iters`.

    Mutates and returns `gauges`.  Also returns list of flipped tile IDs.
    """
    edge_meas = _extract_edge_measurements(
        G, harmonic_idx, min_snr_db=params.fillin_min_snr_db)

    # Precompute per-tile pair-observations: list of (neighbor, Q_ratio_obs, w)
    tile_pair_obs = defaultdict(list)
    for edge, ms in edge_meas.items():
        if len(ms) < 2:
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                if a['tile'] == b['tile']:
                    continue
                if abs(b['Q']) < 1e-12:
                    continue
                obs_ratio = a['Q'] / b['Q']
                snr_lin = 10 ** (min(min(a['snr'], b['snr']), 30) / 10)
                amp_min = min(a['amp'], b['amp'])
                w = amp_min ** 2 * snr_lin
                tile_pair_obs[a['tile']].append((b['tile'], obs_ratio, w))
                tile_pair_obs[b['tile']].append((a['tile'], 1.0 / obs_ratio, w))

    def z_dict_local():
        return {t: complex(g['z_re'], g['z_im']) for t, g in gauges.items()}

    def tile_residual(t: int, zdict: Dict[int, complex]) -> Optional[float]:
        if t not in zdict:
            return None
        z_t = zdict[t]
        rs = []
        ws = []
        for (nbr, ratio_obs, w) in tile_pair_obs[t]:
            if nbr not in zdict:
                continue
            ratio_pred = z_t / zdict[nbr]
            r = np.angle(ratio_obs / ratio_pred)
            rs.append(np.degrees(r))
            ws.append(w)
        if not rs:
            return None
        return float(np.average(np.abs(rs), weights=ws))

    flipped: List[int] = []
    for iteration in range(1, params.rescue_max_iters + 1):
        zdict = z_dict_local()
        n_this = 0
        for t in sorted(gauges):
            r_before = tile_residual(t, zdict)
            if r_before is None:
                continue
            zdict_alt = dict(zdict)
            zdict_alt[t] = -zdict[t]
            r_after = tile_residual(t, zdict_alt)
            if r_after is None:
                continue
            if ((r_before - r_after) / max(r_before, 1e-6)
                    >= params.rescue_min_improvement
                    and r_after < params.rescue_final_max_deg):
                zdict[t] = -zdict[t]
                gauges[t]['z_re'] = float(zdict[t].real)
                gauges[t]['z_im'] = float(zdict[t].imag)
                gauges[t]['psi_rad'] = float(np.angle(zdict[t]))
                gauges[t]['sign_flipped_rescue'] = True
                flipped.append(t)
                n_this += 1
        if n_this == 0:
            break
    _emit(progress, 'sign-flip rescue', 0.75)
    return gauges, flipped


# =============================================================================
# STEP 4: Amplitude gauges (PI_rel, CO_rel, plus tile f0)
# =============================================================================


def sync_amp_gauges(G, params: NormalizeParams,
                    progress: ProgressCb = None) -> Dict[int, dict]:
    """Recover per-tile PI_rel (pulsatility drift) and CO_rel (mean-Q drift)
    via weighted-L+ leveling on the tile-pair graph.

    Also records median f0_hz per tile.  Returns
    `{tile_id: {'PI_rel', 'CO_rel', 'f0_hz'}}`.
    """
    edge_data = defaultdict(list)
    tile_f0 = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        for m in (ed.get('measurements_piv') or []):
            mv = MeasurementView(m)
            tid = mv.tile_id
            f0 = mv.harm.f0_hz
            mean_Q = mv.harm.mean_Q
            # For CO_rel (=G_DC) leveling we gate on DC SNR, not H1 SNR
            # — tiles with strong DC but weak pulsatility (low H1) should
            # still enter this solve, since their mean_Q is well-measured.
            snr_dc = mv.snr.dc_db
            if any(x is None for x in (f0, mean_Q, snr_dc)):
                continue
            if not (np.isfinite(f0) and np.isfinite(mean_Q)
                    and np.isfinite(snr_dc)):
                continue
            if abs(mean_Q) < params.amp_min_mean_Q:
                continue
            edge_data[(u, v)].append(dict(
                tile=int(tid), mean_Q=abs(float(mean_Q)),
                f0=float(f0), snr=float(snr_dc)))
            tile_f0[int(tid)].append(float(f0))

    def _pair_ratios(kind: str):
        out = defaultdict(list)
        for edge, ms in edge_data.items():
            if len(ms) < 2:
                continue
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    if ms[i]['tile'] == ms[j]['tile']:
                        continue
                    if (ms[i]['snr'] < params.amp_min_snr_db
                            or ms[j]['snr'] < params.amp_min_snr_db):
                        continue
                    si, ti = ((i, j) if ms[i]['tile'] < ms[j]['tile']
                              else (j, i))
                    s, t = ms[si]['tile'], ms[ti]['tile']
                    out[(s, t)].append(
                        np.log(ms[si][kind] / ms[ti][kind]))
        return out

    def _summarize(pair_ratios):
        out = {}
        for k, rs in pair_ratios.items():
            if len(rs) < params.amp_n_min:
                continue
            rs_clip = np.clip(rs, -3.0, 3.0)
            med = float(np.median(rs_clip))
            out[k] = dict(n=len(rs_clip), med=med)
        return out

    def _solve(psum):
        if not psum:
            return [], np.zeros(0)
        tiles = sorted({t for p in psum for t in p})
        tG = nx.Graph()
        tG.add_edges_from(psum.keys())
        tG.add_nodes_from(tiles)
        giant = max(nx.connected_components(tG), key=len)
        tiles = sorted(giant)
        T = len(tiles)
        tidx = {t: i for i, t in enumerate(tiles)}
        ps = {p: v for p, v in psum.items()
              if p[0] in giant and p[1] in giant}
        rows, cols, vals = [], [], []
        b = np.zeros(T)
        for (s, t), r in ps.items():
            i, j = tidx[s], tidx[t]
            w = float(r['n'])
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [w, w, -w, -w]
            b[i] += w * r['med']
            b[j] -= w * r['med']
        L = sp.csr_matrix((vals, (rows, cols)), shape=(T, T))
        pin = 0
        keep = np.array([k for k in range(T) if k != pin])
        log_val = np.zeros(T)
        if len(keep) > 0:
            log_val[keep] = spla.spsolve(
                L.tolil()[keep][:, keep].tocsc(), b[keep])
        log_val = log_val - np.mean(log_val)
        return tiles, log_val

    _emit(progress, 'amp gauges (G_DC)', 0.60)
    co_pairs = _summarize(_pair_ratios('mean_Q'))
    tiles_co, logCO = _solve(co_pairs)
    idx_co = {t: i for i, t in enumerate(tiles_co)}

    # PI leveling is DERIVED — after normalizing DC by G_DC=CO_rel and H1
    # by |G_H1|=|z_H1|, the leveled pulsatility 2·|H1_norm|/|Q̄_norm| is
    # automatically constant across tiles.  We don't need (and no longer
    # compute) a separate PI_rel gauge.

    tile_f0_med = {t: float(np.median(fs)) for t, fs in tile_f0.items()}
    out = {}
    all_ids = set(idx_co) | set(tile_f0_med)
    for t in sorted(all_ids):
        entry = {'f0_hz': tile_f0_med.get(t), 'source_amp': 'primary'}
        if t in idx_co:
            entry['CO_rel'] = float(np.exp(logCO[idx_co[t]]))
        out[int(t)] = entry
    return out


def fill_missing_amp_gauges(amp_gauges: Dict[int, dict], G,
                            params: NormalizeParams,
                            progress: ProgressCb = None) -> Dict[int, dict]:
    """Multi-hop + neighbor-borrow fill-in for CO_rel and PI_rel — mirrors
    the phase-gauge pipeline so low-SNR tiles (which fail the amp-gauge
    SNR gate) still get amp gauges by inheriting from their neighbors.

    Uses shared-vessel per-tile mean_Q and PI (measured under a looser
    SNR gate `fillin_min_snr_db`) to estimate each missing tile's gauge:

        CO_rel_miss   = CO_rel_ref  * (mean_Q_miss / mean_Q_ref)
        PI_rel_miss   = PI_rel_ref  * (PI_miss     / PI_ref)

    across all shared vessels with weighted geometric mean.  Tiles still
    missing after multi-hop borrow from the strongest gauged neighbor.

    Also fills missing `f0_hz` (median across the tile's own measurements)
    for any tile that was seen but not entered in the amp gauges.

    Mutates and returns `amp_gauges`.
    """
    # Rebuild per-edge PI + mean_Q lists at the fill-in gate
    edge_data = defaultdict(list)
    tile_all_f0 = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        for m in (ed.get('measurements_piv') or []):
            mv = MeasurementView(m)
            tid = mv.tile_id
            f0 = mv.harm.f0_hz
            mean_Q = mv.harm.mean_Q
            snr_dc = mv.snr.dc_db   # gate on DC SNR (not H1) — see sync_amp_gauges docstring
            if any(x is None for x in (f0, mean_Q, snr_dc)):
                continue
            if not (np.isfinite(f0) and np.isfinite(mean_Q)
                    and np.isfinite(snr_dc)):
                continue
            if abs(mean_Q) < params.amp_min_mean_Q:
                continue
            if snr_dc < params.fillin_min_snr_db:
                continue
            edge_data[(u, v)].append(dict(
                tile=int(tid), mean_Q=abs(float(mean_Q)),
                snr=float(snr_dc)))
            tile_all_f0[int(tid)].append(float(f0))

    # Which tiles are missing the CO_rel (G_DC) gauge?
    all_tiles_seen = set()
    for ms in edge_data.values():
        for m in ms:
            all_tiles_seen.add(m['tile'])
    remaining = {t for t in all_tiles_seen
                 if amp_gauges.get(t, {}).get('CO_rel') is None}

    # Multi-hop fill for CO_rel (G_DC)
    for hop in range(1, params.max_hops + 1):
        if not remaining:
            break
        ref_tiles = {t for t, g in amp_gauges.items()
                     if g.get('CO_rel') is not None}
        newly = {}
        for t_miss in sorted(remaining):
            estimates = []
            for edge, ms in edge_data.items():
                me = [m for m in ms if m['tile'] == t_miss]
                if not me:
                    continue
                others = [m for m in ms if m['tile'] in ref_tiles]
                if not others:
                    continue
                for m_me in me:
                    for m_ref in others:
                        if m_ref['mean_Q'] <= 0:
                            continue
                        log_r = np.log(m_me['mean_Q'] / m_ref['mean_Q'])
                        ref_val = amp_gauges[m_ref['tile']]['CO_rel']
                        log_est = np.log(ref_val) + log_r
                        snr_min = min(m_me['snr'], m_ref['snr'])
                        snr_lin = 10 ** (min(snr_min, 30) / 10)
                        w = snr_lin * (0.5 ** (hop - 1))
                        estimates.append((log_est, w))
            if not estimates:
                continue
            log_ests = np.array([e for e, _ in estimates])
            ws = np.array([w for _, w in estimates])
            log_bar = float(np.average(np.clip(log_ests, -3.0, 3.0),
                                        weights=ws))
            newly[t_miss] = float(np.exp(log_bar))
        for t, val in newly.items():
            # Cap to the same physiologically-plausible range as the
            # z-gauge fill-in.  A ~9× tile-to-tile leveling factor
            # (which the raw estimator can produce from noisy shared
            # vessels) blows up the normalized DC display; capping
            # keeps low-SNR fill-ins in a defensible range at the cost
            # of leaving some residual scale bias.
            capped = float(np.clip(val, *params.fillin_zmag_cap))
            entry = amp_gauges.setdefault(int(t), {})
            entry['CO_rel'] = capped
            entry['source_amp'] = 'fillin'
            entry['CO_fillin_hop'] = hop
            if capped != val:
                entry['CO_capped_from'] = float(val)
        remaining -= set(newly.keys())
        if not newly:
            break

    # Nearest-neighbor borrow for what's left
    if remaining:
        gauged = {t for t, g in amp_gauges.items()
                  if g.get('CO_rel') is not None}
        for t_miss in sorted(remaining):
            nbr_scores = defaultdict(float)
            for edge, ms in edge_data.items():
                me = [m for m in ms if m['tile'] == t_miss]
                if not me:
                    continue
                for m_me in me:
                    for m_n in ms:
                        if m_n['tile'] == t_miss: continue
                        if m_n['tile'] not in gauged: continue
                        snr_min = min(m_me['snr'], m_n['snr'])
                        snr_lin = 10 ** (min(snr_min, 30) / 10)
                        nbr_scores[m_n['tile']] += snr_lin
            if not nbr_scores:
                continue
            best = max(nbr_scores, key=nbr_scores.get)
            entry = amp_gauges.setdefault(int(t_miss), {})
            entry['CO_rel'] = amp_gauges[best]['CO_rel']
            entry['source_amp'] = 'borrowed'
            entry['CO_borrowed_from'] = int(best)

    # Fill missing f0_hz from own measurements (no interpolation needed)
    for t in all_tiles_seen:
        entry = amp_gauges.setdefault(int(t), {})
        if entry.get('f0_hz') is None and tile_all_f0.get(t):
            entry['f0_hz'] = float(np.median(tile_all_f0[t]))

    _emit(progress, 'amp fill-in', 0.68)
    return amp_gauges


# =============================================================================
# STEP 5: Apply gauges to graph (side-by-side normalized fields)
# =============================================================================


HARM_CACHE_KEYS = ('_h_amp_DC_norm', '_h_Z_DC_norm',
                   '_h_amp_H1_norm', '_h_phase_H1_norm', '_h_Z_H1_norm',
                   '_h_amp_H2_norm', '_h_phase_H2_norm', '_h_Z_H2_norm',
                   '_h_amp_H3_norm', '_h_phase_H3_norm', '_h_Z_H3_norm')


def _tile_gauge_summary(gauges_h1: Dict[int, dict],
                        gauges_h2: Dict[int, dict],
                        amp_gauges: Dict[int, dict]) -> Dict[int, dict]:
    """Merge the per-harmonic gauges + amp gauges into a single per-tile
    dict, keyed as `tile_gauges` on G.graph."""
    out = {}
    for tid in set(gauges_h1) | set(gauges_h2) | set(amp_gauges):
        g_h1 = gauges_h1.get(tid, {})
        g_h2 = gauges_h2.get(tid, {})
        g_amp = amp_gauges.get(tid, {})
        z1 = (complex(g_h1['z_re'], g_h1['z_im'])
              if 'z_re' in g_h1 else None)
        z2 = (complex(g_h2['z_re'], g_h2['z_im'])
              if 'z_re' in g_h2 else None)
        out[int(tid)] = dict(
            z_re=(z1.real if z1 else None),
            z_im=(z1.imag if z1 else None),
            z_mag=(abs(z1) if z1 else None),
            z_arg_deg=(float(np.degrees(np.angle(z1))) if z1 else None),
            z_H2_re=(z2.real if z2 else None),
            z_H2_im=(z2.imag if z2 else None),
            z_H2_mag=(abs(z2) if z2 else None),
            z_H2_arg_deg=(float(np.degrees(np.angle(z2))) if z2 else None),
            CO_rel=g_amp.get('CO_rel'),
            f0_hz=g_amp.get('f0_hz'),
            source_h1=g_h1.get('source'),
            source_h2=g_h2.get('source'),
            source_amp=g_amp.get('source_amp'),
        )
    return out


def _canonical_f0(tile_gauges: Dict[int, dict],
                  ref_tile_id=None,
                  fringe_tile_cutoff: int = 6) -> float:
    """Canonical f0 for the normalization step.

    Priority:
      1. `ref_tile_id`  (if supplied and that tile has a finite f0) —
         used when the graph's boundary conditions are anchored to a
         specific tile (e.g. canonical_2's `direct_arm_tile14`).  Using
         the anchor tile's f0 keeps the boundary waveforms consistent
         between measured, normalized, and simulated frames.
      2. Median f0 across plateau tiles (id > `fringe_tile_cutoff`),
         which excludes the frequency-anomalous fringe (tiles 1..6 in
         Stage-21 anatomy).
      3. Median across all tiles if there is no plateau.
      4. Fallback constant 2.85 Hz.
    """
    if ref_tile_id is not None:
        try:
            g = tile_gauges.get(int(ref_tile_id))
            if g is not None and g.get('f0_hz') is not None \
                    and np.isfinite(g['f0_hz']):
                return float(g['f0_hz'])
        except (TypeError, ValueError):
            pass
    plateau = [g['f0_hz'] for tid, g in tile_gauges.items()
               if tid > fringe_tile_cutoff and g.get('f0_hz') is not None]
    if not plateau:
        plateau = [g['f0_hz'] for g in tile_gauges.values()
                   if g.get('f0_hz') is not None]
    if not plateau:
        return 2.85
    return float(np.median(plateau))


def _infer_reference_tile(G):
    """Try to determine which tile the boundary-node bc_* were drawn
    from.  Priority:
      1. `G.graph['reference_tile_id']` if explicitly set.
      2. Regex-scan every boundary node's `bc_source` string for a
         `tile\\d+` token (e.g. `direct_arm_tile14`).  If all boundary
         tokens agree, use that tile id.  If they disagree, return None
         (mixed-gauge graph — canonical fallback is safer than picking
         one arbitrarily).
    Returns tile id (int) or None.
    """
    explicit = G.graph.get('reference_tile_id')
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    import re
    ids = set()
    for n, d in G.nodes(data=True):
        if d.get('boundary_type') not in ('source', 'sink'):
            continue
        bs = str(d.get('bc_source') or '')
        m = re.search(r'tile\s*(\d+)', bs, re.IGNORECASE)
        if m:
            try:
                ids.add(int(m.group(1)))
            except ValueError:
                pass
    if len(ids) == 1:
        return ids.pop()
    return None


def apply_gauges_to_graph(G,
                          gauges_h1: Dict[int, dict],
                          gauges_h2: Dict[int, dict],
                          amp_gauges: Dict[int, dict],
                          progress: ProgressCb = None,
                          level_ac_amplitude: bool = False,
                          fringe_tile_cutoff: int = 0,
                          radius_consistency_rel_thresh: float = 10.0) -> None:
    """Write normalized measurements SIDE-BY-SIDE with raw.

    Adds:
      - `measurements_piv_normalized` on each edge (list mirroring
        `measurements_piv`, with per-measurement fields normalized)
      - `_h_{amp,phase,Z}_H{k}_norm`, `_h_amp_DC_norm` on each edge
        (cache the viewer picks up when source='normalized')
      - `G.graph['tile_gauges']`
      - `G.graph['f0_canonical_hz']`

    Raw `measurements_piv` and existing `_h_*` cache are unchanged.
    """
    tile_gauges = _tile_gauge_summary(gauges_h1, gauges_h2, amp_gauges)
    G.graph['tile_gauges'] = tile_gauges
    # Prefer the tile the graph's BCs were drawn from (canonical_2
    # anchors everything to tile 14 → normalize should use tile 14's
    # f0 so measured/normalized/sim frames share a common cardiac
    # clock).  Falls back to plateau median if none can be inferred.
    ref_tile = _infer_reference_tile(G)
    F0 = _canonical_f0(tile_gauges, ref_tile_id=ref_tile)
    G.graph['f0_canonical_hz'] = F0
    if ref_tile is not None:
        G.graph['reference_tile_id'] = int(ref_tile)
    DT = 1.0 / 250.0

    _emit(progress, 'apply gauges', 0.85)
    n_touched = 0
    n_meas = 0
    n_dropped = 0
    n_h2_fallback = 0

    for u, v, ed in G.edges(data=True):
        ms = ed.get('measurements_piv') or []
        if not ms:
            # Wipe any stale normalized cache
            ed.pop('measurements_piv_normalized', None)
            for k in HARM_CACHE_KEYS:
                ed.pop(k, None)
            continue
        ms_norm = []
        for m in ms:
            tid = m.get('tile_id')
            if tid is None:
                n_dropped += 1
                continue
            g = tile_gauges.get(int(tid), {})
            z_mag = g.get('z_mag')
            z_arg_deg = g.get('z_arg_deg')
            CO_rel = g.get('CO_rel')
            f0_t = g.get('f0_hz')
            if z_mag is None or CO_rel is None or f0_t is None:
                n_dropped += 1
                continue
            # Per-edge orientation canonicalization: the sync solves
            # gauges assuming per-tile physio-canonical Q sign, but the
            # raw harm.phi is measured in the (flow_from -> flow_to)
            # direction — which may point OPPOSITE to the tile's u,v
            # tuple orientation. Without a per-edge fix, edges whose
            # (u, v) tuple runs backward from flow direction land π away
            # from their neighbours, producing intra-tile phase scatter.
            # Fix: multiply the phasor by orient_sign = -1 (i.e., add π
            # to phi) when flow_from is not the tuple's u node.
            #   Historical caveat (2026-08-13): enabling this on the
            #   PerTileFlow sim breaks _sim_tmp_phase_Q comparisons —
            #   the sim was authored against the uncanonical apply.
            #   plexusflow's sim is being re-authored around this
            #   corrected convention, so we enable it here.
            ff = m.get('flow_from')
            if ff == u:
                orient_sign = 1.0
            elif ff == v:
                orient_sign = -1.0
            else:
                orient_sign = 1.0
            z_H2_mag = g.get('z_H2_mag')
            z_H2_arg_deg = g.get('z_H2_arg_deg')
            if z_H2_mag is None or z_H2_arg_deg is None:
                z_H2_mag, z_H2_arg_deg = z_mag, z_arg_deg
                n_h2_fallback += 1
            # Phase-alignment convention: use H1's gauge for EVERY harmonic
            # (i.e., normalize as a pure time-shift).  This preserves the
            # raw waveform's H2/H1 phase relationship (which encodes the
            # cardiac shape asymmetry — sharp systolic upstroke vs slow
            # diastolic decay).  Applying an independent z_H2 rotation
            # decouples H1 and H2, which time-reverses the asymmetry when
            # z_H2_arg != 2 * z_arg_H1.  H2 magnitude still uses z_H2_mag
            # (per-harmonic amplitude leveling is fine — only phases must
            # be time-shift-consistent).
            per_h_gauge = [(z_mag,    z_arg_deg),                # k=1
                            (z_H2_mag, 2.0 * z_arg_deg),          # k=2: linear-in-k phase
                            (z_mag,    3.0 * z_arg_deg)]          # k=3: same rule

            # Deep-ish copy of the measurement, harm dict, and harmonics list
            m_norm = dict(m)
            harm = dict(m.get('harm') or {})
            m_norm['harm'] = harm
            # AC amplitude normalizer:
            #   `level_ac_amplitude=False` (default) — uniform scaling by
            #     CO_rel (same divisor as DC).  This treats CO_rel as a
            #     per-tile MEASUREMENT-SCALE bias (PIV calibration, contrast,
            #     etc.) that multiplies all flow values.  Under this model,
            #     PI = 2|H1|/|Q̄|, |H2/H1|, and waveform SHAPE are all
            #     invariant under normalization.  |z_H1|/CO_rel becomes a
            #     diagnostic: if far from 1, the tile's AC amplitude
            #     scales differently from its DC — a real biological
            #     signal (regional PI variation).
            #   `True` — per-harmonic amp leveling (÷ |z_H_k|).  Forces
            #     normalized PI ≈ constant across tiles.  Useful for
            #     cross-tile shape-check, but destroys regional PI.
            harms_src = harm.get('harmonics') or []
            harms_new = []
            for h_idx, h in enumerate(harms_src):
                h_new = dict(h)
                if h_idx < len(per_h_gauge):
                    g_mag, g_arg = per_h_gauge[h_idx]
                else:
                    g_mag, g_arg = z_mag, z_arg_deg
                # Choose the amp divisor: uniform CO_rel (preserve PI)
                # or per-harmonic |z_H_k| (level PI).
                amp_div = float(g_mag) if level_ac_amplitude else float(CO_rel)
                amp = h_new.get('amp')
                phi = h_new.get('phi')
                if amp is not None and np.isfinite(amp):
                    h_new['amp'] = float(amp) / amp_div
                if phi is not None and np.isfinite(phi):
                    p = float(phi) - np.radians(g_arg)
                    if orient_sign < 0:
                        p += np.pi
                    h_new['phi'] = float(((p + np.pi) % (2*np.pi)) - np.pi)
                # Keep A, B in sync with the flipped phi (Q̂ = A - iB
                # convention → flipping sign flips both A and B).
                if orient_sign < 0:
                    for _k in ('A', 'B'):
                        _val = h_new.get(_k)
                        if _val is not None and np.isfinite(_val):
                            h_new[_k] = -float(_val)
                if (h_new.get('sigma_amp') is not None
                        and np.isfinite(h_new['sigma_amp'])):
                    h_new['sigma_amp'] = float(h_new['sigma_amp']) / amp_div
                harms_new.append(h_new)
            harm['harmonics'] = harms_new
            harm['f0_hz'] = float(F0)

            mean_Q = harm.get('mean_Q')
            if mean_Q is not None and np.isfinite(mean_Q):
                harm['mean_Q'] = float(mean_Q) / CO_rel

            # Q_t: `level_ac_amplitude=False` uniformly scales the whole
            # signal by 1/CO_rel (Q_norm = Q_raw / CO_rel), preserving
            # any DC:AC ratio inside a tile.  `True` uses non-uniform
            # DC/CO_rel + AC/|z_H1|.
            Q_t_raw = m.get('Q_t')
            if Q_t_raw is not None:
                arr = np.asarray(Q_t_raw, dtype=float)
                if arr.size > 0 and np.isfinite(arr).all():
                    if level_ac_amplitude:
                        DC = float(np.nanmean(arr))
                        AC = arr - DC
                        Q_amp = (DC / CO_rel) + (AC / z_mag)
                    else:
                        Q_amp = arr / CO_rel
                    N = len(Q_amp)
                    t_orig = np.arange(N)
                    M = max(2, int(round(N * f0_t / F0)))
                    t_source = np.arange(M) * (F0 / f0_t)
                    t_source = np.clip(t_source, 0, N - 1)
                    Q_warped = np.interp(t_source, t_orig, Q_amp)
                    # Per-harmonic phase rotation without spectral-leakage
                    # artifacts.  A single time-shift (old np.roll) or an
                    # FFT single-bin rotation both fail:
                    #   - np.roll shifts every harmonic k by -k*z_arg_H1,
                    #     leaving H2/H3 inconsistent with the per-harmonic
                    #     phasor updates below.
                    #   - FFT bin rotation only touches one bin per k,
                    #     which rings in time as a discontinuity.
                    # Fit the (rotated) harmonic reconstruction and add
                    # back the residual — this keeps noise and content
                    # beyond H3 intact, while H1/H2/H3 in Q_t exactly
                    # match the stored (amp, phi_new) after rotation.
                    per_h_gauge_rad = [
                        np.radians(z_arg_deg),                # k=1 uses H1 gauge
                        np.radians(z_H2_arg_deg),             # k=2 uses H2 gauge
                        np.radians(z_arg_deg),                # k=3 falls back to H1
                    ]
                    T_arr = np.arange(len(Q_warped)) * DT
                    # Reuse the ORIGINAL fitted coefficients (already on
                    # the measurement) — time-warping preserves (amp, phi)
                    # exactly (pure horizontal rescaling), so no re-fit
                    # needed.  DC is `harm.mean_Q`; H_k use the values in
                    # `harm.harmonics`.
                    dc_orig = (m.get('harm') or {}).get('mean_Q')
                    a0 = (float(dc_orig)
                          if dc_orig is not None and np.isfinite(dc_orig)
                          else float(np.mean(Q_warped)))
                    Q_fit_orig = np.full_like(Q_warped, a0)
                    Q_fit_rot = np.full_like(Q_warped, a0)
                    for h in (m.get('harm', {}).get('harmonics') or []):
                        k_h = int(h.get('k', 0))
                        a = h.get('amp'); p = h.get('phi')
                        if (k_h < 1 or a is None or p is None
                            or not np.isfinite(a) or not np.isfinite(p)):
                            continue
                        a = float(a); p = float(p)
                        # Per-k gauge (magnitude and phase).  Falls back
                        # to H1's gauge for k > 2 (same policy as the
                        # phasor update above).
                        if k_h - 1 < len(per_h_gauge):
                            g_m_k, _ = per_h_gauge[k_h - 1]
                        else:
                            g_m_k = z_mag
                        amp_div_k = (float(g_m_k) if level_ac_amplitude
                                      else float(CO_rel))
                        a_scaled = a / amp_div_k
                        phi_g = (per_h_gauge_rad[k_h - 1]
                                  if k_h - 1 < len(per_h_gauge_rad)
                                  else per_h_gauge_rad[0])
                        Q_fit_orig += a_scaled * np.cos(
                            2.0 * np.pi * k_h * F0 * T_arr + p)
                        Q_fit_rot += a_scaled * np.cos(
                            2.0 * np.pi * k_h * F0 * T_arr + p - phi_g)
                    residual = Q_warped - Q_fit_orig
                    m_norm['Q_t'] = (Q_fit_rot + residual).tolist()

            ms_norm.append(m_norm)
            n_meas += 1

        if ms_norm:
            ed['measurements_piv_normalized'] = ms_norm
            n_touched += 1

            # ── R-consistency gate ─────────────────────────────────────────
            # Q ~ R²·V, so a per-tile R inference error inflates that tile's
            # DC and H_k by ~2× relative to the tile's true value.  If that
            # tile then happens to have high H1 SNR, it dominates the SNR-
            # weighted aggregate and biases the canonical waveform.  Compute
            # per-edge R_profile_um median across tiles that pass fringe_tile_cutoff
            # and skip aggregation for tiles whose R deviates by > threshold.
            _r_by_tid = {}
            for _mm in ms_norm:
                _tid = _mm.get('tile_id')
                if _tid is None: continue
                if fringe_tile_cutoff and int(_tid) <= fringe_tile_cutoff:
                    continue
                _r = _mm.get('R_profile_um')
                if _r is not None and np.isfinite(_r):
                    _r_by_tid[int(_tid)] = float(_r)
            _r_keep_tids = None    # None = no R filter (default open)
            if _r_by_tid and radius_consistency_rel_thresh < 10.0:
                _r_med = float(np.median(list(_r_by_tid.values())))
                _r_keep_tids = {
                    tid for tid, rr in _r_by_tid.items()
                    if abs(rr - _r_med) / max(_r_med, 1e-9)
                       <= radius_consistency_rel_thresh}

            # Precompute edge-level `_h_*_norm` cache from an SNR-
            # weighted MEAN across all normalized measurements on this
            # edge (once normalized, gauges are aligned so phasors can
            # be coherently averaged; noisy tiles get down-weighted).
            #
            # Coherent averaging:
            #   H_k_mean = Σ_t w_t H_k(t) / Σ_t w_t
            # with power weight  w_t = 10^(SNR_k(t) / 10)  (variance-
            # inverse if SNR is defined as signal²/σ²).  Composite SNR
            # in dB is  10 log10 Σ w_t  — coherent stacking gives +3 dB
            # per doubled tile count at equal quality.
            #
            # Also written: `_h_norm_mean_phasors` — dict with the same
            # numbers packed as complex phasors, for the popup overlay.
            def _resolve_signs_max_coherence(phasors, weights):
                """Find s ∈ {±1}^N maximizing |Σ w_i · s_i · z_i|.
                This is per-edge Z/2 sign resolution: each measurement's
                harmonic phasor has an inherent ± amplitude-sign
                ambiguity ((A, φ) ↔ (−A, φ+π)) that the per-tile gauge
                may not have caught for this particular edge. We pick
                the sign configuration that maximizes coherence of the
                weighted sum, then invert the whole solution if the
                majority came out negative (so we keep the convention
                consistent across edges).

                Brute-force for N ≤ 8 (2⁸=256 tries, ~edges typically
                have N ≤ 5). Greedy fallback for larger N — rare.
                Returns list of signs; multiply each z_i by s_i to align."""
                n = len(phasors)
                if n < 2:
                    return [1] * n
                if n <= 8:
                    best_s = [1] * n; best_val = -1.0
                    for mask in range(1 << n):
                        s = [1 if (mask >> i) & 1 == 0 else -1 for i in range(n)]
                        val = abs(sum(w * si * z for w, si, z in
                                       zip(weights, s, phasors)))
                        if val > best_val:
                            best_val = val; best_s = s
                    # Majority-invert so most signs are +1 (stable convention)
                    if sum(1 for si in best_s if si < 0) > n // 2:
                        best_s = [-si for si in best_s]
                    return best_s
                # greedy fallback: init all +1, iteratively flip any
                # single sign that improves |Σ|. Stop when stable.
                s = [1] * n
                for _ in range(4 * n):
                    changed = False
                    cur_val = abs(sum(w * si * z for w, si, z in
                                       zip(weights, s, phasors)))
                    for i in range(n):
                        s[i] = -s[i]
                        alt_val = abs(sum(w * sj * z for w, sj, z in
                                           zip(weights, s, phasors)))
                        if alt_val > cur_val:
                            cur_val = alt_val; changed = True
                        else:
                            s[i] = -s[i]
                    if not changed: break
                if sum(1 for si in s if si < 0) > n // 2:
                    s = [-si for si in s]
                return s

            def _stat_mean_H(harm_ix):
                """SNR-weighted complex mean H at harm_ix across ms_norm,
                with per-edge Z/2 sign resolution applied BEFORE the mean
                (see _resolve_signs_max_coherence). Prevents a single
                high-SNR sign-flipped outlier from dragging the mean into
                the wrong quadrant.

                Returns (H_mean, snr_db, sign_map) where sign_map is a
                {tile_id: +1 or -1} dict recording which tiles were flipped
                by the Z/2 resolution. Consumers (e.g. diagnostic plots)
                use sign_map to mark flipped tiles."""
                zs = []; ws = []; tids = []
                for mm in ms_norm:
                    tid_ = mm.get('tile_id')
                    if (fringe_tile_cutoff and tid_ is not None
                            and int(tid_) <= fringe_tile_cutoff):
                        continue
                    if (_r_keep_tids is not None and tid_ is not None
                            and int(tid_) not in _r_keep_tids):
                        continue
                    hs = ((mm.get('harm') or {}).get('harmonics') or [])
                    if harm_ix >= len(hs) or hs[harm_ix] is None:
                        continue
                    h = hs[harm_ix]
                    a = h.get('amp'); p = h.get('phi'); s = h.get('snr_db')
                    if (a is None or p is None or not np.isfinite(a)
                        or not np.isfinite(p)):
                        continue
                    w = 10 ** (float(s) / 10.0) if (s is not None
                                                     and np.isfinite(s)) else 1.0
                    zs.append(complex(float(a) * np.exp(1j * float(p))))
                    ws.append(w)
                    tids.append(mm.get('tile_id'))
                if not zs:
                    return complex('nan'), float('nan'), {}
                signs = _resolve_signs_max_coherence(zs, ws)
                num = sum(w * si * z for w, si, z in zip(ws, signs, zs))
                den = sum(ws)
                if den <= 0:
                    return complex('nan'), float('nan'), {}
                H = num / den
                snr_db = 10.0 * np.log10(den)
                sign_map = {int(t): int(s) for t, s in zip(tids, signs)
                             if t is not None}
                return H, snr_db, sign_map

            def _stat_mean_DC():
                """SNR-weighted mean of DC across ms_norm.  If DC-SNR is
                unavailable, fall back to H1 SNR as a proxy weight."""
                num = 0.0
                den = 0.0
                for mm in ms_norm:
                    tid_ = mm.get('tile_id')
                    if (fringe_tile_cutoff and tid_ is not None
                            and int(tid_) <= fringe_tile_cutoff):
                        continue
                    if (_r_keep_tids is not None and tid_ is not None
                            and int(tid_) not in _r_keep_tids):
                        continue
                    dc_v = (mm.get('harm') or {}).get('mean_Q')
                    if dc_v is None or not np.isfinite(dc_v):
                        continue
                    snr_dc = (mm.get('snr') or {}).get('dc_db')
                    if snr_dc is None or not np.isfinite(snr_dc):
                        hs = ((mm.get('harm') or {}).get('harmonics') or [])
                        snr_dc = (hs[0].get('snr_db')
                                   if hs and hs[0] is not None else None)
                    w = (10 ** (float(snr_dc) / 10.0)
                         if snr_dc is not None and np.isfinite(snr_dc)
                         else 1.0)
                    num += w * float(dc_v)
                    den += w
                if den <= 0:
                    return float('nan'), float('nan')
                return num / den, 10.0 * np.log10(den)

            dc_mean, dc_snr = _stat_mean_DC()
            ed['_h_amp_DC_norm'] = (float(dc_mean) if np.isfinite(dc_mean)
                                    else float('nan'))
            # Preserve the historical convention `Z_DC = |DC| * 10` so
            # downstream readers of _h_Z_DC_norm see the same scale.
            ed['_h_Z_DC_norm'] = (float(abs(dc_mean) * 10)
                                  if np.isfinite(dc_mean) else float('nan'))
            mean_phasors = {'DC': complex(dc_mean) if np.isfinite(dc_mean)
                             else complex('nan')}
            for k in (1, 2, 3):
                H, snr_db, sign_map = _stat_mean_H(k - 1)
                ed[f'_h_amp_H{k}_norm'] = (float(abs(H))
                                           if np.isfinite(H.real) else float('nan'))
                ed[f'_h_phase_H{k}_norm'] = (float(np.angle(H))
                                             if np.isfinite(H.real) else float('nan'))
                ed[f'_h_Z_H{k}_norm'] = (float(10 ** (min(snr_db, 60) / 20.0))
                                          if np.isfinite(snr_db) else float('nan'))
                mean_phasors[f'H{k}'] = H
                mean_phasors[f'H{k}_snr_db'] = snr_db
                # Per-tile Z/2 sign decisions: which tiles got flipped in
                # the aggregate (−1) vs kept as-is (+1). Consumed by the
                # diagnostic figure to mark flipped tiles.
                mean_phasors[f'H{k}_signs'] = sign_map
            ed['_h_norm_mean_phasors'] = mean_phasors
            ed['_h_norm_mean_n'] = len(ms_norm)
        else:
            ed.pop('measurements_piv_normalized', None)
            for k in HARM_CACHE_KEYS:
                ed.pop(k, None)
            ed.pop('_h_norm_mean_phasors', None)
            ed.pop('_h_norm_mean_n', None)

    # ── Unit-CO rescale (2026-07-28) ─────────────────────────────────
    # After per-tile gauges, apply ONE additional global scalar so that
    # the sum of |DC| across arterial (source) boundary arms equals
    # 1 nL/s.  This makes normalized flows directly comparable to the
    # sim default of Q_A_total=1.  Every normalized DC, harmonic amp,
    # sigma, cache field, mean phasor, and Q_t sample is scaled.
    _unit_co_scale = 1.0
    _sum_a_dc = 0.0
    for n, nd in G.nodes(data=True):
        if nd.get('boundary_type') != 'source':
            continue
        for nb in G.neighbors(n):
            ms_n = G.edges[n, nb].get('measurements_piv_normalized') or []
            best = None
            best_snr = -np.inf
            for mm in ms_n:
                hs = ((mm.get('harm') or {}).get('harmonics') or [])
                snr = hs[0].get('snr_db') if hs and hs[0] else None
                if snr is not None and np.isfinite(snr) and snr > best_snr:
                    best_snr = float(snr); best = mm
            if best is None and ms_n:
                best = ms_n[0]
            if best is None:
                continue
            dc = (best.get('harm') or {}).get('mean_Q')
            if dc is not None and np.isfinite(dc):
                _sum_a_dc += abs(float(dc))
    if _sum_a_dc > 1e-12:
        _unit_co_scale = 1.0 / _sum_a_dc
        for _u, _v, _d in G.edges(data=True):
            for _mm in (_d.get('measurements_piv_normalized') or []):
                _harm = _mm.get('harm') or {}
                if 'mean_Q' in _harm and _harm['mean_Q'] is not None \
                        and np.isfinite(_harm['mean_Q']):
                    _harm['mean_Q'] = float(_harm['mean_Q']) * _unit_co_scale
                for _h in (_harm.get('harmonics') or []):
                    for _k in ('amp', 'sigma_amp', 'A', 'B',
                                'sigma_A', 'sigma_B'):
                        _val = _h.get(_k)
                        if _val is not None and np.isfinite(_val):
                            _h[_k] = float(_val) * _unit_co_scale
                _qt = _mm.get('Q_t')
                if _qt is not None:
                    _mm['Q_t'] = (np.asarray(_qt, dtype=float)
                                   * _unit_co_scale).tolist()
            for _cache_key in ('_h_amp_DC_norm',
                                '_h_amp_H1_norm', '_h_amp_H2_norm',
                                '_h_amp_H3_norm'):
                if _cache_key in _d and _d[_cache_key] is not None \
                        and np.isfinite(_d[_cache_key]):
                    _d[_cache_key] = float(_d[_cache_key]) * _unit_co_scale
            # Z_DC_norm follows |DC|·10 convention — rescale linearly.
            if '_h_Z_DC_norm' in _d and _d['_h_Z_DC_norm'] is not None \
                    and np.isfinite(_d['_h_Z_DC_norm']):
                _d['_h_Z_DC_norm'] = float(_d['_h_Z_DC_norm']) * _unit_co_scale
            # Mean-phasor cache — the amplitude of each phasor scales.
            _mp = _d.get('_h_norm_mean_phasors')
            if _mp:
                for _k in ('DC', 'H1', 'H2', 'H3'):
                    _val = _mp.get(_k)
                    if _val is not None and np.isfinite(complex(_val).real):
                        _mp[_k] = complex(_val) * _unit_co_scale
    # Extend the unit-CO rescale to the boundary nodes' bc_harmonics /
    # bc_Q_t so the sim's boundary drive stays on the same scale as the
    # normalized interior measurements.  Preserves originals under
    # `bc_harmonics_pre_unitCO` / `bc_Q_t_pre_unitCO` so the raw-scale
    # values can be recovered if the user needs them.
    if _unit_co_scale != 1.0:
        for _bn, _nd in G.nodes(data=True):
            if _nd.get('boundary_type') not in ('source', 'sink'):
                continue
            _bch = _nd.get('bc_harmonics')
            if _bch is not None:
                if 'bc_harmonics_pre_unitCO' not in _nd:
                    _nd['bc_harmonics_pre_unitCO'] = np.asarray(
                        _bch, dtype=complex).copy()
                _nd['bc_harmonics'] = (np.asarray(_bch, dtype=complex)
                                        * _unit_co_scale)
            _qt = _nd.get('bc_Q_t')
            if _qt is not None:
                if 'bc_Q_t_pre_unitCO' not in _nd:
                    _nd['bc_Q_t_pre_unitCO'] = np.asarray(_qt, dtype=float).copy()
                _nd['bc_Q_t'] = (np.asarray(_qt, dtype=float)
                                  * _unit_co_scale)

    G.graph['normalize_unit_co_scale'] = float(_unit_co_scale)
    G.graph['normalize_sum_a_dc_pre_rescale'] = float(_sum_a_dc)
    _emit(progress, f'unit-CO rescale: ×{_unit_co_scale:.4f}  '
                     f'(pre-rescale ΣA_DC={_sum_a_dc:.3f} nL/s; '
                     f'also applied to bc_harmonics / bc_Q_t)', 0.97)

    # ------------------------------------------------------------------
    # Rotate bc_harmonics into the anchor-tile phase gauge.
    # bc_harmonics is stored in tile-14 raw frame (see
    # `run_sim_carved_outlets._extract_DA_harmonics` and
    # `kcl_reconstruct.compute_bc_harmonics_recursive`).  To make the
    # carved sim's propagated phases live in the SAME frame as the
    # per-edge `_h_phase_H{k}_norm`, each H_k must be rotated by
    # exp(-i·k·z_arg_14_rad) where z_arg_14 is tile 14's gauge in the
    # anchor-tile frame.  For the default tile-14 anchor z_arg_14 ≈ 0
    # (identity), so this is a no-op for the historical pipeline.
    # ------------------------------------------------------------------
    _tg = G.graph.get('tile_gauges', {}) or {}
    _z_arg_14 = _tg.get(14, {}).get('z_arg_deg')
    if _z_arg_14 is not None:
        _dpsi_14 = np.radians(float(_z_arg_14))
        _n_rot = 0
        for _bn, _nd in G.nodes(data=True):
            if _nd.get('boundary_type') not in ('source', 'sink'):
                continue
            _bch = _nd.get('bc_harmonics')
            if _bch is None: continue
            _bch_arr = np.asarray(_bch, dtype=complex)
            for _k in range(1, len(_bch_arr)):
                _bch_arr[_k] = _bch_arr[_k] * np.exp(-1j * _k * _dpsi_14)
            _nd['bc_harmonics'] = _bch_arr
            _n_rot += 1
        _emit(progress,
                f'bc_harmonics phase rotation: Δψ = {_z_arg_14:+.1f}°  '
                f'→ {_n_rot} boundary nodes',
                0.98)

    G.graph['tile_normalize_stats'] = dict(
        n_edges_touched=n_touched,
        n_measurements=n_meas,
        n_dropped=n_dropped,
        n_h2_fallback=n_h2_fallback,
        f0_canonical_hz=float(F0),
        unit_co_scale=float(_unit_co_scale),
        sum_a_dc_pre_rescale=float(_sum_a_dc),
    )
    _emit(progress, 'apply gauges done', 0.98)


# =============================================================================
# DIAGNOSTICS: post-hoc residuals against the stored gauges
# =============================================================================


def compute_gauge_diagnostics(G, params: NormalizeParams) -> Dict:
    """Post-hoc residuals against `G.graph['tile_gauges']`.  Returns a
    dict of arrays suitable for plotting:

    {
      'pair_phase_H1':  [(tile_a, tile_b, resid_deg, weight, R_bar), ...],
      'pair_phase_H2':  same for H2 (empty if no H2 gauges),
      'pair_amp_H1':    [(tile_a, tile_b, log_resid, weight), ...],
      'pair_amp_H2':    same,
      'pair_amp_DC':    [(tile_a, tile_b, log_resid, weight, n_shared), ...],
      'per_tile_phase_H1': {tile_id: (weighted_median_deg, tot_weight)},
      'per_tile_phase_H2': same,
      'per_tile_amp_H1':   {tile_id: (median_abs_log_resid, tot_weight)},
      'per_tile_amp_DC':   same for CO_rel,
      'obs_count_H1':   {tile_id: number of shared-vessel observations above fillin gate},
      'obs_count_H2':   same,
      'obs_count_DC':   same for DC data,
      'gauges':         G.graph['tile_gauges']  (echoed for convenience),
    }

    Reads gauges + params + graph; does NOT run any solves.  Cheap.
    """
    tg = G.graph.get('tile_gauges') or {}
    out = {'gauges': tg}

    def _phase_diag(harmonic_idx: int):
        """Pair residuals and per-tile summary for the phase-gauge sync."""
        edge_meas = _extract_edge_measurements(G, harmonic_idx,
                                               min_snr_db=params.fillin_min_snr_db)
        # Read the appropriate z_gauge for this harmonic
        def _z_of(tid):
            g = tg.get(int(tid))
            if g is None: return None
            if harmonic_idx == 0:
                zr, zi = g.get('z_re'), g.get('z_im')
            else:
                zr, zi = g.get('z_H2_re'), g.get('z_H2_im')
            if zr is None or zi is None: return None
            return complex(zr, zi)

        pair_recs = []
        per_tile_r = defaultdict(list)  # tile → list of (|resid_deg|, weight)
        obs_count = defaultdict(int)
        for edge, ms in edge_meas.items():
            if len(ms) < 2:
                continue
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = ms[i], ms[j]
                    if a['tile'] == b['tile']:
                        continue
                    z_a = _z_of(a['tile'])
                    z_b = _z_of(b['tile'])
                    if z_a is None or z_b is None:
                        continue
                    if abs(b['Q']) < 1e-12:
                        continue
                    obs_ratio = a['Q'] / b['Q']
                    pred_ratio = z_a / z_b
                    r = float(np.angle(obs_ratio / pred_ratio))
                    snr_min = min(a['snr'], b['snr'])
                    snr_lin = 10 ** (min(snr_min, 30) / 10)
                    amp_min = min(a['amp'], b['amp'])
                    w = amp_min ** 2 * snr_lin
                    pair_recs.append((int(a['tile']), int(b['tile']),
                                      float(np.degrees(r)), float(w), 1.0))
                    per_tile_r[a['tile']].append((abs(float(np.degrees(r))), w))
                    per_tile_r[b['tile']].append((abs(float(np.degrees(r))), w))
                    obs_count[a['tile']] += 1
                    obs_count[b['tile']] += 1

        per_tile = {}
        for t, recs in per_tile_r.items():
            rs = np.array([r for r, _ in recs])
            ws = np.array([w for _, w in recs])
            per_tile[int(t)] = (float(np.average(rs, weights=ws)),
                                float(sum(ws)))
        return pair_recs, per_tile, dict(obs_count)

    def _amp_diag(harmonic_idx: int):
        """Pair log-amp residuals for an AC harmonic gauge (|z|)."""
        edge_meas = _extract_edge_measurements(G, harmonic_idx,
                                               min_snr_db=params.fillin_min_snr_db)
        def _mag_of(tid):
            g = tg.get(int(tid))
            if g is None: return None
            if harmonic_idx == 0:
                m = g.get('z_mag')
            else:
                m = g.get('z_H2_mag')
            return float(m) if m and np.isfinite(m) else None

        pair_recs = []
        per_tile_r = defaultdict(list)
        for edge, ms in edge_meas.items():
            if len(ms) < 2:
                continue
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = ms[i], ms[j]
                    if a['tile'] == b['tile']:
                        continue
                    ma, mb = _mag_of(a['tile']), _mag_of(b['tile'])
                    if ma is None or mb is None or ma == 0 or mb == 0:
                        continue
                    if abs(b['Q']) < 1e-12:
                        continue
                    obs = np.log(abs(a['Q']) / abs(b['Q']))
                    pred = np.log(ma / mb)
                    r = float(obs - pred)
                    snr_min = min(a['snr'], b['snr'])
                    snr_lin = 10 ** (min(snr_min, 30) / 10)
                    amp_min = min(a['amp'], b['amp'])
                    w = amp_min ** 2 * snr_lin
                    pair_recs.append((int(a['tile']), int(b['tile']),
                                      float(r), float(w)))
                    per_tile_r[a['tile']].append((abs(r), w))
                    per_tile_r[b['tile']].append((abs(r), w))
        per_tile = {}
        for t, recs in per_tile_r.items():
            rs = np.array([r for r, _ in recs])
            ws = np.array([w for _, w in recs])
            per_tile[int(t)] = (float(np.average(rs, weights=ws)),
                                float(sum(ws)))
        return pair_recs, per_tile

    # DC amp diagnostic — DC is real-valued, so this uses mean_Q ratios
    def _dc_diag():
        edge_data = defaultdict(list)
        for u, v, ed in G.edges(data=True):
            for m in (ed.get('measurements_piv') or []):
                mv = MeasurementView(m)
                snr_dc = mv.snr.dc_db
                mean_Q = mv.harm.mean_Q
                if snr_dc is None or mean_Q is None: continue
                if not (np.isfinite(snr_dc) and np.isfinite(mean_Q)): continue
                if abs(mean_Q) < params.amp_min_mean_Q: continue
                if snr_dc < params.fillin_min_snr_db: continue
                edge_data[(u, v)].append(dict(
                    tile=int(mv.tile_id), mean_Q=abs(float(mean_Q)),
                    snr=float(snr_dc)))
        pair_recs = []
        per_tile_r = defaultdict(list)
        obs_count = defaultdict(int)
        for edge, ms in edge_data.items():
            if len(ms) < 2:
                continue
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = ms[i], ms[j]
                    if a['tile'] == b['tile']: continue
                    co_a = tg.get(a['tile'], {}).get('CO_rel')
                    co_b = tg.get(b['tile'], {}).get('CO_rel')
                    if co_a is None or co_b is None: continue
                    obs = np.log(a['mean_Q'] / b['mean_Q'])
                    pred = np.log(co_a / co_b)
                    r = float(obs - pred)
                    snr_min = min(a['snr'], b['snr'])
                    snr_lin = 10 ** (min(snr_min, 30) / 10)
                    w = snr_lin
                    pair_recs.append((int(a['tile']), int(b['tile']),
                                      float(r), float(w), 1))
                    per_tile_r[a['tile']].append((abs(r), w))
                    per_tile_r[b['tile']].append((abs(r), w))
                    obs_count[a['tile']] += 1
                    obs_count[b['tile']] += 1
        per_tile = {}
        for t, recs in per_tile_r.items():
            rs = np.array([r for r, _ in recs])
            ws = np.array([w for _, w in recs])
            per_tile[int(t)] = (float(np.average(rs, weights=ws)),
                                float(sum(ws)))
        return pair_recs, per_tile, dict(obs_count)

    ph1, pt_ph1, obs_h1 = _phase_diag(0)
    am1, pt_am1 = _amp_diag(0)
    if any(g.get('z_H2_mag') is not None for g in tg.values()):
        ph2, pt_ph2, obs_h2 = _phase_diag(1)
        am2, pt_am2 = _amp_diag(1)
    else:
        ph2, pt_ph2, obs_h2 = [], {}, {}
        am2, pt_am2 = [], {}
    adc, pt_adc, obs_dc = _dc_diag()

    out['pair_phase_H1'] = ph1
    out['pair_phase_H2'] = ph2
    out['pair_amp_H1']   = am1
    out['pair_amp_H2']   = am2
    out['pair_amp_DC']   = adc
    out['per_tile_phase_H1'] = pt_ph1
    out['per_tile_phase_H2'] = pt_ph2
    out['per_tile_amp_H1']   = pt_am1
    out['per_tile_amp_H2']   = pt_am2
    out['per_tile_amp_DC']   = pt_adc
    out['obs_count_H1'] = obs_h1
    out['obs_count_H2'] = obs_h2
    out['obs_count_DC'] = obs_dc
    return out


# =============================================================================
# STEP 7: One-shot convenience
# =============================================================================


def _h2_global_branch_flip(G) -> int:
    """Post-normalize pass: check whether the aggregated H2 phase is on
    the same branch as the tile-14 raw H2 phase at DA-anchor stubs.
    If they systematically disagree by π, rotate every edge's H2 phasor
    by π to bring the aggregation onto the physically-correct branch.

    Returns the number of edges rotated (0 if no flip was applied; the
    total edge count if a global flip was applied).

    Rationale: `_stat_mean_H` runs `_resolve_signs_max_coherence`
    separately per harmonic.  H1's global branch is fixed later by
    `_boundary_anchor_edge_flip`, but H2 has no such anchor.  Empirically
    (2026-09-30 diagnostic on Somites21) the H2 aggregation for outlet
    templates landed π off from the tile-14 raw, giving a bumpy-rise
    Q(t) shape that flipped to correct monotonic-rise on H2 rotation.
    """
    # Gather (aggregated_H2, tile-N_raw_H2) pairs across EVERY edge that
    # has both.  DA-anchor stubs alone are too few and dominated by the
    # anchoring tile — the interior aggregation is where the branch bug
    # actually surfaces.  We use ANY tile's raw H2 as the truth (each tile
    # measured its edges locally; per-tile-normalized values should track
    # the same underlying phase modulo the tile's z_arg gauge, which
    # already sits inside the aggregated value).
    agg_phasors = []
    raw_phasors = []
    for u, v, d in G.edges(data=True):
        amp2 = d.get('_h_amp_H2_norm')
        ph2 = d.get('_h_phase_H2_norm')
        if amp2 is None or ph2 is None: continue
        if not (np.isfinite(amp2) and np.isfinite(ph2)): continue
        if float(amp2) < 1e-6: continue    # zero-amp edges carry no phase info
        agg = complex(float(amp2) * np.exp(1j * float(ph2)))
        # Sum tile-14 (or any high-SNR tile) raw phasor for THIS edge
        for m in (d.get('measurements_piv_normalized') or []):
            harm = m.get('harm') or {}
            h1 = next((h for h in harm.get('harmonics') or []
                        if h.get('k') == 1), None)
            h2 = next((h for h in harm.get('harmonics') or []
                        if h.get('k') == 2), None)
            if h1 is None or h2 is None: continue
            snr1 = h1.get('snr_db')
            if snr1 is None or float(snr1) < 15.0: continue
            a2 = h2.get('amp'); p2 = h2.get('phi')
            if a2 is None or p2 is None: continue
            if not (np.isfinite(a2) and np.isfinite(p2)): continue
            if float(a2) < 1e-6: continue
            raw = complex(float(a2) * np.exp(1j * float(p2)))
            agg_phasors.append(agg)
            raw_phasors.append(raw)
            break    # one representative tile per edge is enough
    if not agg_phasors:
        return 0
    # Coherence: <agg · conj(raw)> — its phase tells us the mean angle
    # difference.  If |agg - raw| angle is closer to 0 → same branch.
    # If closer to π → opposite branch, flip needed.
    coh = complex(0)
    for a, r in zip(agg_phasors, raw_phasors):
        # Normalize each contribution so amplitude bias doesn't dominate
        if abs(a) < 1e-30 or abs(r) < 1e-30: continue
        coh += (a / abs(a)) * np.conj(r / abs(r))
    if abs(coh) < 1e-30:
        return 0
    diff_angle = float(np.angle(coh))
    diff_deg = float(np.degrees(diff_angle))
    print(f'[normalize] H2 branch check: DA-anchor mean diff = '
          f'{diff_deg:+.1f}° across {len(agg_phasors)} stubs '
          f'(|coh|={abs(coh):.2f})')
    # If diff is closer to π than to 0, need to flip
    if abs(diff_angle) <= np.pi / 2:
        return 0    # already on correct branch

    def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
    n_flipped = 0
    for u, v, d in G.edges(data=True):
        ph = d.get('_h_phase_H2_norm')
        if ph is None or not np.isfinite(ph):
            continue
        d['_h_phase_H2_norm'] = float(wrap(float(ph) + np.pi))
        Z = d.get('_h_Z_H2_norm')
        if Z is not None:
            try:
                d['_h_Z_H2_norm'] = -complex(Z)
            except Exception:
                pass
        # Also flip the mean-phasor cache entries so the viewer's popup agrees
        mp = d.get('_h_norm_mean_phasors')
        if isinstance(mp, dict) and 'H2' in mp:
            try:
                mp['H2'] = -complex(mp['H2'])
            except Exception:
                pass
        n_flipped += 1
    return n_flipped


def _boundary_anchor_edge_flip(G, thresh_deg: float = 90.0,
                                flip_h2: bool = True) -> int:
    """Post-normalize pass. Every boundary anchor is in phase at H1
    (by user convention). Every interior edge's normalized H1 phase
    should sit within `thresh_deg` of that anchor — anything more
    than that is a wrong-branch pi-flip artifact from the sync. Edges
    with |wrap(phase - anchor)| > thresh_deg get their H1 (and, if
    flip_h2, H2) phasors negated in place, moving them onto the
    near-anchor branch. This codifies the "phase within one quadrant
    of boundary" convention as a normalize-pipeline step.
    """
    import numpy as np

    src_H1 = []
    for n, d in G.nodes(data=True):
        if d.get('boundary_type') == 'source':
            bch = d.get('bc_harmonics')
            if bch is not None:
                arr = np.asarray(bch, dtype=complex)
                if len(arr) > 1 and abs(arr[1]) > 0:
                    src_H1.append(arr[1] / abs(arr[1]))
    if not src_H1:
        return 0
    anchor_ph = float(np.angle(np.mean(src_H1)))
    thresh_rad = np.radians(thresh_deg)

    def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
    n_flipped = 0
    for u, v, d in G.edges(data=True):
        ph = d.get('_h_phase_H1_norm')
        if ph is None:
            continue
        if abs(wrap(float(ph) - anchor_ph)) > thresh_rad:
            d['_h_phase_H1_norm'] = float(wrap(float(ph) + np.pi))
            Z = d.get('_h_Z_H1_norm')
            if Z is not None:
                try:
                    d['_h_Z_H1_norm'] = -complex(Z)
                except Exception:
                    pass
            if flip_h2:
                ph2 = d.get('_h_phase_H2_norm')
                if ph2 is not None:
                    d['_h_phase_H2_norm'] = float(wrap(float(ph2) + np.pi))
                Z2 = d.get('_h_Z_H2_norm')
                if Z2 is not None:
                    try:
                        d['_h_Z_H2_norm'] = -complex(Z2)
                    except Exception:
                        pass
            n_flipped += 1
    return n_flipped


def normalize_graph_in_place(G, params: Optional[NormalizeParams] = None,
                             progress: ProgressCb = None) -> Dict:
    """Run the full pipeline on `G` and populate normalized fields
    side-by-side with raw.  Returns a summary dict for logging.

    Phase gauges are solved by `tile_sync_u1.spectral_sync` — a
    single observation-level connection-Laplacian eigenvector,
    replacing the old `sync_tile_gauges` (BFS 2-coloring, blind to
    frustrated cycles) + `fill_missing_gauges` + `sign_flip_rescue`
    (single-site Ising, cannot escape cluster minima). See that module
    for the design notes.
    """
    from .tile_sync_u1 import spectral_sync
    import dataclasses

    if params is None:
        params = NormalizeParams()

    # If caller left manual_pi_flip_tiles empty and the graph carries a
    # saved list under G.graph['normalization_params'], inherit it. Makes
    # a saved-and-reloaded graph reproduce its correction on the next
    # Normalize click without needing UI plumbing.
    if False and (not params.manual_pi_flip_tiles
            and G.graph.get('normalization_params', {}).get('manual_pi_flip_tiles')):   # Recipe-B migration: don't auto-inherit persisted manual overrides
        saved = frozenset(
            int(t) for t in
            G.graph['normalization_params']['manual_pi_flip_tiles'])
        params = dataclasses.replace(params, manual_pi_flip_tiles=saved)
        print(f'[normalize] inherited manual_pi_flip_tiles from graph metadata: '
              f'{sorted(saved)}')

    # Use the OLD pipeline's fill-in SNR (typically -6 dB) so all
    # tiles the OLD pipeline could reach via multi-hop fill remain
    # reachable in the observation-level solve. The primary-sync
    # min_snr_db is not applicable here — spectral sync has no
    # n>=3 gate and no separate primary/fill split.
    _emit(progress, 'spectral sync H1', 0.0)
    gauges_h1 = spectral_sync(G, harmonic_idx=0,
                               min_snr_db=params.fillin_min_snr_db,
                               cluster_pi_flip=params.cluster_pi_flip,
                               cluster_pi_flip_ref_tile=params.cluster_pi_flip_ref_tile,
                               boundary_pin=params.boundary_pin,
                               boundary_pin_boost=params.boundary_pin_boost,
                               manual_pi_flip_tiles=params.manual_pi_flip_tiles,
                               progress=progress)
    flipped_h1 = []      # deprecated diagnostic — see tile_sync_diagnostics

    if params.compute_h2:
        _emit(progress, 'spectral sync H2', 0.5)
        gauges_h2 = spectral_sync(G, harmonic_idx=1,
                                   min_snr_db=params.fillin_min_snr_db,
                                   cluster_pi_flip=params.cluster_pi_flip,
                                   cluster_pi_flip_ref_tile=params.cluster_pi_flip_ref_tile,
                                   boundary_pin=params.boundary_pin,
                                   boundary_pin_boost=params.boundary_pin_boost,
                                   manual_pi_flip_tiles=params.manual_pi_flip_tiles,
                                   progress=progress)
        flipped_h2 = []
    else:
        gauges_h2 = {}
        flipped_h2 = []

    _emit(progress, 'amp gauges (PI/CO)', 0.6)
    amp_gauges = sync_amp_gauges(G, params=params, progress=progress)
    fill_missing_amp_gauges(amp_gauges, G, params=params, progress=progress)

    apply_gauges_to_graph(G, gauges_h1, gauges_h2, amp_gauges,
                          progress=progress,
                          level_ac_amplitude=params.level_ac_amplitude,
                          fringe_tile_cutoff=params.fringe_tile_cutoff,
                          radius_consistency_rel_thresh=(
                              params.radius_consistency_rel_thresh))

    if params.boundary_anchor_edge_flip:
        # flip_h2=False is a physical correction (2026-09-29): an edge
        # H1-flip corresponds to Q → −Q for that edge, and Q → −Q rotates
        # H_k phase by k·π.  So H1 gets +π but H2 gets 2π ≡ 0 — H2 phase
        # must NOT be flipped.  The prior default (flip_h2=params.compute_h2)
        # was adding a spurious π to H2 on every H1-flipped edge, causing
        # a systematic ~π offset in aggregated H2 phases for return-side
        # (comp-0) plexus outlets whose H1 sits far from the source-anchor
        # mean.  See scratchpad/NORMALIZATION_TRACE.md §4 for the full
        # derivation.
        n_edge_flipped = _boundary_anchor_edge_flip(
            G, thresh_deg=params.boundary_anchor_edge_flip_thresh_deg,
            flip_h2=False)
        print(f'[normalize] boundary-anchor edge flip: '
              f'{n_edge_flipped} edges rotated pi to match "phase lags '
              f'from boundary" convention')

    if params.h2_global_branch_check:
        # H2 has an independent Z/2 sign ambiguity from H1 (both come from
        # a separate per-harmonic `_resolve_signs_max_coherence` call).
        # The H1 branch is anchored by boundary_anchor_edge_flip above,
        # but H2 has no anchor — the aggregation can land the entire H2
        # graph on the wrong branch (systematic π offset on every edge).
        # Empirically observed 2026-09-30: per-edge Q(t) reconstructions
        # from `_h_*_norm` had bumpy-rise + monotonic-fall, which flipped
        # to the correct monotonic-rise + bumpy-fall (matching DA source
        # tile-14 raw) after adding π to every `_h_phase_H2_norm`.
        #
        # Fix: compare aggregated H2 phase at DA-anchor stubs to the
        # tile-14 raw H2 phase.  If they disagree by ~π on average, apply
        # a GLOBAL π-flip to every edge's H2 phasor.
        n_h2_flipped = _h2_global_branch_flip(G)
        if n_h2_flipped:
            print(f'[normalize] H2 global branch flip: {n_h2_flipped} edges '
                  f'rotated pi (aggregated H2 was on wrong branch vs tile-14 '
                  f'raw at DA anchors)')

    G.graph['normalization_params'] = dict(
        min_snr_db_h1=params.min_snr_db_h1,
        min_snr_db_h2=params.min_snr_db_h2,
        fillin_min_snr_db=params.fillin_min_snr_db,
        fillin_min_rbar=params.fillin_min_rbar,
        max_hops=params.max_hops,
        run_sign_flip_rescue=params.run_sign_flip_rescue,
        compute_h2=params.compute_h2,
        amp_min_snr_db=params.amp_min_snr_db,
        level_ac_amplitude=params.level_ac_amplitude,
        manual_pi_flip_tiles=sorted(int(t) for t in params.manual_pi_flip_tiles),
        boundary_anchor_edge_flip=params.boundary_anchor_edge_flip,
        boundary_anchor_edge_flip_thresh_deg=params.boundary_anchor_edge_flip_thresh_deg,
    )

    _emit(progress, 'done', 1.0)
    stats = dict(G.graph.get('tile_normalize_stats', {}))
    stats.update(
        n_tiles_h1=len(gauges_h1),
        n_tiles_h1_primary=sum(1 for g in gauges_h1.values()
                               if g.get('source') == 'primary'),
        n_tiles_h1_fillin=sum(1 for g in gauges_h1.values()
                              if g.get('source') == 'fillin'),
        n_tiles_h1_borrowed=sum(1 for g in gauges_h1.values()
                                if g.get('source') == 'borrowed'),
        n_tiles_h2=len(gauges_h2),
        n_flipped_h1=len(flipped_h1),
        n_flipped_h2=len(flipped_h2),
        flipped_tiles_h1=list(flipped_h1),
        flipped_tiles_h2=list(flipped_h2),
        n_tiles_amp=len(amp_gauges),
        n_tiles_amp_primary=sum(1 for g in amp_gauges.values()
                                if g.get('source_amp') == 'primary'),
        n_tiles_amp_fillin=sum(1 for g in amp_gauges.values()
                               if g.get('source_amp') == 'fillin'),
        n_tiles_amp_borrowed=sum(1 for g in amp_gauges.values()
                                 if g.get('source_amp') == 'borrowed'),
    )
    return stats
