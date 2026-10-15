"""Boundary forcing for the (legacy) carved transmission-line simulation: outlet phase references and DA anchor
harmonics.  Both read the legacy normalized fields (``_h_*_norm_v2``, ``tile_gauges``).

The carve itself (``find_porous_and_outlets``, ``build_carved_graph``) lives in ``plexusflow.reconcile.carve``.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import networkx as nx

from plexusflow.reconcile.carve import find_porous_and_outlets, build_carved_graph  # noqa: F401  (re-export)

N_HARM = 3
D_AREAL = 3.5e-4   # 1/Pa areal distensibility, Somites21 median


def _best_h1_snr(d: dict) -> float:
    """Best H1 SNR (dB) across the per-tile measurements on this edge."""
    best = -np.inf
    for m in (d.get("measurements_piv") or []):
        for h in (m.get("harm", {}).get("harmonics") or []):
            if h.get("k") == 1:
                s = h.get("snr_db", -np.inf)
                if s and np.isfinite(s) and s > best:
                    best = float(s)
    return best


def phi_shift_per_outlet(G: nx.Graph,
                          outlets: List[int],
                          min_snr_db_for_phi: float = 15.0) -> Dict[int, float]:
    """Coherent H1 phase reference at each outlet.

    Reads `_h_amp_H1_norm_v2` / `_h_phase_H1_norm_v2` on each outlet's
    TRUSTED (untagged, high-SNR) incident edges and SNR-weights them.
    Returns dict outlet → phase (rad).
    """
    out: Dict[int, float] = {}
    for o in outlets:
        num = 0.0 + 0.0j; den = 0.0
        for m in G.neighbors(o):
            d = G.edges[o, m]
            if d.get('radial_uncertain'):
                continue
            snr = _best_h1_snr(d)
            if snr < min_snr_db_for_phi:
                continue
            amp = d.get('_h_amp_H1_norm_v2')
            phi = d.get('_h_phase_H1_norm_v2')
            if amp is None or phi is None:
                continue
            w = 10 ** (snr / 10.0)
            num += w * amp * np.exp(1j * float(phi))
            den += w
        out[o] = float(np.angle(num / den)) if den > 0 else 0.0
    return out


def extract_DA_harmonics(G: nx.Graph,
                          A_anchors,
                          K: int = N_HARM) -> np.ndarray:
    """Symmetrized DA phasors [H_DC, H_1, ..., H_K] in canonical CO nL/s,
    using tile-14 raw Q_t on the A anchor stubs.

    Must be called BEFORE the porous carve so A anchors are still in G.
    Returns np.ndarray of complex[K+1].
    """
    CO_ref = G.graph['tile_gauges'][14]['CO_rel']
    H = np.zeros(K + 1, dtype=complex); den = 0.0
    for A in A_anchors:
        for nb in G.neighbors(A):
            d = G.edges[A, nb]
            for m in (d.get('measurements_piv') or []):
                if int(m.get('tile_id', -1)) != 14:
                    continue
                harm = m.get('harm') or {}
                mQ = harm.get('mean_Q')
                if mQ is None:
                    continue
                snr = None
                for h in harm.get('harmonics') or []:
                    if h.get('k') == 1:
                        snr = h.get('snr_db')
                        break
                w = 10 ** (float(snr) / 10.0) if snr is not None else 1.0
                H[0] += w * complex(mQ / CO_ref)
                for h in harm.get('harmonics') or []:
                    k_ = h.get('k')
                    if k_ is None or k_ > K:
                        continue
                    H[k_] += w * ((float(h['amp']) / CO_ref)
                                   * np.exp(1j * float(h['phi'])))
                den += w
    return H / den if den > 0 else H
