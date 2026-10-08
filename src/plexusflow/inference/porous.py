"""Porous-region identification + carved-graph construction.

Ported from ``plexusflow/scripts/run_sim_carved_outlets.py`` on 2026-09-30
as part of the migration into ``plexusflow.inference``.

Terminology (see ``carve_algorithm`` memory for the full recipe):
  * ``radial_uncertain`` edge tag → the porous region.
  * ``U`` = every node touching any tagged edge.
  * outlets = subset of ``U`` with at least one non-tagged edge going to
    a NON-porous, non-anchor node. These are where blood enters/leaves
    the reduced (carved) network.
  * The carve builds a reduced graph ``Gc`` = ``G`` minus tagged edges
    minus A-anchors (and their edges), with outlets promoted to
    ``boundary_type='source'`` and V-anchors kept as ``'sink'``.

Public API:
    find_porous_and_outlets(G) → (U, outlets, outlet_comp, A_a, V_a, comps)
    build_carved_graph(G, U, outlets, A_a) → Gc
    extract_DA_harmonics(G, A_anchors, K=3) → np.ndarray of complex[K+1]
    phi_shift_per_outlet(G, outlets, min_snr_db=15) → dict outlet → phase (rad)
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
import networkx as nx

# Default number of AC harmonics used throughout the carved pipeline.
# Kept at module level so callers get the same value that runs behind
# the CLI.
N_HARM = 3

# Physics constants (matches simulation_bc_recipe_LOCKED and viewer)
D_AREAL = 3.5e-4   # 1/Pa areal distensibility, Somites21 median


def find_porous_and_outlets(G: nx.Graph):
    """Identify porous nodes, outlets, and their connected-component
    membership from the ``radial_uncertain`` edge tags on G.

    Returns (U, outlets, outlet_comp, A_anchors, V_anchors, components).
      * U : set[int] — every node touching any tagged edge.
      * outlets : list[int] — porous nodes with at least one un-tagged
        neighbor that isn't itself porous / a boundary anchor.
      * outlet_comp : np.ndarray[int] — per-outlet component id.
      * A_anchors, V_anchors : set[int] — nodes tagged
        ``boundary_type='source'`` / ``'sink'``.
      * components : list[set[int]] — connected components of the
        porous subgraph, sorted largest first.
    """
    tagged = [(u, v) for u, v, d in G.edges(data=True)
              if d.get('radial_uncertain')]
    U: Set[int] = set()
    for u, v in tagged:
        U.add(u); U.add(v)
    A_anchors = {n for n, d in G.nodes(data=True)
                 if d.get('boundary_type') == 'source'}
    V_anchors = {n for n, d in G.nodes(data=True)
                 if d.get('boundary_type') == 'sink'}

    outlets: List[int] = []
    for n in U:
        for m in G.neighbors(n):
            if m in U or m in A_anchors or m in V_anchors:
                continue
            if G.edges[n, m].get('radial_uncertain'):
                continue
            outlets.append(n)
            break
    outlets = sorted(set(outlets))

    H = nx.Graph(); H.add_edges_from(tagged)
    comps = sorted(nx.connected_components(H), key=lambda c: -len(c))
    node_to_comp: Dict[int, int] = {}
    for ci, c in enumerate(comps):
        for n in c:
            node_to_comp[n] = ci
    outlet_comp = np.array([node_to_comp.get(o, -1) for o in outlets])
    return U, outlets, outlet_comp, A_anchors, V_anchors, comps


def build_carved_graph(G: nx.Graph,
                        U: Set[int],
                        outlets: List[int],
                        A_a) -> nx.Graph:
    """Return a new graph with:
      * tagged (uncertain) edges removed
      * A anchors and their incident edges removed
      * stranded porous nodes (degree 0 after removals) removed
      * outlets promoted to ``boundary_type='source'``
      * V anchors preserved as ``boundary_type='sink'``.

    `A_a` may be a set or a sorted list — either works.
    """
    Gc = G.copy()
    # Drop tagged edges
    for u, v, d in list(Gc.edges(data=True)):
        if d.get('radial_uncertain'):
            Gc.remove_edge(u, v)
    # Drop A anchors
    for a in A_a:
        if Gc.has_node(a):
            Gc.remove_node(a)
    # Drop stranded porous-region nodes
    for n in list(U):
        if Gc.has_node(n) and Gc.degree(n) == 0:
            Gc.remove_node(n)
    # Promote outlets to sources (overwrites any prior boundary_type)
    for o in outlets:
        if Gc.has_node(o):
            Gc.nodes[o]['boundary_type'] = 'source'
    return Gc


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
