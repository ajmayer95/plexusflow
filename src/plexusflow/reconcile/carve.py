"""Carve: the porous (radial_uncertain) region and the reduced network that reconciliation runs on.

Ported from ``scripts/run_sim_carved_outlets.py`` (2026-09-30); moved here from ``plexusflow.inference`` in the
2026-10 package split, since the carved domain is part of the measurement reconstruction.

Terminology (see the ``carve_algorithm`` note for the full recipe):
  * ``radial_uncertain`` edge tag → the porous region.
  * ``U`` = every node touching any tagged edge.
  * outlets = subset of ``U`` with at least one non-tagged edge going to a NON-porous, non-anchor node — where
    blood enters/leaves the carved network.
  * The carved graph ``Gc`` = ``G`` minus tagged edges minus A-anchors (and their edges), outlets promoted to
    ``boundary_type='source'``, V-anchors kept as ``'sink'``.
"""
from __future__ import annotations
from typing import Dict, List, Set
import numpy as np
import networkx as nx


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
