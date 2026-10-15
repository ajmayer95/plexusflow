"""BC convention helpers used by the transmission-line solver.

Under the **legacy** convention (default), boundary types are read from
`G.nodes[n]['boundary_type']` on the anatomical leaves themselves. This
is what plexusflow uses in practice (per `first_junction_bc_design.md`
memory, the first-junction convention was rolled back on 2026-07-02).

Under the **first_junction** convention (available on graphs tagged
`G.graph['bc_convention'] == 'first_junction'`), boundary types are
instead read from `boundary_type_effective` on promoted junction nodes,
and edges marked `scaffold=True` are excluded from the physics solve.
This convention is preserved here for backward compatibility but is not
used by any current plexusflow pipeline.

Ported from `pertile/mosaic/scaffold.py` (~40 lines from 382).
"""
from __future__ import annotations
from typing import List, Optional
import networkx as nx


def is_first_junction_convention(G: nx.Graph) -> bool:
    """Return True iff the graph is tagged as using the first-junction
    BC convention (``G.graph['bc_convention'] == 'first_junction'``)."""
    return G.graph.get("bc_convention") == "first_junction"


def effective_boundary_nodes(G: nx.Graph) -> List[int]:
    """Node ids that carry the actual BC the solver applies.

    Under `first_junction`: promoted junctions with `boundary_type_effective`.
    Under legacy (default): the anatomical leaves themselves.
    """
    if is_first_junction_convention(G):
        return [n for n, d in G.nodes(data=True)
                 if d.get("boundary_type_effective") in {"source", "sink"}]
    return [n for n, d in G.nodes(data=True)
             if d.get("boundary_type") in {"source", "sink"}]


def effective_boundary_type(G: nx.Graph, node) -> Optional[str]:
    """The `boundary_type` the solver should use at `node`.

    Returns None if the node is not a boundary in the active convention.
    """
    d = G.nodes[node]
    if is_first_junction_convention(G):
        return d.get("boundary_type_effective")
    return d.get("boundary_type")


def is_active_edge(edge_data: dict) -> bool:
    """True iff the edge participates in the physics solver's admittance
    assembly.  Edges tagged `scaffold=True` are excluded."""
    return not edge_data.get("scaffold")
