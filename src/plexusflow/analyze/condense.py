"""Contract a raw net3 graph into a clean VESSEL graph.

For every maximal degree-2 chain between two "real" endpoints (junction
of degree >= 3, or degree-1 leaf, or a preserved anchor), collapse the
chain into a single edge carrying the concatenated centerline as
``path`` (mosaic-frame graph pixels), plus ``radius``, ``length``,
``n_segments``, and a provenance ``seg_chain``.

Preserved anchors: any node with ``true_tip=True`` or ``boundary_type``
set survives condensation as an endpoint even if its degree is 2.
This is how the user protects the ~4 anatomical anchors from being
absorbed into a chain.

Parallel edges (same A,B via different chains) — keep the longest,
drop the rest.  Self-loops — drop.

The result is a simple ``nx.Graph`` where every interior node is a
genuine junction, and every edge is a whole "vessel".
:mod:`plexusflow.analyze.mosaic_piv` runs Farneback OF along each
vessel's ``path`` to fill ``measurements_piv``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from ..compat import loads_graph
import pickle

import numpy as np
import networkx as nx

from ..config import StageConfig

VESSEL_SCHEMA = "plexusflow-vessel-1"
KEEP_NODE_ATTRS = ("x", "y", "radius", "boundary_type", "true_tip")


def _pos(G, n) -> np.ndarray:
    return np.array([float(G.nodes[n].get("x", 0.0)),
                     float(G.nodes[n].get("y", 0.0))], dtype=float)


def _edge_polyline(G, u, v) -> np.ndarray:
    """Return a 2-point straight-line polyline for one raw edge if it
    doesn't carry an explicit ``path``/``pts`` attribute.  Otherwise
    return the stored polyline."""
    e = G.edges[u, v]
    for k in ("path", "pts"):
        p = e.get(k)
        if p is not None:
            return np.asarray(p, dtype=float)
    return np.stack([_pos(G, u), _pos(G, v)])


def _chain_polyline(G, chain) -> np.ndarray:
    """Concatenate polylines for a chain of edges [(A,n1), (n1,n2), ...,
    (nk,B)] into one dense (M, 2) mosaic-graph-frame polyline.  Handles
    orientation so successive segments meet head-to-tail."""
    pts_out: list[np.ndarray] = []
    prev_end = _pos(G, chain[0][0])
    for u, v in chain:
        seg = _edge_polyline(G, u, v)
        # Orient so seg[0] is near prev_end
        if np.linalg.norm(seg[0] - prev_end) > np.linalg.norm(seg[-1] - prev_end):
            seg = seg[::-1]
        if pts_out:
            # Drop first sample to avoid duplicating the junction between
            # successive segments.
            seg = seg[1:]
        if len(seg):
            pts_out.append(seg)
            prev_end = seg[-1]
    if not pts_out:
        return np.stack([_pos(G, chain[0][0]), _pos(G, chain[-1][1])])
    return np.vstack(pts_out)


def _maximal_chains(UG: nx.Graph, keep: set):
    """Every maximal degree-2 chain, ordered as edges [(A,n1),...,(nk,B)]
    with deg(A) != 2 and deg(B) != 2 (or A/B in ``keep``)."""
    used: set = set()
    chains: list = []

    def walk(a, b):
        chain = [(a, b)]
        prev, cur = a, b
        while UG.degree(cur) == 2 and cur not in keep:
            nxt = next((n for n in UG.neighbors(cur) if n != prev), None)
            if nxt is None:
                break
            chain.append((cur, nxt))
            if nxt == a:
                break
            prev, cur = cur, nxt
        return chain

    for A in UG.nodes():
        if UG.degree(A) == 2 and A not in keep:
            continue
        for nb in UG.neighbors(A):
            if frozenset((A, nb)) in used:
                continue
            ch = walk(A, nb)
            for (u, v) in ch:
                used.add(frozenset((u, v)))
            chains.append(ch)
    rings = sum(1 for u, v in UG.edges()
                 if frozenset((u, v)) not in used)
    return chains, rings


def _contract_residual_deg2(H: nx.Graph, keep: set) -> int:
    """After dropping self-loops / parallel duplicates a junction can
    end up with degree 2.  Absorb it into a longer vessel (unless it's
    a preserved anchor).  Loops until convergence."""
    merged = 0
    changed = True
    while changed:
        changed = False
        for n in [x for x in H.nodes()
                   if H.degree(x) == 2 and x not in keep]:
            nbrs = list(H.neighbors(n))
            if len(nbrs) != 2:
                continue
            a, b = nbrs
            if a == b or H.has_edge(a, b):
                continue
            da, db = H.edges[a, n], H.edges[n, b]
            pa, pb = da.get("path"), db.get("path")
            if pa is not None and pb is not None:
                pa, pb = np.asarray(pa), np.asarray(pb)
                npos = _pos(H, n)
                if np.linalg.norm(pa[0] - npos) < np.linalg.norm(pa[-1] - npos):
                    pa = pa[::-1]
                if np.linalg.norm(pb[-1] - npos) < np.linalg.norm(pb[0] - npos):
                    pb = pb[::-1]
                path = np.vstack([pa, pb[1:] if len(pb) > 1 else pb])
            else:
                path = None
            L = (float(np.hypot(np.diff(path[:, 0]),
                                 np.diff(path[:, 1])).sum())
                 if path is not None and len(path) > 1
                 else float(da.get("length", 0.0) + db.get("length", 0.0)))
            radii = [r for r in (da.get("radius"), db.get("radius"))
                     if r is not None]
            H.add_edge(
                a, b,
                path=path,
                radius=(float(np.median(radii)) if radii else None),
                length=L,
                n_segments=int(da.get("n_segments", 1)
                                + db.get("n_segments", 1)),
                seg_chain=list(da.get("seg_chain", []))
                            + list(db.get("seg_chain", [])),
                measurements_piv=[])
            H.remove_node(n)
            merged += 1
            changed = True
            break
    return merged


def condense_graph(G: nx.Graph) -> tuple[nx.Graph, dict]:
    """Return (vessel_graph, stats) from a raw net3 graph."""
    UG = G.to_undirected() if G.is_directed() else G

    keep = {n for n, a in G.nodes(data=True)
            if a.get("boundary_type") or a.get("true_tip")}
    chains, rings = _maximal_chains(UG, keep)

    # Build candidate vessels
    cand: list = []
    for ch in chains:
        A, B = ch[0][0], ch[-1][1]
        path = _chain_polyline(G, ch)
        length = (float(np.hypot(np.diff(path[:, 0]),
                                   np.diff(path[:, 1])).sum())
                  if len(path) > 1 else 0.0)
        # Radius: length-weighted mean over interior (non-junction) edges,
        # falling back to endpoint edges if the chain has <= 2 edges.
        radii, weights = [], []
        for i, (u, v) in enumerate(ch):
            r = G.edges[u, v].get("radius")
            if r is None:
                # edges added by hand (editor connect, patch merge) carry no radius: use their end nodes'
                rn = [G.nodes[x].get("radius") for x in (u, v) if G.nodes[x].get("radius") is not None]
                r = float(np.mean(rn)) if rn else None
            if r is None:
                continue
            w = float(G.edges[u, v].get("length",
                        G.edges[u, v].get("weight", 1.0)))
            radii.append(float(r)); weights.append(w)
        if radii:
            w = np.asarray(weights, dtype=float)
            if w.sum() > 0:
                mean_r = float(np.average(radii, weights=w))
            else:
                mean_r = float(np.mean(radii))
        else:
            mean_r = None
        cand.append((A, B, path, mean_r, length, ch))

    H = nx.Graph()
    for n in UG.nodes():
        if UG.degree(n) != 2 or n in keep:
            attrs = {k: G.nodes[n][k] for k in KEEP_NODE_ATTRS
                     if k in G.nodes[n]}
            H.add_node(n, **attrs)

    by_pair: dict = defaultdict(list)
    self_loops = 0
    for c in cand:
        A, B = c[0], c[1]
        if A == B:
            self_loops += 1
            continue
        by_pair[frozenset((A, B))].append(c)

    dropped_parallel = 0
    for pair, group in by_pair.items():
        if len(group) > 1:
            group.sort(key=lambda c: c[4], reverse=True)   # keep longest
            dropped_parallel += len(group) - 1
        A, B, path, rad, length, ch = group[0]
        H.add_edge(A, B,
                    path=path,
                    radius=rad, length=length,
                    n_segments=len(ch),
                    seg_chain=[tuple(e) for e in ch],
                    measurements_piv=[])

    n_reabsorbed = _contract_residual_deg2(H, keep)

    H.graph.update({
        "schema": VESSEL_SCHEMA,
        "source_nodes": UG.number_of_nodes(),
        "source_edges": UG.number_of_edges(),
    })
    for k in ("tile_f0_piv",):
        if k in G.graph:
            H.graph[k] = G.graph[k]

    stats = dict(
        rings=rings,
        self_loops=self_loops,
        dropped_parallel=dropped_parallel,
        reabsorbed=n_reabsorbed,
        seg_edges=UG.number_of_edges(),
    )
    return H, stats


def condense_stage(cfg: StageConfig,
                    src_path: Path | None = None,
                    out_path: Path | None = None,
                    verbose: bool = True) -> Path:
    """Condense the stage's edited graph into ``graph_vessels.gpickle``."""
    src = Path(src_path) if src_path else cfg.graph_path
    if not src.exists():
        raise FileNotFoundError(f'no source graph at {src}. '
                                 f'Run `plexusflow vectorize` first.')
    out = Path(out_path) if out_path else cfg.graph_vessels_path
    out.parent.mkdir(parents=True, exist_ok=True)

    G = loads_graph(src.read_bytes())
    UG = G.to_undirected() if G.is_directed() else G
    if verbose:
        d2 = sum(1 for n in UG if UG.degree(n) == 2)
        print(f'source: {src.name}  '
              f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges '
              f'(deg-2 = {d2})')

    H, st = condense_graph(G)

    if verbose:
        degs = Counter(dict(H.degree()).values())
        nseg = np.array([d["n_segments"] for _, _, d in H.edges(data=True)])
        n_kept_anchors = sum(1 for _, a in H.nodes(data=True)
                              if a.get("true_tip") or a.get("boundary_type"))
        print(f'\nvessel graph: {H.number_of_nodes()} nodes, '
              f'{H.number_of_edges()} vessels')
        print(f'  {st["seg_edges"]} segments → {H.number_of_edges()} vessels '
              f'({st["seg_edges"]/max(H.number_of_edges(),1):.1f}× compression)')
        print(f'  degree dist: {dict(sorted(degs.items()))}')
        print(f'  multi-segment vessels: {(nseg>1).sum()} '
              f'(max {int(nseg.max()) if len(nseg) else 0} segments)')
        print(f'  dropped: {st["self_loops"]} self-loops, '
              f'{st["dropped_parallel"]} parallel duplicates, '
              f'{st["rings"]} deg-2 rings; '
              f'reabsorbed {st["reabsorbed"]} residual deg-2 nodes')
        print(f'  preserved anchors (true_tip/boundary_type): {n_kept_anchors}')

    out.write_bytes(pickle.dumps(H))
    if verbose:
        print(f'\nwrote {out.name}')
    cfg.write_provenance("graph_vessels", extra={
        "source_graph": str(src.name),
        "n_vessels": H.number_of_edges(),
        "n_nodes": H.number_of_nodes(),
        **st,
    })
    return out
