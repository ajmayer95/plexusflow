"""Re-vectorise one region of the stage graph from an edited mask and splice it into the existing graph.

The rest of the graph is untouched — node IDs, vessel geometry, measurements and tags outside the region stay exactly as
they are — so only the spliced vessels need measuring again.  Coordinates: graph (x, y_up), image row = H − y.

Algorithm (``splice_region``):
1. **Vectorise a crop** of the (edited) mask around the region polygon, with a margin, using the stage's mask prep
   (close 1-px slits, fill holes ≤ ``fill_holes`` px) and net3 (degree-2 nodes kept, as the mask/vectorise editor does).
   The margin makes the new skeleton continuous across the polygon border.
2. **Old graph**: vessels entirely inside the polygon are removed; vessels crossing the border are cut where they cross —
   the outside parts are kept and end at a *port* (the last outside centreline point).  Old nodes inside are removed.
   A polygon that contains a boundary anchor (DA / V node) is refused.
3. **New graph**: net3 segments with both ends inside the polygon are kept; the inside end of a segment that crosses
   the border is a *new port*.
4. **Joins**: old and new ports are matched one-to-one by distance (≤ ``max(join_px, 2 × vessel radius)``, Hungarian
   assignment); an unmatched old port falls back to the nearest new node within twice that distance.  Each match becomes
   a short straight connector.  Unmatched old ports leave a dead end (listed in the report for review).  Dead-end spurs
   of the new skeleton (including border chains that found no old vessel) are pruned back to the first junction
   (``prune_tips``; the plexus has no real dead ends and the stage graph had them removed when it was edited).
5. **Condense** (plexusflow.analyze.condense): degree-2 chains through ports and new nodes merge into vessel edges whose
   end nodes are old nodes or new junctions.  New vessels get radius = median distance-to-wall along the centreline on the
   prepped mask (the rule used for every stored radius), no measurements, and the region tag (open / venous / topology)
   of the old vessels they overlap (majority, within 2 radii).

Returns a SpliceResult: the new graph (a copy), the old→new edge map and a report.  Apply / record / revert through
plexusflow.edit.patches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import networkx as nx
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FILL_HOLES, CLOSE_SLITS = 20, True          # stage vectorise defaults (plexusflow.yaml vectorize_fill_holes / close_slits)
MIN_FEATURE, PRUNE_ORDER = 100, 3           # stage net3 settings (vectorize_min_feature_size / prune_order)
JOIN_PX = 10.0                              # minimum port-matching distance
MIN_COMPONENT_PX = 10.0                     # new interior pieces shorter than this (total centreline) are dropped
TMP_KEEP = "_splice_keep"


@dataclass
class SpliceResult:
    G: nx.Graph
    emap: dict                               # old edge (u, v) → list of new edge keys (empty list = removed)
    new_edges: list                          # edge keys added
    removed_edges: list                      # old edge keys removed (inside or cut)
    report: dict = field(default_factory=dict)


# ── mask prep / vectorisation ─────────────────────────────────────────────────────────────────────────────────────

def prep_mask(b: np.ndarray, fill_holes: int = FILL_HOLES, close_slits: bool = CLOSE_SLITS) -> np.ndarray:
    """The stage's pre-net3 mask prep on a boolean array (plexusflow.vectorize._prep_mask_for_net3)."""
    b = np.asarray(b, bool)
    if close_slits:
        b = ndi.binary_closing(b, structure=ndi.generate_binary_structure(2, 1)) | b
    if fill_holes > 0:
        lab, n = ndi.label(~b)
        if n:
            sizes = np.bincount(lab.ravel()); small = sizes <= fill_holes; small[0] = False
            border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
            small[border] = False
            b = b | small[lab]
    return b


def net3_vectorizer(crop: np.ndarray, min_feature_size: int = MIN_FEATURE, prune_order: int = PRUNE_ORDER) -> nx.Graph:
    """Run net3 on a boolean crop; returns its raw graph in CROP coordinates (x = col, y_up = crop height − row)."""
    import tempfile, os
    import tifffile
    from net3.pipeline import vectorize
    fd, tmp = tempfile.mkstemp(prefix="splice_crop_", suffix=".tif"); os.close(fd)
    try:
        tifffile.imwrite(tmp, (np.asarray(crop, bool) * 255).astype(np.uint8))
        return vectorize(tmp, min_feature_size=int(min_feature_size), prune_order=int(prune_order),
                         remove_redundant="none", verbose=False)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _xy(G, n):
    return np.array([float(G.nodes[n]["x"]), float(G.nodes[n]["y"])])


def _oriented_path(G, u, v):
    p = np.asarray(G.edges[u, v]["path"], float)
    if np.linalg.norm(p[0] - _xy(G, u)) > np.linalg.norm(p[0] - _xy(G, v)):
        p = p[::-1]
    return p


def _runs(flag):
    """[(start, stop_inclusive)] of consecutive True runs."""
    idx = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    return [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]


def _sample(edt, p, H):
    r = np.clip(np.round(H - p[:, 1]).astype(int), 0, edt.shape[0] - 1)
    c = np.clip(np.round(p[:, 0]).astype(int), 0, edt.shape[1] - 1)
    return edt[r, c]


def _length(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def splice_region(G: nx.Graph, mask: np.ndarray, polygon_xy, *, margin: Optional[float] = None,
                  vectorizer: Optional[Callable] = None, fill_holes: int = FILL_HOLES, close_slits: bool = CLOSE_SLITS,
                  join_px: float = JOIN_PX, prune_tips: bool = True) -> SpliceResult:
    """Re-vectorise the region ``polygon_xy`` ((K, 2) graph x, y_up) of ``mask`` (full-size, >0 / True = vessel) and
    splice it into a copy of ``G``.  ``vectorizer(crop_bool) → raw graph in crop coordinates`` defaults to net3."""
    from ..analyze.condense import condense_graph
    from ..reconcile.regions import region_class, set_region
    vectorizer = vectorizer or net3_vectorizer
    poly = np.asarray(polygon_xy, float)
    if len(poly) < 3:
        raise ValueError("region polygon needs ≥ 3 vertices")
    region = MplPath(poly)
    Hm, Wm = mask.shape[:2]
    inside = lambda P: region.contains_points(np.atleast_2d(P))

    # 1. old graph: classify
    anchors = [n for n, d in G.nodes(data=True) if d.get("boundary_type") and inside(_xy(G, n))[0]]
    if anchors:
        raise ValueError(f"region contains boundary anchor node(s) {anchors}; draw it so the DA / V nodes stay outside")
    rmax = 0.0; removed, trimmed = [], []                      # trimmed: (old edge, [(a_node|None, b_node|None, path)])
    for u, v, d in G.edges(data=True):
        p = _oriented_path(G, u, v); f = inside(p)
        if not f.any():
            continue
        rmax = max(rmax, float(d.get("radius") or 0.0)); removed.append((u, v))
        out_runs = _runs(~f)
        if out_runs:
            pieces = []
            for a, b in out_runs:
                pieces.append((u if a == 0 else None, v if b == len(p) - 1 else None, p[a:b + 1]))
            trimmed.append(((u, v), pieces))
    if not removed:
        raise ValueError("no vessel of the graph crosses the region")
    old_inside_nodes = {n for n in G.nodes if inside(_xy(G, n))[0]}

    # 2. vectorise the crop
    m = margin if margin is not None else max(60.0, 4.0 * rmax)
    x0, y0 = poly.min(0) - m; x1, y1 = poly.max(0) + m
    c0, c1 = int(max(np.floor(x0), 0)), int(min(np.ceil(x1), Wm)); r0, r1 = int(max(np.floor(Hm - y1), 0)), int(min(np.ceil(Hm - y0), Hm))
    crop = prep_mask(np.asarray(mask[r0:r1, c0:c1]) > 0, fill_holes, close_slits)
    R = vectorizer(crop)
    dy = Hm - r1                                               # crop y_up → stage y_up
    for n, d in R.nodes(data=True):
        d["x"] = float(d["x"]) + c0; d["y"] = float(d["y"]) + dy
    full = np.zeros((Hm, Wm), bool); full[r0:r1, c0:c1] = crop
    edt = np.zeros((Hm, Wm)); edt[r0:r1, c0:c1] = ndi.distance_transform_edt(crop)

    # 3. new graph inside the polygon; ports = inside ends of border-crossing segments
    rin = {n for n in R.nodes if inside(_xy(R, n))[0]}
    keep_R = nx.Graph(); keep_R.add_nodes_from(rin)
    keep_R.add_edges_from((u, v) for u, v in R.edges if u in rin and v in rin)
    new_ports = {u if u in rin else v for u, v in R.edges if (u in rin) != (v in rin)}
    for comp in list(nx.connected_components(keep_R)):         # drop tiny interior islands
        L = sum(np.linalg.norm(_xy(R, a) - _xy(R, b)) for a, b in keep_R.subgraph(comp).edges)
        if L < MIN_COMPONENT_PX and not (comp & new_ports):
            keep_R.remove_nodes_from(comp)
    new_ports &= set(keep_R.nodes)

    # 4. local graph: trimmed old pieces + new interior + connectors
    nxt = [max(G.nodes) + 1]
    def fresh():
        nxt[0] += 1; return nxt[0] - 1
    Lg = nx.Graph(); rid = {n: fresh() for n in keep_R.nodes}
    for n in keep_R.nodes:
        Lg.add_node(rid[n], x=float(R.nodes[n]["x"]), y=float(R.nodes[n]["y"]))
    for a, b in keep_R.edges:
        Lg.add_edge(rid[a], rid[b])
    old_ports = []                                             # (port id, xy, radius, old edge)
    for e, pieces in trimmed:
        r = float(G.edges[e].get("radius") or 1.0)
        for a, b, p in pieces:
            if a is None and b is None:                        # floating outside piece: ports at both ends
                pa, pb = fresh(), fresh()
                Lg.add_node(pa, x=p[0][0], y=p[0][1]); Lg.add_node(pb, x=p[-1][0], y=p[-1][1])
                Lg.add_edge(pa, pb, path=p); old_ports += [(pa, p[0], r, e), (pb, p[-1], r, e)]
                continue
            end = a if a is not None else b
            if end in old_inside_nodes:
                continue
            Lg.add_node(end, x=G.nodes[end]["x"], y=G.nodes[end]["y"], **{TMP_KEEP: True})
            pt = fresh(); q = p[-1] if a is not None else p[0]
            Lg.add_node(pt, x=float(q[0]), y=float(q[1])); Lg.add_edge(end, pt, path=p if a is not None else p[::-1])
            old_ports.append((pt, q, r, e))
    # old vessels with both ends outside but no outside piece (whole path inside): their end nodes may lose all edges
    from scipy.optimize import linear_sum_assignment
    npl = sorted(new_ports); matched_old, matched_new, joins = set(), set(), []
    if old_ports and npl:
        P_old = np.array([q for _, q, _, _ in old_ports]); P_new = np.array([_xy(R, n) for n in npl])
        Dm = np.linalg.norm(P_old[:, None] - P_new[None], axis=2)
        tol = np.array([max(join_px, 2.0 * r) for _, _, r, _ in old_ports])[:, None]
        cost = np.where(Dm <= tol, Dm, 1e9); ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 1e9:
                joins.append((old_ports[i][0], rid[npl[j]], float(Dm[i, j]))); matched_old.add(i); matched_new.add(npl[j])
    if len(keep_R):
        kn = sorted(keep_R.nodes); T = cKDTree(np.array([_xy(R, n) for n in kn]))
        for i, (pt, q, r, e) in enumerate(old_ports):
            if i in matched_old:
                continue
            dd, k = T.query(q)
            if dd <= 2 * max(join_px, 2.0 * r):
                joins.append((pt, rid[kn[k]], float(dd))); matched_old.add(i)
    for a, b, dd in joins:
        Lg.add_edge(a, b)
    dead_old = [dict(edge=list(map(int, old_ports[i][3])), x=float(old_ports[i][1][0]), y=float(old_ports[i][1][1]))
                for i in range(len(old_ports)) if i not in matched_old]
    # prune dead ends of the NEW skeleton (net3 spurs, border chains that found no old vessel): the plexus has no real
    # dead ends and the stage graph had them removed when it was edited.  Walk in from every new degree-1 node to the
    # first junction.  Old ports left unmatched are NOT pruned (reported as dead ends).
    newset = set(rid.values()); unmatched_new = [n for n in npl if n not in matched_new]; pruned, tips = 0, []
    if prune_tips:
        stack = [n for n in Lg.nodes if n in newset and Lg.degree(n) <= 1]
        while stack:
            cur = stack.pop()
            if cur not in Lg or Lg.degree(cur) > 1:
                continue
            nb = list(Lg.neighbors(cur)); tips.append(_xy(Lg, cur)); Lg.remove_node(cur); pruned += 1
            if nb and nb[0] in newset and Lg.degree(nb[0]) <= 1:
                stack.append(nb[0])
    # 5. condense the local graph; old nodes are chain ends
    for n, d in Lg.nodes(data=True):
        if d.get(TMP_KEEP):
            d["true_tip"] = True
    Hc, cstats = condense_graph(Lg)
    G2 = G.copy()
    G2.remove_edges_from(removed)
    G2.remove_nodes_from([n for n in old_inside_nodes if n in G2])
    # tags of the removed vessels, for inheritance
    tagpts, tagcls = [], []
    for e in removed:
        c = region_class(G.edges[e])
        p = np.asarray(G.edges[e]["path"], float); tagpts.append(p); tagcls += [c] * len(p)
    TT = cKDTree(np.vstack(tagpts)); tagcls = np.array(tagcls, object)
    edge_owner = np.concatenate([[i] * len(np.asarray(G.edges[e]["path"])) for i, e in enumerate(removed)])
    new_edges, emap = [], {e: [] for e in removed}
    for a, b, d in Hc.edges(data=True):
        for n in (a, b):
            if n not in G2:
                q = np.array([Hc.nodes[n]["x"], Hc.nodes[n]["y"]]) if n in Hc else _xy(Lg, n)
                G2.add_node(n, x=float(q[0]), y=float(q[1]), radius=float(_sample(edt, q[None], Hm)[0]))
        p = np.asarray(d["path"], float)
        if np.linalg.norm(p[0] - _xy(G2, a)) > np.linalg.norm(p[0] - _xy(G2, b)):
            p = p[::-1]
        if a == b or G2.has_edge(a, b):
            continue
        rad = float(np.median(_sample(edt, p, Hm))) if len(p) else 1.0
        attrs = dict(path=p, radius=max(rad, 0.5), length=_length(p), n_segments=int(d.get("n_segments", 1)), seg_chain=[],
                     measurements_piv=[], _meas_dissipation=float("nan"), harmonic_class=None, spliced=True)
        dist, k = TT.query(p, distance_upper_bound=2.0 * max(rad, 1.0))
        ok = np.isfinite(dist)
        if ok.any():
            cls, cnt = np.unique([str(c) for c in tagcls[k[ok]]], return_counts=True); best = cls[np.argmax(cnt)]
            if best != "None":
                set_region(attrs, best)
            for i in np.unique(edge_owner[k[ok]]):
                emap[removed[int(i)]].append((a, b))
        G2.add_edge(a, b, **attrs); new_edges.append((a, b))
    iso = [n for n in G2.nodes if G2.degree(n) == 0 and n not in G]      # stray new nodes
    G2.remove_nodes_from(iso)
    orphan = [n for n in G2.nodes if G2.degree(n) == 0 and n in G and G.degree(n) > 0]
    n_meas = sum(1 for e in removed if G.edges[e].get("measurements_piv"))
    report = dict(polygon=poly.tolist(), crop_rc=[r0, c0, r1, c1], margin=float(m),
                  old_removed=len(removed), old_cut=len(trimmed), old_nodes_removed=len(old_inside_nodes),
                  new_edges=len(new_edges), new_nodes=sum(1 for n in G2.nodes if n not in G),
                  ports_old=len(old_ports), ports_new=len(npl), joins=len(joins),
                  dead_ends=dead_old, new_border_chains_unmatched=len(unmatched_new), pruned_tip_nodes=pruned,
                  orphaned_old_nodes=[int(n) for n in orphan],
                  measured_vessels_replaced=n_meas,
                  recordings_dropped=int(sum(len(G.edges[e].get("measurements_piv") or []) for e in removed)),
                  condense=cstats, mask_px_in_crop=int(crop.sum()))
    return SpliceResult(G=G2, emap=emap, new_edges=new_edges, removed_edges=removed, report=report)
