"""Gross-error detection for the DC reconstruction: which measurements, vessels and places to review by eye.

One rule throughout (data reconciliation, Crowe 1983; Narasimhan & Mah 1987): compare each quantity that has redundancy with
what the rest of the data predicts for it, standardised by its calibrated uncertainty; a large |z| is a suspect, not a verdict.
  measurement   vessel input y_e vs its reconstruction Q'_e on the scale of the multiplicative error model:
                z = ln(|y| / |Q'|) / (c_e √r_e), r_e = 1 − leverage of the projection linearised at the solution (r → 0: not
                checkable by conservation, no test).  Symmetric: flags inflation (Q' ≫ y) and over-reads (y ≫ Q') alike — a
                variance evaluated at Q' would hide inflation.  Opposite signs with both values direction-resolved
                (|·| > 2 σ) are flagged as 'reversal'.
  repeat        a recording vs the 1/c²-weighted mean of the same vessel's OTHER recordings: z = ln(|y_t| / |ȳ|) / √(c_t² + c̄² + τ²),
                τ = between-tile scatter of repeat recordings (acquisition time, physiology; not a measurement error); opposite
                direction-resolved signs → 'repeat_sign'.
  nodal         raw imbalance of the measured inputs at an interior junction whose incident vessels are all measured,
                z = Σ s y / √Σ σ², σ at the input: where conservation fails locally (phantom branch, missing outlet).
  radius        log(median Kloosterman radius / segmentation radius) as a robust z within radius classes (mis-segmentation,
                merged or doubled centrelines), and vessels more than ×2 thicker than every neighbour at both ends.
  structural    (no redundancy) flow on directed loops (cycle cancelling: impossible for a pressure-driven mean flow) and
                unmeasured vessels carrying more than the radius class's 99th percentile of measured flux.
Flags close together in the graph are merged into hotspots, ranked by the reconstructed flow they involve.  Flags are for
review (video, segmentation); they never feed back into the reconstruction automatically — choosing regions where Q' looks
wrong would be circular.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .network import EPS_UNIMAGED
from .reliability import c_of

Z_FLAG = 4.0          # |z| beyond which a test flags (≈ 1 false flag per 16 000 Gaussian tests; the observed tails are heavier)
TAU_TILE = 0.4        # between-tile log-scale scatter of repeat recordings, 21S (docs/dc_method.md, "Scope of the error model")


def redundancy(D, dom):
    """r_e = 1 − leverage of each measured vessel in the projection linearised at Q' (weights 1/σ², unmeasured: ε ridge)."""
    im = np.asarray(D["imaged"], bool); sig = np.asarray(D["sigma"], float)
    w = np.where(im, 1 / sig ** 2, 0.0); w[~im] = EPS_UNIMAGED * np.median(w[im])
    B = dom["B"].tocsr(); inter = np.setdiff1d(np.arange(B.shape[0]), dom["bnd"]); C = B[inter].tocsc()
    lu = splu((C @ sp.diags(1 / w) @ C.T).tocsc()); cache = {}
    def col(a):
        if a not in cache:
            e = np.zeros(len(inter)); e[a] = 1.0; cache[a] = lu.solve(e)
        return cache[a]
    r = np.full(len(w), np.nan)
    for k in np.flatnonzero(im):
        rows = C.indices[C.indptr[k]:C.indptr[k + 1]]; vals = C.data[C.indptr[k]:C.indptr[k + 1]]
        r[k] = 0.0 if len(rows) == 0 else float(sum(v * col(a)[rows] @ vals for a, v in zip(rows, vals)) / w[k])
    return np.clip(r, 0.0, 1.0)


def measurement_test(D, dom):
    """(z, reversal, r): log-ratio z of input vs reconstruction (same sign), direction-resolved sign disagreement, redundancy."""
    y, Qp, c = (np.asarray(D[k], float) for k in ("Qm", "Qp", "c")); im = np.asarray(D["imaged"], bool)
    s0 = float(D.get("sigma0_floor", 0.0)); r = redundancy(D, dom); ok = im & (r > 0.05)
    same = np.sign(y) == np.sign(Qp)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ok & same, np.log((np.abs(y) + s0) / (np.abs(Qp) + s0)) / (c * np.sqrt(r)), np.nan)
    resolved = lambda q: np.abs(q) > 2 * np.sqrt((c * q) ** 2 + s0 ** 2)
    reversal = ok & ~same & resolved(y) & resolved(Qp)
    return z, reversal, r


def repeat_test(M, model, s0, tau=TAU_TILE):
    """{(tile, edge): (z, sign_disagrees)} for every recording of a vessel with ≥ 2 recordings."""
    by = {}
    for (t, e), m in M.items():
        by.setdefault(e, []).append((t, m["Q"], c_of(model, m)))
    out = {}
    for e, L in by.items():
        if len(L) < 2:
            continue
        for i, (t, q, c) in enumerate(L):
            oth = [x for j, x in enumerate(L) if j != i]; w = np.array([1 / x[2] ** 2 for x in oth]); qo = np.array([x[1] for x in oth])
            ybar = float(np.sum(w * qo) / w.sum()); cbar = float(np.sqrt(np.sum(w * np.array([x[2] for x in oth]) ** 2) / w.sum()))
            if np.sign(q) == np.sign(ybar):
                out[(t, e)] = (float(np.log((abs(q) + s0) / (abs(ybar) + s0)) / np.sqrt(c ** 2 + cbar ** 2 + tau ** 2)), False)
            else:                                    # both direction-resolved and opposite: a direction disagreement
                out[(t, e)] = (np.nan, bool(abs(q) > 2 * np.hypot(c * q, s0) and abs(ybar) > 2 * np.hypot(cbar * ybar, s0)))
    return out


def nodal_test(D, dom):
    """{node: z} for interior junctions (degree ≥ 3) whose incident vessels are all measured: imbalance of the inputs."""
    y, im = np.asarray(D["Qm"], float), np.asarray(D["imaged"], bool); c = np.asarray(D["c"], float)
    s0 = float(D.get("sigma0_floor", 0.0)); sb = np.asarray(D.get("s_between", np.zeros(len(y))), float)
    var = (c * y) ** 2 + s0 ** 2 + sb ** 2
    B = dom["B"].tocsr(); nr = dom.get("n_real", B.shape[0]); bnd = set(np.asarray(dom["bnd"]).tolist()); out = {}
    for n in range(nr):
        if n in bnd:
            continue
        cols = B.indices[B.indptr[n]:B.indptr[n + 1]]; s = B.data[B.indptr[n]:B.indptr[n + 1]]
        if len(cols) >= 3 and im[cols].all():
            out[dom["nodes"][n]] = float(np.sum(s * y[cols]) / np.sqrt(np.sum(var[cols])))
    return out


def radius_test(G, D, px_um=1.7):
    """Robust z of log(median R_kl / R_graph) within radius classes; and 'thicker than every neighbour by ×2' (bool)."""
    es = [tuple(e) for e in D["edges"]]; rg = np.array([G.edges[e]["radius"] for e in es], float); lr = np.full(len(es), np.nan)
    for i, e in enumerate(es):
        rk = [m["R_kl_um"] / px_um for m in G.edges[e].get("measurements_piv") or [] if m.get("R_kl_um") and np.isfinite(m["R_kl_um"])]
        if rk:
            lr[i] = np.log(np.median(rk) / rg[i])
    z = np.full(len(es), np.nan); edges_q = np.quantile(rg, [0, .2, .4, .6, .8, 1.0])
    for lo, hi in zip(edges_q[:-1], edges_q[1:]):
        s = (rg >= lo) & (rg <= hi) & np.isfinite(lr)
        if s.sum() > 20:
            med = np.median(lr[s]); mad = 1.4826 * np.median(np.abs(lr[s] - med)); z[s] = (lr[s] - med) / max(mad, 1e-6)
    idx = {e: i for i, e in enumerate(es)}; thick = np.zeros(len(es), bool)
    for i, (u, v) in enumerate(es):
        nb = [rg[idx[f]] for n in (u, v) for f in ((n, w) if (n, w) in idx else (w, n) for w in G.neighbors(n)) if f in idx and f != (u, v)]
        thick[i] = bool(nb) and rg[i] > 2 * max(nb)
    return z, lr, thick


def loop_flow(D):
    """Per-edge flow carried on directed loops (cycle cancelling of the reconstructed field)."""
    import networkx as nx
    es = [tuple(e) for e in D["edges"]]; Q = np.asarray(D["Qp"], float); ix = {e: i for i, e in enumerate(es)}; DG = nx.DiGraph(); fl = {}
    for (u, v), q in zip(es, Q):
        if abs(q) > 1e-9:
            a, b = (u, v) if q > 0 else (v, u); fl[(a, b)] = fl.get((a, b), 0.0) + abs(q); DG.add_edge(a, b)
    pe = np.zeros(len(es))
    while True:
        try:
            cyc = nx.find_cycle(DG)
        except nx.NetworkXNoCycle:
            break
        ce = [(a, b) for a, b, *_ in cyc]; m = min(fl[x] for x in ce)
        for x in ce:
            pe[ix[x] if x in ix else ix[(x[1], x[0])]] += m; fl[x] -= m
            if fl[x] < 1e-12:
                DG.remove_edge(*x)
    return pe


def run_qc(G, D, dom, M, model, loops=True, z_flag=Z_FLAG, loop_min=0.05, big_pct=99.0):
    """All tests → (rows, hotspots, summary).  rows: one dict per flag (kind, edge or node, tile, statistic, values)."""
    es = [tuple(e) for e in D["edges"]]; y, Qp = np.asarray(D["Qm"], float), np.asarray(D["Qp"], float); im = np.asarray(D["imaged"], bool)
    rg = np.array([G.edges[e]["radius"] for e in es], float); rows = []; eset = set(es); eix = {e: i for i, e in enumerate(es)}
    def mid(e):
        p = G.edges[e].get("path")
        return (np.asarray(p, float)[len(p) // 2] if p is not None and len(p) else np.array([np.nan, np.nan]))
    zm, rev, r = measurement_test(D, dom); tl = np.asarray(D["tile"]) if "tile" in D else np.full(len(es), -1)
    for i in np.flatnonzero(np.abs(np.nan_to_num(zm)) > z_flag):
        rows.append(dict(kind="measurement", edge=es[i], tile=int(tl[i]), z=float(zm[i]), y=float(y[i]), Qp=float(Qp[i]), redundancy=float(r[i])))
    for i in np.flatnonzero(rev):
        rows.append(dict(kind="reversal", edge=es[i], tile=int(tl[i]), y=float(y[i]), Qp=float(Qp[i]), redundancy=float(r[i])))
    s0 = float(D.get("sigma0_floor", 0.0))
    for (t, e), (z, sgn) in repeat_test(M, model, s0).items():
        if e in eset and (abs(np.nan_to_num(z)) > z_flag or sgn):
            rows.append(dict(kind="repeat_sign" if sgn else "repeat", edge=e, tile=int(t), z=float(z), y=float(M[(t, e)]["Q"]), Qp=float(Qp[eix[e]])))
    for n, z in nodal_test(D, dom).items():
        if abs(z) > z_flag:
            rows.append(dict(kind="nodal", node=int(n), z=float(z)))
    zr, lr, thick = radius_test(G, D)
    for i in np.flatnonzero((np.abs(np.nan_to_num(zr)) > z_flag) | thick):
        rows.append(dict(kind="radius", edge=es[i], z=float(zr[i]) if np.isfinite(zr[i]) else np.nan, R_graph_px=float(rg[i]),
                         R_kl_over_graph=float(np.exp(lr[i])) if np.isfinite(lr[i]) else np.nan, thicker_than_neighbours=bool(thick[i])))
    thr = np.full(len(es), np.inf); q = np.quantile(rg, [0, .2, .4, .6, .8, 1.0])
    for lo, hi in zip(q[:-1], q[1:]):
        s = (rg >= lo) & (rg <= hi) & im & (y != 0)
        if s.sum() > 20:
            thr[(rg >= lo) & (rg <= hi)] = np.percentile(np.abs(y[s]), big_pct)
    for i in np.flatnonzero(~im & (np.abs(Qp) > thr)):
        rows.append(dict(kind="unmeasured_large", edge=es[i], Qp=float(Qp[i]), threshold=float(thr[i]), R_graph_px=float(rg[i])))
    pe = loop_flow(D) if loops else np.zeros(len(es))
    for i in np.flatnonzero(pe > loop_min):
        rows.append(dict(kind="loop", edge=es[i], loop_flow=float(pe[i]), Qp=float(Qp[i])))
    for rw in rows:
        e = rw.get("edge"); p = mid(e) if e is not None else np.array([G.nodes[rw["node"]].get("y", np.nan), G.nodes[rw["node"]].get("x", np.nan)])
        rw["path_mid"] = [float(p[0]), float(p[1])]
    # hotspots: flagged edges (and the edges of flagged junctions), merged when they touch or are one junction step apart
    import networkx as nx
    flag_e = {rw["edge"] for rw in rows if rw.get("edge") is not None} | {e for rw in rows if rw["kind"] == "nodal" for e in G.edges(rw["node"]) if e in eset or e[::-1] in eset}
    H = nx.Graph(); H.add_edges_from(flag_e)
    comps = list(nx.connected_components(H)); node_comp = {n: k for k, c in enumerate(comps) for n in c}
    # merge components whose nodes are ≤ 1 step apart in the full graph
    parent = list(range(len(comps)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for n, k in node_comp.items():
        for w in G.neighbors(n):
            k2 = node_comp.get(w)
            if k2 is not None and find(k2) != find(k):
                parent[find(k2)] = find(k)
    hid = {n: find(k) for n, k in node_comp.items()}; ix = {e: i for i, e in enumerate(es)}; hs = {}
    for rw in rows:
        n = rw["edge"][0] if rw.get("edge") is not None else rw["node"]; h = hid.get(n, -1); rw["hotspot"] = h
        d = hs.setdefault(h, dict(kinds={}, edges=set(), tiles=set()))
        d["kinds"][rw["kind"]] = d["kinds"].get(rw["kind"], 0) + 1
        if rw.get("edge") is not None:
            d["edges"].add(rw["edge"])
        if rw.get("tile", -1) >= 0:
            d["tiles"].add(rw["tile"])
    hot = []
    for h, d in hs.items():
        flow = float(sum(abs(Qp[ix[e]]) for e in d["edges"] if e in ix))
        hot.append(dict(hotspot=h, n_flags=sum(d["kinds"].values()), kinds=d["kinds"], flow=flow, tiles=sorted(d["tiles"]), n_edges=len(d["edges"])))
    hot.sort(key=lambda x: -x["flow"]); rank = {x["hotspot"]: i + 1 for i, x in enumerate(hot)}
    for rw in rows:
        rw["hotspot"] = rank.get(rw["hotspot"], 0)
    for x in hot:
        x["hotspot"] = rank[x["hotspot"]]
    summary = dict(n_flags=len(rows), by_kind={k: sum(1 for rw in rows if rw["kind"] == k) for k in sorted({rw["kind"] for rw in rows})},
                   n_hotspots=len(hot), z_flag=z_flag, measurement_z_rms=float(np.sqrt(np.nanmean(zm ** 2))), n_reversal=int(rev.sum()), n_testable=int(np.sum(np.isfinite(zm))),
                   loop_share=float(pe.sum() / max(np.abs(Qp).sum(), 1e-30)) if loops else None)
    return rows, hot, summary
