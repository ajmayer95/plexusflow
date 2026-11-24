"""Consolidate cluttered stretches of the vessel graph using the vessel mask.

Vectorisation (net3 triangulates the mask outline) turns every enclosed mask hole into a ring of triangles — a small loop
with several junction nodes — and a wide confluence into several junction nodes a few pixels apart.  Neither is
resolvable vasculature.  This step:

1. **Fills pillar holes.**  A hole is a pillar inside one lumen when its length l is shorter than the thickness t of the
   channel beside it (distance from the hole's rim to the outer wall, with only that hole filled): the side channels are
   too short to develop their own flow profile, so they are not separate vessels.  Holes of area < ``SLIVER`` px are
   filled as well.  Longer holes (l ≥ t) separate real tube-like channels and are kept (21S: vessels running beside holes
   with l/t < 1 have less parabolic profiles, p 0.36 vs 0.46 for l/t ≥ 2).
2. **Finds regions**: graph nodes around each filled hole, plus clusters of junction nodes split out of one junction lumen
   (``junction_clusters``).  Overlapping regions are united only while the union stays compact (span ≤ ``K_SPAN`` ×
   median lumen radius) — no cascades.  Regions touching a boundary anchor or an open region are left alone.
3. **Redraws each region** on the hole-filled mask (minimum-cost paths, cost 1/(distance to wall + 0.5); Gaussian-smoothed
   joins): two vessels leaving → one vessel; k ≥ 3 → one junction node at the widest point of the region, each vessel
   extended to it.  Radius of a redrawn vessel = median distance transform along its new path.
4. **Checks every redraw** and leaves the region unchanged (listed for manual review) if a redrawn path doubles back
   (> 120° turn within 10 px), comes back on itself, leaves the mask, or two redrawn arms run together / cross away from
   the junction (two or more points closer than ARM_GAP × the local radius: they share a lumen or meet again in a loop).

Recordings on the vessels leaving a region are kept with their flow direction relabelled; recordings on internal strands
are dropped; redrawn vessels are marked ``consolidated=True`` (re-measure them: ``plexusflow analyze --skip-analyzed``
after clearing their recordings, or a full re-measure).  Graph coordinates: x = image column, y up (image row = H − y).
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage as ndi

LT_MAX, SLIVER, AREA_CAP = 1.0, 30, 3000      # fill if l < LT_MAX·t or area < SLIVER; holes > AREA_CAP px never filled
K_SPAN = 4.0                                   # region span ≤ K_SPAN × median lumen radius at its nodes
JC_K_LEN, JC_K_LEN_MEASURED, JC_K_SPAN = 1.0, 0.5, 2.5   # junction clusters: edge length / cluster span × lumen radius
HAIRPIN_COS, HAIRPIN_PX = -0.5, 5              # a path 'doubles back' if tangents 2·HAIRPIN_PX apart turn by > 120°
SMOOTH_SIGMA = 3.0
ARM_GAP = 0.5                                  # two redrawn arms closer than ARM_GAP × local radius share a lumen


# ----------------------------------------------------------------------------------------------------------- holes
def pillar_holes(m, lt_max=LT_MAX, sliver=SLIVER, area_cap=AREA_CAP):
    """(label image of enclosed holes, keep[label] = fill it, l/t[label]).  m: boolean vessel mask (True = lumen)."""
    from skimage.measure import regionprops
    holes = ndi.binary_fill_holes(m) & ~m; lab, n = ndi.label(holes)
    keep = np.zeros(n + 1, bool); lt = np.full(n + 1, np.nan)
    for r in regionprops(lab):
        if r.area > area_cap:
            continue
        sl = r.slice; M = 60; r0, c0 = max(sl[0].start - M, 0), max(sl[1].start - M, 0)
        sub = lab[r0:sl[0].stop + M, c0:sl[1].stop + M] == r.label
        ew = ndi.distance_transform_edt(m[r0:sl[0].stop + M, c0:sl[1].stop + M] | sub)
        rim = ndi.binary_dilation(sub, iterations=2) & ~sub
        lt[r.label] = (r.axis_major_length if hasattr(type(r), "axis_major_length") else r.major_axis_length) / max(float(np.median(ew[rim])), 1e-6)
        keep[r.label] = (lt[r.label] < lt_max) or (r.area < sliver)
    return lab, keep, lt


# ------------------------------------------------------------------------------------------------- junction clusters
def junction_clusters(G, sub, protect=(), measured=(), k_len=JC_K_LEN, k_len_measured=JC_K_LEN_MEASURED, k_span=JC_K_SPAN):
    """Clusters of nodes that vectorisation split out of one junction lumen: ({node: representative}, internal edges).

    Candidate edges are shorter than k_len × the lumen radius at BOTH ends (at each end, the largest radius of the OTHER
    vessels meeting there) — k_len_measured for an edge with a flux measurement.  Merged shortest first; a merge is
    accepted only if the cluster's span stays within k_span × the median lumen radius of its members and it does not touch
    another merged cluster, so neither one cluster nor a chain of clusters can cascade along a vessel."""
    inc = {}
    for e in sub:
        for n in e:
            inc.setdefault(n, []).append(e)
    rad = {e: float(G.edges[e]["radius"]) for e in sub}; xy = lambda n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], float)
    lumen_at = lambda n, skip=None: max([rad[f] for f in inc[n] if f != skip] or [0.0])
    nbrs = {n: {f[0] if f[1] == n else f[1] for f in inc[n]} for n in inc}
    protect = set(protect); members = {n: [n] for n in inc}; root = {n: n for n in inc}

    def find(n):
        while root[n] != n:
            root[n] = root[root[n]]; n = root[n]
        return n
    cand = sorted((float(G.edges[e]["length"]), e) for e in sub if e[0] not in protect and e[1] not in protect)
    for L, (u, v) in cand:
        lum = min(lumen_at(u, (u, v)), lumen_at(v, (u, v)))
        if lum <= 0 or L >= (k_len_measured if (u, v) in measured else k_len) * lum:
            continue
        a, b = find(u), find(v)
        if a == b:
            continue
        mem = members[a] + members[b]; ms = set(mem); P = np.array([xy(n) for n in mem])
        if float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1))) > k_span * float(np.median([lumen_at(n) for n in mem])):
            continue
        if any(len(members.get(find(w), ())) > 1 for n in mem for w in nbrs[n] if w not in ms):
            continue
        root[b] = a; members[a] = mem; members.pop(b, None)
    rep = {n: find(n) for n in inc}
    return rep, {e for e in sub if rep[e[0]] == rep[e[1]]}


# ------------------------------------------------------------------------------------------------------ path checks
def _resample(p, step=1.0):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s[-1] < step:
        return p
    t = np.arange(0, s[-1], step); return np.vstack([np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]), p[-1]])


def path_problem(p, on_mask_frac=1.0):
    """None if the path is clean, else a short reason: 'hairpin', 'self-loop', 'off-mask'."""
    q = _resample(np.asarray(p, float))
    if on_mask_frac < 0.95:
        return "off-mask"
    k = HAIRPIN_PX
    if len(q) > 2 * k + 1:
        d = q[k:] - q[:-k]; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        if np.any(np.sum(d[k:] * d[:-k], axis=1) < HAIRPIN_COS):
            return "hairpin"
    if len(q) > 14:
        D = np.linalg.norm(q[:, None] - q[None], axis=-1); i, j = np.triu_indices(len(q), 13)
        if np.any(D[i, j] < 1.5):
            return "self-loop"
    return None


# ----------------------------------------------------------------------------------------------------- consolidate
def consolidate(G, m, protect=(), log=None):
    """Return (H, emap, report, filled_mask).  ``log(str)``: optional progress messages.  G is not modified.  m: boolean vessel mask in image rows/cols.
    emap: old edge → new edge key (absent if the edge was removed).  report: counts, 'regions' (applied, node lists),
    'review' (regions left unchanged, with reason and centroid in graph coordinates)."""
    from skimage.graph import route_through_array
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree
    from ..reconcile.regions import is_open, is_venous, region_class
    log = log or (lambda s: None)
    import time as _time; _t0 = _time.time()
    log("[consolidate] measuring mask holes (each hole filled on its own; the slow step) …")
    Hm, Wm = m.shape; lab, keep, _ = pillar_holes(m)
    log(f"[consolidate] {int(keep.sum())} of {len(keep) - 1} holes to fill ({_time.time() - _t0:.0f} s)")
    mf = m | keep[lab]; edt = ndi.distance_transform_edt(mf); cost = np.where(mf, 1.0 / (edt + 0.5), 1e3)
    rc = lambda Gr, n: (int(np.clip(round(Hm - Gr.nodes[n]["y"]), 0, Hm - 1)), int(np.clip(round(Gr.nodes[n]["x"]), 0, Wm - 1)))
    on_mask = lambda p: float(mf[np.clip(np.round(Hm - p[:, 1]).astype(int), 0, Hm - 1), np.clip(np.round(p[:, 0]).astype(int), 0, Wm - 1)].mean())
    anchors = {n for n, d in G.nodes(data=True) if d.get("boundary_type")}
    opened = {n for u, v, d in G.edges(data=True) if is_open(d) or is_venous(d) for n in (u, v)}
    protect = set(protect) | anchors | opened | {w for a in anchors for w in G.neighbors(a)}
    node_rc = {n: rc(G, n) for n in G}

    # regions: (a) graph nodes around each filled hole
    regions = []; objs = ndi.find_objects(lab)
    edt_f = edt
    pos = np.array(list(node_rc.values())); ids = np.array(list(node_rc.keys())); tree = cKDTree(pos)
    for i in np.flatnonzero(keep):
        sl = objs[i - 1]; h = lab[sl] == i; a = float(h.sum())
        sl2 = tuple(slice(max(s.start - 15, 0), s.stop + 15) for s in sl); hh = lab[sl2] == i
        rloc = float(edt_f[sl2][ndi.binary_dilation(hh, iterations=10)].max())
        pad = int(np.ceil(rloc + np.sqrt(a / np.pi))) + 3
        r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        fp = ndi.binary_dilation(lab[r0:sl[0].stop + pad, c0:sl[1].stop + pad] == i, iterations=pad)
        cand = tree.query_ball_point([(sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2], r=pad + max(h.shape))
        mem = {int(ids[k]) for k in cand if 0 <= pos[k][0] - r0 < fp.shape[0] and 0 <= pos[k][1] - c0 < fp.shape[1]
               and fp[pos[k][0] - r0, pos[k][1] - c0]}
        if len(mem) >= 2:
            regions.append(mem)
    log(f"[consolidate] {len(regions)} hole regions; finding junction clusters ({_time.time() - _t0:.0f} s)")
    # (b) junction clusters
    sub = [e for e in G.edges() if not region_class(G.edges[e]) and e[0] not in anchors and e[1] not in anchors]
    measured = {e for e in sub if any((x.get("quality") or {}).get("profile_ok") for x in G.edges[e].get("measurements_piv") or [])}
    rep, _ = junction_clusters(G, sub, protect, measured); cl = {}
    for n, r in rep.items():
        cl.setdefault(r, set()).add(n)
    regions += [s for s in cl.values() if len(s) > 1]

    def compact(s):
        P = np.array([node_rc[n] for n in s], float)
        return np.linalg.norm(P[:, None] - P[None], axis=2).max() <= K_SPAN * max(float(np.median([edt[node_rc[n]] for n in s])), 1.0)
    accepted, too_large = [], 0
    for s in sorted(regions, key=len):
        if not compact(s):
            too_large += 1; continue
        hit = [k for k, a in enumerate(accepted) if a & s]; u = set(s).union(*[accepted[k] for k in hit])
        if hit and not compact(u):
            too_large += 1; continue
        accepted = [a for k, a in enumerate(accepted) if k not in hit] + [u]
    regions = [s for s in accepted if not (s & protect)]

    H = copy.deepcopy(G); emap = {e: e for e in G.edges()}
    report = dict(holes_filled=int(keep.sum()), holes_total=int(len(keep) - 1), one_vessel=0, junction=0, too_large=too_large,
                  skipped=0, dropped_recordings=0, regions=[], review=[])

    def route(a_rc, b_rc):
        idx, _ = route_through_array(cost, a_rc, b_rc, fully_connected=True, geometric=True)
        idx = np.array(idx, float); return np.column_stack([idx[:, 1], Hm - idx[:, 0]])
    def oriented(k, start):
        p = np.asarray(H.edges[k]["path"], float); s = np.array([H.nodes[start]["x"], H.nodes[start]["y"]])
        return p if np.linalg.norm(p[0] - s) <= np.linalg.norm(p[-1] - s) else p[::-1]
    def trim(p, target_rc, R):
        t = np.array([target_rc[1], Hm - target_rc[0]], float); k = len(p)
        while k > 2 and np.linalg.norm(p[k - 1] - t) < R:
            k -= 1
        return p[:k]
    def end_rc(p):
        return int(np.clip(round(Hm - p[-1, 1]), 0, Hm - 1)), int(np.clip(round(p[-1, 0]), 0, Wm - 1))
    def smooth(q, i0, i1):
        """Gaussian-smooth points i0..i1 of a 1-px-resampled path, blending in and out over 3σ; ends fixed."""
        k = int(3 * SMOOTH_SIGMA); a, b = max(1, i0 - k), min(len(q) - 1, i1 + k + 1)
        if b - a < 3:
            return q
        seg = q[a - 1:b + 1]; sm = np.column_stack([gaussian_filter1d(seg[:, j], SMOOTH_SIGMA, mode="nearest") for j in (0, 1)])
        ix = np.arange(a - 1, b + 1); w = np.clip(np.minimum(ix - (a - 1), b - ix) / max(k, 1), 0, 1)
        out = q.copy(); out[a - 1:b + 1] = w[:, None] * sm + (1 - w[:, None]) * seg
        return out if on_mask(out) >= 0.98 else q
    def finish(d, p):
        d["path"] = p; d["length"] = float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
        rr = np.clip(np.round(Hm - p[:, 1]).astype(int), 0, Hm - 1); cc = np.clip(np.round(p[:, 0]).astype(int), 0, Wm - 1)
        d["radius"] = float(np.median(edt[rr, cc])); d["consolidated"] = True
    def relink(old, new):
        for e0 in [e for e, v in emap.items() if v is not None and set(v) == set(old)]:
            emap[e0] = new
    def review(reg, why):
        P = np.array([[H.nodes[n]["x"], H.nodes[n]["y"]] for n in reg if H.has_node(n)])
        report["review"].append(dict(nodes=sorted(int(n) for n in reg), reason=why, x=float(P[:, 0].mean()), y=float(P[:, 1].mean())))

    log(f"[consolidate] redrawing {len(regions)} regions ({_time.time() - _t0:.0f} s)")
    for _k, reg in enumerate(regions):
        if _k and _k % 100 == 0:
            log(f"[consolidate]   {_k}/{len(regions)} regions ({_time.time() - _t0:.0f} s)")
        if not all(H.has_node(n) for n in reg):
            report["skipped"] += 1; continue
        ext = [(n, w) for n in reg for w in H.neighbors(n) if w not in reg]
        internal = [(a, b) for a in reg for b in H.neighbors(a) if b in reg and a < b]
        others = [w for _, w in ext]
        if len(ext) < 2 or len(set(others)) < len(others):
            report["skipped"] += 1; continue
        n_int = sum(1 for a, b in internal for x in H.edges[a, b].get("measurements_piv") or [] if (x.get("quality") or {}).get("profile_ok"))
        key = lambda a, b: (a, b) if H.has_edge(a, b) else (b, a)
        if len(ext) == 2:
            (na, wa), (nb, wb) = ext
            if wa == wb or H.has_edge(wa, wb):
                report["skipped"] += 1; continue
            ka, kb = key(na, wa), key(nb, wb)
            Rt = float(edt[node_rc[na]])
            pa = trim(_resample(oriented(ka, wa)), node_rc[na], Rt); pb = trim(_resample(oriented(kb, wb)), node_rc[nb], Rt)
            mid = route(end_rc(pa), end_rc(pb)); p = np.vstack([pa, mid[1:], pb[::-1][1:]])
            p = smooth(p, len(pa) - 1, len(pa) + len(mid) - 2)
            why = path_problem(p, on_mask(p))
            if why:
                review(reg, why); continue
            d = copy.deepcopy(dict(H.edges[ka])); recs = {}
            for x in (H.edges[ka].get("measurements_piv") or []) + (H.edges[kb].get("measurements_piv") or []):
                x = copy.deepcopy(x)
                for f in ("flow_from", "flow_to"):
                    x[f] = {na: wb, nb: wa}.get(x.get(f), x.get(f))
                t = int(x.get("tile_id", -1))
                if t not in recs or (x.get("quality") or {}).get("p", 0) > (recs[t].get("quality") or {}).get("p", 0):
                    recs[t] = x
            d["measurements_piv"] = list(recs.values()); finish(d, p)
            for k in (ka, kb):
                relink(k, (wa, wb))
            for e in internal:
                relink(e, None)
            H.remove_nodes_from(reg); H.add_edge(wa, wb, **d); report["one_vessel"] += 1
        else:
            pts = np.array([node_rc[n] for n in reg], float); c = pts.mean(0)
            R = max(4.0, float(np.max(np.linalg.norm(pts - c, axis=1))) + 3)
            r0, c0 = int(max(c[0] - R, 0)), int(max(c[1] - R, 0)); win = edt[r0:int(c[0] + R) + 1, c0:int(c[1] + R) + 1]
            jr, jc = np.unravel_index(np.argmax(win), win.shape); j_rc = (jr + r0, jc + c0); rJ = max(3.0, float(edt[j_rc]))
            J = min(reg, key=lambda n: np.hypot(*(np.array(node_rc[n]) - j_rc)))
            new, arms, why = [], [], None
            for n, w in ext:
                k = key(n, w); pk = trim(_resample(oriented(k, w)), j_rc, rJ)
                seg = route(end_rc(pk), j_rc); q = np.vstack([pk, seg[1:]]); q = smooth(q, len(pk) - 1, len(q) - 1)
                why = path_problem(q, on_mask(q))
                if why:
                    break
                d = copy.deepcopy(dict(H.edges[k]))
                for x in d.get("measurements_piv") or []:
                    for f in ("flow_from", "flow_to"):
                        if x.get(f) == n:
                            x[f] = J
                finish(d, q); new.append((k, w, d)); arms.append(q)
            if not why:                                        # arms must not share a lumen or cross away from the junction
                Jxy = np.array([j_rc[1], Hm - j_rc[0]], float); reach = R + 3 * rJ
                near = [a[(np.linalg.norm(a - Jxy, axis=1) > rJ + 3) & (np.linalg.norm(a - Jxy, axis=1) < reach)] for a in arms]
                loc = [edt[np.clip(np.round(Hm - a[:, 1]).astype(int), 0, Hm - 1), np.clip(np.round(a[:, 0]).astype(int), 0, Wm - 1)]
                       for a in near]
                for i in range(len(near)):
                    for j in range(len(near)):
                        if i == j or not len(near[i]) or not len(near[j]):
                            continue
                        tol = np.maximum(1.5, ARM_GAP * loc[i])           # closer than ~half the local radius = same lumen
                        if (cKDTree(near[j]).query(near[i])[0] < tol).sum() >= 2:
                            why = "arms-touch"
            if why:
                review(reg, why); continue
            H.remove_nodes_from(reg - {J}); H.remove_edges_from(list(H.edges(J)))
            H.nodes[J]["x"], H.nodes[J]["y"] = float(j_rc[1]), float(Hm - j_rc[0])
            for k, w, d in new:
                H.add_edge(J, w, **d); relink(k, (J, w))
            for e in internal:
                relink(e, None)
            report["junction"] += 1
        report["dropped_recordings"] += n_int; report["regions"].append(sorted(int(n) for n in reg))
    return H, {e: v for e, v in emap.items() if v is not None}, report, mf
