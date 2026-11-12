"""Carved network, conservative projection, emergent boundary and passive (Poiseuille) flow.

Conventions: edges are (u, v) with fluxes positive u→v (tail→head); incidence B[tail, e] = +1,
B[head, e] = −1, so (BQ)_n = net flux OUT of node n (+ injection into the network, − withdrawal).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS_UNIMAGED = 1e-6     # relative weight of unmeasured edges in the projection


def carved_domain(G, topology=True):
    """Carved network the reconstruction runs on (region classes: ``reconcile.regions``).

    Open regions (``open_region`` / legacy ``radial_uncertain``) and the DA anchor stubs are removed; the nodes where the
    remaining network meets them are boundary nodes whose exchange counts as inflow (``inj``, labelled L/R by x relative
    to the mean DA-anchor x).  V anchors are the outflow boundary (``out``).

    Topology-uncertain regions (``topology_uncertain``; ignored when ``topology=False``) are also removed, but each
    connected region is a mass-conserving black box: its cut nodes are free boundary nodes (``topo_cut``; how flow is
    distributed around the region is unknown), and one extra conservation row per region (the sum of its cut-node rows,
    appended below the ``n_real`` node rows of B) forces its net exchange to zero.  A region touching an open region or
    an anchor is 'leaky': it is fed through the open region, so it has no conservation row and its cut nodes join the
    inflow boundary (``inj``, listed in ``inj_from_topology``) — accounted exactly as if the region were tagged open.  Without topology tags the domain is
    identical to the open-region-only carve."""
    from .carve import find_porous_and_outlets
    from .regions import is_open, is_topology
    _, _, _, A_anch, V_anch, _ = find_porous_and_outlets(G)
    edges = list(G.edges())
    topo = np.array([bool(topology and is_topology(G.edges[e])) for e in edges])
    opn = np.array([is_open(G.edges[e]) or e[0] in A_anch or e[1] in A_anch for e in edges]) & ~topo
    carved = ~(opn | topo)
    touched = {n for e, k in zip(edges, opn) if k for n in e}
    cnodes = {n for e, k in zip(edges, carved) if k for n in e}
    carve_bnd = sorted(n for n in cnodes if n in touched and n not in V_anch)
    sub = [e for e, k in zip(edges, carved) if k]
    nodes = sorted({n for e in sub for n in e}); ix = {n: i for i, n in enumerate(nodes)}; m_ = len(sub)
    B = sp.csr_matrix((np.r_[np.ones(m_), -np.ones(m_)], (np.r_[[ix[u] for u, v in sub], [ix[v] for u, v in sub]],
                                                       np.r_[np.arange(m_), np.arange(m_)])), shape=(len(nodes), m_))
    mid = float(np.mean([G.nodes[n]["x"] for n in A_anch]))
    inj = np.array([ix[n] for n in carve_bnd]); out = np.array([ix[n] for n in sorted(V_anch) if n in ix])
    side = np.array(["L" if G.nodes[n]["x"] < mid else "R" for n in carve_bnd])
    side_out = np.array(["L" if G.nodes[n]["x"] < mid else "R" for n in sorted(V_anch) if n in ix])     # V nodes, same rule
    dom = dict(name="carved", edges=sub, nodes=nodes, B=B, inj=inj, out=out, bnd=np.r_[inj, out], side=side, side_out=side_out,
               V=[n for n in sorted(V_anch) if n in ix], DA=sorted(A_anch), n_real=len(nodes), topo_cut=np.array([], int), regions=[])
    if topo.any():
        import networkx as nx
        H = nx.Graph(); H.add_edges_from([e for e, k in zip(edges, topo) if k])
        fixed = set(carve_bnd) | set(V_anch) | set(A_anch); rows, cut_all, leaky_cut = [], [], []
        for comp in sorted(nx.connected_components(H), key=lambda c: -len(c)):
            cut = sorted(n for n in comp if n in cnodes and n not in fixed)
            leaky = bool(comp & (touched | set(A_anch) | set(V_anch)))
            vol = float(sum(np.pi * G.edges[e]["radius"] ** 2 * G.edges[e]["length"] for e in H.subgraph(comp).edges()))
            reg = dict(n_edges=H.subgraph(comp).number_of_edges(), nodes=sorted(comp), cut=cut, leaky=leaky, row=None, volume_px3=vol)
            if cut and not leaky:
                reg["row"] = len(nodes) + len(rows); rows.append(np.asarray(B[[ix[n] for n in cut]].sum(0)).ravel())
            if leaky:                                  # fed through an open region: its interface exchange is inflow
                leaky_cut += cut
            else:
                cut_all += [ix[n] for n in cut]
            dom["regions"].append(reg)
        if rows:
            dom["B"] = sp.vstack([B, sp.csr_matrix(np.array(rows))]).tocsr()
        if leaky_cut:
            extra = sorted(set(leaky_cut) - set(carve_bnd)); carve_bnd = carve_bnd + extra
            dom["inj"] = inj = np.array([ix[n] for n in carve_bnd])
            dom["side"] = np.array(["L" if G.nodes[n]["x"] < mid else "R" for n in carve_bnd])
            dom["inj_from_topology"] = np.array([ix[n] for n in extra], int)
        dom["topo_cut"] = np.array(sorted(set(cut_all)), int); dom["bnd"] = np.r_[inj, out, dom["topo_cut"]]
    return dom


def project(Qm, w, dom):
    """Q' = argmin ½ Σ w (Q − Qm)²  s.t. B_I Q = 0 (KKT system)."""
    B = dom["B"]; nN = B.shape[0]
    C = B[np.setdiff1d(np.arange(nN), dom["bnd"])]
    KKT = sp.bmat([[sp.diags(w), C.T], [C, None]], format="csc")
    return spla.spsolve(KKT, np.r_[w * Qm, np.zeros(C.shape[0])])[:len(Qm)]


def emergent(Qp, dom):
    """f = B Q' restricted to boundary nodes (+ = injection, − = withdrawal)."""
    f = dom["B"] @ Qp; fb = np.zeros_like(f); fb[dom["bnd"]] = f[dom["bnd"]]
    return fb


def _gauge_nodes(dom):
    """One pressure-gauge node per connected piece of the carved network (uncertain tags can split it): a V node
    if the piece has one, else a boundary node, else its first node.  A piece without a gauge makes L singular."""
    import scipy.sparse.csgraph as cg
    B = dom["B"][:dom.get("n_real", dom["B"].shape[0])]; n, lab = cg.connected_components(abs(B) @ abs(B).T, directed=False)
    pref = list(dom["out"]) + list(dom["inj"]); g = {}
    for i in pref:
        g.setdefault(lab[i], i)
    for i in range(B.shape[0]):
        g.setdefault(lab[i], i)
    return np.array(sorted(g.values()))


def passive(K, fb, dom):
    """Passive flow with flux fb imposed at every node (Σ fb = 0 per connected piece); one pressure pinned per piece
    (a V node where available) as gauge."""
    nr = dom.get("n_real", dom["B"].shape[0]); B = dom["B"][:nr]; L = (B @ sp.diags(K) @ B.T).tocsc(); fb = np.asarray(fb)[:nr]
    free = np.setdiff1d(np.arange(B.shape[0]), _gauge_nodes(dom))
    P = np.zeros(B.shape[0]); P[free] = spla.spsolve(L[free][:, free], fb[free])
    return K * (B.T @ P)



MU_PA_S = 3.5e-3        #: blood viscosity (Pa·s) used for the venous-pressure penalty
UM_PER_PX = 1.7         #: mosaic pixel size (µm); radii and lengths in the reconciliation are in pixels
NL = 1e-12              #: 1 nL in m³


def venous_pressure_penalty(fb, dom, radius_px, length_px, sigma, sigma_P=2.0, mu=MU_PA_S, um_per_px=UM_PER_PX):
    """Soft equal-pressure constraint at the two vitelline-vein outlets (paper Fig 7, passive prediction).

    The two outflows (f_V1, f_V2) are re-split, at a fixed total, to minimise
        Σ_V (f_V − f_V^meas)² / σ_V²  +  (P_V1 − P_V2)² / σ_P²,
    where P is the Poiseuille pressure for conductances πr⁴/(8μℓ) with V1 as gauge and σ_V the root-sum-square measurement
    uncertainty of the vessels at each outlet.  Linear in the shift, so solved in closed form; σ_P = inf leaves fb as is.
    Returns (fb_adjusted [nL/s], info dict with shift, ΔP before / after [Pa]).  Arterial boundary fluxes are untouched."""
    B = dom["B"]; px = um_per_px * 1e-6
    Kp = np.pi * (np.asarray(radius_px) * px) ** 4 / (8 * mu * np.asarray(length_px) * px)
    Lp = (B @ sp.diags(Kp) @ B.T).tocsc()
    v1, v2 = dom["out"]; free = np.setdiff1d(np.arange(B.shape[0]), [v1]); lu = spla.splu(Lp[free][:, free])

    def pres(f):
        p_ = np.zeros(B.shape[0]); p_[free] = lu.solve(f[free]); return p_
    Bc = B.tocsc()
    sV = [float(np.sqrt(np.sum(np.asarray(sigma)[Bc[v].nonzero()[1]] ** 2))) for v in (v1, v2)]
    d_ = np.zeros(B.shape[0]); d_[v1], d_[v2] = -1.0, 1.0
    pA, pD = pres(np.asarray(fb) * NL), pres(d_ * NL); dP0, kk = pA[v2] - pA[v1], pD[v2] - pD[v1]
    shift = 0.0 if not np.isfinite(sigma_P) else -(dP0 * kk / sigma_P ** 2) / (1 / sV[0] ** 2 + 1 / sV[1] ** 2 + kk ** 2 / sigma_P ** 2)
    out = np.array(fb, float); out[v1] -= shift; out[v2] += shift
    return out, dict(shift=float(shift), dP_before=float(dP0), dP_after=float(dP0 + kk * shift), sigma_V=sV, sigma_P=float(sigma_P))


def contract_short_unimaged(dom, imaged, length, radius, k=2.0):
    """Unimaged edges shorter than k radii lie inside a junction's lumen (vectorisation splits one crossing into two nearby
    nodes).  Contracting them is exact for the reconstruction: such an edge's flow only links the balances of its two end nodes,
    so eliminating it equals summing them (up to the ε ridge on unimaged edges).  Returns the mask of these edges and vessel /
    junction counts of the contracted network (for reporting; the estimator is unchanged)."""
    short = (~np.asarray(imaged, bool)) & (np.asarray(length) < k * np.asarray(radius))
    B = dom["B"][:dom.get("n_real", dom["B"].shape[0])].tocsc(); nN = B.shape[0]; par = np.arange(nN)
    def root(n):
        while par[n] != n:
            par[n] = par[par[n]]; n = par[n]
        return n
    for j in np.flatnonzero(short):
        u, v = B[:, j].indices
        ru, rv = root(u), root(v)
        if ru != rv:
            par[ru] = rv
    roots = np.array([root(n) for n in range(nN)]); keep = ~short
    deg = np.bincount(np.r_[roots[B[:, keep].tocoo().row]], minlength=nN)          # degree of each merged node, kept edges only
    bnd = set(roots[[b for b in dom["bnd"] if b < nN]])
    return dict(short_unimaged=short, k_radii=float(k), n_short_unimaged=int(short.sum()), n_vessels=int(keep.sum()),
                n_imaged=int(np.sum(keep & imaged)), n_unimaged=int(np.sum(keep & ~imaged)),
                n_junctions=int(sum(1 for r in np.unique(roots) if deg[r] >= 3 and r not in bnd)),
                n_merged_junctions=int(sum(1 for r, c in zip(*np.unique(roots, return_counts=True)) if c > 1)))


def metrics(Qm, Qp, Qs, fb, dom, imaged):
    """Boundary totals, L/R split, |Q'| vs |Q_sim| correlation, loop share, movement and sign flips."""
    a, b = np.abs(Qp[imaged]), np.abs(Qs[imaged]); ok = (a > 1e-15) & (b > 1e-15)
    inj = fb[dom["inj"]]; L = inj[dom["side"] == "L"].sum(); Rr = inj[dom["side"] == "R"].sum()
    return dict(
        total_in=float(inj.sum()), inj_L=float(L), inj_R=float(Rr), LR=float(L / Rr) if Rr else np.nan,
        V=[float(x) for x in fb[dom["out"]]], n_withdraw=int(np.sum(inj < 0)),
        corr_lin=float(np.corrcoef(a, b)[0, 1]), corr_log=float(np.corrcoef(np.log(a[ok]), np.log(b[ok]))[0, 1]),
        loop_share=float(np.linalg.norm(Qp - Qs) / np.linalg.norm(Qp)),
        move_p50=float(np.median(np.abs(Qp[imaged] - Qm[imaged]) / np.abs(Qm[imaged]))),
        flips=float(np.mean(np.sign(Qp[imaged]) != np.sign(Qm[imaged]))),
        n_measured=int(imaged.sum()))
