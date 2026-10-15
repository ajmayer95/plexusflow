"""Carved network, conservative projection, emergent boundary and passive (Poiseuille) flow.

Conventions: edges are (u, v) with fluxes positive u→v (tail→head); incidence B[tail, e] = +1,
B[head, e] = −1, so (BQ)_n = net flux OUT of node n (+ injection into the network, − withdrawal).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS_UNIMAGED = 1e-6     # relative weight of unmeasured edges in the projection


def carved_domain(G):
    """Carved network: radial_uncertain (porous) edges and DA anchor stubs removed.  Boundary = carve-boundary nodes
    (nodes of the carved network touching a removed edge, excluding V anchors) + V nodes.  Carve nodes are labelled
    L/R by x relative to the mean DA-anchor x."""
    from .carve import find_porous_and_outlets
    _, _, _, A_anch, V_anch, _ = find_porous_and_outlets(G)
    edges = list(G.edges())
    carved = np.array([not G.edges[e].get("radial_uncertain") and e[0] not in A_anch and e[1] not in A_anch for e in edges])
    touched = {n for e, k in zip(edges, carved) if not k for n in e}
    cnodes = {n for e, k in zip(edges, carved) if k for n in e}
    carve_bnd = sorted(n for n in cnodes if n in touched and n not in V_anch)
    sub = [e for e, k in zip(edges, carved) if k]
    nodes = sorted({n for e in sub for n in e}); ix = {n: i for i, n in enumerate(nodes)}; m_ = len(sub)
    B = sp.csr_matrix((np.r_[np.ones(m_), -np.ones(m_)], (np.r_[[ix[u] for u, v in sub], [ix[v] for u, v in sub]],
                                                       np.r_[np.arange(m_), np.arange(m_)])), shape=(len(nodes), m_))
    mid = float(np.mean([G.nodes[n]["x"] for n in A_anch]))
    inj = np.array([ix[n] for n in carve_bnd]); out = np.array([ix[n] for n in sorted(V_anch) if n in ix])
    side = np.array(["L" if G.nodes[n]["x"] < mid else "R" for n in carve_bnd])
    return dict(name="carved", edges=sub, nodes=nodes, B=B, inj=inj, out=out, bnd=np.r_[inj, out], side=side,
                V=[n for n in sorted(V_anch) if n in ix], DA=sorted(A_anch))


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
    B = dom["B"]; n, lab = cg.connected_components(abs(B) @ abs(B).T, directed=False)
    pref = list(dom["out"]) + list(dom["inj"]); g = {}
    for i in pref:
        g.setdefault(lab[i], i)
    for i in range(B.shape[0]):
        g.setdefault(lab[i], i)
    return np.array(sorted(g.values()))


def passive(K, fb, dom):
    """Passive flow with flux fb imposed at every node (Σ fb = 0 per connected piece); one pressure pinned per piece
    (a V node where available) as gauge."""
    B = dom["B"]; L = (B @ sp.diags(K) @ B.T).tocsc()
    free = np.setdiff1d(np.arange(B.shape[0]), _gauge_nodes(dom))
    P = np.zeros(B.shape[0]); P[free] = spla.spsolve(L[free][:, free], fb[free])
    return K * (B.T @ P)


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
