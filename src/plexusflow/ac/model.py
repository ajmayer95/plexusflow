"""Compliant network model for the harmonic field on the carved network.

Wall: complex areal distensibility D_eff = D (1 − iη) (constant loss tangent).  Exact distributed two-port per vessel:
    Y_self = G ξ coth ξ,  Y_trans = G ξ csch ξ,  ξ² = iω D_eff 8μℓ²/r²,  G = π r⁴ / (8 μ ℓ)
Vessel-mean flux H_e = ½[(Y_s P_t − Y_t P_h) + (Y_t P_t − Y_s P_h)].
Boundary forcing at carve + vein nodes: f_n^(k) = f_DC,n · r_n^(k), r = local measured ratio
Σ s·Ĥ_k / Σ s·Q' over AC-eligible edges within ≤ 4 hops of n (stop once ≥ 3 edges found).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU, PX_M, NL = 3.5e-3, 1.7e-6, 1e-12      # Pa·s, m per pixel, m³ per nL


def build_context(dc, field, dom):
    """Arrays for the harmonic solves: topology, geometry (consensus radii), DC field, eligible Ĥ, boundary forcing F."""
    nodes = dom["nodes"]; ix = {n: i for i, n in enumerate(nodes)}; sub = dom["edges"]
    tail = np.array([ix[u] for u, v in sub]); head = np.array([ix[v] for u, v in sub]); nE = len(sub); nN = len(nodes)
    Qp, fb, elig = dc["Qp"], dc["fb"], field["elig"]
    H = {1: field["H"][0], 2: field["H"][1]}
    adj = {n: [] for n in range(nN)}
    for e in range(nE):
        adj[tail[e]].append((e, 1.0, head[e])); adj[head[e]].append((e, -1.0, tail[e]))

    def local_r(n0, k):
        seen = {n0}; frontier = [n0]; ah = aq = 0j; cnt = 0
        for _ in range(4):
            nxt = []
            for n_ in frontier:
                for e_, val, nb in adj[n_]:
                    if nb in seen:
                        continue
                    seen.add(nb); nxt.append(nb)
                    if elig[e_]:
                        ah += val * H[k][e_]; aq += val * Qp[e_]; cnt += 1
            frontier = nxt
            if cnt >= 3:
                break
        return ah / aq if cnt and abs(aq) > 0 else 0.0
    F = {(k, m): np.zeros(nN, complex) for k in (1, 2) for m in ("P=0", "flux")}
    for k in (1, 2):
        for n_ in dom["inj"]:
            v = fb[n_] * local_r(n_, k); F[(k, "P=0")][n_] = v; F[(k, "flux")][n_] = v
        for n_ in dom["out"]:
            F[(k, "flux")][n_] = fb[n_] * local_r(n_, k)
    return dict(f0=field["f_ref"], nN=nN, nE=nE, dom=dom, Qp=Qp, Qs_dc=dc["Qs"],
                H={1: np.where(elig, H[1], np.nan), 2: np.where(elig, H[2], np.nan)},
                elig=elig, F=F, tail=tail, head=head, r_m=dc["radius"] * PX_M, l_m=dc["length"] * PX_M)


def _admittances(C, k, D, eta):
    r, l = C["r_m"], C["l_m"]; Gc = np.pi * r ** 4 / (8 * MU * l); om = 2 * np.pi * k * C["f0"]
    ks = 1j * om * (D * (1 - 1j * eta)) * 8 * MU * l ** 2 / r ** 2
    xi = np.sqrt(ks.astype(complex)); small = np.abs(xi) < 1e-3
    Ys = Gc * np.where(small, 1 + ks / 3, xi / np.tanh(np.where(small, 1, xi)))
    Yt = Gc * np.where(small, 1 - ks / 6, xi / np.sinh(np.where(small, 1, xi)))
    t, h, nN = C["tail"], C["head"], C["nN"]
    Y = sp.csr_matrix((np.r_[Ys, Ys, -Yt, -Yt], (np.r_[t, h, t, h], np.r_[t, h, h, t])), shape=(nN, nN)).tocsc()
    return Ys, Yt, Y


def solve(C, k, D, eta, vmode="flux"):
    """Vessel-mean harmonic k (nL/s) for the context's forcing.  vmode 'flux': all boundary nodes driven;
    'P=0': veins held at zero pressure."""
    Ys, Yt, Y = _admittances(C, k, D, eta); t, h, nN = C["tail"], C["head"], C["nN"]
    f = C["F"][(k, vmode)] * NL
    if vmode == "P=0":
        free = np.setdiff1d(np.arange(nN), C["dom"]["out"])
        P = np.zeros(nN, complex); P[free] = spla.spsolve(Y[free][:, free], f[free])
    else:
        P = spla.spsolve(Y, f)
    return 0.5 * (Ys + Yt) * (P[t] - P[h]) / NL


def response(C, k, D, eta, bnd):
    """M (nE × nB): vessel-mean H_k per unit complex flux (nL/s) injected at each boundary node."""
    Ys, Yt, Y = _admittances(C, k, D, eta); t, h, nN = C["tail"], C["head"], C["nN"]
    RHS = np.zeros((nN, len(bnd)), complex); RHS[bnd, np.arange(len(bnd))] = NL
    Pm = spla.splu(Y).solve(RHS)
    return 0.5 * ((Ys[:, None] * Pm[t] - Yt[:, None] * Pm[h]) + (Yt[:, None] * Pm[t] - Ys[:, None] * Pm[h])) / NL


def ac_reliability(G, field, model):
    """DC reliability c (reconcile.reliability) of each edge's AC-winner measurement; nan where not AC-eligible."""
    from ..reconcile.reliability import c_of_measurement
    c = np.full(len(field["edges"]), np.nan)
    for i, (e, t) in enumerate(zip(field["edges"], field["tile"])):
        if t >= 0:
            c[i] = c_of_measurement(model, next(m for m in G.edges[e]["measurements_piv"] if m["tile_id"] == t))
    return c
