"""Compliant network model for the harmonic field on the carved network.

Wall: complex areal distensibility D_eff = D (1 − iη) (constant loss tangent).  Exact distributed two-port per vessel:
    Y_self = G ξ coth ξ,  Y_trans = G ξ csch ξ,  ξ² = iω D_eff 8μℓ²/r²,  G = π r⁴ / (8 μ ℓ)
Vessel-mean flux H_e = ½[(Y_s P_t − Y_t P_h) + (Y_t P_t − Y_s P_h)].
Boundary forcing at carve + vein nodes: f_n^(k) = f_DC,n · r_n^(k), r = local measured ratio
Σ s·Ĥ_k / Σ s·Q' over AC-eligible edges within ≤ 4 hops of n (stop once ≥ 3 edges found).
Vein nodes can instead be driven by their stub waveforms (vein_forcing): the measured mean-normalised harmonics
Ĥ_k/Q' = H_k/a0 of the vessel(s) ending at the node — 'own' (each node its own stub) or 'mean' (one waveform, the
complex mean over the stubs, at every vein node).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU, PX_M, NL = 3.5e-3, 1.7e-6, 1e-12      # Pa·s, m per pixel, m³ per nL


TOPO_MODES = ("sealed", "flux", "supernode")


RATIO_MODES = ("weighted", "sum")
RATIO_FLAG = 2.0          # |r| above this at a boundary node is reported in C["forcing_flags"]


def build_context(dc, field, dom, topo="flux", supernode_scale=1.0, ratio="weighted"):
    """Arrays for the harmonic solves: topology, geometry (consensus radii), DC field, eligible Ĥ, boundary forcing F.

    Boundary waveform r_n^(k) at a boundary node n (f_n^(k) = f_DC,n · r_n^(k)) from the AC-eligible vessels e within
    4 hops (stop once ≥ 3 are found):
      'weighted' (default since 2026-11-16)  r = Σ |Q'_e| h_e / Σ |Q'_e|,  h_e = Ĥ_e / Q'_e — the flux-weighted mean of the
                 vessels' own normalised waveforms; |r| ≤ max_e |h_e|.
      'sum'      r = Σ s_e Ĥ_e / Σ s_e Q'_e (s = orientation w.r.t. n) — the previous rule; blows up where the nearby mean
                 flows cancel (21S: one node at 0.6 % of the left inflow had |r| ≈ 130 and set 82 % of the left forcing).

    Topology-uncertain regions (dom['regions'], conserving black boxes in the DC reconstruction) need a pulsatile
    closure — the DC black box says nothing about how they transmit pressure.  Sensitivity options (none is primary):
      'sealed'     no oscillatory exchange at the interfaces (diagnostic baseline; unphysical)
      'flux'       interfaces driven like the carve boundary: f_n^(k) = f_DC,n · r_n^(k), r the local measured ratio —
                   fixed inputs (boundary group 4, never corrected by the fit)
      'supernode'  each region a perfectly conducting compliant compartment: its interfaces tied by a near-zero
                   resistance, lumped compliance D_eff · V_region · supernode_scale (V from the region's traced vessels)
    Leaky regions (fed through an open region) are already inflow boundary nodes and are driven by the local ratio."""
    if topo not in TOPO_MODES:
        raise ValueError(f"topo must be one of {TOPO_MODES}, not {topo!r}")
    if ratio not in RATIO_MODES:
        raise ValueError(f"ratio must be one of {RATIO_MODES}, not {ratio!r}")
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
                        if ratio == "sum":
                            ah += val * H[k][e_]; aq += val * Qp[e_]
                        else:                                        # sgn(Q')·Ĥ / |Q'| summed = Σ|Q'| h_e, h_e = Ĥ/Q'
                            ah += np.sign(Qp[e_]) * H[k][e_]; aq += abs(Qp[e_])
                        cnt += 1
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
        if topo == "flux":
            for n_ in dom.get("topo_cut", []):
                v = fb[n_] * local_r(n_, k); F[(k, "P=0")][n_] = v; F[(k, "flux")][n_] = v
    flags = sorted({int(nodes[n_]) for k in (1, 2) for (kk, m), F_ in F.items() if kk == k and m == "flux"
                    for n_ in np.flatnonzero(np.abs(F_) > RATIO_FLAG * np.maximum(np.abs(fb), 1e-12))})
    sup = [(np.array([ix[n] for n in r["cut"]]), r["volume_px3"] * PX_M ** 3 * supernode_scale)
           for r in dom.get("regions", []) if not r["leaky"] and r["cut"]] if topo == "supernode" else []
    return dict(topo=topo, supernode_scale=supernode_scale, super=sup, ratio=ratio, forcing_flags=flags, f0=field["f_ref"], nN=nN, nE=nE, dom=dom, Qp=Qp, Qs_dc=dc["Qs"],
                H={1: np.where(elig, H[1], np.nan), 2: np.where(elig, H[2], np.nan)},
                elig=elig, F=F, tail=tail, head=head, r_m=dc["radius"] * PX_M, l_m=dc["length"] * PX_M,
                rc_m=np.asarray(dc.get("radius_compliance", dc["radius"])) * PX_M)


def _admittances(C, k, D, eta):
    r, l = C["r_m"], C["l_m"]; Gc = np.pi * r ** 4 / (8 * MU * l); om = 2 * np.pi * k * C["f0"]
    rc = C.get("rc_m", r)                                # compliance radius a_C (dc["radius_compliance"]); default = r
    # κ² = iω R'C' l², R' = 8μ/(π a_R⁴), C' = D π a_C²  →  iω D 8μ l² a_C² / a_R⁴  (= 8μ D l² / r² for one radius)
    ks = 1j * om * (D * (1 - 1j * eta)) * 8 * MU * l ** 2 * rc ** 2 / r ** 4
    xi = np.sqrt(ks.astype(complex)); small = np.abs(xi) < 1e-3
    Ys = Gc * np.where(small, 1 + ks / 3, xi / np.tanh(np.where(small, 1, xi)))
    Yt = Gc * np.where(small, 1 - ks / 6, xi / np.sinh(np.where(small, 1, xi)))
    t, h, nN = C["tail"], C["head"], C["nN"]
    Y = sp.csr_matrix((np.r_[Ys, Ys, -Yt, -Yt], (np.r_[t, h, t, h], np.r_[t, h, h, t])), shape=(nN, nN)).tocsc()
    if C.get("super"):                                   # topology regions as conducting compliant compartments
        gbig = 1e3 * float(np.max(Gc)); rows, cols, vals = [], [], []
        for cut, V in C["super"]:
            rep = cut[0]
            for n in cut[1:]:
                rows += [rep, n, rep, n]; cols += [rep, n, n, rep]; vals += [gbig, gbig, -gbig, -gbig]
            rows.append(rep); cols.append(rep); vals.append(1j * om * D * (1 - 1j * eta) * V)
        Y = (Y + sp.csr_matrix((np.array(vals, complex), (rows, cols)), shape=(nN, nN))).tocsc()
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
    from plexusflow.reconcile.reliability import c_of_measurement
    c = np.full(len(field["edges"]), np.nan)
    for i, (e, t) in enumerate(zip(field["edges"], field["tile"])):
        if t >= 0:
            c[i] = c_of_measurement(model, next(m for m in G.edges[e]["measurements_piv"] if m["tile_id"] == t))
    return c


def vein_forcing(C, field, mode="mean", gauge_rad=None):
    """Replace the vein-node forcing of context C by stub waveforms.  field: the FULL AC field (all eligible vessels —
    stubs are usually outside the measurement-supported fit set because their DC is under-read).  Per vein node n the stub
    ratio r_n,k = Σ |Q'| (Ĥ_k/Q') / Σ |Q'| over its incident AC-eligible vessels (Ĥ gauge-corrected by gauge_rad[tile] if
    given), and f_n,k = f_DC,n · r with
        'own'   r = r_n (nodes without an eligible stub keep the local-ratio forcing)
        'mean'  r = mean_n r_n at every vein node (one shared venous waveform)
    Returns (C', info) with info = dict(stubbed=[node indices], ratio={k: {node: r}}, mean={k: r̄})."""
    out = np.asarray(C["dom"]["out"]); t, h = C["tail"], C["head"]; Qp = C["Qp"]; fdc = C["dom"]["B"] @ Qp
    el = field["elig"]; g = gauge_rad or {}
    ratio = {1: {}, 2: {}}
    for n in out:
        es = [(e, 1.0) for e in np.flatnonzero(t == n)] + [(e, -1.0) for e in np.flatnonzero(h == n)]
        es = [(e, s_) for e, s_ in es if el[e]]
        if not es:
            continue
        for k in (1, 2):
            Hh = sum(np.sign(Qp[e]) * field["H"][k - 1][e] * np.exp(-1j * k * g.get(int(field["tile"][e]), 0.0)) for e, s_ in es)
            ratio[k][int(n)] = Hh / sum(abs(Qp[e]) for e, s_ in es)            # flux-weighted mean of Ĥ/Q' (as build_context)
    stubbed = sorted(ratio[1]); C2 = dict(C); C2["F"] = dict(C["F"]); rbar = {}
    if not stubbed:
        raise ValueError("no vein node has an AC-eligible stub")
    for k in (1, 2):
        F = C["F"][(k, "flux")].copy(); rbar[k] = np.mean([ratio[k][n] for n in stubbed])
        for n in out:
            if mode == "mean":
                F[n] = fdc[n] * rbar[k]
            elif mode == "own" and int(n) in ratio[k]:
                F[n] = fdc[n] * ratio[k][int(n)]
            elif mode != "own":
                raise ValueError(f"unknown vein forcing mode {mode!r}")
        C2["F"][(k, "flux")] = F
    return C2, dict(stubbed=stubbed, ratio=ratio, mean=rbar)


def group_response(C, k, D, eta, gid, el):
    """Vessel-mean H_k (rows: eligible edges) driven by the context's flux forcing restricted to each boundary group.
    C["vmode"] = "P=0" holds the vein nodes at zero pressure instead (their flux forcing is then unused)
    (columns g = 0..3; column 4 = the fixed, never-corrected forcing at topology-region interfaces).  Equals response(C, k, D, eta, bnd)[el][:, gid == g] @ f[bnd][gid == g], with 4 right-hand sides
    instead of one per boundary node."""
    Ys, Yt, Y = _admittances(C, k, D, eta); t, h, nN = C["tail"], C["head"], C["nN"]; bnd = np.asarray(C["dom"]["bnd"])
    f = C["F"][(k, "flux")][bnd]; RHS = np.zeros((nN, 5), complex)
    for g in range(5):                                   # column 4: fixed forcing (topology-region interfaces, flux closure)
        RHS[bnd[gid == g], g] = f[gid == g] * NL
    if C.get("vmode", "flux") == "P=0":                  # veins held at zero pressure (needed for the rigid limit D → 0)
        out = np.asarray(C["dom"]["out"]); free = np.setdiff1d(np.arange(nN), out); P = np.zeros((nN, 5), complex)
        P[free] = spla.splu(Y[free][:, free].tocsc()).solve(RHS[free])
    else:
        P = spla.splu(Y).solve(RHS)
    return (0.5 * (Ys + Yt)[:, None] * (P[t] - P[h]) / NL)[el]
