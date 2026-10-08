"""Option-A AC field on the carved network, from the measured graph and the DC reconciliation.

1. Per Kloosterman-valid measurement in the f0 band: a0, H1..H3 (LSQ at the tile f0; Q_t signed along the
   oriented path u→v), per-beat H1 scatter, coherence ρ.
2. Tile time gauge ψ_t: angular synchronisation of H1 over edges measured in ≥ 2 in-band tiles (ρ ≥ 0.1);
   leading eigenvector of Σ_e u_a ū_b (u = unit H1), ψ_ref = 0.  Applied coherently: H_k → e^{−ikψ_t} H_k.
3. AC winner per carved edge: |a0| > Q_DET, ρ ≥ RHO_MIN, sign(a0) = sign(Q'); smallest per-beat H1 scatter.
4. Ĥ_k = (|Q'_e| / |a0|) e^{−ikψ} H_k,raw   (u→v frame).
"""
from __future__ import annotations
import numpy as np
from ..reconcile.harmonics import phasors, coherence, beat_scatter

REF_TILE, F0_BAND, RHO_MIN, Q_DET, RHO_SYNC = 21, 0.15, 0.02, 1e-4, 0.1


def _measure(q, f0):
    q = np.asarray(q, float); a0, H = phasors(q, f0)
    return a0, H, beat_scatter(q, f0), coherence(q, f0)


def build_ac_field(G, dc, n_jobs=10, ref_tile=REF_TILE, f0_band=F0_BAND):
    """dc: output of reconcile_dc (needs edges, Qp).  Returns the field dict (H is 3 × nE, u→v frame)."""
    from joblib import Parallel, delayed
    tf0 = G.graph["tile_f0_piv"]; f_ref = tf0[ref_tile]
    band = {t for t, f in tf0.items() if abs(f / f_ref - 1) <= f0_band}
    sub = dc["edges"]; eix = {e: i for i, e in enumerate(sub)}
    items = []
    for e in sub:
        for m in G.edges[e].get("measurements_piv") or []:
            if (m.get("quality") or {}).get("kl_ok") and m["tile_id"] in band and m.get("Q_t") is not None:
                items.append((e, m["tile_id"], m["Q_t"]))
    res = Parallel(n_jobs=n_jobs)(delayed(_measure)(q, tf0[t]) for e, t, q in items)
    meas = {}
    for (e, t, _), (a0, H, sc, rho) in zip(items, res):
        meas.setdefault(e, {})[t] = dict(a0=a0, H=H, scatter=sc, rho=rho)

    tiles = sorted(band); ti = {t: i for i, t in enumerate(tiles)}; Mx = np.zeros((len(tiles), len(tiles)), complex); pairs = []
    for e, d in meas.items():
        ok = [t for t, x in d.items() if x["rho"] >= RHO_SYNC and abs(x["H"][0]) > 0]
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]; ua = d[a]["H"][0] / abs(d[a]["H"][0]); ub = d[b]["H"][0] / abs(d[b]["H"][0])
                Mx[ti[a], ti[b]] += ua * np.conj(ub); Mx[ti[b], ti[a]] += ub * np.conj(ua); pairs.append((e, a, b))
    Mx += np.diag(np.full(len(tiles), 1e-9 + np.abs(Mx).sum(1).max()))
    _, V = np.linalg.eigh(Mx); vec = V[:, -1]
    psi = {t: float(np.angle(vec[ti[t]] / vec[ti[ref_tile]])) for t in tiles}

    Qp = dc["Qp"]; nE = len(sub); Hf = np.full((3, nE), np.nan + 0j); win_t = np.full(nE, -1); elig = np.zeros(nE, bool)
    sc_w = np.full(nE, np.nan); rho_w = np.full(nE, np.nan)
    for e, d in meas.items():
        i = eix[e]
        cand = [(t, x) for t, x in d.items() if abs(x["a0"]) > Q_DET and x["rho"] >= RHO_MIN and np.sign(x["a0"]) == np.sign(Qp[i])
                and np.isfinite(x["scatter"])]
        if not cand:
            continue
        t, x = min(cand, key=lambda c: c[1]["scatter"])
        Hf[:, i] = abs(Qp[i]) / abs(x["a0"]) * x["H"] * np.exp(-1j * np.arange(1, 4) * psi[t]); win_t[i] = t; elig[i] = True
        sc_w[i] = x["scatter"]; rho_w[i] = x["rho"]
    return dict(edges=sub, H=Hf, elig=elig, tile=win_t, psi=psi, scatter=sc_w, rho=rho_w, f_ref=f_ref, band=sorted(band),
                Qp=Qp, _meas=meas, _pairs=pairs)


def sync_mismatch(field, aligned=True):
    """Median cross-tile H1 phase mismatch (degrees) over the synchronisation pairs."""
    meas, psi, out = field["_meas"], field["psi"], []
    for e, a, b in field["_pairs"]:
        ha, hb = meas[e][a]["H"][0], meas[e][b]["H"][0]
        if aligned:
            ha, hb = ha * np.exp(-1j * psi[a]), hb * np.exp(-1j * psi[b])
        out.append(abs(np.angle(ha * np.conj(hb))))
    return float(np.degrees(np.median(out)))
