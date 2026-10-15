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


def _sync(pa, pb, pw, pz, nT, ref_i, start=None, n_iter=2000):
    """Tile time gauge from vessel pairs: maximise Σ w·Re(z_ab e^{i(ψ_b − ψ_a)}) with ψ_ref = 0.  Spectral start (or
    `start`), then per-tile normalised fixed-point iteration (generalised power method)."""
    W = np.zeros((nT, nT), complex)
    np.add.at(W, (pa, pb), pw * pz); np.add.at(W, (pb, pa), pw * np.conj(pz))
    if start is None:
        _, V = np.linalg.eigh(W + np.diag(np.full(nT, 1e-12 + np.abs(W).sum(1).max())))
        z = V[:, -1] / np.maximum(np.abs(V[:, -1]), 1e-300)
    else:
        z = np.exp(1j * np.asarray(start))
    for _ in range(n_iter):
        zn = W @ z; nz = np.abs(zn) > 0
        zn = np.where(nz, zn / np.where(nz, np.abs(zn), 1), z)
        done = np.max(np.abs(np.angle(zn * np.conj(z)))) < 1e-10; z = zn
        if done:
            break
    return np.angle(z * np.conj(z[ref_i]))


def build_ac_field(G, dc, n_jobs=10, ref_tile=REF_TILE, f0_band=F0_BAND, n_boot=50):
    """dc: output of reconcile_dc (needs edges, Qp).  Returns the field dict (H is 3 × nE, u→v frame)."""
    from joblib import Parallel, delayed
    tf0 = G.graph["tile_f0_piv"]; f_ref = tf0[ref_tile]
    band = {t for t, f in tf0.items() if abs(f / f_ref - 1) <= f0_band}
    # per-tile heart-rate status (graph['tile_f0_status'], set after review): tiles with no detectable pulsatility carry
    # an assumed rate only — never used for harmonics
    _st = G.graph.get("tile_f0_status") or {}
    band = {t for t in band if (_st.get(t) or {}).get("status") not in ("no_signal", "excluded", "no measurements")}
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

    # Tile time gauge ψ (phase synchronisation over vessels seen in two tiles).  Every pair of valid measurements of the
    # same vessel contributes u_a·conj(u_b) (unit H1 phasors) weighted by ρ_a·ρ_b, so weakly pulsatile tiles still link
    # through many vessels (21S tiles 16–18: individual ρ ~0.01, yet 17↔16 over 228 vessels is coherent at p ~ 1e-23).
    # Solved by per-tile normalised fixed-point iteration (generalised power method) from the spectral start: deterministic,
    # and a weakly linked group's phase is set by its own consistent links rather than by a near-degenerate eigenvector.
    # History: the first Option-A version (2026-10-06) kept only pairs with both ρ ≥ 0.1, unweighted; that discarded most
    # vessels linking the two halves of the 21S yolk sac and put the second half ~65° off the earlier tile_normalize
    # gauge (its "median 5°" check hid the 90th percentile).  Using all pairs, weighted, agrees with tile_normalize (2–5°).
    tiles = sorted(band); ti = {t: i for i, t in enumerate(tiles)}; nT = len(tiles)
    pa, pb, pw, pz, pv, pairs = [], [], [], [], [], []
    for vi, (e, d) in enumerate(meas.items()):
        ok = [t for t, x in d.items() if t in ti and abs(x["H"][0]) > 0 and np.isfinite(x["rho"])]
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]; ua = d[a]["H"][0] / abs(d[a]["H"][0]); ub = d[b]["H"][0] / abs(d[b]["H"][0])
                pa.append(ti[a]); pb.append(ti[b]); pw.append(max(d[a]["rho"], 0.0) * max(d[b]["rho"], 0.0))
                pz.append(ua * np.conj(ub)); pv.append(vi)
                if d[a]["rho"] >= RHO_SYNC and d[b]["rho"] >= RHO_SYNC:
                    pairs.append((e, a, b))
    pa, pb, pw, pz, pv = map(np.asarray, (pa, pb, pw, pz, pv))
    psi_vec = _sync(pa, pb, pw, pz, nT, ti[ref_tile])
    psi = {t: float(psi_vec[ti[t]]) for t in tiles}
    # per-tile gauge uncertainty: bootstrap over vessels (a vessel's pairs are resampled together)
    rng = np.random.default_rng(0); nv = int(pv.max()) + 1 if len(pv) else 0; boots = []
    for _ in range(n_boot if nv else 0):
        cnt = np.bincount(rng.integers(0, nv, nv), minlength=nv)
        boots.append(_sync(pa, pb, pw * cnt[pv], pz, nT, ti[ref_tile], start=psi_vec))
    if boots:
        dev = np.exp(1j * (np.array(boots) - psi_vec[None, :]))
        sd = np.degrees(np.sqrt(-2 * np.log(np.clip(np.abs(dev.mean(axis=0)), 1e-12, 1))))   # circular SD
    else:
        sd = np.full(nT, np.nan)
    gauge_sd_deg = {t: float(sd[ti[t]]) for t in tiles}

    Qp = dc["Qp"]; nE = len(sub); Hf = np.full((3, nE), np.nan + 0j); win_t = np.full(nE, -1); elig = np.zeros(nE, bool)
    sc_w = np.full(nE, np.nan); rho_w = np.full(nE, np.nan)
    for e, d in meas.items():
        i = eix[e]
        cand = [(t, x) for t, x in d.items() if abs(x["a0"]) > Q_DET and x["rho"] >= RHO_MIN and np.sign(x["a0"]) == np.sign(Qp[i])]
        if not cand or abs(Qp[i]) <= Q_DET:     # reconciled flow ~0 (e.g. forced to zero by conservation): Option-A scale vanishes
            continue
        if not cand:
            continue
        # winner: smallest per-beat H1 scatter; recordings with < 3 complete beats (slow early hearts, 15S) have no scatter,
        # so they rank after any candidate that has one, by harmonic coherence ρ
        t, x = min(cand, key=lambda c: (0, c[1]["scatter"]) if np.isfinite(c[1]["scatter"]) else (1, -c[1]["rho"]))
        Hf[:, i] = abs(Qp[i]) / abs(x["a0"]) * x["H"] * np.exp(-1j * np.arange(1, 4) * psi[t]); win_t[i] = t; elig[i] = True
        sc_w[i] = x["scatter"]; rho_w[i] = x["rho"]
    return dict(edges=sub, H=Hf, elig=elig, tile=win_t, psi=psi, scatter=sc_w, rho=rho_w, f_ref=f_ref, band=sorted(band),
                Qp=Qp, gauge_sd_deg=gauge_sd_deg, _meas=meas, _pairs=pairs)


def sync_mismatch(field, aligned=True):
    """Median cross-tile H1 phase mismatch (degrees) over the synchronisation pairs."""
    meas, psi, out = field["_meas"], field["psi"], []
    for e, a, b in field["_pairs"]:
        ha, hb = meas[e][a]["H"][0], meas[e][b]["H"][0]
        if aligned:
            ha, hb = ha * np.exp(-1j * psi[a]), hb * np.exp(-1j * psi[b])
        out.append(abs(np.angle(ha * np.conj(hb))))
    return float(np.degrees(np.median(out)))


def junction_residuals(field, dom, seed=0):
    """Median junction residual |Σ s Ĥ_k| / Σ |Ĥ_k| at interior junctions (degree ≥ 3) whose edges are all AC-eligible.

    Returns (n_junctions, {k: {'aligned', 'unaligned', 'random'}}) for k = 1, 2: the tile-aligned field, the same
    phasors with the tile gauge undone, and a random-phase null with the aligned magnitudes."""
    el, Hf, nE = field["elig"], field["H"], len(field["edges"])
    B = dom["B"].tocsr(); absB = abs(B); interior = np.setdiff1d(np.arange(B.shape[0]), dom["bnd"])
    n_bad = np.asarray(absB @ (~el).astype(float)).ravel(); deg = np.asarray(absB.sum(axis=1)).ravel()
    nodes_ok = interior[(n_bad[interior] == 0) & (deg[interior] >= 3)]; rng = np.random.default_rng(seed)
    r = lambda H: float(np.median(np.abs((B @ H)[nodes_ok]) / np.maximum((absB @ np.abs(H))[nodes_ok], 1e-30)))
    out = {}
    for k in (1, 2):
        Hk = np.where(el, Hf[k - 1], 0)
        Hun = np.where(el, np.array([Hf[k - 1, i] * np.exp(1j * k * field["psi"][field["tile"][i]]) if el[i] else 0
                                     for i in range(nE)]), 0)
        Hrnd = np.where(el, np.abs(Hk) * np.exp(1j * rng.uniform(0, 2 * np.pi, nE)), 0)
        out[k] = dict(aligned=r(Hk), unaligned=r(Hun), random=r(Hrnd))
    return len(nodes_ok), out


def phase_determined(field, max_sd_deg=30.0):
    """AC-eligible vessels whose winner tile has a well-determined time gauge (bootstrap SD ≤ max_sd_deg); only these
    phases are comparable across tiles."""
    gsd = field.get("gauge_sd_deg") or {}
    return field["elig"] & np.array([t >= 0 and gsd.get(int(t), np.inf) <= max_sd_deg for t in field["tile"]])
