"""Option-A AC field on the carved network, from the measured graph and the DC reconciliation.

1. Per Kloosterman-valid measurement in the f0 band: a0, H1..H3 (LSQ at the tile f0; Q_t signed along the
   oriented path u→v), per-beat H1 scatter, coherence ρ, waveform repeatability (harmonics.beat_uncertainty).
2. Tile time gauge ψ_t: angular synchronisation of H1 over edges measured in ≥ 2 in-band tiles (ρ ≥ 0.1);
   leading eigenvector of Σ_e u_a ū_b (u = unit H1), ψ_ref = 0.  Applied coherently: H_k → e^{−ikψ_t} H_k.
3. AC winner per carved edge: |a0| > Q_DET, ρ ≥ RHO_MIN, sign(a0) = sign(Q'); smallest per-beat H1 scatter.
4. Ĥ_k = (|Q'_e| / |a0|) e^{−ikψ} H_k,raw   (u→v frame).
Per winner also stored: a0 (its own mean, for the measurement-supported cut |Q'|/|a0|) and the per-beat uncertainties
var_phi (2 × nE: H1, H2 phase, rad²) and var_logpi (log10 pulsatility) used to weight the AC likelihood.
"""
from __future__ import annotations
import numpy as np
from ..reconcile.harmonics import phasors, coherence, beat_scatter, beat_uncertainty

REF_TILE, F0_BAND, RHO_MIN, Q_DET, RHO_SYNC = 21, 0.15, 0.02, 1e-4, 0.1
P_DETECT, VAR_PHI_FLOOR = 0.01, np.radians(3.0) ** 2
# sync_weight="kappa": every pair is a von Mises observation of ψa − ψb with concentration κ(w), w = ρaρb, κ(w) = κmax·w/(w + w0);
# fitted by von Mises ML on residuals of well-constrained 21S tiles (sandbox/sync_with_continuity.py, 2026-10-23)
# (overlap curve refitted with the flow-direction consistency rule below: κmax 51.4, w0 0.0744; seam unchanged 3.4, 0.0089)
KAPPA_OV, KAPPA_SEAM = (51.4, 0.0744), (3.4, 0.0089)      # sync_weight="detect": F-test level; phase-variance floor (rad²)


def detect_p(rho, n, K=3):
    """p-value of the K-harmonic fit against broadband noise: ρ is the explained-variance fraction of n samples (2K regressors)."""
    from scipy.stats import f as _F
    r = min(max(float(rho), 1e-12), 1 - 1e-12)
    return float(_F.sf((r / (2 * K)) / ((1 - r) / (n - 2 * K - 1)), 2 * K, n - 2 * K - 1)) if n > 2 * K + 1 else 1.0


def _measure(q, f0):
    q = np.asarray(q, float); a0, H = phasors(q, f0)
    return a0, H, beat_scatter(q, f0), coherence(q, f0), beat_uncertainty(q, f0), int(np.isfinite(q).sum())


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


def _sync_objective(pa, pb, pw, pz, psi):
    """Σ w·Re(z_ab e^{i(ψ_b − ψ_a)}): the synchronisation objective (larger = more consistent)."""
    return float(np.sum(pw * np.real(pz * np.exp(1j * (psi[pb] - psi[pa])))))


def sync_diagnostics(pa, pb, pw, pz, nT, ref_i, psi, n_starts=50, step_deg=5.0, seed=0):
    """Is the tile gauge a unique optimum?  (1) multi-start: the solve from n_starts random phases; (2) the weakest cut of the
    tile graph (sign of the Fiedler vector of the weighted Laplacian, weights Σ ρ_aρ_b per tile pair): rotate the side not
    holding the reference tile by θ (0–360°), re-optimise locally, record the objective.  Returns a dict with the objective
    of `psi`, the restart objectives and their circular distance to `psi` (median over tiles), the best restart solution,
    the cut (block tiles, cut weight / total weight) and the rotation profile (θ, objective after refinement)."""
    rng = np.random.default_rng(seed); f0 = _sync_objective(pa, pb, pw, pz, psi); rs = []
    for _ in range(n_starts):
        p_ = _sync(pa, pb, pw, pz, nT, ref_i, start=rng.uniform(-np.pi, np.pi, nT))
        dist = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (p_ - psi)))))))
        rs.append((_sync_objective(pa, pb, pw, pz, p_), dist, p_))
    best = max(rs, key=lambda r: r[0])
    A = np.zeros((nT, nT)); np.add.at(A, (pa, pb), pw); A = A + A.T; deg = A.sum(1)
    weak = deg < 0.05 * np.median(deg)                   # near-unconstrained tiles: reported, left out of the cut
    keep = np.flatnonzero(~weak); Ak = A[np.ix_(keep, keep)]; dk = Ak.sum(1)
    Ln = np.eye(len(keep)) - Ak / np.sqrt(np.outer(dk, dk))   # normalised Laplacian: the cut balances link weight, not tile count
    _, V = np.linalg.eigh(Ln); side = np.zeros(nT, bool); side[keep[V[:, 1] / np.sqrt(dk) > 0]] = True
    if side[ref_i]:
        side[keep] = ~side[keep]
    cut = float(A[np.ix_(side, ~side)].sum() / max(A.sum() / 2, 1e-30)); prof = []
    for th in np.arange(0, 360, step_deg):
        p0 = psi + np.radians(th) * side; p_ = _sync(pa, pb, pw, pz, nT, ref_i, start=p0, n_iter=400)
        prof.append((float(th), _sync_objective(pa, pb, pw, pz, p_), float(np.degrees(np.angle(np.exp(1j * np.median(p_[side] - psi[side])))))))
    rigid = [(float(th), _sync_objective(pa, pb, pw, pz, psi + np.radians(th) * side)) for th in np.arange(0, 360, step_deg)]
    return dict(objective=f0, restarts=[(r[0], r[1]) for r in rs], best=best[2], best_objective=best[0], block=np.flatnonzero(side),
                cut_fraction=cut, rotation=prof, rigid_rotation=rigid, weak=np.flatnonzero(weak))


#: The synchronisation used for the paper (frozen 2026-10-23, results/ac_field_v7): von Mises pair weights with fitted
#: precision curves κ(ρ_a ρ_b) for repeat-vessel and plexus seam-continuity pairs, repeat pairs required to agree on the
#: mean-flow direction, and the dorsal-aorta stubs (outside the carve) as an anchor (ρ ≥ 0.5 measurements, weight ×100).
#: Pass as ``build_ac_field(G, dc, sync_extra_edges=da_stub_edges(G), **CANONICAL_SYNC)``.
CANONICAL_SYNC = dict(sync_weight="kappa", cont_lambda=1.0, sign_check=True, sync_extra_min_rho=0.5, sync_extra_weight=100.0)
CANONICAL_FIELD = "ac_field_v7"     #: results/<name>.pkl written with CANONICAL_SYNC
CANONICAL_F0_BAND = 0.21             #: heart-rate band of the paper (± fraction of the reference tile; Fig 8 "early" = below it)


def da_stub_edges(G):
    """Graph edges at the dorsal-aorta nodes that lie outside the carved network (they enter the synchronisation only)."""
    from ..reconcile import carved_domain
    dom = carved_domain(G); carved = set(dom["edges"])
    e = [x if G.has_edge(*x) else x[::-1] for n in dom["DA"] for x in G.edges(n)]
    return [x for x in e if x not in carved and x[::-1] not in carved]


def build_ac_field(G, dc, n_jobs=10, ref_tile=REF_TILE, f0_band=F0_BAND, n_boot=50, sync_extra_edges=None, stitch_lambda=0.0,
                   sync_extra_min_rho=0.0, sync_extra_weight=1.0, sync_weight="rho", cont_lambda=0.0, cont_radius_um=150.0,
                   cont_radius_pct=60.0, um_per_px=1.7, sign_check=False):
    """dc: output of reconcile_dc (needs edges, Qp).  Returns the field dict (H is 3 × nE, u→v frame).

    sync_extra_edges: edges outside the carved domain whose repeated measurements also constrain the tile gauge (e.g. the
    dorsal-aorta stubs, seen in several tiles); they enter the synchronisation only, never the field.  sync_extra_min_rho drops their
    measurements below that coherence; sync_extra_weight multiplies their pair weights (≫ 1 = an anchor: the same vessel must
    agree across tiles, e.g. the dorsal aorta seen in tiles 14 / 21 / 28).
    stitch_lambda: weight of a spatial-continuity term (phase-field stitching; 0 = off).  Adjacent carved vessels (sharing a
    junction) measured in different tiles add a constraint that their H1 phases agree in each vessel's own flow direction
    (× sgn Q'), weighted λ·ρ_a·ρ_b.  Within-tile pairs do not involve ψ, so in effect this links tiles across their borders;
    it orients blocks that shared vessels connect only weakly.  Bias ≈ the physiological phase step between neighbours."""
    from joblib import Parallel, delayed
    tf0 = G.graph["tile_f0_piv"]; f_ref = tf0[ref_tile]
    band = {t for t, f in tf0.items() if abs(f / f_ref - 1) <= f0_band}
    # per-tile heart-rate status (graph['tile_f0_status'], set after review): tiles with no detectable pulsatility carry
    # an assumed rate only — never used for harmonics
    _st = G.graph.get("tile_f0_status") or {}
    band = {t for t in band if (_st.get(t) or {}).get("status") not in ("no_signal", "excluded", "no measurements")}
    sub = dc["edges"]; eix = {e: i for i, e in enumerate(sub)}
    items = []
    for e in list(sub) + [e for e in (sync_extra_edges or []) if e not in eix]:
        for m in G.edges[e].get("measurements_piv") or []:
            if (m.get("quality") or {}).get("kl_ok") and m["tile_id"] in band and m.get("Q_t") is not None:
                items.append((e, m["tile_id"], m["Q_t"]))
    res = Parallel(n_jobs=n_jobs)(delayed(_measure)(q, tf0[t]) for e, t, q in items)
    meas = {}
    for (e, t, _), (a0, H, sc, rho, bu, n) in zip(items, res):
        meas.setdefault(e, {})[t] = dict(a0=a0, H=H, scatter=sc, rho=rho, var_phi=(bu["var_phi1"], bu["var_phi2"]), var_logpi=bu["var_logpi"],
                                         n=n, p_det=detect_p(rho, n) if np.isfinite(rho) else 1.0)
    # sync_weight: "rho" — w = ρ_a ρ_b over all pairs (canonical); "detect" — only pairs whose fundamental is detected above
    # broadband noise in both recordings (F-test p < P_DETECT), w = 1 / (var_φa + var_φb) from the beat bootstrap (floored)
    def _vphi(x):
        v = x["var_phi"][0]
        if v is None or not np.isfinite(v):                        # < 3 complete beats: white-noise phase variance ≈ 1 / (2 F)
            r = min(max(x["rho"], 1e-12), 1 - 1e-12); v = 1.0 / (2.0 * max((r / 6) / ((1 - r) / max(x["n"] - 7, 1)), 1e-9))
        return max(v, VAR_PHI_FLOOR)
    def _w(xa, xb):
        if sync_weight == "kappa":
            w_ = max(xa["rho"], 0.0) * max(xb["rho"], 0.0); return KAPPA_OV[0] * w_ / (w_ + KAPPA_OV[1])
        if sync_weight == "detect":
            return 1.0 / (_vphi(xa) + _vphi(xb)) if xa["p_det"] < P_DETECT and xb["p_det"] < P_DETECT else 0.0
        return max(xa["rho"], 0.0) * max(xb["rho"], 0.0)

    # Tile time gauge ψ (phase synchronisation over vessels seen in two tiles).  Every pair of valid measurements of the
    # same vessel contributes u_a·conj(u_b) (unit H1 phasors) weighted by ρ_a·ρ_b, so weakly pulsatile tiles still link
    # through many vessels (21S tiles 16–18: individual ρ ~0.01, yet 17↔16 over 228 vessels is coherent at p ~ 1e-23).
    # Solved by per-tile normalised fixed-point iteration (generalised power method) from the spectral start: deterministic,
    # and a weakly linked group's phase is set by its own consistent links rather than by a near-degenerate eigenvector.
    # History: the first Option-A version (2026-10-06) kept only pairs with both ρ ≥ 0.1, unweighted; that discarded most
    # vessels linking the two halves of the 21S yolk sac and put the second half ~65° off the earlier tile_normalize
    # gauge (its "median 5°" check hid the 90th percentile).  Using all pairs, weighted, agrees with tile_normalize (2–5°).
    tiles = sorted(band); ti = {t: i for i, t in enumerate(tiles)}; nT = len(tiles); xset = set(sync_extra_edges or []) - set(eix)
    pa, pb, pw, pz, pv, pairs = [], [], [], [], [], []
    for vi, (e, d) in enumerate(meas.items()):
        ok = [t for t, x in d.items() if t in ti and abs(x["H"][0]) > 0 and np.isfinite(x["rho"])]
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]; ua = d[a]["H"][0] / abs(d[a]["H"][0]); ub = d[b]["H"][0] / abs(d[b]["H"][0])
                if e in xset and min(d[a]["rho"], d[b]["rho"]) < sync_extra_min_rho:
                    continue
                # sign_check: a repeat measurement of one vessel must agree on its mean-flow direction; pairs that disagree are
                # mismatched or orientation-flipped (21S: 50 % disagree across the left-edge / left-block boundary vs 2 % elsewhere)
                if sign_check and np.sign(d[a]["a0"]) != np.sign(d[b]["a0"]):
                    continue
                pa.append(ti[a]); pb.append(ti[b]); pw.append(_w(d[a], d[b]) * (sync_extra_weight if e in xset else 1.0))
                pz.append(ua * np.conj(ub)); pv.append(vi)
                if d[a]["rho"] >= RHO_SYNC and d[b]["rho"] >= RHO_SYNC:
                    pairs.append((e, a, b))
    n_stitch = 0
    if stitch_lambda > 0:
        nb_ = {}
        for i_, (u, v) in enumerate(sub):
            nb_.setdefault(u, []).append(i_); nb_.setdefault(v, []).append(i_)
        sgn = np.sign(dc["Qp"]); vi0 = (max(pv) + 1) if len(pv) else 0
        for node, inc_ in nb_.items():
            for x_ in range(len(inc_)):
                for y_ in range(x_ + 1, len(inc_)):
                    e1, e2 = sub[inc_[x_]], sub[inc_[y_]]; s1, s2 = sgn[inc_[x_]], sgn[inc_[y_]]
                    if e1 not in meas or e2 not in meas or s1 == 0 or s2 == 0:
                        continue
                    for a, xa in meas[e1].items():
                        for b, xb in meas[e2].items():
                            if a == b or a not in ti or b not in ti or not (np.isfinite(xa["rho"]) and np.isfinite(xb["rho"])) or abs(xa["H"][0]) == 0 or abs(xb["H"][0]) == 0:
                                continue
                            ua = s1 * xa["H"][0] / abs(xa["H"][0]); ub = s2 * xb["H"][0] / abs(xb["H"][0])
                            pa.append(ti[a]); pb.append(ti[b]); pw.append(stitch_lambda * max(xa["rho"], 0.0) * max(xb["rho"], 0.0))
                            pz.append(ua * np.conj(ub)); pv.append(vi0 + n_stitch); n_stitch += 1
    # seam continuity (cont_lambda > 0): pairs of DIFFERENT carved plexus vessels (radius ≤ cont_radius_pct percentile), fundamental
    # detected in both recordings (F-test p < P_DETECT), recorded in different tiles, midpoints within cont_radius_um; compared in the
    # reconciled flow-direction frame; weight cont_lambda·ρaρb.  Cross-validated on 84 solved tile pairs (median error 4°, 90th 11°).
    n_cont = 0
    if cont_lambda > 0:
        from scipy.spatial import cKDTree
        sgn = np.sign(dc["Qp"]); rec = []
        for e, d in meas.items():
            i_ = eix.get(e)
            if i_ is None or sgn[i_] == 0:
                continue
            r_ = G.edges[e].get("radius"); pth = G.edges[e].get("path")
            if r_ is None or pth is None or len(pth) == 0:
                continue
            m_ = np.asarray(pth, float)[len(pth) // 2]
            for t, x in d.items():
                if t in ti and np.isfinite(x["rho"]) and abs(x["H"][0]) > 0 and x["p_det"] < P_DETECT:
                    rec.append((t, e, m_, float(r_), sgn[i_] * x["H"][0] / abs(x["H"][0]), x["rho"]))
        if rec:
            rcut = np.percentile([r_[3] for r_ in rec], cont_radius_pct); rec = [r_ for r_ in rec if r_[3] <= rcut]
            kd = cKDTree(np.array([r_[2] for r_ in rec])); vi0 = (int(max(pv)) + 1 if len(pv) else 0) + n_stitch
            ea, eb, ew, ez, ev = [], [], [], [], []
            for i_, j_ in kd.query_pairs(cont_radius_um / um_per_px):
                ra, rb = rec[i_], rec[j_]
                if ra[0] == rb[0] or ra[1] == rb[1]:
                    continue
                w_ = ra[5] * rb[5]; ea.append(ti[ra[0]]); eb.append(ti[rb[0]])
                ew.append(cont_lambda * (KAPPA_SEAM[0] * w_ / (w_ + KAPPA_SEAM[1]) if sync_weight == "kappa" else w_)); ez.append(ra[4] * np.conj(rb[4])); ev.append(vi0 + n_cont); n_cont += 1
            pa = np.r_[np.asarray(pa), ea]; pb = np.r_[np.asarray(pb), eb]; pw = np.r_[np.asarray(pw), ew]; pz = np.r_[np.asarray(pz), ez]; pv = np.r_[np.asarray(pv), ev]
    pa, pb, pw, pz, pv = map(np.asarray, (pa, pb, pw, pz, pv))
    pa, pb, pv = pa.astype(int), pb.astype(int), pv.astype(int)
    psi_vec = _sync(pa, pb, pw, pz, nT, ti[ref_tile])
    diag = sync_diagnostics(pa, pb, pw, pz, nT, ti[ref_tile], psi_vec)
    if diag["best_objective"] > diag["objective"] + 1e-9:          # a restart found a better optimum: use it
        psi_vec = diag["best"]
    diag["block_tiles"] = [tiles[i] for i in diag["block"]]; diag["weak_tiles"] = [tiles[i] for i in diag["weak"]]
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
    sc_w = np.full(nE, np.nan); rho_w = np.full(nE, np.nan); a0_w = np.full(nE, np.nan)
    vphi_w = np.full((2, nE), np.nan); vpi_w = np.full(nE, np.nan)
    for e, d in meas.items():
        if e not in eix:                     # synchronisation-only edge
            continue
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
        sc_w[i] = x["scatter"]; rho_w[i] = x["rho"]; a0_w[i] = x["a0"]; vphi_w[:, i] = x["var_phi"]; vpi_w[i] = x["var_logpi"]
    return dict(edges=sub, H=Hf, elig=elig, tile=win_t, psi=psi, scatter=sc_w, rho=rho_w, a0=a0_w, var_phi=vphi_w,
                var_logpi=vpi_w, f_ref=f_ref, band=sorted(band), ref_tile=ref_tile, Qp=Qp, gauge_sd_deg=gauge_sd_deg,
                _meas=meas, _pairs=pairs, _sync_diag=diag, n_stitch=n_stitch, stitch_lambda=stitch_lambda, sync_weight=sync_weight, n_cont=n_cont, cont_lambda=cont_lambda)


def supported(field, ratio_max=5.0):
    """AC-eligible vessels whose waveform is not promoted to a reconciled flow > ratio_max × its own measured mean
    (|Q'|/|a0| ≤ ratio_max): the measurement-supported set the mechanics are fitted on.  Conservation-dominated vessels
    (e.g. the vein stubs, whose DC is under-read) keep their waveform for boundary forcing but are not fitted."""
    a0 = field["a0"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return field["elig"] & (np.abs(field["Qp"]) / np.abs(a0) <= ratio_max)


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
