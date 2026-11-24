"""Full DC reconciliation on the carved network.

PRODUCTION SINCE 2026-11-19 (ESTIMATOR = "fixed"): static inverse-variance fusion + constrained weighted least squares.

  a. Each recording r has the variance of its MEAN flux.  VARIANCE = "block" (production, the collaborators' recipe of
     2026-11-19 exactly): v_r from heartbeat blocks of the residual of a mean + three-cardiac-harmonic OLS fit at the
     recording's f0 (tile f0 if missing): v_b = var(block means, ddof=1)/B for 1- and 2-beat blocks (complete blocks with
     ≥ 80 % finite samples, B ≥ 3), the larger; floored at 1e-16; unavailable → median of the valid v in the fitted data.
     VARIANCE = "pooled" (alternative): v_r = κ s_r²/N_r with s_r the residual SD (floored at its 1st percentile), N_r
     frames and κ the median block/white-noise ratio (21S 2.36).  Caveat of "block": with ≈ 3 s recordings the median
     recording has 4 blocks, so single v_r scatter ≈ ×10 (21S: weights up to 1.6·10⁴× the median; held-out error +9 %
     vs "pooled").
  b. Per vessel: μ_e = Σ y_r/v_r / Σ 1/v_r,  V_e = 1/Σ 1/v_r  (independent, unbiased estimates of one mean).
  c. Q' = argmin ½ Σ_measured (Q_e − μ_e)²/V_e  s.t.  B_I Q = 0  (unmeasured vessels: weight ε → minimum-norm fill);
     optional symmetric=True adds Σ_L f = Σ_R f (total left arterial inflow = total right; sensitivity analysis).
  s_r is residual temporal variability, not a calibrated error: systematic recording differences (gain) are not modelled.

The quasi-likelihood estimator below (ESTIMATOR = "ql", production 2026-11-03 … 11-18) is kept for comparison.

QUASI-LIKELIHOOD (frozen 2026-11-12; superseded 2026-11-19):

0. Network: network.carved_domain — open-boundary regions removed with free inflow boundaries, topology-uncertain regions
   removed as mass-conserving black boxes (one conservation row per region; reconcile.regions).  Both are graph tags set
   in the workbench viewer; a stage without tags reconstructs on the whole network minus the DA anchors.

1. Per edge the 1/c²-weighted average of its profile-valid recordings with weighted-RMS reliability (vessel_input;
   before 2026-11-10 the single recording with the smallest c), Q signed u→v.  profile-fit-failed / unmeasured edges are unimaged.
2. Q' minimises the quasi-likelihood of the measured flows under exact interior conservation,

       Q' = argmin_q  Σ_imaged ℓ_k(q_k) / m  +  (ε/2) Σ_unimaged q_k²      s.t.  B_I q = 0,
       ℓ_k(q) = ∫_{y_k}^{q} (u − y_k) / (c_k² u² + σ0²) du      (closed form),

   i.e. the measurement variance c²Q′² + σ0,k² (σ0,k² = σ0² + s_between,k², the disagreement of vessel k's recordings —
   vessel_input) is evaluated at the reconciled flow (Wedderburn 1974 quasi-likelihood; data
   reconciliation in the sense of Crowe 1983).  σ0 = 0.01·median|Q_m|; ε = EPS_UNIMAGED is a weak ridge that fixes
   circulations conservation cannot determine (the structural null space ker[B_I; M]: loops of unimaged vessels); m =
   median 1/V at the start makes ε relative to a typical measured weight.  The objective is not convex (ℓ is concave
   beyond q ≈ 2y), so the start is part of the estimator: the uniform-weight conservative projection of the data.
   Solver: projected Newton — each step is one weighted conservative projection with a positive diagonal model of the
   Hessian, followed by an Armijo backtracking line search on the objective — so the objective decreases monotonically
   and the iteration stops at a constrained stationary point (no limit cycle, no averaging).
   Sensitivity: n_starts further solves from perturbed starts (data × (1 + 0.3 N(0, 1)), fixed seeds); Qp_sd = spread;
   `unstable` = a vessel whose direction or magnitude (> stable_rtol) depends on the start.
3. Emergent boundary f = B Q'; passive comparison with K = r⁴/ℓ.  ``radius`` picks r (also handed to the AC model):
   "tube" (default since 2026-11-17) = the uncalibrated tube-interior segmentation radius (reconcile/radius.tube_radii on
   <stage>/mask/mask.tif): median walled half-width for resistance, volume-equivalent √⟨a²⟩ for compliance (returned as
   ``radius_compliance`` for the AC model); "consensus" = √(R_graph · median R_profile) (sensitivity analysis; default before
   2026-11-17); "anatomical" = the calibrated segmentation radius (results/anatomical_radius.pkl); or an array (used for
   both).  The flux estimator is not affected: it keeps the free-radius profile fit.

Before 2026-11-03 step 2 was a damped iteratively re-weighted projection whose iterates settle into a small limit cycle
(21S, 27S); Q' was the average of the last 100 iterates over 8 perturbed runs.  Both estimators have the same
stationary equations ((q − y)/V(q) is the IRLS correction term) and validate identically against independent repeat
recordings; the explicit objective replaced the cycle average.
"""
from __future__ import annotations
import numpy as np
from .reliability import c_of
from .network import carved_domain, project, emergent, passive, metrics, contract_short_unimaged, EPS_UNIMAGED

PX_UM = 1.7


PER_VESSEL = "average"      # since 2026-11-10 (was "winner"): see vessel_input
ESTIMATOR = "fixed"         # since 2026-11-19 (was "ql"): static inverse-variance fusion + constrained weighted least squares
VARIANCE = "residual"       # since 2026-11-20 (was "block"): "residual" (s²/M from the mean + 2-harmonic fit residual) |
                            # "block" (per-recording heartbeat-block variance, the handoff recipe) | "pooled" (κ s²/N)
SYMMETRIC = True            # since 2026-11-20 (was False): total left arterial inflow = total right enforced in the reconciliation
K_FIT = 2                   # harmonics of the heart rate in the per-recording fit (2026-11-20: the third adds ~0.1 % of the power)
RESULT = "dc_reconciled_v3"  #: results/<RESULT>.pkl of the current method (v2 = block variance, free left/right split)
FS_HZ = 250.0


def block_variance(q, f0, fs=FS_HZ):
    """Variance of a recording's mean flux from heartbeat blocks of its harmonic-fit residual (1- and 2-beat blocks,
    var(block means, ddof=1)/B, the larger; ≥ 3 blocks).  Used only to estimate the pooled factor κ.  NaN if unavailable."""
    q = np.asarray(q, float); N = len(q); t = np.arange(N) / fs; ok = np.isfinite(q)
    if not np.isfinite(f0) or f0 <= 0 or f0 >= fs / 6 or ok.sum() < 20:
        return np.nan
    w = 2 * np.pi * f0 * t
    X = np.column_stack([np.ones(N)] + [g(k * w) for k in (1, 2, 3) for g in (np.sin, np.cos)])
    beta, _, rank, _ = np.linalg.lstsq(X[ok], q[ok], rcond=None)
    if rank < 7:
        return np.nan
    r = q - X @ beta; best = np.nan
    for b in (1, 2):
        width = b / f0; means = []
        for j in range(int((N / fs) // width)):
            m = (t >= j * width) & (t < (j + 1) * width); fin = m & ok
            if m.sum() and fin.sum() >= max(2, 0.8 * m.sum()):
                means.append(r[fin].mean())
        if len(means) >= 3:
            v = np.var(means, ddof=1) / len(means)
            best = v if not np.isfinite(best) else max(best, v)
    return best


def autocorrelation_factor(G, M):
    """κ = median over recordings of block_variance / (s_r²/N_r): how much residual autocorrelation inflates the variance
    of the mean relative to white noise (21S ≈ 2.4).  A common factor: it scales every V_e and leaves Q' unchanged."""
    ratios = []
    for (t, e), m in M.items():
        mm = next((x for x in G.edges[e].get("measurements_piv") or [] if int(x.get("tile_id", -1)) == t), None)
        if mm is None or mm.get("Q_t") is None:
            continue
        f0 = (mm.get("harm") or {}).get("f0_hz"); s2 = (m["cv"] * m["Q"]) ** 2 / m["n_frames"]
        vb = block_variance(mm["Q_t"], float(f0) if f0 else np.nan)
        if np.isfinite(vb) and s2 > 0:
            ratios.append(vb / s2)
    return float(np.median(ratios)) if ratios else 1.0


def recording_sd(M):
    """{key: s_r} residual temporal SD of each recording's flux (nL/s), floored at the 1st percentile over recordings."""
    s = {k: abs(m["cv"] * m["Q"]) for k, m in M.items()}
    fl = float(np.percentile([v for v in s.values() if np.isfinite(v) and v > 0], 1))
    return {k: max(v, fl) if np.isfinite(v) else np.nan for k, v in s.items()}, fl


def recording_block_variances(G, M):
    """{key: v_r} per the handoff recipe: block_variance of each recording's Q(t) at its f0 (tile f0 fallback); nonpositive /
    unavailable → median of the valid ones in M; floored at 1e-16.  Also returns the number of unavailable recordings."""
    from .gain import tile_heart_rates
    tf0 = tile_heart_rates(G); v = {}
    for (t, e), m in M.items():
        mm = next((x for x in G.edges[e].get("measurements_piv") or [] if int(x.get("tile_id", -1)) == t), None)
        f0 = (mm.get("harm") or {}).get("f0_hz") if mm else None
        f0 = float(f0) if f0 and np.isfinite(f0) else tf0.get(t, np.nan)
        v[(t, e)] = block_variance(mm["Q_t"], f0) if mm is not None and mm.get("Q_t") is not None else np.nan
    good = [x for x in v.values() if np.isfinite(x) and x > 0]
    if not good:
        raise RuntimeError("no valid recording variance")
    med = float(np.median(good)); n_bad = sum(1 for x in v.values() if not (np.isfinite(x) and x > 0))
    return {k: max(x if np.isfinite(x) and x > 0 else med, 1e-16) for k, x in v.items()}, n_bad


def residual_variance(q, f0, K=K_FIT, fs=FS_HZ):
    """Variance of a recording's fitted mean flux from the residual of its mean + K-harmonic fit at the tile heart rate:
    s² = RSS / (M − (2K + 1)) over the M finite frames, σ² = s² / M (independent residuals, complete cycles).  NaN if unavailable."""
    q = np.asarray(q, float); ok = np.isfinite(q); M = int(ok.sum())
    if not np.isfinite(f0) or f0 <= 0 or M <= 4 * K + 10:
        return np.nan
    w = 2 * np.pi * f0 * np.arange(len(q))[ok] / fs
    X = np.column_stack([np.ones(M)] + [g(k * w) for k in range(1, K + 1) for g in (np.cos, np.sin)])
    r = q[ok] - X @ np.linalg.lstsq(X, q[ok], rcond=None)[0]
    return float(r @ r) / ((M - (2 * K + 1)) * M)


def fitted_mean(q, f0, K=K_FIT, fs=FS_HZ):
    """Mean of the mean + K-harmonic least-squares fit of a recording (the Q̄fit of the Methods; differs from the plain time
    average by ~0.4 % because recordings do not hold a whole number of beats).  NaN if unavailable."""
    q = np.asarray(q, float); ok = np.isfinite(q); M = int(ok.sum())
    if not np.isfinite(f0) or f0 <= 0 or M <= 4 * K + 10:
        return np.nan
    w = 2 * np.pi * f0 * np.arange(len(q))[ok] / fs
    X = np.column_stack([np.ones(M)] + [g(k * w) for k in range(1, K + 1) for g in (np.cos, np.sin)])
    return float(np.linalg.lstsq(X, q[ok], rcond=None)[0][0])


def recording_residual_variances(G, M):
    """{key: σ²_r} = residual_variance of each recording's Q(t) at its tile heart rate; nonpositive / unavailable → median of the
    valid ones in M; floored at 1e-16.  Also returns the number of unavailable recordings."""
    from .gain import tile_heart_rates
    tf0 = tile_heart_rates(G); v = {}
    for (t, e), m in M.items():
        mm = next((x for x in G.edges[e].get("measurements_piv") or [] if int(x.get("tile_id", -1)) == t), None)
        f0 = (mm.get("harm") or {}).get("f0_hz") if mm else None
        f0 = float(f0) if f0 and np.isfinite(f0) else tf0.get(t, np.nan)
        v[(t, e)] = residual_variance(mm["Q_t"], f0) if mm is not None and mm.get("Q_t") is not None else np.nan
    good = [x for x in v.values() if np.isfinite(x) and x > 0]
    if not good:
        raise RuntimeError("no valid recording variance")
    med = float(np.median(good)); n_bad = sum(1 for x in v.values() if not (np.isfinite(x) and x > 0))
    return {k: max(x if np.isfinite(x) and x > 0 else med, 1e-16) for k, x in v.items()}, n_bad


def symmetric_domain(dom):
    """dom with one extra conservation row: total left arterial inflow = total right (Σ_L f − Σ_R f = 0)."""
    import scipy.sparse as sp
    d = dict(dom); B = d["B"]; L = d["inj"][d["side"] == "L"]; R = d["inj"][d["side"] == "R"]
    d["B"] = sp.vstack([B, sp.csr_matrix(np.asarray(B[L].sum(0) - B[R].sum(0)).ravel())]).tocsr()
    return d


def vessel_input(recs, per_vessel=PER_VESSEL):
    """One vessel's (y, c, s_between) from its recordings [(measurement, c), …].

    "average" (production since 2026-11-10): y = Σ w y_t / Σ w with w = 1/c², and the vessel's reliability is the
    weighted-RMS quality of its recordings, c = √(Σ w c² / Σ w) — the typical quality of what was averaged, with no extra
    credit for the number of recordings (the recordings are different moments, not independent samples of one flow).
    Recordings of one vessel come from tiles acquired at different times; the reconstruction targets a conservation-
    consistent reference field summarising the mosaic, so no recording time is privileged.  "winner": the smallest-c
    recording (production before 2026-11-10).
    Evidence (sandbox/agg_compare.py, 21S / 27S): held-out error of the reconciled field vs winner −0.011 [−0.022, +0.003] /
    +0.016 [−0.019, +0.052] (trusted, tile bootstrap); junction calibration RMS z 1.07 / 1.02 (ideal 1) vs 1.43 / 1.32 for
    the independent-repeats combination (Σ 1/c²)^(−1/2) and 1.30 / 1.14 for winner.  Equal weights were worse (27S +0.07);
    1/(c² + τ²) weights and adding the RELATIVE spread to c gave no gain.
    s_between = √(Σ α (y_t − y)²), α = w/Σw: the disagreement between the recordings, an absolute flux scale.  It enters the
    reconciliation variance as a per-vessel floor, σ² = c²Q′² + σ0² + s_between² (every multiply recorded vessel; 0 for one
    recording).  Not calibrated on junctions — a regularisation scale for incompatible observations: without it a vessel
    whose recordings disagree in sign averages to ≈ 0 and gets 4–5× the median weight (21S / 27S: 4.9× / 3.8× → 0.9× / 0.5×);
    held-out error unchanged (21S all −0.021 [−0.045, 0.000], 27S +0.010 [−0.026, +0.049] trusted vs winner)."""
    if per_vessel == "winner" or len(recs) == 1:
        m, c = min(recs, key=lambda x: x[1]); return float(m["Q"]), float(c), 0.0
    assert per_vessel == "average", per_vessel
    cs = np.array([c for _, c in recs]); a = (1 / cs ** 2) / np.sum(1 / cs ** 2); q = np.array([m["Q"] for m, _ in recs]); y = float(a @ q)
    return y, float(np.sqrt(a @ cs ** 2)), float(np.sqrt(a @ (q - y) ** 2))


def repeat_folds(M, n_fold=5, seed=0):
    """Independent references for averaged inputs: per fold, one random recording of every vessel recorded in ≥ 2 tiles is
    held out.  Yields (M without the held recordings, [(edge, held measurement)]).  With per_vessel="average" every recording
    enters the input, so a vessel's other recordings are no longer independent of its reconstructed flow."""
    by = {}
    for k in M:
        by.setdefault(k[1], []).append(k)
    rng = np.random.default_rng(seed)
    for _ in range(n_fold):
        held = {e: ks[int(rng.integers(len(ks)))] for e, ks in by.items() if len(ks) >= 2}
        hs = set(held.values())
        yield {k: m for k, m in M.items() if k not in hs}, [(e, M[k]) for e, k in held.items()]


def anatomical_radius_array(edges, path=None):
    """Calibrated anatomical radius (per-vessel median of R(s)) in mosaic px for ``edges``, from scripts/anatomical_radius.py."""
    import pickle
    from pathlib import Path
    from .. import paths
    A = pickle.loads(Path(path or paths.RESULTS / "anatomical_radius.pkl").read_bytes()); idx = {e: i for i, e in enumerate(A["edges"])}
    return np.array([A["median"][idx[e] if e in idx else idx[e[::-1]]] for e in edges], float)


_TUBE_CACHE = {}


def tube_radius_arrays(G, edges, mask=None):
    """(resistance, compliance, profiled) tube-interior radii for ``edges``; mask defaults to <stage>/mask/mask.tif.
    Cached per (graph object, edge list, mask) so repeated reconciliations (withholding folds) profile once."""
    from .radius import tube_radii
    key = (id(G), G.number_of_edges(), tuple(edges), None if mask is None else id(mask))
    if key not in _TUBE_CACHE:
        if mask is None:
            import tifffile
            from .. import paths
            mask = tifffile.imread(str(paths.STAGE / "mask" / "mask.tif")) > 127
        T = tube_radii(G, edges, mask); _TUBE_CACHE.clear(); _TUBE_CACHE[key] = T
    T = _TUBE_CACHE[key]
    return T["resistance"], T["compliance"], T["profiled"]


def ql_objective(y, C, imaged, s0, m, eps=EPS_UNIMAGED):
    """Objective, gradient and (positive model, exact) diagonal Hessian of the DC quasi-likelihood (see module doc)."""
    c2 = C ** 2
    F = lambda u: np.log(c2 * u ** 2 + s0 ** 2) / (2 * c2) - y * np.arctan(C * u / s0) / (C * s0)
    Fy = F(y)

    def L(Q):
        return float(np.sum((F(Q) - Fy)[imaged]) / m + 0.5 * eps * np.sum(Q[~imaged] ** 2))

    def grad(Q):
        return np.where(imaged, (Q - y) / (c2 * Q ** 2 + s0 ** 2) / m, eps * Q)

    def hess(Q):
        V = c2 * Q ** 2 + s0 ** 2; h = (s0 ** 2 - c2 * Q ** 2 + 2 * c2 * Q * y) / V ** 2
        return np.where(imaged, np.maximum(h, 0.5 / V) / m, eps), np.where(imaged, h / m, eps)
    return L, grad, hess


def ql_solve(Q, y, C, imaged, s0, m, dom, tol=1e-10, max_iter=500):
    """Projected Newton with Armijo backtracking from the conservative start Q → (Q', info)."""
    L, grad, hess = ql_objective(y, C, imaged, s0, m); f = L(Q); converged = False; it = 0
    for it in range(1, max_iter + 1):
        g = grad(Q); G, _ = hess(Q); d = project(Q - g / G, G, dom) - Q; slope = float(g @ d)
        if slope >= 0:                                       # no descent direction left: stationary to machine precision
            converged = True; break
        a = 1.0
        while L(Q + a * d) > f + 1e-4 * a * slope and a > 1e-10:
            a *= 0.5
        Q = Q + a * d; fn = L(Q); rel = np.linalg.norm(a * d) / np.linalg.norm(Q)
        if rel < tol and f - fn <= 1e-14 * abs(f):
            f = fn; converged = True; break
        f = fn
    _, h = hess(Q)
    return Q, dict(objective=f, n_iter=it, converged=converged, n_concave=int(np.sum(imaged & (h < 0))), m=m)


def reconcile_dc(G, model, M, dom=None, topology=True, tol=1e-10, max_iter=500, n_shuffle=10, seed=0, n_starts=4, start_noise=0.3, stable_rtol=0.10,
                 radius="tube", per_vessel=PER_VESSEL, mask=None, estimator=None, symmetric=None, kappa=None, variance=None):
    """estimator: "fixed" (production since 2026-11-19; module docstring a–c) or "ql" (quasi-likelihood, before).
    symmetric: add total left = total right arterial inflow (None → SYMMETRIC).  variance: "residual" | "block" | "pooled"
    (None → VARIANCE).  kappa: pooled autocorrelation factor (None → estimated)."""
    estimator = estimator or ESTIMATOR; symmetric = SYMMETRIC if symmetric is None else bool(symmetric)
    dom = dom or carved_domain(G, topology=topology)
    recs = {}
    for (t, e), m in M.items():
        recs.setdefault(e, []).append((m, c_of(model, m)))
    sub = dom["edges"]; me = len(sub)
    Qm = np.zeros(me); C = np.full(me, np.nan); imaged = np.zeros(me, bool); tile_w = np.full(me, -1); n_rec = np.zeros(me, int); sb = np.zeros(me)
    for i, e in enumerate(sub):
        if e in recs:
            Qm[i], C[i], sb[i] = vessel_input(recs[e], per_vessel); imaged[i] = True; n_rec[i] = len(recs[e])
            tile_w[i] = min(recs[e], key=lambda x: x[1])[0]["tile"]                  # best recording's tile (labels / figures)
    s0 = 0.01 * np.median(np.abs(Qm[imaged])); Cw = np.where(imaged, C, 1.0)
    s0v = np.sqrt(s0 ** 2 + sb ** 2)                                                # per-vessel floor: σ0 + between-recording spread
    dom_fit = symmetric_domain(dom) if symmetric else dom
    fixed = None
    if estimator == "fixed":
        variance = variance or VARIANCE
        if variance == "residual":
            vr, n_bad = recording_residual_variances(G, M); kap, s_floor, sr = np.nan, np.nan, None
        elif variance == "block":
            vr, n_bad = recording_block_variances(G, M); kap, s_floor, sr = np.nan, np.nan, None
        else:
            assert variance == "pooled", variance
            sr, s_floor = recording_sd(M); n_bad = 0
            kap = autocorrelation_factor(G, M) if kappa is None else float(kappa)
            vr = {k: kap * sr[k] ** 2 / M[k]["n_frames"] for k in M}
        mu = np.zeros(me); Ve = np.full(me, np.nan); nrec = np.zeros(me, int)
        sgn = {}; qfit = {}
        if variance == "residual":                         # the recording's value is the mean of the same fit that gives its variance
            from .gain import tile_heart_rates
            tf0_ = tile_heart_rates(G)
            for (t, e), mrow in M.items():
                mm = next((x for x in G.edges[e].get("measurements_piv") or [] if int(x.get("tile_id", -1)) == t), None)
                f0_ = (mm.get("harm") or {}).get("f0_hz") if mm else None; f0_ = float(f0_) if f0_ and np.isfinite(f0_) else tf0_.get(t, np.nan)
                v_ = fitted_mean(mm["Q_t"], f0_) if mm is not None and mm.get("Q_t") is not None else np.nan
                if np.isfinite(v_) and np.sign(v_) == np.sign(mrow["Q"]):
                    qfit[(t, e)] = v_
        for (t, e), mrow in M.items():
            sgn.setdefault(e, []).append((qfit.get((t, e), mrow["Q"]), vr[(t, e)]))
        for i, e in enumerate(sub):
            r = sgn.get(e)
            if r:
                v = np.array([b for _, b in r]); y = np.array([a_ for a_, _ in r]); w = 1 / v
                mu[i] = w @ y / w.sum(); Ve[i] = 1 / w.sum(); nrec[i] = len(r)
        Qm = np.where(imaged, mu, Qm)                                               # vessel input = inverse-variance mean
        W = np.where(imaged, 1 / np.where(imaged, Ve, 1.0), 0.0); wm = float(np.median(W[imaged]))
        Qp = project(Qm, np.where(imaged, W / wm, EPS_UNIMAGED), dom_fit)
        obj = float(0.5 * np.sum((Qp[imaged] - Qm[imaged]) ** 2 / Ve[imaged]))
        info = dict(n_iter=1, converged=True, objective=obj, n_concave=0); m = wm
        Qp_sd = np.zeros(me); unstable = np.zeros(me, bool)
        fixed = dict(variance=variance, kappa=kap, s_floor=s_floor, V=Ve, v_rec=vr, n_variance_unavailable=n_bad, n_fitted_mean=len(qfit))
    else:
        assert estimator == "ql", estimator
        start = lambda q: project(q, np.where(imaged, 1.0, EPS_UNIMAGED), dom_fit)  # uniform-weight conservative projection
        Q0 = start(Qm); m = float(np.median(1 / (Cw[imaged] ** 2 * Q0[imaged] ** 2 + s0 ** 2)))
        Qp, info = ql_solve(Q0, Qm, Cw, imaged, s0v, m, dom_fit, tol=tol, max_iter=max_iter)
        rng = np.random.default_rng(seed); runs = [Qp]
        for k in range(n_starts):
            runs.append(ql_solve(start(Qm * (1 + start_noise * rng.standard_normal(me))), Qm, Cw, imaged, s0v, m, dom_fit, tol=tol, max_iter=max_iter)[0])
        E = np.array(runs); Qp_sd = E.std(axis=0) if n_starts else np.zeros(me)
        unstable = ((np.any(E > 0, axis=0) & np.any(E < 0, axis=0)) | ((E.max(0) - E.min(0)) > stable_rtol * np.maximum(np.abs(Qp), 1e-12))) \
            if n_starts else np.zeros(me, bool)
    n_it, converged = info["n_iter"], info["converged"]
    fb = emergent(Qp, dom)
    f_all = dom["B"] @ Qp; nix = {n: k for k, n in enumerate(dom["nodes"])}
    topo_regions = [dict(n_edges=r["n_edges"], cut=r["cut"], leaky=r["leaky"],
                         net=float(sum(f_all[nix[n]] for n in r["cut"])), throughput=float(0.5 * sum(abs(f_all[nix[n]]) for n in r["cut"])))
                    for r in dom.get("regions", [])]
    rg = np.array([G.edges[e]["radius"] for e in sub], float)
    def _median_rkl(e):
        r = [m["R_profile_um"] / PX_UM for m in G.edges[e].get("measurements_piv") or [] if m.get("R_profile_um") and np.isfinite(m["R_profile_um"])]
        return float(np.median(r)) if r else np.nan
    rt = np.array([_median_rkl(e) for e in sub])
    r_cons = np.sqrt(rg * np.where(np.isfinite(rt), rt, rg)); L = np.maximum(np.array([G.edges[e]["length"] for e in sub], float), 1.0)   # ≥ 1 px: zero-length splice artefacts
    profiled = np.zeros(me, bool)
    if isinstance(radius, str) and radius == "tube":
        r, rc, profiled = tube_radius_arrays(G, sub, mask)
    elif isinstance(radius, str) and radius == "anatomical":
        r = anatomical_radius_array(sub); rc = r
    elif isinstance(radius, str):
        assert radius == "consensus", radius; r = rc = r_cons
    else:
        r = rc = np.asarray(radius, float)
    K = r ** 4 / L; Qs = passive(K, fb, dom)
    met = metrics(Qm, Qp, Qs, fb, dom, imaged)
    con = contract_short_unimaged(dom, imaged, L, r); short_unimaged = con.pop("short_unimaged"); met["contracted"] = con
    rng = np.random.default_rng(seed)
    shuf = np.mean([metrics(Qm, Qp, passive(rng.permutation(K), fb, dom), fb, dom, imaged)["corr_lin"] for _ in range(n_shuffle)])
    return dict(model=model, edges=sub, nodes=dom["nodes"], Qm=Qm, Qp=Qp, Qs=Qs, fb=fb,
                sigma=(np.sqrt(np.where(imaged, fixed["V"], np.nan)) if fixed else np.sqrt(Cw ** 2 * Qp ** 2 + s0v ** 2)), c=C, s_between=sb, s0_vessel=s0v, imaged=imaged, tile=tile_w, radius=r, radius_compliance=rc,
                radius_profiled=profiled, length=L,
                inj=dom["inj"], out=dom["out"], side=dom["side"], metrics=met, shuffled_corr=shuf,
                n_iter=n_it, sigma0_floor=s0, dom=dom,
                estimator=("fixed inverse-variance fusion + constrained WLS (2026-11-19)" if fixed else "quasi-likelihood, projected Newton (2026-11-03)"),
                estimator_key=estimator, symmetric=bool(symmetric), fixed=fixed,
                objective=info["objective"], objective_m=m, n_concave=info["n_concave"],
                converged=converged, Qp_sd=Qp_sd, unstable=unstable, n_starts=n_starts, short_unimaged=short_unimaged,
                radius_model=radius if isinstance(radius, str) else "given", radius_consensus=r_cons, per_vessel=per_vessel, n_rec=n_rec,
                topology_regions=topo_regions, n_topology_edges=int(sum(r["n_edges"] for r in dom.get("regions", []))))
