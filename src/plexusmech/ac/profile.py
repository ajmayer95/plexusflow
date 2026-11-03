"""Profile-likelihood fit of the wall distensibility D (η optional) on the measurement-supported AC field.

Data per vessel e (measurement-supported, plexusflow.ac.field.supported): H1, H2 phase relative to the field's own mean-flow
direction and H1 pulsatility PI = |H1|/|Q̄|.  Model: the RC network (model.py) driven at the boundary by
  carve nodes   local measured ratio, corrected per group (L, R carve: amplitude per harmonic + one coherent timing shift)
  vein nodes    stub waveforms (model.vein_forcing), pinned — no vein parameters (pin_veins=True; False frees the
                L/R vein groups as in group_bc)
Latent tile gauge: φ_obs,n = φ_model,n + n·δ_t for a vessel whose winner tile is t, δ_t ~ von Mises(0, 1/σ_t²), σ_t = the
tile's bootstrap gauge SD (field.gauge_sd_deg; reference tile fixed).  One parameter per tile, so a tile-wide timing
error is not counted as many independent vessel errors; well-gauged tiles barely move.
Optional block gauge (gauge_block = set of tiles): one more offset β, flat prior on the circle, added to every tile of a
block that the synchronisation links only weakly to the rest (field._sync_diag), so the model — not an over-narrow
bootstrap prior — sets that block's orientation: φ_obs,n = φ_model,n + n·(δ_t + β·[t ∈ block]).
NLL = von Mises(H1 phase) + von Mises(H2 phase) + Student-t(log10 PI_model/PI_meas) + gauge prior, with weights
  'waveform'  per-vessel variance = per-beat repeatability (field.var_phi / var_logpi) + (τ c_e)², τ profiled per term
  'c'         κ ∝ 1/c², scale ∝ c (the DC reliability of the AC winner; group_bc's weights)
At each (D, η) the 12 boundary parameters (L-BFGS, warm-started along D) and the δ_t (exact 1-D grid per tile) are
profiled by alternation.  η = 0 by default (purely elastic wall); pass eta_grid to estimate it.
It is a composite likelihood (H1/H2 of one trace are not independent; neighbouring vessels share tiles and boundaries):
point estimates are meaningful, ΔNLL widths overstate the precision — quote D's spread across boundary choices.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import i0e
from .model import group_response
from .group_bc import groups, GROUPS
from .likelihood import nll_t, nll_vm_var, nll_t_var

D_GRID = np.logspace(np.log10(2.5e-4), np.log10(8.0e-4), 21)
_DG = np.linspace(-np.pi, np.pi, 720, endpoint=False)


def _chain(C, c, tile, gauge_sd_deg, ref_tile, var_phi, var_logpi, d_list, eta, weights, gauge, pin_veins, gauge_block=None, beta_fixed=None):
    el = C["elig"]; c = c[el]; gid = groups(C)
    Qp, Qs = C["Qp"][el], C["Qs_dc"][el]; sgnQs = np.sign(Qs)
    phd = {k: np.angle(C["H"][k][el] * np.sign(Qp)) for k in (1, 2)}; pid = np.abs(C["H"][1][el]) / np.abs(Qp)
    tl = np.asarray(tile)[el].astype(int); tiles = np.unique(tl); tix = np.searchsorted(tiles, tl); nT = len(tiles)
    sd = np.radians(np.array([0.0 if t == ref_tile else gauge_sd_deg.get(int(t), 180.0) for t in tiles]))
    kp = np.where(sd > 0, 1 / np.maximum(sd, 1e-3) ** 2, np.inf); fin = np.isfinite(kp); kp0 = np.where(fin, kp, 0.0)
    blk = np.isin(tiles, sorted(gauge_block or [])).astype(float)
    if weights == "waveform":
        vb = [np.asarray(var_phi[k])[el] for k in (0, 1)] + [np.asarray(var_logpi)[el]]
        vb = [np.where(np.isfinite(v), v, np.nanmedian(v)) for v in vb]           # < 3 beats: median repeatability
    elif weights != "c":
        raise ValueError(f"weights must be 'waveform' or 'c', not {weights!r}")
    w = 1 / c ** 2

    def phase_terms(d1, d2):
        """NLLs and per-vessel concentrations of the two phase terms."""
        if weights == "waveform":
            (n1, k1), (n2, k2) = nll_vm_var(d1, vb[0], c), nll_vm_var(d2, vb[1], c)
            return n1 + n2, k1, k2
        out = []
        for d in (d1, d2):
            f = lambda lk: np.sum(-np.exp(lk) * w * np.cos(d) + np.log(2 * np.pi * i0e(np.exp(lk) * w)) + np.exp(lk) * w)
            r = minimize_scalar(f, bounds=(-8, 6), method="bounded"); out.append((r.fun, np.exp(r.x) * w))
        return out[0][0] + out[1][0], out[0][1], out[1][1]

    def model_phases(x, B1, B2):
        q1 = B1 @ np.exp(x[:4] + 1j * x[8:]); q2 = B2 @ np.exp(x[4:8] + 2j * x[8:])
        return q1, np.angle(q1 * sgnQs), np.angle(q2 * sgnQs)

    def nll(x, B1, B2, dl, beta=0.0):
        q1, m1, m2 = model_phases(x, B1, B2); off = (dl + beta * blk)[tix]
        d1 = np.angle(np.exp(1j * (phd[1] - off - m1))); d2 = np.angle(np.exp(1j * (phd[2] - 2 * off - m2)))
        r = np.log10(np.maximum(np.abs(q1) / np.maximum(np.abs(Qs), 1e-30), 1e-12) / pid)
        pi_term = nll_t_var(r, vb[2], c) if weights == "waveform" else nll_t(r, c)
        return phase_terms(d1, d2)[0] + pi_term

    def update_delta(x, B1, B2, dl, beta=0.0):
        _, m1, m2 = model_phases(x, B1, B2)
        r1 = np.angle(np.exp(1j * (phd[1] - m1))); r2 = np.angle(np.exp(1j * (phd[2] - m2))); off = (dl + beta * blk)[tix]
        _, K1, K2 = phase_terms(np.angle(np.exp(1j * (r1 - off))), np.angle(np.exp(1j * (r2 - 2 * off))))
        S1 = np.bincount(tix, K1 * np.cos(r1), nT) + 1j * np.bincount(tix, K1 * np.sin(r1), nT)
        S2 = np.bincount(tix, K2 * np.cos(r2), nT) + 1j * np.bincount(tix, K2 * np.sin(r2), nT)
        if blk.any() and beta_fixed is None:                # block offset β: flat prior, exact grid given the δ_t
            ob = -sum(np.real(np.exp(-1j * (dl[t] + _DG)) * S1[t]) + np.real(np.exp(-2j * (dl[t] + _DG)) * S2[t]) for t in np.flatnonzero(blk))
            beta = float(_DG[np.argmin(ob)])
        sh = beta * blk
        obj = -(np.real(np.exp(-1j * (_DG[None] + sh[:, None])) * S1[:, None]) + np.real(np.exp(-2j * (_DG[None] + sh[:, None])) * S2[:, None])
                + kp0[:, None] * np.cos(_DG)[None])
        new = _DG[np.argmin(obj, axis=1)]; new[~fin] = 0.0
        return new, beta

    prior = lambda dl: float(np.sum((-kp0 * np.cos(dl) + np.log(2 * np.pi * i0e(kp0)) + kp0)[fin]))
    bounds = [(np.log(0.1), np.log(4))] * 8 + [(-np.pi, np.pi)] * 4
    if pin_veins:
        for q in (2, 3, 6, 7, 10, 11):
            bounds[q] = (0.0, 0.0)
    Sv, X, DL, BT, prev, dl, beta = [], [], [], [], None, np.zeros(nT), (0.0 if beta_fixed is None else float(beta_fixed))
    for D_ in d_list:
        B1, B2 = group_response(C, 1, D_, eta, gid, el), group_response(C, 2, D_, eta, gid, el)
        starts = [np.zeros(12)] + ([prev] if prev is not None else [])
        for it in range(4 if gauge else 1):
            rb = min((minimize(nll, x0, args=(B1, B2, dl, beta), method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300)) for x0 in starts),
                     key=lambda r: r.fun)
            starts = [rb.x]
            if gauge:
                new, nb = update_delta(rb.x, B1, B2, dl, beta)
                done = it > 0 and np.max(np.abs(np.angle(np.exp(1j * (new - dl))))) < np.radians(1.0) and abs(np.angle(np.exp(1j * (nb - beta)))) < np.radians(1.0)
                dl, beta = new, nb
                if done:
                    break
        x = starts[0]; prev = x
        Sv.append(nll(x, B1, B2, dl, beta) + (prior(dl) if gauge else 0.0)); X.append(x); DL.append(dl + beta * blk); BT.append(beta)
    return np.array(Sv), X, DL, tiles, BT


def fit_profile(C, c, field, d_grid=D_GRID, eta_grid=(0.0,), weights="waveform", gauge=True, pin_veins=True, n_jobs=1,
                progress=print, gauge_block=None):
    """C: model.build_context on the measurement-supported field (vein forcing already set, e.g. model.vein_forcing);
    c: DC reliability of each edge's AC winner (model.ac_reliability); field: the AC field (tile, gauge_sd_deg, ref_tile,
    var_phi, var_logpi).  Returns dict(S [nη × nD], D_grid, eta_grid, D, eta, i_min, j_min, x, delta_deg {tile: deg},
    nll, profile_D)."""
    from joblib import Parallel, delayed
    d_grid = np.asarray(d_grid, float); eta_grid = np.asarray(eta_grid, float)
    n_seg = max(1, min(len(d_grid), n_jobs // max(len(eta_grid), 1)))
    segs = [s for s in np.array_split(np.arange(len(d_grid)), n_seg) if len(s)]
    jobs = [(i, s) for i in range(len(eta_grid)) for s in segs]
    if progress:
        progress(f"[ac.profile] {len(d_grid)} D × {len(eta_grid)} η, weights '{weights}', gauge {'on' if gauge else 'off'}, "
                 f"veins {'pinned' if pin_veins else 'fitted'}; {len(jobs)} chains on {n_jobs} workers")
    args = (field["tile"], field["gauge_sd_deg"], int(field["ref_tile"]), field.get("var_phi"), field.get("var_logpi"))
    res = Parallel(n_jobs=n_jobs)(delayed(_chain)(C, c, *args, d_grid[s], eta_grid[i], weights, gauge, pin_veins, gauge_block) for i, s in jobs)
    S = np.full((len(eta_grid), len(d_grid)), np.nan); X, DL, BT = {}, {}, {}
    for (i, s), (Sv, Xs, DLs, tiles, BTs) in zip(jobs, res):
        S[i, s] = Sv
        for q, j in enumerate(s):
            X[(i, j)] = Xs[q]; DL[(i, j)] = DLs[q]; BT[(i, j)] = BTs[q]
    i, j = np.unravel_index(np.nanargmin(S), S.shape)
    return dict(S=S, D_grid=d_grid, eta_grid=eta_grid, i_min=int(i), j_min=int(j), D=float(d_grid[j]), eta=float(eta_grid[i]),
                x=X[(i, j)], delta_deg={int(t): float(np.degrees(d)) for t, d in zip(tiles, DL[(i, j)])}, nll=float(S[i, j]),
                profile_D=np.nanmin(S, axis=0) - np.nanmin(S), weights=weights, gauge=gauge, pin_veins=pin_veins,
                block_beta_deg=float(np.degrees(BT[(i, j)])), gauge_block=sorted(gauge_block or []))


def quadratic_min(d_grid, prof):
    """Minimum of a profile by a parabola in log10 D through the grid minimum and its two neighbours, so the vertex is
    bracketed (the grid point itself if the minimum is at a grid end or the three points are not convex).  Before
    2026-11-03 the three LOWEST points were used, which can be one-sided and extrapolate past the bracket."""
    d, p = np.asarray(d_grid, float), np.asarray(prof, float); i = int(np.argmin(p))
    if i == 0 or i == len(p) - 1:
        return float(d[i])
    x = np.log10(d[i - 1:i + 2]); c = np.polyfit(x, p[i - 1:i + 2], 2)
    return float(np.clip(10 ** (-c[1] / (2 * c[0])), d[i - 1], d[i + 1])) if c[0] > 0 else float(d[i])


def fitted_forcing(C, x):
    """Boundary flux forcing (per dom['bnd'] node, k = 1, 2) with the fitted group corrections applied."""
    bnd = np.asarray(C["dom"]["bnd"]); gid = groups(C); x = np.asarray(x)
    return {k: C["F"][(k, "flux")][bnd] * np.exp(x[(k - 1) * 4 + gid] + 1j * k * x[8 + gid]) for k in (1, 2)}


def group_table(C, x):
    """Per group: amplitude corrections, timing (beats) and boundary pulsatility |Σ f_k| / |Σ f_DC|."""
    bnd = np.asarray(C["dom"]["bnd"]); gid = groups(C); f = fitted_forcing(C, x); fdc = (C["dom"]["B"] @ C["Qp"])[bnd]; out = {}
    for n, g in enumerate(GROUPS):
        m = gid == n
        if m.any():
            out[g] = dict(amp1=float(np.exp(x[n])), amp2=float(np.exp(x[4 + n])), timing=float(x[8 + n] / (2 * np.pi)),
                          PI1=float(abs(f[1][m].sum()) / abs(fdc[m].sum())), PI2=float(abs(f[2][m].sum()) / abs(fdc[m].sum())))
    return out
