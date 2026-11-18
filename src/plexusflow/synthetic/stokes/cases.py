"""The three known-truth experiments, each run from a benchmark bundle (bundle.py).

junction_case        real Y-junction × flow split α × seed: one advection, a video per parent speed, every real segment
                     measured with the production code (docs/synthetic_validation.md, "Junction ensemble")
vessel_case          controlled single vessel (straight / arc / S-bend / taper, elliptical sections) × seed, a video per speed;
                     optionally every video measured at several optical-flow frame spacings (dt study), with a background
                     from any stage, contrast-matched tracers and frame jitter (15S-matched conditions)
Truth: quasi-steady Stokes flow u(x, t) = Q0 s(t) û(x); the time-mean flux of every branch is known exactly.
"""
from __future__ import annotations

import numpy as np

from .render import (TRACERS_JUNCTION, TRACERS_VESSEL, FDT, JunctionFrame, VesselFrame, waveform, tau_at_frames,
                     tracer_movie, scale_to_sd, scale_to_contrast, camera)
from .tracers import p2_field, seed_tracers, advect_pseudotime
from .measure import run_measurement, flux_record, h1_over_dc, central_dip

JUNCTION_SPEEDS = (0.1, 0.2, 0.3, 0.5, 0.9)      # parent Poiseuille v_max, px/frame
JUNCTION_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
JUNCTION_SEEDS = (1, 2)


def junction_case(bundle, node, alpha, seed, speeds=JUNCTION_SPEEDS, nf=400, f0=2.9, tracer=TRACERS_JUNCTION, stage="21S"):
    """One (junction, α, seed): returns dict(node, tile, alpha, seed, cases={speed: dict(edges=[...], imbalance, s_t)}).
    Edge records: true signed time-mean flux (outward from the node), measured Q_kl, kept fraction f, slices, radii,
    quality numbers, measured Q(t)."""
    J, Z = bundle.junction(node); br, tile = J["branches"], J["tile"]
    uhat = alpha * Z["uA"] + (1 - alpha) * Z["uB"]; FF = p2_field(Z["p"], Z["t"], uhat)
    s_t = waveform(f0); S = tau_at_frames(s_t, nf)
    Q0s = {v: v * np.pi * br[0]["R"] ** 2 / 2 for v in speeds}; tau_all = np.unique(np.concatenate([Q0s[v] * S for v in speeds]))
    qcap = [-1.0, alpha, 1 - alpha]; caps = []
    for k, b in enumerate(br):
        e, e2 = b["path"][-1], b["path"][-2]; c_ = np.array([e[1], -e[0], 0.0]); t_ = np.array([e[1] - e2[1], -(e[0] - e2[0]), 0.0]); t_ /= np.linalg.norm(t_)
        caps.append((c_, t_, b["R"], qcap[k]))
    rng = np.random.default_rng(seed * 1000 + int(alpha * 10))
    x0 = seed_tracers(FF, tracer["C"], rng)
    rec, _ = advect_pseudotime(FF, caps, tau_all, tracer["C"], rng, x0); idx = {t: k for k, t in enumerate(tau_all)}
    allp = np.vstack([b["path"] for b in br]); pad = 12
    r0, c0 = int(max(allp[:, 0].min() - pad, 0)), int(max(allp[:, 1].min() - pad, 0)); r1, c1 = int(allp[:, 0].max() + pad), int(allp[:, 1].max() + pad)
    B = bundle.background(stage, tile, nf); bg, sd = B.crop(r0, r1, c0, c1); Hh, Ww = bg.shape
    yy, xx = np.mgrid[:Hh, :Ww]; seg = np.zeros((Hh, Ww), bool); true_lum = np.zeros((Hh, Ww), bool)
    for b in br:
        for p in b["path"][:b["n_real"]]:
            d2 = (yy - (p[0] - r0)) ** 2 + (xx - (p[1] - c0)) ** 2; seg |= d2 <= b["R_seg"] ** 2; true_lum |= d2 <= b["R"] ** 2
    target = np.median(sd[seg]); intens = {}; frame = JunctionFrame(r0, c0)
    out = dict(node=node, tile=tile, alpha=alpha, seed=seed, cases={})
    for v in speeds:
        img = tracer_movie(rec, idx, Q0s[v] * S, nf, (Hh, Ww), frame, tracer, rng, intens)
        scale = scale_to_sd(img, bg, B.gain, B.read, target, seg, true_lum)
        vid = camera(img, bg, scale, B.gain, B.read, rng)
        smean = S[-1] / nf; edges = []
        for k, b in enumerate(br):
            qt = (-1 if k == 0 else 1) * [1.0, alpha, 1 - alpha][k] * Q0s[v] * smean
            cl = b["path"][:b["n_real"]] - np.array([r0, c0])
            rec_e = dict(k=k, edge=b["edge"], R_seg=b["R_seg"], R_true=b["R"], Q_true=qt,
                         L_over_R=float(np.hypot(*np.diff(cl, axis=0).T).sum() / b["R"]), v_true=abs(qt) / (np.pi * b["R"] ** 2 / 2))
            try:
                rec_e.update(flux_record(run_measurement(vid, cl, b["R_seg"], f0, seg), b["R_seg"]))
            except Exception as ex:
                rec_e.update(Q_kl=np.nan, f=0.0, err=str(ex))
            edges.append(rec_e)
        qm = np.array([e["Q_kl"] for e in edges])
        out["cases"][v] = dict(edges=edges, imbalance=abs(np.nansum(qm)) / np.nansum(np.abs(qm)) if np.isfinite(qm).all() else np.nan,
                               s_t=s_t(np.arange(nf)))
    return out


def vessel_case(bundle, name, speeds, seed, nf=400, f0=2.9, tracer=TRACERS_VESSEL, background=("21S", 30, (150, 150)),
                target_sd=9.0, contrast=None, jitter=None, dts=None, speed_order=None):
    """One (vessel geometry, seed): a video per speed (centre v_max at the mean section, px/frame).

    background  (stage, tile, (row, col) of the crop corner) — a crop of that tile's stored background
    target_sd   lumen temporal SD (counts) the tracers are scaled to (typical of real clean vessels), unless
    contrast    is given: tracer-only lumen SD = contrast × camera noise SD
    jitter      None or (white σ px per axis, heartbeat amplitude px): rigid per-frame shift, same for every speed
    dts         None → one measurement per video at the batch settings (returns size/shape/accuracy rows);
                list → the video is measured at every frame spacing (int, 'auto', or ('auto', dt_max, target_px)),
                rows carry Q_kl, kept fraction, H1/DC vs truth, dt used
    speed_order iteration order of the speeds (rendering consumes random numbers in this order); default as given
    """
    meta, Z = bundle.vessel(name); F = p2_field(Z["p"], Z["t"], Z["u"])
    s_t = waveform(f0); S = tau_at_frames(s_t, nf)
    a_m, eps = meta["a_meas"], meta["eps"]; Q0s = {v: v * np.pi * a_m * (eps * a_m) / 2 for v in speeds}
    tau_all = np.unique(np.concatenate([Q0s[v] * S for v in speeds]))
    caps = [(c_, t_, max(a, b), q) for c_, t_, a, b, q in meta["caps"]]                # injection disk ≥ section; rejected outside mesh
    rng = np.random.default_rng(seed)
    x0 = seed_tracers(F, tracer["C"], rng)
    rec, _ = advect_pseudotime(F, caps, tau_all, tracer["C"], rng, x0); idx = {t: k for k, t in enumerate(tau_all)}
    pad = 15; xy = meta["P"]; amax = meta["a0"] * max(1.0, 1 / max(eps, 1e-6))
    x0_, y0_ = xy[:, 0].min() - amax - pad, xy[:, 1].min() - amax - pad
    Wd = int(xy[:, 0].max() + amax + pad - x0_); Hd = int(xy[:, 1].max() + amax + pad - y0_)
    stage, tile, rc = background; B = bundle.background(stage, tile, nf)
    bg, _ = B.crop(rc[0], rc[0] + Hd, rc[1], rc[1] + Wd); Hh, Ww = bg.shape; yy, xx = np.mgrid[:Hh, :Ww]
    cl = np.column_stack([xy[meta["meas"], 1] - y0_, xy[meta["meas"], 0] - x0_]); Rseg = a_m / 1.2
    seg = np.zeros((Hh, Ww), bool)
    for p in cl:
        seg |= (yy - p[0]) ** 2 + (xx - p[1]) ** 2 <= Rseg ** 2
    jit = None
    if jitter is not None and (jitter[0] > 0 or jitter[1] > 0):
        jr = np.random.default_rng(1000 + seed); th = jr.uniform(0, 2 * np.pi)
        hb = jitter[1] * np.sin(2 * np.pi * f0 * np.arange(nf) * FDT + jr.uniform(0, 2 * np.pi))
        jit = jr.normal(0, jitter[0], (nf, 2)) + np.outer(hb, [np.cos(th), np.sin(th)])        # (dy, dx) px
    frame = VesselFrame(x0_, y0_); intens = {}; out = []
    for v in (speed_order or speeds):
        img = tracer_movie(rec, idx, Q0s[v] * S, nf, (Hh, Ww), frame, tracer, rng, intens)
        if contrast is None:
            scale = scale_to_sd(img, bg, B.gain, B.read, target_sd, seg, seg)
        else:
            scale = scale_to_contrast(img, bg, B.gain, B.read, contrast, seg)
        vid = camera(img, bg, scale, B.gain, B.read, rng, jitter=jit)
        qt = Q0s[v] * S[-1] / nf
        base = dict(name=name, kind=meta["kind"], a=a_m, eps=eps, v=v, seed=seed, Q_true=qt)
        if dts is None:
            try:
                u = run_measurement(vid, cl, Rseg, f0, seg)
                q = float(u.get("Q_kloosterman", np.nan)); fq = float(u.get("Q_kloosterman_valid_frac", 0.0)) if np.isfinite(q) else 0.0
                rk = float(u.get("R_kloosterman", np.nan))
                _psq = np.asarray(u.get("Q_kloosterman_per_slice", []), float); _psr = np.asarray(u.get("R_kloosterman_per_slice", []), float)
                _ok = np.isfinite(_psq) & np.isfinite(_psr[:len(_psq)]) if len(_psr) >= len(_psq) else np.zeros(0, bool)
                extra = dict(V_CL=float(u.get("V_CL_kloosterman", np.nan)), R_med=float(np.median(_psr[:len(_psq)][_ok])) if _ok.any() else np.nan,
                             U_true=qt / (np.pi * a_m * (eps * a_m) / 2), R_seg=Rseg)
                from ...analyze.mosaic_piv import _flow_quality
                fl = _flow_quality(u.get("vfield_mean"), u.get("cl_used"), float(u.get("R_used", Rseg)))
                fl["dip"] = central_dip(u.get("vfield_mean"), u.get("cl_used"), float(u.get("R_used", Rseg)))
            except Exception:
                q, fq, rk, fl = np.nan, 0.0, np.nan, dict(align=np.nan); extra = {}
            out.append(dict(base, **extra, Q_kl=q, ratio=abs(q) / qt if np.isfinite(q) else np.nan, f=fq, R_kl_over_a=rk / a_m,
                            align=fl.get("align", np.nan), dip=fl.get("dip", np.nan)))
            continue
        for dt in dts:
            kw = dict(dt=dt) if not isinstance(dt, tuple) else dict(dt="auto", dt_max=dt[1], target_disp_px=dt[2])
            try:
                u = run_measurement(vid, cl, Rseg, f0, seg, **kw)
                q = float(u.get("Q_kloosterman", np.nan)); f = float(u.get("Q_kloosterman_valid_frac", 0.0)) if np.isfinite(q) else 0.0
                h1, used = h1_over_dc(u, f0, dt)
                out.append(dict(base, dt=dt, Q_kl=q, f=f, h1=h1, dt_used=used, pilot=float(u.get("of_pilot_disp_px", np.nan))))
            except Exception as ex:
                out.append(dict(base, dt=dt, Q_kl=np.nan, f=0.0, h1=np.nan + 0j, dt_used=-1, pilot=np.nan, err=str(ex)))
    return out
