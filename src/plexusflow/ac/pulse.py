"""Pulsatile flow reconciliation on the carved network (docs/method.md, step 3; replaces ``ac.field.build_ac_field``).

One uncertainty per recording: σ², the variance of its fitted mean from the residual of its own fit
(``reconcile.dc.residual_variance``: s² = RSS / (M − 5), σ² = s² / M).

1. Fit.  Each recording (valid profile fit, tile inside the heart-rate band) is fitted with its mean and K_FIT = 2 harmonics at its
   tile's heart rate:  Q(t) ≈ Q̂_0 + Σ_n Re[Q̂_n e^{2πi n f_α t}].  (A text that writes Σ_n Q̂_n e^{inωt} + c.c. has coefficients half
   of these.)
2. Tile clocks ψ_α (ψ_ref = 0) from the speed pulse of vessels seen in two tiles (``ac.speed``; pairs with a reversing recording
   left out):  d = δ − (ψ_α − ψ_β), variance τ_α² + τ_β².  Weighted circular least squares over all clocks jointly,
   min Σ_pairs [1 − cos d] / variance; spectral start (``ac.field._sync``), then a quasi-Newton minimisation (``tile_clocks``).
   Clock uncertainty: bootstrap over vessels.  It is conditional on the pair variances — not a complete uncertainty of the
   physiological phase.  Tiles whose clock SD exceeds MAX_CLOCK_SD_DEG are not synchronised: their recordings stay in the mean flow
   but are left out of the pulsatile field.
3. Average.  Of a vessel's recordings in synchronised tiles, those whose mean flow has the direction of the reconciled mean flow are
   averaged:  Q̂av_{e,n} = Σ_α Q̂_{α,n} e^{−inψ_α}/σ²_α / Σ_α 1/σ²_α  (n = 0, 1, 2; no rotation for n = 0).  A recording of the
   opposite direction is not used and never sign-flipped: its harmonics are half a cycle off in the vessel's orientation.  On 21S
   such recordings are mostly poorly focused ones (the opposite reading is in the image itself, not in the sign convention, the
   stitching or the optical flow); a real reversal of flow between recordings cannot be excluded.  Vessels with recordings of both directions are flagged
   (``inconsistent``).  The mean-flow reconciliation itself uses every recording.
4. Rescale.  Ĥ_{e,n} = (Q̄*_e / Q̂av_{e,0}) Q̂av_{e,n}: the averaged waveform, rescaled so that its mean is the reconciled mean.
   Not reconstructed where no recording agrees with the reconciled direction or the reconciled flow vanishes.

Earlier variant, kept for the sensitivity comparison (results/ac_field_v8; ``average="all"`` with clocks from ``flux_pairs``): clocks
from the first harmonics of the fluxes of repeated vessels plus two dorsal-aorta segments oriented to their majority direction; all
recordings averaged; rescaled by the combined mean of all mean-flow measurements, vessels with a negative ratio dropped.

The returned dict has the layout of ``build_ac_field`` (H is 3 × n_edges with a zero third row) so the network model and the
viewers read it unchanged.  Driver: ``scripts/pulse_field.py`` → results/ac_field_v9.pkl.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from ..reconcile.harmonics import phasors, coherence, FS
from ..reconcile.dc import residual_variance, K_FIT
from .field import _sync, da_stub_edges, REF_TILE, Q_DET, CANONICAL_F0_BAND

FIELD = "ac_field_v9"        #: results/<FIELD>.pkl written by scripts/pulse_field.py (v8: the earlier clocks and averaging)
MAX_CLOCK_SD_DEG = 30.0      #: tiles whose clock is less certain than this are not synchronised


def heart_rate_band(G, ref_tile=REF_TILE, f0_band=CANONICAL_F0_BAND):
    """Tiles whose heart rate is within ± f0_band of the reference tile's and that carry a detected beat."""
    tf0 = G.graph["tile_f0_piv"]; f_ref = tf0[ref_tile]; st = G.graph.get("tile_f0_status") or {}
    return {int(t) for t, f in tf0.items() if f and np.isfinite(f) and abs(f / f_ref - 1) <= f0_band
            and (st.get(t) or {}).get("status") not in ("no_signal", "excluded", "no measurements")}


def recordings(G, edges, tiles, K=K_FIT):
    """{edge: [recording, …]} for the recordings of `edges` in `tiles` with a valid profile fit: mean a0, harmonics H (K,),
    variance of the mean sigma2 (fit residual), coherence rho (fraction of temporal variance in the K harmonics), frames n."""
    tf0 = G.graph["tile_f0_piv"]; out = {}
    for e in edges:
        e = tuple(e)
        for m in G.edges[e].get("measurements_piv") or []:
            t = int(m["tile_id"])
            if not ((m.get("quality") or {}).get("kl_ok") and t in tiles and m.get("Q_t") is not None):
                continue
            q = np.asarray(m["Q_t"], float); f0 = float(tf0[t]); a0, H = phasors(q, f0, K=K); s2 = residual_variance(q, f0, K=K)
            if not (np.isfinite(a0) and a0 != 0 and np.all(np.isfinite(H)) and abs(H[0]) > 0 and np.isfinite(s2) and s2 > 0):
                continue
            rho = coherence(q, f0, K=K)
            out.setdefault(e, []).append(dict(t=t, a0=float(a0), H=np.asarray(H, complex), sigma2=float(s2),
                                              rho=float(rho) if np.isfinite(rho) else 0.0, n=int(np.isfinite(q).sum())))
    return out


def _fit_clocks(nT, iref, pa, pb, phase, w, x0):
    """min Σ w [1 − cos(phase − ψ_a + ψ_b)] over the clocks jointly (index iref fixed at 0)."""
    free = np.array([k for k in range(nT) if k != iref], int); sc = 1.0 / max(float(np.sum(w)), 1e-300)
    def fg(x):
        psi = np.zeros(nT); psi[free] = x; d = phase - psi[pa] + psi[pb]; g = np.zeros(nT); gd = w * np.sin(d)
        np.add.at(g, pa, -gd); np.add.at(g, pb, gd); return sc * float(np.sum(w * (1 - np.cos(d)))), sc * g[free]
    r = minimize(fg, x0[free], jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
    psi = np.zeros(nT); psi[free] = r.x; return psi, float(r.fun)


def flux_pairs(recs, own_direction=False):
    """Clock measurements from the first harmonics of the FLUX of vessels recorded in two tiles (the earlier method; same layout as
    ``ac.speed.speed_pairs``).  recs: {vessel: [recording, …]} (``recordings``)."""
    rows = []
    for iv, rs in enumerate(recs.values()):
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ra, rb = rs[i], rs[j]
                if ra["t"] == rb["t"]:
                    continue
                Ha, Hb = ra["H"][0], rb["H"][0]
                if own_direction:                               # phase of the pulse relative to the recording's own mean-flow direction
                    Ha, Hb = np.sign(ra["a0"]) * Ha, np.sign(rb["a0"]) * Hb
                rows.append((iv, ra["t"], rb["t"], np.angle(Ha * np.conj(Hb)), 2.0 * ra["sigma2"] / abs(Ha) ** 2 + 2.0 * rb["sigma2"] / abs(Hb) ** 2))
    r = np.array(rows, float).reshape(-1, 5)
    return dict(edges=list(recs), vessel=r[:, 0].astype(int), a=r[:, 1].astype(int), b=r[:, 2].astype(int), d=r[:, 3], var=r[:, 4],
                reversing=np.zeros(len(r), bool))


def tile_clocks(pairs, ref_tile, n_boot=0, seed=0):
    """Tile clocks by weighted circular least squares (module docstring, step 2).  pairs: dict of arrays a, b (tiles), d (rad,
    estimating ψ_a − ψ_b), var (rad²), vessel (index, for the bootstrap) — ``ac.speed.speed_pairs`` or ``flux_pairs``, already
    restricted to the pairs to use.  Returns (psi {tile: rad} for the tiles linked to ref_tile, info) with info = dict(tiles,
    components (largest first), n_pairs, objective, clock_sd_deg {tile: deg} (if n_boot))."""
    ta, tb = np.asarray(pairs["a"], int), np.asarray(pairs["b"], int); d = np.asarray(pairs["d"], float); w = 1.0 / np.asarray(pairs["var"], float)
    ves = np.asarray(pairs["vessel"], int); tiles = sorted(set(ta.tolist()) | set(tb.tolist())); ti = {t: k for k, t in enumerate(tiles)}; nT = len(tiles)
    par = list(range(nT))                                       # connected components of the tile graph
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in zip(ta, tb):
        par[find(ti[a])] = find(ti[b])
    comp = {}
    for k in range(nT):
        comp.setdefault(find(k), []).append(tiles[k])
    comp = sorted(comp.values(), key=len, reverse=True)
    if ref_tile not in ti:
        raise ValueError(f"reference tile {ref_tile} has no clock measurement")
    mt = sorted(next(c for c in comp if ref_tile in c)); mi = {t: k for k, t in enumerate(mt)}; iref = mi[ref_tile]
    keep = np.array([a in mi for a in ta], bool); pa = np.array([mi[a] for a in ta[keep]], int); pb = np.array([mi[b] for b in tb[keep]], int)
    d, w, ves = d[keep], w[keep], ves[keep]
    psi, obj = _fit_clocks(len(mt), iref, pa, pb, d, w, _sync(pa, pb, w, np.exp(1j * d), len(mt), iref))
    info = dict(tiles=tiles, components=comp, n_pairs=int(len(pa)), objective=obj)
    if n_boot:
        rng = np.random.default_rng(seed); uv, vi = np.unique(ves, return_inverse=True); D = []
        for _ in range(n_boot):
            wb = w * np.bincount(rng.integers(0, len(uv), len(uv)), minlength=len(uv))[vi]
            D.append(np.angle(np.exp(1j * (_fit_clocks(len(mt), iref, pa, pb, d, wb, psi)[0] - psi))))
        info["clock_sd_deg"] = {mt[k]: float(v) for k, v in enumerate(np.degrees(np.sqrt(np.mean(np.array(D) ** 2, axis=0))))}
    return {mt[k]: float(np.angle(np.exp(1j * psi[k]))) for k in range(len(mt))}, info


def orient_to_majority(recs, edges):
    """Negate, for the given edges, the recordings whose mean-flow direction is in the 1/σ²-weighted minority (provisional handling
    of the dorsal-aorta segments, module docstring step 2).  Returns (recs, number negated)."""
    out = dict(recs); n = 0
    for e in edges:
        rs = recs.get(e)
        if not rs:
            continue
        sw = np.sign(sum(np.sign(r["a0"]) / r["sigma2"] for r in rs)); new = []
        for r in rs:
            if sw != 0 and np.sign(r["a0"]) != sw:
                r = dict(r, a0=-r["a0"], H=-r["H"], reoriented=True); n += 1
            new.append(r)
        out[e] = new
    return out, n


def build_pulse_field(G, dc, pairs=None, ref_tile=REF_TILE, f0_band=CANONICAL_F0_BAND, average="agree", sync_extra_edges=(), n_boot=100,
                      max_clock_sd_deg=MAX_CLOCK_SD_DEG):
    """The reconciled pulsatile field on the edges of the mean-flow reconciliation `dc` (module docstring).
    pairs: clock measurements (``ac.speed.speed_pairs``); pairs with a tile outside the heart-rate band or a reversing recording are
    left out.  None: the earlier clocks, from the flux harmonics of repeated network vessels and of `sync_extra_edges` (edges outside
    the network, oriented to their majority flow direction; "aorta" = ``da_stub_edges(G)``).
    average: "agree" — recordings with the reconciled flow direction, rescaled by their own mean; "all" — every recording, rescaled by
    the combined mean of all mean-flow measurements, vessels with a negative ratio dropped (the earlier method).
    n_boot: bootstrap resamples for the clock SD (0: none, no tile excluded)."""
    if average not in ("agree", "all"):
        raise ValueError(f"unknown averaging rule {average!r}")
    sub = [tuple(e) for e in dc["edges"]]; eix = {e: i for i, e in enumerate(sub)}; Qp = np.asarray(dc["Qp"], float); nE = len(sub)
    Qm = np.asarray(dc["Qm"], float); imaged = np.asarray(dc["imaged"], bool)      # combined mean of all mean-flow measurements, per vessel
    band = heart_rate_band(G, ref_tile, f0_band); tf0 = G.graph["tile_f0_piv"]; n_reor = 0
    extra = [tuple(e) for e in (da_stub_edges(G) if isinstance(sync_extra_edges, str) and sync_extra_edges == "aorta" else sync_extra_edges)
             if tuple(e) not in eix]
    raw = recordings(G, sub + extra, band)
    if pairs is None:
        raw, n_reor = orient_to_majority(raw, extra); use = flux_pairs(raw); clock_source = "flux harmonics of repeated vessels"
    else:
        m = np.isin(pairs["a"], sorted(band)) & np.isin(pairs["b"], sorted(band)) & ~np.asarray(pairs["reversing"], bool)
        use = {k: np.asarray(pairs[k])[m] for k in ("a", "b", "d", "var", "vessel")}; clock_source = "speed pulse of vessels seen in two tiles"
    psi_all, info = tile_clocks(use, ref_tile, n_boot=n_boot)
    sd = info.get("clock_sd_deg") or {t: 0.0 for t in psi_all}
    excluded = sorted(t for t in psi_all if sd[t] > max_clock_sd_deg); psi = {t: v for t, v in psi_all.items() if t not in excluded}
    N = np.arange(1, K_FIT + 1)
    H = np.full((3, nE), np.nan + 0j); elig = np.zeros(nE, bool); home = np.full(nE, -1); a0 = np.full(nE, np.nan); incons = np.zeros(nE, bool)
    rho = np.full(nE, np.nan); vphi = np.full((2, nE), np.nan); n_rec = np.zeros(nE, int); n_negative = n_undefined = n_no_agree = 0; kept = set()
    for e, rs in raw.items():
        i = eix.get(e); rs = [r for r in rs if r["t"] in psi]
        if i is None or not rs:
            continue
        incons[i] = len({np.sign(r["a0"]) for r in rs}) > 1
        if abs(Qp[i]) <= Q_DET:                                        # the reconciled flow vanishes: no waveform to rescale
            n_undefined += 1; continue
        if average == "agree":
            rs = [r for r in rs if np.sign(r["a0"]) == np.sign(Qp[i])]
            if not rs:
                n_no_agree += 1; continue
        elif not imaged[i] or Qm[i] == 0:
            n_undefined += 1; continue
        elif Qp[i] / Qm[i] < 0:                                        # reconciled and combined means of opposite sign: not reconstructed
            n_negative += 1; continue
        w = np.array([1.0 / r["sigma2"] for r in rs])
        Hb = sum(wi * r["H"] * np.exp(-1j * N * psi[r["t"]]) for wi, r in zip(w, rs)) / w.sum()
        qav = float(np.sum(w * np.array([r["a0"] for r in rs])) / w.sum()) if average == "agree" else Qm[i]
        H[:K_FIT, i] = Qp[i] / qav * Hb; H[K_FIT:, i] = 0.0; elig[i] = True; a0[i] = qav; n_rec[i] = len(rs)
        home[i] = rs[int(np.argmax(w))]["t"]; rho[i] = max(r["rho"] for r in rs); kept |= {(e, r["t"]) for r in rs}
        vphi[:, i] = (1.0 / w.sum()) / np.maximum(np.abs(Hb[:2]) ** 2, 1e-30)      # variance of the averaged coefficient / |coefficient|²
    meas = {e: {r["t"]: dict(a0=r["a0"], H=np.r_[r["H"], np.zeros(3 - len(r["H"]))], rho=r["rho"], sigma2=r["sigma2"], n=r["n"],
                             kept=(e, r["t"]) in kept, scatter=np.nan, var_phi=(np.nan, np.nan), var_logpi=np.nan, p_det=np.nan)
                for r in rs} for e, rs in raw.items()}
    return dict(edges=sub, H=H, elig=elig, tile=home, psi=psi, scatter=np.full(nE, np.nan), rho=rho, a0=a0, var_phi=vphi,
                var_logpi=vphi[0] / np.log(10) ** 2, f_ref=float(tf0[ref_tile]), band=sorted(band), ref_tile=int(ref_tile), Qp=Qp,
                gauge_sd_deg={int(t): float(sd[t]) for t in psi}, n_rec=n_rec, _meas=meas, _pairs=[], _sync_diag=None,
                method=f"pulse reconciliation: mean + 2 harmonics, σ² from the fit residual; joint tile clocks from the {clock_source} (weighted circular "
                       "least squares); 1/σ² average of " + ("the recordings with the reconciled flow direction, rescaled to the reconciled mean"
                                                            if average == "agree" else "all recordings, rescaled by Q̄*/Q̄av (negative ratios dropped)"),
                psi_all=psi_all, clock_sd_deg={int(t): float(v) for t, v in sd.items()}, excluded_tiles=excluded, max_clock_sd_deg=float(max_clock_sd_deg),
                clock_source=clock_source, average=average, inconsistent=incons, n_inconsistent=int((incons & elig).sum()), n_no_agreeing=int(n_no_agree),
                tile_links={(int(a), int(b)): int(n) for (a, b), n in zip(*np.unique(np.c_[np.minimum(use["a"], use["b"]), np.maximum(use["a"], use["b"])], axis=0, return_counts=True))} if len(use["a"]) else {},
                n_sync_extra=len(extra), n_extra_reoriented=int(n_reor), n_negative_ratio=int(n_negative), n_undefined_ratio=int(n_undefined),
                k_fit=K_FIT, n_recordings=int(sum(len(r) for r in raw.values())), n_pairs=info["n_pairs"], components=info["components"],
                unsynchronised_tiles=sorted(set(band) - set(psi)))
