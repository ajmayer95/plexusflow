"""Pulsatile flow reconciliation on the carved network (method of 2026-11-20; replaces ``ac.field.build_ac_field``).

Follows the 'Measurement reconciliation' Methods text.  One uncertainty per recording: σ², the variance of its fitted mean from the
residual of its own fit (``reconcile.dc.residual_variance``: s² = RSS / (M − 5), σ² = s² / M).

1. Fit.  Each recording (valid profile fit, tile inside the heart-rate band) is fitted with its mean and K_FIT = 2 harmonics at its
   tile's heart rate:  Q(t) ≈ Q̂_0 + Σ_n Re[Q̂_n e^{2πi n f_α t}].  (The text writes Σ_n Q̂_n e^{inωt} + c.c.; its coefficients are half
   of these.)
2. Recordings used: those of the vessels of the network, plus — for the clock fit only — the dorsal-aorta segments just outside it.
   PROVISIONAL (2026-11-20, to be revisited): the aorta is the one strongly pulsing vessel recorded on both sides of the seam
   between the left block of 21S tiles (15–20, 29–34, 42–44) and the rest; the plexus vessels alone fix that block's clock only to
   about ±30°.  One aorta recording (tile 20, a poor profile fit) reads the flow backwards, i.e. its whole signal is negated; taken at
   face value it turns the block by half a cycle.  So the recordings of these extra segments are oriented to the segment's
   1/σ²-weighted majority flow direction (a recording in the minority is negated: on 21S exactly that one) before they enter the
   clock fit; network vessels are used exactly as recorded, with φ = arg Q̂_1 in the fixed vessel orientation as in the Methods
   text.  The proper fix is to re-measure the tile-20 aorta recording.
   (Alternative kept as an option, ``clock_own_direction=True``: take every recording's phase relative to its own mean-flow
   direction, φ = arg(Q̂_1 / Q̂_0); no special case, but unreliable mean directions add noise.  21S: D 4.0e-4 instead of 4.3e-4.)
   Optional direction rule (``direction_filter=True``; off by default, not in the Methods text): for a network vessel recorded
   more than once, recordings whose mean-flow direction disagrees with the 1/σ²-weighted majority are not retained.  It removes
   poorly fitted recordings from the averages (21S: D about 4.0 instead of 4.3e-4).
3. Tile clocks ψ_α (ψ_ref = 0), from repeated recordings of the same vessel, first harmonic only:
       min Σ_e Σ_{α<β} w_αβ [1 − cos((φ_α − ψ_α) − (φ_β − ψ_β))],   w_αβ = 1 / (τ_α² + τ_β²),
   with the phase variance τ² = 2σ²/|Q̂_1|² (σ²/(2|Q̂_1|²) in the text's c.c. convention).  Spectral start (``ac.field._sync``), then
   a joint quasi-Newton minimisation over all clocks (a tile-by-tile update cannot move a weakly linked block of tiles).
4. Average.  Q̂av_{e,n} = Σ_α Q̂_{α,n} e^{−inψ_α}/σ²_α / Σ_α 1/σ²_α  (n = 1, 2) over the retained recordings.
5. Rescale.  The waveform is scaled by Q̄*_e / Q̄av_e, the reconciled mean over the combined mean of ALL the vessel's mean-flow
   measurements (the vessel input of the mean-flow reconciliation, dc["Qm"]):  Ĥ_{e,n} = (Q̄*_e / Q̄av_e) Q̂av_{e,n}.
   Not reconstructed where the ratio is negative or undefined, or where no synchronised harmonic is available.

The returned dict has the layout of ``build_ac_field`` (H is 3 × n_edges with a zero third row) so the network model and the
viewers read it unchanged.  Driver: ``scripts/pulse_field.py`` → results/ac_field_v8.pkl.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from ..reconcile.harmonics import phasors, coherence, FS
from ..reconcile.dc import residual_variance, K_FIT
from .field import _sync, da_stub_edges, REF_TILE, Q_DET, CANONICAL_F0_BAND

FIELD = "ac_field_v8"        #: results/<FIELD>.pkl written by scripts/pulse_field.py


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


def direction_rule(recs):
    """Drop, per vessel, the recordings whose mean-flow direction disagrees with the 1/σ²-weighted majority.
    Returns (kept, number dropped)."""
    kept, n_drop = {}, 0
    for e, rs in recs.items():
        sw = sum(np.sign(r["a0"]) / r["sigma2"] for r in rs); keep = [r for r in rs if np.sign(r["a0"]) == np.sign(sw)]
        n_drop += len(rs) - len(keep)
        if keep:
            kept[e] = keep
    return kept, n_drop


def tile_clocks(recs, ref_tile, own_direction=False):
    """Tile clocks from repeated recordings, pairs weighted by 1/(τ_a² + τ_b²) (module docstring, step 3).  Returns (psi {tile: rad} for the tiles linked to
    ref_tile, info) with info = dict(tiles, components (lists of tiles, largest first), n_pairs, objective)."""
    tiles = sorted({r["t"] for rs in recs.values() for r in rs}); ti = {t: k for k, t in enumerate(tiles)}; nT = len(tiles)
    pa, pb, pw, pz = [], [], [], []
    for rs in recs.values():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                ra, rb = rs[a], rs[b]
                if ra["t"] == rb["t"]:
                    continue
                Ha, Hb = ra["H"][0], rb["H"][0]
                if own_direction:                               # phase of the pulse relative to the recording's own mean-flow direction
                    Ha, Hb = np.sign(ra["a0"]) * Ha, np.sign(rb["a0"]) * Hb
                pa.append(ti[ra["t"]]); pb.append(ti[rb["t"]]); pz.append(Ha / abs(Ha) * np.conj(Hb / abs(Hb)))
                pw.append(1.0 / (2.0 * ra["sigma2"] / abs(Ha) ** 2 + 2.0 * rb["sigma2"] / abs(Hb) ** 2))    # 1 / (τ_a² + τ_b²)
    pa, pb, pw, pz = np.asarray(pa, int), np.asarray(pb, int), np.asarray(pw, float), np.asarray(pz, complex)
    par = list(range(nT))                                       # connected components of the tile-overlap graph
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in zip(pa, pb):
        par[find(a)] = find(b)
    comp = {}
    for k in range(nT):
        comp.setdefault(find(k), []).append(tiles[k])
    comp = sorted(comp.values(), key=len, reverse=True)
    if ref_tile not in ti:
        raise ValueError(f"reference tile {ref_tile} has no repeated recording")
    main = next(c for c in comp if ref_tile in c); mset = set(main)
    keep = np.array([tiles[a] in mset for a in pa], bool) if len(pa) else np.zeros(0, bool)
    mt = sorted(main); mi = {t: k for k, t in enumerate(mt)}
    qa = np.array([mi[tiles[a]] for a in pa[keep]], int); qb = np.array([mi[tiles[b]] for b in pb[keep]], int); qw, qz = pw[keep], pz[keep]
    p0 = _sync(qa, qb, qw, qz, len(mt), mi[ref_tile]); free = np.array([k for k in range(len(mt)) if mt[k] != ref_tile], int)
    ph = np.angle(qz); sc = 1.0 / max(float(qw.sum()), 1e-300)
    def fg(x):
        psi = np.zeros(len(mt)); psi[free] = x; d = ph - psi[qa] + psi[qb]; g = np.zeros(len(mt))
        np.add.at(g, qa, -qw * np.sin(d)); np.add.at(g, qb, qw * np.sin(d))
        return sc * float(np.sum(qw * (1 - np.cos(d)))), sc * g[free]
    r = minimize(fg, p0[free], jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
    psi = {mt[k]: float(np.angle(np.exp(1j * v))) for k, v in zip(free, r.x)}; psi[ref_tile] = 0.0
    return psi, dict(tiles=tiles, components=comp, n_pairs=int(len(pa)), objective=float(r.fun))


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


def build_pulse_field(G, dc, ref_tile=REF_TILE, f0_band=CANONICAL_F0_BAND, sync_extra_edges=None, direction_filter=False,
                      clock_own_direction=False):
    """The reconciled pulsatile field on the edges of the mean-flow reconciliation `dc` (module docstring).
    sync_extra_edges: edges outside the network that take part in the clock fit only, oriented to their majority flow direction
    (default ``da_stub_edges(G)``, the dorsal-aorta segments; () for none).  direction_filter: apply the optional direction rule."""
    sub = [tuple(e) for e in dc["edges"]]; eix = {e: i for i, e in enumerate(sub)}; Qp = np.asarray(dc["Qp"], float); nE = len(sub)
    Qav = np.asarray(dc["Qm"], float); imaged = np.asarray(dc["imaged"], bool)     # combined mean of all mean-flow measurements, per vessel
    extra = [tuple(e) for e in (da_stub_edges(G) if sync_extra_edges is None else sync_extra_edges) if tuple(e) not in eix]
    band = heart_rate_band(G, ref_tile, f0_band); tf0 = G.graph["tile_f0_piv"]
    raw = recordings(G, sub + extra, band); raw, n_reor = (raw, 0) if clock_own_direction else orient_to_majority(raw, extra)
    recs, n_drop = direction_rule(raw) if direction_filter else (raw, 0)
    psi, info = tile_clocks(recs, ref_tile, own_direction=clock_own_direction); N = np.arange(1, K_FIT + 1)
    H = np.full((3, nE), np.nan + 0j); elig = np.zeros(nE, bool); home = np.full(nE, -1); a0 = np.full(nE, np.nan)
    rho = np.full(nE, np.nan); vphi = np.full((2, nE), np.nan); n_rec = np.zeros(nE, int); n_negative = n_undefined = 0
    for e, rs in recs.items():
        i = eix.get(e); rs = [r for r in rs if r["t"] in psi]
        if i is None or not rs:
            continue
        if not imaged[i] or Qav[i] == 0 or abs(Qp[i]) <= Q_DET:        # rescaling ratio Q̄*/Q̄av undefined (or the reconciled flow vanishes)
            n_undefined += 1; continue
        if Qp[i] / Qav[i] < 0:                                         # reconciled and combined means of opposite sign: not reconstructed
            n_negative += 1; continue
        w = np.array([1.0 / r["sigma2"] for r in rs])
        Hb = sum(wi * r["H"] * np.exp(-1j * N * psi[r["t"]]) for wi, r in zip(w, rs)) / w.sum()
        H[:K_FIT, i] = Qp[i] / Qav[i] * Hb; H[K_FIT:, i] = 0.0; elig[i] = True; a0[i] = Qav[i]; n_rec[i] = len(rs)
        home[i] = rs[int(np.argmax(w))]["t"]; rho[i] = max(r["rho"] for r in rs)
        vphi[:, i] = (1.0 / w.sum()) / np.maximum(np.abs(Hb[:2]) ** 2, 1e-30)      # variance of the averaged coefficient / |coefficient|²
    kept = {(e, r["t"]) for e, rs in recs.items() for r in rs}
    meas = {e: {r["t"]: dict(a0=r["a0"], H=np.r_[r["H"], np.zeros(3 - len(r["H"]))], rho=r["rho"], sigma2=r["sigma2"], n=r["n"],
                             kept=(e, r["t"]) in kept, scatter=np.nan, var_phi=(np.nan, np.nan), var_logpi=np.nan, p_det=np.nan)
                for r in rs} for e, rs in raw.items()}
    return dict(edges=sub, H=H, elig=elig, tile=home, psi=psi, scatter=np.full(nE, np.nan), rho=rho, a0=a0, var_phi=vphi,
                var_logpi=vphi[0] / np.log(10) ** 2, f_ref=float(tf0[ref_tile]), band=sorted(band), ref_tile=int(ref_tile), Qp=Qp,
                gauge_sd_deg={int(t): 0.0 for t in psi}, n_rec=n_rec, _meas=meas, _pairs=[], _sync_diag=None,
                method="pulse reconciliation 2026-11-20 (Methods text): mean + 2 harmonics, σ² from the fit residual, joint clocks on repeated "
                       "network vessels with pair weight 1/(τa²+τb²), 1/σ² average, rescaled by Q̄*/Q̄av (negative ratios dropped)"
                       + ("; direction rule" if direction_filter else ""),
                direction_filter=bool(direction_filter), n_sync_extra=len(extra), n_extra_reoriented=int(n_reor), clock_own_direction=bool(clock_own_direction),
                n_negative_ratio=int(n_negative), n_undefined_ratio=int(n_undefined),
                k_fit=K_FIT, n_recordings=int(sum(len(r) for r in raw.values())), n_direction_dropped=int(n_drop),
                n_pairs=info["n_pairs"], components=info["components"], unsynchronised_tiles=sorted(set(band) - set(psi)))
