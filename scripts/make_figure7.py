"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  heart rate f0 of every tile vs acquisition time (first-frame time stamp of the analysed recording); tiles within the band
     (canonical ±21 % of the reference tile) are retained (shaded).  Colour = fraction of the tile's coherent vessels (ρ ≥ 0.1) whose
     flow reverses within the beat (|H1| > |Q̄|): the earliest, slow-beating tiles are a distinct, mostly oscillatory flow state.
     Inset: one vessel recorded in tile 3 (9 min) and tile 5 (21 min): its flow reverses every beat early, not later.  (Same-vessel overlap
     check of the early pulsatility decrease: supplement, sandbox/pulsatility_vs_time_overlap.py.)
  b  one overlapping tile pair of the core (most shared coherent vessels), split at a seam (dotted) through the overlap: each side
     shows its own tile's vessels coloured by H1 phase (cyclic).  As recorded the colours jump at the seam (arbitrary clock of each
     tile); after one phase offset per tile, φ_aligned = φ_obs − ψ_t, they run continuously.  No spatial gradient is imposed.
  c  organ scale: circular kernel smoothing (σ 150 µm, ρ-weighted phasors) of the shown tiles' vessel phases, isochrones every 20°.
     Tiles fall into stitching components (linked by ≥ 1 shared vessel coherent in both): the main one (with the reference
     tile) is zeroed at the simultaneous DA/V reference; the second (dashed) is stitched separately and its single offset β to
     the main one is inferred from arterial-inlet symmetry (inlet vessels of both components carry the same H1 phase; tile
     bootstrap SD); isolated tiles and tiles with bootstrap SD > 30° are open dots; stars = V, triangles = DA reference
  d  validation on vessels imaged in two tiles: same-vessel phase difference after alignment, H1 (used to estimate ψ_t) and
     H2 (corrected with 2ψ_t, not fitted), median |Δφ|
  Phases: synchronisation anchored on the dorsal aorta (= results/ac_field_v3); zero = tile-14 physiological boundary
  (2 DA + 2 V stubs, one recording); flow-direction frame (sgn Q′); measurements against the reconciled flow not used.
  (supplement fig7_supp_heldout: held-out overlap prediction)
Reads the graph and results/dc_reconciled_v2.pkl (recomputes the AC field); writes figures/fig7_sync/fig7.{png,pdf,tif}.
"""
from __future__ import annotations
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.ac.field import build_ac_field, F0_BAND
from plexusflow.config import StageConfig
from plexusflow.analyze.geometry import load_tile_geometry

OUT_DIR = _paths.FIGURES / "fig7_sync"
SD_MAX = 30.0
T0 = time.time()


def log(m):
    print(f"[fig7 +{time.time() - T0:4.0f} s] {m}", flush=True)


def tile_polygons(stage):
    """Each tile's footprint in mosaic-image pixels (col, row), from the registration affines."""
    import re
    shp = {}
    for f_ in (stage / "tiles" / "maxmed").glob("vid*_maxmed.tif"):
        with tifffile.TiffFile(f_) as tf_:
            shp[int(re.match(r"vid(\d+)_", f_.name).group(1))] = tf_.pages[0].shape[:2]
    geom = load_tile_geometry(StageConfig.load(stage), video_shapes=shp); out = {}
    global FUSE_UM_PX
    FUSE_UM_PX = 1.7 * geom.downsample                                  # tile pixel 1.7 µm; fused image downsampled
    for t, tg in geom.tiles.items():
        H, W = tg.video_shape; corners = np.array([[0, 0, 1], [0, W, 1], [H, W, 1], [H, 0, 1]], float)
        rm, cm, _ = (tg.A_full @ corners.T)
        out[t] = np.c_[cm / geom.downsample - geom.x0, rm / geom.downsample - geom.y0]       # (col, row) in the fused image
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--f0-band", type=float, default=0.21, help="heart-rate band (fraction of the reference tile's); canonical 0.21 (tiles 7-9 included)")
    ap.add_argument("--tag", default="", help="output name suffix, e.g. _band21"); args = ap.parse_args(); BAND = args.f0_band
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    G = _paths.load_graph(); dc = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    log("rebuilding the AC field (per-tile phasors and synchronisation pairs)")
    # synchronisation anchored on the dorsal aorta (same vessel in tiles 14 / 21 / 28 must agree; ρ ≥ 0.5 measurements, ×100),
    # as results/ac_field_v3 (scripts/ac_field.py --sync-da --sync-da-min-rho 0.5 --sync-da-weight 100)
    from plexusflow.reconcile import carved_domain as _cd
    _dom0 = _cd(G); _carved = set(_dom0["edges"])
    da_edges = [e if G.has_edge(*e) else e[::-1] for n in _dom0["DA"] for e in G.edges(n)]
    da_edges = [e for e in da_edges if e not in _carved and e[::-1] not in _carved]
    F = build_ac_field(G, dc, sync_extra_edges=da_edges, sync_extra_min_rho=0.5, sync_extra_weight=100.0, f0_band=BAND)
    meas, psi, pairs, sd = F["_meas"], F["psi"], F["_pairs"], F["gauge_sd_deg"]
    pairs = [(e, a, b) for e, a, b in pairs if e not in set(da_edges)]          # validation on carved vessels only
    tf0, ref, f_ref, band = G.graph["tile_f0_piv"], F["ref_tile"], F["f_ref"], set(F["band"])
    log(f"{len(band)} of {len(tf0)} tiles in band; {len(pairs)} synchronisation pairs (both ρ ≥ 0.1)")

    # cross-tile phase differences, raw and aligned, H1 and H2
    d = {k: {"raw": [], "aligned": []} for k in (1, 2)}
    for e, a, b in pairs:
        for k in (1, 2):
            ha, hb = meas[e][a]["H"][k - 1], meas[e][b]["H"][k - 1]
            d[k]["raw"].append(np.angle(ha * np.conj(hb)))
            d[k]["aligned"].append(np.angle(ha * np.exp(-1j * k * psi[a]) * np.conj(hb * np.exp(-1j * k * psi[b]))))
    d = {k: {s: np.degrees(np.asarray(v)) for s, v in dd.items()} for k, dd in d.items()}
    for k in (1, 2):
        log(f"  H{k}: median |Δφ| raw {np.median(np.abs(d[k]['raw'])):.1f}°, aligned {np.median(np.abs(d[k]['aligned'])):.1f}°")

    # tile links: Σ ρ_a ρ_b over shared vessels
    link = {}
    for e, x in meas.items():
        ts = [t for t in x if t in band and np.isfinite(x[t]["rho"])]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = sorted((ts[i], ts[j])); link[(a, b)] = link.get((a, b), 0.0) + max(x[a]["rho"], 0) * max(x[b]["rho"], 0)

    mos = tifffile.imread(str(_paths.MOSAIC)); mos = mos.mean(axis=-1) if mos.ndim == 3 else mos; Hm, Wm = mos.shape
    polys = tile_polygons(_paths.STAGE); cen = {t: p.mean(0) for t, p in polys.items()}
    allp = np.vstack([polys[t] for t in band]); pad_ = 15
    XL, YL = (allp[:, 0].min() - pad_, allp[:, 0].max() + pad_), (allp[:, 1].max() + pad_, allp[:, 1].min() - pad_)

    def organ(ax):
        ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), zorder=0, alpha=0.35)
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        [s_.set_visible(False) for s_ in ax.spines.values()]

    # two figures: Fig 7 = choosing a stitchable regime (a heart rate / reversal, b early-tile vessel, c band vessel, d repeat-vessel
    # population); Fig 8 = synchronization (a seam before/after, b H1/H2 validation, c organ phase field).  Panels draw on `fig`,
    # which is switched from figR to figS before the synchronization panels.
    # Fig 7 = three regimes: transient (early tiles), low pulsatility (tiles barely linked to the rest), stitchable.
    #   a organ map of the regimes + example vessels · b heart rate vs acquisition time · c tile link vs pulsatile amplitude ·
    #   d one example vessel per regime (Q/Q̄) · e repeat-vessel waveform-shape difference D, transient vs stitchable
    figR = plt.figure(figsize=(7.5, 6.4), facecolor="white")
    gR = figR.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.36, left=0.075, right=0.975, top=0.94, bottom=0.08)
    gRt = gR[0].subgridspec(1, 2, width_ratios=[1, 1.08], wspace=0.22); gRr = gRt[1].subgridspec(2, 1, hspace=0.62)
    gRb = gR[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.42)
    GMAP, GA, GK7, GX, GT = gRt[0], gRr[0], gRr[1], [gRb[0], gRb[1], gRb[2]], gRb[3]
    figS = plt.figure(figsize=(7.5, 5.9), facecolor="white")
    gS2 = figS.add_gridspec(1, 2, width_ratios=[0.5, 1], wspace=0.06, left=0.08, right=0.97, top=0.93, bottom=0.1)
    gL = gS2[0].subgridspec(2, 1, height_ratios=[3.0, 0.85], hspace=0.2); GB, GD, GC = gL[0], gL[1], gS2[1]
    fig = figR

    def label(cell, letter):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.045, b_.y1 + 0.03, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- a: which tiles are comparable — heart rate per tile, accepted band
    axA = fig.add_subplot(GA); ts = sorted(tf0); f = np.array([tf0[t] for t in ts]); inb = np.array([t in band for t in ts])
    # acquisition time: the camera's record time in each recording's Photron header (.cih "Date"/"Time", minute resolution), minutes
    # after the first tile; falls back to the first raw frame's file time.  (The .irg IRIG stamps are not usable — no IRIG source; they
    # jump between recordings.  Header and file times agree to ≤ 1.1 min for all 53 tiles.)
    import glob as _glob, os as _os, re as _re
    from plexusflow.config import StageConfig as _SC
    _cfg = _SC.load(_paths.STAGE); _rd = _os.path.expanduser(str(_cfg.external_raw_video_dir or "")); _tcih, _tfile = {}, {}
    for t in ts:                                     # the recording the pipeline analysed: raw_video_dir_pattern with this tile id
        _cands = _glob.glob(_os.path.join(_rd, _cfg.raw_video_dir_pattern.format(vid=t)))
        _fs = sorted(_glob.glob(_os.path.join(sorted(_cands)[0], "*.tif")))[:1] if _cands else []
        if _fs:
            _tfile[t] = _os.stat(_fs[0]).st_mtime
        _cih = _glob.glob(_os.path.join(sorted(_cands)[0], "*.cih")) if _cands else []
        if _cih:
            _txt = open(_cih[0], errors="ignore").read(); _md = _re.search(r"Date : (\d+)/(\d+)/(\d+)", _txt); _mt = _re.search(r"Time : (\d+):(\d+)", _txt)
            if _md and _mt:
                import datetime as _dtm
                _tcih[t] = _dtm.datetime(*map(int, _md.groups()), *map(int, _mt.groups())).timestamp()
    tacq = _tcih if all(t in _tcih for t in ts) else _tfile        # one clock for every tile (the two differ by a fixed ~3 h)
    if all(t in tacq for t in ts):
        _t0 = min(tacq.values()); xs_a = np.array([(tacq[t] - _t0) / 60 for t in ts]); xlab = "acquisition time (min)"
    else:
        xs_a = np.array(ts, float); xlab = "acquisition order (tile)"
    xref = xs_a[ts.index(ref)]
    # regimes per tile: transient = heart rate below the band (the first recordings); low pulsatility = band tiles linked to the
    # rest by < 5 % of the median link weight (Σ ρ_a ρ_b over shared vessels); stitchable = the other band tiles
    early = {t for t in ts if t not in band and tf0[t] < f_ref * (1 - BAND)}
    Lt7 = {t: 0.0 for t in band}
    for (a_, b_), w_ in link.items():
        Lt7[a_] += w_; Lt7[b_] += w_
    L_thr7 = 0.05 * np.median(list(Lt7.values())); lowp = {t for t in band if Lt7[t] < L_thr7}; stitch = set(band) - lowp
    RCOL = {"T": "#b23a48", "L": "#8a8a8a", "S": "#2b5c8a"}
    RLAB = {"T": "transient (early)", "L": "low pulsatility", "S": "stitchable"}
    reg = lambda t: "T" if t in early else ("L" if t in lowp else ("S" if t in stitch else None))
    log(f"  regimes: transient {sorted(early)}; low pulsatility {sorted(lowp)}; stitchable {len(stitch)} tiles")
    Fall = build_ac_field(G, dc, n_boot=0, f0_band=10.0); mall = Fall["_meas"]

    # ---- b: heart rate over the acquisition, coloured by regime
    axA.axhspan(f_ref * (1 - BAND), f_ref * (1 + BAND), color="#eef1f5", zorder=0, lw=0)
    for k_ in ("T", "L", "S"):
        m_ = np.array([reg(t) == k_ for t in ts])
        axA.scatter(xs_a[m_], f[m_], s=12, color=RCOL[k_], edgecolors="white", linewidths=0.4, zorder=3, label=RLAB[k_])
    axA.plot(xref, f_ref, "s", ms=6, mfc="none", mec="black", mew=0.9, zorder=4)
    axA.annotate("reference tile", xy=(xref, f_ref), xytext=(xref + 0.08 * xs_a.max(), f_ref - 0.45), fontsize=6.5, color="0.3", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
    axA.text(0.99, f_ref * (1 - BAND) + 0.03, "heart-rate band", transform=axA.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.5, color="0.45")
    axA.set_xlabel(xlab, fontsize=8, labelpad=1); axA.set_ylabel("heart rate $f_0$ (Hz)", fontsize=8); axA.tick_params(labelsize=7)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.1, borderaxespad=0.1, markerscale=1.1)
    label(GA, "b")

    # ---- c: why the low-pulsatility tiles cannot be stitched — link to the other tiles vs pulsatile amplitude
    from scipy.stats import spearmanr as _spr
    h1t7 = {t: np.median([abs(x[t]["H"][0]) for x in meas.values() if t in x and np.isfinite(x[t]["rho"])]) for t in band}
    axK = fig.add_subplot(GK7); tl_k = sorted(band)
    for k_ in ("S", "L"):
        m_ = [t for t in tl_k if reg(t) == k_]
        axK.scatter([h1t7[t] for t in m_], [max(Lt7[t], 0.05) for t in m_], s=12, color=RCOL[k_], edgecolors="white", linewidths=0.4, zorder=3)
    axK.axhline(L_thr7, color="0.6", lw=0.6, ls=":"); axK.set_xscale("log"); axK.set_yscale("log")
    axK.text(0.99, L_thr7 * 1.3, "stitching limit", transform=axK.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.5, color="0.45")
    axK.set_xlabel("tile pulsatile amplitude, median $|H_1|$ (nL/s)", fontsize=8, labelpad=1)
    axK.set_ylabel("link to other tiles\n" + r"$\Sigma\,\rho_a\rho_b$", fontsize=8, labelpad=1); axK.tick_params(labelsize=7)
    axK.spines[["top", "right"]].set_visible(False)
    rs7 = _spr([h1t7[t] for t in tl_k], [Lt7[t] for t in tl_k])[0]
    axK.text(0.02, 0.97, f"Spearman {rs7:+.2f}", transform=axK.transAxes, fontsize=6.5, va="top", color="0.3")
    log(f"  (c) link vs median |H1|: Spearman {rs7:+.2f}")
    label(GK7, "c")

    # waveform-shape spectrum of one recording, mean-normalised and clock-free: s_n = H_n e^{−inφ1} / Q̄ (n = 1..3, flow-direction frame).
    # Shape difference of two recordings D = ‖s_a − s_b‖ / rms(‖s_a‖, ‖s_b‖) — the relative RMS difference of the H1-aligned Q/Q̄
    # waveforms (Parseval); it carries the harmonic amplitudes |H_n|/Q̄ and the inter-harmonic phases Θ21, Θ31 together.
    def _shape(y_):
        H_ = np.asarray(y_["H"][:3]) * np.sign(y_["a0"]); u_ = np.conj(H_[0] / abs(H_[0]))
        return H_ * u_ ** np.arange(1, 4) / abs(y_["a0"])
    def _D(ya, yb):
        sa, sb = _shape(ya), _shape(yb); return np.linalg.norm(sa - sb) / np.sqrt((np.linalg.norm(sa) ** 2 + np.linalg.norm(sb) ** 2) / 2)
    SP = {"E": [], "B": []}; vD = {"E": {}, "B": {}}
    for e_, x_ in mall.items():
        ok_ = [t_ for t_, y_ in x_.items() if np.isfinite(y_["rho"]) and y_["rho"] >= 0.2 and abs(y_["a0"]) > 0 and abs(y_["H"][0]) > 0]
        for i_ in range(len(ok_)):
            for j_ in range(i_ + 1, len(ok_)):
                a_, b_ = ok_[i_], ok_[j_]; ya, yb = x_[a_], x_[b_]
                g_ = "E" if (a_ in early and b_ in early) else ("B" if (a_ in stitch and b_ in stitch) else None)
                if g_ is None or np.sign(ya["a0"]) != np.sign(yb["a0"]):
                    continue
                d_ = _D(ya, yb); SP[g_].append(d_); vD[g_].setdefault(e_, []).append(d_)
    SP = {k: np.array(v) for k, v in SP.items()}; SCOL = {"E": RCOL["T"], "B": RCOL["S"]}
    SLAB = {"E": "transient", "B": "stitchable"}
    for g_ in ("E", "B"):
        log(f"  (e) repeat-vessel shape difference D, both {SLAB[g_]}: {len(SP[g_])} pairs / {len(vD[g_])} vessels, median {np.median(SP[g_]):.2f}, "
            f"90th {np.percentile(SP[g_], 90):.2f}")

    # ---- d: one vessel per regime, Q/Q̄ over two beats (raw trace light, 3-harmonic fit bold); recordings shifted to their own H1 phase
    from plexusflow.reconcile.harmonics import phasors as _phz
    EXV, EXA, EXB = (1206, 1267), 3, 5                     # transient: tiles 3 → 5, 12 min apart
    LPV, LPT = (24146, 24170), 18                           # low pulsatility: large steady vessel, tile 18 (outside every stitchable tile)
    RV = (49550, 49649); RVe = RV if RV in mall else RV[::-1]; xr_ = mall[RVe]
    rv_t = sorted([t_ for t_, y_ in xr_.items() if t_ in stitch and np.isfinite(y_["rho"]) and y_["rho"] >= 0.5], key=lambda t_: xs_a[ts.index(t_)])
    EXe = EXV if EXV in mall else EXV[::-1]
    def _trace(ax_, E_, t_, col_, lab_):
        e_ = E_ if G.has_edge(*E_) else E_[::-1]
        m_ = next(m for m in G.edges[e_]["measurements_piv"] if int(m["tile_id"]) == t_); q_ = np.asarray(m_["Q_t"], float); q_ = -q_ if np.mean(q_) < 0 else q_
        a0_, H_ = _phz(q_, tf0[t_]); tt_ = np.arange(len(q_)) / 250.0; tau_ = tf0[t_] * tt_ + np.angle(H_[0]) / (2 * np.pi)
        tau_ = tau_ - np.floor(tau_[0]); sel_ = (tau_ >= 0) & (tau_ <= 2); tg_ = np.linspace(0, 2, 300)
        fit_ = a0_ + sum((abs(H_[n - 1]) * np.exp(1j * (np.angle(H_[n - 1]) - n * np.angle(H_[0]))) * np.exp(2j * np.pi * n * tg_)).real for n in (1, 2, 3))
        ax_.plot(tau_[sel_], q_[sel_] / a0_, color=col_, lw=0.3, alpha=0.18, zorder=2)
        ax_.plot(tg_, fit_ / a0_, color=col_, lw=1.3, zorder=3, label=lab_)
        return a0_, H_
    axs_d = [fig.add_subplot(g_) for g_ in GX]
    a1_, H1a_ = _trace(axs_d[0], EXV, EXA, "#b23a48", f"tile {EXA}, {xs_a[ts.index(EXA)]:.0f} min")
    a1b_, _ = _trace(axs_d[0], EXV, EXB, "#e08a96", f"tile {EXB}, {xs_a[ts.index(EXB)]:.0f} min")
    aL_, HL_ = _trace(axs_d[1], LPV, LPT, RCOL["L"], f"tile {LPT}, {xs_a[ts.index(LPT)]:.0f} min")
    cRV = plt.cm.viridis(np.linspace(0.05, 0.8, len(rv_t)))
    for t_, c_ in zip(rv_t, cRV):
        _trace(axs_d[2], RV, t_, c_, f"{xs_a[ts.index(t_)]:.0f} min")
    dRV = [_D(xr_[a_], xr_[b_]) for i_, a_ in enumerate(rv_t) for b_ in rv_t[i_ + 1:]]; dEX = _D(mall[EXe][EXA], mall[EXe][EXB])
    LPe = LPV if LPV in mall else LPV[::-1]; yL_ = mall[LPe][LPT]
    ttl_ = [f"1  transient: same vessel 12 min apart\nD = {dEX:.2f}",
            f"2  low pulsatility: small $|H_1|$,\nphase poorly constrained (ρ = {yL_['rho']:.2f})",
            f"3  stitchable: same vessel, {len(rv_t)} tiles\nD = {min(dRV):.2f}–{max(dRV):.2f}"]
    for k_, (ax_, cl_) in enumerate(zip(axs_d, ("T", "L", "S"))):
        ax_.axhline(1, color="0.75", lw=0.4, ls=":"); ax_.axhline(0, color="0.4", lw=0.5); ax_.set_xlim(0, 2); ax_.set_xticks([0, 1, 2])
        ax_.set_ylim(-1.0, 2.6); ax_.set_xlabel("beats (H1-aligned)", fontsize=7.5, labelpad=1); ax_.tick_params(labelsize=7)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.set_title(ttl_[k_], fontsize=6.8, loc="left", color=RCOL[cl_], pad=3, linespacing=1.15)
        if k_ == 0:
            ax_.set_ylabel(r"$Q\,/\,\bar Q$", fontsize=8, labelpad=1)
        else:
            ax_.set_yticklabels([])
        ax_.legend(loc="lower left", fontsize=5.8, frameon=False, handlelength=0.9, borderaxespad=0.1, labelspacing=0.1, ncol=2 if k_ == 2 else 1,
                   columnspacing=0.6)
    log(f"  (d) stitchable vessel {RV}: tiles {rv_t}, pairwise D " + ", ".join(f"{v:.2f}" for v in dRV) + f"; transient vessel D {dEX:.2f}")
    label(GX[0], "d")

    # ---- e: across all repeat vessels — D, both recordings transient vs both stitchable; examples 1 and 3 marked
    axD = fig.add_subplot(GT); ecdf_ = {}
    for g_ in ("E", "B"):
        v_ = np.sort(SP[g_]); ecdf_[g_] = v_
        axD.step(v_, np.arange(1, len(v_) + 1) / len(v_), where="post", color=SCOL[g_], lw=1.3, label=f"{SLAB[g_]} ({len(v_)})")
        m_ = np.median(v_); axD.plot([m_], [0.5], "o", ms=3, color=SCOL[g_], zorder=4)
        axD.text(m_, 0.5, f"  {m_:.2f}", color=SCOL[g_], fontsize=7, ha="left", va="top" if g_ == "E" else "bottom", fontweight="bold")
    _on = lambda g_, x_: np.searchsorted(ecdf_[g_], x_, side="right") / len(ecdf_[g_])
    for g_, xs_, tag_ in (("E", [dEX], "1"), ("B", dRV, "3")):
        for x_ in xs_:
            axD.plot(x_, _on(g_, x_), "o", ms=3.5, mfc="white", mec=SCOL[g_], mew=0.9, zorder=5)
        xm_ = np.median(xs_); axD.annotate(tag_, xy=(xm_, _on(g_, xm_)), xytext=(xm_ + (0.08 if g_ == "E" else -0.08), _on(g_, xm_) + 0.1),
                                           fontsize=7, fontweight="bold", color=SCOL[g_], ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=SCOL[g_]))
    axD.set_xlim(0, 1.2); axD.set_ylim(0, 1.02); axD.set_yticks([0, 0.5, 1]); axD.spines[["top", "right"]].set_visible(False)
    axD.set_xlabel("waveform-shape difference $D$\nbetween recordings of a vessel", fontsize=7.5, labelpad=1)
    axD.set_ylabel("cumulative fraction", fontsize=8, labelpad=1); axD.tick_params(labelsize=7)
    axD.legend(title="repeat vessel,\nboth recordings", title_fontsize=6.3, fontsize=6.3, frameon=False, loc="lower right", handlelength=1.0,
               borderaxespad=0.0, labelspacing=0.15, alignment="left", bbox_to_anchor=(1.02, 0.0))
    label(GT, "e")

    # ---- a: organ map of the three regimes (union of each regime's tiles ∩ organ, smoothed) and the example vessels
    from matplotlib.path import Path as _MPath
    from scipy.ndimage import gaussian_filter as _gf
    allq = np.vstack(list(polys.values())); padm = 20; stp = 12.0
    def _path(e_):
        pth = G.edges[e_].get("path")
        q_ = np.asarray(pth, float) if pth is not None and len(pth) >= 2 else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e_], float)
        return q_[:, 0], Hm - q_[:, 1]
    gxs = np.arange(0, Wm, stp); gys = np.arange(0, Hm, stp); GX_, GY_ = np.meshgrid(gxs, gys); P2 = np.c_[GX_.ravel(), GY_.ravel()]
    nxy = np.array([(G.nodes[n]["x"], Hm - G.nodes[n]["y"]) for n in G.nodes if "x" in G.nodes[n]])
    dens, _, _ = np.histogram2d(nxy[:, 1], nxy[:, 0], bins=[np.r_[gys, Hm], np.r_[gxs, Wm]]); dens = _gf(dens, 60 / stp)
    organ_m = dens > 0.08 * dens.max()
    def _cover(tset):
        m_ = np.zeros(len(P2), bool)
        for t in tset:
            m_ |= _MPath(polys[t]).contains_points(P2)
        return m_.reshape(GX_.shape)
    cS, cL, cT = _cover(stitch), _cover(lowp), _cover(early)
    # one partition of the organ: each point takes one regime (priority stitchable > low pulsatility > transient, i.e. transient =
    # imaged only in the early recordings); each indicator is smoothed and every point takes the strongest, so neighbouring regions
    # share one boundary; islands smaller than ~2 % of the organ are absorbed by their surroundings
    from scipy.ndimage import label as _lab, binary_fill_holes as _fill
    raw_ = np.full(GX_.shape, -1); raw_[organ_m & cT] = 0; raw_[organ_m & cL] = 1; raw_[organ_m & cS] = 2
    organ_f = _fill(_gf(organ_m.astype(float), 30 / stp) > 0.5)
    sm_ = np.stack([_gf((raw_ == k).astype(float), 45 / stp) for k in range(3)]); lab_ = np.where(organ_f, sm_.argmax(0), -1)
    lab_[organ_f & (sm_.max(0) < 1e-3)] = 2
    amin_ = 0.02 * organ_f.sum()
    for _ in range(2):
        for k in range(3):
            cc_, n_ = _lab(lab_ == k)
            for i_ in range(1, n_ + 1):
                blob_ = cc_ == i_
                if blob_.sum() < amin_:
                    ring_ = (_gf(blob_.astype(float), 2) > 0.01) & ~blob_ & (lab_ >= 0)
                    lab_[blob_] = np.bincount(lab_[ring_], minlength=3).argmax() if ring_.any() else -1
    mT, mL, mS = [_gf((lab_ == k).astype(float), 1.0) for k in range(3)]
    axMp = fig.add_subplot(GMAP)
    axMp.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.45, zorder=0)
    plt.rcParams["hatch.linewidth"] = 0.5; plt.rcParams["hatch.color"] = "0.45"
    axMp.contourf(gxs, gys, mS, levels=[0.5, 2], colors=[RCOL["S"]], alpha=0.22, zorder=1)
    axMp.contourf(gxs, gys, mL, levels=[0.5, 2], colors=[RCOL["L"]], alpha=0.18, zorder=1)
    axMp.contourf(gxs, gys, mL, levels=[0.5, 2], colors=["none"], hatches=["/////"], zorder=1)
    axMp.contourf(gxs, gys, mT, levels=[0.5, 2], colors=[RCOL["T"]], alpha=0.3, zorder=1)
    axMp.contour(gxs, gys, _gf((lab_ >= 0).astype(float), 1.0), levels=[0.5], colors=["0.35"], linewidths=0.8, zorder=2)   # organ outline
    for m_ in (mT, mL):                                                                                    # shared internal boundaries
        axMp.contour(gxs, gys, m_, levels=[0.5], colors=["0.2"], linewidths=0.7, zorder=2)
    for E_, tag_, k_, off_ in ((EXV, "1", "T", (-330, -260)), (LPV, "2", "L", (330, 120)), (RV, "3", "S", (330, -260))):
        e_ = E_ if G.has_edge(*E_) else E_[::-1]; xx_, yy_ = _path(e_); cx_, cy_ = np.mean(xx_), np.mean(yy_)
        axMp.plot(xx_, yy_, color="black", lw=2.2, zorder=5, solid_capstyle="round")
        axMp.annotate("", xy=(cx_, cy_), xytext=(cx_ + off_[0], cy_ + off_[1]), arrowprops=dict(arrowstyle="-", lw=0.7, color="black"), zorder=6)
        axMp.plot(cx_ + off_[0], cy_ + off_[1], "o", ms=10, mfc=RCOL[k_], mec="white", mew=0.8, zorder=7)
        axMp.text(cx_ + off_[0], cy_ + off_[1], tag_, fontsize=7.5, ha="center", va="center", zorder=8, fontweight="bold", color="white")
    axMp.set_xlim(allq[:, 0].min() - padm, allq[:, 0].max() + padm); axMp.set_ylim(allq[:, 1].max() + padm, allq[:, 1].min() - padm)
    axMp.set_aspect("equal"); axMp.set_xticks([]); axMp.set_yticks([]); [s_.set_visible(False) for s_ in axMp.spines.values()]
    axMp.legend(handles=[matplotlib.patches.Patch(fc=mcolors.to_rgba(RCOL["T"], 0.35), ec=RCOL["T"], label=f"transient early recordings (tiles {min(early)}–{max(early)}, first {max(xs_a[ts.index(t)] for t in early) + 4:.0f} min)"),
                         matplotlib.patches.Patch(fc=mcolors.to_rgba(RCOL["L"], 0.18), ec=RCOL["L"], hatch="/////", label="low-pulsatility tiles (phase not constrained)"),
                         matplotlib.patches.Patch(fc=mcolors.to_rgba(RCOL["S"], 0.22), ec=RCOL["S"], label="stitchable regime")],
                loc="upper center", bbox_to_anchor=(0.5, -0.0), fontsize=6.8, frameon=False, handlelength=1.6, borderaxespad=0, ncol=1)
    label(GMAP, "a")

    # ---- SI (fig7_supp_shape_map): repeat vessels on the organ coloured by their waveform-shape difference D; early tiles shaded
    figSM = plt.figure(figsize=(4.0, 4.6), facecolor="white"); _fig_main = fig; fig = figSM
    axM = figSM.add_axes([0.03, 0.17, 0.94, 0.8]); allq = np.vstack(list(polys.values())); padm = 20
    axM.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.35, zorder=0)
    axM.set_xlim(allq[:, 0].min() - padm, allq[:, 0].max() + padm); axM.set_ylim(allq[:, 1].max() + padm, allq[:, 1].min() - padm)
    axM.set_aspect("equal"); axM.set_xticks([]); axM.set_yticks([]); [s_.set_visible(False) for s_ in axM.spines.values()]
    def _path(e_):
        pth = G.edges[e_].get("path")
        q_ = np.asarray(pth, float) if pth is not None and len(pth) >= 2 else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e_], float)
        return q_[:, 0], Hm - q_[:, 1]
    for e_, x_ in mall.items():                                          # every vessel with a coherent measurement in some tile
        if any(np.isfinite(y_["rho"]) and y_["rho"] >= 0.2 for y_ in x_.values()) and G.has_edge(*e_):
            axM.plot(*_path(e_), color="0.55", lw=0.35, zorder=1, solid_capstyle="round")
    for t in early:
        axM.add_patch(Polygon(polys[t], closed=True, fc="#b23a48", ec="#b23a48", alpha=0.10, lw=0, zorder=1))
        axM.add_patch(Polygon(polys[t], closed=True, fc="none", ec="#b23a48", lw=0.7, ls="--", zorder=4))
    axM.add_patch(Polygon(polys[ref], closed=True, fc="none", ec="black", lw=0.9, zorder=4))
    cmD = plt.cm.plasma_r; nD = mcolors.Normalize(0, 0.8)
    for g_ in ("B", "E"):
        for e_, dl_ in vD[g_].items():
            if G.has_edge(*e_):
                axM.plot(*_path(e_), color=cmD(nD(np.median(dl_))), lw=1.4, zorder=3, solid_capstyle="round")
    for (E_, tag_) in ((EXV, "1"), (RV, "2")):
        e_ = E_ if G.has_edge(*E_) else E_[::-1]; xx_, yy_ = _path(e_); cx_, cy_ = np.mean(xx_), np.mean(yy_)
        axM.add_patch(matplotlib.patches.Circle((cx_, cy_), 90, fc="none", ec="black", lw=1.0, zorder=5))
        axM.plot(cx_ + 150, cy_ - 150, "o", ms=9, mfc="black", mec="white", mew=0.6, zorder=6)
        axM.text(cx_ + 150, cy_ - 150, tag_, fontsize=7, ha="center", va="center", zorder=7, fontweight="bold", color="white")
    e1 = polys[min(early)]; cE = np.vstack([polys[t] for t in early])
    axM.text(cE[:, 0].mean(), cE[:, 1].min() - 40, f"early tiles {min(early)}–{max(early)} (first {max(xs_a[ts.index(t)] for t in early) + 5:.0f} min)",
             ha="center", va="bottom", fontsize=7, color="#b23a48", zorder=6)
    axM.text(*(polys[ref].mean(0) + np.array([0, 0])), f"ref.\ntile {ref}", ha="center", va="center", fontsize=6, color="0.15", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
    from matplotlib.lines import Line2D as _L2
    axM.legend(handles=[_L2([], [], color="0.55", lw=0.6, label="vessel sampled (ρ ≥ 0.2)"),
                        matplotlib.patches.Patch(fc="#f3dfe2", ec="#b23a48", ls="--", lw=0.7, label="early tiles (excluded)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=6.5, frameon=False, handlelength=1.4, borderaxespad=0, ncol=2, columnspacing=1.2)
    caxM = axM.inset_axes([0.25, -0.04, 0.5, 0.025]); cbM = fig.colorbar(plt.cm.ScalarMappable(norm=nD, cmap=cmD), cax=caxM, orientation="horizontal",
                                                                        ticks=[0, 0.4, 0.8], extend="max")
    cbM.ax.tick_params(labelsize=6.5, pad=1); cbM.outline.set_linewidth(0.4); cbM.ax.xaxis.set_label_position("top")
    cbM.set_label("repeat vessel: shape difference $D$", fontsize=7, labelpad=2)
    fig = _fig_main

    # ---- b: held-out overlap prediction — drop all links between a and b, refit ψ from the rest, predict ψa − ψb
    from plexusflow.ac.field import _sync
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tl = sorted(band); ti_ = {t: i for i, t in enumerate(tl)}; nT_ = len(tl); PA, PB, PW, PZ = [], [], [], []
    for e, x in meas.items():                                  # the same pair set as build_ac_field (all pairs, ρ_aρ_b weights)
        ok = [t for t, y in x.items() if t in ti_ and abs(y["H"][0]) > 0 and np.isfinite(y["rho"])]
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]; ua = x[a]["H"][0] / abs(x[a]["H"][0]); ub = x[b]["H"][0] / abs(x[b]["H"][0])
                PA.append(ti_[a]); PB.append(ti_[b]); PW.append(max(x[a]["rho"], 0.0) * max(x[b]["rho"], 0.0)); PZ.append(ua * np.conj(ub))
    PA, PB, PW, PZ = map(np.asarray, (PA, PB, PW, PZ)); psi0 = np.array([psi[t] for t in tl])
    acc = {}
    for e, a, b in pairs:
        xa, xb = meas[e][a], meas[e][b]; w = max(xa["rho"], 0) * max(xb["rho"], 0)
        acc.setdefault((a, b), []).append((w, xa["H"][0] / abs(xa["H"][0]) * np.conj(xb["H"][0] / abs(xb["H"][0]))))
    NMIN = 5; obs, pred, fitd, nn, n_bridge = [], [], [], [], 0
    for (a, b), L in acc.items():
        if len(L) < NMIN:
            continue
        ia, ib = ti_[a], ti_[b]; keep = ~(((PA == ia) & (PB == ib)) | ((PA == ib) & (PB == ia)))
        A = coo_matrix((np.ones(keep.sum()), (PA[keep], PB[keep])), shape=(nT_, nT_))
        if connected_components(A, directed=False)[0] > connected_components(coo_matrix((np.ones(len(PA)), (PA, PB)), shape=(nT_, nT_)), directed=False)[0]:
            n_bridge += 1; continue
        ps = _sync(PA[keep], PB[keep], PW[keep], PZ[keep], nT_, ti_[ref], start=psi0)
        pr = np.degrees(np.angle(np.exp(1j * (ps[ia] - ps[ib])))); pf = np.degrees(np.angle(np.exp(1j * (psi[a] - psi[b]))))
        w = np.array([l[0] for l in L]); z = np.array([l[1] for l in L]); o = np.degrees(np.angle(np.sum(w * z)))
        obs.append(pr + (o - pr + 180) % 360 - 180); pred.append(pr); fitd.append(pf); nn.append(len(L))
    obs, pred, fitd, nn = map(np.asarray, (obs, pred, fitd, nn)); res = np.abs(obs - pred); resf = np.abs(((obs - fitd) + 180) % 360 - 180)
    log(f"  (b) held out {len(obs)} tile pairs with ≥ {NMIN} shared vessels ({n_bridge} bridges skipped): median |measured − predicted| "
        f"{np.median(res):.1f}°, 90th {np.percentile(res, 90):.1f}°, max {res.max():.1f}°  (in-sample: median {np.median(resf):.1f}°)")
    # b — H1 phase delay from the boundary, tile-averaged.  Phases in the common frame, relative to each vessel's own flow
    #     direction (× sgn Q̄); q ∝ cos(ωt + φ), so a smaller φ = a later peak.  Boundary phase φ_bnd = circular mean over the
    #     field's vessels within 1 hop of the driven boundary (arterial carve + venous outflow); delay_t = wrap(φ_bnd − φ̄_t).
    from plexusflow.reconcile import carved_domain
    from scipy.sparse.csgraph import shortest_path
    dom = carved_domain(G); Ef = F["edges"]; nds = dom["nodes"]; nix_ = {n: i for i, n in enumerate(nds)}
    Ad = coo_matrix((np.ones(len(Ef)), ([nix_[u] for u, v in Ef], [nix_[v] for u, v in Ef])), shape=(len(nds),) * 2)
    dB = shortest_path(Ad, directed=False, indices=[int(i) for i in dom["inj"]] + [int(i) for i in dom["out"]]).min(0)
    eB = np.array([min(dB[nix_[u]], dB[nix_[v]]) for u, v in Ef])
    okf = F["elig"] & np.array([sd.get(int(t), np.inf) <= SD_MAX if t >= 0 else False for t in F["tile"]])
    phf = np.angle(F["H"][0] * np.sign(F["Qp"])); nb = okf & (eB <= 1)
    phi_bnd = np.angle(np.sum(np.exp(1j * phf[nb])))
    # zero = the physiological boundary, all in tile 14 (one recording, no tile offset between them): the two DA inflow stubs and
    # the two V outflow stubs, ρ-weighted H1 phase in each vessel's own flow direction
    from plexusflow.reconcile.harmonics import phasors as _ph, coherence as _coh
    zda, wda = [], []; ZT = 14
    for n_ in list(dom["DA"]) + list(dom["V"]):
        for e_ in G.edges(n_):
            for m in G.edges[e_].get("measurements_piv") or []:
                if int(m["tile_id"]) == ZT and m.get("Q_t") is not None and (m.get("quality") or {}).get("kl_ok"):
                    a0_, H_ = _ph(m["Q_t"], tf0[ZT]); r_ = _coh(m["Q_t"], tf0[ZT])
                    zda.append(np.exp(1j * np.angle(np.sign(a0_) * H_[0] * np.exp(-1j * psi[ZT])))); wda.append(max(r_, 0))
    phi_b = np.angle(np.sum(np.array(wda) * np.array(zda)))
    log(f"  (b) zero = tile-{ZT} physiological boundary (DA + V stubs): {np.degrees(phi_b):.1f}° ({len(zda)} measurements); "
        f"driven boundary {np.degrees(phi_bnd):.1f}° → boundary lags the DA by {np.degrees(np.angle(np.exp(1j * (phi_b - phi_bnd)))):.0f}°")
    # well-synchronised core: drop near-unconstrained tiles (< 5 % of median link weight) and the weakest normalised cut
    #     (the side without the reference tile) — tiles whose orientation relative to the rest the shared vessels do not fix
    tl_ = sorted(band); ix_t = {t: i for i, t in enumerate(tl_)}; Wt = np.zeros((len(tl_),) * 2)
    for e, x in meas.items():
        ok_ = [t for t, y in x.items() if t in ix_t and abs(y["H"][0]) > 0 and np.isfinite(y["rho"])]
        for i in range(len(ok_)):
            for j in range(i + 1, len(ok_)):
                w_ = max(x[ok_[i]]["rho"], 0) * max(x[ok_[j]]["rho"], 0); Wt[ix_t[ok_[i]], ix_t[ok_[j]]] += w_; Wt[ix_t[ok_[j]], ix_t[ok_[i]]] += w_
    deg_ = Wt.sum(1); weak_ = deg_ < 0.05 * np.median(deg_); kp_ = np.flatnonzero(~weak_); Wk = Wt[np.ix_(kp_, kp_)]; dk_ = Wk.sum(1)
    _, Vk = np.linalg.eigh(np.eye(len(kp_)) - Wk / np.sqrt(np.outer(dk_, dk_))); side_ = Vk[:, 1] / np.sqrt(dk_) > 0
    if side_[list(kp_).index(ix_t[ref])]:
        side_ = ~side_
    # stitching components: tiles linked by ≥ 1 shared vessel coherent in both (ρ ≥ 0.1); each component is internally stitched,
    # its offset to the reference component is a single unknown β_k.  Main = the component of the reference tile; components of
    # ≥ 3 tiles get β inferred from arterial-inlet symmetry; smaller ones (isolated tiles) are not shown.
    import networkx as _nx
    _H = _nx.Graph(); _H.add_nodes_from(tl_)
    for e_, a_, b_ in pairs:
        _H.add_edge(a_, b_)
    comps = sorted(_nx.connected_components(_H), key=len, reverse=True); main_c = next(c for c in comps if ref in c)
    side_comps = [c for c in comps if c is not main_c and len(c) >= 3]
    block_t = set().union(*side_comps) if side_comps else set(); masked = set(tl_) - main_c - block_t
    core = [t for t in tl_ if t not in masked]
    log(f"  (c) stitching components: main {len(main_c)} tiles, side {[sorted(c) for c in side_comps]}, isolated {sorted(masked)}")
    # β per side component from arterial-inlet symmetry: inlet vessels (≤ 1 hop of a carve inlet) of the side component should carry
    # the same H1 phase as those of the main component (both fed by the two aortic branches, recorded together in tile 14)
    dIn = shortest_path(Ad, directed=False, indices=[int(i) for i in dom["inj"]]).min(0); eIn = np.array([min(dIn[nix_[u]], dIn[nix_[v]]) for u, v in Ef])
    inl = F["elig"] & (F["rho"] >= 0.3) & (eIn <= 1); wt_ = F["tile"]; beta = {}; beta_sd = {}
    rngb = np.random.default_rng(0)
    selM = inl & np.isin(wt_, sorted(main_c)) & np.array([sd.get(int(t), np.inf) <= SD_MAX for t in wt_])
    for k_, c_ in enumerate(side_comps):
        selS = inl & np.isin(wt_, sorted(c_)) & np.array([sd.get(int(t), np.inf) <= SD_MAX for t in wt_])
        zM, zS = np.exp(1j * phf[selM]), np.exp(1j * phf[selS])
        b_ = np.angle(zM.sum() * np.conj(zS.sum())); bs_ = [np.angle(rngb.choice(zM, len(zM)).sum() * np.conj(rngb.choice(zS, len(zS)).sum())) for _ in range(500)]
        sd_ = np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * (np.array(bs_) - b_)))))))
        for t in c_:
            beta[t] = b_
        beta_sd[k_] = (np.degrees(b_), sd_, int(selM.sum()), int(selS.sum()))
        log(f"  (c) side component {sorted(c_)}: β = {np.degrees(b_):+.1f}° ± {sd_:.1f}° (inlet vessels main {selM.sum()}, side {selS.sum()})")
    tdel, tR, nflip, ntot, tvec = {}, {}, {}, {}, {}; eixf = {e: i for i, e in enumerate(Ef)}; Hm_ = mos.shape[0]
    def mid(e):
        pth = G.edges[e].get("path")
        p_ = np.asarray(pth, float)[len(pth) // 2] if pth is not None and len(pth) else np.array([np.mean([G.nodes[n]["x"] for n in e]), np.mean([G.nodes[n]["y"] for n in e])])
        return np.array([p_[0], Hm_ - p_[1]])
    for t in band:
        zs, ws, xy_ = [], [], []
        for e, x in meas.items():
            y = x.get(t)
            if y is None or not np.isfinite(y["rho"]) or y["rho"] < 0.1 or abs(y["H"][0]) == 0:
                continue
            i_ = eixf.get(e)
            if i_ is None or F["Qp"][i_] == 0:
                continue
            nflip[t] = nflip.get(t, 0) + int(np.sign(y["a0"]) != np.sign(F["Qp"][i_])); ntot[t] = ntot.get(t, 0) + 1
            zs.append(np.exp(1j * np.angle(np.sign(F["Qp"][i_]) * y["H"][0] * np.exp(-1j * psi[t])))); ws.append(y["rho"]); xy_.append(mid(e))
        if len(zs) >= 5:
            S = np.sum(np.array(ws) * np.array(zs)); tdel[t] = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(S))); tR[t] = abs(S) / np.sum(ws)
            if len(zs) >= 8:                     # within-tile delay gradient: weighted plane fit, delay relative to the tile mean
                dv_ = np.degrees(np.angle(np.conj(np.array(zs)) * S / abs(S)))           # delay of each vessel minus the tile's (deg)
                X_ = np.array(xy_) - np.average(np.array(xy_), axis=0, weights=ws); w_ = np.array(ws)
                A_ = X_ * np.sqrt(w_)[:, None]; b_ = (dv_ - np.average(dv_, weights=w_)) * np.sqrt(w_)
                g_, *_ = np.linalg.lstsq(A_, b_, rcond=None); res_ = b_ - A_ @ g_
                cov_ = np.linalg.pinv(A_.T @ A_) * (res_ @ res_) / max(len(b_) - 3, 1)
                tvec[t] = (g_, float(np.sqrt(g_ @ np.linalg.pinv(cov_) @ g_)))            # gradient (deg / fused px), its z
    fr = {t: nflip.get(t, 0) / ntot[t] for t in ntot}
    log("  (b) tiles where the measured mean runs against Q′ for > 30 % of vessels: " + (", ".join(f"{t} ({100 * v:.0f}%)" for t, v in sorted(fr.items()) if v > 0.3) or "none"))
    vals = np.array(list(tdel.values()))
    log(f"  (b) boundary H1 phase {np.degrees(phi_b):.0f}° from {nb.sum()} vessels ≤ 1 hop; tile delays {vals.min():.0f}° … {vals.max():.0f}° "
        f"(median {np.median(vals):.0f}°); within-tile R median {np.median(list(tR.values())):.2f}")
    log("  (b) tile delay° (R): " + ", ".join(f"{t}:{tdel[t]:.0f}({tR[t]:.2f})" for t in sorted(tdel)))
    # ---- vessel delays for the maps: every measurement of a carved vessel in a shown tile (core or the weakly linked block,
    #      offset determined), aligned (ψ_t), flow-direction frame (sgn Q′), delay = wrap(φ_zero − φ) in [−30°, 330°)
    from scipy.ndimage import gaussian_filter
    shown_t = {t for t in band if t not in masked and sd.get(t, np.inf) <= SD_MAX}
    def pts(e, step=2.0):
        pth = G.edges[e].get("path")
        p_ = np.asarray(pth, float) if pth is not None and len(pth) >= 2 else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e], float)
        p_ = np.c_[p_[:, 0], Hm_ - p_[:, 1]]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(p_, axis=0).T))]
        ss = np.linspace(0, sl[-1], max(int(sl[-1] / step), 2)); return np.c_[np.interp(ss, sl, p_[:, 0]), np.interp(ss, sl, p_[:, 1])]
    V, Vraw = [], []                                          # (points, delay°, weight, tile)
    for e, x in meas.items():
        i_ = eixf.get(e)
        if i_ is None or F["Qp"][i_] == 0:
            continue
        for t, y in x.items():
            if t not in shown_t or not np.isfinite(y["rho"]) or y["rho"] < 0.1 or abs(y["H"][0]) == 0 or np.sign(y["a0"]) != np.sign(F["Qp"][i_]):
                continue                                      # (a measurement against the reconciled flow direction is not used, as in the field)
            dl = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(np.sign(F["Qp"][i_]) * y["H"][0] * np.exp(-1j * psi[t]) * np.exp(1j * beta.get(t, 0.0)))))
            V.append((pts(e), (dl + 30) % 360 - 30, y["rho"], t))
            dr = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(np.sign(F["Qp"][i_]) * y["H"][0])))   # as recorded (no ψ_t)
            Vraw.append((V[-1][0], (dr + 30) % 360 - 30, y["rho"], t))
    tdel = {t: (v + 30) % 360 - 30 for t, v in tdel.items()}
    dmax = max(v for t, v in tdel.items() if t in shown_t); vtop = 20 * np.ceil(np.percentile([v[1] for v in V], 98) / 20)
    cmp = plt.cm.cividis_r; nph = mcolors.Normalize(0, vtop); LEV = np.arange(0, vtop + 1, 20)

    def smooth_field(vs, sigma_um, step_px, box):
        """Circular kernel smoothing of vessel delays on a grid: Z = G_σ * Σ w e^{iδ}; masked where the kernel weight is small."""
        (x0, x1), (y0, y1) = box; nx, ny = int((x1 - x0) / step_px) + 1, int((y1 - y0) / step_px) + 1
        Z = np.zeros((ny, nx), complex); Wg = np.zeros((ny, nx)); Wb = np.zeros((ny, nx))
        for p_, dl, w, t in vs:
            ix_ = np.clip(((p_[:, 0] - x0) / step_px).astype(int), 0, nx - 1); iy_ = np.clip(((p_[:, 1] - y0) / step_px).astype(int), 0, ny - 1)
            np.add.at(Z, (iy_, ix_), w * np.exp(1j * np.radians(dl))); np.add.at(Wg, (iy_, ix_), w)
            if t in block_t:
                np.add.at(Wb, (iy_, ix_), w)
        sg = sigma_um / FUSE_UM_PX / step_px
        Zs = gaussian_filter(Z.real, sg) + 1j * gaussian_filter(Z.imag, sg); Ws = gaussian_filter(Wg, sg); Wbs = gaussian_filter(Wb, sg)
        ok = Ws > 0.15 * np.percentile(Ws[Ws > 0], 75)
        dd = (np.degrees(np.angle(Zs)) + 30) % 360 - 30
        return np.ma.masked_where(~ok, dd), np.ma.masked_where(~ok, Wbs / np.maximum(Ws, 1e-12)), (x0, x1, y1, y0)

    fig = figS
    # ---- b: one overlapping tile pair of the core — measured vessel delays and the smoothed isochrones across the seam
    cnt = {}
    for e, a, b in pairs:
        if a in shown_t and b in shown_t and a not in block_t and b not in block_t:
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    (ta, tb), npair_ = max(cnt.items(), key=lambda kv: kv[1])
    pp = np.vstack([polys[ta], polys[tb]]); box = ((pp[:, 0].min(), pp[:, 0].max()), (pp[:, 1].min(), pp[:, 1].max()))
    x0, x1 = box[0]; y0, y1 = box[1]; TCOL = {ta: "#0072B2", tb: "#D55E00"}
    # each vessel shown with the phase from the tile whose centre is nearer (a seam through the overlap); colour = H1 phase (cyclic)
    ca, cb_ = cen[ta], cen[tb]; nrm_s = (cb_ - ca) / np.linalg.norm(cb_ - ca); mid_s = (ca + cb_) / 2
    side = lambda p_: np.dot(p_.mean(0) - mid_s, nrm_s)
    cmc = plt.cm.twilight_shifted; ncy = mcolors.Normalize(-180, 180)
    gBs = GB.subgridspec(3, 1, height_ratios=[1, 1, 0.36], hspace=0.16)
    for j_, (VV, ttl) in enumerate(((Vraw, "before alignment"), (V, "after alignment: one offset per tile"))):
        ax = fig.add_subplot(gBs[j_, 0])
        ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.3, zorder=0)
        for p_, dl, w, t in VV:
            if t not in (ta, tb) or (side(p_) < 0) != (t == ta):
                continue
            ph_ = ((dl + 180) % 360) - 180
            ax.plot(p_[:, 0], p_[:, 1], color=cmc(ncy(ph_)), lw=1.5, solid_capstyle="round", zorder=3)
        sx = np.array([-nrm_s[1], nrm_s[0]]); L_ = 2 * max(x1 - x0, y1 - y0)
        ax.plot([mid_s[0] - L_ * sx[0], mid_s[0] + L_ * sx[0]], [mid_s[1] - L_ * sx[1], mid_s[1] + L_ * sx[1]], color="black", lw=1.0, ls=":", zorder=4)
        for t in (ta, tb):
            ax.add_patch(Polygon(polys[t], closed=True, fc="none", ec=TCOL[t], lw=1.0, ls="--", zorder=5))
            if j_ == 0:
                ax.text(*(polys[t].mean(0) + 0.42 * (polys[t][0] - polys[t].mean(0))), f"tile {t}", color=TCOL[t], fontsize=7, fontweight="bold", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("0.5")
        ax.text(0.02, 0.97, ttl, transform=ax.transAxes, fontsize=7.5, va="top", ha="left", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    axb0, axb1 = fig.axes[-2], fig.axes[-1]; p0_, p1_ = axb0.get_position(), axb1.get_position()
    xm_ = p0_.x0 + 0.5 * p0_.width
    fig.patches.append(matplotlib.patches.FancyArrowPatch((xm_, p0_.y0 + 0.002), (xm_, p1_.y1 - 0.002),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, lw=1.0, color="black", zorder=10))
    fig.text(xm_ + 0.012, (p0_.y0 + p1_.y1) / 2, r"$-\psi_t$", ha="left", va="center", fontsize=8, zorder=10)
    caxb = fig.add_subplot(gBs[2, 0]); caxb.axis("off")
    cbi = caxb.inset_axes([0.15, 0.12, 0.7, 0.12]); cbb = fig.colorbar(plt.cm.ScalarMappable(norm=ncy, cmap=cmc), cax=cbi, orientation="horizontal", ticks=[-180, 0, 180])
    cbb.ax.tick_params(labelsize=6.5); cbb.outline.set_linewidth(0.4); cbb.set_label("H1 phase (deg)", fontsize=7, labelpad=1)
    bb = GB.get_position(fig)
    fig.text(bb.x0 + 0.5 * bb.width, bb.y0 + 0.105 * bb.height, r"$\phi^{\rm aligned}_{e,t} = \phi^{\rm obs}_{e,t} - \psi_t$", ha="center", fontsize=8.5)

    log(f"  (b) tile pair {ta}/{tb} ({npair_} shared coherent vessels)")
    label(GB, "a")

    # ---- c: organ-scale smoothed delay field with isochrones; the second stitching component dashed, offset β from inlet symmetry
    axC = fig.add_subplot(GC); organ(axC)
    big = ((XL[0], XL[1]), (YL[1], YL[0]))
    fo, fb, ext = smooth_field(V, 150.0, 6.0, big)
    gx = np.linspace(big[0][0], big[0][1], fo.shape[1]); gy = np.linspace(big[1][0], big[1][1], fo.shape[0])
    axC.imshow(fo, cmap=cmp, norm=nph, extent=ext, origin="upper", alpha=0.9, zorder=1, interpolation="bilinear")
    cs = axC.contour(gx, gy, fo, levels=LEV, colors="0.1", linewidths=0.45, zorder=3)
    _cl = axC.clabel(cs, levels=[v for v in LEV if v in (40, 80, 160)], fmt="%d°", fontsize=6.5, inline_spacing=1); _seen = set()
    for t_ in _cl:                                        # one label per level
        if t_.get_text() in _seen:
            t_.set_visible(False)
        _seen.add(t_.get_text())
    plt.rcParams["hatch.linewidth"] = 0.3; plt.rcParams["hatch.color"] = "0.5"
    fbf = np.ma.filled(fb, 0)
    axC.contour(gx, gy, fbf, levels=[0.5], colors="0.15", linewidths=0.9, linestyles="--", zorder=4)
    if beta_sd:
        bdeg, bsd, _, _ = beta_sd[0]; ys_, xs_ = np.nonzero(fbf > 0.5)
        axC.text(gx[int(np.median(xs_))], gy[int(np.percentile(ys_, 72))], "second stitching\ncomponent",
                 ha="center", va="center", fontsize=7, color="0.15", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    Lt = {t: 0.0 for t in band}
    for (a_, b_), w_ in link.items():
        Lt[a_] += w_; Lt[b_] += w_
    L_thr = 0.05 * np.median(list(Lt.values())); weak_t = {t for t in band if Lt[t] < L_thr}
    log(f"  (c) weakly constrained tiles (link < 5 % of median, {L_thr:.2f}): {sorted(weak_t)}")
    # low-pulsatility region from Fig 7 (smoothed union of the weakly linked tiles ∩ organ, minus stitchable coverage)
    plt.rcParams["hatch.color"] = "0.45"; plt.rcParams["hatch.linewidth"] = 0.5
    axC.contourf(gxs, gys, mL, levels=[0.5, 2], colors=["white"], alpha=0.6, zorder=4.4)
    axC.contourf(gxs, gys, mL, levels=[0.5, 2], colors=["none"], hatches=["/////"], zorder=4.5)
    axC.contour(gxs, gys, mL, levels=[0.5], colors=["0.45"], linewidths=0.7, zorder=4.6)
    for t in band:
        if t not in shown_t and t not in weak_t:
            axC.plot(*cen[t], "o", ms=3.5, mfc="white", mec="0.55", mew=0.7, zorder=5)
    refpts = [(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"]) for n_ in list(dom["DA"]) + [nds[int(i)] for i in dom["out"]]]
    for n_ in dom["DA"]:
        axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="v", ms=5.5, color="white", mec="black", mew=0.6, zorder=6)
    for i in [int(i) for i in dom["out"]]:
        n_ = nds[i]; axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="*", ms=7, color="white", mec="black", mew=0.6, zorder=6)
    rp = np.array(refpts); axC.annotate("simultaneous DA/V\nreference, 0°", xy=(rp[:, 0].mean(), rp[:, 1].min() - 20),
                                        xytext=(rp[:, 0].mean() + 0.30 * (XL[1] - XL[0]), YL[1] + 0.02 * (YL[0] - YL[1])), ha="center", va="top", fontsize=8,
                                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), zorder=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axC.legend(handles=[Patch(fc="none", ec="0.15", ls="--", label="second stitching component"),
                        Patch(fc="white", ec="0.45", hatch="//////", lw=0.5, label="offset not constrained"),
                        Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="0.55", label="tile phase undetermined")],
               loc="upper left", bbox_to_anchor=(-0.04, 1.0), fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0)
    axC.add_patch(Polygon(polys[ta], closed=True, fc="none", ec="#0072B2", lw=0.8, ls="--", zorder=5))
    axC.add_patch(Polygon(polys[tb], closed=True, fc="none", ec="#D55E00", lw=0.8, ls="--", zorder=5))
    sm = plt.cm.ScalarMappable(norm=nph, cmap=cmp)
    cax = axC.inset_axes([0.2, -0.05, 0.6, 0.028]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=LEV[::3], extend="min")
    cb.set_label("H1 phase lag behind simultaneous DA/V reference (deg)", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
    _pC = axC.get_position(); fig.text(_pC.x0 - 0.01, _pC.y1 + 0.03, "c", fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- e: repeat-vessel agreement after alignment; H2 is an independent check (not used in the synchronization)
    gDs = GD.subgridspec(1, 2, wspace=0.3); bins = np.linspace(-60, 60, 25)
    for c_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gDs[c_]); v = d[k]["aligned"]
        ax.hist(v, bins=bins, density=True, color="#8fb0d3" if k == 1 else "#c9b6d9", ec="white", lw=0.3)
        ax.set_xlim(-60, 60); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-60, 0, 60])
        ax.axvline(0, color="0.3", lw=0.6, zorder=3); ax.tick_params(axis="x", labelsize=7)
        ax.text(0.0, 1.0, f"H{k}", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.text(1.0, 1.0, f"median $|\\Delta\\phi|$ = {np.median(np.abs(v)):.0f}°", transform=ax.transAxes, fontsize=7, va="bottom", ha="right", color="0.2")
        if k == 2:
            ax.text(0.5, 0.82, "not used in\nsynchronization", transform=ax.transAxes, fontsize=6.5, ha="center", va="top", color="#6a4c93", fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    fig.text(GD.get_position(fig).x0 + 0.5 * GD.get_position(fig).width, GD.get_position(fig).y0 - 0.045,
             "repeat-vessel phase difference, aligned (deg)", ha="center", va="top", fontsize=7.5)
    label(GD, "b")



    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # supplement — held-out overlap prediction (fig7_supp_heldout)
    fs = plt.figure(figsize=(2.6, 2.6), facecolor="white"); axP = fs.add_axes([0.24, 0.22, 0.72, 0.72]); axP.set_aspect("equal")
    axP.plot([-200, 200], [-200, 200], color="0.6", lw=0.7, ls="--", zorder=1)
    axP.scatter(obs, pred, s=4 + 18 * np.sqrt(nn / nn.max()), color="0.25", alpha=0.75, lw=0, zorder=3)
    axP.set_xlim(-200, 200); axP.set_ylim(-200, 200); axP.set_xticks([-180, 0, 180]); axP.set_yticks([-180, 0, 180])
    axP.set_xlabel("tile-pair offset measured\nfrom shared vessels (deg)"); axP.set_ylabel("predicted with pair held out (deg)", labelpad=0)
    axP.spines[["top", "right"]].set_visible(False)
    axP.text(0.03, 0.97, f"{len(obs)} tile pairs\nmedian error {np.median(res):.0f}°", transform=axP.transAxes, ha="left", va="top", fontsize=7.5, color="0.3", linespacing=1.4)
    fs.savefig(OUT_DIR / f"fig7_supp_heldout{args.tag}.png", dpi=300, facecolor="white"); fs.savefig(OUT_DIR / f"fig7_supp_heldout{args.tag}.pdf", facecolor="white")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figSM.savefig(OUT_DIR / f"fig7_supp_shape_map{args.tag}.png", dpi=300, facecolor="white"); figSM.savefig(OUT_DIR / f"fig7_supp_shape_map{args.tag}.pdf", facecolor="white")
    for f_, nm_ in ((figR, "fig7_regime"), (figS, "fig8_sync")):
        f_.savefig(OUT_DIR / f"{nm_}{args.tag}.png", dpi=300, facecolor="white"); f_.savefig(OUT_DIR / f"{nm_}{args.tag}.pdf", facecolor="white")
        f_.savefig(OUT_DIR / f"{nm_}{args.tag}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        log(f"wrote {OUT_DIR / f'{nm_}{args.tag}.png'}")


if __name__ == "__main__":
    main()
