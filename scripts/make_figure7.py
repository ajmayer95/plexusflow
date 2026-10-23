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
    # Fig 7 = the two requirements for the phase-aligned composite, as two blocks.
    #   stationarity (top): a heart rate vs acquisition time · b repeat-vessel waveforms, transient vs retained · c D distributions
    #   phase support (bottom, retained only): d organ map, fundamental detected (|H1|) or not · e detection vs |H1|
    # Fig 7 = can asynchronously acquired waveforms be treated as measurements of the same pulsatile state?
    #   a heart rate vs acquisition time · b repeat-vessel examples · c waveform-shape difference D · d clock-free shape phase Θ21
    # Local measurability of the fundamental (organ map + detection vs |H1|) is a measurement-method question → SI (fig7_supp_detection)
    # layout: col 0 = a heart rate (top) over d D (bottom) · col 1 = b organ split into the two regimes, example vessels magnified ·
    # col 2 = c the two examples' waveforms, stacked
    # three full-height columns: a heart rate (acquisition time running down) · b organ, transient vs retained shaded, four example
    # vessels · right: c two transient examples (top), d D distributions with the examples marked (middle), two retained examples (bottom)
    # left: a heart rate (short, horizontal) over d D distributions · centre: b organ · right: c 2×2 repeat-vessel examples
    # top (quantitative): a heart rate vs acquisition time · b D distributions; bottom (concrete): c organ with vessels 1–4 · d their waveforms
    figR = plt.figure(figsize=(7.5, 5.6), facecolor="white")
    gR = figR.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[0.58, 1], wspace=0.24, hspace=0.5,
                           left=0.075, right=0.985, top=0.95, bottom=0.08)
    gRx = gR[1, 1].subgridspec(2, 2, wspace=0.1, hspace=0.55)
    GA, GT, GB7 = gR[0, 0], gR[0, 1], gR[1, 0]; GX = [gRx[0, 0], gRx[0, 1], gRx[1, 0], gRx[1, 1]]
    figT = plt.figure(figsize=(3.6, 2.9), facecolor="white")       # SI: clock-free Θ21 check (fig7_supp_theta21)
    GTH = figT.add_gridspec(1, 1, left=0.17, right=0.97, top=0.95, bottom=0.17)[0]
    figD = plt.figure(figsize=(7.5, 3.6), facecolor="white")
    gDd = figD.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25, left=0.06, right=0.975, top=0.9, bottom=0.14)
    GMAP, GK7 = gDd[0], gDd[1]
    figS = plt.figure(figsize=(7.5, 5.9), facecolor="white")
    gS2 = figS.add_gridspec(1, 2, width_ratios=[0.45, 1], wspace=0.06, left=0.08, right=0.98, top=0.93, bottom=0.1)
    gL = gS2[0].subgridspec(2, 1, height_ratios=[3.0, 0.8], hspace=0.42); GB, GD, GC = gL[0], gL[1], gS2[1]
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
    # regimes per tile, from measurement properties only (no stitching yet): transient = heart rate below the band (the first
    # recordings); low pulsatility = band tiles with fewer than N_MEAS vessels whose H1 phase is measurable (coherence ρ ≥ 0.2);
    # retained = the other band tiles.  (Identical split to the synchronization link weight < 5 % of median: low tiles have ≤ 11
    # phase-measurable vessels, all others ≥ 17.)
    Fall = build_ac_field(G, dc, n_boot=0, f0_band=10.0); mall = Fall["_meas"]
    early = {t for t in ts if t not in band and tf0[t] < f_ref * (1 - BAND)}; N_MEAS = 15
    nmeas = {t: sum(1 for x in mall.values() if t in x and np.isfinite(x[t]["rho"]) and x[t]["rho"] >= 0.2) for t in ts}
    # regimes are temporal only: transient (early) vs retained (all band tiles).  Whether a vessel's phase is measurable is a
    # local, per-vessel property (panels a, c), not a tile class: one tile can hold a detected vein, pulsatile plexus and undetected
    # small vessels.  Whether a tile's clock is identifiable is decided in Fig 8.
    lowp = set(); stitch = set(band)
    # the early-tile footprint is transient as a whole: vessels whose midpoint lies inside it are left out of the retained displays
    # and statistics even where a retained tile's edge also recorded them (8 vessels, tiles 14 / 15); the synchronization is unchanged
    from matplotlib.path import Path as _MPath
    def _mid(e_):
        pth = G.edges[e_].get("path"); q_ = np.asarray(pth, float)[len(pth) // 2] if pth is not None and len(pth) else np.array(
            [np.mean([G.nodes[n]["x"] for n in e_]), np.mean([G.nodes[n]["y"] for n in e_])]); return (q_[0], Hm - q_[1])
    _ep = [_MPath(polys[t]) for t in early]
    ezone_e = {e_ for e_ in mall if G.has_edge(*e_) and any(pp_.contains_point(_mid(e_)) for pp_ in _ep)}
    log(f"  early-tile footprint: {len(ezone_e)} vessels")
    RCOL = {"T": "#b23a48", "L": "#8a8a8a", "S": "#2b5c8a"}
    RLAB = {"T": "transient", "L": "", "S": "retained"}
    # detection of the fundamental above broadband noise, per measurement: F-test of the 3-harmonic fit (ρ = explained-variance
    # fraction of n samples, 6 regressors); detected = p < P_DET
    from scipy.stats import f as _Fd
    P_DET = 0.01; _nsamp = {}
    for e_ in G.edges:
        for m_ in G.edges[e_].get("measurements_piv") or []:
            if m_.get("Q_t") is not None:
                _nsamp[(e_, int(m_["tile_id"]))] = _nsamp[(e_[::-1], int(m_["tile_id"]))] = len(m_["Q_t"])
    def _pdet(e_, t_, y_):
        n_ = _nsamp.get((e_, t_)); r_ = min(max(y_["rho"], 1e-12), 1 - 1e-12)
        return float(_Fd.sf((r_ / 6) / ((1 - r_) / (n_ - 7)), 6, n_ - 7)) if n_ and n_ > 7 else 1.0
    reg = lambda t: "T" if t in early else ("L" if t in lowp else ("S" if t in stitch else None))
    log(f"  regimes: transient {sorted(early)}; low pulsatility {sorted(lowp)}; retained {len(stitch)} tiles")

    # ---- a: heart rate over the acquisition (time runs down), coloured by regime
    for k_ in ("T", "S"):
        m_ = np.array([reg(t) == k_ for t in ts])
        axA.scatter(xs_a[m_], f[m_], s=10, color=RCOL[k_], edgecolors="white", linewidths=0.4, zorder=3, label=RLAB[k_])
    # the retained set is the heart-rate band (f0 within ±21 % of the reference recording); b–d show it is a reproducible regime
    _te = max(xs_a[ts.index(t)] for t in early); _tr = min(xs_a[ts.index(t)] for t in ts if reg(t) == "S")
    axA.axvline((_te + _tr) / 2, color="0.5", lw=0.7, ls=":")
    axA.text((_te + _tr) / 2 + 6, 1.0, "end of\ntransient", fontsize=8, color="0.4", va="bottom", ha="left", linespacing=0.9)
    axA.set_xlabel(xlab, fontsize=8, labelpad=1); axA.set_ylabel("$f_0$ (Hz)", fontsize=8, labelpad=1); axA.set_yticks([1, 2, 3])
    axA.tick_params(labelsize=7); axA.spines[["top", "right"]].set_visible(False)
    axA.legend(loc="lower right", fontsize=8, frameon=False, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15)
    label(GA, "a")

    fig = figD
    # ---- c: phase is measurable where the local pulsatile signal is large enough — per vessel recording (retained tiles):
    #      fraction with the fundamental detected, and the phase uncertainty of detected ones, vs the fundamental's amplitude
    DV_ = []
    for e_, x_ in mall.items():
        if e_ in ezone_e:
            continue
        for t_, y_ in x_.items():
            if t_ in stitch and np.isfinite(y_["rho"]) and abs(y_["H"][0]) > 0:
                vp_ = y_["var_phi"][0]
                DV_.append((abs(y_["H"][0]), _pdet(e_, t_, y_) < P_DET, np.degrees(np.sqrt(vp_)) if vp_ is not None and np.isfinite(vp_) else np.nan))
    DV_ = np.array(DV_, float); hb_ = np.logspace(np.log10(np.percentile(DV_[:, 0], 1)), np.log10(np.percentile(DV_[:, 0], 99.5)), 13)
    hm_, fd_, ps_ = [], [], []
    for lo_, hi_ in zip(hb_[:-1], hb_[1:]):
        m_ = (DV_[:, 0] >= lo_) & (DV_[:, 0] < hi_)
        if m_.sum() >= 15:
            hm_.append(np.sqrt(lo_ * hi_)); fd_.append(DV_[m_, 1].mean()); dd_ = m_ & (DV_[:, 1] == 1)
            ps_.append(np.nanmedian(DV_[dd_, 2]) if dd_.sum() >= 5 else np.nan)
    axK = fig.add_subplot(GK7)
    axK.plot(hm_, np.array(fd_) * 100, "o-", color=RCOL["S"], ms=3.5, lw=1.2, label="detected above noise (%, left)")
    axK.set_xscale("log"); axK.set_ylim(0, 102); axK.set_yticks([0, 50, 100])
    axK.set_xlabel("amplitude of the fundamental $|H_1|$ (nL/s)", fontsize=7.5, labelpad=1)
    axK.set_ylabel("vessels with measurable phase\nof the fundamental (%)", fontsize=7.5, labelpad=1); axK.tick_params(labelsize=7)
    axK.spines[["top", "right"]].set_visible(False)
    i50_ = next((i_ for i_, v_ in enumerate(fd_) if v_ >= 0.5), None)
    if i50_ is not None:
        axK.axvline(hm_[i50_], color="0.7", lw=0.6, ls=":")
        axK.text(hm_[i50_] * 1.08, 40, f"50 % at {hm_[i50_]:.3f} nL/s", fontsize=6.3, color="0.45")
    log(f"  (c) {len(DV_)} retained recordings; detected {DV_[:, 1].mean():.0%}; detection vs |H1|: " + ", ".join(f"{h:.3g}:{v:.0%}" for h, v in zip(hm_, fd_)))
    label(GK7, "b"); fig = figR

    # waveform-shape spectrum of one recording, mean-normalised and clock-free: s_n = H_n e^{−inφ1} / Q̄ (n = 1..3, flow-direction frame).
    # Shape difference of two recordings D = ‖s_a − s_b‖ / rms(‖s_a‖, ‖s_b‖) — the relative RMS difference of the H1-aligned Q/Q̄
    # waveforms (Parseval); it carries the harmonic amplitudes |H_n|/Q̄ and the inter-harmonic phases Θ21, Θ31 together.
    def _shape(y_):
        H_ = np.asarray(y_["H"][:3]) * np.sign(y_["a0"]); u_ = np.conj(H_[0] / abs(H_[0]))
        return H_ * u_ ** np.arange(1, 4) / abs(y_["a0"])
    def _D(ya, yb):
        sa, sb = _shape(ya), _shape(yb); return np.linalg.norm(sa - sb) / np.sqrt((np.linalg.norm(sa) ** 2 + np.linalg.norm(sb) ** 2) / 2)
    SP = {"E": [], "B": []}; vD = {"E": {}, "B": {}}; TH = {"E": [], "B": []}
    for e_, x_ in mall.items():
        ok_ = [t_ for t_, y_ in x_.items() if np.isfinite(y_["rho"]) and y_["rho"] >= 0.2 and abs(y_["a0"]) > 0 and abs(y_["H"][0]) > 0]
        for i_ in range(len(ok_)):
            for j_ in range(i_ + 1, len(ok_)):
                a_, b_ = ok_[i_], ok_[j_]; ya, yb = x_[a_], x_[b_]
                g_ = "E" if (a_ in early and b_ in early) else ("B" if (a_ in stitch and b_ in stitch and e_ not in ezone_e) else None)
                if g_ is None or np.sign(ya["a0"]) != np.sign(yb["a0"]):
                    continue
                d_ = _D(ya, yb); SP[g_].append(d_); vD[g_].setdefault(e_, []).append(d_)
                _th = lambda y_: np.angle(y_["H"][1] * np.conj(y_["H"][0]) ** 2)
                TH[g_].append((abs(np.degrees(np.angle(np.exp(1j * (_th(ya) - _th(yb)))))), min(ya["rho"], yb["rho"])))
    SP = {k: np.array(v) for k, v in SP.items()}; SCOL = {"E": RCOL["T"], "B": RCOL["S"]}
    SLAB = {"E": "transient", "B": "retained"}
    for g_ in ("E", "B"):
        log(f"  (e) repeat-vessel shape difference D, both {SLAB[g_]}: {len(SP[g_])} pairs / {len(vD[g_])} vessels, median {np.median(SP[g_]):.2f}, "
            f"90th {np.percentile(SP[g_], 90):.2f}")

    # ---- d: one vessel per regime, Q/Q̄ over two beats (raw trace light, 3-harmonic fit bold); recordings shifted to their own H1 phase
    from plexusflow.reconcile.harmonics import phasors as _phz
    EXV, EXA, EXB = (1206, 1267), 3, 5                     # transient: tiles 3 → 5, 12 min apart
    LPV, LPT, LPR = (12024, 12102), 16, 15                  # low pulsatility: one vessel, clear in retained tile 15, lost in low tile 16
    RV = (49550, 49649); RVe = RV if RV in mall else RV[::-1]; xr_ = mall[RVe]
    rv_t = sorted([t_ for t_, y_ in xr_.items() if t_ in stitch and np.isfinite(y_["rho"]) and y_["rho"] >= 0.5], key=lambda t_: xs_a[ts.index(t_)])
    # example 3: retained, other half of the organ, recordings 40 min apart (sandbox/find_example3.py)
    RV3 = (71514, 71560); RV3e = RV3 if RV3 in mall else RV3[::-1]; xr3_ = mall[RV3e]   # intermediate D (sandbox/find_retained_mid.py)
    rv3_t = sorted([t_ for t_, y_ in xr3_.items() if t_ in stitch and np.isfinite(y_["rho"]) and y_["rho"] >= 0.5], key=lambda t_: xs_a[ts.index(t_)])
    EXe = EXV if EXV in mall else EXV[::-1]
    def _trace(ax_, E_, t_, col_, lab_):
        e_ = E_ if G.has_edge(*E_) else E_[::-1]
        m_ = next(m for m in G.edges[e_]["measurements_piv"] if int(m["tile_id"]) == t_); q_ = np.asarray(m_["Q_t"], float); q_ = -q_ if np.mean(q_) < 0 else q_
        a0_, H_ = _phz(q_, tf0[t_]); tt_ = np.arange(len(q_)) / 250.0; tau_ = tf0[t_] * tt_ + np.angle(H_[0]) / (2 * np.pi)
        tau_ = tau_ - np.floor(tau_[0]); sel_ = (tau_ >= 0) & (tau_ <= 2); tg_ = np.linspace(0, 2, 300)
        fit_ = a0_ + sum((abs(H_[n - 1]) * np.exp(1j * (np.angle(H_[n - 1]) - n * np.angle(H_[0]))) * np.exp(2j * np.pi * n * tg_)).real for n in (1, 2, 3))
        ax_.plot(tau_[sel_], q_[sel_] / a0_, color=col_, lw=0.3, alpha=0.1, zorder=2)
        ax_.plot(tg_, fit_ / a0_, color=col_, lw=1.3, zorder=3, label=lab_)
        return a0_, H_
    T2V, T2A, T2B = (689, 857), 2, 3                        # second transient vessel: tiles 2 → 3, both ρ ≥ 0.94 (sandbox/find_transient2.py)
    T2e = T2V if T2V in mall else T2V[::-1]
    axs_d = [fig.add_subplot(g_) for g_ in GX]
    _trace(axs_d[0], EXV, EXA, "#b23a48", f"{xs_a[ts.index(EXA)]:.0f} min"); _trace(axs_d[0], EXV, EXB, "#e08a96", f"{xs_a[ts.index(EXB)]:.0f} min")
    _trace(axs_d[1], T2V, T2A, "#b23a48", f"{xs_a[ts.index(T2A)]:.0f} min"); _trace(axs_d[1], T2V, T2B, "#e08a96", f"{xs_a[ts.index(T2B)]:.0f} min")
    for t_, c_ in zip(rv_t, plt.cm.viridis(np.linspace(0.05, 0.8, len(rv_t)))):
        _trace(axs_d[2], RV, t_, c_, f"{xs_a[ts.index(t_)]:.0f}")
    for t_, c_ in zip(rv3_t, plt.cm.viridis(np.linspace(0.05, 0.8, len(rv3_t)))):
        _trace(axs_d[3], RV3, t_, c_, f"{xs_a[ts.index(t_)]:.0f}")
    dEX = _D(mall[EXe][EXA], mall[EXe][EXB]); dT2 = _D(mall[T2e][T2A], mall[T2e][T2B])
    dRV = [_D(xr_[a_], xr_[b_]) for i_, a_ in enumerate(rv_t) for b_ in rv_t[i_ + 1:]]
    dRV3 = [_D(xr3_[a_], xr3_[b_]) for i_, a_ in enumerate(rv3_t) for b_ in rv3_t[i_ + 1:]]
    _sp = lambda tl_: xs_a[ts.index(tl_[-1])] - xs_a[ts.index(tl_[0])]
    ttl_ = [("1", "T", f"transient · D = {dEX:.2f}"), ("2", "T", f"transient · D = {dT2:.2f}"),
            ("3", "S", f"retained · D ≤ {max(dRV):.2f}"), ("4", "S", f"retained · D {'=' if len(dRV3) == 1 else '≤'} {max(dRV3):.2f}")]
    for k_, ax_ in enumerate(axs_d):
        tag_, cl_, sub_ = ttl_[k_]
        ax_.axhline(1, color="0.75", lw=0.4, ls=":"); ax_.axhline(0, color="0.4", lw=0.5); ax_.set_xlim(0, 2); ax_.set_xticks([0, 1, 2])
        ax_.set_ylim(-1.2, 2.6); ax_.set_yticks([0, 1, 2]); ax_.tick_params(labelsize=6, pad=1)
        if k_ in (1, 3):
            ax_.set_yticklabels([])
        else:
            ax_.set_ylabel(r"$Q/\bar Q$", fontsize=8, labelpad=0)
        if k_ in (0, 1):
            ax_.set_xticklabels([])
        if k_ == 2:
            ax_.set_xlabel("beats (aligned on the fundamental)", fontsize=8, labelpad=1, x=1.04)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.text(-0.02, 1.06, tag_, transform=ax_.transAxes, fontsize=6.5, fontweight="bold", color="white", va="bottom", ha="right",
                 bbox=dict(boxstyle="circle,pad=0.15", fc=RCOL[cl_], ec="none"))
        ax_.text(0.06, 1.03, sub_, transform=ax_.transAxes, fontsize=8, color=RCOL[cl_], va="bottom", linespacing=0.95)
        pass
    log(f"  (c) examples: transient D {dEX:.2f}, {dT2:.2f}; retained max D {max(dRV):.2f}, {max(dRV3):.2f}")
    _q0, _q1 = axs_d[0].get_position(), axs_d[1].get_position()
    fig.text((_q0.x0 + _q1.x1) / 2, _q0.y1 + 0.07, "same vessel measured repeatedly", fontsize=8, ha="center", va="bottom", fontweight="bold", color="0.2")
    # the examples belong to panel c (bottom row = map + examples); no separate letter

    # ---- e: across all repeat vessels — D, both recordings transient vs both stitchable; examples 1 and 3 marked
    axD = fig.add_subplot(GT); ecdf_ = {}
    for g_ in ("E", "B"):
        v_ = np.sort(SP[g_]); ecdf_[g_] = v_
        axD.step(v_, np.arange(1, len(v_) + 1) / len(v_), where="post", color=SCOL[g_], lw=1.3, label=f"{SLAB[g_]} ({len(v_)})")
        m_ = np.median(v_); axD.plot([m_], [0.5], "o", ms=3, color=SCOL[g_], zorder=4)
        axD.plot([m_, m_], [0, 0.5], color=SCOL[g_], lw=0.6, ls=":", zorder=2)
        axD.text(m_, 0.5, f" {m_:.2f}", color=SCOL[g_], fontsize=8, ha="left", va="top" if g_ == "E" else "bottom")
    _on = lambda g_, x_: np.searchsorted(ecdf_[g_], x_, side="right") / len(ecdf_[g_])
    for g_, xs_, tag_ in (("E", [dEX], "1"), ("E", [dT2], "2"), ("B", dRV, "3"), ("B", dRV3, "4")):
        for x_ in xs_:
            axD.plot(x_, _on(g_, x_), "o", ms=3.5, mfc="white", mec=SCOL[g_], mew=0.9, zorder=5)
        xm_ = np.median(xs_); axD.annotate(tag_, xy=(xm_, _on(g_, xm_)), xytext=(xm_ + {"1": -0.1, "2": 0.07, "3": -0.04, "4": 0.12}[tag_], _on(g_, xm_) + {"1": 0.05, "2": 0.12, "3": 0.3, "4": -0.06}[tag_]),
                                           fontsize=7, fontweight="bold", color=SCOL[g_], ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=SCOL[g_]))
    axD.set_xlim(0, 1.1); axD.set_ylim(0, 1.02); axD.set_yticks([0, 0.5, 1]); axD.spines[["top", "right"]].set_visible(False)
    axD.set_xlabel("waveform-shape difference $D$", fontsize=8, labelpad=1)
    axD.text(0.0, 1.06, "← more reproducible", transform=axD.transAxes, fontsize=8, color="0.4", va="bottom")
    axD.text(0.98, 0.26, f"retained ({len(SP['B'])})", color=SCOL["B"], fontsize=8, transform=axD.transAxes, va="bottom", ha="right")
    axD.text(0.98, 0.1, f"transient ({len(SP['E'])})", color=SCOL["E"], fontsize=8, transform=axD.transAxes, va="bottom", ha="right")
    axD.set_ylabel("cum. fraction", fontsize=8, labelpad=0); axD.tick_params(labelsize=7)
    _pd = axD.get_position(); fig.text(_pd.x0 - 0.06, _pd.y1 + 0.03, "b", fontsize=10, fontweight="bold", va="bottom")
    pass

    fig = figT
    # ---- d: clock-free check — the shape phase Θ21 = φ2 − 2φ1 is unchanged by any tile clock offset (φn → φn + nψ)
    axTH = fig.add_subplot(GTH)
    for g_, ls_, sel_, lab_ in (("E", "-", 0.2, "transient"), ("B", "-", 0.2, "retained"), ("B", "--", 0.5, "retained, both ρ ≥ 0.5")):
        a_ = np.array(TH[g_]); v_ = np.sort(a_[a_[:, 1] >= sel_, 0])
        axTH.step(v_, np.arange(1, len(v_) + 1) / len(v_), where="post", color=SCOL[g_], lw=1.3 if ls_ == "-" else 1.0, ls=ls_,
                  label=f"{lab_} (median {np.median(v_):.0f}°)")
        log(f"  (d) |ΔΘ21| {lab_}: n {len(v_)}, median {np.median(v_):.0f}°")
    axTH.set_xlim(0, 180); axTH.set_xticks([0, 45, 90, 135, 180]); axTH.set_ylim(0, 1.02); axTH.set_yticks([0, 0.5, 1])
    axTH.set_xlabel(r"change in shape phase $\Theta_{21}=\phi_2-2\phi_1$ (deg)", fontsize=7.5, labelpad=1)
    axTH.set_ylabel("cumulative fraction", fontsize=8, labelpad=1); axTH.tick_params(labelsize=7); axTH.spines[["top", "right"]].set_visible(False)
    axTH.legend(title="repeat vessel, both recordings", title_fontsize=6.3, fontsize=6.3, frameon=False, loc="lower right",
                handlelength=1.6, borderaxespad=0.1, labelspacing=0.15, alignment="left")
    fig = figR

    fig = figD
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
    # one partition of the organ: each point takes one regime (transient = the whole early-tile footprint; elsewhere retained >
    # low pulsatility); each indicator is smoothed and every point takes the strongest, so neighbouring regions
    # share one boundary; islands smaller than ~2 % of the organ are absorbed by their surroundings
    from scipy.ndimage import label as _lab, binary_fill_holes as _fill
    raw_ = np.full(GX_.shape, -1); raw_[organ_m & cL] = 1; raw_[organ_m & cS] = 2; raw_[organ_m & cT] = 0      # early footprint wins
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
    axMp.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.25, zorder=0)
    plt.rcParams["hatch.linewidth"] = 0.5; plt.rcParams["hatch.color"] = "0.45"
    # per vessel (retained recordings; best-ρ recording of each vessel): fundamental detected → coloured by its amplitude, else grey
    cmH = plt.cm.viridis; nH = mcolors.LogNorm(2e-3, 5e-2); vmap_ = []
    for e_, x_ in mall.items():
        if e_ in ezone_e or not G.has_edge(*e_):
            continue
        rr_ = [(y_["rho"], t_) for t_, y_ in x_.items() if t_ in stitch and np.isfinite(y_["rho"]) and abs(y_["H"][0]) > 0]
        if not rr_:
            continue
        t_ = max(rr_)[1]; y_ = x_[t_]; vmap_.append((e_, _pdet(e_, t_, y_) < P_DET, abs(y_["H"][0])))
    for e_, d_, h_ in sorted(vmap_, key=lambda v_: (v_[1], v_[2])):
        xx_, yy_ = _path(e_)
        axMp.plot(xx_, yy_, color=RCOL["S"] if d_ else "0.78", lw=0.8 if d_ else 0.45, zorder=3 if d_ else 2.5, solid_capstyle="round")
    log(f"  (a) map: {len(vmap_)} retained vessels, {sum(v_[1] for v_ in vmap_)} with the fundamental detected")
    axMp.contour(gxs, gys, _gf((lab_ >= 0).astype(float), 1.0), levels=[0.5], colors=["0.3"], linewidths=1.0, zorder=2)   # organ outline
    oy_, ox_ = np.nonzero(lab_ >= 0); pad2 = 15
    axMp.set_xlim(gxs[ox_.min()] - pad2, gxs[ox_.max()] + pad2); axMp.set_ylim(gys[oy_.max()] + pad2, gys[oy_.min()] - pad2)
    axMp.set_aspect("equal"); axMp.set_xticks([]); axMp.set_yticks([]); [s_.set_visible(False) for s_ in axMp.spines.values()]
    from matplotlib.lines import Line2D as _L2a
    axMp.legend(handles=[_L2a([], [], color=RCOL["S"], lw=1.4, label="phase measurable"),
                         _L2a([], [], color="0.78", lw=1.4, label="not detected")],
                loc="lower left", bbox_to_anchor=(-0.14, -0.04), fontsize=6.3, frameon=False, handlelength=1.4, borderaxespad=0,
                handletextpad=0.4, labelspacing=0.25)
    label(GMAP, "a"); fig = figR

    # ---- b (Fig 7): the organ, transient vs retained shaded; the four example vessels numbered
    axB7 = figR.add_subplot(GB7)
    axB7.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.5, zorder=0)
    axB7.contourf(gxs, gys, mT, levels=[0.5, 2], colors=[RCOL["T"]], alpha=0.28, zorder=1)
    axB7.contourf(gxs, gys, _gf((lab_ >= 1).astype(float), 1.0), levels=[0.5, 2], colors=[RCOL["S"]], alpha=0.16, zorder=1)
    axB7.contour(gxs, gys, _gf((lab_ >= 0).astype(float), 1.0), levels=[0.5], colors=["0.35"], linewidths=0.8, zorder=2)
    axB7.contour(gxs, gys, mT, levels=[0.5], colors=[RCOL["T"]], linewidths=0.9, zorder=2)
    oyB, oxB = np.nonzero(lab_ >= 0)
    axB7.set_xlim(gxs[oxB.min()] - 15, gxs[oxB.max()] + 15); axB7.set_ylim(gys[oyB.max()] + 15, gys[oyB.min()] - 15)
    axB7.set_aspect("equal"); axB7.set_xticks([]); axB7.set_yticks([]); [s_.set_visible(False) for s_ in axB7.spines.values()]
    for E_, tag_, k_, off_ in ((EXV, "1", "T", (-160, 260)), (T2V, "2", "T", (320, -200)), (RV, "3", "S", (300, 260)), (RV3, "4", "S", (-320, 300))):
        e_ = E_ if G.has_edge(*E_) else E_[::-1]; xx_, yy_ = _path(e_); cx_, cy_ = np.mean(xx_), np.mean(yy_)
        axB7.plot(xx_, yy_, color="white", lw=6.0, zorder=5, solid_capstyle="round"); axB7.plot(xx_, yy_, color="black", lw=4.0, zorder=5.1, solid_capstyle="round")
        axB7.annotate("", xy=(cx_, cy_), xytext=(cx_ + off_[0], cy_ + off_[1]), arrowprops=dict(arrowstyle="-", lw=0.7, color="black"), zorder=5)
        axB7.plot(cx_ + off_[0], cy_ + off_[1], "o", ms=10, mfc=RCOL[k_], mec="white", mew=0.8, zorder=6)
        axB7.text(cx_ + off_[0], cy_ + off_[1], tag_, fontsize=7, ha="center", va="center", fontweight="bold", color="white", zorder=7)
    axB7.legend(handles=[matplotlib.patches.Patch(fc=mcolors.to_rgba(RCOL["T"], 0.28), ec=RCOL["T"], label="transient early recordings"),
                         matplotlib.patches.Patch(fc=mcolors.to_rgba(RCOL["S"], 0.16), ec="0.35", label="retained recordings")],
                loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, ncol=2, handlelength=1.2, borderaxespad=0, columnspacing=0.8)
    fig = figR; _pb = axB7.get_position(); fig.text(_pb.x0 - 0.02, _pb.y1 + 0.005, "c", fontsize=10, fontweight="bold", va="bottom")

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
            if t not in shown_t or e in ezone_e or not np.isfinite(y["rho"]) or y["rho"] < 0.1 or abs(y["H"][0]) == 0 or np.sign(y["a0"]) != np.sign(F["Qp"][i_]):
                continue                                      # (a measurement against the reconciled flow direction is not used, as in the field)
            dl = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(np.sign(F["Qp"][i_]) * y["H"][0] * np.exp(-1j * psi[t]) * np.exp(1j * beta.get(t, 0.0)))))
            V.append((pts(e), (dl + 30) % 360 - 30, y["rho"], t))
            dr = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(np.sign(F["Qp"][i_]) * y["H"][0])))   # as recorded (no ψ_t)
            Vraw.append((V[-1][0], (dr + 30) % 360 - 30, y["rho"], t))
    tdel = {t: (v + 30) % 360 - 30 for t, v in tdel.items()}
    dmax = max(v for t, v in tdel.items() if t in shown_t); vtop = 20 * np.ceil(np.percentile([v[1] for v in V], 98) / 20)
    cmp = plt.cm.plasma; nph = mcolors.Normalize(0, 160); LEV = np.arange(0, vtop + 1, 20)      # colour capped at 160° (rim saturates)

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
    for j_, (VV, ttl) in enumerate(((Vraw, "before alignment"), (V, "after alignment: one rigid offset per tile"))):
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
    cbb.ax.tick_params(labelsize=6.5); cbb.outline.set_linewidth(0.4); cbb.set_label("phase of the fundamental (deg)", fontsize=7, labelpad=1)
    bb = GB.get_position(fig)
    fig.text(bb.x0 + 0.5 * bb.width, bb.y0 + 0.105 * bb.height, r"$\phi^{\rm aligned}_{e,t} = \phi^{\rm obs}_{e,t} - \psi_t$", ha="center", fontsize=8.5)

    log(f"  (b) tile pair {ta}/{tb} ({npair_} shared coherent vessels)")
    label(GB, "a")

    # ---- c: organ-scale smoothed delay field with isochrones; the second stitching component dashed, offset β from inlet symmetry
    axC = fig.add_subplot(GC); organ(axC)
    big = ((XL[0], XL[1]), (YL[1], YL[0]))
    fo, fb, ext = smooth_field(V, 150.0, 6.0, big)
    gx = np.linspace(big[0][0], big[0][1], fo.shape[1]); gy = np.linspace(big[1][0], big[1][1], fo.shape[0])
    # one exclusive partition of the organ: excluded (transient, Fig 7) · clock undetermined (covered ONLY by tiles whose offset SD >
    # SD_MAX) · reconstructed (field) · no measurable data (organ image only).  The field is masked to the reconstructed part.
    undet_t = sorted(t for t in stitch if sd.get(t, np.inf) > SD_MAX); det_t = {t for t in stitch if sd.get(t, np.inf) <= SD_MAX}
    log(f"  (c) retained tiles with undetermined offset (SD > {SD_MAX:.0f}°): " + ", ".join(f"{t} ({sd.get(t, np.nan):.0f}°)" for t in undet_t))
    mX = _gf((lab_ == 0).astype(float), 1.0)
    mU = _gf(((lab_ == 2) & _cover(set(undet_t)) & ~_cover(det_t)).astype(float), 30 / stp) if undet_t else np.zeros_like(mX)
    _ccU, _nU = _lab(mU > 0.5)                                         # drop slivers (< 1 % of the organ)
    for i_ in range(1, _nU + 1):
        if (_ccU == i_).sum() < 0.01 * (lab_ >= 0).sum():
            mU[_ccU == i_] = 0
    ixg = np.clip(np.round((gx - gxs[0]) / stp).astype(int), 0, len(gxs) - 1); iyg = np.clip(np.round((gy - gys[0]) / stp).astype(int), 0, len(gys) - 1)
    _bad = (mX[np.ix_(iyg, ixg)] > 0.5) | (mU[np.ix_(iyg, ixg)] > 0.5) | (lab_[np.ix_(iyg, ixg)] < 0)
    fo = np.ma.masked_where(np.ma.getmaskarray(fo) | _bad, fo)
    axC.imshow(fo, cmap=cmp, norm=nph, extent=ext, origin="upper", alpha=0.85, zorder=1, interpolation="bilinear")
    LEVc = LEV[::2]
    cs = axC.contour(gx, gy, fo, levels=LEVc, colors="0.1", linewidths=0.5, zorder=3)
    _cl = axC.clabel(cs, levels=[v for v in LEVc if v in (40, 80, 160)], fmt="%d°", fontsize=8, inline_spacing=1); _seen = set()
    for t_ in _cl:                                        # one label per level
        if t_.get_text() in _seen:
            t_.set_visible(False)
        _seen.add(t_.get_text())
    plt.rcParams["hatch.linewidth"] = 0.3; plt.rcParams["hatch.color"] = "0.5"
    fbf = np.ma.filled(fb, 0)
    axC.contour(gx, gy, fbf, levels=[0.5], colors="0.15", linewidths=0.9, linestyles="--", zorder=4)
    if beta_sd:
        bdeg, bsd, _, _ = beta_sd[0]; ys_, xs_ = np.nonzero(fbf > 0.5)
        pass                                                       # block is keyed in the legend (dashed outline)
    Lt = {t: 0.0 for t in band}
    for (a_, b_), w_ in link.items():
        Lt[a_] += w_; Lt[b_] += w_
    L_thr = 0.05 * np.median(list(Lt.values())); weak_t = {t for t in band if Lt[t] < L_thr}
    log(f"  (c) weakly constrained tiles (link < 5 % of median, {L_thr:.2f}): {sorted(weak_t)}")
    # transient region from Fig 7 (excluded from AC reconstruction)
    plt.rcParams["hatch.color"] = "0.45"; plt.rcParams["hatch.linewidth"] = 0.5
    axC.contourf(gxs, gys, mX, levels=[0.5, 2], colors=["white"], alpha=0.6, zorder=4.4)
    axC.contourf(gxs, gys, mX, levels=[0.5, 2], colors=["none"], hatches=["/////"], zorder=4.5)
    axC.contour(gxs, gys, mX, levels=[0.5], colors=["0.5"], linewidths=0.6, zorder=4.6)
    # retained (Fig 7: physiologically usable) but absolute phase not identifiable — area covered only by tiles with offset SD > SD_MAX
    if undet_t:
        axC.contourf(gxs, gys, mU, levels=[0.5, 2], colors=["white"], alpha=0.75, zorder=4.4)
        axC.contourf(gxs, gys, mU, levels=[0.5, 2], colors=["none"], hatches=["...."], zorder=4.5)
        axC.contour(gxs, gys, mU, levels=[0.5], colors=["0.5"], linewidths=0.6, linestyles=":", zorder=4.6)
    refpts = [(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"]) for n_ in list(dom["DA"]) + [nds[int(i)] for i in dom["out"]]]
    for n_ in dom["DA"]:
        axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="v", ms=5.5, color="white", mec="black", mew=0.6, zorder=6)
    for i in [int(i) for i in dom["out"]]:
        n_ = nds[i]; axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="*", ms=7, color="white", mec="black", mew=0.6, zorder=6)
    rp = np.array(refpts); axC.annotate("simultaneous DA/V\nphase reference, 0°", xy=(rp[:, 0].mean(), rp[:, 1].min() - 20),
                                        xytext=(rp[:, 0].mean() + 0.30 * (XL[1] - XL[0]), YL[1] + 0.02 * (YL[0] - YL[1])), ha="center", va="top", fontsize=8,
                                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), zorder=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axC.legend(handles=[Patch(fc="none", ec="0.15", ls="--", label="separately oriented block"),
                        Patch(fc="white", ec="0.45", hatch="//////", lw=0.5, label="excluded from AC reconstruction (Fig. 7)"),
                        Patch(fc="white", ec="0.5", hatch="....", ls=":", lw=0.5, label="tile phase offset undetermined")],
               loc="lower left", bbox_to_anchor=(-0.04, 0.0), fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0)
    # inset: the coherent-overlap tile graph (retained tiles; edge = ≥ 1 shared vessel with ρ ≥ 0.1 in both recordings), nodes at the
    # tile centres; its connected components are the synchronization blocks; the dashed bridge = the inlet-symmetry constraint (β)
    import networkx as _nx
    Tg = _nx.Graph(); Tg.add_nodes_from(sorted(stitch))
    _da_set = set(da_edges) | {e_[::-1] for e_ in da_edges}
    for e_, x_ in meas.items():
        if e_ in _da_set:                                  # the DA stubs anchor the reference, they are not overlap links (as the components)
            continue
        ok_ = [t_ for t_, y_ in x_.items() if t_ in stitch and np.isfinite(y_["rho"]) and y_["rho"] >= 0.1]
        for i_ in range(len(ok_)):
            for j_ in range(i_ + 1, len(ok_)):
                Tg.add_edge(ok_[i_], ok_[j_])
    comps_ = sorted(_nx.connected_components(Tg), key=len, reverse=True)
    log(f"  (c) coherent-overlap graph on retained tiles: components {[sorted(c_) for c_ in comps_]}")
    cmain_ = next(c_ for c_ in comps_ if ref in c_); cother_ = [c_ for c_ in comps_ if c_ is not cmain_]
    GCOL = {"main": "#2b5c8a", "side": "#E69F00"}
    # ---- SI figure fig8_supp_block: a overlap graph · b soft mode of the synchronization Laplacian · c estimates of the block offset β ·
    #      d sensitivity of the phase map to β
    figB = plt.figure(figsize=(7.5, 6.4), facecolor="white")
    gB = figB.add_gridspec(2, 2, wspace=0.18, hspace=0.25, left=0.04, right=0.98, top=0.95, bottom=0.06)
    def labB(ax_, letter):
        b_ = ax_.get_position(); figB.text(b_.x0 - 0.02, b_.y1 + 0.035, letter, fontsize=10, fontweight="bold", va="bottom")
    axG = figB.add_subplot(gB[0, 0]); axG.set_aspect("equal"); axG.set_xticks([]); axG.set_yticks([])
    [sp_.set_visible(False) for sp_ in axG.spines.values()]
    for a_, b_ in Tg.edges():
        axG.plot([cen[a_][0], cen[b_][0]], [cen[a_][1], cen[b_][1]], color="0.65", lw=0.35, zorder=1)
    # overlaps across the gap: tile pairs (main × side) that share vessels, none coherent (ρ ≥ 0.1) in both recordings
    if cother_:
        side_ = max(cother_, key=len); near_ = set()
        for e_, x_ in meas.items():
            if e_ in _da_set:
                continue
            ts_ = [t_ for t_ in x_ if np.isfinite(x_[t_]["rho"])]
            near_ |= {(a_, b_) for a_ in ts_ if a_ in cmain_ for b_ in ts_ if b_ in side_}
        for a_, b_ in near_:
            axG.plot([cen[a_][0], cen[b_][0]], [cen[a_][1], cen[b_][1]], color="#b23a48", lw=0.6, ls=(0, (1, 1.2)), zorder=1.5)
        # β bridge: between the tiles holding the arterial-inlet vessels that β compares (most inlet vessels on each side)
        from collections import Counter as _Ctr
        tM_ = _Ctr(wt_[selM].astype(int)).most_common(1)[0][0]
        tS_ = _Ctr(wt_[inl & np.isin(wt_, sorted(side_))].astype(int)).most_common(1)[0][0]
        axG.annotate("", xy=cen[tM_], xytext=cen[tS_], arrowprops=dict(arrowstyle="-", ls=(0, (3, 1.5)), lw=1.0, color="black",
                                                                   connectionstyle="arc3,rad=0.5"), zorder=2)
        log(f"  (c) inset: cross-gap overlaps (no coherent vessel) {sorted(near_)}; β bridge between inlet tiles {tS_} (side) and {tM_} (main)")
    for t_ in Tg.nodes():
        c_ = GCOL["main"] if t_ in cmain_ else GCOL["side"]
        axG.plot(*cen[t_], "s" if t_ == ref else "o", ms=10 if t_ == ref else 9, mfc=c_, mec="black" if t_ == ref else "white", mew=0.5, zorder=3)
        axG.text(*cen[t_], str(t_), fontsize=5.2, ha="center", va="center", color="white", fontweight="bold", zorder=4)
    from matplotlib.lines import Line2D as _L2g
    axG.legend(handles=[_L2g([], [], color="0.65", lw=0.6, label="coherent shared vessel"),
                        _L2g([], [], color="#b23a48", lw=0.8, ls=(0, (1, 1.2)), label="overlap, not coherent in both"),
                        _L2g([], [], color="black", lw=1.0, ls=(0, (3, 1.5)), label="β: inlet symmetry")],
               loc="lower left", fontsize=6.5, frameon=False, handlelength=1.8, borderaxespad=0.15, labelspacing=0.15)
    allc_ = np.array([cen[t_] for t_ in Tg.nodes()]); pg_ = 140
    axG.set_xlim(allc_[:, 0].min() - pg_, allc_[:, 0].max() + pg_); axG.set_ylim(allc_[:, 1].max() + 5.4 * pg_, allc_[:, 1].min() - 3.2 * pg_)
    axG.set_title("tiles linked by a shared vessel coherent (ρ ≥ 0.1) in both", fontsize=8); labB(axG, "a")

    # b: the synchronization Laplacian (weights Σ ρ_a ρ_b over all shared vessels, DA stubs excluded): soft mode = rotate the block
    tlB = sorted(Tg.nodes()); ixB = {t_: i_ for i_, t_ in enumerate(tlB)}; WB = np.zeros((len(tlB), len(tlB)))
    for e_, x_ in meas.items():
        if e_ in _da_set:
            continue
        ok_ = [t_ for t_ in x_ if t_ in ixB and np.isfinite(x_[t_]["rho"])]
        for i_ in range(len(ok_)):
            for j_ in range(i_ + 1, len(ok_)):
                w_ = max(x_[ok_[i_]]["rho"], 0) * max(x_[ok_[j_]]["rho"], 0); WB[ixB[ok_[i_]], ixB[ok_[j_]]] += w_; WB[ixB[ok_[j_]], ixB[ok_[i_]]] += w_
    dgB = WB.sum(1); evB, VB = np.linalg.eigh(np.eye(len(tlB)) - WB / np.sqrt(np.outer(dgB, dgB))); f2 = VB[:, 1] / np.sqrt(dgB)
    f2 = f2 / np.abs(f2).max() * (np.sign(f2[ixB[ref]]) * -1 if f2[ixB[ref]] != 0 else 1)
    side_set_ = max(cother_, key=len) if cother_ else set()
    cutB = sum(WB[ixB[a_], ixB[b_]] for a_ in side_set_ for b_ in tlB if b_ not in side_set_)
    log(f"  (SI) Laplacian eigenvalues {', '.join(f'{v:.4f}' for v in evB[:4])}; cut weight {cutB:.3f} of {WB.sum() / 2:.1f}")
    axL = figB.add_subplot(gB[0, 1]); axL.set_aspect("equal"); axL.set_xticks([]); axL.set_yticks([]); [sp_.set_visible(False) for sp_ in axL.spines.values()]
    for a_, b_ in Tg.edges():
        axL.plot([cen[a_][0], cen[b_][0]], [cen[a_][1], cen[b_][1]], color="0.8", lw=0.35, zorder=1)
    scL = axL.scatter([cen[t_][0] for t_ in tlB], [cen[t_][1] for t_ in tlB], c=f2, cmap="PuOr", vmin=-1, vmax=1, s=70, edgecolors="0.3", linewidths=0.4, zorder=3)
    axL.set_xlim(*axG.get_xlim()); axL.set_ylim(*axG.get_ylim())
    cbL = figB.colorbar(scL, ax=axL, shrink=0.55, pad=0.01); cbL.set_label("soft-mode amplitude $v_2$", fontsize=7); cbL.ax.tick_params(labelsize=6)
    axE = axL.inset_axes([0.02, 0.06, 0.32, 0.26]); axE.semilogy(np.arange(1, 9), np.maximum(evB[:8], 1e-5), "o-", color="0.25", ms=3, lw=0.8)
    axE.set_xticks([1, 2, 3, 5, 8]); axE.tick_params(labelsize=5.5, length=1.5, pad=1); axE.set_xlabel("mode", fontsize=6, labelpad=0)
    axE.set_ylabel("eigenvalue", fontsize=6, labelpad=0); axE.set_ylim(5e-6, 1)
    axE.annotate(f"$\\lambda_2$ = {evB[1]:.4f}", xy=(2, max(evB[1], 1e-5)), xytext=(3.3, 3e-5), fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.4))
    axL.set_title(f"soft mode: rotate the block (cut {100 * cutB / (WB.sum() / 2):.2f} % of link weight)", fontsize=8); labB(axL, "b")

    # c: estimates of β (main − block, deg) from independent routes
    zB = {}
    def _zz(y_, t_):
        return np.exp(1j * (np.angle(np.sign(y_["a0"]) * y_["H"][0]) - psi[t_]))
    wl_, zl_ = [], []                                      # every shared vessel across the gap, weight ρρ
    for e_, x_ in meas.items():
        if e_ in _da_set:
            continue
        for a_ in [t_ for t_ in x_ if t_ in side_set_ and np.isfinite(x_[t_]["rho"])]:
            for b_ in [t_ for t_ in x_ if t_ in cmain_ and np.isfinite(x_[t_]["rho"])]:
                # same vessel in both tiles: compare in the vessel's fixed orientation (no per-tile flow-sign flip, unreliable at low ρ)
                wl_.append(max(x_[a_]["rho"], 0) * max(x_[b_]["rho"], 0))
                zl_.append(x_[b_]["H"][0] * np.exp(-1j * psi[b_]) * np.conj(x_[a_]["H"][0] * np.exp(-1j * psi[a_])) / max(abs(x_[b_]["H"][0] * x_[a_]["H"][0]), 1e-30))
    wl_, zl_ = np.array(wl_), np.array(zl_); rngB = np.random.default_rng(3)
    def _cmean(w_, z_):
        S_ = (w_ * z_).sum(); bs_ = [np.angle((w_[k_] * z_[k_]).sum() * np.conj(S_)) for k_ in (rngB.integers(0, len(w_), len(w_)) for _ in range(1000))]
        return np.degrees(np.angle(S_)), np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.array(bs_)))))))
    est_weak = _cmean(wl_, zl_)
    from scipy.spatial import cKDTree as _KD                # nearby coherent vessels (different vessels), smoothness calibrated within tiles
    recB = [(t_, e_, np.asarray(_mid(e_)), _zz(y_, t_)) for e_, x_ in meas.items() if G.has_edge(*e_) and e_ not in _da_set
            for t_, y_ in x_.items() if t_ in ixB and np.isfinite(y_["rho"]) and y_["rho"] >= 0.12 and abs(y_["H"][0]) > 0]
    xyB = np.array([r_[2] for r_ in recB]); tB_ = np.array([r_[0] for r_ in recB]); zB_ = np.array([r_[3] for r_ in recB]); eB_ = [r_[1] for r_ in recB]
    kd_ = _KD(xyB); dmx_ = 250.0 / FUSE_UM_PX; xda_ = np.mean([G.nodes[n_]["x"] for n_ in dom["DA"]])
    cal_ = [np.angle(zB_[i_] * np.conj(zB_[j_])) for i_, j_ in kd_.query_pairs(dmx_) if tB_[i_] == tB_[j_] and eB_[i_] != eB_[j_]
            and np.linalg.norm(xyB[i_] - xyB[j_]) >= 150.0 / FUSE_UM_PX]
    sd_cal = np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.array(cal_))))))
    prx_, tpx_ = [], set()
    for i_, j_ in kd_.query_pairs(dmx_):
        a_, b_ = (i_, j_) if tB_[i_] in side_set_ else (j_, i_)
        if tB_[a_] in side_set_ and tB_[b_] in cmain_ and eB_[a_] != eB_[b_] and xyB[a_][0] < xda_ and xyB[b_][0] < xda_:
            prx_.append(zB_[b_] * np.conj(zB_[a_])); tpx_.add((int(tB_[a_]), int(tB_[b_])))
    est_prox = (np.degrees(np.angle(np.sum(prx_))), np.degrees(sd_cal / np.sqrt(max(len(tpx_), 1)))) if prx_ else (np.nan, np.nan)
    stub_ = {}
    for n_ in dom["V"]:
        for m_ in G.neighbors(n_):
            f_ = (n_, m_) if (n_, m_) in meas else (m_, n_)
            if f_ in meas and 14 in meas[f_] and np.isfinite(meas[f_][14]["rho"]):
                y_ = meas[f_][14]; sd_k = "L" if _mid(f_)[0] < xda_ else "R"
                if sd_k not in stub_ or y_["rho"] > stub_[sd_k][0]:
                    stub_[sd_k] = (y_["rho"], np.angle(np.sign(y_["a0"]) * y_["H"][0]))
    lr_ = np.degrees(np.angle(np.exp(1j * (stub_["L"][1] - stub_["R"][1])))) if len(stub_) == 2 else np.nan
    bsym = beta_sd[0][:2] if beta_sd else (np.nan, np.nan)
    log(f"  (SI) β: inlet symmetry {bsym[0]:+.1f} ± {bsym[1]:.1f}; nearby coherent vessels {est_prox[0]:+.0f} ± {est_prox[1]:.0f} "
        f"({len(prx_)} pairs, tile pairs {sorted(tpx_)}); weak shared vessels {est_weak[0]:+.0f} ± {est_weak[1]:.0f} ({len(wl_)}); tile-14 V stubs L − R {lr_:+.0f}")
    axF = figB.add_subplot(gB[1, 0]); rowsF = [("arterial-inlet symmetry (used)", *bsym, "black"),
                                               (f"nearby coherent vessels ({len(prx_)} pairs, ≤ 250 µm)", *est_prox, "#2b5c8a"),
                                               (f"all weak shared vessels ({len(wl_)})", *est_weak, "#2b5c8a"),
                                               ("tile 14 vein stubs, left − right (symmetry check)", lr_, np.nan, "0.5")]
    for k_, (lab_, m_, sdv_, c_) in enumerate(rowsF):
        y_ = len(rowsF) - k_
        if np.isfinite(sdv_):
            axF.plot([m_ - 1.96 * sdv_, m_ + 1.96 * sdv_], [y_, y_], color=c_, lw=1.4, solid_capstyle="butt")
        axF.plot(m_, y_, "o" if k_ < 3 else "D", color=c_, ms=4.5)
        axF.text(-118, y_ + 0.22, lab_, fontsize=6.8, va="bottom", color=c_)
    axF.annotate("DA stub via tile 20 (ρ 0.18): +145°,\nrejected — network model fit much worse", xy=(119, 0.2), xytext=(-118, 0.2), fontsize=6.2, color="#b23a48",
                 va="center", ha="left", arrowprops=dict(arrowstyle="->", lw=0.6, color="#b23a48"))
    axF.axvline(0, color="0.75", lw=0.6, ls=":"); axF.set_xlim(-120, 120); axF.set_ylim(-0.4, len(rowsF) + 0.9); axF.set_yticks([])
    axF.set_xlabel("block offset β (deg; bars 95 %)", fontsize=8); axF.tick_params(labelsize=7); axF.spines[["top", "right", "left"]].set_visible(False)
    axF.set_title("every route that can see β", fontsize=8); labB(axF, "c")

    # d: sensitivity — isochrones of the composite with β from symmetry (used), from nearby vessels, and ± the weak-link 95 % range
    axS = figB.add_subplot(gB[1, 1]); organ(axS)
    axS.imshow(fo, cmap=cmp, norm=nph, extent=ext, origin="upper", alpha=0.35, zorder=1, interpolation="bilinear")
    for d_, col_, ls_, lab_ in ((0.0, "black", "-", f"β = {bsym[0]:+.0f}° (used)"), (est_prox[0] - bsym[0], "#2b5c8a", "--", f"β = {est_prox[0]:+.0f}° (nearby vessels)"),
                                (1.96 * est_weak[1], "#D55E00", ":", f"β ± {1.96 * est_weak[1]:.0f}° (weak-link 95 %)"), (-1.96 * est_weak[1], "#D55E00", ":", None)):
        Valt = [(p_, ((dl_ - d_ + 30) % 360) - 30 if t_ in block_t else dl_, w_, t_) for p_, dl_, w_, t_ in V]
        fa_, _, _ = smooth_field(Valt, 150.0, 6.0, big)
        axS.contour(gx, gy, fa_, levels=LEV[1:6], colors=[col_], linewidths=0.8 if d_ == 0 else 0.7, linestyles=ls_, zorder=3)
        if lab_:
            axS.plot([], [], color=col_, ls=ls_, lw=0.9, label=lab_)
    axS.contour(gx, gy, np.ma.filled(fb, 0), levels=[0.5], colors="0.15", linewidths=0.9, linestyles="--", zorder=4)
    axS.legend(loc="lower left", fontsize=6.3, frameon=False, borderaxespad=0.2, handlelength=2.0)
    axS.set_title("isochrones (20° apart) under alternative β", fontsize=8); labB(axS, "d")

    sm = plt.cm.ScalarMappable(norm=nph, cmap=cmp)
    cax = axC.inset_axes([0.2, -0.05, 0.6, 0.028]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 40, 80, 120, 160], extend="both")
    cb.set_label("phase lag of the fundamental behind DA/V reference (deg)", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
    _pC = axC.get_position(); fig.text(_pC.x0 - 0.01, _pC.y1 + 0.03, "c", fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- e: repeat-vessel agreement after alignment; H2 is an independent check (not used in the synchronization)
    gDs = GD.subgridspec(1, 2, wspace=0.3); bins = np.linspace(-60, 60, 25)
    for c_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gDs[c_]); v = d[k]["aligned"]
        ax.hist(v, bins=bins, density=True, color="#8fb0d3" if k == 1 else "#c9b6d9", ec="white", lw=0.3)
        ax.set_xlim(-60, 60); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-60, 0, 60])
        ax.axvline(0, color="0.3", lw=0.6, zorder=3); ax.tick_params(axis="x", labelsize=7)
        ax.text(0.0, 1.13, "fundamental" if k == 1 else "2nd harmonic", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.text(0.0, 1.01, "used to synchronize" if k == 1 else "not used to synchronize", transform=ax.transAxes, fontsize=8, va="bottom",
                color="0.3" if k == 1 else "#6a4c93")
        ax.text(0.0, 0.97, f"median $|\\Delta\\phi|$ = {np.median(np.abs(v)):.0f}°", transform=ax.transAxes, fontsize=6.5, va="top", ha="left", color="0.25", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
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
    # SI fig8_supp_phase_rho: repeat-vessel aligned phase difference vs the weaker recording's coherence (retained tiles)
    PR_ = []
    for e_, x_ in meas.items():
        ok_ = [t_ for t_ in x_ if t_ in stitch and np.isfinite(x_[t_]["rho"]) and abs(x_[t_]["H"][0]) > 0]
        for i_ in range(len(ok_)):
            for j_ in range(i_ + 1, len(ok_)):
                a_, b_ = ok_[i_], ok_[j_]
                PR_.append((min(x_[a_]["rho"], x_[b_]["rho"]), abs(np.degrees(np.angle(x_[a_]["H"][0] * np.exp(-1j * psi[a_])
                                                                                      * np.conj(x_[b_]["H"][0] * np.exp(-1j * psi[b_])))))))
    PR_ = np.array(PR_); edg_ = np.array([0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
    figP = plt.figure(figsize=(3.4, 2.6), facecolor="white"); axPR = figP.add_axes([0.17, 0.2, 0.78, 0.72])
    mids_, med_, q1_, q3_ = [], [], [], []
    for lo_, hi_ in zip(edg_[:-1], edg_[1:]):
        m_ = (PR_[:, 0] >= lo_) & (PR_[:, 0] < hi_)
        if m_.sum() >= 20:
            mids_.append(np.sqrt(lo_ * hi_)); med_.append(np.median(PR_[m_, 1])); q1_.append(np.percentile(PR_[m_, 1], 25)); q3_.append(np.percentile(PR_[m_, 1], 75))
            log(f"  (SI) weaker ρ {lo_:.3f}–{hi_:.2f}: {int(m_.sum())} pairs, median |Δφ| {med_[-1]:.0f}°")
    axPR.fill_between(mids_, q1_, q3_, color="#8fb0d3", alpha=0.5, lw=0); axPR.plot(mids_, med_, "o-", color="#2b5c8a", ms=3.5, lw=1.2)
    axPR.axhline(90, color="#b23a48", lw=0.8, ls="--"); axPR.text(0.98, 92, "random phase", transform=axPR.get_yaxis_transform(), ha="right", va="bottom",
                                                                    fontsize=6.5, color="#b23a48")
    axPR.axvline(0.2, color="0.6", lw=0.6, ls=":"); axPR.text(0.21, 60, "ρ = 0.2", fontsize=6.5, color="0.45")
    axPR.set_xscale("log"); axPR.set_ylim(0, 120); axPR.set_yticks([0, 30, 60, 90, 120])
    axPR.set_xlabel("coherence of the weaker recording, ρ", fontsize=7.5); axPR.set_ylabel("repeat-vessel phase difference,\naligned (deg)", fontsize=7.5)
    axPR.tick_params(labelsize=7); axPR.spines[["top", "right"]].set_visible(False)
    figP.savefig(OUT_DIR / f"fig8_supp_phase_rho{args.tag}.png", dpi=300, facecolor="white"); figP.savefig(OUT_DIR / f"fig8_supp_phase_rho{args.tag}.pdf", facecolor="white")
    figB.savefig(OUT_DIR / f"fig8_supp_block{args.tag}.png", dpi=300, facecolor="white"); figB.savefig(OUT_DIR / f"fig8_supp_block{args.tag}.pdf", facecolor="white")
    figSM.savefig(OUT_DIR / f"fig7_supp_shape_map{args.tag}.png", dpi=300, facecolor="white"); figSM.savefig(OUT_DIR / f"fig7_supp_shape_map{args.tag}.pdf", facecolor="white")
    def _plos_min_font(f_, lo=8.0):
        """PLOS Comp Biol: figure text 8–12 pt — raise every smaller text element (labels, ticks, legends, annotations) to `lo`."""
        for t_ in f_.findobj(matplotlib.text.Text):
            if t_.get_text() and t_.get_fontsize() < lo:
                t_.set_fontsize(lo)
    for f_, nm_ in ((figR, "fig7_regime"), (figS, "fig8_sync"), (figD, "fig7_supp_detection"), (figT, "fig7_supp_theta21")):
        _plos_min_font(f_)
        f_.savefig(OUT_DIR / f"{nm_}{args.tag}.png", dpi=300, facecolor="white"); f_.savefig(OUT_DIR / f"{nm_}{args.tag}.pdf", facecolor="white")
        f_.savefig(OUT_DIR / f"{nm_}{args.tag}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        log(f"wrote {OUT_DIR / f'{nm_}{args.tag}.png'}")


if __name__ == "__main__":
    main()
