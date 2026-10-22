"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  heart rate f0 of every tile vs tile number; tiles within ±15 % of the reference tile are retained (shaded band)
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

    fig = plt.figure(figsize=(7.5, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.25], wspace=0.3, hspace=0.28, left=0.08, right=0.97, top=0.95, bottom=0.08)
    GA, GB, GC, GD = gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]

    def label(cell, letter):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.045, b_.y1 + 0.03, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- a: which tiles are comparable — heart rate per tile, accepted band
    axA = fig.add_subplot(GA); ts = sorted(tf0); f = np.array([tf0[t] for t in ts]); inb = np.array([t in band for t in ts])
    # acquisition time: time stamp of each tile's first raw frame (file modification time), minutes after the first tile
    import glob as _glob, os as _os, re as _re
    from plexusflow.config import StageConfig as _SC
    _cfg = _SC.load(_paths.STAGE); _rd = _os.path.expanduser(str(_cfg.external_raw_video_dir or "")); tacq = {}
    for t in ts:                                     # the recording the pipeline analysed: raw_video_dir_pattern with this tile id
        _cands = _glob.glob(_os.path.join(_rd, _cfg.raw_video_dir_pattern.format(vid=t)))
        _fs = sorted(_glob.glob(_os.path.join(sorted(_cands)[0], "*.tif")))[:1] if _cands else []
        if _fs:
            tacq[t] = _os.stat(_fs[0]).st_mtime
    if all(t in tacq for t in ts):
        _t0 = min(tacq.values()); xs_a = np.array([(tacq[t] - _t0) / 60 for t in ts]); xlab = "acquisition time (min)"
    else:
        xs_a = np.array(ts, float); xlab = "acquisition order (tile)"
    xref = xs_a[ts.index(ref)]
    axA.axhspan(f_ref * (1 - BAND), f_ref * (1 + BAND), color="#e9eef4", zorder=0, lw=0)
    axA.plot(xs_a[inb], f[inb], "o", ms=2.8, color="0.15", zorder=3)
    axA.plot(xs_a[~inb], f[~inb], "o", ms=2.8, color="0.72", zorder=3)
    axA.plot(xref, f_ref, "s", ms=5, mfc="white", mec="black", mew=0.9, zorder=4)
    axA.text(0.97, f_ref * (1 - BAND) + 0.04, "retained for phase reconstruction", transform=axA.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="0.4")
    axA.annotate("reference tile", xy=(xref, f_ref), xytext=(xref - 0.06 * xs_a.max(), f_ref + 0.22), fontsize=7, color="0.3", ha="right", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
    axA.set_xlabel(xlab); axA.set_ylabel("heart rate $f_0$ (Hz)"); axA.spines[["top", "right"]].set_visible(False)
    label(GA, "a")

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
    gBs = GB.subgridspec(2, 2, height_ratios=[1, 0.18], wspace=0.2, hspace=0.0)
    for j_, (VV, ttl) in enumerate(((Vraw, "as recorded"), (V, "one phase offset per tile"))):
        ax = fig.add_subplot(gBs[0, j_])
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
        ax.set_title(ttl, fontsize=8, pad=3)
    axb0, axb1 = fig.axes[-2], fig.axes[-1]; p0_, p1_ = axb0.get_position(), axb1.get_position()
    fig.patches.append(matplotlib.patches.FancyArrowPatch((p0_.x1 - 0.004, p0_.y0 + 0.5 * p0_.height), (p1_.x0 + 0.004, p0_.y0 + 0.5 * p0_.height),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, lw=1.0, color="black", zorder=10))
    fig.text((p0_.x1 + p1_.x0) / 2, p0_.y0 + 0.5 * p0_.height + 0.012, r"$-\psi_t$", ha="center", va="bottom", fontsize=8, zorder=10)
    caxb = fig.add_subplot(gBs[1, :]); caxb.axis("off")
    cbi = caxb.inset_axes([0.3, 0.55, 0.4, 0.18]); cbb = fig.colorbar(plt.cm.ScalarMappable(norm=ncy, cmap=cmc), cax=cbi, orientation="horizontal", ticks=[-180, 0, 180])
    cbb.ax.tick_params(labelsize=6.5); cbb.outline.set_linewidth(0.4); cbb.set_label("H1 phase (deg)", fontsize=7, labelpad=1)
    bb = GB.get_position(fig)
    fig.text(bb.x0 + 0.5 * bb.width, bb.y0 + 0.235 * bb.height, r"$\phi^{\rm aligned}_{e,t} = \phi^{\rm obs}_{e,t} - \psi_t$", ha="center", fontsize=8.5)

    log(f"  (b) tile pair {ta}/{tb} ({npair_} shared coherent vessels)")
    label(GB, "b")

    # ---- c: organ-scale smoothed delay field with isochrones; the second stitching component dashed, offset β from inlet symmetry
    axC = fig.add_subplot(GC); organ(axC)
    big = ((XL[0], XL[1]), (YL[1], YL[0]))
    fo, fb, ext = smooth_field(V, 150.0, 6.0, big)
    gx = np.linspace(big[0][0], big[0][1], fo.shape[1]); gy = np.linspace(big[1][0], big[1][1], fo.shape[0])
    axC.imshow(fo, cmap=cmp, norm=nph, extent=ext, origin="upper", alpha=0.9, zorder=1, interpolation="bilinear")
    cs = axC.contour(gx, gy, fo, levels=LEV, colors="0.1", linewidths=0.45, zorder=3)
    axC.clabel(cs, levels=[v for v in LEV if v in (40, 80, 160)], fmt="%d°", fontsize=6, inline_spacing=1)
    plt.rcParams["hatch.linewidth"] = 0.3; plt.rcParams["hatch.color"] = "0.5"
    fbf = np.ma.filled(fb, 0)
    axC.contour(gx, gy, fbf, levels=[0.5], colors="0.15", linewidths=0.9, linestyles="--", zorder=4)
    if beta_sd:
        bdeg, bsd, _, _ = beta_sd[0]; ys_, xs_ = np.nonzero(fbf > 0.5)
        axC.text(gx[int(np.median(xs_))], gy[ys_.max()] + 0.03 * (YL[0] - YL[1]), f"separately stitched;\noffset from inlet symmetry\nβ = {bdeg:+.0f}° ± {bsd:.0f}°",
                 ha="center", va="top", fontsize=6, color="0.15", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    for t in band:
        if t not in shown_t:
            axC.plot(*cen[t], "o", ms=3.5, mfc="white", mec="0.55", mew=0.7, zorder=5)
    refpts = [(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"]) for n_ in list(dom["DA"]) + [nds[int(i)] for i in dom["out"]]]
    for n_ in dom["DA"]:
        axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="v", ms=5.5, color="white", mec="black", mew=0.6, zorder=6)
    for i in [int(i) for i in dom["out"]]:
        n_ = nds[i]; axC.plot(G.nodes[n_]["x"], Hm_ - G.nodes[n_]["y"], marker="*", ms=7, color="white", mec="black", mew=0.6, zorder=6)
    rp = np.array(refpts); axC.annotate("simultaneous DA/V\nreference, 0°", xy=(rp[:, 0].mean(), rp[:, 1].min() - 20),
                                        xytext=(rp[:, 0].mean() + 0.30 * (XL[1] - XL[0]), YL[1] + 0.02 * (YL[0] - YL[1])), ha="center", va="top", fontsize=6.5,
                                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"), zorder=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axC.legend(handles=[Patch(fc="none", ec="0.15", ls="--", label="second stitching component"),
                        Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="0.55", label="tile phase undetermined")],
               loc="upper left", bbox_to_anchor=(-0.04, 1.0), fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0)
    axC.add_patch(Polygon(polys[ta], closed=True, fc="none", ec="#0072B2", lw=0.8, ls="--", zorder=5))
    axC.add_patch(Polygon(polys[tb], closed=True, fc="none", ec="#D55E00", lw=0.8, ls="--", zorder=5))
    sm = plt.cm.ScalarMappable(norm=nph, cmap=cmp)
    cax = axC.inset_axes([0.2, -0.05, 0.6, 0.028]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=LEV[::3], extend="min")
    cb.set_label("H1 phase lag behind simultaneous DA/V reference (deg)", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
    label(GC, "c")

    # ---- c: repeated-vessel agreement after alignment
    gD = GD.subgridspec(2, 1, hspace=0.3); bins = np.linspace(-60, 60, 31)
    for r_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gD[r_]); v = d[k]["aligned"]
        ax.hist(v, bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.3)
        ax.set_xlim(-60, 60); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-60, 0, 60])
        ax.axvline(0, color="0.3", lw=0.6, zorder=3)
        ax.text(0.0, 0.97, "H1" if k == 1 else "H2", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        if k == 2:
            ax.text(0.075, 0.965, "not used in synchronization", transform=ax.transAxes, fontsize=6.8, va="top", color="0.35")
        ax.text(1.0, 0.97, f"median $|\\Delta\\phi|$ = {np.median(np.abs(d[k]['aligned'])):.0f}°", transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color="0.2")
        if r_ == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("same-vessel phase difference\nbetween tiles, aligned (deg)")
    label(GD, "d")

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
    fig.savefig(OUT_DIR / f"fig7{args.tag}.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / f"fig7{args.tag}.pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"fig7{args.tag}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    log(f"wrote {OUT_DIR / f'fig7{args.tag}.png'}")


if __name__ == "__main__":
    main()
