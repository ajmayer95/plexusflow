"""Figure 6 — putting asynchronous recordings on one cardiac clock (PLOS, 7.5 in wide).

  a  heart rate of each tile in acquisition order; the band of retained tiles (within ± 21 % of the reference tile) is shaded,
     the tiles outside it (21S: tiles 1–6) are excluded
  b  the tile clocks: tiles linked by the vessels they share (line width ∝ number of clock measurements); tiles whose clock is
     too uncertain are drawn open; one vessel recorded
     in several tiles before and after its recordings are shifted by their tile clocks (each curve divided by its own mean)
  c  waveform agreement of repeated recordings: per vessel, the RMS difference between its clock-aligned recordings and their
     weighted mean (fraction of the mean flow), against the heart-rate spread of the tiles that recorded it.  For this panel
     only, the clock fit is repeated with the excluded tiles added, so their vessels can be scored
  d  fundamental (H1) phase field: delay of every reconstructed vessel behind the vitelline-vein waveform, with isochrones of
     the smoothed field (Gaussian, σ = 250 µm, vessels weighted by the precision of their phase; display only).  No isochrones
     are drawn where the vessel phases around a point do not agree with each other

Everything is read from the pipeline: flux recordings and clock measurements from plexusflow.ac.pulse, the clocks and the
reconstructed field from results/ac_field_v9.pkl (scripts/pulse_field.py).  Writes figures/fig6/fig6.{png,pdf,tif}; nothing to any graph.
"""
from __future__ import annotations
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plos_style
import plexusflow.paths as P
from plexusflow.config import StageConfig
from plexusflow.analyze.geometry import load_tile_geometry
from plexusflow.reconcile import carved_domain
from plexusflow.reconcile.dc import K_FIT
from plexusflow.reconcile.harmonics import phasors, coherence
from plexusflow.ac.pulse import FIELD, recordings, flux_pairs, tile_clocks
from mosaic_display import load_mosaic, show_mosaic

EXAMPLE = (57695, 57893)        # 21S: recorded in tiles 12, 13, 22, 23; its RMS difference is typical for its heart-rate spread
ZERO_TILE = 14                  # the tile that records both vitelline-vein outlets (one movie, so no clock between them)
EXCOL, EXAMPLE_COL = "#d95f02", "#c51b8a"
TILE_COLS = ["#1b9e77", "#7570b3", "#e6ab02", "#666666"]
SIGMA_UM = 250.0                # smoothing of the phase field for the isochrones (display only)
WRAP_DEG = -60.0                # delays are shown in [−60°, 300°): on 21S almost no well-determined vessel lies between 270° and 330°
PHASE_SD_FLOOR_DEG = 10.0       # the smoothed field weights each vessel by 1 / (phase variance + this floor²)
MIN_LOCAL_AGREEMENT = 0.5       # no isochrones where the weighted vessel phases around a point disagree (mean resultant length below this)
LABEL_CLEARANCE_PX = 120.0      # isochrone labels keep this distance (mosaic px) from the boundary markers
LABEL_MIN_LENGTH_PX = 400.0     # isochrone pieces shorter than this get no label


def distinct_tiles(rs):
    return sorted({r["t"] for r in rs})


def repeated(recs):
    """vessels recorded in at least two tiles"""
    return {e: rs for e, rs in recs.items() if len(distinct_tiles(rs)) >= 2}


def waveform_disagreement(recs, psi):
    """{vessel: (d, tiles)} for the vessels recorded in ≥ 2 tiles: d = time-RMS difference between each clock-aligned
    recording (divided by its own mean) and their weighted mean, as a fraction of the mean flow; weights 1/σ² of the
    normalised recording (a0²/σ²)."""
    N = np.arange(1, K_FIT + 1); out = {}
    for e, rs in repeated(recs).items():
        al = np.array([r["H"] / r["a0"] * np.exp(-1j * N * psi[r["t"]]) for r in rs]); w = np.array([r["a0"] ** 2 / r["sigma2"] for r in rs])
        mean = (w[:, None] * al).sum(0) / w.sum()
        out[e] = (float(np.sqrt(0.5 * np.sum(w[:, None] * np.abs(al - mean) ** 2) / w.sum())), [r["t"] for r in rs])
    return out


def vein_reference(G, dom, psi, tf0, tile=ZERO_TILE):
    """Phase of the vitelline-vein waveform (coherence-weighted circular mean of the H1 phases of the vein stubs, each in its
    own flow direction, on the common clock)."""
    z, w = [], []
    for n in dom["V"]:
        for e in G.edges(n):
            for m in G.edges[e].get("measurements_piv") or []:
                if int(m["tile_id"]) == tile and m.get("Q_t") is not None and (m.get("quality") or {}).get("profile_ok"):
                    a0, H = phasors(m["Q_t"], tf0[tile], K=K_FIT); rho = coherence(m["Q_t"], tf0[tile], K=K_FIT)
                    z.append(np.exp(1j * np.angle(np.sign(a0) * H[0] * np.exp(-1j * psi[tile])))); w.append(max(rho, 0.0))
    return float(np.angle(np.sum(np.array(w) * np.array(z)))), len(z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", default=FIELD, help="results/<field>.pkl (scripts/pulse_field.py)")
    ap.add_argument("--example", default=None, help='vessel "u,v" of panel b (default: the paper\'s)')
    a = ap.parse_args()
    plos_style.apply()
    G = P.load_graph(); A = pickle.loads((P.RESULTS / f"{a.field}.pkl").read_bytes())
    E = [tuple(e) for e in A["edges"]]; eix = {e: i for i, e in enumerate(E)}; Qp = np.asarray(A["Qp"], float)
    psi = {int(t): float(v) for t, v in A["psi"].items()}; ref = int(A["ref_tile"]); band = {int(t) for t in A["band"]}
    tf0 = {int(t): float(f) for t, f in G.graph["tile_f0_piv"].items()}; f_ref = float(A["f_ref"])
    ALLT = sorted(tf0); OUTT = [t for t in ALLT if t not in band]; N = np.arange(1, K_FIT + 1)
    dom = carved_domain(G); assert [tuple(e) for e in dom["edges"]] == E

    # clock measurements of every tile pair, out-of-band tiles included (for panel c), exactly as the pipeline forms them
    excl = set(A.get("excluded_tiles") or []); every = E + [tuple(e) for e in G.edges if tuple(e) not in eix and tuple(e)[::-1] not in eix]
    FP = flux_pairs(recordings(G, every, set(ALLT))); lo, hi = np.minimum(FP["a"], FP["b"]), np.maximum(FP["a"], FP["b"])
    links_all = {(int(x), int(y)): int(n) for (x, y), n in zip(*np.unique(np.c_[lo, hi], axis=0, return_counts=True))}
    links = {k: n for k, n in links_all.items() if k[0] in psi and k[1] in psi}; TILES = sorted(psi)
    psi_all, _ = tile_clocks(FP, ref)                                                                    # all tiles, for panel c only
    rec_all = recordings(G, E, set(ALLT)); rec_all = {e: [r for r in rs if r["t"] in psi_all] for e, rs in rec_all.items()}
    rec = {e: [r for r in rs if r["t"] in psi] for e, rs in rec_all.items()}; rec = {e: rs for e, rs in rec.items() if rs}
    dis = waveform_disagreement(rec_all, psi_all)
    shift = np.degrees(np.abs(np.angle(np.exp(1j * np.array([psi_all[t] - psi[t] for t in TILES])))))
    spread = lambda ts: (max(tf0[t] for t in ts) - min(tf0[t] for t in ts)) / np.mean([tf0[t] for t in ts])
    d_in = [d for e, (d, ts) in dis.items() if e in eix and not any(t in OUTT for t in ts)]
    print(f"[fig6] {len(psi)} tile clocks from {sum(links.values())} vessel pairs between retained tiles ({len(FP['d'])} with the out-of-band tiles); not synchronised: {sorted(excl)}; "
          f"clock fit with the out-of-band tiles added moves the retained clocks by median {np.median(shift):.1f}°, max {shift.max():.1f}°")
    print(f"[fig6] RMS waveform difference: retained tiles median {100 * np.median(d_in):.1f} %; per excluded tile " +
          ", ".join(f"{t}: {100 * np.median([d for d, ts in dis.values() if t in ts]):.0f} %" for t in OUTT if any(t in ts for _, ts in dis.values())))

    EX = tuple(int(x) for x in a.example.split(",")) if a.example else EXAMPLE; EX = EX if EX in rec else EX[::-1]
    rex = sorted(rec[EX], key=lambda r: r["t"]); tcol = {r["t"]: c for r, c in zip(rex, TILE_COLS)}
    print(f"[fig6] example vessel {EX}: tiles {[r['t'] for r in rex]}, RMS difference {100 * dis[EX][0]:.1f} %")

    # field: delay behind the vitelline veins, H1 in each vessel's own flow direction
    ok = np.asarray(A["elig"], bool); g1 = np.where(ok, A["H"][0] / Qp, np.nan + 0j)
    zV, nV = vein_reference(G, dom, psi, tf0)
    lag = (np.degrees(np.angle(np.exp(1j * zV) * np.conj(g1))) - WRAP_DEG) % 360 + WRAP_DEG      # [WRAP_DEG, WRAP_DEG + 360): the wrap sits in the empty part of the distribution
    w_e = 1.0 / (np.nan_to_num(np.asarray(A["var_phi"], float)[0], nan=np.inf) + np.radians(PHASE_SD_FLOOR_DEG) ** 2)      # precision of each vessel's H1 phase
    print(f"[fig6] field on {ok.sum()} vessels; zero = {nV} vein recordings in tile {ZERO_TILE}; delay in [0°, 180°) for {100 * np.mean((lag[ok] >= 0) & (lag[ok] < 180)):.0f} %")

    MOS = load_mosaic(P.STAGE); Hm = MOS["outside"].shape[0]; geom = load_tile_geometry(StageConfig.load(P.STAGE))
    def centre(t):
        tg = geom.tiles[int(t)]; Hv, Wv = tg.video_shape; c = np.array([Hv / 2, Wv / 2, 1.0]) @ tg.A_full.T
        r_, c_ = c[:2] / geom.downsample - [geom.y0, geom.x0]; return np.array([c_, r_])
    def footprint(t):
        tg = geom.tiles[int(t)]; Hv, Wv = tg.video_shape
        cr = np.array([[0, 0, 1], [0, Wv, 1], [Hv, Wv, 1], [Hv, 0, 1]], float) @ tg.A_full.T
        rc = cr[:, :2] / geom.downsample - [geom.y0, geom.x0]; return MPath(np.column_stack([rc[:, 1], rc[:, 0]]))
    th = np.linspace(0, 1, 300); wave = lambda g: 1 + sum((g[k] * np.exp(2j * np.pi * (k + 1) * th)).real for k in range(len(g)))

    # layout: a, b, c stacked on the left; the map d on the right, the left column spanning the map height
    x0, x1, y1, y0 = MOS["extent"]; ratio = abs(y1 - y0) / abs(x1 - x0)
    W, xd0, xd1 = 7.5, 0.535, 0.995; mh = (xd1 - xd0) * W * ratio; mb, mt = 0.5, 0.5; Hf = mb + mh + mt
    fig = plt.figure(figsize=(W, Hf))
    L = fig.add_gridspec(1, 1, left=0.085, right=0.505, top=(mb + mh + 0.2) / Hf, bottom=0.45 / Hf)[0, 0].subgridspec(
        3, 1, height_ratios=[0.6, 1.3, 0.95], hspace=0.62)
    B = L[1].subgridspec(2, 2, width_ratios=[1.25, 1], wspace=0.28, hspace=0.45)

    # ---- a: heart rate during acquisition
    axa = ax = fig.add_subplot(L[0])
    ax.axhspan(f_ref * (1 - 0.21), f_ref * (1 + 0.21), color="0.93", lw=0, zorder=0)
    ax.plot(ALLT, [tf0[t] for t in ALLT], color="0.7", lw=0.7, zorder=1)
    for t in ALLT:
        ax.plot(t, tf0[t], "o", ms=3.6, mfc=EXCOL if t in OUTT else "0.25", mec="none", zorder=3)
    ax.plot(ref, tf0[ref], "o", ms=7.5, mfc="none", mec="k", mew=1.2, zorder=4)
    ax.text(ref + 2.5, tf0[ref] - 0.22, r"$f_{\mathrm{ref}}$", fontsize=8, va="top")
    if OUTT:
        ax.text(8, 1.05, f"tiles {min(OUTT)}–{max(OUTT)} excluded", color=EXCOL, fontsize=8, ha="left", va="center")
    ax.set_xlim(0, max(ALLT) + 1); ax.set_ylim(0.8, 3.2); ax.set_xlabel("Tile (acquisition order)", labelpad=1); ax.set_ylabel(r"Heart rate $f_0$ (Hz)")
    ax.set_title("Heart rate during acquisition", fontsize=9)

    # ---- b: tile clocks from shared vessels; one vessel before / after
    axb = ax = fig.add_subplot(B[:, 0]); show_mosaic(ax, MOS); ax.images[0].set_alpha(0.28); ax.axis("off")
    C = {t: centre(t) for t in sorted(set(TILES) | set(OUTT) | excl)}; nmax = max(links.values())
    for (ta, tb), n in links.items():
        ax.plot([C[ta][0], C[tb][0]], [C[ta][1], C[tb][1]], color="0.55", lw=0.2 + 1.4 * np.sqrt(n / nmax), alpha=0.8, zorder=2)
    for (ta, tb), n in links_all.items():
        if ta in OUTT or tb in OUTT:
            ax.plot([C[ta][0], C[tb][0]], [C[ta][1], C[tb][1]], color=EXCOL, lw=0.4 + 1.4 * np.sqrt(n / nmax), alpha=0.85, zorder=2)
    for t in TILES:
        ax.plot(*C[t], "o", ms=4, mfc="0.78", mec="0.4", mew=0.4, zorder=3)
    for t in OUTT:
        ax.plot(*C[t], "o", ms=4, mfc=EXCOL, mec="0.3", mew=0.4, zorder=3)
    for t in excl:                                             # clock too uncertain: not synchronised
        ax.plot(*C[t], "o", ms=4, mfc="white", mec="0.4", mew=0.6, zorder=3)
    ax.plot(*C[ref], "o", ms=6, mfc="white", mec="k", mew=1.3, zorder=4)
    halo = [pe.withStroke(linewidth=2, foreground="white")]
    ax.text(C[ref][0] - 60, C[ref][1], "ref", fontsize=8, ha="right", va="center", zorder=6, path_effects=halo)
    pex = np.asarray(G.edges[EX]["path"], float); xy = np.column_stack([pex[:, 0], Hm - pex[:, 1]])
    ax.plot(xy[:, 0], xy[:, 1], color=EXAMPLE_COL, lw=2.0, zorder=5)
    for t, c_ in tcol.items():
        ax.plot(*C[t], "o", ms=6.5, mfc=c_, mec="k", mew=0.5, zorder=5); dx = 1 if C[t][0] >= xy.mean(0)[0] else -1
        ax.text(C[t][0] + dx * 70, C[t][1] - 40, str(t), fontsize=8, color=c_, fontweight="bold", ha="left" if dx > 0 else "right", va="center",
                zorder=6, path_effects=halo)
    ax.plot(*xy.mean(0), "o", ms=11, mfc="none", mec=EXAMPLE_COL, mew=1.5, zorder=7)
    if OUTT:
        oc = np.array([C[t] for t in OUTT]).mean(0)
        ax.text(oc[0], oc[1] - 230, f"{min(OUTT)}–{max(OUTT)} excluded", color=EXCOL, fontsize=8, ha="center", va="bottom", fontweight="bold", path_effects=halo, zorder=8)
    ax.set_title(f"{len(psi)} tile clocks from shared vessels", fontsize=8, pad=2)
    curves = {rot: [(r["t"], wave(r["H"] / r["a0"] * (np.exp(-1j * N * psi[r["t"]]) if rot else 1))) for r in rex] for rot in (False, True)}
    yy = np.concatenate([y for cs in curves.values() for _, y in cs]); axw = []
    for k, rot in enumerate((False, True)):
        ax = fig.add_subplot(B[k, 1]); axw.append(ax)
        for t, y in curves[rot]:
            ax.plot(th, y, color=tcol[t], lw=1.4)
        ax.axhline(1, color="0.85", lw=0.5); ax.set_xlim(0, 1); ax.set_ylim(yy.min() - 0.1, yy.max() + 0.1); ax.set_yticks([0.5, 1.0, 1.5])
        ax.set_title("After synchronization" if rot else "Before synchronization", fontsize=8, pad=2); ax.set_xticks([0, 0.5, 1])
        if rot:
            ax.set_xlabel(r"Cardiac phase $\theta$", labelpad=0); ax.set_xticklabels(["0", "0.5", "1"])
        else:
            ax.set_xticklabels([])
    fig.canvas.draw(); p1, p2 = axw[0].get_position(), axw[1].get_position()
    fig.text(p2.x0 - 0.045, 0.5 * (p1.y1 + p2.y0), "Normalized flow", rotation=90, ha="center", va="center", fontsize=9)

    # ---- c: waveform agreement vs heart-rate spread
    axc = ax = fig.add_subplot(L[2])
    dv = np.array([d for d, ts in dis.values()]); fv = np.array([spread(ts) for d, ts in dis.values()])
    out = np.array([any(t in OUTT for t in ts) for d, ts in dis.values()]); X = 100 * fv + 0.5
    ax.scatter(X[~out], 100 * dv[~out], s=2, color="0.55", alpha=0.35, lw=0, rasterized=True, label="retained tiles")
    if out.any():
        ax.scatter(X[out], 100 * dv[out], s=6, color=EXCOL, alpha=0.75, lw=0, zorder=3, label=f"tiles {min(OUTT)}–{max(OUTT)} (excluded)")
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]; cen, med = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fv >= lo) & (fv < hi)
        if m.sum() >= 8:
            cen.append(100 * (lo + hi) / 2 + 0.5); med.append(100 * np.median(dv[m]))
    ax.plot(cen, med, "-o", color="k", ms=3.2, lw=1.2, zorder=4, label="median")
    ax.plot(100 * spread(dis[EX][1]) + 0.5, 100 * dis[EX][0], "*", ms=10, color=EXAMPLE_COL, mec="white", mew=0.5, zorder=6, label="vessel in b")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.4, 80); ax.set_ylim(0.3, 2500); ax.set_yticks([1, 10, 100])
    pct = mticker.FuncFormatter(lambda v, _: f"{v:g}%"); ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Heart-rate difference"); ax.set_ylabel("RMS difference", labelpad=1)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.1, borderaxespad=0.2, markerscale=1.6, ncol=2, columnspacing=0.6, handlelength=1.2)
    ax.set_title("Waveform agreement", fontsize=9)

    # ---- d: phase field — vessel delays and isochrones of the smoothed field
    um_px = 1.7 * geom.downsample; stp = 6.0; nx, ny = int((x1 - x0) / stp) + 1, int((y1 - y0) / stp) + 1
    Zg = np.zeros((ny, nx), complex); Cg = np.zeros((ny, nx)); Wg = np.zeros((ny, nx)); segs, cl = [], []; foot_out = [footprint(t) for t in OUTT]; n_hidden = 0
    for i in np.flatnonzero(ok):
        p_ = G.edges[E[i]].get("path")
        if p_ is None:
            continue
        p_ = np.asarray(p_, float); p_ = np.c_[p_[:, 0], Hm - p_[:, 1]]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(p_, axis=0).T))]
        ss = np.linspace(0, sl[-1], max(int(sl[-1] / 2.0), 2)); q = np.c_[np.interp(ss, sl, p_[:, 0]), np.interp(ss, sl, p_[:, 1])]
        if any(fp.contains_point(q.mean(0)) for fp in foot_out):           # inside an excluded tile: not shown
            n_hidden += 1; continue
        segs.append(q); cl.append(lag[i])
        ix = np.clip(((q[:, 0] - x0) / stp).astype(int), 0, nx - 1); iy = np.clip(((q[:, 1] - y0) / stp).astype(int), 0, ny - 1)
        np.add.at(Zg, (iy, ix), w_e[i] * np.exp(1j * np.radians(lag[i]))); np.add.at(Wg, (iy, ix), w_e[i]); np.add.at(Cg, (iy, ix), 1.0)
    print(f"[fig6] d: {n_hidden} vessels inside the excluded tiles not shown")
    sg = SIGMA_UM / um_px / stp; Zs = gaussian_filter(Zg.real, sg) + 1j * gaussian_filter(Zg.imag, sg); Cs = gaussian_filter(Cg, sg)
    coh = np.abs(Zs) / np.maximum(gaussian_filter(Wg, sg), 1e-30)       # local agreement of the vessel phases (1 = all equal, 0 = scattered)
    gx = x0 + stp * np.arange(nx); gy = y0 + stp * np.arange(ny)
    outside = MOS["outside"][np.clip(gy.astype(int), 0, MOS["outside"].shape[0] - 1)][:, np.clip(gx.astype(int), 0, MOS["outside"].shape[1] - 1)]
    cov = gaussian_filter(Cg, 60.0 / um_px / stp)
    hide = (Cs <= 0.15 * np.percentile(Cs[Cs > 0], 75)) | binary_dilation(outside, iterations=3) | (cov < 0.05 * np.percentile(cov[cov > 0], 75)) | (coh < MIN_LOCAL_AGREEMENT)
    F = np.ma.masked_where(hide, (np.degrees(np.angle(Zs)) - WRAP_DEG) % 360 + WRAP_DEG)
    vtop = 20 * np.ceil(np.percentile(np.ma.compressed(np.ma.masked_where(Cs <= 0.15 * np.percentile(Cs[Cs > 0], 75), F.data)), 98) / 20)
    norm = mcolors.Normalize(0, vtop); cmap = plt.cm.plasma
    axd = ax = fig.add_axes([xd0, mb / Hf, xd1 - xd0, mh / Hf]); show_mosaic(ax, MOS); ax.images[0].set_alpha(0.2); ax.axis("off")
    ax.imshow(F, cmap=cmap, norm=norm, extent=(gx[0], gx[-1], gy[-1], gy[0]), origin="upper", alpha=0.2, zorder=1, interpolation="bilinear")
    ax.add_collection(LineCollection(segs, colors=cmap(norm(np.clip(cl, 0, vtop))), linewidths=0.75, zorder=2))
    levels = list(np.arange(30, vtop + 1, 30)); cs = ax.contour(gx, gy, F, levels=levels, colors="0.1", linewidths=0.7, alpha=0.85, zorder=3)
    # one label per isochrone piece, at the vertex farthest from the boundary markers (never on top of them)
    marks = np.array([[G.nodes[dom["nodes"][j]]["x"], Hm - G.nodes[dom["nodes"][j]]["y"]] for j in list(dom["inj"]) + list(dom["out"])]); spots = []
    for segs_l in cs.allsegs:
        for sg_ in segs_l:
            if len(sg_) < 8 or np.sum(np.hypot(*np.diff(sg_, axis=0).T)) < LABEL_MIN_LENGTH_PX:
                continue
            k = np.arange(len(sg_) // 6, len(sg_) - len(sg_) // 6); dm = np.min(np.hypot(sg_[k, None, 0] - marks[None, :, 0], sg_[k, None, 1] - marks[None, :, 1]), axis=1)
            good = k[dm >= LABEL_CLEARANCE_PX]
            if len(good):
                spots.append(tuple(sg_[good[np.argmin(np.abs(good - len(sg_) // 2))]]))       # the clear vertex nearest the middle of the piece
    labels = ax.clabel(cs, levels=levels, fmt=lambda v: f"{v:.0f}°", fontsize=8, inline=True, inline_spacing=3, manual=spots)
    for t_ in labels:
        t_.set_fontweight("bold"); t_.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_anchor("N"); ax.set_aspect("auto")
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); kept = []
    for t_ in sorted(labels, key=lambda t_: float(t_.get_text().rstrip("°"))):          # drop overlapping isochrone labels
        bb = t_.get_window_extent(rend).expanded(1.1, 1.2)
        if any(bb.overlaps(k_) for k_ in kept):
            t_.set_visible(False)
        else:
            kept.append(bb)
    nodes = dom["nodes"]
    for idx, mk, fc, lab in ((dom["inj"], "^", "#d7191c", "arterial inflow"), (dom["out"], "s", "#2c7bb6", "venous outflow")):
        pts = np.array([[G.nodes[nodes[j]]["x"], Hm - G.nodes[nodes[j]]["y"]] for j in idx])
        ax.plot(pts[:, 0], pts[:, 1], mk, ms=5 if mk == "^" else 5.5, mfc=fc, mec="white", mew=0.7, ls="", zorder=7, label=lab)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=2, handletextpad=0.1, borderaxespad=0.0, columnspacing=1.2)
    ax.set_title("Fundamental (H1) phase field", fontsize=9, pad=16)
    cax = ax.inset_axes([0.22, -0.24 / mh, 0.56, 0.09 / mh])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal", ticks=list(range(0, int(vtop) + 1, 60)))
    cb.set_label(r"Delay behind vitelline veins ($^\circ$)", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()      # colour-bar label level with panel c's x label
    yc = axc.xaxis.label.get_window_extent(rend).transformed(inv).y0; yb = cax.xaxis.label.get_window_extent(rend).transformed(inv).y0
    pc = cax.get_position(); cax.set_axes_locator(None); cax.set_position([pc.x0, pc.y0 + (yc - yb), pc.width, pc.height])
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for ax_, letter in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        tb = ax_.title.get_window_extent(rend).transformed(inv)
        fig.text(ax_.get_position().x0 + 0.003 if letter == "d" else 0.006, tb.y0, letter, fontsize=12, fontweight="bold", va="bottom")
    plos_style.save(fig, P.FIGURES / "fig6" / "fig6", finalize_fig=False)


if __name__ == "__main__":
    main()
