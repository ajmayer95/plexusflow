"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  heart rate f0 of every tile vs tile number; tiles within ±15 % of the reference tile are retained (shaded band)
  b  H1 phase delay from the driven boundary, tile-averaged: each accepted tile's ρ-weighted circular mean of its vessels'
     aligned H1 phase (relative to their flow direction), subtracted from the boundary phase (circular mean of the field's
     vessels within 1 hop of the arterial carve boundary and venous outflow); positive = peaks later; star = venous outflow
  supplement (fig7_supp_heldout): held-out overlap prediction — for every overlapping tile pair with ≥ 5 shared coherent
     vessels that is not a bridge, all links between the two tiles removed, ψ refitted, ψ_a − ψ_b predicted vs measured
  c  validation on vessels imaged in two tiles: same-vessel phase difference between tiles after alignment, H1 (fitted)
     and H2 (not fitted; aligned with 2ψ), median |Δφ|; raw medians in the log (for the caption)
  (waveform scaling to Q′ is shown in fig. 9; one vessel's waveform in two tiles is a supplement panel)
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
    for t, tg in geom.tiles.items():
        H, W = tg.video_shape; corners = np.array([[0, 0, 1], [0, W, 1], [H, W, 1], [H, 0, 1]], float)
        rm, cm, _ = (tg.A_full @ corners.T)
        out[t] = np.c_[cm / geom.downsample - geom.x0, rm / geom.downsample - geom.y0]       # (col, row) in the fused image
    return out


def main():
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    G = _paths.load_graph(); dc = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    log("rebuilding the AC field (per-tile phasors and synchronisation pairs)")
    F = build_ac_field(G, dc); meas, psi, pairs, sd = F["_meas"], F["psi"], F["_pairs"], F["gauge_sd_deg"]
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

    fig = plt.figure(figsize=(7.5, 3.3), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.38, left=0.07, right=0.985, top=0.9, bottom=0.17)

    def label(cell, letter):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.045, b_.y1 + 0.03, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- a: which tiles are comparable — heart rate per tile, accepted band
    axA = fig.add_subplot(gs[0]); ts = sorted(tf0); f = np.array([tf0[t] for t in ts]); inb = np.array([t in band for t in ts])
    axA.axhspan(f_ref * (1 - F0_BAND), f_ref * (1 + F0_BAND), color="#e9eef4", zorder=0, lw=0)
    axA.plot(np.array(ts)[inb], f[inb], "o", ms=2.8, color="0.15", zorder=3)
    axA.plot(np.array(ts)[~inb], f[~inb], "o", ms=2.8, color="0.72", zorder=3)
    axA.plot(ref, f_ref, "s", ms=5, mfc="white", mec="black", mew=0.9, zorder=4)
    axA.text(0.97, f_ref * (1 - F0_BAND) + 0.04, "accepted $f_0$ range", transform=axA.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="0.4")
    axA.annotate("reference", xy=(ref, f_ref), xytext=(ref - 3, f_ref + 0.22), fontsize=7, color="0.3", ha="right", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
    axA.set_xlabel("tile number"); axA.set_ylabel("heart rate $f_0$ (Hz)"); axA.spines[["top", "right"]].set_visible(False)
    label(gs[0], "a")

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
    # zero = the dorsal aorta's H1 phase in the reference tile (both DA stubs, ρ-weighted): the arterial flow peak
    from plexusflow.reconcile.harmonics import phasors as _ph, coherence as _coh
    zda, wda = [], []
    for n_ in dom["DA"]:
        for e_ in G.edges(n_):
            for m in G.edges[e_].get("measurements_piv") or []:
                if int(m["tile_id"]) == ref and m.get("Q_t") is not None and (m.get("quality") or {}).get("kl_ok"):
                    a0_, H_ = _ph(m["Q_t"], tf0[ref]); r_ = _coh(m["Q_t"], tf0[ref])
                    zda.append(np.exp(1j * np.angle(np.sign(a0_) * H_[0] * np.exp(-1j * psi[ref])))); wda.append(max(r_, 0))
    phi_b = np.angle(np.sum(np.array(wda) * np.array(zda)))
    log(f"  (b) zero = dorsal aorta H1 phase in tile {ref}: {np.degrees(phi_b):.1f}° ({len(zda)} DA measurements); "
        f"driven boundary {np.degrees(phi_bnd):.1f}° → boundary lags the DA by {np.degrees(np.angle(np.exp(1j * (phi_b - phi_bnd)))):.0f}°")
    tdel, tR, nflip, ntot = {}, {}, {}, {}; eixf = {e: i for i, e in enumerate(Ef)}
    for t in band:
        zs, ws = [], []
        for e, x in meas.items():
            y = x.get(t)
            if y is None or not np.isfinite(y["rho"]) or y["rho"] < 0.1 or abs(y["H"][0]) == 0:
                continue
            i_ = eixf.get(e)
            if i_ is None or F["Qp"][i_] == 0:
                continue
            nflip[t] = nflip.get(t, 0) + int(np.sign(y["a0"]) != np.sign(F["Qp"][i_])); ntot[t] = ntot.get(t, 0) + 1
            zs.append(np.exp(1j * np.angle(np.sign(F["Qp"][i_]) * y["H"][0] * np.exp(-1j * psi[t])))); ws.append(y["rho"])
        if len(zs) >= 5:
            S = np.sum(np.array(ws) * np.array(zs)); tdel[t] = np.degrees(np.angle(np.exp(1j * phi_b) * np.conj(S))); tR[t] = abs(S) / np.sum(ws)
    fr = {t: nflip.get(t, 0) / ntot[t] for t in ntot}
    log("  (b) tiles where the measured mean runs against Q′ for > 30 % of vessels: " + (", ".join(f"{t} ({100 * v:.0f}%)" for t, v in sorted(fr.items()) if v > 0.3) or "none"))
    vals = np.array(list(tdel.values()))
    log(f"  (b) boundary H1 phase {np.degrees(phi_b):.0f}° from {nb.sum()} vessels ≤ 1 hop; tile delays {vals.min():.0f}° … {vals.max():.0f}° "
        f"(median {np.median(vals):.0f}°); within-tile R median {np.median(list(tR.values())):.2f}")
    log("  (b) tile delay° (R): " + ", ".join(f"{t}:{tdel[t]:.0f}({tR[t]:.2f})" for t in sorted(tdel)))
    axM = fig.add_subplot(gs[1]); organ(axM)
    for t, pgon in polys.items():
        if t in tf0:
            axM.add_patch(Polygon(pgon, closed=True, fc="none", ec="0.55" if t in band else "0.8", lw=0.35, zorder=1))
    # delays as positive lags: wrap into [−30°, 330°) so small negatives near the boundary stay ≈ 0; colour range [0, max]
    tdel = {t: (v + 30) % 360 - 30 for t, v in tdel.items()}
    dmax = max(v for t, v in tdel.items() if sd.get(t, np.inf) <= SD_MAX); vtop = 10 * np.ceil(dmax / 10)
    log(f"  (b) max delay over determined tiles {dmax:.0f}° (tile {max((v, t) for t, v in tdel.items() if sd.get(t, np.inf) <= SD_MAX)[1]})")
    cmp = plt.cm.cividis_r; nph = mcolors.Normalize(0, vtop)
    for t, v in tdel.items():
        ok = sd.get(t, np.inf) <= SD_MAX; col = cmp(nph(v))
        if ok:
            axM.plot(*cen[t], "o", ms=3 + 5 * tR[t], mfc=col, mec="0.15", mew=0.4, zorder=4)
        else:                                                     # tile offset undetermined (SD > 30°): no phase shown
            axM.plot(*cen[t], "o", ms=5, mfc="white", mec="0.55", mew=0.9, zorder=4)
    for i in [int(i) for i in dom["out"]]:                                   # venous outflow nodes, for orientation
        n_ = nds[i]; xy = (G.nodes[n_]["x"], mos.shape[0] - G.nodes[n_]["y"])
        axM.plot(*xy, marker="*", ms=7, color="white", mec="black", mew=0.6, zorder=6)
    sm = plt.cm.ScalarMappable(norm=nph, cmap=cmp)
    cb = fig.colorbar(sm, ax=axM, fraction=0.045, pad=0.01, shrink=0.8, ticks=[0, 60, 120, 180] if vtop >= 180 else [0, vtop / 2, vtop], extend="min")
    cb.set_label("H1 phase delay from dorsal aorta (deg)", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
    label(gs[1], "b")

    # ---- c: repeated-vessel agreement after alignment
    gD = gs[2].subgridspec(2, 1, hspace=0.3); bins = np.linspace(-60, 60, 31)
    for r_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gD[r_]); v = d[k]["aligned"]
        ax.hist(v, bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.3)
        ax.set_xlim(-60, 60); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-60, 0, 60])
        ax.text(0.0, 0.97, "H1" if k == 1 else "H2", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(1.0, 0.97, f"{np.median(np.abs(d[k]['aligned'])):.0f}°", transform=ax.transAxes, fontsize=8, va="top", ha="right", color="0.2")
        if r_ == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("same-vessel phase difference\nbetween tiles, aligned (deg)")
    label(gs[2], "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # supplement — held-out overlap prediction (fig7_supp_heldout)
    fs = plt.figure(figsize=(2.6, 2.6), facecolor="white"); axP = fs.add_axes([0.24, 0.22, 0.72, 0.72]); axP.set_aspect("equal")
    axP.plot([-200, 200], [-200, 200], color="0.6", lw=0.7, ls="--", zorder=1)
    axP.scatter(obs, pred, s=4 + 18 * np.sqrt(nn / nn.max()), color="0.25", alpha=0.75, lw=0, zorder=3)
    axP.set_xlim(-200, 200); axP.set_ylim(-200, 200); axP.set_xticks([-180, 0, 180]); axP.set_yticks([-180, 0, 180])
    axP.set_xlabel("tile-pair offset measured\nfrom shared vessels (deg)"); axP.set_ylabel("predicted with pair held out (deg)", labelpad=0)
    axP.spines[["top", "right"]].set_visible(False)
    axP.text(0.03, 0.97, f"{len(obs)} tile pairs\nmedian error {np.median(res):.0f}°", transform=axP.transAxes, ha="left", va="top", fontsize=7.5, color="0.3", linespacing=1.4)
    fs.savefig(OUT_DIR / "fig7_supp_heldout.png", dpi=300, facecolor="white"); fs.savefig(OUT_DIR / "fig7_supp_heldout.pdf", facecolor="white")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig7.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / "fig7.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig7.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    log(f"wrote {OUT_DIR / 'fig7.png'}")


if __name__ == "__main__":
    main()
