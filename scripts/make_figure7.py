"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  tiles on the organ, coloured by their heart rate f0; tiles outside the accepted band (|f0/f_ref − 1| > 0.15) hatched;
     reference tile outlined
  b  one vessel imaged in two overlapping tiles: its waveform Q(t)/Q̄ in each, as recorded (raw time origins) and after
     the tile gauges
  c  the synchronisation network: tiles linked by the vessels they share (line width ∝ Σ ρ_a ρ_b), coloured by ψ_t;
     open circle = gauge less certain than 30° (bootstrap SD)
  d  validation on vessels imaged in two tiles (both ρ ≥ 0.1): cross-tile phase difference of H1 (used to fit ψ) and H2
     (not used; aligned with 2ψ), raw vs aligned
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

    # example vessel for b: both tiles coherent, the largest raw H1 offset
    best = None
    for e, a, b in pairs:
        xa, xb = meas[e][a], meas[e][b]
        if min(xa["rho"], xb["rho"]) < 0.5 or abs(xa["a0"]) < 0.05:
            continue
        off = abs(np.angle(xa["H"][0] * np.conj(xb["H"][0]))); al = abs(np.angle(xa["H"][0] * np.exp(-1j * psi[a]) * np.conj(xb["H"][0] * np.exp(-1j * psi[b]))))
        score = off - 2 * al
        if best is None or score > best[0]:
            best = (score, e, a, b)
    _, ex_e, ex_a, ex_b = best
    log(f"  (b) example {ex_e}: tiles {ex_a}/{ex_b}, ρ {meas[ex_e][ex_a]['rho']:.2f}/{meas[ex_e][ex_b]['rho']:.2f}")

    mos = tifffile.imread(str(_paths.MOSAIC)); mos = mos.mean(axis=-1) if mos.ndim == 3 else mos; Hm, Wm = mos.shape
    polys = tile_polygons(_paths.STAGE); cen = {t: p.mean(0) for t, p in polys.items()}

    fig = plt.figure(figsize=(7.5, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, 1], hspace=0.28, wspace=0.12, left=0.07, right=0.93, top=0.96, bottom=0.07)

    def label(cell, letter):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.012, b_.y1 + 0.012, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    allp = np.vstack([polys[t] for t in polys if t in tf0]); pad_ = 20
    XL, YL = (allp[:, 0].min() - pad_, allp[:, 0].max() + pad_), (allp[:, 1].max() + pad_, allp[:, 1].min() - pad_)

    def organ(ax):
        ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), zorder=0, alpha=0.55)
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]

    # ---- a: tile heart rates
    axA = fig.add_subplot(gs[0, 0]); organ(axA)
    fin = [tf0[t] for t in band]; nf = mcolors.Normalize(min(fin), max(fin)); cmf = plt.cm.cividis
    for t, p in polys.items():
        if t not in tf0:
            continue
        inb = t in band
        axA.add_patch(Polygon(p, closed=True, fc=cmf(nf(tf0[t])) if inb else "none", ec="0.25" if inb else "0.4", lw=0.4, alpha=0.45 if inb else 1,
                              hatch=None if inb else "////", zorder=2))
    axA.add_patch(Polygon(polys[ref], closed=True, fc="none", ec="black", lw=1.4, zorder=4))
    sm = plt.cm.ScalarMappable(norm=nf, cmap=cmf); cb = fig.colorbar(sm, ax=axA, fraction=0.04, pad=0.01, shrink=0.7); cb.set_label("tile heart rate $f_0$ (Hz)"); cb.ax.tick_params(labelsize=7)
    axA.text(0.02, 0.02, f"reference tile ({f_ref:.2f} Hz) outlined\nhatched: outside ±{100 * F0_BAND:.0f}% of reference", transform=axA.transAxes, fontsize=7, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
    label(gs[0, 0], "a")

    # ---- c: synchronisation network coloured by ψ
    axC = fig.add_subplot(gs[0, 1]); organ(axC)
    for t in band:
        axC.add_patch(Polygon(polys[t], closed=True, fc="none", ec="0.6", lw=0.3, zorder=1))
    wmax = max(link.values()); segs, lws = [], []
    for (a, b), w in link.items():
        segs.append([cen[a], cen[b]]); lws.append(0.3 + 2.2 * np.sqrt(w / wmax))
    axC.add_collection(LineCollection(segs, linewidths=lws, colors="0.25", alpha=0.6, zorder=2))
    cmp = plt.cm.twilight; npsi = mcolors.Normalize(-180, 180)
    for t in band:
        ok = sd.get(t, np.inf) <= SD_MAX; col = cmp(npsi(np.degrees(psi[t])))
        axC.plot(*cen[t], "o", ms=6.5, mfc=col if ok else "white", mec=col if not ok else "0.15", mew=1.4 if not ok else 0.5, zorder=4)
    axC.plot(*cen[ref], "s", ms=8, mfc="none", mec="black", mew=1.2, zorder=5)
    sm = plt.cm.ScalarMappable(norm=npsi, cmap=cmp); cb = fig.colorbar(sm, ax=axC, fraction=0.04, pad=0.01, shrink=0.7, ticks=[-180, -90, 0, 90, 180])
    cb.set_label("tile time gauge $\\psi_t$ (deg)"); cb.ax.tick_params(labelsize=7)
    n_unc = sum(1 for t in band if sd.get(t, np.inf) > SD_MAX)
    axC.text(0.02, 0.02, f"links: vessels shared by two tiles\nopen: gauge SD > {SD_MAX:.0f}° ({n_unc} tiles)", transform=axC.transAxes, fontsize=7, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
    label(gs[0, 1], "c")

    # ---- b: one vessel in two tiles, raw and aligned
    gB = gs[1, 0].subgridspec(1, 2, wspace=0.16); t_ = np.linspace(0, 1, 300); TC = ["#0072B2", "#D55E00"]
    for j, mode in enumerate(("raw", "aligned")):
        ax = fig.add_subplot(gB[j])
        for k_, t in enumerate((ex_a, ex_b)):
            x = meas[ex_e][t]; H = x["H"] * (np.exp(-1j * np.arange(1, 4) * psi[t]) if mode == "aligned" else 1)
            q = x["a0"] + sum((H[n - 1] * np.exp(2j * np.pi * n * t_)).real for n in (1, 2, 3)); q = q / x["a0"]
            ax.plot(t_, q, color=TC[k_], lw=1.3, label=f"tile {t}")
        ax.set_xlabel("$t\\,/\\,T$"); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("as recorded" if mode == "raw" else "after tile gauges", fontsize=8.5, pad=3)
        if j == 0:
            ax.set_ylabel("$Q(t)\\,/\\,\\bar Q$"); ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.5)
        else:
            ax.set_yticklabels([])
    yl = [min(a_.get_ylim()[0] for a_ in fig.axes[-2:]), max(a_.get_ylim()[1] for a_ in fig.axes[-2:])]
    for a_ in fig.axes[-2:]:
        a_.set_ylim(*yl)
    label(gs[1, 0], "b")

    # ---- d: validation, cross-tile phase differences
    gD = gs[1, 1].subgridspec(2, 1, hspace=0.18); bins = np.linspace(-180, 180, 37)
    for r_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gD[r_])
        ax.hist(d[k]["raw"], bins=bins, density=True, histtype="step", color="0.5", lw=1.0, label="as recorded")
        ax.hist(d[k]["aligned"], bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.3, label="after tile gauges")
        ax.set_xlim(-180, 180); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-180, -90, 0, 90, 180])
        ax.text(0.01, 0.95, f"H{k}" + ("" if k == 1 else "  (not used to fit $\\psi$)"), transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(0.99, 0.95, f"median |Δφ|  {np.median(np.abs(d[k]['raw'])):.0f}° → {np.median(np.abs(d[k]['aligned'])):.0f}°", transform=ax.transAxes, fontsize=7.5, ha="right", va="top")
        if r_ == 0:
            ax.set_xticklabels([]); ax.legend(fontsize=7, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
        else:
            ax.set_xlabel("phase difference between tiles, same vessel (deg)")
    label(gs[1, 1], "d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig7.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / "fig7.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig7.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    log(f"wrote {OUT_DIR / 'fig7.png'}")


if __name__ == "__main__":
    main()
