"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  heart rate f0 of every tile vs tile number; tiles within ±15 % of the reference tile are retained (shaded band)
  b  how ψ_t is obtained: for the three tile pairs sharing the most coherent vessels (both ρ ≥ 0.1), each shared vessel's
     raw H1 phase difference Δφ_e with the fitted offset ψ_a − ψ_b; then the fitted ψ_t of every retained tile with its
     bootstrap SD (open: SD > 30°; square: reference, ψ = 0)
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

    fig = plt.figure(figsize=(7.5, 2.9), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.6, 0.9], wspace=0.32, left=0.07, right=0.985, top=0.9, bottom=0.17)

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

    # ---- b: shared-vessel phase constraints → fitted tile gauges
    gB = gs[1].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.42)
    npair = {}
    for e, a, b in pairs:
        npair[(a, b)] = npair.get((a, b), 0) + 1
    top3 = [k for k, _ in sorted(npair.items(), key=lambda kv: -kv[1])[:3]]
    axP = fig.add_subplot(gB[0])
    for r_, (a, b) in enumerate(top3):
        dv = np.degrees([np.angle(meas[e][a]["H"][0] * np.conj(meas[e][b]["H"][0])) for e, aa, bb in pairs if (aa, bb) == (a, b)])
        fit = np.degrees(np.angle(np.exp(1j * (psi[a] - psi[b]))))
        dv = (dv + 180) % 360 - 180
        yj = r_ + np.random.default_rng(r_).uniform(-0.18, 0.18, len(dv))
        axP.plot(dv, yj, "o", ms=1.6, color="0.45", alpha=0.6, mew=0, zorder=2)
        axP.plot([fit, fit], [r_ - 0.32, r_ + 0.32], color="#2b5c8a", lw=1.6, zorder=3)
    axP.set_yticks(range(len(top3))); axP.set_yticklabels([f"{a}–{b}" for a, b in top3], fontsize=7); axP.set_ylim(-0.6, len(top3) - 0.4)
    axP.set_xlabel("shared-vessel $\\Delta\\phi_e$ (deg)"); axP.set_ylabel("tile pair", labelpad=1); axP.spines[["top", "right"]].set_visible(False)
    axP.tick_params(axis="y", length=0)
    axP.set_xlim(-180, 180); axP.set_xticks([-180, 0, 180])
    axP.plot([], [], "-", color="#2b5c8a", lw=1.6, label="fitted $\\psi_a - \\psi_b$"); axP.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 0.98), handlelength=1.0, borderaxespad=0)
    axG = fig.add_subplot(gB[1]); tb_ = sorted(band); pg = np.degrees([np.angle(np.exp(1j * psi[t])) for t in tb_]); sg_ = np.array([sd.get(t, np.nan) for t in tb_])
    okg = sg_ <= SD_MAX
    axG.errorbar(np.array(tb_)[okg], pg[okg], yerr=sg_[okg], fmt="o", ms=2.6, color="0.15", elinewidth=0.6, capsize=0, zorder=3)
    axG.errorbar(np.array(tb_)[~okg], pg[~okg], yerr=np.minimum(sg_[~okg], 180), fmt="o", ms=2.6, mfc="white", mec="0.45", ecolor="0.6", elinewidth=0.6, capsize=0, zorder=3)
    axG.plot(ref, 0, "s", ms=5, mfc="white", mec="black", mew=0.9, zorder=4)
    axG.set_ylim(-190, 190); axG.set_yticks([-180, -90, 0, 90, 180]); axG.set_xlabel("tile number"); axG.set_ylabel("tile gauge $\\psi_t$ (deg)", labelpad=1)
    axG.spines[["top", "right"]].set_visible(False)
    label(gs[1], "b")

    # ---- c: repeated-vessel agreement after alignment
    gD = gs[2].subgridspec(2, 1, hspace=0.3); bins = np.linspace(-90, 90, 37)
    for r_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gD[r_]); v = d[k]["aligned"]
        ax.hist(v, bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.3)
        ax.set_xlim(-90, 90); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-90, 0, 90])
        ax.text(0.0, 0.97, "H1" if k == 1 else "H2 validation", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(1.0, 0.97, f"median\n|Δφ| {np.median(np.abs(d[k]['aligned'])):.0f}°", transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color="0.25")
        if r_ == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("same-vessel phase difference\nbetween tiles, aligned (deg)")
    label(gs[2], "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig7.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / "fig7.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig7.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    log(f"wrote {OUT_DIR / 'fig7.png'}")


if __name__ == "__main__":
    main()
