"""Figure 7 — cross-tile phase synchronisation: putting asynchronous pulsatile measurements on one time reference.

Every tile is a separate recording with its own (arbitrary) time origin, so a vessel's harmonic phases are only
comparable within a tile.  The AC field (plexusflow.ac.field) fixes one time gauge ψ_t per tile from the vessels imaged
in two or more tiles and applies it coherently, Ĥ_n → e^{−inψ_t} Ĥ_n.

  a  heart rate f0 of every tile vs tile number; tiles within ±15 % of the reference tile are retained (shaded band)
  b  one vessel's assembled reference waveform: measured in its AC winner tile t (cardiac phase τ = f0·t mod 1) →
     shifted by the tile gauge (Δτ = ψ_t/2π) → scaled to the reconciled mean, Q̃_e(τ) = (Q′_e / Q̄_{e,t}) Q_{e,t}(τ − ψ_t/2π)
  c  validation on vessels imaged in two tiles (both ρ ≥ 0.1): same-vessel phase difference between tiles for H1 (fitted)
     and H2 (not fitted; aligned with 2ψ), as recorded vs aligned, with the median |Δφ|
  (supplement: one vessel's waveform in two tiles before / after the gauges)
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

    fig = plt.figure(figsize=(7.5, 3.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1], wspace=0.3, left=0.07, right=0.985, top=0.9, bottom=0.17)

    def label(cell, letter):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.045, b_.y1 + 0.03, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ---- a: heart rate per tile and the accepted band
    axA = fig.add_subplot(gs[0]); ts = sorted(tf0); f = np.array([tf0[t] for t in ts]); inb = np.array([t in band for t in ts])
    axA.axhspan(f_ref * (1 - F0_BAND), f_ref * (1 + F0_BAND), color="#dfe8f2", zorder=0, lw=0)
    axA.axhline(f_ref, color="0.45", lw=0.6, ls="--", zorder=1)
    axA.plot(np.array(ts)[inb], f[inb], "o", ms=3.2, color="#2b5c8a", zorder=3, label="retained")
    axA.plot(np.array(ts)[~inb], f[~inb], "o", ms=3.2, mfc="white", mec="0.45", mew=0.7, zorder=3, label="excluded")
    axA.plot(ref, f_ref, "s", ms=5, mfc="none", mec="black", mew=1.0, zorder=4, label=f"reference (tile {ref})")
    axA.set_xlabel("tile number"); axA.set_ylabel("heart rate $f_0$ (Hz)"); axA.spines[["top", "right"]].set_visible(False)
    axA.text(0.98, f_ref * (1 + F0_BAND), f"±{100 * F0_BAND:.0f}%", transform=axA.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="0.35")
    axA.legend(fontsize=7, frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    label(gs[0], "a")

    # ---- b: assembling one vessel's reference waveform: measured → phase-shifted → scaled to Q′
    el, wt, rho_w, a0_w, Qp_w = F["elig"], F["tile"], F["rho"], F["a0"], F["Qp"]; best = None
    for i in np.flatnonzero(el & np.isfinite(rho_w)):
        t = int(wt[i]); ratio = abs(Qp_w[i] / a0_w[i]); dtau = abs(np.angle(np.exp(1j * psi[t]))) / (2 * np.pi)
        if rho_w[i] < 0.8 or not (0.2 <= dtau <= 0.45) or not (1.3 <= ratio <= 2.2) or sd.get(t, np.inf) > SD_MAX or abs(Qp_w[i]) < 0.05:
            continue
        if best is None or rho_w[i] > best[0]:
            best = (rho_w[i], i, t)
    _, ib, tb = best; eb = F["edges"][ib]; x = meas[eb][tb]; sg = np.sign(x["a0"]); tau = np.linspace(0, 1, 300)
    wave = lambda H, a0: sg * (a0 + sum((H[n - 1] * np.exp(2j * np.pi * n * tau)).real for n in (1, 2, 3)))
    q_raw = wave(x["H"], x["a0"]); Hs = x["H"] * np.exp(-1j * np.arange(1, 4) * psi[tb]); q_sh = wave(Hs, x["a0"])
    q_fin = abs(Qp_w[ib]) / abs(x["a0"]) * q_sh
    log(f"  (b) vessel {eb}, tile {tb}: ρ {rho_w[ib]:.2f}, Δτ = ψ/2π {psi[tb] / (2 * np.pi):+.2f}, Q′/Q̄ {abs(Qp_w[ib] / x['a0']):.2f}")
    gB = gs[1].subgridspec(2, 3, height_ratios=[1, 0.36], wspace=0.75, hspace=0.42)
    steps = [(q_raw, f"measured, tile {tb}", "0.45"), (q_sh, "phase-aligned", "0.45"), (q_fin, "assembled $\\tilde Q_e$", "#2b5c8a")]
    ylo, yhi = min(q.min() for q, *_ in steps), max(q.max() for q, *_ in steps); pad = 0.08 * (yhi - ylo); axs_ = []
    for j, (q, ttl, col) in enumerate(steps):
        ax = fig.add_subplot(gB[0, j]); axs_.append(ax)
        ax.plot(tau, q, color=col, lw=1.3); ax.axhline(q.mean(), color=col, lw=0.5, ls=":")
        ax.set_ylim(0, yhi + pad); ax.set_xlim(0, 1); ax.set_xticks([0, 1]); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(ttl, fontsize=7.5, pad=2); ax.set_xlabel("$\\tau$", labelpad=0)
        if j == 0:
            ax.set_ylabel("flow (nL/s)")
        else:
            ax.set_yticklabels([])
    for j, txt in enumerate(("shift\n$\\psi_t/2\\pi$", "scale\n$Q′_e/\\bar Q_{e,t}$")):
        a_, b_ = axs_[j].get_position(), axs_[j + 1].get_position(); xm = (a_.x1 + b_.x0) / 2; ym = (a_.y0 + a_.y1) / 2
        fig.patches.append(matplotlib.patches.FancyArrowPatch((a_.x1 + 0.004, ym), (b_.x0 - 0.004, ym), transform=fig.transFigure,
                                                              arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.3"))
        fig.text(xm, ym + 0.025, txt, ha="center", va="bottom", fontsize=6.5, color="0.25", linespacing=1.2)
    axe = fig.add_subplot(gB[1, :]); axe.axis("off")
    axe.text(0.5, 0.45, "$\\tilde Q_e(\\tau) = \\dfrac{Q′_e}{\\bar Q_{e,t}}\; Q_{e,t}\\!\\left(\\tau - \\dfrac{\\psi_t}{2\\pi}\\right)$", ha="center", va="center", fontsize=8.5)
    axe.text(0.5, -0.5, "$\\psi_t$ inferred from vessels shared between tiles", ha="center", va="center", fontsize=7, color="0.35")
    label(gs[1], "b")

    # ---- c: validation, cross-tile phase differences of the same vessel
    gD = gs[2].subgridspec(2, 1, hspace=0.25); bins = np.linspace(-180, 180, 37)
    for r_, k in enumerate((1, 2)):
        ax = fig.add_subplot(gD[r_])
        ax.hist(d[k]["raw"], bins=bins, density=True, histtype="step", color="0.5", lw=0.8, label="as recorded")
        ax.hist(d[k]["aligned"], bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.3, label="aligned")
        ax.set_xlim(-180, 180); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xticks([-180, -90, 0, 90, 180])
        ax.text(0.0, 0.97, f"H{k}" + ("" if k == 1 else "  (not fitted)"), transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(1.0, 0.97, f"{np.median(np.abs(d[k]['raw'])):.0f}° → {np.median(np.abs(d[k]['aligned'])):.0f}°", transform=ax.transAxes, fontsize=8, ha="right", va="top")
        if r_ == 0:
            ax.set_xticklabels([]); ax.legend(fontsize=7, frameon=False, loc="center right", bbox_to_anchor=(1.03, 0.5), handlelength=1.2)
        else:
            ax.set_xlabel("same-vessel phase difference\nbetween tiles (deg)")
    label(gs[2], "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig7.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / "fig7.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig7.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    log(f"wrote {OUT_DIR / 'fig7.png'}")


if __name__ == "__main__":
    main()
