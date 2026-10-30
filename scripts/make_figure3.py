"""Figure 3 — measuring flow in one vessel of one tile (PLOS, 7.5 in wide).

The representative vessel (edge 75489–75638) in its own tile (36), the tile of Fig 2c:
  a  time-mean optical flow (arrows) around the vessel; centreline from the mapped graph; cross-sections (kept: white)
  b  axial-speed profiles across five cross-sections and their Poiseuille fits v(ρ) = v_CL (1 − ρ²/R_s²)
  c  per-cross-section mean flux Q̄_s = (π/2) R_s² v_CL,s along the vessel (radius R_s on the right axis); the vessel's mean
     flow is their median (Kloosterman et al. 2014)
  d  Q(t) over the recording (250 fps) and its harmonic model Q̂(t) = Q̄ + Σ_k Re(H_k e^{2πik f0 t}), k = 1..3
  e  every beat folded on the cardiac phase (grey) against the model, and the model's fundamental / 2nd / 3rd harmonic

Inputs: batch_kl_inputs/tile_36.pkl.gz, tiles/maxmed/vid36_maxmed.tif, the canonical graph (measurements_piv Q_t,
tile_f0_piv).  Writes figures/fig3/fig3.{png,pdf,tif}.
"""
from __future__ import annotations
import sys, gzip, pickle
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.analyze.kloosterman import kloosterman_scalar
from plexusflow.analyze.centerline import _centerline_geometry
from plexusflow.analyze.constants import PX_SIZE_UM, FRAME_DT_S
from plexusflow.reconcile.harmonics import phasors, FS
import plos_style

EDGE, TILE = (75489, 75638), 36
INK, MEAS, MODEL, VCOL = "#1a1a1a", "#2ca02c", "#1f4e8c", "#d62728"
HCOL = ["#1f4e8c", "#d95f02", "#7570b3"]
KW = dict(dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5, R_slice_lo=0.5,
          R_slice_hi=2.0, min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)
UMS = PX_SIZE_UM / FRAME_DT_S
HALO = [pe.withStroke(linewidth=1.6, foreground="black")]


def main():
    plos_style.apply()
    P = _paths.STAGE; G = _paths.load_graph(); tf0 = G.graph["tile_f0_piv"]
    ki = (pickle.load(gzip.open(P / "batch_kl_inputs" / f"tile_{TILE}.pkl.gz"))["kl_inputs"]).get(EDGE)
    vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R0 = float(ki["R_prior_px"]); off = np.asarray(ki["roi_offset"], float)
    res = kloosterman_scalar(vf, cl, R0, mask=ki.get("ridge_mask_roi"), **KW)
    valid = np.asarray(res["valid_mask"], bool); conv = float(ki["conv"])
    Qs = np.asarray(res["per_slice_Q"]) * conv; Rs = np.asarray(res["per_slice_R"]) * PX_SIZE_UM; arc = np.asarray(res["per_slice_arc"]) * PX_SIZE_UM
    Qbar = float(res["Q_char"]) * conv
    m = next(x for x in G.edges[EDGE]["measurements_piv"] if x["tile_id"] == TILE)
    q = np.asarray(m["Q_t"], float); f0 = float(tf0[TILE]); t = np.arange(len(q)) / FS
    a0, H = phasors(q, f0)
    model = lambda tt, ks=(0, 1, 2, 3): a0 * (0 in ks) + sum((H[k - 1] * np.exp(2j * np.pi * k * f0 * tt)).real for k in ks if k > 0)
    print(f"[fig3] {EDGE} tile {TILE}: {valid.sum()}/{len(valid)} sections, Q̄ {Qbar:.3f} nL/s (trace mean {a0:.3f}), f0 {f0:.2f} Hz, "
          f"|H| {np.round(np.abs(H), 3)}")

    fig = plt.figure(figsize=(7.5, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.42, left=0.07, right=0.94, top=0.95, bottom=0.08)
    gT = gs[0].subgridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.42)
    gB = gs[1].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.28)
    letter = lambda ax, l, dx=-0.18, dy=1.03: ax.text(dx, dy, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: vessel in its tile, time-mean OF, cross-sections
    img = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{TILE}_maxmed.tif")).astype(float); img = img.mean(-1) if img.ndim == 3 else img
    _, tan, nrm = _centerline_geometry(cl); clt = cl + off
    lo, hi = clt.min(0) - 2.2 * R0, clt.max(0) + 2.2 * R0; ctr = (lo + hi) / 2; h_ = (hi - lo).max() / 2
    axA = fig.add_subplot(gT[0])
    axA.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7), interpolation="nearest")
    axA.set_xlim(ctr[1] - h_, ctr[1] + h_); axA.set_ylim(ctr[0] + h_, ctr[0] - h_); axA.set_xticks([]); axA.set_yticks([])
    for i in range(len(cl)):
        a_, b_ = clt[i] - R0 * 1.3 * nrm[i], clt[i] + R0 * 1.3 * nrm[i]
        axA.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white" if valid[i] else "0.6", lw=0.4, alpha=0.8, zorder=2)
    axA.plot(clt[:, 1], clt[:, 0], color=VCOL, lw=1.0, zorder=5)
    Pq, Uq = [], []
    for i in range(3, len(cl) - 2, 10):                       # sparse along the vessel, dense across it (shows the profile)
        for s_ in np.linspace(-0.8, 0.8, 7):
            p = cl[i] + s_ * R0 * nrm[i]; r_, c_ = int(round(p[0])), int(round(p[1]))
            if 0 <= r_ < vf.shape[0] and 0 <= c_ < vf.shape[1]:
                Pq.append(p + off); Uq.append(vf[r_, c_])
    Pq, Uq = np.array(Pq), np.array(Uq)
    gap = np.median(np.hypot(*np.diff(cl, axis=0).T)) * 10      # px between arrow rows along the vessel
    k_ = 0.85 * gap / np.percentile(np.linalg.norm(Uq, axis=1), 98)   # longest arrow fits between rows: lengths show the profile
    axA.quiver(Pq[:, 1], Pq[:, 0], Uq[:, 0] * k_, Uq[:, 1] * k_, angles="xy", scale_units="xy", scale=1, color="white", edgecolor=INK,
               linewidth=0.3, width=0.009, headwidth=3.5, headlength=3.5, headaxislength=3, zorder=4)
    sel = np.flatnonzero(valid); picks = sel[np.linspace(2, len(sel) - 3, 5).astype(int)]
    for j, i in enumerate(picks):
        p = clt[i] + R0 * 1.9 * nrm[i]
        axA.text(p[1], p[0], str(j + 1), color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6, path_effects=HALO)
    x0, x1 = axA.get_xlim(); y0, y1 = axA.get_ylim(); L = 20 / PX_SIZE_UM; xe = x0 + 0.95 * (x1 - x0); ya = y0 + 0.07 * (y1 - y0)
    axA.plot([xe - L, xe], [ya, ya], color="white", lw=2, solid_capstyle="butt", path_effects=HALO)
    axA.annotate("20 µm", (xe, ya), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom", color="white", fontsize=8, path_effects=HALO)
    axA.set_title("Time-mean optical flow", fontsize=8, pad=2); letter(axA, "a", -0.04)

    # b: profiles + Poiseuille fits
    axB = fig.add_subplot(gT[1]); rho = np.asarray(res["rho"], float) * PX_SIZE_UM; prof = np.asarray(res["per_slice_profiles"], float) * UMS
    vcl = np.asarray(res["per_slice_v_CL"], float) * UMS; offs = np.asarray(res.get("per_slice_offset", np.zeros(len(cl))), float) * PX_SIZE_UM
    step = float(np.round(np.nanmax(vcl[picks]) * 0.75, -2))
    for j, i in enumerate(picks):
        y0_ = j * step; ok = np.isfinite(prof[i])
        axB.plot(rho[ok], prof[i][ok] + y0_, "o", ms=1.6, color=INK, mew=0, zorder=3)
        rr = np.linspace(-Rs[i], Rs[i], 100)
        axB.plot(rr + offs[i], vcl[i] * (1 - (rr / Rs[i]) ** 2) + y0_, color="#2b70b4", lw=1.0, zorder=2)
        axB.axhline(y0_, color="0.88", lw=0.4, zorder=1)
        axB.text(rho.max() * 1.02, y0_ + 0.12 * step, str(j + 1), fontsize=8, fontweight="bold", va="bottom")
    axB.set_xlabel("Radial position (µm)", labelpad=1); axB.set_ylabel("Axial speed (µm/s)", labelpad=1)
    axB.set_yticks([0, step]); axB.set_yticklabels(["0", f"{step:.0f}"]); axB.spines[["top", "right"]].set_visible(False)
    axB.plot([], [], "o", ms=2.5, color=INK, label="Optical flow"); axB.plot([], [], color="#2b70b4", lw=1.0, label="Poiseuille fit")
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    letter(axB, "b", -0.3)

    # c: per-section mean flux + radius
    axC = fig.add_subplot(gT[2])
    axC.plot(arc[valid], Qs[valid], "o-", ms=2.0, lw=0.6, color=INK, zorder=3, label="Cross-sections")
    axC.axhline(Qbar, color=VCOL, lw=1.0, ls="--", zorder=2, label=f"Median {Qbar:.2f} nL/s")
    axC.set_ylim(0, np.nanmax(Qs[valid]) * 1.35); axC.set_xlabel("Distance along vessel (µm)", labelpad=1)
    axC.set_ylabel("Mean flux (nL/s)", labelpad=1); axC.spines[["top"]].set_visible(False)
    axC2 = axC.twinx(); axC2.plot(arc[valid], Rs[valid], color="0.55", lw=0.8, ls=":")
    axC2.set_ylabel("Radius (µm)", color="0.45", labelpad=2); axC2.tick_params(colors="0.45"); axC2.set_ylim(0, np.nanmax(Rs[valid]) * 1.6)
    axC2.spines[["top"]].set_visible(False)
    axC.plot([], [], color="0.55", lw=0.8, ls=":", label="Radius")
    axC.legend(loc="lower center", frameon=False, borderaxespad=0.2, handlelength=1.5, labelspacing=0.2)
    letter(axC, "c", -0.3)

    # d: Q(t) + harmonic model
    axD = fig.add_subplot(gB[0])
    axD.plot(t, q, color=MEAS, lw=0.7, label="Measured")
    tt = np.linspace(0, t[-1], 2000); axD.plot(tt, model(tt), color=MODEL, lw=1.1, label="Harmonic model")
    axD.axhline(a0, color="0.45", lw=0.6, ls="--")
    axD.set_xlim(0, t[-1]); axD.set_xlabel("Time (s)", labelpad=1); axD.set_ylabel("Flux (nL/s)", labelpad=1)
    axD.spines[["top", "right"]].set_visible(False)
    axD.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2, handlelength=1.5)
    letter(axD, "d", -0.1)

    # e: folded beats + harmonic components
    gE = gB[1].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.32)
    axE = fig.add_subplot(gE[0]); ph = (t * f0) % 1.0; cyc = np.floor(t * f0).astype(int); nb = 0
    for c in np.unique(cyc):
        s_ = cyc == c
        if s_.sum() > 0.9 * FS / f0:
            axE.plot(ph[s_], q[s_], color="0.65", lw=0.5); nb += 1
    pp = np.linspace(0, 1, 300); axE.plot(pp, model(pp / f0), color=MODEL, lw=1.3)
    axE.set_xlim(0, 1); axE.tick_params(labelbottom=False); axE.set_ylabel("Flux (nL/s)", labelpad=1); axE.spines[["top", "right"]].set_visible(False)
    axE.text(0.03, 0.04, f"{nb} beats", transform=axE.transAxes, ha="left", va="bottom", fontsize=8, color="0.35")
    letter(axE, "e", -0.3, 1.05)
    axK = fig.add_subplot(gE[1], sharex=axE)
    for k, nm in zip((1, 2, 3), ("Fundamental", "2nd", "3rd")):
        axK.plot(pp, model(pp / f0, (k,)), color=HCOL[k - 1], lw=1.0, label=nm)
    axK.axhline(0, color="0.85", lw=0.5); axK.set_xlabel("Cardiac phase", labelpad=1); axK.set_ylabel("nL/s", labelpad=1)
    axK.spines[["top", "right"]].set_visible(False)
    axK.legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, borderaxespad=0.1, handlelength=1.0, columnspacing=0.6)
    axK.set_xticks([0, 0.5, 1])
    plos_style.save(fig, _paths.FIGURES / "fig3" / "fig3")


if __name__ == "__main__":
    main()
