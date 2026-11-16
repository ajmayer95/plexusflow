"""Figure 3 — measuring flow in one vessel of one tile (PLOS, 7.5 in wide).

The representative vessel (edge 75489–75638) in its own tile (36), the tile of Fig 2c:
  a  one representative raw frame (closest to the movie median) with the time-mean optical flow (arrows, centred on their
     sample points), the centreline from the mapped graph and every 5th cross-section
  b  axial-speed profiles across three cross-sections and their Poiseuille fits v(ρ) = v_CL (1 − ρ²/R_s²); the vessel's
     mean flux Q̄ (mean of the shape-valid sections' (π/2) R_s² v_CL,s) is stated in the panel
  c  pulsatile flux: Q(t) over the recording (250 fps) with its harmonic model Q̂(t) = Q̄ + Σ_k Re(H_k e^{2πik f0 t}),
     k = 1..3 (left); every beat folded on the cardiac phase against the model, and the model's harmonics (right)
The per-section flux along the vessel and the fixed-radius peak/trough profiles are in the SI figure
(make_figure_S_flux_estimator.py), which reuses load_vessel() from here.

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
from plexusflow.analyze.kloosterman import kloosterman_scalar, PRODUCTION_KW
from plexusflow.analyze.centerline import _centerline_geometry
from plexusflow.analyze.constants import PX_SIZE_UM, FRAME_DT_S
from plexusflow.reconcile.harmonics import phasors, FS
import plos_style

EDGE, TILE = (75489, 75638), 36
INK, MEAS, MODEL, VCOL = "#1a1a1a", "#1b9e3a", "#1f4e8c", "#d62728"
HCOL = ["#1f4e8c", "#d95f02", "#7570b3"]
KW = PRODUCTION_KW
UMS = PX_SIZE_UM / FRAME_DT_S
HALO = [pe.withStroke(linewidth=1.6, foreground="black")]


def load_vessel():
    """Everything both Fig 3 and its SI companion need for the representative vessel."""
    P = _paths.STAGE; G = _paths.load_graph(); tf0 = G.graph["tile_f0_piv"]
    ki = (pickle.load(gzip.open(P / "batch_kl_inputs" / f"tile_{TILE}.pkl.gz"))["kl_inputs"]).get(EDGE)
    vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R0 = float(ki["R_prior_px"]); off = np.asarray(ki["roi_offset"], float)
    res = kloosterman_scalar(vf, cl, R0, mask=ki.get("ridge_mask_roi"), **KW)
    valid = np.asarray(res["valid_mask"], bool); conv = float(ki["conv"])
    m = next(x for x in G.edges[EDGE]["measurements_piv"] if x["tile_id"] == TILE)
    q = np.asarray(m["Q_t"], float); f0 = float(tf0[TILE]); t = np.arange(len(q)) / FS
    a0, H = phasors(q, f0)
    model = lambda tt, ks=(0, 1, 2, 3): a0 * (0 in ks) + sum((H[k - 1] * np.exp(2j * np.pi * k * f0 * tt)).real for k in ks if k > 0)
    return dict(P=P, G=G, ki=ki, vf=vf, cl=cl, R0=R0, off=off, res=res, valid=valid, Qs=np.asarray(res["per_slice_Q"]) * conv,
                Rs=np.asarray(res["per_slice_R"]) * PX_SIZE_UM, arc=np.asarray(res["per_slice_arc"]) * PX_SIZE_UM,
                Qbar=float(res["Q_char"]) * conv, q=q, f0=f0, t=t, a0=a0, H=H, model=model)


def main():
    plos_style.apply()
    V = load_vessel(); P, G, vf, cl, R0, off, res, valid = (V[k] for k in ("P", "G", "vf", "cl", "R0", "off", "res", "valid"))
    Rs, Qbar, q, f0, t, a0, H, model = (V[k] for k in ("Rs", "Qbar", "q", "f0", "t", "a0", "H", "model"))
    print(f"[fig3] {EDGE} tile {TILE}: {valid.sum()}/{len(valid)} sections, Q̄ {Qbar:.3f} nL/s (trace mean {a0:.3f}), f0 {f0:.2f} Hz, "
          f"|H| {np.round(np.abs(H), 3)}")

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.40, left=0.07, right=0.985, top=0.93, bottom=0.09)
    gT = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.30)
    gB = gs[1].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.28)

    # a: vessel in its tile, time-mean OF, cross-sections
    from plexusflow.config import StageConfig
    from plexusflow.io.acquisition import load_representative_frame       # raw recording, or the copy in a data release
    img, k_frame, n_raw = load_representative_frame(P, StageConfig.load(P), TILE)
    print(f"[fig3] a: raw frame {k_frame} of {n_raw} (closest to the movie median)")
    _, tan, nrm = _centerline_geometry(cl); clt = cl + off
    lo, hi = clt.min(0) - 2.2 * R0, clt.max(0) + 2.2 * R0; ctr = (lo + hi) / 2; h_ = (hi - lo).max() / 2
    axA = fig.add_subplot(gT[0])
    axA.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7), interpolation="nearest")
    axA.set_xlim(ctr[1] - h_, ctr[1] + h_); axA.set_ylim(ctr[0] + h_, ctr[0] - h_); axA.set_xticks([]); axA.set_yticks([])
    for i in range(0, len(cl), 5):                            # every 5th cross-section drawn (all are fitted)
        a_, b_ = clt[i] - R0 * 1.3 * nrm[i], clt[i] + R0 * 1.3 * nrm[i]
        axA.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white" if valid[i] else "0.6", lw=0.5, alpha=0.75, zorder=2)
    axA.plot(clt[:, 1], clt[:, 0], color=VCOL, lw=2.6, alpha=0.6, solid_capstyle="round", zorder=5)   # wide translucent centreline under the arrows
    Pq, Uq = [], []
    for i in range(5, len(cl) - 4, 12):                       # sparse along the vessel, dense across it (shows the profile)
        for s_ in np.linspace(-0.75, 0.75, 5):
            p = cl[i] + s_ * R0 * nrm[i]; r_, c_ = int(round(p[0])), int(round(p[1]))
            if 0 <= r_ < vf.shape[0] and 0 <= c_ < vf.shape[1]:
                Pq.append(p + off); Uq.append(vf[r_, c_])
    Pq, Uq = np.array(Pq), np.array(Uq)
    gap = np.median(np.hypot(*np.diff(cl, axis=0).T)) * 12      # px between arrow rows along the vessel
    k_ = 0.85 * gap / np.percentile(np.linalg.norm(Uq, axis=1), 98)   # longest arrow fits between rows: lengths show the profile
    axA.quiver(Pq[:, 1], Pq[:, 0], Uq[:, 0] * k_, Uq[:, 1] * k_, angles="xy", scale_units="xy", scale=1, color=(1, 1, 1, 0.75), edgecolor=(0.1, 0.1, 0.1, 0.6),
               linewidth=0.25, width=0.008, headwidth=3.5, headlength=3.5, headaxislength=3, pivot="middle", zorder=6)
    sel = np.flatnonzero(valid); picks = sel[np.linspace(2, len(sel) - 3, 3).astype(int)]
    for j, i in enumerate(picks):
        p = clt[i] + R0 * 1.9 * nrm[i]
        axA.text(p[1], p[0], str(j + 1), color="0.92", fontsize=8, ha="center", va="center", zorder=6, path_effects=[pe.withStroke(linewidth=1.2, foreground="0.15")])
    x0, x1 = axA.get_xlim(); y0, y1 = axA.get_ylim(); L = 20 / PX_SIZE_UM; xe = x0 + 0.95 * (x1 - x0); ya = y0 + 0.07 * (y1 - y0)
    axA.plot([xe - L, xe], [ya, ya], color="white", lw=2, solid_capstyle="butt", path_effects=HALO)
    axA.annotate("20 µm", (xe, ya), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom", color="white", fontsize=8, path_effects=HALO)

    # b: three profiles + Poiseuille fits; the vessel's mean flux
    axB = fig.add_subplot(gT[1]); rho = np.asarray(res["rho"], float) * PX_SIZE_UM; prof = np.asarray(res["per_slice_profiles"], float) * UMS
    vcl = np.asarray(res["per_slice_v_CL"], float) * UMS; offs = np.asarray(res.get("per_slice_offset", np.zeros(len(cl))), float) * PX_SIZE_UM
    step = float(np.round(np.nanmax(vcl[picks]) * 0.85, -2))
    for j, i in enumerate(picks):
        y0_ = j * step; ok = np.isfinite(prof[i])
        axB.plot(rho[ok], prof[i][ok] + y0_, "o", ms=1.8, color=INK, mew=0, zorder=3)
        rr = np.linspace(-Rs[i], Rs[i], 100)
        axB.plot(rr + offs[i], vcl[i] * (1 - (rr / Rs[i]) ** 2) + y0_, color="#2b70b4", lw=1.1, zorder=2)
        axB.axhline(y0_, color="0.88", lw=0.4, zorder=1)
        axB.text(rho.max() * 1.02, y0_ + 0.08 * step, str(j + 1), fontsize=8, color="0.35", va="bottom")
    axB.set_xlabel("Radial position (µm)", labelpad=1); axB.set_ylabel("Axial speed (µm/s, offset)", labelpad=1)
    axB.set_yticks([0, step]); axB.set_yticklabels(["0", f"{step:.0f}"]); axB.spines[["top", "right"]].set_visible(False)
    axB.plot([], [], "o", ms=2.5, color=INK, label="Optical flow"); axB.plot([], [], color="#2b70b4", lw=1.1, label="Parabolic fit")
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    axB.set_ylim(-0.12 * step, (len(picks) - 1) * step + np.nanmax(vcl[picks]) + 0.75 * step)   # headroom for the flux note
    axB.text(0.02, 0.98, r"$\bar Q_s = \frac{\pi}{2}R_s^2\,\bar v_{\rm CL,s}$" + f"\n$\\bar Q$ = {Qbar:.2f} nL/s ({valid.sum()} sections)",
             transform=axB.transAxes, ha="left", va="top", fontsize=9, linespacing=1.5)

    # c (left): Q(t) + harmonic model
    axD = fig.add_subplot(gB[0])
    axD.plot(t, q, "o", ms=1.6, mew=0, color=MEAS, alpha=0.85, zorder=2, label="Estimated flux")
    tt = np.linspace(0, t[-1], 2000); axD.plot(tt, model(tt), color=MODEL, lw=1.1, zorder=3, label="Harmonic fit")
    axD.axhline(a0, color="0.7", lw=0.5, ls="--")
    axD.set_xlim(0, t[-1]); axD.set_xlabel("Time (s)", labelpad=1); axD.set_ylabel("Flux (nL/s)", labelpad=1)
    axD.spines[["top", "right"]].set_visible(False)
    axD.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2, handlelength=1.5)
    yl = axD.get_ylim(); axD.set_ylim(yl[0], yl[0] + 1.22 * (yl[1] - yl[0]))           # headroom for the waveform equation
    axD.text(0.015, 0.985, r"$Q(t) = \bar Q\,v_0(t)\,/\,\langle v_0\rangle_t$",
             transform=axD.transAxes, ha="left", va="top", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

    # c (right): folded beats + harmonic components
    gE = gB[1].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.32)
    axE = fig.add_subplot(gE[0]); ph = (t * f0) % 1.0; cyc = np.floor(t * f0).astype(int)
    full = [c for c in np.unique(cyc) if (cyc == c).sum() > 0.9 * FS / f0]; nb = len(full)
    bins_ = np.linspace(0, 1, 41); sel_ = np.isin(cyc, full); idx_ = np.clip(np.digitize(ph[sel_], bins_) - 1, 0, 39)
    mu = np.array([q[sel_][idx_ == b].mean() for b in range(40)]); sd = np.array([q[sel_][idx_ == b].std() for b in range(40)])
    bc_ = 0.5 * (bins_[:-1] + bins_[1:])
    axE.fill_between(bc_, mu - sd, mu + sd, color="0.72", alpha=0.6, lw=0); axE.plot(bc_, mu, "o", ms=2.8, color=MEAS, mew=0, zorder=3)
    pp = np.linspace(0, 1, 300); axE.plot(pp, model(pp / f0), color=MODEL, lw=1.3, zorder=2)
    axE.set_xlim(0, 1); axE.tick_params(labelbottom=False); axE.set_ylabel("Flux (nL/s)", labelpad=1); axE.spines[["top", "right"]].set_visible(False)
    axK = fig.add_subplot(gE[1], sharex=axE)
    for k, nm in zip((1, 2, 3), ("Fundamental", "2nd", "3rd")):
        axK.plot(pp, model(pp / f0, (k,)), color=HCOL[k - 1], lw=1.0, label=nm)
    axK.axhline(0, color="0.85", lw=0.5); axK.set_xlabel("Cardiac phase", labelpad=1); axK.set_ylabel("nL/s", labelpad=1)
    axK.spines[["top", "right"]].set_visible(False)
    axK.legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, borderaxespad=0.1, handlelength=1.0, columnspacing=0.6)
    axK.set_xticks([0, 0.5, 1])
    # panel letters: a/b share the top row's y, c sits on the bottom row; a and c share x
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    edge = lambda ax_: inv.transform(ax_.get_tightbbox(rr))           # [[x0, y0], [x1, y1]] in figure fraction
    tops = {k: max(edge(ax_)[1, 1] for ax_ in row) for k, row in (("top", (axA, axB)), ("bot", (axD, axE)))}
    x_left = min(edge(axA)[0, 0], edge(axD)[0, 0]) - 0.005
    for ax_, l_, row in ((axA, "a", "top"), (axB, "b", "top"), (axD, "c", "bot")):
        x_ = x_left if l_ in "ac" else edge(ax_)[0, 0] - 0.005
        fig.text(x_, min(tops[row] + 0.004, 0.993), l_, fontsize=12, fontweight="bold", va="top" if tops[row] + 0.004 > 0.95 else "bottom", ha="left")
    plos_style.save(fig, _paths.FIGURES / "fig3" / "fig3")


if __name__ == "__main__":
    main()
