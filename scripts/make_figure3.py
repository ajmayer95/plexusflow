"""Figure 3 — measuring flow in one vessel of one tile (PLOS, 7.5 in wide).

The representative vessel (edge 75489–75638) in its own tile (36), the tile of Fig 2c:
  a  one representative raw frame (closest to the movie median) with the time-mean optical flow (arrows, centred on their
     sample points), the centreline from the mapped graph and every 5th cross-section
  b  axial-speed profiles across three cross-sections and their Poiseuille fits v(ρ) = v_CL (1 − ρ²/R_s²); the vessel's
     mean flux Q̄ (mean of the shape-valid sections' (π/2) R_s² v_CL,s) is stated in the panel.  Only the samples each
     section's fit used are drawn (per_slice_fit_bins): the cross-section sampling reaches past the vessel edge, and at
     section 1 the mask also covers a neighbouring vessel's lumen — those samples never enter the flux (2026-11-19).
  c  pulsatile flux: per-frame Q(t) = (π/2)⟨R_s² U_s(t)⟩_s (geometry frozen from the time-mean fit, centreline speed U_s
     per frame; kloosterman.per_frame operator) with its truncated Fourier series Q_N(t) = Q̂_0 + Σ_{n=1..N} Q̂_n cos(2πn f0 t + φ_n), N = 3 (the fit),
     A_k = |H_k|, φ_k = arg H_k (Q̂_n = |H_n|, φ_n = arg H_n, H_n the complex amplitude); in the background the modes: Q̂_0 and each
     harmonic drawn about it, Q̂_0 + Q̂_n cos(2πn f0 t + φ_n), at half the fit's line width. right: every beat folded on the
     cardiac phase (mean ± SD) against the model.
The per-section flux along the vessel and the fixed-radius peak/trough profiles are in the SI figure
(script in the git history), which reused load_vessel() from here.

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
N_FIT = 2                       # harmonics kept in the pulsatile fit (2026-11-20: the third adds ~0.1 % of the power, panel d)
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
    a0, H = phasors(q, f0, K=N_FIT)
    model = lambda tt, ks=tuple(range(N_FIT + 1)): a0 * (0 in ks) + sum((H[k - 1] * np.exp(2j * np.pi * k * f0 * tt)).real for k in ks if k > 0)
    return dict(P=P, G=G, ki=ki, vf=vf, cl=cl, R0=R0, off=off, res=res, valid=valid, Qs=np.asarray(res["per_slice_Q"]) * conv,
                Rs=np.asarray(res["per_slice_R"]) * PX_SIZE_UM, arc=np.asarray(res["per_slice_arc"]) * PX_SIZE_UM,
                Qbar=float(res["Q_char"]) * conv, q=q, f0=f0, t=t, a0=a0, H=H, model=model)


def truncated_power_fractions(G, K=3):
    """Fraction of the measured power captured by the truncated series Q_k(t) = Q̂_0 + Σ_{n=1..k} q_n(t), k = 0..K, pooled
    over every measured vessel (recordings with a valid profile fit in a tile inside the heart-rate band; a property of
    the measurement only).  Per recording: nested LSQ fits of mean + k harmonics at the tile heart rate; power of the fit
    = ⟨Q_obs²⟩ − ⟨residual²⟩ (exact for a least-squares fit).  Per vessel: mean over its recordings (repeatedly imaged
    vessels not over-represented).  Pooled: Σ_vessels power(Q_k) / Σ_vessels ⟨Q_obs²⟩.
    Returns (fractions [K + 1], number of vessels)."""
    from plexusflow.ac.field import CANONICAL_F0_BAND, REF_TILE
    tf0 = G.graph["tile_f0_piv"]; st = G.graph.get("tile_f0_status") or {}
    band = {t for t, f in tf0.items() if f and abs(f / tf0[REF_TILE] - 1) <= CANONICAL_F0_BAND
            and (st.get(t) or {}).get("status") not in ("no_signal", "excluded", "no measurements")}
    num, den, n_ves = np.zeros(K + 1), 0.0, 0
    for _, _, d in G.edges(data=True):
        Pv, Tv = [], []
        for m in d.get("measurements_piv") or []:
            if not ((m.get("quality") or {}).get("kl_ok") and m["tile_id"] in band and m.get("Q_t") is not None):
                continue
            q = np.asarray(m["Q_t"], float); t = np.arange(len(q)) / FS; ok = np.isfinite(q)
            if ok.sum() < 200:
                continue
            q, t = q[ok], t[ok]; f0 = tf0[m["tile_id"]]
            X = np.column_stack([np.ones(len(t))] + [fn(2 * np.pi * k * f0 * t) for k in range(1, K + 1) for fn in (np.cos, np.sin)])
            ms = np.array([np.mean((q - X[:, :2 * k + 1] @ np.linalg.lstsq(X[:, :2 * k + 1], q, rcond=None)[0]) ** 2) for k in range(K + 1)])
            Tv.append(np.mean(q ** 2)); Pv.append(Tv[-1] - ms)
        if Pv:
            num += np.mean(Pv, axis=0); den += np.mean(Tv); n_ves += 1
    return num / den, n_ves


def main():
    plos_style.apply()
    V = load_vessel(); P, G, vf, cl, R0, off, res, valid = (V[k] for k in ("P", "G", "vf", "cl", "R0", "off", "res", "valid"))
    Rs, Qbar, q, f0, t, a0, H, model = (V[k] for k in ("Rs", "Qbar", "q", "f0", "t", "a0", "H", "model"))
    print(f"[fig3] {EDGE} tile {TILE}: {valid.sum()}/{len(valid)} sections, Q̄ {Qbar:.3f} nL/s (trace mean {a0:.3f}), f0 {f0:.2f} Hz, "
          f"|H| {np.round(np.abs(H), 3)}")

    fig = plt.figure(figsize=(7.5, 5.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30, left=0.08, right=0.985, top=0.94, bottom=0.07)
    gT = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.30)

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
        y0_ = j * step; ok = np.isfinite(prof[i]) & np.asarray(res["per_slice_fit_bins"][i], bool)   # samples the fit used
        o_ = offs[i] if np.isfinite(offs[i]) else 0.0
        axB.plot((rho[ok] - o_) / Rs[i], prof[i][ok] + y0_, "o", ms=1.8, color=INK, mew=0, zorder=3)   # ρ / R_s (section's own radius)
        xx = np.linspace(-1, 1, 100)
        axB.plot(xx, vcl[i] * (1 - xx ** 2) + y0_, color="#2b70b4", lw=1.1, zorder=2)
        axB.axhline(y0_, color="0.88", lw=0.4, zorder=1)
        axB.text(1.08, y0_ + 0.08 * step, str(j + 1), fontsize=8, color="0.35", va="bottom")
    axB.set_xlim(-1.2, 1.2); axB.set_xticks([-1, -0.5, 0, 0.5, 1]); axB.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
    axB.set_xlabel(r"Radial position / fitted radius, $r/R_s$", labelpad=1); axB.set_ylabel("Axial speed (µm/s)", labelpad=1)
    axB.set_yticks([0, step]); axB.set_yticklabels(["0", f"{step:.0f}"]); axB.spines[["top", "right"]].set_visible(False)
    axB.plot([], [], "o", ms=2.5, color=INK, label=r"$v_{\rm obs}(r)$"); axB.plot([], [], color="#2b70b4", lw=1.1, label=r"$v(r) = v_{\rm CL}\,(1 - r^2/R_s^2)$")
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    axB.set_ylim(-0.12 * step, (len(picks) - 1) * step + 1.12 * np.nanmax(vcl[picks]))

    # c: per-frame Q(t) (foreground), the fit Q_3 (foreground) and its modes (background): the mean Q̂0 and each harmonic n
    #    drawn about the mean, Q̂0 + Q̂_n cos(2πn f0 t + φ_n), so the modes share the data's scale
    gB = gs[1].subgridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.27)
    axD = fig.add_subplot(gB[0])
    tt = np.linspace(0, t[-1], 2000); LW_FIT = 1.6; LW_MODE = LW_FIT / 2; A_MODE = 0.55
    MCOL = {0: "#009E73", 1: "#0072B2", 2: "#E69F00", 3: "#9467bd"}
    axD.plot(tt, np.full_like(tt, a0), color=MCOL[0], lw=LW_MODE, alpha=A_MODE, zorder=2, label=r"$\hat Q_0$")
    for n in range(1, N_FIT + 1):
        axD.plot(tt, a0 + model(tt, (n,)), color=MCOL[n], lw=LW_MODE, alpha=A_MODE, zorder=2, label=rf"$\hat Q_0 + q_{n}$")
    axD.plot(t, q, "o", ms=2.0, mew=0, color="0.25", zorder=4, label=r"$Q_{\rm obs}(t)$")
    axD.plot(tt, model(tt), color="#C2185B", lw=LW_FIT, zorder=5, label=rf"$Q_{N_FIT}(t)$")
    axD.set_xlim(0, t[-1]); axD.set_xlabel("Time (s)", labelpad=1); axD.set_ylabel("Flux (nL/s)", labelpad=1)
    axD.spines[["top", "right"]].set_visible(False)
    h_, l_ = axD.get_legend_handles_labels(); order = [N_FIT + 1, N_FIT + 2] + list(range(N_FIT + 1))
    LEG_C = axD.legend([h_[k] for k in order], [l_[k] for k in order], loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=6,
                       frameon=False, borderaxespad=0.0, borderpad=0.0, handlelength=1.2, columnspacing=0.9, handletextpad=0.4, fontsize=8)
    yl = axD.get_ylim(); axD.set_ylim(yl[0], yl[0] + 1.06 * (yl[1] - yl[0]))

    # d: fraction of the measured power in the truncated series Q_k = Q̂_0 + Σ_{n ≤ k} q_n, pooled over all measured vessels
    FR, n_ves = truncated_power_fractions(G, K=N_FIT); FR = 100 * FR; kk = np.arange(len(FR))
    print(f"[fig3] d: {n_ves} vessels; fraction of measured power in Q_0..Q_{len(FR) - 1} (%): {np.round(FR, 2)}")
    axE = fig.add_subplot(gB[1])
    axE.bar(kk, FR, width=0.66, color=[MCOL[0] if k == 0 else "#C2185B" if k == N_FIT else "0.7" if k < N_FIT else "0.88" for k in kk], edgecolor="0.15", lw=0.5, zorder=3)
    for k in kk:
        axE.annotate(f"{FR[k]:.1f}", (k, FR[k]), xytext=(0, 2), textcoords="offset points", fontsize=8, ha="center", va="bottom")
    axE.set_xlim(-0.6, len(FR) - 0.4); axE.set_xticks(kk); axE.set_xticklabels([rf"$Q_{k}$" for k in kk])
    axE.set_ylim(0, 100); axE.set_yticks([0, 25, 50, 75, 100]); axE.set_ylabel("Fraction of power (%)", labelpad=1)
    axE.spines[["top", "right"]].set_visible(False)

    # panel letters: a/b share the top row's y, c sits on the bottom row; a and c share x
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    edge = lambda ax_: inv.transform(ax_.get_tightbbox(rr))           # [[x0, y0], [x1, y1]] in figure fraction
    tops = {k: max(edge(ax_)[1, 1] for ax_ in row) for k, row in (("top", (axA, axB)), ("bot", (axD,)))}
    x_left = min(edge(axA)[0, 0], edge(axD)[0, 0]) - 0.005
    for ax_, l_, row in ((axA, "a", "top"), (axB, "b", "top"), (axD, "c", "bot")):
        x_ = x_left if l_ in "ac" else edge(ax_)[0, 0] - 0.005
        if l_ == "c":                                         # c: letter and the series definition on one baseline
            lb = LEG_C.get_window_extent(rr).transformed(fig.transFigure.inverted())
            y_c = lb.y0 + 0.3 * (lb.y1 - lb.y0)                   # baseline of the legend row
            fig.text(x_, y_c, l_, fontsize=12, fontweight="bold", va="baseline", ha="left")
            fig.text(axE.get_position().x0 - 0.075, y_c, "d", fontsize=12, fontweight="bold", va="baseline", ha="left")
            fig.text(axE.get_position().x0, y_c, rf"$N = {n_ves:,}$ vessels".replace(",", "{,}"), fontsize=9, va="baseline", ha="left")
            continue
        fig.text(x_, min(tops[row] + 0.004, 0.993), l_, fontsize=12, fontweight="bold", va="top" if tops[row] + 0.004 > 0.95 else "bottom", ha="left")
    plos_style.save(fig, _paths.FIGURES / "fig3" / "fig3")


if __name__ == "__main__":
    main()
