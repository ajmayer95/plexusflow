"""Figure 3 — from the mosaic to vessel-resolved mean flow (PLOS, 7.5 in wide; old Figs 3 + 4 merged).

  a  whole-organ vessel graph (net3 vectorization of the curated mask), edges coloured by radius; the crop of b boxed
  b  one crop through the pipeline: projection → ilastik vessel probability → curated mask → graph (radius-encoded);
     the representative vessel (edge 75489–75638, tile 36) highlighted
  c  that vessel in its tile: time-mean optical flow (arrows), refined centreline and the cross-sections (kept: white)
  d  axial-speed profiles across five of the cross-sections with their Poiseuille fits v(ρ) = v_CL (1 − ρ²/R_s²)
  e  per-cross-section mean flux Q̄_s = (π/2) R_s² v_CL,s along the vessel; the vessel's mean flow is their median
     (Kloosterman et al. 2014), the number every later figure uses

Inputs: fuse/maxmed.tif, fuse/probs_logit.tif, mask/mask.tif, the canonical graph, batch_kl_inputs/tile_36.pkl.gz,
tiles/maxmed/vid36_maxmed.tif.  Writes figures/fig3/fig3.{png,pdf,tif}.
"""
from __future__ import annotations
import sys, gzip, pickle, argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.analyze.kloosterman import kloosterman_scalar
from plexusflow.analyze.centerline import _centerline_geometry
from plexusflow.analyze.constants import PX_SIZE_UM, FRAME_DT_S
import plos_style

EDGE, TILE = (75489, 75638), 36
VCOL, INK = "#d62728", "#1a1a1a"
KW = dict(dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5, R_slice_lo=0.5,
          R_slice_hi=2.0, min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)
UMS = PX_SIZE_UM / FRAME_DT_S          # px/frame → µm/s


def noframe(ax, ec=None, lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(ec is not None)
        if ec is not None:
            s.set_color(ec); s.set_linewidth(lw)


def scalebar(ax, um, label, color="white", x=0.07, y=0.9, px_um=PX_SIZE_UM, lw=2.0):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); L = um / px_um
    xa, ya = x0 + x * (x1 - x0), y0 + y * (y1 - y0)
    ax.plot([xa, xa + L], [ya, ya], color=color, lw=lw, solid_capstyle="butt", zorder=9)
    ax.annotate(label, (xa + L / 2, ya), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color=color, fontsize=8, zorder=9)


def graph_segments(G, Hm, edges=None):
    segs, rad, keys = [], [], []
    for u, v, d in (G.edges(data=True) if edges is None else ((u, v, G.edges[u, v]) for u, v in edges)):
        p = np.asarray(d["path"], float) if d.get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in (u, v)])
        segs.append(np.column_stack([p[:, 0], Hm - p[:, 1]])); rad.append(float(d.get("radius") or np.nan)); keys.append((u, v))
    return segs, np.array(rad), keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crop-um", type=float, default=650.0, help="side of the b crop (µm), centred on the vessel")
    args = ap.parse_args()
    plos_style.apply()
    P = _paths.STAGE
    G = _paths.load_graph()
    maxmed = tifffile.imread(str(P / "fuse" / "maxmed.tif")).astype(float); maxmed = maxmed.mean(-1) if maxmed.ndim == 3 else maxmed
    prob = tifffile.imread(str(P / "fuse" / "probs_logit.tif")).astype(float)
    mask = tifffile.imread(str(P / "mask" / "mask.tif")) > 127
    Hm, Wm = maxmed.shape
    segs, rad, keys = graph_segments(G, Hm)
    rad_um = rad * PX_SIZE_UM; rnorm = Normalize(np.nanpercentile(rad_um, 2), np.nanpercentile(rad_um, 98)); rcmap = plt.cm.viridis
    lw_of = lambda r: 0.25 + 1.2 * np.clip(np.nan_to_num(r / np.nanpercentile(rad_um, 98), nan=0.3), 0, 1)
    ve = keys.index(EDGE) if EDGE in keys else keys.index(EDGE[::-1]); vmid = segs[ve][len(segs[ve]) // 2]
    hs = args.crop_um / PX_SIZE_UM / 2; cx0, cx1 = int(vmid[0] - hs), int(vmid[0] + hs); cy0, cy1 = int(vmid[1] - hs), int(vmid[1] + hs)
    print(f"[fig3] graph {G.number_of_nodes()} nodes / {G.number_of_edges()} edges; crop centred on {EDGE} at ({vmid[0]:.0f}, {vmid[1]:.0f})")

    fig = plt.figure(figsize=(7.5, 6.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.36, left=0.035, right=0.93, top=0.955, bottom=0.075)
    gT = gs[0].subgridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.08)
    gB = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.42)
    letter = lambda ax, l, dx=0.0, dy=1.02: ax.text(dx, dy, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: whole-organ graph
    axA = fig.add_subplot(gT[0])
    axA.add_collection(LineCollection(segs, colors=rcmap(rnorm(np.nan_to_num(rad_um, nan=rnorm.vmin))), linewidths=lw_of(rad_um) * 0.6))
    axA.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=INK, lw=1.0, zorder=5))
    axA.set_xlim(0, Wm); axA.set_ylim(Hm, 0); axA.set_aspect("equal"); noframe(axA)
    scalebar(axA, 1000, "1 mm", color=INK, x=0.02, y=0.04)
    axA.set_title(f"vessel graph ({G.number_of_edges():,} vessels)", fontsize=9, pad=2); letter(axA, "a")
    pA = axA.get_position()

    # b: crop pipeline (2 × 2)
    gb = gT[1].subgridspec(2, 2, wspace=0.05, hspace=0.12)
    sub = lambda im: im[cy0:cy1, cx0:cx1]
    panels = [("projection", sub(maxmed), "gray"), ("vessel probability", sub(prob), "magma"), ("curated mask", sub(mask), None), ("graph", None, None)]
    for k, (ttl, im, cmap) in enumerate(panels):
        ax = fig.add_subplot(gb[k // 2, k % 2])
        if k < 2:
            ax.imshow(im, cmap=cmap, vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="antialiased")
        elif k == 2:
            ax.imshow(np.where(im, 0.35, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(np.where(sub(mask), 0.88, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            inside = [i for i, s_ in enumerate(segs) if ((s_[:, 0] > cx0) & (s_[:, 0] < cx1) & (s_[:, 1] > cy0) & (s_[:, 1] < cy1)).any()]
            ax.add_collection(LineCollection([segs[i] - [cx0, cy0] for i in inside], colors=rcmap(rnorm(np.nan_to_num(rad_um[inside], nan=rnorm.vmin))),
                                             linewidths=lw_of(rad_um[inside]) * 1.4))
            ax.plot(*(segs[ve] - [cx0, cy0]).T, color=VCOL, lw=2.0, solid_capstyle="round", zorder=4)
        if k < 3:
            ax.plot(*(segs[ve] - [cx0, cy0]).T, color=VCOL, lw=0.8, ls=(0, (2, 1.5)), zorder=4)
        ax.set_xlim(0, cx1 - cx0); ax.set_ylim(cy1 - cy0, 0); ax.set_aspect("equal"); noframe(ax, "0.6")
        ax.set_title(ttl, fontsize=8, pad=2)
        if k == 0:
            letter(ax, "b", -0.02); scalebar(ax, 200, "200 µm", y=0.1)
    cax = fig.add_axes([pA.x0 + 0.1 * pA.width, pA.y0 - 0.012, 0.8 * pA.width, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap), cax=cax, orientation="horizontal")
    cb.set_label("vessel radius (µm)", labelpad=1); cb.outline.set_linewidth(0.5)

    # c–e: the representative vessel
    kl = pickle.load(gzip.open(P / "batch_kl_inputs" / f"tile_{TILE}.pkl.gz"))["kl_inputs"]; ki = kl.get(EDGE) or kl.get(EDGE[::-1])
    vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R0 = float(ki["R_prior_px"]); off = np.asarray(ki["roi_offset"], float)
    res = kloosterman_scalar(vf, cl, R0, mask=ki.get("ridge_mask_roi"), **KW)
    valid = np.asarray(res["valid_mask"], bool); conv = float(ki["conv"])
    Qs = np.asarray(res["per_slice_Q"]) * conv; Rs = np.asarray(res["per_slice_R"]) * PX_SIZE_UM; arc = np.asarray(res["per_slice_arc"]) * PX_SIZE_UM
    Qbar = float(res["Q_char"]) * conv
    print(f"[fig3] vessel {EDGE}: {valid.sum()}/{len(valid)} cross-sections kept, Q̄ = {Qbar:.3f} nL/s, R = {res['R_char'] * PX_SIZE_UM:.1f} µm")
    img = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{TILE}_maxmed.tif")).astype(float); img = img.mean(-1) if img.ndim == 3 else img
    _, tan, nrm = _centerline_geometry(cl); clt = cl + off
    lo, hi = clt.min(0) - 2.2 * R0, clt.max(0) + 2.2 * R0; ctr = (lo + hi) / 2; h_ = (hi - lo).max() / 2
    axC = fig.add_subplot(gB[0])
    axC.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7), interpolation="nearest")
    axC.set_xlim(ctr[1] - h_, ctr[1] + h_); axC.set_ylim(ctr[0] + h_, ctr[0] - h_); noframe(axC)
    for i in range(len(cl)):
        a_, b_ = clt[i] - R0 * 1.3 * nrm[i], clt[i] + R0 * 1.3 * nrm[i]
        axC.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white" if valid[i] else "0.6", lw=0.4, alpha=0.8, zorder=2)
    axC.plot(clt[:, 1], clt[:, 0], color=VCOL, lw=1.2, zorder=3)
    Pq, Uq = [], []
    for i in range(1, len(cl), 4):
        for s_ in (-0.45, 0.45):
            p = cl[i] + s_ * R0 * nrm[i]; r_, c_ = int(round(p[0])), int(round(p[1]))
            if 0 <= r_ < vf.shape[0] and 0 <= c_ < vf.shape[1]:
                Pq.append(p + off); Uq.append(vf[r_, c_])
    Pq, Uq = np.array(Pq), np.array(Uq); k_ = 0.8 * R0 / np.percentile(np.linalg.norm(Uq, axis=1), 90)
    axC.quiver(Pq[:, 1], Pq[:, 0], Uq[:, 0] * k_, Uq[:, 1] * k_, angles="xy", scale_units="xy", scale=1, color="white", edgecolor=INK,
               linewidth=0.3, width=0.012, headwidth=3.5, headlength=3.5, headaxislength=3, zorder=4)
    scalebar(axC, 20, "20 µm", x=0.62, y=0.92)
    axC.set_title("time-mean optical flow", fontsize=8, pad=2); letter(axC, "c", -0.02)
    # mark five cross-sections used in d
    sel = np.flatnonzero(valid); picks = sel[np.linspace(2, len(sel) - 3, 5).astype(int)]
    for j, i in enumerate(picks):
        p = clt[i] + R0 * 1.9 * nrm[i]
        axC.text(p[1], p[0], str(j + 1), color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)

    # d: stacked profiles + Poiseuille fits
    axD = fig.add_subplot(gB[1]); rho = np.asarray(res["rho"], float) * PX_SIZE_UM; prof = np.asarray(res["per_slice_profiles"], float) * UMS
    vcl = np.asarray(res["per_slice_v_CL"], float) * UMS; offs = np.asarray(res.get("per_slice_offset", np.zeros(len(cl))), float) * PX_SIZE_UM
    step = float(np.round(np.nanmax(vcl[picks]) * 0.75, -2))
    for j, i in enumerate(picks):
        y0 = j * step; ok = np.isfinite(prof[i])
        axD.plot(rho[ok], prof[i][ok] + y0, "o", ms=1.8, color=INK, mew=0, zorder=3)
        rr = np.linspace(-Rs[i], Rs[i], 100)
        axD.plot(rr + offs[i], vcl[i] * (1 - (rr / Rs[i]) ** 2) + y0, color="#2b70b4", lw=1.0, zorder=2)
        axD.axhline(y0, color="0.85", lw=0.4, zorder=1)
        axD.text(rho.max() * 1.02, y0 + 0.15 * step, str(j + 1), fontsize=8, fontweight="bold", color=INK, va="bottom")
    axD.set_xlabel("radial position ρ (µm)", labelpad=1); axD.set_ylabel("axial speed (µm/s, offset)", labelpad=1)
    axD.set_yticks([0, step]); axD.set_yticklabels(["0", f"{step:.0f}"]); axD.spines[["top", "right"]].set_visible(False)
    axD.plot([], [], "o", ms=2.5, color=INK, label="optical flow"); axD.plot([], [], color="#2b70b4", lw=1.0, label="Poiseuille fit")
    axD.legend(loc="upper left", frameon=False, borderaxespad=0, handlelength=1.2, bbox_to_anchor=(0.0, 1.13), ncol=2, columnspacing=0.8)
    letter(axD, "d", -0.3, 1.07)

    # e: per-slice mean flux along the vessel, and the median
    axE = fig.add_subplot(gB[2])
    axE.plot(arc[valid], Qs[valid], "o-", ms=2.2, lw=0.6, color=INK, zorder=3, label="cross-sections")
    axE.axhline(Qbar, color=VCOL, lw=1.0, ls="--", zorder=2, label=f"median {Qbar:.2f} nL/s")
    axE.plot([], [], color="0.55", lw=0.8, ls=":", label="$R_s$ (right axis)")
    axE.legend(loc="lower center", frameon=False, borderaxespad=0.2, handlelength=1.6, labelspacing=0.2)
    axE.set_ylim(0, np.nanmax(Qs[valid]) * 1.35); axE.set_xlabel("distance along vessel (µm)", labelpad=1); axE.set_ylabel("mean flux $\\bar Q_s$ (nL/s)", labelpad=1)
    axE.spines[["top"]].set_visible(False)
    axE2 = axE.twinx(); axE2.plot(arc[valid], Rs[valid], color="0.55", lw=0.8, ls=":"); axE2.set_ylabel("$R_s$ (µm)", color="0.45", labelpad=2)
    axE2.tick_params(colors="0.45"); axE2.set_ylim(0, np.nanmax(Rs[valid]) * 1.6); axE2.spines[["top"]].set_visible(False)
    letter(axE, "e", -0.3, 1.07)
    plos_style.save(fig, _paths.FIGURES / "fig3" / "fig3")


if __name__ == "__main__":
    main()
