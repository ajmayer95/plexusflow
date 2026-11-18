"""Figure 5 — effective vascular domain.  a: anatomical schematic (<stage>/assets/fig5_anatomy.png, rendered from fig5_anatomy.svg).  b: the domain on the mosaic: retained (carved) network vs excluded vessels, and the effective boundaries —
arterial cut vertices grouped as A1 (left) / A2 (right) and the vitelline-vein outlets V1 / V2.

Writes figures/fig_domain/fig_domain.{png,pdf,tif}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.reconcile import carved_domain
import plos_style
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

PX_UM = 1.7
RET, EXC, ART, VEN, INK = "#ffd23f", "#9a9a9a", "#d40000", "#2b35a8", "#1a1a1a"


def main():
    plos_style.apply()
    G = _paths.load_graph(); dom = carved_domain(G)
    mos = tifffile.imread(str(_paths.STAGE / "fuse" / "organ_fusion.tif")).astype(float); Hm, Wm = mos.shape
    out = ~ndi.binary_fill_holes(mos > 0)
    seg = lambda e: (lambda p: np.column_stack([p[:, 0], Hm - p[:, 1]]))(
        np.asarray(G.edges[e]["path"], float) if G.edges[e].get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e]))
    keep = set(map(tuple, dom["edges"]))
    ret = [seg(e) for e in dom["edges"]]
    exc = [seg(e) for e in G.edges() if e not in keep and e[::-1] not in keep]
    xy = lambda n: np.array([G.nodes[n]["x"], Hm - G.nodes[n]["y"]])
    inj = [dom["nodes"][i] for i in dom["inj"]]; side = np.asarray(dom["side"])
    print(f"[domain] retained {len(ret)} vessels, excluded {len(exc)}; arterial cuts L {np.sum(side == 'L')}, R {np.sum(side == 'R')}; V {dom['V']}")

    vmin_, vmax_ = np.percentile(mos[~out], 1), np.percentile(mos[~out], 99.5)
    Pa = np.array([xy(n) for n in inj]); Pv = np.array([xy(n) for n in dom["V"]])
    def crop(ax, ctr, half, show_A, show_V, sb_um, show_exc=True, net_col=RET):
        ax.imshow(np.ma.masked_where(out, mos), cmap="gray", vmin=vmin_, vmax=vmax_, alpha=0.8, interpolation="antialiased")
        ax.add_collection(LineCollection(ret, colors=net_col, linewidths=0.7, alpha=0.95, zorder=2))
        if show_exc:
            ax.add_collection(LineCollection(exc, colors=EXC, linewidths=1.8, alpha=0.95, zorder=3))
        if show_A:
            nL, nR = int(np.sum(side == "L")), int(np.sum(side == "R"))
            for s_, lab, dx in (("L", r"$A_{\mathrm{L}}$", -1), ("R", r"$A_{\mathrm{R}}$", 1)):   # individual cut nodes (21S, 11-17: 20 L / 13 R): caption
                P_ = np.array([xy(n) for n, s in zip(inj, side) if s == s_])
                ax.scatter(*P_.T, s=18, color=ART, edgecolor="white", linewidth=0.5, zorder=5)
                ax.text(P_[:, 0].max() + 40 if dx < 0 else P_[:, 0].min() - 40, P_[:, 1].mean(), lab, ha="left" if dx < 0 else "right", va="center",
                        fontsize=9, fontweight="bold", color=ART, path_effects=HALO, zorder=6)
        if show_V:
            for n, lab, dx in zip(dom["V"], ("V$_1$", "V$_2$"), (-1, 1)):
                p_ = xy(n); ax.scatter(*p_, s=90, marker="s", color="white", edgecolor=VEN, linewidth=1.4, zorder=5)
                ax.text(p_[0] + dx * 18, p_[1] + 18, lab, ha="right" if dx < 0 else "left", va="top", fontsize=9, fontweight="bold", color=VEN,
                        path_effects=HALO, zorder=6)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] + half, ctr[1] - half); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("0.6"); sp_.set_linewidth(0.6)
        L = sb_um / PX_UM; xe = ctr[0] + 0.92 * half; ya = ctr[1] + 0.88 * half
        ax.plot([xe - L, xe], [ya, ya], color="white", lw=2, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=3.2, foreground="black")])
        ax.text(xe, ya - 0.04 * half, f"{sb_um:g} µm", color="white", ha="right", va="bottom", fontsize=8, path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
    def smooth_line(P_, sigma_px=12.0):
        """Display-only smoothing of a skeleton polyline: resample at 1 px along the arc, Gaussian-smooth, keep both ends."""
        P_ = np.asarray(P_, float); d_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]
        if len(P_) < 3 or d_[-1] < 3 * sigma_px: return P_
        t_ = np.linspace(0, d_[-1], int(d_[-1]) + 1); Q_ = np.column_stack([np.interp(t_, d_, P_[:, 0]), np.interp(t_, d_, P_[:, 1])])
        S_ = ndi.gaussian_filter1d(Q_, sigma_px, axis=0, mode="nearest"); w_ = np.clip(np.minimum(t_, d_[-1] - t_) / (2 * sigma_px), 0, 1)[:, None]
        return w_ * S_ + (1 - w_) * Q_                                 # blend back to the raw ends so the termini stay on their nodes
    da_stubs = [smooth_line(seg(e)) for n in dom["DA"] for e in G.edges(n) if e not in keep and e[::-1] not in keep]
    def draw_da(ax):                                               # dorsal-aorta skeleton (excluded stubs) + termini, in arterial red
        ax.add_collection(LineCollection(da_stubs, colors=ART, linewidths=1.0, alpha=0.95, zorder=4))
        for n, lab, dx in zip(dom["DA"], (r"A$_{\mathrm{DA},1}$", r"A$_{\mathrm{DA},2}$"), (-1, 1)):
            p_ = xy(n); ax.scatter(*p_, s=40, color=ART, edgecolor="white", linewidth=0.6, zorder=6)
            ax.text(p_[0] + dx * 22, p_[1] + 10, lab, ha="right" if dx < 0 else "left", va="center", fontsize=9, fontweight="bold",
                    color=ART, path_effects=HALO, zorder=6)
    fig = plt.figure(figsize=(5.2, 4.3))                       # a full height left; b, c stacked right
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.14, wspace=0.05, left=0.01, right=0.99, top=0.93, bottom=0.01)
    axA = fig.add_subplot(gs[:, 0]); sch = _paths.STAGE / "assets" / "fig5_anatomy.png"     # drawing (assets/fig5_anatomy.svg), an input
    if not sch.exists():
        raise FileNotFoundError(f"{sch} missing: the drawing is an input (data release <stage>/assets/)")
    if True:
        axA.imshow(plt.imread(str(sch)), interpolation="antialiased")
        # anatomical labels on the cartoon (pixel coordinates of anatomy.png, 2412 × 3015)
        # label coordinates were laid out on the old raster (stretched to 2412 × 3015); the current raster keeps the drawing's
        # true aspect (2412 × H), so y is rescaled by H/3015
        k_ = 1.5075; ky_ = plt.imread(str(sch)).shape[0] / 3015
        P_ = lambda x, y: (x * k_, y * k_ * ky_)
        lab_kw = dict(fontsize=8, color=INK, ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
        axA.annotate("Heart", P_(840, 230), xytext=P_(1300, 230), **lab_kw)
        axA.annotate("Left vitelline\nvein", P_(380, 690), xytext=P_(240, 470), **lab_kw)      # image left = embryo left
        axA.annotate("Right vitelline\nvein", P_(1250, 690), xytext=P_(1400, 470), **lab_kw)   # (heart loops to image right)
        axA.annotate("Dorsal\naortae", P_(700, 900), xytext=P_(320, 950), **lab_kw)
        axA.annotate("", P_(890, 900), xytext=P_(420, 940), arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
        axA.annotate("Developing\nvitelline arteries", P_(560, 1700), xytext=P_(800, 2080), **lab_kw)
        axA.annotate("", P_(1090, 1700), xytext=P_(860, 2040), arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
    axA.axis("off")
    half_a = 0.62 * max(np.ptp(Pa[:, 0]), np.ptp(Pa[:, 1])); ca = Pa.mean(0)
    Pda = np.array([xy(n) for n in dom["DA"]])                       # dorsal-aorta stub ends sit just below the veins
    cv = np.array([Pv[:, 0].mean(), (Pv[:, 1].min() + Pda[:, 1].max()) / 2 + 40]); half_v = 0.62 * (Pda[:, 1].max() + 120 - Pv[:, 1].min())
    axB = fig.add_subplot(gs[1, 1]); crop(axB, ca, half_a, True, False, 200, show_exc=False); axB.set_title("Effective arterial boundaries", fontsize=9, pad=3)
    axC = fig.add_subplot(gs[0, 1]); crop(axC, cv, half_v, False, True, 100, show_exc=False); draw_da(axC); axC.set_title("Anatomical boundaries", fontsize=9, pad=3)
    yb = axC.get_position().y1 + 0.015
    fig.text(0.01, yb, "a", fontsize=12, fontweight="bold", ha="left", va="bottom")                         # level with b
    fig.text(0.5 * (axA.get_position().x0 + axA.get_position().x1), yb + 0.003, "Computational domain", fontsize=9, ha="center", va="bottom")
    for ax_, l_ in ((axC, "b"), (axB, "c")):
        ax_.text(-0.02, 1.015, l_, transform=ax_.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    plos_style.save(fig, _paths.FIGURES / "fig5" / "fig5")


if __name__ == "__main__":
    main()
