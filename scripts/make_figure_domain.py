"""Effective vascular domain.  a: anatomical schematic (figures/fig5/anatomy.png, from Alex's anatomy.svg).  b: the domain on the mosaic: retained (carved) network vs excluded vessels, and the effective boundaries —
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
            for s_, lab, dx in (("L", r"$A^{\mathrm{L}}_{1\ldots%d}$" % nL, -1), ("R", r"$A^{\mathrm{R}}_{1\ldots%d}$" % nR, 1)):
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
    da_stubs = [seg(e) for n in dom["DA"] for e in G.edges(n) if e not in keep and e[::-1] not in keep]
    def draw_da(ax):                                               # dorsal-aorta skeleton (excluded stubs) + termini, in arterial red
        ax.add_collection(LineCollection(da_stubs, colors=ART, linewidths=1.0, alpha=0.95, zorder=4))
        for n, lab, dx in zip(dom["DA"], ("A$_{\mathrm{true},1}$", "A$_{\mathrm{true},2}$"), (-1, 1)):
            p_ = xy(n); ax.scatter(*p_, s=40, color=ART, edgecolor="white", linewidth=0.6, zorder=6)
            ax.text(p_[0] + dx * 22, p_[1] + 10, lab, ha="right" if dx < 0 else "left", va="center", fontsize=9, fontweight="bold",
                    color=ART, path_effects=HALO, zorder=6)
    fig = plt.figure(figsize=(7.5, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.0], wspace=0.06, left=0.01, right=0.99, top=0.9, bottom=0.02)
    axA = fig.add_subplot(gs[0]); sch = _paths.FIGURES / "fig5" / "anatomy.png"     # Alex's drawing (figures/fig5/anatomy.svg)
    if sch.exists():
        axA.imshow(plt.imread(str(sch)), interpolation="antialiased")
    axA.axis("off")
    half_a = 0.62 * max(np.ptp(Pa[:, 0]), np.ptp(Pa[:, 1])); ca = Pa.mean(0)
    Pda = np.array([xy(n) for n in dom["DA"]])                       # dorsal-aorta stub ends sit just below the veins
    cv = np.array([Pv[:, 0].mean(), (Pv[:, 1].min() + Pda[:, 1].max()) / 2 + 40]); half_v = 0.62 * (Pda[:, 1].max() + 120 - Pv[:, 1].min())
    axB = fig.add_subplot(gs[1]); crop(axB, ca, half_a, True, False, 200, show_exc=False); axB.set_title("Arterial cuts", fontsize=9, pad=3)
    axC = fig.add_subplot(gs[2]); crop(axC, cv, half_v, False, True, 100, show_exc=False); draw_da(axC); axC.set_title("Venous outlets", fontsize=9, pad=3)
    for ax_, l_ in ((axA, "a"), (axB, "b"), (axC, "c")):
        ax_.text(0.0, 1.03, l_, transform=ax_.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    plos_style.save(fig, _paths.FIGURES / "fig_domain" / "fig_domain")


if __name__ == "__main__":
    main()
