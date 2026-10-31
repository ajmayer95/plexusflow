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
RET, EXC, ART, VEN, INK = "#f2c94c", "#ff4d4d", "#d7191c", "#2c7bb6", "#1a1a1a"


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

    fig = plt.figure(figsize=(7.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.0], wspace=0.04, left=0.01, right=0.99, top=0.94, bottom=0.01)
    axA = fig.add_subplot(gs[0]); sch = _paths.FIGURES / "fig5" / "anatomy.png"     # Alex's drawing (figures/fig5/anatomy.svg)
    if sch.exists():
        axA.imshow(plt.imread(str(sch)), interpolation="antialiased")
    axA.axis("off"); axA.text(0.0, 1.0, "a", transform=axA.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    ax = fig.add_subplot(gs[1]); ax.text(0.0, 1.0, "b", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    ax.imshow(np.ma.masked_where(out, mos), cmap="gray", vmin=np.percentile(mos[~out], 1), vmax=np.percentile(mos[~out], 99.5), alpha=0.75)
    ax.add_collection(LineCollection(ret, colors=RET, linewidths=0.35, alpha=0.9, zorder=2))
    ax.add_collection(LineCollection(exc, colors=EXC, linewidths=1.6, alpha=0.95, zorder=3))
    for s_, lab, dx in (("L", "A$_1$", -1), ("R", "A$_2$", 1)):
        P = np.array([xy(n) for n, s in zip(inj, side) if s == s_])
        ax.scatter(*P.T, s=9, color=ART, edgecolor="white", linewidth=0.4, zorder=5)
        c = P.mean(0); ax.annotate(lab, c, xytext=(dx * 26, 10), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color=ART, path_effects=HALO,
                                   arrowprops=dict(arrowstyle="-", color=ART, lw=0.6), zorder=6)
    for n, lab, dx in zip(dom["V"], ("V$_1$", "V$_2$"), (-1, 1)):
        p = xy(n); ax.scatter(*p, s=40, marker="s", color=VEN, edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(lab, p, xytext=(dx * 18, -12), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color=VEN, path_effects=HALO, zorder=6)
    ys, xs = np.nonzero(~out); ax.set_xlim(xs.min() - 60, xs.max() + 60); ax.set_ylim(ys.max() + 60, ys.min() - 60); ax.axis("off")
    x0 = xs.min(); L = 1000 / PX_UM; y = ys.max() + 30
    ax.plot([x0, x0 + L], [y, y], color=INK, lw=2, solid_capstyle="butt"); ax.text(x0 + L / 2, y - 25, "1 mm", ha="center", va="bottom", fontsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=RET, lw=1.5, label="Retained network"), Line2D([], [], color=EXC, lw=2, label="Excluded"),
                       Line2D([], [], marker="o", ls="", color=ART, label="Arterial cut (A)"), Line2D([], [], marker="s", ls="", color=VEN, label="Venous outlet (V)")],
              loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", fontsize=8, handlelength=1.2, borderaxespad=0.2)
    plos_style.save(fig, _paths.FIGURES / "fig_domain" / "fig_domain")


if __name__ == "__main__":
    main()
