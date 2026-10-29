"""Figure 2 — from the mosaic to a vessel graph (PLOS, 7.5 in wide).

  a  one crop through the pipeline: projection → ilastik vessel probability → curated mask → vectorized graph
     (net3; edges coloured by radius); the crop is boxed in b
  b  the whole-organ vessel graph, coloured by radius
  c  the graph mapped back into one tile's own recorded frame (tile 36, its max − median projection): every edge path
     transformed by the inverse of the tile's registration affine

Inputs: fuse/maxmed.tif, fuse/probs_logit.tif, mask/mask.tif, the canonical graph, tiles/maxmed, stitch geometry.
Writes figures/fig2/fig2.{png,pdf,tif}.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.config import StageConfig
from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch
import plos_style

PX_UM = 1.7
INK = "#1a1a1a"
HALO = [pe.withStroke(linewidth=1.6, foreground="black")]


def scalebar(ax, um, px_um, x_frac=0.95, y_frac=0.93, color="white", halo=True):
    """Bottom-right bar, label right-aligned above it (axes in image coordinates, y down)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); L = um / px_um
    top, bot = min(y0, y1), max(y0, y1)                  # works for image (y down) and data axes
    xe = x0 + x_frac * (x1 - x0); ya = top + y_frac * (bot - top) if y0 > y1 else bot - y_frac * (bot - top)
    kw = dict(path_effects=HALO) if halo else {}
    ax.plot([xe - L, xe], [ya, ya], color=color, lw=2, solid_capstyle="butt", **kw)
    ax.annotate(f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm", (xe, ya), xytext=(0, 2), textcoords="offset points",
                ha="right", va="bottom", color=color, fontsize=8, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=36)
    ap.add_argument("--crop-um", type=float, default=650.0)
    ap.add_argument("--edge", type=int, nargs=2, default=[75489, 75638], help="crop centred on this vessel")
    args = ap.parse_args()
    plos_style.apply()
    P = _paths.STAGE; G = _paths.load_graph()
    mm = tifffile.imread(str(P / "fuse" / "maxmed.tif")).astype(float); mm = mm.mean(-1) if mm.ndim == 3 else mm
    prob = tifffile.imread(str(P / "fuse" / "probs_logit.tif")).astype(float)
    mask = tifffile.imread(str(P / "mask" / "mask.tif")) > 127
    from scipy import ndimage as _ndi0
    _lab, _n = _ndi0.label(mask); _sz = np.bincount(_lab.ravel()); _sz[0] = 0
    mask = _lab == int(np.argmax(_sz))                    # display: keep the connected vascular network, drop isolated specks
    print(f"[fig2] mask: kept largest component, dropped {_n - 1} specks ({int(_sz.sum() - _sz.max())} px)")
    Hm, Wm = mm.shape
    segs, rad, keys, paths = [], [], [], []
    for u, v, d in G.edges(data=True):
        q = np.asarray(d["path"], float) if d.get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in (u, v)])
        paths.append(q); segs.append(np.column_stack([q[:, 0], Hm - q[:, 1]])); rad.append(float(d.get("radius") or np.nan)); keys.append((u, v))
    rad_um = np.array(rad) * PX_UM
    rnorm = Normalize(np.nanpercentile(rad_um, 2), np.nanpercentile(rad_um, 98)); rcmap = plt.cm.viridis
    rcol = lambda r: rcmap(rnorm(np.nan_to_num(r, nan=rnorm.vmin)))
    lw_of = lambda r: 0.2 + 1.0 * np.clip(np.nan_to_num(r / np.nanpercentile(rad_um, 98), nan=0.3), 0, 1)
    e = tuple(args.edge); ve = keys.index(e) if e in keys else keys.index(e[::-1]); vm = segs[ve][len(segs[ve]) // 2]
    hs = args.crop_um / PX_UM / 2; cx0, cx1, cy0, cy1 = int(vm[0] - hs), int(vm[0] + hs), int(vm[1] - hs), int(vm[1] + hs)
    print(f"[fig2] {G.number_of_edges():,} edges; crop centred on {e}")

    from matplotlib.patches import ConnectionPatch
    from scipy import ndimage as _ndi
    organ = _ndi.binary_fill_holes(_ndi.binary_closing(mask, iterations=25))           # organ outline from the curated mask
    geom = load_tile_geometry(StageConfig.load(P)); tg = geom.tiles[args.tile]
    ti = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{args.tile}_maxmed.tif")).astype(float); ti = ti.mean(-1) if ti.ndim == 3 else ti
    Ht, Wt = ti.shape; ds = geom.downsample
    tc = np.array([[0, 0, 1], [0, Wt, 1], [Ht, Wt, 1], [Ht, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
    tcx, tcy = tc[:, 1] / ds - geom.x0, tc[:, 0] / ds - geom.y0
    ys, xs = np.nonzero(organ); pad = 40
    ox0, ox1, oy0, oy1 = max(xs.min() - pad, 0), min(xs.max() + pad, Wm), max(ys.min() - pad, 0), min(ys.max() + pad, Hm)

    fig = plt.figure(figsize=(7.5, 4.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.12, wspace=0.08, left=0.03, right=0.985, top=0.95, bottom=0.11)
    letter = lambda ax, l, dx=0.0: ax.text(dx, 1.03, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    def frame(ax, on):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(on); sp.set_color("0.6"); sp.set_linewidth(0.6)
    sub = lambda im: im[cy0:cy1, cx0:cx1]
    inside = [k for k, s_ in enumerate(segs) if ((s_[:, 0] > cx0) & (s_[:, 0] < cx1) & (s_[:, 1] > cy0) & (s_[:, 1] < cy1)).any()]
    vmm = np.percentile(mm[organ], (1, 99.5)); vpr = np.percentile(prob[organ], (1, 99.5))

    # a: whole-organ stages
    top = []
    for k, ttl in enumerate(("Stitched mosaic", "Probability", "Mask", "Graph")):
        ax = fig.add_subplot(gs[0, k]); top.append(ax)
        if k == 0:
            ax.imshow(np.ma.masked_where(~organ, mm), cmap="gray", vmin=vmm[0], vmax=vmm[1], interpolation="antialiased")
        elif k == 1:
            ax.imshow(np.ma.masked_where(~organ, prob), cmap="magma", vmin=vpr[0], vmax=vpr[1], interpolation="antialiased")
        elif k == 2:
            ax.imshow(np.where(mask, 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        else:
            ax.add_collection(LineCollection(segs, colors=rcol(rad_um), linewidths=lw_of(rad_um) * 0.45))
            ax.plot(tcx, tcy, color=INK, lw=0.8, ls=(0, (2.5, 1.5)), zorder=6)
        ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=INK, lw=0.9, zorder=5))
        ax.set_xlim(ox0, ox1); ax.set_ylim(oy1, oy0); ax.set_aspect("equal"); frame(ax, False)
        ax.set_title(ttl, fontsize=8, pad=2)
        if k < 3:
            ax.annotate("", xy=(1.07, 0.5), xytext=(0.99, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8))
    letter(top[0], "a")
    scalebar(top[2], 1000, PX_UM, x_frac=0.32, y_frac=1.0, color=INK, halo=False)

    # b: matched local detail (probability, mask, graph) of the boxed region
    bot = []
    for k, ttl in enumerate(("Probability", "Mask", "Graph")):
        ax = fig.add_subplot(gs[1, k]); bot.append(ax)
        if k == 0:
            ax.imshow(sub(prob), cmap="magma", vmin=vpr[0], vmax=vpr[1], interpolation="antialiased")
        elif k == 1:
            ax.imshow(np.where(sub(mask), 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(np.where(sub(mask), 0.9, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.add_collection(LineCollection([segs[i] - [cx0, cy0] for i in inside], colors=rcol(rad_um[inside]), linewidths=lw_of(rad_um[inside]) * 1.5))
        ax.set_xlim(0, cx1 - cx0); ax.set_ylim(cy1 - cy0, 0); ax.set_aspect("equal"); frame(ax, True)
    letter(bot[0], "b"); scalebar(bot[0], 200, PX_UM)
    fig.canvas.draw(); pZ = bot[2].get_position()
    cax = fig.add_axes([pZ.x0 + 0.1 * pZ.width, pZ.y0 - 0.035, 0.8 * pZ.width, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap), cax=cax, orientation="horizontal"); cb.set_label("Radius (µm)", labelpad=1)
    cb.outline.set_linewidth(0.5)

    # c: tile 36's own frame with the globally derived graph mapped in
    axC = fig.add_subplot(gs[1, 3])
    axC.imshow(ti, cmap="gray", vmin=np.percentile(ti, 1), vmax=np.percentile(ti, 99.5), interpolation="antialiased")
    loc, lr = [], []
    for q, r in zip(paths, rad_um):
        rc = graph_to_tile_batch(q, geom, args.tile)
        if ((rc[:, 0] > -5) & (rc[:, 0] < Ht + 5) & (rc[:, 1] > -5) & (rc[:, 1] < Wt + 5)).any():
            loc.append(rc[:, ::-1]); lr.append(r)
    lr = np.array(lr)
    axC.add_collection(LineCollection(loc, colors=rcol(lr), linewidths=lw_of(lr) * 1.2, alpha=0.95))
    axC.set_xlim(0, Wt); axC.set_ylim(Ht, 0); axC.set_aspect("equal"); frame(axC, False)
    scalebar(axC, 100, PX_UM); letter(axC, "c")
    k_top = int(np.argmax(tcx[:4] - tcy[:4] * 0)); kb = int(np.argmax(tcy[:4]))
    fig.add_artist(ConnectionPatch((tcx[kb], tcy[kb]), (0.5, 1.0), coordsA=top[3].transData, coordsB=axC.transAxes, color=INK, lw=0.6,
                                   ls=(0, (2.5, 1.5)), arrowstyle="-|>", mutation_scale=7))
    print(f"[fig2] tile {args.tile}: {len(loc)} edges mapped into its frame")
    plos_style.save(fig, _paths.FIGURES / "fig2" / "fig2")


if __name__ == "__main__":
    main()
