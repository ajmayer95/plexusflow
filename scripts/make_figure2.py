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

    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1.0], hspace=0.1, left=0.02, right=0.98, top=0.965, bottom=0.02)
    gA = gs[0].subgridspec(1, 4, wspace=0.12)
    gB = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    letter = lambda ax, l, dx=0.0: ax.text(dx, 1.02, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: crop pipeline
    sub = lambda im: im[cy0:cy1, cx0:cx1]
    inside = [k for k, s in enumerate(segs) if ((s[:, 0] > cx0) & (s[:, 0] < cx1) & (s[:, 1] > cy0) & (s[:, 1] < cy1)).any()]
    for k, ttl in enumerate(("Projection", "Probability", "Mask", "Graph")):
        ax = fig.add_subplot(gA[k])
        if k == 0:
            im = sub(mm); ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="antialiased")
        elif k == 1:
            im = sub(prob); ax.imshow(im, cmap="magma", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="antialiased")
        elif k == 2:
            ax.imshow(np.where(sub(mask), 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(np.where(sub(mask), 0.9, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.add_collection(LineCollection([segs[i] - [cx0, cy0] for i in inside], colors=rcol(rad_um[inside]), linewidths=lw_of(rad_um[inside]) * 1.8))
        ax.set_xlim(0, cx1 - cx0); ax.set_ylim(cy1 - cy0, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.6"); sp.set_linewidth(0.6)
        ax.set_title(ttl, fontsize=8, pad=2)
        if k == 0:
            letter(ax, "a", -0.02); scalebar(ax, 200, PX_UM)
        if k < 3:
            ax.annotate("", xy=(1.11, 0.5), xytext=(1.01, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8))

    # b: whole graph by radius
    axB = fig.add_subplot(gB[0])
    axB.add_collection(LineCollection(segs, colors=rcol(rad_um), linewidths=lw_of(rad_um) * 0.7))
    axB.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=INK, lw=0.8))
    geom = load_tile_geometry(StageConfig.load(P)); tg = geom.tiles[args.tile]
    Ht, Wt = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{args.tile}_maxmed.tif")).shape[:2]
    corners = np.array([[0, 0, 1], [0, Wt, 1], [Ht, Wt, 1], [Ht, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
    ds = geom.downsample; axB.plot(corners[:, 1] / ds - geom.x0, corners[:, 0] / ds - geom.y0, color="#e8711a", lw=1.0)
    axB.set_xlim(0, Wm); axB.set_ylim(Hm, 0); axB.set_aspect("equal"); axB.axis("off")
    scalebar(axB, 1000, PX_UM, x_frac=0.3, y_frac=0.98, color=INK, halo=False)
    fig.canvas.draw(); pB = axB.get_position()
    fig.text(pB.x0, pB.y1 + 0.005, "b", fontsize=12, fontweight="bold", va="bottom", ha="left")
    cax = fig.add_axes([pB.x0 + 0.62 * pB.width, pB.y0 + 0.04, 0.32 * pB.width, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap), cax=cax, orientation="horizontal"); cb.set_label("Radius (µm)", labelpad=1)
    cb.outline.set_linewidth(0.5)

    # c: graph in tile 36's own frame
    axC = fig.add_subplot(gB[1])
    ti = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{args.tile}_maxmed.tif")).astype(float); ti = ti.mean(-1) if ti.ndim == 3 else ti
    axC.imshow(ti, cmap="gray", vmin=np.percentile(ti, 1), vmax=np.percentile(ti, 99.5), interpolation="antialiased")
    loc, lr = [], []
    for q, r in zip(paths, rad_um):
        rc = graph_to_tile_batch(q, geom, args.tile)
        if ((rc[:, 0] > -5) & (rc[:, 0] < Ht + 5) & (rc[:, 1] > -5) & (rc[:, 1] < Wt + 5)).any():
            loc.append(rc[:, ::-1]); lr.append(r)
    lr = np.array(lr)
    axC.add_collection(LineCollection(loc, colors=rcol(lr), linewidths=lw_of(lr) * 1.6, alpha=0.95))
    axC.set_xlim(0, Wt); axC.set_ylim(Ht, 0); axC.set_aspect("equal"); axC.set_xticks([]); axC.set_yticks([])
    for sp in axC.spines.values():
        sp.set_color("#e8711a"); sp.set_linewidth(1.2)
    scalebar(axC, 100, PX_UM)
    letter(axC, "c", 0.0)
    print(f"[fig2] tile {args.tile}: {len(loc)} edges mapped into its frame")
    plos_style.save(fig, _paths.FIGURES / "fig2" / "fig2")


if __name__ == "__main__":
    main()
