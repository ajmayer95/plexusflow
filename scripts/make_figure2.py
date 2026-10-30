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


def radius_ticks(paths_rc, radii_px, step_px=22.0, margin_px=3.0, end_gap=1.5):
    """Perpendicular bars of length 2R along each centreline, placed so that no bar crosses (or comes within margin_px of)
    another vessel's centreline or another bar.  Candidates every step_px/3 of arc, skipping end_gap·R near junctions;
    greedy acceptance from each vessel's middle outwards, wider vessels first; bars on one vessel ≥ max(step_px, 2R) apart."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    lines = [LineString(q) if len(q) >= 2 else None for q in paths_rc]
    cand = []
    for k, (q, r) in enumerate(zip(paths_rc, radii_px)):
        if lines[k] is None or not np.isfinite(r) or r <= 0:
            continue
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]; L = d[-1]
        lo, hi = end_gap * r, L - end_gap * r
        if hi <= lo:
            continue
        mid = (lo + hi) / 2; pos = np.arange(lo, hi, step_px / 3.0)
        for a in sorted(pos, key=lambda a: abs(a - mid)):
            i = min(np.searchsorted(d, a), len(q) - 1); i0 = max(i - 1, 0)
            t = q[i] - q[i0]; n_ = np.hypot(*t)
            if n_ == 0:
                continue
            t = t / n_; nrm = np.array([-t[1], t[0]]); c = q[i0] + (a - d[i0]) * t
            cand.append((-r, abs(a - mid), k, a, c - r * nrm, c + r * nrm))
    cand.sort(key=lambda x: (x[0], x[1]))
    tree = STRtree([l for l in lines if l is not None]); idx = [k for k, l in enumerate(lines) if l is not None]
    kept, kept_geom, last = [], [], {}
    for _, _, k, a, p0, p1 in cand:
        r = np.hypot(*(p1 - p0)) / 2
        if any(abs(a - b) < max(step_px, 2 * r) for b in last.get(k, [])):
            continue
        g = LineString([p0, p1]); gb = g.buffer(margin_px)
        hit = [idx[j] for j in tree.query(gb)]
        if any(h != k and lines[h].intersects(gb) for h in hit):
            continue
        if any(gb.intersects(o) for o in kept_geom):
            continue
        kept.append([p0, p1]); kept_geom.append(g); last.setdefault(k, []).append(a)
    return kept

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=36)
    ap.add_argument("--crop-um", type=float, default=650.0)
    ap.add_argument("--edge", type=int, nargs=2, default=[75489, 75638], help="crop centred on this vessel")
    ap.add_argument("--center", type=float, nargs=2, default=[2800, 1300], help="crop centre (mosaic col, row) for b; pass none to auto-place inside --tile")
    ap.add_argument("--out-name", default="fig2")
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
    if args.center:
        vm = np.array(args.center, float)
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
    from scipy.ndimage import map_coordinates
    def tile_on_grid(r0, r1, c0, c1):
        """Tile projection resampled onto the fused-mosaic pixel grid for rows r0:r1, cols c0:c1 (nan outside the tile)."""
        rr, cc = np.mgrid[r0:r1, c0:c1].astype(float)
        hom = np.stack([(rr + geom.y0) * ds, (cc + geom.x0) * ds, np.ones_like(rr)], -1) @ tg.A_full_inv.T
        v = map_coordinates(ti, [hom[..., 0], hom[..., 1]], order=1, cval=np.nan)
        inside_ = (hom[..., 0] >= 0) & (hom[..., 0] <= Ht - 1) & (hom[..., 1] >= 0) & (hom[..., 1] <= Wt - 1)
        return np.where(inside_, v, np.nan)
    if args.center is None:                                  # auto: crop inside the tile, where fusion changes it most
        hs = args.crop_um / PX_UM / 2; best = None
        cy_, cx_ = np.mean(tcy[:4]), np.mean(tcx[:4])
        for fy in np.linspace(-0.3, 0.3, 7):
            for fx in np.linspace(-0.3, 0.3, 7):
                yc, xc = cy_ + fy * Ht / ds, cx_ + fx * Wt / ds
                r0, r1, c0, c1 = int(yc - hs), int(yc + hs), int(xc - hs), int(xc + hs)
                T_ = tile_on_grid(r0, r1, c0, c1)
                if np.isnan(T_).any():
                    continue
                F_ = mm[r0:r1, c0:c1]; d_ = 1 - np.corrcoef(T_.ravel(), F_.ravel())[0, 1]
                if best is None or d_ > best[0]:
                    best = (d_, r0, r1, c0, c1)
        _, cy0, cy1, cx0, cx1 = best
        print(f"[fig2] auto crop inside tile {args.tile}: rows {cy0}:{cy1}, cols {cx0}:{cx1} (tile vs fused 1 − r = {best[0]:.2f})")
        inside = [k for k, s_ in enumerate(segs) if ((s_[:, 0] > cx0) & (s_[:, 0] < cx1) & (s_[:, 1] > cy0) & (s_[:, 1] < cy1)).any()]
    ys, xs = np.nonzero(organ); pad = 40
    ox0, ox1, oy0, oy1 = max(xs.min() - pad, 0), min(xs.max() + pad, Wm), max(ys.min() - pad, 0), min(ys.max() + pad, Hm)

    fig = plt.figure(figsize=(7.5, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.14, wspace=0.08, left=0.03, right=0.925, top=0.95, bottom=0.03)
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
            ax.plot(tcx, tcy, color="#e0a800", lw=1.0, zorder=6)
        ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=INK, lw=0.9, zorder=5))
        ax.set_xlim(ox0, ox1); ax.set_ylim(oy1, oy0); ax.set_aspect("equal"); frame(ax, False)
        ax.set_title(ttl, fontsize=8, pad=2)
        if k < 3:
            ax.annotate("", xy=(1.06, 0.5), xytext=(1.0, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, mutation_scale=6))
    letter(top[0], "a")
    scalebar(top[0], 1000, PX_UM, x_frac=0.22, y_frac=1.0, color=INK, halo=False)

    # bottom row: b graph close-up (radius bars) · c mosaic space around the tile · A⁻¹ · d the tile's own frame
    gB = gs[1, :].subgridspec(1, 4, width_ratios=[1.0, 1.15, 0.22, 1.05], wspace=0.05)
    axb = fig.add_subplot(gB[0])
    axb.imshow(np.where(sub(mask), 0.9, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cl = [segs[i] - [cx0, cy0] for i in inside]
    axb.add_collection(LineCollection(radius_ticks(cl, rad_um[inside] / PX_UM), colors="0.3", linewidths=0.5))
    axb.add_collection(LineCollection(cl, colors=rcol(rad_um[inside]), linewidths=1.1))
    axb.set_xlim(0, cx1 - cx0); axb.set_ylim(cy1 - cy0, 0); axb.set_aspect("equal"); frame(axb, True)
    letter(axb, "b"); scalebar(axb, 200, PX_UM, color=INK, halo=False)

    # c: mosaic space — fused image + graph around tile 36, its footprint as placed (rotated, scaled)
    axm = fig.add_subplot(gB[1]); m_ = 0.18 * max(Ht, Wt) / ds
    X0, X1, Y0, Y1 = tcx[:4].min() - m_, tcx[:4].max() + m_, tcy[:4].min() - m_, tcy[:4].max() + m_
    r0, r1, c0, c1 = int(max(Y0, 0)), int(min(Y1, Hm)), int(max(X0, 0)), int(min(X1, Wm))
    reg = mm[r0:r1, c0:c1]
    axm.imshow(reg, cmap="gray", vmin=np.percentile(reg, 1), vmax=np.percentile(reg, 99.5), extent=(c0, c1, r1, r0), interpolation="antialiased")
    near = [k for k, s_ in enumerate(segs) if ((s_[:, 0] > c0) & (s_[:, 0] < c1) & (s_[:, 1] > r0) & (s_[:, 1] < r1)).any()]
    axm.add_collection(LineCollection([segs[k] for k in near], colors="white", linewidths=0.35, alpha=0.45))
    axm.plot(tcx, tcy, color="#ffd23f", lw=1.2, zorder=6)
    axm.plot(tcx[0], tcy[0], "o", ms=4, color="#ffd23f", zorder=7)                         # tile origin corner, to read the rotation
    axm.set_xlim(c0, c1); axm.set_ylim(r1, r0); axm.set_aspect("equal"); frame(axm, False)
    axm.set_title("Mosaic frame", fontsize=8, pad=2); letter(axm, "c"); scalebar(axm, 200, PX_UM)

    # arrow: the inverse of the tile's registration
    axa = fig.add_subplot(gB[2]); axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axa.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK, mutation_scale=10))
    axa.text(0.5, 0.56, f"$A_{{{args.tile}}}^{{-1}}$", ha="center", va="bottom", fontsize=9)

    # d: the tile's own frame (its recorded movie's projection) with the graph mapped in
    axC = fig.add_subplot(gB[3])
    axC.imshow(ti, cmap="gray", vmin=np.percentile(ti, 1), vmax=np.percentile(ti, 99.5), interpolation="antialiased")
    loc, lr = [], []
    for q, r in zip(paths, rad_um):
        rc = graph_to_tile_batch(q, geom, args.tile)
        if ((rc[:, 0] > -5) & (rc[:, 0] < Ht + 5) & (rc[:, 1] > -5) & (rc[:, 1] < Wt + 5)).any():
            loc.append(rc[:, ::-1]); lr.append(r)
    lr = np.array(lr)
    from plexusflow.analyze.geometry import warp_mask_to_tile
    if getattr(tg, "video_shape", None) is None or tuple(tg.video_shape) != (Ht, Wt):
        tg.video_shape = (Ht, Wt)
    mt = warp_mask_to_tile(mask, geom, args.tile)                      # the tile-level vessel mask used for flow
    ov = np.zeros((Ht, Wt, 4)); ov[mt] = (0.0, 0.78, 1.0, 0.35)
    axC.imshow(ov, interpolation="nearest", zorder=3)
    axC.add_collection(LineCollection(loc, colors="#ffd23f", linewidths=0.5, alpha=0.95, zorder=4))
    axC.plot([0, Wt, Wt, 0, 0], [0, 0, Ht, Ht, 0], color="#ffd23f", lw=0.9, zorder=6, clip_on=False)
    axC.plot(0, 0, "o", ms=4, color="#ffd23f", zorder=7, clip_on=False)
    axC.set_xlim(0, Wt); axC.set_ylim(Ht, 0); axC.set_aspect("equal"); frame(axC, False)
    axC.set_title("Tile frame", fontsize=8, pad=2); letter(axC, "d"); scalebar(axC, 100, PX_UM)
    print(f"[fig2] tile {args.tile}: {len(loc)} edges mapped into its frame")
    plos_style.save(fig, _paths.FIGURES / "fig2" / args.out_name)


if __name__ == "__main__":
    main()
