"""Figure 1 — from microscopy to a whole-organ vessel graph (PLOS, 7.5 in wide; old Figs 1–3 combined).

  a  acquisition: imaging setup (figures/fig1/setup.png, rendered from setup.svg; BioRender elements — Created with
     BioRender.com) and one tile's 250 fps recording (tile 36) as a stack of frames → max over time
  b  stitching: the fused whole-organ mosaic with the acquisition order (tile centres in recording order, coloured by
     elapsed time from the Photron .cih clocks); the crop of c boxed
  c  segmentation and vectorization of one crop: mosaic → ilastik vessel probability → extracted graph (radius-encoded)
  d  the whole-organ vessel graph (net3 vectorization of the curated mask), coloured by radius; crop of c boxed

Writes figures/fig1/fig1.{png,pdf,tif}.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Polygon, Ellipse, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.config import StageConfig
import plos_style

PX_UM_RAW = 1.7                 # µm per raw camera pixel (10× objective)
ACCENT = "#e8711a"
EDGE = (75489, 75638)             # representative vessel (tile 36), followed in Figs 2+
INK = "#1a1a1a"


def tile_acq_minutes(raw_dir: Path, pattern: str, tids) -> dict:
    """Elapsed acquisition time per tile (min from the first tile) from each tile's Photron .cih header.  The camera's
    absolute date is a wrong default but its clock runs monotonically through the scan, so relative time is exact."""
    import re, datetime as dt
    times = {}
    for t in tids:
        for d in sorted(raw_dir.glob(pattern.replace("{vid}", str(int(t))).replace("{id}", str(int(t))))):
            cs = list(d.glob("*.cih")) if d.is_dir() else []
            if not cs:
                continue
            txt = cs[0].read_text(errors="ignore")
            dm, tm = re.search(r"Date\s*:\s*([\d/]+)", txt), re.search(r"Time\s*:\s*([\d:]+)", txt)
            if dm and tm:
                for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
                    try:
                        times[int(t)] = dt.datetime.strptime(dm.group(1) + " " + tm.group(1), fmt); break
                    except ValueError:
                        continue
            break
    if not times:
        return {}
    t0 = min(times.values())
    return {t: (v - t0).total_seconds() / 60.0 for t, v in times.items()}


def tile_centres(cfg, mosaic):
    """Tile centres in mosaic pixels.  positions.json is stitch-relative, so the centre cloud is anchored to the organ's
    centroid (the scan covers the organ uniformly), which removes the constant offset."""
    from net3.edit.video_overlay import load_tile_positions
    from scipy import ndimage as ndi
    pos = load_tile_positions(cfg.positions_path); tids = np.array(sorted(pos))
    tl = np.array([[pos[t]["top_left_x"], pos[t]["top_left_y"]] for t in tids], float)
    cy, cx = ndi.center_of_mass(mosaic > 0)
    return tids, tl + np.array([cx - tl[:, 0].mean(), cy - tl[:, 1].mean()])


def scalebar(ax, um_per_px, um, label, color="white", x=0.06, y=0.9, lw=2.0):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); L = um / um_per_px
    xa, ya = x0 + x * (x1 - x0), y0 + y * (y1 - y0)
    ax.plot([xa, xa + L], [ya, ya], color=color, lw=lw, solid_capstyle="butt", zorder=9)
    ax.annotate(label, (xa + L / 2, ya), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                color=color, fontsize=8, zorder=9)


def graph_segments(G, Hm):
    """Edge polylines in mosaic pixel coordinates (row = Hm − y) and radii (px)."""
    segs, rad, keys = [], [], []
    for u, v, d in G.edges(data=True):
        q = np.asarray(d["path"], float) if d.get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in (u, v)])
        segs.append(np.column_stack([q[:, 0], Hm - q[:, 1]])); rad.append(float(d.get("radius") or np.nan)); keys.append((u, v))
    return segs, np.array(rad), keys


def draw_grid(ax, x0, x1, ytop, ybot, nr=5, nc=10):
    """5 rows × 10 columns of overlapping fields of view (~25 % overlap), scanned column by column (boustrophedon):
    top-left → down to bottom-left → one column right → up to the top → …; translucent grey fills, light → dark in
    acquisition order, so overlaps read as darker bands."""
    cols = np.linspace(x0, x1, nc); rows = np.linspace(ytop, ybot, nr)
    w, h = 1.3 * (cols[1] - cols[0]), 1.3 * (rows[0] - rows[1])
    path = np.array([(c, r) for j, c in enumerate(cols) for r in (rows if j % 2 == 0 else rows[::-1])])
    for k, (xx, yy) in enumerate(path):
        ax.add_patch(Rectangle((xx - w / 2, yy - h / 2), w, h, fc=plt.cm.Greys(0.25 + 0.55 * k / (len(path) - 1)), ec="white", lw=0.3,
                               alpha=0.45, zorder=2))
    ax.add_patch(Rectangle((cols[6] - w / 2, rows[1] - h / 2), w, h, fc="none", ec=ACCENT, lw=1.1, zorder=4))
    ax.plot(*path.T, color=INK, lw=0.6, alpha=0.8, zorder=3)
    ax.plot(*path[0], "o", ms=3.5, mfc="white", mec=INK, mew=0.7, zorder=5)
    ax.annotate("", xy=path[-1], xytext=path[-2], arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6), zorder=5)


def draw_setup(fig, cell, img):
    """BioRender setup (objective over the windowed egg) with a callout to the top-down scan grid of the exposed plexus.
    Returns the axes the panel letter is attached to."""
    from matplotlib.patches import ConnectionPatch
    g = cell.subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.05)
    axI = fig.add_subplot(g[0]); axI.imshow(img, interpolation="antialiased"); axI.axis("off")
    Hi, Wi = img.shape[:2]
    axI.text(0.80 * Wi, 0.15 * Hi, "10×\nobjective", ha="left", va="center", fontsize=8, linespacing=1.0)
    axG = fig.add_subplot(g[1]); axG.set_xlim(0, 1); axG.set_ylim(0, 1); axG.set_aspect("equal", adjustable="datalim"); axG.axis("off")
    axG.add_patch(Ellipse((0.5, 0.52), 0.98, 0.50, fc="#e7b9a8", ec="#b06a5a", lw=0.6, alpha=0.9))
    draw_grid(axG, 0.2, 0.8, 0.63, 0.41)
    axG.text(0.5, 0.80, "top view", ha="center", va="bottom", fontsize=8)
    axG.text(0.5, 0.22, "plexus scanned one\nfield of view at a time\n(250 fps movie each)", ha="center", va="top", fontsize=8, linespacing=1.05)
    # zoom lines from the exposed yolk surface to the top view
    for (xi, yi), (xg, yg) in (((0.80 * Wi, 0.46 * Hi), (0.02, 0.62)), ((0.80 * Wi, 0.56 * Hi), (0.02, 0.42))):
        fig.add_artist(ConnectionPatch((xi, yi), (xg, yg), coordsA=axI.transData, coordsB=axG.transData, color="0.45", lw=0.6, ls=(0, (2, 1.5))))
    return axI


def draw_schematic(ax):
    """Plain vector schematic: camera → objective → yolk sac, scanned one field of view at a time."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.17, 0.84), 0.66, 0.14, fc="0.25", ec=INK, lw=0.8))
    ax.text(0.5, 0.91, "high-speed camera\n250 fps", ha="center", va="center", color="white", fontsize=8, linespacing=1.05)
    ax.plot([0.5, 0.5], [0.84, 0.78], color=INK, lw=0.8)
    ax.add_patch(Polygon([[0.38, 0.78], [0.62, 0.78], [0.56, 0.64], [0.44, 0.64]], closed=True, fc="0.7", ec=INK, lw=0.8))
    ax.text(0.66, 0.71, "10×\nobjective", ha="left", va="center", fontsize=8, linespacing=1.0)
    ax.add_patch(Polygon([[0.44, 0.64], [0.56, 0.64], [0.62, 0.42], [0.53, 0.42]], closed=True, fc="#ffe08a", ec="none", alpha=0.6))
    ax.add_patch(Ellipse((0.5, 0.35), 0.99, 0.36, fc="#f3e1c7", ec="#b08850", lw=0.8))
    ax.add_patch(Ellipse((0.5, 0.35), 0.84, 0.27, fc="#e7b9a8", ec="#b06a5a", lw=0.6, alpha=0.9))
    draw_grid(ax, 0.16, 0.84, 0.45, 0.25)
    ax.text(0.5, 0.06, "yolk-sac plexus, scanned\none field of view at a time", ha="center", va="center", fontsize=8, linespacing=1.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=36, help="example tile for a")
    ap.add_argument("--crop-um", type=float, default=650.0, help="side of the c crop (µm), centred on the representative vessel")
    args = ap.parse_args()
    plos_style.apply()
    cfg = StageConfig.load(_paths.STAGE)
    out = _paths.FIGURES / "fig1"
    mos = tifffile.imread(str(_paths.STAGE / "fuse" / "organ_fusion.tif")).astype(float)
    mos = mos.mean(-1) if mos.ndim == 3 else mos
    tids, cen = tile_centres(cfg, mos)
    mins = tile_acq_minutes(Path(cfg.raw_video_dir).expanduser(), cfg.raw_video_dir_pattern, tids)
    tmin = np.array([mins.get(int(t), np.nan) for t in tids])
    print(f"[fig1] {len(tids)} tiles; clocks for {np.isfinite(tmin).sum()}, scan {np.nanmax(tmin):.0f} min")

    vp = next(Path(cfg.videos_realtime_dir).glob(f"*loc{args.tile}_realtime.tif"))
    st = tifffile.imread(str(vp)).astype(np.float32); T = st.shape[0]
    um_rt = PX_UM_RAW * getattr(cfg, "videos_realtime_spatial_ds", 2)
    mip = np.clip(st - np.median(st, axis=0), 0, None).max(0)
    print(f"[fig1] tile {args.tile}: {T} frames {st.shape[1:]} at {um_rt:.1f} µm/px")

    G = _paths.load_graph(); Hm, Wm = mos.shape
    maxmed = tifffile.imread(str(_paths.STAGE / "fuse" / "maxmed.tif")).astype(float); maxmed = maxmed.mean(-1) if maxmed.ndim == 3 else maxmed
    prob = tifffile.imread(str(_paths.STAGE / "fuse" / "probs_logit.tif")).astype(float)
    mask = tifffile.imread(str(_paths.STAGE / "mask" / "mask.tif")) > 127
    segs, rad, keys = graph_segments(G, Hm); rad_um = rad * PX_UM_RAW
    rnorm = Normalize(np.nanpercentile(rad_um, 2), np.nanpercentile(rad_um, 98)); rcmap = plt.cm.viridis
    rcol = lambda r: rcmap(rnorm(np.nan_to_num(r, nan=rnorm.vmin)))
    lw_of = lambda r: 0.2 + 1.0 * np.clip(np.nan_to_num(r / np.nanpercentile(rad_um, 98), nan=0.3), 0, 1)
    # crop of c: centred on the representative vessel (Figs 2+), side args.crop_um
    ve = keys.index(EDGE) if EDGE in keys else keys.index(EDGE[::-1]); vmid = segs[ve][len(segs[ve]) // 2]
    hs = args.crop_um / PX_UM_RAW / 2
    cx0, cx1, cy0, cy1 = int(vmid[0] - hs), int(vmid[0] + hs), int(vmid[1] - hs), int(vmid[1] + hs)

    fig = plt.figure(figsize=(7.5, 8.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.9, 1.75], hspace=0.12, left=0.015, right=0.985, top=0.975, bottom=0.01)
    gTop = gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.8], wspace=0.06)
    letter = lambda ax, l, dx=0.0, dy=1.01: ax.text(dx, dy, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    head = lambda ax, t: ax.text(0.5, 1.01, t, transform=ax.transAxes, fontsize=9, va="bottom", ha="center")

    # a: acquisition — setup drawing + frame stack → max-projection
    gA = gTop[0].subgridspec(1, 2, width_ratios=[0.36, 1.0], wspace=0.02)
    axA = fig.add_subplot(gA[0]); setup = out / "setup.png"
    if setup.exists():
        axA.imshow(plt.imread(str(setup)), interpolation="antialiased"); axA.axis("off")
    else:
        draw_schematic(axA)
    letter(axA, "a")
    axB = fig.add_subplot(gA[1]); axB.axis("off")
    H_, W_ = st.shape[1:]; fw = 1.0; fh = fw * H_ / W_
    n_show, dx, dy = 6, 0.13, 0.11
    lo, hi = np.percentile(st[:20], 1), np.percentile(st[:20], 99.6)
    for k in range(n_show)[::-1]:
        x0, y0 = k * dx, k * dy
        axB.add_patch(Rectangle((x0 + 0.025, y0 - 0.025), fw, fh, fc="0.2", ec="none", alpha=0.18, zorder=2 * (n_show - k)))
        axB.imshow(st[k * 6], cmap="gray", vmin=lo, vmax=hi, extent=(x0, x0 + fw, y0, y0 + fh), zorder=2 * (n_show - k) + 1,
                   interpolation="antialiased")
        axB.add_patch(Rectangle((x0, y0), fw, fh, fc="none", ec="white", lw=0.8, zorder=2 * (n_show - k) + 1.5))
    axB.add_patch(Rectangle((0, 0), fw, fh, fc="none", ec=INK, lw=0.6, zorder=2 * n_show + 2))
    t0, t1 = np.array([fw + 0.06, -0.02]), np.array([fw + 0.06 + (n_show - 1) * dx + 0.06, (n_show - 1) * dy + 0.03])
    axB.annotate("", xy=t1, xytext=t0, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=7))
    axB.text(*(t0 + t1) / 2 + np.array([0.08, -0.06]), "t", fontsize=8, style="italic", ha="left", va="top")
    axB.text(0.5 * (fw + (n_show - 1) * dx), fh + (n_show - 1) * dy + 0.06, f"one tile: {T} frames, 250 fps", ha="center", va="bottom", fontsize=8)
    xm = fw + (n_show - 1) * dx + 0.42; ym = 0.25
    axB.annotate("", xy=(xm - 0.04, fh / 2 + 0.25), xytext=(xm - 0.3, fh / 2 + 0.25), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8))
    axB.text(xm - 0.17, fh / 2 + 0.31, "max\nover t", ha="center", va="bottom", fontsize=8, linespacing=1.0)
    axB.imshow(mip, cmap="magma", vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.7), extent=(xm, xm + fw, ym, ym + fh),
               interpolation="antialiased", zorder=3)
    axB.add_patch(Rectangle((xm, ym), fw, fh, fc="none", ec=ACCENT, lw=1.0, zorder=4))
    L = 200 / um_rt / W_ * fw
    axB.plot([xm + 0.16, xm + 0.16 + L], [ym + 0.07, ym + 0.07], color="white", lw=2, solid_capstyle="butt", zorder=5)
    axB.text(xm + 0.16 + L / 2, ym + 0.1, "200 µm", color="white", ha="center", va="bottom", fontsize=8, zorder=5)
    axB.text(xm + fw / 2, ym + fh + 0.06, "max-projection", ha="center", va="bottom", fontsize=8)
    axB.set_xlim(-0.05, xm + fw + 0.03); axB.set_ylim(-0.1, fh + (n_show - 1) * dy + 0.25); axB.set_aspect("equal")

    # b: stitching — fused mosaic + acquisition order
    axS = fig.add_subplot(gTop[1])
    axS.imshow(np.ma.masked_equal(mos, 0), cmap="gray", vmin=np.percentile(mos[mos > 0], 1), vmax=np.percentile(mos, 99.7), interpolation="antialiased")
    ok = np.isfinite(tmin); norm = Normalize(0, np.nanmax(tmin)); cmap = plt.cm.plasma
    seg = np.stack([cen[:-1], cen[1:]], axis=1); sv = (tmin[:-1] + tmin[1:]) / 2
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=0.9, zorder=2); lc.set_array(np.where(np.isfinite(sv), sv, 0)); axS.add_collection(lc)
    axS.scatter(cen[ok, 0], cen[ok, 1], c=tmin[ok], cmap=cmap, norm=norm, s=5, edgecolor="none", zorder=3)
    axS.scatter(*cen[0], s=30, facecolor="none", edgecolor="white", linewidth=0.8, zorder=4)
    axS.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec="white", lw=0.8, zorder=5))
    axS.set_xlim(0, Wm); axS.set_ylim(Hm, 0); axS.set_aspect("equal"); axS.axis("off")
    scalebar(axS, PX_UM_RAW, 1000, "1 mm", color=INK, x=0.0, y=0.03)
    letter(axS, "b"); head(axS, f"stitched mosaic ({len(tids)} tiles)")
    pS = axS.get_position(); cax = fig.add_axes([pS.x1 - 0.005, pS.y0 + 0.2 * pS.height, 0.01, 0.6 * pS.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("acquisition time (min)", labelpad=2); cb.outline.set_linewidth(0.5)

    # c: segmentation and vectorization of one crop
    gC = gs[1].subgridspec(1, 3, wspace=0.22)
    sub = lambda im: im[cy0:cy1, cx0:cx1]
    inside = [k for k, s_ in enumerate(segs) if ((s_[:, 0] > cx0) & (s_[:, 0] < cx1) & (s_[:, 1] > cy0) & (s_[:, 1] < cy1)).any()]
    for k, ttl in enumerate(("mosaic crop", "vessel probability", "extracted graph")):
        ax = fig.add_subplot(gC[k])
        if k == 0:
            im = sub(maxmed); ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="antialiased")
            letter(ax, "c", -0.02, 1.07); scalebar(ax, PX_UM_RAW, 200, "200 µm", x=0.07, y=0.1)
        elif k == 1:
            im = sub(prob); ax.imshow(im, cmap="magma", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5), interpolation="antialiased")
        else:
            ax.imshow(np.where(sub(mask), 0.9, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.add_collection(LineCollection([segs[i] - [cx0, cy0] for i in inside], colors=rcol(rad_um[inside]), linewidths=lw_of(rad_um[inside]) * 1.6))
        ax.set_xlim(0, cx1 - cx0); ax.set_ylim(cy1 - cy0, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.6"); sp.set_linewidth(0.6)
        ax.set_title(ttl, fontsize=9, pad=2)
        if k < 2:
            ax.annotate("", xy=(1.17, 0.5), xytext=(1.04, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25", mutation_scale=9))
    fig.text(0.5, gC[0].get_position(fig).y1 + 0.035, "segmentation and vectorization", ha="center", va="bottom", fontsize=9)

    # d: whole-organ vessel graph
    axD = fig.add_subplot(gs[2])
    axD.add_collection(LineCollection(segs, colors=rcol(rad_um), linewidths=lw_of(rad_um) * 0.75))
    axD.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec=INK, lw=0.8, zorder=5))
    axD.set_xlim(0, Wm); axD.set_ylim(Hm, 0); axD.set_aspect("equal"); axD.axis("off")
    scalebar(axD, PX_UM_RAW, 1000, "1 mm", color=INK, x=0.02, y=0.03)
    fig.canvas.draw(); pD = axD.get_position()                 # the equal-aspect box actually drawn
    fig.text(pD.x0 - 0.01, pD.y1 + 0.004, "d", fontsize=12, fontweight="bold", va="bottom", ha="left")
    head(axD, f"whole-organ vessel graph ({G.number_of_edges():,} vessels)")
    cax = fig.add_axes([pD.x1 + 0.02, pD.y0 + 0.25 * pD.height, 0.013, 0.5 * pD.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap), cax=cax); cb.set_label("vessel radius (µm)", labelpad=2); cb.outline.set_linewidth(0.5)
    plos_style.save(fig, out / "fig1")


if __name__ == "__main__":
    main()
