"""Figure 1 — acquisition: the organ is recorded tile by tile, one high-speed movie per tile (PLOS, 7.5 in wide).

  a  imaging setup: figures/fig1/setup_biorender.png (objective over the windowed egg; Created with BioRender.com) with a
     callout to the top view of the plexus, covered by a 5 × 10 grid of overlapping fields of view scanned column by
     column; without that file a plain drawn schematic is used
  b  one tile (default tile 36, the representative vessel's tile of Figs 3–4): a few consecutive frames of its 250 fps
     recording and the moving max-projection (frame − temporal median, max over time) that shows the perfused lumen
  c  acquisition order: tile centres joined in recording order (boustrophedon), coloured by elapsed acquisition time
     (Photron .cih clocks, relative to the first tile) over the fused organ; the tile of b boxed

Writes figures/fig1/fig1.{png,pdf,tif}.  Ported from the archived hand-assembly scripts (make_figure1.py,
make_acquisition_assets.py, 2026-08) into one code-generated figure.
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
    ap.add_argument("--tile", type=int, default=36, help="example tile for b")
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

    fig = plt.figure(figsize=(7.5, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.15, 1.05], wspace=0.14, left=0.01, right=0.885, top=0.9, bottom=0.03)
    # a
    setup = out / "setup_biorender.png"
    if setup.exists():                                    # BioRender setup + top-view callout (credit: Created with BioRender.com)
        axA = draw_setup(fig, gs[0], plt.imread(str(setup)))
    else:
        axA = fig.add_subplot(gs[0]); draw_schematic(axA)
    axA.text(0.0, 1.02, "a", transform=axA.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    # b: frames (offset deck) and the moving max-projection
    gB = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    axF = fig.add_subplot(gB[0]); axF.set_xlim(0, 1); axF.set_ylim(0, 1); axF.axis("off")
    H_, W_ = st.shape[1:]; asp = H_ / W_; fw = 0.74; fh = fw * asp * (axF.bbox.width / axF.bbox.height)
    lo, hi = np.percentile(st[0], 1), np.percentile(st[0], 99.5)
    n_show = 4
    for k in range(n_show)[::-1]:
        x0, y0 = 0.02 + 0.075 * k, 0.97 - fh - 0.075 * k
        axF.imshow(st[k * 3], cmap="gray", vmin=lo, vmax=hi, extent=(x0, x0 + fw, y0, y0 + fh), zorder=10 - k)
        axF.add_patch(Rectangle((x0, y0), fw, fh, fc="none", ec="white", lw=0.5, zorder=10.5 - k))
    axF.set_aspect("auto")
    axF.set_title(f"{T} frames, 250 fps", fontsize=8, pad=2, loc="right")
    axF.text(-0.3, 1.02, "b", transform=axF.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    axM = fig.add_subplot(gB[1])
    axM.imshow(mip, cmap="magma", vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.7), interpolation="nearest")
    axM.set_xticks([]); axM.set_yticks([])
    for s in axM.spines.values():
        s.set_color(ACCENT); s.set_linewidth(1.0)
    axM.set_title("max-projection", fontsize=8, pad=2)
    scalebar(axM, um_rt, 200, "200 µm", x=0.08, y=0.08)
    # c: acquisition order
    axC = fig.add_subplot(gs[2])
    vis = np.ma.masked_equal(mos, 0)
    axC.imshow(vis, cmap="gray", vmin=np.percentile(mos[mos > 0], 1), vmax=np.percentile(mos, 99.7), alpha=0.55, interpolation="antialiased")
    ok = np.isfinite(tmin); norm = Normalize(0, np.nanmax(tmin)); cmap = plt.cm.viridis
    seg = np.stack([cen[:-1], cen[1:]], axis=1); sv = (tmin[:-1] + tmin[1:]) / 2
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=1.0, zorder=2); lc.set_array(np.where(np.isfinite(sv), sv, 0)); axC.add_collection(lc)
    axC.scatter(cen[ok, 0], cen[ok, 1], c=tmin[ok], cmap=cmap, norm=norm, s=9, edgecolor=INK, linewidth=0.3, zorder=3)
    axC.scatter(*cen[0], s=45, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
    axC.annotate("start", cen[0], xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    i = int(np.flatnonzero(tids == args.tile)[0]); box = W_ * 2 * 0.95       # realtime px × ds → mosaic px
    hb = H_ * 2 * 0.95
    axC.add_patch(Rectangle((cen[i, 0] - box / 2, cen[i, 1] - hb / 2), box, hb, fill=False, ec=ACCENT, lw=1.1, zorder=5))
    axC.set_xlim(0, mos.shape[1]); axC.set_ylim(mos.shape[0], 0); axC.set_aspect("equal"); axC.axis("off")
    scalebar(axC, PX_UM_RAW, 1000, "1 mm", color=INK, x=0.0, y=0.03)
    axC.text(0.0, 1.02, "c", transform=axC.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
    pC = axC.get_position(); cax = fig.add_axes([pC.x1 + 0.01, pC.y0 + 0.15 * pC.height, 0.012, 0.7 * pC.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("acquisition time (min)", labelpad=2)
    cb.outline.set_linewidth(0.5)
    plos_style.save(fig, out / "fig1")


if __name__ == "__main__":
    main()
