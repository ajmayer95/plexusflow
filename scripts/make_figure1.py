"""Figure 1 — acquisition, per-tile filtering and stitching (PLOS, 7.5 in wide).

  a  imaging setup (figures/fig1/setup.png, rendered from Alex's setup.svg; BioRender elements — Created with
     BioRender.com); the yolk-sac plexus is recorded one overlapping field of view at a time
  b  one 250 fps movie per tile (stacks; tile 36 in front) → per-tile image used for registration: temporal max − median
     projection (tiles/maxmed) → background-filtered (minus a σ = 80 px Gaussian, percentile-normalised)
  c  stitching one tile: tile 15 in its own frame (top left) and placed by its stored affine A_15 over the rigid anchor
     tiles (yellow); arrows decompose A_15 relative to the anchors' rotation (green: stretch along rows, magenta:
     contraction along columns, yellow: shear, blue: rotation); stars = the same three vessel landmarks before/after
  d  the fused whole-organ mosaic with every tile footprint and the acquisition order (arrows coloured by tile index)

Inputs: stitch/positions.json, tiles/maxmed, videos_realtime, fuse/organ_fusion.tif.  Writes figures/fig1/fig1.{png,pdf,tif}.
Stitching panels ported from the 2026-09-02 session code (plexusflow-sandbox/recovered_fig2_2026-09-02/) and the
archived make_figure2.py panel F.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Polygon, Ellipse, FancyArrowPatch, Arc
from scipy.ndimage import gaussian_filter, affine_transform, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.config import StageConfig
from plexusflow.fuse.feather import _load_positions_and_affines, compute_world_bbox, feather_kernel_from_tile, _load_tile, _warp_into_canvas
import plos_style

PX_UM_RAW = 1.7
ACCENT = "#e8711a"
INK = "#1a1a1a"
T15, ANCHORS = 15, (7, 14, 21, 17, 18, 16, 19, 20, 22)
CORNERS = lambda H, W: np.array([[0, 0, 1], [H, 0, 1], [H, W, 1], [0, W, 1], [0, 0, 1]], float).T


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



def hp_norm(im, sigma=80.0):
    """The registration image: background-filtered (minus a broad Gaussian), percentile-normalised to [0, 1]."""
    x = im - gaussian_filter(im, sigma); lo, hi = np.percentile(x, (1, 95)); hi = max(hi, lo + 1e-9)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def feather_render(tiles, affs, render, bbox_tiles, alpha=0.5):
    y0, y1, x0, x1 = compute_world_bbox({t: tiles[t].shape for t in bbox_tiles}, {t: affs[t] for t in bbox_tiles}, pad=15)
    H, W = y1 - y0, x1 - x0; num = np.zeros((H, W)); den = np.zeros((H, W))
    for t in render:
        w = _warp_into_canvas(feather_kernel_from_tile(tiles[t].shape, alpha=alpha), affs[t], (H, W), y0, x0)
        num += _warp_into_canvas(hp_norm(tiles[t]), affs[t], (H, W), y0, x0) * w; den += w
    out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0); lo, hi = np.percentile(out[out > 0], (1, 95))
    return np.clip((out - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.9, y0, x0, H, W


def landmarks(img, n=3, sep=0.3):
    """n bright, well-separated vessel points (local maxima of the smoothed image), away from the tile edge."""
    sm = gaussian_filter(img, 3); H, W = sm.shape; pk = (sm == maximum_filter(sm, 25)) & (sm > np.percentile(sm, 97))
    pk[:int(0.15 * H)] = pk[-int(0.15 * H):] = False; pk[:, :int(0.15 * W)] = pk[:, -int(0.15 * W):] = False
    ys, xs = np.nonzero(pk); order = np.argsort(-sm[ys, xs]); pts = []
    for i in order:
        p = np.array([ys[i], xs[i]], float)
        if all(np.hypot(*(p - q)) > sep * min(H, W) for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    return np.array(pts)


def stitch_panel(ax, positions, tile_dir, ds=2):
    _, affines = _load_positions_and_affines(positions, ds)
    tiles = {t: _load_tile(tile_dir / f"vid{t}_maxmed.tif", ds) for t in sorted(set(ANCHORS) | {T15})}
    A15 = affines[T15]; H15, W15 = tiles[T15].shape
    mosaic, y0, x0, H, W = feather_render(tiles, affines, set(ANCHORS) | {T15}, {7, 14, 21, 17, 18, 16, 19, 20, T15})
    ax.imshow(mosaic, cmap="gray", interpolation="antialiased")
    # tile 15 in its own frame, rotated only by the anchors' rotation (so what remains is scale + shear + Δθ)
    R = affines[7][:2, :2].copy(); src = hp_norm(tiles[T15])
    rc = CORNERS(H15, W15)[:2, :4].T @ R.T; lo_, hi_ = rc.min(0), rc.max(0); rH, rW = (np.ceil(hi_ - lo_) + 4).astype(int)
    Ar = np.eye(3); Ar[:2, :2] = R; Ar[:2, 2] = -lo_ + 2; inv = np.linalg.inv(Ar)
    rot = affine_transform(src, inv[:2, :2], offset=inv[:2, 2], output_shape=(rH, rW), order=1, cval=0.0)
    py, px = 0.035 * H, 0.012 * W
    ax.imshow(np.clip(rot, 0, 1) ** 0.9, cmap="gray", extent=(px, px + rW, py + rH, py), interpolation="antialiased", zorder=3)
    ol = rc - lo_ + 2 + [py, px]; ax.plot(*np.vstack([ol, ol[:1]])[:, ::-1].T, color="0.6", lw=0.8, zorder=4)
    src_c = ol.mean(0); dst_c = (A15 @ [H15 / 2, W15 / 2, 1])[:2] - [y0, x0]
    # affine decomposition relative to the anchors
    L = A15[:2, :2]; sx = np.hypot(L[0, 1], L[1, 1]); sy = np.hypot(L[0, 0], L[1, 0])
    sh = np.degrees(np.arcsin(np.clip((L[0, 0] * L[0, 1] + L[1, 0] * L[1, 1]) / max(sx * sy, 1e-12), -1, 1)))
    dth = np.degrees(np.arctan2(L[1, 0], L[0, 0])) - np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    ux = np.array([R[0, 1], R[1, 1]]); ux /= np.linalg.norm(ux); uy = np.array([R[0, 0], R[1, 0]]); uy /= np.linalg.norm(uy)
    eff = np.array([abs(sx - 1), abs(sy - 1), abs(np.sin(np.radians(sh))), abs(np.sin(np.radians(dth)))]); k = 0.30 * W15 / eff.max()
    arrow = lambda a, b, c, lw=1.4, ms=8: ax.add_patch(FancyArrowPatch(a[::-1], b[::-1], arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, zorder=9))
    for sg in (1, -1):                                             # rows: stretch (out) if sy > 1
        a, b = src_c + sg * 3 * uy, src_c + sg * k * eff[1] * uy
        arrow(a, b, "#3cb44b") if sy > 1 else arrow(b, a, "#3cb44b")
        out_, in_ = src_c + sg * max(k * eff[0], 32) * ux, src_c + sg * 6 * ux      # columns: contraction (in) if sx < 1
        arrow(out_, in_, "#c94a91") if sx < 1 else arrow(in_, out_, "#c94a91")
    sgn = np.sign(sh) or 1; top = src_c - H15 / 2 * uy; bot = src_c + H15 / 2 * uy
    arrow(top, top + k * eff[2] * sgn * ux, "#f0ad00", 1.2, 7); arrow(bot, bot - k * eff[2] * sgn * ux, "#f0ad00", 1.2, 7)
    r_arc = 0.28 * W15; deg = np.clip(np.degrees(k * eff[3] / r_arc) * 3, 15, 80) * np.sign(dth)
    ax.add_patch(Arc((src_c[1], src_c[0]), 2 * r_arc, 2 * r_arc, theta1=min(-30, -30 + deg), theta2=max(-30, -30 + deg), color="#0d5aa7", lw=1.2, zorder=10))
    # anchors + tile numbers, final tile-15 outline, source → placed arrow
    for t in ANCHORS:
        Hn, Wn = tiles[t].shape; w = affines[t] @ CORNERS(Hn, Wn)
        ax.plot(w[1] - x0, w[0] - y0, color="#ffd23f", lw=0.6, alpha=0.85, zorder=5)
        ax.text(w[1, :-1].mean() - x0, w[0, :-1].mean() - y0, str(t), color="white", fontsize=8, fontweight="bold", ha="center", va="center",
                bbox=dict(fc="black", ec="#ffd23f", lw=0.5, alpha=0.75, boxstyle="round,pad=0.2"), zorder=6)
    w15 = A15 @ CORNERS(H15, W15); ax.plot(w15[1] - x0, w15[0] - y0, color="#e6194B", lw=1.2, zorder=7)
    ax.text(px + 10, py + 12, "15", color="white", fontsize=8, fontweight="bold", ha="left", va="top",
            bbox=dict(fc="#e6194B", ec="white", lw=0.5, boxstyle="round,pad=0.25"), zorder=12)
    ax.add_patch(FancyArrowPatch(src_c[::-1], dst_c[::-1], arrowstyle="-|>", mutation_scale=11, color="#e6194B", lw=1.3,
                                 connectionstyle="arc3,rad=-0.2", zorder=8))
    # the same three landmarks before (in tile 15's frame) and after (in the mosaic)
    lm = landmarks(src)
    for p in lm:
        q_src = Ar @ [p[0], p[1], 1]; q_dst = A15 @ [p[0], p[1], 1]
        ax.plot(q_src[1] + px, q_src[0] + py, "*", ms=7, color="#ff4d4d", mec="white", mew=0.4, zorder=11)
        ax.plot(q_dst[1] - x0, q_dst[0] - y0, "*", ms=7, color="#ff4d4d", mec="white", mew=0.4, zorder=11)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_visible(False)
    print(f"[fig1] tile 15 vs anchors: Δθ {dth:+.1f}°, s_rows {sy:.3f}, s_cols {sx:.3f}, shear {sh:+.1f}°")
    return dict(sx=sx, sy=sy, sh=sh, dth=dth)


def registration_panel(axs, positions, tile_dir, ds=2, pad=0.08):
    """Tile 15 against its already-placed neighbours, in their overlap: magenta = tile 15, green = neighbours (white where
    they agree).  Left: tile 15 placed rigidly (neighbours' rotation, same centre, no scaling/shear); right: the stored
    affine from the manual registration.  Same crop in both."""
    _, affines = _load_positions_and_affines(positions, ds)
    nb = (14, 16, 19, 20, 21)
    tiles = {t: _load_tile(tile_dir / f"vid{t}_maxmed.tif", ds) for t in nb + (T15,)}
    norm01 = lambda im: np.clip((im - np.percentile(im, 1)) / max(np.percentile(im, 99.5) - np.percentile(im, 1), 1e-9), 0, 1)
    A15 = affines[T15]; H15, W15 = tiles[T15].shape; c = np.array([H15 / 2, W15 / 2])
    R = affines[7][:2, :2]; Ar = np.eye(3); Ar[:2, :2] = R; Ar[:2, 2] = A15[:2, :2] @ c + A15[:2, 2] - R @ c
    y0, y1, x0, x1 = compute_world_bbox({t: tiles[t].shape for t in nb + (T15,)}, {**{t: affines[t] for t in nb}, T15: A15}, pad=15)
    Hc, Wc = y1 - y0, x1 - x0
    num = np.zeros((Hc, Wc)); den = np.zeros((Hc, Wc))
    for t in nb:
        w = _warp_into_canvas(feather_kernel_from_tile(tiles[t].shape, 0.5), affines[t], (Hc, Wc), y0, x0)
        num += _warp_into_canvas(norm01(tiles[t]), affines[t], (Hc, Wc), y0, x0) * w; den += w
    G = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    # crop: the part of tile 15's footprint that the neighbours also cover
    cov15 = _warp_into_canvas(np.ones_like(tiles[T15]), A15, (Hc, Wc), y0, x0) > 0.5
    ov = (cov15 & np.isfinite(G)).astype(float)
    # square window fully inside the overlap, as far from tile 15's centre as possible (the rigid error grows outward)
    win = int(0.42 * min(H15, W15)); I = np.pad(ov.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    frac = (I[win:, win:] - I[:-win, win:] - I[win:, :-win] + I[:-win, :-win]) / win ** 2
    ctr = (A15 @ [H15 / 2, W15 / 2, 1])[:2] - [y0, x0]
    rr, cc_ = np.mgrid[0:frac.shape[0], 0:frac.shape[1]]
    score = np.where(frac > 0.97, np.hypot(rr + win / 2 - ctr[0], cc_ + win / 2 - ctr[1]), -1)
    r_, c_ = np.unravel_index(np.argmax(score), score.shape); box = (r_, r_ + win, c_, c_ + win)
    for ax, A, lab in ((axs[0], Ar, "rigid"), (axs[1], A15, "affine")):
        M = _warp_into_canvas(norm01(tiles[T15]), A, (Hc, Wc), y0, x0)
        cov = _warp_into_canvas(np.ones_like(tiles[T15]), A, (Hc, Wc), y0, x0) > 0.5
        g = np.nan_to_num(G); m = np.where(cov, M, 0)
        rgb = np.dstack([m, g, m])[box[0]:box[1], box[2]:box[3]]
        ax.imshow(np.clip(rgb * 1.15, 0, 1), interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.5"); sp.set_linewidth(0.6)
        ax.set_title(lab, fontsize=8, pad=2)
    L = 100 / (PX_UM_RAW * ds); n = box[1] - box[0]
    axs[0].plot([0.07 * n, 0.07 * n + L], [0.93 * n] * 2, color="white", lw=2, solid_capstyle="butt")
    axs[0].text(0.07 * n + L / 2, 0.90 * n, "100 µm", color="white", ha="center", va="bottom", fontsize=8)


def order_panel(fig, ax, mos, positions):
    """Fused mosaic, all tile footprints, acquisition order as arrows coloured by tile index (archived make_figure2 F)."""
    from net3.edit.video_overlay import load_tile_positions
    pos = load_tile_positions(str(positions))
    A = {t: np.asarray(e["affine_matrix"], float) for t, e in pos.items() if e.get("affine_matrix") is not None}
    tids = sorted(A); C = np.array([352.0, 320.0])                 # full-res tile centre (H/2, W/2)
    ax.imshow(np.ma.masked_equal(mos, 0), cmap="gray", vmin=np.percentile(mos[mos > 0], 1), vmax=np.percentile(mos, 99.7), interpolation="antialiased")
    for t in tids:
        q = (np.array([[0, 0], [0, 640], [704, 640], [704, 0], [0, 0]], float) @ A[t][:2, :2].T + A[t][:2, 2])[:, [1, 0]]
        ax.plot(*q.T, color="#e8c85a", lw=0.3, alpha=0.35)
    pts = np.array([(A[t][:2, :2] @ C + A[t][:2, 2])[[1, 0]] for t in tids]); norm = Normalize(tids[0], tids[-1]); cmap = plt.cm.plasma
    lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), colors=cmap(norm(np.array(tids[:-1]))), linewidths=1.1, zorder=4)
    ax.add_collection(lc)
    for i in range(2, len(tids) - 1, 6):                        # a few direction arrowheads only
        ax.add_patch(FancyArrowPatch(pts[i], (pts[i] + pts[i + 1]) / 2, arrowstyle="-|>", mutation_scale=8, lw=0, color=cmap(norm(tids[i])), zorder=5))
    ax.scatter(*pts.T, c=tids, cmap=cmap, norm=norm, s=5, zorder=5, edgecolor="white", linewidth=0.25)
    ax.scatter(*pts[0], s=40, facecolor="none", edgecolor="white", linewidth=1.0, zorder=6)
    ax.set_xlim(0, mos.shape[1]); ax.set_ylim(mos.shape[0], 0); ax.set_aspect("equal"); ax.axis("off")
    return norm, cmap, len(tids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=36, help="front tile of the stacks in b")
    args = ap.parse_args()
    plos_style.apply()
    cfg = StageConfig.load(_paths.STAGE); P = _paths.STAGE; out = _paths.FIGURES / "fig1"
    mos = tifffile.imread(str(P / "fuse" / "organ_fusion.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos
    vids = {t: next(Path(cfg.videos_realtime_dir).glob(f"*loc{t}_realtime.tif")) for t in (args.tile, 21, 7)}
    st = tifffile.imread(str(vids[args.tile])).astype(np.float32); T = st.shape[0]
    mm = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{args.tile}_maxmed.tif")).astype(float); mm = mm.mean(-1) if mm.ndim == 3 else mm

    fig = plt.figure(figsize=(7.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.06, left=0.01, right=0.9, top=0.965, bottom=0.02)
    gL = gs[0].subgridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.14)
    letter = lambda ax, l, dx=0.0, dy=1.01: ax.text(dx, dy, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: setup → one tile's movie → projection
    gA = gL[0].subgridspec(1, 2, width_ratios=[0.42, 1.0], wspace=0.0)
    axA = fig.add_subplot(gA[0]); setup = out / "setup.png"
    if setup.exists():
        axA.imshow(plt.imread(str(setup)), interpolation="antialiased"); axA.axis("off")
    else:
        draw_schematic(axA)
    letter(axA, "a")
    axB = fig.add_subplot(gA[1]); axB.axis("off")
    H_, W_ = st.shape[1:]; fw = 1.0; fh = fw * H_ / W_; n_show, dx, dy = 6, 0.09, 0.08
    lo, hi = np.percentile(st[:20], 1), np.percentile(st[:20], 99.6)
    for k in range(n_show)[::-1]:
        x0, y0 = k * dx, k * dy; z = 2 * (n_show - k)
        axB.add_patch(Rectangle((x0 + 0.02, y0 - 0.02), fw, fh, fc="0.2", ec="none", alpha=0.16, zorder=z))
        axB.imshow(st[k * 8], cmap="gray", vmin=lo, vmax=hi, extent=(x0, x0 + fw, y0, y0 + fh), zorder=z + 1, interpolation="antialiased")
        axB.add_patch(Rectangle((x0, y0), fw, fh, fc="none", ec="white", lw=0.6, zorder=z + 1.5))
    t0 = np.array([fw + 0.04, -0.03]); t1 = t0 + [(n_show - 1) * dx + 0.05, (n_show - 1) * dy + 0.05]
    axB.annotate("", xy=t1, xytext=t0, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=7))
    axB.text(t0[0] + 0.06, t0[1] - 0.02, "250 fps", ha="left", va="top", fontsize=8)
    xs = fw + (n_show - 1) * dx + 0.42; fh2 = fw * mm.shape[0] / mm.shape[1]; y2 = (fh + (n_show - 1) * dy - fh2) / 2
    axB.annotate("", xy=(xs - 0.04, y2 + fh2 / 2), xytext=(xs - 0.28, y2 + fh2 / 2), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8))
    lo, hi = np.percentile(mm, (1, 99.5)); axB.imshow(mm, cmap="gray", vmin=lo, vmax=hi, extent=(xs, xs + fw, y2, y2 + fh2), interpolation="antialiased")
    axB.add_patch(Rectangle((xs, y2), fw, fh2, fc="none", ec="0.5", lw=0.6))
    axB.text(xs + fw / 2, y2 + fh2 + 0.05, "Projection", ha="center", va="bottom", fontsize=8)
    L = 200 / PX_UM_RAW / mm.shape[1] * fw
    axB.plot([xs + fw - 0.2 - L, xs + fw - 0.2], [y2 + 0.08, y2 + 0.08], color="white", lw=2, solid_capstyle="butt")
    axB.text(xs + fw - 0.2 - L / 2, y2 + 0.11, "200 µm", color="white", ha="center", va="bottom", fontsize=8)
    axB.set_xlim(-0.03, xs + fw + 0.03); axB.set_ylim(-0.25, fh + (n_show - 1) * dy + 0.12); axB.set_aspect("equal")

    # b: registration of one tile against its neighbours (before / after)
    gR = gL[1].subgridspec(1, 2, wspace=0.05)
    axR = [fig.add_subplot(gR[0]), fig.add_subplot(gR[1])]
    registration_panel(axR, P / "stitch" / "positions.json", P / "tiles" / "maxmed")
    letter(axR[0], "b", 0.0, 1.07)

    # c: whole-organ mosaic + acquisition order
    axD = fig.add_subplot(gs[1]); norm, cmap, nt = order_panel(fig, axD, mos, P / "stitch" / "positions.json")
    x0_, y0_ = mos.shape[1] * 0.03, mos.shape[0] * 0.975; Lm = 1000 / PX_UM_RAW
    axD.plot([x0_, x0_ + Lm], [y0_, y0_], color=INK, lw=2, solid_capstyle="butt")
    axD.annotate("1 mm", (x0_ + Lm / 2, y0_), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.canvas.draw(); pD = axD.get_position()
    fig.text(pD.x0, pD.y1 - 0.01, "c", fontsize=12, fontweight="bold", va="bottom", ha="left")
    cax = fig.add_axes([pD.x1 + 0.012, pD.y0 + 0.25 * pD.height, 0.012, 0.5 * pD.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("Acquisition order", labelpad=2); cb.outline.set_linewidth(0.5)
    plos_style.save(fig, out / "fig1")


if __name__ == "__main__":
    main()
