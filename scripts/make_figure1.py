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
from scipy.ndimage import gaussian_filter, affine_transform, maximum_filter, laplace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.config import StageConfig
from plexusflow.fuse.feather import _load_positions_and_affines, compute_world_bbox, feather_kernel_from_tile, _load_tile, _warp_into_canvas
import plos_style

PX_UM_RAW = 1.7
ACCENT = "#e8711a"
INK = "#1a1a1a"
import matplotlib.patheffects as _pe
HALO = [_pe.withStroke(linewidth=1.6, foreground="black")]   # legible white scale bars on bright tissue
N_ZOOM = 3
ACQ_TILES = (1, 2, 3)              # panel a: the first three recorded tiles
FORCE_SEAMS = ((18, 19),)           # tile 31: one zoom where 31, 18 and 19 all overlap
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


def best_rigid_shift(positions, tile_dir, tile, nb, ds=2):
    """Rigid baseline for the registration panel: the tile's manual rotation, the neighbours' common scale, translation
    chosen to maximise overlap NCC with the (fixed) neighbours.  Returns (shift in full-res px, NCC rigid, NCC manual affine)."""
    from scipy.optimize import minimize
    _, A = _load_positions_and_affines(positions, ds)
    TL = {t: _load_tile(tile_dir / f"vid{t}_maxmed.tif", ds) for t in tuple(nb) + (tile,)}
    hp = lambda im: (lambda x: (x - x.mean()) / (x.std() + 1e-9))(gaussian_filter(im - gaussian_filter(im, 20), 1.5))
    y0, y1, x0, x1 = compute_world_bbox({u: TL[u].shape for u in TL}, {u: A[u] for u in TL}, pad=40); sh = (y1 - y0, x1 - x0)
    num = np.zeros(sh); den = np.zeros(sh)
    for u in nb:
        num += _warp_into_canvas(hp(TL[u]), A[u], sh, y0, x0); den += _warp_into_canvas(np.ones_like(TL[u]), A[u], sh, y0, x0)
    G = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan); Ht = hp(TL[tile]); one = np.ones_like(TL[tile])
    def ncc(Aa):
        M = _warp_into_canvas(Ht, Aa, sh, y0, x0); o = (_warp_into_canvas(one, Aa, sh, y0, x0) > 0.99) & np.isfinite(G)
        if o.sum() < 2000:
            return -1.0
        a, b = M[o] - M[o].mean(), G[o] - G[o].mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    H, W = TL[tile].shape; c = np.array([H / 2, W / 2]); U, _, Vt = np.linalg.svd(A[tile][:2, :2]); R0 = U @ Vt
    s_nb = float(np.median([np.sqrt(abs(np.linalg.det(A[u][:2, :2]))) for u in nb])); L = s_nb * R0
    ctr0 = A[tile][:2, :2] @ c + A[tile][:2, 2]
    def rigid(d):
        Aa = np.eye(3); Aa[:2, :2] = L; Aa[:2, 2] = ctr0 + d - L @ c; return Aa
    best = min((minimize(lambda d: -ncc(rigid(d)), np.array(q, float), method="Powell", options=dict(xtol=0.2, ftol=1e-4, maxfev=200))
                for q in ((0, 0), (-10, 0), (10, 0), (0, -10), (0, 10))), key=lambda r: r.fun)
    return best.x * ds, -best.fun, ncc(A[tile])


def registration_panel(axF, axZ, positions, tile_dir, tile=T15, ds=1, zoom_um=280.0, n_lm=3, zoom_with=None, nb=None):
    """Tile 15 and its neighbours.  axF: full footprints at the stored placement (magenta = tile 15, green = neighbours,
    white = agreement), tile outlines + numbers, zoom box on the 15 | 19–20 overlap.  axZ = (rigid, affine): that zoom with
    tile placed by the NCC-optimal rigid (manual rotation, neighbours' scale, best translation) vs by the stored affine; stars = vessel junctions
    of the neighbours (illustrative — the manual registration stored no point correspondences)."""
    _, affines = _load_positions_and_affines(positions, ds)
    T15 = tile                                                     # (name kept from the tile-15 version)
    cen = {t: (A @ [176.0 * 2 / ds, 160.0 * 2 / ds, 1])[:2] for t, A in affines.items()}
    nb = tuple(nb) if nb else tuple(t for t in sorted(affines, key=lambda u: np.hypot(*(cen[u] - cen[tile]))) if t != tile
               and np.hypot(*(cen[t] - cen[tile])) < 0.85 * 704 / ds)[:5]
    tiles = {t: _load_tile(tile_dir / f"vid{t}_maxmed.tif", ds) for t in nb + (T15,)}
    def norm01(im):
        x = im - gaussian_filter(im, 40.0 / ds); lo, hi = np.percentile(x, (2, 99.7))
        return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    A15 = affines[T15]; H15, W15 = tiles[T15].shape; c = np.array([H15 / 2, W15 / 2])
    U, S, Vt = np.linalg.svd(A15[:2, :2]); s_nb = np.median([np.sqrt(abs(np.linalg.det(affines[u][:2, :2]))) for u in nb])
    shift, n_r, n_a = best_rigid_shift(positions, tile_dir, tile, nb)        # NCC-optimal rigid (translation; manual rotation)
    print(f"[fig1] tile {tile}: optimal rigid NCC {n_r:.3f} vs manual affine {n_a:.3f} (shift {shift.round(1)} px)")
    Rp = (U @ Vt) * s_nb; Ar = np.eye(3); Ar[:2, :2] = Rp; Ar[:2, 2] = A15[:2, :2] @ c + A15[:2, 2] + shift / ds - Rp @ c
    y0, y1, x0, x1 = compute_world_bbox({t: tiles[t].shape for t in nb + (T15,)}, {**{t: affines[t] for t in nb}, T15: A15}, pad=10)
    sh = (y1 - y0, x1 - x0); num = np.zeros(sh); den = np.zeros(sh); cov = {}
    for t in nb:
        cov[t] = _warp_into_canvas(np.ones_like(tiles[t]), affines[t], sh, y0, x0) > 0.5
        w = _warp_into_canvas(feather_kernel_from_tile(tiles[t].shape, 0.5), affines[t], sh, y0, x0)
        num += _warp_into_canvas(norm01(tiles[t]), affines[t], sh, y0, x0) * w; den += w
    G = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    m15 = {k: (_warp_into_canvas(norm01(tiles[T15]), A, sh, y0, x0), _warp_into_canvas(np.ones_like(tiles[T15]), A, sh, y0, x0) > 0.5)
           for k, A in (("rigid", Ar), ("affine", A15))}
    rgb = lambda M, cv: np.clip(np.dstack([M * cv, G, M * cv]), 0, 1)
    # zooms: one per requested neighbour seam; window inside that overlap where the affine raises local agreement most
    win = int(zoom_um / (PX_UM_RAW * ds))
    def pick(nbr):
        ov = (m15["affine"][1] & m15["rigid"][1] & (np.all([cov[t] for t in nbr], axis=0) if isinstance(nbr, tuple) else cov[nbr] if nbr in cov else np.any([cov[t] for t in nb], axis=0))).astype(float)
        I = np.pad(ov.cumsum(0).cumsum(1), ((1, 0), (1, 0))); frac = (I[win:, win:] - I[:-win, win:] - I[win:, :-win] + I[:-win, :-win]) / win ** 2
        thr = 0.98 if (frac >= 0.98).any() else 0.9 * frac.max(); cand = []
        for r_ in range(0, frac.shape[0], max(win // 8, 1)):
            for c_ in range(0, frac.shape[1], max(win // 8, 1)):
                if frac[r_, c_] < thr:
                    continue
                g = G[r_:r_ + win, c_:c_ + win]; a_ = m15["affine"][0][r_:r_ + win, c_:c_ + win]
                ra = np.corrcoef(a_.ravel(), g.ravel())[0, 1]
                rg = np.corrcoef(m15["rigid"][0][r_:r_ + win, c_:c_ + win].ravel(), g.ravel())[0, 1]
                sharp = min(np.mean(laplace(gaussian_filter(g, 1.0)) ** 2), np.mean(laplace(gaussian_filter(a_, 1.0)) ** 2))
                if np.isfinite(ra) and np.isfinite(rg):
                    cand.append((sharp, (ra - rg) + 0.5 * ra, (r_, r_ + win, c_, c_ + win), (rg, ra)))
        sh_cut = np.percentile([c_[0] for c_ in cand], 70)            # only the sharpest 30 % of windows
        return [c_ for c_ in cand if c_[0] >= sh_cut]
    if not zoom_with or list(zoom_with) == [0]:
        # automatic: the two best non-overlapping sharp windows anywhere in the overlaps, ranked by rigid → affine gain
        Ainv = np.linalg.inv(A15)
        def disp(zb):                                   # rigid vs affine position of the window centre's tile-31 point (canvas px)
            q = Ainv @ [(zb[0] + zb[1]) / 2 + y0, (zb[2] + zb[3]) / 2 + x0, 1]
            return float(np.hypot(*((Ar @ q)[:2] - (A15 @ q)[:2])))
        cand = sorted([c_ for n in nb for c_ in pick(n) if c_[3][1] > 0.45], key=lambda c_: -disp(c_[2])); chosen = []
        for c_ in cand:
            if all(max(abs(c_[2][0] - z[2][0]), abs(c_[2][2] - z[2][2])) >= 1.5 * win for z in chosen):
                chosen.append(c_)
            if len(chosen) == N_ZOOM - len(FORCE_SEAMS):
                break
        for n in FORCE_SEAMS:                          # one zoom on a required seam: biggest rigid → affine gain there
            pool = [c_ for c_ in pick(n) if all(max(abs(c_[2][0] - z[2][0]), abs(c_[2][2] - z[2][2])) >= win for z in chosen)]
            wa_ = (A15 @ CORNERS(H15, W15))[:2, :4] - np.array([[y0], [x0]])
            tr = wa_[:, np.argmax(wa_[1] - wa_[0])]                          # tile's top-right corner on the canvas
            near = lambda c_: np.hypot((c_[2][0] + c_[2][1]) / 2 - tr[0], (c_[2][2] + c_[2][3]) / 2 - tr[1])
            chosen.append(min([c_ for c_ in pool if c_[3][1] > 0.4] or pool, key=near))
    else:
        chosen = [max(pick(n), key=lambda c_: c_[1]) for n in zoom_with]
    for c_ in chosen:
        print(f"[fig1] zoom: local r rigid {c_[3][0]:.2f} → affine {c_[3][1]:.2f}" + (f", offset {disp(c_[2]) * ds * PX_UM_RAW:.0f} µm" if 'disp' in dir() else ""))
    zbs = [c_[2] for c_ in chosen]
    # full view (stored placement)
    axF.imshow(rgb(*m15["affine"]), interpolation="antialiased")
    for t in nb + (T15,):
        w = (A15 if t == T15 else affines[t]) @ CORNERS(*tiles[t].shape)
        axF.plot(w[1] - x0, w[0] - y0, color="#ff66ff" if t == T15 else "#7dff7d", lw=0.6, alpha=0.9)
        cw = w[:2, :-1].mean(1); c31 = (A15 @ [H15 / 2, W15 / 2, 1])[:2]
        pos_ = cw if t == T15 else cw + 0.45 * (cw - c31) / max(np.hypot(*(cw - c31)), 1) * min(tiles[t].shape)
        axF.text(pos_[1] - x0, pos_[0] - y0, str(t), color="white", fontsize=8, fontweight="bold", ha="center", va="center",
                 bbox=dict(fc="black", ec="none", alpha=0.55, boxstyle="round,pad=0.15"), zorder=8)
    wr, wa = Ar @ CORNERS(H15, W15), A15 @ CORNERS(H15, W15)
    # best-rigid footprint dashed, affine footprint (solid magenta outline above)
    axF.plot(wr[1] - x0, wr[0] - y0, color="white", lw=0.9, ls=(0, (3, 2)), zorder=6)
    disp = np.hypot(*(wa[:2, :4] - wr[:2, :4])) * ds * PX_UM_RAW
    print(f"[fig1] tile {tile} corner displacement rigid → affine: {np.round(disp, 0)} µm")
    ZC = "#ffd23f"                                                 # zoom boxes: yellow, numbered inside their corner
    for k, zb in enumerate(zbs):
        axF.add_patch(Rectangle((zb[2], zb[0]), win, win, fill=False, ec=ZC, lw=1.1, zorder=9))
        if len(zbs) > 1:
            axF.text(zb[2] + 0.08 * win, zb[0] + 0.1 * win, str(k + 1), color="black", fontsize=8, fontweight="bold", ha="left", va="top",
                     bbox=dict(fc=ZC, ec="none", pad=0.6), zorder=10)
    axF.set_xticks([]); axF.set_yticks([]); [sp.set_visible(False) for sp in axF.spines.values()]
    for k, zb in enumerate(zbs):
        sub = gaussian_filter(G[zb[0]:zb[1], zb[2]:zb[3]], 2.5); pk = (sub == maximum_filter(sub, int(win * 0.18))) & (sub > np.percentile(sub, 96))
        m = int(win * 0.15); pk[:m] = pk[-m:] = False; pk[:, :m] = pk[:, -m:] = False
        ys, xs = np.nonzero(pk); o = np.argsort(-sub[ys, xs]); lm = []
        for k_ in o:
            q = np.array([ys[k_], xs[k_]], float)
            if all(np.hypot(*(q - z)) > 0.3 * win for z in lm):
                lm.append(q)
            if len(lm) == n_lm:
                break
        for row, key in enumerate(("rigid", "affine")):
            ax = axZ[row][k]
            ax.imshow(rgb(*m15[key])[zb[0]:zb[1], zb[2]:zb[3]], interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(ZC); sp.set_linewidth(1.1)
            if k == 0:
                ax.text(-0.06, 0.5, key, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8)
            if row == 0 and len(zbs) > 1:
                ax.text(0.05, 0.95, str(k + 1), transform=ax.transAxes, color="black", fontsize=8, fontweight="bold", ha="left", va="top",
                        bbox=dict(fc=ZC, ec="none", pad=0.6))
    L = 100 / (PX_UM_RAW * ds); axs_ = axZ[1][-1]; xe = win * 0.94
    axs_.plot([xe - L, xe], [win * 0.92] * 2, color="white", lw=2, solid_capstyle="butt", path_effects=HALO)
    axs_.text(xe, win * 0.89, "100 µm", color="white", ha="right", va="bottom", fontsize=8, path_effects=HALO)
    r = {}
    for key in ("rigid", "affine"):
        M, cv = m15[key]; oo = cv & (den > 0.5)
        r[key] = np.corrcoef(M[oo], G[oo])[0, 1]
    print(f"[fig1] tile {tile} vs {nb}: overlap r rigid {r['rigid']:.2f} → affine {r['affine']:.2f}")
    return zbs


def order_panel(fig, ax, mos, positions):
    """Fused mosaic, all tile footprints, acquisition order as arrows coloured by tile index (archived make_figure2 F)."""
    from net3.edit.video_overlay import load_tile_positions
    pos = load_tile_positions(str(positions))
    A = {t: np.asarray(e["affine_matrix"], float) for t, e in pos.items() if e.get("affine_matrix") is not None}
    tids = sorted(A); C = np.array([352.0, 320.0])                 # full-res tile centre (H/2, W/2)
    from scipy import ndimage as _ndi
    outside = ~_ndi.binary_fill_holes(mos > 0)                     # mask only the exterior
    # display only: saturated (255) blobs and zero-valued holes inside the organ are filled by a masked Gaussian average of the
    # surrounding tissue (normalised convolution, σ = 6 px) — stated in the caption
    bad = ~outside & ((mos >= mos.max()) | (mos <= 0)); bad = _ndi.binary_dilation(bad, iterations=1) & ~outside
    good = (~bad & ~outside).astype(float); num = gaussian_filter(mos * good, 6.0); den = gaussian_filter(good, 6.0)
    disp = np.where(bad, num / np.maximum(den, 1e-6), mos)
    print(f"[fig1] mosaic display fill: {100 * bad.sum() / (~outside).sum():.1f}% of organ pixels (saturated or empty)")
    ok = ~outside & ~bad
    ax.imshow(np.ma.masked_where(outside, disp), cmap="gray", vmin=np.percentile(disp[ok], 1), vmax=np.percentile(disp[ok], 99.5),
              interpolation="antialiased")
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
    ap.add_argument("--tile", type=int, default=36, help="tile whose movie is shown in a")
    ap.add_argument("--reg-tile", type=int, default=31, help="tile shown registered against its neighbours in b")
    ap.add_argument("--reg-nb", type=int, nargs="+", default=[18, 19, 32, 33], help="its (fixed, rigid) neighbours")
    ap.add_argument("--zoom-with", type=int, nargs="+", default=[0], help="zoom on the overlap with these neighbours")
    ap.add_argument("--out-name", default="fig1")
    args = ap.parse_args()
    plos_style.apply()
    cfg = StageConfig.load(_paths.STAGE); P = _paths.STAGE; out = _paths.FIGURES / "fig1"
    mos = tifffile.imread(str(P / "fuse" / "organ_fusion.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos
    vids = {t: next(Path(cfg.videos_realtime_dir).glob(f"*loc{t}_realtime.tif")) for t in (args.tile, 21, 7)}
    st = tifffile.imread(str(vids[args.tile])).astype(np.float32); T = st.shape[0]
    mm = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{args.tile}_maxmed.tif")).astype(float); mm = mm.mean(-1) if mm.ndim == 3 else mm

    fig = plt.figure(figsize=(7.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.06, left=0.01, right=0.9, top=0.965, bottom=0.02)
    gL = gs[0].subgridspec(2, 1, height_ratios=[0.9, 1.0], hspace=0.08)
    letter = lambda ax, l, dx=0.0, dy=1.01: ax.text(dx, dy, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: setup → one tile's movie → projection
    gA = gL[0].subgridspec(1, 2, width_ratios=[0.34, 1.0], wspace=0.04)
    axA = fig.add_subplot(gA[0]); setup = out / "setup.png"
    if setup.exists():
        axA.imshow(plt.imread(str(setup)), interpolation="antialiased"); axA.axis("off")
    else:
        draw_schematic(axA)
    letter(axA, "a")
    axB = fig.add_subplot(gA[1]); axB.axis("off")
    # the first three tiles: each one 250 fps movie (stack, top) projected down to its max − median image (bottom)
    H_, W_ = st.shape[1:]; fw = 1.0; fh = fw * H_ / W_; n_show, dx, dy = 8, 0.04, 0.035; pitch = 1.42
    ytop = fh + 0.55; xmax = 0
    for j, t in enumerate(ACQ_TILES):
        s_ = tifffile.imread(str(next(Path(cfg.videos_realtime_dir).glob(f"*loc{t}_realtime.tif")))).astype(np.float32)
        p_ = tifffile.imread(str(P / "tiles" / "maxmed" / f"vid{t}_maxmed.tif")).astype(float); p_ = p_.mean(-1) if p_.ndim == 3 else p_
        # display: flat-field each frame by the smoothed temporal mean (removes vignetting / haze), then stretch
        flat = gaussian_filter(s_.mean(0), 25) + 1e-6
        fr = [gaussian_filter(s_[k * 5] / flat, 0.6) for k in range(n_show)]
        lo, hi = np.percentile(fr[0], 2), np.percentile(fr[0], 99.7); ox = j * pitch
        for k in range(n_show)[::-1]:
            x0, y0 = ox + k * dx, ytop + k * dy; z = 2 * (n_show - k)
            dim = 1.0 - 0.45 * k / (n_show - 1)                                         # frames further back are dimmer
            axB.imshow(dim * np.clip((fr[k] - lo) / (hi - lo), 0, 1) ** 0.85, cmap="gray", vmin=0, vmax=1, extent=(x0, x0 + fw, y0, y0 + fh),
                       zorder=z + 1, interpolation="antialiased")
            axB.add_patch(Rectangle((x0, y0), fw, fh, fc="none", ec="0.15" if k else INK, lw=0.35 if k else 0.7, zorder=z + 1.5))
        b0, b1 = (ox, ytop), (ox + (n_show - 1) * dx, ytop + (n_show - 1) * dy)
        for cx, cy in ((0, fh), (fw, fh), (fw, 0)):                                     # depth edges front → back corner
            axB.plot([b0[0] + cx, b1[0] + cx], [b0[1] + cy, b1[1] + cy], color=INK, lw=0.6, zorder=2 * n_show + 3)
        fh2 = fw * p_.shape[0] / p_.shape[1]; lo, hi = np.percentile(p_, (1, 99.5))
        axB.imshow(p_, cmap="gray", vmin=lo, vmax=hi, extent=(ox, ox + fw, 0, fh2), interpolation="antialiased")
        axB.add_patch(Rectangle((ox, 0), fw, fh2, fc="none", ec="0.5", lw=0.6))
        axB.annotate("", xy=(ox + fw / 2, fh2 + 0.04), xytext=(ox + fw / 2, ytop - 0.06),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8))
        xmax = ox + fw + (n_show - 1) * dx
    for q in range(3):                                            # "more tiles" ellipsis
        xd = (len(ACQ_TILES) - 1) * pitch + fw + (n_show - 1) * dx + 0.1 + q * 0.11
        axB.plot(xd, ytop + 0.5 * fh, "o", ms=3.2, color=INK, zorder=50)
        axB.plot(xd, 0.5 * fh2, "o", ms=3.2, color=INK, zorder=50)
    axB.text(-0.06, ytop + 0.5 * (fh + (n_show - 1) * dy), "Movie", rotation=90, ha="right", va="center", fontsize=8)
    axB.text(0.5 * (fw + (n_show - 1) * dx), ytop + fh + (n_show - 1) * dy + 0.05, "250 fps · ~3 s", ha="center", va="bottom", fontsize=8)
    axB.text(-0.06, fh2 / 2, "Projection", rotation=90, ha="right", va="center", fontsize=8)
    L = 500 / PX_UM_RAW / p_.shape[1] * fw; xr = (len(ACQ_TILES) - 1) * pitch + fw - 0.06
    axB.plot([xr - L, xr], [0.08, 0.08], color="white", lw=2, solid_capstyle="butt", path_effects=HALO)
    axB.text(xr, 0.11, "500 µm", color="white", ha="right", va="bottom", fontsize=8, path_effects=HALO)
    axB.set_xlim(-0.2, (len(ACQ_TILES) - 1) * pitch + fw + (n_show - 1) * dx + 0.45); axB.set_ylim(-0.05, ytop + fh + (n_show - 1) * dy + 0.22); axB.set_aspect("equal")

    # b: registration of one tile against its neighbours (before / after)
    gR = gL[1].subgridspec(1, 2, width_ratios=[0.95, 1.2], wspace=0.08)
    gZ = gR[1].subgridspec(4, N_ZOOM, height_ratios=[0.45, 1, 1, 0.45], hspace=0.04, wspace=0.04)
    axF = fig.add_subplot(gR[0]); axZ = [[fig.add_subplot(gZ[r + 1, c]) for c in range(N_ZOOM)] for r in range(2)]
    registration_panel(axF, axZ, P / "stitch" / "positions.json", P / "tiles" / "maxmed", tile=args.reg_tile, zoom_with=args.zoom_with, nb=args.reg_nb)
    letter(axF, "b", 0.0, 1.02)

    # c: whole-organ mosaic + acquisition order
    axD = fig.add_subplot(gs[1]); norm, cmap, nt = order_panel(fig, axD, mos, P / "stitch" / "positions.json")
    x0_, y0_ = mos.shape[1] * 0.03, mos.shape[0] * 0.975; Lm = 1000 / PX_UM_RAW
    axD.plot([x0_, x0_ + Lm], [y0_, y0_], color=INK, lw=2, solid_capstyle="butt")
    axD.annotate("1 mm", (x0_ + Lm / 2, y0_), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.canvas.draw(); pD = axD.get_position()
    fig.text(pD.x0, pD.y1 - 0.01, "c", fontsize=12, fontweight="bold", va="bottom", ha="left")
    cax = fig.add_axes([pD.x1 + 0.012, pD.y0 + 0.25 * pD.height, 0.012, 0.5 * pD.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("Acquisition order", labelpad=2); cb.outline.set_linewidth(0.5)
    plos_style.save(fig, out / args.out_name)


if __name__ == "__main__":
    main()
