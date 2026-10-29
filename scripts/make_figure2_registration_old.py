"""Figure 2 — registration: overlapping tiles recorded at different times are placed by a growth-aware affine map into
one whole-organ mosaic (PLOS, 7.5 in wide).

  a  one tile: movie frame → temporal-std/CLAHE projection (the image that is registered) → ilastik vessel mask
  b  why the map is affine: tiles recorded Δt apart differ by growth, so each is placed by x_mosaic = A_i x_i + b_i
  c  tile 23 (recorded later) over reference tiles 11/12: vessel masks before (translation only) and
  d  after the affine fit; magenta = mask disagreement in the overlap (percent of overlap), arrows = corner displacement
  e  the fused whole-organ mosaic with every tile footprint

Inputs: videos_realtime, per-tile ilastik probabilities (Somites21/Tiles/Probs), stitch/positions_test.json (the 11/12/23
hand fit) and stitch/positions.json (canonical).  Writes figures/fig2/fig2.{png,pdf,tif}.  Ported from the archived
hand-assembly script (make_figure2.py, 2026-08/09).
"""
from __future__ import annotations
import sys, os, re, glob, argparse, datetime as dt
from pathlib import Path

import numpy as np
import tifffile, h5py
from scipy.ndimage import affine_transform, gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
import plos_style

H, W = 704, 640
CEN = np.array([H / 2, W / 2])
CORNERS = np.array([[0, 0], [0, W], [H, W], [H, 0]], float)
COL = {11: "#d62728", 12: "#2ca02c", 23: "#1f77b4"}
PX_UM_RAW = 1.7
MIS = "#c51b8a"          # mask disagreement
PROBS = Path("/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/Somites21/Tiles/Probs")
RAW = Path("/Users/amayer/Desktop/ImageData/21Somites/emb1")
CACHE = Path("/private/tmp/plexus_stdclahe_cache")


def stdclahe(t):
    CACHE.mkdir(parents=True, exist_ok=True); cf = CACHE / f"sc2_{t}.npy"
    if cf.exists():
        return np.load(cf)
    im = tifffile.imread(glob.glob(str(_paths.STAGE / "videos_realtime" / f"*loc{t}_*"))[0]).astype(np.float32)
    s = gaussian_filter(im.std(0), 1.0); s = (s - s.min()) / (np.ptp(s) + 1e-6)
    s = resize(equalize_adapthist(s, clip_limit=0.015), (H, W), order=1, preserve_range=True).astype(np.float32)
    np.save(cf, s); return s


def prob(t, ch=0):
    with h5py.File(PROBS / f"vid{t}_maxmed_probs.h5", "r") as f:
        return np.asarray(f["exported_data"])[..., ch].astype(np.float32)


def acq_minutes(tids):
    def clock(t):
        for d in sorted(RAW.glob(f"10x 250fps loc{t}_*")):
            cs = list(d.glob("*.cih")) if d.is_dir() else []
            if cs:
                txt = cs[0].read_text(errors="ignore")
                dm, tm = re.search(r"Date\s*:\s*([\d/]+)", txt), re.search(r"Time\s*:\s*([\d:]+)", txt)
                for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
                    try:
                        return dt.datetime.strptime(dm.group(1) + " " + tm.group(1), fmt)
                    except (ValueError, AttributeError):
                        pass
    t0 = clock(1)
    return {t: (clock(t) - t0).total_seconds() / 60 for t in tids}


def canvas_of(A, tiles):
    allc = np.concatenate([CORNERS @ A[t][:2, :2].T + A[t][:2, 2] for t in tiles])
    oy0, ox0 = allc.min(0); oy1, ox1 = allc.max(0)
    return oy0, ox0, int(np.ceil(oy1 - oy0)), int(np.ceil(ox1 - ox0))


def warp(img, Aff, oy0, ox0, Hc, Wc, order):
    Mi = np.linalg.inv(Aff[:2, :2]); off = Mi @ (np.array([oy0, ox0]) - Aff[:2, 2])
    return affine_transform(img, Mi, offset=off, output_shape=(Hc, Wc), order=order)


def quad(Aff, oy0=0.0, ox0=0.0):
    return (CORNERS @ Aff[:2, :2].T + Aff[:2, 2] - np.array([oy0, ox0]))[:, [1, 0]]


def frame_axes(ax, col="0.5", lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(lw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", type=int, nargs=3, default=[11, 12, 23], help="two reference tiles + the registered tile")
    ap.add_argument("--a-tile", type=int, default=12)
    args = ap.parse_args()
    plos_style.apply()
    from net3.edit.video_overlay import load_tile_positions
    tiles = args.tiles; ref_ids, odd = tiles[:2], tiles[2]
    times = acq_minutes(tiles)
    print(f"[fig2] tiles {tiles}, times " + ", ".join(f"{t}: {times[t]:.0f} min" for t in tiles))

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.32], hspace=0.16, left=0.025, right=0.985, top=0.94, bottom=0.03)
    gT = gs[0].subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.08)
    gB = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1.02], wspace=0.06)
    letter = lambda ax, l, dx=0.0: ax.text(dx, 1.03, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: frame → projection → mask
    st = tifffile.imread(glob.glob(str(_paths.STAGE / "videos_realtime" / f"*loc{args.a_tile}_*"))[0])
    frame = st[len(st) // 2].astype(float); proj = stdclahe(args.a_tile); mask = prob(args.a_tile) > 0.5
    gA = gT[0].subgridspec(1, 3, wspace=0.22)
    for k, (im, cmap, ttl) in enumerate(((frame, "gray", "movie frame"), (proj, "gray", "temporal projection"),
                                         (mask, "gray_r", "vessel mask"))):
        ax = fig.add_subplot(gA[k])
        vmn, vmx = (0, 1) if im.dtype == bool else np.percentile(im, (1, 99.5))
        ax.imshow(im, cmap=cmap, vmin=vmn, vmax=vmx, interpolation="antialiased"); frame_axes(ax)
        ax.set_title(ttl, fontsize=8, pad=2)
        if k == 0:
            letter(ax, "a", -0.02)
            L = 200 / (PX_UM_RAW * W / frame.shape[1])            # 200 µm in realtime-frame px
            x0, y0 = 0.08 * frame.shape[1], 0.92 * frame.shape[0]
            ax.plot([x0, x0 + L], [y0, y0], color="white", lw=2, solid_capstyle="butt")
            ax.annotate("200 µm", (x0 + L / 2, y0), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color="white", fontsize=8)
        if k < 2:
            ax.annotate("", xy=(1.2, 0.5), xytext=(1.03, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.25", mutation_scale=8))

    # c, d: before / after affine registration of the later tile
    pos = load_tile_positions(str(_paths.STAGE / "stitch" / "positions_test.json"))
    A = {t: np.asarray(pos[t]["affine_matrix"], float) for t in tiles}
    oy0, ox0, Hc, Wc = canvas_of(A, tiles)
    mk = {t: (prob(t) > 0.5).astype(np.float32) for t in tiles}
    cov = lambda Aff: warp(np.ones((H, W), np.float32), Aff, oy0, ox0, Hc, Wc, 0) > 0.5
    mref = np.zeros((Hc, Wc), bool); crefcov = np.zeros((Hc, Wc), bool)
    for t in ref_ids:
        m = warp(mk[t], A[t], oy0, ox0, Hc, Wc, 1) > 0.5; cc = cov(A[t]); mref |= m & cc; crefcov |= cc
    ca = A[odd][:2, :2] @ CEN + A[odd][:2, 2]
    Ab = np.eye(3); Ab[:2, :2] = A[ref_ids[0]][:2, :2]; Ab[:2, 2] = ca - A[ref_ids[0]][:2, :2] @ CEN    # translation only
    r = np.linalg.inv(A[ref_ids[0]][:2, :2]) @ A[odd][:2, :2]
    scale, angle = float(np.sqrt(abs(np.linalg.det(r)))), float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    fr = {}
    for k, (Aodd, tag, l) in enumerate(((Ab, "before: translation only", "c"), (A[odd], "after: affine", "d"))):
        ax = fig.add_subplot(gB[k])
        m = warp(mk[odd], Aodd, oy0, ox0, Hc, Wc, 1) > 0.5; cc = cov(Aodd)
        base = (mref | (m & cc)).astype(float); gray = 1 - 0.55 * base; gray[~(crefcov | cc)] = 1.0
        ov = crefcov & cc; xor = np.full((Hc, Wc), np.nan); xor[ov] = (mref[ov] != (m & cc)[ov]).astype(float)
        fr[tag] = np.nanmean(xor) * 100
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        ax.imshow(np.ma.masked_invalid(np.where(xor == 1, 1.0, np.nan)), cmap=matplotlib.colors.ListedColormap([MIS]), vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
        lab = {**{t: A[t] for t in ref_ids}, odd: Aodd}
        for t in tiles:
            q = quad(lab[t], oy0, ox0); ax.add_patch(Polygon(q, closed=True, fill=False, ec=COL[t], lw=1.0))
            c_ = q.mean(0)
            ax.text(c_[0], c_[1], f"tile {t}\n{times[t]:.0f} min", ha="center", va="center", fontsize=8, color=COL[t], fontweight="bold",
                    linespacing=1.0, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=COL[t], lw=0.5, alpha=0.85))
        ax.text(0.02, 0.02, f"mask disagreement {fr[tag]:.0f}%", transform=ax.transAxes, fontsize=8, va="bottom", color=MIS, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        if l == "d":
            for pb, pa in zip(quad(Ab, oy0, ox0), quad(A[odd], oy0, ox0)):
                ax.add_patch(FancyArrowPatch(pb, pa, arrowstyle="-|>", color=COL[odd], mutation_scale=7, lw=0.9))
        ax.set_xlim(0, Wc); ax.set_ylim(Hc, 0); frame_axes(ax, "0.6")
        ax.set_title(tag, fontsize=8, pad=2); letter(ax, l, -0.02)
    print(f"[fig2] mask disagreement before {fr['before: translation only']:.1f}% → after {fr['after: affine']:.1f}%; "
          f"tile {odd} scale {scale:.3f}, rotation {angle:.1f}°")

    # b: growth / affine schematic
    ax = fig.add_subplot(gT[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off"); letter(ax, "b", 0.0)
    ref = ref_ids[0]
    ax.add_patch(Rectangle((0.4, 3.0), 2.8, 2.8, fill=False, ec=COL[ref], lw=1.2))
    ax.text(1.8, 2.6, f"tiles {ref_ids[0]}, {ref_ids[1]}\n{times[ref]:.0f} min", ha="center", va="top", fontsize=8, color=COL[ref], linespacing=1.05)
    ax.add_patch(FancyArrowPatch((3.5, 4.4), (5.6, 4.4), arrowstyle="-|>", mutation_scale=10, color="0.3", lw=0.9))
    ax.text(4.55, 4.75, f"Δt = {times[odd] - times[ref]:.0f} min\ngrowth", ha="center", va="bottom", fontsize=8, color="0.25", linespacing=1.05)
    sq = 2.8 * scale; off = (2.8 - sq) / 2
    ax.add_patch(Rectangle((6.3 + off, 3.0 + off), sq, sq, fill=False, ec=COL[odd], lw=1.2, angle=angle, rotation_point="center"))
    ax.text(7.7, 2.6, f"tile {odd}\n{times[odd]:.0f} min\nscale {scale:.2f}×", ha="center", va="top", fontsize=8, color=COL[odd], linespacing=1.05)
    ax.text(5.0, 0.35, r"$\mathbf{x}_{\mathrm{mosaic}} = A_i\,\mathbf{x}_i + \mathbf{b}_i$", ha="center", va="center", fontsize=9)

    # e: whole-organ mosaic with every tile footprint
    ax = fig.add_subplot(gB[2])
    mos = tifffile.imread(str(_paths.STAGE / "fuse" / "organ_fusion.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos
    posc = load_tile_positions(str(_paths.STAGE / "stitch" / "positions.json"))
    Ac = {t: np.asarray(e["affine_matrix"], float) for t, e in posc.items() if e.get("affine_matrix") is not None}
    ax.imshow(np.ma.masked_equal(mos, 0), cmap="gray", vmin=np.percentile(mos[mos > 0], 1), vmax=np.percentile(mos, 99.7), interpolation="antialiased")
    for t in sorted(Ac):
        ax.add_patch(Polygon(quad(Ac[t]), closed=True, fill=False, ec=COL.get(t, "#e8c85a"), lw=1.0 if t in COL else 0.4,
                             alpha=1.0 if t in COL else 0.7, zorder=3 if t in COL else 2))
    x0, y0 = mos.shape[1] * 0.05, mos.shape[0] * 0.965; L = 1000 / PX_UM_RAW
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=2, solid_capstyle="butt")
    ax.annotate("1 mm", (x0 + L / 2, y0), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, mos.shape[1]); ax.set_ylim(mos.shape[0], 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"fused mosaic ({len(Ac)} tiles)", fontsize=8, pad=2); letter(ax, "e", 0.0)
    plos_style.save(fig, _paths.FIGURES / "fig2" / "fig2")


if __name__ == "__main__":
    main()
