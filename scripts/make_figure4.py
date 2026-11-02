"""Figure 4 — pulsatility changes between vessels recorded simultaneously (PLOS, 7.5 in wide).

Two tiles; in each, two vessels A and B recorded in the same movie (one clock), joined by a flow-directed path:
  a  outbound from the dorsal aorta (tile 27): relative pulsatility |H1|/Q̄ decreases A → B
  b  return toward the vitelline vein (tile 13): relative pulsatility increases A → B
Top: the tile (max − median motion projection) with A, B and the flow-directed path between them.
Bottom: one cardiac cycle of each vessel from its 3-harmonic fit, (Q(φ) − Q̄)/Q̄, in its own mean-flow direction
(no cross-tile synchronisation or model involved).  Annotations: relative first-harmonic amplitude |H1|/|Q̄| for A and B, and the fundamental's phase
difference Δφ1 = arg H1,B − arg H1,A.

Ported from PerTileFlow/scripts/make_figure5_two_regions.py (old Fig 5); harmonic bars dropped.
Writes figures/fig4/fig4.{png,pdf,tif}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.config import StageConfig
from plexusflow.analyze.geometry import load_tile_geometry, tile_video_shape, graph_to_tile_batch
from plexusflow.reconcile.harmonics import phasors, FS
import plos_style

REGIONS = [dict(tile=27, A=(74413, 74421), B=(75268, 75296), title="Arterial-side path", col="#00838f"),
           dict(tile=13, A=(57078, 57118), B=(53492, 53651), title="Venous-return path", col="#7b3f99")]
COL_A, COL_B, PATHCOL, INK = "#2166ac", "#d73027", "#f0ad00", "#1a1a1a"
PX_UM = 1.7


def meas(G, e, tile):
    return next(m for m in G.edges[e]["measurements_piv"] if int(m["tile_id"]) == tile)


def flow_frame(m, f0):
    """Q̄ > 0 and harmonics in the vessel's own mean-flow direction (DC sign flipped first)."""
    q = np.asarray(m["Q_t"], float); a0, H = phasors(q, f0); s = np.sign(a0) or 1.0
    return abs(a0), H * s


def tile_panel(ax, G, cfg, geom, R):
    t = R["tile"]; ds = float(cfg.videos_realtime_spatial_ds)
    v = tifffile.imread(str(next(Path(cfg.videos_realtime_dir).glob(f"*loc{t}_realtime.tif")))).astype(np.float32)
    med = np.median(v, 0); mot = np.clip(v.max(0) - med, 0, None); img = mot if mot.std() > med.std() else med
    lo, hi = np.percentile(img, (1, 99)); ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.65, cmap="gray", vmin=0, vmax=1)
    # flow-directed path A → B among this tile's measured vessels
    D = nx.DiGraph(); U = nx.Graph(); look = {}
    for u, w, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            if int(m.get("tile_id", -1)) == t:
                D.add_edge(int(m["flow_from"]), int(m["flow_to"])); U.add_edge(u, w)
                look[(int(m["flow_from"]), int(m["flow_to"]))] = (u, w); look[(u, w)] = look[(w, u)] = (u, w)
    mA, mB = meas(G, R["A"], t), meas(G, R["B"], t)
    try:
        nodes = nx.shortest_path(D, int(mA["flow_to"]), int(mB["flow_from"]))
    except (nx.NetworkXNoPath, nx.NodeNotFound):                  # no flow-consistent route: shortest graph route
        cands = [nx.shortest_path(U, a, b) for a in R["A"] for b in R["B"] if nx.has_path(U, a, b)]
        nodes = min(cands, key=len) if cands else []
        print(f"[fig4] tile {t}: no flow-directed A→B route; showing the shortest graph route ({len(nodes)} nodes)")
    pts = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        p = np.asarray(G.edges[look[(a, b)]]["path"], float); ap = np.array([G.nodes[a]["x"], G.nodes[a]["y"]])
        if np.linalg.norm(p[-1] - ap) < np.linalg.norm(p[0] - ap):
            p = p[::-1]
        rc = graph_to_tile_batch(p, geom, t) / ds; pts.append(rc); ax.plot(rc[:, 1], rc[:, 0], color=PATHCOL, lw=2.2, zorder=3)
    if pts:
        allp = np.vstack(pts)
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(allp, axis=0).T))]; w = 0.08 * cum[-1]       # tangent over ±8 % of the path
        for f in ((0.5,) if cum[-1] < 150 else (0.3, 0.7)):                                      # direction arrowheads along the path
            i = int(np.searchsorted(cum, f * cum[-1])); i0, i1 = np.searchsorted(cum, cum[i] - w), min(np.searchsorted(cum, cum[i] + w), len(allp) - 1)
            d_ = allp[i1] - allp[i0]; d_ = d_ / np.linalg.norm(d_); a_, b_ = allp[i] - 16 * d_, allp[i] + 16 * d_
            ax.add_patch(FancyArrowPatch(a_[::-1], b_[::-1], arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.4, zorder=4, shrinkA=0, shrinkB=0,
                                          path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]))
    for e, col, nm in ((R["A"], COL_A, "A"), (R["B"], COL_B, "B")):
        rc = graph_to_tile_batch(np.asarray(G.edges[e]["path"], float), geom, t) / ds
        ax.plot(rc[:, 1], rc[:, 0], color="white", lw=3.4, zorder=5); ax.plot(rc[:, 1], rc[:, 0], color=col, lw=2.0, zorder=6)
        mid = rc[len(rc) // 2]
        ax.annotate(nm, (mid[1], mid[0]), xytext=(7, -9 if mid[0] < 0.4 * img.shape[0] else 7), textcoords="offset points", color=col,
                    fontsize=9, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1), zorder=7)
    L = 200 / (PX_UM * ds); W, H = img.shape[1], img.shape[0]
    ax.plot([0.95 * W - L, 0.95 * W], [0.11 * H] * 2, color="white", lw=2, solid_capstyle="butt")
    ax.text(0.95 * W, 0.08 * H, "200 µm", color="white", ha="right", va="bottom", fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.35"); s.set_linewidth(0.6)


def locator(ax, geom):
    mos = tifffile.imread(str(_paths.STAGE / "fuse" / "organ_fusion.tif")).astype(float)
    from scipy import ndimage as ndi
    out = ~ndi.binary_fill_holes(mos > 0)
    ax.imshow(np.ma.masked_where(out, mos), cmap="gray", vmin=np.percentile(mos[~out], 1), vmax=np.percentile(mos[~out], 99.5))
    for R in REGIONS:
        tg = geom.tiles[R["tile"]]; Hr, Wr = tg.video_shape
        fp = np.array([[0, 0, 1], [0, Wr, 1], [Hr, Wr, 1], [Hr, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
        fp = fp[:, :2] / geom.downsample - [geom.y0, geom.x0]
        ax.plot(fp[:, 1], fp[:, 0], color=R["col"], lw=1.4)
    ys, xs = np.nonzero(~out); ax.set_xlim(xs.min() - 30, xs.max() + 30); ax.set_ylim(ys.max() + 30, ys.min() - 30)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    plos_style.apply()
    P = _paths.STAGE; G = _paths.load_graph(); tf0 = G.graph["tile_f0_piv"]; cfg = StageConfig.load(P)
    shapes = {i: tile_video_shape(cfg, i) for i in (cfg.discover_raw_tile_ids() or sorted(G.graph['tile_f0_piv']))}
    geom = load_tile_geometry(cfg, video_shapes={k: v for k, v in shapes.items() if v is not None})

    fig = plt.figure(figsize=(5.2, 4.6))                     # text-column width; no organ locator
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.2, wspace=0.34, left=0.11, right=0.985, top=0.93, bottom=0.09)
    ph = np.linspace(0, 1, 400); axs = []
    for j, (R, letter) in enumerate(zip(REGIONS, "ab")):
        t = R["tile"]; f0 = float(tf0[t])
        axT = fig.add_subplot(gs[0, j]); tile_panel(axT, G, cfg, geom, R); axs.append(axT)
        axT.set_title(R["title"], fontsize=9, pad=3)
        axT.text(-0.08, 1.03, letter, transform=axT.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
        axW = fig.add_subplot(gs[1, j]); pi, h1 = {}, {}
        for e, c_, nm in ((R["A"], COL_A, "A"), (R["B"], COL_B, "B")):
            qb, H = flow_frame(meas(G, e, t), f0)
            w = sum((H[k - 1] * np.exp(2j * np.pi * k * ph)).real for k in (1, 2, 3)) / qb
            axW.plot(ph, w, color=c_, lw=1.4, label=f"Vessel {nm}")
            pi[nm] = abs(H[0]) / qb; h1[nm] = H[0]
        dphi = np.degrees(np.angle(h1["B"] * np.conj(h1["A"])))
        axW.axhline(0, color="0.8", lw=0.5); axW.set_xlim(0, 1); axW.set_xticks([0, 0.5, 1]); axW.set_ylim(-1.25, 0.75)
        axW.set_xlabel("Cardiac phase", labelpad=1); axW.spines[["top", "right"]].set_visible(False)
        axW.set_ylabel(r"$(Q - \bar Q)\,/\,\bar Q$", labelpad=1)
        if j == 0:
            axW.legend(loc="upper right", frameon=False, borderaxespad=0.1, handlelength=1.4)
        axW.text(0.02, 0.03, rf"$|H_1|/|\bar Q|$: {pi['A']:.2f} → {pi['B']:.2f}" + "\n" + rf"$\Delta\varphi_1$ = {dphi:+.0f}°", transform=axW.transAxes,
                 ha="left", va="bottom", fontsize=8, linespacing=1.3)
        print(f"[fig4] tile {t}: |H1|/Q̄ A {pi['A']:.2f} → B {pi['B']:.2f} (×{pi['B'] / pi['A']:.2f}), Δφ1 {dphi:+.0f}°")
    plos_style.save(fig, _paths.FIGURES / "fig4" / "fig4")


if __name__ == "__main__":
    main()
