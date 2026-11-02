"""Figure 4 — pulsatility changes between vessels recorded simultaneously (PLOS, 7.5 in wide).

Two tiles; in each, two vessels A and B recorded in the same movie (one clock), joined by a flow-directed path:
  a  outbound from the dorsal aorta (tile 27): relative pulsatility decreases A → B, more so at higher harmonics
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

REGIONS = [dict(tile=27, A=(58635, 58696), B=(59043, 59075), title="Arterial-side path", short="arterial side", col="#00838f"),
           dict(tile=13, A=(57078, 57118), B=(53492, 53651), title="Venous-return path", short="venous return", col="#7b3f99")]
COL_A, COL_B, PATHCOL, INK = "#2166ac", "#d73027", "#f0ad00", "#1a1a1a"
PX_UM = 1.7


def meas(G, e, tile):
    return next(m for m in G.edges[e]["measurements_piv"] if int(m["tile_id"]) == tile)


def flow_frame(m, f0):
    """Q̄ > 0 and harmonics in the vessel's own mean-flow direction (DC sign flipped first)."""
    q = np.asarray(m["Q_t"], float); a0, H = phasors(q, f0); s = np.sign(a0) or 1.0
    return abs(a0), H * s


def beat_pi(m, f0, K=3):
    """Normalised harmonic amplitudes Π_n = |H_n|/|Q̄| (n = 1..K) of the whole trace and their standard error across complete
    cardiac cycles (each cycle fitted separately by least squares)."""
    q = np.asarray(m["Q_t"], float); t = np.arange(len(q)) / FS; cyc = np.floor(t * f0).astype(int); per = []
    for c in np.unique(cyc):
        sel = cyc == c
        if sel.sum() < 0.9 * FS / f0:
            continue
        X = np.column_stack([np.ones(sel.sum())] + [fn(2 * np.pi * k * f0 * t[sel]) for k in range(1, K + 1) for fn in (np.cos, np.sin)])
        b, *_ = np.linalg.lstsq(X, q[sel], rcond=None)
        per.append([np.hypot(b[2 * k - 1], b[2 * k]) / abs(b[0]) for k in range(1, K + 1)])
    per = np.array(per)
    a0, H = phasors(q, f0, K)
    return np.abs(H) / abs(a0), per.std(0, ddof=1) / np.sqrt(len(per)), len(per)


def noise_floor(m, f0):
    """Median amplitude at off-harmonic frequencies (k + ½)·f0, k = 1..5, relative to |Q̄|: the trace's noise floor on the Π scale."""
    q = np.asarray(m["Q_t"], float); t = np.arange(len(q)) / FS; a0 = q.mean()
    return float(np.median([abs(2 * np.mean((q - a0) * np.exp(-2j * np.pi * (k + 0.5) * f0 * t))) for k in range(1, 6)]) / abs(a0))


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
    pts = []; path_edges = [look[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]
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
        s.set_color(R["col"]); s.set_linewidth(1.2)
    return path_edges


def context(ax, G, geom, paths_):
    """Whole organ (fused mosaic) with both tile footprints and the two A→B paths."""
    mos = tifffile.imread(str(_paths.STAGE / "fuse" / "organ_fusion.tif")).astype(float)
    from scipy import ndimage as ndi
    out = ~ndi.binary_fill_holes(mos > 0); Hm = mos.shape[0]
    ax.imshow(np.ma.masked_where(out, mos), cmap="gray", vmin=np.percentile(mos[~out], 1), vmax=np.percentile(mos[~out], 99.5))
    fps = {}
    for R in REGIONS:
        tg = geom.tiles[R["tile"]]; Hr, Wr = tg.video_shape
        fp = np.array([[0, 0, 1], [0, Wr, 1], [Hr, Wr, 1], [Hr, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
        fp = fp[:, :2] / geom.downsample - [geom.y0, geom.x0]; fps[R["tile"]] = fp
        ax.plot(fp[:, 1], fp[:, 0], color=R["col"], lw=2.0, zorder=4)
        for e in paths_[R["tile"]] + [R["A"], R["B"]]:
            e = e if G.has_edge(*e) else e[::-1]; q = np.asarray(G.edges[e]["path"], float)
            col = COL_A if e in (R["A"], R["A"][::-1]) else (COL_B if e in (R["B"], R["B"][::-1]) else PATHCOL)
            ax.plot(q[:, 0], Hm - q[:, 1], color=col, lw=1.6, zorder=5, solid_capstyle="round")
        right = R is REGIONS[0]                          # tile number beside its footprint, on the side of its enlarged panel
        k = int(np.argmax(fp[:4, 1])) if right else int(np.argmin(fp[:4, 1]))
        ax.text(fp[k, 1] + (25 if right else -25), fp[k, 0], f"tile {R['tile']}", color=R["col"], fontsize=8, ha="left" if right else "right",
                va="center", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ys, xs = np.nonzero(~out); ax.set_xlim(xs.min() - 30, xs.max() + 30); ax.set_ylim(ys.max() + 30, ys.min() - 30)
    ax.set_aspect("equal"); ax.axis("off")
    L = 1000 / PX_UM; x1 = xs.max() - 40; y1 = ys.max() - 10
    ax.plot([x1 - L, x1], [y1, y1], color=INK, lw=2, solid_capstyle="butt"); ax.text(x1 - L / 2, y1 - 40, "1 mm", ha="center", va="bottom", fontsize=8)
    return fps


def main():
    from matplotlib.patches import ConnectionPatch
    plos_style.apply()
    P = _paths.STAGE; G = _paths.load_graph(); tf0 = G.graph["tile_f0_piv"]; cfg = StageConfig.load(P)
    shapes = {i: tile_video_shape(cfg, i) for i in (cfg.discover_raw_tile_ids() or sorted(G.graph['tile_f0_piv']))}
    geom = load_tile_geometry(cfg, video_shapes={k: v for k, v in shapes.items() if v is not None})

    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.78], hspace=0.32, left=0.07, right=0.985, top=0.93, bottom=0.09)
    gT = gs[0].subgridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.12)
    gB = gs[1].subgridspec(1, 2, wspace=0.22); gBs = [gB[j].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.42) for j in (0, 1)]
    ph = np.linspace(0, 1, 400); tiles_ax, paths_, rows = {}, {}, {}
    for R, col in zip(REGIONS, (0, 2)):
        axT = fig.add_subplot(gT[col]); paths_[R["tile"]] = tile_panel(axT, G, cfg, geom, R); tiles_ax[R["tile"]] = axT
        axT.set_title(f"{R['title']} (tile {R['tile']})", fontsize=9, pad=3, color=R["col"])
    axM = fig.add_subplot(gT[1]); fps = context(axM, G, geom, paths_)
    for R in REGIONS:                                   # connect each footprint to its tile panel
        fp = fps[R["tile"]]; axT = tiles_ax[R["tile"]]; left = R is REGIONS[0]
        k = int(np.argmin(fp[:4, 1])) if left else int(np.argmax(fp[:4, 1]))
        fig.add_artist(ConnectionPatch(xyA=(fp[k, 1], fp[k, 0]), coordsA=axM.transData, xyB=(1.0 if left else 0.0, 0.5),
                                       coordsB=axT.transAxes, color=R["col"], lw=0.6, alpha=0.45, ls=(0, (3, 2))))
    for j, R in enumerate(REGIONS):
        t = R["tile"]; f0 = float(tf0[t]); axW = fig.add_subplot(gBs[j][0]); axP = fig.add_subplot(gBs[j][1]); h, pis = {}, {}
        for e, c_, nm in ((R["A"], COL_A, "A"), (R["B"], COL_B, "B")):
            m = meas(G, e, t); qb, H = flow_frame(m, f0); pis[nm] = beat_pi(m, f0); h[nm] = H; pis[nm + "_floor"] = noise_floor(m, f0)
            w = sum((H[k - 1] * np.exp(2j * np.pi * k * ph)).real for k in (1, 2, 3)) / qb
            axW.plot(ph, w, color=c_, lw=1.4, label=f"Vessel {nm}")
        dphi = np.degrees(np.angle(h["B"][0] * np.conj(h["A"][0])))
        axW.axhline(0, color="0.8", lw=0.5); axW.set_xlim(0, 1); axW.set_xticks([0, 0.5, 1]); axW.set_ylim(-1.25, 0.8)
        axW.set_xlabel("Cardiac phase", labelpad=1); axW.spines[["top", "right"]].set_visible(False)
        axW.set_ylabel(r"$(Q - \bar Q)\,/\,\bar Q$", labelpad=1)
        axW.set_title(R["title"], fontsize=9, pad=3, color=R["col"], loc="left")
        axW.text(0.03, 0.04, rf"$\Delta\varphi_1$ = {dphi:+.0f}°", transform=axW.transAxes, ha="left", va="bottom", fontsize=8)
        if j == 0:
            axW.legend(loc="upper right", frameon=False, borderaxespad=0.1, handlelength=1.4)
        # normalised harmonic amplitudes A → B with per-beat standard errors
        n_ = np.arange(1, 4); (pa, sa, na), (pb, sb, nb) = pis["A"], pis["B"]
        fl = (pis["A_floor"], pis["B_floor"])
        for k in range(3):
            axP.plot([n_[k] - 0.12, n_[k] + 0.12], [pa[k], pb[k]], color="0.6", lw=0.8, zorder=1)
        for x_, p_, s_, c_, f_ in ((n_ - 0.12, pa, sa, COL_A, fl[0]), (n_ + 0.12, pb, sb, COL_B, fl[1])):
            axP.errorbar(x_, p_, yerr=s_, fmt="none", ecolor=c_, capsize=1.5, lw=0.8, zorder=3)
            weak = p_ < 3 * f_                               # open marker: within 3× the vessel's own off-harmonic noise floor
            axP.scatter(x_, p_, s=18, zorder=4, facecolors=np.where(weak, "white", c_), edgecolors=c_, linewidths=1.0)
        for k in range(3):
            axP.text(n_[k], max(pa[k] + sa[k], pb[k] + sb[k]) * 1.45, f"{100 * (pb[k] / pa[k] - 1):+.0f}%", ha="center", va="bottom", fontsize=8,
                     color=COL_B, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        axP.set_yscale("log"); axP.set_ylim(0.007, 2.0); axP.set_xlim(0.35, 3.5); axP.set_xticks(n_)
        axP.set_xlabel("Harmonic $n$", labelpad=1); axP.set_ylabel(r"$\Pi^{(n)} = |\hat Q^{(n)}|\,/\,\bar Q$", labelpad=1)
        axP.spines[["top", "right"]].set_visible(False)
        print(f"[fig4] tile {t}: Π_A {np.round(pa, 3)} ± {np.round(sa, 3)} ({na} beats); Π_B {np.round(pb, 3)} ± {np.round(sb, 3)} ({nb} beats); "
              f"change {np.round(100 * (pb / pa - 1))}%; Δφ1 {dphi:+.0f}°; noise floor A {fl[0]:.4f} B {fl[1]:.4f}")
        rows[j] = (axW, axP)
    # panel letters on a grid
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); tb = lambda a_: inv.transform(a_.get_tightbbox(rr))
    x_a = tb(tiles_ax[REGIONS[0]["tile"]])[0, 0]; y_top = max(tb(a_)[1, 1] for a_ in list(tiles_ax.values()) + [axM]); y_bot = max(tb(a_)[1, 1] for r_ in rows.values() for a_ in r_)
    for x_, y_, l_ in ((min(x_a, tb(rows[0][0])[0, 0]) - 0.005, min(y_top + 0.004, 0.99), "a"), (min(x_a, tb(rows[0][0])[0, 0]) - 0.005, y_bot + 0.004, "b"),
                       (tb(rows[1][0])[0, 0] - 0.005, y_bot + 0.004, "c")):
        fig.text(x_, y_, l_, fontsize=12, fontweight="bold", va="bottom", ha="left")
    plos_style.save(fig, _paths.FIGURES / "fig4" / "fig4")

if __name__ == "__main__":
    main()
