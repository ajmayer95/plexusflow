"""Figure 6 v3 — construction of a conservation-consistent reference field Q′ (the whole-organ field is fig. 8).

  a    measurement quality → uncertainty: two vessels of the junction's tile — clean (vessel 1 of the junction) and unreliable
       (lowest alignment with ≥ 40 % sections kept): crop with kept cross-sections and mean-flow arrows; sections kept, alignment, c
  b    calibration: held-out standardized same-tile junction residuals z (model fitted on the other tiles, 5 folds by tile),
       standard normal, ±1.96; RMS z and coverage (tail beyond 3 in the caption)
  c    reconciliation at one junction: tile image with the vessels; signed dumbbells measured → reconciled with c; residual
       before / after; inset: median adjustment and direction changes over all measured vessels
Junction (c), by rule: interior node with 3 vessels — one short (1.5–3 radii), two ≥ 3 radii; measurement centrelines ≥ 0.5 R apart
away from the ends; no vessel's kept cross-sections all in one end zone (outer 20 %); largest c ≥ 2 × smallest; every vessel measured with its winner from the same tile, measured
imbalance ≥ 10 % of the throughput, the vessel with the largest c takes ≥ 70 % of the total |correction|, the others move
≤ 10 %, no direction change; then largest throughput.
Methods / supplement: c = c0 + c_f (1 − f) + c_a (1 − align); weights w = 1/(c² Q² + s0²); weighting ablation; damping.
Reads the graph and results/{dc_reconciled_v2.pkl, reliability_model_v1.json}; writes figures/fig6v3/fig6v3.{png,pdf}.
"""
from __future__ import annotations
import sys
import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import measurement_table, same_tile_junctions, carved_domain
from plexusflow.reconcile.reliability import _arrays, _fit_theta

OUT_DIR = _paths.FIGURES / "fig6v3"
T0 = time.time()
VESSEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def log(m):
    print(f"[fig6v3 +{time.time() - T0:4.0f} s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--junction", type=int, default=None, help="junction node for panel b (default: rule)")
    args = ap.parse_args()
    G = _paths.load_graph(); dom = carved_domain(G); E = dom["edges"]; eix = {e: i for i, e in enumerate(E)}
    D = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    model = json.loads((_paths.RESULTS / "reliability_model_v1.json").read_text())
    M = measurement_table(G); y, im, Qp, win = D["Qm"], D["imaged"], D["Qp"], D["tile"]; c = np.where(im, D["c"], 1.0)
    nodes = dom["nodes"]; nix = {n: k for k, n in enumerate(nodes)}; bnd = set(int(b) for b in dom["bnd"]); inc = {}
    for i, (u, v) in enumerate(E):
        inc.setdefault(nix[u], []).append((i, 1.0)); inc.setdefault(nix[v], []).append((i, -1.0))
    log(f"{_paths.STAGE.name}: {len(E)} carved vessels, {int(im.sum())} measured")

    # a — held-out standardized residuals (folds by tile)
    tl = sorted({t for t, _ in M}); JJ = []
    for t in tl:
        JJ += [(t, j) for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t})]
    ix_, dc_, f_, al_, _, net_ = _arrays(JJ, M, tl); jt = np.array([tl.index(t) for t, _ in JJ])
    fold = np.random.default_rng(0).permutation(np.arange(len(tl)) % 5)[jt]; zs = []
    for k in range(5):
        def part(sel):
            jj = np.flatnonzero(sel); rm = -np.ones(len(JJ), int); rm[jj] = np.arange(len(jj)); kp = sel[ix_]
            return (rm[ix_[kp]], dc_[kp], f_[kp], al_[kp], np.zeros(kp.sum(), int), net_[jj])
        th = np.exp(_fit_theta(part(fold != k), 1).x); d = part(fold == k); cc = th[0] + th[1] * (1 - d[2]) + th[2] * (1 - d[3])
        zs.append(d[5] / np.sqrt(np.bincount(d[0], d[1] ** 2 * cc ** 2, minlength=len(d[5])) + th[3] ** 2))
    z = np.concatenate(zs); cov = (np.sqrt(np.mean(z ** 2)), np.mean(np.abs(z) < 1), np.mean(np.abs(z) < 1.96), np.mean(np.abs(z) > 3))
    log(f"  (a) {len(z)} junctions held out by tile: RMS(z) {cov[0]:.3f}, |z|<1 {cov[1]:.3f}, |z|<1.96 {cov[2]:.3f}, |z|>3 {cov[3]:.4f}")

    # b — the junction
    if args.junction is not None:
        jn = nix[args.junction]
    else:
        cands = []
        for n, es in inc.items():
            if n in bnd or len(es) != 3 or not all(im[i] for i, _ in es) or len({int(win[i]) for i, _ in es}) != 1:
                continue
            asp = sorted(G.edges[E[i]]["length"] / G.edges[E[i]]["radius"] for i, _ in es)
            if not (1.5 <= asp[0] < 3 and asp[1] >= 3):                     # one short vessel, two of medium length or more
                continue
            thr = sum(abs(y[i]) for i, _ in es) / 2; imb = abs(sum(s * y[i] for i, s in es))
            if thr <= 0 or imb < 0.10 * thr:
                continue
            corr = np.array([abs(Qp[i] - y[i]) for i, _ in es]); cs = np.array([c[i] for i, _ in es]); w_ = int(np.argmax(cs))
            rel = np.array([abs(Qp[i] - y[i]) / abs(y[i]) for i, _ in es])
            if corr[w_] < 0.7 * corr.sum() or np.any(np.delete(rel, w_) > 0.10) or np.any(np.sign(Qp[[i for i, _ in es]]) != np.sign(y[[i for i, _ in es]])):
                continue
            if cs.max() < 2 * cs.min():                                    # a real difference in reliability to reconcile
                continue
            cands.append((thr, n))
        kcache = {}
        def separate(n):
            """The vessels' measurement centrelines stay ≥ 0.5 R apart away from the junction (no shared flow ridge)."""
            t = int(win[inc[n][0][0]])
            if t not in kcache:
                kcache[t] = pickle.load(gzip.open(_paths.STAGE / "batch_kl_inputs" / f"tile_{t}.pkl.gz"))["kl_inputs"]
            cls = []
            for i, _ in inc[n]:
                ki = kcache[t].get(E[i]) or kcache[t].get(E[i][::-1])
                if ki is None:
                    return False
                cl = np.asarray(ki["cl_used"], float) + np.asarray(ki["roi_offset"], float); cls.append((cl, float(ki["R_prior_px"])))
            for a in range(3):
                for b in range(a + 1, 3):
                    (ca, Ra), (cb, Rb) = cls[a], cls[b]
                    core = lambda cl: cl[len(cl) // 5: len(cl) - len(cl) // 5] if len(cl) >= 10 else cl
                    if np.min(np.linalg.norm(core(ca)[:, None] - cb[None], axis=2)) < 0.5 * min(Ra, Rb):
                        return False
            from plexusflow.analyze.kloosterman import kloosterman_scalar
            for (i, _), (cl, R) in zip(inc[n], cls):                      # no vessel whose kept cross-sections all sit in one end zone
                ki = kcache[t].get(E[i]) or kcache[t].get(E[i][::-1]); cl0 = np.asarray(ki["cl_used"], float)
                vm = np.asarray(kloosterman_scalar(np.asarray(ki["vfield_mean"], float), cl0, R, mask=ki["ridge_mask_roi"], dt=1, slice_window=2, n_bins=60,
                                                   fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5, R_slice_lo=0.5, R_slice_hi=2.0,
                                                   min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)["valid_mask"], bool)
                sa = np.r_[0, np.cumsum(np.hypot(*np.diff(cl0, axis=0).T))]; kp = sa[vm]
                if len(kp) == 0 or (kp <= 0.2 * sa[-1]).all() or (kp >= 0.8 * sa[-1]).all():
                    return False
            return True
        for thr_, n_ in sorted(cands, reverse=True):
            log(f"     candidate {nodes[n_]}: throughput {thr_:.3f}, c " + "/".join(f"{c[i]:.2f}" for i, _ in inc[n_]) + ", aspect "
                + "/".join(f"{G.edges[E[i]]['length'] / G.edges[E[i]]['radius']:.1f}" for i, _ in inc[n_]) + f", separate {separate(n_)}")
        best = next(((thr, n) for thr, n in sorted(cands, reverse=True) if separate(n)), None)
        log(f"  (b) {len(cands)} junctions match the rule; first with separate centrelines (largest throughput) used")
        jn = best[1]
    es = inc[jn]; tj = int(win[es[0][0]])
    before = sum(s * y[i] for i, s in es); after = sum(s * Qp[i] for i, s in es); thr = sum(abs(y[i]) for i, _ in es) / 2
    log(f"  (b) junction {nodes[jn]} (tile {tj}): " + ", ".join(f"{E[i]} y {s * y[i]:+.3f} c {c[i]:.2f} Q′ {s * Qp[i]:+.3f}" for i, s in es)
        + f";  net {before:+.4f} → {after:+.1e} (throughput {thr:.3f})")

    # ---- shared: Kloosterman kept cross-sections (batch settings) and drawing of one vessel
    from plexusflow.analyze.kloosterman import kloosterman_scalar
    from plexusflow.analyze.centerline import _centerline_geometry
    KW = dict(dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5, R_slice_lo=0.5,
              R_slice_hi=2.0, min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)   # reproduces n_kept
    kdj = pickle.load(gzip.open(_paths.STAGE / "batch_kl_inputs" / f"tile_{tj}.pkl.gz"))["kl_inputs"]
    img = tifffile.imread(str(_paths.STAGE / "tiles" / "maxmed" / f"vid{tj}_maxmed.tif")).astype(float)

    def vessel(e):
        ki = kdj.get(e) or kdj.get(e[::-1]); vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R = float(ki["R_prior_px"])
        vm = np.asarray(kloosterman_scalar(vf, cl, R, mask=ki["ridge_mask_roi"], **KW)["valid_mask"], bool)
        return dict(vf=vf, cl=cl, R=R, off=np.asarray(ki["roi_offset"], float), vm=vm)

    def draw(ax, v, col, pad=1.6, arrows=True, ticks=True):
        cl = v["cl"] + v["off"]; R = v["R"]; vf = v["vf"]; H, W = vf.shape[:2]; _, tang, nrm = _centerline_geometry(v["cl"])
        lo, hi = cl.min(0) - pad * R, cl.max(0) + pad * R; side = max(max(hi - lo) / 2, 3.5 * R); ctr = (lo + hi) / 2
        r0, r1 = int(max(ctr[0] - side, 0)), int(min(ctr[0] + side, img.shape[0])); c0_, c1_ = int(max(ctr[1] - side, 0)), int(min(ctr[1] + side, img.shape[1]))
        crop = img[r0:r1, c0_:c1_]; ax.imshow(img, cmap="gray", vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 99.5))
        ax.plot(cl[:, 1], cl[:, 0], color=col, lw=3.2, zorder=4, solid_capstyle="round")
        if ticks:
            for q in np.flatnonzero(v["vm"]):
                a_, b_ = cl[q] - 0.75 * R * nrm[q], cl[q] + 0.75 * R * nrm[q]; ax.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white", lw=1.6, zorder=5)
        if arrows:
            qs = np.arange(1, len(v["cl"]) - 1, max(2, len(v["cl"]) // 7)); ys_, xs_ = [], []
            for q in qs:
                for rho_ in (-0.5 * R, 0.5 * R):
                    ys_.append(int(round(v["cl"][q, 0] + rho_ * nrm[q, 0]))); xs_.append(int(round(v["cl"][q, 1] + rho_ * nrm[q, 1])))
            ys_, xs_ = np.array(ys_), np.array(xs_); okp = (ys_ >= 0) & (ys_ < H) & (xs_ >= 0) & (xs_ < W); ys_, xs_ = ys_[okp], xs_[okp]
            if len(ys_):
                nn = np.argmin((v["cl"][None, :, 0] - ys_[:, None]) ** 2 + (v["cl"][None, :, 1] - xs_[:, None]) ** 2, axis=1)
                U, V = vf[ys_, xs_, 0], vf[ys_, xs_, 1]; sp_ = np.hypot(U, V); Ls = np.nanpercentile(sp_, 90) or 1.0
                ax.quiver(xs_ + v["off"][1], ys_ + v["off"][0], U / Ls * 0.8 * R, V / Ls * 0.8 * R, np.abs(V * tang[nn, 0] + U * tang[nn, 1]) / np.maximum(sp_, 1e-12),
                          cmap="RdYlGn", clim=(0.3, 1), angles="xy", scale_units="xy", scale=1, width=0.012, headwidth=3.5, zorder=6)
        ax.set_xlim(c0_, c1_); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)

    vinfo = [vessel(E[i]) for i, _ in es]
    # the third example for (a): a vessel in the same tile, well sampled but with flow misaligned to its centreline
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0
    jset = {E[i] for i, _ in es}; mis = None
    for i in np.flatnonzero(im & (win == tj)):
        m = M.get((tj, E[i]))
        if m is None or E[i] in jset or al(m) > 0.75 or m["f"] < 0.4:
            continue
        ki = kdj.get(E[i]) or kdj.get(E[i][::-1])
        if ki is None or len(ki["cl_used"]) < 12:
            continue
        if mis is None or al(m) - 0.3 * m["f"] < mis[1]:
            mis = (i, al(m) - 0.3 * m["f"])
    ex = [(es[0][0], VESSEL_COLORS[0], "1", vinfo[0], "clean measurement"),
          (es[1][0], VESSEL_COLORS[1], "2", vinfo[1], "few cross-sections usable")]
    if mis is not None:
        ex.append((mis[0], "#ff7f00", "x", vessel(E[mis[0]]), "flow not along the vessel"))
    log(f"  (a) examples: " + ", ".join(f"{E[i]} f {M[(tj, E[i])]['f']:.2f} align {al(M[(tj, E[i])]):.2f} c {c[i]:.2f}" for i, *_ in ex))

    # ---- publication layout: PLOS Comp Biol, 7.5 in wide, Arial 8–11 pt
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    VC = ["#0072B2", "#E69F00", "#CC79A7"]                 # vessels 1–3 (Okabe–Ito), used in c only
    ACC, ARR = "#5b84b1", "white"                          # calibration accent; flow arrows (white, dark edge)
    CL_OTHER = "#009E73"                                   # centreline of the low-quality example (not in the junction)
    fig = plt.figure(figsize=(7.5, 5.3), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.0], hspace=0.42, left=0.025, right=0.985, top=0.94, bottom=0.075)
    gTop = gs[0].subgridspec(1, 2, width_ratios=[1.12, 1], wspace=0.16)

    def label(cell, letter, heading=""):
        b_ = cell.get_position(fig); fig.text(b_.x0 - 0.012, b_.y1 + 0.012, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")

    UM_PX = 1.7                                            # tile pixel size (µm)

    def scalebar(ax, um=20):
        x0_, x1_ = ax.get_xlim(); y0_, y1_ = ax.get_ylim(); L = um / UM_PX; pad = 0.06 * (x1_ - x0_)
        ax.plot([x1_ - pad - L, x1_ - pad], [y0_ - pad, y0_ - pad], color="white", lw=1.6, solid_capstyle="butt", zorder=8)
        import matplotlib.patheffects as pe
        ax.text(x1_ - pad - L / 2, y0_ - pad - 0.03 * (y0_ - y1_), f"{um} µm", color="white", fontsize=7, ha="center", va="bottom", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="0.15")])

    def crop_vessel(ax, v, col, side_R=3.6):
        cl = v["cl"] + v["off"]; R = v["R"]; vf = v["vf"]; H, W = vf.shape[:2]; _, tang, nrm = _centerline_geometry(v["cl"])
        ctr = (cl.min(0) + cl.max(0)) / 2; side = max(max(cl.max(0) - cl.min(0)) / 2 + 1.4 * R, side_R * R)
        r0, r1 = int(max(ctr[0] - side, 0)), int(min(ctr[0] + side, img.shape[0])); c0_, c1_ = int(max(ctr[1] - side, 0)), int(min(ctr[1] + side, img.shape[1]))
        cr = img[r0:r1, c0_:c1_]; ax.imshow(img, cmap="gray", vmin=np.percentile(cr, 1), vmax=np.percentile(cr, 99.5), interpolation="nearest")
        ax.plot(cl[:, 1], cl[:, 0], color=col, lw=1.4, zorder=4, solid_capstyle="round")
        for q in np.flatnonzero(v["vm"]):
            a_, b_ = cl[q] - 0.7 * R * nrm[q], cl[q] + 0.7 * R * nrm[q]; ax.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white", lw=0.5, alpha=0.85, zorder=5)
        qs = np.arange(1, len(v["cl"]) - 1, max(2, len(v["cl"]) // 6)); ys_, xs_ = [], []
        for q in qs:
            for rho_ in (-0.5 * R, 0.5 * R):
                ys_.append(int(round(v["cl"][q, 0] + rho_ * nrm[q, 0]))); xs_.append(int(round(v["cl"][q, 1] + rho_ * nrm[q, 1])))
        ys_, xs_ = np.array(ys_), np.array(xs_); okp = (ys_ >= 0) & (ys_ < H) & (xs_ >= 0) & (xs_ < W); ys_, xs_ = ys_[okp], xs_[okp]
        if len(ys_):
            U, V = vf[ys_, xs_, 0], vf[ys_, xs_, 1]; Ls = np.nanpercentile(np.hypot(U, V), 90) or 1.0
            ax.quiver(xs_ + v["off"][1], ys_ + v["off"][0], U / Ls * 0.85 * R, V / Ls * 0.85 * R, color=ARR, edgecolor="0.1", linewidth=0.4, angles="xy",
                      scale_units="xy", scale=1, width=0.014, headwidth=3.4, headlength=3.6, headaxislength=3.3, zorder=6)
        ax.set_xlim(c0_, c1_); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([]); scalebar(ax)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("0.5")

    # ---- a: measurement quality
    pair = [ex[0], ex[-1]]; gA = gTop[0].subgridspec(2, 2, height_ratios=[1, 0.6], hspace=0.05, wspace=0.08)
    for k, (i, *_r) in enumerate(pair):
        axv = fig.add_subplot(gA[0, k]); crop_vessel(axv, _r[2], VC[0] if k == 0 else CL_OTHER)
    axtab = fig.add_subplot(gA[1, :]); axtab.axis("off"); T = axtab.transAxes
    xs_ = [0.5, 0.25, 0.75]; hdr_y, rows_y = 0.87, [0.65, 0.49, 0.33]
    axtab.text(xs_[1], hdr_y, "High quality", ha="center", va="center", fontsize=8.5, fontweight="bold", transform=T)
    axtab.text(xs_[2], hdr_y, "Low quality", ha="center", va="center", fontsize=8.5, fontweight="bold", transform=T)
    tab_rows = [("$f$ (sections kept)", [f"{v['vm'].mean():.2f}  ({int(v['vm'].sum())}/{len(v['vm'])})" for i, _, _, v, _ in pair]),
                ("alignment", [f"{al(M[(tj, E[i])]):.2f}" for i, *_ in pair]), ("$c$", [f"{c[i]:.2f}" for i, *_ in pair])]
    for yv, (nm, vals) in zip(rows_y, tab_rows):
        axtab.text(xs_[0], yv, nm, ha="center", va="center", fontsize=8, color="0.35", transform=T)
        for x_, vv in zip(xs_[1:], vals):
            axtab.text(x_, yv, vv, ha="center", va="center", fontsize=8, fontweight="bold" if nm == "$c$" else "normal", transform=T)
    for yl, lw_ in ((0.98, 0.6), (0.76, 0.35), (0.22, 0.6)):
        axtab.plot([0.03, 0.97], [yl, yl], color="0.45", lw=lw_, transform=T)
    axtab.text(0.5, 0.07, "$c = c_0 + c_f\\,(1-f) + c_a\\,(1-\\mathrm{align})$", ha="center", va="center", fontsize=7.5, color="0.25", transform=T)
    label(gTop[0], "a")

    # ---- b: held-out calibration
    axB = fig.add_subplot(gTop[1]); bins = np.linspace(-5, 5, 51)
    axB.hist(np.clip(z, -5, 5), bins=bins, density=True, color="#8fb0d3", ec="white", lw=0.4)
    xg = np.linspace(-5, 5, 400); pdf = np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi); axB.plot(xg, pdf, "-", color="0.45", lw=1.0)
    axB.text(2.12, 0.085, "N(0, 1)", fontsize=7.5, color="0.4", ha="left", va="center")
    for v_ in (-1.96, 1.96):
        axB.axvline(v_, color="0.6", ls="--", lw=0.6)
    axB.set_xlim(-5, 5); axB.set_ylim(0, 0.62); axB.set_yticks([]); axB.spines[["top", "right", "left"]].set_visible(False)
    axB.set_xlabel("standardized junction residual $z_j$")
    axB.text(0.99, 0.99, f"RMS $z$ = {cov[0]:.2f}\n95% within $|z|$ < 1.96", transform=axB.transAxes, ha="right", va="top", fontsize=8)
    axS = axB.inset_axes([0.0, 0.42, 0.34, 0.58]); axS.set_xlim(0, 1); axS.set_ylim(0, 1); axS.axis("off"); axS.set_zorder(10)
    jx, jy = 0.26, 0.84
    for (dx, dy, lab_, inward) in ((-0.14, 0.08, "$y_1$", True), (-0.14, -0.08, "$y_2$", True), (0.16, 0.0, "$y_3$", False)):
        x0_, y0_ = jx + dx, jy + dy
        axS.annotate("", xy=(jx, jy) if inward else (x0_, y0_), xytext=(x0_, y0_) if inward else (jx, jy),
                     arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3", shrinkA=0, shrinkB=2, mutation_scale=5))
        axS.text(x0_ + (-0.02 if dx < 0 else 0.02), y0_, lab_, ha="right" if dx < 0 else "left", va="center", fontsize=7)
    axS.plot(jx, jy, "o", ms=3, color="0.3")
    for yv, eq in ((0.6, "$r_j = \\Sigma_e\\, s_e\\, y_e$"), (0.36, "$\\sigma_j = (\\Sigma_e\\, c_e^2\\, y_e^2 + \\sigma_0^2)^{1/2}$"), (0.12, "$z_j = r_j\\, /\\, \\sigma_j$")):
        axS.text(0.04, yv, eq, fontsize=7, va="center", color="0.15", bbox=dict(fc="white", ec="none", pad=0.6))
    label(gTop[1], "b")

    # ---- c: conservative reconciliation at one junction
    gC = gs[1].subgridspec(1, 2, width_ratios=[0.8, 1.3], wspace=0.07); axj = fig.add_subplot(gC[0])
    allcl = np.vstack([v["cl"] + v["off"] for v in vinfo]); Rm = max(v["R"] for v in vinfo)
    lo2, hi3 = allcl.min(0) - 1.2 * Rm, allcl.max(0) + 1.2 * Rm; side = max(hi3 - lo2) / 2; ctr = (lo2 + hi3) / 2
    r_lo, r_hi = int(max(ctr[0] - side, 0)), int(min(ctr[0] + side, img.shape[0])); c_lo, c_hi = int(max(ctr[1] - side, 0)), int(min(ctr[1] + side, img.shape[1]))
    cr = img[r_lo:r_hi, c_lo:c_hi]; axj.imshow(img, cmap="gray", vmin=np.percentile(cr, 1), vmax=np.percentile(cr, 99.5), interpolation="nearest")
    # raw (unsmoothed) graph skeleton — every edge's path, mapped from the graph frame into this tile's pixels
    import re as _re
    from plexusflow.config import StageConfig
    from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch
    shp = {}
    for f_ in (_paths.STAGE / "tiles" / "maxmed").glob("vid*_maxmed.tif"):
        with tifffile.TiffFile(f_) as tf_:
            shp[int(_re.match(r"vid(\d+)_", f_.name).group(1))] = tf_.pages[0].shape[:2]
    mgeom = load_tile_geometry(StageConfig.load(_paths.STAGE), video_shapes=shp)
    for u_, v_, pth in G.edges(data="path"):
        if pth is None or len(pth) < 2:
            continue
        loc = graph_to_tile_batch(np.asarray(pth, float), mgeom, tj)
        if ((loc[:, 0] > r_lo) & (loc[:, 0] < r_hi) & (loc[:, 1] > c_lo) & (loc[:, 1] < c_hi)).any():
            axj.plot(loc[:, 1], loc[:, 0], color="white", lw=0.4, alpha=0.28, zorder=3)
    jpt = graph_to_tile_batch(np.array([[G.nodes[nodes[jn]]["x"], G.nodes[nodes[jn]]["y"]]]), mgeom, tj)[0]
    for k, (v, (i_e, _s)) in enumerate(zip(vinfo, es)):
        cl = v["cl"] + v["off"]; _, _, nrm = _centerline_geometry(v["cl"])
        # colour the stretch of the raw skeleton that the measurement covers (between the projections of its centreline ends)
        sk = graph_to_tile_batch(np.asarray(G.edges[E[i_e]]["path"], float), mgeom, tj)
        sa = np.r_[0, np.cumsum(np.hypot(*np.diff(sk, axis=0).T))]; ss = np.linspace(0, sa[-1], max(int(sa[-1] / 0.5), 2))
        skd = np.c_[np.interp(ss, sa, sk[:, 0]), np.interp(ss, sa, sk[:, 1])]
        i0_, i1_ = sorted(int(np.argmin(np.linalg.norm(skd - p_, axis=1))) for p_ in (cl[0], cl[-1]))
        seg = skd[i0_:i1_ + 1]; axj.plot(seg[:, 1], seg[:, 0], color=VC[k], lw=2.2, solid_capstyle="round", zorder=4)
        q_ = len(cl) // 2; lb = cl[q_] + 1.4 * v["R"] * nrm[q_]
        axj.text(lb[1], lb[0], str(k + 1), color="white", fontsize=8, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.18", fc=VC[k], ec="white", lw=0.6))
    axj.plot(jpt[1], jpt[0], "o", ms=4.5, color="white", mec="0.1", mew=0.7, zorder=6)
    axj.set_xlim(c_lo, c_hi); axj.set_ylim(r_hi, r_lo); axj.set_xticks([]); axj.set_yticks([]); scalebar(axj)
    for sp in axj.spines.values():
        sp.set_linewidth(0.5); sp.set_color("0.5")
    axd = fig.add_subplot(gC[1])
    for k, (i, s) in enumerate(es):
        yk = len(es) - 1 - k; a_, b_ = s * y[i], s * Qp[i]
        axd.plot([a_, b_], [yk, yk], color=VC[k], lw=2.0, zorder=2, solid_capstyle="butt")
        axd.plot(b_, yk, "D", ms=7, mfc="white", mec=VC[k], mew=1.3, zorder=3)
        axd.plot(a_, yk, "o", ms=4.5, color=VC[k], mec="white", mew=0.4, zorder=4)
    axd.set_yticks(range(len(es))); axd.set_yticklabels([f"$\\bf{{{len(es) - j}}}$   $c$ = {c[es[len(es) - 1 - j][0]]:.2f}" for j in range(len(es))], fontsize=8)
    for j, tl_ in enumerate(axd.get_yticklabels()):
        tl_.set_color(VC[len(es) - 1 - j])
    axd.tick_params(axis="y", length=0, pad=3); axd.axvline(0, color="0.8", lw=0.4, zorder=1); axd.set_ylim(-0.6, len(es) - 0.4)
    lim = 1.12 * max(abs(v_) for i, s in es for v_ in (y[i], Qp[i])); axd.set_xlim(-lim, lim)
    axd.spines[["top", "right"]].set_visible(False); axd.spines["left"].set_color("0.6"); axd.set_xlabel("flow (nL/s)   (− into junction, + out)")
    axd.plot([], [], "o", ms=4, color="0.4", label="measured"); axd.plot([], [], "D", ms=5, mfc="white", mec="0.3", mew=1.1, label="reconciled")
    axd.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25, markerscale=0.85)
    axd.text(0.5, 1.02, f"$|\\Sigma_e\\, s_e\\, y_e|$ = {abs(before):.3f} nL/s   →   $|\\Sigma_e\\, s_e\\, Q′_e|$ = 0", transform=axd.transAxes, ha="center", va="bottom", fontsize=8.5)
    label(gs[1], "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6v3.png", dpi=300, facecolor="white"); fig.savefig(OUT_DIR / "fig6v3.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig6v3.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})     # submission (PLOS: 300–600 dpi)
    log(f"wrote {OUT_DIR / 'fig6v3.png'}")


if __name__ == "__main__":
    main()
