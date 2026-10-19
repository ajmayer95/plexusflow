"""Figure 6 v3 — construction of a conservation-consistent reference field Q′ (the whole-organ field is fig. 8).

  a    each measurement earns an error bar: three vessels of the junction's tile — a clean one, one with few usable cross-sections,
       one whose flow is not along the vessel (lowest alignment with ≥ 40 % kept) — image crop with the kept cross-sections (ticks)
       and the mean flow (arrows, coloured by alignment), fraction kept, alignment, c, and the measured flow ± c·|y|
  b    conservation at one junction: the junction's vessels on the tile image; each flow measured ± c·|y| and reconciled; inflow vs
       outflow stacked, measured (mismatch) and reconciled (equal)
  c    across the organ, the relative correction |Q′ − y| / |y| per measured vessel, in five equal groups of c (quartiles, 10–90 %)
Junction (b), by rule: interior node with 3 vessels — one short (1.5–3 radii), two ≥ 3 radii; measurement centrelines ≥ 0.5 R apart
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

    fig = plt.figure(figsize=(20, 11.5), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.0], hspace=0.3, left=0.04, right=0.98, top=0.95, bottom=0.06)
    c0, cf, ca = model["c0"], model["c_f"], model["c_a"]

    # ---- a: what makes a measurement uncertain
    gA = gs[0].subgridspec(1, len(ex), wspace=0.1); xmax = 1.05 * max(abs(y[j]) * (1 + c[j]) for j, *_ in ex)
    for k, (i, col, lab, v, what) in enumerate(ex):
        m = M[(tj, E[i])]; g2 = gA[k].subgridspec(1, 2, width_ratios=[1, 0.95], wspace=0.08); axv = fig.add_subplot(g2[0]); draw(axv, v, col)
        g3 = g2[1].subgridspec(3, 1, height_ratios=[1.6, 0.5, 0.35], hspace=0.05); axt = fig.add_subplot(g3[0]); axt.axis("off")
        axt.text(0, 1.0, f"{'vessel ' + lab if lab != 'x' else 'another vessel'}", fontsize=13, fontweight="bold", color=col, va="top", transform=axt.transAxes)
        axt.text(0, 0.8, what, fontsize=12, style="italic", va="top", transform=axt.transAxes)
        axt.text(0, 0.55, f"cross-sections kept  {int(v['vm'].sum())}/{len(v['vm'])}\nalignment  {al(m):.2f}", fontsize=11.5, va="top", transform=axt.transAxes, linespacing=1.5)
        axt.text(0, 0.1, f"c = {c[i]:.2f}   (± {100 * c[i]:.0f} %)", fontsize=13, fontweight="bold", va="top", transform=axt.transAxes)
        axe = fig.add_subplot(g3[1]); yy = abs(y[i]); sd = c[i] * yy
        axe.errorbar([yy], [0], xerr=[sd], fmt="o", color=col, ms=9, capsize=7, lw=3)
        axe.set_xlim(0, xmax); axe.set_ylim(-1, 1); axe.set_yticks([]); axe.spines[["left", "right", "top"]].set_visible(False)
        axe.set_xlabel("measured flow (nL/s)", fontsize=10); axe.tick_params(labelsize=9)
    fig.text(0.04, gs[0].get_position(fig).y1 + 0.012, "a   Each measurement earns an error bar from how well its flow field fits the vessel", fontsize=14, fontweight="bold")
    fig.text(0.04, gs[0].get_position(fig).y0 - 0.03, f"white ticks: cross-sections kept;  arrows: mean flow (green = along the vessel)        "
             f"c = {c0:.2f} + {cf:.2f}·(1 − fraction kept) + {ca:.2f}·(1 − alignment), calibrated on junctions imaged within one tile", fontsize=11, color="0.25")

    # ---- b: conservation at the junction
    gBC = gs[1].subgridspec(1, 2, width_ratios=[2.75, 1], wspace=0.16)
    gB = gBC[0].subgridspec(1, 3, width_ratios=[0.8, 1.25, 0.95], wspace=0.25)
    gBl = gB[0].subgridspec(1, 1); axj = fig.add_subplot(gBl[0])
    allcl = np.vstack([v["cl"] + v["off"] for v in vinfo]); Rm = max(v["R"] for v in vinfo)
    lo2, hi3 = allcl.min(0) - 1.3 * Rm, allcl.max(0) + 1.3 * Rm; side = max(hi3 - lo2) / 2; ctr = (lo2 + hi3) / 2
    r_lo, r_hi = int(max(ctr[0] - side, 0)), int(min(ctr[0] + side, img.shape[0])); c_lo, c_hi = int(max(ctr[1] - side, 0)), int(min(ctr[1] + side, img.shape[1]))
    crop = img[r_lo:r_hi, c_lo:c_hi]; axj.imshow(img, cmap="gray", vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 99.5))
    for k, v in enumerate(vinfo):
        cl = v["cl"] + v["off"]; _, _, nrm = _centerline_geometry(v["cl"]); axj.plot(cl[:, 1], cl[:, 0], color=VESSEL_COLORS[k], lw=4, solid_capstyle="round")
        q_ = len(cl) // 2; lb = cl[q_] + 1.3 * v["R"] * nrm[q_]
        axj.text(lb[1], lb[0], str(k + 1), color="white", fontsize=14, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.25", fc=VESSEL_COLORS[k], ec="white"))
    axj.set_xlim(c_lo, c_hi); axj.set_ylim(r_hi, r_lo); axj.set_xticks([]); axj.set_yticks([]); axj.set_title(f"the junction (tile {tj})", fontsize=12)

    axf = fig.add_subplot(gB[1]); order = list(range(len(es)))
    for k in order:
        i, s = es[k]; yy, qq, sd = abs(y[i]), abs(Qp[i]), c[i] * abs(y[i]); yk = len(es) - 1 - k
        axf.errorbar([yy], [yk], xerr=[sd], fmt="o", color=VESSEL_COLORS[k], ms=11, capsize=8, lw=3, zorder=3)
        axf.plot([qq], [yk], "D", ms=12, color="k", mfc="white", mew=2.2, zorder=4)
        if abs(qq - yy) > 0.15 * sd:
            axf.annotate("", xy=(qq, yk - 0.22), xytext=(yy, yk - 0.22), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="0.2"))
        axf.text(-0.02, yk, f"{k + 1}  {'out' if s * y[i] > 0 else 'in '}", transform=axf.get_yaxis_transform(), ha="right", va="center", fontsize=13, fontweight="bold", color=VESSEL_COLORS[k])
    axf.errorbar([], [], xerr=[], fmt="o", color="0.4", ms=9, capsize=6, lw=2.5, label="measured ± c·flow")
    axf.plot([], [], "D", ms=10, color="k", mfc="white", mew=2, label="reconciled")
    axf.set_yticks([]); axf.set_ylim(-0.7, len(es) - 0.3); axf.set_xlim(0, None); axf.legend(fontsize=11, loc="upper right", frameon=False)
    axf.set_xlabel("flow (nL/s)", fontsize=12); axf.spines[["top", "right", "left"]].set_visible(False)
    axf.set_title("each flow moves in proportion to its error bar", fontsize=12)

    axb = fig.add_subplot(gB[2]); X = {"measured": y, "reconciled": Qp}
    for xk, (nm, arr) in enumerate(X.items()):
        for side_, sgn, dx in (("in", -1, -0.19), ("out", 1, 0.19)):
            bot = 0.0
            for k, (i, s) in enumerate(es):
                if s * np.sign(y[i]) != sgn:
                    continue
                h = abs(arr[i]); axb.bar(xk + dx, h, bottom=bot, width=0.34, color=VESSEL_COLORS[k], ec="white", lw=1.5)
                axb.text(xk + dx, bot + h / 2, str(k + 1), ha="center", va="center", color="white", fontsize=12, fontweight="bold"); bot += h
            axb.text(xk + dx, bot + 0.002, side_, ha="center", va="bottom", fontsize=11.5)
    axb.set_xticks([0, 1]); axb.set_xticklabels([f"measured\nin ≠ out  ({abs(before) / thr:.0%} mismatch)", "reconciled\nin = out"], fontsize=12)
    axb.set_ylabel("flow (nL/s)", fontsize=12); axb.spines[["top", "right"]].set_visible(False); axb.set_xlim(-0.6, 1.6)
    axb.set_title("conservation is enforced", fontsize=12)
    fig.text(0.04, gs[1].get_position(fig).y1 + 0.04, "b   At each junction inflow must equal outflow: the mismatch goes to the least certain flows", fontsize=14, fontweight="bold")

    # ---- c: over all vessels, the correction tracks c
    axc = fig.add_subplot(gBC[1]); s_ = im & (np.abs(y) > 1e-5)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(Qp - y) / np.abs(y)
    edges = np.quantile(c[s_], np.linspace(0, 1, 6)); grp, mids = [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        sel = s_ & (c >= b0) & (c <= b1); grp.append(100 * rel[sel]); mids.append(np.median(c[sel]))
    bp = axc.boxplot(grp, positions=range(len(grp)), widths=0.6, whis=(10, 90), showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=2))
    for k, b_ in enumerate(bp["boxes"]):
        b_.set_facecolor(plt.cm.viridis_r(0.15 + 0.7 * k / (len(grp) - 1))); b_.set_alpha(0.8)
    axc.set_xticks(range(len(grp))); axc.set_xticklabels([f"{a:.2f}–\n{b:.2f}" for a, b in zip(edges[:-1], edges[1:])], fontsize=10)
    axc.set_xlabel("uncertainty c  (five equal groups of vessels)", fontsize=11.5); axc.set_ylabel("correction  |Q′ − y| / |y|  (%)", fontsize=11.5)
    axc.set_yscale("log"); axc.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); axc.spines[["top", "right"]].set_visible(False)
    axc.text(0.03, 0.97, f"{int(s_.sum())} measured vessels\nmedian correction {100 * np.nanmedian(rel[s_]):.0f} %\nboxes: quartiles; whiskers 10–90 %",
             transform=axc.transAxes, ha="left", va="top", fontsize=10.5)
    fig.text(gBC[1].get_position(fig).x0, gs[1].get_position(fig).y1 + 0.04, "c   Across the organ, corrections go to uncertain flows", fontsize=14, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6v3.png", dpi=140, bbox_inches="tight", facecolor="white"); fig.savefig(OUT_DIR / "fig6v3.pdf", bbox_inches="tight", facecolor="white")
    log(f"wrote {OUT_DIR / 'fig6v3.png'}")


if __name__ == "__main__":
    main()
