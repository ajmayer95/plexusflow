"""Figure 6 v3 — construction of a conservation-consistent reference field Q′ (the whole-organ field is fig. 8).

  top  per-tile flow → one measurement per vessel → conservative reconciliation → reference field Q′
  a    same-tile junctions calibrate the local uncertainty: standardized Kirchhoff residuals z = (net flux at a junction
       imaged within one tile) / (expected RMS imbalance from the uncertainty model), with the model fitted on the other
       tiles (5 folds by tile); standard normal overlaid; RMS, coverage within 1 and 1.96, beyond 3
  b    reconciliation enforces conservation at a junction: one junction, its vessels' measured means y (all from one tile),
       uncertainty c, reconciled Q′, and the net flux before (Σ s y) and after (Σ s Q′ = 0); on the tile image each vessel's mean
       flow (arrows, coloured by alignment) and the cross-sections the Kloosterman fit kept (ticks; recomputed with the batch settings)
  c    globally the corrections are modest: measured vs reconciled mean flow per vessel, coloured by c
  d    one measurement per vessel: a vessel seen in several tiles, the lowest-c measurement is kept (--vessel u,v;
       paper 61854,61988)
Junction (b), by rule: interior node with 3 vessels — one short (1.5–3 radii), two ≥ 3 radii; measurement centrelines ≥ 0.5 R apart
away from the ends; largest c ≥ 2 × smallest; every vessel measured with its winner from the same tile, measured
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
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import measurement_table, same_tile_junctions, carved_domain, c_of
from plexusflow.reconcile.reliability import _arrays, _fit_theta

OUT_DIR = _paths.FIGURES / "fig6v3"
T0 = time.time()
TILE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
VESSEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def log(m):
    print(f"[fig6v3 +{time.time() - T0:4.0f} s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vessel", default="61854,61988", help="vessel 'u,v' for panel d")
    ap.add_argument("--junction", type=int, default=None, help="junction node for panel b (default: rule)")
    args = ap.parse_args()
    G = _paths.load_graph(); dom = carved_domain(G); E = dom["edges"]; eix = {e: i for i, e in enumerate(E)}
    D = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    model = json.loads((_paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
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

    fig = plt.figure(figsize=(20, 14.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.2, 1.0, 1.0], width_ratios=[1, 1.35], hspace=0.38, wspace=0.18, left=0.05, right=0.97, top=0.97, bottom=0.05)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    steps = [("Per-tile flow", "optical flow → mean flux of\neach vessel in each tile"),
             ("One measurement per vessel", "the lowest-uncertainty tile (d);\nuncertainty calibrated on junctions (a)"),
             ("Conservative reconciliation", "$\\min_{B_I Q=0}\\;\\sum_e w_e\\,(Q_e-y_e)^2$\n\nweights reflect local uncertainty and flow scale (b)"),
             ("Reference field  $Q'$", "modest corrections (c);\nwhole organ: fig. 8")]
    xs = np.linspace(0.01, 0.775, len(steps)); bw = 0.215
    for k, (title, body) in enumerate(steps):
        ax0.add_patch(FancyBboxPatch((xs[k], 0.04), bw, 0.92, boxstyle="round,pad=0.006,rounding_size=0.02", fc=plt.cm.Blues(0.15 + 0.15 * k), ec="#2b5c8a", lw=1.3))
        ax0.text(xs[k] + bw / 2, 0.8, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax0.text(xs[k] + bw / 2, 0.36, body, ha="center", va="center", fontsize=10.5)
        if k < len(steps) - 1:
            ax0.annotate("", xy=(xs[k + 1] - 0.004, 0.5), xytext=(xs[k] + bw + 0.004, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2b5c8a"))

    # a
    ax = fig.add_subplot(gs[1, 0]); bins = np.linspace(-6, 6, 61)
    ax.hist(np.clip(z, -6, 6), bins=bins, density=True, color="#9ecae1", ec="white", label=f"{len(z)} same-tile junctions\n(model fitted on the other tiles)")
    xg = np.linspace(-6, 6, 400); ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1.8, label="standard normal")
    for v in (-1.96, 1.96):
        ax.axvline(v, color="0.5", ls=":", lw=1)
    ax.set_xlabel("z = observed net flux / expected imbalance from measurement uncertainty", fontsize=11); ax.set_ylabel("density", fontsize=11)
    ax.legend(fontsize=10, loc="upper left"); ax.set_xlim(-6, 6)
    ax.text(0.98, 0.97, f"RMS(z) = {cov[0]:.2f}\n|z| < 1:  {100 * cov[1]:.1f} %  (normal 68.3 %)\n|z| < 1.96:  {100 * cov[2]:.1f} %  (95 %)\n|z| > 3:  {100 * cov[3]:.1f} %  (0.3 %)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox=dict(fc="white", ec="0.6"))
    ax.set_title("a   Same-tile junctions calibrate the local uncertainty", loc="left", fontsize=13, fontweight="bold")

    # b — one image: the junction in tile tj, each vessel's mean flow (arrows, coloured by alignment) and its kept cross-sections
    from plexusflow.analyze.kloosterman import kloosterman_scalar
    from plexusflow.analyze.centerline import _centerline_geometry
    gB = gs[1, 1].subgridspec(1, 2, width_ratios=[1.15, 1], wspace=0.06); ax = fig.add_subplot(gB[0])
    kdj = pickle.load(gzip.open(_paths.STAGE / "batch_kl_inputs" / f"tile_{tj}.pkl.gz"))["kl_inputs"]
    img = tifffile.imread(str(_paths.STAGE / "tiles" / "maxmed" / f"vid{tj}_maxmed.tif")).astype(float)
    vinfo = []
    for k, (i, s) in enumerate(es):
        ki = kdj.get(E[i]) or kdj.get(E[i][::-1]); vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R = float(ki["R_prior_px"])
        kr = kloosterman_scalar(vf, cl, R, dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5,
                                R_slice_lo=0.5, R_slice_hi=2.0, min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0,
                                mask=ki["ridge_mask_roi"])                 # the batch settings (plexusflow.analyze.profile); reproduces n_kept
        vinfo.append(dict(vf=vf, cl=cl, R=R, off=np.asarray(ki["roi_offset"], float), vm=np.asarray(kr["valid_mask"], bool)))
    allcl = np.vstack([v["cl"] + v["off"] for v in vinfo]); Rmax = max(v["R"] for v in vinfo)
    lo2, hi3 = allcl.min(0) - 1.3 * Rmax, allcl.max(0) + 1.3 * Rmax; side = max(hi3 - lo2) / 2; ctr = (lo2 + hi3) / 2
    r_lo, r_hi = int(max(ctr[0] - side, 0)), int(min(ctr[0] + side, img.shape[0])); c_lo, c_hi = int(max(ctr[1] - side, 0)), int(min(ctr[1] + side, img.shape[1]))
    crop = img[r_lo:r_hi, c_lo:c_hi]
    ax.imshow(img, cmap="gray", vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 99.5))
    for k, ((i, s), v) in enumerate(zip(es, vinfo)):
        col = VESSEL_COLORS[k]; cl = v["cl"] + v["off"]; R = v["R"]; vf = v["vf"]; H, W = vf.shape[:2]; _, tang, nrm = _centerline_geometry(v["cl"])
        ax.plot(cl[:, 1], cl[:, 0], color=col, lw=3.0, alpha=0.95, zorder=4)
        for q in np.flatnonzero(v["vm"])[::2]:                         # kept cross-sections (every 2nd): short ticks across the vessel
            a_, b_ = cl[q] - 0.3 * R * nrm[q], cl[q] + 0.3 * R * nrm[q]
            ax.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white", lw=1.3, zorder=5, solid_capstyle="butt")
        qs = np.arange(1, len(v["cl"]) - 1, max(2, len(v["cl"]) // 9))     # flow sampled near the centreline only
        ys_, xs_ = [], []
        for q in qs:
            for rho_ in (-0.45 * R, 0.45 * R):
                ys_.append(int(round(v["cl"][q, 0] + rho_ * nrm[q, 0]))); xs_.append(int(round(v["cl"][q, 1] + rho_ * nrm[q, 1])))
        ys_, xs_ = np.array(ys_), np.array(xs_); okp = (ys_ >= 0) & (ys_ < H) & (xs_ >= 0) & (xs_ < W); ys_, xs_ = ys_[okp], xs_[okp]
        if len(ys_):
            nn = np.argmin((v["cl"][None, :, 0] - ys_[:, None]) ** 2 + (v["cl"][None, :, 1] - xs_[:, None]) ** 2, axis=1)
            U, V = vf[ys_, xs_, 0], vf[ys_, xs_, 1]; sp_ = np.hypot(U, V); Ls = np.nanpercentile(sp_, 90) or 1.0; L_ = 0.55 * R
            ax.quiver(xs_ + v["off"][1], ys_ + v["off"][0], U / Ls * L_, V / Ls * L_, np.abs(V * tang[nn, 0] + U * tang[nn, 1]) / np.maximum(sp_, 1e-12),
                      cmap="RdYlGn", clim=(0, 1), angles="xy", scale_units="xy", scale=1, width=0.006, headwidth=4, zorder=6)
        q_ = len(cl) // 3; lab = cl[q_] + 1.0 * R * nrm[q_]
        ax.text(lab[1], lab[0], str(k + 1), color="white", fontsize=13, fontweight="bold", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="white"))
    ends = np.array([[v["cl"][0] + v["off"], v["cl"][-1] + v["off"]] for v in vinfo]); best_j = None
    for cand in ends.reshape(-1, 2):                                  # junction ≈ the centreline ends that meet
        d_ = sum(np.min(np.linalg.norm(e_ - cand, axis=1)) for e_ in ends)
        if best_j is None or d_ < best_j[0]:
            best_j = (d_, cand)
    jpt = np.mean([e_[np.argmin(np.linalg.norm(e_ - best_j[1], axis=1))] for e_ in ends], axis=0)
    ax.plot(jpt[1], jpt[0], "o", color="white", mec="k", ms=10, zorder=9)
    ax.set_xlim(c_lo, c_hi); ax.set_ylim(r_hi, r_lo); ax.axis("off")
    ax.text(0.02, 0.02, f"tile {tj}:  arrows = mean flow (green: along the vessel);  white ticks = cross-sections kept", transform=ax.transAxes, fontsize=9.5,
            color="white", bbox=dict(fc="black", ec="none", alpha=0.6, pad=2))
    axt = fig.add_subplot(gB[1]); axt.axis("off")
    rows = []
    for k, ((i, s), v) in enumerate(zip(es, vinfo)):
        m = M[(tj, E[i])]
        rows.append([str(k + 1), f"{s * y[i]:+.3f}", f"{v['vm'].sum()}/{len(v['vm'])}", f"{m['align']:.2f}", f"{c[i]:.2f}", f"{s * Qp[i]:+.3f}"])
    tab = axt.table(cellText=rows, colLabels=["", "measured\ny (nL/s)", "kept\nsections", "align-\nment", "c", "reconciled\nQ′ (nL/s)"],
                    loc="upper center", cellLoc="center", bbox=[0.0, 0.44, 1.0, 0.48])
    tab.auto_set_font_size(False); tab.set_fontsize(11)
    for j_ in range(6):
        tab[(0, j_)].get_text().set_fontsize(9.5)
    for k in range(len(es)):
        tab[(k + 1, 0)].set_facecolor(VESSEL_COLORS[k]); tab[(k + 1, 0)].get_text().set_color("white"); tab[(k + 1, 0)].get_text().set_fontweight("bold")
    for j_ in range(6):
        tab[(0, j_)].set_facecolor("#e8eef5")
    axt.text(0.5, 0.38, "sign: + leaves the junction", ha="center", fontsize=9.5, color="0.35", transform=axt.transAxes)
    axt.text(0.5, 0.26, f"measured:  $\\sum s\\,y$ = {before:+.3f} nL/s\n({abs(before) / thr:.0%} of the flow through the junction)", ha="center", va="center", fontsize=11.5,
             transform=axt.transAxes)
    axt.text(0.5, 0.11, "reconciled:  $\\sum s\\,Q'$ = 0", ha="center", va="center", fontsize=12, fontweight="bold", transform=axt.transAxes)
    axt.text(0.5, -0.02, "the least reliable vessel (largest c)\nabsorbs most of the correction", ha="center", va="center",
             fontsize=10.5, style="italic", transform=axt.transAxes)
    fig.text(gs[1, 1].get_position(fig).x0, gs[1, 1].get_position(fig).y1 + 0.012, "b   Reconciliation enforces conservation at each junction", fontsize=13, fontweight="bold")

    # c
    ax = fig.add_subplot(gs[2, 0]); s_ = im & (np.abs(y) > 1e-5) & (np.abs(Qp) > 1e-5)
    sc = ax.scatter(np.abs(y[s_]), np.abs(Qp[s_]), c=c[s_], s=4, cmap="viridis_r", alpha=0.6, norm=mcolors.LogNorm(*np.percentile(c[s_], [1, 99])))
    fl = s_ & (np.sign(y) != np.sign(Qp)); ax.scatter(np.abs(y[fl]), np.abs(Qp[fl]), s=12, facecolors="none", edgecolors="k", lw=0.5, label=f"direction reversed ({fl.sum() / s_.sum():.1%})")
    lo_, hi2 = 1e-5, np.abs(np.r_[y[s_], Qp[s_]]).max() * 2; ax.plot([lo_, hi2], [lo_, hi2], "k--", lw=0.8)
    for k, (i, s) in enumerate(es):
        ax.plot(abs(y[i]), abs(Qp[i]), "o", ms=10, color=VESSEL_COLORS[k], mec="white", mew=1.2)
    ax.plot([], [], "o", color="0.4", ms=8, label="vessels of the junction in b")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi2); ax.set_ylim(lo_, hi2); ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("|y|  measured mean flow (nL/s)", fontsize=11); ax.set_ylabel("|Q′|  reconciled (nL/s)", fontsize=11)
    plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("uncertainty c", fontsize=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        a2 = (np.abs(Qp - y) / np.abs(y))[s_]
    ax.text(0.97, 0.05, f"median correction {np.median(a2):.0%}\n(c < 0.3: {np.median(a2[c[s_] < 0.3]):.0%})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, bbox=dict(fc="white", ec="0.6"))
    ax.set_title("c   Globally, the corrections are modest", loc="left", fontsize=13, fontweight="bold")

    # d — winner selection, thumbnails only
    u_, v_ = (int(x) for x in args.vessel.split(",")); i0 = eix.get((u_, v_), eix.get((v_, u_))); e0 = E[i0]
    ms0 = sorted([(t, m, c_of(model, m)) for (t, e), m in M.items() if e == e0], key=lambda r: r[2]); nT = len(ms0)
    gD = gs[2, 1].subgridspec(1, nT, wspace=0.08)
    from plexusflow.analyze.centerline import _centerline_geometry
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0
    for k, (t, m, cc) in enumerate(ms0):
        axv = fig.add_subplot(gD[k]); colr = TILE_COLORS[k % 4]; winner = k == 0
        p = _paths.STAGE / "batch_kl_inputs" / f"tile_{t}.pkl.gz"; kd = pickle.load(gzip.open(p))["kl_inputs"] if p.exists() else {}; ki = kd.get(e0) or kd.get(e0[::-1])
        if ki is not None:
            vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R = float(ki["R_prior_px"]); H, W = vf.shape[:2]; r0, c0 = ki["roi_offset"]
            tif = _paths.STAGE / "tiles" / "maxmed" / f"vid{t}_maxmed.tif"
            if tif.exists():
                img = tifffile.imread(str(tif)).astype(float)[r0:r0 + H, c0:c0 + W]; axv.imshow(img, cmap="gray", vmin=np.percentile(img, 2), vmax=np.percentile(img, 99.5))
            rr, cc_ = np.mgrid[0:H, 0:W]; tube = np.zeros((H, W), bool)
            for p_ in cl:
                tube |= (rr - p_[0]) ** 2 + (cc_ - p_[1]) ** 2 <= R ** 2
            axv.contour(tube.astype(float), levels=[0.5], colors="#ffd54f", linewidths=1.2)
            _, tang, _ = _centerline_geometry(cl); st_ = max(2, int(min(H, W) / 13)); ys_, xs_ = np.mgrid[0:H:st_, 0:W:st_]
            ys_, xs_ = ys_[tube[0:H:st_, 0:W:st_]], xs_[tube[0:H:st_, 0:W:st_]]
            if len(ys_):
                nn = np.argmin((cl[None, :, 0] - ys_[:, None]) ** 2 + (cl[None, :, 1] - xs_[:, None]) ** 2, axis=1)
                U, V = vf[ys_, xs_, 0], vf[ys_, xs_, 1]; sp_ = np.hypot(U, V); Ls = np.nanpercentile(sp_, 90) or 1.0
                axv.quiver(xs_, ys_, U / Ls * st_ * 1.1, V / Ls * st_ * 1.1, np.abs(V * tang[nn, 0] + U * tang[nn, 1]) / np.maximum(sp_, 1e-12), cmap="RdYlGn", clim=(0, 1),
                           angles="xy", scale_units="xy", scale=1, width=0.013)
            axv.set_xlim(-0.5, W - 0.5); axv.set_ylim(H - 0.5, -0.5)
        axv.set_xticks([]); axv.set_yticks([])
        for sp in axv.spines.values():
            sp.set_edgecolor(colr); sp.set_linewidth(4 if winner else 2)
        axv.set_title(f"tile {t}:  c = {cc:.2f}\n{'★ kept' if winner else 'not used'}   mean {m['Q']:+.3f} nL/s", fontsize=11, color=colr, fontweight="bold" if winner else "normal")
        axv.set_xlabel(f"cross-sections kept {m['f']:.0%}\nflow–vessel alignment {al(m):.2f}", fontsize=10)
    fig.text(gs[2, 1].get_position(fig).x0, gs[2, 1].get_position(fig).y1 + 0.012, f"d   One measurement per vessel: the lowest-uncertainty tile is kept  (reconciled Q′ = {Qp[i0]:+.3f} nL/s)",
             fontsize=13, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6v3.png", dpi=140, bbox_inches="tight", facecolor="white"); fig.savefig(OUT_DIR / "fig6v3.pdf", bbox_inches="tight", facecolor="white")
    log(f"wrote {OUT_DIR / 'fig6v3.png'}")


if __name__ == "__main__":
    main()
