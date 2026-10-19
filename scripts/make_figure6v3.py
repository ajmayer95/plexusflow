"""Figure 6 v3 — how the conservation-consistent reference field Q′ is built, and why the construction is defensible.

The counterpart of fig 6 v2 (per-tile scaling) for the conservative reconciliation; the whole-organ field itself is fig 8.
  top  per-tile flow → reliability + one measurement per vessel → conservation-constrained reconciliation → Q′ (fig 8)
  a    reliability (c = c0 + c_f (1 − f) + c_a (1 − align)) is calibrated on data, with held-out coverage (whole tiles
       left out, 5 folds): at simultaneously imaged (same-tile) junctions inside the carve the true flows
       conserve, so the observed net flux is measurement error; its size vs the expected RMS imbalance from the error
       model, √(Σ c² y² + σ0²) (errors ≈ c|y|, independent) — c_T, c_f, c_a are fitted so that these agree
  b    choosing among repeated measurements: one vessel seen in several tiles — each tile's image (vessel mask, ±R tube,
       mean flow coloured by |cos θ| to the centreline), Q(t), mean and reliability c; the smallest c is used (winner),
       the others are held out; the reconciled Q′ of that vessel
  c    why weights relative to the reconciled flow: held-out error (other tiles' independent measurements of the same
       vessels) and measured directions reversed, for weights relative to the measured value, uniform weights, and
       weights relative to the reconciled flow (production) — same winners
  d    what reconciliation does to the measurements: measured vs reconciled flow per vessel, coloured by c
Example vessel (b): --vessel u,v (paper: 61854,61988), else by rule: measured in ≥ 3 tiles; winner c in the best 25 %; reconciled within 25 % of the winner;
largest spread of the tile means (same direction), then largest |Q′|.
In Methods / supplement: w_e = 1/(c_e² Q_e² + s0²); the damped re-weighting and its long-run mean (fig_dc_method).
Reads the graph and results/{dc_reconciled_v2.pkl, reliability_model_v1.json}; writes figures/fig6v3/fig6v3.{png,pdf}.
"""
from __future__ import annotations
import sys
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
from plexusflow.reconcile.network import project, EPS_UNIMAGED
from plexusflow.reconcile.harmonics import phasors, FS
from plexusflow.reconcile.reliability import _arrays, _fit_theta

OUT_DIR = _paths.FIGURES / "fig6v3"
T0 = time.time()
TILE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def log(m):
    print(f"[fig6v3 +{time.time() - T0:4.0f} s] {m}", flush=True)


def main():
    G = _paths.load_graph(); dom = carved_domain(G); E = dom["edges"]; eix = {e: i for i, e in enumerate(E)}; nE = len(E)
    D = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    model = json.loads((_paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    M = measurement_table(G); tf0 = G.graph["tile_f0_piv"]
    y, im, Qp, s0, win = D["Qm"], D["imaged"], D["Qp"], D["sigma0_floor"], D["tile"]; c = np.where(im, D["c"], 1.0)
    meas_by_e = {}
    for (t, e), m in M.items():
        if e in eix:
            meas_by_e.setdefault(eix[e], []).append((t, m, c_of(model, m)))
    log(f"{_paths.STAGE.name}: {nE} carved vessels, {int(im.sum())} measured, {len(M)} measurements")

    # example vessel (b): --vessel u,v, else by rule
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--vessel", default=None, help="example vessel 'u,v' for panel b (default: rule)")
    args = ap.parse_args()
    c25 = np.percentile(c[im], 25); best = None
    for i, ms in meas_by_e.items():
        if len(ms) < 3 or not im[i] or c[i] > c25 or abs(Qp[i] - y[i]) > 0.25 * abs(y[i]):
            continue
        qs = np.array([m["Q"] for _, m, _ in ms])
        if not (np.all(qs > 0) or np.all(qs < 0)):
            continue
        key = (round(np.log(np.abs(qs).max() / np.abs(qs).min()), 1), abs(Qp[i]))
        if best is None or key > best[0]:
            best = (key, i)
    i0 = best[1]
    if args.vessel:
        u_, v_ = (int(x) for x in args.vessel.split(",")); i0 = eix.get((u_, v_), eix.get((v_, u_)))
        assert i0 is not None and len(meas_by_e.get(i0, [])) >= 2, f"vessel {args.vessel} is not a carved vessel with ≥ 2 measurements"
    e0 = E[i0]; ms0 = sorted(meas_by_e[i0], key=lambda r: r[2])
    log(f"  example vessel {e0}: " + ", ".join(f"tile {t} y {m['Q']:+.3f} c {cc:.2f}" for t, m, cc in ms0) + f";  Q′ {Qp[i0]:+.3f}")

    # (a) calibration
    pred, obs = [], []
    for t in sorted({t for t, _ in M}):
        meas = {e for (tt, e) in M if tt == t and e in eix}
        for j in same_tile_junctions(G, t, meas):
            mm = [M[(t, e)] for e, _ in j]
            pred.append(np.sqrt(sum((m["Q"] * c_of(model, m)) ** 2 for m in mm) + model["sigma0"] ** 2)); obs.append(sum(s * M[(t, e)]["Q"] for e, s in j))
    pred, obs = np.array(pred), np.array(obs)
    # held-out coverage: fit c0, c_f, c_a with whole tiles left out (5 folds), z = net flux / expected RMS imbalance
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
    zc = np.concatenate(zs); cov = (np.sqrt(np.mean(zc ** 2)), np.mean(np.abs(zc) < 1), np.mean(np.abs(zc) < 1.96), np.mean(np.abs(zc) > 3))
    log(f"  held-out coverage (by tile, {len(JJ)} junctions): RMS(z) {cov[0]:.3f}, |z|<1 {cov[1]:.3f}, |z|<1.96 {cov[2]:.3f}, |z|>3 {cov[3]:.4f}")

    # (c) weighting ablation (same winners)
    held = [(eix[k[1]], m["Q"]) for k, m in M.items() if k[1] in eix and im[eix[k[1]]] and k[0] != win[eix[k[1]]]]
    hi = np.array([h[0] for h in held]); hq = np.array([h[1] for h in held])
    def score(Q):
        ok = (np.abs(Q[hi]) > 0) & (np.abs(hq) > 0); big = im & (np.abs(y) > 1e-3)
        return np.median(np.abs(np.log(np.abs(Q[hi][ok]) / np.abs(hq[ok])))), np.mean(np.sign(Q[big]) != np.sign(y[big]))
    wsolve = lambda w: project(y, np.where(im, w / np.median(w[im]), EPS_UNIMAGED), dom)
    variants = [("relative to the\nmeasured value", wsolve(1 / (c ** 2 * y ** 2 + s0 ** 2))), ("uniform\n(absolute error)", wsolve(np.ones(nE))),
                ("relative to the\nreconciled flow", Qp)]
    vscore = [score(Q) for _, Q in variants]
    log("  (c) " + "; ".join(f"{lab.replace(chr(10), ' ')}: held-out {s[0]:.3f}, reversed {s[1]:.1%}" for (lab, _), s in zip(variants, vscore)))

    kl = {}
    def kl_of(t, e):
        if t not in kl:
            p = _paths.STAGE / "batch_kl_inputs" / f"tile_{t}.pkl.gz"; kl[t] = pickle.load(gzip.open(p))["kl_inputs"] if p.exists() else {}
        return kl[t].get(e) or kl[t].get((e[1], e[0]))
    from plexusflow.analyze.centerline import _centerline_geometry
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0

    fig = plt.figure(figsize=(20, 15.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.22, 1, 1], width_ratios=[1, 1.55], hspace=0.36, wspace=0.2, left=0.06, right=0.97, top=0.97, bottom=0.05)
    # top strip
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    steps = [("Per-tile flow", "optical flow → flux of each\nvessel in each tile"),
             ("Reliability + winner", "data-calibrated reliability (a);\nmost reliable measurement (b)"),
             ("Conservative reconciliation", "$\\min_{B_I Q=0}\\;\\sum_e w_e\\,(Q_e-y_e)^2$\n\nrelative-error weights updated from Q (c)"),
             ("Reference field  $Q'$", "measurements corrected, not\nrewritten (d); whole organ: fig. 8")]
    xs = np.linspace(0.01, 0.775, len(steps)); bw = 0.215
    for k, (title, body) in enumerate(steps):
        ax0.add_patch(FancyBboxPatch((xs[k], 0.04), bw, 0.92, boxstyle="round,pad=0.006,rounding_size=0.02", fc=plt.cm.Blues(0.15 + 0.15 * k), ec="#2b5c8a", lw=1.3))
        ax0.text(xs[k] + bw / 2, 0.8, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax0.text(xs[k] + bw / 2, 0.36, body, ha="center", va="center", fontsize=10.5)
        if k < len(steps) - 1:
            ax0.annotate("", xy=(xs[k + 1] - 0.004, 0.5), xytext=(xs[k] + bw + 0.004, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2b5c8a"))

    # a
    ax = fig.add_subplot(gs[1, 0]); ok = pred > 0; bins = np.quantile(pred[ok], np.linspace(0, 1, 13)); cen, rms, lo, hi_ = [], [], [], []
    rng = np.random.default_rng(0)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        s_ = ok & (pred >= b0) & (pred < b1)
        if s_.sum() > 20:
            cen.append(np.median(pred[s_])); rms.append(np.sqrt(np.mean(obs[s_] ** 2)))
            bs = [np.sqrt(np.mean(rng.choice(obs[s_], s_.sum()) ** 2)) for _ in range(200)]; lo.append(np.percentile(bs, 5)); hi_.append(np.percentile(bs, 95))
    ax.scatter(pred[ok], np.abs(obs[ok]), s=3, color="0.75", alpha=0.4, label="observed |net flux|, one junction")
    ax.errorbar(cen, rms, yerr=[np.array(rms) - lo, np.array(hi_) - rms], fmt="o-", color="#d62728", lw=2, ms=6, label="observed RMS imbalance (binned, 90 % CI)")
    l0, l1 = np.percentile(pred[ok], [1, 99.5]); ax.plot([l0, l1], [l0, l1], "k--", lw=1, label="1 : 1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(l0 * 0.05, l1 * 5); ax.legend(fontsize=10, loc="upper left")
    fm = "c = {:.2f} + {:.2f} (1 − f) + {:.2f} (1 − align)".format(model.get("c0", np.nan), model["c_f"], model["c_a"])
    ax.text(0.97, 0.04, f"{fm}\nwhole tiles held out:  {100 * cov[1]:.1f} % within 1σ (Gaussian 68.3 %)\n"
            f"{100 * cov[2]:.1f} % within 1.96σ (95 %);  {100 * cov[3]:.1f} % beyond 3σ (0.3 %)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, bbox=dict(fc="white", ec="0.6"))
    ax.set_xlabel("expected RMS junction imbalance from measurement uncertainty (nL/s)\n" "$\\sqrt{\\sum_{e\\in j} c_e^2\\,y_e^2+\\sigma_0^2}$   (errors ≈ c|y|, independent; true flows conserve)", fontsize=11)
    ax.set_ylabel("observed |net flux| at the junction (nL/s)", fontsize=11)
    ax.set_title(f"a   Reliability is calibrated on {len(pred)} simultaneously imaged junctions", loc="left", fontsize=13, fontweight="bold")

    # b
    nT = len(ms0); gB = gs[1, 1].subgridspec(2, nT, height_ratios=[0.9, 1], hspace=0.32, wspace=0.1)
    axq = fig.add_subplot(gB[1, :])
    for k, (t, m, cc) in enumerate(ms0):
        colr = TILE_COLORS[k % len(TILE_COLORS)]; winner = k == 0
        axv = fig.add_subplot(gB[0, k]); ki = kl_of(t, e0)
        if ki is not None:
            vf = np.asarray(ki["vfield_mean"], float); cl = np.asarray(ki["cl_used"], float); R = float(ki["R_prior_px"]); H, W = vf.shape[:2]; r0, c0 = ki["roi_offset"]
            tif = _paths.STAGE / "tiles" / "maxmed" / f"vid{t}_maxmed.tif"
            if tif.exists():
                img = tifffile.imread(str(tif)).astype(float)[r0:r0 + H, c0:c0 + W]; axv.imshow(img, cmap="gray", vmin=np.percentile(img, 2), vmax=np.percentile(img, 99.5))
            if ki.get("ridge_mask_roi") is not None:
                axv.contour(np.asarray(ki["ridge_mask_roi"], float), levels=[0.5], colors="#4fc3f7", linewidths=0.8, linestyles="--")
            rr, cc_ = np.mgrid[0:H, 0:W]; tube = np.zeros((H, W), bool)
            for p_ in cl:
                tube |= (rr - p_[0]) ** 2 + (cc_ - p_[1]) ** 2 <= R ** 2
            axv.contour(tube.astype(float), levels=[0.5], colors="#ffd54f", linewidths=1.2)
            _, tang, _ = _centerline_geometry(cl); st_ = max(2, int(min(H, W) / 14)); ys_, xs_ = np.mgrid[0:H:st_, 0:W:st_]
            ys_, xs_ = ys_[tube[0:H:st_, 0:W:st_]], xs_[tube[0:H:st_, 0:W:st_]]
            if len(ys_):
                nn = np.argmin((cl[None, :, 0] - ys_[:, None]) ** 2 + (cl[None, :, 1] - xs_[:, None]) ** 2, axis=1)
                U, V = vf[ys_, xs_, 0], vf[ys_, xs_, 1]; sp_ = np.hypot(U, V); Ls = np.nanpercentile(sp_, 90) or 1.0
                cos = np.abs(V * tang[nn, 0] + U * tang[nn, 1]) / np.maximum(sp_, 1e-12)
                axv.quiver(xs_, ys_, U / Ls * st_ * 1.1, V / Ls * st_ * 1.1, cos, cmap="RdYlGn", clim=(0, 1), angles="xy", scale_units="xy", scale=1, width=0.012)
            axv.set_xlim(-0.5, W - 0.5); axv.set_ylim(H - 0.5, -0.5)
        axv.set_xticks([]); axv.set_yticks([])
        for s in axv.spines.values():
            s.set_edgecolor(colr); s.set_linewidth(4 if winner else 2)
        axv.set_title(f"tile {t}:  c = {cc:.2f}{'   ★ used' if winner else '   held out'}\nkept {m['f']:.0%} of cross-sections, alignment {al(m):.2f}", fontsize=10.5,
                      color=colr, fontweight="bold" if winner else "normal")
        mr = next((x for x in G.edges[e0].get("measurements_piv") or [] if int(x["tile_id"]) == t and x.get("Q_t") is not None), None)
        if mr is not None:
            qt = np.asarray(mr["Q_t"], float); tt = np.arange(len(qt)) / FS          # Q_t is stored signed u→v (graph edge order)
            a0_, Hh = phasors(qt, tf0[t]); fit = a0_ + sum((Hh[j] * np.exp(2j * np.pi * (j + 1) * tf0[t] * tt)).real for j in range(3))
            axq.plot(tt, qt, color=colr, lw=0.5, alpha=0.3); axq.plot(tt, fit, color=colr, lw=3 if winner else 1.6)
        axq.axhline(m["Q"], color=colr, ls="--", lw=2.2 if winner else 1.2, label=f"tile {t}: mean {m['Q']:+.3f} nL/s{'  (used)' if winner else ''}")
    axq.axhline(Qp[i0], color="k", lw=2.4, label=f"after reconciliation  Q′ = {Qp[i0]:+.3f}"); axq.axhline(0, color="k", lw=0.4)
    axq.set_xlabel("time (s)", fontsize=11); axq.set_ylabel("Q (nL/s)", fontsize=11); axq.legend(fontsize=10, loc="lower right", ncol=2, framealpha=0.92)
    axq.set_title("Q(t) in each tile (thin: raw, thick: mean + 3 harmonics) and its mean", fontsize=10.5)
    fig.text(gs[1, 1].get_position(fig).x0, gs[1, 1].get_position(fig).y1 + 0.045, f"b   One vessel, {nT} tiles: the most reliable measurement is used",
             fontsize=13, fontweight="bold")

    # c
    ax = fig.add_subplot(gs[2, 0]); xk = np.arange(3); cols = ["#9467bd", "0.55", "#2ca02c"]
    ax.bar(xk - 0.2, [s[0] for s in vscore], 0.38, color=cols)
    ax2 = ax.twinx(); ax2.bar(xk + 0.2, [100 * s[1] for s in vscore], 0.38, color=cols, alpha=0.45, hatch="//", edgecolor="k")
    for x, s in zip(xk, vscore):
        ax.text(x - 0.2, s[0] * 1.02, f"{s[0]:.2f}", ha="center", va="bottom", fontsize=11); ax2.text(x + 0.2, 100 * s[1] * 1.02, f"{100 * s[1]:.0f} %", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xk); ax.set_xticklabels([v[0] for v in variants], fontsize=11); ax.set_xlabel("weights", fontsize=11)
    ax.set_ylabel("held-out error vs other tiles (solid)", fontsize=11); ax2.set_ylabel("measured directions reversed, % (hatched)", fontsize=11)
    ax.set_ylim(0, max(s[0] for s in vscore) * 1.25); ax2.set_ylim(0, max(100 * s[1] for s in vscore) * 1.3)
    ax.set_title("c   Weights relative to the reconciled flow", loc="left", fontsize=13, fontweight="bold")

    # d
    ax = fig.add_subplot(gs[2, 1]); s_ = im & (np.abs(y) > 1e-5) & (np.abs(Qp) > 1e-5)
    sc = ax.scatter(np.abs(y[s_]), np.abs(Qp[s_]), c=c[s_], s=4, cmap="viridis_r", alpha=0.6, norm=mcolors.LogNorm(*np.percentile(c[s_], [1, 99])))
    fl = s_ & (np.sign(y) != np.sign(Qp)); ax.scatter(np.abs(y[fl]), np.abs(Qp[fl]), s=12, facecolors="none", edgecolors="k", lw=0.5, label=f"direction reversed ({fl.sum() / s_.sum():.1%})")
    lo_, hi2 = 1e-5, np.abs(np.r_[y[s_], Qp[s_]]).max() * 2; ax.plot([lo_, hi2], [lo_, hi2], "k--", lw=0.8)
    ax.plot(abs(y[i0]), abs(Qp[i0]), "o", ms=12, color="#1f77b4", mec="k", mew=1.2, label="vessel in b")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi2); ax.set_ylim(lo_, hi2); ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("|y|  measured (nL/s)", fontsize=11); ax.set_ylabel("|Q′|  reconciled (nL/s)", fontsize=11); plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("reliability c", fontsize=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        a2 = (np.abs(Qp - y) / np.abs(y))[s_]
    ax.text(0.97, 0.05, f"median adjustment {np.median(a2):.0%}\nreliable (c < 0.3): {np.median(a2[c[s_] < 0.3]):.0%}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, bbox=dict(fc="white", ec="0.6"))
    ax.set_title("d   Reconciliation corrects measurements rather than rewriting them", loc="left", fontsize=13, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6v3.png", dpi=140, bbox_inches="tight", facecolor="white"); fig.savefig(OUT_DIR / "fig6v3.pdf", bbox_inches="tight", facecolor="white")
    log(f"wrote {OUT_DIR / 'fig6v3.png'}")


if __name__ == "__main__":
    main()
