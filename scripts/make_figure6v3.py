"""Figure 6 v3 — DC reconciliation: reliability score, one measurement per vessel, flow conservation (any stage).

The counterpart of fig 6 v2 (per-tile scaling of a shared vessel) for the conservative reconciliation.
  A  the reliability score c = c_T + c_f (1 − f) + c_a (1 − align) (tile, kept cross-section fraction, flow–centreline
     alignment), calibrated on simultaneously imaged junctions: observed net flux vs √(Σ y² c² + σ0²)
  B  one vessel seen in several tiles: each tile's image (vessel mask, ±R tube, mean flow coloured by |cos θ|), its Q(t)
     and mean, its c — the smallest c is the vessel's measurement (winner); the others are held out
  C  flow conservation around that vessel: the vessel and the vessels meeting it at both ends, labelled measured y →
     reconciled Q′; bars: net flux at the two junctions before (measured) and after (reconciled = 0)
  D  measured vs reconciled flow for all measured vessels, coloured by c
  E  the reference field |Q′| (unmeasured and conservation-dominated vessels faint), location of B / C
Example vessel (rule): measured in ≥ 3 tiles; both end junctions interior with every incident vessel measured; winner c
in the best 25 %; reconciled within 25 % of the winner; largest spread of the tile means (same direction) — then largest |Q′|.
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
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import measurement_table, same_tile_junctions, carved_domain, c_of
from plexusflow.reconcile.network import emergent
from plexusflow.reconcile.harmonics import phasors, FS

IMPUTED_RATIO = 5.0
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
    y, im, Qp, win = D["Qm"], D["imaged"], D["Qp"], D["tile"]; c = np.where(im, D["c"], 1.0)
    nodes = dom["nodes"]; nix = {n: k for k, n in enumerate(nodes)}; bnd = set(int(b) for b in dom["bnd"])
    inc = {}
    for i, (u, v) in enumerate(E):
        inc.setdefault(nix[u], []).append((i, 1.0)); inc.setdefault(nix[v], []).append((i, -1.0))
    meas_by_e = {}
    for (t, e), m in M.items():
        if e in eix:
            meas_by_e.setdefault(eix[e], []).append((t, m, c_of(model, m)))
    log(f"{_paths.STAGE.name}: {nE} carved vessels, {int(im.sum())} measured, {len(M)} measurements")

    # ---- example vessel
    cw = c[im]; c25 = np.percentile(cw, 25); best = None
    for i, ms in meas_by_e.items():
        if len(ms) < 3 or not im[i] or c[i] > c25 or abs(Qp[i] - y[i]) > 0.25 * abs(y[i]):
            continue
        ends = [nix[E[i][0]], nix[E[i][1]]]
        if any(n in bnd or len(inc[n]) < 3 or not all(im[j] for j, _ in inc[n]) for n in ends):
            continue
        qs = np.array([m["Q"] for _, m, _ in ms])
        if not (np.all(qs > 0) or np.all(qs < 0)):
            continue
        key = (round(np.log(np.abs(qs).max() / np.abs(qs).min()), 1), abs(Qp[i]))
        if best is None or key > best[0]:
            best = (key, i)
    if best is None:
        sys.exit("no vessel matches the example rule")
    i0 = best[1]; e0 = E[i0]; ends = [nix[e0[0]], nix[e0[1]]]
    ms0 = sorted(meas_by_e[i0], key=lambda r: r[2])
    log(f"  example vessel {e0}: " + ", ".join(f"tile {t} y {m['Q']:+.3f} c {cc:.2f}" for t, m, cc in ms0) + f";  Q′ {Qp[i0]:+.3f}")

    # ---- calibration
    pred, obs = [], []
    for t in sorted({t for t, _ in M}):
        meas = {e for (tt, e) in M if tt == t and e in eix}
        for j in same_tile_junctions(G, t, meas):
            mm = [M[(t, e)] for e, _ in j]
            pred.append(np.sqrt(sum((m["Q"] * c_of(model, m)) ** 2 for m in mm) + model["sigma0"] ** 2)); obs.append(sum(s * M[(t, e)]["Q"] for e, s in j))
    pred, obs = np.array(pred), np.array(obs)

    mos = tifffile.imread(str(_paths.MOSAIC)); mos = mos.mean(-1) if mos.ndim == 3 else mos; Hm = mos.shape[0]
    segs = []
    for e in E:
        d = G.edges[e]; p = np.asarray(d["path"], float) if d.get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e])
        segs.append(np.column_stack([p[:, 0], Hm - p[:, 1]]))
    kl = {}
    def kl_of(t, e):
        if t not in kl:
            p = _paths.STAGE / "batch_kl_inputs" / f"tile_{t}.pkl.gz"; kl[t] = pickle.load(gzip.open(p))["kl_inputs"] if p.exists() else {}
        return kl[t].get(e) or kl[t].get((e[1], e[0]))
    from plexusflow.analyze.centerline import _centerline_geometry

    fig = plt.figure(figsize=(22, 15.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1, 1], hspace=0.28, wspace=0.24, left=0.05, right=0.97, top=0.95, bottom=0.05)

    # A — reliability score
    gA = gs[0, 0].subgridspec(2, 1, height_ratios=[0.5, 1], hspace=0.12)
    axf = fig.add_subplot(gA[0]); axf.axis("off")
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0
    parts = np.array([(model["c_T"][m["tile"]], model["c_f"] * (1 - m["f"]), model["c_a"] * (1 - al(m))) for (t, e), m in M.items() if e in eix])
    fr = parts.mean(0) / parts.sum(1).mean()
    axf.text(0.0, 0.95, "$c\\;=\\;c_T\\;+\\;c_f\\,(1-f)\\;+\\;c_a\\,(1-\\mathrm{align})$", fontsize=17, va="top")
    axf.text(0.0, 0.52, f"$c_T$  tile (imaging) term, median {np.median(list(model['c_T'].values())):.2f}  — share {fr[0]:.0%}\n"
             f"$f$     fraction of cross-sections kept ($c_f$ = {model['c_f']:.2f})  — share {fr[1]:.0%}\n"
             f"align  flow parallel to the centreline ($c_a$ = {model['c_a']:.2f})  — share {fr[2]:.0%}\n"
             "c ≈ relative error of the mean flux;  fitted to junction mass balance (below)", fontsize=11, va="top", linespacing=1.6)
    ax = fig.add_subplot(gA[1]); ok = pred > 0; bins = np.quantile(pred[ok], np.linspace(0, 1, 13)); cen, rms, lo, hi_ = [], [], [], []
    rng = np.random.default_rng(0)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        s_ = ok & (pred >= b0) & (pred < b1)
        if s_.sum() > 20:
            cen.append(np.median(pred[s_])); rms.append(np.sqrt(np.mean(obs[s_] ** 2)))
            bs = [np.sqrt(np.mean(rng.choice(obs[s_], s_.sum()) ** 2)) for _ in range(200)]; lo.append(np.percentile(bs, 5)); hi_.append(np.percentile(bs, 95))
    ax.scatter(pred[ok], np.abs(obs[ok]), s=3, color="0.75", alpha=0.4, label="|net flux| at a junction")
    ax.errorbar(cen, rms, yerr=[np.array(rms) - lo, np.array(hi_) - rms], fmt="o-", color="#d62728", lw=2, ms=6, label="RMS (binned, 90 % CI)")
    l0, l1 = np.percentile(pred[ok], [1, 99.5]); ax.plot([l0, l1], [l0, l1], "k--", lw=1, label="1 : 1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(l0 * 0.05, l1 * 5); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("predicted imbalance  $\\sqrt{\\sum_{e\\in j} y_e^2 c_e^2+\\sigma_0^2}$  (nL/s)"); ax.set_ylabel("observed net flux (nL/s)")
    ax.set_title(f"{len(pred)} simultaneously imaged junctions", fontsize=10.5)
    axf.set_title("A   Reliability score c", loc="left", fontsize=13.5, fontweight="bold")

    # B — one vessel, several tiles
    nT = len(ms0); gB = gs[0, 1:].subgridspec(2, nT, height_ratios=[0.85, 1], hspace=0.3, wspace=0.12)
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
        axv.set_title(f"tile {t}{'   ★ winner' if winner else '   held out'}\nc = {cc:.2f}  (f {m['f']:.2f}, align {al(m):.2f})", fontsize=10.5,
                      color=colr, fontweight="bold" if winner else "normal")
        mr = next((x for x in G.edges[e0].get("measurements_piv") or [] if int(x["tile_id"]) == t and x.get("Q_t") is not None), None)
        if mr is not None:
            qt = np.asarray(mr["Q_t"], float); tt = np.arange(len(qt)) / FS          # Q_t is stored signed u→v (graph edge order)
            a0_, Hh = phasors(qt, tf0[t]); fit = a0_ + sum((Hh[j] * np.exp(2j * np.pi * (j + 1) * tf0[t] * tt)).real for j in range(3))
            axq.plot(tt, qt, color=colr, lw=0.5, alpha=0.35); axq.plot(tt, fit, color=colr, lw=3 if winner else 1.6)
        axq.axhline(m["Q"], color=colr, ls="--", lw=2.2 if winner else 1.2, label=f"tile {t}: mean {m['Q']:+.3f}, c {cc:.2f}{'  (winner y)' if winner else ''}")
    axq.axhline(Qp[i0], color="k", lw=2.2, label=f"reconciled Q′ = {Qp[i0]:+.3f}"); axq.axhline(0, color="k", lw=0.4)
    axq.set_xlabel("time (s)"); axq.set_ylabel("Q (nL/s), signed along the vessel"); axq.legend(fontsize=9.5, loc="upper right", ncol=2, framealpha=0.92)
    axq.set_title("each tile's Q(t) (thin: 250 Hz, thick: mean + 3 harmonics at that tile's heart rate) and mean", fontsize=10.5)
    fig.text(gs[0, 1].get_position(fig).x0, 0.985, f"B   One vessel, {nT} tile measurements: the most reliable one is used", fontsize=13.5, fontweight="bold")

    # C — conservation around the vessel
    gC = gs[1, 0].subgridspec(1, 2, width_ratios=[2.3, 1], wspace=0.3); ax = fig.add_subplot(gC[0])
    loc = sorted({j for n in ends for j, _ in inc[n]}); pts = np.vstack([segs[j] for j in loc]); cx, cy = pts.mean(0)
    hw = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.55 + 15
    ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98)); ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy + hw, cy - hw); ax.axis("off")
    for j in loc:
        p = segs[j]; ax.plot(p[:, 0], p[:, 1], color="#d62728" if j == i0 else "#ffd54f", lw=4 if j == i0 else 2.5, solid_capstyle="round")
        m_ = p[len(p) // 2]; d_ = p[min(len(p) - 1, len(p) // 2 + 1)] - p[max(0, len(p) // 2 - 1)]
        ax.annotate("", xy=m_ + 0.18 * hw * d_ / max(np.linalg.norm(d_), 1e-9) * np.sign(Qp[j]), xytext=m_, arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
        nrm_ = np.array([-d_[1], d_[0]]) / max(np.linalg.norm(d_), 1e-9); off = m_ + nrm_ * 0.16 * hw
        ax.text(*off, f"{abs(y[j]):.3f}\n→ {abs(Qp[j]):.3f}", color="white", fontsize=9.5, fontweight="bold", ha="center", va="center",
                bbox=dict(fc="#d62728" if j == i0 else "black", ec="none", alpha=0.7, pad=1.5))
    for n in ends:
        ax.plot(G.nodes[nodes[n]]["x"], Hm - G.nodes[nodes[n]]["y"], "o", color="white", mec="k", ms=9)
    ax.set_title("vessels at both ends of the example (red):\n|measured y| → |reconciled Q′| (nL/s); arrow = reconciled direction", fontsize=10)
    axb = fig.add_subplot(gC[1]); lab, before, after = [], [], []
    for k, n in enumerate(ends):
        before.append(sum(s * y[j] for j, s in inc[n])); after.append(sum(s * Qp[j] for j, s in inc[n])); lab.append(f"junction {k + 1}")
        throughput = sum(abs(y[j]) for j, _ in inc[n]) / 2
        log(f"  junction {k + 1}: net flux measured {before[-1]:+.4f}, reconciled {after[-1]:+.1e} nL/s (throughput {throughput:.3f})")
    xk = np.arange(len(ends)); axb.bar(xk - 0.18, np.abs(before), 0.34, color="#1f77b4", label="measured y")
    axb.bar(xk + 0.18, np.maximum(np.abs(after), 1e-12), 0.34, color="k", label="reconciled Q′")
    for x_, b_ in zip(xk, before):
        axb.text(x_ - 0.18, abs(b_) * 1.03, f"{abs(b_):.3f}", ha="center", va="bottom", fontsize=10)
    for x_ in xk:
        axb.text(x_ + 0.18, 0.0, "0", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axb.set_xticks(xk); axb.set_xticklabels(lab); axb.set_ylabel("|net flux| into the junction (nL/s)"); axb.legend(fontsize=9, loc="upper right")
    axb.set_ylim(0, max(np.abs(before)) * 1.25); axb.set_title("mass imbalance at the\ntwo junctions", fontsize=10.5)
    fig.text(gs[1, 0].get_position(fig).x0, gs[1, 0].get_position(fig).y1 + 0.012, "C   Flow conservation adjusts the field", fontsize=13.5, fontweight="bold")

    # D — measured vs reconciled
    ax = fig.add_subplot(gs[1, 1]); s_ = im & (np.abs(y) > 1e-5) & (np.abs(Qp) > 1e-5)
    sc = ax.scatter(np.abs(y[s_]), np.abs(Qp[s_]), c=c[s_], s=4, cmap="viridis_r", alpha=0.6, norm=mcolors.LogNorm(*np.percentile(c[s_], [1, 99])))
    fl = s_ & (np.sign(y) != np.sign(Qp)); ax.scatter(np.abs(y[fl]), np.abs(Qp[fl]), s=12, facecolors="none", edgecolors="k", lw=0.5, label=f"direction changed ({fl.sum() / s_.sum():.1%})")
    lo_, hi2 = 1e-5, np.abs(np.r_[y[s_], Qp[s_]]).max() * 2; ax.plot([lo_, hi2], [lo_, hi2], "k--", lw=0.8)
    for j in loc:
        ax.plot(abs(y[j]), abs(Qp[j]), "o", ms=11 if j == i0 else 7, color="#d62728" if j == i0 else "#ffd54f", mec="k", mew=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi2); ax.set_ylim(lo_, hi2); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("|y|  measured (winner, nL/s)"); ax.set_ylabel("|Q′|  reconciled (nL/s)"); plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label("reliability c")
    with np.errstate(divide="ignore", invalid="ignore"):
        a2 = (np.abs(Qp - y) / np.abs(y))[s_]
    ax.text(0.97, 0.05, f"median adjustment {np.median(a2):.0%}\n(c < 0.3: {np.median(a2[c[s_] < 0.3]):.0%})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, bbox=dict(fc="white", ec="0.6"))
    ax.set_title("D   Reliable measurements are kept, unreliable ones corrected", loc="left", fontsize=13.5, fontweight="bold")

    # E — reference field
    ax = fig.add_subplot(gs[1, 2]); faint = ~im | (np.abs(Qp) > IMPUTED_RATIO * np.abs(y))
    a_ = np.abs(Qp); vmin, vmax = max(np.percentile(a_[im], 2), 1e-4), np.percentile(a_[im], 98); nrm = mcolors.LogNorm(vmin, vmax)
    ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98)); ax.axis("off")
    ax.add_collection(LineCollection(segs, colors=[(*plt.cm.viridis(nrm(np.clip(a_[i], vmin, vmax)))[:3], 0.3 if faint[i] else 1.0) for i in range(nE)],
                                     linewidths=np.where(faint, 0.6, 1.2)))
    f = emergent(Qp, dom)
    for k in dom["bnd"]:
        if abs(f[k]) > 0.02:
            ax.scatter(G.nodes[nodes[k]]["x"], Hm - G.nodes[nodes[k]]["y"], s=20 + 60 * min(abs(f[k]) / 2, 1.5), color="#d62728" if f[k] > 0 else "#1f5fa8",
                       edgecolor="white", lw=0.8, zorder=5)
    mid = segs[i0][len(segs[i0]) // 2]; ax.plot(*mid, "s", ms=16, mfc="none", mec="#d62728", mew=2.5, zorder=8)
    ax.scatter([], [], s=50, color="#d62728", label="boundary injection"); ax.scatter([], [], s=50, color="#1f5fa8", label="boundary withdrawal")
    ax.plot([], [], color=plt.cm.viridis(0.6), alpha=0.3, lw=1, label=f"faint: unmeasured or conservation-dominated (|Q′| > {IMPUTED_RATIO:g}|y|)")
    ax.plot([], [], "s", mfc="none", mec="#d62728", mew=2, ms=10, label="example vessel (B, C)")
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=True)
    plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="viridis"), ax=ax, fraction=0.04, pad=0.01).set_label("|Q′| (nL/s)")
    ax.set_title("E   Conservation-consistent reference field", loc="left", fontsize=13.5, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6v3.png", dpi=140, bbox_inches="tight", facecolor="white"); fig.savefig(OUT_DIR / "fig6v3.pdf", bbox_inches="tight", facecolor="white")
    log(f"wrote {OUT_DIR / 'fig6v3.png'}")


if __name__ == "__main__":
    main()
