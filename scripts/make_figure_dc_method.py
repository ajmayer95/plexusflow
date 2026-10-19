"""DC method figure — how the conservation-consistent reference field is built (one stage, default $PLEXUSFLOW_STAGE).

  top  flowchart: per-tile optical flow → Kloosterman flux y → quality (f, align, tile) → reliability c → one winner per
       vessel → weighted projection onto B_I Q = 0 with weights at the reconciled flow → long-run mean Q′, emergent boundary
  a    calibration of c on simultaneously imaged (same-tile) junctions inside the carve: observed RMS imbalance vs the
       imbalance the reliability model predicts, σ_j = √(Σ Q² c² + σ0²)
  b    reliability c of all measurements vs the per-vessel winners; contributions of tile, kept fraction f, alignment
  c    why weights relative to the reconciled flow: held-out error (independent other-tile measurements) and measured-direction
       flips for weights relative to the measured value, uniform weights, and the reconciled scale (same winners)
  d    the damped re-weighting map Q ← (1 − α) Q + α F(Q): inflow and step change per iteration for α = 0.2 / 0.3 / 0.5;
       the estimator is the long-run mean (window shaded)
  e    measured vs reconciled flow per vessel, coloured by c
  f    the reference field |Q′| (unmeasured and conservation-dominated vessels faint), boundary injection / withdrawal
Reads the graph and results/{dc_reconciled_v2.pkl, reliability_model_v1.json}; writes figures/fig_dc_method/fig_dc_method.{png,pdf}.
Writes nothing to any graph.
"""
from __future__ import annotations
import sys
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
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import measurement_table, same_tile_junctions, carved_domain, c_of
from plexusflow.reconcile.network import project, emergent, EPS_UNIMAGED

IMPUTED_RATIO = 5.0
ALPHAS = (0.2, 0.3, 0.5)
N_ITER, BURN = 400, 300
OUT_DIR = _paths.FIGURES / "fig_dc_method"
T0 = time.time()


def log(m):
    print(f"[fig_dc_method +{time.time() - T0:4.0f} s] {m}", flush=True)


def main():
    G = _paths.load_graph(); dom = carved_domain(G); E = dom["edges"]; eix = {e: i for i, e in enumerate(E)}
    D = pickle.loads((_paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    model = json.loads((_paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    M = measurement_table(G); y, im, c, s0, Qp = D["Qm"], D["imaged"], np.where(D["imaged"], D["c"], 1.0), D["sigma0_floor"], D["Qp"]
    log(f"{_paths.STAGE.name}: {len(E)} carved vessels, {int(im.sum())} measured, {len(M)} measurements")

    # (a) calibration on same-tile junctions inside the carve
    pred, obs = [], []
    for t in sorted({t for t, _ in M}):
        meas = {e for (tt, e) in M if tt == t and e in eix}
        for j in same_tile_junctions(G, t, meas):
            ms = [M[(t, e)] for e, _ in j]
            pred.append(np.sqrt(sum((m["Q"] * c_of(model, m)) ** 2 for m in ms) + model["sigma0"] ** 2))
            obs.append(sum(s * M[(t, e)]["Q"] for e, s in j))
    pred, obs = np.array(pred), np.array(obs)
    log(f"(a) {len(pred)} same-tile junctions inside the carve")

    # (b) c for all measurements and for winners
    c_all = np.array([c_of(model, m) for (t, e), m in M.items() if e in eix]); c_win = c[im]
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0
    parts = np.array([(model["c_T"][m["tile"]], model["c_f"] * (1 - m["f"]), model["c_a"] * (1 - al(m))) for (t, e), m in M.items() if e in eix])

    # (c) weighting comparison on the same winners
    held = [(eix[k[1]], m["Q"]) for k, m in M.items() if k[1] in eix and im[eix[k[1]]] and abs(m["Q"] - y[eix[k[1]]]) > 1e-15]
    hi = np.array([h[0] for h in held]); hq = np.array([h[1] for h in held])
    def score(Q):
        ok = (np.abs(Q[hi]) > 0) & (np.abs(hq) > 0); big = im & (np.abs(y) > 1e-3)
        return np.median(np.abs(np.log(np.abs(Q[hi][ok]) / np.abs(hq[ok])))), np.mean(np.sign(Q[big]) != np.sign(y[big]))
    def wsolve(w):
        return project(y, np.where(im, w / np.median(w[im]), EPS_UNIMAGED), dom)
    variants = [("relative to the\nmeasured value", wsolve(1 / (c ** 2 * y ** 2 + s0 ** 2))),
                ("uniform\n(absolute error)", wsolve(np.ones(len(y)))),
                ("relative to the\nreconciled flow", Qp)]
    vscore = [score(Q) for _, Q in variants]; vin = [emergent(Q, dom)[dom["inj"]].sum() for _, Q in variants]
    log("(c) weighting variants: " + "; ".join(f"{lab.replace(chr(10), ' ')} held-out {s[0]:.3f} flips {s[1]:.3f}" for (lab, _), s in zip(variants, vscore)))

    # (d) damped re-weighting map
    traj = {}
    for a in ALPHAS:
        q = wsolve(np.ones(len(y))); inf, ch = [], []
        for k in range(N_ITER):
            new = wsolve(1 / (c ** 2 * q ** 2 + s0 ** 2)); qn = (1 - a) * q + a * new
            ch.append(np.linalg.norm(qn - q) / np.linalg.norm(qn)); q = qn; inf.append(emergent(q, dom)[dom["inj"]].sum())
        traj[a] = (np.array(inf), np.array(ch))
    log("(d) iterations done: long-run mean inflow " + ", ".join(f"α {a}: {traj[a][0][BURN:].mean():.2f}" for a in ALPHAS))

    # figure
    fig = plt.figure(figsize=(21, 16.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.23, 1, 1.25], hspace=0.24, wspace=0.26, left=0.05, right=0.97, top=0.96, bottom=0.04)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    steps = [("Per-tile optical flow", "Farnebäck OF on each tile's\nvideo; Kloosterman flux\nthrough every cross-section"),
             ("Measurements  $y_{te}$", "time-mean flux of vessel e\nin tile t (signed u→v);\nquality: kept fraction f,\nflow–centreline alignment"),
             ("Reliability  $c$", "$c = c_T + c_f(1-f) + c_a(1-\\mathrm{align})$\ncalibrated on same-tile\njunctions (a, b)"),
             ("One winner per vessel", "the measurement with the\nsmallest c; other tiles\nare held out (c)"),
             ("Conservative projection", "$w_e^{(k)} = 1/(c_e^2 (q_e^{(k)})^2 + s_0^2)$\n$\\tilde q^{(k+1)} = \\arg\\min_{B_I q = 0}\\sum_e w_e^{(k)}(q_e-y_e)^2$\n(interior nodes conserve)"),
             ("Reference field  $Q'$", "$q^{(k+1)} = (1-\\alpha)\\,q^{(k)} + \\alpha\\,\\tilde q^{(k+1)}$\n$Q' = $ long-run mean of $q^{(k)}$ (d)\nboundary flux $f = BQ'$ (f)")]
    xs = np.linspace(0.005, 0.84, len(steps)); bw = 0.152
    for k, (title, body) in enumerate(steps):
        x = xs[k]
        ax0.add_patch(FancyBboxPatch((x, 0.06), bw, 0.86, boxstyle="round,pad=0.008,rounding_size=0.02", fc=plt.cm.Blues(0.12 + 0.12 * k),
                                     ec="#2b5c8a", lw=1.4))
        ax0.text(x + bw / 2, 0.82, title, ha="center", va="center", fontsize=12.5, fontweight="bold")
        ax0.text(x + bw / 2, 0.40, body, ha="center", va="center", fontsize=10.5)
        if k < len(steps) - 1:
            ax0.annotate("", xy=(xs[k + 1] - 0.004, 0.5), xytext=(x + bw + 0.004, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2, color="#2b5c8a"))

    # a
    ax = fig.add_subplot(gs[1, 0]); ok = pred > 0
    bins = np.quantile(pred[ok], np.linspace(0, 1, 13)); cen, rms, lo, hi_ = [], [], [], []
    rng = np.random.default_rng(0)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        s_ = ok & (pred >= b0) & (pred < b1)
        if s_.sum() > 20:
            cen.append(np.median(pred[s_])); rms.append(np.sqrt(np.mean(obs[s_] ** 2)))
            bs = [np.sqrt(np.mean(rng.choice(obs[s_], s_.sum()) ** 2)) for _ in range(200)]; lo.append(np.percentile(bs, 5)); hi_.append(np.percentile(bs, 95))
    ax.scatter(pred[ok], np.abs(obs[ok]), s=3, color="0.75", alpha=0.4, label="junctions |net flux|")
    ax.errorbar(cen, rms, yerr=[np.array(rms) - lo, np.array(hi_) - rms], fmt="o-", color="#d62728", lw=2, ms=6, label="RMS net flux (binned, 90 % CI)")
    l0, l1 = np.percentile(pred[ok], [1, 99.5]); ax.plot([l0, l1], [l0, l1], "k--", lw=1, label="1 : 1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(l0 * 0.05, l1 * 5)
    ax.set_xlabel("predicted junction imbalance  $\\sqrt{\\sum_{e \\in j} y_e^2 c_e^2 + \\sigma_0^2}$  (nL/s)"); ax.set_ylabel("observed net flux at junction (nL/s)")
    ax.set_title(f"a   Reliability is calibrated on {len(pred)} simultaneously imaged junctions", loc="left", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # b
    ax = fig.add_subplot(gs[1, 1]); bb = np.logspace(np.log10(max(min(c_all.min(), c_win.min()), 0.02)), np.log10(max(c_all.max(), 3)), 45)
    ax.hist(c_all, bins=bb, color="0.65", alpha=0.8, density=True, label=f"all measurements ({len(c_all)})")
    ax.hist(c_win, bins=bb, color="#1f77b4", alpha=0.6, density=True, label=f"per-vessel winners ({len(c_win)})")
    ax.set_xscale("log"); ax.set_xlabel("reliability c (relative error scale)"); ax.set_ylabel("density")
    fr = parts.mean(0) / parts.sum(1).mean()
    ax.text(0.97, 0.95, f"$c_f$ = {model['c_f']:.2f},  $c_a$ = {model['c_a']:.2f},  median $c_T$ = {np.median(list(model['c_T'].values())):.2f}\n"
            f"mean share of c: tile {fr[0]:.0%}, kept fraction {fr[1]:.0%}, alignment {fr[2]:.0%}", transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(fc="white", ec="0.6"))
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.8))
    ax.set_title("b   Reliability determines per-vessel measurement selection", loc="left", fontsize=12.5, fontweight="bold")

    # c
    ax = fig.add_subplot(gs[1, 2]); xk = np.arange(3); cols = ["#9467bd", "0.55", "#2ca02c"]
    ax.bar(xk - 0.2, [s[0] for s in vscore], 0.38, color=cols, alpha=0.95)
    ax2 = ax.twinx(); ax2.bar(xk + 0.2, [100 * s[1] for s in vscore], 0.38, color=cols, alpha=0.45, hatch="//", edgecolor="k")
    for x, s, i_ in zip(xk, vscore, vin):
        ax.text(x - 0.2, s[0] + 0.01, f"{s[0]:.2f}", ha="center", fontsize=10); ax2.text(x + 0.2, 100 * s[1] + 0.4, f"{100 * s[1]:.0f} %", ha="center", fontsize=10)
        ax.text(x, -0.13, f"inflow {i_:.1f} nL/s", ha="center", fontsize=9.5, transform=ax.get_xaxis_transform())
    ax.set_xticks(xk); ax.set_xticklabels([v[0] for v in variants], fontsize=10)
    ax.set_ylabel("held-out error  median |log(Q′/y_other)|  (solid)"); ax2.set_ylabel("measured directions flipped, %  (hatched)")
    ax.set_ylim(0, max(s[0] for s in vscore) * 1.25); ax2.set_ylim(0, max(100 * s[1] for s in vscore) * 1.3)
    ax.set_title("c   Weighting-scale ablation: held-out measurements", loc="left", fontsize=12.5, fontweight="bold")

    # d
    gd = gs[2, 0].subgridspec(2, 1, hspace=0.12); axd1 = fig.add_subplot(gd[0]); axd2 = fig.add_subplot(gd[1], sharex=axd1)
    for a, colr in zip(ALPHAS, ("#1f77b4", "#2ca02c", "#d62728")):
        inf, ch = traj[a]; it = np.arange(1, N_ITER + 1)
        axd1.plot(it, inf, color=colr, lw=1.3, label=f"α = {a}{' (production)' if a == 0.3 else ''}  long-run mean {inf[BURN:].mean():.2f}")
        axd2.plot(it, ch, color=colr, lw=1.0)
    for a_ in (axd1, axd2):
        a_.axvspan(BURN, N_ITER, color="0.85", zorder=0)
    axd1.set_ylabel("inflow (nL/s)"); axd1.legend(fontsize=9, loc="lower right"); plt.setp(axd1.get_xticklabels(), visible=False)
    axd2.set_yscale("log"); axd2.set_ylabel("step change ‖ΔQ‖/‖Q‖"); axd2.set_xlabel("re-weighting iteration")
    axd1.text((BURN + N_ITER) / 2, 1.0, "averaging\nwindow", transform=axd1.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="0.35")
    axd1.set_title("d   Long-run mean is insensitive to damping", loc="left", fontsize=12.5, fontweight="bold")

    # e
    ax = fig.add_subplot(gs[2, 1]); s_ = im & (np.abs(y) > 1e-5) & (np.abs(Qp) > 1e-5)
    sc = ax.scatter(np.abs(y[s_]), np.abs(Qp[s_]), c=c[s_], s=4, cmap="viridis_r", alpha=0.6, norm=mcolors.LogNorm(*np.percentile(c[s_], [1, 99])))
    fl = s_ & (np.sign(y) != np.sign(Qp)); ax.scatter(np.abs(y[fl]), np.abs(Qp[fl]), s=12, facecolors="none", edgecolors="k", lw=0.5,
                                                       label=f"direction changed ({fl.sum()}, {fl.sum() / s_.sum():.1%})")
    lo_, hi2 = 1e-5, np.abs(np.r_[y[s_], Qp[s_]]).max() * 2; ax.plot([lo_, hi2], [lo_, hi2], "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi2); ax.set_ylim(lo_, hi2)
    ax.set_xlabel("|y|  winner measurement (nL/s)"); ax.set_ylabel("|Q′|  reconciled (nL/s)"); ax.legend(fontsize=9, loc="upper left")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label("reliability c (log scale)")
    adj = np.abs(Qp[s_] - y[s_]) / np.abs(y[s_])
    ax.text(0.97, 0.05, f"median adjustment: {np.median(adj):.0%}\nc < 0.3: {np.median(adj[c[s_] < 0.3]):.0%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, bbox=dict(fc="white", ec="0.6"))
    ax.set_title("e   Reconciliation mostly corrects, rarely rewrites", loc="left", fontsize=12.5, fontweight="bold")

    # f
    ax = fig.add_subplot(gs[2, 2]); mos = tifffile.imread(str(_paths.MOSAIC)); mos = mos.mean(-1) if mos.ndim == 3 else mos; Hm = mos.shape[0]
    segs = []
    for e in E:
        d = G.edges[e]; p = np.asarray(d["path"], float) if d.get("path") is not None else np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in e])
        segs.append(np.column_stack([p[:, 0], Hm - p[:, 1]]))
    imputed = ~im | (np.abs(Qp) > IMPUTED_RATIO * np.abs(y))
    a_ = np.abs(Qp); vmin, vmax = max(np.percentile(a_[im], 2), 1e-4), np.percentile(a_[im], 98); nrm = mcolors.LogNorm(vmin, vmax)
    ax.imshow(mos, cmap="gray", vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98)); ax.axis("off")
    cols = [(*plt.cm.viridis(nrm(np.clip(a_[i], vmin, vmax)))[:3], 0.3 if imputed[i] else 1.0) for i in range(len(E))]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=np.where(imputed, 0.6, 1.2)))
    f = emergent(Qp, dom); nodes = dom["nodes"]
    for i in dom["bnd"]:
        if abs(f[i]) > 0.02:
            ax.scatter(G.nodes[nodes[i]]["x"], Hm - G.nodes[nodes[i]]["y"], s=20 + 60 * min(abs(f[i]) / 2, 1.5),
                       color="#d62728" if f[i] > 0 else "#1f5fa8", edgecolor="white", lw=0.8, zorder=5)
    ax.scatter([], [], s=60, color="#d62728", label=f"injection (Σ {f[dom['inj']][f[dom['inj']] > 0].sum():.2f} nL/s)")
    ax.scatter([], [], s=60, color="#1f5fa8", label=f"withdrawal (Σ {f[dom['bnd']][f[dom['bnd']] < 0].sum():.2f} nL/s)")
    ax.plot([], [], color=plt.cm.viridis(0.6), alpha=0.3, lw=1, label=f"faint: unmeasured or conservation-dominated (|Q′| > {IMPUTED_RATIO:g}|y|)")
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
    cbf = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="viridis"), ax=ax, fraction=0.04, pad=0.01); cbf.set_label("|Q′| (nL/s)")
    ax.set_title("f   Conservation-consistent reference field", loc="left", fontsize=12.5, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_dc_method.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig_dc_method.pdf", bbox_inches="tight", facecolor="white")
    log(f"wrote {OUT_DIR / 'fig_dc_method.png'}")


if __name__ == "__main__":
    main()
