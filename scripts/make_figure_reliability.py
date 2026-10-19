"""Reliability figure — what the reliability score c sees, and what it does not (one stage, default $PLEXUSFLOW_STAGE).

c = c_T + c_f (1 − f) + c_a (1 − align) is a trust score for a measurement's TIME-MEAN flux: is the optical flow along the
vessel (align = speed-weighted |cos θ| between the time-mean flow and the centreline within ±R), and how many Kloosterman
cross-sections survived (f).  It is not a waveform-quality score — that is the harmonic coherence ρ (and per-beat scatter),
used for the AC field.

Rows (examples picked by rule, the measurement nearest the group's median c — not by hand):
  1 most reliable 10 % (median ρ within it)      2 median c                    3 poor alignment, clean waveform (align < 0.4, ρ > 0.5)
  4 few cross-sections kept (f < 0.2, align > 0.85)                            5 reliable mean, no pulsatility (c in best 10 %, ρ < 0.05)
Columns: time-mean optical-flow field over the vessel (speed, arrows, centreline, ±R band) | Q(t) with DC + 3 harmonics | the
three parts of c.  Last panel: c vs ρ over all measurements.
Reads the graph, results/reliability_model_v1.json and batch_kl_inputs/ (per-measurement flow fields); writes
figures/fig_reliability/fig_reliability.{png,pdf}.  Writes nothing to any graph.
"""
from __future__ import annotations
import sys
import json
import gzip
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import measurement_table, carved_domain, c_of
from plexusflow.reconcile.harmonics import phasors, coherence, FS

OUT_DIR = _paths.FIGURES / "fig_reliability"


def main():
    G = _paths.load_graph(); dom = carved_domain(G); carved = set(map(tuple, dom["edges"]))
    model = json.loads((_paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    M = measurement_table(G); tf0 = G.graph["tile_f0_piv"]
    raw = {(int(m["tile_id"]), e): m for e in carved for m in G.edges[e].get("measurements_piv") or []}
    keys = [k for k in M if k[1] in carved and k in raw and raw[k].get("Q_t") is not None and len(raw[k]["Q_t"]) > 100]
    al = lambda m: m["align"] if np.isfinite(m["align"]) else 1.0
    c = np.array([c_of(model, M[k]) for k in keys]); f = np.array([M[k]["f"] for k in keys]); a = np.array([al(M[k]) for k in keys])
    rho = np.array(Parallel(n_jobs=8)(delayed(coherence)(raw[k]["Q_t"], tf0.get(k[0])) for k in keys)); rho = np.where(np.isfinite(rho), rho, 0.0)
    print(f"[fig_reliability] {_paths.STAGE.name}: {len(keys)} measurements of carved vessels", flush=True)
    c10 = np.percentile(c, 10)
    groups = [("most reliable 10 %", c <= c10, "median ρ"),
              ("median reliability", (c >= np.percentile(c, 45)) & (c <= np.percentile(c, 55)), None),
              ("poor alignment, clean waveform", (a < 0.4) & (rho > 0.5), None),
              ("few cross-sections kept", (f < 0.2) & (a > 0.85), None),
              ("reliable mean, no pulsatility", (c <= c10) & (rho < 0.05), None)]
    picks = []
    for lab, sel, how in groups:
        idx = np.flatnonzero(sel)
        if not len(idx):
            picks.append((lab, None)); continue
        target = np.median(rho[idx]) if how == "median ρ" else np.median(c[idx])
        score = np.abs(rho[idx] - target) if how == "median ρ" else np.abs(c[idx] - target)
        picks.append((lab, int(idx[np.argmin(score)])))
    kl = {}
    def kl_of(t, e):
        if t not in kl:
            p = _paths.STAGE / "batch_kl_inputs" / f"tile_{t}.pkl.gz"
            kl[t] = pickle.load(gzip.open(p))["kl_inputs"] if p.exists() else {}
        return kl[t].get(e) or kl[t].get((e[1], e[0]))

    n = len(picks)
    fig = plt.figure(figsize=(19, 3.3 * n + 4.2), facecolor="white")
    gs = fig.add_gridspec(n + 1, 3, width_ratios=[1.0, 2.3, 0.75], height_ratios=[1] * n + [1.25], hspace=0.42, wspace=0.18,
                          left=0.06, right=0.98, top=0.95, bottom=0.04)
    for r_, (lab, i) in enumerate(picks):
        axv, axq, axc = fig.add_subplot(gs[r_, 0]), fig.add_subplot(gs[r_, 1]), fig.add_subplot(gs[r_, 2])
        if i is None:
            for a_ in (axv, axq, axc):
                a_.axis("off")
            axq.text(0.5, 0.5, f"{lab}: no measurement matches", ha="center"); continue
        t, e = keys[i]; m = M[keys[i]]; q = np.asarray(raw[keys[i]]["Q_t"], float); tt = np.arange(len(q)) / FS; f0 = tf0.get(t)
        ki = kl_of(t, e)
        if ki is not None:
            vf = np.asarray(ki["vfield_mean"], float); sp = np.hypot(vf[..., 0], vf[..., 1]); cl = np.asarray(ki["cl_used"], float)
            axv.imshow(sp, cmap="magma", origin="upper"); H, W = sp.shape; st_ = max(2, int(min(H, W) / 14))
            yy, xx = np.mgrid[0:H:st_, 0:W:st_]
            # vf[..., 0] = column (x) and vf[..., 1] = row (y, downward) displacement; angles='xy' draws them in image coordinates
            U, V = vf[0:H:st_, 0:W:st_, 0], vf[0:H:st_, 0:W:st_, 1]; L = np.nanpercentile(np.hypot(U, V), 95) or 1.0
            axv.quiver(xx, yy, U / L * st_ * 0.9, V / L * st_ * 0.9, color="white", angles="xy", scale_units="xy", scale=1, width=0.006, alpha=0.9)
            axv.plot(cl[:, 1], cl[:, 0], color="#4fc3f7", lw=1.6)
            if "ridge_mask_roi" in ki:
                axv.contour(np.asarray(ki["ridge_mask_roi"], float), levels=[0.5], colors="#4fc3f7", linewidths=0.8, linestyles="--")
            axv.set_xlim(0, W); axv.set_ylim(H, 0)
            from plexusflow.analyze.mosaic_piv import _flow_quality
            rq = _flow_quality(vf, cl, float(ki["R_prior_px"]))["align"]
            print(f"[fig_reliability]   {lab}: tile {t} edge {e} stored align {al(m):.2f}, recomputed from the plotted field {rq:.2f}; "
                  f"centreline span (row, col) {np.round(cl[-1] - cl[0], 1)}", flush=True)
        axv.set_xticks([]); axv.set_yticks([])
        if r_ == 0:
            axv.set_title("time-mean optical flow (speed, arrows)\ncentreline and vessel mask in blue", fontsize=10)
        axv.set_ylabel(lab, fontsize=10.5, fontweight="bold")
        axq.plot(tt, q, color="0.45", lw=0.6)
        if f0:
            a0, Hh = phasors(q, f0); fit = a0 + sum((Hh[k] * np.exp(2j * np.pi * (k + 1) * f0 * tt)).real for k in range(3))
            axq.plot(tt, fit, color="#d62728", lw=1.5)
        axq.axhline(np.nanmean(q), color="#1f77b4", ls="--", lw=1); axq.axhline(0, color="k", lw=0.5)
        axq.text(0.005, 0.97, f"tile {t}   mean {np.nanmean(q):+.4f} nL/s   f {m['f']:.2f}   align {al(m):.2f}   ρ {rho[i]:.2f}",
                 transform=axq.transAxes, va="top", fontsize=9.5, bbox=dict(fc="white", ec="0.7", alpha=0.9))
        axq.set_ylabel("Q (nL/s)", fontsize=9); axq.tick_params(labelsize=8.5)
        if r_ == 0:
            axq.set_title("flux Q(t)  (grey raw 250 Hz, red DC + 3 harmonics at the tile heart rate, blue mean)", fontsize=10)
        if r_ == n - 1:
            axq.set_xlabel("time (s)", fontsize=9)
        parts = [model["c_T"][t], model["c_f"] * (1 - m["f"]), model["c_a"] * (1 - al(m))]
        axc.barh([2, 1, 0], parts, color=["0.6", "#ff7f0e", "#9467bd"]); axc.set_yticks([2, 1, 0]); axc.set_yticklabels(["tile", "kept fraction", "alignment"], fontsize=9)
        axc.set_xlim(0, max(1.3, sum(parts) * 1.05)); axc.tick_params(labelsize=8.5)
        axc.set_title(f"c = {c[i]:.2f}  ({100 * np.mean(c <= c[i]):.0f}th percentile)", fontsize=10, fontweight="bold")
    ax = fig.add_subplot(gs[n, :2]); hb = ax.hexbin(rho, c, gridsize=60, bins="log", cmap="Greys", mincnt=1)
    ax.set_xlabel("harmonic coherence ρ (waveform quality: share of the trace's variance at the heart-rate harmonics)")
    ax.set_ylabel("reliability c (mean-flux trust)"); sr = spearmanr(rho, c)[0]
    for lab, i in picks:
        if i is not None:
            ax.plot(rho[i], c[i], "o", color="#d62728", ms=8, mec="white"); ax.annotate(lab, (rho[i], c[i]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_title(f"c (mean-flux trust) and ρ (waveform quality) are related but distinct  (Spearman {sr:+.2f}, {len(keys)} measurements)", fontsize=11, fontweight="bold", loc="left")
    plt.colorbar(hb, ax=ax, fraction=0.03, pad=0.01).set_label("measurements")
    ax = fig.add_subplot(gs[n, 2]); ax.axis("off")
    ax.text(0.0, 0.95, f"$c = c_T + c_f\\,(1-f) + c_a\\,(1-\\mathrm{{align}})$\n\n$c_f$ = {model['c_f']:.2f},  $c_a$ = {model['c_a']:.2f}\n"
            f"median $c_T$ = {np.median(list(model['c_T'].values())):.2f}\n\nc ranks measurements for the DC\nreconciliation (winner = smallest c);\n"
            f"the AC field picks its winner by\nper-beat repeatability instead.", va="top", fontsize=11)
    fig.suptitle(f"{_paths.STAGE.name}: what the reliability score c sees — the geometry behind the mean flux", fontsize=13.5, fontweight="bold")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_reliability.png", dpi=140, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig_reliability.pdf", bbox_inches="tight", facecolor="white")
    print(f"[fig_reliability] Spearman(c, ρ) {sr:+.2f}; picks: " + "; ".join(f"{lab} c {c[i]:.2f} ρ {rho[i]:.2f}" for lab, i in picks if i is not None))
    print(f"[fig_reliability] wrote {OUT_DIR / 'fig_reliability.png'}")


if __name__ == "__main__":
    main()
