"""Figure 4 — the pulsatile flux of one vessel and how it is summarised (PLOS, 7.5 in wide).

  a  Q(t) of the representative vessel (edge 75489–75638, tile 36; Figs 3c–e): the ridge Poiseuille trace scaled to the
     vessel's Kloosterman mean, 250 fps, with its harmonic model Q̂(t) = Q̄ + Σ_k Re(H_k e^{2πik f0 t}), k = 1..3
  b  its power spectrum: peaks at the tile heart rate f0 and its multiples above a broadband floor
  c  every recorded beat folded on the cardiac phase (grey) against the model, and the model split into the
     fundamental, 2nd and 3rd harmonics
  d  harmonic coherence ρ (variance explained by the three harmonics) of every Kloosterman-valid measurement; the dashed
     line is the detection level of the fundamental (F test, p = 0.01, at the median trace length); this vessel marked

Inputs: the canonical graph (measurements_piv Q_t, tile_f0_piv).  Writes figures/fig4/fig4.{png,pdf,tif}.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.reconcile.harmonics import phasors, coherence, FS
from plexusflow.ac.field import detect_p
import plos_style

EDGE, TILE = (75489, 75638), 36
INK, MEAS, MODEL = "#1a1a1a", "#2ca02c", "#1f4e8c"
HCOL = ["#1f4e8c", "#d95f02", "#7570b3"]
HNAME = ["fundamental", "2nd harmonic", "3rd harmonic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    plos_style.apply()
    G = _paths.load_graph(); tf0 = G.graph["tile_f0_piv"]
    m = next(x for x in G.edges[EDGE]["measurements_piv"] if x["tile_id"] == TILE)
    q = np.asarray(m["Q_t"], float); f0 = float(tf0[TILE]); t = np.arange(len(q)) / FS
    a0, H = phasors(q, f0); rho_v = coherence(q, f0)
    model = lambda tt, ks=(1, 2, 3): a0 * (0 in ks) + sum((H[k - 1] * np.exp(2j * np.pi * k * f0 * tt)).real for k in ks if k > 0)
    print(f"[fig4] vessel {EDGE} tile {TILE}: f0 {f0:.2f} Hz, Q̄ {a0:.3f} nL/s, |H| {np.round(np.abs(H), 3)}, ρ {rho_v:.3f}, {len(q)} samples")

    # organ-wide coherence of every Kloosterman-valid measurement (at its tile's f0)
    rhos, ns = [], []
    for u, v, d in G.edges(data=True):
        for x in d.get("measurements_piv") or []:
            if (x.get("quality") or {}).get("kl_ok") and x.get("Q_t") is not None and tf0.get(x["tile_id"]):
                qq = np.asarray(x["Q_t"], float); r = coherence(qq, float(tf0[x["tile_id"]]))
                if np.isfinite(r):
                    rhos.append(r); ns.append(int(np.isfinite(qq).sum()))
    rhos, ns = np.array(rhos), np.array(ns); n_med = int(np.median(ns))
    rho_det = brentq(lambda r: detect_p(r, n_med) - 0.01, 1e-6, 0.5)
    det = np.mean([detect_p(r, n) < 0.01 for r, n in zip(rhos, ns)])
    print(f"[fig4] {len(rhos)} measurements; median ρ {np.median(rhos):.2f}; detection ρ* {rho_det:.3f} (n {n_med}); detected {100 * det:.0f}%")

    fig = plt.figure(figsize=(7.5, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.1], hspace=0.55, wspace=0.3,
                          left=0.085, right=0.985, top=0.93, bottom=0.1)
    letter = lambda ax, l, dx=-0.1: ax.text(dx, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # a: trace + model
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(t, q, color=MEAS, lw=0.7, label="measured Q(t)")
    tt = np.linspace(0, t[-1], 2000); axA.plot(tt, model(tt, (0, 1, 2, 3)), color=MODEL, lw=1.2, label="harmonic model (k ≤ 3)")
    axA.axhline(a0, color="0.4", lw=0.7, ls="--"); axA.text(0.012 * t[-1], a0, f"Q̄ = {a0:.2f} nL/s", va="bottom", ha="left", fontsize=8, color="0.3", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    axA.set_xlim(0, t[-1]); axA.set_xlabel("time (s)", labelpad=1); axA.set_ylabel("flux Q (nL/s)", labelpad=1)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(loc="upper left", frameon=False, ncol=2, borderaxespad=0, bbox_to_anchor=(0.0, 1.16), handlelength=1.6, columnspacing=1.0)
    letter(axA, "a", -0.08)

    # b: spectrum
    axB = fig.add_subplot(gs[0, 1])
    fr, pw = periodogram(q - a0, fs=FS, window="hann", detrend=False)
    axB.semilogy(fr, pw, color=INK, lw=0.7)
    for k in (1, 2, 3):
        axB.axvline(k * f0, color=HCOL[k - 1], lw=0.8, ls=":", zorder=0)
        axB.text(k * f0, pw.max() * 3, f"{k}f$_0$", color=HCOL[k - 1], ha="center", va="bottom", fontsize=8)
    axB.set_xlim(0, 15); axB.set_xticks([0, 5, 10, 15]); axB.set_ylim(pw[fr > 0.3].min() * 0.5, pw.max() * 12)
    axB.set_xlabel("frequency (Hz)", labelpad=1); axB.set_ylabel("power (nL²/s²/Hz)", labelpad=1)
    axB.spines[["top", "right"]].set_visible(False); letter(axB, "b", -0.3)

    # c: beats folded on phase + harmonic components
    gc = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.35)
    axC = fig.add_subplot(gc[0]); ph = (t * f0) % 1.0; cyc = np.floor(t * f0).astype(int)
    for c in np.unique(cyc):
        s_ = cyc == c
        if s_.sum() > 0.9 * FS / f0:
            axC.plot(ph[s_], q[s_], color="0.6", lw=0.5, alpha=0.8)
    pp = np.linspace(0, 1, 300); axC.plot(pp, model(pp / f0, (0, 1, 2, 3)), color=MODEL, lw=1.4)
    axC.set_xlim(0, 1); axC.set_xlabel("cardiac phase (cycles)", labelpad=1); axC.set_ylabel("flux Q (nL/s)", labelpad=1)
    axC.spines[["top", "right"]].set_visible(False)
    axC.text(0.98, 0.97, f"{len([c for c in np.unique(cyc) if (cyc == c).sum() > 0.9 * FS / f0])} beats", transform=axC.transAxes, ha="right", va="top", fontsize=8, color="0.35")
    letter(axC, "c", -0.32)
    axK = fig.add_subplot(gc[1])
    for k in (1, 2, 3):
        axK.plot(pp, model(pp / f0, (k,)), color=HCOL[k - 1], lw=1.1, label=f"{HNAME[k - 1]}  {abs(H[k - 1]):.2f}")
    axK.axhline(0, color="0.8", lw=0.5); axK.set_xlim(0, 1); axK.set_xlabel("cardiac phase (cycles)", labelpad=1)
    axK.set_ylabel("component (nL/s)", labelpad=1); axK.spines[["top", "right"]].set_visible(False)
    axK.legend(loc="upper center", frameon=False, borderaxespad=0, bbox_to_anchor=(0.5, 1.3), handlelength=1.2, labelspacing=0.15)

    # d: organ-wide coherence
    axD = fig.add_subplot(gs[1, 1])
    bins = np.logspace(-3, 0, 40); axD.hist(np.clip(rhos, 1e-3, 1), bins=bins, color="#9ab7d3", edgecolor="white", lw=0.3)
    axD.set_xscale("log"); axD.axvline(rho_det, color=INK, lw=0.8, ls="--")
    axD.text(rho_det * 1.25, axD.get_ylim()[1] * 0.95, f"detection\nlevel", ha="left", va="top", fontsize=8, linespacing=1.0)
    axD.annotate("this vessel", (rho_v, 0), xytext=(rho_v * 0.75, axD.get_ylim()[1] * 0.55), ha="right", fontsize=8, color=MEAS,
                 arrowprops=dict(arrowstyle="-|>", color=MEAS, lw=0.8, mutation_scale=7))
    axD.set_xlabel("harmonic coherence ρ", labelpad=1); axD.set_ylabel("measurements", labelpad=1)
    axD.text(0.02, 0.97, f"n = {len(rhos):,}\n{100 * det:.0f}% detected", transform=axD.transAxes, ha="left", va="top", fontsize=8)
    axD.spines[["top", "right"]].set_visible(False); letter(axD, "d", -0.3)
    plos_style.save(fig, _paths.FIGURES / "fig4" / "fig4")


if __name__ == "__main__":
    main()
