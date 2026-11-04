"""SI figure — known-truth benchmark of the DC reconstruction (scripts/benchmark/dc_reconstruction.py →
results/dc_synthetic_benchmark.json): conservation-only vs passive-resistance-constrained reconstruction under measurement
noise × radius uncertainty, on the real 21S network.

  a, b  ΔE = E(passive-constrained QL) − E(conservation-only QL), median over realisations, measured (a) / hidden (b) vessels;
        E = median |log(|Q|/|Q*|)| against the known truth.  Red: conservation-only closer to the truth.
  c, d  at the experimental noise level (κ = 1): error (c) and direction agreement (d) vs radius error for every method
        (IQR over realisations); passive-constrained with independent (solid) and tile-correlated (dashed) radius errors.
Writes figures/fig6/fig6_S_synthetic.{png,pdf,tif}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches
import plexusflow.paths as _paths
import plos_style

INK, GREY = "#1a1a1a", "0.55"
STYLE = {"kcl_ql": ("conservation only, quasi-likelihood (used)", "#1f5fa8", "-"),
         "kcl_relative": ("conservation only, weights 1/(c²y²)", "#6a9fd4", "-"),
         "kcl_uniform": ("conservation only, equal weights", "0.6", "-"),
         "pas_ql": ("passive-constrained, quasi-likelihood", "#d95f02", "-")}


def main():
    plos_style.apply()
    J = json.loads((_paths.RESULTS / "dc_synthetic_benchmark.json").read_text()); T = J["table"]
    K, S = J["kappas"], J["sigma_a"]; rerr = J.get("radius_error_estimate") or {}; s_est = rerr.get("sigma_consensus")
    get = lambda mode, k, s, m, st: next(t for t in T if (t["mode"], t["kappa"], t["sigma_a"], t["method"], t["set"]) == (mode, k, s, m, st))
    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.3, left=0.09, right=0.92, top=0.93, bottom=0.2)
    norm = mcolors.TwoSlopeNorm(0, -0.6, 0.6); cmap = plt.get_cmap("RdBu_r")
    for j, st in enumerate(("measured", "hidden")):
        ax = fig.add_subplot(gs[0, j]); M = np.array([[get("vessel", k, s, "pas_ql", st)["err_median"] - get("vessel", k, s, "kcl_ql", st)["err_median"] for s in S] for k in K])
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for a_ in range(len(K)):
            for b_ in range(len(S)):
                ax.text(b_, a_, f"{M[a_, b_]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M[a_, b_]) > 0.35 else INK)
        ax.set_xticks(range(len(S))); ax.set_xticklabels([f"{int(100 * s)}%" for s in S]); ax.set_yticks(range(len(K))); ax.set_yticklabels([f"{k:g}" for k in K])
        ax.set_xlabel("radius error σ_a (per vessel)", fontsize=8, labelpad=1); ax.set_ylabel("noise κ (1 = experiment)", fontsize=8, labelpad=1)
        ax.set_title(f"{st.capitalize()} vessels: ΔE = E_passive − E_cons. (red: cons. closer)", fontsize=8.5, pad=3, loc="left")
        if s_est:
            xe = np.interp(s_est, S, np.arange(len(S))); ax.axvline(xe, color=INK, lw=1.0, ls=(0, (2, 2)))
            ax.add_patch(matplotlib.patches.Rectangle((-0.5, K.index(1.0) - 0.5), len(S), 1, fill=False, ec=INK, lw=0.8))   # experimental noise row
        ax.tick_params(labelsize=8, length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cax = fig.add_axes([0.935, 0.66, 0.009, 0.25]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
    cb.set_label("ΔE", fontsize=8, labelpad=2)
    xs = np.array(S) * 100
    for j, (key, lab) in enumerate((("err", "error vs truth, median |log(Q/Q*)|"), ("dir", "direction correct (%)"))):
        ax = fig.add_subplot(gs[1, j])
        for st, ls_m in (("hidden", "o"), ("measured", "s")):
            if key == "dir" and st == "measured":
                continue
            for m, (name, col, _) in STYLE.items():
                ys = np.array([get("vessel", 1.0, s, m, st)[key + ("_median" if key == "err" else "_mean")] for s in S]) * (100 if key == "dir" else 1)
                ax.plot(xs, ys, color=col, lw=1.4 if st == "hidden" else 0.9, ls="-" if st == "hidden" else ":", marker=ls_m, ms=3.5,
                        label=name if (st == "hidden") else None)
                if m == "pas_ql":
                    yt = [get("tile+vessel", 1.0, s, m, st)[key + ("_median" if key == "err" else "_mean")] * (100 if key == "dir" else 1) for s in S[1:]]
                    ax.plot(xs[1:], yt, color=col, lw=1.0, ls="--", marker=ls_m, ms=3, mfc="white")
        if s_est:
            ax.axvline(100 * s_est, color=INK, lw=1.0, ls=(0, (2, 2)))
            ax.text(100 * s_est + 0.6, ax.get_ylim()[1] if key == "dir" else 0.97, "estimated\nfor 21S", fontsize=8, va="top", ha="left", color=INK,
                    transform=ax.get_xaxis_transform() if key != "dir" else ax.transData)
        ax.set_xlabel("radius error σ_a (%)", fontsize=8, labelpad=1); ax.set_ylabel(lab, fontsize=8); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("κ = 1: error vs truth (solid hidden, dotted measured)" if key == "err" else "κ = 1: direction, hidden vessels", fontsize=9, pad=3, loc="left")
        if key == "err":
            ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 0))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c_, lw=1.4, marker="o", ms=3.5, label=n_) for n_, c_, _ in STYLE.values()]
    hs.append(Line2D([], [], color=STYLE["pas_ql"][1], lw=1.0, ls="--", marker="o", ms=3, mfc="white", label="passive-constrained, tile-correlated radius errors"))
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=8, frameon=False, handlelength=2.2, columnspacing=1.5, bbox_to_anchor=(0.5, 0.0))
    for x_, y_, l_ in ((0.005, 0.95, "a"), (0.5, 0.95, "b"), (0.005, 0.5, "c"), (0.5, 0.5, "d")):
        fig.text(x_, y_, l_, fontsize=12, fontweight="bold", va="bottom")
    plos_style.save(fig, _paths.FIGURES / "fig6" / "fig6_S_synthetic")


if __name__ == "__main__":
    main()
