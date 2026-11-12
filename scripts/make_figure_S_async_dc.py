"""SI figure — DC reconstruction when tiles see different flow fields (scripts/benchmark/dc_async.py →
results/dc_async_benchmark.json).  x = departure of the fields the tiles saw from the time-average (median |log(Q*(t)/Q̄)|).
  a  reconstruction error vs the time-average Q̄: measured and hidden (determined by conservation) vessels
  b  direction accuracy of the 2σ-mask vessels vs Q̄ (and vs the field their own tile saw)
  c  same-tile junction RMS z under the calibrated c (flat: the calibration cannot see between-tile variation) and the
     synthetic repeat disagreement (trusted references) against the real value — the shaded band marks the departures
     consistent with the data.
Marker fill: share of the variation that is a per-tile scale (τ_global) vs redistribution (τ_local).
Writes figures/fig6/fig6_S_async.{png,pdf,tif}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plexusflow.paths as _paths
import plos_style

INK, GREY, BLUE, ORANGE = "#1a1a1a", "0.55", "#1f5fa8", "#d95f02"


def main():
    plos_style.apply()
    J = json.loads((_paths.RESULTS / "dc_async_benchmark.json").read_text()); T = J["table"]; real = J["real_repeat"]["trusted"]
    x = np.array([t["departure"] for t in T]); share = np.array([t["tau_global"] / max(t["tau_global"] + t["tau_local"], 1e-9) for t in T])
    # departures consistent with the real trusted repeat disagreement (linear fit of repeat vs departure)
    rp = np.array([t["repeat_trusted"] for t in T]); cf = np.polyfit(x, rp, 1); resid = np.std(rp - np.polyval(cf, x))
    lo, hi = sorted(((real - resid - cf[1]) / cf[0], (real + resid - cf[1]) / cf[0]))
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9), gridspec_kw=dict(wspace=0.45, left=0.08, right=0.985, top=0.84, bottom=0.2))
    def sc(ax, y, col, mk, lab):
        ax.scatter(x, y, s=22, marker=mk, c=[plt.matplotlib.colors.to_rgba(col, 0.25 + 0.75 * s_) for s_ in share], edgecolors=col, linewidths=0.8, label=lab, zorder=3)
    for ax in axs:
        ax.axvspan(lo, hi, color="0.9", lw=0, zorder=0); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
        ax.set_xlabel("tile fields vs time-average,\nmedian |log(Q*(t)/Q̄)|", fontsize=8, labelpad=1)
    sc(axs[0], [t["err_measured"] for t in T], BLUE, "s", "measured vessels"); sc(axs[0], [t["err_hidden_det"] for t in T], ORANGE, "o", "hidden (determined)")
    axs[0].set_ylabel("error vs Q̄, median |log(Q′/Q̄)|", fontsize=8); axs[0].set_ylim(0, None); axs[0].legend(fontsize=8, frameon=False, loc="upper left", handletextpad=0.2)
    axs[0].set_title("Reconstruction vs time-average", fontsize=9, loc="left", pad=3)
    sc(axs[1], [100 * t["mask_dir"] for t in T], BLUE, "o", "vs time-average Q̄"); sc(axs[1], [100 * t["mask_dir_inst"] for t in T], GREY, "^", "vs field its tile saw")
    axs[1].set_ylim(95, 100.2); axs[1].set_ylabel("2σ-mask vessels with correct direction (%)", fontsize=8)
    axs[1].legend(fontsize=8, frameon=False, loc="lower left", handletextpad=0.2); axs[1].set_title("2σ direction mask", fontsize=9, loc="left", pad=3)
    sc(axs[2], [t["repeat_trusted"] for t in T], BLUE, "o", "repeat disagreement (trusted)"); sc(axs[2], [t["junction_rms_z"] - 1 for t in T], GREY, "D", "junction RMS z − 1")
    axs[2].axhline(real, color=BLUE, lw=0.8, ls=(0, (3, 2))); axs[2].text(x.max(), real + 0.01, "real 21S", fontsize=8, color=BLUE, ha="right", va="bottom")
    axs[2].set_ylim(0, None); axs[2].legend(fontsize=8, frameon=False, loc="upper left", handletextpad=0.2); axs[2].set_title("What the data can see", fontsize=9, loc="left", pad=3)
    for x_, l_ in ((0.005, "a"), (0.335, "b"), (0.665, "c")):
        fig.text(x_, 0.93, l_, fontsize=12, fontweight="bold")
    print(f"[S_async] departures consistent with the real trusted repeat disagreement: {lo:.3f}–{hi:.3f}")
    plos_style.save(fig, _paths.FIGURES / "fig6" / "fig6_S_async")


if __name__ == "__main__":
    main()
