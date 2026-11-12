"""SI figure — conservation-only vs passive-resistance-constrained DC reconstruction, scored against independent repeat
recordings.  Paired differences (passive − conservation-only) with 95 % vessel-bootstrap intervals, from
results/dc_passive_comparison.json (scripts/dc_validate_passive.py) of one or more stages.

Run: python3 scripts/make_figure_S_passive.py [STAGE_DIR ...]      (default: $PLEXUSFLOW_STAGE; the paper uses 21S and 27S)
Writes <figures of the first stage>/fig6/fig6_S_passive.{png,pdf,tif}.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plexusflow.paths as _paths
import plos_style

INK, GREY = "#1a1a1a", "0.55"; COL = {"trusted": "#1f5fa8", "all": "0.45"}


def stage_label(p):
    n = Path(p).name; return n.replace("PlexusFlow", "").replace("_release", "")


def main():
    plos_style.apply()
    stages = [Path(s) for s in sys.argv[1:]] or [_paths.STAGE]
    res = []
    for s in stages:
        rdir = _paths.RESULTS if s == _paths.STAGE and os.environ.get("PLEXUSFLOW_RESULTS") else s / "results"
        res.append((stage_label(s), json.loads((rdir / "dc_passive_comparison.json").read_text())))
    rows = [(st, test, ref) for st, _ in res for test in ("measured", "hidden") for ref in ("trusted", "all")]
    R = dict(res); y = np.arange(len(rows))[::-1].astype(float)
    for k in range(len(rows)):                                            # small gaps between stage / test blocks
        y[k] -= 0.6 * (k // 4) + 0.25 * (k // 2)
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 0.42 * len(rows) + 1.1), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.25, right=0.985, top=0.86, bottom=0.17))
    for ax, key, lab, scale in ((axs[0], "d_err", "Δ median |log(Q / Q_repeat)|", 1.0), (axs[1], "d_dir", "Δ correct direction (percentage points)", 100.0)):
        ax.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
        for yy, (st, test, ref) in zip(y, rows):
            r = R[st][test][ref]; v = r[key] * scale; lo, hi = (np.array(r[key + "_ci"]) * scale)
            ax.plot([lo, hi], [yy, yy], color=COL[ref], lw=1.6, solid_capstyle="round"); ax.scatter([v], [yy], s=24, color=COL[ref], zorder=3,
                                                                                                 facecolors=COL[ref] if ref == "trusted" else "white", linewidths=1.2)
        ax.set_xlabel(lab, fontsize=8, labelpad=2); ax.tick_params(labelsize=8); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{st}  {test} vessels · {ref}" for st, test, ref in rows], fontsize=8)
    axs[0].set_title("worse with passive constraint →", fontsize=8, color=GREY, loc="right", pad=2)
    axs[1].set_title("better with passive constraint →", fontsize=8, color=GREY, loc="right", pad=2)
    fig.text(0.25, 0.975, "Passive-resistance-constrained minus conservation-only reconstruction (paired, 95 % intervals)",
             fontsize=9, va="top", ha="left", color=INK)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL["trusted"], ms=5, label="trusted repeats (c < 0.5)"),
                        Line2D([], [], marker="o", ls="", color=COL["all"], mfc="white", ms=5, mew=1.2, label="all repeats")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, frameon=False, fontsize=8, handletextpad=0.2)
    for k, (st, d) in enumerate(res):
        t = d["totals"]; print(f"[S_passive] {st}: inflow conservation-only {t['conservation_only']['inflow']:.2f} vs passive {t['passive']['inflow']:.2f} nL/s; "
                                f"measured directions reversed {100 * t['conservation_only']['flips']:.1f} vs {100 * t['passive']['flips']:.1f} %")
    plos_style.save(fig, _paths.FIGURES / "fig6" / "fig6_S_passive")


if __name__ == "__main__":
    main()
