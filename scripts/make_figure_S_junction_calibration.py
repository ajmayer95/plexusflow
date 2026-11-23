"""SI figure — same-tile junction calibration of the measurement-discrepancy scale, per stage (companion of Fig 6b).

Top row: relative junction imbalance δ = (in − out)/in (Kloosterman-style; robust SD and the single fractional error it implies).
Bottom row: out-of-fold RMS(r_j / σ_j,pred) by quintile of junction quality (mean c_k), quality-dependent c_k vs one constant c,
with 95 % tile-bootstrap intervals.  Input: results/dc_junction_calibration.json (scripts/dc_junction_calibration.py).

Run: python3 scripts/make_figure_S_junction_calibration.py [STAGE_DIR ...]   (default: $PLEXUSFLOW_STAGE; the paper uses 21S and 27S)
Writes <figures of the first stage>/fig6/fig6_S_calibration.{png,pdf,tif}.
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

INK, GREY, BLUE, MUTED = "#1a1a1a", "0.55", "#1f5fa8", "#6b6b6b"


def stage_label(p):
    return Path(p).name.replace("PlexusFlow", "").replace("_release", "")


def main():
    plos_style.apply()
    stages = [Path(s) for s in sys.argv[1:]] or [_paths.STAGE]
    res = []
    for s in stages:
        rdir = _paths.RESULTS if s == _paths.STAGE and os.environ.get("PLEXUSFLOW_RESULTS") else s / "results"
        res.append((stage_label(s), json.loads((rdir / "dc_junction_calibration.json").read_text())))
    n = len(res)
    fig, axs = plt.subplots(2, n, figsize=(7.5, 5.0), squeeze=False, gridspec_kw=dict(hspace=0.55, wspace=0.3, left=0.1, right=0.98, top=0.93, bottom=0.1))
    for col, (st, J) in enumerate(res):
        K = J["kloosterman"]; ax = axs[0, col]; e = np.array(K["hist_edges"]); h = np.array(K["hist"], float)
        ax.bar(0.5 * (e[1:] + e[:-1]), h / (h.sum() * np.diff(e)), width=np.diff(e), color="0.75", edgecolor="white", lw=0.3)
        xx = np.linspace(-1, 1, 300); sd = K["delta_robust_sd"]
        ax.plot(xx, np.exp(-0.5 * (xx / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=INK, lw=1.0)
        ax.set_xlim(-1, 1); ax.set_xlabel("relative junction imbalance (in − out) / in", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{st}: {K['n']} same-tile junctions", fontsize=9, loc="left")
        ax.text(0.98, 0.95, f"robust SD {sd:.2f}\nmedian {K['delta_median']:+.3f}\nsingle fractional error {K['single_fractional_error']:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top", color=MUTED)
        ax = axs[1, col]; xq = np.arange(1, 6)
        for lab, colr, mk, dx, name in (("constant", GREY, "s", 0.09, "one constant c"), ("calibrated", BLUE, "o", -0.09, "quality-dependent $c_k$")):
            r, ci = np.array(J[lab]["rms_z"]), np.array(J[lab]["rms_z_ci"])
            ax.vlines(xq + dx, ci[:, 0], ci[:, 1], color=colr, lw=1.0)
            ax.plot(xq + dx, r, marker=mk, ms=5, color=colr, lw=1.2, ls="-" if lab == "calibrated" else (0, (3, 2)), label=name)
        ax.axhline(1, color=INK, lw=0.7, ls=":"); ax.set_xticks(xq); ax.set_xticklabels([f"{c:.2f}" for c in J["quintile_median_c"]])
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.3, 1.7); ax.set_xlabel("junction mean $c_k$ (quintiles)  →  poorer", fontsize=8)
        ax.set_ylabel("RMS($r_j$ / σ$_{j,\\mathrm{pred}}$), held-out tiles", fontsize=8)
        ax.text(0.98, 0.03, f"held-out NLL: constant − $c_k$ = {J['constant']['heldout_nll'] - J['calibrated']['heldout_nll']:+.0f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color=MUTED)
        if col == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        for a in axs[:, col]:
            a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
    for k, a in enumerate(axs.ravel(order="F")):
        a.text(-0.16, 1.06, "abcdef"[k], transform=a.transAxes, fontsize=12, fontweight="bold", va="bottom")
    out = stages[0] / "figures" / "fig6" / "fig6_S_calibration" if not os.environ.get("PLEXUSFLOW_FIGURES") else _paths.FIGURES / "fig6" / "fig6_S_calibration"
    out.parent.mkdir(parents=True, exist_ok=True); plos_style.save(fig, out); print(f"wrote {out}")


if __name__ == "__main__":
    main()
