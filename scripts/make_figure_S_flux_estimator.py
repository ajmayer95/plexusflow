"""SI figure — the flux estimator in detail, for the representative vessel of Fig 3 (edge 75489–75638, tile 36).

  a  per-cross-section mean flux Q̄_s = (π/2) R_s² v_CL,s along the vessel, with the vessel mean (mean of the shape-valid
     sections; Kloosterman et al. 2014 section fits): consistency of the flux estimate along one vessel
  b  fitted section radius R_s along the vessel, against the segmented radius
  c  the per-frame model: one parabola of FIXED radius whose centreline speed v_CL(t) follows the waveform, so
     Q(t) = Q̄ v_CL(t)/⟨v_CL⟩; profiles shown at the waveform's peak and trough (reconstructed from Q(t)/Q̄)

Inputs as Fig 3 (load_vessel in make_figure3.py).  Writes figures/figS_flux_estimator/figS_flux_estimator.{png,pdf,tif}.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as _paths
from plexusflow.analyze.constants import PX_SIZE_UM
import plos_style
from make_figure3 import load_vessel, EDGE, INK, VCOL


def main():
    plos_style.apply()
    V = load_vessel(); G, valid, arc, Qs, Rs, Qbar, t, f0, a0, model = (V[k] for k in ("G", "valid", "arc", "Qs", "Rs", "Qbar", "t", "f0", "a0", "model"))
    R_seg = float(G.edges[EDGE].get("radius") or np.nan) * PX_SIZE_UM
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.5, 2.4), gridspec_kw=dict(wspace=0.45, left=0.08, right=0.985, top=0.86, bottom=0.2))

    axA.plot(arc[valid], Qs[valid], "o-", ms=2.0, lw=0.6, color=INK, zorder=3, label="Cross-sections")
    axA.axhline(Qbar, color=VCOL, lw=1.0, ls="--", zorder=2, label=f"Vessel mean {Qbar:.2f} nL/s")
    axA.set_ylim(0, np.nanmax(Qs[valid]) * 1.3); axA.set_xlabel("Distance along vessel (µm)", labelpad=1)
    axA.set_ylabel("Mean flux (nL/s)", labelpad=1); axA.legend(loc="lower center", frameon=False, handlelength=1.5, labelspacing=0.2)

    axB.plot(arc[valid], Rs[valid], "o-", ms=2.0, lw=0.6, color=INK, label="Fitted $R_s$")
    axB.axhline(R_seg, color="0.5", lw=1.0, ls=":", label=f"Segmented {R_seg:.0f} µm")
    axB.set_ylim(0, np.nanmax(Rs[valid]) * 1.35); axB.set_xlabel("Distance along vessel (µm)", labelpad=1)
    axB.set_ylabel("Section radius (µm)", labelpad=1); axB.legend(loc="lower center", frameon=False, handlelength=1.5, labelspacing=0.2)

    beat = (t >= 1.0 / f0) & (t < 2.0 / f0); tb = t[beat]; Qm = model(tb)
    rr = np.linspace(-R_seg, R_seg, 100)
    for tt_, col, nm in ((tb[np.argmax(Qm)], "#c0392b", "peak"), (tb[np.argmin(Qm)], "#7fb3d5", "trough")):
        axC.plot(rr, model(np.array([tt_]))[0] / a0 * (1 - (rr / R_seg) ** 2), color=col, lw=1.2, label=nm)
    axC.set_xlim(-1.1 * R_seg, 1.1 * R_seg); axC.set_ylim(0, None)
    axC.set_xlabel("Radial position (µm)", labelpad=1); axC.set_ylabel(r"$v\,/\,\langle v_{\rm CL}\rangle$", labelpad=1)
    axC.set_ylim(0, axC.get_ylim()[1] * 1.3)
    axC.legend(loc="upper right", frameon=False, handlelength=1.2, labelspacing=0.2)
    axC.set_title(r"$Q(t)=\bar Q\,v_{\rm CL}(t)/\langle v_{\rm CL}\rangle$, fixed radius", fontsize=8, pad=3)
    for ax, l in zip((axA, axB, axC), "abc"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.28, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    plos_style.save(fig, _paths.FIGURES / "figS_flux_estimator" / "figS_flux_estimator")


if __name__ == "__main__":
    main()
