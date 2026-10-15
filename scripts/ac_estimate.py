"""Joint (D, η) estimate with group-corrected boundary forcing (plexusflow.ac.group_bc).

Needs results/dc_reconciled_v2.pkl, results/ac_field_v2.pkl, results/reliability_model_v1.json.
Reports the minimum for the fixed measurement-derived forcing and for the group-corrected forcing, ΔNLL(η = 0),
the fitted group corrections and implied boundary pulsatility.
Writes results/ac_group_bc_estimate.pkl and figures/ac/ac_group_bc_estimate.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plexusflow.paths as paths
from plexusflow.reconcile import carved_domain
from plexusflow.ac.model import build_context, ac_reliability
from plexusflow.ac.group_bc import estimate, group_summary


def load():
    G = paths.load_graph()
    D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    A = pickle.loads((paths.RESULTS / "ac_field_v2.pkl").read_bytes())
    model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    return G, D, A, model, build_context(D, A, carved_domain(G))


def main():
    G, D, A, model, C = load()
    R = estimate(C, ac_reliability(G, A, model))
    Dg, Eg = R["D_grid"], R["eta_grid"]; nD = len(Dg)
    for lab, M_ in (("fixed measured BC", R["S_fixed"]), ("group-corrected BC", R["S"])):
        i, j = np.unravel_index(np.nanargmin(M_), M_.shape); dd = M_ - np.nanmin(M_)
        print(f"{lab:20s}: min D {Dg[j]:.2e}, η {Eg[i]:.1f};  NLL {np.nanmin(M_):.0f};  ΔNLL(η=0) {dd.min(axis=1)[0]:.0f};  "
              f"D profile ±1 step {dd.min(axis=0)[max(j - 1, 0)]:.0f}/{dd.min(axis=0)[min(j + 1, nD - 1)]:.0f}")
    for g, r in group_summary(C, D["fb"], R["x_min"]).items():
        print(f"  {g:8s} timing {r['timing']:+.3f} beat;  H1 ×{r['amp1']:.2f} → PI {r['PI1']:.2f};  H2 ×{r['amp2']:.2f} → PI {r['PI2']:.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (lab, M_) in zip(axs, (("fixed measured BC", R["S_fixed"]), ("group-corrected BC (L, R, V1, V2)", R["S"]))):
        dd = np.minimum(M_ - np.nanmin(M_), 200); cs = ax.contourf(Dg, Eg, dd, levels=30, cmap="viridis_r")
        ii, jj = np.unravel_index(np.nanargmin(M_), M_.shape); ax.plot(Dg[jj], Eg[ii], "r*", ms=16, mec="k")
        ax.set_xscale("log"); ax.set_xlabel("D (1/Pa)"); ax.set_ylabel("η"); fig.colorbar(cs, ax=ax, label="ΔNLL (capped 200)")
        ax.set_title(f"{lab}: min D {Dg[jj]:.1e}, η {Eg[ii]:.1f}", fontsize=10)
    fig.tight_layout(); out = paths.FIGURES / "ac" / "ac_group_bc_estimate.png"; out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120)
    (paths.RESULTS / "ac_group_bc_estimate.pkl").write_bytes(pickle.dumps(dict(S=R["S"], S_fixed=R["S_fixed"], best=R["best"], D=Dg, eta=Eg)))
    print(f"wrote {out}, {paths.RESULTS / 'ac_group_bc_estimate.pkl'}")


if __name__ == "__main__":
    main()
