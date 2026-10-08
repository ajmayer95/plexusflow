"""DC reconciliation (final method, 2026-10-07) from the measured graph.

1. Reliability scale c = c_T + c_f(1 − f) + c_a(1 − align), ML-fitted on same-tile junction balance.
2. Per edge the Kloosterman-valid measurement with the smallest c; carved network; reliability-weighted damped IRLS
   projection onto B_I Q = 0; emergent boundary f = B Q'; passive comparison (consensus radii) + radius shuffle.
See plexusflow.reconcile.  Writes results/reliability_model_v1.json and results/dc_reconciled_v2.pkl.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json, pickle
import numpy as np
import plexusflow.paths as paths
from plexusflow.reconcile import measurement_table, fit_reliability, reconcile_dc


def main():
    G = pickle.loads(paths.GRAPH.read_bytes())
    M = measurement_table(G)
    model, J = fit_reliability(G, M)
    cT = np.array(list(model["c_T"].values()))
    print(f"[1] reliability fit: {model['n_junctions']} junctions, {len(cT)} tiles:  c_f {model['c_f']:.2f}  c_a {model['c_a']:.2f}  "
          f"σ0 {model['sigma0']:.4f}  c_T median {np.median(cT):.2f} [IQR {np.percentile(cT, 25):.2f}–{np.percentile(cT, 75):.2f}]")
    D = reconcile_dc(G, model, M)
    dom, met, C, im = D.pop("dom"), D["metrics"], D["c"], D["imaged"]
    print(f"[2] carved network: {len(D['edges'])} edges, boundary {len(dom['inj'])} carve + {len(dom['out'])} V;  measured {im.sum()} "
          f"({im.mean():.0%}): trusted {np.sum(im & (C < 0.5))}, weak {np.sum(im & (C >= 0.5) & (C < 0.8))}, untrusted {np.sum(im & (C >= 0.8))}")
    V1, V2 = met["V"]
    print(f"[3] IRLS {D['n_iter']} iterations;  inflow {met['total_in']:.2f} nL/s (L {met['inj_L']:.2f} / R {met['inj_R']:.2f}, "
          f"withdrawing carve nodes {met['n_withdraw']});  V1/V2 {V1:+.2f}/{V2:+.2f} (L {100 * V1 / (V1 + V2):.1f}%)")
    print(f"    moved median {met['move_p50']:.2f}, sign flips {met['flips']:.1%};  passive corr {met['corr_lin']:.3f} "
          f"(log {met['corr_log']:.3f}), shuffled radii {D['shuffled_corr']:.3f};  loop share {met['loop_share']:.2f}")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "reliability_model_v1.json").write_text(json.dumps(model, indent=1))
    (paths.RESULTS / "dc_reconciled_v2.pkl").write_bytes(pickle.dumps(D, protocol=4))
    print(f"wrote {paths.RESULTS / 'reliability_model_v1.json'}, {paths.RESULTS / 'dc_reconciled_v2.pkl'}")


if __name__ == "__main__":
    main()
