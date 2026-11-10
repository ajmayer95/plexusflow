"""DC reconciliation (final method, 2026-10-07) from the measured graph.

1. Reliability scale c = c0 + c_p(1 − p) + c_a(1 − align) (p = parabolicity; f before 2026-11-10), ML-fitted on same-tile junction balance (--tile-terms: per-tile c_T).
2. Per edge the Kloosterman-valid measurement with the smallest c; carved network; quasi-likelihood reconciliation under
   B_I Q = 0 (variance c²Q′² + σ0², projected Newton from the uniform projection; plexusflow.reconcile.dc); emergent
   boundary f = B Q'; passive comparison (--radius: consensus or calibrated anatomical radii) + radius shuffle.
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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile-terms", action="store_true", help="per-tile c_T (the pre-2026-10-19 model) instead of one c0")
    ap.add_argument("--radius", default="consensus", choices=("consensus", "anatomical"),
                    help="radius for the passive model and the AC model: consensus √(R_graph·R_kl) or the calibrated "
                         "segmentation radius (scripts/anatomical_radius.py)")
    a = ap.parse_args()
    G = paths.load_graph()
    M = measurement_table(G)
    model, J = fit_reliability(G, M, tile_terms=a.tile_terms)
    cT = np.array(list(model["c_T"].values()))
    print(f"[1] reliability fit: {model['n_junctions']} junctions, {len(cT)} tiles:  c_{model['profile_term']} {model['c_' + model['profile_term']]:.2f}  c_a {model['c_a']:.2f}  "
          f"σ0 {model['sigma0']:.4f}  " + (f"c0 {model['c0']:.3f}" if 'c0' in model else
          f"c_T median {np.median(cT):.2f} [IQR {np.percentile(cT, 25):.2f}–{np.percentile(cT, 75):.2f}]"))
    D = reconcile_dc(G, model, M, radius=a.radius)
    dom, met, C, im = D.pop("dom"), D["metrics"], D["c"], D["imaged"]
    print(f"[2] carved network: {len(D['edges'])} edges, boundary {len(dom['inj'])} carve + {len(dom['out'])} V;  measured {im.sum()} "
          f"({im.mean():.0%}): trusted {np.sum(im & (C < 0.5))}, weak {np.sum(im & (C >= 0.5) & (C < 0.8))}, untrusted {np.sum(im & (C >= 0.8))}")
    cc = met["contracted"]
    print(f"    contracted network (unimaged edges < {cc['k_radii']:g} radii merged into junctions: {cc['n_short_unimaged']}): {cc['n_vessels']} vessels, "
          f"{cc['n_imaged']} imaged ({cc['n_imaged'] / cc['n_vessels']:.0%}), {cc['n_unimaged']} unimaged; {cc['n_junctions']} interior junctions "
          f"({cc['n_merged_junctions']} merged)")
    V = np.array(met["V"])
    v_txt = (f"V1/V2 {V[0]:+.2f}/{V[1]:+.2f} (L {100 * V[0] / V.sum():.1f}%)" if len(V) == 2 else
             f"{len(V)} V nodes: total {V.sum():+.2f} nL/s ({np.sum(V < 0)} withdrawing, {np.sum(V > 0)} injecting)")
    print(f"[3] quasi-likelihood: {D['n_iter']} projected-Newton steps, objective {D['objective']:.4f};  inflow {met['total_in']:.2f} nL/s (L {met['inj_L']:.2f} / R {met['inj_R']:.2f}, "
          f"withdrawing carve nodes {met['n_withdraw']});  {v_txt}")
    un = D["unstable"]
    print(f"    {'converged' if D['converged'] else 'NOT converged'}; start-dependent vessels ({D['n_starts']} perturbed starts) {un.sum()} of {len(un)} "
          f"({np.sum(un & im)} measured, {np.sum(un & ~im)} unmeasured)")
    print(f"    moved median {met['move_p50']:.2f}, sign flips {met['flips']:.1%};  passive corr {met['corr_lin']:.3f} "
          f"(log {met['corr_log']:.3f}, {a.radius} radii), shuffled radii {D['shuffled_corr']:.3f};  loop share {met['loop_share']:.2f}")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "reliability_model_v1.json").write_text(json.dumps(model, indent=1))
    (paths.RESULTS / "dc_reconciled_v2.pkl").write_bytes(pickle.dumps(D, protocol=4))
    print(f"wrote {paths.RESULTS / 'reliability_model_v1.json'}, {paths.RESULTS / 'dc_reconciled_v2.pkl'}")


if __name__ == "__main__":
    main()
