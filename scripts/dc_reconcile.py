"""Mean-flow (DC) reconstruction of one stage from its measured graph.

PRODUCTION SINCE 2026-11-19: per recording the variance of its mean, v = κ s²/N (s = residual temporal SD about a mean +
3-harmonic fit, N frames, κ pooled heartbeat-block autocorrelation factor); per vessel the inverse-variance mean; constrained
weighted least squares under interior conservation (plexusflow.reconcile.dc, ESTIMATOR = "fixed").  --estimator ql reproduces
the quasi-likelihood method below (production 2026-11-03 … 11-18); the reliability model c is still fitted and stored for it.
--symmetric adds total left = total right arterial inflow (sensitivity; writes dc_reconciled_v2_sym.pkl).

Quasi-likelihood method (before 2026-11-19):

0. (Sensitivity only, off by default) heart-rate normalisation of recording fluxes, plexusflow.reconcile.gain.
1. Reliability scale c = c0 + c_a(1 − align) (alignment only since 2026-11-18), ML-fitted on same-tile junction balance of
   this stage (--tile-terms: per-tile c_T).
2. Per vessel the 1/c²-weighted average of its Kloosterman-valid recordings (between-recording spread kept as a variance
   floor); carved network; quasi-likelihood reconciliation under interior conservation (variance c²Q′² + σ0² evaluated at
   the reconciled flow, projected Newton from the uniform projection; plexusflow.reconcile.dc); emergent boundary
   f = B Q'; passive comparison (--radius: tube-interior [default], consensus or calibrated anatomical radii) + radius shuffle.
Graph regions (plexusflow.reconcile.regions): open-boundary regions are carved with free inflow boundaries; topology-uncertain
regions are carved as mass-conserving black boxes (--no-topology ignores them, reproducing the open-region-only carve).
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
    ap.add_argument("--radius", default="tube", choices=("tube", "consensus", "anatomical"),
                    help="model radius for the passive and AC models: uncalibrated tube-interior segmentation radius (median "
                         "for resistance, √⟨a²⟩ for compliance; default), consensus √(R_graph·R_kl) (sensitivity), or the "
                         "calibrated segmentation radius (scripts/anatomical_radius.py)")
    ap.add_argument("--no-topology", action="store_true", help="ignore topology-uncertain region tags (open-region carve only)")
    ap.add_argument("--estimator", default=None, choices=("fixed", "ql"), help="fixed (production) or ql (quasi-likelihood, before 2026-11-19)")
    ap.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=None,
                    help="total left = total right arterial inflow (default: on, plexusflow.reconcile.dc.SYMMETRIC; --no-symmetric = free split)")
    ap.add_argument("--variance", default=None, choices=("residual", "block", "pooled"),
                    help="variance of each recording's mean: residual of the mean + 2-harmonic fit (default), heartbeat blocks, or pooled κ s²/N")
    a = ap.parse_args()
    G = paths.load_graph()
    M = measurement_table(G)
    from plexusflow.reconcile.reliability import LAST_GAIN
    gfit = LAST_GAIN.get(id(G))
    if gfit:
        print(f"[0] heart-rate correction: flux ∝ f^β, β = {gfit['beta']:.2f} (f_ref {gfit['f_ref']:.2f} Hz; {gfit['n_recordings']} recordings of "
              f"{gfit['n_vessels']} repeat vessels)")
    model, J = fit_reliability(G, M, tile_terms=a.tile_terms)
    cT = np.array(list(model["c_T"].values()))
    print(f"[1] reliability fit: {model['n_junctions']} junctions, {len(cT)} tiles:  {model['formula'].split(';')[0]}:  c_a {model['c_a']:.2f}  "
          f"σ0 {model['sigma0']:.4f}  " + (f"c0 {model['c0']:.3f}" if 'c0' in model else
          f"c_T median {np.median(cT):.2f} [IQR {np.percentile(cT, 25):.2f}–{np.percentile(cT, 75):.2f}]"))
    D = reconcile_dc(G, model, M, radius=a.radius, topology=not a.no_topology, estimator=a.estimator, symmetric=a.symmetric, variance=a.variance)
    D["gain"] = gfit
    if a.radius == "tube":
        pr = D["radius_profiled"]
        print(f"    tube-interior radius profiled for {pr.sum()} of {len(pr)} vessels ({pr.mean():.0%}; others keep the segmented radius); "
              f"median radius / segmented {np.median(D['radius'][pr] / np.array([G.edges[e]['radius'] for e in D['edges']])[pr]):.3f}, "
              f"compliance / resistance radius {np.median(D['radius_compliance'][pr] / D['radius'][pr]):.3f}")
    dom, met, C, im = D.pop("dom"), D["metrics"], D["c"], D["imaged"]
    print(f"[2] carved network: {len(D['edges'])} edges, boundary {len(dom['inj'])} carve + {len(dom['out'])} V;  measured {im.sum()} "
          f"({im.mean():.0%}): trusted {np.sum(im & (C < 0.5))}, weak {np.sum(im & (C >= 0.5) & (C < 0.8))}, untrusted {np.sum(im & (C >= 0.8))}")
    TR = D["topology_regions"]
    print(f"    topology-uncertain regions: {len(TR)} ({D['n_topology_edges']} edges removed{', IGNORED (--no-topology)' if a.no_topology else ''})"
          + "".join(f"\n      region {k}: {r['n_edges']} edges, {len(r['cut'])} cut nodes{' LEAKY' if r['leaky'] else ''}, "
                    f"net exchange {r['net']:+.1e}, throughput {r['throughput']:.2f} nL/s" for k, r in enumerate(TR)))
    cc = met["contracted"]
    print(f"    contracted network (unimaged edges < {cc['k_radii']:g} radii merged into junctions: {cc['n_short_unimaged']}): {cc['n_vessels']} vessels, "
          f"{cc['n_imaged']} imaged ({cc['n_imaged'] / cc['n_vessels']:.0%}), {cc['n_unimaged']} unimaged; {cc['n_junctions']} interior junctions "
          f"({cc['n_merged_junctions']} merged)")
    V = np.array(met["V"])
    v_txt = (f"V1/V2 {V[0]:+.2f}/{V[1]:+.2f} (L {100 * V[0] / V.sum():.1f}%)" if len(V) == 2 else
             f"{len(V)} V nodes: total {V.sum():+.2f} nL/s ({np.sum(V < 0)} withdrawing, {np.sum(V > 0)} injecting)")
    if D.get("fixed"):
        fx = D["fixed"]; Ve = fx["V"][im]
        vtxt = (f"per-recording residual variance s²/M of the mean + 2-harmonic fit ({fx['n_variance_unavailable']} unavailable → median)" if fx["variance"] == "residual"
                else f"per-recording heartbeat-block variance ({fx['n_variance_unavailable']} unavailable → median)" if fx["variance"] == "block"
                else f"pooled κ = {fx['kappa']:.2f}, s floor {fx['s_floor']:.4f} nL/s")
        print(f"[3] fixed inverse-variance fusion + constrained WLS{' (SYMMETRIC: total L = total R)' if D['symmetric'] else ''}: "
              f"{vtxt}; vessel SD of the mean median "
              f"{np.median(np.sqrt(Ve)):.4f} nL/s [IQR {np.percentile(np.sqrt(Ve), 25):.4f}–{np.percentile(np.sqrt(Ve), 75):.4f}]")
    print(f"    {D['estimator']}: objective {D['objective']:.4f};  inflow {met['total_in']:.2f} nL/s (L {met['inj_L']:.2f} / R {met['inj_R']:.2f}, "
          f"withdrawing carve nodes {met['n_withdraw']});  {v_txt}")
    un = D["unstable"]
    print(f"    {'converged' if D['converged'] else 'NOT converged'}; start-dependent vessels ({D['n_starts']} perturbed starts) {un.sum()} of {len(un)} "
          f"({np.sum(un & im)} measured, {np.sum(un & ~im)} unmeasured)")
    print(f"    moved median {met['move_p50']:.2f}, sign flips {met['flips']:.1%};  passive corr {met['corr_lin']:.3f} "
          f"(log {met['corr_log']:.3f}, {a.radius} radii), shuffled radii {D['shuffled_corr']:.3f};  loop share {met['loop_share']:.2f}")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "reliability_model_v1.json").write_text(json.dumps(model, indent=1))
    from plexusflow.reconcile.dc import RESULT
    current = D["symmetric"] and (D.get("fixed") or {}).get("variance") == "residual"      # the method since 2026-11-20
    out = paths.RESULTS / (f"{RESULT}.pkl" if current else
                           f"{RESULT}_{(D.get('fixed') or {}).get('variance', D['estimator_key'])}_{'sym' if D['symmetric'] else 'free'}.pkl")
    out.write_bytes(pickle.dumps(D, protocol=4))
    print(f"wrote {paths.RESULTS / 'reliability_model_v1.json'}, {out}")


if __name__ == "__main__":
    main()
