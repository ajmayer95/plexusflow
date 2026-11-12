"""How much does the distensibility fit depend on the pulsatile closure of topology-uncertain regions?

Closures (plexusmech.ac.model.build_context topo=…): 'sealed' (no oscillatory exchange — diagnostic baseline), 'flux'
(interfaces driven by the local measured ratio, fixed inputs), 'supernode' (conducting compliant compartment, compliance
D·V_region × s, s ∈ --scales).  Same field, eligibility, weights, vein forcing and profile fit as scripts/mech/ac_estimate.py.
Per closure: best D (grid and parabola), ΔNLL profile; residuals at the best fit (H1 / H2 phase, |log10 PI|) for vessels
near (≤ --near px of a region interface) vs far; --folds k: k-fold over supported vessels (fit on k−1 folds, predict the
held-out vessels' H1/H2 phase and PI at the training optimum, tile gauge from training).  Writes
results/ac_topology_sensitivity.json.  Usage: python scripts/mech/ac_topology_sensitivity.py [--scales 0.25,1,4] [--folds 3]
"""
import sys, json, pickle, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from scipy.spatial import cKDTree
import plexusflow.paths as paths
from plexusflow.reconcile import carved_domain
from plexusflow.ac.field import supported
from plexusmech.ac.model import build_context, ac_reliability, vein_forcing, group_response
from plexusmech.ac.profile import fit_profile, quadratic_min
from plexusmech.ac.group_bc import groups


def predict(C, R, el_eval):
    """H1/H2 phase residuals and log10 PI residual for edges el_eval at the fit R (its D, η, x, tile gauge)."""
    gid = groups(C); x = np.asarray(R["x"]); D_, eta = R["D"], R["eta"]
    B1, B2 = group_response(C, 1, D_, eta, gid, el_eval), group_response(C, 2, D_, eta, gid, el_eval)
    q1 = B1[:, :4] @ np.exp(x[:4] + 1j * x[8:]) + B1[:, 4]; q2 = B2[:, :4] @ np.exp(x[4:8] + 2j * x[8:]) + B2[:, 4]
    Qp, Qs = C["Qp"][el_eval], C["Qs_dc"][el_eval]; s = np.sign(Qs)
    tile = np.asarray(C["_tile"])[el_eval]; dl = np.radians([R["delta_deg"].get(int(t), 0.0) for t in tile])
    o1 = np.angle(C["H"][1][el_eval] * np.sign(Qp)); o2 = np.angle(C["H"][2][el_eval] * np.sign(Qp))
    d1 = np.angle(np.exp(1j * (o1 - dl - np.angle(q1 * s)))); d2 = np.angle(np.exp(1j * (o2 - 2 * dl - np.angle(q2 * s))))
    pi_m = np.abs(q1) / np.maximum(np.abs(Qs), 1e-30); pi_o = np.abs(C["H"][1][el_eval]) / np.abs(Qp)
    return np.degrees(np.abs(d1)), np.degrees(np.abs(d2)), np.abs(np.log10(np.maximum(pi_m, 1e-12) / pi_o))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", default="ac_field_v7"); ap.add_argument("--scales", default="0.25,1,4")
    ap.add_argument("--near", type=float, default=300.0); ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--d-range", nargs=3, type=float, default=(2.5e-4, 8e-4, 15)); ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    from plexusflow.config import StageConfig
    cfg = StageConfig.load(paths.STAGE); veins = cfg.ac_veins
    block = [int(t) for t in cfg.ac_gauge_block] if cfg.ac_gauge_block else None
    G = paths.load_graph(); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    A = pickle.loads((paths.RESULTS / f"{a.field}.pkl").read_bytes())
    model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    dom = carved_domain(G); sup = supported(A); Am = dict(A, elig=sup); c = ac_reliability(G, Am, model)
    d_grid = np.logspace(np.log10(a.d_range[0]), np.log10(a.d_range[1]), int(a.d_range[2]))
    cut = [n for r in dom["regions"] for n in r["cut"]]
    Tc = cKDTree(np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in cut])) if cut else None
    mid = np.array([np.asarray(G.edges[e]["path"], float)[len(G.edges[e]["path"]) // 2] for e in dom["edges"]])
    near = (Tc.query(mid)[0] <= a.near) if Tc else np.zeros(len(mid), bool)
    print(f"{paths.STAGE.name}: {sup.sum()} supported vessels ({np.sum(sup & near)} within {a.near:g} px of a region interface); "
          f"regions {[(r['n_edges'], 'leaky' if r['leaky'] else 'sealed') for r in dom['regions']]}", flush=True)
    variants = [("sealed", 1.0), ("flux", 1.0)] + [("supernode", float(s)) for s in a.scales.split(",")]
    out = {}
    def ctx(mode, s, elig):
        C0 = build_context(D, dict(A, elig=elig), dom, topo=mode, supernode_scale=s)
        C, _ = vein_forcing(C0, A, mode=veins) if veins != "local" else (C0, None)
        C["_tile"] = A["tile"]; return C
    for mode, s in variants:
        name = mode if mode != "supernode" else f"supernode ×{s:g}"
        C = ctx(mode, s, sup)
        R = fit_profile(C, c, A, d_grid=d_grid, n_jobs=a.n_jobs, progress=None, gauge_block=block)
        prof = np.nanmin(R["S"], axis=0) - np.nanmin(R["S"])
        e1, e2, ep = predict(C, R, sup)
        n_, f_ = near[sup], ~near[sup]
        res = dict(D=R["D"], D_parabola=quadratic_min(d_grid, prof), nll=R["nll"], profile=prof.tolist(), D_grid=d_grid.tolist(),
                   near=dict(h1_deg=float(np.median(e1[n_])), h2_deg=float(np.median(e2[n_])), logpi=float(np.median(ep[n_])), n=int(n_.sum())),
                   far=dict(h1_deg=float(np.median(e1[f_])), h2_deg=float(np.median(e2[f_])), logpi=float(np.median(ep[f_])), n=int(f_.sum())))
        if a.folds > 1:
            idx = np.flatnonzero(sup); rng = np.random.default_rng(0); fold = rng.integers(0, a.folds, len(idx)); H1, H2, PI, NR = [], [], [], []
            for k in range(a.folds):
                tr = np.zeros(len(sup), bool); tr[idx[fold != k]] = True; te = np.zeros(len(sup), bool); te[idx[fold == k]] = True
                Ct = ctx(mode, s, tr); Rt = fit_profile(Ct, c, A, d_grid=d_grid, n_jobs=a.n_jobs, progress=None, gauge_block=block)
                Cp = ctx(mode, s, sup); h1, h2, pp = predict(Cp, Rt, te); H1 += list(h1); H2 += list(h2); PI += list(pp); NR += list(near[te])
            H1, H2, PI, NR = map(np.asarray, (H1, H2, PI, NR))
            res["heldout"] = dict(h1_deg=float(np.median(H1)), h2_deg=float(np.median(H2)), logpi=float(np.median(PI)),
                                  h1_deg_far=float(np.median(H1[~NR])), h1_deg_near=float(np.median(H1[NR])) if NR.any() else None)
        out[name] = res
        ho = res.get("heldout", {})
        print(f"{name:16s}: D {res['D']:.2e} (parabola {res['D_parabola']:.2e}) | fit residual H1/H2 phase, |log PI|: near {res['near']['h1_deg']:.0f}°/{res['near']['h2_deg']:.0f}°/"
              f"{res['near']['logpi']:.2f}, far {res['far']['h1_deg']:.0f}°/{res['far']['h2_deg']:.0f}°/{res['far']['logpi']:.2f}"
              + (f" | held-out H1 {ho['h1_deg']:.0f}° (near {ho['h1_deg_near']:.0f}°, far {ho['h1_deg_far']:.0f}°), H2 {ho['h2_deg']:.0f}°, |log PI| {ho['logpi']:.2f}" if ho else ""), flush=True)
    (paths.RESULTS / "ac_topology_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {paths.RESULTS / 'ac_topology_sensitivity.json'}")


if __name__ == "__main__":
    main()
