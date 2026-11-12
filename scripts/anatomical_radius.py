"""Anatomical radius R_k(s) for every carved vessel: float mask EDT along the centreline + the calibrated boundary offset Δ.

Δ is calibrated on the lumen-width measurements (results/lumen_width_calib.pkl of the stage, or --calib-stage for another stage of
the same imaging setup): R_true = k_est · R_seg for each calibration vessel, Δ = median(R_true − mask-boundary radius).
Writes results/anatomical_radius.pkl  {edges, median, cond, taper, scatter, fallback, offset_px, calibration}.

    PLEXUSFLOW_STAGE=.../PlexusFlow21S python3 scripts/anatomical_radius.py [--calib-stage STAGE_DIR] [--offset PX | --factor F | --net3]
"""
import sys, argparse, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, tifffile
import plexusflow.paths as P
from plexusflow.reconcile.network import carved_domain
from plexusflow.reconcile.radius import mask_edt, calibrate_offset, anatomical_radii


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calib-stage", default=None, help="stage whose lumen-width calibration to use (default: this stage)")
    ap.add_argument("--offset", type=float, default=None, help="use this Δ (px) instead of calibrating")
    ap.add_argument("--factor", type=float, default=None, help="multiplicative calibration instead: R = factor · mask boundary (Δ = 0)")
    ap.add_argument("--net3", action="store_true", help="sensitivity: the stored net3 radius as is (no mask profile, no calibration)")
    a = ap.parse_args()
    G = P.load_graph(); dom = carved_domain(G); edges = list(dom["edges"])
    edt = mask_edt(tifffile.imread(str(P.STAGE / "mask" / "mask.tif")))
    cal = None
    if a.factor is not None or a.net3:
        off = 0.0; print(f"{P.STAGE.name}: " + ("stored net3 radius" if a.net3 else f"R = {a.factor:.3f} × mask boundary"))
    elif a.offset is None:
        cst = Path(a.calib_stage) if a.calib_stage else P.STAGE; calib = cst / "results" / "lumen_width_calib.pkl"
        calib_G, calib_edt = (G, edt) if cst.resolve() == P.STAGE.resolve() else \
            (P.load_graph(cst / "graph" / "graph_vessels_piv.gpickle"), mask_edt(tifffile.imread(str(cst / "mask" / "mask.tif"))))
        rows = []
        for r in pickle.loads(calib.read_bytes()):
            e = tuple(r["edge"]); e = e if calib_G.has_edge(*e) else e[::-1]
            if np.isfinite(r.get("k_est", np.nan)) and calib_G.has_edge(*e):
                rows.append(dict(edge=e, R_true=float(r["k_est"]) * float(calib_G.edges[e]["radius"])))
        cal = calibrate_offset(calib_edt, calib_G, rows); off = cal["offset_px"]
        print(f"{P.STAGE.name}: boundary → lumen offset Δ = {off:+.2f} px (95 % {cal['offset_ci'][0]:+.2f} to {cal['offset_ci'][1]:+.2f}; "
              f"per-vessel SD {cal['offset_sd']:.2f} px; n {cal['n']}; multiplicative alternative ×{cal['factor']:.2f}) from {calib}")
    else:
        off = a.offset; print(f"{P.STAGE.name}: boundary → lumen offset Δ = {off:+.2f} px (given)")
    A = anatomical_radii(G, edt, edges, off)
    if a.factor is not None:
        A = {e: {**v, "median": v["median"] * a.factor, "cond": v["cond"] * a.factor} for e, v in A.items()}
    if a.net3:
        A = {e: dict(median=float(G.edges[e]["radius"]), cond=float(G.edges[e]["radius"]), taper=1.0, scatter=np.nan, fallback=False) for e in edges}
    med = np.array([A[e]["median"] for e in edges]); cond = np.array([A[e]["cond"] for e in edges]); fb = np.array([A[e]["fallback"] for e in edges])
    rg = np.array([G.edges[e]["radius"] for e in edges], float)
    print(f"  {len(edges)} carved vessels ({fb.sum()} without a usable mask profile → net3 radius − 1 px + Δ)")
    print(f"  anatomical median radius / net3 radius: median ×{np.median(med / rg):.3f}; conduction-equivalent / median: "
          f"median {np.median(cond / med):.3f}, < 0.9 for {np.mean(cond / med < 0.9):.1%}")
    out = dict(edges=edges, median=med, cond=cond, taper=np.array([A[e]["taper"] for e in edges]),
               scatter=np.array([A[e]["scatter"] for e in edges]), fallback=fb, offset_px=off, factor=a.factor, net3=a.net3, calibration=cal, calib_stage=str(a.calib_stage or P.STAGE))
    (P.RESULTS / "anatomical_radius.pkl").write_bytes(pickle.dumps(out))
    print(f"  wrote {P.RESULTS / 'anatomical_radius.pkl'}")


if __name__ == "__main__":
    main()
