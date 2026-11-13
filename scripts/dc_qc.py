"""Review list for the DC reconstruction: gross-error tests and structural flags (plexusflow.reconcile.qc).

Needs results/reliability_model_v1.json (scripts/dc_reconcile.py).  By default the field is reconstructed here on the whole
network with topology-uncertain region tags ignored (flags inside those areas are then visible too); --topology reviews the
production field (results/dc_reconciled_v2.pkl) instead.  Writes results/dc_qc.csv (one
row per flag: kind, vessel or junction, tile, statistic, location, hotspot) and results/dc_qc.json (summary + hotspots ranked
by the reconstructed flow they involve).  Flags are for review in the workbench (video, segmentation, region tags); they never
change the reconstruction.  Usage: python scripts/dc_qc.py [--z 4] [--no-loops] [--topology]
"""
import sys, csv, json, pickle, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as paths
from plexusflow.reconcile import measurement_table, carved_domain, reconcile_dc
from plexusflow.reconcile.qc import run_qc, Z_FLAG


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--z", type=float, default=Z_FLAG, help=f"|z| flag threshold for the statistical tests (default {Z_FLAG:g})")
    ap.add_argument("--no-loops", action="store_true", help="skip the loop-flow decomposition (the slow part, minutes on a large stage)")
    ap.add_argument("--top", type=int, default=10, help="hotspots to print")
    ap.add_argument("--topology", action="store_true", help="review the production field (with topology-uncertain regions) instead")
    a = ap.parse_args()
    G = paths.load_graph(); M = measurement_table(G)
    model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model.get("c_T", {}).items()}
    if a.topology:
        D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes()); dom = carved_domain(G, topology=D.get("n_topology_edges", 0) > 0)
        if [tuple(e) for e in dom["edges"]] != [tuple(e) for e in D["edges"]]:
            sys.exit("results/dc_reconciled_v2.pkl was built on a different graph or region tags: rerun scripts/dc_reconcile.py first")
    else:
        dom = carved_domain(G, topology=False); D = reconcile_dc(G, model, M, dom=dom, n_starts=0, n_shuffle=0); D.pop("dom", None)
    print(f"{paths.STAGE.name}: reviewing the field {'with' if a.topology else 'without'} topology-uncertain regions ({len(D['edges'])} vessels)")
    rows, hot, summ = run_qc(G, D, dom, M, model, loops=not a.no_loops, z_flag=a.z)
    keys = ["hotspot", "kind", "edge", "node", "tile", "z", "y", "Qp", "redundancy", "loop_flow", "threshold", "R_graph_px", "R_kl_over_graph",
            "thicker_than_neighbours", "path_mid"]
    rows.sort(key=lambda r: (r["hotspot"] or 10 ** 6, r["kind"], -abs(r.get("z") or r.get("loop_flow") or r.get("Qp") or 0)))
    with open(paths.RESULTS / "dc_qc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    (paths.RESULTS / "dc_qc.json").write_text(json.dumps(dict(summary=summ, hotspots=hot), indent=1, default=str))
    print(f"{paths.STAGE.name}: {summ['n_flags']} flags in {summ['n_hotspots']} hotspots (|z| > {a.z:g}): " + ", ".join(f"{k} {v}" for k, v in summ["by_kind"].items()))
    print(f"  measurement test: {summ['n_testable']} vessels testable by conservation, RMS z {summ['measurement_z_rms']:.2f} (calibrated model: ≈ 1)"
          + (f"; loop share {summ['loop_share']:.1%}" if summ["loop_share"] is not None else ""))
    for h in hot[:a.top]:
        print(f"  hotspot {h['hotspot']:3d}: {h['n_flags']:3d} flags ({', '.join(f'{k} {v}' for k, v in h['kinds'].items())}), "
              f"{h['n_edges']} vessels, Σ|Q'| {h['flow']:.2f} nL/s, tiles {h['tiles']}")
    print(f"wrote {paths.RESULTS / 'dc_qc.csv'}, {paths.RESULTS / 'dc_qc.json'}")


if __name__ == "__main__":
    main()
