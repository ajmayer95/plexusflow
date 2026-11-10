"""Centreline-refinement warning: did the speed-ridge refinement pull the centreline toward another vessel?

For every recording with cached batch inputs, the vessel's graph path is mapped into the tile as the batch maps it and
compared with the refined centreline the flux was measured on (cl_used):
  shift_R  = largest distance of a refined point from the mapped graph centreline, in sampling radii (the refinement is
             clipped at ±1 R, so ≈ 1 means it hit the limit somewhere);
  nbr_frac = share of refined points that moved > 0.5 R from their own centreline AND lie closer to another vessel's;
  flag     = nbr_frac > 0.25 (21S 4 %, 27S 7 % of Kloosterman-valid recordings; 5 % / 10 % of all cached ones).
Flagged recordings disagree more with repeat recordings of the same vessel at equal reliability c (21S +10–20 % in
|log ratio| per c quintile; 27S mainly in the best quintile), so the flag is a diagnostic for inspection, not a weight.
Writes results/refine_check.pkl  {(tile, (u, v)): dict(shift_R, mean_shift_R, nbr_frac, flag, R)}.

    PLEXUSFLOW_STAGE=.../PlexusFlow21S python3 scripts/refine_check.py
"""
import sys, gzip, pickle, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import plexusflow.paths as P

NBR_FLAG = 0.25


def prior_cl(G, A, B, geom, tid, H, W):
    from plexusflow.analyze.mosaic_piv import _path_a_to_b, _largest_inbounds_segment
    from plexusflow.analyze.geometry import graph_to_tile_batch
    seg = _largest_inbounds_segment(graph_to_tile_batch(_path_a_to_b(G, A, B, np.asarray(G.edges[A, B]["path"], float)), geom, tid), H, W)
    return np.asarray(seg, float)


def dense(p, step=0.5):
    if len(p) < 2:
        return p
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]); n = max(int(s[-1] / step), 2)
    t = np.linspace(0, s[-1], n); return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def work(fn, stage):
    import os; os.environ["PLEXUSFLOW_STAGE"] = str(stage)
    from plexusflow.config import StageConfig
    from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch
    G = P.load_graph(); geom = load_tile_geometry(StageConfig.load(P.STAGE))
    d = pickle.load(gzip.open(fn)); tid = int(d["tile"]); kl = d["kl_inputs"]
    H, W = 10 ** 5, 10 ** 5                       # in-bounds is decided by the batch's crop; ROI offsets handle the rest
    # every graph edge's centreline in this tile's frame, labelled
    pts, lab = [], []
    for i, (u, v, dd) in enumerate(G.edges(data=True)):
        p = dd.get("path")
        if p is None or len(p) < 2:
            continue
        q = dense(graph_to_tile_batch(np.asarray(p, float), geom, tid))
        pts.append(q); lab.append(np.full(len(q), i))
    eid = {(u, v): i for i, (u, v) in enumerate(G.edges())}
    pts = np.vstack(pts); lab = np.concatenate(lab); T = cKDTree(pts); out = {}
    for (A, B), ki in kl.items():
        try:
            r0, c0 = ki.get("roi_offset", (0, 0)); cl = np.asarray(ki["cl_used"], float) + [r0, c0]; R = float(ki["R_prior_px"])
            own = dense(prior_cl(G, A, B, geom, tid, H, W)); To = cKDTree(own)
            d_own = To.query(cl)[0]
            k = eid.get((A, B), eid.get((B, A)))
            dd, ii = T.query(cl, k=12); other = np.where(lab[ii] != k, dd, np.inf).min(axis=1)
            out[(tid, (A, B))] = dict(shift_R=float(d_own.max() / R), mean_shift_R=float(d_own.mean() / R),
                                      nbr_frac=float(np.mean((d_own > 0.5 * R) & (other < d_own))), R=R)
        except Exception as e:
            out[(tid, (A, B))] = dict(err=str(e))
    return out


def main():
    D = {}
    for o in Parallel(n_jobs=4)(delayed(work)(fn, P.STAGE) for fn in sorted(glob.glob(str(P.STAGE / "batch_kl_inputs" / "tile_*.pkl.gz")))):
        D.update(o)
    err = [k for k, v in D.items() if "err" in v]; D = {k: {**v, "flag": v["nbr_frac"] > NBR_FLAG} for k, v in D.items() if "err" not in v}
    s = np.array([v["shift_R"] for v in D.values()]); nb = np.array([v["nbr_frac"] for v in D.values()])
    print(f"{P.STAGE.name}: {len(D)} recordings ({len(err)} failed); max shift ≥ 0.9 R: {np.mean(s >= 0.9):.1%}; "
          f"pulled toward a neighbour (nbr_frac > {NBR_FLAG}): {np.mean(nb > NBR_FLAG):.1%}")
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "refine_check.pkl").write_bytes(pickle.dumps(D))
    print(f"  wrote {P.RESULTS / 'refine_check.pkl'}")


if __name__ == "__main__":
    main()
