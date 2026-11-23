"""Clock measurements between tiles from the speed pulse of shared vessels (plexusflow.ac.speed) — the one step of the pulse chain
that reads the raw movies.

For every tile with a detected heart rate: dense optical flow of each frame (the settings of the flux measurement), sampled along and
across every vessel of the graph that the tile images, giving the velocity along the vessel at each path point in each frame.  For
every vessel seen in two tiles: the speed of the velocity pooled over the common piece of vessel, fitted with mean + 2 harmonics;
the phase difference of the first harmonics and its variance.  Each movie uses the same frames as the flux measurement.
Writes results/speed_pairs_v1.pkl (a few hundred kB): one entry per (vessel, tile pair) — tiles a, b, d (rad, estimates ψ_a − ψ_b),
var, reversing, n_points — plus the settings.  It is part of a data release, so the rest of the chain runs without the movies.
"""
import sys
import argparse
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import tifffile
from joblib import Parallel, delayed
import plexusflow.paths as paths
from plexusflow.config import StageConfig
from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch, graph_radius_to_tile_px
from plexusflow.ac import speed as SP


def one_tile(files, layout):
    return SP.along_vessel_velocity(np.stack([tifffile.imread(f) for f in files]), layout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=4, help="tiles processed at once (each holds one movie in memory)")
    a = ap.parse_args()
    cfg = StageConfig.load(paths.STAGE); G = paths.load_graph(); geom = load_tile_geometry(cfg); tf0 = G.graph["tile_f0_piv"]; n_frames = {}
    for e in G.edges:                                           # frames of the flux series of each tile (movie cut before its fade)
        for m in G.edges[e].get("measurements_piv") or []:
            if m.get("Q_t") is not None:
                n_frames[int(m["tile_id"])] = len(m["Q_t"])
    tiles = sorted(int(t) for t, f in tf0.items() if f and np.isfinite(f) and int(t) in n_frames)
    pts = {tuple(e): SP.path_points(G.edges[e]["path"]) for e in G.edges if G.edges[e].get("path") is not None}
    pts = {e: p for e, p in pts.items() if p is not None}; radii = {e: float(G.edges[e]["radius"]) for e in pts}; jobs = []
    for t in tiles:
        files = [str(f) for f in cfg.raw_tile_frames(t)][:n_frames[t] + 1]
        if len(files) < 50:
            sys.exit(f"raw movie of tile {t} not found (external_raw_video_dir in plexusflow.yaml)")
        jobs.append((files, SP.tile_layout(pts, radii, lambda xy, t=t: graph_to_tile_batch(xy, geom, t), lambda r, t=t: graph_radius_to_tile_px(r, geom, t),
                                           geom.tiles[t].video_shape)))
    print(f"{paths.STAGE.name}: {len(tiles)} tiles, {len(pts)} vessels; optical flow {SP.flow_settings()[0]}")
    V = dict(zip(tiles, Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_tile)(f, lay) for f, lay in jobs)))
    out = SP.speed_pairs(V, {t: float(tf0[t]) for t in tiles}); out["tiles"] = tiles
    out["settings"] = dict(step=SP.STEP, n_across=SP.N_ACROSS, width_frac=SP.WIDTH_FRAC, min_points=SP.MIN_POINTS, rev_frac=SP.REV_FRAC, rev_smooth=SP.REV_SMOOTH,
                           flow=SP.flow_settings()[0], bg_percentile=SP.flow_settings()[1], frame_spacing=1, n_frames={t: n_frames[t] for t in tiles})
    paths.RESULTS.mkdir(parents=True, exist_ok=True); f = paths.RESULTS / f"{SP.RESULT}.pkl"; f.write_bytes(pickle.dumps(out, protocol=4))
    rev = out["reversing"]
    print(f"{len(rev)} pairs on {len(out['edges'])} vessels; {int(rev.sum())} ({rev.mean():.0%}) with a reversing recording; "
          f"median common piece {SP.STEP * np.median(out['n_points']):.0f} px")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
