"""Build a frozen, self-contained data release of a stage (graph, viewer inputs, final results).

Writes <out>/ (default: <stage>_release_v1 next to the stage folder):
  plexusflow.yaml            stage config without raw-video dependence (viewer runs read-only)
  graph/graph_vessels_piv.gpickle   the frozen measured graph
  graph/export/              portable copy: nodes.csv, edges.csv, measurements.csv, paths.npz, traces.npz, SCHEMA.md
  fuse/organ_fusion.tif, stitch/positions.json (+ provenance)      mosaic + tile placement for the viewer
  videos_realtime/           every tile downsampled SPATIAL× in space, ~N_FRAMES frames (the viewer plays ~90)
  results/                   final DC / AC / reliability outputs (+ CSV exports)
  batch_kl_inputs/           Kloosterman sidecars (refit the reliability model without videos)
  README.md, MANIFEST.json   contents, sha256 checksums, code commit, measurement-batch provenance
Usage: python scripts/make_release.py [--out PATH] [--spatial 4] [--frames 90] [--no-sidecars]
"""
import sys, json, csv, shutil, hashlib, pickle, subprocess, datetime, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plexusflow.paths as paths

RESULTS = ("reliability_model_v1.json", "dc_reconciled_v2.pkl", "ac_field_v2.pkl", "ac_group_bc_estimate.pkl")
NODE_KEYS = ("x", "y", "boundary_type", "side")
EDGE_KEYS = ("radius", "length", "radial_uncertain", "in_carved_network")
MEAS_SCALARS = ("tile_id", "flow_from", "flow_to", "Q_kl", "Q_kl_valid_frac", "Q_kl_n_valid", "R_kl_um", "V_CL_kl_ums",
                "Q_pois_seg", "Q_ridge_mean", "Q_direct", "Q_pois_free")
QUALITY_KEYS = ("kl_ok", "f", "n_slices", "n_kept", "slice_cv", "R_ratio_med", "align", "consist", "centre")


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def scalar(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else ""
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return int(v) if not isinstance(v, (bool, np.bool_)) else int(bool(v))
    return "" if v is None else v


def export_graph(G, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("node",) + NODE_KEYS)
        for n, d in G.nodes(data=True):
            w.writerow((n,) + tuple(scalar(d.get(k)) for k in NODE_KEYS))
    paths_npz = {}
    with open(out / "edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("edge_id", "u", "v") + EDGE_KEYS + ("n_measurements",))
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            w.writerow((i, u, v) + tuple(scalar(d.get(k)) for k in EDGE_KEYS) + (len(d.get("measurements_piv") or []),))
            if d.get("path") is not None:
                paths_npz[f"e{i}"] = np.asarray(d["path"], np.float32)
    np.savez_compressed(out / "paths.npz", **paths_npz)
    traces = {}; mid = 0
    with open(out / "measurements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("meas_id", "edge_id", "u", "v") + MEAS_SCALARS + ("mean_Q", "f0_hz", "H1_amp", "H1_phase", "H2_amp", "H2_phase",
                                                                     "H3_amp", "H3_phase", "snr_dc_db", "snr_h1_db")
                   + tuple(f"q_{k}" for k in QUALITY_KEYS))
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            for m in d.get("measurements_piv") or []:
                h = m.get("harm") or {}; hs = list(h.get("harmonics") or [])
                hv = []
                for k in range(3):
                    x = hs[k] if k < len(hs) else None
                    if isinstance(x, dict):
                        hv += [scalar(x.get("amp", x.get("amplitude"))), scalar(x.get("phi", x.get("phase")))]
                    elif x is not None and np.iscomplexobj(x):
                        hv += [float(abs(x)), float(np.angle(x))]
                    elif isinstance(x, (list, tuple)) and len(x) >= 2:
                        hv += [scalar(x[0]), scalar(x[1])]
                    else:
                        hv += ["", ""]
                q = m.get("quality") or {}; sn = m.get("snr") or {}
                w.writerow((mid, i, u, v) + tuple(scalar(m.get(k)) for k in MEAS_SCALARS) + (scalar(h.get("mean_Q")), scalar(h.get("f0_hz")))
                           + tuple(hv) + (scalar(sn.get("dc_db")), scalar(sn.get("h1_db"))) + tuple(scalar(q.get(k)) for k in QUALITY_KEYS))
                if m.get("Q_t") is not None:
                    traces[f"m{mid}"] = np.asarray(m["Q_t"], np.float32)
                mid += 1
    np.savez_compressed(out / "traces.npz", **traces)
    (out / "SCHEMA.md").write_text(f"""# Graph export schema

Portable copy of `graph_vessels_piv.gpickle` ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {mid} measurements).
Coordinates are in the fused-mosaic graph frame (pixels; ~1.7 µm/px at 10×).

- **nodes.csv** — `node` id, `x`, `y`, `boundary_type` (source / sink / empty), `side`.
- **edges.csv** — `edge_id` (row order of the gpickle), `u`, `v` (positive flux = u→v), `radius` (segmented, px), `length` (px),
  `radial_uncertain` (porous zone, excluded from the carved network), `in_carved_network`, `n_measurements`.
  Centreline polylines: `paths.npz`, key `e<edge_id>`, (N, 2) [x, y].
- **measurements.csv** — one row per (edge, tile) optical-flow measurement. `flow_from`/`flow_to` give the direction of the
  positive mean flow; `mean_Q` (nL/s) is the DC of the harmonic fit (magnitude; sign from flow_from/flow_to); `H*_amp`/`H*_phase`
  the harmonic fit at the tile heart rate `f0_hz`; `Q_kl` Kloosterman flux (nL/s), `R_kl_um` its fitted radius, `V_CL_kl_ums`
  centreline speed; `q_*` the per-measurement quality record (`q_kl_ok`, kept-slice fraction `q_f`, `q_align` flow/centreline
  alignment …) from which the reliability scale c is computed (results/reliability_model_v1.json).
  Flux time series Q(t) (250 Hz, nL/s): `traces.npz`, key `m<meas_id>`.
""")
    return mid


def prime_viewer_cache(graph_path):
    """Run the viewer's per-edge harmonic_class precompute headlessly and save it into the release graph, so opening the
    release never has to compute (≈3 min) or write it (the viewer does not write into folders with a MANIFEST.json)."""
    if int(np.__version__.split(".")[0]) >= 2:
        sys.exit("make_release must run under numpy < 2: a gpickle written by numpy 2 cannot be read by numpy 1.x "
                 "(the viewer's Python). Use the system python3.")
    from plexusflow.viewer.app import ReadOnlyMosaicViewer
    v = object.__new__(ReadOnlyMosaicViewer)
    v.G = pickle.loads(Path(graph_path).read_bytes()); v.graph_path = Path(graph_path)
    v.cache_harmonic_class = True; v.force_recompute_harmonic_class = False
    v._precompute_harmonic_classes()


def downsample_videos(src_dir, dst_dir, spatial, n_frames):
    import tifffile
    dst_dir.mkdir(parents=True, exist_ok=True); n = 0
    for f in sorted(src_dir.glob("*.tif")):
        X = tifffile.imread(f)                       # existing realtime preview: (T, H, W) uint8
        step = max(1, int(round(len(X) / n_frames))); X = X[::step][:n_frames].astype(np.float32)
        s = max(1, spatial // 2)                      # the preview is already 2× downsampled from raw
        if s > 1:
            T, H, W = X.shape; X = X[:, :H // s * s, :W // s * s].reshape(T, H // s, s, W // s, s).mean((2, 4))
        tifffile.imwrite(dst_dir / f.name, X.round().clip(0, 255).astype(np.uint8), compression="zlib"); n += 1
    return n


def export_results(res_dir, out):
    D = pickle.loads((res_dir / "dc_reconciled_v2.pkl").read_bytes())
    with open(out / "dc_reconciled_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("u", "v", "Q_measured", "Q_reconciled", "Q_passive", "c", "sigma", "imaged", "winner_tile", "radius_px", "length_px"))
        for i, (u, v) in enumerate(D["edges"]):
            w.writerow((u, v) + tuple(scalar(D[k][i]) for k in ("Qm", "Qp", "Qs", "c", "sigma", "imaged", "tile", "radius", "length")))
    with open(out / "dc_boundary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("node", "kind", "side", "flux_out_nL_s"))
        for k, i in enumerate(D["inj"]):
            w.writerow((D["nodes"][i], "carve", D["side"][k], scalar(D["fb"][i])))
        for i in D["out"]:
            w.writerow((D["nodes"][i], "vein", "", scalar(D["fb"][i])))
    A = pickle.loads((res_dir / "ac_field_v2.pkl").read_bytes())
    with open(out / "ac_field_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("u", "v", "eligible", "winner_tile", "H1_amp", "H1_phase", "H2_amp", "H2_phase", "beat_scatter_H1", "rho"))
        for i, (u, v) in enumerate(A["edges"]):
            H = A["H"][:, i]
            w.writerow((u, v, int(A["elig"][i]), int(A["tile"][i]), scalar(abs(H[0])), scalar(np.angle(H[0])), scalar(abs(H[1])), scalar(np.angle(H[1])),
                        scalar(A["scatter"][i]), scalar(A["rho"][i])))
    json.dump(dict(metrics=D["metrics"], shuffled_corr=float(D["shuffled_corr"]), f_ref_hz=float(A["f_ref"]),
                   tile_time_gauge_psi=A["psi"]), open(out / "summary.json", "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--spatial", type=int, default=4)
    ap.add_argument("--frames", type=int, default=90); ap.add_argument("--no-sidecars", action="store_true"); a = ap.parse_args()
    st = paths.STAGE; out = Path(a.out) if a.out else st.parent / f"{st.name}_release_v1"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty — choose another --out")
    out.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip())
    print(f"release → {out}  (code {commit[:10]}{' + uncommitted changes' if dirty else ''})", flush=True)
    (out / "graph").mkdir(); gdst = out / "graph" / paths.GRAPH.name; shutil.copy2(paths.GRAPH, gdst)
    prime_viewer_cache(gdst)
    G = pickle.loads(gdst.read_bytes()); n_meas = export_graph(G, out / "graph" / "export"); print(f"  graph + export: {n_meas} measurements", flush=True)
    (out / "fuse").mkdir()
    for f in ("organ_fusion.tif", "organ_fusion_close.tif"):
        if (st / "fuse" / f).exists():
            shutil.copy2(st / "fuse" / f, out / "fuse" / f)
    (out / "stitch").mkdir()
    for f in ("positions.json", "provenance.json"):
        shutil.copy2(st / "stitch" / f, out / "stitch" / f)
    nv = downsample_videos(st / "videos_realtime", out / "videos_realtime", a.spatial, a.frames); print(f"  videos: {nv} tiles", flush=True)
    (out / "results").mkdir()
    for f in RESULTS:
        shutil.copy2(paths.RESULTS / f, out / "results" / f)
    export_results(paths.RESULTS, out / "results")
    if not a.no_sidecars:
        shutil.copytree(st / "batch_kl_inputs", out / "batch_kl_inputs")
    cfg = (st / "plexusflow.yaml").read_text().splitlines()
    cfg = [("# " + l if l.startswith("external_") else l) for l in cfg]
    (out / "plexusflow.yaml").write_text("# Frozen release: raw videos are not included (viewer is read-only; fresh optical flow disabled).\n"
                                         + "\n".join(cfg) + f"\nvideos_realtime_spatial_ds: {a.spatial}\n")
    batch = G.graph.get("measurement_batch")
    (out / "README.md").write_text(f"""# {st.name} — frozen release v1 ({datetime.date.today()})

Self-contained snapshot of the Somites21 yolk-sac vessel-flow analysis: the measured vessel graph, everything the read-only
viewer needs, and the final DC / AC results.  No mask or graph editing and no raw videos — the graph is frozen.

| folder | contents |
|---|---|
| `graph/` | `graph_vessels_piv.gpickle` (networkx; includes the viewer's derived per-edge harmonic cache) and `export/` — the same graph as CSV + npz (see `export/SCHEMA.md`) |
| `fuse/`, `stitch/` | organ mosaic and tile placement |
| `videos_realtime/` | every tile, {a.spatial}× downsampled in space, {a.frames} frames (display only) |
| `results/` | final DC reconciliation, AC field, group-BC (D, η) estimate, reliability model; CSV exports and `summary.json` |
| `batch_kl_inputs/` | Kloosterman sidecars (time-mean flow per vessel) — refit the reliability model without videos |

**Use** (code: https://github.com/ajmayer95/plexusflow, commit `{commit}`):
```bash
export PLEXUSFLOW_STAGE="/path/to/{out.name}"
python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"        # read-only viewer
python scripts/dc_reconcile.py                          # reproduce results/ from the graph
python scripts/ac_field.py && python scripts/ac_estimate.py
```
Reproducibility: the scripts reproduce `results/` exactly in the environment recorded in `MANIFEST.json` (Python / numpy / scipy);
other versions agree to ~0.3 % (the DC reweighting loop stops at its iteration cap, so last-digit numerical differences persist).
Methods: `docs/dc_method.md`, `docs/synthetic_validation.md` in the code repository.  Checksums: `MANIFEST.json`.
""")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    import platform, scipy, networkx
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, networkx=networkx.__version__,
               results_environment="results/ were produced with system Python 3.9 (numpy/scipy as installed on the main Mac, 2026-10-08)")
    man = dict(name=out.name, created=datetime.datetime.now().isoformat(timespec="seconds"), code_commit=commit, code_dirty=dirty, environment=env,
               source_stage=str(st), measurement_batch=batch, graph=dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), measurements=n_meas),
               files={str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in files})
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"done: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB → {out}")


if __name__ == "__main__":
    main()
