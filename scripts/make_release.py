"""Build a self-contained data release of a stage (graph, viewer inputs, final results, figure inputs).

Writes <out>/ (default: <stage>_release_<version> next to the stage folder), in two tiers:
  core
    plexusflow.yaml            stage config without the raw-video paths (the viewer runs read-only)
    graph/graph_vessels_piv.gpickle   the measured graph, reduced to plexusflow.release.RELEASE_FIELDS, current field
                               names; Q(t) traces are stored once, in export/traces.npz, and re-attached on load
    graph/export/              the same graph as CSV + npz (nodes, edges, measurements, tiles, paths, traces, SCHEMA.md)
    graph/provenance.json      how the recordings were measured
    fuse/, stitch/, mask/, tiles/maxmed/, assets/   mosaic, tile placement and the other figure inputs
    batch_profile_inputs/tile_<FIG3_TILE>.pkl.gz   the profile-fit inputs Fig 3 uses
    results/                   final results (+ CSV exports)
    README.md, MANIFEST.json   contents, sha256 checksums and tier per file, code commit and environment
  extras
    videos_realtime/           per-tile videos: a prepared folder (--videos) or a downsample of the stage's (--preview-videos)
    batch_profile_inputs/      all profile-fit sidecars (unless --no-sidecars)
With --pack, also writes <out>.core.tar.gz and <out>.extras.tar.gz (same root folder; extract both to merge).
Usage: python scripts/make_release.py [--out PATH] [--version v4] [--videos DIR | --preview-videos] [--no-sidecars] [--pack]
Run with numpy < 2 (system python3) so every environment can read the gpickle.
"""
import sys, json, csv, shutil, hashlib, pickle, subprocess, datetime, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from plexusflow.compat import sidecar_dir, loads_graph
import plexusflow.paths as paths
from plexusflow.release import SCHEMA_VERSION, to_release_schema

RESULTS = ("dc_reconciled_v3.pkl", "ac_field_v9.pkl", "ac_estimate_v11.json")
RESULTS_OPTIONAL = ("reliability_model_v1.json", "dc_qc.json", "dc_qc.csv")
# inputs of the paper figures (scripts/make_figure*.py), copied as-is from the stage (core tier)
FIGURE_INPUTS = ("fuse/organ_fusion.tif", "fuse/maxmed.tif", "fuse/probs_logit.tif", "mask/mask.tif", "tiles/maxmed", "assets")
FIG3_TILE = 36                # its representative raw frame is stored (raw recordings are not released)
EXTRAS = ("videos_realtime", "batch_profile_inputs")
NODE_KEYS = ("x", "y", "boundary_type")
EDGE_KEYS = ("radius", "length", "radial_uncertain", "in_carved_network")
MEAS_SCALARS = ("tile_id", "flow_from", "flow_to", "Q_profile", "R_profile_um")
QUALITY_KEYS = ("profile_ok", "f", "align", "p")


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
    from plexusflow.reconcile import carved_domain
    carved = {frozenset(e) for e in carved_domain(G)["edges"]}
    with open(out / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("node",) + NODE_KEYS)
        for n, d in G.nodes(data=True):
            w.writerow((n,) + tuple(scalar(d.get(k)) for k in NODE_KEYS))
    paths_npz = {}
    with open(out / "edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("edge_id", "u", "v") + EDGE_KEYS + ("n_measurements",))
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            vals = {**d, "in_carved_network": frozenset((u, v)) in carved}
            w.writerow((i, u, v) + tuple(scalar(vals.get(k)) for k in EDGE_KEYS) + (len(d.get("measurements_piv") or []),))
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
    tf, ts = G.graph.get("tile_f0_piv") or {}, G.graph.get("tile_f0_status") or {}
    with open(out / "tiles.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("tile_id", "f0_hz", "f0_status", "beats", "beat_r2", "user_note"))
        for t in sorted(set(tf) | set(ts)):
            s_ = ts.get(t) or {}
            w.writerow((t, scalar(tf.get(t)), s_.get("status", ""), scalar(s_.get("beats")), scalar(s_.get("beat_r2")), s_.get("user_note", "")))
    (out / "SCHEMA.md").write_text(f"""# Graph export

CSV and npz copy of `graph_vessels_piv.gpickle`: {G.number_of_nodes()} nodes, {G.number_of_edges()} vessels,
{mid} recordings.  Coordinates are in the mosaic frame, in pixels (about 1.7 µm per pixel).

- **nodes.csv**: `node`, `x`, `y`, `boundary_type` (`source` = dorsal-aorta inflow, `sink` = vitelline-vein outflow, empty otherwise).
- **edges.csv**: `edge_id` (row order of the gpickle), `u`, `v`, `radius` (px), `length` (px), `radial_uncertain`
  (region left out of the reconstructed network), `in_carved_network`, `n_measurements`.  Centrelines are in `paths.npz`,
  key `e<edge_id>`, an (N, 2) array of [x, y].
- **measurements.csv**: one row per recording (vessel and tile).  `flow_from` / `flow_to` give the direction of positive
  mean flow; `mean_Q` (nL/s) is the recording's mean flux and `H1_*`, `H2_*` its first two harmonics at the tile's heart
  rate `f0_hz`; `Q_profile` is the profile-fit flux and `R_profile_um` its radius; `q_*` is the quality record
  (`q_profile_ok`, `q_f`, `q_align`, `q_p`).  The flux time series Q(t) (250 frames/s, nL/s) are in `traces.npz`, key
  `m<meas_id>`.
- **tiles.csv**: heart rate `f0_hz` of each tile, with its status and any note from review.
""")

    return mid


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
    D = pickle.loads((res_dir / "dc_reconciled_v3.pkl").read_bytes())
    with open(out / "dc_reconciled_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("u", "v", "Q_measured", "Q_reconciled", "Q_passive", "c", "sigma", "imaged", "winner_tile", "radius_px", "length_px",
                                       "Q_reconciled_sd", "unstable"))
        un = D.get("unstable", np.zeros(len(D["edges"]), bool)); sd = D.get("Qp_sd", np.full(len(D["edges"]), np.nan))
        for i, (u, v) in enumerate(D["edges"]):
            w.writerow((u, v) + tuple(scalar(D[k][i]) for k in ("Qm", "Qp", "Qs", "c", "sigma", "imaged", "tile", "radius", "length")) + (scalar(sd[i]), int(un[i])))
    with open(out / "dc_boundary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("node", "kind", "side", "flux_out_nL_s"))
        for k, i in enumerate(D["inj"]):
            w.writerow((D["nodes"][i], "carve", D["side"][k], scalar(D["fb"][i])))
        for i in D["out"]:
            w.writerow((D["nodes"][i], "vein", "", scalar(D["fb"][i])))
    A = pickle.loads((res_dir / "ac_field_v9.pkl").read_bytes())
    with open(out / "ac_field_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("u", "v", "eligible", "winner_tile", "H1_amp", "H1_phase", "H2_amp", "H2_phase", "beat_scatter_H1", "rho",
                                       "tile_gauge_sd_deg", "phase_determined"))
        gsd = A.get("gauge_sd_deg") or {}
        for i, (u, v) in enumerate(A["edges"]):
            H = A["H"][:, i]; t = int(A["tile"][i]); sd = gsd.get(t, np.nan) if t >= 0 else np.nan
            w.writerow((u, v, int(A["elig"][i]), t, scalar(abs(H[0])), scalar(np.angle(H[0])), scalar(abs(H[1])), scalar(np.angle(H[1])),
                        scalar(A["scatter"][i]), scalar(A["rho"][i]), scalar(sd), int(bool(A["elig"][i]) and np.isfinite(sd) and sd <= 30.0)))
    json.dump(dict(metrics=D["metrics"], shuffled_corr=float(D["shuffled_corr"]), f_ref_hz=float(A["f_ref"]),
                   dc_converged=bool(D.get("converged", False)), dc_unstable_vessels=int(np.sum(D.get("unstable", []))),
                   tile_time_gauge_psi=A["psi"], tile_time_gauge_sd_deg=A.get("gauge_sd_deg", {}),
                   phase_determined_rule="tile gauge bootstrap SD <= 30 deg"), open(out / "summary.json", "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--spatial", type=int, default=4)
    ap.add_argument("--frames", type=int, default=90); ap.add_argument("--no-sidecars", action="store_true")
    ap.add_argument("--version", default="v4"); ap.add_argument("--pack", action="store_true")
    ap.add_argument("--preview-videos", action="store_true", help="downsample videos (--spatial/--frames) instead of copying them as used by the figures")
    ap.add_argument("--videos", default=None, help="copy this prepared video folder (e.g. from a raw-video downsample) as videos_realtime/; "
                                                    "a videos_realtime.json next to it (factors, frame timing) is copied too")
    a = ap.parse_args()
    st = paths.STAGE; out = Path(a.out) if a.out else st.parent / f"{st.name}_release_{a.version}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty — choose another --out")
    out.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip())
    print(f"release → {out}  (code {commit[:10]}{' + uncommitted changes' if dirty else ''})", flush=True)
    (out / "graph").mkdir(); gdst = out / "graph" / paths.GRAPH.name
    G = to_release_schema(loads_graph(paths.GRAPH.read_bytes()))      # current field names, release fields only
    if (paths.GRAPH.parent / "provenance.json").exists():              # how the measurements were made
        shutil.copy2(paths.GRAPH.parent / "provenance.json", out / "graph" / "provenance.json")
    n_meas = export_graph(G, out / "graph" / "export"); print(f"  graph + export: {n_meas} measurements", flush=True)
    mid = 0                                   # Q(t) stored once: in export/traces.npz, keyed by the export's meas_id
    for *_, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            m["meas_id"] = mid; m.pop("Q_t", None); mid += 1
    G.graph["external_traces"] = "export/traces.npz"
    gdst.write_bytes(pickle.dumps(G, protocol=4))
    G2 = paths.load_graph(gdst); chk = [m for *_, d in G2.edges(data=True) for m in d.get("measurements_piv") or []]
    n_att = sum(1 for m in chk if m.get("Q_t") is not None); print(f"  traces: {n_att}/{len(chk)} re-attach from export/traces.npz", flush=True)
    n_had = sum(1 for *_, d in pickle.loads((paths.GRAPH).read_bytes()).edges(data=True) for m in d.get("measurements_piv") or [] if m.get("Q_t") is not None)
    if n_att != n_had:
        sys.exit(f"trace re-attachment mismatch: {n_att} attached vs {n_had} in the stage graph")
    (out / "fuse").mkdir()
    if (st / "fuse" / "organ_fusion_close.tif").exists():          # viewer / Figs 7, 9, 10 background (organ_fusion.tif: figure inputs)
        shutil.copy2(st / "fuse" / "organ_fusion_close.tif", out / "fuse" / "organ_fusion_close.tif")
    (out / "stitch").mkdir()
    for f in ("positions.json", "provenance.json"):
        shutil.copy2(st / "stitch" / f, out / "stitch" / f)
    vids_ds = None
    if a.videos:                                                   # prepared bundle
        shutil.copytree(a.videos, out / "videos_realtime")
        vj = Path(a.videos).parent / "videos_realtime.json"
        if vj.exists():
            shutil.copy2(vj, out / "videos_realtime" / "videos_realtime.json")
            vids_ds = json.loads(vj.read_text()).get("spatial_downsample")
        print(f"  videos: {len(list((out / 'videos_realtime').glob('*.tif')))} tiles from {a.videos}", flush=True)
    elif a.preview_videos:                                         # small viewer-only copy (data-v2 behaviour)
        nv = downsample_videos(st / "videos_realtime", out / "videos_realtime", a.spatial, a.frames); print(f"  videos (preview): {nv} tiles", flush=True)
    else:                                                          # as used by the figures (Figs 1, 4)
        shutil.copytree(st / "videos_realtime", out / "videos_realtime"); print("  videos: copied as-is", flush=True)
    for rel in FIGURE_INPUTS:                                      # figure inputs (core)
        src, dst = st / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    from plexusflow.config import StageConfig
    from plexusflow.io.acquisition import write_acquisition_times, write_representative_frame
    cfg_ = StageConfig.load(st)
    write_acquisition_times(out, cfg_, sorted(G.graph["tile_f0_piv"]))   # Figs 1, 8 (raw headers are not released)
    write_representative_frame(out, cfg_, FIG3_TILE)                      # Fig 3a
    from plexusflow.analyze.geometry import tile_video_shape, VIDEO_SHAPES_FILE
    (out / VIDEO_SHAPES_FILE).write_text(json.dumps({str(t): list(tile_video_shape(cfg_, t)) for t in sorted(G.graph["tile_f0_piv"])
                                                      if tile_video_shape(cfg_, t)}))              # tile geometry (raw not released)
    print(f"  figure inputs: {', '.join(FIGURE_INPUTS)}, acquisition_times.json, tile_video_shapes.json, raw_subset/", flush=True)
    (out / "results").mkdir()
    for f in RESULTS:
        shutil.copy2(paths.RESULTS / f, out / "results" / f)
    for f in RESULTS_OPTIONAL:
        if (paths.RESULTS / f).exists():
            shutil.copy2(paths.RESULTS / f, out / "results" / f)
    export_results(paths.RESULTS, out / "results")
    if not a.no_sidecars:
        shutil.copytree(sidecar_dir(st), out / "batch_profile_inputs")
    else:                                                          # Fig 3 needs only its tile's sidecar
        (out / "batch_profile_inputs").mkdir()
        shutil.copy2(sidecar_dir(st) / f"tile_{FIG3_TILE}.pkl.gz", out / "batch_profile_inputs" / f"tile_{FIG3_TILE}.pkl.gz")
    cfg = (st / "plexusflow.yaml").read_text().splitlines()
    cfg = [("# " + l if l.startswith("external_") else l) for l in cfg]
    (out / "plexusflow.yaml").write_text("# Frozen release: raw videos are not included (viewer is read-only; fresh optical flow disabled).\n"
                                         + "\n".join(cfg) + (f"\nvideos_realtime_spatial_ds: {vids_ds or a.spatial}\n" if (a.preview_videos or vids_ds) else "\n"))
    batch = G.graph.get("measurement_batch")
    (out / "README.md").write_text(f"""# {st.name} — data release {a.version} ({datetime.date.today()})

Data for the {st.name.replace("PlexusFlow", "").replace("S", "-somite")} yolk-sac analysis: the measured vessel graph, the
reconstructed flows, and the inputs needed to redraw the figures and to open the viewer.  The raw microscope recordings are
not included.

## Downloads

The release comes as two archives that unpack into the same folder (`{out.name}/`):

- **core**: graph, results, mosaic, masks and the other figure inputs.  Enough to rerun the analysis, draw Figs 2, 3, 5,
  6 and 7, and use the viewer without video.
- **extras**: one short video per tile for playback in the viewer and for Figs 1 and 4
  (see `videos_realtime/videos_realtime.json` for the downsampling and frame timing).

## Contents

| Path | What it is |
|---|---|
| `graph/graph_vessels_piv.gpickle` | Vessel graph (networkx): centrelines, radii, region tags, and every flow recording per vessel and tile. Q(t) traces are stored once, in `graph/export/traces.npz`, and re-attached when the graph is loaded with plexusflow. |
| `graph/export/` | The same graph as CSV and npz, readable without Python (`SCHEMA.md` lists the columns). |
| `graph/provenance.json` | How the recordings were measured. |
| `results/` | Reconciled mean flow (`dc_reconciled_v3.pkl`), pulsatile field (`ac_field_v9.pkl`), distensibility fit (`ac_estimate_v11.json`), with CSV copies and `summary.json`. |
| `fuse/`, `stitch/`, `mask/`, `tiles/maxmed/`, `assets/` | Organ mosaic, tile positions, segmentation, per-tile projections and figure drawings. |
| `acquisition_times.json`, `tile_video_shapes.json`, `raw_subset/`, `batch_profile_inputs/` | Recording times and frame sizes of every tile, and the inputs Fig 3 needs from the raw data. |
| `videos_realtime/` (extras) | Per-tile videos. |
| `MANIFEST.json` | Checksum and size of every file, and the code version and environment used to build the release. |

## Running the code

The code is at https://github.com/ajmayer95/plexusflow; this release was built from commit `{commit[:10]}`.
It needs Python 3.10 or newer.

```bash
git clone https://github.com/ajmayer95/plexusflow.git
cd plexusflow
git checkout {commit[:10]}
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[figures]"
export PLEXUSFLOW_STAGE="/path/to/{out.name}"
export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results
export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures
bash scripts/reproduce_paper.sh
```

`reproduce_paper.sh` recomputes `results/` from the graph and draws Figs 1–7 into `$PLEXUSFLOW_FIGURES` (a few minutes on
a laptop; `--figures` skips the analysis).  Nothing is written into this folder.  For the viewer, `pip install -e ".[gui]"`
and `python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"`.

Recomputed results match the files in `results/` up to small numerical differences between library versions.  The method
is described in `docs/method.md` in the code repository.
""")

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    import platform, scipy, networkx
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, networkx=networkx.__version__)
    man = dict(name=out.name, schema_version=SCHEMA_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"), code_commit=commit, code_dirty=dirty, environment=env,
               source_stage=str(st), measurement_batch=batch, graph=dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), measurements=n_meas),
               files={str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha256(p), tier=tier(p.relative_to(out))) for p in files})
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    mb = lambda t: sum(p.stat().st_size for p in files if tier(p.relative_to(out)) == t) / 1e6
    print(f"done: {len(files)} files → {out}  (core {mb('core'):.0f} MB, extras {mb('extras'):.0f} MB)")
    if a.pack:
        import tarfile
        for t in ("core", "extras"):
            arc = out.parent / f"{out.name}.{t}.tar.gz"
            with tarfile.open(arc, "w:gz") as tf_:
                for p in sorted(out.rglob("*")):
                    if p.is_file() and (tier(p.relative_to(out)) == t or (t == "core" and p.name == "MANIFEST.json")):
                        tf_.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
            print(f"  packed {arc.name}: {arc.stat().st_size / 1e6:.0f} MB")


def tier(rel):
    rel = Path(rel)
    if rel == Path("batch_profile_inputs") / f"tile_{FIG3_TILE}.pkl.gz":    # Fig 3's sidecar ships with the core
        return "core"
    return "extras" if rel.parts[0] in EXTRAS else "core"


if __name__ == "__main__":
    main()
