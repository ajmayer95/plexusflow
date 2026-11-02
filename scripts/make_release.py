"""Build a frozen, self-contained data release of a stage (graph, viewer inputs, final results).

Writes <out>/ (default: <stage>_release_v2 next to the stage folder), in two tiers:
  core
    plexusflow.yaml            stage config without raw-video dependence (viewer runs read-only)
    graph/graph_vessels_piv.gpickle   the frozen measured graph; Q(t) traces are NOT inside it (stored once, in
                               export/traces.npz, and re-attached by plexusflow.paths.load_graph / the viewer)
    graph/export/              portable copy: nodes.csv, edges.csv, measurements.csv, tiles.csv, paths.npz, traces.npz, SCHEMA.md
    fuse/organ_fusion_close.tif (or organ_fusion.tif), stitch/positions.json (+ provenance)   mosaic + tile placement
    results/                   final DC / AC / reliability outputs (+ CSV exports)
    README.md, MANIFEST.json   contents, sha256 checksums and tier per file, code commit, measurement-batch provenance
  extras (optional)
    videos_realtime/           every tile downsampled SPATIAL× in space, ~N_FRAMES frames (viewer playback only)
    batch_kl_inputs/           Kloosterman sidecars (re-run the Kloosterman step / refit the reliability model)
With --pack, also writes <out>.core.tar.gz and <out>.extras.tar.gz (same root folder; extract both to merge).
Usage: python scripts/make_release.py [--out PATH] [--version v2] [--spatial 4] [--frames 90] [--no-sidecars] [--pack]
Run with numpy < 2 (system python3) so every environment can read the gpickle.
"""
import sys, json, csv, shutil, hashlib, pickle, subprocess, datetime, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plexusflow.paths as paths
from plexusflow.release import SCHEMA_VERSION

RESULTS = ("reliability_model_v1.json", "dc_reconciled_v2.pkl", "ac_field_v7.pkl", "ac_estimate_v8.json")
RESULTS_OPTIONAL = ()
# inputs of the paper figures (scripts/make_figure*.py), copied as-is from the stage (core tier)
FIGURE_INPUTS = ("fuse/organ_fusion.tif", "fuse/maxmed.tif", "fuse/probs_logit.tif", "mask/mask.tif", "tiles/maxmed", "assets")
FIG3_TILE = 36                # its representative raw frame is stored (raw recordings are not released)
EXTRAS = ("videos_realtime", "batch_kl_inputs")
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
    tf, ts = G.graph.get("tile_f0_piv") or {}, G.graph.get("tile_f0_status") or {}
    with open(out / "tiles.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(("tile_id", "f0_hz", "f0_status", "beats", "beat_r2", "user_note"))
        for t in sorted(set(tf) | set(ts)):
            s_ = ts.get(t) or {}
            w.writerow((t, scalar(tf.get(t)), s_.get("status", ""), scalar(s_.get("beats")), scalar(s_.get("beat_r2")), s_.get("user_note", "")))
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
  Flux time series Q(t) (250 Hz, nL/s): `traces.npz`, key `m<meas_id>` (the only copy; the gpickle refers to it).
- **tiles.csv** — per-tile heart rate `f0_hz` used for the harmonic fits, and where reviewed, its status
  (resolved / corrected / user_confirmed / user_estimate / user_uncertain / no_signal / excluded), beats per recording,
  beat strength (median R² of a DC+H1+H2 fit over the tile's strongest vessels) and the reviewer's note.
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
    v.workbench = True                      # write the cache into this release copy of the graph (its only writer)
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
    A = pickle.loads((res_dir / "ac_field_v7.pkl").read_bytes())
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
    ap.add_argument("--version", default="v3"); ap.add_argument("--pack", action="store_true")
    ap.add_argument("--preview-videos", action="store_true", help="downsample videos (--spatial/--frames) instead of copying them as used by the figures")
    a = ap.parse_args()
    st = paths.STAGE; out = Path(a.out) if a.out else st.parent / f"{st.name}_release_{a.version}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty — choose another --out")
    out.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip())
    print(f"release → {out}  (code {commit[:10]}{' + uncommitted changes' if dirty else ''})", flush=True)
    (out / "graph").mkdir(); gdst = out / "graph" / paths.GRAPH.name; shutil.copy2(paths.GRAPH, gdst)
    prime_viewer_cache(gdst)
    G = pickle.loads(gdst.read_bytes()); n_meas = export_graph(G, out / "graph" / "export"); print(f"  graph + export: {n_meas} measurements", flush=True)
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
    if a.preview_videos:                                           # small viewer-only copy (data-v2 behaviour)
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
    print(f"  figure inputs: {', '.join(FIGURE_INPUTS)}, acquisition_times.json, raw_subset/", flush=True)
    (out / "results").mkdir()
    for f in RESULTS:
        shutil.copy2(paths.RESULTS / f, out / "results" / f)
    for f in RESULTS_OPTIONAL:
        if (paths.RESULTS / f).exists():
            shutil.copy2(paths.RESULTS / f, out / "results" / f)
    export_results(paths.RESULTS, out / "results")
    if not a.no_sidecars:
        shutil.copytree(st / "batch_kl_inputs", out / "batch_kl_inputs")
    cfg = (st / "plexusflow.yaml").read_text().splitlines()
    cfg = [("# " + l if l.startswith("external_") else l) for l in cfg]
    (out / "plexusflow.yaml").write_text("# Frozen release: raw videos are not included (viewer is read-only; fresh optical flow disabled).\n"
                                         + "\n".join(cfg) + f"\nvideos_realtime_spatial_ds: {a.spatial}\n")
    batch = G.graph.get("measurement_batch")
    (out / "README.md").write_text(f"""# {st.name} — frozen release {a.version} ({datetime.date.today()})

Self-contained snapshot of the {st.name.replace("PlexusFlow", "")} yolk-sac vessel-flow analysis: the measured vessel graph,
everything the read-only viewer needs, the final DC / AC results and the inputs of every paper figure.  No raw videos — the graph is frozen.

Two downloads, same root folder (extract both to merge): **core** (graph, exports, mosaic, results, figure inputs) is all
the analysis needs; **extras** adds the per-tile realtime videos (viewer playback, Figs 1 and 4) and the Kloosterman inputs
(Figs 3 and 6, refitting the reliability model).  All ten paper figures need both.

| folder | tier | contents |
|---|---|---|
| `graph/` | core | `graph_vessels_piv.gpickle` (networkx; includes the viewer's derived per-edge harmonic cache; Q(t) traces are re-attached from `export/traces.npz` on load) and `export/` — the same graph as CSV + npz, plus per-tile heart rates (see `export/SCHEMA.md`) |
| `fuse/`, `stitch/` | core | organ mosaic and tile placement |
| `results/` | core | final DC reconciliation, synchronised AC (waveform) field `ac_field_v7`, distensibility fit `ac_estimate_v8`, reliability model; CSV exports and `summary.json` |
| `fuse/`, `mask/`, `tiles/maxmed/`, `assets/` | core | mosaic projections, segmentation probability and mask, per-tile projections, figure drawings |
| `acquisition_times.json`, `raw_subset/` | core | recording times of every tile (from the camera headers) and the representative raw frame of tile 36 (Fig 3) |
| `videos_realtime/` | extras | every tile's realtime video as used by the figures and the viewer |
| `batch_kl_inputs/` | extras | Kloosterman sidecars (time-mean flow per vessel) — refit the reliability model without videos |

**Use** (code: https://github.com/ajmayer95/plexusflow, commit `{commit}`; Python ≥ 3.10).  Paste-safe commands:
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
`reproduce_paper.sh` re-derives `results/` and draws Figs 1–10 into `$PLEXUSFLOW_FIGURES` (about 4 minutes; `--figures` to skip
the analysis); `python examples/quickstart.py` is a one-minute tour of the DC reconciliation.  For the read-only viewer,
`pip install -e ".[gui]"` and `python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"`.  This folder is never written to:
outputs go to PLEXUSFLOW_RESULTS / PLEXUSFLOW_FIGURES.
No Python?  `graph/export/` (CSV + npz, see `SCHEMA.md`) and the CSVs in `results/` contain everything.
Reproducibility: re-running the scripts reproduces `results/` up to ~0.3 % across Python / numpy / scipy versions (the DC
reweighting loop stops at its iteration cap, so last-digit numerical differences persist); the build environment is in `MANIFEST.json`.
Methods: `docs/dc_method.md`, `docs/synthetic_validation.md` in the code repository.  Checksums: `MANIFEST.json`.
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
    return "extras" if Path(rel).parts[0] in EXTRAS else "core"


if __name__ == "__main__":
    main()
