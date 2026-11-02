"""Build the known-truth benchmark bundle (see plexusflow.synthetic.stokes.bundle).

Junction geometries: either copied from an earlier build (--from DIR with geom/ and vessels/, e.g. results/stokes_bench)
or rebuilt from the stage graph (--rebuild: pick junctions, mesh with gmsh, solve Stokes; and the single-vessel sweep set).
Backgrounds: from each stage's raw videos (plexusflow.yaml external_raw_video_dir) — the junction tiles at 400 frames,
plus the backgrounds of the single-vessel and frame-spacing studies (21S tile 30 at 400 and 600 frames, 15S tile 37 at 600).

Usage:
  python scripts/benchmark/make_bundle.py --out ../PlexusFlow21S_benchmark_v1 --from "$PLEXUSFLOW_STAGE/results/stokes_bench"
  python scripts/benchmark/make_bundle.py --out ... --rebuild [--n-junctions 30]          (needs gmsh, meshio)
  add --pack to also write <out>.tar.gz
"""
import sys, argparse, json, shutil, hashlib, subprocess, datetime, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.config import StageConfig
from plexusflow.synthetic.stokes.bundle import SCHEMA_VERSION, background_from_raw

STAGES = {"21S": "PlexusFlow21S", "15S": "PlexusFlow15S"}
EXTRA_BACKGROUNDS = {("21S", 30): (400, 600), ("15S", 37): (600,)}
VESSEL_KINDS, VESSEL_AS, VESSEL_EPS = ("straight", "arc", "sbend", "taper"), (5, 8, 12), (1.0, 0.8, 0.6, 0.4)


def stage_root(tag):
    return paths.STAGE.parent / STAGES[tag]


def one_background(tag, tile, nfs, out):
    cfg = StageConfig.load(stage_root(tag))
    files = [str(f) for f in cfg.raw_tile_frames(tile)]
    np.savez_compressed(out / f"{tag}_tile{tile}.npz", **background_from_raw(files, nfs))
    return tag, tile


def rebuild_junction(n, t, br, out):
    from plexusflow.synthetic.stokes.geometry import prep_branches, build_junction_mesh, solve_junction_basis
    try:
        br = prep_branches(br); msh = out / f"{n}.msh"; node = build_junction_mesh(br, msh)
        mesh, ub, U, _ = solve_junction_basis(msh, br); msh.unlink()
        np.savez_compressed(out / f"{n}.npz", p=mesh.p, t=mesh.t, uA=U["A"], uB=U["B"])
        (out / f"{n}.pkl").write_bytes(pickle.dumps(dict(node=n, tile=t, branches=br, node_xy=node)))
        return n
    except Exception as ex:
        print(f"  junction {n}: geometry/Stokes failed: {ex}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="src", help="earlier build with geom/ (junctions) and vessels/")
    g.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-junctions", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); (out / "junctions").mkdir(parents=True, exist_ok=True)
    (out / "vessels").mkdir(exist_ok=True); (out / "backgrounds").mkdir(exist_ok=True)

    if a.src:
        src = Path(a.src).expanduser()
        for sub, dst in (("geom", "junctions"), ("vessels", "vessels")):
            n = 0
            for f in sorted((src / sub).glob("*")):
                if f.suffix in (".npz", ".pkl"):
                    shutil.copy2(f, out / dst / f.name); n += 1
            print(f"copied {n} files {sub}/ → {dst}/")
    else:
        from plexusflow.synthetic.stokes.geometry import list_junctions, build_vessel
        G = paths.load_graph(stage_root("21S") / "graph" / "graph_vessels_piv.gpickle")
        picks = list_junctions(G, stage_root("21S"), a.n_junctions)
        ok = [n for n in Parallel(n_jobs=a.jobs)(delayed(rebuild_junction)(n, t, br, out / "junctions") for n, t, br in picks) if n]
        print(f"junctions: {len(ok)} built")
        names = Parallel(n_jobs=a.jobs)(delayed(build_vessel)(out / "vessels", k, s, e)
                                        for k in VESSEL_KINDS for s in VESSEL_AS for e in VESSEL_EPS)
        for f in (out / "vessels").glob("*.msh"):
            f.unlink()
        print(f"vessels: {len(names)} built")

    tiles = sorted({pickle.loads(p.read_bytes())["tile"] for p in (out / "junctions").glob("*.pkl")})
    want = {("21S", t): (400,) for t in tiles}
    for k, nfs in EXTRA_BACKGROUNDS.items():
        want[k] = tuple(sorted(set(want.get(k, ())) | set(nfs)))
    todo = [(tag, t, nfs) for (tag, t), nfs in sorted(want.items()) if not (out / "backgrounds" / f"{tag}_tile{t}.npz").exists()]
    print(f"backgrounds: {len(want)} tiles ({len(todo)} to compute)", flush=True)
    Parallel(n_jobs=min(a.jobs, 4))(delayed(one_background)(tag, t, nfs, out / "backgrounds") for tag, t, nfs in todo)

    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "README.md"))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = dict(name=out.name, schema_version=SCHEMA_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
               code_commit=commit, backgrounds={f"{tag}_tile{t}": list(nfs) for (tag, t), nfs in sorted(want.items())},
               files={str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    (out / "README.md").write_text(f"""# {out.name} — known-truth benchmark bundle

Everything the synthetic benchmark of the per-vessel flux measurement reads (method and results:
`docs/synthetic_validation.md` in the plexusflow repository).  No raw videos or mesher needed to rerun it.

- `backgrounds/` — per real tile: static background (per-pixel 10th percentile) and temporal SD of the first 400 / 600
  raw frames at full resolution, and the camera noise model (gain, read noise).
- `junctions/` — {len(list((out / 'junctions').glob('*.pkl')))} real Y-junctions: branch geometry (tile pixels) and two
  Stokes basis solutions each.
- `vessels/` — {len(list((out / 'vessels').glob('*.pkl')))} controlled single vessels with their Stokes solutions.

Rerun (needs `pip install -e ".[bench]"`):

    python scripts/benchmark/junctions.py --bundle {out.name}
    python scripts/benchmark/vessels.py --bundle {out.name}
    python scripts/benchmark/frame_spacing.py --bundle {out.name}
""")
    print(f"done: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB → {out}")
    if a.pack:
        import tarfile
        with tarfile.open(f"{out}.tar.gz", "w:gz") as tf:
            tf.add(out, arcname=out.name)
        print(f"packed {out}.tar.gz")


if __name__ == "__main__":
    main()
