"""plexusflow command-line interface.

Pipeline (each command reads the previous stage's output):

    plexusflow init       <STAGE_ROOT>    # scaffold folder + example config
    plexusflow info       <STAGE_ROOT>    # show resolved paths + status

    plexusflow preprocess <STAGE_ROOT>    # raw videos → per-tile projections
                                            # → PlexusFlow<stage>/tiles/{maxmed,bandpass}/
    plexusflow stitch     <STAGE_ROOT>    # napari alignment on per-tile projections
                                            # → PlexusFlow<stage>/stitch/positions.json
    plexusflow fuse       <STAGE_ROOT>    # feather per-tile projections into mosaic
                                            # → PlexusFlow<stage>/fuse/*.tif
    plexusflow mask       <STAGE_ROOT>    # napari threshold + paint
                                            # → PlexusFlow<stage>/mask/mask.tif
    plexusflow vectorize  <STAGE_ROOT>    # mask → graph (net3)
                                            # → PlexusFlow<stage>/graph/graph.gpickle
    plexusflow edit       <STAGE_ROOT>    # napari graph editor (net3)
    plexusflow condense   <STAGE_ROOT>    # graph → vessel graph
                                            # → PlexusFlow<stage>/graph/graph_vessels.gpickle
    plexusflow analyze    <STAGE_ROOT>    # Farneback OF → Q(t) per vessel
                                            # → PlexusFlow<stage>/graph/graph_vessels_piv.gpickle

The stage folder holds everything plexusflow generates. Raw videos and
optionally other tile inputs live externally, referenced by the yaml.
See `plexusflow init` for the scaffold.
"""
from __future__ import annotations

from pathlib import Path
import typer

from . import __version__
from .config import StageConfig

app = typer.Typer(add_completion=False, help="Whole-organ vascular reconstruction.")


@app.command()
def version() -> None:
    """Print the installed plexusflow version."""
    typer.echo(f"plexusflow {__version__}")


@app.command()
def info(stage_root: Path = typer.Argument(...,
                        help="Stage folder (e.g. PlexusFlow21S/)")) -> None:
    """Show resolved paths + which artifacts exist for this stage."""
    stage_root = _require_stage(stage_root)
    cfg = StageConfig.load(stage_root)
    typer.echo(cfg.summary())
    n_raw = len(cfg.discover_raw_tile_ids())
    n_std = len(cfg.discover_tile_ids('std_clahe')) if cfg.tiles_std_clahe_dir.exists() else 0
    n_max = len(cfg.discover_tile_ids('maxmed'))    if cfg.tiles_maxmed_dir.exists() else 0
    n_prb = len(cfg.discover_tile_ids('probs'))     if cfg.tiles_probs_dir.exists() else 0
    typer.echo(f"  tile counts: raw={n_raw}, std_clahe={n_std}, maxmed={n_max}, probs={n_prb}")
    typer.echo("  analysis:")
    for line in _analysis_summary(Path(cfg.root)):
        typer.echo(line)


def _require_stage(stage_root: Path) -> Path:
    """Exit with a one-line message (no traceback) unless stage_root is an existing folder."""
    p = Path(stage_root).expanduser()
    if not p.is_dir():
        typer.echo(f"error: {p} is not a folder. Pass a stage folder or an extracted data release.", err=True)
        raise typer.Exit(1)
    return p


def _analysis_summary(root: Path) -> list[str]:
    """Measured graph, derived results and (for a data release) its manifest."""
    import json
    out = []
    mf = root / "MANIFEST.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        out.append(f"  data release: {m.get('name', root.name)}, schema {m.get('schema_version', '?')}, "
                   f"code {str(m.get('code_commit', '?'))[:10]} (read-only)")
    g = root / "graph" / "graph_vessels_piv.gpickle"
    out.append(f"    {'✓' if g.exists() else ' '} measured graph   {g}")
    res = root / "results"
    for f in ("dc_reconciled_v3.pkl", "ac_field_v9.pkl", "ac_estimate_v10.json"):
        out.append(f"    {'✓' if (res / f).exists() else ' '} results/{f}")
    vr = root / "videos_realtime"
    n_v = len(list(vr.glob("*.tif"))) if vr.is_dir() else 0
    out.append(f"  realtime videos: {n_v}  (viewer playback, Figs 1 and 4)")
    return out


@app.command()
def init(stage_root: Path = typer.Argument(...),
         force: bool = typer.Option(False, help='Overwrite existing plexusflow.yaml')
         ) -> None:
    """Scaffold folder layout + write a starter plexusflow.yaml."""
    stage_root = Path(stage_root).expanduser().resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    for sub in ('tiles/std_clahe', 'tiles/maxmed', 'tiles/probs',
                 'stitch', 'fuse', 'mask', 'graph'):
        (stage_root / sub).mkdir(parents=True, exist_ok=True)
    cfg_path = stage_root / 'plexusflow.yaml'
    if cfg_path.exists() and not force:
        typer.echo('plexusflow.yaml exists — pass --force to overwrite')
    else:
        cfg_path.write_text(_STARTER_YAML.strip() + '\n')
        typer.echo(f'wrote {cfg_path}')
    typer.echo(f'stage scaffold ready at {stage_root}')
    typer.echo('  next: drop tile files into tiles/{std_clahe,maxmed,probs}/,')
    typer.echo('        then `plexusflow stitch <STAGE_ROOT>`')


_STARTER_YAML = """
# plexusflow stage config. All fields are optional; defaults match the
# doc in plexusflow.config.StageConfig.
#
# Point at externally-located tile files (e.g. legacy PerTileFlow layout)
# by uncommenting these:
# external_raw_video_dir: ~/Desktop/ImageData/21Somites/emb1
# external_std_clahe_dir: ~/path/to/Mosaic/StdFilter
# external_maxmed_dir:    ~/path/to/Tiles/MaxMedImages
# external_probs_dir:     ~/path/to/Tiles/Probs
#
# raw_video_dir_pattern: '10x 250fps loc{vid}_*'
# raw_video_frame_glob:  '*.tif'
# raw_fps: 250.0
#
# tile_std_clahe_pattern: 'vid{vid}_std_clahe.tif'
# tile_maxmed_pattern:    'vid{vid}_maxmed.tif'
# tile_probs_pattern:     'vid{vid}_maxmed_probs.h5'
#
# Downsample factors (see plexusflow.config for the model):
stitch_downsample: 2      # napari display only; positions saved at FULL RES
fuse_downsample:   2      # output resolution of feathered images
fuse_feather_alpha: 0.7
probs_channel: 0          # which ilastik output channel is vessel
#
# Mask defaults (used by the threshold editor when starting fresh):
mask_default_threshold: 0.5
mask_default_low: 0.25
mask_default_min_size: 100
mask_default_close_radius: 1
#
# Vectorization (net3) defaults:
vectorize_min_feature_size: 100
vectorize_prune_order: 3
vectorize_fill_holes: 20     # fill mask holes ≤ N px before net3; 0 to disable
"""


@app.command()
def preprocess(stage_root: Path = typer.Argument(...),
                which: str = typer.Option("maxmed,bandpass",
                    help='Comma-separated projections to compute. '
                            'Options: maxmed, bandpass, std, hp_std, std_clahe'),
                max_frames: int = typer.Option(600,
                    help='Frames per tile to use (600 @ 250fps = 2.4s). '
                            'Set to 0 for all frames.'),
                overwrite: bool = typer.Option(False,
                    help='Recompute even if output tile file already exists'),
                ) -> None:
    """Compute per-tile projections from raw videos into the stage folder.

    After this runs, `tiles/<projection>/vid{N}_<projection>.tif` files
    exist for each requested projection, and downstream `stitch` / `fuse`
    commands can consume them without needing the raw videos.

    Common workflow:

        plexusflow init      PlexusFlow21S
        # edit plexusflow.yaml to set external_raw_video_dir
        plexusflow preprocess PlexusFlow21S --which maxmed,bandpass
        plexusflow stitch    PlexusFlow21S
        plexusflow fuse      PlexusFlow21S
    """
    from .preprocess import preprocess_tiles, PROJECTIONS
    cfg = StageConfig.load(stage_root)
    names = [n.strip() for n in which.split(',') if n.strip()]
    unknown = [n for n in names if n not in PROJECTIONS]
    if unknown:
        typer.echo(f'error: unknown projections {unknown}. '
                    f'Valid: {sorted(PROJECTIONS)}', err=True)
        raise typer.Exit(2)
    if not cfg.raw_video_dir.exists():
        typer.echo(f'error: raw video dir not found: {cfg.raw_video_dir}\n'
                    f'  set `external_raw_video_dir` in plexusflow.yaml',
                    err=True)
        raise typer.Exit(2)
    mf = max_frames if max_frames > 0 else None
    typer.echo(f'preprocessing {names} from {cfg.raw_video_dir} '
                f'(max_frames={mf})...')
    dirs = preprocess_tiles(cfg, which=names, max_frames=mf, overwrite=overwrite)
    for name, d in dirs.items():
        n = len(list(d.glob('*.tif')))
        typer.echo(f'  {name}: {n} tiles → {d}')


@app.command()
def stitch(stage_root: Path = typer.Argument(...),
           tile_source: str = typer.Option("maxmed",
                                            help='"maxmed" (default, matches ilastik '
                                                    'training input) or "std_clahe" '
                                                    '(legacy high-contrast variant)')
           ) -> None:
    """Launch the napari tile stitcher (step 2 — run `preprocess` first).

    Expects per-tile projections to already exist (via `preprocess`, or via
    the `external_*_dir` overrides in `plexusflow.yaml`).
    """
    from .stitch import launch_stitcher
    cfg = StageConfig.load(stage_root)
    rc = launch_stitcher(cfg, tile_source=tile_source)
    raise typer.Exit(rc)


@app.command()
def fuse(stage_root: Path = typer.Argument(...),
         which: str = typer.Option("all",
             help='"intensity" | "maxmed" | "bandpass" | "probs" | "all"'),
         fusion: str = typer.Option("logit",
             help='For probs: "logit" (default, correct) or "linear"')
         ) -> None:
    """Feather-blend per-tile images into single stitched images."""
    from .fuse import feather_intensity, feather_probabilities
    cfg = StageConfig.load(stage_root)
    cfg.ensure_dirs()
    if not cfg.positions_path.exists():
        typer.echo(f'error: no positions.json at {cfg.positions_path}. '
                    'Run `plexusflow stitch` first.', err=True)
        raise typer.Exit(2)
    todo = ({which} if which != "all"
            else {"intensity", "maxmed", "bandpass", "probs"})
    if "intensity" in todo and cfg.tiles_std_clahe_dir.exists():
        typer.echo(f'[fuse] intensity → {cfg.fuse_intensity_path}')
        feather_intensity(cfg.positions_path, cfg.tiles_std_clahe_dir,
                            cfg.tile_std_clahe_pattern, cfg.fuse_intensity_path,
                            downsample=cfg.fuse_downsample,
                            alpha=cfg.fuse_feather_alpha)
    if "maxmed" in todo and cfg.tiles_maxmed_dir.exists():
        typer.echo(f'[fuse] maxmed → {cfg.fuse_maxmed_path}')
        feather_intensity(cfg.positions_path, cfg.tiles_maxmed_dir,
                            cfg.tile_maxmed_pattern, cfg.fuse_maxmed_path,
                            downsample=cfg.fuse_downsample,
                            alpha=cfg.fuse_feather_alpha, dtype='uint8')
    if "bandpass" in todo and cfg.tiles_bandpass_dir.exists():
        typer.echo(f'[fuse] bandpass → {cfg.fuse_bandpass_path}')
        feather_intensity(cfg.positions_path, cfg.tiles_bandpass_dir,
                            cfg.tile_bandpass_pattern, cfg.fuse_bandpass_path,
                            downsample=cfg.fuse_downsample,
                            alpha=cfg.fuse_feather_alpha, dtype='uint8')
    if "probs" in todo and cfg.tiles_probs_dir.exists():
        out = (cfg.fuse_probs_logit_path if fusion == "logit"
               else cfg.fuse_probs_linear_path)
        typer.echo(f'[fuse] probs ({fusion}) → {out}')
        feather_probabilities(cfg.positions_path, cfg.tiles_probs_dir,
                                cfg.tile_probs_pattern, cfg.probs_channel, out,
                                fusion=fusion, downsample=cfg.fuse_downsample,
                                alpha=cfg.fuse_feather_alpha)
    cfg.write_provenance("fuse", extra={"fusion": fusion, "which": which})


@app.command()
def mask(stage_root: Path = typer.Argument(...),
         prob_source: str = typer.Option("probs_logit",
             help='"probs_logit" | "probs_linear" | "intensity" | "maxmed"'),
         video: str = typer.Option(None, help='Optional TIFF stack to overlay'),
         invert: bool = typer.Option(False, help='Invert probability on load')
         ) -> None:
    """Launch the threshold+paint mask editor."""
    from .mask import launch_mask_editor
    cfg = StageConfig.load(stage_root)
    rc = launch_mask_editor(cfg, prob_source=prob_source,
                              video=video, invert_prob=invert)
    raise typer.Exit(rc)


@app.command()
def mve(stage_root: Path = typer.Argument(...)) -> None:
    """Unified Mask + Vectorize + Edit napari session.

    One window that opens the probability map + mask + graph + optional
    per-tile video overlay, with docks for threshold sliders, a
    Vectorize button (mask → graph on-demand), and a Save button that
    writes mask.tif and graph.gpickle to the stage paths.
    """
    from .mve import launch_unified_editor
    cfg = StageConfig.load(stage_root)
    rc = launch_unified_editor(cfg)
    raise typer.Exit(rc)


@app.command()
def vectorize(stage_root: Path = typer.Argument(...),
              for_editor: bool = typer.Option(True,
                    help='Keep degree-2 nodes for centerline (needed by net3 edit)')
              ) -> None:
    """Vectorize mask.tif into graph.gpickle via net3."""
    from .vectorize import run_vectorize
    cfg = StageConfig.load(stage_root)
    rc = run_vectorize(cfg, for_editor=for_editor)
    raise typer.Exit(rc)


@app.command()
def condense(stage_root: Path = typer.Argument(...),
              src: Path = typer.Option(None,
                    help='Source .gpickle (default: graph/graph.gpickle)'),
              out: Path = typer.Option(None,
                    help='Output .gpickle (default: graph/graph_vessels.gpickle)')
              ) -> None:
    """Merge each chain of degree-2 nodes into one vessel edge.

    The merged edge keeps the concatenated centreline `path`.  Nodes flagged `true_tip=True` or with
    `boundary_type` set stay as chain endpoints.  Writes graph/graph_vessels.gpickle by default."""
    from .analyze import condense_stage
    cfg = StageConfig.load(stage_root)
    condense_stage(cfg, src_path=src, out_path=out)


@app.command()
def analyze(stage_root: Path = typer.Argument(...),
             tiles: str = typer.Option(None,
                    help='Tile range/list (e.g. "1-53" or "1,3,5"). '
                          'Default = all discoverable tiles.'),
             max_frames: int = typer.Option(None,
                    help='Frames per tile (default: all). 500 gives a 2s window.'),
             workers: int = typer.Option(None,
                    help='joblib workers (default = cpu_count - 1)'),
             default_f0_hz: float = typer.Option(2.6,
                    help='Default cardiac frequency when graph carries no '
                          'per-tile tile_f0_piv.'),
             skip_analyzed: bool = typer.Option(False,
                    help='Skip vessels already measured on the current tile '
                          '(resume-mode; default: overwrite)'),
             dry_run: bool = typer.Option(False,
                    help='Do not write the output graph'),
             kl_sidecar: Path = typer.Option(None,
                    help='Directory for per-tile Kloosterman-input sidecars '
                          '(time-mean flow field, CL, R, mask ROI; float16, gzip) so '
                          'Kloosterman/QC can be re-run without re-batching. '
                          'Default: <stage>/batch_kl_inputs. Pass "none" to skip.'),
             out: Path = typer.Option(None,
                    help='Write the measured graph here instead of '
                          '<stage>/graph/graph_vessels_piv.gpickle (e.g. a local disk); '
                          'resumes from it if it exists, else starts from the stage graph.')
             ) -> None:
    """Farneback OF per tile → Q(t) per vessel → harmonics → store on graph.

    Reads graph/graph_vessels.gpickle (run `plexusflow condense` first).
    Writes to graph/graph_vessels_piv.gpickle with incremental saves
    after every tile."""
    from .analyze import analyze_stage
    cfg = StageConfig.load(stage_root)
    tile_ids = None
    if tiles:
        tile_ids = set()
        for part in tiles.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                tile_ids.update(range(int(a), int(b) + 1))
            else:
                tile_ids.add(int(part))
        tile_ids = sorted(tile_ids)
    side = None if (kl_sidecar is not None and str(kl_sidecar).lower() == 'none') \
        else (kl_sidecar or (Path(stage_root) / 'batch_kl_inputs'))
    analyze_stage(cfg, tile_ids=tile_ids, max_frames=max_frames,
                    n_workers=workers, default_f0_hz=default_f0_hz,
                    skip_analyzed=skip_analyzed, dry_run=dry_run,
                    kl_sidecar_dir=side, out_path=out)


@app.command('refine-eval')
def refine_eval_cmd(stage_root: Path = typer.Argument(...),
                    pending: Path = typer.Argument(..., help="graph/patches/pending/<stamp> candidate folder"),
                    no_measure: bool = typer.Option(False, "--no-measure", help="skip measuring the new vessels"),
                    workers: int = typer.Option(None, help="joblib workers for the measurement")) -> None:
    """Evaluate a refinement candidate (workbench Refine panel): measure its new vessels into the candidate only,
    refit current vs candidate, write result.json / result.pkl next to it.  Writes nothing else."""
    import os
    os.environ.setdefault("PLEXUSFLOW_STAGE", str(stage_root))
    from .edit.refine import evaluate
    evaluate(stage_root, pending, measure=not no_measure, workers=workers, log=lambda s: typer.echo(s))


@app.command('carry-measurements')
def carry_measurements_cmd(stage_root: Path = typer.Argument(...),
                           tol_px: float = typer.Option(1.0, help='Centreline match tolerance (px).')) -> None:
    """After editing + condense: copy flow measurements from the previous analysed graph
    (graph_vessels_piv) onto the edited graph (graph_vessels) for every vessel whose centreline is
    unchanged, then save it as the new graph_vessels_piv.  Follow with `analyze --skip-analyzed` to
    measure only edited/new vessels.  The previous analysed graph is backed up first."""
    import datetime, pickle, shutil
    from .analyze.carryover import carry_measurements, remap_sidecars
    cfg = StageConfig.load(stage_root)
    if not cfg.graph_piv_path.exists():
        typer.echo(f'no analysed graph at {cfg.graph_piv_path} — nothing to carry'); raise typer.Exit(2)
    G_old = pickle.loads(cfg.graph_piv_path.read_bytes())
    G_new = pickle.loads(cfg.graph_vessels_path.read_bytes())
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = cfg.graph_piv_path.with_name(cfg.graph_piv_path.name + f'.pre_carry_{stamp}')
    shutil.copy2(cfg.graph_piv_path, backup)
    stats, emap = carry_measurements(G_new, G_old, tol_px=tol_px)
    side = Path(stage_root) / 'batch_kl_inputs'
    if side.exists():
        shutil.copytree(side, side.with_name(side.name + f'.pre_carry_{stamp}'))
        n_sc = remap_sidecars(side, emap)
        typer.echo(f'  sidecars remapped: {n_sc} tiles')
    cfg.graph_piv_path.write_bytes(pickle.dumps(G_new, protocol=pickle.HIGHEST_PROTOCOL))
    typer.echo(f'carried measurements → {cfg.graph_piv_path.name}  (backup: {backup.name})')
    for k, v in stats.items():
        typer.echo(f'  {k}: {v}')
    typer.echo('next: plexusflow analyze <stage> --skip-analyzed   (measures only the vessels listed as to_measure)')


@app.command('legacy-normalize')
def legacy_normalize(stage_root: Path = typer.Argument(...),
              src: Path = typer.Option(None,
                    help='Source .gpickle (default: graph/graph_vessels_piv.gpickle)'),
              out: Path = typer.Option(None,
                    help='Output .gpickle (default: graph/graph_vessels_piv_normalized.gpickle)'),
              fringe_tile_cutoff: int = typer.Option(5,
                    help='Exclude tiles with id <= this value from the cross-tile '
                          'edge aggregation. Default 5 (Stage-21 "fringe" tiles).'),
              radius_consistency: float = typer.Option(0.30,
                    help='Drop per-tile aggregation contributions whose R_kl_um '
                          'deviates from the per-edge median by more than this '
                          'relative fraction. Default 0.30. Use 10.0 to disable.'),
              compute_h2: bool = typer.Option(True,
                    help='Include H2 harmonic in the sync'),
              exclude_tiles: str = typer.Option(None,
                    help='Comma-separated tile ids to remove from measurements_piv '
                          'BEFORE normalization (e.g. "1,2,3,4,5,6"). Applied in '
                          'addition to fringe_tile_cutoff, which only filters '
                          'the aggregation step.'),
              anchor_tile: int = typer.Option(None,
                    help='Tile whose H1 phase gauge gets pinned to 0 during '
                          'sync (defaults to tiles[0], historically tile 14 or '
                          'the first in input order). Pass 21 (or any valid '
                          'tile id) to re-express all per-edge phases in that '
                          'tile\'s H1 frame.'),
              tag: str = typer.Option('',
                    help='Optional output tag.  Inserted into the output '
                          '.gpickle filename as "<stem>_<tag>.gpickle".'),
              ) -> None:
    """LEGACY: run the 2026-09 tile_normalize on the PIV-analyzed graph (superseded by scripts/dc_reconcile.py + scripts/pulse_field.py).

    Applies per-tile CO_rel + phi_H1 gauges + sign resolution, then
    aggregates per-tile phasors into edge-level `_h_*_norm` fields.

    Calls `plexusflow.legacy.tile_normalize.normalize_graph_in_place`.  Kept only to reproduce
    2026-09 figures; the viewer no longer offers it."""
    import pickle
    from .legacy.tile_normalize import (
        NormalizeParams, normalize_graph_in_place)
    cfg = StageConfig.load(stage_root)
    src = src or cfg.graph_piv_path
    out = out or cfg.graph_normalized_path
    if not src.exists():
        raise FileNotFoundError(f"Source graph not found: {src}\n"
                                f"Run `plexusflow analyze` first.")
    print(f"[normalize] loading  {src.name}   ({src.stat().st_size/1024/1024:.0f} MB)")
    G = pickle.loads(src.read_bytes())
    print(f"[normalize] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Optional per-tile strip
    if exclude_tiles:
        exclude = {int(t.strip()) for t in exclude_tiles.split(',') if t.strip()}
        n_removed = 0
        for u, v, d in G.edges(data=True):
            ms = d.get('measurements_piv') or []
            kept = [m for m in ms if int(m.get('tile_id', -1)) not in exclude]
            if len(kept) != len(ms):
                d['measurements_piv'] = kept
                n_removed += len(ms) - len(kept)
            # wipe stale caches
            for k in list(d.keys()):
                if (k == 'measurements_piv_normalized'
                        or k.endswith('_norm') or k.startswith('_h_norm_')):
                    d.pop(k, None)
        G.graph.pop('tile_gauges', None)
        print(f"[normalize] stripped {n_removed} measurements from tiles "
              f"{sorted(exclude)}")

    params = NormalizeParams(
        fringe_tile_cutoff=fringe_tile_cutoff,
        radius_consistency_rel_thresh=radius_consistency,
        compute_h2=compute_h2,
        anchor_tile=anchor_tile,
    )
    # Tag suffix on output path
    if tag:
        out = out.with_name(f'{out.stem}_{tag}{out.suffix}')
    if anchor_tile is not None:
        print(f"[normalize] anchor_tile = {anchor_tile}  (phase gauge pinned there)")
    print(f"[normalize] params: fringe_tile_cutoff={params.fringe_tile_cutoff}  "
          f"radius_consistency_rel_thresh={params.radius_consistency_rel_thresh}  "
          f"compute_h2={params.compute_h2}")
    stats = normalize_graph_in_place(G, params=params,
                                       progress=lambda s, p: print(f"  [{p*100:>3.0f}%] {s}"))
    print(f"[normalize] done.  tile_gauges = {len(G.graph.get('tile_gauges') or {})} tiles")
    print(f"[normalize] writing  {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(G, f)
    print(f"[normalize] wrote {out.stat().st_size/1024/1024:.0f} MB")


@app.command()
def edit(stage_root: Path = typer.Argument(...),
         from_mask: bool = typer.Option(False,
             help='Vectorize + edit in one step (net3 edit-mask)')
         ) -> None:
    """Launch the net3 graph editor."""
    from .edit import launch_editor
    cfg = StageConfig.load(stage_root)
    rc = launch_editor(cfg, from_mask=from_mask)
    raise typer.Exit(rc)


def _launch_viewer(stage_root: Path, field: str, mode: str) -> None:
    """Resolve graph + fuse image + tile positions + per-tile videos from the stage folder's StageConfig and open
    the mosaic viewer in the given capability mode ('view' or 'workbench')."""
    from .viewer import ReadOnlyMosaicViewer
    stage_root = _require_stage(stage_root)
    cfg = StageConfig.load(stage_root)
    graph_path = cfg.graph_piv_path if cfg.graph_piv_path.exists() else cfg.graph_vessels_path
    if not graph_path.exists():
        typer.echo(f"error: no vessel graph in {stage_root} (looked for {cfg.graph_piv_path.name} and "
                   f"{cfg.graph_vessels_path.name} in graph/). Run `plexusflow info {stage_root}` to see what is there.", err=True)
        raise typer.Exit(1)
    tiff_path = None
    # Prefer the exterior-zeroed organ fusion (plexus + embryo space, background
    # masked out) when present — cleaner background than the raw feathered fusion.
    _organ = cfg.fuse_maxmed_path.parent / "organ_fusion.tif"
    if _organ.exists():
        tiff_path = _organ
    else:
        for attr in ("fuse_bandpass_path", "fuse_intensity_path", "fuse_maxmed_path"):
            p = getattr(cfg, attr, None)
            if p is not None and p.exists():
                tiff_path = p
                break
    tile_positions = cfg.positions_path if cfg.positions_path.exists() else None
    raw_video_dir = cfg.raw_video_dir if cfg.raw_video_dir.exists() else None
    # Display video: prefer the downsampled `videos_realtime/` bundle when
    # it exists (fast playback); OF-compute video: always the raw folder.
    if cfg.videos_realtime_dir.exists() and any(cfg.videos_realtime_dir.iterdir()):
        display_video_dir = cfg.videos_realtime_dir
        of_video_dir      = raw_video_dir
        typer.echo(f'  display videos: {cfg.videos_realtime_dir.name}/ '
                    f'(realtime downsample)')
    else:
        display_video_dir = raw_video_dir
        of_video_dir      = raw_video_dir
    video_pattern = cfg.raw_video_dir_pattern
    ReadOnlyMosaicViewer(
        graph_path=graph_path,
        tiff_path=tiff_path,
        tile_positions=tile_positions,
        video_dir=display_video_dir,
        video_pattern=video_pattern,
        of_video_dir=of_video_dir,
        of_video_pattern=video_pattern,
        initial_field=field,
        mode=mode,
    ).run()


@app.command()
def view(stage_root: Path = typer.Argument(...),
         field: str = typer.Option("mean_Q",
             help='Field to colour edges by on startup')
         ) -> None:
    """Open the read-only mosaic viewer: Measured, Reconciled and Model sources, time series, video overlay.

    Model = the Fig 7 RC network driven by the reconciled boundary fluxes, at a distensibility set with a slider
    (recomputed in memory).  Never writes anything and runs no optical flow — safe on frozen releases.
    For tagging, boundary edits, refinement, fresh OF and simulations use `plexusflow workbench`.
    """
    _launch_viewer(stage_root, field, 'view')


@app.command()
def workbench(stage_root: Path = typer.Argument(...),
              field: str = typer.Option("mean_Q",
                  help='Field to colour edges by on startup')
              ) -> None:
    """Open the viewer with editing tools: fresh OF on a vessel, radial-uncertainty
    tagging, boundary marking (saved to the stage graph).  Refuses frozen releases
    (opens them read-only).  Simulation tabs appear when plexusmech is installed."""
    _launch_viewer(stage_root, field, 'workbench')


@app.command('downsample-videos')
def downsample_videos(
    stage_root: Path = typer.Argument(...),
    spatial: int = typer.Option(None,
        help='Spatial downsample factor (default: cfg.videos_realtime_spatial_ds).'),
    temporal: int = typer.Option(None,
        help='Temporal downsample factor — keep every Nth frame '
              '(default: cfg.videos_realtime_temporal_ds).'),
    tiles: str = typer.Option(None,
        help='Tile range/list (e.g. "1-53" or "1,3,5"). Default = all discovered.'),
    max_frames: int = typer.Option(0,
        help='Cap on frames after temporal decimation. 0 = no cap.'),
    overwrite: bool = typer.Option(False,
        help='Regenerate even if the output tile file already exists.'),
) -> None:
    """Build downsampled per-tile videos for viewer realtime playback.

    Reads the raw folder-of-frames for each tile, decimates spatially
    (default 2×) and optionally temporally, writes one multi-page TIFF
    per tile to ``<stage>/videos_realtime/<same pattern>.tif``.  The
    viewer picks these up automatically for display; OF compute keeps
    using the raw frames.
    """
    import numpy as np, tifffile
    cfg = StageConfig.load(stage_root)
    sd = int(spatial or cfg.videos_realtime_spatial_ds)
    td = int(temporal or cfg.videos_realtime_temporal_ds)
    if sd < 1 or td < 1:
        typer.echo('spatial and temporal factors must be ≥ 1', err=True)
        raise typer.Exit(2)
    if not cfg.raw_video_dir.exists():
        typer.echo(f'raw video dir not found: {cfg.raw_video_dir}\n'
                    f'  set external_raw_video_dir in plexusflow.yaml', err=True)
        raise typer.Exit(2)
    all_ids = cfg.discover_raw_tile_ids()
    if tiles:
        want = set()
        for part in tiles.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                want.update(range(int(a), int(b) + 1))
            else:
                want.add(int(part))
        all_ids = [t for t in all_ids if t in want]
    cfg.videos_realtime_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f'downsampling {len(all_ids)} tiles → {cfg.videos_realtime_dir} '
                f'(spatial ×{sd}, temporal ×{td}'
                f'{f", max {max_frames} frames" if max_frames else ""})')
    for tid in all_ids:
        # Take the raw pattern (e.g. "10x 250fps loc{vid}_*") and swap the
        # trailing `_*` glob for `_realtime.tif` so the file has a clean
        # name while still matching the viewer's `loc{vid}_*` glob resolver.
        raw_pat = cfg.raw_video_dir_pattern.format(vid=tid)
        base = raw_pat.rstrip('*').rstrip('_')
        out_name = f'{base}_realtime.tif'
        out_path = cfg.videos_realtime_dir / out_name
        if out_path.exists() and not overwrite:
            typer.echo(f'  tile {tid}: exists, skip ({out_name})')
            continue
        try:
            folder = cfg.raw_tile_folder(tid)
        except FileNotFoundError:
            typer.echo(f'  tile {tid}: no raw folder, skip')
            continue
        frames = sorted(folder.glob(cfg.raw_video_frame_glob))
        if not frames:
            typer.echo(f'  tile {tid}: no frames, skip')
            continue
        if td > 1:
            frames = frames[::td]
        if max_frames and len(frames) > max_frames:
            frames = frames[:max_frames]
        stack = []
        for fp in frames:
            im = tifffile.imread(str(fp))
            if sd > 1:
                im = im[::sd, ::sd]
            stack.append(im)
        arr = np.stack(stack, axis=0)
        tifffile.imwrite(str(out_path), arr, compression='zlib')
        typer.echo(f'  tile {tid}: wrote {arr.shape} → {out_name} '
                    f'({out_path.stat().st_size / 1e6:.1f} MB)')
    typer.echo('done')


if __name__ == "__main__":
    app()
