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
    cfg = StageConfig.load(stage_root)
    typer.echo(cfg.summary())
    n_raw = len(cfg.discover_raw_tile_ids())
    n_std = len(cfg.discover_tile_ids('std_clahe')) if cfg.tiles_std_clahe_dir.exists() else 0
    n_max = len(cfg.discover_tile_ids('maxmed'))    if cfg.tiles_maxmed_dir.exists() else 0
    n_prb = len(cfg.discover_tile_ids('probs'))     if cfg.tiles_probs_dir.exists() else 0
    typer.echo(f"  tile counts: raw={n_raw}, std_clahe={n_std}, maxmed={n_max}, probs={n_prb}")


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
        typer.echo(f'plexusflow.yaml exists — pass --force to overwrite')
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
    """Contract every degree-2 chain in the edited graph into one vessel
    edge (with concatenated centerline `path`).  Preserves nodes flagged
    `true_tip=True` or with `boundary_type` set as chain endpoints.
    Writes to graph/graph_vessels.gpickle by default."""
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


@app.command()
def normalize(stage_root: Path = typer.Argument(...),
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
    """Run tile_normalize on the PIV-analyzed graph.

    Applies per-tile CO_rel + phi_H1 gauges + sign resolution, then
    aggregates per-tile phasors into edge-level `_h_*_norm` fields.

    This is the same code path as the viewer's 'Normalize' button — both
    use `plexusflow.mosaic.tile_normalize.normalize_graph_in_place`."""
    import pickle
    from .mosaic.tile_normalize import (
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
def simulate(stage_root: Path = typer.Argument(...),
              src: Path = typer.Option(None,
                    help='Source .gpickle (default: graph/graph_vessels_piv_normalized.gpickle)'),
              out: Path = typer.Option(None,
                    help='Output .gpickle (default: same as src, sim fields added in place)'),
              D: float = typer.Option(3.5e-4,
                    help='Areal distensibility (1/Pa). Default 3.5e-4.'),
              eta: float = typer.Option(0.0,
                    help='Constant-loss-tangent wall η. D_eff = D · (1 − iη).'),
              da_dc: float = typer.Option(1.593,
                    help='Symmetric DA anchor DC injection (nL/s).'),
              da_h1_amp: float = typer.Option(0.512,
                    help='Symmetric DA anchor |H1| injection (nL/s).'),
              da_h1_phase_deg: float = typer.Option(139.6,
                    help='DA H1 phase (degrees) — cardiac pulse phase reference.'),
              ) -> None:
    """Run the intact two-pass sim (LOCKED 2026-09-30 recipe).

    Pass 1: sink_pressure_bc=0 at V (network+DA determine drainage DC).
    Pass 2: V injection = ratio × Pass-1 V DC with cross-tile v_ratios
    from `_h_*_norm` fields.

    Same code path as the viewer's Simulate button — both call
    `plexusflow.analyze.transmission_line.solve_transmission_line`.
    Writes `_h_{amp,phase,Z}_H*_sim_intact` on every edge."""
    import math, pickle
    import numpy as np
    from .analyze import transmission_line as tl
    cfg = StageConfig.load(stage_root)
    src = src or cfg.graph_normalized_path
    out = out or src
    if not src.exists():
        raise FileNotFoundError(f"Source graph not found: {src}\n"
                                f"Run `plexusflow normalize` first.")
    print(f"[simulate] loading  {src.name}   ({src.stat().st_size/1024/1024:.0f} MB)")
    G = pickle.loads(src.read_bytes())
    print(f"[simulate] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    A = sorted([n for n,d in G.nodes(data=True)
                if d.get('boundary_type')=='source'],
               key=lambda n: G.nodes[n]['x'])
    V = sorted([n for n,d in G.nodes(data=True)
                if d.get('boundary_type')=='sink'],
               key=lambda n: G.nodes[n]['x'])
    if len(A) < 1 or len(V) < 1:
        raise RuntimeError(f"Need source + sink anchors; got A={A} V={V}")
    print(f"[simulate] anchors: DA={A}  V={V}")

    N_HARM = 3
    f0 = float(G.graph.get('f0_canonical_hz', 2.467))
    DA_PHASE = math.radians(da_h1_phase_deg)

    # η patch on _vessel_admittance
    _orig_va = tl._vessel_admittance
    def _patched(R_m, L_m, omega, mu=tl.MU_DEFAULT, rho=tl.RHO_BLOOD, D=None):
        return _orig_va(R_m, L_m, omega, mu, rho,
                          (D if D is not None else 1e-6) * (1.0 - 1j*eta))
    tl._vessel_admittance = _patched

    # Symmetric DA BCs
    bc_pass1 = {}
    for a in A:
        arr = np.zeros(N_HARM + 1, dtype=complex)
        arr[0] = float(da_dc)
        arr[1] = da_h1_amp * np.exp(1j*DA_PHASE)
        bc_pass1[a] = arr
    print(f"[simulate] Pass 1: sink_pressure_bc=0.0 at V, DAs Q-BC "
          f"(DC={da_dc:+.3f}, |H1|={da_h1_amp:.3f}∠{da_h1_phase_deg:+.1f}°)")
    r1 = tl.solve_transmission_line(
        G, D=D, n_harmonics=N_HARM, f0_hz=f0,
        bc_harmonics_override=bc_pass1,
        sink_pressure_bc=0.0, verbose=False)
    print(f"[simulate] Pass 1 done ({len(r1.edge_flows)} edge flows)")

    # v_ratios from cross-tile _h_*_norm (matches pipeline_outlet_agg)
    def _norm_meas(d):
        dc = d.get('_h_amp_DC_norm')
        if dc is None or not np.isfinite(dc) or abs(dc) < 1e-30: return None
        out = {'DC': float(dc)}
        for k in (1, 2, 3):
            a = d.get(f'_h_amp_H{k}_norm'); p = d.get(f'_h_phase_H{k}_norm')
            if a is None or p is None or not (np.isfinite(a) and np.isfinite(p)):
                continue
            out[f'H{k}'] = complex(float(a)*np.exp(1j*float(p)))
        return out if 'H1' in out else None
    def _stub_flow_dc(sim, node):
        nb = list(G.neighbors(node))[0]
        if (node, nb) in sim.edge_flows:
            return float(complex(sim.edge_flows[(node, nb)][0]).real)
        ef = sim.edge_flows[(nb, node)]
        return -float(complex(ef[0]).real)

    v_ratios = {}
    for v_anchor in V:
        for _, nb in G.edges(v_anchor):
            d = G.edges[v_anchor, nb]
            mm = _norm_meas(d)
            if mm is None: continue
            v_ratios[v_anchor] = {
                k: (mm.get(f'H{k}') / mm['DC']) if mm.get(f'H{k}') is not None
                    else complex(0)
                for k in (1, 2, 3)}
            break

    # Pass 2 BCs
    bc_pass2 = dict(bc_pass1)
    for v_anchor in V:
        v1_dc = _stub_flow_dc(r1, v_anchor)
        ratios = v_ratios.get(v_anchor, {})
        arr = np.zeros(N_HARM + 1, dtype=complex)
        arr[0] = v1_dc
        for k in (1, 2, 3):
            if k in ratios:
                arr[k] = v1_dc * ratios[k]
        bc_pass2[v_anchor] = arr
        print(f"[simulate] Pass 1 V DC at {v_anchor}: {v1_dc:+.4f}  →  "
              f"Pass 2 |H1|={abs(arr[1]):.3f}∠{math.degrees(np.angle(arr[1])):+.1f}°")

    print(f"[simulate] Pass 2: solving with full Q-BC set …")
    r2 = tl.solve_transmission_line(
        G, D=D, n_harmonics=N_HARM, f0_hz=f0,
        bc_harmonics_override=bc_pass2, verbose=False)
    tl._vessel_admittance = _orig_va

    # Write intact-sim fields onto every edge
    n_written = 0
    for (u, v), ef in r2.edge_flows.items():
        d = G.edges[u, v]
        for k in range(N_HARM + 1):
            z = complex(ef[k])
            suffix = 'DC' if k == 0 else f'H{k}'
            d[f'_h_Z_{suffix}_sim_intact']    = z
            d[f'_h_amp_{suffix}_sim_intact']  = float(abs(z))
            if k > 0:
                d[f'_h_phase_{suffix}_sim_intact'] = float(np.angle(z))
        n_written += 1
    print(f"[simulate] wrote _h_*_sim_intact on {n_written} edges")

    print(f"[simulate] writing  {out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(G, f)
    print(f"[simulate] wrote {out.stat().st_size/1024/1024:.0f} MB")


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


@app.command()
def view(stage_root: Path = typer.Argument(...),
         field: str = typer.Option("mean_Q",
             help='Field to colour edges by on startup')
         ) -> None:
    """Launch the read-only mosaic viewer.

    Resolves graph + fuse image + tile positions + per-tile videos
    from the stage folder's StageConfig, then invokes the
    ported-from-pertile ReadOnlyMosaicViewer with those paths.
    """
    from .viewer import ReadOnlyMosaicViewer
    cfg = StageConfig.load(stage_root)
    graph_path = cfg.graph_piv_path if cfg.graph_piv_path.exists() else cfg.graph_vessels_path
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
        # A frozen release (has MANIFEST.json) is read-only: never write caches back into its graph.
        cache_harmonic_class=not (Path(stage_root) / "MANIFEST.json").exists(),
    ).run()


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
