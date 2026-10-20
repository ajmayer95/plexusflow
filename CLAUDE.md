# plexusflow — Claude Code Instructions

**plexusflow is the primary codebase as of 2026-08-21** (see memory
`plexusflow_is_primary`).  All new figure, inference, and normalization
work goes here.  The sibling `../PerTileFlow` tree is kept only for the
read-only napari viewer, a few legacy inference-diagnostic scripts, and
the pertile source still referenced from the viewer.

## Launching the viewer

Plexusflow has its own viewer stack under `plexusflow.cli`.  Full list
of commands is in [`LAUNCH.md`](LAUNCH.md) — copy/paste commands.
Most useful:

```bash
# Read-only mosaic viewer (default stage root path shown)
python3 -m plexusflow.cli view "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"

# Threshold + paint mask editor
python3 -m plexusflow.cli mask ".../PlexusFlow21S"

# Unified mask + vectorize + preview (one window)
python3 -m plexusflow.cli mve  ".../PlexusFlow21S"

# Graph editor (net3)
python3 -m plexusflow.cli edit ".../PlexusFlow21S"
```

The viewer entry point in source is
`plexusflow/src/plexusflow/viewer/app.py` (one large file; CLI dispatch
is in `plexusflow/src/plexusflow/cli.py`).

The legacy napari viewer at `../PerTileFlow/pertile/viewer/mosaic_readonly_app.py`
is still functional and some diagnostic scripts import from it — don't
delete, just don't start new work on it.

## Layout

- `src/plexusflow/` — library (`viewer/`, `mosaic/`, `inference/`,
  `normalize/`, `simulate/`, etc.)
- `scripts/` — ALL figure-generation scripts, diagnostics,
  experiments, re-aggregators.  Everything new goes here (never in a
  scratchpad).  Representative current scripts (2026-10):
  - main-paper figures, order as of 2026-10-20:
    - Fig 6 `make_figure6v3.py` — DC reliability c and conservative reconciliation (→ Q′)
    - Fig 7 `make_figure7.py` — cross-tile phase synchronisation (tile gauges ψ_t, H1/H2 validation)
    - Fig 8 `make_figure8alt_v3.py` — organ-scale DC field Q′, emergent boundary f = BQ′, passive network (DC only)
    - Fig 9 (to build) — boundary forcing f_b·r_b^(n), RC model, D profile (old Fig 7v3 boundary-waveform panel
      belongs here)
    - Fig 10 `mech/make_figure9alt_v3.py` (to rename) — measured vs simulated pulsatility, H1/H2 phase, waveforms
  - `renormalize_to_tile21.py` — rerun normalization anchored at tile 21
  - `shift_tile21_anchor_by_half_period.py` — cosmetic T/2 shift
  - `reaggregate_coherent_h2_gauge.py` — fixes the independent-H2
    gauge bug (see memory [[coherent_h2_gauge_2026-10-05]])
  - `exp_pairwise_h1h2_coherence.py` — gauge-free H1/H2 pairwise test
  - `trace_single_vessel.py`, `trace_outlet_23072.py`,
    `diagnose_outlet_waveform_aggregation.py` — diagnostics
  - `plot_reconciled_bcs_tile14.py`, `plot_bcs_and_adjacent_vessels.py`,
    `plot_da_vs_carve_outlets.py` — reconciliation plots
  - `run_sim_carved_outlets.py` — carved TL sim driver
- `notebooks/` — exploration (now including
  `explore_normalization.ipynb` — interactive gauge/display playground)
- `docs/`, `examples/`, `tests/`, `runs/`, `renders/` — standard

Scripts expect to be run with the current working directory being
either `plexusflow/` itself or the sibling `PerTileFlow/`.  They use
`ROOT = Path(__file__).resolve().parents[2]` plus hard-coded absolute
`STAGE` paths, so cwd rarely matters — just invoke as
`python3 scripts/make_figure....py`.

## Canonical data location

Stage data lives OUTSIDE this folder, at:

```
/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S/
  graph/      — gpickles (base, normalized, carved sims)
  fuse/       — organ_fusion_close.tif (mosaic)
  figures/    — all figure outputs, including figures/diagnostics/
```

**Current canonical graphs** (see memory [[canonical_files_2026-10-01]]
for the base set; [[coherent_h2_gauge_2026-10-05]] for the recent H2
fix):

- `graph_vessels_piv_normalized_tile21anchor_REAL.gpickle` — baseline
  tile-21-anchored normalized graph.
- `graph_vessels_piv_normalized_tile21anchor_REAL_cohH2.gpickle` —
  same, but every edge's H2 phasor re-aggregated with the coherent
  gauge (ψ⁽²⁾ := 2·ψ⁽¹⁾).  DA/V bc_harmonics are preserved from the
  KCL reconstruction, with coherent-gauge phase applied to the H2/H3
  components (magnitude kept).
- `graph_vessels_piv_normalized_tile21anchor_REAL_cohH2_t5shifted.gpickle` —
  cohH2 plus a cosmetic T/2 time-origin shift so cardiac peaks land
  near t=0.5 (cosmetic only; a global shift that preserves all
  relative phase).
- `graph_vessels_piv_state.carved_outlets_tile21_local_r.gpickle` —
  carved sim built from the pre-cohH2 graph (needs re-running with
  cohH2 to propagate the fix downstream).

## Memory

Project memory lives at
`~/.claude/projects/<project-dir>/memory/`.  The memory folder is
symlinked between the `PerTileFlow` and `plexusflow` project dirs so
both sessions see the same notes.  **ALWAYS READ** `methods.md` before
touching batch analysis, SNR gating, or sign conventions.

Current-default notes to know (in `MEMORY.md`):

- [[coherent_h2_gauge_2026-10-05]] — THE active gauge-bug fix.
  tile_normalize solves z_H2 independently of z_H1; this changes
  waveform shape.  Validated by pairwise ε test; coherent gauge
  ψ⁽ⁿ⁾ := n·ψ⁽¹⁾ is correct.
- [[fig9v3_bc_pipeline_2026-10-02]] — Fig 9v3 master BC pipeline.
- [[canonical_files_2026-10-01]] — current canonical set.
- [[simple_normalization_recipe]] — per-tile has ONLY CO_rel +
  z_arg_H1 (reinforced by coherent-gauge finding).
- [[plexusflow_is_primary]] — plexusflow is primary since 2026-08-21.

When modifying normalization, batch analysis, SNR gating, quality
tiers, or sign conventions, update `memory/methods.md` immediately.

## Inference tab in the viewer

The viewer's inference tab (`_inference_run` in `app.py`) implements:

- Forward model: distributed two-port polynomial O(C_e) form:
  `Y_self = G·(1 + κ²/3)`, `Y_trans = G·(1 − κ²/6)`.  Edge flow
  `Q = Y_self·p_u − Y_trans·p_v`.
- D-grid cap: `max_e C_e ≤ 0.3`.
- Noise: white-residual phasor variance; σ_b log-uniform grid for the
  boundary forcing prior; `s²` noise-floor marginalisation.
- A/V→BC promotion: nodes with `boundary_type ∈ {'source','sink'}`
  in the carve interior get promoted to boundary.
- Per-tile matched (Q, SNR) read from `measurements_piv[tile_id]`,
  never top-level canonical fields.

Known identifiability: data identify `D·|b̂_H1|` and the SHAPE of b̂
across BC nodes.  Absolute D₀ falls in `[10⁻⁴, 10⁻³] /Pa` across σ_b ∈
`[3, 30] Pa` — point value is one slice of a structural ridge, see
"Identifiability note" in the Show Priors panel.

Shapes of b̂_H1, b̂_DC, and D̂ tile-ratios are σ_b-invariant and are
the publishable identified quantities.
