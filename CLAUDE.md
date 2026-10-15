# plexusflow — instructions for Claude Code (any machine)

Whole-organ vascular flow from yolk-sac videos: per-tile optical flow → per-vessel flux → conservation-consistent
reference field (paper 1). Modelling on top of it (two-port network, D/η inference) is `plexusmech` (paper 2).
Owner: Alex Mayer. Two machines: office Mac (raw videos live there) and laptop. Code syncs through GitHub, data through
Google Drive.

## Where everything is

| What | Where |
|---|---|
| Code (this repo) | `~/code/plexusflow` — GitHub `ajmayer95/plexusflow` (private). Pull at start; Alex pushes at the end. |
| Stage data (graphs, mosaics, results, figures) | Google Drive `My Drive/PlexusFlow15S`, `PlexusFlow21S`, `PlexusFlow27S` (`graph/`, `fuse/`, `results/`, `figures/`). Office Mac path: `/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/` — on the laptop find it with `ls ~/Library/CloudStorage/`. |
| Frozen data releases (read-only!) | `My Drive/PlexusFlow{15S,21S,27S}_release_v2` + GitHub release `data-v2`. **Never write into a release folder.** |
| Diagnostics / experiments | `sandbox/` = symlink to Drive `My Drive/Python/Yolk Sac Code/plexusflow-sandbox` (gitignored). On a new machine: `ln -s "<Drive>/My Drive/Python/Yolk Sac Code/plexusflow-sandbox" ~/code/plexusflow/sandbox`. Logs are `sandbox/run_*.log`. |
| Raw videos | office Mac only (`~/Desktop/ImageData/...`). Fresh OF, `analyze` and benchmark-bundle building need them; nothing else does. |
| Claude memory notes (decisions, gotchas) | snapshot on Drive: `My Drive/Python/Yolk Sac Code/claude-memory/` (start with `MEMORY.md`; `README.txt` says how to link it). Read the relevant notes before changing methods. |
| Old Drive code folder `My Drive/Python/Yolk Sac Code/plexusflow` | archive only — do not develop there. `../PerTileFlow` = legacy. |

Scripts find data through `plexusflow.paths`: set `PLEXUSFLOW_STAGE` to a stage folder (results default to
`<stage>/results`, figures to `<stage>/figures`; override with `PLEXUSFLOW_RESULTS` / `PLEXUSFLOW_FIGURES`, e.g. when
working from a release or writing to scratch).

## Environments

- **System `python3` (3.9, numpy 1.26)** — default for pipeline scripts and the napari viewer:
  `PYTHONPATH=src python3 scripts/...` (the package says ≥ 3.10, so it is not pip-installed there).
  **Write graph gpickles only from this Python** — pickles saved under numpy 2 are unreadable by numpy 1.26.
- **`.venv` (3.11, numpy 2)** — tests, ruff, the synthetic benchmark (`pip install -e ".[dev,bench]"`).
  Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q tests`; lint (as CI): `.venv/bin/ruff check src tests`.
  On a machine without the venv: `python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,bench]"`.

## Working rules (agreed with Alex)

- Work on a branch, never commit to `main` directly. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. **Alex runs `git push`** and merges PRs.
- The repo holds functional pipeline code only. One-off diagnostics, sweeps and comparisons go in `sandbox/`, never committed.
- Long jobs (> ~30 s): run in the background with `python3 -u`, timestamped progress lines, hard caps on every
  loop/optimizer, output to `sandbox/run_<name>.log`; tell Alex the log path and an ETA. Never run long jobs in the
  foreground through `| grep | tee` (buffers everything; looks frozen).
- Prefer established, citable formulations over bespoke numerics; justify any method change by a measured failure of the
  simpler version, one change at a time.
- Alex is Alex Mayer (not Adam). Use they/them for people whose pronouns are unknown.

## Layout (2026-10)

```
src/plexusflow/   analyze/ (OF + Kloosterman batch), reconcile/ (DC), ac/ (AC field), release.py (release loader),
                  viewer/ (view = read-only, workbench = editing), synthetic/ (known-truth benchmark), legacy/ (2026-09
                  tile_normalize / KCL, unused), cli.py, paths.py, config.py, upstream stages (preprocess … condense)
src/plexusmech/   modelling: ac/ (two-port model, likelihood, (D, η) estimate), tl/ (TL simulator), inference/ (legacy carved sim)
scripts/          dc_reconcile.py → ac_field.py → make_figure8alt_v3.py, make_figure_ac_field.py; make_release.py;
                  benchmark/ (known-truth benchmark); mech/ (ac_estimate.py, make_figure9alt_v3.py)
tests/            pytest; test_package_boundary.py: plexusflow never imports plexusmech or legacy (except lazy hooks)
```

Viewer: `PYTHONPATH=src python3 -m plexusflow.cli view "<stage>"` (read-only) or `... workbench "<stage>"` (fresh OF,
tagging, boundary edits; refuses release folders). Other commands: `LAUNCH.md`.

## Current methods and decisions (read before changing anything)

- **DC reconciliation is FROZEN** (`reconcile/dc.py`): one winner measurement per vessel (smallest reliability c);
  weights relative to the *reconciled* flow, w = 1/(c²Q′² + s0²); damped re-weighting settles into a small limit cycle →
  cycle average over an 8-run ensemble, `unstable` flags. Output = a **conservation-consistent reference field**
  (measurements are asynchronous; not an instantaneous snapshot). Justification (sandbox ladder, 2026-10-15): manuscript
  weights 0.2|y| bias the field low (21S right arterial entry 1.4 vs 4.75 measured); uniform / quality-only weights flip
  20–29 % of measured directions; reconciled-scale relative weights are best on held-out data, 3–4 % flips.
  Heteroscedastic ML, joint calibration, fixed-σ variants, boundary sign constraints, bounded re-weighting: tested and
  rejected — do not reopen without new evidence (memory `dc_estimator_frozen_2026-10-15`).
- **The venous boundary is effectively unmeasured.** 21S: arterial entry measured 6.5 nL/s, the two V stubs 0.5 (DC
  under-read ~10×; their waveforms are trusted). Conservation routes the venous return to the head-end veins; that routing
  in the upper lobes is not determined by the data (repeat measurements of one vessel differ by a median 2–2.6×). Fig 8
  panel A draws "conservation-imputed" vessels (unmeasured, or |Q′| > 5× their measurement) faint. Leave V-stub values as they are.
- **AC field** (`ac/field.py`): tile time gauge ψ by ρ-weighted angular synchronisation (all valid pairs, bootstrap SD;
  phases compared only where SD ≤ 30°), coherent k·ψ; Ĥ_k = (|Q′|/|a0|) e^{−ikψ} H_k from the AC winner (smallest per-beat
  scatter). Harmonics are deliberately NOT conservation-projected (distributed compliance: endpoint flows differ at ω > 0);
  the junction residual is an independent check.
- **Fig 9 control** (sandbox `ac_boundary_distance_control.py`): forcing donors do not carry the agreement; compare against
  a constant-phase baseline — 21S H1 skill over constant phase is small, H2 large; 27S beats all baselines; 15S has no
  interior skill (its agreement comes from near-boundary vessels).
- Calibration junctions touching porous vessels balance 2× worse — restrict to the carve if the reliability fit is ever refit.

## Open items (2026-10-15)

1. 15S: label uncertain regions arterial vs venous (its venous return leaves through carve regions; inflow ≈ 0 until labelled).
2. 27S: 4 injecting V nodes / strong-oscillation boundary nodes; fit D for 27S and 15S.
3. AC inference weights: replace DC-based c with AC uncertainties (harmonic-fit covariance + k²σ_ψ²) before the (D, η) fit.
4. Benchmark bundle: attach `PlexusFlow21S_benchmark_v1` to the `data-v2` release (rebuild on the office Mac with
   `scripts/benchmark/make_bundle.py` — needs raw videos). Branch `synthetic-benchmark` holds the benchmark code.
5. Condense keeps parallel loops; viewer `app.py` (14k lines) has dead tab builders to remove.
