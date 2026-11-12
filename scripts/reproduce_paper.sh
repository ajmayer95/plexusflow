#!/usr/bin/env bash
# Reproduce every derived result and all main-text figures of the paper from a data release.
#
#   export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release      # the extracted data release (read-only is fine)
#   export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results      # where derived results go (default <stage>/results)
#   export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures      # where figures go (default <stage>/figures)
#   bash scripts/reproduce_paper.sh            # results + figures   (~5 min on a laptop)
#   bash scripts/reproduce_paper.sh --figures  # figures only, from existing results
#
# Steps: DC reconciliation → weighting validation → aligned pulsatile (AC) field → distensibility fit → Figs 1–9 → figures/latest/.
# Point RESULTS and FIGURES at your own folders so a frozen release is never modified.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLEXUSFLOW_STAGE:?set PLEXUSFLOW_STAGE to the extracted data release (see README)}"
PY=${PYTHON:-python3}
run() { echo; echo "=== $* ==="; "$PY" "$@"; }

if [[ "${1:-}" != "--figures" ]]; then
  mkdir -p "${PLEXUSFLOW_RESULTS:-$PLEXUSFLOW_STAGE/results}"
  # (after a change of the Kloosterman section rule: python3 scripts/kl_reaggregate.py — numpy<2 — rewrites the graph first)
  run scripts/dc_reconcile.py          # reliability model + reconciled mean flows   → reliability_model_v1.json, dc_reconciled_v2.pkl
  RES="${PLEXUSFLOW_RESULTS:-$PLEXUSFLOW_STAGE/results}"
  if [[ -d "$RES/stokes_bench/ensemble" ]]; then run scripts/benchmark/c_truth_calibration.py; fi   # c vs known truth → c_truth_calibration.json (SI)
  run scripts/dc_validate_weighting.py # validation vs repeat recordings            → dc_weighting_validation.json (SI weighting)
  run scripts/dc_junction_calibration.py # c_k vs constant c on held-out tiles     → dc_junction_calibration.json (Fig 6b)
  run scripts/dc_withholding_curve.py  # withhold 5–75 % of measured vessels       → dc_withholding.json (Fig 6e)
  run scripts/dc_validate_passive.py   # conservation-only vs passive-constrained      → dc_passive_comparison.json (SI)
  run scripts/benchmark/dc_reconstruction.py  # known-truth benchmark, noise × radius error → dc_synthetic_benchmark.json (SI)
  run scripts/benchmark/dc_async.py    # known-truth benchmark, tiles see different fields → dc_async_benchmark.json (SI)
  run scripts/ac_field.py              # tile-synchronised harmonic field (paper preset) → ac_field_v7.pkl
  run scripts/mech/ac_estimate.py      # profiled distensibility fit                 → ac_estimate_v8.json
fi

for f in 1 2 3 4 5 6 7 8 9; do        # make_figureN.py draws Fig N (make_figure8.py also the pulsatile-regime SI figure)
  run "scripts/make_figure$f.py"
done
run scripts/make_figure_S_passive.py   # SI: passive-constrained comparison (add a second stage dir for 21S + 27S)
run scripts/make_figure_S_junction_calibration.py   # SI: junction calibration per stage, Fig 6b companion (add a second stage dir for 21S + 27S)
run scripts/make_figure_S_synthetic_dc.py  # SI: known-truth benchmark
run scripts/make_figure_S_async_dc.py      # SI: asynchronous acquisition / tile-scale bias
run scripts/collect_figures.py         # copies Fig1–Fig9 to <figures>/latest/
