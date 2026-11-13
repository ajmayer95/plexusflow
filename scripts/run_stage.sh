#!/usr/bin/env bash
# Analysis chain for ONE stage (any embryo), from its measured graph to the results the read-only viewer shows.
#
#   bash scripts/run_stage.sh /path/to/PlexusFlowNNS              # reconstruction + validation + pulsatile field + D fit
#   bash scripts/run_stage.sh /path/to/PlexusFlowNNS --no-ac       # mean flow only
#   bash scripts/run_stage.sh /path/to/PlexusFlowNNS --no-validate # skip the validation runs
#   PLEXUSFLOW_RESULTS=/elsewhere bash scripts/run_stage.sh ...    # write results somewhere other than <stage>/results
#
# Prerequisites (see docs/pipeline.md): graph/graph_vessels_piv.gpickle from `plexusflow analyze`, with boundary anchors
# (DA 'source', veins 'sink') and region tags (open / topology uncertain) set in `plexusflow workbench`; for the AC steps,
# ac_ref_tile (and optionally ac_f0_band, ac_veins, ac_gauge_block) in <stage>/plexusflow.yaml.
# The paper's figures are made by scripts/reproduce_paper.sh, which runs the same chain for the 21S release.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?usage: run_stage.sh /path/to/stage [--no-ac] [--no-validate]}"; shift
AC=1; VALIDATE=1
for a in "$@"; do
  case "$a" in
    --no-ac) AC=0 ;;
    --no-validate) VALIDATE=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
export PLEXUSFLOW_STAGE="$(cd "$STAGE" && pwd)"
RES="${PLEXUSFLOW_RESULTS:-$PLEXUSFLOW_STAGE/results}"; mkdir -p "$RES"
PY=${PYTHON:-python3}
run() { echo; echo "=== $* ==="; "$PY" "$@"; }
echo "stage $PLEXUSFLOW_STAGE → results $RES"

run scripts/dc_reconcile.py                 # reliability model + reconciled mean flow → reliability_model_v1.json, dc_reconciled_v2.pkl
run scripts/dc_qc.py                        # review list: gross-error tests, loops, hotspots → dc_qc.csv, dc_qc.json
if [[ $VALIDATE == 1 ]]; then
  run scripts/dc_validate_weighting.py      # leave-one-recording-out validation       → dc_weighting_validation.json
  run scripts/dc_junction_calibration.py    # calibration on held-out tiles            → dc_junction_calibration.json
  run scripts/dc_withholding_curve.py       # whole-vessel withholding, 5–75 %         → dc_withholding.json
fi
if [[ $AC == 1 ]]; then
  run scripts/ac_field.py                   # tile-synchronised harmonic field         → ac_field_v7.pkl
  run scripts/mech/ac_estimate.py           # profiled distensibility fit              → ac_estimate_v8.json
fi
echo
echo "done. View:  plexusflow view \"$PLEXUSFLOW_STAGE\""
