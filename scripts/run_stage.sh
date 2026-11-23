#!/usr/bin/env bash
# Analysis chain for ONE stage (any embryo), from its measured graph to the results the read-only viewer shows.
#
#   bash scripts/run_stage.sh /path/to/PlexusFlowNNS              # mean flow + pulsatile field + distensibility fit
#   bash scripts/run_stage.sh /path/to/PlexusFlowNNS --no-ac       # mean flow only
#   PLEXUSFLOW_RESULTS=/elsewhere bash scripts/run_stage.sh ...    # write results somewhere other than <stage>/results
#
# Prerequisites (see docs/pipeline.md): graph/graph_vessels_piv.gpickle from `plexusflow analyze`, with boundary anchors
# (DA 'source', veins 'sink') and region tags (open / topology uncertain) set in `plexusflow workbench`; for the AC steps,
# ac_ref_tile (and optionally ac_f0_band, ac_veins, ac_gauge_block) in <stage>/plexusflow.yaml.
# The paper's figures are made by scripts/reproduce_paper.sh, which runs the same chain for the 21S release.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?usage: run_stage.sh /path/to/stage [--no-ac]}"; shift
AC=1
for a in "$@"; do
  case "$a" in
    --no-ac) AC=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
export PLEXUSFLOW_STAGE="$(cd "$STAGE" && pwd)"
RES="${PLEXUSFLOW_RESULTS:-$PLEXUSFLOW_STAGE/results}"; mkdir -p "$RES"
PY=${PYTHON:-python3}
run() { echo; echo "=== $* ==="; "$PY" "$@"; }
echo "stage $PLEXUSFLOW_STAGE → results $RES"

run scripts/dc_reconcile.py                 # mean flow: 1/σ² average, conservation, left = right inflow → dc_reconciled_v3.pkl
run scripts/dc_qc.py                        # review list: gross-error tests, loops, hotspots            → dc_qc.csv, dc_qc.json
if [[ $AC == 1 ]]; then
  if [[ ! -f "$RES/speed_pairs_v1.pkl" ]]; then
    run scripts/speed_pairs.py              # clock measurements between tiles (raw movies)             → speed_pairs_v1.pkl
  fi
  run scripts/pulse_field.py                # reconciled pulsatile field (plexusflow.ac.pulse)           → ac_field_v9.pkl
  run scripts/mech/ac_estimate.py           # distensibility fit, forcing as measured                    → ac_estimate_v10.json
fi
echo
echo "done. View:  plexusflow view \"$PLEXUSFLOW_STAGE\""
