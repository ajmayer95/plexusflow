#!/usr/bin/env bash
# Reproduce every derived result and all main-text figures of the paper from a data release.
#
#   export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release      # the extracted data release (read-only is fine)
#   export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results      # where derived results go (default <stage>/results)
#   export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures      # where figures go (default <stage>/figures)
#   bash scripts/reproduce_paper.sh            # results + figures   (~5 min on a laptop)
#   bash scripts/reproduce_paper.sh --figures  # figures only, from existing results
#
# Steps: mean-flow reconciliation → pulsatile-flow reconciliation → distensibility fit → Figs 1–7 → figures/latest/.
# Point RESULTS and FIGURES at your own folders so a frozen release is never modified.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLEXUSFLOW_STAGE:?set PLEXUSFLOW_STAGE to the extracted data release (see README)}"
PY=${PYTHON:-python3}
run() { echo; echo "=== $* ==="; "$PY" "$@"; }

if [[ "${1:-}" != "--figures" ]]; then
  mkdir -p "${PLEXUSFLOW_RESULTS:-$PLEXUSFLOW_STAGE/results}"
  run scripts/dc_reconcile.py          # mean flow: 1/σ² average of the recordings, conservation, left = right inflow → dc_reconciled_v3.pkl
  run scripts/pulse_field.py           # tile clocks, averaged harmonics, rescaled to the reconciled mean             → ac_field_v9.pkl
  run scripts/mech/ac_estimate.py      # distensibility of the RC network, boundary forcing as measured                → ac_estimate_v11.json
fi

for f in 1 2 3 4 5 6 7; do             # make_figureN.py draws Fig N
  run "scripts/make_figure$f.py"
done
run scripts/collect_figures.py         # copies Fig1–Fig7 to <figures>/latest/
