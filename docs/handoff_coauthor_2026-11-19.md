# Co-author handoff: state on 19 Nov 2026

This note gets you from nothing to regenerating every figure and tracing any number back to its code. Read
[AGENTS.md](../AGENTS.md) first (conventions, where things live), then this.

## 1. Access

| what | where | how you get it |
|---|---|---|
| code | GitHub `ajmayer95/plexusflow`, branch **`paper-revision-2026-11`** | collaborator invite |
| data (stage) | Google Drive folder `PlexusFlow21S/` | shared Drive folder |
| manuscript | Overleaf | Overleaf share |

Raw tile movies are not shared (office Mac only). Everything downstream of the measured graph can be regenerated without them;
re-running optical flow / Kloosterman measurement cannot.

## 2. Setup

```bash
git clone https://github.com/ajmayer95/plexusflow.git && cd plexusflow && git checkout paper-revision-2026-11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e ".[figures,gui]" --no-deps     # numpy 1.x: required to WRITE graphs
export PLEXUSFLOW_STAGE="<your Drive>/PlexusFlow21S"
export PLEXUSFLOW_RESULTS="$HOME/pf_out/results"      # your own outputs — do not write into the shared stage
export PLEXUSFLOW_FIGURES="$HOME/pf_out/figures"
```

Point `PLEXUSFLOW_RESULTS` / `PLEXUSFLOW_FIGURES` at your own folders. Scripts read inputs from the stage and write only there.
Before running figure scripts that read results, either copy `PlexusFlow21S/results/` into your results folder or run the
chain (below). Do not edit the stage graph (`graph/graph_vessels_piv.gpickle`) directly: topology edits go through the
workbench patch log (`graph/patches/`), coordinate those with Alex.

## 3. Regenerating everything

```bash
bash scripts/run_stage.sh "$PLEXUSFLOW_STAGE"          # DC reconstruction, QC, validations, pulsatile field, D fit (~5 min)
python scripts/dc_reconcile.py --symmetric              # 50/50 left/right sensitivity
for n in 1 2 3 4 5; do python scripts/make_figure$n.py; done
python scripts/make_figure6_draft.py                    # DRAFT, will change substantially
python scripts/make_figure8.py; python scripts/make_figure9.py   # see numbering below
python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"       # read-only viewer
```

Each figure script's docstring lists its panels and inputs; each prints the numbers it uses.

## 4. Figure numbering (revision in progress)

The manuscript is moving to 8 figures. Script numbers do not match yet:

| paper | subject | script |
|---|---|---|
| 1 | imaging, registration, acquisition order | `make_figure1.py` (revised 19 Nov) |
| 2 | mosaic → probability → mask → graph; tile mapping | `make_figure2.py` (revised 19 Nov) |
| 3 | one vessel: optical flow, Poiseuille fits, harmonic decomposition | `make_figure3.py` |
| 4 | arterial / venous path examples | `make_figure4.py` |
| 5 | domain schematic and boundaries | `make_figure5.py` |
| 6 | mean flow (DC): fusion + conservation (+ passive comparison) | `make_figure6_draft.py` (draft) + passive panels of `make_figure7.py` |
| 7 | cardiac phase reconstruction | `make_figure8.py` |
| 8 | RC-network pulsatility, distensibility D | `make_figure9.py` |

Revised exports are collected in `PlexusFlow21S/figures_revised_8/` as `Fig1.png`, `Fig2.png`, …; the original `figures/` is
untouched. Figs 1 and 2 share one mosaic image via `scripts/mosaic_display.py`.

## 5. Method change on 19 Nov: mean-flow (DC) estimator

Production DC is now static inverse-variance fusion followed by constrained weighted least squares
(`src/plexusflow/reconcile/dc.py`, `ESTIMATOR = "fixed"`, `VARIANCE = "block"`; written up in
[dc_method.md](dc_method.md)):

1. per recording, the variance of its mean flux from heartbeat-block means of the residual about a mean + 3-harmonic fit;
2. per vessel, the inverse-variance mean of its recordings;
3. the conservative field closest to those means (interior conservation exact; unmeasured vessels minimum-norm).

It replaced the quasi-likelihood estimator (`estimator="ql"`, still available), which inflated weak readings into long
high-flux routes. 21S: inflow 3.76 nL/s (44.5 % left; forced 50/50: 3.91), D = 4.47e-4 /Pa.

Open concern (our review response to the recipe): with ~3 s recordings the per-recording block variance rests on ~4 blocks
and is noisy; a pooled form `v = κ s²/N` (`variance="pooled"`) predicts held-out recordings better and gives 5.18 nL/s,
40 % left. Not yet decided. `V_e` is not a usable uncertainty (repeat recordings differ far more than it implies).

Graph edits on 19 Nov (patch log, reversible): 0006 deleted edge (65434, 65449); 0007 straightened and re-measured edge
(65039, 65350).

## 6. Open items

- Fig 6 redesign (current draft will change substantially).
- Figs 3–5 and 7–8 per the manuscript checklist; Fig 8 D/E: audit of the profiled distensibility curves (what is
  scanned / re-optimised, why the curves are sharp, whether any interval is justified) before strengthening claims.
- Same-vessel waveform comparison for Fig 7: `src/plexusflow/analyze/waveform_shape.py`; the viewer shows it
  (Tile filter "All tiles", no video, "Plot Q(t) for selected").
- Decide block vs pooled recording variance.

## 7. Where to look

| question | file |
|---|---|
| how a recording's flux is measured | `src/plexusflow/analyze/kloosterman.py`, `profile.py` |
| measurement table, recording noise | `src/plexusflow/reconcile/reliability.py` (`measurement_table`, `temporal_cv`) |
| DC reconstruction | `src/plexusflow/reconcile/dc.py`, `network.py`; driver `scripts/dc_reconcile.py` |
| pulsatile field | `src/plexusflow/ac/`, `scripts/ac_field.py` |
| distensibility fit | `src/plexusmech/`, `scripts/mech/ac_estimate.py` |
| full chain | `scripts/run_stage.sh` |
