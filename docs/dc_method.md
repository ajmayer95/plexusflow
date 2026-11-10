# Mean flow (DC): conservative reconciliation and passive-geometry comparison

## Measured flux per vessel
Each vessel segment $e$ is imaged by one or more tiles $\alpha$, each giving a flux time series
$f_{e,\alpha}(t)$ (250 Hz, a few cardiac cycles) at that tile's heart rate.  The flux is the Kloosterman
per-slice Poiseuille estimate (centred parabola per cross-section, $R^2 \ge 0.5$, area from the slice's own
fitted radius, slices whose flow direction disagrees with the median rejected — no flow-rate band since
2026-11-09; $f_{e,\alpha}$ = fraction of slices kept).  The fitted slice radius is a nuisance parameter of the
flux fit (depth averaging and optical-flow smoothing widen it), not vessel geometry.  A measurement where Kloosterman fails is treated as unimaged — there is no fallback estimator.

**Reliability.**  Each measurement gets a reliability scale
$$c_{e,\alpha}=c_0+c_p\,(1-p_{e,\alpha})+c_a\,(1-a_{e,\alpha}),$$
where $p$ is the parabolicity — the mean over all cross-sections of the parabolic fit's $R^2$ clipped to $[0,1]$
(sections without a fit count 0; since 2026-11-10, replacing the kept fraction $f$: the two rank recordings almost
identically, Spearman 0.95–0.97, but $p$ predicts held-out junction imbalance better, tile-CV $\Delta$NLL −22 / −24 in
21S / 27S, and $f$ adds nothing once $p$ is in; $p$ does not penalise slow vessels beyond $f$) — and $a$ is the speed-weighted $|\cos\theta|$ between the time-mean optical-flow vector and the centreline
tangent within $\pm R$ (does the flow run along the traced vessel?).  The parameters are fitted by maximum
likelihood on same-tile junction balance — at every junction whose incident segments were all measured in the
same tile, the net flux $\sum_e s_e Q_e \sim \mathcal N(0, \sum_e Q_e^2 c_e^2 + \sigma_0^2)$ — giving
21S $c_0 \approx 0.05$, $c_p \approx 0.43$, $c_a \approx 1.43$ (4,478 junctions; one $c_0$ for all tiles — per-tile terms did not improve
held-out coverage, `scripts/dc_reconcile.py --tile-terms` reproduces the older model).  $c$ is an empirical
relative-reliability scale, not a standard error; the labels trusted ($c<0.5$), weak ($0.5$–$0.8$) and
untrusted ($\ge 0.8$) are for reporting only — every Kloosterman-valid measurement enters with its weight.

Per segment the input aggregates all its recordings (since 2026-11-10; before, the single recording with the smallest $c$):
$y_e=\sum_t\alpha_{e,t}y_{e,t}$ with $\alpha_{e,t}\propto c_{e,t}^{-2}$; the effective recording quality is the weighted RMS
$c_e=(\sum_t\alpha_{e,t}c_{e,t}^2)^{1/2}$ (no extra confidence for more recordings — they are different moments, not
independent samples of one flow); and the disagreement of the recordings,
$s_{e}^2=\sum_t\alpha_{e,t}(y_{e,t}-y_e)^2$, enters the reconciliation variance as an absolute floor,
$\sigma_e^2=c_e^2Q_e'^2+\sigma_0^2+s_e^2$.  $c_{e,t}$ is calibrated on simultaneous junctions; $s_e$ is not calibrated — it
is a regularisation scale for incompatible observations (without it, a vessel whose recordings disagree in sign averages
to ≈ 0 and receives 4–5× the median weight).  The reconstruction is a conservation-consistent reference field
summarising the mosaic, not a simultaneous physiological state.  Validation against a vessel's own recordings holds one
recording out per fold (`dc.repeat_folds`).  No per-tile gain is applied.  Every edge carries an explicit orientation (tail → head); positive flux runs tail → head.

## Domain
The two porous zones adjacent to the dorsal aortae, where optical-flow fluxes are not reliable,
are removed together with the aortic stubs.  The analysis uses the remaining trusted plexus
(9,544 segments, 7,493 imaged).  Its boundary $\mathcal B$ is the 35 vertices where flow enters
from the porous zones plus the two vitelline veins; mass is conserved at every other vertex.

## Conservative field and emergent boundary
The measured field does not conserve mass.  We take the conservative field that best fits the measurements under the
reliability model, as a quasi-likelihood with the measurement variance evaluated at the reconciled flow,
$$Q'=\arg\min_Q \sum_{e\ \rm imaged}\int_{Q^{\rm meas}_e}^{Q_e}\frac{u-Q^{\rm meas}_e}{c_e^2u^2+\sigma_0^2}\,du
\;+\;\frac{\varepsilon}{2}\sum_{e\ \rm unimaged}Q_e^2\quad\text{s.t.}\quad B_{\mathcal I}Q=0,$$
with $B$ the signed incidence matrix ($+1$ at the tail, $-1$ at the head) restricted to interior
vertices $\mathcal I$ ($\sigma_0$ = 1 % of the median measured flux, a floor only).  Near the data each term is the
weighted square $(Q_e-Q^{\rm meas}_e)^2/2\sigma_e^2$ with $\sigma_e^2=c_e^2Q_e^2+\sigma_0^2$; setting σ from the
reconciled rather than the measured flux means segments that happen to read low are not given spurious confidence.
Unimaged segments are filled by the constraint; the weak ridge ε only fixes circulations that conservation cannot
determine (loops of unimaged segments: the null space of $[B_{\mathcal I};M]$, dimension 44 in 21S, 170 vessels).
Most unimaged segments are not vessels in their own right: 68 % (21S; 27S similar) are shorter than two radii and lie inside a
junction's lumen, where vectorisation split one crossing into two nearby nodes.  Contracting them is exact for the
reconstruction (the segment's flow only links its two node balances), so reported counts use the contracted network
(`network.contract_short_unimaged`, stored as `metrics["contracted"]`): 21S 8146 vessels, 92 % imaged (653 unimaged);
27S 11469 vessels, 92 % imaged (885 unimaged).
The objective is not convex, so the start is part of the method: the uniform-weight conservative projection of the data,
followed by projected-Newton descent (each step one weighted projection, Armijo line search) to a constrained stationary
point.  Sensitivity to the start is reported (`unstable`, `Q_reconciled_sd`: four perturbed starts).  The boundary
fluxes are an output, not a fit: $f=BQ'$ is non-zero only on $\mathcal B$.  No left/right symmetry
is imposed (imposing it changes the agreement below by < 0.05 and is reported as a check).

## Passive-geometry comparison
We ask how much of the reconciled flow the vessel calibres alone explain by driving a passive
Poiseuille network through the same emergent boundary,
$$L_KP=f,\qquad Q^{\rm sim}=K\odot B^\top P,\qquad K_e=r_e^4/\ell_e,$$
with $f$ imposed at every boundary vertex (one pressure fixed as gauge).  Each segment radius is
the geometric mean of two independent estimates — the radius of the traced vessel and the
median optical-flow profile radius across the tiles that imaged it — which reduces per-vessel
radius error (the two disagree by ~30 % in log scale).  Alternative (`dc_reconcile.py --radius anatomical`): the
anatomical radius from the segmentation alone — per-vessel median of the medial-axis distance transform of the mask, plus
a boundary offset Δ = 3.0 px [2.0, 3.7] calibrated on lumen widths (`scripts/anatomical_radius.py`).  The passive
agreement is nearly the same (21S 0.78 vs 0.79); the AC distensibility scales roughly as $R^2$ (21S $D$ 5.6e-4
[4.7, 6.3] over the Δ interval vs 5.0e-4 consensus).  Agreement is
$\mathrm{corr}(|Q'|,|Q^{\rm sim}|)$ over imaged segments, compared with the same network after
randomly permuting the radii across segments.

## Result
| | |
|---|---|
| total inflow through the carve boundary (L / R) | 7.20 nL/s (2.86 / 4.34) |
| vitelline-vein outflow V1 / V2 | 2.62 / 4.59 nL/s |
| corr(\|Q′\|, \|Q^sim\|), linear / log (7,493 segments) | 0.837 / 0.728 |
| same, radii shuffled across segments | 0.38 |
| segments whose direction the conservative correction reverses | 3.1 % |
| median relative change of a measured flux | 25 % |

(`results/dc_reconciled_v2.pkl` → `metrics`; passive flow without the venous-pressure penalty.  Paper Fig 7 adds the soft
equal-venous-pressure penalty, `plexusflow.reconcile.venous_pressure_penalty`, σ_P = 2 Pa: r_log 0.719 over measured vessels,
0.78 and 96 % direction agreement over direction-resolved vessels, |Q′| > 2σ.)

The measured calibres explain the reconciled flow pattern well beyond what topology alone gives.
The weighting was validated on synthetic networks with known truth and on held-out real vessels
(hide 10 % of clean segments, refit the reliability model without them, reconcile, predict): median
relative error 0.29 with the reliability weights vs 0.34 for σ ∝ |Q′| alone, 0.47 for σ ∝ c|Q^meas|
and 0.37 with the reliabilities shuffled.

*Code: `plexusflow.reconcile` (driver `scripts/dc_reconcile.py`); figures `scripts/make_figure6.py` (reliability, reconciliation) and `scripts/make_figure7dc.py` (whole-organ DC, Fig 7).*

## Scope of the error model (diagnostics, 21S, 2026-11-10)
The reliability model is calibrated on junctions whose vessels were imaged simultaneously (one tile); it describes
measurement discrepancy, not physiological change between acquisitions or structural uncertainty in the graph.
Diagnostics of the production reconstruction (scripts in the gitignored `sandbox/`: `redundancy.py`, `correlated_error.py`):
- Within tiles the model is calibrated: fully measured same-tile junctions RMS z = 0.91 (Fig 6a, held-out 0.97–1.05).
- Across tiles discrepancies are larger: junctions whose best recordings come from different tiles RMS z = 1.49. A
  per-tile log-scale term fitted to them gives τ ≈ 0.43 (≈ ×1.5 per tile, 1 SD), matching the between-tile excess
  (τ ≈ 0.37) of repeat recordings. This mixes physiological change during the ~3 h sequential acquisition with any
  tile-shared measurement error; it is not corrected (a per-tile factor would normalise away real state changes), and
  the reconstruction is a conservation-consistent reference field, not a simultaneous snapshot. The spread floor
  s_between covers disagreement among a vessel's own recordings but not errors shared by all vessels of one tile, so
  the reconciled uncertainties are optimistic across tiles.
- Standardized reconciliation adjustments (linearised; z/√r with r = 1 − leverage): robust scale 1.10 overall, 1.00–1.05
  for the most reliable and ≈ 1.16 for the least reliable vessels; median signed ln(Q′/y) = 0.000 (no normalisation
  bias); direction reversals 0.1 % (most reliable quintile) → 11 % (least reliable).
- The quasi-likelihood penalises inflating a flux only logarithmically (variance evaluated at Q′). Interior vessels are
  held by conservation at both ends; a vessel free at one end is not. Measured dead ends must therefore stay closed (27S
  test: opening 18 measured dead ends let several near-zero measurements reconcile to ±0.7–1.9 nL/s; reverted). 21S has
  no measured dead ends (5 unmeasured, closed).
