# Mean flow (DC): conservative reconciliation and passive-geometry comparison

## Measured flux per vessel
Each vessel segment $e$ is imaged by one or more tiles $\alpha$, each giving a flux time series
$f_{e,\alpha}(t)$ (250 Hz, a few cardiac cycles) at that tile's heart rate.  The flux is the Kloosterman
per-slice Poiseuille estimate (centred parabola per cross-section, $R^2 \ge 0.5$, area from the slice's own
fitted radius, slices outside $[0.5, 1.5]\times$ the median rejected; $f_{e,\alpha}$ = fraction of slices
kept).  A measurement where Kloosterman fails is treated as unimaged — there is no fallback estimator.

**Reliability.**  Each measurement gets a reliability scale
$$c_{e,\alpha}=c_0+c_f\,(1-f_{e,\alpha})+c_a\,(1-a_{e,\alpha}),$$
where $a$ is the speed-weighted $|\cos\theta|$ between the time-mean optical-flow vector and the centreline
tangent within $\pm R$ (does the flow run along the traced vessel?).  The parameters are fitted by maximum
likelihood on same-tile junction balance — at every junction whose incident segments were all measured in the
same tile, the net flux $\sum_e s_e Q_e \sim \mathcal N(0, \sum_e Q_e^2 c_e^2 + \sigma_0^2)$ — giving
$c_0 = 0.118$, $c_f = 0.316$, $c_a = 1.454$ (4,478 junctions; one $c_0$ for all tiles — per-tile terms did not improve
held-out coverage, `scripts/dc_reconcile.py --tile-terms` reproduces the older model).  $c$ is an empirical
relative-reliability scale, not a standard error; the labels trusted ($c<0.5$), weak ($0.5$–$0.8$) and
untrusted ($\ge 0.8$) are for reporting only — every Kloosterman-valid measurement enters with its weight.

Per segment we use the measurement with the smallest $c$ and its oriented mean $Q^{\rm meas}_e$.  No per-tile
gain is applied.  Every edge carries an explicit orientation (tail → head); positive flux runs tail → head.

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
radius error (the two disagree by ~30 % in log scale).  Agreement is
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
