# Mean flow (DC): conservative reconciliation and passive-geometry comparison

*Final method, frozen 2026-11-12.  Numbers are 21S production (`results/dc_reconciled_v2.pkl`).*

## Measured flux per vessel
Each vessel segment $e$ is imaged by one or more tiles $\alpha$, each giving a flux time series
$f_{e,\alpha}(t)$ (250 Hz, a few cardiac cycles) at that tile's heart rate.  The flux is the Kloosterman
per-slice Poiseuille estimate (centred parabola per cross-section, $R^2 \ge 0.5$, area from the slice's own
fitted radius, slices whose flow direction disagrees with the median rejected — no flow-rate band since
2026-11-09; $f_{e,\alpha}$ = fraction of slices kept).  The fitted slice radius is a nuisance parameter of the
flux fit (depth averaging and optical-flow smoothing widen it), not vessel geometry.  A measurement where Kloosterman fails is treated as unimaged — there is no fallback estimator; this includes recordings with
fewer than two direction-consistent sections (until 2026-11-11 these fell back to averaging all sections regardless of sign;
104 21S recordings, most grossly off their repeats).

**Fractional discrepancy.**  Each measurement gets a calibrated fractional-discrepancy scale
$$c_{e,\alpha}=c_0+c_p\,(1-p_{e,\alpha})+c_a\,(1-a_{e,\alpha}),$$
where $p$ is the parabolicity — the mean over all cross-sections of the parabolic fit's $R^2$ clipped to $[0,1]$
(sections without a fit count 0; since 2026-11-10, replacing the kept fraction $f$: the two rank recordings almost
identically, Spearman 0.95–0.97, but $p$ predicts held-out junction imbalance better, tile-CV $\Delta$NLL −22 / −24 in
21S / 27S, and $f$ adds nothing once $p$ is in; $p$ does not penalise slow vessels beyond $f$) — and $a$ is the speed-weighted $|\cos\theta|$ between the time-mean optical-flow vector and the centreline
tangent within $\pm R$ (does the flow run along the traced vessel?).  The parameters are fitted by maximum
likelihood on same-tile junction balance — at every junction whose incident segments were all measured in the
same tile, the net flux $\sum_e s_e Q_e \sim \mathcal N(0, \sum_e Q_e^2 c_e^2 + \sigma_0^2)$ — giving
21S $c_0 \approx 0.054$, $c_p \approx 0.42$, $c_a \approx 1.45$ (4,411 junctions; one $c_0$ for all tiles — per-tile terms did not improve
held-out coverage, `scripts/dc_reconcile.py --tile-terms` reproduces the older model).  $c$ is the modelled
fractional standard deviation of the measurement discrepancy ($\sigma_\epsilon \approx c\,|q|$, ignoring $\sigma_0$): $c = 0.3$
means an error SD of about 30 % of the flux under the working model — not a bound or a confidence interval.  What the
calibration establishes is that $c$ predicts the combined discrepancy at simultaneously imaged junctions, including on held-out
tiles; it does not verify each vessel's own error SD, and an error shared by all vessels at a junction (e.g. a common depth-averaging
bias) cancels there and is invisible to it.  The aggregated $c_{\mathrm{eff}}$ (weighted RMS) summarises the recording-level
discrepancy and is not the standard error of the weighted mean; the reconciliation variance adds the between-recording spread,
$\sigma_k^2=c_{k,\mathrm{eff}}^2Q_k'^2+s_{k,\mathrm{between}}^2+s_0^2$.  At a junction,
$c_{j,\mathrm{eff}}=\sqrt{\sum_k c_k^2y_k^2/\sum_k y_k^2}$: the predicted imbalance SD as a fraction of the root-sum-square
measured flux (Fig 6a axis); the labels trusted ($c<0.5$), weak ($0.5$–$0.8$) and
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
The reconstruction network is the vessel graph minus two kinds of tagged region (see *Graph regions* below): the open
regions beside the dorsal aortae, where blood enters the plexus and optical-flow fluxes are not reliable (removed with the
aortic stubs; the nodes where the plexus meets them are inflow boundary nodes), and topology-uncertain regions (removed as
mass-conserving black boxes).  21S: 8,969 segments (7,713 vessels after contracting short unimaged junction splits,
7,120 imaged); boundary $\mathcal B$ = 38 inflow nodes plus the two vitelline veins; three topology regions (575 edges).
Mass is conserved at every other vertex and over each closed topology region.

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
point.  Sensitivity to the start is reported (`unstable`, `Qp_sd`: four further solves from starts with the data perturbed by
30 %).  21S: median start-to-start SD 1.6 % of the flux (trusted vessels 0.8 %, 90th percentile 5 %); 4.5 % of segments,
mostly small or unimaged and carrying 0.65 % of Σ|Q′|, have an SD larger than their flux (direction not determined by the
data).  The stricter `unstable` flag (any run differing by > 10 % or in direction) marks 33 %.  The boundary
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
| total inflow through the carve boundary (L / R) | 6.32 nL/s (2.27 / 4.05) |
| vitelline-vein outflow V1 / V2 | 2.08 / 4.23 nL/s |
| boundary nodes withdrawing flow | 9 |
| corr(\|Q′\|, \|Q^sim\|), linear / log (7,120 imaged vessels) | 0.918 / 0.771 |
| same, radii shuffled across segments | 0.47 |
| imaged segments whose direction the reconciliation reverses | 4.1 % |
| median relative change of a measured flux | 24 % |

(`results/dc_reconciled_v2.pkl` → `metrics`; passive flow without the venous-pressure penalty.  Paper Fig 7 adds the soft
equal-venous-pressure penalty, `plexusflow.reconcile.venous_pressure_penalty`, σ_P = 2 Pa.)

**Validation.**  Whole-vessel withholding (10 % of measured vessels hidden with all their recordings, 3 repeats; error of
the reconstruction at the hidden vessels against each hidden recording): median ×1.73, direction 79 %.

**Choice of estimator (2026-11-11).**  Starting from the simple reference reconstruction (one recording per vessel chosen by
harmonic coherence, σ = 0.2|y|, convex weighted projection, imposed left/right symmetry) one decision was changed at a time
on the same data and the same held-out vessels:

| step | held-out error | direction | circulation | inflow |
|---|---|---|---|---|
| reference (convex, σ = 0.2\|y\|, symmetric) | ×2.24 | 70.8 % | 9.0 % | 2.03 |
| − symmetry | ×2.28 | 72.2 % | 6.3 % | 2.29 |
| recording chosen by c instead of coherence | ×2.25 | 72.8 % | 5.6 % | 2.36 |
| σ = c\|y\| (calibrated) | ×1.92 | 75.9 % | 2.8 % | 3.74 |
| 1/c²-average of recordings + spread | ×1.90 | 75.2 % | 2.8 % | 3.40 |
| quasi-likelihood (σ at Q′) | ×1.75 | 78.9 % | 4.8 % | 6.45 |
| + topology regions (production) | ×1.73 | 79.0 % | 2.2 % | 6.32 |

(circulation = share of Σ|Q′| removed by cycle cancelling.)  The calibrated reliability is the largest gain.  The
quasi-likelihood is the second; its advantage over the convex projection with the same inputs and regions comes from the
most reliable, clearly flowing held-out vessels (×1.30 vs ×1.62, better in 69 % [65, 72] of paired recordings, vessel
bootstrap) and it loses on the noisiest, smallest readings.  Its cost: weakly measured vessels can be raised well above
their measurement (210 vessels above the 99th percentile of measured flux for their radius; of the 53 with a reliable
recording, 49 agree with it within ×2 and none is reversed; the other 157 are supported by conservation only).  Imposed
left/right symmetry and coherence-based selection do not improve prediction; with no symmetry imposed every variant is
right-dominant (left 36–40 %).  Extent of the open region (none, ±1, ±2 junction steps): inflow 6.43–6.46 nL/s, left 37–38 %,
held-out ×1.73–1.75; only the withdrawing boundary nodes change (fewest with the current carve).

*Code: `plexusflow.reconcile` (driver `scripts/dc_reconcile.py`); figures `scripts/make_figure6.py` (reliability, reconciliation) and `scripts/make_figure7.py` (whole-organ DC, Fig 7).*

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


## Graph regions: open boundary vs topology uncertain

Two edge-tag classes (`plexusflow.reconcile.regions`), set in the workbench viewer (View tab → region tools, *Tag class*), remove
parts of the graph from the reconstruction network with different physics:

| class | tag key | meaning | treatment in `carved_domain` |
|---|---|---|---|
| open boundary | `open_region` (legacy `radial_uncertain`, still read and written) | blood enters / leaves the network here (the high-flux zones beside the dorsal aortae) | cut nodes are free boundary nodes; their exchange is inflow (`inj`, L/R split) |
| topology uncertain | `topology_uncertain` | vessels exist and mass is conserved, but the graph does not represent them (unresolvable marginal band, sheet-like flow, untraceable dense plexus) | each connected region is a mass-conserving black box: cut nodes free (`topo_cut`), one extra conservation row per region (net exchange 0); internal routing unknown |

A topology region touching an open region or an anchor is *leaky*: it has no conservation row and its cut nodes join the
inflow boundary, exactly as if it were tagged open (tested in `tests/test_regions.py`).  Without topology tags the domain is
identical to the open-region carve (checked: Q′ reproduced to 4e-9 nL/s).  `scripts/dc_reconcile.py --no-topology` ignores
topology tags; `scripts/dc_topology_sensitivity.py` compares the field with and without them (grown regions `--dilate`,
leave-one-region-out `--loo`, whole-vessel withholding, `--watch` vessels; map `figures/diagnostics/topology_sensitivity.png`).
Motivation: shunts of 1–4 nL/s through phantom edges in the marginal band (no motion in kymographs, judged non-vessels on
video) that also reversed reliable neighbours (e.g. 50136–50144, four concordant recordings).
