# Method: from per-tile recordings to one flow field and a network distensibility

This is the method of the paper as the code runs it (since 2026-11-20). Each step names the module and the script that
implements it. Numbers are for the 21-somite embryo (21S).

Input: the measured vessel graph. Every vessel carries one flux time series $Q^{\rm obs}_{e,\alpha}(t)$ per tile
$\alpha$ that recorded it (250 frames/s; flux from parabolic velocity-profile fits, see [pipeline.md](pipeline.md)).
Tiles are separate movies, taken one after another, so each has its own clock and its own heart rate $f_\alpha$.

## 1. Fit per recording

`plexusflow.reconcile.dc.residual_variance`, `fitted_mean`

Each recording with a valid profile fit is fitted by least squares with a mean and two harmonics of its tile's heart
rate:

$$Q^{\rm obs}_{e,\alpha}(t) \approx \hat Q_0 + \sum_{n=1}^{2} {\rm Re}\,[\hat Q_n e^{2\pi i n f_\alpha t}].$$

The third harmonic carries about 0.1 % of the power and is not used (Fig 3d). One uncertainty is kept per recording:
the variance of the fitted mean, $\sigma^2 = s^2/M$ with $s^2 = {\rm RSS}/(M-5)$ the residual variance of the fit over
$M$ frames. Every weight below is $1/\sigma^2$.

## 2. Mean flow

`plexusflow.reconcile.dc.reconcile_dc` · `scripts/dc_reconcile.py` → `results/dc_reconciled_v3.pkl`

1. Per vessel, the $1/\sigma^2$-weighted average of the fitted means of all its recordings (all tiles).
2. Weighted least-squares projection of these averages onto the flows that conserve mass at every interior junction of
   the network, with total left arterial inflow equal to total right. Unmeasured vessels enter with negligible weight
   and are determined by conservation.

The network is the effective vascular domain of Fig 5 (`plexusflow.reconcile.carved_domain`): the plexus with the
dorsal aorta carved away, 33 arterial inflow nodes and 2 vitelline-vein outlets. 21S: 8,467 vessels, 7,009 measured;
inflow 5.74 nL/s.

## 3. Pulsatile flow

`plexusflow.ac.pulse.build_pulse_field` · `scripts/pulse_field.py` → `results/ac_field_v9.pkl`

1. **Tiles used.** Tiles whose heart rate is within 21 % of the reference tile's (21S: tile 21, 2.69 Hz; tiles 1–6 fall
   outside).
2. **Clock measurements.** Every vessel of the graph recorded in two tiles gives one measurement per tile pair, whether
   it belongs to the network or to the open region beside the aortae. Each recording is taken in its own mean-flow
   direction, $\varphi = \arg({\rm sgn}(\hat Q_0)\,\hat Q_1)$: a recording that reads the flow the other way has its mean
   and its pulse negated together, so its signed first harmonic would be half a cycle off while $\varphi$ is not. The
   measurement is $\varphi_\alpha - \varphi_\beta$, with variance $\tau_\alpha^2+\tau_\beta^2$,
   $\tau^2 = 2\sigma^2/|\hat Q_1|^2$.
3. **Tile clocks.** One offset $\psi_\alpha$ per tile ($\psi_{\rm ref}=0$) by weighted circular least squares over all
   clocks jointly,
   $$\min_\psi \sum_{\rm pairs} \frac{1-\cos((\varphi_\alpha-\psi_\alpha)-(\varphi_\beta-\psi_\beta))}{\tau_\alpha^2+\tau_\beta^2},$$
   from a spectral start followed by a quasi-Newton minimisation. Clock uncertainty: bootstrap over vessels; it is
   conditional on the pair variances and is not a complete uncertainty of the physiological phase. A tile whose clock SD
   exceeds 30° is not synchronised: its recordings stay in the mean flow but are left out of the pulsatile field.

   Two things make this work on 21S. Taking each recording in its own direction removes half-cycle conflicts: two
   recordings of a vessel disagree in direction in about 20 % of pairs, mostly where the direction is not resolved (with
   r = |mean flux| / residual SD of one frame, 47 % for r < 0.1, 10 % at r ≈ 1, 0 % for r > 3). Including the open-region
   vessels ties the left-plexus tiles (20, 29) to tile 21 through vessels that carry a strong pulse.
4. **Average.** All flux recordings of a vessel in
   synchronised tiles are averaged with weights $1/\sigma^2$, in the vessel's fixed orientation, harmonic $n$ rotated by
   $e^{-in\psi_\alpha}$ (a pure time shift). No recording is selected or sign-flipped by its direction.
5. **Rescale.** The averaged waveform is multiplied by the reconciled mean over the averaged measured mean of all the
   vessel's recordings, $\bar Q^*_e/\bar Q^{\rm av}_e$. A vessel is not reconstructed where this ratio is negative
   (reconciled and measured means of opposite direction) or undefined. Vessels with recordings of both directions are
   flagged as temporally inconsistent.

The field is a composite of recordings made at different times, not a simultaneous observation.

21S: 8,357 clock measurements between retained tiles; 46 of 47 clocks synchronised (tile 53, at the lower rim, has a clock SD
of 77° and is left out; the next largest is 15°).
5,204 vessels reconstructed, 679 of them flagged as temporally inconsistent; 786 vessels are not reconstructed because their
reconciled and measured means have opposite signs, and 35 because they have no usable mean.

**Sensitivity to the averaging rule** (measured with an earlier, equivalent set of clocks). Averaging only the recordings
whose mean flow has the reconciled direction (`pulse_field.py --agree`) changed $\hat D$ by 0.5 % and added about 200
vessels; that rule selects recordings by a direction they helped to determine, so it is not used. Averaging every
recording and dropping nothing changed $\hat D$ by 1 %, with about 3 % of vessels half a cycle off.

The weights $1/\sigma^2$ are relative weights from the temporal scatter of each recording. They are not standard errors
of the mean: judged by how often two recordings of a vessel disagree in direction, the error of a recording's mean is
about 0.7 times its per-frame residual SD, some twenty times $s/\sqrt M$.

**Sensitivity to the clocks.** `scripts/pulse_field.py --previous` → `ac_field_v8.pkl` gives the earlier clocks: signed
first harmonics of repeated network vessels plus two dorsal-aorta segments outside the network (oriented to their majority
direction). They differ for the tiles of the left plexus: ten tiles by about +31° and tiles 16–18 and 31 by about −10°. A
third route, the speed of the optical-flow velocity pooled over the piece of vessel common to two tiles (computed from the
raw movies), gave the same clocks as the ones in use to a median of 2° (90 % of tiles within 5°).

## 4. Distensibility

`plexusmech.ac` · `scripts/mech/ac_estimate.py` → `results/ac_estimate_v10.json`

The network is modelled as a distributed RC line per vessel (Poiseuille resistance, wall compliance set by one areal
distensibility $D$).

- **Forcing, as measured, not fitted.** At each arterial node: the reconciled mean boundary flux times the
  flux-weighted mean waveform of reconstructed vessels within 4 junctions. At the two vein outlets: one shared
  measured waveform.
- **Loss.** For each fitted vessel, the model is compared with the reconstructed field in three terms: first-harmonic
  phase and second-harmonic phase (von Mises) and first-harmonic amplitude relative to the mean (Student-t, 4 degrees
  of freedom, on $\log_{10}$). Per-vessel variance = variance of the averaged coefficient + one common variance fitted
  per term.
- **Estimate.** The loss is evaluated on 21 values of $D$ between 2.5 and 8 × 10⁻⁴ /Pa; $\hat D$ is the minimum of
  a parabola through the lowest points.

21S: 4,961 vessels fitted; $\hat D$ = 4.0 × 10⁻⁴ /Pa (4.02; 4.28 with the earlier clocks).

## Figures

`scripts/make_figure6.py` (clock synchronisation, phase field) and `scripts/make_figure7.py` (reconstructed field
against the RC network) read the result files above; see [FIGURES.md](../FIGURES.md).

## Earlier methods in the code

`plexusflow.reconcile` still contains the earlier mean-flow estimators (quasi-likelihood with a fitted reliability
model; heartbeat-block variances) and `plexusflow.ac.field` the earlier synchronisation. No script in the paper chain
uses them; their drivers, validation runs and figure scripts were removed on 2026-11-23 and are in the git history.
