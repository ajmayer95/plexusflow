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

`plexusflow.ac.speed` · `scripts/speed_pairs.py` → `results/speed_pairs_v1.pkl` (reads the raw movies)
`plexusflow.ac.pulse.build_pulse_field` · `scripts/pulse_field.py` → `results/ac_field_v9.pkl`

1. **Tiles used.** Tiles whose heart rate is within 21 % of the reference tile's (21S: tile 21, 2.69 Hz; tiles 1–6 fall
   outside).
2. **Clock measurements.** A vessel seen in two tiles shows the same heartbeat in both movies. For each such vessel and
   tile pair:
   - the vessel path is mapped into both tiles with the stitching transforms; at every path point (2 px apart) the
     optical-flow velocity (the settings of the flux measurement) is sampled at 7 positions across the vessel
     (± 0.8 radius), projected on the path tangent and averaged;
   - $u_\alpha(t)$ is the average over the path points that lie inside **both** frames (at least 5, i.e. 10 px), so the
     two recordings describe the same piece of vessel;
   - the **speed** $|u_\alpha(t)|$ is fitted with a mean and two harmonics at the tile's heart rate, giving the first
     harmonic $\hat s_{\alpha,1}$ and the variance of the fitted mean $\sigma^2$;
   - $\delta = \arg(\hat s_{\alpha,1}\,\hat s_{\beta,1}^{*})$ estimates $\psi_\alpha-\psi_\beta$, with variance
     $\tau_\alpha^2+\tau_\beta^2$, $\tau^2 = 2\sigma^2/|\hat s_1|^2$.

   No profile is fitted and no flow direction is needed: the magnitude is the same whichever way a recording reads the
   flow. That matters because some recordings read a vessel's flow in the opposite direction; their signed harmonics are
   then half a cycle off, their speed pulses are not. On 21S this happens in 4.4 % of the pairs in which both recordings
   are clearly one-way (and in 24 % of all pairs, most of them slow vessels where the direction is not determined). It is
   not a sign convention (every flux is stored in the edge's fixed orientation), not the stitching (tiles agree to within
   about 8 px) and not the optical flow (the raw image pattern moves the same way). It goes with focus: the rate rises
   from 2 % when both recordings are equally sharp to 21 % when one is much blurrier, and it is concentrated in the
   out-of-focus top strips of a few tiles (16–18, 31).
   A magnitude is a faithful pulse only if the velocity keeps its sign through the beat, so a recording whose pooled
   velocity (smoothed over 5 frames) is on each side of zero for at least 15 % of the frames is left out of the clocks.
3. **Tile clocks.** One offset $\psi_\alpha$ per tile ($\psi_{\rm ref}=0$) by weighted circular least squares over all
   clocks jointly,
   $$\min_\psi \sum_{\rm pairs} \frac{1-\cos(\delta-\psi_\alpha+\psi_\beta)}{\tau_\alpha^2+\tau_\beta^2},$$
   from a spectral start followed by a quasi-Newton minimisation. Clock uncertainty: bootstrap over vessels; it is
   conditional on the pair variances and is not a complete uncertainty of the physiological phase. A tile whose clock SD
   exceeds 30° is not synchronised: its recordings stay in the mean flow but are left out of the pulsatile field.
4. **Average.** The waveforms come from the signed flux, not from the speed. Of a vessel's flux recordings, those whose
   mean flow has the direction of the reconciled mean flow are averaged with weights $1/\sigma^2$, harmonic $n$ rotated
   by $e^{-in\psi_\alpha}$ (a pure time shift). A recording of the opposite direction is not used and is never
   sign-flipped. Such recordings are mostly poorly focused ones; a real reversal of flow between recordings cannot be
   excluded. Vessels with recordings of both
   directions are flagged as temporally inconsistent. The mean-flow reconciliation (step 2 of this document) still uses
   every recording.
5. **Rescale.** The averaged waveform is rescaled so that its mean equals the reconciled mean. A vessel is not
   reconstructed when none of its recordings agrees with the reconciled direction.

The field is a composite of recordings made at different times, not a simultaneous observation, and comparing speeds
does not establish the phase of the signed flow in a vessel whose direction changes between recordings.

21S: 15,018 clock measurements on 6,882 vessels (all tiles, network and open-region vessels alike), 5,726 of them between
retained tiles and without a reversing recording; all 47 clocks synchronised (largest clock SD 25°, tile 53); 83 % of the
flux recordings agree with the reconciled direction; 5,415 vessels reconstructed, 902 of them flagged as temporally
inconsistent; 588 vessels have no recording in the reconciled direction.

**Sensitivity.** `scripts/pulse_field.py --previous` → `ac_field_v8.pkl` is the earlier method: clocks from the first
harmonics of the fluxes of repeated network vessels plus two dorsal-aorta segments outside the network (oriented to their
majority direction), every recording averaged, rescaled by the combined mean of all measurements with negative ratios
dropped. Its clocks differ for the tiles of the left plexus (ten tiles by about +28°, tiles 16–18 and 31 by about −4°).

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

21S: 5,232 vessels fitted; $\hat D$ = 4.1 × 10⁻⁴ /Pa (4.3 with the earlier method).

## Figures

`scripts/make_figure6.py` (clock synchronisation, phase field) and `scripts/make_figure7.py` (reconstructed field
against the RC network) read the result files above; see [FIGURES.md](../FIGURES.md).

## Earlier methods in the code

`plexusflow.reconcile` still contains the earlier mean-flow estimators (quasi-likelihood with a fitted reliability
model; heartbeat-block variances) and `plexusflow.ac.field` the earlier synchronisation. No script in the paper chain
uses them; their drivers, validation runs and figure scripts were removed on 2026-11-23 and are in the git history.
