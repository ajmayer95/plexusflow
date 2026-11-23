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

`plexusflow.ac.pulse.build_pulse_field` · `scripts/pulse_field.py` → `results/ac_field_v8.pkl`

1. **Tiles used.** Tiles whose heart rate is within 21 % of the reference tile's (21S: tile 21, 2.69 Hz; tiles 1–6 fall
   outside).
2. **Tile clocks.** One phase offset $\psi_\alpha$ per tile ($\psi_{\rm ref}=0$), from vessels recorded in more than
   one tile, first harmonic only:
   $$\min_\psi \sum_e \sum_{\alpha<\beta} w_{\alpha\beta}\,[1-\cos((\varphi_\alpha-\psi_\alpha)-(\varphi_\beta-\psi_\beta))],
   \qquad w_{\alpha\beta} = \frac{1}{\tau_\alpha^2+\tau_\beta^2},\quad \tau^2 = \frac{2\sigma^2}{|\hat Q_1|^2},$$
   with $\varphi = \arg \hat Q_1$ in the vessel's fixed orientation. Start from a spectral solution, then minimise
   over all clocks jointly (a tile-by-tile update cannot move a weakly linked block of tiles).
3. **Average.** Harmonic $n$ of each recording is rotated by $e^{-in\psi_\alpha}$ (a pure time shift) and the
   recordings of a vessel are averaged with weights $1/\sigma^2$.
4. **Rescale.** The averaged waveform is multiplied by the reconciled mean over the averaged measured mean,
   $\bar Q^*_e/\bar Q^{\rm av}_e$. A vessel is not reconstructed where this ratio is negative or undefined.

21S: 47 tile clocks; 5,212 vessels reconstructed.

**Provisional.** Two dorsal-aorta segments just outside the network take part in the clock fit only. The aorta is the
one strongly pulsing vessel recorded on both sides of the seam between the left block of tiles and the rest; the plexus
vessels alone fix that block's clock only to about ± 30°. One aorta recording (tile 20) reads the flow backwards, so
the recordings of these segments are oriented to the segment's weighted majority direction before the fit. The proper
fix is to re-measure that recording.

An optional rule (`--direction-rule`, off) drops recordings whose mean-flow direction disagrees with their vessel's
weighted majority.

## 4. Distensibility

`plexusmech.ac` · `scripts/mech/ac_estimate.py` → `results/ac_estimate_v9.json`

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

21S: 4,969 vessels fitted; $\hat D$ = 4.3 × 10⁻⁴ /Pa.

## Figures

`scripts/make_figure6.py` (clock synchronisation, phase field) and `scripts/make_figure7.py` (reconstructed field
against the RC network) read the three result files above; see [FIGURES.md](../FIGURES.md).

## Earlier methods in the code

`plexusflow.reconcile` still contains the earlier mean-flow estimators (quasi-likelihood with a fitted reliability
model; heartbeat-block variances) and `plexusflow.ac.field` the earlier synchronisation. No script in the paper chain
uses them; their drivers, validation runs and figure scripts were removed on 2026-11-23 and are in the git history.
