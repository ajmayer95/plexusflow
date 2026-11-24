# Known-truth validation of the per-vessel flux estimator (2026-10-09)

The question: how accurate is the production per-vessel measurement — Farnebäck optical flow → profile-fit per-slice
Poiseuille flux — and does the DC reliability scale *c* describe its real errors?  Same-tile junction balance (used to fit
*c*) cannot see errors shared by all vessels at a junction, so this needs ground truth.  We built synthetic videos with
exactly known, exactly conserving flow and ran them through the unchanged production code.

## Method

**Geometry.**  Real Y-junctions from the graph: the three edges' centrelines (tile pixel frame, lightly smoothed) and radii,
swept into circular 3D tubes and fused with a ball at the node (gmsh / OpenCASCADE), plus straight 4–5R extensions so the
imposed inflow/outflow profiles sit away from the measured segments.  Single vessels for controlled sweeps: elliptical
cross-sections (in-plane semi-axis *a*, depth *b* = ε·*a*) lofted along straight, arc, S-bend and tapered centrelines.

**Flow.**  Steady Stokes (scikit-fem, Taylor–Hood P2–P1).  No-slip walls; fully developed Poiseuille velocity prescribed on
every cut plane, fluxes summing to zero, so Kirchhoff holds exactly by construction (checked: cap fluxes by facet quadrature
to 3e-5 of Q0; Gauss-quadrature sections along each segment conserve to 2e-4).  Two basis solutions per junction give any
split α (linearity).  Pulsatile flow is quasi-steady scaling, u(x,t) = Q0·s(t)·û(x), valid since the Womersley number is
0.03–0.1 (≤ 0.5 worst case) for these vessels at 3 Hz; s(t) = 1 + 0.4 cos ωt + 0.12 cos(2ωt + 0.375π).

**Tracers and images.**  Tracers seeded uniformly in the lumen and injected at inflow caps at a rate ∝ Q(t), advected on the
exact P2 field (containing tetrahedron + basis evaluation; pseudo-time integration along streamlines).  Each frame renders
tracers as Gaussians whose width and peak brightness change with depth, stretched along the 4 ms in-frame displacement
(exposure = frame interval, from the camera header).  Videos are **hybrids**: the tile's own static background (per-pixel
10th percentile of the real clip: tissue texture, stuck puncta) + scaled synthetic tracers + camera noise from a tile-wide
photon-transfer fit (variance vs mean).  The pipeline receives the **segmented** radius and mask, as on real data.

**Calibrations from the real videos.**

| Quantity | How | Result |
|---|---|---|
| True lumen width | temporal-SD band width of 34 clean real vessels matched by synthetics of known radius | R_true ≈ 1.2 × R_seg (1.35 for R_seg 6–8 px, 1.2 above) |
| Depth of correlation | shape of the same SD profiles | not identifiable; results insensitive (z_c 4–16 px changes Q bias by ~4 %); depth-saturated z_c = 16 px used |
| Tracer density | resolvable moving tracer images in real lumens | 1.2 / 4.7 / 19 per 8² / 16² / 32² px window; synthetic matched (0.033 /px³) |
| Camera noise | photon-transfer lower envelope, per tile | gain ≈ 30 counts⁻¹, shot-noise limited |

## Results

**Junction ensemble** — 27 real junctions (26 tiles) × 5 splits × 5 speeds × 2 seeds; 4080 edge measurements:

| True branch speed (px/frame) | n | Q̂/Q_true (median) | robust scatter (log) | profile-fit fail |
|---|---|---|---|---|
| < 0.05 | 281 | 0.61 | 0.56 | 38 % |
| 0.05–0.1 | 620 | 0.78 | 0.22 | 17 % |
| 0.1–0.2 | 768 | 0.77 | 0.21 | 7 % |
| 0.2–0.3 | 549 | 0.74 | 0.20 | 6 % |
| 0.3–0.5 | 601 | 0.68 | 0.23 | 3 % |
| 0.5–0.9 | 533 | 0.60 | 0.28 | 4 % |
| > 0.9 | 191 | 0.46 | 0.54 | 10 % |

Overall ×0.71, scatter 0.28, sign errors 1.3 %.  Size matters little (×0.68–0.79 over R_seg 4–17 px).  Isolated long vessels
read ×0.8–1.0, so roughly 20 % of the under-read comes from short segments near branch points (skewed, off-centre profiles;
×0.65 for L < 2R vs ×0.75 for 3–4R).  Synthetic junction imbalance (median 0.06–0.09) is about half the real one (0.12–0.16):
real data have error sources the synthetics lack (flowing neighbours, tissue motion, tracer heterogeneity).

**The reliability scale *c*** (fitted on real same-tile junctions) tracks the true error: binned robust scatter of
log(Q̂/Q) rises 0.24 → 0.30 → 0.43 → 0.68 over c < 0.3, 0.3–0.5, 0.5–0.8, ≥ 0.8.  An error model fitted directly on the
synthetic truth has the same form, log|error| = −1.82 + 1.09(1−f) + 2.83(1−align) + 0.36 log v, alignment dominant —
independent support for how *c* is constructed.

**Flow-estimator comparison** (6 junctions × 3 speeds; same downstream profile-fit code for every method):

| Method | Q̂/Q at v 0.1 / 0.3 / 0.9 | scatter | profile-fit fail | H1 amplitude ratio / phase error |
|---|---|---|---|---|
| Farnebäck, lag 1 (production) | 0.64 / 0.77 / 0.57 | 0.25–0.48 | 0–22 % | 0.93–0.99 / 3–11° |
| Farnebäck, lag 2 | 0.84 / 0.75 / 0.58 | 0.24–0.40 | 0–6 % | 0.83–0.96 / 3–5° |
| OpenPIV multipass window deformation (32-16-8 or 16-8, lags 1–8) | 0.03–0.51 | 0.3–2.2 | 17–56 % | 22–34 % vectors replaced |
| Phase-binned ensemble PIV, 16 px (lags 1–8) | 1.0–1.55 | 0.08–0.21 (v ≤ 0.3) | 33–50 % | 0.83–1.09 / 5–19° |
| Phase-binned ensemble PIV, 32 px | — | — | 78–94 % | window ≥ vessel |

Instantaneous correlation PIV fails at this tracer density (≈ 1 tracer per 8 px window).  Phase-binned ensemble PIV has the
lowest scatter but over-reads (window smoothing inflates the fitted radius) and often fails the profile-fit gates — both
profile-fit problems, not velocity problems.  Farnebäck at lag 2 helps slow vessels.

**Cross-section shape** (single vessels, 48 geometries × 3 speeds × 2 seeds).  Flattening (depth/width ε) over-reads by
≈ 1/ε × the circular factor, with no failures or quality-metric warning:

| ε | v 0.1 | v 0.3 | v 0.9 |
|---|---|---|---|
| 1.0 | 0.96 | 0.92 | 0.81 |
| 0.8 | 1.15 | 1.16 | 1.06 |
| 0.6 | 1.53 | 1.44 | 1.28 |
| 0.4 | 2.31 | 2.19 | 1.81 |

Curvature (R_curv = 4a), S-bends and taper change circular-vessel results by ≤ ±10 %.

**Is flattening size-dependent?**  Tracer-fluctuation variance at vessel centres (206 vessels, 16 tiles, normalised by
circular synthetics) falls with width (γ = d log(depth/width)/d log(width) = −0.56 to −1.08 depending on the assumed depth
of correlation).  But junction conservation does not support a matching flux correction Q ← Q̂·(R/8)^γ: real junctions
balance best at γ ≈ −0.1 to −0.2 (95 % CI −0.4 to 0.0), the same as a synthetic no-bias null (−0.1 to −0.3), and γ = −1
clearly worsens balance.  The variance trend is therefore more likely concentration, focus or lumen-calibration effects than
size-dependent flux bias; a size-dependent correction is bounded to |γ| ≲ 0.4 and not applied.

## Conclusions

1. **Relative flows are robust.**  The DC reconciliation needs no size-dependent correction; the speed-dependent under-read
   (×0.78 → ×0.60 from 0.1 to 0.9 px/frame) distorts slow vs fast vessels by ~20–25 % at most.
2. ***c* is validated** in structure (alignment ≫ kept fraction) and in rank against true error.
3. **Absolute scale is uncertain by a common factor.**  The speed/proximity under-read (≈ ×0.7–0.8) and a possible uniform
   flattening over-read (1/ε, unknown without depth information) act in opposite directions: true flows are likely
   ×0.8–1.4 of the reported values (×1.35 if vessels are circular).  Ratios (splits, pulsatility index, phases) and the D, η
   inference do not depend on this factor.
4. **Correlation PIV is not competitive** with Farnebäck at this tracer density; phase-binned ensemble PIV is the only
   variant worth revisiting, with a flux estimator suited to smoothed profiles.

## Caveats

One tracer/optics model (depth-saturated, 'H'-type tracers matched in density and temporal SD); rigid junctions with
in-phase pulsatility in all branches; prescribed-flux boundary conditions; 27 junctions (filters required all three edges
in one tile).  The depth-to-width ratio of real vessels is not measurable from these 2D videos.

## Reproducing

The benchmark is part of the package (`plexusflow.synthetic.stokes`) and runs from a **benchmark bundle**
(`PlexusFlow21S_benchmark_v1`, ~100 MB): per-tile full-resolution backgrounds (per-pixel 10th percentile and temporal SD
of the first 400 / 600 raw frames) with each tile's camera noise model, and the saved Stokes solutions of the 27 junctions
and 51 single vessels.  No raw videos or mesher are needed; runs from the bundle reproduce runs from the raw data bit for bit.

```bash
pip install -e ".[bench]"
python scripts/benchmark/vessels.py       --bundle PlexusFlow21S_benchmark_v1    # single vessels: shape, size, flattening
python scripts/benchmark/frame_spacing.py --bundle PlexusFlow21S_benchmark_v1    # OF frame spacing, 15S-matched conditions
python scripts/benchmark/junctions.py     --bundle PlexusFlow21S_benchmark_v1    # junction ensemble (≈ 1 h on 8 cores)
```

`scripts/benchmark/make_bundle.py` builds the bundle from a stage folder and its raw videos (`--rebuild` also re-picks the
junctions and re-meshes / re-solves the geometry with gmsh + scikit-fem).  The PIV comparison and the lumen-width,
depth-of-correlation and tracer-density calibrations were exploratory analyses and are not part of this repository
(available from the author on request).
