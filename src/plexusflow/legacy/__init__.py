"""LEGACY (2026-09) normalization path — kept for reproducing earlier figures, not used by the pipeline or viewer.

  - ``tile_normalize``  — per-tile CO_rel + complex gauges, cross-tile aggregation into ``_h_*_norm`` fields
  - ``tile_sync_u1``    — spectral U(1) tile synchronisation used by tile_normalize
  - ``kcl_reconstruct`` — SNR-gated KCL reconstruction of boundary harmonics
  - ``dc_conservative_projection`` — first conservative DC projection (2026-10-05)

Superseded by ``plexusflow.reconcile`` (DC: reliability-weighted conservative projection) and
``plexusflow.ac.field`` (tile time gauge by weighted angular synchronisation, coherent n·ψ).  Driver for the
legacy path: ``plexusflow legacy-normalize``.
"""
