"""AC wall-property inference on the carved network (consumes ``plexusflow.ac.field``).

Modules:
  - ``model``       — exact distributed two-port network model, measurement-derived boundary forcing
                       (local measured ratio, ≤ 4 hops; vein stub waveforms), harmonic solves, boundary group responses
  - ``likelihood``  — von Mises phase NLL and Student-t log-pulsatility NLL, scales profiled; variance-based
                       versions (κ from the circular variance)
  - ``group_bc``    — boundary groups (L/R carve, L/R veins) and the earlier joint (D, η) grid estimate
  - ``profile``     — the current fit: profile likelihood of D (η optional, default 0) with L/R carve corrections,
                       veins pinned to the stub waveforms (model.vein_forcing), a latent tile gauge correction and
                       waveform-based weights (per-beat repeatability + (τ·c)²)

Driver: ``scripts/mech/ac_estimate.py``.
"""
