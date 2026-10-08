"""AC (pulsatile) field and wall-property inference on the carved network.

Modules:
  - ``field``       — Option-A harmonic field: tile time gauge ψ by angular synchronisation (coherent, k·ψ),
                       AC winner per edge, Ĥ_k = (|Q'|/|a0|) e^{−ikψ} H_k,raw.  No conservation imposed.
  - ``model``       — exact distributed two-port network model, measurement-derived boundary forcing
                       (local measured ratio, ≤ 4 hops), harmonic solves and boundary response matrices
  - ``likelihood``  — von Mises phase NLL and Student-t log-pulsatility NLL, scales profiled
  - ``group_bc``    — joint (D, η) estimate with group-corrected boundary forcing (L carve, R carve, V1, V2:
                       amplitude per harmonic + one coherent timing shift each)

Drivers: ``scripts/ac_field.py``, ``scripts/ac_estimate.py``.
"""
