"""AC wall-property inference on the carved network (consumes ``plexusflow.ac.field``).

Modules:
  - ``model``       — exact distributed two-port network model, measurement-derived boundary forcing
                       (local measured ratio, ≤ 4 hops), harmonic solves and boundary response matrices
  - ``likelihood``  — von Mises phase NLL and Student-t log-pulsatility NLL, scales profiled
  - ``group_bc``    — joint (D, η) estimate with group-corrected boundary forcing (L carve, R carve, V1, V2:
                       amplitude per harmonic + one coherent timing shift each)

Driver: ``scripts/mech/ac_estimate.py``.
"""
