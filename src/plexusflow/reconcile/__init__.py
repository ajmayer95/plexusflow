"""DC reconciliation of per-tile vessel flows (method decided 2026-10-07).

Modules:
  - ``harmonics``    — per-trace DC + harmonic phasors, harmonic coherence ρ, per-beat H1 scatter
  - ``reliability``  — measurement table, same-tile junctions, reliability scale
                        c = c0 + c_f (1 − f) + c_a (1 − align), fitted by ML on junction balance (per-tile c_T optional)
  - ``network``      — carved domain, conservative projection (B_I Q = 0), emergent boundary, passive flow
  - ``dc``           — the full DC reconciliation (winner = min c, reliability-weighted damped IRLS)

Driver: ``scripts/dc_reconcile.py``.
"""
from .reliability import measurement_table, same_tile_junctions, fit_reliability, c_of
from .network import carved_domain, project, emergent, passive, metrics
from .dc import reconcile_dc

__all__ = ["measurement_table", "same_tile_junctions", "fit_reliability", "c_of",
           "carved_domain", "project", "emergent", "passive", "metrics", "reconcile_dc"]
