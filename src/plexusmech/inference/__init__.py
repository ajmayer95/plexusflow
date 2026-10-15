"""Legacy carved-simulation inference state (2026-09 pipeline; reads the legacy normalized fields).

Superseded for new work by ``plexusmech.ac`` on top of the reconciled field.
"""
from __future__ import annotations

__all__ = ["PipelineState", "extract_DA_harmonics", "phi_shift_per_outlet"]


def __getattr__(name):
    # Lazy re-exports so importing the package stays light.
    if name == "PipelineState":
        from .state import PipelineState
        return PipelineState
    if name in ("extract_DA_harmonics", "phi_shift_per_outlet"):
        from . import porous_bc
        return getattr(porous_bc, name)
    raise AttributeError(f"module 'plexusmech.inference' has no attribute {name!r}")
