"""Inference recipes for the porous-region carved transmission-line sim.

This package consolidates what used to live as 24 loose scripts in
``plexusflow/scripts/inference/``.  Historically each script duplicated
graph-loading + porous-region identification + outlet templates + α_j
fit + BC construction.  Here, those primitives live in one place.

Modules (some deferred pending future ports):
  - ``state``          — graph loading + porous/outlet identification +
                          cache-keyed-on-tagging (from ``pipeline_state.py``)
  - ``porous``         — reduced-network builders, outlet aggregation
                          (from ``run_sim_carved_outlets.py``)
  - ``templates``      — outlet-template + v-ratio construction
                          (from ``pipeline_outlet_agg.py``, 2026-09-30
                          canonical recipe)
  - ``alpha_fit``      — per-outlet α_j DC LSQ + Green's functions
                          (from ``fit_dc_porous_sources.py``) — deferred

Legacy driver scripts in ``plexusflow/scripts/inference/`` still exist
and can be run directly; they can also ``from plexusflow.inference
import <symbol>`` as they get migrated one at a time.

Related memory:
  - ``sim_intact_two_pass_2026-09-30`` — LOCKED intact sim recipe.
  - ``carve_algorithm`` — full carved-sim recipe end-to-end.
  - ``simulation_bc_recipe_LOCKED`` — two-pass P=0 recipe.
"""
from __future__ import annotations

# Package-level re-exports for convenience.
# Sub-modules are lazily imported to keep startup light for CLI verbs.

__all__ = [
    'PipelineState',
    'find_porous_and_outlets',
    'build_carved_graph',
    'extract_DA_harmonics',
]


def __getattr__(name):
    # Lazy re-exports so `from plexusflow.inference import PipelineState`
    # works without eagerly importing every submodule.
    if name == 'PipelineState':
        from .state import PipelineState
        return PipelineState
    if name == 'find_porous_and_outlets':
        from .porous import find_porous_and_outlets
        return find_porous_and_outlets
    if name == 'build_carved_graph':
        from .porous import build_carved_graph
        return build_carved_graph
    if name == 'extract_DA_harmonics':
        from .porous import extract_DA_harmonics
        return extract_DA_harmonics
    raise AttributeError(f"module 'plexusflow.inference' has no attribute {name!r}")
