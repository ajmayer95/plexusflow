"""Per-tile preprocessing (raw folder → registration or measurement input).

Two output flavours live here, keeping them explicitly separate:

* ``register_prep`` — heavily-transformed image intended ONLY for stitching
  registration (SIFT features).  CLAHE + blur + 8-bit are appropriate here
  because the transforms recovered by SIFT are then applied to
  ``measure_prep`` output.

* ``measure_prep`` — lightly-processed image intended for per-tile
  segmentation and downstream quantitative work.  No CLAHE, no contrast
  stretching, minimal transformation.  Pixel values remain interpretable.

The two share the same raw-loading and projection stages.
"""
from __future__ import annotations

from .load    import load_raw_tile, load_frame_range
from .project import temporal_median, temporal_projections, ProjectionSet
from .projections import (
    PROJECTIONS,
    preprocess_tiles,
    proj_max_median3,
    proj_bandpass_rms,
    proj_std,
    proj_highpass_std,
    proj_std_clahe,
)

# register_prep referenced in an older scaffold docstring but the module
# hasn't landed yet. Import lazily / conditionally to avoid failing here.
try:
    from .register_prep import RegisterPrepParams, register_prep, estimate_shared_bias  # noqa
    _HAVE_REGISTER_PREP = True
except ImportError:
    _HAVE_REGISTER_PREP = False

__all__ = [
    "load_raw_tile",
    "load_frame_range",
    "temporal_median",
    "temporal_projections",
    "ProjectionSet",
    # Batch preprocess API (raw → per-tile projections in stage dir)
    "PROJECTIONS",
    "preprocess_tiles",
    "proj_max_median3",
    "proj_bandpass_rms",
    "proj_std",
    "proj_highpass_std",
    "proj_std_clahe",
]
if _HAVE_REGISTER_PREP:
    __all__ += ["RegisterPrepParams", "register_prep", "estimate_shared_bias"]
