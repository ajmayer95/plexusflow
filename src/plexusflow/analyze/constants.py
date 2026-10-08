"""Physical constants for flow analysis.

Intentionally small — just what the analyze pipeline needs.  These
are the calibration values for the 21-somite HH-stage yolk-sac imaging
setup (10× air objective, 250 fps).  Override by monkey-patching this
module or by passing the value explicitly where a function accepts it.
"""
from __future__ import annotations

# ── Imaging calibration ─────────────────────────────────────────────
PX_SIZE_UM: float = 1.7          # pixel size in µm (10× objective)
FRAME_DT_S: float = 0.004        # inter-frame period in s (250 fps)

# ── Harmonic fit ────────────────────────────────────────────────────
N_HARMONICS: int = 3             # number of harmonics of the heart rate
                                  # to include in the Q(t) fit

# ── Fluid ────────────────────────────────────────────────────────────
MU_DEFAULT: float = 3.5e-3       # embryonic-blood dynamic viscosity (Pa·s)
