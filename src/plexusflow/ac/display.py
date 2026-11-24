"""Display conventions of the paper's pulsatile maps (Fig 7), shared by figures and the viewer.

Phase reference: the ρ-weighted H1 (H2) phase of the vitelline-vein stub recordings of tile 14 (one clock), taken into
the field's common clock (ψ_14 and an optional fitted tile correction), each recording in its own mean-flow direction.
Phase lag of a vessel: reference minus its harmonic taken in its own flow direction, in degrees —
  H1 wrapped to [−60°, 300°) (the wrap falls in the empty part of the distribution), drawn 0–220° on plasma;
  H2 cyclic, [0°, 360°), drawn on a cyclic map (the H2 lag passes a full cycle across the network).
Relative pulsatile power (|H1|² + |H2|²) / Q̄², log scale 0.05–0.6 on magma.  Mean flow |Q|, log scale on viridis.
"""
from __future__ import annotations

import numpy as np

REF_TILE = 14
H2_CYCLE = ['#0d0887', '#9c179e', '#ed7953', '#f0f921', '#1b9e77', '#0d0887']
PHASE_LAG = {1: dict(wrap_lo=-60.0, vmin=0.0, vmax=220.0, ticks=(0, 90, 180), cmap="plasma"),
             2: dict(wrap_lo=0.0, vmin=0.0, vmax=360.0, ticks=(0, 180, 360), cmap="h2cycle")}
PULSE_POWER = dict(vmin=0.05, vmax=0.6, ticks=(0.05, 0.2, 0.6), cmap="magma")


def vein_phase_reference(G, field, dom, tile: int = REF_TILE, gauge_correction_deg=None) -> dict:
    """{1: φ_ref,H1, 2: φ_ref,H2} (rad) in the field's clock from the vitelline-vein stub recordings of `tile`."""
    from ..reconcile.harmonics import phasors, coherence
    tf0 = G.graph["tile_f0_piv"]; psi = float(field["psi"][tile])
    d = np.radians((gauge_correction_deg or {}).get(str(tile), 0.0))
    z = {1: 0j, 2: 0j}
    for n in list(dom["V"]):
        for e in G.edges(n):
            for m in G.edges[e].get("measurements_piv") or []:
                if int(m["tile_id"]) == tile and m.get("Q_t") is not None and (m.get("quality") or {}).get("profile_ok"):
                    a0, H = phasors(m["Q_t"], tf0[tile]); w = max(coherence(m["Q_t"], tf0[tile]), 0)
                    for k in (1, 2):
                        z[k] += w * np.exp(1j * np.angle(np.sign(a0) * H[k - 1] * np.exp(-1j * k * (psi + d))))
    return {k: float(np.angle(z[k])) for k in (1, 2)}


def phase_lag_deg(H, dc, k: int, phref: dict):
    """Lag (deg) of harmonic k behind the reference, each vessel in its own mean-flow direction (sign of dc)."""
    lo = PHASE_LAG[k]["wrap_lo"]
    return (np.degrees(np.angle(np.exp(1j * phref[k]) * np.conj(np.asarray(H) * np.sign(dc)))) - lo) % 360 + lo


def pulse_power(H1, H2, dc):
    return (np.abs(H1) ** 2 + np.abs(H2) ** 2) / np.maximum(np.abs(dc), 1e-30) ** 2


def cmap(name: str):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    if name == "h2cycle":
        return mcolors.LinearSegmentedColormap.from_list("h2cycle", H2_CYCLE, N=512)
    return plt.get_cmap(name)
