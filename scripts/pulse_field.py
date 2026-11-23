"""Reconciled pulsatile field on the carved network (plexusflow.ac.pulse; the method since 2026-11-20).

Needs results/dc_reconciled_v3.pkl (scripts/dc_reconcile.py).  Each recording is fitted with its mean and two harmonics and
weighted by 1/σ² (σ² from the fit residual); the tile clocks are fitted jointly on repeatedly recorded network vessels;
recordings are averaged and the waveform is rescaled to the reconciled mean flow.  Writes results/ac_field_v8.pkl (the input of scripts/mech/ac_estimate.py).
The previous synchronisation (seam continuity, dorsal-aorta anchor, one recording per vessel) is scripts/ac_field.py → ac_field_v7.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pickle
import numpy as np
import plexusflow.paths as paths
from plexusflow.ac.field import REF_TILE, CANONICAL_F0_BAND, supported
from plexusflow.ac.pulse import build_pulse_field, FIELD
from plexusflow.reconcile.dc import RESULT


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-tile", type=int, default=None, help=f"reference tile for the heart-rate band and the clock origin (default: plexusflow.yaml ac_ref_tile, else {REF_TILE} for 21S)")
    ap.add_argument("--f0-band", type=float, default=None, help=f"keep tiles whose heart rate is within ± this fraction of the reference tile's (default: plexusflow.yaml ac_f0_band, else {CANONICAL_F0_BAND})")
    ap.add_argument("--direction-rule", action="store_true", help="drop recordings whose mean-flow direction disagrees with their vessel's 1/σ²-weighted "
                                                               "majority (off by default; not in the Methods text)")
    ap.add_argument("--dc", default=RESULT, help=f"results/<dc>.pkl (default {RESULT})")
    ap.add_argument("--out", default=FIELD, help=f"results/<out>.pkl (default {FIELD})")
    a = ap.parse_args()
    from plexusflow.config import StageConfig
    cfg = StageConfig.load(paths.STAGE)
    ref = a.ref_tile if a.ref_tile is not None else cfg.ac_ref_tile
    if ref is None:
        if paths.STAGE.name.startswith("PlexusFlow21S"):
            ref = REF_TILE
        else:
            sys.exit(f"{paths.STAGE.name}: no reference tile. Set ac_ref_tile in plexusflow.yaml or pass --ref-tile.")
    band = a.f0_band if a.f0_band is not None else (cfg.ac_f0_band if cfg.ac_f0_band is not None else CANONICAL_F0_BAND)
    G = paths.load_graph(); D = pickle.loads((paths.RESULTS / f"{a.dc}.pkl").read_bytes())
    A = build_pulse_field(G, D, ref_tile=ref, f0_band=band, direction_filter=a.direction_rule); A["f0_band"] = band; A["dc"] = a.dc
    el = A["elig"]; lag = np.degrees(np.angle(A["H"][0][el] * np.sign(A["Qp"][el])))
    print(f"reference tile {ref} ({A['f_ref']:.2f} Hz), heart-rate band ±{band:.0%}: {len(A['band'])} tiles, {len(A['psi'])} synchronised"
          + (f"; NOT linked to the reference: {A['unsynchronised_tiles']}" if A["unsynchronised_tiles"] else ""))
    print(f"recordings {A['n_recordings']}" + (f" ({A['n_direction_dropped']} dropped by the direction rule)" if A['direction_filter'] else "") + f"; repeat pairs {A['n_pairs']}; "
          f"tile-overlap components {[len(c) for c in A['components']]}")
    print(f"clock fit also uses {A['n_sync_extra']} dorsal-aorta segment(s) outside the network (provisional); first-harmonic phases taken "
          + ("relative to each recording's own mean-flow direction" if A["clock_own_direction"] else f"in the fixed vessel orientation, {A['n_extra_reoriented']} aorta recording(s) negated to the majority direction"))
    print(f"not reconstructed: {A['n_negative_ratio']} vessels with reconciled and combined means of opposite sign, {A['n_undefined_ratio']} without a usable mean")
    print(f"field: {el.sum()} of {len(el)} vessels ({supported(A).sum()} measurement-supported); recordings per vessel median {np.median(A['n_rec'][el]):.0f}; "
          f"|H1|/|Q'| median {np.median(np.abs(A['H'][0][el]) / np.abs(A['Qp'][el])):.2f}, |H2|/|Q'| median {np.median(np.abs(A['H'][1][el]) / np.abs(A['Qp'][el])):.2f}")
    paths.RESULTS.mkdir(parents=True, exist_ok=True); out = paths.RESULTS / f"{a.out}.pkl"
    out.write_bytes(pickle.dumps(A, protocol=4)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
