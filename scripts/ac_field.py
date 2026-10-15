"""AC Option-A harmonic field on the carved network (plexusflow.ac.field).

Needs results/dc_reconciled_v2.pkl (scripts/dc_reconcile.py).  Prints coverage, the cross-tile H1 phase mismatch
before/after the tile time gauge, and the junction residual |Σ s Ĥ| / Σ|Ĥ| at fully-eligible interior junctions
(aligned / unaligned / random-phase null).  Writes results/ac_field_v2.pkl.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pickle
import numpy as np
import plexusflow.paths as paths
from plexusflow.ac.field import build_ac_field, sync_mismatch, junction_residuals, REF_TILE, F0_BAND
from plexusflow.reconcile import carved_domain


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-tile", type=int, default=REF_TILE,
                    help=f"reference tile for the f0 band and the phase gauge (default {REF_TILE}, the 21S choice)")
    ap.add_argument("--f0-band", type=float, default=F0_BAND,
                    help=f"keep tiles whose heart rate is within ± this fraction of the reference tile's (default {F0_BAND}); "
                         "large (e.g. 10) = every tile with a beat — for stages whose rate wanders between tiles (15S)")
    args = ap.parse_args(); ref = args.ref_tile
    G = paths.load_graph(); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    A = build_ac_field(G, D, ref_tile=ref, f0_band=args.f0_band)
    A["f0_band"] = args.f0_band
    A["ref_tile"] = ref
    print(f"f0 band (±{args.f0_band:.0%} of tile {ref}, {A['f_ref']:.2f} Hz): {len(A['band'])} tiles;  "
          f"H1 phase mismatch over {len(A['_pairs'])} cross-tile pairs {sync_mismatch(A, False):.1f}° → aligned {sync_mismatch(A):.1f}°")
    el, nE = A["elig"], len(A["edges"])
    print(f"AC-eligible carved edges {el.sum()} of {nE} ({el.mean():.0%});  winner per-beat H1 scatter median "
          f"{np.nanmedian(A['scatter']):.2f}, ρ median {np.nanmedian(A['rho']):.2f}")
    n_j, res = junction_residuals(A, carved_domain(G))
    for k in (1, 2):
        print(f"  H{k} junction residual ({n_j} junctions): aligned {res[k]['aligned']:.3f}  "
              f"unaligned {res[k]['unaligned']:.3f}  random phase {res[k]['random']:.3f}")
    out = {k: v for k, v in A.items() if not k.startswith("_")}
    (paths.RESULTS / "ac_field_v2.pkl").write_bytes(pickle.dumps(out, protocol=4)); print(f"wrote {paths.RESULTS / 'ac_field_v2.pkl'}")


if __name__ == "__main__":
    main()
