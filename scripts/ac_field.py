"""AC Option-A harmonic field on the carved network (plexusflow.ac.field).

Needs results/dc_reconciled_v2.pkl (scripts/dc_reconcile.py).  Prints coverage, the cross-tile H1 phase mismatch
before/after the tile time gauge, and the junction residual |Σ s Ĥ| / Σ|Ĥ| at fully-eligible interior junctions
(aligned / unaligned / random-phase null).  Defaults = the paper's synchronisation (plexusflow.ac.field.CANONICAL_SYNC) and
write results/ac_field_v7.pkl; --no-sync-da, --sync-weight rho, --cont-lambda 0, --no-sign-check give the older variants.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pickle
import numpy as np
import plexusflow.paths as paths
from plexusflow.ac.field import (build_ac_field, sync_mismatch, junction_residuals, da_stub_edges, REF_TILE,
                                 CANONICAL_SYNC as CS, CANONICAL_FIELD, CANONICAL_F0_BAND)
from plexusflow.reconcile import carved_domain


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-tile", type=int, default=None,
                    help=f"reference tile for the f0 band and the phase gauge (default: plexusflow.yaml ac_ref_tile, else {REF_TILE})")
    ap.add_argument("--f0-band", type=float, default=None,
                    help=f"keep tiles whose heart rate is within ± this fraction of the reference tile's (default: plexusflow.yaml ac_f0_band, else {CANONICAL_F0_BAND}); "
                         "large (e.g. 10) = every tile with a beat — for stages whose rate wanders between tiles (15S)")
    ap.add_argument("--sync-da", action=argparse.BooleanOptionalAction, default=True,
                    help="also synchronise on the dorsal-aorta stubs (outside the carve; default on)")
    ap.add_argument("--out", default=CANONICAL_FIELD, help=f"results/<out>.pkl (default {CANONICAL_FIELD})")
    ap.add_argument("--sync-da-min-rho", type=float, default=CS["sync_extra_min_rho"], help="with --sync-da: use only DA measurements with ρ ≥ this")
    ap.add_argument("--sync-da-weight", type=float, default=CS["sync_extra_weight"], help="with --sync-da: multiply the DA pair weights (≫ 1 = anchor)")
    ap.add_argument("--sign-check", action=argparse.BooleanOptionalAction, default=CS["sign_check"],
                    help="drop repeat-vessel pairs whose mean-flow directions disagree (default on)")
    ap.add_argument("--cont-lambda", type=float, default=CS["cont_lambda"], help="weight of plexus seam-continuity pairs in the sync (0 = off; 1 = calibrated)")
    ap.add_argument("--sync-weight", choices=("rho", "detect", "kappa"), default=CS["sync_weight"],
                    help="pair weights: kappa = fitted von Mises precision (paper), rho = ρ_aρ_b, detect = detected in both")
    args = ap.parse_args()
    from plexusflow.config import StageConfig
    cfg = StageConfig.load(paths.STAGE)                      # per-stage defaults (plexusflow.yaml)
    ref = args.ref_tile if args.ref_tile is not None else cfg.ac_ref_tile
    if ref is None:
        if paths.STAGE.name.startswith("PlexusFlow21S"):
            ref = REF_TILE                                   # the paper's 21S reference tile
        else:
            sys.exit(f"{paths.STAGE.name}: no reference tile. Set ac_ref_tile in plexusflow.yaml (a clearly beating tile near the "
                     f"aortae; its heart rate centres the f0 band and fixes the phase gauge) or pass --ref-tile.")
    if args.f0_band is None:
        args.f0_band = cfg.ac_f0_band if cfg.ac_f0_band is not None else CANONICAL_F0_BAND
    print(f"reference tile {ref}, f0 band ±{args.f0_band:g}")
    G = paths.load_graph(); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    extra = None
    if args.sync_da:
        extra = da_stub_edges(G)
        print(f"synchronising also on {len(extra)} dorsal-aorta stub edges: {extra}")
    A = build_ac_field(G, D, ref_tile=ref, f0_band=args.f0_band, sync_extra_edges=extra,
                       sync_extra_min_rho=args.sync_da_min_rho, sync_extra_weight=args.sync_da_weight, sync_weight=args.sync_weight, cont_lambda=args.cont_lambda, sign_check=args.sign_check)
    A["sync"] = dict(sync_da=bool(args.sync_da), min_rho=args.sync_da_min_rho, weight=args.sync_da_weight, pair_weight=args.sync_weight, cont_lambda=args.cont_lambda, sign_check=args.sign_check)
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
    # the complete field, including the per-recording phasors (_meas), synchronisation pairs (_pairs) and diagnostics
    # (_sync_diag), so Fig 8 draws exactly the field the D fit and Fig 9 use instead of rebuilding it
    (paths.RESULTS / f"{args.out}.pkl").write_bytes(pickle.dumps(A, protocol=4)); print(f"wrote {paths.RESULTS / (args.out + '.pkl')}")


if __name__ == "__main__":
    main()
