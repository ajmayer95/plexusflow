"""Reconciled pulsatile field on the carved network (plexusflow.ac.pulse; docs/method.md, step 3).

Needs results/dc_reconciled_v3.pkl (scripts/dc_reconcile.py).  Each flux recording is fitted with its mean and two harmonics and
weighted by 1/σ² (σ² from the fit residual).  Tile clocks: weighted circular least squares on every vessel recorded in two tiles,
each recording taken in its own mean-flow direction; tiles whose clock is too uncertain are left out.  A vessel's recordings are
averaged on the common clock and rescaled to the reconciled mean flow (not reconstructed where the reconciled and measured means
have opposite signs).  Writes results/ac_field_v9.pkl (the input of scripts/mech/ac_estimate.py).
--previous reproduces the earlier clocks (signed flux harmonics of network vessels plus the dorsal-aorta segments;
results/ac_field_v8.pkl) for the sensitivity comparison.  --agree averages only the recordings with the reconciled flow direction
(sensitivity; writes results/ac_field_v9_agree.pkl).
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
    ap.add_argument("--dc", default=RESULT, help=f"results/<dc>.pkl (default {RESULT})")
    ap.add_argument("--out", default=None, help=f"results/<out>.pkl (default {FIELD}; ac_field_v8 with --previous)")
    ap.add_argument("--previous", action="store_true", help="the earlier clocks: signed flux harmonics and the dorsal-aorta segments")
    ap.add_argument("--agree", action="store_true", help="sensitivity: average only recordings with the reconciled flow direction")
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
    a.out = a.out or ("ac_field_v8" if a.previous else FIELD + ("_agree" if a.agree else ""))
    kw = dict(clocks="previous", n_boot=0) if a.previous else dict(average="agree" if a.agree else "all")
    A = build_pulse_field(G, D, ref_tile=ref, f0_band=band, **kw); A["f0_band"] = band; A["dc"] = a.dc; el = A["elig"]
    print(f"reference tile {ref} ({A['f_ref']:.2f} Hz), heart-rate band ±{band:.0%}: {len(A['band'])} tiles, {len(A['psi'])} synchronised"
          + (f"; NOT linked to the reference: {sorted(set(A['band']) - set(A['psi_all']))}" if set(A['band']) - set(A['psi_all']) else ""))
    print(f"clocks from the {A['clock_source']}: {A['n_pairs']} pairs; tile components {[len(c) for c in A['components']]}")
    if a.previous:
        print(f"   {A['n_extra_reoriented']} dorsal-aorta recording(s) negated to the majority direction")
    else:
        sd = A["clock_sd_deg"]
        print(f"   clock SD (bootstrap) median {np.median(list(sd.values())):.1f}°, largest " + ", ".join(f"{t}: {v:.0f}°" for t, v in sorted(sd.items(), key=lambda kv: -kv[1])[:3])
              + f"; tiles above {A['max_clock_sd_deg']:.0f}° left out of the field: " + (", ".join(str(t) for t in A["excluded_tiles"]) or "none"))
    print(f"recordings {A['n_recordings']}; not reconstructed: {A['n_no_agreeing']} vessels with no recording in the reconciled direction, "
          f"{A['n_negative_ratio']} with reconciled and combined means of opposite sign, {A['n_undefined_ratio']} without a usable mean")
    print(f"field: {el.sum()} of {len(el)} vessels ({supported(A).sum()} measurement-supported), {A['n_inconsistent']} of them with recordings of both directions; "
          f"recordings per vessel median {np.median(A['n_rec'][el]):.0f}; |H1|/|Q'| median {np.median(np.abs(A['H'][0][el]) / np.abs(A['Qp'][el])):.2f}, "
          f"|H2|/|Q'| median {np.median(np.abs(A['H'][1][el]) / np.abs(A['Qp'][el])):.2f}")
    paths.RESULTS.mkdir(parents=True, exist_ok=True); out = paths.RESULTS / f"{a.out}.pkl"
    out.write_bytes(pickle.dumps(A, protocol=4)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
