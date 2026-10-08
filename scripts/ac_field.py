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
from plexusflow.ac.field import build_ac_field, sync_mismatch, REF_TILE, F0_BAND
from plexusflow.reconcile import carved_domain


def main():
    G = pickle.loads(paths.GRAPH.read_bytes()); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    A = build_ac_field(G, D)
    print(f"f0 band (±{F0_BAND:.0%} of tile {REF_TILE}, {A['f_ref']:.2f} Hz): {len(A['band'])} tiles;  "
          f"H1 phase mismatch over {len(A['_pairs'])} cross-tile pairs {sync_mismatch(A, False):.1f}° → aligned {sync_mismatch(A):.1f}°")
    el, Hf, nE = A["elig"], A["H"], len(A["edges"])
    print(f"AC-eligible carved edges {el.sum()} of {nE} ({el.mean():.0%});  winner per-beat H1 scatter median "
          f"{np.nanmedian(A['scatter']):.2f}, ρ median {np.nanmedian(A['rho']):.2f}")
    dom = carved_domain(G); B = dom["B"].tocsr(); absB = abs(B); interior = np.setdiff1d(np.arange(B.shape[0]), dom["bnd"])
    n_bad = np.asarray(absB @ (~el).astype(float)).ravel(); deg = np.asarray(absB.sum(axis=1)).ravel()
    nodes_ok = interior[(n_bad[interior] == 0) & (deg[interior] >= 3)]; rng = np.random.default_rng(0)
    r = lambda H: np.median(np.abs((B @ H)[nodes_ok]) / np.maximum((absB @ np.abs(H))[nodes_ok], 1e-30))
    for k in (1, 2):
        Hk = np.where(el, Hf[k - 1], 0)
        Hun = np.where(el, np.array([Hf[k - 1, i] * np.exp(1j * k * A["psi"][A["tile"][i]]) if el[i] else 0 for i in range(nE)]), 0)
        Hrnd = np.where(el, np.abs(Hk) * np.exp(1j * rng.uniform(0, 2 * np.pi, nE)), 0)
        print(f"  H{k} junction residual ({len(nodes_ok)} junctions): aligned {r(Hk):.3f}  unaligned {r(Hun):.3f}  random phase {r(Hrnd):.3f}")
    out = {k: v for k, v in A.items() if not k.startswith("_")}
    (paths.RESULTS / "ac_field_v2.pkl").write_bytes(pickle.dumps(out, protocol=4)); print(f"wrote {paths.RESULTS / 'ac_field_v2.pkl'}")


if __name__ == "__main__":
    main()
