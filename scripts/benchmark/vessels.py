"""Single-vessel known-truth sweep: shape (straight / arc / S-bend / taper) × size × flattening ε = depth/width × speed × seed.

Prints Q̂/Q_true by flattening and speed (the depth-averaging bias ≈ 1/ε), by kind and by size; Kloosterman failures.
Writes <out>/vessel_sweep.pkl.
Usage: python scripts/benchmark/vessels.py --bundle PATH [--out DIR] [--jobs 6]
"""
import sys, argparse, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.synthetic.stokes.bundle import Bundle
from plexusflow.synthetic.stokes.cases import vessel_case

KINDS, AS, EPS, SPEEDS, SEEDS = ("straight", "arc", "sbend", "taper"), (5, 8, 12), (1.0, 0.8, 0.6, 0.4), (0.1, 0.3, 0.9), (1, 2)


def one(bundle_root, kind, a, e, speeds=None):
    name = f"{kind}_a{a:g}_e{e:g}"
    try:
        return sum((vessel_case(Bundle(bundle_root), name, speeds or SPEEDS, seed=s) for s in SEEDS), [])
    except Exception as ex:
        print(f"  {name} failed: {ex}", flush=True); return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", default=str(paths.RESULTS / "benchmark"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--speeds", default=None, help="comma list (px/frame); default 0.1,0.3,0.9")
    ap.add_argument("--kinds", default=None, help="comma list of straight,arc,sbend,taper (default all)")
    ap.add_argument("--eps", default=None, help="comma list of flattenings (default 1,0.8,0.6,0.4)")
    a = ap.parse_args()
    global SPEEDS, KINDS, EPS
    if a.speeds: SPEEDS = tuple(float(x) for x in a.speeds.split(","))
    if a.kinds: KINDS = tuple(a.kinds.split(","))
    if a.eps: EPS = tuple(float(x) for x in a.eps.split(","))
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    res = sum(Parallel(n_jobs=a.jobs)(delayed(one)(a.bundle, k, s, e, SPEEDS) for k in KINDS for s in AS for e in EPS), [])
    (out / "vessel_sweep.pkl").write_bytes(pickle.dumps(res)); print(f"{len(res)} measurements in {time.time() - t0:.0f} s")
    R = {k: np.array([r[k] for r in res], float) for k in ("a", "eps", "v", "ratio", "f", "R_kl_over_a", "align")}
    kind = np.array([r["kind"] for r in res])
    med = lambda s: f"{np.exp(np.nanmedian(np.log(R['ratio'][s]))):.2f}" if np.isfinite(R["ratio"][s]).sum() >= 2 else "  — "
    print("\nQ̂/Q_true (median over seeds, sizes, kinds) by flattening ε = b/a and speed:")
    print("ε      " + "  ".join(f"v {v}" for v in SPEEDS) + "    1/ε    fail")
    for e in EPS:
        s = R["eps"] == e
        print(f"{e:4.1f}   " + "   ".join(med(s & (R["v"] == v)) for v in SPEEDS) + f"    {1 / e:.2f}   {np.mean(~np.isfinite(R['ratio'][s])):.0%}")
    print("\ncircular vessels (ε = 1) by kind × speed:")
    for k in KINDS:
        s = (R["eps"] == 1.0) & (kind == k)
        print(f"  {k:8s} " + "   ".join(f"v {v}: {med(s & (R['v'] == v))}" for v in SPEEDS)
              + f"   align {np.nanmedian(R['align'][s]):.2f}  f {np.nanmedian(R['f'][s]):.2f}")
    print("\ncircular vessels by size × speed:")
    for sz in AS:
        s = (R["eps"] == 1.0) & (R["a"] == sz)
        print(f"  a {sz:2d} px  " + "   ".join(f"v {v}: {med(s & (R['v'] == v))}" for v in SPEEDS) + f"   R_kl/a {np.nanmedian(R['R_kl_over_a'][s]):.2f}")


if __name__ == "__main__":
    main()
