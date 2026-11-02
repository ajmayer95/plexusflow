"""Optical-flow frame spacing (dt) vs speed on known-truth single vessels, for slow (15S-like) recordings.

Each video (geometry × speed × seed × condition) is rendered once and measured at every dt (and the adaptive 'auto').
Conditions (background, tracer contrast = tracer lumen SD / camera noise SD, whole-frame jitter):
  clean        21S tile 30, lumen SD matched to clean real vessels, no jitter            (the dt sweep)
  A original   21S tile 30, contrast 3.7, no jitter
  B low contr. 15S tile 37, contrast 0.8 (measured on 15S tile 37), no jitter
  C jitter     15S tile 37, contrast 3.7, jitter 0.02 px white + 0.08 px heartbeat
  D both       15S tile 37, contrast 0.8, jitter as C
Prints per condition: mean flux / truth (signed: negative = wrong direction), wrong-direction fraction, |H1/DC| recovery,
H1 phase error, and the dt the adaptive chooser picks.  Writes <out>/frame_spacing.pkl.
Usage: python scripts/benchmark/frame_spacing.py --bundle PATH [--out DIR] [--jobs 6] [--conditions clean,A,B,C,D]
"""
import sys, argparse, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.synthetic.stokes.bundle import Bundle
from plexusflow.synthetic.stokes.cases import vessel_case
from plexusflow.synthetic.stokes.render import A1

GEOMS = ("straight_a5_e1", "straight_a8_e1", "straight_a12_e1")
SEEDS = (1, 2)
F0, NF = 1.0, 600                                          # 15S-like: ~1 Hz heart, 2.4 s clip
COND = {"clean": dict(bg=("21S", 30, (150, 150)), contrast=None, jitter=None, speeds=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5),
                      dts=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, "auto"), order="asc"),
        "A": dict(bg=("21S", 30, (150, 150)), contrast=3.7, jitter=None),
        "B": dict(bg=("15S", 37, (150, 150)), contrast=0.8, jitter=None),
        "C": dict(bg=("15S", 37, (150, 150)), contrast=3.7, jitter=(0.02, 0.08)),
        "D": dict(bg=("15S", 37, (150, 150)), contrast=0.8, jitter=(0.02, 0.08))}
MATCHED_SPEEDS, MATCHED_DTS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5), (1, 2, 4, 6, 8, 12, 16, 24, "auto")


def run(bundle_root, cond, name, seed):
    C = COND[cond]; speeds = C.get("speeds", MATCHED_SPEEDS); dts = C.get("dts", MATCHED_DTS)
    order = list(speeds) if C.get("order") == "asc" else sorted(speeds, reverse=True)   # fastest first = direction reference
    rows = vessel_case(Bundle(bundle_root), name, speeds, seed, nf=NF, f0=F0, background=C["bg"], contrast=C["contrast"],
                       jitter=C["jitter"], dts=dts, speed_order=order)
    ref = next((r for r in rows if r["v"] == max(speeds) and r["dt"] == 1 and np.isfinite(r["Q_kl"])), None)
    for r in rows:
        r["cond"] = cond; r["ref_sign"] = np.sign(ref["Q_kl"]) if ref else None
    print(f"  {cond} {name} s{seed}: done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", default=str(paths.RESULTS / "benchmark"))
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--conditions", default=",".join(COND))
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    conds = a.conditions.split(",")
    rows = sum(Parallel(n_jobs=a.jobs)(delayed(run)(a.bundle, c, g, s) for c in conds for g in GEOMS for s in SEEDS), [])
    (out / "frame_spacing.pkl").write_bytes(pickle.dumps(rows)); print(f"{len(rows)} rows in {time.time() - t0:.0f} s")
    for cond in conds:
        R = [r for r in rows if r["cond"] == cond]; speeds = sorted({r["v"] for r in R}); dts = list(dict.fromkeys(r["dt"] for r in R))
        for title, fn in (("signed Q / Q_true  (median; negative = wrong direction)",
                           lambda rs: np.nanmedian([r["ref_sign"] * r["Q_kl"] / r["Q_true"] if r["ref_sign"] else np.nan for r in rs])),
                          ("wrong direction (fraction)",
                           lambda rs: np.nanmean([float(np.sign(r["Q_kl"]) != r["ref_sign"]) if np.isfinite(r["Q_kl"]) and r["ref_sign"] else np.nan for r in rs])),
                          ("|H1/DC| measured / true", lambda rs: np.nanmedian([abs(r["h1"]) / A1 for r in rs])),
                          ("H1 phase error (deg)", lambda rs: np.nanmedian([abs(np.degrees(np.angle(r["h1"]))) for r in rs]))):
            print(f"\n[{cond}] {title}\n v_max   " + "".join(f"{str(d):>7s}" for d in dts))
            for v in speeds:
                print(f"{v:6.2f}  " + "".join(f"{fn([r for r in R if r['v'] == v and r['dt'] == d]):7.2f}" for d in dts))
        print(f"[{cond}] auto dt used: " + ", ".join(f"v{v}: {np.median([r['dt_used'] for r in R if r['v'] == v and r['dt'] == 'auto']):.0f}"
                                                 for v in speeds))


if __name__ == "__main__":
    main()
