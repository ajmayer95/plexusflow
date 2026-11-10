"""Re-run the Kloosterman mean-flux estimator on the cached batch inputs and update the measured graph in place.

The batch keeps, per recording, the time-mean optical-flow field, the refined centreline and the sampling radius
(``<stage>/batch_kl_inputs/tile_*.pkl.gz``).  When the section-aggregation rule changes (2026-11-09: the 0.5–1.5×
median flow-rate band was dropped; ``kloosterman.PRODUCTION_KW``) this re-estimates every vessel's flux without
recomputing optical flow, and rescales the rest of that measurement to it:

  * Q_kl, R_kl_um, V_CL_kl_ums, Q_kl_valid_frac, Q_kl_n_valid, quality.f / n_kept / p (parabolicity) — from the new estimate;
  * Q_t, harm.mean_Q, Q_nls.best and the harmonic amplitudes  — × s = Q_kl,new / Q_kl,old (Q(t) is normalised to the
    Kloosterman mean, and the Huber harmonic fit is scale-equivariant, so this equals re-running it);
  * flow direction, SNRs, phases, alignment  — unchanged.

Idempotent (s = 1 on a second run).  The previous graph is kept as ``<graph>.pre_kl_<mode>_<date>``.  Must run under
numpy < 2 so the graph stays readable everywhere (see paths.load_graph).

    PLEXUSFLOW_STAGE=.../PlexusFlow21S python3 scripts/kl_reaggregate.py [--dry-run]
"""
from __future__ import annotations
import sys, gzip, pickle, glob, shutil, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from joblib import Parallel, delayed
import plexusflow.paths as P
from plexusflow.analyze.kloosterman import kloosterman_scalar, PRODUCTION_KW, parabolicity

LINEAR = ("A", "B", "amp", "sigma_A", "sigma_B", "sigma_amp")


def estimate_tile(fn):
    d = pickle.load(gzip.open(fn)); out = {}
    for e, ki in d["kl_inputs"].items():
        try:
            r = kloosterman_scalar(np.asarray(ki["vfield_mean"], float), np.asarray(ki["cl_used"], float),
                                   float(ki["R_prior_px"]), mask=ki.get("ridge_mask_roi"), **PRODUCTION_KW)
        except Exception:
            continue
        out[e] = dict(Q=float(r["Q_char"]) * ki["conv"], R_px=float(r["R_char"]), V_px=float(r["V_CL_char"]),
                      f=float(r["Q_char_valid_frac"]), n=int(r["n_valid"]), p=parabolicity(r.get("per_slice_r2", [])))
    return int(d["tile"]), out


def main():
    dry = "--dry-run" in sys.argv
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0" and not dry:
        sys.exit("run under numpy < 2 (system python3): graphs written by numpy 2 cannot be read by numpy 1.x")
    from plexusflow.analyze.constants import PX_SIZE_UM, FRAME_DT_S     # as the batch used them
    pt = dict(PX_SIZE_UM=PX_SIZE_UM, FRAME_DT_S=FRAME_DT_S)
    G = P.load_graph()
    files = sorted(glob.glob(str(P.STAGE / "batch_kl_inputs" / "tile_*.pkl.gz")))
    est = dict(Parallel(n_jobs=8)(delayed(estimate_tile)(fn) for fn in files))
    n_upd = n_miss = 0; ratios, df = [], []
    for u, v, dd in G.edges(data=True):
        for m in dd.get("measurements_piv") or []:
            if not (m.get("quality") or {}).get("kl_ok") or (m.get("Q_nls") or {}).get("best_key") != "kl":
                continue
            t = int(m["tile_id"]); new = est.get(t, {}).get((u, v)) or est.get(t, {}).get((v, u))
            if new is None or not np.isfinite(new["Q"]) or not np.isfinite(m["Q_kl"]) or m["Q_kl"] <= 0:
                n_miss += 1; continue
            s = abs(new["Q"]) / m["Q_kl"]; ratios.append(s); df.append(new["f"] - m["quality"]["f"])
            m["Q_kl"] = abs(new["Q"]); m["R_kl_um"] = new["R_px"] * pt["PX_SIZE_UM"]
            m["V_CL_kl_ums"] = new["V_px"] * pt["PX_SIZE_UM"] / pt["FRAME_DT_S"]
            m["Q_kl_valid_frac"] = new["f"]; m["Q_kl_n_valid"] = new["n"]
            m["quality"]["f"] = new["f"]; m["quality"]["n_kept"] = new["n"]; m["quality"]["p"] = new["p"]
            m["harm"]["mean_Q"] *= s; m["Q_nls"]["best"] *= s
            if m.get("Q_t") is not None:
                m["Q_t"] = (np.asarray(m["Q_t"]) * s).astype(np.float32)
            for h in m["harm"].get("harmonics") or []:
                for k in LINEAR:
                    if k in h:
                        h[k] *= s
                if "P_sig" in h:
                    h["P_sig"] *= s * s
            m.setdefault("prov", {})["kl_mode"] = PRODUCTION_KW["outlier_mode"]
            n_upd += 1
    r = np.log(ratios)
    print(f"{P.STAGE.name}: {n_upd} measurements re-estimated ({n_miss} without cached inputs, left unchanged)")
    print(f"  Q new/old: median ×{np.exp(np.median(r)):.3f}, IQR ×{np.exp(np.percentile(r, 25)):.3f}–×{np.exp(np.percentile(r, 75)):.3f}, "
          f"changed > 10 %: {np.mean(np.abs(r) > np.log(1.1)):.1%}, > 20 %: {np.mean(np.abs(r) > np.log(1.2)):.1%}")
    print(f"  f: mean change {np.mean(df):+.3f}; median f now {np.median([m['quality']['f'] for *_, d in G.edges(data=True) for m in d.get('measurements_piv') or [] if (m.get('quality') or {}).get('kl_ok')]):.3f}")
    if dry:
        print("  dry run — graph not written"); return
    gp = Path(P.GRAPH); bak = gp.with_name(gp.name + f".pre_kl_{PRODUCTION_KW['outlier_mode']}_{datetime.date.today():%Y%m%d}")
    if not bak.exists():
        shutil.copy2(gp, bak)
    tmp = gp.with_name(gp.name + ".tmp"); tmp.write_bytes(pickle.dumps(G, protocol=4)); tmp.replace(gp)
    print(f"  wrote {gp.name} (previous kept as {bak.name})")


if __name__ == "__main__":
    main()
