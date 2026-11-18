"""How well does conservation reconstruct vessels whose measurements are missing?  (Also scored: the passive r⁴/ℓ network
with the same boundary flows — what geometry alone predicts for the withheld vessels, the null reference.)  (input to Fig 6e)

For a fraction p of the carved vessels that have a usable recording, withhold EVERY recording of those vessels, reconcile the
rest (production quasi-likelihood, reliability model fixed), and compare each withheld vessel's reconstructed flow with its
own trusted recordings (c < 0.5): error = |log(|Q′| / |y|)|, direction = same sign.  p ∈ FRACS, N_REP random draws each.
The same draws are also reconciled with a CONSTANT relative uncertainty (one c and σ0 fitted by the same junction likelihood,
c_p = c_a = 0) — does the quality-dependent uncertainty model improve the reconstruction?  (curve_constant)
Reference line: disagreement between two trusted recordings of the same vessel from different tiles (what a second
measurement would give).  95 % intervals: bootstrap over reference tiles (recordings in a tile share errors and physiology),
pooled over draws.  Writes $PLEXUSFLOW_RESULTS/dc_withholding.json.  ~3 min (21S).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plexusflow.paths as P
from plexusflow.reconcile.reliability import measurement_table, c_of, same_tile_junctions
from plexusflow.reconcile.dc import reconcile_dc
from plexusflow.reconcile.network import carved_domain

FRACS = (0.05, 0.1, 0.25, 0.5, 0.75)
N_REP, N_BOOT, TRUST, SEED = 3, 1000, 0.5, 0
T0 = time.time()
log = lambda s: print(f"[+{time.time() - T0:4.0f} s] {s}", flush=True)


def tile_boot(rows, rng):
    """rows (err, ok, tile) → median err, direction rate, and their tile-bootstrap 95 % intervals"""
    a = np.array(rows, float); tiles = np.unique(a[:, 2]); grp = [np.flatnonzero(a[:, 2] == t) for t in tiles]; be, bd = [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([grp[k] for k in rng.integers(0, len(grp), len(grp))]); be.append(np.median(a[ix, 0])); bd.append(a[ix, 1].mean())
    return dict(err=float(np.median(a[:, 0])), err_ci=np.percentile(be, [2.5, 97.5]).tolist(), dir=float(a[:, 1].mean()),
                dir_ci=np.percentile(bd, [2.5, 97.5]).tolist(), n=int(len(a)), n_tiles=int(len(tiles)))


def fit_constant(G, M, model):
    """Constant relative uncertainty: one c (and σ0) maximising the same Gaussian junction likelihood as the reliability model,
    V_j = c² Σ y_k² + σ0², on all same-tile junctions; returned as a reliability-model dict with c_p = c_a = 0."""
    from scipy.optimize import minimize
    S2, R2 = [], []
    for t in sorted({t for t, _ in M}):
        for J in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t}):
            y = np.array([s * M[(t, e)]["Q"] for e, s in J]); S2.append(np.sum(y ** 2)); R2.append(y.sum() ** 2)
    S2, R2 = np.array(S2), np.array(R2)
    nll = lambda th: 0.5 * np.sum(np.log(np.exp(2 * th[0]) * S2 + np.exp(2 * th[1])) + R2 / (np.exp(2 * th[0]) * S2 + np.exp(2 * th[1])))
    th = minimize(nll, [np.log(0.4), np.log(1e-3)], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000)).x
    c, s0 = float(np.exp(th[0])), float(np.exp(th[1])); pt = [k for k in model if k.startswith("c_") and k not in ("c_T", "c_a")][0]
    return dict(model, c0=c, c_T={k: c for k in model["c_T"]}, **{pt: 0.0}, c_a=0.0, sigma0=s0), len(S2)


def main():
    G = P.load_graph(); M = measurement_table(G); model = json.loads((P.RESULTS / "reliability_model_v1.json").read_text()); dom = carved_domain(G)
    E = list(dom["edges"]); eix = {e: i for i, e in enumerate(E)}
    by_e = {}
    for (t, e), m in M.items():
        if e in eix:
            by_e.setdefault(e, []).append(m)
    usable = sorted(by_e)                                                    # vessels with ≥ 1 Kloosterman-valid recording
    ref_ok = [e for e in usable if any(c_of(model, m) < TRUST and abs(m["Q"]) > 0 for m in by_e[e])]
    log(f"{P.STAGE.name}: {len(usable)} measured carved vessels, {len(ref_ok)} with a trusted recording")
    rng = np.random.default_rng(SEED); out = dict(fracs=list(FRACS), n_rep=N_REP, trust_c=TRUST, n_measured=len(usable), curve=[])
    # reference: two trusted recordings of the same vessel from different tiles
    rep = []
    for e, ms in by_e.items():
        tr = [m for m in ms if c_of(model, m) < TRUST and abs(m["Q"]) > 0]
        for i, a in enumerate(tr):
            for b in tr[i + 1:]:
                rep.append((abs(np.log(abs(a["Q"]) / abs(b["Q"]))), float(np.sign(a["Q"]) == np.sign(b["Q"])), a["tile"]))
    out["repeat"] = tile_boot(rep, np.random.default_rng(SEED + 1))
    log(f"  trusted repeat-recording disagreement: median {out['repeat']['err']:.3f}, same sign {out['repeat']['dir']:.1%} ({len(rep)} pairs)")
    out["curve_passive"] = []; out["curve_constant"] = []
    model_c, nJc = fit_constant(G, M, model); out["constant_model"] = dict(c=model_c["c0"], sigma0=model_c["sigma0"], n_junctions=nJc)
    log(f"  constant relative uncertainty (same junction likelihood, {nJc} junctions): c {model_c['c0']:.3f}, σ0 {model_c['sigma0']:.1e}")
    for p in FRACS:
        rows = []; rows_pas = []; rows_con = []                             # reconstruction | passive network (geometry only) | constant uncertainty
        for r in range(N_REP):
            hide = set(rng.choice(len(usable), int(round(p * len(usable))), replace=False)); hide = {usable[k] for k in hide}
            Mk = {k: m for k, m in M.items() if k[1] not in hide}
            Df = reconcile_dc(G, model, Mk, dom=dom, n_starts=0, n_shuffle=1); Dc = reconcile_dc(G, model_c, Mk, dom=dom, n_starts=0, n_shuffle=1)
            for e in hide:
                i = eix[e]
                for m in by_e[e]:
                    if c_of(model, m) < TRUST and abs(m["Q"]) > 0:
                        rows.append((abs(np.log(max(abs(Df["Qp"][i]), 1e-12) / abs(m["Q"]))), float(np.sign(Df["Qp"][i]) == np.sign(m["Q"])), m["tile"]))
                        rows_con.append((abs(np.log(max(abs(Dc["Qp"][i]), 1e-12) / abs(m["Q"]))), float(np.sign(Dc["Qp"][i]) == np.sign(m["Q"])), m["tile"]))
                        rows_pas.append((abs(np.log(max(abs(Df["Qs"][i]), 1e-12) / abs(m["Q"]))), float(np.sign(Df["Qs"][i]) == np.sign(m["Q"])), m["tile"]))
            log(f"  p {p:.2f} draw {r + 1}/{N_REP}: {len(hide)} vessels withheld, inflow {Df['metrics']['total_in']:.2f} nL/s")
        s = tile_boot(rows, np.random.default_rng(SEED + 2)); s["frac"] = p; out["curve"].append(s)
        sp = tile_boot(rows_pas, np.random.default_rng(SEED + 3)); sp["frac"] = p; out["curve_passive"].append(sp)
        sc = tile_boot(rows_con, np.random.default_rng(SEED + 4)); sc["frac"] = p; out["curve_constant"].append(sc)
        log(f"  p {p:.2f}: constant uncertainty error median {sc['err']:.3f} [{sc['err_ci'][0]:.3f}, {sc['err_ci'][1]:.3f}], same sign {sc['dir']:.1%}")
        log(f"  p {p:.2f}: passive network (radii only) error median {sp['err']:.3f}, same sign {sp['dir']:.1%}")
        log(f"  p {p:.2f}: withheld-vessel error median {s['err']:.3f} [{s['err_ci'][0]:.3f}, {s['err_ci'][1]:.3f}], same sign {s['dir']:.1%} (n {s['n']})")
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "dc_withholding.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {P.RESULTS / 'dc_withholding.json'}")


if __name__ == "__main__":
    main()
