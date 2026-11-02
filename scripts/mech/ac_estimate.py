"""Wall distensibility D (η optional) from the AC field: profile likelihood of plexusmech.ac.profile.

Needs results/dc_reconciled_v2.pkl, results/ac_field_v2.pkl (scripts/ac_field.py, with the per-winner a0 / var_phi /
var_logpi fields), results/reliability_model_v1.json.
Defaults = the frozen 2026-10-19 method: measurement-supported vessels (|Q'|/|a0| ≤ 5); carve forcing = local measured
ratio with L/R group corrections; veins pinned to the stub waveforms (--veins mean: one shared waveform, 21S; own: each
vein node its own stub, 27S); latent tile gauge; waveform-based weights; η = 0 (pass --eta 0 0.2 0.4 … to estimate it).
The stub waveforms are gauge-corrected with the fitted tile corrections, so the fit is run twice (--gauge-passes).
Writes results/ac_estimate_v8.json (D, η, profile, fitted boundary forcing per node, tile gauge corrections — the input of
make_figure10.py --bc-json) and figures/ac/ac_estimate_v8.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse, json, pickle, time
import numpy as np
import plexusflow.paths as paths
from plexusflow.reconcile import carved_domain
from plexusflow.ac.field import supported
from plexusmech.ac.model import build_context, ac_reliability, vein_forcing
from plexusmech.ac.profile import fit_profile, quadratic_min, fitted_forcing, group_table

T0 = time.time()
def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f} s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--veins", choices=("mean", "own", "local"), default="mean",
                    help="vein forcing: one shared stub waveform (21S), each node its own stub (27S), or the local ratio")
    ap.add_argument("--fit-veins", action="store_true", help="free the L/R vein group corrections (default: pinned)")
    ap.add_argument("--eta", type=float, nargs="+", default=[0.0], help="η grid (default 0: purely elastic wall)")
    ap.add_argument("--weights", choices=("waveform", "c"), default="waveform")
    ap.add_argument("--no-gauge", action="store_true", help="no latent tile gauge correction")
    ap.add_argument("--d-range", type=float, nargs=3, default=[2.5e-4, 8.0e-4, 21], metavar=("MIN", "MAX", "N"))
    ap.add_argument("--gauge-passes", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default="ac_estimate_v8", help="results/<out>.json (default ac_estimate_v8, the paper fit)")
    ap.add_argument("--field", default="ac_field_v7", help="results/<field>.pkl (default ac_field_v7 = scripts/ac_field.py defaults)")
    ap.add_argument("--gauge-block", default=None, help="tiles sharing one extra gauge offset with a flat prior, e.g. 15,18,19,20,29,30,31,32,33,34")
    a = ap.parse_args()
    G = paths.load_graph(); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    A = pickle.loads((paths.RESULTS / f"{a.field}.pkl").read_bytes())
    if "var_phi" not in A:
        sys.exit("results/ac_field_v2.pkl predates the per-beat uncertainties — rerun scripts/ac_field.py")
    model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text()); model["c_T"] = {int(k): v for k, v in model["c_T"].items()}
    dom = carved_domain(G); sup = supported(A); Am = dict(A, elig=sup)
    C0 = build_context(D, Am, dom); c = ac_reliability(G, Am, model)
    log(f"{paths.STAGE.name}: {sup.sum()} measurement-supported of {A['elig'].sum()} AC-eligible vessels; veins '{a.veins}'"
        f"{' (fitted)' if a.fit_veins else ' (pinned)'}; η {a.eta}; weights '{a.weights}'")
    d_grid = np.logspace(np.log10(a.d_range[0]), np.log10(a.d_range[1]), int(a.d_range[2])); gauge_rad = None
    for p in range(max(1, a.gauge_passes if a.veins != "local" else 1)):
        if a.veins == "local":
            C = C0
        else:
            C, info = vein_forcing(C0, A, mode=a.veins, gauge_rad=gauge_rad)
            log(f"  pass {p + 1}: {len(info['stubbed'])} of {len(dom['out'])} vein nodes have a stub;  shared waveform "
                + ", ".join(f"H{k} {abs(info['mean'][k]):.3f}∠{np.degrees(np.angle(info['mean'][k])):+.0f}°" for k in (1, 2)))
        R = fit_profile(C, c, A, d_grid=d_grid, eta_grid=a.eta, weights=a.weights, gauge=not a.no_gauge, pin_veins=not a.fit_veins,
                        n_jobs=a.n_jobs, progress=lambda m: log("  " + m),
                        gauge_block=[int(t) for t in a.gauge_block.split(",")] if a.gauge_block else None)
        gauge_rad = {t: np.radians(v) for t, v in R["delta_deg"].items()}
        prof = np.nanmin(R["S"], axis=0) - np.nanmin(R["S"])
        log(f"  pass {p + 1}: best D {R['D']:.3e} (parabola {quadratic_min(d_grid, prof):.3e}), η {R['eta']:.2f}, NLL {R['nll']:.1f}"
            + (f";  block gauge β {R['block_beta_deg']:+.1f}°" if R["gauge_block"] else ""))
    eta_prof = np.nanmin(R["S"], axis=1) - np.nanmin(R["S"])
    if len(a.eta) > 1:
        log("  η profile ΔNLL: " + "  ".join(f"{e:.2f}:{v:.1f}" for e, v in zip(a.eta, eta_prof)))
    for g, r in group_table(C, R["x"]).items():
        log(f"    {g:8s} H1 ×{r['amp1']:.2f}  H2 ×{r['amp2']:.2f}  timing {r['timing']:+.3f} beat  →  boundary pulsatility H1 {r['PI1']:.2f}, H2 {r['PI2']:.2f}")
    big = sorted(R["delta_deg"].items(), key=lambda kv: -abs(kv[1]))[:6]
    log("  largest tile gauge corrections: " + ", ".join(f"{t}: {v:+.0f}° [SD {A['gauge_sd_deg'].get(t, np.nan):.0f}°]" for t, v in big))
    f = fitted_forcing(C, R["x"]); bnd = np.asarray(dom["bnd"])
    out = dict(stage=paths.STAGE.name, D=R["D"], D_parabola=quadratic_min(d_grid, prof), eta=R["eta"], nll=R["nll"],
               forcing=f"carve: local ratio + L/R group corrections; veins: {a.veins}{' (fitted)' if a.fit_veins else ' (pinned)'}; "
                       f"weights {a.weights}; tile gauge {'latent' if not a.no_gauge else 'off'}",
               settings=vars(a), D_grid=d_grid.tolist(), eta_grid=list(a.eta), S=R["S"].tolist(),
               nodes=[int(dom["nodes"][b]) for b in bnd], f1=[[float(z.real), float(z.imag)] for z in f[1]],
               f2=[[float(z.real), float(z.imag)] for z in f[2]], x=[float(v) for v in R["x"]],
               gauge_correction_deg={str(t): v for t, v in R["delta_deg"].items()}, groups=group_table(C, R["x"]),
               n_supported=int(sup.sum()), gauge_block=R["gauge_block"], block_beta_deg=R["block_beta_deg"])
    (paths.RESULTS / f"{a.out}.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, e in enumerate(a.eta):
        ax.plot(d_grid, R["S"][i] - np.nanmin(R["S"]), "o-", ms=3, label=f"η = {e:g}")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10); ax.set_xlabel("D (1/Pa)"); ax.set_ylabel("ΔNLL (profiled)")
    ax.axvline(R["D"], color="k", lw=0.8); ax.legend(fontsize=8)
    ax.set_title(f"{paths.STAGE.name}: D = {out['D_parabola']:.2e} /Pa (veins {a.veins})", fontsize=10)
    o = paths.FIGURES / "ac" / f"{a.out}.png"; o.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(o, dpi=120)
    log(f"wrote {paths.RESULTS / (a.out + '.json')}, {o}")


if __name__ == "__main__":
    main()
