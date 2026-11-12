"""Sensitivity of the DC reconstruction to topology-uncertain regions (plexusflow.reconcile.regions).

Baseline = the same graph with topology tags ignored (open-region carve only).  Variants: the tagged regions as drawn,
optionally grown by --dilate r px (every non-open edge with an endpoint or midpoint within r px of a tagged edge's path),
and --loo (each region dropped in turn).  Same inputs, reliability model and solver (production quasi-likelihood, no
multistart).  Reports per variant: arterial inflow / left share / venous outflows; regions, cut nodes, max |net exchange|
(must be 0) and throughput; large reversals (|Q'| > 0.5 nL/s against the measured direction); field change of kept vessels
near (≤ 300 px of a region) vs far; --watch vessels; whole-vessel withholding (--withhold repeats, 10 % of measured vessels
kept in every variant, identical held sets).  Writes results/topology_sensitivity.json and
figures/diagnostics/topology_sensitivity.png.

Usage: python scripts/dc_topology_sensitivity.py [--graph PATH] [--dilate 0,40,80] [--loo] [--withhold 3] [--watch 50136-50144,...]
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.spatial import cKDTree
import plexusflow.paths as paths
from plexusflow.reconcile import measurement_table
from plexusflow.reconcile.reliability import c_of
from plexusflow.reconcile.dc import vessel_input, ql_solve
from plexusflow.reconcile.network import carved_domain, project, emergent, EPS_UNIMAGED
from plexusflow.reconcile.regions import is_open, is_topology, set_region, counts, TOPOLOGY


def solve(G, dom, recs, mod):
    es = [tuple(e) for e in dom["edges"]]; ix = {e: i for i, e in enumerate(es)}; me = len(es); by = {}
    for t, e, m in recs:
        if e in ix: by.setdefault(e, []).append((m, c_of(mod, m)))
    y = np.zeros(me); C = np.ones(me); im = np.zeros(me, bool); sb = np.zeros(me)
    for e, L in by.items():
        i = ix[e]; y[i], C[i], sb[i] = vessel_input(L); im[i] = True
    s0 = 0.01 * np.median(np.abs(y[im])); s0v = np.sqrt(s0 ** 2 + sb ** 2); Cw = np.where(im, C, 1.0)
    Q0 = project(y, np.where(im, 1.0, EPS_UNIMAGED), dom); m = float(np.median(1 / (Cw[im] ** 2 * Q0[im] ** 2 + s0 ** 2)))
    return dict(zip(es, ql_solve(Q0, y, Cw, im, s0v, m, dom)[0])), dict(zip(es, np.where(im, y, np.nan)))


def grow(G, r):
    """Copy of G with topology regions grown by r px."""
    G2 = G.copy()
    if r <= 0: return G2
    T = cKDTree(np.vstack([np.asarray(d["path"], float) for _, _, d in G.edges(data=True) if is_topology(d)]))
    for u, v, d in G2.edges(data=True):
        if is_open(d) or is_topology(d): continue
        p = np.asarray(d["path"], float); pts = np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]], p[len(p) // 2]])
        if T.query(pts)[0].min() <= r: set_region(d, TOPOLOGY)
    return G2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default=None, help="graph with region tags (default: the stage graph)")
    ap.add_argument("--dilate", default="0", help="comma list of px to grow the tagged regions (0 = as drawn)")
    ap.add_argument("--loo", action="store_true", help="also drop each tagged region in turn")
    ap.add_argument("--withhold", type=int, default=3, help="whole-vessel withholding repeats (0 = skip)")
    ap.add_argument("--watch", default="", help="comma list of u-v edges to report")
    a = ap.parse_args()
    G = paths.load_graph(a.graph); M = measurement_table(G); mod = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text())
    REC = [(t, e, m) for (t, e), m in M.items()]
    c = counts(G); print(f"tags: {c['topology']} topology-uncertain edges, {c['open']} open-boundary edges")
    if c["topology"] == 0:
        print("no topology-uncertain tags — tag regions in the workbench (class: Topology uncertain) or pass --graph"); return
    watch = [tuple(int(x) for x in w.split("-")) for w in a.watch.split(",") if w]
    pos = lambda n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]])
    variants = [("baseline (no topology)", G, False)] + [(f"regions{'' if r == 0 else f' +{r} px'}", grow(G, int(r)), True) for r in a.dilate.split(",")]
    if a.loo:
        import networkx as nx
        H = nx.Graph(); H.add_edges_from([(u, v) for u, v, d in G.edges(data=True) if is_topology(d)])
        for k, comp in enumerate(sorted(nx.connected_components(H), key=len, reverse=True)):
            G2 = G.copy()
            for u, v in H.subgraph(comp).edges(): set_region(G2.edges[u, v], None)
            variants.append((f"drop region {k} ({H.subgraph(comp).number_of_edges()} edges)", G2, True))
    out = {}; base = None
    for name, Gv, topo in variants:
        dom = carved_domain(Gv, topology=topo); Q, y = solve(Gv, dom, REC, mod)
        q = np.array([Q[e] for e in dom["edges"]]); f = dom["B"] @ q; fb = emergent(q, dom)
        inj = fb[dom["inj"]]; tot = float(inj.sum()); V = [float(x) for x in fb[dom["out"]]]
        nix = {n: k for k, n in enumerate(dom["nodes"])}
        regs = [dict(edges=r["n_edges"], cut=len(r["cut"]), leaky=r["leaky"], net=float(sum(f[nix[n]] for n in r["cut"])),
                     throughput=float(0.5 * sum(abs(f[nix[n]]) for n in r["cut"]))) for r in dom["regions"]]
        rev = [e for e in Q if np.isfinite(y[e]) and np.sign(Q[e]) != np.sign(y[e]) and abs(Q[e]) > 0.5]
        res = dict(inflow=tot, left=float(inj[dom["side"] == "L"].sum() / tot), V=V, regions=regs, n_reversals=len(rev),
                   reversals=[[int(u), int(v), float(y[(u, v)]), float(Q[(u, v)])] for u, v in rev],
                   watch={f"{u}-{v}": (float(Q[(u, v)]) if (u, v) in Q else None) for u, v in watch}, removed=int(sum(r["n_edges"] for r in dom["regions"])))
        if base is None:
            base = (Q, dom)
        else:
            cutpos = np.array([pos(n) for r in dom["regions"] for n in r["cut"]]) if dom["regions"] else np.zeros((0, 2))
            Tc = cKDTree(cutpos) if len(cutpos) else None
            ks = [e for e in Q if e in base[0] and abs(base[0][e]) > 1e-4]
            lr = np.array([abs(np.log(abs(Q[e]) / abs(base[0][e]))) if Q[e] != 0 else np.inf for e in ks])
            fl = np.array([np.sign(Q[e]) != np.sign(base[0][e]) for e in ks])
            mid = np.array([np.asarray(Gv.edges[e]["path"], float)[len(Gv.edges[e]["path"]) // 2] for e in ks])
            near = Tc.query(mid)[0] <= 300 if Tc else np.zeros(len(ks), bool)
            res.update(change_near_median=float(np.median(lr[near])) if near.any() else None, change_far_median=float(np.median(lr[~near])),
                       gt2x_near=float(np.mean(lr[near] > np.log(2))) if near.any() else None, gt2x_far=float(np.mean(lr[~near] > np.log(2))),
                       flips_near=float(fl[near].mean()) if near.any() else None, flips_far=float(fl[~near].mean()), n_near=int(near.sum()))
        out[name] = res
        print(f"{name:34s}: removed {res['removed']:4d} edges, {len(regs)} regions (max |net| {max([abs(r['net']) for r in regs], default=0):.0e}, "
              f"throughput {sum(r['throughput'] for r in regs):.2f}) | inflow {tot:.2f} L {res['left']:.1%} V {', '.join(f'{x:+.2f}' for x in V)} | "
              f"reversals {len(rev)}" + (f" | change near {res['change_near_median']:.3f} far {res['change_far_median']:.3f}, >2× near {res['gt2x_near']:.1%} far {res['gt2x_far']:.1%}"
                                         if 'change_far_median' in res and res['change_near_median'] is not None else "")
              + "".join(f" | {k} {v:+.3f}" if v is not None else f" | {k} removed" for k, v in res["watch"].items()), flush=True)
    if a.withhold:
        doms = [(name, Gv, carved_domain(Gv, topology=topo)) for name, Gv, topo in variants]
        common = set.intersection(*[set(map(tuple, d["edges"])) for _, _, d in doms])
        meas = sorted({e for _, e, _ in REC if e in common}); rng = np.random.default_rng(1); W = {n: [] for n, _, _ in doms}
        for rep in range(a.withhold):
            hid = set(map(tuple, rng.permutation(meas)[: int(0.1 * len(meas))]))
            tr = [x for x in REC if x[1] not in hid]; te = [x for x in REC if x[1] in hid]
            for name, Gv, dom in doms:
                Q, _ = solve(Gv, dom, tr, mod); W[name] += [(Q[e], m["Q"]) for t, e, m in te]
            print(f"  withholding repeat {rep + 1}/{a.withhold}", flush=True)
        for name, pq in W.items():
            A = np.array(pq, float); ok = np.sign(A[:, 0]) == np.sign(A[:, 1]); e = np.abs(np.log(np.abs(A[ok, 0] / A[ok, 1])))
            out[name]["withheld"] = dict(median_abs_log=float(np.median(e)), direction=float(ok.mean()), n=int(len(A)))
            print(f"withheld {name:34s}: ×{np.exp(np.median(e)):.3f}, direction {ok.mean():.1%} (n {len(A)})")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "topology_sensitivity.json").write_text(json.dumps(out, indent=1))
    _figure(G, variants, out)
    print(f"wrote {paths.RESULTS / 'topology_sensitivity.json'}")


def _figure(G, variants, out):
    """Mosaic overview: open regions orange, topology regions cyan, baseline large reversals red, reversals left after the
    first region variant black."""
    import tifffile
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mos = tifffile.imread(str(paths.STAGE / "fuse" / "maxmed.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos; Hm = mos.shape[0]
    fig, ax = plt.subplots(figsize=(9, 9.5)); ax.imshow(mos, cmap="gray", vmin=np.percentile(mos[mos > 0], 1), vmax=np.percentile(mos[mos > 0], 99.5))
    _, Gv, _ = variants[1]
    for u, v, d in Gv.edges(data=True):
        p = np.asarray(d.get("path", []), float)
        if not len(p): continue
        col = "#00d0ff" if is_topology(d) else ("#ff8800" if is_open(d) else None)
        if col: ax.plot(p[:, 0], Hm - p[:, 1], color=col, lw=1.2)
    names = list(out)
    for name, col, lw in ((names[0], "red", 2.5), (names[1], "black", 1.5)):
        for u, v, _, _ in out[name]["reversals"]:
            p = np.asarray(G.edges[u, v]["path"], float); ax.plot(p[:, 0], Hm - p[:, 1], color=col, lw=lw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"orange: open-boundary regions · cyan: topology-uncertain regions ({names[1]})\n"
                 f"red: large reversals without topology regions ({out[names[0]]['n_reversals']}) · black: remaining ({out[names[1]]['n_reversals']})", fontsize=9)
    d = paths.FIGURES / "diagnostics"; d.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(d / "topology_sensitivity.png", dpi=130)
    print(f"wrote {d / 'topology_sensitivity.png'}")


if __name__ == "__main__":
    main()
