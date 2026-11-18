"""Fig 6 — reliability-weighted DC reconstruction, in the order of the method: score reliability → calibrate it on junctions →
reconcile the network → validate on withheld measurements.  PLOS: 7.5 × 8.75 in, all text ≥ 8 pt Arial.

  a  one real recording (tile 24): dense time-mean flow, cross-section ticks shaded by their parabolic fit R²; parabolicity
     p = mean clip(R², 0, 1), directional alignment a = mean |cos θ|, reliability c = c0 + c_p(1 − p) + c_a(1 − a)
  b  calibration: normalized junction imbalance |in − out| / √Σy² (RMS) by junction measurement quality (quintile of mean c_k),
     measured vs the quality-dependent and constant-error models fitted on other tiles (dc_junction_calibration.json)
  c  measured network around the tile 22/26/27 overlap (vessels drawn to the frame edge): best recording per vessel, colour =
     c_k, width = |y|; rings (area ∝ |in − out|, largest quarter emphasised); unimaged segments < 2 radii = one junction
  d  reconciled network: width = |Q′| (same scale), colour = reconstructed / measured flux magnitude (neutral grey);
     conservation imposed at every interior junction
  e  prediction of withheld measurements: flux measurements removed (network kept), disagreement with the vessels' own
     independent recordings vs fraction withheld, repeat-recording benchmark, direction agreement (dc_withholding.json)
SI: fig6_S_example (9-vessel patch, repeated recordings, per-vessel flows, one single-tile junction), fig6_S_weighting (four
weightings), fig6_S_corrections, fig6_S_reliability (real good / poor recordings), fig6_S_heldout.

Run: python3 -u scripts/make_figure6.py   (writes figures/fig6/fig6.{png,pdf,tif} and the SI figures)
"""
import sys, json, gzip, pickle, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors, matplotlib.cm as cm, matplotlib.patheffects, matplotlib.transforms
import plexusflow.paths as P
from plexusflow.reconcile.reliability import profile_coef, measurement_table, same_tile_junctions, c_of, FS_HZ
from plexusflow.analyze.kloosterman import kloosterman_scalar, PRODUCTION_KW
from plexusflow.analyze.centerline import _centerline_geometry
from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch
from plexusflow.config import StageConfig

ST = P.STAGE
_ap = argparse.ArgumentParser(description="Fig 6v5 (DC reliability and reconciliation)")
_ap.parse_args()
OUT = P.FIGURES / "fig6" / "fig6"
Z_S = 5.0                                         # junction-circle area per z²
TILE, NODE = 30, 28329
SHOW_IMBALANCE = False                            # Fig 6c junction-imbalance labels (off: reviewer, panel a overloaded)
JNODE, JTILE = 58340, 22                          # SI fig6_S_example: patch vessels 2/1/3, all three recorded in tile 22
EX_TILE, EX_EDGE = 24, (64960, 65010)             # Fig 6a example recording: long, straight, f 0.67, align 0.68
REGION_HALF = 160                                 # Fig 6c/d: view = vessels within ± this many graph px (≈ 270 µm) of PATCH; every vessel entering the view is drawn
PATCH, PATCH_HOPS = 59219, 2                      # Fig 6c/d: 9 vessels each recorded in ≥ 2 of tiles 22/26/27; Q′ within every vessel's recordings
import os as _os
EX_PATCH = int(_os.environ.get("FIG6_EX_PATCH", 19781))         # Fig 6c worked example: 9 vessels, Q′ above and below the means
PATCH = int(_os.environ.get("FIG6_PATCH", PATCH)); PATCH_HOPS = int(_os.environ.get("FIG6_PATCH_HOPS", PATCH_HOPS))
HIGH, LOW = (28304, 28329), (35956, 35996)          # LOW replaced 11-17: (35992, 36030) removed by consolidation patch 0001
PX_UM = 1.7
SEED_FIG = 0                                       # bootstrap seed for figure bands
KW = PRODUCTION_KW
NEUT = "#4a6378"                                   # measured-network vessels (neutral: the rings carry the message)
COL = ["#0072B2", "#E69F00", "#CC79A7"]          # Okabe–Ito, as fig6v3
LOWCOL = "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plos_style
plos_style.apply()
plt.rcParams["mathtext.default"] = "regular"

t0 = time.time()
log = lambda s: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:4.0f} s] {s}", flush=True)


def crop(img, pts, pad):
    lo, hi = pts.min(0) - pad, pts.max(0) + pad; ctr, h = (lo + hi) / 2, max(hi - lo) / 2
    r0, c0 = int(max(ctr[0] - h, 0)), int(max(ctr[1] - h, 0))
    return r0, min(int(ctr[0] + h), img.shape[0]), c0, min(int(ctr[1] + h), img.shape[1])


def show(ax, img, box):
    r0, r1, c0, c1 = box; sub = img[r0:r1, c0:c1]
    ax.imshow(img, cmap="gray", vmin=np.percentile(sub, 1), vmax=np.percentile(sub, 99.7), interpolation="nearest")
    ax.set_xlim(c0 - 0.5, c1 - 0.5); ax.set_ylim(r1 - 0.5, r0 - 0.5); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, box, um=20):
    r0, r1, c0, c1 = box; L = um / PX_UM; x1 = c1 - 0.07 * (c1 - c0); y = r1 - 0.08 * (r1 - r0)
    ax.plot([x1 - L, x1], [y, y], color="white", lw=2, solid_capstyle="butt", zorder=12)
    ax.text(x1 - L / 2, y - 0.03 * (r1 - r0), f"{um} µm", color="white", ha="center", va="bottom", fontsize=8, zorder=12,
            bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.55))


def draw_measurement(ax, img, ki, color, valid):
    cl = np.asarray(ki["cl_used"], float); off = np.asarray(ki["roi_offset"], float); R = float(ki["R_prior_px"])
    vf = np.asarray(ki["vfield_mean"], float); _, tan, nrm = _centerline_geometry(cl); clt = cl + off
    box = crop(img, clt, 2.0 * R); show(ax, img, box)
    for i in range(len(cl)):
        a_, b_ = clt[i] - R * nrm[i], clt[i] + R * nrm[i]
        ax.plot([a_[1], b_[1]], [a_[0], b_[0]], color="white" if valid[i] else "#9a9a9a", lw=0.35 if valid[i] else 0.35,
                alpha=0.45 if valid[i] else 0.45, ls="-" if valid[i] else (0, (1.5, 1.5)), zorder=2)
    ax.plot(clt[:, 1], clt[:, 0], color=color, lw=1.6, zorder=3, solid_capstyle="round")
    # time-mean OF arrows on every other section, three points across the lumen
    P_, U = [], []
    for i in range(1, len(cl), 3):                       # every 3rd section, 5 across the lumen
        for s in np.linspace(-0.7, 0.7, 5):
            p = cl[i] + s * R * nrm[i]; r, c = int(round(p[0])), int(round(p[1]))
            if 0 <= r < vf.shape[0] and 0 <= c < vf.shape[1]:
                P_.append(p + off); U.append(vf[r, c])
    P_, U = np.array(P_), np.array(U); sp = np.linalg.norm(U, axis=1); k = 0.5 * R / np.percentile(sp, 90)
    ax.quiver(P_[:, 1], P_[:, 0], U[:, 0] * k, U[:, 1] * k, angles="xy", scale_units="xy", scale=1, color="#ffd23f",
              edgecolor=INK, linewidth=0.4, width=0.011, headwidth=3.5, headlength=3.5, headaxislength=3, pivot="middle", zorder=4)
    scalebar(ax, box)
    return box


def heldout_calibration(r, jt, jQ, jf, ja, n_fold=5, n_bins=8, n_boot=1000, seed=0):
    """Tile-held-out calibration: refit (c0, c_f, c_a, σ0) by ML on the junctions of 4/5 of the tiles (whole tiles per fold),
    predict σ_j for the held-out tiles' junctions; bin by predicted σ_j → RMS predicted vs RMS observed r_j (bootstrap CI)."""
    from scipy.optimize import minimize
    idx = np.concatenate([np.full(len(q), k) for k, q in enumerate(jQ)]); Q = np.concatenate(jQ); f = np.concatenate(jf); a = np.concatenate(ja)
    nJ, jt = len(r), np.asarray(jt); tiles = np.unique(jt); fold = dict(zip(tiles, np.random.default_rng(seed).permutation(len(tiles)) % n_fold))
    jfold = np.array([fold[t] for t in jt]); ef = jfold[idx]

    def V(th, sel_e):
        c = np.exp(th[0]) + np.exp(th[1]) * (1 - f[sel_e]) + np.exp(th[2]) * (1 - a[sel_e])
        return np.bincount(idx[sel_e], (Q[sel_e] * c) ** 2, minlength=nJ) + np.exp(2 * th[3])
    pred = np.empty(nJ); x0 = np.log([0.1, 0.3, 1.4, 1e-3])
    for k in range(n_fold):
        tr, te = jfold != k, jfold == k
        nll = lambda th: 0.5 * np.sum((np.log(V(th, ef != k)) + r ** 2 / V(th, ef != k))[tr])
        th = minimize(nll, x0, method="L-BFGS-B").x
        pred[te] = np.sqrt(V(th, ef == k)[te])
        log(f"  fold {k}: c0 {np.exp(th[0]):.3f} c_f {np.exp(th[1]):.3f} c_a {np.exp(th[2]):.3f} σ0 {np.exp(th[3]):.1e}; {te.sum()} held-out junctions")
    # bins by predicted σ_j; 95 % intervals from a TILE-level bootstrap (junctions within a tile are not independent)
    zh = r / pred; qb = np.quantile(pred, np.linspace(0, 1, n_bins + 1)); rng = np.random.default_rng(seed + 1)
    jb = np.clip(np.searchsorted(qb, pred, side="right") - 1, 0, n_bins - 1); ti = np.searchsorted(tiles, jt)
    S = np.zeros((len(tiles), n_bins)); N = np.zeros((len(tiles), n_bins))
    np.add.at(S, (ti, jb), r ** 2); np.add.at(N, (ti, jb), 1)
    boot = np.array([np.sqrt(S[k].sum(0) / np.maximum(N[k].sum(0), 1)) for k in rng.integers(0, len(tiles), (n_boot, len(tiles)))])
    rows = np.array([(np.sqrt(np.mean(pred[jb == k] ** 2)), np.sqrt(S[:, k].sum() / N[:, k].sum()), *np.percentile(boot[:, k], [2.5, 97.5]),
                      N[:, k].sum()) for k in range(n_bins)])
    log(f"  {len(tiles)} tiles, {nJ} held-out junctions; tile-bootstrap 95 % CI half-widths (% of RMS): "
        + ", ".join(f"{50 * (h - l) / o:.0f}" for o, l, h in rows[:, 1:4]))
    log(f"  held-out: RMS z {np.sqrt(np.mean(zh ** 2)):.3f}, within ±1.96 {np.mean(np.abs(zh) < 1.96):.3f}; bins (pred → obs): "
        + ", ".join(f"{p_:.4f}→{o_:.4f}" for p_, o_ in rows[:, :2]))
    return dict(rows=rows, z=zh, fold=fold, n_per_fold=np.bincount(jfold, minlength=n_fold))


def main():
    log("loading graph + reliability model")
    G = P.load_graph(ST / "graph" / "graph_vessels_piv.gpickle"); M = measurement_table(G)
    mod = json.loads((ST / "results" / "reliability_model_v1.json").read_text()); mod["c_T"] = {int(k): v for k, v in mod["c_T"].items()}
    s0 = mod["sigma0"]

    # b: standardized junction residuals
    z = []; rs_ = []; jt_, jQ_, jf_, ja_ = [], [], [], []
    for t in sorted({t for t, _ in M}):
        for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t}):
            y = np.array([s * M[(t, e)]["Q"] for e, s in j]); c = np.array([c_of(mod, M[(t, e)]) for e, s in j])
            sj = np.sqrt(np.sum((y * c) ** 2) + s0 ** 2); z.append(y.sum() / sj); rs_.append((y.sum(), sj))
            jt_.append(t); jQ_.append(np.abs(y)); jf_.append([M[(t, e)][profile_coef(mod)[0]] for e, _ in j])
            ja_.append([M[(t, e)]["align"] if np.isfinite(M[(t, e)]["align"]) else 1.0 for e, _ in j])
    z = np.array(z); rms, p95 = np.sqrt(np.mean(z ** 2)), np.mean(np.abs(z) < 1.96)
    rr_, ss_ = np.array(rs_).T; qb = np.quantile(ss_, np.linspace(0, 1, 9))
    log("  calibration (sigma octiles): " + ", ".join(f"σ {np.sqrt(np.mean(ss_[m] ** 2)):.4f} → RMS r {np.sqrt(np.mean(rr_[m] ** 2)):.4f}"
        for m in [(ss_ >= a_) & (ss_ <= b_) for a_, b_ in zip(qb[:-1], qb[1:])]))
    cal = heldout_calibration(rr_, jt_, jQ_, jf_, ja_)
    log(f"{len(z)} junctions: RMS z {rms:.3f}, within |z|<1.96 {p95:.3f}")

    # c: the junction
    jn = next(j for j in same_tile_junctions(G, TILE, {e for (tt, e) in M if tt == TILE}) if any(NODE in e for e, _ in j)
              and all(NODE in e for e, _ in j))
    order = [HIGH] + sorted([e for e, _ in jn if e != HIGH], key=lambda e: -abs(M[(TILE, e)]["Q"]) * c_of(mod, M[(TILE, e)]))
    jn = sorted(jn, key=lambda es: order.index(es[0]))
    y = np.array([s * M[(TILE, e)]["Q"] for e, s in jn]); cj = np.array([c_of(mod, M[(TILE, e)]) for e, _ in jn])
    sig2 = (y * cj) ** 2 + s0 ** 2; r = y.sum(); yp = y - sig2 * r / sig2.sum()
    log(f"junction {NODE}: y {np.round(y, 4)}, c {np.round(cj, 3)}, Σy {r:+.4f} → Σ Q' {yp.sum():+.1e}; Q' {np.round(yp, 4)}")
    D = pickle.load(open(ST / "results" / "dc_reconciled_v2.pkl", "rb")); eix = {tuple(e): i for i, e in enumerate(D["edges"])}
    yl = yp; yp = np.empty_like(y)
    for k, ((e, s), yy) in enumerate(zip(jn, y)):
        i = eix.get(e, eix.get(e[::-1])); sg = 1 if e in eix else -1; yp[k] = s * sg * D["Qp"][i]
        assert D.get("per_vessel", "winner") == "average" or abs(s * sg * D["Qm"][i] - yy) < 1e-9   # winner input = this recording
        log(f"  {e}: measured {yy:+.4f}, single-junction projection {yl[k]:+.4f}, network-reconciled {yp[k]:+.4f}")
    log(f"  network-reconciled Σ s Q' = {yp.sum():+.1e}")

    log("Kloosterman inputs + images")
    kl = pickle.load(gzip.open(ST / "batch_kl_inputs" / f"tile_{TILE}.pkl.gz"))["kl_inputs"]
    img = tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{TILE}_maxmed.tif")).astype(float)
    img = img.mean(-1) if img.ndim == 3 else img
    ex = {}
    for e in (HIGH, LOW):
        ki = kl.get(e) or kl.get(e[::-1])
        res = kloosterman_scalar(np.asarray(ki["vfield_mean"], float), np.asarray(ki["cl_used"], float), float(ki["R_prior_px"]),
                                 mask=ki.get("ridge_mask_roi"), **KW)
        valid = np.asarray(res["valid_mask"], bool); m = M[(TILE, e)]
        ex[e] = dict(ki=ki, valid=valid, f=m[profile_coef(mod)[0]], align=m["align"], c=c_of(mod, m), cv=m.get("cv", np.nan))
        log(f"  {e}: kept {valid.sum()}/{len(valid)} (stored f {m['f']:.3f}), align {m['align']:.2f}, c {ex[e]['c']:.2f}")

    # ---------------------------------------------------------------- layout (7.5 in wide, PLOS max)
    # One idea: inconsistent measurements → a conservation-consistent reconstruction (a, maps; b, one junction) → that
    # predicts independent recordings (c).  The reliability score c (examples, held-out junction calibration, known-truth
    # check) is the SI figure fig6_S_reliability; the quasi-likelihood is in Methods.
    BLUE, GREY, PINK = "#1f5fa8", "0.55", "#c51b8a"            # blue = lower uncertainty, magenta = higher / imbalance
    LET = dict(fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")
    TIT = dict(fontsize=9, color=INK, pad=4, loc="left")
    c0_ = mod.get("c0", np.median(list(mod["c_T"].values())))
    geom = load_tile_geometry(StageConfig.load(ST)); E = [tuple(e) for e in D["edges"]]; eidx = {e: i for i, e in enumerate(E)}
    tile_xy = lambda xy: graph_to_tile_batch(np.asarray(xy, float), geom, TILE)
    r0, r1, c0, c1 = box = (0, img.shape[0], 0, img.shape[1])
    inside = lambda q: np.any((q[:, 0] > r0) & (q[:, 0] < r1) & (q[:, 1] > c0) & (q[:, 1] < c1))
    path = {}
    for u, v, d in G.edges(data=True):
        q = tile_xy(d["path"]) if d.get("path") is not None else None
        if q is not None and inside(q):
            path[(u, v)] = q
    pe = [i for i, e in enumerate(E) if e in path]; ps = set(E[i] for i in pe)
    Qm, Qp, im, sig = D["Qm"], D["Qp"], D["imaged"].astype(bool), D["sigma"]
    AVG = D.get("per_vessel", "winner") == "average"                  # vessel input = weighted average of all its recordings
    AGG_TEXT = (r"repeat recordings aggregated before reconstruction:  $Q_k = \sum_t w_{k,t}\,Q_{k,t}\,/\sum_t w_{k,t}$,  $w_{k,t} = 1/c_{k,t}^2$,"
                r"  $c_k = (\sum_t w_{k,t}\,c_{k,t}^2\,/\sum_t w_{k,t})^{1/2}$" if AVG else "")
    recs_all = {}
    for (t_, e_), m_ in M.items():
        if tuple(e_) in eix:
            recs_all.setdefault(eix[tuple(e_)], []).append((int(m_["tile"]),))
    rad = {e: G.edges[e].get("radius", 5.0) for e in path}; rmax = np.percentile(list(rad.values()), 95)
    lw_of = lambda e: 0.7 + 2.0 * min(rad[e] / rmax, 1.0)
    qa = np.abs(np.r_[Qm[[i for i in pe if im[i]]], Qp[pe]]); qa = qa[qa > 0]
    mag_norm = mcolors.LogNorm(vmin=np.percentile(qa, 3), vmax=np.percentile(qa, 99))
    mag_cmap = mcolors.ListedColormap(matplotlib.colormaps["viridis"](np.linspace(0.18, 1.0, 256)))
    vlo, vhi = np.percentile(img, 1), np.percentile(img, 99.7) * 1.25
    jpos = tile_xy([[G.nodes[NODE]["x"], G.nodes[NODE]["y"]]])[0]
    # standardized junction imbalance z = Σ s Q / √(Σ σ²) at interior junctions inside the tile whose vessels are all measured
    zj = {}
    for n in {n for e in ps for n in e}:
        if G.degree(n) < 3 or G.nodes[n].get("boundary_type") in ("source", "sink"):
            continue
        inc = []
        for nb in G.neighbors(n):
            e = (n, nb) if (n, nb) in ps else ((nb, n) if (nb, n) in ps else None)
            if e is None:
                break
            inc.append((eidx[e], 1.0 if e[0] == n else -1.0))
        else:
            p_ = tile_xy([[G.nodes[n]["x"], G.nodes[n]["y"]]])[0]
            if r0 < p_[0] < r1 and c0 < p_[1] < c1 and all(im[i] for i, _ in inc):
                zj[n] = (p_, sum(s_ * Qm[i] for i, s_ in inc) / np.sqrt(sum(sig[i] ** 2 for i, _ in inc)), sum(s_ * Qp[i] for i, s_ in inc))
    big = {n: v for n, v in zj.items() if abs(v[1]) > 2}
    log(f"tile {TILE}: {len(zj)} fully measured junctions, {len(big)} with |z| > 2 in the measured field; max |Σ Q′| {max(abs(v[2]) for v in zj.values()):.1e}")

    q0 = float(np.percentile(np.abs(Qm[im]), 10))                    # floor for relative changes of near-zero flows
    rel = np.where(im, (np.abs(Qp) - np.abs(Qm)) / np.maximum(np.abs(Qm), q0), np.nan); rev = im & (np.sign(Qp) != np.sign(Qm))

    # ---- main figure: a recordings of the same vessels from different tiles (different times) disagree; the reconstruction
    #      is one conservative field through them;  b prediction of independent recordings
    import networkx as nx
    from plexusflow.io.acquisition import tile_acquisition_minutes
    Hn = nx.Graph(); Hn.add_edges_from(E); pn_ = nx.single_source_shortest_path_length(Hn, PATCH, cutoff=PATCH_HOPS)
    pidx = sorted({eidx[e] if e in eidx else eidx[e[::-1]] for e in Hn.subgraph(pn_).edges()})
    pjun = [n for n in pn_ if Hn.degree(n) >= 3 and pn_[n] < PATCH_HOPS]
    recs = {i: [] for i in pidx}
    for (tt, e), m in M.items():
        if e in eidx and eidx[e] in recs:
            recs[eidx[e]].append((int(m["tile"]), abs(m["Q"]), c_of(mod, m), np.sign(m["Q"])))
    ptiles = sorted({r[0] for i in pidx for r in recs[i]}); tmin = tile_acquisition_minutes(ST, StageConfig.load(ST), ptiles)
    TCOL = dict(zip(ptiles, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]))
    xy = lambda n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], float)
    mids = {i: np.asarray(G.edges[E[i]]["path"], float)[len(G.edges[E[i]]["path"]) // 2] for i in pidx}
    order = sorted(pidx, key=lambda i: (mids[i][0], mids[i][1])); num = {i: k + 1 for k, i in enumerate(order)}
    log(f"patch {PATCH}: {len(pidx)} vessels, tiles {ptiles} at {[tmin[t_] for t_ in ptiles]} min; "
        + "; ".join(f"{num[i]}: Q′ {abs(Qp[i]):.3f} vs " + "/".join(f"{r[1]:.3f}" for r in recs[i]) for i in order))

    # graph (x, y) → mosaic image (column, row) = (x, H − y): draw everything in tissue orientation
    mos = tifffile.imread(str(ST / "fuse" / "maxmed.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos; Hm = mos.shape[0]
    toimg = lambda q: np.column_stack([q[:, 0], Hm - q[:, 1]]); xy0 = xy; xy = lambda n: toimg(xy0(n)[None])[0]
    mids = {i: toimg(np.asarray(G.edges[E[i]]["path"], float))[len(G.edges[E[i]]["path"]) // 2] for i in pidx}
    fig = plt.figure(figsize=(7.5, 7.0), facecolor="white")
    gt = fig.add_gridspec(1, 2, left=0.02, right=0.985, top=0.95, bottom=0.465, width_ratios=[0.9, 1.1], wspace=0.14)
    gr = gt[1].subgridspec(2, 1, height_ratios=[0.85, 1.15], hspace=0.5)
    # a: the vessels in the tissue (max − median mosaic), numbered, with flow direction and the three tile footprints
    allp = toimg(np.vstack([np.asarray(G.edges[E[i]]["path"], float) for i in pidx]))
    axC = fig.add_subplot(gt[0]); ctr_ = (allp.min(0) + allp.max(0)) / 2; half_ = 0.5 * (allp.max(0) - allp.min(0)).max() + 28
    c0_, c1_, r0_, r1_ = int(ctr_[0] - half_), int(ctr_[0] + half_), int(ctr_[1] - half_), int(ctr_[1] + half_)
    crop_ = mos[max(r0_, 0):r1_, max(c0_, 0):c1_]; lo_, hi_ = np.percentile(crop_[crop_ > 0], (1, 99.5))
    axC.imshow(mos, cmap="gray", vmin=lo_, vmax=hi_ * 1.3, interpolation="antialiased"); axC.set_xlim(c0_, c1_); axC.set_ylim(r1_, r0_)
    axC.set_aspect("equal"); axC.set_xticks([]); axC.set_yticks([])
    for s_ in axC.spines.values():
        s_.set_color("0.7"); s_.set_linewidth(0.5)
    placed = []
    plen_ = {i: float(np.sum(np.linalg.norm(np.diff(np.asarray(G.edges[E[i]]["path"], float), axis=0), axis=1))) for i in pidx}
    arrow_set = set(pidx)                                                         # direction on every vessel (reconstructed sign), kept subordinate
    for i in pidx:
        q = toimg(np.asarray(G.edges[E[i]]["path"], float))
        if np.linalg.norm(q[0] - xy(E[i][0])) > np.linalg.norm(q[-1] - xy(E[i][0])):
            q = q[::-1]
        axC.plot(q[:, 0], q[:, 1], color="black", lw=4.6, zorder=4, solid_capstyle="round"); axC.plot(q[:, 0], q[:, 1], color="#ffd400", lw=3.2, zorder=5, solid_capstyle="round")
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; a_ = int(np.searchsorted(cum, 0.5 * cum[-1]))
        lo_i, hi_i = max(a_ - 2, 0), min(a_ + 2, len(q) - 1); d_ = q[hi_i] - q[lo_i]
        if Qp[i] < 0:
            d_ = -d_
        lab_xy = mids[i]
        if np.linalg.norm(d_) > 0:
            u_ = d_ / np.linalg.norm(d_)
        if np.linalg.norm(d_) > 0 and i in arrow_set:
            i0, i1 = int(np.searchsorted(cum, 0.22 * cum[-1])), min(int(np.searchsorted(cum, 0.78 * cum[-1])), len(q) - 1)
            seg = q[i0:i1 + 1] if Qp[i] >= 0 else q[i0:i1 + 1][::-1]                    # shaft follows the vessel, head at the downstream end
            HL_, HW_ = 6.5, 3.2                                                   # head length / half-width in mosaic pixels
            if len(seg) < 3:                                                       # short vessel: its own chord, downstream end
                a0_, a1_ = (q[0], q[-1]) if Qp[i] >= 0 else (q[-1], q[0]); seg = np.array([a0_, a0_ + 0.85 * (a1_ - a0_)])
            # smooth the traced path (jagged pixel steps), then resample by arc length
            sm_ = seg.copy()
            if len(seg) >= 5:
                k_ = np.ones(3) / 3; sm_ = np.column_stack([np.convolve(np.pad(seg[:, d], 1, mode="edge"), k_, "valid") for d in (0, 1)])
            cl_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sm_, axis=0), axis=1))]; Lt_ = cl_[-1]
            at_ = lambda L: np.array([np.interp(L, cl_, sm_[:, 0]), np.interp(L, cl_, sm_[:, 1])])
            tip_ = at_(Lt_); back_ = at_(max(Lt_ - 16, 0))                          # head direction from the last ~16 px of the shaft
            dv_ = tip_ - back_; dv_ = dv_ / max(np.linalg.norm(dv_), 1e-9); pv_ = np.array([-dv_[1], dv_[0]])
            base_ = tip_ - HL_ * dv_
            Ls_ = np.linspace(0, max(Lt_ - HL_, 0), 40); shaft_ = np.array([at_(L) for L in Ls_])
            if len(shaft_) and Lt_ > HL_ + 2:
                shaft_[-1] = base_                                                 # shaft ends exactly at the head's base centre
                axC.plot(shaft_[:, 0], shaft_[:, 1], color="0.25", lw=1.0, solid_capstyle="butt", zorder=6)
            tri_ = np.array([tip_, base_ + HW_ * pv_, base_ - HW_ * pv_])
            axC.add_patch(matplotlib.patches.Polygon(tri_, closed=True, fc="0.25", ec="none", zorder=7))
        if np.linalg.norm(d_) > 0:
            nrm = np.array([-u_[1], u_[0]]); cands = [q[a_] + 12 * nrm, q[a_] - 12 * nrm]      # number beside the vessel, not on it
            lab_xy = max(cands, key=lambda c: min([np.linalg.norm(c - o) for o in placed] + [1e9]))
        placed.append(lab_xy)
        axC.text(*lab_xy, str(num[i]), ha="center", va="center", fontsize=9, color=INK, fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.6))
    # imbalance of the used recordings at each junction, before reconstruction (all 0 after) — logged only (SI / caption)
    for n in (pjun if SHOW_IMBALANCE else []):
        inc = [(eidx[(n, m)], 1.0) if (n, m) in eidx else (eidx[(m, n)], -1.0) for m in Hn.neighbors(n)]
        if not all(im[i_] for i_, _ in inc):
            continue
        r_ = sum(s_ * Qm[i_] for i_, s_ in inc); thr_ = sum(abs(Qm[i_]) for i_, _ in inc) / 2; nt_ = len({int(D["tile"][i_]) for i_, _ in inc})
        pj_ = xy(n); obst_ = np.vstack([allp] + [np.array(placed)] * 3 if placed else [allp])
        cands_ = [pj_ + (30 if nt_ == 1 else 22) * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        lp_ = max(cands_, key=lambda c: np.min(np.linalg.norm(obst_ - c, axis=1))); placed.append(lp_)
        jcol_ = TCOL.get(int(D["tile"][inc[0][0]]), PINK) if nt_ == 1 else PINK      # single-tile junction: in that tile's colour
        axC.plot([pj_[0], lp_[0]], [pj_[1], lp_[1]], color=jcol_, lw=0.9, zorder=8)
        axC.scatter(*pj_, s=70, facecolor="none", edgecolor=jcol_, linewidth=1.4, zorder=8)
        axC.text(*lp_, f"{100 * abs(r_) / thr_:.0f}%" + ("\none tile" if nt_ == 1 else ""), ha="center", va="center", fontsize=8, linespacing=1.0,
                 color=jcol_, fontweight="bold", zorder=9, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=jcol_, lw=0.6 if nt_ > 1 else 1.2))
        log(f"  junction {n}: imbalance of used recordings {100 * abs(r_) / thr_:.0f}% ({nt_} tile(s))")
    L_ = 50 / PX_UM; xb_, yb_ = c1_ - 8 - L_, r1_ - 8
    axC.plot([xb_, xb_ + L_], [yb_, yb_], color="white", lw=2.5, solid_capstyle="butt", zorder=8)
    axC.text(xb_ + L_ / 2, yb_ - 5, "50 µm", color="white", fontsize=8, ha="center", va="bottom", zorder=8,
             path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
    axC.set_title("Vessels in the overlap of three tiles", fontsize=9, loc="left", pad=3)
    axI = axC.inset_axes([0.0, 0.0, 0.36, 0.36]); ipad = 230
    ic0, ic1, ir0, ir1 = int(ctr_[0] - ipad), int(ctr_[0] + ipad), int(ctr_[1] - ipad), int(ctr_[1] + ipad)
    axI.imshow(mos, cmap="gray", vmin=lo_, vmax=hi_ * 1.3, interpolation="antialiased"); axI.set_xlim(ic0, ic1); axI.set_ylim(ir1, ir0)
    axI.set_xticks([]); axI.set_yticks([])
    for s_ in axI.spines.values():
        s_.set_color("white"); s_.set_linewidth(1.2)
    for t_ in ptiles:
        tg = geom.tiles[int(t_)]; Hv, Wv = tg.video_shape if tg.video_shape else tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{t_}_maxmed.tif")).shape[:2]
        fp = np.array([[0, 0, 1], [0, Wv, 1], [Hv, Wv, 1], [Hv, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
        fp = fp[:, :2] / geom.downsample - [geom.y0, geom.x0]                  # (row, column) in the mosaic
        axI.plot(fp[:, 1], fp[:, 0], color=TCOL[t_], lw=1.6, zorder=3)
    axI.add_patch(matplotlib.patches.Rectangle((c0_, r0_), c1_ - c0_, r1_ - r0_, fill=False, ec="#ffd400", lw=1.4, zorder=4))

    # b: one vessel, three recordings
    from plexusflow.reconcile.harmonics import FS as _FS
    wv = max((i for i in order if len(recs[i]) >= 3), key=lambda i: max(r[1] for r in recs[i]) / min(r[1] for r in recs[i]))
    axW = fig.add_subplot(gr[0]); p_ = axW.get_position(); axW.set_position([p_.x0 + 0.05, p_.y0, p_.width - 0.05, p_.height])
    sgn = np.sign(Qp[wv]) or 1.0
    for m in G.edges[E[wv]]["measurements_piv"]:
        t_ = int(m["tile_id"])
        if t_ not in TCOL or m.get("Q_t") is None:
            continue
        q = sgn * np.asarray(m["Q_t"], float); tt = np.arange(len(q)) / _FS; sel = tt <= 2.0; used = AVG or t_ == int(D["tile"][wv])
        axW.plot(tt[sel], q[sel], color=TCOL[t_], lw=1.4 if used else 0.9, alpha=1.0 if used else 0.7, zorder=3 if used else 2)
        axW.axhline(q.mean(), color=TCOL[t_], lw=1.0, ls=(0, (3, 2)), zorder=2)
    axW.axhline(abs(Qp[wv]), color=INK, lw=2.0, zorder=4)
    from matplotlib.lines import Line2D as _L2b
    axW.legend(handles=[_L2b([], [], color=INK, lw=2.0, label="reconstructed mean Q′")], loc="upper left", frameon=False, fontsize=8,
               handlelength=1.6, borderaxespad=0.1)
    axW.set_ylim(top=axW.get_ylim()[1] * 1.22)                                   # room for the key above the traces
    axW.set_xlim(0, 2.0); axW.set_xlabel("time since recording start (s)", fontsize=8, labelpad=1); axW.set_ylabel("flow (nL/s)", fontsize=8, labelpad=1)
    axW.tick_params(labelsize=8); axW.spines[["top", "right"]].set_visible(False)
    axW.set_title(f"Repeated recordings of vessel {num[wv]} (dashed: means)", fontsize=9, loc="left", pad=3)
    log(f"  waveform panel: vessel {num[wv]} {E[wv]}, means " + ", ".join(f"{r[0]}: {r[1]:.3f}" for r in sorted(recs[wv])) + f"; Q′ {abs(Qp[wv]):.3f}")

    # c: all nine vessels — selected recording, held-out recordings, reconstruction
    axS = fig.add_subplot(gr[1]); xs_ = np.arange(1, len(order) + 1)
    p_ = axS.get_position(); axS.set_position([p_.x0 + 0.05, p_.y0, p_.width - 0.05, p_.height])
    for k, i in enumerate(order):
        rs = recs[i]; offs = np.linspace(-0.18, 0.18, len(rs)) if len(rs) > 1 else [0.0]
        used = int(D["tile"][i])                                             # winner input: the only recording the reconstruction uses
        assert used in [r[0] for r in rs] and min(rs, key=lambda r: r[2])[0] == used, (i, used, rs)
        for (t_, q_, c_, s_), dx in zip(sorted(rs), offs):
            u_ = AVG or t_ == used                                           # averaged input: every recording is used
            axS.scatter(k + 1 + dx, q_, s=52 if u_ else 24, facecolors=TCOL[t_] if u_ else "white", edgecolors=TCOL[t_], linewidths=1.3,
                        alpha=1.0 if u_ else 0.6, zorder=3)
        if AVG:
            axS.plot([k + 1 - 0.38, k + 1 + 0.38], [abs(Qm[i])] * 2, color="0.6", lw=1.6, zorder=3.5, solid_capstyle="butt")
        axS.plot([k + 1 - 0.38, k + 1 + 0.38], [abs(Qp[i])] * 2, color=INK, lw=3.2, zorder=4, solid_capstyle="butt")
    axS.set_yscale("log"); axS.set_xticks(xs_); axS.set_xticklabels([str(x) for x in xs_], fontsize=8); axS.set_xlim(0.4, len(order) + 0.6)
    axS.set_xlabel("Vessel ID", fontsize=8, labelpad=1); axS.set_ylabel("|mean flow| (nL/s)", fontsize=8, labelpad=1); axS.tick_params(labelsize=8)
    axS.spines[["top", "right"]].set_visible(False); axS.set_title("Conservation-consistent reference flows", fontsize=9, loc="left", pad=3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="", ms=7, color=TCOL[t_], label=f"Tile {t_}") for t_ in ptiles]
    hs += ([Line2D([], [], marker="o", ls="", ms=7, color="0.35", label="Recorded flux"), Line2D([], [], color="0.6", lw=1.6, label="Weighted average")]
           if AVG else [Line2D([], [], marker="o", ls="", ms=7, color="0.35", label="Used"),
                        Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec="0.35", mew=1.3, label="Not used")]) \
          + [Line2D([], [], color=INK, lw=3.2, label="Reconstructed")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.425), ncol=6, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.4)

    # ---- bottom row: how the reconstruction is calibrated and checked.  d one simultaneously recorded junction (imbalance r_j vs
    #      the σ_j its vessels' c_k predict) · e c_k vs one constant c on held-out tiles (dc_junction_calibration.py) · f prediction
    #      of independent recordings, selected model only (dc_validate_weighting.py; the four-weighting comparison is SI)
    V = json.loads((P.RESULTS / "dc_weighting_validation.json").read_text()); JC = json.loads((P.RESULTS / "dc_junction_calibration.json").read_text())
    BOT, TOPB, TT = 0.075, 0.32, 0.352                                # axes bottom / top, title line (below the tile legend)
    def draw_calibration(axE, big=False):
        """held-out RMS(r_j / σ_j,pred) by quality quintile: quality-dependent c_k vs one constant c (dc_junction_calibration.json)"""
        xq = np.arange(1, 6); ms_ = 4.5
        for lab, colr, mk, dx, name in (("constant", GREY, "s", 0.09, "constant fractional discrepancy"), ("calibrated", BLUE, "o", -0.09, "quality-dependent")):
            rms_, ci_ = np.array(JC[lab]["rms_z"]), np.array(JC[lab]["rms_z_ci"])
            axE.vlines(xq + dx, ci_[:, 0], ci_[:, 1], color=colr, lw=1.0, zorder=2)
            axE.plot(xq + dx, rms_, marker=mk, ms=ms_, color=colr, lw=1.3, ls="-" if lab == "calibrated" else (0, (3, 2)), mfc=colr, mec=colr, zorder=3, label=name)
        axE.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1); axE.text(0.55, 1.02, "expected = 1", fontsize=8, color=MUTED, ha="left", va="bottom")
        axE.set_xticks(xq); axE.set_xticklabels(["low", "2", "3", "4", "high"], fontsize=8); axE.set_xlim(0.5, 5.5); axE.set_ylim(0.3, 1.65)
        axE.set_xlabel("Mean discrepancy score $c$ at junction (quintile)", fontsize=9, labelpad=2)
        axE.set_ylabel("Observed / predicted\njunction error (RMS)", fontsize=9, labelpad=2)
        axE.tick_params(labelsize=8); axE.spines[["top", "right"]].set_visible(False)
        axE.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.1)

    def draw_calibration_cum(axE, n_boot=500):
        """unbinned calibration: junctions ranked by predicted relative error (production c_eff); cumulative S(m) = Σ_{j≤m} z_j² / N vs
        m/N for both models (held-out z); a calibrated model follows the diagonal; 95 % tile-bootstrap bands"""
        PJ = JC["per_junction"]; q = np.array(PJ["c_eff"]); t = np.array(PJ["tile"]); ut = np.unique(t); tix = {u: np.flatnonzero(t == u) for u in ut}
        grid = np.linspace(0, 1, 201); rng_ = np.random.default_rng(SEED_FIG)
        def cum(qq, z2):
            o = np.argsort(qq, kind="stable"); N = len(z2); return np.r_[0, np.arange(1, N + 1) / N], np.r_[0, np.cumsum(z2[o]) / N]
        axE.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1, label="Expected under calibrated uncertainty")
        for lab, colr, ls_, name in (("z_constant", GREY, (0, (3, 2)), "Constant relative uncertainty"), ("z_calibrated", BLUE, "-", "Quality-dependent relative uncertainty")):
            z2 = np.array(PJ[lab]) ** 2; xs, ys = cum(q, z2); B = []
            for _ in range(n_boot):
                ix = np.concatenate([tix[u] for u in rng_.choice(ut, len(ut))]); xb, yb = cum(q[ix], z2[ix]); B.append(np.interp(grid, xb, yb))
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0); axE.fill_between(grid, lo, hi, color=colr, alpha=0.12 if lab == "z_constant" else 0.2, lw=0, zorder=2)
            axE.plot(xs, ys, color=colr, lw=1.4, ls=ls_, zorder=3, label=name)
            log(f"panel a cumulative ({lab}): S at 20/50/80/100 % = " + " ".join(f"{np.interp(f_, xs, ys):.2f}" for f_ in (0.2, 0.5, 0.8, 1.0)))
        axE.set_xlim(0, 1); axE.set_ylim(0, max(1.15, axE.get_ylim()[1])); axE.set_aspect("auto"); axE.tick_params(labelsize=8)
        axE.set_xlabel("Fraction of junctions (lowest predicted error first)", fontsize=9, labelpad=2)
        axE.set_ylabel("Cumulative squared\nstandardized imbalance", fontsize=9, labelpad=2)
        axE.text(0.98, 0.04, f"{len(q):,} junctions\nheld-out recordings (5-fold CV)", transform=axE.transAxes, fontsize=7.5, color=MUTED, ha="right", va="bottom")
        axE.spines[["top", "right"]].set_visible(False); axE.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.1)

    def draw_calibration_bins(axE, nbin=5, n_boot=1000):
        """held-out normalized junction imbalance, RMS z by quintile of predicted junction error (production c_eff ordering), constant
        vs quality-dependent relative error side by side; 95 % tile-bootstrap intervals; ideal = 1"""
        PJ = JC["per_junction"]; q = np.array(PJ["c_eff"]); t = np.array(PJ["tile"]); ut = np.unique(t); tix = {u: np.flatnonzero(t == u) for u in ut}
        edges_ = np.quantile(q, np.linspace(0, 1, nbin + 1)); bi = np.clip(np.searchsorted(edges_, q, side="right") - 1, 0, nbin - 1)
        rng_ = np.random.default_rng(SEED_FIG); xq = np.arange(1, nbin + 1)
        for lab, colr, mk, dx, name in (("z_constant", GREY, "s", -0.13, "Constant relative error"), ("z_calibrated", BLUE, "o", 0.13, "Quality-dependent relative error")):
            z2 = np.array(PJ[lab]) ** 2; rms = np.array([np.sqrt(z2[bi == b].mean()) for b in range(nbin)]); B = []
            for _ in range(n_boot):
                ix = np.concatenate([tix[u] for u in rng_.choice(ut, len(ut))]); B.append([np.sqrt(z2[ix][bi[ix] == b].mean()) for b in range(nbin)])
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
            axE.errorbar(xq + dx, rms, yerr=[rms - lo, hi - rms], fmt=mk, ms=5.5 if mk == "o" else 5, color=colr, mfc=colr if mk == "o" else "white",
                         mec=colr, mew=1.2, elinewidth=1.0, capsize=2, zorder=3, label=name)
            log(f"panel a ({lab}): RMS z by quintile " + " ".join(f"{v:.2f}" for v in rms))
        axE.axhline(1, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1, label="Ideal calibration")
        med_ = [np.median(q[bi == b]) for b in range(nbin)]
        axE.set_xticks(xq); axE.set_xticklabels([f"{m_:.2f}" for m_ in med_]); axE.set_xlim(0.5, nbin + 0.5); axE.set_ylim(0, 1.75); axE.tick_params(labelsize=8)
        axE.set_xlabel("Predicted relative measurement error\n(junction quintiles, median)", fontsize=9, labelpad=2)
        axE.set_ylabel("RMS normalized\njunction imbalance", fontsize=9, labelpad=2)
        axE.spines[["top", "right"]].set_visible(False); axE.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.4, borderaxespad=0.1)

    def draw_calibration_cont(axE, k=900, n_boot=300):
        """held-out standardized junction residuals, rolling RMS over the k junctions nearest in c_eff (flux-weighted effective
        discrepancy score of the production model, same ordering for both models); tile-bootstrap 95 % bands"""
        PJ = JC["per_junction"]; q = np.array(PJ["c_eff"]); t = np.array(PJ["tile"]); ut = np.unique(t); tix = {u: np.flatnonzero(t == u) for u in ut}
        def roll(qq, z2):
            o = np.argsort(qq); qq, z2 = qq[o], z2[o]; cs = np.r_[0, np.cumsum(z2)]; st = max(k // 6, 20)
            a_ = np.arange(0, len(qq) - k + 1, st); return np.array([np.median(qq[i:i + k]) for i in a_]), np.sqrt((cs[a_ + k] - cs[a_]) / k)
        xs0, _ = roll(q, np.ones_like(q)); grid = np.geomspace(xs0[0], xs0[-1], 40); rng_ = np.random.default_rng(SEED_FIG)   # bands on the curve's own range
        for lab, colr, ls_, name in (("z_constant", GREY, (0, (3, 2)), "Constant relative error"), ("z_calibrated", BLUE, "-", "Quality-dependent relative error")):
            z2 = np.array(PJ[lab]) ** 2; xs, ys = roll(q, z2)
            B = []
            for _ in range(n_boot):
                b = np.concatenate([tix[u] for u in rng_.choice(ut, len(ut))]); xb, yb = roll(q[b], z2[b]); B.append(np.interp(grid, xb, yb))
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0); axE.fill_between(grid, lo, hi, color=colr, alpha=0.12 if lab == "z_constant" else 0.2, lw=0, zorder=2)
            axE.plot(xs, ys, color=colr, lw=1.4, ls=ls_, zorder=3, label=name)
            log(f"panel a ({lab}): rolling RMS (k {k}) {ys.min():.2f}–{ys.max():.2f}")
        axE.axhline(1, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1, label="Ideal calibration")
        axE.set_xscale("log"); axE.set_xticks([0.2, 0.3, 0.5, 0.8]); axE.set_xticklabels(["0.2", "0.3", "0.5", "0.8"]); axE.minorticks_off()
        axE.set_xlim(grid[0] * 0.93, grid[-1] * 1.07); axE.set_ylim(0.25, 1.65); axE.tick_params(labelsize=8)
        axE.set_xlabel("Predicted relative measurement error at the junction, $c_{j}$", fontsize=9, labelpad=2)
        axE.set_ylabel("RMS normalized\njunction imbalance", fontsize=9, labelpad=2)
        axE.text(0.98, 0.04, f"held-out recordings (5-fold CV)\n{len(q):,} junctions", transform=axE.transAxes, fontsize=7.5, color=MUTED, ha="right", va="bottom")
        axE.spines[["top", "right"]].set_visible(False); axE.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.1)

    def draw_relative(axR):
        """observed RMS relative junction imbalance |in − out| / √Σy² by quality quintile vs the scale each model predicts (held-out tiles)"""
        R_ = JC["relative"]; xq = np.arange(1, 6); ci_ = np.array(R_["observed_ci"])
        axR.plot(xq, R_["predicted_constant"], color=GREY, lw=1.4, ls=(0, (3, 2)), marker="s", ms=5, label="constant fractional-discrepancy model", zorder=2)
        axR.plot(xq, R_["predicted_calibrated"], color=BLUE, lw=1.6, marker="o", ms=5, mfc="white", mew=1.4, label="quality-dependent model", zorder=3)
        axR.vlines(xq, ci_[:, 0], ci_[:, 1], color=INK, lw=1.2, zorder=4)
        axR.plot(xq, R_["observed"], ls="", marker="o", ms=6.5, color=INK, label="measured imbalance", zorder=5)
        axR.set_xticks(xq); axR.set_xticklabels(["best", "2", "3", "4", "worst"], fontsize=8); axR.set_xlim(0.6, 5.4)
        axR.set_ylim(0, max(1.0, 1.05 * max(R_["predicted_constant"]))); axR.tick_params(labelsize=8)
        axR.set_xlabel("mean recording quality at junction", fontsize=8, labelpad=1)
        axR.set_ylabel("RMS conservation residual\n/ flux scale", fontsize=8, labelpad=1)
        dnll = JC["constant"]["heldout_nll"] - JC["calibrated"]["heldout_nll"]
        axR.text(0.98, 0.03, f"{JC['n_junctions']} same-tile junctions,\nheld-out tiles; constant model\nworse by {dnll:.0f} log-likelihood units", transform=axR.transAxes,
                 fontsize=8, color=MUTED, ha="right", va="bottom")
        axR.spines[["top", "right"]].set_visible(False); axR.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.4, labelspacing=0.25, handletextpad=0.4, borderaxespad=0.1)

    def draw_obs_pred(axR):
        """observed vs predicted relative junction imbalance |r_j| / √Σy² (RMS per bin of held-out prediction); diagonal = perfect calibration"""
        B_ = JC["by_predicted"]["calibrated"]; x = np.array(B_["predicted"]); y = np.array(B_["observed"]); ci_ = np.array(B_["observed_ci"])
        hi = 1.1 * max(x.max(), ci_.max()); axR.plot([0, hi], [0, hi], color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
        axR.text(0.97 * hi, 0.04 * hi, "dashed: perfect calibration", fontsize=8, color=MUTED, ha="right", va="bottom")
        axR.vlines(x, ci_[:, 0], ci_[:, 1], color=BLUE, lw=1.2, zorder=2)
        axR.plot(x, y, ls="", marker="o", ms=5.5, color=BLUE, zorder=3)
        axR.set_xlim(0, hi); axR.set_ylim(0, hi); axR.set_aspect("equal", adjustable="box"); axR.tick_params(labelsize=8)
        axR.set_xlabel("predicted imbalance (RMS)", fontsize=8, labelpad=1)
        axR.set_ylabel("observed imbalance (RMS)", fontsize=8, labelpad=1)
        axR.text(0.03, 0.97, f"{JC['n_junctions']} same-tile junctions,\nheld-out tiles", transform=axR.transAxes, fontsize=8, color=MUTED, ha="left", va="top")
        axR.spines[["top", "right"]].set_visible(False)
        log(f"panel b: held-out NLL constant − quality-dependent {JC['constant']['heldout_nll'] - JC['calibrated']['heldout_nll']:+.0f} (caption)")

    def draw_withheld_fold(axW2):
        """withheld-vessel validation as median fold error exp(median |ln(predicted / recorded)|) vs fraction withheld"""
        WH = json.loads((P.RESULTS / "dc_withholding.json").read_text()); cv = WH["curve"]; rp_ = WH["repeat"]
        x = np.array([100 * c_["frac"] for c_ in cv]); y = np.exp([c_["err"] for c_ in cv]); ci_ = np.exp(np.array([c_["err_ci"] for c_ in cv]))
        axW2.axhspan(*np.exp(rp_["err_ci"]), color=GREY, alpha=0.09, lw=0, zorder=1)
        axW2.axhline(np.exp(rp_["err"]), color="0.65", lw=0.8, ls=(0, (1, 2)), zorder=2, label="Typical repeat-recording disagreement")
        axW2.fill_between(x, ci_[:, 0], ci_[:, 1], color=BLUE, alpha=0.18, lw=0, zorder=2)
        if WH.get("curve_constant"):                                         # same draws, constant relative uncertainty
            cc_ = WH["curve_constant"]; yc = np.exp([c_["err"] for c_ in cc_]); cic = np.exp(np.array([c_["err_ci"] for c_ in cc_]))
            axW2.fill_between(x, cic[:, 0], cic[:, 1], color=GREY, alpha=0.12, lw=0, zorder=2)
            axW2.plot(x, yc, color=GREY, lw=1.2, ls=(0, (3, 2)), marker="s", ms=4, mfc="white", zorder=3, label="Constant relative uncertainty")
        axW2.plot(x, y, color=BLUE, lw=1.3, marker="D", ms=4.5, zorder=3, label="Quality-dependent relative uncertainty")
        axW2.set_xlim(0, 78); axW2.set_ylim(1, 2.25); axW2.set_xticks([0, 25, 50, 75]); axW2.set_yticks([1, 1.25, 1.5, 1.75, 2, 2.25])
        axW2.set_yticklabels(["1", "1.25", "1.5", "1.75", "2", ""]); axW2.tick_params(labelsize=8)
        axW2.set_xlabel("Vessels withheld (%)", fontsize=9, labelpad=2)
        axW2.set_ylabel("Median fold error in\nwithheld-vessel flux", fontsize=9, labelpad=2)
        axW2.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.1); axW2.spines[["top", "right"]].set_visible(False)
        log(f"caption: direction agreement {100 * cv[0]['dir']:.0f} % → {100 * cv[-1]['dir']:.0f} %; repeat ×{np.exp(rp_['err']):.2f}; "
            + ", ".join(f"{100 * c_['frac']:.0f} %: ×{np.exp(c_['err']):.2f}" for c_ in cv))

    def draw_withheld_dir(axW3):
        """direction agreement of withheld vessels vs fraction withheld; repeat-recording reference"""
        WH = json.loads((P.RESULTS / "dc_withholding.json").read_text()); cv = WH["curve"]; rp_ = WH["repeat"]
        x = np.array([100 * c_["frac"] for c_ in cv]); y = 100 * np.array([c_["dir"] for c_ in cv]); ci_ = 100 * np.array([c_["dir_ci"] for c_ in cv])
        axW3.axhspan(*(100 * np.array(rp_["dir_ci"])), color=GREY, alpha=0.09, lw=0, zorder=1)
        axW3.axhline(100 * rp_["dir"], color="0.65", lw=0.8, ls=(0, (3, 2)), zorder=2, label="Repeat recordings")
        axW3.fill_between(x, ci_[:, 0], ci_[:, 1], color=BLUE, alpha=0.18, lw=0, zorder=2)
        axW3.plot(x, y, color=BLUE, lw=1.3, marker="D", ms=4.5, zorder=3, label="Withheld vessels")
        axW3.set_xlim(0, 78); axW3.set_ylim(60, 101); axW3.set_xticks([0, 25, 50, 75]); axW3.tick_params(labelsize=8)
        axW3.set_xlabel("Vessels withheld (%)", fontsize=9, labelpad=2); axW3.set_ylabel("Same flow direction\nas recorded (%)", fontsize=9, labelpad=2)
        axW3.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.1); axW3.spines[["top", "right"]].set_visible(False)

    def draw_withholding(axW2):
        """withheld-vessel error vs fraction of measured vessels withheld (dc_withholding_curve.py)"""
        WH = json.loads((P.RESULTS / "dc_withholding.json").read_text()); cv = WH["curve"]
        x = np.array([100 * c_["frac"] for c_ in cv]); y = np.array([c_["err"] for c_ in cv]); ci_ = np.array([c_["err_ci"] for c_ in cv])
        rp = WH["repeat"]
        axW2.axhspan(*rp["err_ci"], color=GREY, alpha=0.18, lw=0, zorder=1)
        axW2.axhline(rp["err"], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=2)
        axW2.fill_between(x, ci_[:, 0], ci_[:, 1], color=BLUE, alpha=0.18, lw=0, zorder=2)
        axW2.plot(x, y, color=BLUE, lw=1.6, marker="D", ms=5.5, zorder=3, label="reconstructed, vessel withheld")
        from matplotlib.patches import Patch as _Pt
        axW2.legend(handles=[(_Pt(color=BLUE, alpha=0.18, lw=0), matplotlib.lines.Line2D([], [], color=BLUE, lw=1.6, marker="D", ms=5)),
                             (_Pt(color=GREY, alpha=0.18, lw=0), matplotlib.lines.Line2D([], [], color=GREY, lw=1.0, ls=(0, (3, 2))))],
                    labels=["reconstruction, withheld vessels (95 % CI)", "independent recordings of the same vessel"], loc="lower right", frameon=False, fontsize=8, handlelength=2.0, borderaxespad=0.1)
        log(f"caption: direction agreement {100 * cv[0]['dir']:.0f} % (5 % withheld) → {100 * cv[-1]['dir']:.0f} % (75 %)")
        un = D["metrics"]["contracted"]; fu = 100 * un["n_unimaged"] / un["n_vessels"]
        axW2.axvline(fu, color=INK, lw=0.8, ls=":", zorder=1)
        log(f"caption: dotted line = vessels already unmeasured ({fu:.0f} %)")
        axW2.text(fu + 0.8, 0.93, f"never measured ({fu:.0f} %)", fontsize=8, color=MUTED, ha="left", va="top")
        axW2.set_xlim(0, 78); axW2.set_ylim(0, 0.95); axW2.set_xticks([0, 25, 50, 75]); axW2.tick_params(labelsize=8)
        axW2.set_xlabel("measured vessels withheld (%)", fontsize=8, labelpad=1)
        ax2_ = axW2.secondary_yaxis("right", functions=(np.exp, np.log)); ax2_.set_yticks([1.25, 1.5, 2.0, 2.5]); ax2_.set_yticklabels(["×1.25", "×1.5", "×2", "×2.5"])
        ax2_.tick_params(labelsize=8); ax2_.set_ylabel("fold error", fontsize=8, labelpad=2)
        axW2.set_ylabel("median |ln(predicted /\nrecorded flux)|", fontsize=8, labelpad=2)
        axW2.spines[["top", "right"]].set_visible(False)

    def draw_check(axF):
        """prediction of independent recordings (trusted references, c < 0.5; vessel-bootstrap 95 % intervals)"""
        rowsF = (("raw measurement", V["measured"]["raw"]["trusted"], GREY, "o", GREY),
                 ("reconstructed, measured vessel", V["measured"]["calibrated"]["trusted"], BLUE, "o", "white"),
                 ("reconstructed, withheld vessel", V["imputed"]["calibrated"]["trusted"], BLUE, "D", BLUE))
        for k_, (lab, q_, colr, mk_, fc_) in enumerate(rowsF):
            yk = 2 - k_
            axF.plot(q_["err_ci"], [yk, yk], color=colr, lw=2.2, solid_capstyle="butt", alpha=0.45, zorder=2)
            axF.plot(q_["err"], yk, marker=mk_, ms=6.5, mfc=fc_, mec=colr, mew=1.4, ls="", zorder=3)
            axF.text(0.01, yk + 0.3, lab, fontsize=8, color=INK if colr == GREY else BLUE, ha="left", va="center")
            axF.text(0.6, yk, f"{100 * q_['dir']:.0f} %", fontsize=8, color=colr, ha="right", va="center")
            log(f"panel check: {lab}: err {q_['err']:.3f} [{q_['err_ci'][0]:.3f}, {q_['err_ci'][1]:.3f}], same sign {100 * q_['dir']:.1f} %, n {q_['n']} refs / {q_['n_vessels']} vessels")
        axF.text(0.6, 2.62, "same sign", fontsize=8, color=MUTED, ha="right", va="bottom")
        axF.set_yticks([]); axF.set_ylim(-0.45, 2.75); axF.spines[["top", "right", "left"]].set_visible(False)
        axF.set_xlim(0, 0.6); axF.set_xticks([0, 0.2, 0.4]); axF.tick_params(labelsize=8)
        axF.set_xlabel("median |log ratio| to an\nindependent recording", fontsize=8, labelpad=1)

    # d: junction JNODE in tile JTILE — measured fluxes of all incident vessels from one recording
    jinc = [((JNODE, m_), 1.0) if (JNODE, m_) in eidx else ((m_, JNODE), -1.0) for m_ in Hn.neighbors(JNODE)]
    yj = np.array([s_ * M[(JTILE, e)]["Q"] for e, s_ in jinc]); cjk = np.array([c_of(mod, M[(JTILE, e)]) for e, _ in jinc])
    sj = float(np.sqrt(np.sum((cjk * yj) ** 2) + s0 ** 2)); rj = -yj.sum(); qin = -yj[yj < 0].sum()            # r = in − out
    log(f"panel d: junction {JNODE} tile {JTILE}: y (out +) {np.round(yj, 4)}, c {np.round(cjk, 2)}, in − out {rj:+.4f} ({100 * rj / qin:.0f} % of inflow), σ_j {sj:.4f}, z {rj / sj:+.2f}")
    axJ = fig.add_axes([0.345, 0.02, 0.31, TOPB - 0.02]); axJ.set_ylim(-1.5, 1.42); axJ.set_xlim(-1.4, 1.4); axJ.set_aspect("equal", adjustable="datalim"); axJ.axis("off")
    pj = xy(JNODE); jc = TCOL[JTILE]; ymax = np.abs(yj).max()
    for (e, s_), y_, c_ in zip(jinc, yj, cjk):
        other = e[1] if e[0] == JNODE else e[0]; pq = toimg(np.asarray(G.edges[e]["path"], float))
        far = pq[-1] if np.linalg.norm(pq[-1] - pj) > np.linalg.norm(pq[0] - pj) else pq[0]
        u_ = (far - pj) / np.linalg.norm(far - pj); u_ = np.array([u_[0], -u_[1]])                           # image → axes (y up)
        a0, a1 = 0.13 * u_, 0.74 * u_; tail, head = (a0, a1) if y_ > 0 else (a1, a0)                         # y > 0: flow leaves the node
        axJ.annotate("", xy=head, xytext=tail, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", lw=1.2 + 4.0 * abs(y_) / ymax,
                                                                color=jc, shrinkA=0, shrinkB=0), zorder=3)
        lab = 0.86 * u_
        axJ.text(*lab, f"vessel {num[eidx[e]]}\n{abs(y_):.4f} nL/s\nc = {c_:.2f}", fontsize=8, color=INK, ha="center",
                 va="bottom" if u_[1] > 0.3 else ("top" if u_[1] < -0.3 else "center"), linespacing=1.05)
    axJ.add_patch(Circle((0, 0), 0.07, fc="white", ec=INK, lw=0.9, zorder=4))
    axJ.text(0.0, -1.36, f"observed  $r_j$ = in − out = {rj:+.4f} nL/s\npredicted scale  σ$_{{j,\\mathrm{{pred}}}}$ = {sj:.4f} nL/s", fontsize=8, color=INK, ha="center", va="bottom", linespacing=1.15)
    fig.text(0.38, TT, f"One junction, one recording (tile {JTILE})", fontsize=9, color=INK, ha="left", va="bottom")
    fig.canvas.draw(); rr_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); tb = lambda a_: inv.transform(a_.get_tightbbox(rr_))
    xr = min(tb(axW)[0, 0], tb(axS)[0, 0]) - 0.012
    yT = max(tb(axC)[1, 1], tb(axW)[1, 1])                                 # a and b on one line; d, e, f on one line
    yB = TT - 0.004                                                       # d, e, f letters on the bottom-row title line
    for x_, y_, l_ in ((0.005, yT, "a"), (xr, yT, "b"), (xr, tb(axS)[1, 1], "c"), (0.345, yB, "d")):
        fig.text(x_, min(y_ + 0.004, 0.985), l_, **LET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plos_style.save(fig, OUT.parent / "fig6_S_example")                 # SI: the worked example (formerly the main figure)
    log("wrote fig6_S_example")

    # ================================================================ main figure: the network before / after, then the evidence
    #  a measured field (best recording per vessel): colour = reliability c_k, width = |y|, circles = junction imbalance |r_j|
    #  b reconstructed field: width = |Q′| (same scale), colour = correction Q′/y; imputed (unimaged) vessels marked; no imbalance
    #  c junction calibration c_k vs constant c (held-out tiles) · d prediction of independent recordings
    bnodes = {D["nodes"][i] for i in np.r_[D["inj"], D["out"]]}                   # network boundary: flow enters / leaves, no balance imposed
    p0 = np.array([G.nodes[PATCH]["x"], G.nodes[PATCH]["y"]]); inreg = lambda n: np.all(np.abs(np.array([G.nodes[n]["x"], G.nodes[n]["y"]]) - p0) < REGION_HALF)
    reg = [i for i, e in enumerate(E) if inreg(e[0]) and inreg(e[1])]; rnodes = {n for i in reg for n in E[i]}
    rjun = []
    for n in rnodes:
        if Hn.degree(n) < 3 or n in bnodes or not all(inreg(m_) for m_ in Hn.neighbors(n)) or G.nodes[n].get("boundary_type") in ("source", "sink"):
            continue
        inc = [(eidx[(n, m_)], 1.0) if (n, m_) in eidx else (eidx[(m_, n)], -1.0) for m_ in Hn.neighbors(n)]
        if all(im[i_] for i_, _ in inc):
            rjun.append((n, sum(s_ * Qm[i_] for i_, s_ in inc), sum(s_ * Qp[i_] for i_, s_ in inc), len({int(D["tile"][i_]) for i_, _ in inc})))
    rpath = {i: toimg(np.asarray(G.edges[E[i]]["path"], float)) for i in reg}
    exE = [(u, v) for u, v in G.edges() if (inreg(u) or inreg(v)) and (u, v) not in eidx and (v, u) not in eidx and G.edges[u, v].get("path") is not None]
    ijun = [n for n in rnodes if Hn.degree(n) >= 3 and n not in bnodes and all(inreg(m_) for m_ in Hn.neighbors(n))]; rb = [n for n in bnodes if inreg(n)]
    corr = np.where(im, np.log2(np.maximum(np.abs(Qp), 1e-9) / np.maximum(np.abs(Qm), 1e-9)), np.nan); flips = [i for i in reg if im[i] and np.sign(Qp[i]) != np.sign(Qm[i])]
    log(f"main a/b: region ±{REGION_HALF} px around {PATCH}: {len(reg)} vessels ({sum(im[i] for i in reg)} imaged), {len(rjun)} fully measured junctions "
        f"(single-tile {sum(1 for j in rjun if j[3] == 1)}); median |r_j| {np.median([abs(j[1]) for j in rjun]):.4f}, max |Σ Q′| {max(abs(j[2]) for j in rjun):.1e}; "
        f"|correction| median ×{2 ** np.nanmedian(np.abs(corr[reg])):.2f}, sign flips {len(flips)}")
    allq = np.vstack(list(rpath.values())); ctr = (allq.min(0) + allq.max(0)) / 2; hw = 0.5 * (allq.max(0) - allq.min(0)).max() + 12
    # everything that enters the view is drawn (clipped at the frame); junctions = nodes inside the frame
    inbox = lambda q: np.any((np.abs(q[:, 0] - ctr[0]) < hw) & (np.abs(q[:, 1] - ctr[1]) < hw))
    reg = [i for i in range(len(E)) if inbox(toimg(np.asarray(G.edges[E[i]]["path"], float)))]; rpath = {i: toimg(np.asarray(G.edges[E[i]]["path"], float)) for i in reg}
    # unimaged segments shorter than two radii lie inside a junction's lumen (vectorisation splits one crossing into two degree-3
    # nodes): drawn as ONE effective junction.  Exact for the reconstruction — the segment's flow only links the two node balances,
    # so eliminating it = summing them (up to the ε ridge on unimaged vessels).
    SH = {i for i in reg if D["short_unimaged"][i]}                     # network.contract_short_unimaged (stored by reconcile_dc)
    par = {}
    def root(n):
        while par.get(n, n) != n:
            n = par[n]
        return n
    for i in SH:
        a_, b_ = root(E[i][0]), root(E[i][1])
        if a_ != b_:
            par[a_] = b_
    grp = {}
    for n in {n for i in reg for n in E[i]}:
        grp.setdefault(root(n), []).append(n)
    gxy = {k: np.mean([xy(n) for n in v], axis=0) for k, v in grp.items()}
    for i in reg:                                                     # vessel ends meet at the effective junction
        if i in SH:
            continue
        q = rpath[i]; k0, k1 = root(E[i][0]), root(E[i][1])
        if np.linalg.norm(q[0] - xy(E[i][0])) > np.linalg.norm(q[-1] - xy(E[i][0])):
            q = q[::-1]
        if len(grp[k0]) > 1:
            q = np.vstack([gxy[k0], q])
        if len(grp[k1]) > 1:
            q = np.vstack([q, gxy[k1]])
        rpath[i] = q
    reg = [i for i in reg if i not in SH]
    rjun, ijun = [], []
    for k, ns in grp.items():
        if not np.all(np.abs(gxy[k] - ctr) < hw) or any(n in bnodes for n in ns):
            continue
        inc = [(eidx[(n, m_)], 1.0) if (n, m_) in eidx else (eidx[(m_, n)], -1.0) for n in ns for m_ in Hn.neighbors(n)]
        inc = [(i_, s_) for i_, s_ in inc if i_ not in SH]
        if len(inc) < 3:
            continue
        ijun.append(k)
        if all(im[i_] for i_, _ in inc):
            rjun.append((k, sum(s_ * Qm[i_] for i_, s_ in inc), sum(s_ * Qp[i_] for i_, s_ in inc), len({int(D["tile"][i_]) for i_, _ in inc})))
    nin = lambda n: np.all(np.abs(xy(n) - ctr) < hw); rb = [n for n in bnodes if nin(n)]
    log(f"  {len(SH)} unimaged segments < 2 radii drawn as effective junctions ({sum(len(v) > 1 for v in grp.values())} merged junctions)")
    exE = [(u, v) for u, v in G.edges() if (u, v) not in eidx and (v, u) not in eidx and G.edges[u, v].get("path") is not None
           and inbox(toimg(np.asarray(G.edges[u, v]["path"], float)))]
    log(f"  view: {len(reg)} vessels drawn to the frame ({sum(im[i] for i in reg)} imaged), {len(rjun)} fully measured junctions in view, "
        f"{len(exE)} excluded vessels, {len(rb)} boundary nodes; median |r_j| {np.median([abs(j[1]) for j in rjun]):.4f}")
    qref = float(np.percentile(np.abs(np.r_[Qm[[i for i in reg if im[i]]], Qp[reg]]), 95))
    lw_q = lambda q: 0.6 + 4.2 * np.sqrt(min(abs(q) / qref, 1.3))
    cmap_c = mcolors.LinearSegmentedColormap.from_list("rel", ["#1f5fa8", "#7fb0d8", "#e9c46a", "#d1495b", "#8c1c4a"]); norm_c = mcolors.Normalize(0.1, 1.0)
    cmap_r = mcolors.LinearSegmentedColormap.from_list("corr", ["#2d004b", "#8073ac", "#b8b8b8", "#e08214", "#7f3b08"])   # neutral grey stays visible
    norm_r = mcolors.TwoSlopeNorm(0, -1.5, 1.5)
    # reading order = the method: a reliability of one recording → b its calibration at junctions → c measured network →
    # d reconciled network (conservation imposed) → e independent validation on withheld measurements
    HF = 4.95                                                            # SI fig6_S_networks: measured | reconstructed
    figM = plt.figure(figsize=(7.5, HF), facecolor="white")
    W_, H_ = 0.45, 0.45 * 7.5 / HF; YM = 0.8 / HF; XA, XB = 0.03, 0.525
    axA = figM.add_axes([XA, YM, W_, H_]); axB = figM.add_axes([XB, YM, W_, H_])
    crop2 = mos[int(ctr[1] - hw):int(ctr[1] + hw), int(ctr[0] - hw):int(ctr[0] + hw)]; lo2, hi2 = np.percentile(crop2[crop2 > 0], (1, 99.5))
    for ax in (axA, axB):
        ax.imshow(mos, cmap="gray", vmin=lo2, vmax=hi2 * 1.2, interpolation="antialiased", alpha=0.13)
        ax.set_xlim(ctr[0] - hw, ctr[0] + hw); ax.set_ylim(ctr[1] + hw, ctr[1] - hw); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("0.75"); s_.set_linewidth(0.5)
    zc = np.where(im, (np.abs(Qp) - np.abs(Qm)) / D["sigma"], np.nan)          # magnitude correction in units of the vessel's σ
    norm_z = mcolors.TwoSlopeNorm(0, -3, 3); revs = [i for i in reg if im[i] and np.sign(Qp[i]) != np.sign(Qm[i])]
    log(f"panel d: |z| median {np.nanmedian(np.abs(zc[reg])):.2f}, |z| > 2 for {100 * np.nanmean(np.abs(zc[reg]) > 2):.0f} % of measured vessels in view; "
        f"{len(revs)} direction reversals; Spearman(|z|, c) in view {spearmanr(np.abs(zc[[i for i in reg if im[i]]]), D['c'][[i for i in reg if im[i]]])[0]:+.2f}")
    for i in reg:
        q = rpath[i]
        if im[i]:
            axA.plot(q[:, 0], q[:, 1], color=NEUT, lw=lw_q(Qm[i]), solid_capstyle="round", zorder=3)
            axB.plot(q[:, 0], q[:, 1], color=cmap_r(norm_r(corr[i])), lw=lw_q(Qp[i]), solid_capstyle="round", zorder=3)
        else:
            axA.plot(q[:, 0], q[:, 1], color="0.45", lw=0.9, ls=(0, (1.2, 1.2)), zorder=3)
            axB.plot(q[:, 0], q[:, 1], color="black", lw=lw_q(Qp[i]) + 1.4, solid_capstyle="round", zorder=3)
            axB.plot(q[:, 0], q[:, 1], color="white", lw=lw_q(Qp[i]), solid_capstyle="round", zorder=3.1)
    rsc = 260.0 / np.percentile([abs(j[1]) for j in rjun], 90)                                   # circle area ∝ |r_j|
    r75 = np.percentile([abs(j[1]) for j in rjun], 75)                                           # emphasise the largest quarter
    for n, r_, rq_, nt_ in rjun:
        big_ = abs(r_) >= r75
        axA.scatter(*gxy[n], s=rsc * abs(r_), facecolor="none", edgecolor=PINK, linewidth=1.0 if big_ else 0.6, alpha=0.9 if big_ else 0.4, zorder=5)
    for n in ijun:
        axB.scatter(*gxy[n], s=2.5, color="0.35", linewidths=0, zorder=5)
    for i in revs:                                                            # flow direction reversed by the reconciliation
        q = rpath[i]; mq = q[len(q) // 2]; axB.scatter(*mq, s=26, marker="x", color=INK, linewidths=1.1, zorder=6)
    EXSTY = dict(color="0.2", lw=0.9, ls=(0, (2.5, 1.5)), zorder=2.5)                          # not part of the reconstruction
    for u, v in exE:
        q = toimg(np.asarray(G.edges[u, v]["path"], float))
        for ax in (axA, axB):
            ax.plot(q[:, 0], q[:, 1], **EXSTY)
    for n in rb:
        for ax in (axA, axB):
            ax.scatter(*xy(n), s=34, marker="s", facecolor="white", edgecolor=INK, linewidth=1.1, zorder=6)
    # region box on the patch of the SI example (vessels 1–9)
    for ax in (axA, axB):
        L_ = 50 / PX_UM; xb_, yb_ = ctr[0] + hw - 10 - L_, ctr[1] + hw - 10
        ax.add_patch(matplotlib.patches.Rectangle((xb_ - 5, yb_ - 24), L_ + 10, 30, fc="white", ec="none", alpha=0.85, zorder=7.9))
        ax.plot([xb_, xb_ + L_], [yb_, yb_], color=INK, lw=2.2, solid_capstyle="butt", zorder=8)
        ax.text(xb_ + L_ / 2, yb_ - 5, "50 µm", color=INK, fontsize=8, ha="center", va="bottom", zorder=8)
    # keys below the maps (nothing drawn over the network); median values go to the caption
    from matplotlib.lines import Line2D
    med_r = float(np.median([abs(j[1]) for j in rjun])); med_q = float(np.median(np.abs(Qp[reg])))
    log(f"caption: median junction imbalance (measured, in view) {med_r:.3f} nL/s; median vessel |flow| {med_q:.3f} nL/s")
    r_lo, r_hi = float(f"{med_r:.2g}"), float(f"{4 * med_r:.2g}")
    hA = [Line2D([], [], color=NEUT, lw=2.4, label=r"width $\propto$ |measured flux|"),
          Line2D([], [], color="0.45", lw=0.9, ls=(0, (1.2, 1.2)), label="not measured"),
          Line2D([], [], ls="", marker="o", ms=np.sqrt(rsc * r_lo), mfc="none", mec=PINK, mew=1.0, label=f"imbalance {r_lo:g} nL/s"),
          Line2D([], [], ls="", marker="o", ms=np.sqrt(rsc * r_hi), mfc="none", mec=PINK, mew=1.0, label=f"{r_hi:g} nL/s")]
    hA += ([Line2D([], [], ls="", marker="s", ms=6, mfc="white", mec=INK, mew=1.1, label="network boundary")] if rb else [])
    hA += ([Line2D([], [], color=EXSTY["color"], lw=EXSTY["lw"], ls=EXSTY["ls"], label="not reconstructed")] if exE else [])
    figM.legend(handles=hA, loc="upper center", bbox_to_anchor=(XA + W_ / 2, YM - 0.04 / HF), ncol=2, frameon=False, fontsize=8,
               handlelength=2.6, handletextpad=0.6, columnspacing=1.6, labelspacing=1.1)
    caxB = figM.add_axes([XB + 0.12 * W_, YM - 0.24 / HF, 0.76 * W_, 0.075 / HF]); cbB = figM.colorbar(cm.ScalarMappable(norm_r, cmap_r), cax=caxB, orientation="horizontal", extend="both")
    cbB.set_ticks([-1.5, -1, 0, 1, 1.5]); cbB.set_ticklabels(["", "×½", "×1", "×2", ""]); cbB.ax.tick_params(labelsize=8)
    cbB.set_label("reconstructed / measured |flux|", fontsize=8, labelpad=1)
    cbB.ax.text(0.0, 1.15, "decreased", transform=cbB.ax.transAxes, fontsize=8, color="#542788", ha="left", va="bottom")
    cbB.ax.text(1.0, 1.15, "increased", transform=cbB.ax.transAxes, fontsize=8, color="#b35806", ha="right", va="bottom")
    if any(not im[i] for i in reg):
        cbB.ax.text(0.5, -4.0, "width ∝ |reconstructed flux|;  white: not measured;  ×: direction reversed", transform=cbB.ax.transAxes, fontsize=8, color=MUTED, ha="center", va="top")
    TA = YM + H_ + 0.05 / HF
    ntl = len({t_ for i in reg if im[i] for t_, *_ in recs_all.get(i, [])})
    log(f"caption: measured fluxes from {ntl} tiles")
    figM.text(0.025, TA, "Measured fluxes", fontsize=9, color=INK, ha="left", va="bottom")
    figM.text(XB + 0.025, TA, "Reconstructed fluxes", fontsize=9, color=INK, ha="left", va="bottom")
    plos_style.save(figM, OUT.parent / "fig6_S_networks"); log("wrote fig6_S_networks")

    def draw_recon(ax):
        """compact reconstructed field: width ∝ |Q′|, colour = log2(|Q′| / |y|), white = not measured, × = direction reversed"""
        ax.imshow(mos, cmap="gray", vmin=lo2, vmax=hi2 * 1.2, interpolation="antialiased", alpha=0.13)
        ax.set_xlim(ctr[0] - hw, ctr[0] + hw); ax.set_ylim(ctr[1] + hw, ctr[1] - hw); ax.set_aspect("equal", anchor="NW"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("0.75"); s_.set_linewidth(0.5)
        for i in reg:
            q = rpath[i]
            if im[i]:
                ax.plot(q[:, 0], q[:, 1], color=cmap_r(norm_r(corr[i])), lw=lw_q(Qp[i]), solid_capstyle="round", zorder=3)
            else:
                ax.plot(q[:, 0], q[:, 1], color="black", lw=lw_q(Qp[i]) + 1.4, solid_capstyle="round", zorder=3)
                ax.plot(q[:, 0], q[:, 1], color="white", lw=lw_q(Qp[i]), solid_capstyle="round", zorder=3.1)
        for i in revs:
            q = rpath[i]; ax.scatter(*q[len(q) // 2], s=26, marker="x", color=INK, linewidths=1.1, zorder=6)
        for u, v in exE:
            ax.plot(*toimg(np.asarray(G.edges[u, v]["path"], float)).T, **EXSTY)
        L_ = 50 / PX_UM; xb_, yb_ = ctr[0] + hw - 10 - L_, ctr[1] + hw - 10
        ax.add_patch(matplotlib.patches.Rectangle((xb_ - 5, yb_ - 24), L_ + 10, 30, fc="white", ec="none", alpha=0.85, zorder=7.9))
        ax.plot([xb_, xb_ + L_], [yb_, yb_], color=INK, lw=2.2, solid_capstyle="butt", zorder=8)
        ax.text(xb_ + L_ / 2, yb_ - 5, "50 µm", color=INK, fontsize=8, ha="center", va="bottom", zorder=8)
    # a (calibration plot, top right) and c (withheld validation, bottom)
    # ---- main figure, two panels: a calibration of the discrepancy model (with a same-tile junction inset), b prediction of
    #      withheld fluxes.  The reconstructed network itself is shown at organ scale in Fig 7 (pair: SI fig6_S_networks).
    # ---- main figure (11-18): a physiological variability (heart rate) · b measurement uncertainty (alignment) ·
    #      c conservative reconstruction (worked example) · d predictive validation (ablation).  Withholding → SI.
    def draw_hr(ax_t, ax_r):
        from plexusflow.reconcile.gain import tile_heart_rates, fit_hr_gain
        from plexusflow.io.acquisition import tile_acquisition_minutes
        from plexusflow.config import StageConfig as _SC6
        f0_ = tile_heart_rates(G); R0_ = measurement_table(G, gain=None); fit_ = fit_hr_gain(R0_, f0_); ts_ = sorted(f0_)
        tq_ = tile_acquisition_minutes(ST, _SC6.load(ST), ts_)
        xs_ = np.array([tq_[t] for t in ts_]) if tq_ else np.array(ts_, float)
        ax_t.plot(xs_, [f0_[t] for t in ts_], "o", ms=3.2, color=BLUE, mec="white", mew=0.4)
        ax_t.set_xlabel("Acquisition time (min)" if tq_ else "Recording (acquisition order)", fontsize=9, labelpad=2); ax_t.set_ylabel("Heart rate (Hz)", fontsize=9, labelpad=2)
        ax_t.tick_params(labelsize=8); ax_t.spines[["top", "right"]].set_visible(False)
        # the data the exponent is estimated from: pairs of recordings of the same vessel, log flux ratio vs log heart-rate ratio
        by_ = {}
        for (t, e), m in R0_.items():
            if m["Q"] != 0: by_.setdefault(e, {})[t] = abs(m["Q"])
        xr_, yr_ = [], []
        for r in by_.values():
            tt = sorted(r)
            for i in range(len(tt)):
                for j in range(i + 1, len(tt)):
                    a_, b2_ = (tt[i], tt[j]) if f0_[tt[i]] >= f0_[tt[j]] else (tt[j], tt[i])
                    xr_.append(f0_[a_] / f0_[b2_]); yr_.append(r[a_] / r[b2_])
        xr_, yr_ = np.array(xr_), np.array(yr_)
        ax_r.scatter(xr_, yr_, s=1.5, color="0.7", alpha=0.18, lw=0, rasterized=True, zorder=1)
        edges_ = np.array([1.0, 1.03, 1.08, 1.15, 1.3, 1.6, 2.2, 3.5]); bc, bm, bl, bh, bn = [], [], [], [], []
        for lo_, hi_ in zip(edges_[:-1], edges_[1:]):
            m_ = (xr_ >= lo_) & (xr_ < hi_)
            if m_.sum() >= 15:
                bc.append(np.exp(np.median(np.log(xr_[m_])))); q_ = np.log(yr_[m_]); bm.append(np.exp(np.median(q_)))
                bl.append(np.exp(np.percentile(q_, 25))); bh.append(np.exp(np.percentile(q_, 75))); bn.append(int(m_.sum()))
        bc, bm, bl, bh = map(np.array, (bc, bm, bl, bh))
        ax_r.errorbar(bc, bm, yerr=[bm - bl, bh - bm], fmt="o", ms=4.5, color=BLUE, mec="white", mew=0.5, ecolor=BLUE, elinewidth=1, capsize=0, zorder=4, label="binned median (IQR)")
        xx_ = np.linspace(1, xr_.max(), 50); ax_r.plot(xx_, xx_ ** fit_["beta"], color=INK, lw=1.2, zorder=5, label=r"$Q \propto f^{\,%.1f}$" % fit_["beta"])
        ax_r.axhline(1, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=2)
        ax_r.set_xscale("log"); ax_r.set_yscale("log"); ax_r.set_xlim(0.98, xr_.max() * 1.05); ax_r.set_ylim(0.1, 10)
        ax_r.set_xticks([1, 1.5, 2, 3]); ax_r.set_xticklabels(["1", "1.5", "2", "3"]); ax_r.set_yticks([0.1, 1, 10]); ax_r.set_yticklabels(["0.1", "1", "10"]); ax_r.minorticks_off()
        ax_r.set_xlabel("Heart-rate ratio, $f_2/f_1$", fontsize=9, labelpad=2); ax_r.set_ylabel("Flux ratio, $Q_2/Q_1$\n(same vessel)", fontsize=9, labelpad=2)
        ax_r.tick_params(labelsize=8); ax_r.spines[["top", "right"]].set_visible(False); ax_r.legend(loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.3, borderaxespad=0.1)
        Vv, ok_ = list(by_), np.ones(1, bool)
        log(f"panel a: {len(xr_)} repeat pairs; bins (n) {bn}")
        log(f"panel a: β {fit_['beta']:.2f} (f_ref {fit_['f_ref']:.2f} Hz)")
    def draw_norm_effect(axN):
        """held-out prediction error without / with heart-rate normalisation (alignment weighting; dc_ablation.json)"""
        A_ = json.loads((P.RESULTS / "dc_ablation.json").read_text()); V_ = {(v["hr"], v["weighting"], v["reconstruction"]): v for v in A_["variants"]}
        for g_, (rec, lab) in enumerate((("average", "per-vessel\naverage"), ("conservation", "conservative\nreconstruction"))):
            pts = []
            for dx, hr, fc in ((-0.14, False, "white"), (0.14, True, BLUE)):
                v = V_[(hr, "alignment", rec)]; y_ = v["fold_error"]; lo_, hi_ = v["fold_error_ci"]; pts.append((g_ + dx, y_))
                axN.errorbar(g_ + dx, y_, yerr=[[y_ - lo_], [hi_ - y_]], fmt="o", ms=5.5, mfc=fc, mec=BLUE, mew=1.2, ecolor=BLUE, elinewidth=1, capsize=0, zorder=4)
            axN.plot(*zip(*pts), color="0.6", lw=0.8, zorder=2)
        axN.set_xticks([0, 1]); axN.set_xticklabels(["average\nonly", "conser-\nvation"], fontsize=8); axN.set_xlim(-0.5, 1.5)
        axN.set_ylabel("Median fold error,\nheld-out recordings", fontsize=9, labelpad=2); axN.tick_params(labelsize=8); axN.spines[["top", "right"]].set_visible(False)
        from matplotlib.lines import Line2D as _L2n
        axN.legend(handles=[_L2n([], [], ls="", marker="o", ms=5.5, mfc="white", mec=BLUE, mew=1.2, label="raw"),
                            _L2n([], [], ls="", marker="o", ms=5.5, mfc=BLUE, mec=BLUE, label="HR-normalised")],
                   loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0.0, bbox_to_anchor=(1.12, 1.08))

    def draw_validation(axV):
        """held-out prediction (no heart-rate normalisation): constant vs alignment-dependent uncertainty, each as per-vessel
        average only (open) and conservative reconstruction (filled); dc_ablation.json"""
        A_ = json.loads((P.RESULTS / "dc_ablation.json").read_text()); V_ = {(v["hr"], v["weighting"], v["reconstruction"]): v for v in A_["variants"]}
        for g_, wl in enumerate(("constant", "alignment")):
            pts = []
            for dx, rec, fc in ((-0.14, "average", "white"), (0.14, "conservation", BLUE)):
                v = V_[(False, wl, rec)]; y_ = v["fold_error"]; lo_, hi_ = v["fold_error_ci"]; pts.append((g_ + dx, y_))
                axV.errorbar(g_ + dx, y_, yerr=[[y_ - lo_], [hi_ - y_]], fmt="o", ms=5.5, mfc=fc, mec=BLUE, mew=1.2, ecolor=BLUE, elinewidth=1, capsize=0, zorder=4)
            axV.plot(*zip(*pts), color="0.6", lw=0.8, zorder=2)
        WH = json.loads((P.RESULTS / "dc_withholding.json").read_text()); ry = np.exp(WH["repeat"]["err"])
        axV.axhline(ry, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1); axV.text(1.45, ry, "repeat\nrecordings", fontsize=7, color=MUTED, va="center", ha="left")
        axV.set_xticks([0, 1]); axV.set_xticklabels(["constant\nuncertainty", "CV-\ndependent"], fontsize=8); axV.set_xlim(-0.5, 1.4)
        axV.set_ylabel("Median fold error,\nheld-out recordings", fontsize=9, labelpad=2); axV.tick_params(labelsize=8); axV.spines[["top", "right"]].set_visible(False)
        from matplotlib.lines import Line2D as _L2v
        axV.legend(handles=[_L2v([], [], ls="", marker="o", ms=5.5, mfc="white", mec=BLUE, mew=1.2, label="per-vessel average"),
                            _L2v([], [], ls="", marker="o", ms=5.5, mfc=BLUE, mec=BLUE, label="conservative reconstruction")],
                   loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0.0, bbox_to_anchor=(1.05, 1.1))
        log("panel b: " + "; ".join(f"{wl} {rec}: ×{V_[(False, wl, rec)]['fold_error']:.3f}" for wl in ("constant", "alignment") for rec in ("average", "conservation")))

    def draw_ablation(axA):
        A_ = json.loads((P.RESULTS / "dc_ablation.json").read_text()); V_ = {(v["hr"], v["weighting"], v["reconstruction"]): v for v in A_["variants"]}
        groups = [(False, "constant"), (True, "constant"), (False, "alignment"), (True, "alignment")]
        labs = ["constant\nraw", "constant\nHR-norm.", "CV\nraw", "CV\nHR-norm."]
        for g_, (hr, wl) in enumerate(groups):
            pts = []
            for dx, rec, mk, fc in ((-0.13, "average", "o", "white"), (0.13, "conservation", "o", BLUE)):
                v = V_[(hr, wl, rec)]; y_ = v["fold_error"]; lo_, hi_ = v["fold_error_ci"]; pts.append((g_ + dx, y_))
                axA.errorbar(g_ + dx, y_, yerr=[[y_ - lo_], [hi_ - y_]], fmt=mk, ms=6, mfc=fc, mec=BLUE, mew=1.2, ecolor=BLUE, elinewidth=1, capsize=0, zorder=4)
            axA.plot(*zip(*pts), color="0.6", lw=0.8, zorder=2)
        WH = json.loads((P.RESULTS / "dc_withholding.json").read_text())
        axA.axhline(np.exp(WH["repeat"]["err"]), color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
        axA.text(3.45, np.exp(WH["repeat"]["err"]), "repeat\nrecordings", fontsize=7.5, color=MUTED, va="center", ha="left")
        axA.set_xticks(range(4)); axA.set_xticklabels(labs, fontsize=8); axA.set_xlim(-0.5, 3.4)
        axA.set_ylabel("Median fold error,\nheld-out recordings", fontsize=9, labelpad=2); axA.tick_params(labelsize=8); axA.spines[["top", "right"]].set_visible(False)
        from matplotlib.lines import Line2D as _L2a
        axA.legend(handles=[_L2a([], [], ls="", marker="o", ms=6, mfc="white", mec=BLUE, mew=1.2, label="per-vessel average only"),
                            _L2a([], [], ls="", marker="o", ms=6, mfc=BLUE, mec=BLUE, label="conservative reconstruction")],
                   loc="upper right", frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.1, bbox_to_anchor=(1.0, 1.02))
        log(f"panel d: {A_['n_heldout']} held-out recordings ({A_['n_tiles']} tiles); " + "; ".join(f"{k}: ×{V_[k]['fold_error']:.3f}" for k in V_))
    HF = 6.2; fig = plt.figure(figsize=(7.5, HF), facecolor="white")
    yA = 4.05 / HF; hA = 1.75 / HF
    # SI (sensitivity, not applied): heart rate over acquisition + pairwise flux / heart-rate ratios; normalisation effect
    figS = plt.figure(figsize=(7.2, 2.7), facecolor="white"); axS1 = figS.add_axes([0.08, 0.2, 0.25, 0.72]); axS2 = figS.add_axes([0.42, 0.2, 0.25, 0.72])
    draw_hr(axS1, axS2); axS3 = figS.add_axes([0.78, 0.2, 0.2, 0.72]); draw_norm_effect(axS3)
    plos_style.save(figS, OUT.parent / "fig6_S_hr_scaling"); log("wrote fig6_S_hr_scaling")
    # a (left): two real recordings' flux traces Q(t)/mean — a clean and a noisy one (CV near the 10th / 90th percentile of
    #    recordings with 0.15 < |Q| < 0.6 nL/s on the reconstruction network); grey = per-frame flux, colour = mean + 3 cardiac harmonics
    _mg = {(int(m_["tile_id"]), (u_, v_)): m_ for u_, v_, d_ in G.edges(data=True) for m_ in (d_.get("measurements_piv") or [])}
    _cand = sorted([(r_["cv"], k_) for k_, r_ in M.items() if 0.15 < abs(r_["Q"]) < 0.6 and k_[1] in eidx and k_ in _mg
                    and _mg[k_].get("Q_t") is not None and (_mg[k_].get("harm") or {}).get("f0_hz")])
    _cvs = np.array([c_ for c_, _ in _cand])
    def _spiky(key):                                                        # one glitch dominating the residual → not representative
        mg = _mg[key]; q = np.asarray(mg["Q_t"], float); t = np.arange(len(q)) / FS_HZ; w = 2 * np.pi * mg["harm"]["f0_hz"]
        X = np.column_stack([np.ones(len(q))] + [g(k * w * t) for k in (1, 2, 3) for g in (np.cos, np.sin)]); r = q - X @ np.linalg.lstsq(X, q, rcond=None)[0]
        return np.max(np.abs(r)) > 4.5 * np.std(r)
    TRACE = []
    for q_ in (0.1, 0.9):
        for j_ in np.argsort(np.abs(_cvs - np.quantile(_cvs, q_))):
            if not _spiky(_cand[j_][1]):
                TRACE.append(_cand[j_][1]); break
    def draw_trace(ax, key, colr, lab):
        mg = _mg[key]; q = np.asarray(mg["Q_t"], float); f0 = mg["harm"]["f0_hz"]; t = np.arange(len(q)) / FS_HZ; w = 2 * np.pi * f0
        X = np.column_stack([np.ones(len(q))] + [g(k * w * t) for k in (1, 2, 3) for g in (np.cos, np.sin)]); b = np.linalg.lstsq(X, q, rcond=None)[0]
        mu = b[0]; sh = t <= 3.0 / f0                                                  # three beats
        ax.plot(t[sh], q[sh] / mu, color="0.65", lw=0.6, zorder=2)
        ax.plot(t[sh], (X @ b)[sh] / mu, color=colr, lw=1.4, zorder=3)
        ax.axhline(1, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        r_ = M[key]; ax.text(0.99, 0.97, f"CV = {r_['cv']:.2f}   c = {c_of(mod, r_):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=colr,
                             bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
        ax.set_ylim(-0.6, 2.6); ax.set_yticks([0, 1, 2])
        ax.set_title(lab, fontsize=8, color=colr, pad=2, loc="left"); ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        log(f"panel a trace {lab}: tile {key[0]} {key[1]} |Q| {abs(r_['Q']):.3f} CV {r_['cv']:.2f} c {c_of(mod, r_):.2f}")
    _ha = (hA - 0.3 / HF) / 2
    for k_, (key_, lab_, colr_) in enumerate(((TRACE[0], "clean recording", BLUE), (TRACE[1], "noisy recording", PINK))):
        axX = fig.add_axes([0.06, yA + 0.42 / HF + (1 - k_) * (_ha + 0.12 / HF), 0.2, _ha - 0.06 / HF]); draw_trace(axX, key_, colr_, lab_)
        if k_ == 1:
            axX.set_xlabel("time (s)", fontsize=8, labelpad=1)
        else:
            axX.set_xticklabels([])
    fig.text(0.012, yA + 0.42 / HF + _ha, "flux / mean", rotation=90, fontsize=8, ha="center", va="center")
    fig.text(0.16, yA - 0.1 / HF, f"$c = {mod['c0']:.2f} + {mod.get('c_cv', 0):.2f}\\,\\mathrm{{CV}}$", fontsize=8.5, ha="center", va="center")
    axC2 = fig.add_axes([0.335, yA, 0.17, hA]); draw_calibration_cum(axC2)
    axVb = fig.add_axes([0.71, yA, 0.2, hA]); draw_validation(axVb)
    axC2.set_xlabel("Fraction of junctions\n(lowest error first)", fontsize=8.5, labelpad=2); axC2.set_ylabel("Cumulative squared\nimbalance", fontsize=8.5, labelpad=1)
    for t_ in list(axC2.texts): t_.remove()
    axC2.set_xticks([0, 0.5, 1]); axC2.set_xticklabels(["0", "0.5", "1"])
    axC2.legend(handles=axC2.get_legend().legend_handles, labels=["ideal", "constant", "CV-dependent"], loc="upper left", frameon=False, fontsize=7, handlelength=1.5, borderaxespad=0.05)
    figA = plt.figure(figsize=(5.0, 2.6), facecolor="white"); draw_ablation(figA.add_axes([0.15, 0.22, 0.6, 0.72]))   # SI
    plos_style.save(figA, OUT.parent / "fig6_S_ablation"); log("wrote fig6_S_ablation")
    T1, T2 = 6.0 / HF, 3.02 / HF
    figW = plt.figure(figsize=(3.75, 2.6), facecolor="white"); draw_withheld_fold(figW.add_axes([0.2, 0.2, 0.75, 0.72]))           # SI
    plos_style.save(figW, OUT.parent / "fig6_S_withholding"); log("wrote fig6_S_withholding")
    figDir = plt.figure(figsize=(3.6, 2.6), facecolor="white"); draw_withheld_dir(figDir.add_axes([0.2, 0.2, 0.75, 0.72]))   # SI
    plos_style.save(figDir, OUT.parent / "fig6_S_direction"); log("wrote fig6_S_direction")
    # c: worked example — the nine-vessel patch at the tile 22/26/27 overlap: every recording (tile colour), the 1/c²-weighted
    #    aggregate (diamond) and the full-network reconciled flux Q′ (bar), signed along each vessel's reconstructed flow direction
    _saved = (pidx, recs, ptiles, TCOL, order, num)                   # the worked example has its own patch (EX_PATCH): a mixed one
    pn_c = nx.single_source_shortest_path_length(Hn, EX_PATCH, cutoff=PATCH_HOPS)
    pidx = sorted({eidx[e] if e in eidx else eidx[e[::-1]] for e in Hn.subgraph(pn_c).edges()}); recs = {i: [] for i in pidx}
    for (tt, e), m in M.items():
        if e in eidx and eidx[e] in recs:
            recs[eidx[e]].append((int(m["tile"]), abs(m["Q"]), c_of(mod, m), np.sign(m["Q"])))
    ptiles = sorted({r[0] for i in pidx for r in recs[i]}); _pal = iter(["#d95f02", "#e7298a", "#66a61e", "#a6761d"])
    TCOL = {t_: _saved[3][t_] if t_ in _saved[3] else next(_pal) for t_ in ptiles}
    _mid = {i: np.asarray(G.edges[E[i]]["path"], float)[len(G.edges[E[i]]["path"]) // 2] for i in pidx}
    order = sorted(pidx, key=lambda i: (-_mid[i][1], _mid[i][0])); num = {i: k + 1 for k, i in enumerate(order)}
    axP = fig.add_axes([0.025, 0.66 / HF, 0.265, 2.0 / HF]); axQ = fig.add_axes([0.375, 0.5 / HF, 0.605, 2.3 / HF])
    pts_ = np.vstack([toimg(np.asarray(G.edges[E[i]]["path"], float)) for i in pidx]); hh = 0.72 * np.ptp(pts_, axis=0).max(); cx_, cy_ = (pts_.min(0) + pts_.max(0)) / 2 + [0.2 * hh, -0.2 * hh]   # patch lower-left: room for the locator
    cropP = mos[int(cy_ - hh):int(cy_ + hh), int(cx_ - hh):int(cx_ + hh)]
    axP.imshow(mos, cmap="gray", vmin=np.percentile(cropP, 1), vmax=np.percentile(cropP, 99.5), alpha=0.55)
    axP.set_xlim(cx_ - hh, cx_ + hh); axP.set_ylim(cy_ + hh, cy_ - hh); axP.set_aspect("equal"); axP.set_xticks([]); axP.set_yticks([])
    inview = lambda q: np.any((np.abs(q[:, 0] - cx_) < hh) & (np.abs(q[:, 1] - cy_) < hh))
    def parrow(q, fwd, col, hl, ec="none"):                                          # arrowhead only, on the path at mid-length, along its tangent
        if len(q) >= 5:                                                    # smooth the pixel-stepped trace before taking the tangent
            k_ = np.ones(5) / 5; q = np.column_stack([np.convolve(np.pad(q[:, d], 2, mode="edge"), k_, "valid") for d in (0, 1)])
        cl_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; L = cl_[-1]
        hl = min(hl, L / 1.6)                                              # short vessel: smaller head rather than none
        if hl < 2:
            return
        at_ = lambda l: np.array([np.interp(l, cl_, q[:, 0]), np.interp(l, cl_, q[:, 1])])
        m_ = at_(L / 2); w_ = min(0.6 * hl, L / 4); d_ = at_(L / 2 + w_) - at_(L / 2 - w_)
        if np.linalg.norm(d_) == 0 or not np.all(np.abs(m_ - [cx_, cy_]) < hh - hl):
            return                                                         # degenerate, or the head would leave the view
        d_ = (d_ if fwd else -d_) / np.linalg.norm(d_); n_ = np.array([-d_[1], d_[0]])
        tip_ = m_ + 0.5 * hl * d_; base_ = m_ - 0.5 * hl * d_
        axP.add_patch(matplotlib.patches.Polygon([tip_, base_ + 0.42 * hl * n_, base_ - 0.42 * hl * n_], closed=True, fc=col, ec=ec, lw=0.5, zorder=6))
    fps_ = {}
    for t_ in ptiles:                                                     # recording footprints: the patch sits in their overlap
        tg = geom.tiles[int(t_)]; Hv, Wv = tg.video_shape if tg.video_shape else tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{t_}_maxmed.tif")).shape[:2]
        fp = np.array([[0, 0, 1], [0, Wv, 1], [Hv, Wv, 1], [Hv, 0, 1], [0, 0, 1]], float) @ tg.A_full.T
        fp = fp[:, :2] / geom.downsample - [geom.y0, geom.x0]
        axP.plot(fp[:, 1], fp[:, 0], color="white", lw=0.6, ls=(0, (3, 3)), alpha=0.6, zorder=2); fps_[t_] = fp
    CCMAP = matplotlib.colors.LinearSegmentedColormap.from_list("cdisc", plt.cm.plasma(np.linspace(0.0, 0.82, 64)))
    CNORM = matplotlib.colors.LogNorm(0.15, 1.2)                          # vessel colour: c_eff, network 5–95 % range
    pset = set(pidx)
    for i, e in enumerate(E):                                             # rest of the reconstructed network in view: skeleton + flow
        if i in pset:
            continue
        q = toimg(np.asarray(G.edges[e]["path"], float))
        if inview(q):
            axP.plot(q[:, 0], q[:, 1], color="0.45", lw=0.9, solid_capstyle="round", zorder=3)
            parrow(q, Qp[i] >= 0, "0.4", 0.045 * hh)
    for i in pidx:
        q = toimg(np.asarray(G.edges[E[i]]["path"], float)); axP.plot(q[:, 0], q[:, 1], color=CCMAP(CNORM(D["c"][i])), lw=2.6, solid_capstyle="round", zorder=4)
        parrow(q, Qp[i] >= 0, "white", 0.065 * hh, ec=INK)
        k_ = len(q) // 2; d_ = q[min(k_ + 2, len(q) - 1)] - q[max(k_ - 2, 0)]; d_ = d_ / max(np.linalg.norm(d_), 1e-9)
        cands = [q[k_] + 11 * np.array([-d_[1], d_[0]]), q[k_] - 11 * np.array([-d_[1], d_[0]])]
        mq = max(cands, key=lambda c: np.min(np.linalg.norm(pts_ - c, axis=1)))
        axP.text(*mq, str(num[i]), fontsize=7, fontweight="bold", color=INK, ha="center", va="center", zorder=7,
                 bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec=INK, lw=0.5))
    axPI = axP.inset_axes([0.64, 0.64, 0.36, 0.36]); axPI.set_zorder(10)
    _fa = np.vstack(list(fps_.values()))[:, ::-1]; icx_, icy_ = (_fa.min(0) + _fa.max(0)) / 2; ip_ = 0.5 * np.ptp(_fa, axis=0).max() * 1.04   # all footprints                # locator: the three recording footprints and this view
    axPI.imshow(mos, cmap="gray", vmin=np.percentile(cropP, 1), vmax=np.percentile(cropP, 99.5) * 1.2)
    axPI.set_xlim(icx_ - ip_, icx_ + ip_); axPI.set_ylim(icy_ + ip_, icy_ - ip_); axPI.set_xticks([]); axPI.set_yticks([])
    for t_, fp in fps_.items():
        axPI.add_patch(matplotlib.patches.Polygon(fp[:, ::-1], closed=True, fc="white", ec="none", alpha=0.1, zorder=2))
        axPI.plot(fp[:, 1], fp[:, 0], color="white", lw=0.6, alpha=0.7, zorder=3)
    axPI.add_patch(matplotlib.patches.Rectangle((cx_ - hh, cy_ - hh), 2 * hh, 2 * hh, fill=False, ec=INK, lw=1.2, zorder=4))
    for sp_ in axPI.spines.values():
        sp_.set_color("white"); sp_.set_linewidth(1.5)
    axCB = fig.add_axes([0.05, 0.42 / HF, 0.215, 0.07 / HF])
    cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(norm=CNORM, cmap=CCMAP), cax=axCB, orientation="horizontal")
    cb_.set_ticks([0.2, 0.3, 0.5, 0.8]); cb_.set_ticklabels(["0.2", "0.3", "0.5", "0.8"]); cb_.minorticks_off(); axCB.tick_params(labelsize=8, length=2, pad=1)
    cb_.outline.set_linewidth(0.5); axCB.set_xlabel("Estimated relative measurement error, $c_k$", fontsize=8, labelpad=1)
    log("panel c vessel c_eff: " + ", ".join(f"{num[i]}: {D['c'][i]:.2f}" for i in order))
    L_ = 50 / PX_UM; xb_, yb_ = cx_ + hh - 8 - L_, cy_ + hh - 8
    axP.plot([xb_, xb_ + L_], [yb_, yb_], color=INK, lw=2, solid_capstyle="butt", zorder=6); axP.text(xb_ + L_ / 2, yb_ - 4, "50 µm", fontsize=8, ha="center", va="bottom")
    for k_, i in enumerate(order):
        o_ = np.sign(Qp[i]) or 1.0; yrow = len(order) - k_
        for (t_, q_, c_, s_) in sorted(recs[i]):
            axQ.scatter(o_ * s_ * q_, yrow, s=22, facecolor="white", edgecolor="0.6", linewidth=1.0, zorder=3)
        axQ.scatter(o_ * Qm[i], yrow, s=34, marker="D", color=BLUE, zorder=4)
        axQ.plot([o_ * Qp[i]] * 2, [yrow - 0.3, yrow + 0.3], color=INK, lw=2.4, solid_capstyle="butt", zorder=5)
        axQ.plot([o_ * Qm[i], o_ * Qp[i]], [yrow, yrow], color="0.55", lw=0.7, zorder=2.5)   # adjustment: weighted mean → Q′
    axQ.axvline(0, color="0.8", lw=0.6, zorder=1); axQ.set_yticks(range(1, len(order) + 1)); axQ.set_yticklabels([str(num[i]) for i in order[::-1]], fontsize=8)
    axQ.set_ylim(0.4, len(order) + 0.6); axQ.set_xlim(left=min(0, axQ.get_xlim()[0])); axQ.tick_params(labelsize=8)
    axQ.set_ylabel("Vessel", fontsize=9, labelpad=2); axQ.set_xlabel("Mean flux (nL/s)", fontsize=9, labelpad=2)
    axQ.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D as _L2c
    hq = [_L2c([], [], ls="", marker="o", ms=5, mfc="white", mec="0.6", mew=1.0, label="Recorded flux"),
          _L2c([], [], ls="", marker="D", ms=5, color=BLUE, label="Quality-weighted average"), _L2c([], [], color=INK, lw=2.4, label="Reconciled flux Q′")]
    axQ.legend(handles=hq, loc="best", frameon=False, fontsize=8, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.1)
    from collections import Counter as _Cn                                   # highlight one junction of the patch: conservation before / after
    _cnt = _Cn(n for i in pidx for n in E[i]); _jn = [n for n, k in _cnt.items() if k >= 3]
    if _jn:
        _inc = lambda n: [i for i in pidx if n in E[i]]
        n_ = max(_jn, key=lambda n: sum(abs(Qp[i]) for i in _inc(n)))
        into = lambda i, q: (E[i][1] == n_) == (q > 0)                    # flow along E[i] (u→v) enters n_
        ins = [i for i in _inc(n_) if into(i, Qp[i])]; outs = [i for i in _inc(n_) if not into(i, Qp[i])]
        o_ = lambda i: np.sign(Qp[i]) * Qm[i]                                # weighted average along the reconciled direction
        fmt = lambda L_, f_: " + ".join(f"{f_(i):.3f}" for i in L_)
        ltx = lambda L_: " + ".join(f"$Q_{{{num[i]}}}$" for i in L_)
        log(f"panel b junction {n_}: in {[num[i] for i in ins]} out {[num[i] for i in outs]}; averages {fmt(ins, o_)} vs {fmt(outs, o_)}; reconciled {fmt(ins, lambda i: abs(Qp[i]))} = {fmt(outs, lambda i: abs(Qp[i]))}")
    log("panel c: " + "; ".join(f"{num[i]}: recs " + "/".join(f"{np.sign(Qp[i]) * s_ * q_:+.3f}" for (_, q_, _, s_) in sorted(recs[i]))
                                 + f", mean {np.sign(Qp[i]) * Qm[i]:+.3f}, Q′ {abs(Qp[i]):.3f}" for i in order))
    pidx, recs, ptiles, TCOL, order, num = _saved
    # a: one real recording (EX_TILE, EX_EDGE), rotated so the vessel runs left → right along its flow: dense time-mean optical
    #    flow inside the lumen, every Kloosterman cross-section (kept solid, rejected dashed), and the definition of c_k
    from scipy.ndimage import map_coordinates
    from scipy.optimize import curve_fit
    c0f, (pterm, cff), caf = mod["c0"], profile_coef(mod), mod["c_a"]
    klx = pickle.load(gzip.open(ST / "batch_kl_inputs" / f"tile_{EX_TILE}.pkl.gz"))["kl_inputs"]; kx = klx.get(EX_EDGE) or klx.get(EX_EDGE[::-1])
    imx = tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{EX_TILE}_maxmed.tif")).astype(float); imx = imx.mean(-1) if imx.ndim == 3 else imx
    clx = np.asarray(kx["cl_used"], float); offx = np.asarray(kx["roi_offset"], float); Rx = float(kx["R_prior_px"]); vfx = np.asarray(kx["vfield_mean"], float)
    rx_ = kloosterman_scalar(vfx, clx, Rx, mask=kx.get("ridge_mask_roi"), **KW); validx = np.asarray(rx_["valid_mask"], bool)
    r2x = np.clip(np.nan_to_num(np.asarray(rx_["per_slice_r2"], float), nan=0.0), 0, 1)              # section fit quality (0 = no fit)
    mx = M.get((EX_TILE, EX_EDGE)) or M.get((EX_TILE, EX_EDGE[::-1])); fx, ax_, cx = mx["f"], mx["align"], c_of(mod, mx)
    vec_rc = lambda q: np.array([vfx[int(round(q[0])), int(round(q[1]))][1], vfx[int(round(q[0])), int(round(q[1]))][0]])   # (drow, dcol)
    tdir = clx[-1] - clx[0]; tdir /= np.linalg.norm(tdir)
    profx = np.asarray(rx_["per_slice_profiles"], float); rhox = np.asarray(rx_["rho"], float)
    from plexusflow.analyze.centerline import _centerline_geometry
    _, _, nlx = _centerline_geometry(clx)                                    # each section's own normal (the estimator's ρ axis)
    if np.mean([vec_rc(q) @ tdir for q in clx if 0 <= q[0] < vfx.shape[0] - 0.5 and 0 <= q[1] < vfx.shape[1] - 0.5]) < 0:
        clx, validx, r2x, profx, nlx, tdir = clx[::-1], validx[::-1], r2x[::-1], profx[::-1], nlx[::-1], -tdir   # flow runs left → right
    sgx = np.sign(np.nanmedian(np.asarray(rx_["per_slice_Q"], float))) or 1.0            # axial velocity positive along the flow
    ndir = np.array([-tdir[1], tdir[0]]); org = clx[0]; us_ = (clx - org) @ tdir; vs_ = (clx - org) @ ndir; Lx = float(us_[-1])
    nfr = np.column_stack([nlx @ tdir, nlx @ ndir])                          # section normals in the drawing frame
    inside = lambda q: 0 <= q[0] < vfx.shape[0] - 0.5 and 0 <= q[1] < vfx.shape[1] - 0.5
    # 3-D view: the straightened vessel is the x–y plane (image, cross-section ticks shaded by fit R², flow arrows); three
    # cross-sections (good / middle / poor fit) stand on it — measured axial velocity (points) and the fitted parabola
    u_ = np.arange(-0.6 * Rx, Lx + 0.6 * Rx, 1.0); v_ = np.arange(-2.0 * Rx, 2.0 * Rx + 0.1, 1.0); UU, VV = np.meshgrid(u_, v_)
    rot = map_coordinates(imx, [org[0] + UU * tdir[0] + VV * ndir[0] + offx[0], org[1] + UU * tdir[1] + VV * ndir[1] + offx[1]], order=1)
    gim = np.clip(0.35 + 0.6 * (rot - np.percentile(rot, 1)) / (np.percentile(rot, 99.5) - np.percentile(rot, 1)), 0, 1)
    cand = np.arange(8, len(us_) - 8); picks = []
    for tgt in (0.95, 0.6, 0.25):                                              # spread along the vessel
        ok_ = [i for i in cand if all(abs(i - a_) > len(us_) // 5 for a_ in picks) and np.isfinite(profx[i]).sum() > 8]
        picks.append(min(ok_, key=lambda i: abs(r2x[i] - tgt)))
    picks = sorted(picks, key=lambda i: us_[i]); poi = lambda x, v0, Rf: v0 * np.maximum(0.0, 1 - (x / Rf) ** 2)
    zs = 2.6 * Rx / np.nanmax([np.nanmax(sgx * profx[i]) for i in picks]); cmap_r2 = mcolors.LinearSegmentedColormap.from_list("r2", ["#d9d9d9", BLUE])
    figP = plt.figure(figsize=(7.5, 3.4), facecolor="white")                 # SI fig6_S_profile_quality (Fig 3 territory)
    axS = figP.add_axes([0.1, -0.05, 0.8, 1.0], projection="3d", computed_zorder=False)
    axS.plot_surface(UU, VV, np.zeros_like(UU), facecolors=plt.cm.gray(gim), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=0)
    vmx = np.nanmax(np.linalg.norm(vfx, axis=-1))
    for uu in np.arange(4, Lx - 12, 16.0):
        for vv in (-0.5 * Rx, 0.5 * Rx):
            q = org + uu * tdir + vv * ndir
            if inside(q):
                w_ = vec_rc(q); a_, b_ = w_ @ tdir, w_ @ ndir; sc_ = 9.0 / max(np.hypot(a_, b_), 1e-12)          # direction only (alignment); fixed length
                axS.quiver(uu, vv, 0.05, a_ * sc_, b_ * sc_, 0, color="#ffc300", lw=1.3, arrow_length_ratio=0.35, zorder=2)
    axS.plot(us_, vs_, np.full_like(us_, 0.03), color="white", lw=0.8, ls=(0, (3, 2)), zorder=2.5)    # refined centreline
    for i in picks:
        p_ = sgx * profx[i]; okp = np.isfinite(p_) & (np.abs(rhox) <= 1.6 * Rx); good = r2x[i] >= 0.5; col_ = BLUE if good else "#c0392b"
        X_ = lambda rr: us_[i] + rr * nfr[i, 0]; Y_ = lambda rr: vs_[i] + rr * nfr[i, 1]      # section line: centreline point + ρ·normal
        x0 = us_[i]
        axS.plot(X_(np.array([-1.6, 1.6]) * Rx), Y_(np.array([-1.6, 1.6]) * Rx), [0, 0], color=INK, lw=0.8, zorder=3)
        for rr, zz in zip(rhox[okp], p_[okp] * zs):
            axS.plot([X_(rr)] * 2, [Y_(rr)] * 2, [0, max(zz, 0)], color="0.6", lw=0.3, zorder=3)
        axS.scatter(X_(rhox[okp]), Y_(rhox[okp]), p_[okp] * zs, s=3, color=INK, depthshade=False, zorder=5)
        ft = np.isfinite(p_) & (np.abs(rhox) <= Rx)
        try:
            (v0_, Rf_), _ = curve_fit(poi, rhox[ft], p_[ft], p0=[np.nanmax(p_), Rx], bounds=([0, 0.5 * Rx], [np.inf, 2 * Rx]))
            xx = np.linspace(-Rf_, Rf_, 60); axS.plot(X_(xx), Y_(xx), poi(xx, v0_, Rf_) * zs, color=col_, lw=1.5, zorder=6)
        except Exception:
            pass
        axS.text(x0, vs_[i], max(np.nanmax(p_) * zs, 0) + 0.35 * Rx, f"$R^2$ {r2x[i]:.2f}", fontsize=8, color=col_, ha="center", zorder=7)
    L5 = 20 / PX_UM; axS.plot([Lx - L5, Lx], [-1.85 * Rx] * 2, [0.05, 0.05], color="white", lw=2, solid_capstyle="butt", zorder=4)
    axS.text(Lx - L5 / 2, -1.85 * Rx, 0.2, "20 µm", color="white", fontsize=7, ha="center", va="bottom", zorder=4,
             path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")])
    axS.set_box_aspect((Lx + 1.2 * Rx, 4 * Rx, 3.2 * Rx), zoom=1.3); axS.view_init(elev=22, azim=-58); axS.set_axis_off()
    axS.set_xlim(u_[0], u_[-1]); axS.set_ylim(v_[0], v_[-1]); axS.set_zlim(0, 3.2 * Rx)
    log(f"panel a: tile {EX_TILE} {EX_EDGE}: kept {validx.sum()}/{len(validx)} (f {fx:.3f}), align {ax_:.3f}, c {cx:.3f}; example sections "
        + ", ".join(f"{int(i)} (R² {r2x[i]:.2f})" for i in picks))
    figP.text(0.03, 0.94, f"parabolicity $p$ = {np.mean(r2x):.2f} (mean section $R^2$)    alignment $a$ = {ax_:.2f}    →  $c$ = {cx:.2f}", fontsize=9, color=INK, ha="left", va="center")
    plos_style.save(figP, OUT.parent / "fig6_S_profile_quality"); log("wrote fig6_S_profile_quality")
    # a: one junction whose vessels were all measured in the same tile — the calibration signal (tile image, refined
    #    centrelines, flow directions, measured fluxes and each recording's p, a, c; r_j and the predicted σ_j)
    kl22 = pickle.load(gzip.open(ST / "batch_kl_inputs" / f"tile_{JTILE}.pkl.gz"))["kl_inputs"]
    imj = tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{JTILE}_maxmed.tif")).astype(float); imj = imj.mean(-1) if imj.ndim == 3 else imj
    pn = graph_to_tile_batch(np.array([[G.nodes[JNODE]["x"], G.nodes[JNODE]["y"]]], float), geom, JTILE)[0]
    cls = []
    for (e, s_), y_, c_ in zip(jinc, yj, cjk):
        ki = kl22.get(e) or kl22.get(e[::-1]); q_ = np.asarray(ki["cl_used"], float) + np.asarray(ki["roi_offset"], float)
        if np.linalg.norm(q_[0] - pn) > np.linalg.norm(q_[-1] - pn):
            q_ = q_[::-1]                                                      # from the junction outward
        cls.append((q_, y_, c_, M[(JTILE, e)]))
    hj = min(1.6 * max(np.linalg.norm(q_ - pn, axis=1).max() for q_, *_ in cls), 110)
    figJ = plt.figure(figsize=(3.2, 3.0), facecolor="white"); axJ = figJ.add_axes([0.03, 0.17, 0.94, 0.8]); r0j, c0j = int(pn[0] - hj), int(pn[1] - hj)
    cropj = imj[max(r0j, 0):int(pn[0] + hj), max(c0j, 0):int(pn[1] + hj)]
    axJ.imshow(cropj, cmap="gray", extent=[max(c0j, 0), max(c0j, 0) + cropj.shape[1], max(r0j, 0) + cropj.shape[0], max(r0j, 0)],
               vmin=np.percentile(cropj, 1), vmax=np.percentile(cropj, 99.5), alpha=0.85)
    axJ.set_xlim(pn[1] - hj, pn[1] + hj); axJ.set_ylim(pn[0] + hj, pn[0] - hj); axJ.set_aspect("equal", anchor="C"); axJ.set_xticks([]); axJ.set_yticks([])
    HAL = [matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")]
    for q_, y_, c_, m_ in cls:
        axJ.plot(q_[:, 1], q_[:, 0], color="#ffd23f", lw=1.6, zorder=3)
        k_ = int(0.55 * (len(q_) - 1)); a_, b_ = (q_[k_ - 3], q_[k_ + 3]) if y_ > 0 else (q_[k_ + 3], q_[k_ - 3])   # y > 0: leaves the junction
        axJ.annotate("", xy=(b_[1], b_[0]), xytext=(a_[1], a_[0]), arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", color="#ffd23f", lw=1.6), zorder=4)
        far = q_[-1]; d_ = (far - pn) / np.linalg.norm(far - pn); lab = pn + 0.68 * hj * d_
    axJ.scatter(pn[1], pn[0], s=22, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=6)
    L5 = 20 / PX_UM; axJ.plot([pn[1] + hj - 6 - L5, pn[1] + hj - 6], [pn[0] + hj - 6] * 2, color="white", lw=2, solid_capstyle="butt", zorder=6)
    pJ = axJ.get_position(); qin_ = -yj[yj < 0].sum()
    yin_ = [abs(v_) for v_ in yj if v_ < 0]; yout_ = [v_ for v_ in yj if v_ > 0]
    figJ.text(0.5, 0.06, f"in {sum(yin_):.3f}, out {sum(yout_):.3f} nL/s;  $r_j$ = {abs(rj):.3f} nL/s", fontsize=8, color=INK, ha="center", va="center")
    plos_style.save(figJ, OUT.parent / "fig6_S_junction"); log("wrote fig6_S_junction (example same-tile junction)")
    log(f"caption: junction in {sum(yin_):.3f} ({' + '.join(f'{v_:.3f}' for v_ in yin_)}), out {sum(yout_):.3f}, |r_j| {abs(rj):.3f} ({100 * abs(rj) / qin_:.0f} %)")
    log(f"panel a formula (caption): c = {c0f:.2f} + {cff:.2f}(1 − {pterm}) + {caf:.2f}(1 − a) = {cx:.2f}; p = {np.mean(r2x):.3f} (stored {mx.get('p', float('nan')):.3f})")
    for x_, y_, l_ in ((0.0, T1, "a"), (0.6, T1, "b"), (0.0, T2, "c")):  # panel letters only
        fig.text(x_, y_ - 0.004, l_, **LET)
    plos_style.save(fig, OUT)
    figD = plt.figure(figsize=(3.6, 3.2), facecolor="white"); draw_obs_pred(figD.add_axes([0.2, 0.17, 0.75, 0.75]))   # SI: diagonal calibration
    plos_style.save(figD, OUT.parent / "fig6_S_calibration_diagonal"); log("wrote fig6_S_calibration_diagonal")
    log(f"wrote {OUT}.png/.pdf")

    # ---- SI: the four weightings (formerly main panel d) — reconciliation and imputation error, trusted references
    fw = plt.figure(figsize=(7.5, 2.9), facecolor="white"); axd = fw.add_axes([0.09, 0.2, 0.88, 0.68])
    kinds = ("uniform", "quality", "measured_scale", "calibrated"); xl = ("Uniform", "Quality", "Relative", "Calibrated")
    xs = np.arange(len(kinds)); raw = V["measured"]["raw"]["trusted"]
    axd.axhline(raw["err"], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axd.text(-0.38, raw["err"] + 0.012, "repeat-recording disagreement", fontsize=8, color=GREY, va="bottom", ha="left")
    for test, dx, mk, lab in (("measured", -0.12, "o", "Reconciliation (vessel used)"), ("imputed", 0.12, "D", "Imputation (vessel withheld)")):
        r_ = [V[test][k]["trusted"] for k in kinds]; e_ = np.array([q["err"] for q in r_]); ci_ = np.array([q["err_ci"] for q in r_])
        cl = [BLUE if k == "calibrated" else GREY for k in kinds]
        for x_, (lo_, hi_), c_ in zip(xs + dx, ci_, cl):
            axd.plot([x_, x_], [lo_, hi_], color=c_, lw=1.0, zorder=2)
        axd.scatter(xs + dx, e_, marker=mk, s=46, facecolors=["white" if test == "measured" else c_ for c_ in cl], edgecolors=cl, linewidths=1.4, zorder=3)
        axd.scatter([], [], marker=mk, s=40, facecolors="white" if test == "measured" else GREY, edgecolors=GREY, linewidths=1.4, label=lab)
    axd.set_xticks(xs); axd.set_xticklabels(xl, fontsize=9); axd.get_xticklabels()[-1].set_color(BLUE); axd.get_xticklabels()[-1].set_fontweight("bold")
    for x_, w_ in zip(xs, ("all equal", "1/c²", "1/(cy)²", "1/(cQ′)²")):
        axd.annotate(w_, (x_, 0), xycoords=("data", "axes fraction"), xytext=(0, -20), textcoords="offset points", ha="center", va="top",
                     fontsize=8, color=BLUE if x_ == xs[-1] else GREY)
    axd.annotate("weight:", (0, 0), xycoords="axes fraction", xytext=(-4, -20), textcoords="offset points", fontsize=8, color=GREY, ha="right", va="top")
    axd.set_xlim(-0.45, len(kinds) - 0.55); axd.set_ylim(0, 0.75); axd.set_ylabel("median |log ratio| to an\nindependent recording", fontsize=8, labelpad=2)
    axd.tick_params(labelsize=8); axd.legend(loc="upper right", frameon=False, fontsize=8, handletextpad=0.3, ncol=2)
    axd.spines[["top", "right"]].set_visible(False)
    plos_style.save(fw, OUT.parent / "fig6_S_weighting"); log("wrote fig6_S_weighting")

    # ---- SI: the full-tile correction map with the example junction (formerly the main panels)
    sel_t = np.array([i for i in pe if im[i]])
    log(f"tile {TILE}: relative change median {np.median(np.abs(rel[sel_t])):.2f}, >50 % {np.mean(np.abs(rel[sel_t]) > 0.5):.0%}, reversed {rev[sel_t].mean():.1%}; q0 {q0:.4f}")
    fig = plt.figure(figsize=(7.5, 5.3), facecolor="white")  # SI
    gs = fig.add_gridspec(1, 2, left=0.015, right=0.985, top=0.93, bottom=0.17, width_ratios=[1.12, 1.0], wspace=0.08)
    # a: where the reconstruction changes the measurements
    axM = fig.add_subplot(gs[0]); cor_norm = mcolors.TwoSlopeNorm(0, -1.0, 1.0); cor_cmap = matplotlib.colormaps["RdBu_r"]
    axM.imshow(img * 0.55, cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest"); axM.set_xlim(c0, c1); axM.set_ylim(r1, r0)
    axM.set_xticks([]); axM.set_yticks([])
    for s_ in axM.spines.values():
        s_.set_color("#bdbdbd"); s_.set_linewidth(0.5)
    for e, q in path.items():
        if e not in ps:
            axM.plot(q[:, 1], q[:, 0], color="white", lw=0.4, alpha=0.3, zorder=2)
    for i in pe:
        e = E[i]; q = path[e]; lw = lw_of(e)
        if not im[i]:
            axM.plot(q[:, 1], q[:, 0], color="#c8c8c8", lw=lw, ls=(0, (1.0, 1.0)), zorder=3); continue
        axM.plot(q[:, 1], q[:, 0], color=cor_cmap(cor_norm(np.clip(rel[i], -1, 1))), lw=lw + 0.4, zorder=4, solid_capstyle="round")
        if rev[i]:
            axM.plot(q[:, 1], q[:, 0], color=INK, lw=0.9, ls=(0, (2.5, 1.5)), zorder=5)
    axM.scatter(jpos[1], jpos[0], s=230, marker="s", facecolor="none", edgecolor="white", linewidth=3.2, zorder=8)
    axM.scatter(jpos[1], jpos[0], s=230, marker="s", facecolor="none", edgecolor="#f0ad00", linewidth=1.6, zorder=9)
    scalebar(axM, box, um=100)
    axM.set_title("Where the reconstruction changes the measured flows", fontsize=9, pad=3, loc="left")
    fig.canvas.draw(); PM = axM.get_position()
    cbx = fig.add_axes([PM.x0 + 0.1 * PM.width, PM.y0 - 0.045, 0.8 * PM.width, 0.014])
    cb = plt.colorbar(cm.ScalarMappable(norm=cor_norm, cmap=cor_cmap), cax=cbx, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_xticklabels(["−100%", "−50%", "0", "+50%", "+100%"]); cb.ax.tick_params(labelsize=8, length=2, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("change in flow magnitude, (|Q′| − |y|) / |y|", fontsize=8, labelpad=1)
    from matplotlib.lines import Line2D
    ky = PM.y0 - 0.125; kx = PM.x0 + 0.02
    fig.add_artist(Line2D([kx, kx + 0.03], [ky, ky], color="#c8c8c8", lw=2.2, ls=(0, (1, 1)), transform=fig.transFigure))
    fig.text(kx + 0.036, ky, "not imaged (inferred)", fontsize=8, va="center")
    fig.add_artist(Line2D([kx + 0.24, kx + 0.27], [ky, ky], color=INK, lw=0.9, ls=(0, (2.5, 1.5)), transform=fig.transFigure))
    fig.text(kx + 0.276, ky, "direction reversed", fontsize=8, va="center")
    fig.add_artist(Line2D([kx + 0.45], [ky], marker="s", ms=8, mfc="none", mec="#f0ad00", mew=1.6, ls="", transform=fig.transFigure))
    fig.text(kx + 0.465, ky, "junction in b", fontsize=8, va="center")

    # b: the marked junction, measured vs reconstructed, branch width ∝ flow
    gR = gs[1].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
    gb = gR[0].subgridspec(1, 2, wspace=0.0); axJs = [fig.add_subplot(gb[0]), fig.add_subplot(gb[1])]
    pn = np.array([G.nodes[NODE]["x"], G.nodes[NODE]["y"]], float); dirs = []
    for e_, s_ in jn:
        nb_ = e_[1] if e_[0] == NODE else e_[0]; d_ = np.array([G.nodes[nb_]["x"], G.nodes[nb_]["y"]], float) - pn; th = np.arctan2(-d_[1], d_[0])
        dirs.append(np.array([np.cos(th), np.sin(th)]))
    wmax = max(np.abs(y).max(), np.abs(yp).max())
    for ax, vals, title in ((axJs[0], y, "Measured"), (axJs[1], yp, "Reconstructed")):
        ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.1, 2.3); ax.set_aspect("equal"); ax.axis("off")
        for k, (u, v) in enumerate(zip(dirs, vals)):
            cl = PINK if cj[k] >= 0.3 else BLUE
            ax.plot([0, 1.35 * u[0]], [0, 1.35 * u[1]], color=cl, lw=1.5 + 9 * abs(v) / wmax, solid_capstyle="butt", alpha=0.85, zorder=2)
            tip = 1.55 * u if v > 0 else 0.55 * u; tail = 0.95 * u if v > 0 else 1.45 * u            # s·y > 0: out of the junction
            ax.annotate("", xy=tip, xytext=tail, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=9), zorder=4)
            lab = f"{abs(v):.3f}"
            if ax is axJs[1]:
                dq = abs(v) - abs(y[k]); lab += f"\n({dq:+.3f})"
            ax.text(*(1.95 * u), lab, ha="center", va="center", fontsize=8, color=cl, linespacing=1.0, fontweight="bold" if cj[k] >= 0.3 else "normal")
        ax.scatter([0], [0], s=28, color=INK, zorder=5); ax.set_title(title, fontsize=9, pad=0)
    ins = [k for k in range(3) if y[k] < 0]; outs = [k for k in range(3) if y[k] > 0]
    for ax, vals, rel_ in ((axJs[0], y, "≠"), (axJs[1], yp, "=")):
        ax.text(0, -2.05, f"in {' + '.join(f'{abs(vals[k]):.3f}' for k in ins)} {rel_} out {' + '.join(f'{abs(vals[k]):.3f}' for k in outs)}",
                ha="center", va="bottom", fontsize=8, color=PINK if rel_ == "≠" else INK)
    fig.text((axJs[0].get_position().x0 + axJs[1].get_position().x1) / 2, axJs[0].get_position().y0 - 0.02,
             "higher-uncertainty branch (c = %.2f) changes most" % cj.max(), ha="center", va="top", fontsize=8, color=PINK)

    # c: independent validation (results/dc_weighting_validation.json, scripts/dc_validate_weighting.py)
    V = json.loads((P.RESULTS / "dc_weighting_validation.json").read_text())
    axd = fig.add_subplot(gR[1]); pos = axd.get_position(); axd.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    kinds = ("uniform", "quality", "measured_scale", "calibrated"); xl = ("Uniform", "Quality", "Relative", "Calibrated")
    xs = np.arange(len(kinds)); raw = V["measured"]["raw"]["trusted"]
    axd.axhline(raw["err"], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axd.text(-0.38, raw["err"] + 0.012, "repeat-recording disagreement", fontsize=7.5, color=GREY, va="bottom", ha="left")
    for test, dx, mk, lab in (("measured", -0.15, "o", "measured"), ("imputed", 0.15, "D", "withheld")):
        r_ = [V[test][k]["trusted"] for k in kinds]; e_ = np.array([q["err"] for q in r_]); ci_ = np.array([q["err_ci"] for q in r_])
        cl = [BLUE if k == "calibrated" else GREY for k in kinds]
        axd.vlines(xs + dx, ci_[:, 0], ci_[:, 1], color=cl, lw=1.3, zorder=2)
        axd.scatter(xs + dx, e_, marker=mk, s=36, facecolors=["white" if test == "measured" else c_ for c_ in cl], edgecolors=cl, linewidths=1.3, zorder=3)
        axd.scatter([], [], marker=mk, s=36, facecolors="white" if test == "measured" else GREY, edgecolors=GREY, linewidths=1.3, label=lab + " vessels")
    axd.set_xticks(xs); axd.set_xticklabels(xl, fontsize=8); axd.get_xticklabels()[-1].set_color(BLUE); axd.get_xticklabels()[-1].set_fontweight("bold")
    axd.set_xlim(-0.45, len(kinds) - 0.55); axd.set_ylim(0, 0.75)
    axd.set_ylabel("error vs independent\nrecordings, median |log ratio|", fontsize=8, labelpad=2); axd.tick_params(labelsize=8)
    axd.legend(loc="lower left", frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.1)
    axd.set_title("Prediction of independent recordings", **TIT)
    for s_ in ("top", "right"):
        axd.spines[s_].set_visible(False)

    fig.canvas.draw(); rr_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); tb = lambda a_: inv.transform(a_.get_tightbbox(rr_))
    yT = tb(axM)[1, 1]; xr = min(tb(axJs[0])[0, 0], tb(axd)[0, 0]) - 0.01
    for x_, y_, l_ in ((0.005, yT, "a"), (xr, yT, "b"), (xr, tb(axd)[1, 1], "c")):
        fig.text(x_, min(y_ + 0.004, 0.985), l_, **LET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plos_style.save(fig, OUT.parent / "fig6_S_corrections")
    log("wrote fig6_S_corrections")

    # SI: the reliability score c — (a) two measurements, (b) held-out junction calibration, (c) known truth
    fs2 = plt.figure(figsize=(7.5, 3.3), facecolor="white")
    g2 = fs2.add_gridspec(1, 3, left=0.015, right=0.985, top=0.88, bottom=0.2, width_ratios=[1.5, 1, 1], wspace=0.45)
    ga = g2[0].subgridspec(1, 2, wspace=0.05); axA = []
    for k, (e, lab, colr) in enumerate(((HIGH, "Lower uncertainty", BLUE), (LOW, "Higher uncertainty", PINK))):
        ax = fs2.add_subplot(ga[k]); x = ex[e]; draw_measurement(ax, img, x["ki"], colr, x["valid"]); axA.append(ax)
        ax.set_title(lab, fontsize=8.5, color=colr, pad=2)
        ax.text(0.5, -0.04, f"c = {x['c']:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=colr, fontweight="bold")
        ax.text(0.5, -0.19, (f"CV {x['cv']:.2f}" if profile_coef(mod)[0] == "cv" else f"{profile_coef(mod)[0]} {x['f']:.2f}, align {x['align']:.2f}"), transform=ax.transAxes, ha="center", va="top", fontsize=8)
    pa0, pa1 = axA[0].get_position(), axA[1].get_position()
    fs2.text((pa0.x0 + pa1.x1) / 2, 0.04, ((r"$c = %.2f + %.2f\,\mathrm{CV}$" % (c0_, profile_coef(mod)[1])) if profile_coef(mod)[0] == "cv" else
              (r"$c = %.2f + %.2f\,(1 - " + profile_coef(mod)[0] + r") + %.2f\,(1 - \mathrm{align})$") % (c0_, profile_coef(mod)[1], mod["c_a"])), ha="center", fontsize=8.5)
    axc = fs2.add_subplot(g2[1]); R_ = cal["rows"]; lo_, hi_ = R_[:, 0].min() / 1.6, R_[:, 0].max() * 1.6
    axc.plot([lo_, hi_], [lo_, hi_], color=GREY, lw=0.8, ls=(0, (3, 2)))
    axc.errorbar(R_[:, 0], R_[:, 1], yerr=[R_[:, 1] - R_[:, 2], R_[:, 3] - R_[:, 1]], fmt="o", ms=4, color=BLUE, mec="white", mew=0.5, ecolor=BLUE, capsize=0)
    axc.set_xscale("log"); axc.set_yscale("log"); axc.set_xlim(lo_, hi_); axc.set_ylim(lo_, hi_)
    axc.set_xlabel("predicted imbalance (nL/s)", fontsize=8); axc.set_ylabel("observed imbalance, RMS (nL/s)", fontsize=8); axc.tick_params(labelsize=8)
    axc.set_title("Held-out junctions", fontsize=8.5, loc="left"); axc.spines[["top", "right"]].set_visible(False)
    T_ = json.loads((P.RESULTS / "c_truth_calibration.json").read_text()); bns = T_["bins"]; axb = fs2.add_subplot(g2[2])
    xb_ = np.array([q["c_mean"] for q in bns]); yb_ = np.array([q["rms"] for q in bns]); cib = np.array([q["rms_ci"] for q in bns])
    axb.plot([0, 1.3], [0, 1.3], color=GREY, lw=0.8, ls=(0, (3, 2)))
    axb.errorbar(xb_, yb_, yerr=[yb_ - cib[:, 0], cib[:, 1] - yb_], fmt="o", ms=4, color=BLUE, mec="white", mew=0.5, ecolor=BLUE, capsize=0)
    axb.set_xlim(0, 1.3); axb.set_ylim(0, 1.8); axb.set_xlabel("reliability score c", fontsize=8); axb.set_ylabel("true fractional error, RMS", fontsize=8)
    axb.tick_params(labelsize=8); axb.spines[["top", "right"]].set_visible(False); axb.set_title("Known truth (synthetic videos)", fontsize=8.5, loc="left")
    fs2.canvas.draw(); rr2 = fs2.canvas.get_renderer(); inv2 = fs2.transFigure.inverted(); tb2 = lambda a_: inv2.transform(a_.get_tightbbox(rr2))
    for x_, l_ in ((0.005, "a"), (tb2(axc)[0, 0] - 0.015, "b"), (tb2(axb)[0, 0] - 0.015, "c")):
        fs2.text(x_, 0.93, l_, fontsize=12, fontweight="bold")
    plos_style.save(fs2, OUT.parent / "fig6_S_reliability")

    # supplement: (a) the five tile folds on the mosaic, (b) held-out standardized residuals
    fs = plt.figure(figsize=(7.5, 3.4), facecolor="white"); gS = fs.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.30,
                                                                               left=0.02, right=0.98, top=0.93, bottom=0.25)
    ax = fs.add_subplot(gS[0]); fuse = tifffile.imread(str(ST / "fuse" / "maxmed.tif")).astype(float)
    fuse = fuse.mean(-1) if fuse.ndim == 3 else fuse
    ax.imshow(fuse, cmap="gray", vmin=np.percentile(fuse[fuse > 0], 1), vmax=np.percentile(fuse[fuse > 0], 99.5), interpolation="antialiased")
    FC = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]; ds = geom.downsample; polys, cen, fk, tid = [], [], [], []
    for t, k in sorted(cal["fold"].items()):
        tg = geom.tiles[int(t)]; H, W = tg.video_shape if tg.video_shape else tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{t}_maxmed.tif")).shape[:2]
        cn = np.array([[0, 0, 1], [0, W, 1], [H, W, 1], [H, 0, 1]], float) @ tg.A_full.T
        rc = np.c_[cn[:, 0] / ds - geom.y0, cn[:, 1] / ds - geom.x0]; polys.append(rc); cen.append(rc.mean(0)); fk.append(k); tid.append(t)
    # each pixel → nearest tile centre among tiles covering it (clean cells instead of stacked overlaps)
    from matplotlib.path import Path as MPath
    from scipy.spatial import cKDTree
    st_ = 4; rr, cc = np.mgrid[0:fuse.shape[0]:st_, 0:fuse.shape[1]:st_]; pts = np.c_[rr.ravel(), cc.ravel()].astype(float)
    inside = np.array([MPath(pg).contains_points(pts) for pg in polys])            # (n_tiles, n_pts)
    d2 = ((pts[None, :, :] - np.array(cen)[:, None, :]) ** 2).sum(-1); d2[~inside] = np.inf
    own = np.where(inside.any(0), d2.argmin(0), -1).reshape(rr.shape)
    rgba = np.zeros(own.shape + (4,)); fold_px = np.where(own >= 0, np.array(fk)[np.maximum(own, 0)], -1)
    for k in range(len(FC)):
        rgba[fold_px == k] = (*mcolors.to_rgb(FC[k]), 0.42)
    edge = np.zeros(own.shape, bool); edge[:-1] |= own[:-1] != own[1:]; edge[:, :-1] |= own[:, :-1] != own[:, 1:]
    rgba[edge & (own >= 0)] = (1, 1, 1, 0.8)
    ax.imshow(rgba, extent=(-0.5, fuse.shape[1] - 0.5 + (st_ - 1), fuse.shape[0] - 0.5 + (st_ - 1), -0.5), interpolation="nearest")
    for t, c_ in zip(tid, cen):
        ax.text(c_[1], c_[0], str(t), ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2, foreground="black")])
    allc = np.vstack(polys); ax.set_xlim(allc[:, 1].min() - 10, allc[:, 1].max() + 10); ax.set_ylim(allc[:, 0].max() + 10, allc[:, 0].min() - 10)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for k in range(len(FC)):
        ax.plot([], [], "s", ms=6, color=FC[k], alpha=0.75, label=f"fold {k + 1}: {sum(v == k for v in cal['fold'].values())} tiles, {cal['n_per_fold'][k]} junctions")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=2, frameon=False, fontsize=8, handletextpad=0.2, columnspacing=1.0)
    ax.text(-0.01, 1.02, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax = fs.add_subplot(gS[1]); zh = cal["z"]
    ax.hist(zh, bins=np.linspace(-4.5, 4.5, 55), density=True, color="#8fb3d9", edgecolor="white", linewidth=0.4)
    xx = np.linspace(-4.5, 4.5, 400); ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.0)
    for v in (-1.96, 1.96):
        ax.axvline(v, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.98, 0.97, f"RMS = {np.sqrt(np.mean(zh ** 2)):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlim(-4.5, 4.5); ax.set_yticks([]); ax.set_xlabel(r"standardized junction residual $z_j$ (held-out tiles)", fontsize=8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.text(-0.02, 1.02, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    so = OUT.parent / "fig6_S_heldout"
    plos_style.save(fs, so)
    log(f"wrote {so}.png/.pdf")


if __name__ == "__main__":
    main()
