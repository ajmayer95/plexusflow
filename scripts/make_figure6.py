"""Fig 6 — DC reconciliation: measurement reliability → held-out calibration → weighted conservative projection → result and
validation.  PLOS: 7.5 in wide, all text ≥ 8 pt Arial.

  a  reliable vs unreliable Kloosterman measurement (tile 30): kept cross-sections, time-mean OF arrows; c, f, align under
     each; the reliability formula with the fitted coefficients (results/reliability_model_v1.json).
  b  tile-held-out (5-fold) calibration: RMS predicted junction uncertainty σ_j vs RMS observed imbalance r_j.
  c  the projection (min Σ w (q − y)² s.t. B_int q = 0, w = 1/(c² Q′² + s0²)) and tile-30 junction 28329 before / after
     (network-reconciled values from results/dc_reconciled_v2.pkl).
  d, e  measured / reconciled |flux| in tile 30 (junction-imbalance circles on d).
  f  validation against independent repeat recordings (results/dc_weighting_validation.json, dc_validate_weighting.py):
     error of measured (reconciled) and hidden (imputed) vessels for four weightings, trusted references c < 0.5.

Run: python3 -u scripts/make_figure6.py   (writes figures/fig6/fig6.{png,pdf,tif} and the SI figure fig6_S_heldout)
"""
import sys, json, gzip, pickle, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors, matplotlib.cm as cm, matplotlib.patheffects
import plexusflow.paths as P
from plexusflow.reconcile.reliability import measurement_table, same_tile_junctions, c_of
from plexusflow.analyze.kloosterman import kloosterman_scalar
from plexusflow.analyze.centerline import _centerline_geometry
from plexusflow.analyze.geometry import load_tile_geometry, graph_to_tile_batch
from plexusflow.config import StageConfig

ST = P.STAGE
_ap = argparse.ArgumentParser(description="Fig 6v5 (DC reliability and reconciliation)")
_ap.parse_args()
OUT = P.FIGURES / "fig6" / "fig6"
Z_S = 5.0                                         # junction-circle area per z²
TILE, NODE = 30, 28329
PATCH, PATCH_HOPS = 59219, 2                      # Fig 6a: 9 vessels each recorded in ≥ 2 of tiles 22/26/27; Q′ within every vessel's recordings
import os as _os
PATCH = int(_os.environ.get("FIG6_PATCH", PATCH)); PATCH_HOPS = int(_os.environ.get("FIG6_PATCH_HOPS", PATCH_HOPS))
HIGH, LOW = (28304, 28329), (35992, 36030)
PX_UM = 1.7
KW = dict(dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="median_band", median_lo=0.5, median_hi=1.5, R_slice_lo=0.5,
          R_slice_hi=2.0, min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)
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
            jt_.append(t); jQ_.append(np.abs(y)); jf_.append([M[(t, e)]["f"] for e, _ in j])
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
        assert abs(s * sg * D["Qm"][i] - yy) < 1e-9
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
        ex[e] = dict(ki=ki, valid=valid, f=m["f"], align=m["align"], c=c_of(mod, m))
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
    fig = plt.figure(figsize=(7.5, 6.6), facecolor="white")
    gt = fig.add_gridspec(1, 2, left=0.02, right=0.985, top=0.95, bottom=0.44, width_ratios=[0.9, 1.1], wspace=0.14)
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
    for i in pidx:
        q = toimg(np.asarray(G.edges[E[i]]["path"], float))
        if np.linalg.norm(q[0] - xy(E[i][0])) > np.linalg.norm(q[-1] - xy(E[i][0])):
            q = q[::-1]
        axC.plot(q[:, 0], q[:, 1], color="black", lw=7.0, zorder=4, solid_capstyle="round"); axC.plot(q[:, 0], q[:, 1], color="#ffd400", lw=5.0, zorder=5, solid_capstyle="round")
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; a_ = int(np.searchsorted(cum, 0.5 * cum[-1]))
        lo_i, hi_i = max(a_ - 2, 0), min(a_ + 2, len(q) - 1); d_ = q[hi_i] - q[lo_i]
        if Qp[i] < 0:
            d_ = -d_
        lab_xy = mids[i]
        if np.linalg.norm(d_) > 0:
            u_ = d_ / np.linalg.norm(d_)
            i0, i1 = int(np.searchsorted(cum, 0.22 * cum[-1])), min(int(np.searchsorted(cum, 0.78 * cum[-1])), len(q) - 1)
            seg = q[i0:i1 + 1] if Qp[i] >= 0 else q[i0:i1 + 1][::-1]                    # shaft follows the vessel, head at the downstream end
            for col_, lw_ in (("black", 1.8),):
                axC.plot(seg[:-1, 0], seg[:-1, 1], color=col_, lw=lw_, solid_capstyle="round", zorder=6)
            if len(seg) >= 3:
                tip_, back_ = seg[-1], seg[-3]
            else:                                                                  # short vessel: its own chord, downstream end
                a0_, a1_ = (q[0], q[-1]) if Qp[i] >= 0 else (q[-1], q[0]); tip_, back_ = a0_ + 0.85 * (a1_ - a0_), a0_
            dv_ = tip_ - back_; dv_ = dv_ / max(np.linalg.norm(dv_), 1e-9); pv_ = np.array([-dv_[1], dv_[0]])
            HL_, HW_ = 11.0, 5.5                                                   # head length / half-width in mosaic pixels
            tri_ = np.array([tip_, tip_ - HL_ * dv_ + HW_ * pv_, tip_ - HL_ * dv_ - HW_ * pv_])
            axC.add_patch(matplotlib.patches.Polygon(tri_, closed=True, fc="black", ec="white", lw=0.6, zorder=7))
            nrm = np.array([-u_[1], u_[0]]); cands = [q[a_] + 13 * nrm, q[a_] - 13 * nrm]      # number beside the vessel, not on it
            lab_xy = max(cands, key=lambda c: min([np.linalg.norm(c - o) for o in placed] + [1e9]))
        placed.append(lab_xy)
        axC.text(*lab_xy, str(num[i]), ha="center", va="center", fontsize=9, color=INK, fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.6))
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
        q = sgn * np.asarray(m["Q_t"], float); tt = np.arange(len(q)) / _FS; sel = tt <= 2.0; used = t_ == int(D["tile"][wv])
        axW.plot(tt[sel], q[sel], color=TCOL[t_], lw=1.4 if used else 0.9, alpha=1.0 if used else 0.7, zorder=3 if used else 2)
        axW.axhline(q.mean(), color=TCOL[t_], lw=1.0, ls=(0, (3, 2)), zorder=2)
    axW.axhline(abs(Qp[wv]), color=INK, lw=2.0, zorder=4)
    axW.text(1.98, abs(Qp[wv]) * 0.93, "reconstructed mean flow", fontsize=8, color=INK, va="top", ha="right", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    axW.set_xlim(0, 2.0); axW.set_xlabel("time since recording start (s)", fontsize=8, labelpad=1); axW.set_ylabel("flow (nL/s)", fontsize=8, labelpad=1)
    axW.tick_params(labelsize=8); axW.spines[["top", "right"]].set_visible(False)
    axW.set_title(f"Repeated recordings of vessel {num[wv]} (dashed: means)", fontsize=9, loc="left", pad=3)
    log(f"  waveform panel: vessel {num[wv]} {E[wv]}, means " + ", ".join(f"{r[0]}: {r[1]:.3f}" for r in sorted(recs[wv])) + f"; Q′ {abs(Qp[wv]):.3f}")

    # c: all nine vessels — selected recording, held-out recordings, reconstruction
    axS = fig.add_subplot(gr[1]); xs_ = np.arange(1, len(order) + 1)
    p_ = axS.get_position(); axS.set_position([p_.x0 + 0.05, p_.y0, p_.width - 0.05, p_.height])
    for k, i in enumerate(order):
        rs = recs[i]; offs = np.linspace(-0.18, 0.18, len(rs)) if len(rs) > 1 else [0.0]
        used = int(D["tile"][i])                                             # the winner (smallest c): the only recording the reconstruction uses
        assert used in [r[0] for r in rs] and min(rs, key=lambda r: r[2])[0] == used, (i, used, rs)
        for (t_, q_, c_, s_), dx in zip(sorted(rs), offs):
            axS.scatter(k + 1 + dx, q_, s=48 if t_ == used else 26, facecolors=TCOL[t_] if t_ == used else "white", edgecolors=TCOL[t_], linewidths=1.3, zorder=3)
        axS.plot([k + 1 - 0.38, k + 1 + 0.38], [abs(Qp[i])] * 2, color=INK, lw=3.2, zorder=4, solid_capstyle="butt")
    axS.set_yscale("log"); axS.set_xticks(xs_); axS.set_xticklabels([str(x) for x in xs_], fontsize=8); axS.set_xlim(0.4, len(order) + 0.6)
    axS.set_xlabel("Vessel ID", fontsize=8, labelpad=1); axS.set_ylabel("mean flow (nL/s)", fontsize=8, labelpad=1); axS.tick_params(labelsize=8)
    axS.spines[["top", "right"]].set_visible(False); axS.set_title("Conservation-consistent reference flows", fontsize=9, loc="left", pad=3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="", ms=7, color=TCOL[t_], label=f"Tile {t_}") for t_ in ptiles]
    hs += [Line2D([], [], marker="o", ls="", ms=7, color="0.35", label="Used"),
           Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec="0.35", mew=1.3, label="Independent repeat"),
           Line2D([], [], color=INK, lw=3.2, label="Reconstructed mean")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.405), ncol=6, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.4)

    # d: independent validation (results/dc_weighting_validation.json, scripts/dc_validate_weighting.py)
    V = json.loads((P.RESULTS / "dc_weighting_validation.json").read_text())
    axd = fig.add_axes([0.09, 0.07, 0.88, 0.25])
    kinds = ("uniform", "quality", "measured_scale", "calibrated"); xl = ("Uniform", "Quality", "Relative", "Calibrated")
    xs = np.arange(len(kinds)); raw = V["measured"]["raw"]["trusted"]
    axd.axhline(raw["err"], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axd.text(-0.38, raw["err"] - 0.015, "two recordings of the same vessel", fontsize=8, color=GREY, va="top", ha="left")
    for test, dx, mk, lab in (("measured", -0.12, "o", "measured vessels"), ("imputed", 0.12, "D", "withheld vessels")):
        r_ = [V[test][k]["trusted"] for k in kinds]; e_ = np.array([q["err"] for q in r_]); ci_ = np.array([q["err_ci"] for q in r_])
        cl = [BLUE if k == "calibrated" else GREY for k in kinds]
        axd.vlines(xs + dx, ci_[:, 0], ci_[:, 1], color=cl, lw=1.4, zorder=2)
        axd.scatter(xs + dx, e_, marker=mk, s=64, facecolors=["white" if test == "measured" else c_ for c_ in cl], edgecolors=cl, linewidths=1.5, zorder=3)
        axd.scatter([], [], marker=mk, s=50, facecolors="white" if test == "measured" else GREY, edgecolors=GREY, linewidths=1.5, label=lab)
    axd.set_xticks(xs); axd.set_xticklabels(xl, fontsize=9); axd.get_xticklabels()[-1].set_color(BLUE); axd.get_xticklabels()[-1].set_fontweight("bold")
    axd.set_xlim(-0.45, len(kinds) - 0.55); axd.set_ylim(0, 0.75)
    axd.set_ylabel("median absolute\nlog-ratio error", fontsize=8, labelpad=2); axd.tick_params(labelsize=8)
    axd.legend(loc="upper right", frameon=False, fontsize=8, handletextpad=0.3, ncol=2)
    axd.set_title("Held-out prediction error", fontsize=9, pad=4, loc="left")
    for s_ in ("top", "right"):
        axd.spines[s_].set_visible(False)
    fig.canvas.draw(); rr_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); tb = lambda a_: inv.transform(a_.get_tightbbox(rr_))
    xr = min(tb(axW)[0, 0], tb(axS)[0, 0]) - 0.012
    for x_, y_, l_ in ((0.005, tb(axC)[1, 1], "a"), (xr, tb(axW)[1, 1], "b"), (xr, tb(axS)[1, 1], "c"), (0.005, tb(axd)[1, 1], "d")):
        fig.text(x_, min(y_ + 0.004, 0.985), l_, **LET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plos_style.save(fig, OUT)
    log(f"wrote {OUT}.png/.pdf")

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
    axd.text(-0.38, raw["err"] - 0.015, "two recordings of the same vessel", fontsize=8, color=GREY, va="top", ha="left")
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
        ax.text(0.5, -0.19, f"f {x['f']:.2f}, align {x['align']:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    pa0, pa1 = axA[0].get_position(), axA[1].get_position()
    fs2.text((pa0.x0 + pa1.x1) / 2, 0.04, r"$c = %.2f + %.2f\,(1 - f) + %.2f\,(1 - \mathrm{align})$" % (c0_, mod["c_f"], mod["c_a"]), ha="center", fontsize=8.5)
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
