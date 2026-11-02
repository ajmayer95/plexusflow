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
    # three ideas: calibrate the uncertainty (a) → project onto conservation with it (b, c) → validate independently (d).
    # The two measurement examples (old panel a) are the SI figure fig6_S_examples.
    BLUE, GREY, PINK = "#1f5fa8", "0.55", "#c51b8a"
    fig = plt.figure(figsize=(7.5, 6.3), facecolor="white")
    LET = dict(fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")
    TIT = dict(fontsize=9, color=INK, pad=4, loc="left")
    top = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.93, bottom=0.585, width_ratios=[1, 1.55], wspace=0.22)

    # a: held-out calibration
    axa = fig.add_subplot(top[0]); R_ = cal["rows"]
    lo_, hi_ = R_[:, 0].min() / 1.6, R_[:, 0].max() * 1.6
    axa.plot([lo_, hi_], [lo_, hi_], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axa.errorbar(R_[:, 0], R_[:, 1], yerr=[R_[:, 1] - R_[:, 2], R_[:, 3] - R_[:, 1]], fmt="o", ms=5, color=BLUE, mec="white", mew=0.6,
                 ecolor=BLUE, elinewidth=1.0, capsize=0, zorder=3)
    axa.set_xscale("log"); axa.set_yscale("log"); axa.set_xlim(lo_, hi_); axa.set_ylim(lo_, hi_); axa.set_aspect("equal")
    axa.set_xlabel("predicted junction uncertainty (nL/s)", fontsize=8, labelpad=1)
    axa.set_ylabel("observed junction imbalance,\nRMS (nL/s)", fontsize=8, labelpad=1); axa.tick_params(labelsize=8)
    axa.set_title("Uncertainty calibration (held-out tiles)", **TIT)
    axa.text(0.04, 0.96, r"$\sigma_j^2 = \sum_{k\in j} c_k^2\,y_k^2 + \sigma_0^2$", transform=axa.transAxes, ha="left", va="top", fontsize=9)
    axa.text(0.04, 0.84, r"$c_k$: per-vessel reliability", transform=axa.transAxes, ha="left", va="top", fontsize=8, color=GREY)
    for s_ in ("top", "right"):
        axa.spines[s_].set_visible(False)

    # b: projection — one equation, the real junction measured → reconciled
    axB = fig.add_subplot(top[1]); axB.axis("off"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_title("Conservative projection", **TIT)
    axB.text(0.5, 0.93, r"$\mathbf{Q}' = \arg\min_{\mathbf{q}}\ \sum_k\, (q_k - y_k)^2 / \sigma_k^2$", ha="center", va="center", fontsize=10, color=INK)
    axB.text(0.5, 0.81, "subject to inflow = outflow at every junction", ha="center", va="center", fontsize=8, color=GREY)
    pn = np.array([G.nodes[NODE]["x"], G.nodes[NODE]["y"]], float); ang = []
    for e_, s_ in jn:
        nb_ = e_[1] if e_[0] == NODE else e_[0]; d_ = np.array([G.nodes[nb_]["x"], G.nodes[nb_]["y"]], float) - pn; ang.append(np.arctan2(-d_[1], d_[0]))
    weak = int(np.argmax(cj))

    def junction(ax, vals, title, show_c):
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.75, 2.45); ax.set_aspect("equal"); ax.axis("off")
        for k, (th, v) in enumerate(zip(ang, vals)):
            u = np.array([np.cos(th), np.sin(th)]); cl = PINK if k == weak else BLUE
            ax.plot([0, 1.25 * u[0]], [0, 1.25 * u[1]], color="0.85", lw=7, solid_capstyle="round", zorder=1)
            a0, a1 = (0.15 * u, 1.15 * u) if v > 0 else (1.15 * u, 0.15 * u)          # s·y > 0 = flow out of the node
            ax.annotate("", xy=a1, xytext=a0, arrowprops=dict(arrowstyle="-|>", color=cl, lw=2.0, mutation_scale=16, shrinkA=0, shrinkB=0), zorder=3)
            lab = f"{abs(v):.3f}" + (f"\n(c = {cj[k]:.2f})" if show_c else "")
            ax.text(*(1.75 * u), lab, ha="center", va="center", fontsize=8, color=cl, linespacing=1.0,
                    fontweight="bold" if k == weak else "normal")
        ax.scatter([0], [0], s=30, color=INK, zorder=4)
        tot = abs(np.sum(vals))
        ax.text(0, -2.25, title, ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(0, -2.72, f"imbalance {tot:.3f} nL/s" if tot > 5e-4 else "imbalance 0", ha="center", va="bottom", fontsize=8,
                color=PINK if tot > 5e-4 else INK)
    pB = axB.get_position(); w_ = 0.36 * pB.width
    axJ1 = fig.add_axes([pB.x0 + 0.04 * pB.width, pB.y0 - 0.03, w_, 0.78 * pB.height]); junction(axJ1, y, "Measured y", True)
    axJ2 = fig.add_axes([pB.x1 - 0.04 * pB.width - w_, pB.y0 - 0.03, w_, 0.78 * pB.height]); junction(axJ2, yp, "Reconciled Q′", False)
    fig.add_artist(matplotlib.patches.FancyArrowPatch((pB.x0 + 0.43 * pB.width, pB.y0 + 0.39 * pB.height), (pB.x0 + 0.57 * pB.width, pB.y0 + 0.39 * pB.height),
                                                      transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=1.6, color=GREY))
    axB.text(0.5, 0.5, "least reliable\nbranch moves most", ha="center", va="center", fontsize=8, color=PINK, linespacing=1.0)

    # c: measured / reconciled maps (tile 30), one colour bar
    geom = load_tile_geometry(StageConfig.load(ST)); E = [tuple(e) for e in D["edges"]]
    tile_xy = lambda xy: graph_to_tile_batch(np.asarray(xy, float), geom, TILE)
    r0, r1, c0, c1 = box = (0, img.shape[0], 0, img.shape[1])
    inside = lambda q: np.any((q[:, 0] > r0) & (q[:, 0] < r1) & (q[:, 1] > c0) & (q[:, 1] < c1))
    path = {}
    for u, v, d in G.edges(data=True):
        q = tile_xy(d["path"]) if d.get("path") is not None else None
        if q is not None and inside(q):
            path[(u, v)] = q
    pe = [i for i, e in enumerate(E) if e in path]; ps = set(E[i] for i in pe)
    Qm, Qp, im = D["Qm"], D["Qp"], D["imaged"].astype(bool)
    rad = {e: G.edges[e].get("radius", 5.0) for e in path}; rmax = np.percentile(list(rad.values()), 95)
    lw_of = lambda e: 0.6 + 1.8 * min(rad[e] / rmax, 1.0)
    qa = np.abs(np.r_[Qm[[i for i in pe if im[i]]], Qp[pe]]); qa = qa[qa > 0]
    mag_norm = mcolors.LogNorm(vmin=np.percentile(qa, 3), vmax=np.percentile(qa, 99))
    mag_cmap = mcolors.ListedColormap(matplotlib.colormaps["viridis"](np.linspace(0.18, 1.0, 256)))
    vlo, vhi = np.percentile(img, 1), np.percentile(img, 99.7) * 1.25
    jpos = tile_xy([[G.nodes[NODE]["x"], G.nodes[NODE]["y"]]])[0]

    def draw_map(ax, kind, title):
        ax.imshow(img * 0.55, cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#bdbdbd"); s_.set_linewidth(0.5)
        for e, q in path.items():
            if e not in ps:
                ax.plot(q[:, 1], q[:, 0], color="white", lw=0.4, alpha=0.3, zorder=2)
        for i in pe:
            e = E[i]; q = path[e]; lw = lw_of(e)
            if kind == "meas" and not im[i]:
                ax.plot(q[:, 1], q[:, 0], color="#d0d0d0", lw=lw, ls=(0, (1.0, 1.0)), zorder=3); continue
            col = mag_cmap(mag_norm(max(abs(Qm[i] if kind == "meas" else Qp[i]), mag_norm.vmin)))
            ax.plot(q[:, 1], q[:, 0], color=col, lw=lw, zorder=4, solid_capstyle="round")
        ax.scatter(jpos[1], jpos[0], s=90, marker="o", facecolor="none", edgecolor=PINK, linewidth=1.4, zorder=8)
        ax.set_title(title, fontsize=9, color=INK, pad=3)
        return ax

    bot = fig.add_gridspec(1, 3, left=0.02, right=0.985, top=0.47, bottom=0.12, width_ratios=[1, 1, 1.75], wspace=0.05)
    axs = [draw_map(fig.add_subplot(bot[k]), kd, tt) for k, (kd, tt) in enumerate((("meas", "Measured"), ("rec", "Reconciled")))]
    scalebar(axs[0], box, um=100)
    fig.canvas.draw()
    P0, P1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([P0.x0 + 0.08 * P0.width, P0.y0 - 0.035, P1.x1 - P0.x0 - 0.16 * P0.width, 0.011])
    cb = plt.colorbar(cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap), cax=cax, orientation="horizontal")
    cb.set_label("|flux| (nL/s);  dotted: not imaged", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8, length=2, pad=1, which="both")
    cb.outline.set_linewidth(0.4)

    # d: independent validation (results/dc_weighting_validation.json, scripts/dc_validate_weighting.py)
    V = json.loads((P.RESULTS / "dc_weighting_validation.json").read_text())
    gd = bot[2].subgridspec(1, 1); axd = fig.add_subplot(gd[0]); pos = axd.get_position()
    axd.set_position([pos.x0 + 0.075, pos.y0 + 0.015, pos.width - 0.075, pos.height - 0.015])
    kinds = ("uniform", "quality", "measured_scale", "calibrated"); xl = ("Uniform", "Quality-\nweighted", "Measured-\nrelative", "Calibrated")
    xs = np.arange(len(kinds)); raw = V["measured"]["raw"]["trusted"]
    axd.axhline(raw["err"], color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    axd.text(-0.45, raw["err"] - 0.012, "repeat recording", fontsize=8, color=GREY, va="top", ha="left")
    for test, dx, mk, lab in (("measured", -0.14, "o", "measured vessels"), ("imputed", 0.14, "D", "hidden vessels (imputed)")):
        r_ = [V[test][k]["trusted"] for k in kinds]; e_ = np.array([q["err"] for q in r_]); ci_ = np.array([q["err_ci"] for q in r_])
        cl = [BLUE if k == "calibrated" else GREY for k in kinds]
        axd.vlines(xs + dx, ci_[:, 0], ci_[:, 1], color=cl, lw=1.2, zorder=2)
        axd.scatter(xs + dx, e_, marker=mk, s=34, facecolors=["white" if test == "measured" else c_ for c_ in cl], edgecolors=cl,
                    linewidths=1.3, zorder=3)
        axd.scatter([], [], marker=mk, s=34, facecolors="white" if test == "measured" else GREY, edgecolors=GREY, linewidths=1.3, label=lab)
    axd.set_xticks(xs); axd.set_xticklabels(xl, fontsize=8); axd.get_xticklabels()[-1].set_color(BLUE); axd.get_xticklabels()[-1].set_fontweight("bold")
    axd.set_xlim(-0.55, len(kinds) - 0.45); axd.set_ylim(0, 0.75)
    axd.set_ylabel("error vs trusted repeat recordings\n(median |log ratio|)", fontsize=8, labelpad=2); axd.tick_params(labelsize=8)
    axd.legend(loc="lower left", frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    axd.set_title("Independent validation", **TIT)
    for s_ in ("top", "right"):
        axd.spines[s_].set_visible(False)
    log("  d: " + "; ".join(f"{k} meas {V['measured'][k]['trusted']['err']:.3f} imp {V['imputed'][k]['trusted']['err']:.3f} ({100 * V['imputed'][k]['trusted']['dir']:.0f}%)" for k in kinds)
        + f"; repeat {raw['err']:.3f}")

    fig.canvas.draw(); rr_ = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); tb = lambda a_: inv.transform(a_.get_tightbbox(rr_))
    yT = max(tb(axa)[1, 1], tb(axB)[1, 1]); yB = max(tb(a_)[1, 1] for a_ in axs + [axd]); xL = 0.005
    xb = tb(axB)[0, 0] - 0.01; xd = tb(axd)[0, 0] - 0.012
    for x_, y_, l_ in ((xL, yT, "a"), (xb, yT, "b"), (xL, yB, "c"), (xd, yB, "d")):
        fig.text(x_, min(y_ + 0.004, 0.985), l_, **LET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plos_style.save(fig, OUT)
    log(f"wrote {OUT}.png/.pdf")

    # SI: the two measurement examples with the reliability score
    fx = plt.figure(figsize=(5.0, 3.0), facecolor="white"); gx = fx.add_gridspec(1, 2, left=0.02, right=0.98, top=0.9, bottom=0.27, wspace=0.06)
    for k, (e, lab, colr) in enumerate(((HIGH, "Reliable measurement", COL[0]), (LOW, "Unreliable measurement", LOWCOL))):
        ax = fx.add_subplot(gx[k]); x = ex[e]; draw_measurement(ax, img, x["ki"], colr, x["valid"])
        ax.set_title(lab, fontsize=9, color=colr, pad=3)
        ax.text(0.5, -0.03, f"c = {x['c']:.2f}  (f = {x['f']:.2f}, align = {x['align']:.2f})", transform=ax.transAxes, ha="center", va="top", fontsize=8, color=colr)
    c0_ = mod.get("c0", np.median(list(mod["c_T"].values())))
    fx.text(0.5, 0.09, r"$c_k = c_0 + c_f\,(1 - f_k) + c_a\,(1 - \mathrm{align}_k)$" + f"   ($c_0$ = {c0_:.2f}, $c_f$ = {mod['c_f']:.2f}, $c_a$ = {mod['c_a']:.2f})",
            ha="center", va="center", fontsize=9)
    plos_style.save(fx, OUT.parent / "fig6_S_examples")
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
