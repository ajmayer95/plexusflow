"""DRAFT Fig 6 (mean flow, 2026-11-19) — WILL CHANGE SUBSTANTIALLY.  Kept in the repo so the current version is reproducible.

Main figure (figures/fig6/fig6_abs.*): one strip — a micrograph of a 10-vessel patch in the tile 12/13/22 overlap (inset: tile
footprints), b one vessel recorded in three tiles (flux, harmonic fit, ± residual SD s, mean), c every recording (± s), the
inverse-variance weighted mean and the reconstructed flux per vessel.  Reconstructed values are read from the production
results (results/dc_reconciled_v2.pkl, reconcile.dc ESTIMATOR="fixed").  Patch: env FIG6_PATCH (default node 56794).
SI (figures/fig6/fig6_S_heldout_abs.*): held-out recordings predicted by one repeat recording, the weighted mean of the
others, and the reconstruction — computed here with frame-variance weights (s_r²), NOT the production block variance.

Run: python scripts/make_figure6_draft.py   (PLEXUSFLOW_STAGE required; FIGURES / RESULTS env as for every figure)
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as P
from plexusflow.reconcile import measurement_table, carved_domain
from plexusflow.reconcile.network import project, EPS_UNIMAGED
G = P.load_graph(); M = measurement_table(G, gain=None)
from collections import defaultdict
import numpy as np
import networkx as nx
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from plexusflow.io.measurement_view import MeasurementView
from plexusflow.reconcile.dc import repeat_folds

HERE = P.FIGURES / "fig6"; HERE.mkdir(parents=True, exist_ok=True)
ST = P.STAGE
import os
PATCH, PATCH_HOPS = int(os.environ.get("FIG6_PATCH", 56794)), 2
FS = 250.0
PX_UM = 1.7
INK, MUTED = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.default": "regular", "axes.linewidth": 0.8})
T0 = time.time()


def log(s):
    line = f"[+{time.time() - T0:5.0f} s] {s}"
    print(line, flush=True)
    with open(HERE / "run_fig6_abs.log", "a") as fh:
        fh.write(line + "\n")


(HERE / "run_fig6_abs.log").write_text("")

# ------------------------------------------------------------------ the method
dom = carved_domain(G); E = [tuple(e) for e in dom["edges"]]; me = len(E); eix = {e: i for i, e in enumerate(E)}
S_ABS = {k: abs(m["cv"]) * abs(m["Q"]) for k, m in M.items()}
FLOOR = float(np.percentile([v for v in S_ABS.values() if v > 0], 1))


def s_of(k):
    v = S_ABS.get(k, np.nan)
    return max(v, FLOOR) if np.isfinite(v) else np.nan


def edge_index(e):
    i = eix.get(e)
    return (i, 1.0) if i is not None else ((eix[e[::-1]], -1.0) if e[::-1] in eix else (None, 0.0))


def reconstruct(Mx):
    """Noise-weighted means ȳ_e, weights W_e = Σ_r 1/s_r², and the conservation-constrained Q′."""
    recs = defaultdict(list)
    for k, m in Mx.items():
        i, sg = edge_index(k[1])
        if i is None or not np.isfinite(s_of(k)):
            continue
        recs[i].append((sg * m["Q"], s_of(k)))
    y = np.zeros(me); W = np.zeros(me)
    for i, r in recs.items():
        q = np.array([a for a, _ in r]); w = 1 / np.array([b for _, b in r]) ** 2
        y[i] = w @ q / w.sum(); W[i] = w.sum()
    im = W > 0
    w_proj = np.where(im, W / np.median(W[im]), EPS_UNIMAGED)
    return project(y, w_proj, dom), y, W, im


Qp, ybar, W, IM = reconstruct(M)
# use the PRODUCTION reconstruction (results/dc_reconciled_v2.pkl: per-recording block variance, the 2026-11-19 recipe)
import pickle as _pk
_D = _pk.loads((P.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
assert [tuple(e) for e in _D["edges"]] == [tuple(e) for e in E], "production edges differ from the sandbox domain"
Qp, ybar, IM = np.asarray(_D["Qp"], float), np.asarray(_D["Qm"], float), np.asarray(_D["imaged"], bool)
VREC = _D["fixed"]["v_rec"]
fi = dom["B"] @ Qp; inj = fi[dom["inj"]]
log(f"reconstruction: {IM.sum()} measured of {me} vessels; inflow {inj.sum():.2f} nL/s, "
    f"left {100 * inj[dom['side'] == 'L'].sum() / inj.sum():.1f} %; max |interior imbalance| "
    f"{np.max(np.abs(np.delete(fi, dom['bnd']))):.1e}")

# ------------------------------------------------------------------ c: held-out recordings
err_avg, err_rec, dir_avg, dir_rec, qheld = [], [], [], [], []
err_one, dir_one = [], []
rng_one = np.random.default_rng(1)
for Mf, held in repeat_folds(M, n_fold=5, seed=0):
    Qf, yf, Wf, imf = reconstruct(Mf)
    others = defaultdict(list)
    for k, mm in Mf.items():
        others[k[1]].append(mm["Q"])
    for e, m in held:
        i, sg = edge_index(e)
        if i is None or not imf[i] or m["Q"] == 0 or not others.get(e):
            continue
        h = sg * m["Q"]; one = sg * others[e][int(rng_one.integers(len(others[e])))]
        for pred, err, dr in ((one, err_one, dir_one), (yf[i], err_avg, dir_avg), (Qf[i], err_rec, dir_rec)):
            err.append(abs(np.log(max(abs(pred), 1e-6) / abs(h)))); dr.append(np.sign(pred) == np.sign(h))
        qheld.append(abs(h))
err_avg, err_rec, dir_avg, dir_rec, qheld, err_one, dir_one = map(np.array, (err_avg, err_rec, dir_avg, dir_rec, qheld, err_one, dir_one))
log(f"single repeat recording ×{np.exp(np.median(err_one)):.3f}, direction {100 * np.mean(dir_one):.1f} %")
err_imp, dir_imp = [], []
_USABLE = sorted({e for (t, e) in M})
DRAWS25 = [{_USABLE[k] for k in np.random.default_rng(10 + r).choice(len(_USABLE), int(0.25 * len(_USABLE)), replace=False)} for r in range(3)]
for hide in DRAWS25:                                     # 25 % of vessels with every recording hidden (3 draws)
    Qf, yf, Wf, imf = reconstruct({k: m for k, m in M.items() if k[1] not in hide})
    for (t_, e), m in M.items():
        if e not in hide or m["Q"] == 0:
            continue
        i, sg = edge_index(e)
        if i is None:
            continue
        h = sg * m["Q"]; err_imp.append(abs(np.log(max(abs(Qf[i]), 1e-6) / abs(h)))); dir_imp.append(np.sign(Qf[i]) == np.sign(h))
err_imp, dir_imp = np.array(err_imp), np.array(dir_imp)
fold = lambda x: float(np.exp(np.median(x)))
log(f"held-out recordings: n = {len(err_avg)} (5 folds); weighted mean of other recordings ×{fold(err_avg):.3f}, "
    f"direction {100 * dir_avg.mean():.1f} %; reconstruction ×{fold(err_rec):.3f}, direction {100 * dir_rec.mean():.1f} %; "
    f"reconstruction closer in {100 * np.mean(err_rec < err_avg):.1f} % of cases; unmeasured vessels (25 % hidden): "
    f"×{fold(err_imp):.3f}, direction {100 * dir_imp.mean():.1f} % (n {len(err_imp)})")
QB = [0, 0.03, 0.1, 0.3, np.inf]
for lo, hi in zip(QB[:-1], QB[1:]):
    s = (qheld >= lo) & (qheld < hi)
    log(f"  held-out |Q| {lo}–{hi}: n {s.sum()}: mean ×{fold(err_avg[s]):.3f} dir {100*dir_avg[s].mean():.0f} % | "
        f"reconstruction ×{fold(err_rec[s]):.3f} dir {100*dir_rec[s].mean():.0f} %")

# ------------------------------------------------------------------ a: two recordings
def trace_fit(q, f0, K=3):
    t = np.arange(len(q)) / FS; w = 2 * np.pi * f0 * t
    X = np.column_stack([np.ones_like(t)] + [f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])
    b, *_ = np.linalg.lstsq(X, q, rcond=None)
    return t, X @ b


cand = []
for (t_, e), m in M.items():
    if not (0.25 <= abs(m["Q"]) <= 0.6) or not np.isfinite(m["cv"]):
        continue
    mm = next((x for x in G.edges[e].get("measurements_piv") or [] if MeasurementView(x).tile_id == t_), None)
    if mm is None or MeasurementView(mm).Q_t is None or not MeasurementView(mm).harm.f0_hz:
        continue
    cand.append((m["cv"], t_, e, abs(m["Q"]), mm))
cand.sort(key=lambda x: x[0])
clean = cand[int(0.03 * len(cand))]
noisy = min((c for c in cand if c[0] >= np.percentile([x[0] for x in cand], 92)), key=lambda c: abs(c[3] - clean[3]))
EXA = []
for cv_, t_, e, qa, mm in (clean, noisy):
    mv = MeasurementView(mm); q = np.asarray(mv.Q_t, float); q = q * np.sign(np.nanmean(q))
    t, fit = trace_fit(q, float(mv.harm.f0_hz)); s = float(np.std(q - fit))
    EXA.append(dict(tile=t_, e=e, t=t, q=q, fit=fit, s=s, mean=q.mean(), f0=float(mv.harm.f0_hz)))
    log(f"  example recording tile {t_} edge {e}: mean {q.mean():.3f} nL/s, s_r {s:.4f} nL/s (cv {cv_:.3f}), f0 {mv.harm.f0_hz:.2f} Hz")

# ------------------------------------------------------------------ b: the nine-vessel patch
Hn = nx.Graph(); Hn.add_edges_from(E)
pn = nx.single_source_shortest_path_length(Hn, PATCH, cutoff=PATCH_HOPS)
pidx = sorted({edge_index(e)[0] for e in Hn.subgraph(pn).edges()})
recs = {i: [] for i in pidx}
for k, m in M.items():
    i, sg = edge_index(k[1])
    if i in recs:
        recs[i].append((int(k[0]), sg * m["Q"], s_of(k)))
mos = tifffile.imread(str(ST / "fuse" / "maxmed.tif")).astype(float); mos = mos.mean(-1) if mos.ndim == 3 else mos; Hm = mos.shape[0]
toimg = lambda q: np.column_stack([q[:, 0], Hm - q[:, 1]])
pth = {i: toimg(np.asarray(G.edges[E[i]]["path"], float)) for i in pidx}
mids = {i: pth[i][len(pth[i]) // 2] for i in pidx}
order = sorted(pidx, key=lambda i: (mids[i][0], mids[i][1])); num = {i: k + 1 for k, i in enumerate(order)}
ptiles = sorted({r[0] for i in pidx for r in recs[i]})
TCOL = dict(zip(ptiles, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]))
for i in order:
    sg = np.sign(Qp[i]) or 1.0
    log(f"  vessel {num[i]} {E[i]}: Q′ {abs(Qp[i]):.3f} | mean {sg * ybar[i]:+.3f} | recordings "
        + ", ".join(f"t{t_} {sg * q:+.3f}±{s:.3f}" for t_, q, s in sorted(recs[i])))

# ------------------------------------------------------------------ figure: a noise · b held-out · c local reconstruction
# a: two recordings of one patch vessel with the largest weight ratio (both |mean| > 0.03 nL/s)
# panel b vessel: recorded by the most tiles (all three if possible), similar means, largest spread of residual noise
best = None
for i in order:
    rs = recs[i]
    if len({r[0] for r in rs}) < 2 or min(abs(r[1]) for r in rs) < 0.03:
        continue
    key = (len({r[0] for r in rs}), max(abs(r[1]) for r in rs) / min(abs(r[1]) for r in rs) < 2.0,
           max(r[2] for r in rs) / min(r[2] for r in rs))
    if best is None or key > best[0]:
        best = (key, i)
AV = best[1]
EXA = []
for t_ in sorted({r[0] for r in recs[AV]}, key=lambda t: s_of((t, E[AV]))):
    mm = next(x for x in G.edges[E[AV]]["measurements_piv"] if MeasurementView(x).tile_id == t_)
    mv = MeasurementView(mm); q = np.asarray(mv.Q_t, float)
    q = q * (np.sign(np.nanmean(q)) if np.sign(np.nanmean(q)) != 0 else 1.0)
    t, fit = trace_fit(q, float(mv.harm.f0_hz))
    EXA.append(dict(tile=t_, t=t, q=q, fit=fit, s=s_of((t_, E[AV])), mean=q.mean(), f0=float(mv.harm.f0_hz)))
log(f"panel b: vessel {num[AV]} {E[AV]}: " + " | ".join(f"tile {x['tile']} s {x['s']:.3f} mean {x['mean']:.3f}" for x in EXA))

OUT = P.FIGURES / "fig6"; OUT.mkdir(parents=True, exist_ok=True)

# SI: held-out discrepancies
figS, axB = plt.subplots(figsize=(3.6, 2.0), facecolor="white")
rows = [("one repeat recording", err_one, "0.65"), ("noise-weighted mean", err_avg, "0.4"), ("reconstruction", err_rec, INK)]
for k, (lab, err, col) in enumerate(rows):
    f = np.exp(err); y0 = len(rows) - 1 - k
    q1, q2, q3 = np.percentile(f, [25, 50, 75]); w1, w9 = np.percentile(f, [10, 90])
    axB.plot([w1, w9], [y0, y0], color=col, lw=1.0)
    axB.add_patch(matplotlib.patches.Rectangle((q1, y0 - 0.22), q3 - q1, 0.44, fc=col, ec="none", alpha=0.85))
    axB.plot([q2, q2], [y0 - 0.22, y0 + 0.22], color="white", lw=1.6)
    axB.text(w9 * 1.08, y0, f"×{q2:.2f}", va="center", fontsize=7.5, color=INK)
axB.set_xscale("log"); axB.set_xlim(1, 14); axB.set_xticks([1, 2, 5, 10]); axB.set_xticklabels(["1", "2", "5", "10"])
axB.set_yticks(range(len(rows))); axB.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5); axB.set_ylim(-0.6, len(rows) - 0.4)
axB.set_xlabel("discrepancy with a held-out recording (fold)"); axB.spines["left"].set_visible(False); axB.tick_params(axis="y", length=0)
figS.tight_layout()
for ext in ("png", "pdf"):
    figS.savefig(OUT / f"fig6_S_heldout_abs.{ext}", dpi=300)
log(f"saved {OUT}/fig6_S_heldout_abs.png / .pdf")


# ================================================================== MAIN FIGURE: one junction — measure → average → reconcile
# caption junction: an end of the panel-b vessel whose three vessels are all in the patch (largest weighted-mean imbalance)
_cands = []
for n_ in list(E[AV]) + [n for i in order for n in E[i]]:
    vs_ = [i for i in order if n_ in E[i]]
    if G.degree(n_) == 3 and len(vs_) == 3:
        _cands.append((abs(sum((ybar[i] if E[i][1] == n_ else -ybar[i]) for i in vs_)), n_, vs_))
_cands.sort(key=lambda c: (c[1] in E[AV], c[0]), reverse=True)
_, JN, jv = _cands[0] if _cands else (0, None, [])


def into(i, q):                                             # flux into JN along vessel i (q signed along E[i])
    return q if E[i][1] == JN else -q


before = {i: into(i, ybar[i]) for i in jv}; after = {i: into(i, Qp[i]) for i in jv}
for name, d in (("weighted means", before), ("reconstructed", after)):
    fin = sum(v for v in d.values() if v > 0); fout = -sum(v for v in d.values() if v < 0)
    log(f"junction {JN} {name}: " + ", ".join(f"v{num[i]} {d[i]:+.3f}" for i in jv) + f" | in {fin:.3f} out {fout:.3f} imbalance {fin - fout:+.3f}")

import plos_style as ps
ps.apply()
W_IN, H_IN = 7.5, 2.75
fig = plt.figure(figsize=(W_IN, H_IN))
ax_at = lambda x, y, w, h: fig.add_axes([x / W_IN, y / H_IN, w / W_IN, h / H_IN])     # positions in inches
LETTER = dict(fontsize=12, fontweight="bold", ha="left", va="top")
TOP = H_IN - 0.05
fig.text(0.03 / W_IN, TOP / H_IN, "a", **LETTER); fig.text(2.42 / W_IN, TOP / H_IN, "b", **LETTER); fig.text(4.6 / W_IN, TOP / H_IN, "c", **LETTER)
MEANC, RECC = "0.6", INK

# a: local network (wider view), nine vessels numbered, vessel 3 marked
axM = ax_at(0.2, 0.12, 2.2, 2.5)
allp = np.vstack([pth[i] for i in pidx]); ctr = (allp.min(0) + allp.max(0)) / 2
hx = 0.5 * (allp.max(0) - allp.min(0)).max() + 45; hy = hx * 2.5 / 2.2
c0, c1, r0, r1 = ctr[0] - hx, ctr[0] + hx, ctr[1] - hy, ctr[1] + hy
crop = mos[int(max(r0, 0)):int(r1), int(max(c0, 0)):int(c1)]; lo, hi = np.percentile(crop[crop > 0], (1, 99.5))
axM.imshow(mos, cmap="gray", vmin=lo - 0.45 * (hi - lo), vmax=hi + 0.15 * (hi - lo), interpolation="bicubic")
axM.set_xlim(c0, c1); axM.set_ylim(r1, r0); axM.set_xticks([]); axM.set_yticks([])
for s_ in axM.spines.values():
    s_.set_linewidth(0.6); s_.set_color("0.6")
placed = []
for i in pidx:
    q = pth[i]; tail = toimg(np.array([[G.nodes[E[i][0]]["x"], G.nodes[E[i][0]]["y"]]]))[0]
    if np.linalg.norm(q[0] - tail) > np.linalg.norm(q[-1] - tail):
        q = q[::-1]
    if len(q) > 7:
        q = np.column_stack([np.convolve(np.pad(q[:, d], 2, mode="edge"), np.ones(5) / 5, "valid") for d in (0, 1)])
    lw = 1.6 if i == AV else 1.0
    axM.plot(q[:, 0], q[:, 1], color="white", lw=lw + 1.4, zorder=4, solid_capstyle="round")
    axM.plot(q[:, 0], q[:, 1], color=INK, lw=lw, zorder=5, solid_capstyle="round")
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    a0, a1 = np.searchsorted(cum, 0.42 * cum[-1]), min(np.searchsorted(cum, 0.62 * cum[-1]), len(q) - 1)
    p0, p1 = (q[a0], q[a1]) if Qp[i] >= 0 else (q[a1], q[a0])
    axM.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7), zorder=6)
    d = q[a1] - q[a0]; d = d / max(np.linalg.norm(d), 1e-9); nrm = np.array([-d[1], d[0]]); m_ = q[len(q) // 2]
    lab = max([m_ + 11 * nrm, m_ - 11 * nrm], key=lambda c: min([np.linalg.norm(c - o) for o in placed] + [1e9])); placed.append(lab)
    axM.text(*lab, str(num[i]), ha="center", va="center", fontsize=8, color="white" if i == AV else INK, zorder=8,
             fontweight="bold" if i == AV else "normal",
             bbox=dict(boxstyle="circle,pad=0.15", fc=INK if i == AV else "white", ec=INK, lw=0.6))
# tile footprints (mosaic row/column), outlined in tile colours
from plexusflow.analyze.geometry import load_tile_geometry
from plexusflow.config import StageConfig
geom = load_tile_geometry(StageConfig.load(ST))
from matplotlib.patches import Polygon as _Poly
for t_ in ptiles:
    tg = geom.tiles[int(t_)]
    Hv, Wv = tg.video_shape if tg.video_shape else tifffile.imread(str(ST / "tiles" / "maxmed" / f"vid{t_}_maxmed.tif")).shape[:2]
    fp = np.array([[0, 0, 1], [0, Wv, 1], [Hv, Wv, 1], [Hv, 0, 1]], float) @ tg.A_full.T
    fp = fp[:, :2] / geom.downsample - [geom.y0, geom.x0]                         # (row, column) in the mosaic
    xy_ = np.column_stack([fp[:, 1], fp[:, 0]])
    FPXY = globals().setdefault("FPXY", {}); FPXY[t_] = xy_
    log(f"  tile {t_} footprint x {xy_[:, 0].min():.0f}–{xy_[:, 0].max():.0f}, y {xy_[:, 1].min():.0f}–{xy_[:, 1].max():.0f}; view x {c0:.0f}–{c1:.0f} y {r0:.0f}–{r1:.0f}")
L = 50 / PX_UM; xb, yb = c1 - 8 - L, r1 - 8
axM.plot([xb, xb + L], [yb, yb], color="white", lw=2.0, solid_capstyle="butt", zorder=8)
axM.text(xb + L / 2, yb - 4, "50 µm", color="white", fontsize=8, ha="center", va="bottom", zorder=8)
# overview inset: the tiles' footprints and the zoomed view
axI = axM.inset_axes([0.0, 0.66, 0.34, 0.34]); axI.set_facecolor("0.15")
allfp = np.vstack(list(FPXY.values())); ic = (allfp.min(0) + allfp.max(0)) / 2; ih = 0.5 * (allfp.max(0) - allfp.min(0)).max() * 1.05
axI.imshow(mos, cmap="gray", vmin=lo - 0.45 * (hi - lo), vmax=hi + 0.15 * (hi - lo), interpolation="antialiased")
axI.set_xlim(ic[0] - ih, ic[0] + ih); axI.set_ylim(ic[1] + ih, ic[1] - ih); axI.set_xticks([]); axI.set_yticks([])
for t_, xy_ in FPXY.items():
    axI.add_patch(_Poly(xy_, closed=True, fc="none", ec=TCOL[t_], lw=1.1, zorder=3))
axI.add_patch(matplotlib.patches.Rectangle((c0, r0), c1 - c0, r1 - r0, fc="white", alpha=0.35, ec="white", lw=1.0, zorder=5))
for s_ in axI.spines.values():
    s_.set_color("white"); s_.set_linewidth(1.0)

# b: vessel 3 recorded in two tiles
nb = len(EXA); hb = (2.12 - 0.12 * (nb - 1)) / nb
axs = [ax_at(2.95, 0.5 + (nb - 1 - k) * (hb + 0.12), 1.45, hb) for k in range(nb)]
tmax = 3.0 / min(e["f0"] for e in EXA)
ylo_ = min(np.percentile(x["q"][x["t"] <= tmax], 1) for x in EXA); yhi_ = max(np.percentile(x["q"][x["t"] <= tmax], 99.5) for x in EXA)
pad_ = 0.12 * (yhi_ - ylo_); ylo_, yhi_ = ylo_ - pad_, yhi_ + 2.2 * pad_
for ax, x in zip(axs, EXA):
    col = TCOL[x["tile"]]; sel = x["t"] <= tmax
    ax.fill_between(x["t"][sel], (x["fit"] - x["s"])[sel], (x["fit"] + x["s"])[sel], color=col, alpha=0.22, lw=0, zorder=1)
    ax.plot(x["t"][sel], x["q"][sel], color=col, lw=0.45, alpha=0.5, zorder=2)
    ax.plot(x["t"][sel], x["fit"][sel], color=col, lw=1.3)
    ax.axhline(x["mean"], color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(0, tmax); ax.set_ylim(ylo_, yhi_); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.99, f"tile {x['tile']}", transform=ax.transAxes, ha="left", va="top", color=col, fontsize=8)
    ax.text(0.99, 0.99, f"±s = {x['s']:.3f} nL/s", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=8)
for ax in axs[:-1]:
    ax.tick_params(labelbottom=False)
axs[-1].set_xlabel("time (s)")
fig.text(2.52 / W_IN, 1.56 / H_IN, f"vessel {num[AV]} flux (nL/s)", rotation=90, ha="center", va="center", fontsize=9)

# c: all nine vessels — recordings (± s), weighted mean (grey), reconstructed (black)
axS = ax_at(5.0, 0.5, 2.42, 2.12)
for k, i in enumerate(order):
    yk = len(order) - k; sg = np.sign(Qp[i]) or 1.0; rs = sorted(recs[i])
    offs = np.linspace(-0.2, 0.2, len(rs)) if len(rs) > 1 else [0.0]
    for (t_, q, s), dy in zip(rs, offs):
        axS.errorbar(sg * q, yk + dy, xerr=s, fmt="o", ms=2.8, color=TCOL[t_], alpha=0.8, mec="none", elinewidth=0.8, capsize=0, zorder=3)
    axS.plot([sg * ybar[i], abs(Qp[i])], [yk, yk], color=MEANC, lw=0.7, zorder=2)
    axS.plot([sg * ybar[i]] * 2, [yk - 0.32, yk + 0.32], color=MEANC, lw=1.8, zorder=2.5, solid_capstyle="butt")
    axS.plot([abs(Qp[i])] * 2, [yk - 0.36, yk + 0.36], color=RECC, lw=2.0, zorder=4, solid_capstyle="butt")
axS.set_yticks(range(1, len(order) + 1)); axS.set_yticklabels([str(num[order[len(order) - y]]) for y in range(1, len(order) + 1)])
for tl in axS.get_yticklabels():
    if tl.get_text() == str(num[AV]):
        tl.set_fontweight("bold")
xmax_ = max(abs(r[1]) + r[2] for i in order for r in recs[i]) * 1.45
axS.set_ylim(0.4, len(order) + 0.6); axS.set_xlim(-0.01 * xmax_, xmax_); axS.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axS.set_ylabel("vessel", labelpad=1); axS.set_xlabel("mean flux (nL/s)"); axS.spines[["top", "right"]].set_visible(False)
hs = [Line2D([], [], marker="o", ls="-", lw=0.8, ms=3, alpha=0.85, color=TCOL[t_], label=f"tile {t_}") for t_ in ptiles]
hs += [Line2D([], [], color=MEANC, lw=1.8, label="weighted mean"), Line2D([], [], color=RECC, lw=2.0, label="reconstructed")]
axS.legend(handles=hs, loc="upper right", frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.4, labelspacing=0.15,
           borderaxespad=0.1)

ps.save(fig, OUT / os.environ.get("FIG6_NAME", "fig6_abs"))
log(f"saved {OUT}/fig6_abs.png / .pdf / .tif (strip layout)")
