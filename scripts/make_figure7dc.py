"""Figure 8 alt v3 — fig8v4 layout on the FINAL DC reconciliation (2026-10-08).

Identical panels to fig8alt v2; data now from results/dc_reconciled_v2.pkl (scripts/dc_reconcile_v2.py):
re-batched Kloosterman-valid measurements, per-edge winner = smallest reliability c, reliability-weighted
IRLS (w = 1/(c²|Q'|² + σ0²)), emergent carve/V boundary, consensus radii.  (Below: v2 description.)

Figure 8 alt v2 — fig8v4 layout on the conservative projection (locked DC method).

Same six panels as make_figure8v4.py, but every field comes from the locked DC
method (docs/dc_method.md) instead of the fitted-outlet carved sim:

  data      winner-take-all on harmonic coherence ρ (K=3), raw oriented mean, no gains
  domain    carved network (porous zones + DA stubs removed); boundary = 35 carve-boundary
            vertices + V1/V2
  Q'        IRLS-weighted conservative projection (σ = 0.2|Q'| + σ0, 6 iterations)
  boundary  EMERGENT  f = B Q'  (nothing fitted)
  Q_sim     passive Poiseuille, K = r⁴/ℓ, consensus radii, f imposed at every boundary vertex

  A  conservative mean-flow field |Q'| (unimaged/imputed edges drawn faint)
  B  passive-network prediction |Q_sim| + emergent carve-boundary injections (circles)
  C  local error E_e = log10(|Q_sim|/|Q'|) on measured edges, direction mismatches marked
  D  measured (Q') vs predicted hexbin, r_log, P(direction agree | filtered), n
  E  true vs uniform vs shuffled radii (solved on the fly; shuffle = mean ± sd of 10)
  F  agreement vs minimum hop distance from the driven carve boundary

Paper Fig 6 (PLOS, 7.5 × 7.6 in; panels a–f as A–F above).  Writes figures/fig8alt_v3/fig8alt_v3.{png,pdf,tif}.  Writes nothing to any graph.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import shortest_path
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths  # stage + results locations (PLEXUSFLOW_STAGE)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plos_style
from plexusflow.reconcile import carved_domain, passive

STAGE = _paths.STAGE
MOSAIC_P = _paths.MOSAIC
OUT_DIR = _paths.FIGURES / 'fig7'
import os
SIGMA_P = float(os.environ.get('FIG7_SIGMA_P', '2.0'))   # Pa; 'inf' = no pressure penalty (measured venous split)  #: soft equal-venous-pressure penalty (any σ_P ≲ 5 Pa gives the same split)
SCHEME, RADIUS, N_SHUFFLE = 'irls', 'consensus', 10


def main():
    import pickle
    plos_style.apply()
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    G = pickle.loads(_paths.GRAPH.read_bytes())
    dom = carved_domain(G)
    G.graph['boundary'] = dict(DA=dom['DA'], V=dom['V'])
    edges = dom['edges']; assert edges == D['edges']
    idx = np.arange(len(edges)); em = np.ones(len(edges), bool)
    rec = dict(length=D['length'])
    Qp, Qs, fb, imaged, r = D['Qp'], D['Qs'], D['fb'], D['imaged'], D['radius']
    K = r ** 4 / D['length']; nodes = dom['nodes']
    # passive model with a soft equal-pressure penalty at the two vitelline-vein outlets: the outflow split (f_V1, f_V2) is
    # chosen to minimise Σ_V (f_V − f_V^meas)²/σ_V² + (P_V1 − P_V2)²/σ_P², σ_V = stub-flow uncertainty (dc_reconciled sigma)
    import scipy.sparse.linalg as spla
    from plexusflow.reconcile.network import _gauge_nodes
    MU, PX, NL = 3.5e-3, 1.7e-6, 1e-12
    Kp = np.pi * (r * PX) ** 4 / (8 * MU * D['length'] * PX); Bm = dom['B']; Lp = (Bm @ sp.diags(Kp) @ Bm.T).tocsc()
    v1, v2 = dom['out']; free = np.setdiff1d(np.arange(Bm.shape[0]), [v1]); lu = spla.splu(Lp[free][:, free])
    def pres(f):
        p_ = np.zeros(Bm.shape[0]); p_[free] = lu.solve(f[free]); return p_
    sV = [float(np.sqrt(np.sum(D['sigma'][Bm.tocsc()[v].nonzero()[1]] ** 2))) for v in (v1, v2)]
    d_ = np.zeros(Bm.shape[0]); d_[v1], d_[v2] = -1.0, 1.0
    pA, pD = pres(fb * NL), pres(d_ * NL); dP0, kk = pA[v2] - pA[v1], pD[v2] - pD[v1]
    shift = 0.0 if not np.isfinite(SIGMA_P) else -(dP0 * kk / SIGMA_P ** 2) / (1 / sV[0] ** 2 + 1 / sV[1] ** 2 + kk ** 2 / SIGMA_P ** 2)
    fb = fb.copy(); fb[v1] -= shift; fb[v2] += shift
    Qs = passive(K, fb, dom)
    print(f"[fig7] soft venous penalty σ_P {SIGMA_P} Pa: outflow V1 {fb[v1]:.2f}, V2 {fb[v2]:.2f} nL/s (shift {shift:+.2f}); "
          f"P_V2 − P_V1 {dP0 + kk * shift:+.2f} Pa (was {dP0:+.2f})")
    print(f"[fig7] carved: {len(idx)} edges ({int(imaged.sum())} measured); "
          f"Σ in {fb[dom['inj']].sum():.2f}; V {fb[dom['out']]}")

    mos = tifffile.imread(str(MOSAIC_P)); mos = mos.mean(axis=-1) if mos.ndim == 3 else mos
    Hm, Wm = mos.shape
    segs = []
    for i in idx:
        d = G.edges[edges[i]]
        p = np.asarray(d['path'], float) if d['path'] is not None else \
            np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in edges[i]])
        segs.append(np.column_stack([p[:, 0], Hm - p[:, 1]]))
    nxy = {n: (G.nodes[n]['x'], Hm - G.nodes[n]['y']) for n in G.nodes}

    # shared colour scale for A & B (measured edges of both fields)
    allq = np.abs(np.r_[Qp[imaged], Qs[imaged]])
    vmin = max(float(np.percentile(allq, 2)), 1e-4); vmax = float(np.percentile(allq, 98))
    nAB = mcolors.LogNorm(vmin, vmax); cAB = plt.cm.viridis
    nC = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1); cC = plt.cm.PuOr_r
    MIS = '#c51b8a'; INKC = '#1a1a1a'

    def bg(ax):
        ax.imshow(mos, cmap='gray', vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), aspect='equal', zorder=0, alpha=0.55)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(0, Wm); ax.set_ylim(Hm, 0)

    def markers(ax):
        for i_ in dom['inj']:                               # Fig 5 convention: arterial cuts red, venous outlets blue
            ax.plot(*nxy[nodes[i_]], 'o', color='#d40000', ms=2.2, mec='white', mew=0.3, zorder=6)
        for n in G.graph['boundary']['V']:
            ax.plot(*nxy[n], 's', color='#2b35a8', ms=4, mec='white', mew=0.5, zorder=6)

    resolved = np.abs(Qp) > 2 * D['sigma']                     # direction determined: |Q'| > 2σ_e (Fig 6 noise model)
    print(f"[fig7] direction resolved for {resolved.sum()} of {len(Qp)} vessels (|Q'| > 2σ)")
    print(f"[fig7] resolved vessels carry {100 * np.abs(Qp[resolved]).sum() / np.abs(Qp).sum():.0f}% of Σ|Q'|; length-weighted {100 * (np.abs(Qp) * D['length'])[resolved].sum() / (np.abs(Qp) * D['length']).sum():.0f}%; passive Σ|Q_sim| on them {100 * np.abs(Qs[resolved]).sum() / np.abs(Qs).sum():.0f}%")
    def field(ax, Q, mask_unresolved):
        und = ~resolved if mask_unresolved else np.zeros(len(Q), bool)
        ax.add_collection(LineCollection([segs[c] for c in np.flatnonzero(und)], colors=[(0.6, 0.6, 0.6, 0.5)], linewidths=0.25, zorder=2))
        keep = np.flatnonzero(~und)
        ax.add_collection(LineCollection([segs[c] for c in keep], colors=[cAB(nAB(min(max(abs(Q[c]), vmin), vmax))) for c in keep],
                                         linewidths=0.5, zorder=3))

    fig = plt.figure(figsize=(7.5, 6.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.8], hspace=0.04, left=0.06, right=0.9, top=0.97, bottom=0.08)
    gT = gs[0].subgridspec(1, 2, wspace=0.08); gBm = gs[1].subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.8], wspace=0.45)
    def letter(ax, l, title, dx=-0.02):
        ax.text(dx, 1.01, l, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')
        ax.set_title(title, fontsize=9, loc='left', pad=3)

    # a
    axA = fig.add_subplot(gT[0]); bg(axA); field(axA, Qp, True); markers(axA)
    letter(axA, 'a', "Reference field Q′")
    axA.plot([], [], color=(0.6, 0.6, 0.6), lw=1.2, label='direction undetermined (|Q′| < 2σ)')
    axA.legend(loc='lower right', fontsize=8, frameon=False, borderaxespad=0, handlelength=1.4)

    # b
    axB = fig.add_subplot(gT[1]); bg(axB); field(axB, Qs, True); markers(axB)     # same measurement-support mask as a
    inj = fb[dom['inj']]
    for j, nidx in enumerate(dom['inj']):
        xy = nxy[nodes[nidx]]; a = inj[j]
        col = cAB(nAB(min(max(abs(a), vmin), vmax)))
        if a >= 0:
            axB.scatter(*xy, s=22, facecolor=col, edgecolor=INKC, linewidths=0.5, zorder=7)
        else:
            axB.scatter(*xy, s=22, facecolor='white', edgecolor=INKC, linewidths=0.7, zorder=7)
    axB.scatter([], [], s=22, facecolor=cAB(0.9), edgecolor=INKC, linewidths=0.5, label='inflow (emergent)')
    axB.scatter([], [], s=22, facecolor='white', edgecolor=INKC, linewidths=0.7, label='outflow (emergent)')
    axB.legend(loc='lower right', fontsize=8, frameon=False, borderaxespad=0, handletextpad=0.2, labelspacing=0.2)
    letter(axB, 'b', r'Passive prediction Q$_{\rm sim}$')
    pB = axB.get_position(); cax = fig.add_axes([pB.x1 + 0.012, pB.y0 + 0.1 * pB.height, 0.013, 0.8 * pB.height])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nAB, cmap=cAB), cax=cax)
    cb.set_label(r'$|\bar Q|$ (nL/s)', fontsize=9, labelpad=2); cb.outline.set_linewidth(0.5)

    # c
    axC = fig.add_subplot(gBm[0]); bg(axC)
    m = imaged & (np.abs(Qp) > 1e-12) & (np.abs(Qs) > 1e-12)
    E = np.log10(np.abs(Qs) / np.maximum(np.abs(Qp), 1e-12))
    mr = m & (np.abs(Qp) > 2 * D['sigma'])
    axC.add_collection(LineCollection([segs[c] for c in np.flatnonzero(~mr)], colors=[(0.6, 0.6, 0.6, 0.5)], linewidths=0.25, zorder=2))
    axC.add_collection(LineCollection([segs[c] for c in np.flatnonzero(mr)],
                                      colors=[cC(nC(E[c])) for c in np.flatnonzero(mr)], linewidths=0.5, zorder=3))
    q_floor = max(vmin * 3, 3e-3)
    # noise level = each vessel's calibrated measurement uncertainty σ_e = sqrt((c_e y_e)² + σ0²) (Fig 6b); a direction is only
    # judged where the reconciled flow is clearly above it, |Q'| > 2σ_e
    sig_ok = np.abs(Qp) > 2 * D['sigma']
    mism = m & sig_ok & (np.sign(Qp) != np.sign(Qs))
    for c in np.flatnonzero(mism):
        s = segs[c]; mid = s[len(s) // 2]
        axC.plot(*mid, 'o', color=MIS, ms=2.0, markeredgewidth=0, zorder=6, alpha=0.9)
    markers(axC)
    letter(axC, 'c', 'Local agreement')
    caxC = axC.inset_axes([0.15, -0.06, 0.7, 0.035])
    cbC = plt.colorbar(plt.cm.ScalarMappable(norm=nC, cmap=cC), cax=caxC, ticks=[-1, 0, 1], orientation='horizontal')
    cbC.set_label(r'log$_{10}$(|Q$_{\rm sim}$| / |Q′|)', labelpad=1); cbC.outline.set_linewidth(0.5)
    axC.plot([], [], 'o', color=MIS, ms=3, label='direction mismatch')
    axC.legend(fontsize=8, loc='lower right', frameon=False, borderaxespad=0, handletextpad=0.1)

    # D
    axD = fig.add_subplot(gBm[1])
    lm, ls_ = np.log10(np.maximum(np.abs(Qp[m]), 1e-5)), np.log10(np.maximum(np.abs(Qs[m]), 1e-5))
    axD.hexbin(lm, ls_, gridsize=45, cmap='Greys', mincnt=1, linewidths=0, zorder=2, alpha=0.7)
    lo, hi = min(lm.min(), ls_.min()), max(lm.max(), ls_.max())
    axD.plot([lo, hi], [lo, hi], color=INKC, ls='--', lw=0.9, zorder=3)
    mm = mism[m]
    axD.scatter(lm[mm], ls_[mm], s=5, facecolors='none', edgecolors=MIS, linewidths=0.5, alpha=0.8,
                zorder=4, label='direction mismatch')
    r_log = float(np.corrcoef(lm, ls_)[0, 1])
    consider = m & sig_ok
    p_agree = float(np.mean(np.sign(Qp[consider]) == np.sign(Qs[consider])))
    axD.text(0.03, 0.9, f'$r_{{\\log}} = {r_log:.3f}$\n'
                         f'direction agrees: {100 * p_agree:.0f}%\n(|Q′| > 2σ, n = {int(consider.sum())})',
             transform=axD.transAxes, va='top', ha='left', fontsize=8, bbox=dict(fc='white', ec='0.7', lw=0.5, pad=2.5))
    axD.set_xlabel(r"measured log$_{10}$|Q′| (nL/s)", labelpad=1)
    axD.set_ylabel(r'predicted log$_{10}$|Q$_{\rm sim}$| (nL/s)', labelpad=1)
    letter(axD, 'd', 'Vessel-by-vessel', dx=-0.2); axD.title.set_position((0, 1.06))
    axD.legend(fontsize=8, loc='lower right', frameon=False, borderaxespad=0, handletextpad=0.1); axD.set_aspect('equal')
    axD.spines[['top', 'right']].set_visible(False)
    axD.set_xlim(lo - 0.1, hi + 0.1); axD.set_ylim(lo - 0.1, hi + 0.1)

    # E  geometry controls
    ln = rec['length'][em]
    def r_of(Kx):
        q = passive(Kx, fb, dom); a, b = np.abs(Qp[m]), np.abs(q[m]); ok = (a > 1e-12) & (b > 1e-12)
        return float(np.corrcoef(np.log10(a[ok]), np.log10(b[ok]))[0, 1])
    rng = np.random.default_rng(0)
    r_uni = r_of(np.mean(r) ** 4 / ln)
    r_sh = np.array([r_of(rng.permutation(r) ** 4 / ln) for _ in range(N_SHUFFLE)])
    print(f'  r_log true {r_log:.3f}, uniform {r_uni:.3f}, shuffled {r_sh.mean():.3f} ± {r_sh.std():.3f}')
    axE = fig.add_subplot(gBm[2])
    for yi, lab, rv, col, err in ((2, 'true\nradii', r_log, '#2b70b4', 0), (1, 'uniform\nradii', r_uni, '#9e9e9e', 0),
                                  (0, 'shuffled\nradii', r_sh.mean(), '#e08214', r_sh.std())):
        axE.hlines(yi, 0, rv, color=col, lw=2, alpha=0.8)
        axE.errorbar(rv, yi, xerr=err if err else None, fmt='o', color=col, markersize=5, capsize=2, zorder=4)
        axE.text(rv + 0.03, yi + 0.28, f'{rv:.2f}', va='bottom', ha='center', fontsize=8, color=col)
    axE.set_yticks([2, 1, 0], ['true radii', 'uniform radii', 'shuffled radii'])
    axE.set_xlim(0, 1.0); axE.set_ylim(-0.6, 2.9); axE.set_box_aspect(1.0)
    axE.set_xlabel(r"$r_{\log}$", labelpad=1)
    axE.spines['right'].set_visible(False); axE.spines['top'].set_visible(False)
    letter(axE, 'e', 'Radius controls', dx=-0.3)

    # F  distance from the driven carve boundary
    B = dom['B']; Bc = B.tocoo()
    tails = np.asarray((B.multiply(B > 0)).argmax(axis=0)).ravel(); heads = np.asarray((B.multiply(B < 0)).argmin(axis=0)).ravel()
    A = sp.csr_matrix((np.ones(len(idx)), (tails, heads)), shape=(B.shape[0], B.shape[0]))
    hop = shortest_path(A, directed=False, unweighted=True, indices=dom['inj']).min(axis=0)
    ehop = np.minimum(hop[tails], hop[heads])
    d_grid = np.arange(0, int(np.nanpercentile(ehop[m], 95)) + 1)
    r_vals, n_vals = [], []
    for dmin in d_grid:
        mk = m & (ehop >= dmin)
        n_vals.append(int(mk.sum()))
        r_vals.append(float(np.corrcoef(np.log10(np.abs(Qp[mk])), np.log10(np.abs(Qs[mk])))[0, 1]) if mk.sum() >= 20 else np.nan)
    r_vals = np.array(r_vals)
    figF = plt.figure(figsize=(3.6, 2.4)); axF = figF.add_axes([0.16, 0.2, 0.68, 0.72])
    axF.plot(d_grid, r_vals, '-', color='#2b70b4', lw=1.2, zorder=3)
    axF.fill_between(d_grid, r_vals, where=~np.isnan(r_vals), color='#2b70b4', alpha=0.15, zorder=2)
    axF.set_xlim(0, d_grid.max()); axF.set_ylim(0, 1); axF.axhline(0.5, color='0.65', lw=0.6, ls=':')
    axF.set_xlabel('distance from the driven boundary (hops)', labelpad=1)
    axF.set_ylabel(r"$r_{\log}$", labelpad=1)
    axF2 = axF.twinx(); axF2.plot(d_grid, n_vals, '--', color='0.45', lw=0.8, alpha=0.8)
    axF2.set_ylabel('vessels retained', color='0.45', labelpad=2); axF2.tick_params(colors='0.45'); axF2.spines['top'].set_visible(False); axF.spines['top'].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # SI: sensitivity of the direction-confidence threshold k (|Q'| > kσ): coverage, directed loops, model agreement
    import networkx as nx
    tl_ = np.asarray((Bm.multiply(Bm > 0)).argmax(0)).ravel(); hd_ = np.asarray((Bm.multiply(Bm < 0)).argmin(0)).ravel()
    zz = np.abs(Qp) / D['sigma']; ks = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]); rows = []
    for k_ in ks:
        keep = zz > k_ if k_ > 0 else np.abs(Qp) > 1e-12
        DG = nx.DiGraph(); [DG.add_edge(tl_[e], hd_[e]) if Qp[e] > 0 else DG.add_edge(hd_[e], tl_[e]) for e in np.flatnonzero(keep)]
        Sc = [c for c in nx.strongly_connected_components(DG) if len(c) > 1]; Su = set().union(*Sc) if Sc else set()
        nl = sum(keep[e] and tl_[e] in Su and hd_[e] in Su for e in range(len(Qp)))
        mk = keep & imaged & (np.abs(Qs) > 1e-12)
        rows.append((100 * keep.mean(), 100 * np.abs(Qp[keep]).sum() / np.abs(Qp).sum(), 100 * nl / max(keep.sum(), 1),
                     100 * np.mean(np.sign(Qp[mk]) == np.sign(Qs[mk])), np.corrcoef(np.log10(np.abs(Qp[mk])), np.log10(np.abs(Qs[mk])))[0, 1]))
    rows = np.array(rows)
    for k_, rw in zip(ks, rows):
        print(f"[fig7 SI] k={k_:.1f}: vessels {rw[0]:.0f}%, flux {rw[1]:.0f}%, on directed loops {rw[2]:.1f}%, direction agree {rw[3]:.1f}%, r_log {rw[4]:.3f}")
    figT, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8)); figT.subplots_adjust(wspace=0.55, left=0.08, right=0.91, bottom=0.18, top=0.92)
    ax1.plot(ks, rows[:, 0], 'o-', color='0.3', ms=3, label='vessels retained'); ax1.plot(ks, rows[:, 1], 's-', color='#2b70b4', ms=3, label='flux carried')
    ax1.plot(ks, rows[:, 2], '^-', color=MIS, ms=3, label='on directed loops')
    ax1.set_xlabel("threshold k  (|Q′| > kσ)", labelpad=1); ax1.set_ylabel("%", labelpad=1); ax1.set_ylim(0, 102)
    ax1.legend(frameon=False, loc='upper right', borderaxespad=0.1); ax1.axvline(2, color='0.7', lw=0.6, ls=':')
    ax2.plot(ks, rows[:, 3], 'o-', color='#2b70b4', ms=3); ax2.set_ylabel('direction agreement (%)', color='#2b70b4', labelpad=1)
    ax2b = ax2.twinx(); ax2b.plot(ks, rows[:, 4], 's--', color='0.4', ms=3); ax2b.set_ylabel(r'$r_{\log}$', color='0.4', labelpad=2)
    ax2.set_xlabel(r"threshold $k$", labelpad=1); ax2.axvline(2, color='0.7', lw=0.6, ls=':')
    for ax_, l_ in ((ax1, 'a'), (ax2, 'b')):
        ax_.text(-0.18, 1.02, l_, transform=ax_.transAxes, fontsize=12, fontweight='bold', va='bottom')
    plos_style.save(figT, OUT_DIR / 'fig7_S_threshold')
    # SI: forcing sensitivity — redistribute arterial inflow across the cut vertices at fixed L/R totals (and fixed V split)
    inj_ = np.asarray(dom['inj']); side_ = np.asarray(dom['side']); Bcr = Bm.tocsr()
    def redistribute(w):
        f_ = fb.copy()
        for s_ in ('L', 'R'):
            mk_ = side_ == s_; tot_ = fb[inj_[mk_]].sum(); ww = np.maximum(np.asarray(w, float)[mk_], 0); f_[inj_[mk_]] = tot_ * ww / ww.sum()
        return f_
    def score(f_):
        q_ = passive(K, f_, dom); mk = m & (np.abs(q_) > 1e-12); mr = mk & (np.abs(Qp) > 2 * D['sigma'])
        rl_ = lambda x: np.corrcoef(np.log10(np.abs(Qp[x])), np.log10(np.abs(q_[x])))[0, 1]
        return rl_(mk), rl_(mr), 100 * np.mean(np.sign(Qp[mr]) == np.sign(q_[mr]))
    stub = lambda fn: np.array([fn(Bcr[i].indices) for i in inj_])
    cases = [('reconstructed', fb), ('uniform', redistribute(np.ones(len(inj_)))),
             ('∝ radius²', redistribute(stub(lambda ix: (r[ix] ** 2).sum()))), ('∝ conductance', redistribute(stub(lambda ix: K[ix].sum())))]
    sc = np.array([score(f_) for _, f_ in cases])
    rngF = np.random.default_rng(0); rnd = np.array([score(redistribute(rngF.dirichlet(np.ones(len(inj_))))) for _ in range(40)])
    fx = fb.copy(); fx[inj_] = 0; fx[inj_[side_ == 'L']] = fb[inj_].sum() / (side_ == 'L').sum(); ext = np.array(score(fx))
    figF2, axs_ = plt.subplots(1, 2, figsize=(7.0, 2.8)); figF2.subplots_adjust(wspace=0.3, left=0.2, right=0.98, bottom=0.18, top=0.9)
    labs = ['reconstructed', 'uniform', r'$\propto r^2$', r'$\propto r^4/L$', 'random (n = 40)', 'all inflow from L']; y_ = np.arange(len(labs))[::-1]
    for ax_, j_, xl in ((axs_[0], 1, r'$r_{\log}$ (|Q′| > 2σ)'), (axs_[1], 2, 'direction agreement (%)')):
        vals = list(sc[:, j_]) + [rnd[:, j_].mean(), ext[j_]]; errs = [0] * len(cases) + [rnd[:, j_].std(), 0]
        cols = ['#2b70b4'] + ['0.45'] * (len(cases) - 1) + ['0.65', '#e08214']
        ax_.barh(y_, vals, color=cols, height=0.6)
        ax_.errorbar(vals[-2], y_[-2], xerr=errs[-2], fmt='none', ecolor='k', lw=0.8, capsize=2)
        ax_.set_yticks(y_, labs if j_ == 1 else [''] * len(labs)); ax_.set_xlabel(xl, labelpad=1)
        ax_.set_xlim(0.4 if j_ == 1 else 70, 0.85 if j_ == 1 else 100); ax_.spines[['top', 'right']].set_visible(False)
        for yy, vv in zip(y_, vals):
            ax_.text(vv + (0.012 if j_ == 1 else 0.8) + (errs[list(y_).index(yy)] if True else 0), yy, f'{vv:.2f}' if j_ == 1 else f'{vv:.1f}', va='center', fontsize=8)
    axs_[0].text(-0.45, 1.03, 'a', transform=axs_[0].transAxes, fontsize=12, fontweight='bold', va='bottom')
    axs_[1].text(-0.1, 1.03, 'b', transform=axs_[1].transAxes, fontsize=12, fontweight='bold', va='bottom')
    print('[fig7 SI forcing] ' + '; '.join(f'{l_}: r {v[1]:.3f}, dir {v[2]:.1f}%' for l_, v in zip(labs, list(sc) + [rnd.mean(0), ext])))
    plos_style.save(figF2, OUT_DIR / 'fig7_S_forcing')
    plos_style.save(fig, OUT_DIR / ('fig7' if np.isfinite(SIGMA_P) else 'fig7_no_vein_penalty')); plos_style.save(figF, OUT_DIR / 'fig7_S_boundary_distance')
    print(f"[fig7] r_log {r_log:.3f}, P(dir agree|filtered) {p_agree:.2f}, mismatches {int(mism.sum())}, "
          f"n {int(m.sum())}; wrote {OUT_DIR / 'fig7.png'}")


if __name__ == '__main__':
    main()
