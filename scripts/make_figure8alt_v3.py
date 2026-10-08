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

Writes figures/fig8alt_v2/fig8alt_v2.{png,pdf}.  Writes nothing to any graph.
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
from plexusflow.reconcile import carved_domain, passive

STAGE = _paths.STAGE
MOSAIC_P = STAGE / 'fuse' / 'organ_fusion_close.tif'
OUT_DIR = STAGE / 'figures' / 'fig8alt_v3'
SCHEME, RADIUS, N_SHUFFLE = 'irls', 'consensus', 10


def main():
    import pickle
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    G = pickle.loads(_paths.GRAPH.read_bytes())
    dom = carved_domain(G)
    G.graph['boundary'] = dict(DA=dom['DA'], V=dom['V'])
    edges = dom['edges']; assert edges == D['edges']
    idx = np.arange(len(edges)); em = np.ones(len(edges), bool)
    rec = dict(length=D['length'])
    Qp, Qs, fb, imaged, r = D['Qp'], D['Qs'], D['fb'], D['imaged'], D['radius']
    K = r ** 4 / D['length']; nodes = dom['nodes']
    print(f"[fig8alt_v3] carved: {len(idx)} edges ({int(imaged.sum())} measured); "
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
    nC = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1); cC = plt.cm.RdBu_r

    def bg(ax):
        ax.imshow(mos, cmap='gray', vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), aspect='equal', zorder=0)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(0, Wm); ax.set_ylim(Hm, 0)

    def markers(ax):
        for n in G.graph['boundary']['DA']:
            ax.plot(*nxy[n], marker='v', color='#8b0000', ms=11, markeredgecolor='white', mew=1.2, zorder=6, alpha=0.6)
        for n in G.graph['boundary']['V']:
            ax.plot(*nxy[n], marker='*', color='#1f5fa8', ms=15, markeredgecolor='white', mew=1.2, zorder=6)

    def field(ax, Q, faint_unimaged):
        cols, lws = [], []
        for c in range(len(idx)):
            col = cAB(nAB(min(max(abs(Q[c]), vmin), vmax)))
            if faint_unimaged and not imaged[c]:
                col = (*col[:3], 0.35); lws.append(0.6)
            else:
                lws.append(1.3)
            cols.append(col)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=3))

    fig = plt.figure(figsize=(17, 15.5), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.14, wspace=0.11, left=0.03, right=0.94, top=0.94, bottom=0.06)
    gs_C = gs[1, 0].subgridspec(2, 1, height_ratios=[1.0, 0.32], hspace=0.42)
    gs_D = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, 0.32], hspace=0.42)

    # A
    axA = fig.add_subplot(gs[0, 0]); bg(axA); field(axA, Qp, True); markers(axA)
    axA.set_title("A   Conservative mean-flow field  $|Q'|$", fontsize=14, loc='left', fontweight='bold', pad=6)
    axA.plot([], [], color=cAB(0.6), lw=0.6, alpha=0.35, label='imputed (unimaged) edge')
    axA.legend(loc='lower right', fontsize=10, frameon=True)

    # B
    axB = fig.add_subplot(gs[0, 1]); bg(axB); field(axB, Qs, False); markers(axB)
    inj = fb[dom['inj']]
    for j, nidx in enumerate(dom['inj']):
        xy = nxy[nodes[nidx]]; a = inj[j]
        col = cAB(nAB(min(max(abs(a), vmin), vmax)))
        if a >= 0:
            axB.scatter(*xy, s=55, facecolor=col, edgecolor='white', linewidths=0.9, zorder=5)
        else:
            axB.scatter(*xy, s=55, facecolor='none', edgecolor=col, linewidths=1.6, zorder=5)
    axB.scatter([], [], s=55, facecolor=cAB(0.9), edgecolor='white', label='emergent injection (color = |f|)')
    axB.scatter([], [], s=55, facecolor='none', edgecolor=cAB(0.6), linewidths=1.6, label='emergent withdrawal')
    axB.legend(loc='lower right', fontsize=10, frameon=True)
    axB.set_title('B   Passive-network prediction (emergent boundary)', fontsize=14, loc='left', fontweight='bold', pad=6)
    cax = fig.add_axes([0.945, 0.52, 0.012, 0.38])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nAB, cmap=cAB), cax=cax)
    cb.set_label(r'$|\bar Q|$   (nL/s)', fontsize=11); cb.ax.tick_params(labelsize=9)

    # C
    axC = fig.add_subplot(gs_C[0]); bg(axC)
    m = imaged & (np.abs(Qp) > 1e-12) & (np.abs(Qs) > 1e-12)
    E = np.log10(np.abs(Qs) / np.maximum(np.abs(Qp), 1e-12))
    axC.add_collection(LineCollection([segs[c] for c in np.flatnonzero(m)],
                                      colors=[cC(nC(E[c])) for c in np.flatnonzero(m)], linewidths=1.5, zorder=3))
    q_floor = max(vmin * 3, 3e-3)
    mism = m & (np.sign(Qp) != np.sign(Qs)) & ((np.abs(Qp) >= q_floor) | (np.abs(Qs) >= q_floor))
    for c in np.flatnonzero(mism):
        s = segs[c]; mid = s[len(s) // 2]
        axC.plot(*mid, 'o', color='#ff2bd6', ms=2.2, markeredgewidth=0, zorder=6, alpha=0.85)
    markers(axC)
    axC.set_title(r"C   Local model error  $E_e = \log_{10}(|Q^{sim}_e|/|Q'_e|)$", fontsize=14, loc='left',
                  fontweight='bold', pad=6)
    caxC = axC.inset_axes([-0.08, 0.10, 0.028, 0.80])
    cbC = plt.colorbar(plt.cm.ScalarMappable(norm=nC, cmap=cC), cax=caxC, ticks=[-1, -0.5, 0, 0.5, 1])
    cbC.ax.set_yticklabels(['−1\n(sim 10×\n under)', '−0.5', '0\n(agree)', '+0.5', '+1\n(sim 10×\n over)'])
    cbC.ax.yaxis.tick_left(); cbC.ax.tick_params(labelsize=9)
    axC.plot([], [], 'o', color='#ff2bd6', ms=6, label=f'direction mismatch  ({int(mism.sum())})')
    axC.legend(fontsize=10, loc='lower left', frameon=True)

    # D
    axD = fig.add_subplot(gs_D[0])
    lm, ls_ = np.log10(np.maximum(np.abs(Qp[m]), 1e-5)), np.log10(np.maximum(np.abs(Qs[m]), 1e-5))
    axD.hexbin(lm, ls_, gridsize=55, cmap='Greys', mincnt=1, linewidths=0, zorder=2)
    lo, hi = min(lm.min(), ls_.min()), max(lm.max(), ls_.max())
    axD.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.7, zorder=3)
    mm = mism[m]
    axD.scatter(lm[mm], ls_[mm], s=24, facecolors='none', edgecolors='#c22a3f', linewidths=0.9, alpha=0.85,
                zorder=4, label=f'direction mismatch  ({int(mm.sum())})')
    r_log = float(np.corrcoef(lm, ls_)[0, 1])
    consider = m & ((np.abs(Qp) >= q_floor) | (np.abs(Qs) >= q_floor))
    p_agree = float(np.mean(np.sign(Qp[consider]) == np.sign(Qs[consider])))
    axD.text(0.03, 0.97, f'$r_{{\\log}} = {r_log:.3f}$\n'
                         f'$P(\\mathrm{{dir\\ agree}}\\,|\\,\\mathrm{{filtered}}) = {p_agree:.2f}$\n$n = {int(m.sum())}$',
             transform=axD.transAxes, va='top', ha='left', fontsize=12,
             bbox=dict(fc='white', ec='0.5', lw=0.8, alpha=0.95, boxstyle='round,pad=0.35'))
    axD.set_xlabel(r"conservative measured  $\log_{10}|Q'|$", fontsize=12)
    axD.set_ylabel(r'predicted  $\log_{10}|\bar Q^{sim}|$', fontsize=12)
    axD.set_title('D   Measured vs predicted flow magnitude', fontsize=14, loc='left', fontweight='bold', pad=6)
    axD.legend(fontsize=10, loc='lower right', frameon=True); axD.set_aspect('equal')
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
    axE = fig.add_subplot(gs_C[1])
    for yi, lab, rv, col, err in ((2, 'true\nradii', r_log, '#2b70b4', 0), (1, 'uniform\nradii', r_uni, '#9e9e9e', 0),
                                  (0, 'shuffled\nradii', r_sh.mean(), '#c22a3f', r_sh.std())):
        axE.hlines(yi, 0, rv, color=col, lw=4, alpha=0.8)
        axE.errorbar(rv, yi, xerr=err if err else None, fmt='o', color=col, markersize=11, capsize=4, zorder=4)
        axE.text(rv + 0.015, yi, f'  r = {rv:.3f}' + (f' ± {err:.3f}' if err else ''), va='center', ha='left',
                 fontsize=11, fontweight='bold', color=col)
    axE.set_yticks([2, 1, 0], ['true\nradii', 'uniform\nradii', 'shuffled\nradii'], fontsize=10)
    axE.set_xlim(0, 1.0); axE.set_ylim(-0.6, 2.6)
    axE.set_xlabel(r"$r(\log|Q'|,\log|Q_{sim}|)$", fontsize=11)
    axE.spines['right'].set_visible(False); axE.spines['top'].set_visible(False)
    axE.set_title('E   Measured vessel calibers carry predictive information', fontsize=13, loc='left',
                  fontweight='bold', pad=4)

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
    axF = fig.add_subplot(gs_D[1])
    axF.plot(d_grid, r_vals, '-', color='#2b70b4', lw=2.0, zorder=3)
    axF.fill_between(d_grid, r_vals, where=~np.isnan(r_vals), color='#2b70b4', alpha=0.15, zorder=2)
    axF.set_xlim(0, d_grid.max()); axF.set_ylim(0, 1); axF.axhline(0.5, color='0.65', lw=0.6, ls=':')
    axF.set_xlabel('minimum graph distance from the driven carve boundary (hops)', fontsize=11)
    axF.set_ylabel(r"$r(\log|Q'|,\log|Q_{sim}|)$", fontsize=11)
    axF.set_title('F   Agreement persists away from the driven boundary', fontsize=13, loc='left', fontweight='bold', pad=4)
    axF2 = axF.twinx(); axF2.plot(d_grid, n_vals, '--', color='0.45', lw=1.1, alpha=0.8)
    axF2.set_ylabel('number of retained edges', fontsize=10, color='0.45'); axF2.tick_params(labelsize=9, colors='0.45')
    if len(d_grid) >= 3:
        k3 = len(d_grid) // 3
        axF2.annotate('retained edges', xy=(d_grid[k3], n_vals[k3]), xytext=(0, -14), textcoords='offset points',
                      color='0.4', fontsize=9, fontstyle='italic')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / 'fig8alt_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'fig8alt_v3.pdf', bbox_inches='tight', facecolor='white')
    print(f"[fig8alt_v3] r_log {r_log:.3f}, P(dir agree|filtered) {p_agree:.2f}, mismatches {int(mism.sum())}, "
          f"n {int(m.sum())}; wrote {OUT_DIR / 'fig8alt_v3.png'}")


if __name__ == '__main__':
    main()
