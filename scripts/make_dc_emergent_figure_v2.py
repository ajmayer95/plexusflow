"""DC figure v2 — final method (2026-10-08), from results/dc_reconciled_v2.pkl (scripts/dc_reconcile_v2.py).

  Q_meas  re-batched Kloosterman-valid measurements; per edge the tile with the smallest reliability c,
          c = c_T + c_f(1−f) + c_a(1−align)
  Q'      argmin Σ w (Q − Q_meas)²  s.t. B_I Q = 0,   w = 1/(c²|Q'|² + σ0²)  (reliability-weighted IRLS)
  f       = B Q'  (emergent injections at carve nodes, outflow at V)
  Q_sim   passive Poiseuille, K = r⁴/ℓ, consensus radii, flux f at every boundary node
Panels: A Q' on the mosaic, B Q_sim on the mosaic, C emergent boundary fluxes, D |Q'| vs |Q_sim|
(trusted vessels highlighted; radius-shuffle control in the title).
Writes figures/dc_emergent/dc_emergent_v2.{png,pdf}.  Writes nothing to the graph.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths  # stage + results locations (PLEXUSFLOW_STAGE)

STAGE = _paths.STAGE
MOSAIC = _paths.MOSAIC
OUT = _paths.FIGURES / 'dc_emergent'
ARROW_PCT = 95


def main():
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    G = pickle.loads(_paths.GRAPH.read_bytes())
    sub, nodes = D['edges'], D['nodes']; ix = {n: i for i, n in enumerate(nodes)}
    XY = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nodes])
    Qp, Qs, IM, C, Rcons = D['Qp'], D['Qs'], D['imaged'], D['c'], D['radius']
    BN, VN, side = D['inj'], D['out'], D['side']
    paths = []
    for u, v in sub:
        p = G.edges[u, v].get('path')
        p = np.asarray(p, float)[:, :2] if p is not None and len(p) >= 2 else \
            np.vstack([[G.nodes[u]['x'], G.nodes[u]['y']], [G.nodes[v]['x'], G.nodes[v]['y']]])
        if np.linalg.norm(p[0] - XY[ix[u]]) > np.linalg.norm(p[-1] - XY[ix[u]]): p = p[::-1]   # tail(u) → head(v)
        paths.append(p)
    f = np.zeros(len(nodes)); f[np.r_[BN, VN]] = D['fb'][np.r_[BN, VN]]
    alpha = f[BN]; aL, aR = alpha[side == 'L'].sum(), alpha[side == 'R'].sum()
    a_, b_ = np.abs(Qp[IM]), np.abs(Qs[IM]); ok = (a_ > 1e-12) & (b_ > 1e-12)
    r_lin = np.corrcoef(a_, b_)[0, 1]; r_log = np.corrcoef(np.log(a_[ok]), np.log(b_[ok]))[0, 1]
    tr = IM & (C < 0.5); r_tr = np.corrcoef(np.abs(Qp[tr]), np.abs(Qs[tr]))[0, 1]
    print(f'[dc-fig v2] Σα = {alpha.sum():.2f}  L = {aL:.2f}  R = {aR:.2f}  V1 {f[VN[0]]:+.2f}  V2 {f[VN[1]]:+.2f};  '
          f'corr {r_lin:.3f} (log {r_log:.3f}, trusted {r_tr:.3f}), shuffled {D["shuffled_corr"]:.3f}')

    mos = tifffile.imread(str(MOSAIC))
    if mos.ndim == 3: mos = mos.mean(axis=-1)
    Hm = mos.shape[0]
    segs = [np.column_stack([p[:, 0], Hm - p[:, 1]]) for p in paths]
    allq = np.abs(np.r_[Qp[IM], Qs[IM]])
    norm = mcolors.LogNorm(np.percentile(allq, 3), np.percentile(allq, 99.7))
    cmap = plt.cm.plasma
    lw = 0.35 + 3.2 * np.clip(Rcons / np.percentile(Rcons, 99), 0, 1)
    pts = XY[np.r_[BN, VN]]
    box = (XY[:, 0].min() - 60, XY[:, 0].max() + 60, Hm - XY[:, 1].max() - 60, Hm - XY[:, 1].min() + 60)

    def field(ax, Q, title, show_unimaged):
        ax.imshow(mos, cmap='gray', vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), alpha=0.55)
        col = [cmap(norm(max(abs(q), norm.vmin))) if (IM[i] or not show_unimaged) else (0.25, 0.25, 0.25, 1)
               for i, q in enumerate(Q)]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw))
        thr = np.percentile(np.abs(Q), ARROW_PCT)
        X, Y, U, V = [], [], [], []
        for i, s in enumerate(segs):
            if abs(Q[i]) < thr or len(s) < 2: continue
            k = len(s) // 2 - (len(s) == 2); d = s[k + 1] - s[k]; n = np.linalg.norm(d)
            if n == 0: continue
            m = s[k] + 0.5 * d; d = np.sign(Q[i]) * d / n
            X.append(m[0]); Y.append(m[1]); U.append(d[0]); V.append(d[1])
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1 / 110, width=0.0026, headwidth=4, headlength=4.5,
                  headaxislength=4, pivot='middle', color='white', edgecolor='k', linewidth=0.5, zorder=5)
        for n_ in VN: ax.plot(XY[n_, 0], Hm - XY[n_, 1], '*', color='#0a84ff', ms=16, mec='w', mew=1.2, zorder=6)
        ax.scatter(XY[BN, 0], Hm - XY[BN, 1], s=14, facecolor='none', edgecolor='#30d158', lw=1, zorder=6)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold')

    fig = plt.figure(figsize=(22, 18), facecolor='white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.12, wspace=0.06)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    field(axA, Qp, "A   Reconciled measured field Q′  (reliability-weighted, emergent boundary)", True)
    field(axB, Qs, "B   Passive Poiseuille flow Q_sim  (same emergent boundary)", False)
    cax = fig.add_axes([0.915, 0.47, 0.01, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='|Q| (nL/s)')

    axC = fig.add_subplot(gs[1, 0])
    order = {s_: np.argsort(-XY[BN][side == s_][:, 1]) for s_ in ('L', 'R')}
    xL = np.arange((side == 'L').sum()); xR = np.arange((side == 'R').sum()) + len(xL) + 2
    aLv = alpha[side == 'L'][order['L']]; aRv = alpha[side == 'R'][order['R']]
    axC.bar(xL, aLv, color=np.where(aLv >= 0, '#8e44ad', '#bfa2d6'), width=0.8)
    axC.bar(xR, aRv, color=np.where(aRv >= 0, '#d35400', '#f0b78e'), width=0.8)
    xv = xR[-1] + 3 + 2.0 * np.arange(2)
    axC.bar(xv, f[VN], color='#1f5fa8', width=0.8)
    axC.axhline(0, color='0.3', lw=0.8)
    axC.set_xticks(np.r_[xL.mean(), xR.mean(), xv])
    axC.set_xticklabels([f'left carve boundary\n({int((side == "L").sum())} nodes, top→bottom)\nΣ = {aL:.2f}',
                         f'right carve boundary\n({int((side == "R").sum())} nodes, top→bottom)\nΣ = {aR:.2f}', 'V1', 'V2'], fontsize=10)
    axC.set_ylabel('emergent boundary flux (nL/s; + = into plexus)')
    axC.set_title(f'C   Emergent boundary fluxes — left : right = {aL/(aL+aR)*100:.0f} : {aR/(aL+aR)*100:.0f}  '
                  f'(Σ in = {alpha.sum():.2f}, Σ V = {f[VN].sum():.2f})', loc='left', fontsize=13, fontweight='bold')
    for x_, val in zip(xv, f[VN]): axC.text(x_, val - 0.1, f'{val:.2f}', ha='center', va='top', fontsize=9)

    axD = fig.add_subplot(gs[1, 1])
    fl = np.percentile(allq, 1)
    axD.loglog(np.maximum(a_, fl), np.maximum(b_, fl), '.', ms=2.5, alpha=0.25, color='0.6', label='all measured')
    axD.loglog(np.maximum(np.abs(Qp[tr]), fl), np.maximum(np.abs(Qs[tr]), fl), '.', ms=2.5, alpha=0.45, color='#5e3c99', label='trusted (c < 0.5)')
    axD.legend(loc='upper left', fontsize=9, markerscale=4)
    lim = [fl, allq.max() * 1.5]; axD.plot(lim, lim, 'k--', lw=1)
    axD.set_xlim(lim); axD.set_ylim(lim); axD.set_aspect('equal')
    axD.set_xlabel('|Q′|  conservative measured (nL/s)'); axD.set_ylabel('|Q_sim|  passive geometry (nL/s)')
    axD.set_title(f'D   corr = {r_lin:.3f} (log {r_log:.3f}; trusted {r_tr:.3f}),  shuffled radii {D["shuffled_corr"]:.3f},  '
                  f'{int(IM.sum())} measured edges', loc='left',
                  fontsize=13, fontweight='bold')

    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / 'dc_emergent_v2.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT / 'dc_emergent_v2.pdf', bbox_inches='tight', facecolor='white')
    print(f'[dc-fig v2] wrote {OUT}/dc_emergent_v2.png/.pdf')


if __name__ == '__main__':
    main()
