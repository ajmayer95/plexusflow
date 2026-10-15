"""AC (waveform) field figure — the measured, tile-aligned harmonic field on the carved network, no model.

Needs results/dc_reconciled_v2.pkl (scripts/dc_reconcile.py) and results/ac_field_v2.pkl (scripts/ac_field.py).
The model comparison (RC network vs measured) is scripts/mech/make_figure9alt_v3.py.

  A  pulsatility fraction (|H1|²+|H2|²)/|Q̄|², H1 phase, H2 phase (mod π; phase-determined vessels only)
  B  tile time gauge: bootstrap SD of each vessel's winner tile; junction residual |Σ s Ĥ| / Σ|Ĥ| for the aligned field,
     the field with the tile gauge undone, and a random-phase null
  C  measured waveforms (DC + H1 + H2) at six vessels: --picks "u,v;u,v;…", else chosen automatically

Writes figures/fig_ac_field/fig_ac_field.{png,pdf}.  Writes nothing to any graph.
"""
from __future__ import annotations
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import carved_domain
from plexusflow.ac.field import junction_residuals, phase_determined

GAUGE_SD_MAX = 30.0     # deg: phases of tiles with a less certain time gauge are greyed
OUT_DIR = _paths.FIGURES / 'fig_ac_field'


def auto_picks(segs, e, rho, Qp, n=6):
    """Well-measured, coherent vessels spread over the network (farthest-point sampling on midpoints)."""
    cand = np.flatnonzero(e & np.isfinite(rho))
    cand = cand[(rho[cand] >= np.percentile(rho[cand], 75)) & (np.abs(Qp[cand]) >= np.percentile(np.abs(Qp[cand]), 50))]
    mid = np.array([segs[c][len(segs[c]) // 2] for c in cand])
    chosen = [int(np.argmax(rho[cand]))]
    for _ in range(n - 1):
        d = np.min(np.linalg.norm(mid[:, None, :] - mid[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return [int(cand[i]) for i in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--picks', default=None, help='six edges "u,v;u,v;…" for panel C (default: automatic)')
    args = ap.parse_args()

    G = _paths.load_graph()
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    A = pickle.loads((_paths.RESULTS / 'ac_field_v2.pkl').read_bytes())
    dom = carved_domain(G); assert dom['edges'] == D['edges'] == A['edges']
    edges = dom['edges']; nE = len(edges)
    e, Qp, rho = A['elig'], np.asarray(D['Qp'], float), A['rho']
    H = {k: np.where(e, A['H'][k - 1], np.nan) for k in (1, 2)}
    ph_ok = phase_determined(A, GAUGE_SD_MAX)
    gsd = A.get('gauge_sd_deg') or {}
    sd_e = np.array([gsd.get(int(t), np.nan) if t >= 0 else np.nan for t in A['tile']], float)
    n_j, res = junction_residuals(A, dom)
    print(f'[fig_ac_field] {_paths.STAGE.name}: carved edges {nE}, AC-eligible {int(e.sum())}, phase-determined '
          f'{int(ph_ok.sum())} (tile gauge SD ≤ {GAUGE_SD_MAX:.0f}°); ref tile {A.get("ref_tile", "?")} ({A["f_ref"]:.2f} Hz)')
    print('[fig_ac_field] junction residual (' + str(n_j) + ' junctions): ' + '; '.join(
        f'H{k} aligned {r["aligned"]:.3f} unaligned {r["unaligned"]:.3f} random {r["random"]:.3f}' for k, r in res.items()))

    pf = (np.abs(H[1]) ** 2 + np.abs(H[2]) ** 2) / np.maximum(np.abs(Qp), 1e-30) ** 2
    # phase relative to the vessel's mean-flow direction, so flow reversal along a path does not read as a π jump
    ph = lambda Hk: np.mod(np.angle(Hk * np.sign(Qp)), np.pi)

    mos = tifffile.imread(str(_paths.MOSAIC)); mos = mos.mean(axis=-1) if mos.ndim == 3 else mos
    Hmos, Wmos = mos.shape
    segs = []
    for uv in edges:
        d = G.edges[uv]
        p = np.asarray(d['path'], float) if d.get('path') is not None else \
            np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in uv])
        segs.append(np.column_stack([p[:, 0], Hmos - p[:, 1]]))
    pos = {}
    for c, (u, v) in enumerate(edges):
        pos[(u, v)] = c; pos[(v, u)] = c
    picks = ([pos.get(tuple(int(x) for x in s.split(','))) for s in args.picks.split(';')] if args.picks
             else auto_picks(segs, ph_ok, rho, Qp))
    for k, c in enumerate(picks):
        print(f'  pick {k + 1}: ' + ('not in carved network' if c is None else
                                     f'edge {edges[c]}, eligible={bool(e[c])}, ρ={rho[c]:.2f}, Q\'={Qp[c]:+.3f}'))

    allxy = np.vstack(segs); pad = 60
    x0, x1 = max(allxy[:, 0].min() - pad, 0), min(allxy[:, 0].max() + pad, Wmos)
    y0, y1 = max(allxy[:, 1].min() - pad, 0), min(allxy[:, 1].max() + pad, Hmos)

    def bg(ax):
        ax.imshow(mos, cmap='gray', vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), aspect='equal', zorder=0)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)

    def edgemap(ax, val, mask, norm, cmap, lw=1.2):
        ax.add_collection(LineCollection([segs[c] for c in np.flatnonzero(~mask)], colors=(0.35, 0.35, 0.35, 0.35),
                                         linewidths=0.4, zorder=2))
        sel = np.flatnonzero(mask & np.isfinite(val))
        ax.add_collection(LineCollection([segs[c] for c in sel], colors=cmap(norm(val[sel])), linewidths=lw, zorder=3))

    def pick_marks(ax):
        for k, c in enumerate(picks):
            if c is None: continue
            mid = segs[c][len(segs[c]) // 2]
            ax.plot(*mid, 'o', color='#f5d400', markersize=16, markeredgecolor='black', markeredgewidth=1.3, zorder=8)
            ax.text(*mid, str(k + 1), ha='center', va='center', fontsize=9, fontweight='bold', zorder=9)

    def cbar(ax, norm, cmap, label):
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax.inset_axes([0.1, -0.07, 0.8, 0.025]),
                          orientation='horizontal')
        cb.set_label(label, fontsize=10); cb.ax.tick_params(labelsize=8)
        return cb

    aspect = (y1 - y0) / (x1 - x0)
    fig = plt.figure(figsize=(21, 7.5 + 2 * 5.2 * aspect), facecolor='white')
    outer = fig.add_gridspec(3, 1, height_ratios=[aspect, aspect, 1.0], hspace=0.30, left=0.03, right=0.98,
                             top=0.97, bottom=0.04)
    gA = outer[0].subgridspec(1, 3, wspace=0.04)
    gB = outer[1].subgridspec(1, 3, wspace=0.10, width_ratios=[1.0, 0.65, 1.0])
    gC = outer[2].subgridspec(1, 6, wspace=0.30)

    nPF = mcolors.LogNorm(0.05, 0.6); cPF = plt.cm.magma
    nPH = mcolors.Normalize(0, np.pi); cPH = plt.cm.hsv
    for col, (val, mask, nrm, cm_, title, lab) in enumerate((
            (pf, e, nPF, cPF, 'pulsatility fraction', r'$(|H_1|^2 + |H_2|^2)/|\bar Q|^2$'),
            (ph(H[1]), ph_ok, nPH, cPH, r'$H_1$ phase', r'$H_1$ phase  (rad, mod $\pi$)'),
            (ph(H[2]), ph_ok, nPH, cPH, r'$H_2$ phase', r'$H_2$ phase  (rad, mod $\pi$)'))):
        ax = fig.add_subplot(gA[col]); bg(ax); edgemap(ax, val, mask, nrm, cm_, 1.2)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=5)
        cb = cbar(ax, nrm, cm_, lab)
        if nrm is nPF:
            cb.set_ticks([0.05, 0.1, 0.2, 0.4, 0.6]); cb.set_ticklabels(['0.05', '0.1', '0.2', '0.4', '0.6']); cb.ax.minorticks_off()
        if col == 0:
            ax.text(-0.04, 1.03, 'A', transform=ax.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')
            pick_marks(ax)

    # B: gauge certainty map, per-tile SD histogram, junction residuals
    nSD = mcolors.Normalize(0, 60); cSD = plt.cm.viridis_r
    axB1 = fig.add_subplot(gB[0]); bg(axB1); edgemap(axB1, sd_e, e, nSD, cSD, 1.2)
    axB1.set_title('tile time gauge uncertainty', fontsize=13, fontweight='bold', pad=5)
    cbar(axB1, nSD, cSD, 'bootstrap SD of the winner tile gauge (deg)')
    axB1.text(-0.04, 1.03, 'B', transform=axB1.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')
    axB2 = fig.add_subplot(gB[1])
    sds = np.array([v for v in gsd.values() if np.isfinite(v)])
    axB2.hist(np.clip(sds, 0, 90), bins=np.arange(0, 95, 5), color='0.45', edgecolor='white')
    axB2.axvline(GAUGE_SD_MAX, color='#d62728', ls='--', lw=1.2)
    axB2.set_xlabel('tile gauge SD (deg)', fontsize=10); axB2.set_ylabel('tiles', fontsize=10)
    axB2.text(0.97, 0.95, f'{int(np.sum(sds <= GAUGE_SD_MAX))} of {len(sds)} tiles\n≤ {GAUGE_SD_MAX:.0f}°',
              transform=axB2.transAxes, ha='right', va='top', fontsize=10)
    axB3 = fig.add_subplot(gB[2])
    labels = ('aligned', 'gauge undone', 'random phase'); keys = ('aligned', 'unaligned', 'random')
    xk = np.arange(2); w = 0.26
    for i, (lab, key, colr) in enumerate(zip(labels, keys, ('#2ca02c', '#ff7f0e', '0.6'))):
        vals = [res[k][key] for k in (1, 2)]
        axB3.bar(xk + (i - 1) * w, vals, w, label=lab, color=colr)
        for x_, v_ in zip(xk + (i - 1) * w, vals):
            axB3.text(x_, v_ + 0.01, f'{v_:.2f}', ha='center', fontsize=9)
    axB3.set_xticks(xk); axB3.set_xticklabels([r'$H_1$', r'$H_2$'], fontsize=12)
    axB3.set_ylabel(r'junction residual  $|\sum s\hat H| / \sum|\hat H|$', fontsize=10)
    axB3.set_title(f'flow balance of the harmonics ({n_j} junctions)', fontsize=11)
    axB3.legend(fontsize=9, frameon=False, loc='upper left'); axB3.set_ylim(0, 0.8)

    # C: measured waveforms
    t = np.linspace(0, 1, 400)
    for k, c in enumerate(picks):
        ax = fig.add_subplot(gC[k])
        if c is None or not e[c]:
            ax.text(0.5, 0.5, 'not AC-eligible', transform=ax.transAxes, ha='center', color='0.4'); continue
        q = np.sign(Qp[c]) * (Qp[c] + sum((H[j][c] * np.exp(2j * np.pi * j * t)).real for j in (1, 2)))
        ax.plot(t, q, color='#2ca02c', lw=2.2); ax.axhline(abs(Qp[c]), color='0.5', lw=0.8, ls=':')
        ax.set_xlabel('t / T', fontsize=10); ax.tick_params(labelsize=9)
        if k == 0:
            ax.set_ylabel('|Q| along mean flow (nL/s)', fontsize=10)
            ax.text(-0.12, 1.08, 'C', transform=ax.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')
        ax.text(0.08, 0.92, str(k + 1), transform=ax.transAxes, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(fc='#f5d400', ec='black', lw=1.3, boxstyle='circle,pad=0.25'))
        if not ph_ok[c]:
            ax.text(0.97, 0.05, 'phase not determined', transform=ax.transAxes, ha='right', fontsize=8, color='0.4')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / 'fig_ac_field.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'fig_ac_field.pdf', bbox_inches='tight', facecolor='white')
    print(f"[fig_ac_field] wrote {OUT_DIR / 'fig_ac_field.png'}")


if __name__ == '__main__':
    main()
