"""Figure 9 alt v3 — fig9alt_v2 panels on the repo pipeline (plexusflow.ac), for any stage.

Needs results/dc_reconciled_v2.pkl (scripts/dc_reconcile.py) and results/ac_field_v2.pkl (scripts/ac_field.py).

  measured  DC  = Q' (final DC reconciliation);  H_k = Option-A field (coherent tile gauge, eligible edges only)
  simulated distributed RC network (exact two-port, plexusflow.ac.model), consensus radii, D = 3.5e-4 /Pa, η = 0;
            boundary: emergent DC f = BQ' × local measured harmonic ratio at carve vertices and veins ('flux')

  A  2 × 3 maps (rows measured / simulated): pulsatility fraction (|H1|²+|H2|²)/|Q̄|², H1 phase (mod π), H2 phase (mod π)
  B  PI residual map  sqrt(|ΔPI1|²+|ΔPI2|²), PI_k = H_k·sgn(Q̄)/|Q̄|  + ‖Q(t)‖_RMS measured vs simulated
  C  waveforms (DC + H1 + H2) at six vessels: --picks "u,v;u,v;…", else chosen automatically (high coherence,
     well-measured, spread across the network)

  --bc-json results/ac_estimate_v3.json (scripts/mech/ac_estimate.py): use the profiled best fit instead (D, η and the boundary forcing with
            its fitted group corrections, from the (D, η) profile likelihood of plexusmech.ac.group_bc); with --tag
            the output is fig9alt_v3_<tag>.{png,pdf}.  If the file has 'gauge_correction_deg' (a fitted correction δ_t of
            each tile's time gauge), the measured harmonics are corrected H_n·e^{−inδ_t} and the gauge-uncertain tiles are
            drawn in the measured phase maps (their absolute phase is then set by the fit; statistics still use only
            tiles with a determined gauge)

Writes figures/fig9alt_v3/fig9alt_v3[_<tag>].{png,pdf}.  Writes nothing to any graph.
"""
from __future__ import annotations
import sys
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import plexusflow.paths as _paths
from plexusflow.reconcile import carved_domain
from plexusmech.ac.model import build_context, solve

D_MAIN, ETA_MAIN, VEIN_BC = 3.5e-4, 0.0, 'flux'
GAUGE_SD_MAX = 30.0     # deg: tiles with a less certain time gauge are greyed in the measured phase maps
OUT_DIR = _paths.FIGURES / 'fig9alt_v3'


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
    ap.add_argument('--bc-json', default=None, help='profiled fit (D, η, boundary forcing per node), e.g. results/ac_profiled_bc_local_ratio.json')
    ap.add_argument('--field', default='ac_field_v5_cont', help='results/<field>.pkl (canonical 2026-10-23: ac_field_v5_cont, DA anchor + seam continuity)')
    ap.add_argument('--tag', default=None, help='output name suffix: fig9alt_v3_<tag>')
    args = ap.parse_args()

    G = _paths.load_graph()
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    A = pickle.loads((_paths.RESULTS / f'{args.field}.pkl').read_bytes())
    dom = carved_domain(G); assert dom['edges'] == D['edges'] == A['edges']
    J = json.loads(Path(args.bc_json).read_text()) if args.bc_json else None
    gcorr = (J or {}).get('gauge_correction_deg')
    if gcorr:
        dlt = np.radians(np.array([gcorr.get(str(int(t)), 0.0) if t >= 0 else 0.0 for t in A['tile']]))
        A = dict(A, H=A['H'] * np.exp(-1j * np.arange(1, A['H'].shape[0] + 1)[:, None] * dlt[None, :]))
        print(f'[fig9alt_v3] tile gauge corrections applied: {sum(abs(v) > 1 for v in gcorr.values())} tiles moved > 1°, '
              f'largest ' + ', '.join(f'{t}: {v:+.0f}°' for t, v in sorted(gcorr.items(), key=lambda kv: -abs(kv[1]))[:5]))
    C = build_context(D, A, dom)
    d_main, eta_main, bc_lab = D_MAIN, ETA_MAIN, 'local measured ratio'
    if J:
        d_main, eta_main = J['D'], J['eta']; bc_lab = J['forcing']
        nix = {n: i for i, n in enumerate(dom['nodes'])}; C['F'] = dict(C['F'])
        for k in (1, 2):
            F = np.zeros(C['nN'], complex); F[[nix[n] for n in J['nodes']]] = [complex(a, b) for a, b in J[f'f{k}']]
            C['F'][(k, 'flux')] = F
    e, Qp, Qdc_s = C['elig'], C['Qp'], C['Qs_dc']
    Hm_ = C['H']; rho = A['rho']
    # phases are only comparable across tiles whose time gauge is well determined (bootstrap SD ≤ GAUGE_SD_MAX)
    gsd = A.get('gauge_sd_deg') or {}
    ph_ok = e & np.array([gsd.get(int(t), 0.0) <= GAUGE_SD_MAX if t >= 0 else False for t in A['tile']])
    Hs = {k: solve(C, k, d_main, eta_main, VEIN_BC) for k in (1, 2)}
    edges = dom['edges']; nE = len(edges)
    print(f'[fig9alt_v3] phase-determined vessels {int(ph_ok.sum())} of {int(e.sum())} eligible (tile gauge SD ≤ {GAUGE_SD_MAX:.0f}°)')
    print(f'[fig9alt_v3] {_paths.STAGE.name}: carved edges {nE}, AC-eligible {int(e.sum())}; ref tile '
          f'{A.get("ref_tile", "?")} ({C["f0"]:.2f} Hz); D={d_main:g}, η={eta_main}; boundary: {bc_lab}')

    pulsfrac = lambda H1, H2, dc: (np.abs(H1) ** 2 + np.abs(H2) ** 2) / np.maximum(np.abs(dc), 1e-30) ** 2
    pf_m, pf_s = pulsfrac(Hm_[1], Hm_[2], Qp), pulsfrac(Hs[1], Hs[2], Qdc_s)
    ph = lambda H: np.mod(np.angle(H), np.pi)
    PI = lambda H, dc: H * np.sign(dc) / np.maximum(np.abs(dc), 1e-30)
    pires = np.sqrt(sum(np.abs(PI(Hm_[k], Qp) - PI(Hs[k], Qdc_s)) ** 2 for k in (1, 2)))
    rms_m = np.sqrt(sum(np.abs(Hm_[k]) ** 2 for k in (1, 2)))
    rms_s = np.sqrt(sum(np.abs(Hs[k]) ** 2 for k in (1, 2)))
    lc = lambda a, b: np.corrcoef(np.log(a), np.log(b))[0, 1]
    dphi = lambda k: np.degrees(np.median(np.abs(np.angle(np.exp(1j * (np.angle(Hm_[k][ph_ok] * np.sign(Qp[ph_ok]))
                                                                        - np.angle(Hs[k][ph_ok] * np.sign(Qdc_s[ph_ok]))))))))
    ok = e & (pf_m > 0) & (pf_s > 0)
    stats = dict(pf_corr=lc(pf_m[ok], pf_s[ok]), rms_corr=lc(rms_m[e & (rms_s > 0)], rms_s[e & (rms_s > 0)]),
                 h1_dphi_deg=dphi(1), h2_dphi_deg=dphi(2), pires=np.median(pires[e]))
    print('[fig9alt_v3] ' + ', '.join(f'{k} {v:.3f}' for k, v in stats.items()))

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
    if args.picks:
        picks = [pos.get(tuple(int(x) for x in s.split(','))) for s in args.picks.split(';')]
    else:
        picks = auto_picks(segs, e, rho, Qp)
    for k, c in enumerate(picks):
        print(f'  pick {k + 1}: ' + ('not in carved network' if c is None else
                                     f'edge {edges[c]}, eligible={bool(e[c])}, ρ={rho[c]:.2f}, Q\'={Qp[c]:+.3f}'))

    # map extent: crop to the network (+ margin) so tall/wide embryos fill their panels
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
            ax.plot(*mid, 'o', color='#f5d400', markersize=18, markeredgecolor='black', markeredgewidth=1.4, zorder=8)
            ax.text(*mid, str(k + 1), ha='center', va='center', fontsize=10, fontweight='bold', zorder=9)

    aspect = (y1 - y0) / (x1 - x0)          # >1 for tall embryos
    fig = plt.figure(figsize=(21, 11.5 + 6.5 * aspect), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.6 * aspect, 1.0], hspace=0.10, left=0.03, right=0.98,
                             top=0.97, bottom=0.04)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.11)
    gA = top[0].subgridspec(2, 3, hspace=0.06, wspace=0.04)
    gB = top[1].subgridspec(2, 1, hspace=0.22)
    gC = outer[1].subgridspec(2, 3, hspace=0.35, wspace=0.22)

    nPF = mcolors.LogNorm(0.05, 0.6); cPF = plt.cm.magma
    nPH = mcolors.Normalize(0, np.pi); cPH = plt.cm.hsv
    allmask = np.ones(nE, bool); axes_A = {}
    for row, (side, pf, H1, H2, mask, pmask) in enumerate((('measured', pf_m, Hm_[1], Hm_[2], e, e if gcorr else ph_ok),
                                                           ('simulated', pf_s, Hs[1], Hs[2], allmask, allmask))):
        for col in range(3):
            ax = fig.add_subplot(gA[row, col]); bg(ax); axes_A[row, col] = ax
            if col == 0: edgemap(ax, pf, mask, nPF, cPF, 1.3)
            elif col == 1: edgemap(ax, ph(H1), pmask, nPH, cPH, 1.1)
            else: edgemap(ax, ph(H2), pmask, nPH, cPH, 1.1)
            if row == 0:
                ax.set_title(['pulsatility fraction', r'$H_1$ phase lag', r'$H_2$ phase lag'][col],
                             fontsize=13, fontweight='bold', pad=5)
            if col == 0:
                ax.text(-0.04, 0.5, side, transform=ax.transAxes, rotation=90, ha='center', va='center',
                        fontsize=12, fontweight='bold')
            if row == 0 and col == 0:
                ax.text(-0.05, 1.03, 'A', transform=ax.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')
    for col, (nrm, cm_, lab) in enumerate(((nPF, cPF, r'$(|H_1|^2 + |H_2|^2)/|\bar Q|^2$'),
                                           (nPH, cPH, r'$H_1$ phase  (rad, mod $\pi$)'),
                                           (nPH, cPH, r'$H_2$ phase  (rad, mod $\pi$)'))):
        cax = axes_A[1, col].inset_axes([0.1, -0.07, 0.8, 0.025])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation='horizontal')
        cb.set_label(lab, fontsize=10); cb.ax.tick_params(labelsize=8)
        if nrm is nPF:
            cb.set_ticks([0.05, 0.1, 0.2, 0.4, 0.6]); cb.set_ticklabels(['0.05', '0.1', '0.2', '0.4', '0.6'])
            cb.ax.minorticks_off()

    # B
    nPI = mcolors.LogNorm(0.05, 1.5); cPI = plt.cm.magma
    axB1 = fig.add_subplot(gB[0]); bg(axB1); edgemap(axB1, pires, e, nPI, cPI, 1.1); pick_marks(axB1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nPI, cmap=cPI), cax=axB1.inset_axes([1.03, 0.10, 0.03, 0.80]))
    cb.set_label(r'PI residual: $\sqrt{|\Delta PI_1|^2 + |\Delta PI_2|^2}$', fontsize=10); cb.ax.tick_params(labelsize=8)
    axB1.text(-0.05, 1.03, 'B', transform=axB1.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')
    axB2 = fig.add_subplot(gB[1])
    sel = e & (rms_s > 0) & (rms_m > 0)
    axB2.scatter(rms_m[sel], rms_s[sel], c=pires[sel], s=5, cmap=cPI, norm=nPI, alpha=0.6, linewidths=0)
    lo = max(min(np.percentile(rms_m[sel], 2), np.percentile(rms_s[sel], 2)) * 0.5, 1e-4)
    hi = max(np.percentile(rms_m[sel], 99.5), np.percentile(rms_s[sel], 99.5),
             *[max(rms_m[c], rms_s[c]) for c in picks if c is not None and sel[c]]) * 2      # keep every pick on scale
    axB2.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.6)
    axB2.set_xscale('log'); axB2.set_yscale('log'); axB2.set_xlim(lo, hi); axB2.set_ylim(lo, hi); axB2.set_aspect('equal')
    axB2.set_xlabel(r'$\|Q(t)\|_{\mathrm{RMS}}$  measured  (nL/s)', fontsize=10)
    axB2.set_ylabel(r'$\|Q(t)\|_{\mathrm{RMS}}$  RC network  (nL/s)', fontsize=10)
    axB2.text(0.03, 0.97, f"r(log) = {stats['rms_corr']:.3f}\nn = {int(sel.sum())}", transform=axB2.transAxes,
              va='top', fontsize=10, bbox=dict(fc='white', ec='0.6', alpha=0.9))
    for k, c in enumerate(picks):
        if c is None or not sel[c]: continue
        axB2.plot(rms_m[c], rms_s[c], 'o', color='#f5d400', markersize=15, markeredgecolor='black', markeredgewidth=1.3, zorder=7)
        axB2.text(rms_m[c], rms_s[c], str(k + 1), ha='center', va='center', fontsize=9, fontweight='bold', zorder=8)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nPI, cmap=cPI), cax=axB2.inset_axes([1.03, 0.10, 0.03, 0.80]))
    cb.set_label('PI residual', fontsize=10); cb.ax.tick_params(labelsize=8)

    # C waveforms (DC + H1 + H2)
    t = np.linspace(0, 1, 400)
    def wave(dc_, Hk):
        return np.sign(dc_) * (dc_ + sum((Hk[k] * np.exp(2j * np.pi * k * t)).real for k in (1, 2)))
    for k, c in enumerate(picks):
        ax = fig.add_subplot(gC[k // 3, k % 3])
        if c is None:
            ax.text(0.5, 0.5, 'not in carved network', transform=ax.transAxes, ha='center', color='red'); continue
        if e[c]:
            ax.plot(t, wave(Qp[c], {j: Hm_[j][c] for j in (1, 2)}), color='#2ca02c', lw=2.6, zorder=2,
                    label='measured (conservative)' if k == 0 else None)
        else:
            ax.text(0.5, 0.9, 'measured: not AC-eligible', transform=ax.transAxes, ha='center', fontsize=9, color='0.4')
        ax.plot(t, wave(Qdc_s[c], {j: Hs[j][c] for j in (1, 2)}), color='#d62728', lw=1.8, ls='--', zorder=3,
                label=f'RC network (D={d_main:.2g}, η={eta_main:g})' if k == 0 else None)
        ax.set_xlabel('t / T', fontsize=10); ax.set_ylabel('Q (nL/s)', fontsize=10); ax.tick_params(labelsize=9)
        ax.text(0.03, 0.95, str(k + 1), transform=ax.transAxes, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(fc='#f5d400', ec='black', lw=1.3, boxstyle='circle,pad=0.25'))
        if k == 0:
            ax.legend(fontsize=9, loc='lower left', frameon=False)
            ax.text(-0.07, 1.08, 'C', transform=ax.transAxes, fontsize=22, fontweight='bold', va='bottom', ha='right')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = 'fig9alt_v3' + (f'_{args.tag}' if args.tag else '')
    plt.savefig(OUT_DIR / f'{stem}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    print(f"[fig9alt_v3] wrote {OUT_DIR / f'{stem}.png'}")


if __name__ == '__main__':
    main()
