"""Figure 9 — pulsatile flow: measured field vs distributed RC network, and the distensibility fit.

Needs results/dc_reconciled_v2.pkl, results/<field>.pkl (AC field, default ac_field_v7) and the profiled fit
results/ac_estimate_v8.json (scripts/mech/ac_estimate.py: D, η, fitted boundary forcing, tile-gauge corrections and the
profiled objective over a D grid).

  a  2 × 3 maps (rows measured / RC network): pulsatility fraction (|H1|²+|H2|²)/Q̄², H1 and H2 phase lag (deg) behind the
     Fig 8 reference (ρ-weighted tile-14 dorsal-aorta + vitelline-vein stub phase, H_n in each vessel's own flow direction);
     both rows on the same reference.  Measured row: AC-eligible vessels; RC-network row: every vessel of the carved
     network, unmeasured ones included (the model is solved everywhere)
  b  normalised-harmonic residual sqrt(Σ_k=1,2 |h_k,meas − h_k,sim|²), h_k = H_k sgn(Q̄)/|Q̄| (complex; not the classical
     PI = (Q_max − Q_min)/Q̄); examples of e marked
  c  pulsatile RMS measured vs RC network (log), with r_log, n and median H1/H2 phase errors
  d  profiled fitting objective vs D (boundary group corrections re-fitted at each D; η = 0) — a composite pseudo-likelihood,
     conditional on the radii, resistance scale and boundary model; no interval drawn
  e  Q/Q̄ over one beat at three well-measured vessels (marked in b), each titled with the RC network's mean flow relative to the measured mean (Q/Q̄ divides each
     curve by its own mean)
SI fig9_S_distensibility: H1 lag and pulsatility fraction, measured vs the network at low / fitted / high D (boundary
forcing held at the D̂ fit).  Writes figures/fig9/fig9[_<tag>].*; nothing to any graph.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import plos_style
import plexusflow.paths as _paths
from plexusflow.reconcile import carved_domain
from plexusmech.ac.model import build_context, solve

D_MAIN, ETA_MAIN, VEIN_BC = 3.5e-4, 0.0, 'flux'
GAUGE_SD_MAX = 30.0     # deg: tiles with a less certain time gauge are greyed in the measured phase maps
OUT_DIR = _paths.FIGURES / 'fig9'                          # paper Fig 9 (Fig 10 until 2026-11-12)


PAPER_PICKS = '67236,67239;36453,36993;93211,93245'   # 21S panel e vessels (auto rule on the 2026-11-02 field)


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
    ap.add_argument('--picks', default=PAPER_PICKS, help='three edges "u,v;u,v;…" for panel e (default: the paper\'s; "auto" = '
                    'good / typical / poor by harmonic-residual percentile 20/50/80 among well-measured vessels)')
    ap.add_argument('--bc-json', default=None, help='profiled fit (D, η, boundary forcing, D profile); default results/ac_estimate_v8.json')
    ap.add_argument('--field', default='ac_field_v7', help='results/<field>.pkl (canonical 2026-10-23: ac_field_v5_cont, DA anchor + seam continuity)')
    ap.add_argument('--tag', default=None, help='output name suffix: fig9_<tag>')
    ap.add_argument('--abs-q', action='store_true', help='panel e in nL/s (each vessel in its own flow direction, own y axis) '
                    'instead of Q/Q̄; the model curve then also shows its mean-flow mismatch')
    args = ap.parse_args()

    plos_style.apply()
    G = _paths.load_graph()
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v2.pkl').read_bytes())
    A = pickle.loads((_paths.RESULTS / f'{args.field}.pkl').read_bytes())
    dom = carved_domain(G); assert dom['edges'] == D['edges'] == A['edges']
    bc = Path(args.bc_json) if args.bc_json else _paths.RESULTS / 'ac_estimate_v8.json'
    J = json.loads(bc.read_text()) if bc.exists() else None
    if J is None:
        print(f'[fig9] {bc} not found: falling back to D = {D_MAIN:g}, no profile panel (run scripts/mech/ac_estimate.py)')
    gcorr = (J or {}).get('gauge_correction_deg')
    if gcorr:
        dlt = np.radians(np.array([gcorr.get(str(int(t)), 0.0) if t >= 0 else 0.0 for t in A['tile']]))
        A = dict(A, H=A['H'] * np.exp(-1j * np.arange(1, A['H'].shape[0] + 1)[:, None] * dlt[None, :]))
        print(f'[fig9] tile gauge corrections applied: {sum(abs(v) > 1 for v in gcorr.values())} tiles moved > 1°, '
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
    print(f'[fig9] phase-determined vessels {int(ph_ok.sum())} of {int(e.sum())} eligible (tile gauge SD ≤ {GAUGE_SD_MAX:.0f}°)')
    print(f'[fig9] {_paths.STAGE.name}: carved edges {nE}, AC-eligible {int(e.sum())}; ref tile '
          f'{A.get("ref_tile", "?")} ({C["f0"]:.2f} Hz); D={d_main:g}, η={eta_main}; boundary: {bc_lab}')

    # phase reference = Fig 8's zero: ρ-weighted H1 (H2) phase of the dorsal-aorta and vitelline-vein stubs, all recorded in
    # tile 14 (one clock), taken into this field's gauge (ψ_14 and the fitted correction δ_14), each in its own flow direction
    from plexusflow.reconcile.harmonics import phasors as _phz, coherence as _coh
    ZT = 14; tf0 = G.graph['tile_f0_piv']; psi14 = float(A['psi'][ZT]); d14 = np.radians((gcorr or {}).get(str(ZT), 0.0))
    zr = {1: 0j, 2: 0j}
    for n_ in list(dom['DA']) + list(dom['V']):
        for e_ in G.edges(n_):
            for m in G.edges[e_].get('measurements_piv') or []:
                if int(m['tile_id']) == ZT and m.get('Q_t') is not None and (m.get('quality') or {}).get('kl_ok'):
                    a0_, H_ = _phz(m['Q_t'], tf0[ZT]); w_ = max(_coh(m['Q_t'], tf0[ZT]), 0)
                    for k in (1, 2):
                        zr[k] += w_ * np.exp(1j * np.angle(np.sign(a0_) * H_[k - 1] * np.exp(-1j * k * (psi14 + d14))))
    phref = {k: np.angle(zr[k]) for k in (1, 2)}
    print(f'[fig9] phase reference (tile-14 DA/V stubs): H1 {np.degrees(phref[1]):.0f}°, H2 {np.degrees(phref[2]):.0f}°')
    lag = lambda H, dc, k: (np.degrees(np.angle(np.exp(1j * phref[k]) * np.conj(H * np.sign(dc)))) + 30) % 360 - 30
    pulsfrac = lambda H1, H2, dc: (np.abs(H1) ** 2 + np.abs(H2) ** 2) / np.maximum(np.abs(dc), 1e-30) ** 2
    pf_m, pf_s = pulsfrac(Hm_[1], Hm_[2], Qp), pulsfrac(Hs[1], Hs[2], Qdc_s)
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
    print('[fig9] ' + ', '.join(f'{k} {v:.3f}' for k, v in stats.items()))
    print(f'[fig9] n: RMS correlation {int((e & (rms_s > 0) & (rms_m > 0)).sum())}, phase errors {int(ph_ok.sum())}')

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
    if args.picks and args.picks != 'auto' and all(tuple(int(x) for x in s_.split(',')) in pos for s_ in args.picks.split(';')):
        picks = [pos.get(tuple(int(x) for x in s.split(','))) for s in args.picks.split(';')]
    else:                                                       # good / typical / poor agreement among well-measured vessels
        cand = np.flatnonzero(e & np.isfinite(rho) & np.isfinite(pires))
        cand = cand[(rho[cand] >= np.percentile(rho[cand], 75)) & (np.abs(Qp[cand]) >= np.percentile(np.abs(Qp[cand]), 50))]
        picks = [int(cand[np.argmin(np.abs(pires[cand] - np.percentile(pires[cand], q_)))]) for q_ in (20, 50, 80)]
    for k, c in enumerate(picks):
        print(f'  pick {k + 1}: ' + ('not in carved network' if c is None else
                                     f'edge {edges[c]}, eligible={bool(e[c])}, ρ={rho[c]:.2f}, Q\'={Qp[c]:+.3f}, harmonic residual '
                                     f'{pires[c]:.2f} (percentile {100 * np.mean(pires[np.isfinite(pires) & e] <= pires[c]):.0f} of eligible)'))

    # map extent: crop to the network (+ margin) so tall/wide embryos fill their panels
    allxy = np.vstack(segs); pad = 60
    x0, x1 = max(allxy[:, 0].min() - pad, 0), min(allxy[:, 0].max() + pad, Wmos)
    y0, y1 = max(allxy[:, 1].min() - pad, 0), min(allxy[:, 1].max() + pad, Hmos)

    def bg(ax):
        ax.imshow(mos, cmap='gray', vmin=np.percentile(mos, 2), vmax=np.percentile(mos, 98), aspect='equal', zorder=0, alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)

    def edgemap(ax, val, mask, norm, cmap, lw=1.2):
        ax.add_collection(LineCollection([segs[c] for c in np.flatnonzero(~mask)], colors=(0.35, 0.35, 0.35, 0.35),
                                         linewidths=0.2, zorder=2))
        sel = np.flatnonzero(mask & np.isfinite(val))
        ax.add_collection(LineCollection([segs[c] for c in sel], colors=cmap(norm(val[sel])), linewidths=lw, zorder=3))

    def pick_marks(ax):
        for k, c in enumerate(picks):
            if c is None: continue
            mid = segs[c][len(segs[c]) // 2]
            ax.plot(*mid, 'o', color='#f5d400', markersize=8, markeredgecolor='black', markeredgewidth=0.6, zorder=8)
            ax.text(*mid, str(k + 1), ha='center', va='center', fontsize=8, fontweight='bold', zorder=9)

    aspect = (y1 - y0) / (x1 - x0)          # >1 for tall embryos
    fig = plt.figure(figsize=(7.5, 3.1 + 2.6 * aspect), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45 * aspect, 0.62], hspace=0.42, left=0.085, right=0.97, top=0.93, bottom=0.1)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.2)
    gA = top[0].subgridspec(2, 3, hspace=0.05, wspace=0.04)
    gB = top[1].subgridspec(2, 1, hspace=0.62)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 3.0], wspace=0.28)
    gE = bot[1].subgridspec(1, 3, wspace=0.32 if args.abs_q else 0.12)

    nPF = mcolors.LogNorm(0.05, 0.6); cPF = plt.cm.magma
    nPH = mcolors.Normalize(-30, 330); cPH = plt.cm.twilight_shifted          # cyclic: −30° ≡ 330°
    allmask = np.ones(nE, bool); axes_A = {}
    for row, (side, pf, Hr, dcr, mask, pmask) in enumerate((('Measured', pf_m, Hm_, Qp, e, e if gcorr else ph_ok),
                                                            ('RC network', pf_s, Hs, Qdc_s, allmask, allmask))):   # whole carved network
        for col in range(3):
            ax = fig.add_subplot(gA[row, col]); bg(ax); axes_A[row, col] = ax
            if col == 0: edgemap(ax, pf, mask, nPF, cPF, 0.5)
            else: edgemap(ax, lag(Hr[col], dcr, col), pmask, nPH, cPH, 0.45)
            if row == 0:
                ax.set_title(['Pulsatility fraction', 'Fundamental phase lag', '2nd-harmonic phase lag'][col], fontsize=9, pad=3)
            if col == 0:
                ax.text(-0.05, 0.5, side, transform=ax.transAxes, rotation=90, ha='center', va='center', fontsize=9)
    for col, (nrm, cm_, lab) in enumerate(((nPF, cPF, 'Pulsatile / mean power'), (nPH, cPH, 'Lag (°)'), (nPH, cPH, 'Lag (°)'))):
        cax = axes_A[1, col].inset_axes([0.1, -0.08, 0.8, 0.035])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation='horizontal')
        cb.set_label(lab, fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
        if nrm is nPF:
            cb.set_ticks([0.05, 0.1, 0.2, 0.6]); cb.set_ticklabels(['0.05', '0.1', '0.2', '0.6']); cb.ax.minorticks_off()
        else:
            cb.set_ticks([0, 90, 180, 270])

    # b  PI residual map
    nPI = mcolors.LogNorm(0.05, 1.5); cPI = plt.cm.magma
    axB = fig.add_subplot(gB[0]); bg(axB); edgemap(axB, pires, e, nPI, cPI, 0.45); pick_marks(axB)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nPI, cmap=cPI), cax=axB.inset_axes([1.03, 0.10, 0.05, 0.80]))
    cb.set_label('Harmonic residual', fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
    axB.set_title('Harmonic residual', fontsize=9, pad=3)

    # c  pulsatile RMS scatter + statistics
    axC = fig.add_subplot(gB[1])
    sel = e & (rms_s > 0) & (rms_m > 0)
    axC.scatter(rms_m[sel], rms_s[sel], c=pires[sel], s=1.5, cmap=cPI, norm=nPI, alpha=0.6, linewidths=0, rasterized=True)
    lo = max(min(np.percentile(rms_m[sel], 2), np.percentile(rms_s[sel], 2)) * 0.5, 1e-4)
    hi = max(np.percentile(rms_m[sel], 99.5), np.percentile(rms_s[sel], 99.5)) * 2
    axC.plot([lo, hi], [lo, hi], 'k--', lw=0.5, alpha=0.6)
    axC.set_xscale('log'); axC.set_yscale('log'); axC.set_xlim(lo, hi); axC.set_ylim(lo, hi); axC.set_aspect('equal')
    axC.set_xlabel('Measured pulsatile RMS (nL/s)', fontsize=8, labelpad=1)
    axC.set_ylabel('RC network (nL/s)', fontsize=8, labelpad=1); axC.tick_params(labelsize=8)
    axC.text(0.0, 1.03, f"$r_{{\\log}}$ = {stats['rms_corr']:.2f}\nmedian $|\\Delta\\phi_1|$ = {stats['h1_dphi_deg']:.0f}°, $|\\Delta\\phi_2|$ = {stats['h2_dphi_deg']:.0f}°",
             transform=axC.transAxes, va='bottom', fontsize=8, linespacing=1.2)
    axC.spines[['top', 'right']].set_visible(False)

    # d  profiled objective vs D
    axD = fig.add_subplot(bot[0])
    if J and J.get('D_grid') and J.get('S'):
        Dg = np.array(J['D_grid']); Sg = np.array(J['S'][0]); dS = Sg - Sg.min()
        axD.plot(Dg * 1e4, dS, 'o-', color='0.2', ms=2.5, lw=1.0)
        axD.axvline(d_main * 1e4, color='#d62728', lw=0.8, ls='--')
        axD.text(d_main * 1e4, 0.97 * dS.max(), f" $\\hat D$ = {d_main * 1e4:.1f}", color='#d62728', fontsize=8, va='top', ha='left')
        axD.set_xlabel('Distensibility $D$ ($10^{-4}$ Pa$^{-1}$)', fontsize=8, labelpad=1)
        axD.set_ylabel(r'$\Delta\mathcal{L}(D)$, profiled', fontsize=8, labelpad=1); axD.tick_params(labelsize=8)
        axD.set_ylim(-0.03 * dS.max(), 1.05 * dS.max())
        print(f'[fig9] D profile: min at {Dg[np.argmin(Sg)]:.3g} (fit {d_main:.3g}); Δ at grid ends {dS[0]:.0f} / {dS[-1]:.0f}')
    axD.set_title('Distensibility fit', fontsize=9, pad=3); axD.spines[['top', 'right']].set_visible(False)

    # e  waveforms, Q/Q̄ over one beat
    t = np.linspace(0, 1, 400)
    def wave(dc_, Hk):
        q = dc_ + sum((Hk[k] * np.exp(2j * np.pi * k * t)).real for k in (1, 2))
        return q * np.sign(dc_) if args.abs_q else q / dc_
    axs_e = []
    for k, c in enumerate(picks):
        ax = fig.add_subplot(gE[k], sharey=None if args.abs_q or not axs_e else axs_e[0]); axs_e.append(ax)
        if c is None:
            ax.text(0.5, 0.5, 'not in carved network', transform=ax.transAxes, ha='center', color='red', fontsize=8); continue
        if not args.abs_q:
            ax.axhline(1, color='0.85', lw=0.5, zorder=1)
        ax.plot(t, wave(Qp[c], {j: Hm_[j][c] for j in (1, 2)}), color='#2ca02c', lw=1.4, zorder=2, label='Measured')
        ax.plot(t, wave(Qdc_s[c], {j: Hs[j][c] for j in (1, 2)}), color='#d62728', lw=1.0, ls='--', zorder=3, label='RC network')
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1']); ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        dm = 100 * (abs(Qdc_s[c]) / abs(Qp[c]) - 1)          # RC-network mean flow vs the measured (reconciled) mean
        ax.set_title(f'Model mean {dm:+.0f} %'.replace('-', '\u2212'), fontsize=8, pad=3, color='#d62728')
        if args.abs_q:
            ax.set_ylim(bottom=0)
            if k == 0:
                ax.set_ylabel('$Q$ (nL/s)', fontsize=8, labelpad=1)
        elif k > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r'$Q/\bar Q$', fontsize=8, labelpad=1)
        if k == 1:
            ax.set_xlabel('$t/T$', fontsize=8, labelpad=1)
        ax.text(0.04, 0.95, str(k + 1), transform=ax.transAxes, ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(fc='#f5d400', ec='black', lw=0.6, boxstyle='circle,pad=0.2'))
        print(f"[fig9] e{k + 1} ({('good', 'typical', 'poor')[k]}): edge {edges[c]}, harmonic residual {pires[c]:.2f}")
    axs_e[0].legend(fontsize=8, loc='lower left', frameon=False, borderaxespad=0.1, handlelength=1.8)

    # panel letters on a grid
    pa, pb, pc, pd, pe = axes_A[0, 0].get_position(), axB.get_position(), axC.get_position(), axD.get_position(), axs_e[0].get_position()
    xl = pa.x0 - 0.045; xr = pb.x0 - 0.04; yt = max(pa.y1, pb.y1) + 0.02; yb = pd.y1 + 0.03
    for x_, y_, l_ in ((xl, yt, 'a'), (xr, yt, 'b'), (xr, pc.y1 + 0.055, 'c'), (xl, yb, 'd'), (pe.x0 - 0.035, yb, 'e')):
        fig.text(x_, y_, l_, fontsize=12, fontweight='bold', va='bottom', ha='left')

    # SI: stiff / fitted / compliant (boundary forcing held at the D̂ fit)
    if J and J.get('D_grid'):
        Dlo, Dhi = float(min(J['D_grid'])), float(max(J['D_grid']))
        figS = plt.figure(figsize=(7.5, 3.9 * aspect + 0.8), facecolor='white')
        gS = figS.add_gridspec(2, 4, hspace=0.06, wspace=0.04, left=0.05, right=0.99, top=0.91, bottom=0.17)
        cols = [('Measured', Hm_, Qp, e if gcorr else ph_ok, e)]
        for nm_, Dv in ((f'Stiff, $D$ = {Dlo * 1e4:.1f}', Dlo), (f'Fitted, $D$ = {d_main * 1e4:.1f}', d_main), (f'Compliant, $D$ = {Dhi * 1e4:.1f}', Dhi)):
            Hd = {k: solve(C, k, Dv, eta_main, VEIN_BC) for k in (1, 2)}; cols.append((nm_, Hd, Qdc_s, allmask, allmask))
            if Dv != d_main:
                _dd = np.angle(np.exp(1j * (np.angle(Hm_[1][ph_ok] * np.sign(Qp[ph_ok])) - np.angle(Hd[1][ph_ok] * np.sign(Qdc_s[ph_ok])))))
                print(f'[fig9 SI] D {Dv:.2g}: median |Δφ1| {np.degrees(np.median(np.abs(_dd))):.0f}°')
        for j_, (nm_, Hr, dcr, pmask, mask) in enumerate(cols):
            for row in range(2):
                ax = figS.add_subplot(gS[row, j_]); bg(ax)
                if row == 0:
                    edgemap(ax, lag(Hr[1], dcr, 1), pmask, nPH, cPH, 0.45); ax.set_title(nm_, fontsize=9, pad=3)
                else:
                    edgemap(ax, pulsfrac(Hr[1], Hr[2], dcr), mask, nPF, cPF, 0.5)
                if j_ == 0:
                    ax.text(-0.05, 0.5, ['Fundamental lag', 'Pulsatility fraction'][row], transform=ax.transAxes, rotation=90, ha='center', va='center', fontsize=8)
        for row, (nrm, cm_, lab) in enumerate(((nPH, cPH, 'Lag (°)'), (nPF, cPF, 'Pulsatile / mean power'))):
            cax = figS.add_axes([0.3 + 0.38 * row, 0.09, 0.25, 0.022])
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation='horizontal')
            cb.set_label(lab, fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
            if nrm is nPF:
                cb.set_ticks([0.05, 0.1, 0.2, 0.6]); cb.set_ticklabels(['0.05', '0.1', '0.2', '0.6']); cb.ax.minorticks_off()
            else:
                cb.set_ticks([0, 90, 180, 270])
        figS.text(0.99, 0.965, '$D$ in $10^{-4}$ Pa$^{-1}$; boundary forcing held at the fitted values', ha='right', fontsize=8, color='0.3')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plos_style.save(figS, OUT_DIR / ('fig9_S_distensibility' + (f'_{args.tag}' if args.tag else '')))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = 'fig9' + (f'_{args.tag}' if args.tag else '')
    plos_style.save(fig, OUT_DIR / stem)
    print(f"[fig9] wrote {OUT_DIR / f'{stem}.png'}")


if __name__ == '__main__':
    main()
