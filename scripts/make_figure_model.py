"""Revised Fig 7 (2026-11-20) — mean and pulsatile flow in the compliant network: reconstructed field vs distributed RC network.

  a  2 × 4 maps (rows reconstructed / RC network): mean flow |Q|, pulsatile / mean power (|H1|² + |H2|²)/Q̄², and the H1 and H2
     phase lags behind the vitelline veins; the three example vessels and the boundary nodes are marked
  b  loss of the distensibility fit against D (scripts/mech/ac_estimate.py, --method measured)
  c  one cardiac cycle of three vessels, reconstructed vs RC network, each normalised by its own mean; the relative error of
     the modelled mean flux is written in each panel
Inputs (all written by scripts/run_stage.sh): results/dc_reconciled_v3.pkl, results/ac_field_v8.pkl, results/ac_estimate_v9.json.
Writes figures/fig7_model/fig7_model.{png,pdf}.

Derived from the earlier RC-network figure, whose description follows (panel letters there are the old ones; the boundary-
forcing panel is still built, for the node markers, but hidden).

Figure 9 — pulsatile flow: measured field vs distributed RC network, and the distensibility fit.

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
OUT_DIR = None                                                     # set in main(): figures/fig7_model


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
    ap.add_argument('--field', default='ac_field_v8', help='results/<field>.pkl (canonical 2026-10-23: ac_field_v5_cont, DA anchor + seam continuity)')
    ap.add_argument('--tag', default=None, help='output name suffix: fig9_<tag>')
    ap.add_argument('--abs-q', action='store_true', help='panel e in nL/s (each vessel in its own flow direction, own y axis) '
                    'instead of Q/Q̄; the model curve then also shows its mean-flow mismatch')
    args = ap.parse_args()

    plos_style.apply()
    G = _paths.load_graph()
    global OUT_DIR
    OUT_DIR = _paths.FIGURES / 'fig7_model'; OUT_DIR.mkdir(parents=True, exist_ok=True)
    D = pickle.loads((_paths.RESULTS / 'dc_reconciled_v3.pkl').read_bytes())
    A = pickle.loads((_paths.RESULTS / f'{args.field}.pkl').read_bytes())
    dom = carved_domain(G); assert dom['edges'] == D['edges'] == A['edges']
    bc = Path(args.bc_json) if args.bc_json else _paths.RESULTS / 'ac_estimate_v9.json'
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
    from plexusmech.ac.model import vein_forcing
    from plexusmech.ac.group_bc import groups as _groups
    _Cv, VINFO = vein_forcing(C, A, mode='mean', gauge_rad=None)
    F0 = {k: _Cv['F'][(k, 'flux')].copy() for k in (1, 2)}               # reconstructed boundary forcing (before group corrections)
    d_main, eta_main, bc_lab = D_MAIN, ETA_MAIN, 'local measured ratio'
    if J:
        d_main, eta_main = J.get('D_parabola', J['D']), J['eta']; bc_lab = J['forcing']
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
    for n_ in list(dom['V']):                                   # zero = vitelline veins, as Fig 7
        for e_ in G.edges(n_):
            for m in G.edges[e_].get('measurements_piv') or []:
                if int(m['tile_id']) == ZT and m.get('Q_t') is not None and (m.get('quality') or {}).get('kl_ok'):
                    a0_, H_ = _phz(m['Q_t'], tf0[ZT]); w_ = max(_coh(m['Q_t'], tf0[ZT]), 0)
                    for k in (1, 2):
                        zr[k] += w_ * np.exp(1j * np.angle(np.sign(a0_) * H_[k - 1] * np.exp(-1j * k * (psi14 + d14))))
    phref = {k: np.angle(zr[k]) for k in (1, 2)}
    print(f'[fig9] phase reference (tile-14 DA/V stubs): H1 {np.degrees(phref[1]):.0f}°, H2 {np.degrees(phref[2]):.0f}°')
    lag = lambda H, dc, k: (np.degrees(np.angle(np.exp(1j * phref[k]) * np.conj(H * np.sign(dc)))) + (90 if k == 1 else 30)) % 360 - (90 if k == 1 else 30)   # H1: [−90°, 270°), the wrap in the empty part of the distribution
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
        picks = [int(cand[np.argmin(np.abs(pires[cand] - np.percentile(pires[cand], q_)))]) for q_ in (10, 50, 90)]
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
    WIDE = 1.31                                                 # no boundary-forcing panel (2026-11-20): the maps span the full width
    fig = plt.figure(figsize=(7.5, 3.1 + 1.95 * aspect * WIDE), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45 * aspect * WIDE, 0.85], hspace=0.34, left=0.075, right=0.985, top=0.95, bottom=0.07)
    gA = outer[0].subgridspec(2, 4, hspace=0.05, wspace=0.03); gF = outer[0]   # gF: boundary-forcing axes, built for the node markers then hidden
    gB = None
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.25, 2.9], wspace=0.2)
    gE = bot[1].subgridspec(1, 3, wspace=0.32 if args.abs_q else 0.12)

    nPF = mcolors.LogNorm(0.05, 0.6); cPF = plt.cm.magma
    nPH = mcolors.Normalize(0, 220); nPH2 = mcolors.Normalize(0, 330); cPH = plt.cm.plasma          # as Fig 7 (H1); H2 spans a full cycle
    allmask = np.ones(nE, bool); axes_A = {}
    qa_ = np.abs(np.r_[Qp, Qdc_s]); qa_ = qa_[qa_ > 0]; nQ = mcolors.LogNorm(max(np.percentile(qa_, 2), 1e-4), np.percentile(qa_, 98)); cQ = plt.cm.viridis
    TT = ['Mean flow', 'Relative pulsatile power', 'H1 phase lag', 'H2 phase lag']
    for row, (side, pf, Hr, dcr, mask, pmask) in enumerate((('Reconstructed', pf_m, Hm_, Qp, e, e if gcorr else ph_ok),
                                                            ('RC network', pf_s, Hs, Qdc_s, allmask, allmask))):   # whole carved network
        for col in range(4):
            ax = fig.add_subplot(gA[row, col]); bg(ax); axes_A[row, col] = ax
            if col == 0: edgemap(ax, np.abs(dcr), allmask & (np.abs(dcr) > 0), nQ, cQ, 0.45)
            elif col == 1: edgemap(ax, pf, mask, nPF, cPF, 0.45)
            else: edgemap(ax, np.clip(lag(Hr[col - 1], dcr, col - 1), 0, None), pmask, nPH if col == 2 else nPH2, cPH, 0.4)
            if row == 0:
                ax.set_title(TT[col], fontsize=9, pad=3)
            if col == 0:
                ax.text(-0.06, 0.5, side, transform=ax.transAxes, rotation=90, ha='center', va='center', fontsize=9)
    for col, (nrm, cm_, lab) in enumerate(((nQ, cQ, r'$|Q|$ (nL/s)'), (nPF, cPF, 'Pulsatile / mean power'),
                                           (nPH, cPH, r'H1 delay ($^\circ$)'), (nPH2, cPH, r'H2 delay ($^\circ$)'))):
        cax = axes_A[1, col].inset_axes([0.08, -0.08, 0.84, 0.035])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation='horizontal')
        cb.set_label(lab, fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
        if nrm is nPF:
            cb.set_ticks([0.05, 0.2, 0.6]); cb.set_ticklabels(['0.05', '0.2', '0.6']); cb.ax.minorticks_off()
        elif nrm is nPH:
            cb.set_ticks([0, 90, 180])
        elif nrm is nPH2:
            cb.set_ticks([0, 180])

    # c  pulsatile RMS scatter + statistics
    axC = fig.add_axes([0, 0, 0.01, 0.01]); axC.set_visible(False)
    sel = e & (rms_s > 0) & (rms_m > 0)
    lo, hi = 3e-4, 1.0
    axC.hexbin(np.clip(rms_m[sel], lo, hi), np.clip(rms_s[sel], lo, hi), gridsize=28, xscale='log', yscale='log', bins='log',
               cmap='Greys', mincnt=1, linewidths=0, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)))
    axC.plot([lo, hi], [lo, hi], color='#d62728', lw=0.9)
    axC.set_xlim(lo, hi); axC.set_ylim(lo, hi); axC.set_aspect('equal')
    axC.set_xticks([1e-3, 1e-2, 1e-1]); axC.set_yticks([1e-3, 1e-2, 1e-1]); axC.minorticks_off()
    axC.set_xlabel('Measured (nL/s)', fontsize=8, labelpad=1); axC.set_ylabel('RC network (nL/s)', fontsize=8, labelpad=1)
    axC.tick_params(labelsize=8); axC.set_title('Pulse amplitude', fontsize=9, pad=3)
    axC.text(0.05, 0.95, f"r = {stats['rms_corr']:.2f}", transform=axC.transAxes, va='top', fontsize=8)
    axC.spines[['top', 'right']].set_visible(False)

    # d  profiled objective vs D
    axD = fig.add_subplot(bot[0])
    if J and J.get('D_grid') and J.get('S'):
        Dg = np.array(J['D_grid']); Sg = np.array(J['S'][0]); dS = Sg - Sg.min()
        axD.plot(Dg * 1e4, dS, 'o-', color='0.2', ms=2.5, lw=1.0)
        axD.axvline(d_main * 1e4, color='#d62728', lw=0.8, ls='--')
        axD.text(d_main * 1e4, 0.97 * dS.max(), f" $\\hat D$ = {d_main * 1e4:.1f}", color='#d62728', fontsize=8, va='top', ha='left')
        axD.set_xlabel('$D$ ($10^{-4}$ Pa$^{-1}$)', fontsize=8, labelpad=1)
        axD.set_ylabel(r'Loss, $\Delta\mathcal{L}(D)$', fontsize=8, labelpad=1); axD.tick_params(labelsize=8)
        axD.set_ylim(-0.03 * dS.max(), 1.05 * dS.max())
        print(f'[fig9] D profile: min at {Dg[np.argmin(Sg)]:.3g} (fit {d_main:.3g}); Δ at grid ends {dS[0]:.0f} / {dS[-1]:.0f}')
    axD.set_title('Effective network distensibility', fontsize=9, pad=3); axD.spines[['top', 'right']].set_visible(False)

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
        ax.plot(t, wave(Qp[c], {j: Hm_[j][c] for j in (1, 2)}), color='#2ca02c', lw=1.4, zorder=2, label='Reconstructed')
        ax.plot(t, wave(Qdc_s[c], {j: Hs[j][c] for j in (1, 2)}), color='#d62728', lw=1.0, ls='--', zorder=3, label='RC network')
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1']); ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        dm = 100 * (abs(Qdc_s[c]) / abs(Qp[c]) - 1)          # RC-network mean flow vs the measured (reconciled) mean
        ax.text(0.97, 0.97, f'mean {dm:+.0f} %'.replace('-', '\u2212'), transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#d62728')   # top right: clear of every waveform
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
    if not args.abs_q:                                          # headroom so the 'mean' notes clear every waveform (shared y axis)
        _yl = axs_e[0].get_ylim(); axs_e[0].set_ylim(_yl[0], _yl[1] + 0.13 * (_yl[1] - _yl[0]))
    axs_e[1].legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, borderaxespad=0.0, handlelength=1.8)
    axs_e[1].set_title(' ', fontsize=9, pad=12)

    pick_marks(axes_A[0, 1])
    # a  boundary forcing: inputs of the model (thin = reconstructed from nearby vessels, thick = after the fitted group corrections)
    GCOL = {0: '#1f78b4', 1: '#e66101', 2: '#1b9e77', 3: '#1b9e77'}; GLAB = {0: 'arterial L', 1: 'arterial R', 2: 'venous (shared)', 3: 'venous (shared)'}
    bnd = np.asarray(dom['bnd']); gid = _groups(C); fdc = (dom['B'] @ C['Qp'])[bnd]; th_ = np.linspace(0, 1, 300)
    axF = fig.add_subplot(gF); bg(axF); axF.images[0].set_alpha(0.6)
    edgemap(axF, np.zeros(nE), np.zeros(nE, bool), nPF, cPF, 0.3)
    allxy = np.array([[G.nodes[dom['nodes'][j]]['x'], Hmos - G.nodes[dom['nodes'][j]]['y']] for j in bnd[gid < 4]])
    cx_, cy_ = allxy.mean(0); hh_ = 0.5 * (allxy[:, 1].max() - allxy[:, 1].min()) + 700
    pfA = axes_A[0, 0].get_position(); pfB = axes_A[1, 0].get_position(); pF0 = axF.get_position()
    xF0 = 0.008; hfig = pfA.y1 - pfB.y0; wfig = pF0.x1 - xF0                 # left edge shared with panel c's letter
    axF.set_axes_locator(None); axF.set_position([xF0, pfB.y0, wfig, hfig]); axF.set_aspect('auto')
    W_, H_ = fig.get_size_inches(); hw_ = hh_ * (wfig * W_) / (hfig * H_)
    axF.set_xlim(cx_ - hw_, cx_ + hw_); axF.set_ylim(cy_ + hh_, cy_ - hh_)
    FF = {k: C['F'][(k, 'flux')] for k in (1, 2)}
    nxy = lambda j: np.array([G.nodes[dom['nodes'][j]]['x'], Hmos - G.nodes[dom['nodes'][j]]['y']])
    def wv(r1, r2):
        return 1 + (r1 * np.exp(2j * np.pi * th_)).real + (r2 * np.exp(4j * np.pi * th_)).real
    for g_ in (0, 1, 2, 3):                                           # all boundary nodes, small
        m = gid == g_
        if m.any():
            xy_ = np.array([nxy(j) for j in bnd[m]])
            for axm in (axF, axes_A[0, 2]):
                axm.plot(xy_[:, 0], xy_[:, 1], '^' if g_ < 2 else 's', ms=3.5 if axm is axF else 3.0, mfc=GCOL[g_], mec='white', mew=0.4, ls='', zorder=9)
    # insets: VL, VR (own stub waveform thin, shared forcing thick); two largest arterial sources per side (reconstructed thin, fitted thick)
    vnodes = list(np.asarray(dom['out'])); vside = list(dom['side_out'])
    big = {}
    for sd in (0, 1):
        order = [bnd[i] for i in np.argsort(-np.abs(fdc)) if gid[i] == sd]
        first = order[0]; second = next(j for j in order[1:] if sd == 1 or np.linalg.norm(nxy(j) - nxy(first)) > 150)   # left: > 150 px apart
        big[sd] = [first, second]
    PANELS = []
    for j, sd in zip(vnodes, vside):
        own = (VINFO['ratio'][1].get(int(j), np.nan), VINFO['ratio'][2].get(int(j), np.nan)); shr = (VINFO['mean'][1], VINFO['mean'][2])
        PANELS.append(('V' + sd, j, '#1b9e77', [(shr, 1.8, 1.0)], [0.02, 0.835, 0.40, 0.115] if sd == 'L' else [0.58, 0.835, 0.40, 0.115]))
    for sd, lab, col, xs in ((0, 'A', '#1f78b4', 0.0), (1, 'A', '#e66101', 0.0)):
        for k_, j in enumerate(big[sd]):
            i_ = int(np.flatnonzero(bnd == j)[0])
            rec = (F0[1][j] / fdc[i_], F0[2][j] / fdc[i_]); fit = (FF[1][j] / fdc[i_], FF[2][j] / fdc[i_])
            nm = f"A{('L', 'R')[sd]}{k_ + 1} {100 * abs(fdc[i_]) / np.abs(fdc[gid == sd]).sum():.0f}%"
            yf = (axF.transLimits.transform(nxy(big[sd][0]))[1]); y2 = (axF.transLimits.transform(nxy(big[sd][1]))[1])
            yb_ = min(max(yf - 0.07, 0.2), 0.62) + (0.17 if (sd == 0 and y2 > yf) else -0.17) * k_
            PANELS.append((nm, j, col, [(fit, 1.8, 1.0)], [0.015 if sd == 0 else 0.735, yb_, 0.25, 0.115]))
    for nm, j, col, curves, box in PANELS:
        ai = axF.inset_axes(box); ai.set_facecolor((1, 1, 1, 0.92))
        for (r1, r2), lw_, al_ in curves:
            if np.isfinite(r1):
                ai.plot(th_, wv(r1, r2), color=col, lw=lw_ + 0.3, alpha=al_)
        ai.axhline(1, color='0.8', lw=0.5); ai.set_xlim(0, 1); ai.set_ylim(-0.12, 1.85); ai.set_xticks([]); ai.set_yticks([])
        for sp_ in ai.spines.values():
            sp_.set_edgecolor(col); sp_.set_linewidth(0.9)
        import matplotlib.patheffects as _pe
        ai.text(0.0, 1.05, nm, transform=ai.transAxes, fontsize=8, color=col, va='bottom', ha='left', fontweight='bold', zorder=10,   # label above the box: never over the curve
                path_effects=[_pe.withStroke(linewidth=2.2, foreground='white')])
        p_ = nxy(j); axF.plot(*p_, 'o', ms=7, mfc='none', mec='k', mew=0.9, zorder=10)
        ex = box[0] + box[2] if box[0] < 0.5 else box[0]; ey = box[1] + box[3] / 2
        axF.annotate('', xy=p_, xytext=(ex, ey), textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color=col, lw=0.6), zorder=8)
        log_ = f"{nm}: node {dom['nodes'][j]}"; print('[fig8a]', log_)
    axF.set_title('Boundary forcing', fontsize=9, pad=3); axF.set_visible(False)
    from matplotlib.lines import Line2D as _L2
    axF.text(0.98, 0.02, '', transform=axF.transAxes)
    pa, pc, pd, pe = axes_A[0, 0].get_position(), axC.get_position(), axD.get_position(), axs_e[0].get_position(); pf = axF.get_position()
    xl = pa.x0 - 0.045; yt = pa.y1 + 0.02; yb = pd.y1 + 0.03
    fig.canvas.draw(); _r = fig.canvas.get_renderer(); _inv = fig.transFigure.inverted(); yb = axD.title.get_window_extent(_r).transformed(_inv).y0
    fig.canvas.draw(); _r = fig.canvas.get_renderer(); _inv = fig.transFigure.inverted()
    _tb = lambda ax_: ax_.title.get_window_extent(_r).transformed(_inv).y0
    yab = _tb(axes_A[0, 0]); ycd = _tb(axD)
    for x_, y_, l_ in ((0.008, yab, 'a'), (0.008, ycd, 'b'), (pe.x0 - 0.04, ycd, 'c')):
        fig.text(x_, y_, l_, fontsize=12, fontweight='bold', va='baseline', ha='left')
    # (main figure saved below; SI off)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'fig7_model.{ext}', dpi=300)
    return
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
