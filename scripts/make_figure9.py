"""Figure 9 — pulsatile flow: measured field vs distributed RC network, and the distensibility fit.

Needs results/dc_reconciled_v2.pkl, results/<field>.pkl (AC field, default ac_field_v7) and the profiled fit
results/ac_estimate_v8.json (scripts/mech/ac_estimate.py: D, η, fitted boundary forcing, tile-gauge corrections and the
profiled objective over a D grid).

  (layout 2026-11-16) a  boundary forcing: boundary nodes (arterial L / R, venous, topology interfaces) and the normalised
     waveforms g_b(t) = F_b(t)/F̄_b (thin: individual arterial nodes; bold: flux-weighted group), F_b = F̄_b g_b
  b (was a)  2 × 3 maps (rows measured / RC network): pulsatility fraction (|H1|²+|H2|²)/Q̄², H1 and H2 phase lag (deg) behind the
     Fig 8 reference (ρ-weighted tile-14 dorsal-aorta + vitelline-vein stub phase, H_n in each vessel's own flow direction);
     both rows on the same reference.  Measured row: AC-eligible vessels; RC-network row: every vessel of the carved
     network, unmeasured ones included (the model is solved everywhere)
  (SI fig9_S_agreement) normalised-harmonic residual sqrt(Σ_k=1,2 |h_k,meas − h_k,sim|²), h_k = H_k sgn(Q̄)/|Q̄| (complex; not the classical
     PI = (Q_max − Q_min)/Q̄); examples of e marked
  c  pulsatility fraction (dimensionless: the reconstructed mean-flow scale cancels) measured vs RC network, median H1/H2 phase
     errors; the dimensional pulsatile RMS scatter (shares the DC scale) is in SI fig9_S_agreement
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
import matplotlib.patheffects
import matplotlib.text
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
    # layout (2026-11-16, reviewer): a boundary forcing · b measured vs RC maps · c dimensionless agreement · d D fit · e waveforms;
    # the harmonic-residual map and the dimensional RMS scatter (shares the DC scale) move to SI fig9_S_agreement
    fig = plt.figure(figsize=(7.5, 2.9 + 2.75 * aspect), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.62 * aspect, 0.7], hspace=0.5, left=0.075, right=0.975, top=0.94, bottom=0.085)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.25], wspace=0.13)
    gF = top[0].subgridspec(2, 1, height_ratios=[1.0, 0.10], hspace=0.04)
    gA = top[1].subgridspec(2, 3, hspace=0.05, wspace=0.04)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.95, 0.95, 2.6], wspace=0.42)
    gE = bot[2].subgridspec(1, 3, wspace=0.32 if args.abs_q else 0.12)

    # ---- a  boundary forcing: where the model is driven and with what waveform -------------------------------------------
    nodes_ = dom['nodes']; nxy = np.array([[G.nodes[n]['x'], Hmos - G.nodes[n]['y']] for n in nodes_])
    fb_ = np.asarray(D['fb']); Fk = {k: np.asarray(C['F'][(k, 'flux')]) for k in (1, 2)}
    side_in, side_out = np.asarray(dom['side']), np.asarray(dom['side_out'])
    GRP = [('Arterial, left', np.asarray(dom['inj'])[side_in == 'L'], '#0072B2', '^'),
           ('Arterial, right', np.asarray(dom['inj'])[side_in == 'R'], '#D55E00', '^'),
           ('Venous (shared)', np.asarray(dom['out']), '#009E73', '*')]
    axF = fig.add_subplot(gF[0]); bg(axF)
    axF.add_collection(LineCollection(segs, colors=(0.3, 0.3, 0.3, 0.6), linewidths=0.35, zorder=2))
    # arterial open regions (inflow zones beside the aortae) and the aorta anchors, for orientation
    from plexusflow.reconcile.regions import is_open as _is_open
    opn_ = [np.column_stack([np.asarray(d_['path'], float)[:, 0], Hmos - np.asarray(d_['path'], float)[:, 1]])
            for _, _, d_ in G.edges(data=True) if _is_open(d_) and d_.get('path') is not None]
    axF.add_collection(LineCollection(opn_, colors=(0.85, 0.55, 0.2, 0.55), linewidths=0.9, zorder=2.5))
    for n_ in dom['DA']:
        axF.plot(G.nodes[n_]['x'], Hmos - G.nodes[n_]['y'], marker='v', ms=6, color='white', mec='black', mew=0.6, zorder=6)
    bidx = np.r_[np.asarray(dom['inj']), np.asarray(dom['out'])].astype(int)
    bxy = np.vstack([nxy[bidx], [[G.nodes[n_]['x'], Hmos - G.nodes[n_]['y']] for n_ in dom['DA']]])
    cx_, cy_ = (bxy.min(0) + bxy.max(0)) / 2
    # crop: the boundary nodes with room on both sides for the waveform insets; height follows the panel's aspect ratio
    pos_ = axF.get_position(); ar_ = (pos_.height * fig.get_figheight()) / (pos_.width * fig.get_figwidth())
    hw = 0.5 * np.ptp(bxy[:, 0]) + 640; hh = max(hw * ar_, 0.5 * np.ptp(bxy[:, 1]) + 120); hw = hh / ar_
    axF.set_xlim(cx_ - hw, cx_ + hw); axF.set_ylim(cy_ + hh, cy_ - hh)
    # highlighted arterial nodes: per side, of the 6 largest-inflow nodes the two farthest apart (A1–A4, colour = side)
    hi_nodes = []
    _tt = np.linspace(0, 1, 100)
    for (lab_, idx_, col_, mk_), tag_ in zip(GRP[:2], ('L', 'R')):
        idx_ = idx_[fb_[idx_] > 0]
        gmin_ = np.array([min(1 + sum((Fk[k][i_] / fb_[i_] * np.exp(2j * np.pi * k * _tt)).real for k in (1, 2))) for i_ in idx_])
        cand_ = idx_[gmin_ > 0][np.argsort(-fb_[idx_[gmin_ > 0]])][:6]       # the 6 largest-inflow nodes (no reversal) …
        if len(cand_) >= 2:                                                  # … the two farthest apart
            dd_ = np.linalg.norm(nxy[cand_][:, None] - nxy[cand_][None], axis=2); a_, b_ = np.unravel_index(np.argmax(dd_), dd_.shape)
            chosen = sorted([int(cand_[a_]), int(cand_[b_])], key=lambda i_: nxy[i_, 1])
        else:
            chosen = [int(c_) for c_ in cand_]
        hi_nodes += [(tag_, i_, col_) for i_ in chosen]
    hi_nodes = [(f'A{k_ + 1}', i_, col_) for k_, (_, i_, col_) in enumerate(hi_nodes)]   # A1–A4; colour gives the side
    for lab_, idx_, col_, mk_ in GRP:
        if len(idx_):
            axF.scatter(nxy[idx_, 0], nxy[idx_, 1], s=40 if mk_ == '*' else 16, marker=mk_, c=col_, edgecolors='black', linewidths=0.35, zorder=5)
    _sb = 200 / 1.7                                                                    # 200 µm at 1.7 µm/px (as Fig 8)
    axF.plot([cx_ - hw + 40, cx_ - hw + 40 + _sb], [cy_ + hh - 45] * 2, color='black', lw=1.5, zorder=7)
    axF.text(cx_ - hw + 40 + _sb / 2, cy_ + hh - 60, '200 µm', ha='center', va='bottom', fontsize=8, zorder=7)
    axF.set_title(r'Boundary forcing  $F_b(t) = \bar F_b\, g_b(t)$', fontsize=9, pad=3)
    # waveform insets on the crop: g_b(t) over one beat at each highlighted node (and the shared venous waveform)
    tt = np.linspace(0, 1, 200)
    nwave = lambda i_: 1 + sum((Fk[k][i_] / fb_[i_] * np.exp(2j * np.pi * k * tt)).real for k in (1, 2))
    vidx_ = np.asarray(dom['out'])
    gV = 1 + sum((Fk[k][vidx_].sum() / fb_[vidx_].sum() * np.exp(2j * np.pi * k * tt)).real for k in (1, 2))
    curves = [(nm_, [i_], col_, nwave(i_)) for nm_, i_, col_ in hi_nodes] + [('V1 = V2', list(vidx_), '#009E73', gV)]
    sideof_ = {nm_: ('L' if col_ == GRP[0][2] else 'R') for nm_, _, col_ in hi_nodes}
    gmax = max(float(c_[3].max()) for c_ in curves) * 1.08
    from matplotlib.patches import ConnectionPatch
    to_ax = lambda pt: axF.transAxes.inverted().transform(axF.transData.transform(pt))
    IW, IH = 0.27, 0.12
    # inset positions: beside their node, pushed apart on each side so they do not overlap
    ypos = {}
    for sd_ in ('L', 'R'):
        cs_ = sorted([c_ for c_ in curves if sideof_.get(c_[0]) == sd_], key=lambda c_: -to_ax(nxy[c_[1][0]])[1])   # top first
        ys_ = [float(np.clip(to_ax(nxy[c_[1][0]])[1] - IH / 2, 0.02, 0.97 - IH)) for c_ in cs_]
        for k_ in range(1, len(ys_)):
            gap_ = ys_[k_ - 1] - ys_[k_] - (IH + 0.03)
            if gap_ < 0:
                ys_[k_ - 1] = min(ys_[k_ - 1] - gap_ / 2, 0.97 - IH); ys_[k_] = ys_[k_ - 1] - IH - 0.03
        ypos.update({c_[0]: y_ for c_, y_ in zip(cs_, ys_)})
    for nm_, ids_, col_, g_ in curves:
        pts = [to_ax(nxy[i_]) for i_ in ids_]; ya = float(np.mean([p_[1] for p_ in pts]))
        if nm_.startswith('V'):                                            # centred above the two outlets
            xa = float(np.mean([p_[0] for p_ in pts])); x0i, y0i = float(np.clip(xa - IW / 2, 0.02, 0.98 - IW)), min(0.97 - IH, ya + 0.07)
        else:
            x0i = 0.02 if sideof_[nm_] == 'L' else 0.98 - IW; y0i = ypos[nm_]
        ins = axF.inset_axes([x0i, y0i, IW, IH], zorder=9)
        ins.set_facecolor((1, 1, 1, 0.88)); ins.axhline(1, color='0.65', lw=0.6, zorder=1)
        ins.plot(tt, g_, color=col_, lw=1.4, zorder=2); ins.set_xlim(0, 1); ins.set_ylim(0, gmax * 1.38)   # headroom for the label
        ins.set_xticks([]); ins.set_yticks([]); [sp_.set_color(col_) for sp_ in ins.spines.values()]; [sp_.set_linewidth(0.8) for sp_ in ins.spines.values()]
        if nm_.startswith('V'):
            ins.text(-0.04, 0.5, nm_, transform=ins.transAxes, ha='right', va='center', fontsize=8, fontweight='bold', color=col_,
                     path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])
        else:
            ins.text(0.04, 0.93, nm_, transform=ins.transAxes, ha='left', va='top', fontsize=8, fontweight='bold', color=col_)
        for i_ in ids_:
            if not nm_.startswith('V'):
                axF.scatter(*nxy[i_], s=70, facecolors='none', edgecolors='black', linewidths=1.0, zorder=7)
                xyB = (1.0, 0.5) if x0i < 0.5 else (0.0, 0.5)
            else:                                                          # label each outlet, connect to the inset's lower edge
                k_ = ids_.index(i_); xyB = (0.3 + 0.4 * (to_ax(nxy[i_])[0] > np.mean([p_[0] for p_ in pts])), 0.0)
                _left = to_ax(nxy[i_])[0] <= np.mean([p_[0] for p_ in pts])
                axF.annotate(f'V{k_ + 1}', nxy[i_], xytext=(-8 if _left else 8, 0), textcoords='offset points', ha='right' if _left else 'left',
                             va='center', fontsize=8,
                             fontweight='bold', color='#009E73', zorder=8,
                             path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])
            fig.add_artist(ConnectionPatch(xyA=tuple(nxy[i_]), coordsA=axF.transData, xyB=xyB, coordsB=ins.transAxes,
                                           color=col_, lw=0.7, zorder=8))
    axL = fig.add_subplot(gF[1]); axL.axis('off')
    hnd = [plt.Line2D([], [], color=c_, marker=m_, ms=5 if m_ != '*' else 7, mec='black', mew=0.3, lw=0) for _, _, c_, m_ in GRP]
    axL.legend(hnd, [g_[0] for g_ in GRP], fontsize=8, loc='upper center', ncol=3, frameon=False, handlelength=1.0,
               columnspacing=0.6, borderaxespad=0, handletextpad=0.3)

    # ---- b  measured vs RC network maps ---------------------------------------------------------------------------------
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
    pick_marks(axes_A[0, 0])
    for col, (nrm, cm_, lab) in enumerate(((nPF, cPF, 'Pulsatile / mean power'), (nPH, cPH, 'Lag (°)'), (nPH, cPH, 'Lag (°)'))):
        cax = axes_A[1, col].inset_axes([0.1, -0.08, 0.8, 0.035])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm_), cax=cax, orientation='horizontal')
        cb.set_label(lab, fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
        if nrm is nPF:
            cb.set_ticks([0.05, 0.1, 0.2, 0.6]); cb.set_ticklabels(['0.05', '0.1', '0.2', '0.6']); cb.ax.minorticks_off()
        else:
            cb.set_ticks([0, 90, 180, 270])

    # ---- c  dimensionless agreement: pulsatility fraction (DC scale cancels) + phase errors -------------------------------
    nPI = mcolors.LogNorm(0.05, 1.5); cPI = plt.cm.magma
    axC = fig.add_subplot(bot[0]); selp = ok
    axC.scatter(pf_m[selp], pf_s[selp], s=1.5, color='0.25', alpha=0.35, linewidths=0, rasterized=True)
    lo_ = max(min(np.percentile(pf_m[selp], 1), np.percentile(pf_s[selp], 1)) * 0.7, 1e-3)
    hi_ = max(np.percentile(pf_m[selp], 99), np.percentile(pf_s[selp], 99)) * 1.3
    axC.plot([lo_, hi_], [lo_, hi_], 'k--', lw=0.5, alpha=0.6)
    axC.set_xscale('log'); axC.set_yscale('log'); axC.set_xlim(lo_, hi_); axC.set_ylim(lo_, hi_); axC.set_aspect('equal')
    axC.set_xlabel('Measured pulsatility fraction', fontsize=8, labelpad=1); axC.set_ylabel('RC network', fontsize=8, labelpad=1)
    axC.tick_params(labelsize=8); axC.spines[['top', 'right']].set_visible(False)
    axC.text(0.03, 0.97, f"$r_{{\\log}}$ = {stats['pf_corr']:.2f}\nmedian $|\\Delta\\phi_1|$ {stats['h1_dphi_deg']:.0f}°\nmedian $|\\Delta\\phi_2|$ {stats['h2_dphi_deg']:.0f}°",
             transform=axC.transAxes, va='top', ha='left', fontsize=8, linespacing=1.15, zorder=5,
             bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    print(f"[fig9] c: pulsatility fraction r_log {stats['pf_corr']:.3f} (n {int(selp.sum())}); dimensional RMS r_log {stats['rms_corr']:.3f} → SI")

    # ---- d  profiled objective vs D -------------------------------------------------------------------------------------
    axD = fig.add_subplot(bot[1])
    if J and J.get('D_grid') and J.get('S'):
        Dg = np.array(J['D_grid']); Sg = np.array(J['S'][0]); dS = Sg - Sg.min()
        axD.plot(Dg * 1e4, dS, 'o-', color='0.2', ms=2.5, lw=1.0)
        axD.axvline(d_main * 1e4, color='#d62728', lw=0.8, ls='--')
        axD.text(d_main * 1e4, 0.97 * dS.max(), f" $\\hat D$ = {d_main * 1e4:.1f}", color='#d62728', fontsize=8, va='top', ha='left')
        axD.set_xlabel('$D$ ($10^{-4}$ Pa$^{-1}$)', fontsize=8, labelpad=1)
        axD.set_ylabel(r'$\Delta\mathcal{L}(D)$, profiled', fontsize=8, labelpad=1); axD.tick_params(labelsize=8)
        axD.set_ylim(-0.03 * dS.max(), 1.05 * dS.max())
        print(f'[fig9] D profile: min at {Dg[np.argmin(Sg)]:.3g} (fit {d_main:.3g}); Δ at grid ends {dS[0]:.0f} / {dS[-1]:.0f}')
    axD.set_title('Distensibility fit', fontsize=9, pad=3); axD.spines[['top', 'right']].set_visible(False)

    # ---- e  waveforms, Q/Q̄ over one beat -------------------------------------------------------------------------------
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
        ax.text(0.97, 0.03, f'mean {dm:+.0f} %'.replace('-', '−'), transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#d62728')
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
    axs_e[1].legend(*axs_e[0].get_legend_handles_labels(), fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
                    frameon=False, borderaxespad=0.1, handlelength=1.8, columnspacing=1.2)

    # panel letters
    pf_, pa, pc, pd, pe = axF.get_position(), axes_A[0, 0].get_position(), axC.get_position(), axD.get_position(), axs_e[0].get_position()
    yt = max(pf_.y1, pa.y1) + 0.022; yb = max(pc.y1, pd.y1) + 0.03
    for x_, y_, l_ in ((pf_.x0 - 0.05, yt, 'a'), (pa.x0 - 0.04, yt, 'b'), (pc.x0 - 0.065, yb, 'c'), (pd.x0 - 0.06, yb, 'd'),
                       (pe.x0 - 0.045, yb, 'e')):
        fig.text(x_, y_, l_, fontsize=12, fontweight='bold', va='bottom', ha='left')

    # SI: harmonic residual map + dimensional pulsatile RMS (shares the DC scale of the reconstruction)
    figA = plt.figure(figsize=(7.5, 3.3 * aspect + 0.4), facecolor='white')
    gAg = figA.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35, left=0.03, right=0.97, top=0.9, bottom=0.14)
    axB = figA.add_subplot(gAg[0]); bg(axB); edgemap(axB, pires, e, nPI, cPI, 0.45); pick_marks(axB)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nPI, cmap=cPI), cax=axB.inset_axes([1.03, 0.10, 0.04, 0.80]))
    cb.set_label('Harmonic residual', fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.4)
    axB.set_title('Harmonic residual', fontsize=9, pad=3)
    axR = figA.add_subplot(gAg[1]); sel = e & (rms_s > 0) & (rms_m > 0)
    axR.scatter(rms_m[sel], rms_s[sel], c=pires[sel], s=1.5, cmap=cPI, norm=nPI, alpha=0.6, linewidths=0, rasterized=True)
    lo = max(min(np.percentile(rms_m[sel], 2), np.percentile(rms_s[sel], 2)) * 0.5, 1e-4)
    hi = max(np.percentile(rms_m[sel], 99.5), np.percentile(rms_s[sel], 99.5)) * 2
    axR.plot([lo, hi], [lo, hi], 'k--', lw=0.5, alpha=0.6)
    axR.set_xscale('log'); axR.set_yscale('log'); axR.set_xlim(lo, hi); axR.set_ylim(lo, hi); axR.set_aspect('equal')
    axR.set_xlabel('Measured pulsatile RMS (nL/s)', fontsize=8, labelpad=1); axR.set_ylabel('RC network (nL/s)', fontsize=8, labelpad=1)
    axR.tick_params(labelsize=8); axR.spines[['top', 'right']].set_visible(False)
    axR.text(0.0, 1.03, f"$r_{{\\log}}$ = {stats['rms_corr']:.2f}  (both scale with the reconstructed mean flow)", transform=axR.transAxes,
             va='bottom', fontsize=8)
    for x_, l_ in ((0.01, 'a'), (0.55, 'b')):
        figA.text(x_, 0.93, l_, fontsize=12, fontweight='bold', va='bottom')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plos_style.save(figA, OUT_DIR / ('fig9_S_agreement' + (f'_{args.tag}' if args.tag else '')))

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

    # text-collision check: every pair of visible text boxes (labels, titles, ticks, legends, annotations) must not overlap
    fig.canvas.draw(); _r = fig.canvas.get_renderer(); _tb = []
    _ticks_off = set()
    for ax_ in fig.axes:                       # tick labels outside the view limits are kept by matplotlib but never drawn
        for axis_, (lo_, hi_) in ((ax_.xaxis, sorted(ax_.get_xlim())), (ax_.yaxis, sorted(ax_.get_ylim()))):
            for tk_ in axis_.get_major_ticks() + axis_.get_minor_ticks():
                if not (lo_ - 1e-9 * abs(hi_ - lo_) <= tk_.get_loc() <= hi_ + 1e-9 * abs(hi_ - lo_)):
                    _ticks_off.update({id(tk_.label1), id(tk_.label2)})
    for t_ in fig.findobj(matplotlib.text.Text):
        if t_.get_visible() and t_.get_text().strip() and id(t_) not in _ticks_off:
            try:
                _tb.append((t_.get_text().strip(), t_.get_window_extent(_r)))
            except Exception:
                pass
    _hits = [(a_[0], b_[0]) for i_, a_ in enumerate(_tb) for b_ in _tb[i_ + 1:] if a_[1].overlaps(b_[1])
             and a_[1].width > 0 and b_[1].width > 0]
    print(f'[fig9] text overlaps: {len(_hits)}' + ('' if not _hits else ' — ' + '; '.join(f'{a_!r}/{b_!r}' for a_, b_ in _hits[:12])))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = 'fig9' + (f'_{args.tag}' if args.tag else '')
    plos_style.save(fig, OUT_DIR / stem)
    print(f"[fig9] wrote {OUT_DIR / f'{stem}.png'}")


if __name__ == '__main__':
    main()
