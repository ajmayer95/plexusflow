"""Figure 7 — whole-organ DC: reconstructed reference field Q′ vs the passive Poiseuille prediction Q_sim.

Inputs: results/dc_reconciled_v2.pkl (scripts/dc_reconcile.py: reliability-weighted IRLS projection of the measured mean
flows onto mass conservation on the carved network, emergent boundary fluxes f = BQ′) and the canonical graph.
Passive model: conductance K = r⁴/ℓ (consensus radii), the emergent boundary fluxes imposed at every boundary vertex, and a
soft equal-pressure penalty at the two vitelline-vein outlets (plexusflow.reconcile.venous_pressure_penalty, σ_P = 2 Pa;
FIG7_SIGMA_P=inf renders the unpenalised variant).

  a  Q′ on direction-resolved vessels (|Q′| > 2σ_e), measured-uncertain and unmeasured vessels in two greys, boundary-flux
     key, direction arrows, posterior rim flow divide (stagnation point)
  b  Q_sim on the same vessels, arterial / venous boundary markers, the model's own divide
  c  local agreement log10(|Q_sim|/|Q′|) on resolved vessels (clipped ±1), direction mismatches marked
  d  measured vs predicted |Q| (log hexbin): r_log on all measured and on resolved vessels, direction agreement
  e  radius controls: measured (consensus) vs uniform vs shuffled radii (r_log, all measured vessels)
SI: fig7_S_boundary_distance, fig7_S_threshold (k = 0–3σ), fig7_S_forcing (arterial-inflow redistributions).
Writes figures/fig7/; nothing to any graph.
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
import matplotlib.patheffects as pe
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
    # passive model with a soft equal-pressure penalty at the two vitelline-vein outlets (plexusflow.reconcile.venous_pressure_penalty)
    from plexusflow.reconcile import venous_pressure_penalty
    v1, v2 = dom['out']; Bm = dom['B']
    fb, vp = venous_pressure_penalty(fb, dom, r, D['length'], D['sigma'], sigma_P=SIGMA_P)
    shift, dP0, dP1 = vp['shift'], vp['dP_before'], vp['dP_after']
    Qs = passive(K, fb, dom)
    print(f"[fig7] soft venous penalty σ_P {SIGMA_P} Pa: outflow V1 {fb[v1]:.2f}, V2 {fb[v2]:.2f} nL/s (shift {shift:+.2f}); "
          f"P_V2 − P_V1 {dP1:+.2f} Pa (was {dP0:+.2f})")
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
        ax.set_xlim(0, Wm); ax.set_ylim(Hm, -0.06 * Hm)          # headroom above the organ for the boundary-flux key

    ART_S = lambda f: 4 + 60 * abs(f)                                       # marker area per nL/s of boundary inflow
    def markers(ax, sized=False):
        for i_ in dom['inj']:                               # Fig 5 convention: arterial cuts red, venous outlets blue
            if sized:                                       # area ∝ inflow of the reference field; hollow = net withdrawal
                f_ = D['fb'][i_]
                ax.scatter(*nxy[nodes[i_]], s=ART_S(f_), facecolor='#d40000' if f_ > 0 else 'white', edgecolor='#d40000' if f_ <= 0 else 'white',
                           linewidths=0.4 if f_ > 0 else 0.8, zorder=6, alpha=0.9)
            else:
                ax.plot(*nxy[nodes[i_]], 'o', color='#d40000', ms=2.2, mec='white', mew=0.3, zorder=6)
        for n in G.graph['boundary']['V']:
            ax.plot(*nxy[n], 's', color='#2b35a8', ms=4, mec='white', mew=0.5, zorder=6)

    resolved = np.abs(Qp) > 2 * D['sigma']                     # direction determined: |Q'| > 2σ_e (Fig 6 noise model)
    print(f"[fig7] direction resolved for {resolved.sum()} of {len(Qp)} vessels (|Q'| > 2σ)")
    print(f"[fig7] resolved vessels carry {100 * np.abs(Qp[resolved]).sum() / np.abs(Qp).sum():.0f}% of Σ|Q'|; length-weighted {100 * (np.abs(Qp) * D['length'])[resolved].sum() / (np.abs(Qp) * D['length']).sum():.0f}%; passive Σ|Q_sim| on them {100 * np.abs(Qs[resolved]).sum() / np.abs(Qs).sum():.0f}%")
    # posterior divide of the peripheral (rim) vein: rim = largest connected set of vessels within 70 px of the organ outline;
    # each rim vessel's flow is expressed as angular motion about the organ centroid; the divide is the angle θc on the
    # posterior arc maximising min(flux leaving θc on each side) over ±30° windows, i.e. flow away from θc on BOTH sides
    from scipy import ndimage as ndi
    import networkx as nx
    org_ = tifffile.imread(str(STAGE / 'fuse' / 'organ_fusion.tif')); org_ = org_.mean(-1) if org_.ndim == 3 else org_
    organ = ndi.binary_fill_holes(ndi.binary_closing(org_ > 0, iterations=10)); dist_ = ndi.distance_transform_edt(organ)
    cyo, cxo = ndi.center_of_mass(organ)
    mids = np.array([sg_[len(sg_) // 2] for sg_ in segs])
    near = np.array([dist_[int(np.clip(y_, 0, Hm - 1)), int(np.clip(x_, 0, Wm - 1))] < 70 for x_, y_ in mids])
    Hr = nx.Graph(); [Hr.add_edge(*edges[k], k=k) for k in np.flatnonzero(near)]
    comp_ = max(nx.connected_components(Hr), key=len)
    rim = near & np.array([edges[k][0] in comp_ and edges[k][1] in comp_ for k in range(len(edges))])
    rim &= r >= np.percentile(r[rim], 50)                                   # the rim vein proper: larger half of rim vessels
    th_ = np.degrees(np.arctan2(mids[:, 1] - cyo, mids[:, 0] - cxo))         # 90° = posterior midline (image y down)
    len_ok = np.array([len(sg_) >= 8 for sg_ in segs])
    def rim_divide(Q):
        dth = np.zeros(len(Q))
        for k in np.flatnonzero(rim):
            sg_ = segs[k]; u_ = np.array(nxy[edges[k][0]])
            a_, b_ = (sg_[0], sg_[-1]) if np.linalg.norm(sg_[0] - u_) < np.linalg.norm(sg_[-1] - u_) else (sg_[-1], sg_[0])
            if Q[k] < 0: a_, b_ = b_, a_
            ta, tb = (np.degrees(np.arctan2(v_[1] - cyo, v_[0] - cxo)) for v_ in (a_, b_))
            dth[k] = np.sign((tb - ta + 180) % 360 - 180)                      # +1: flow toward larger θ
        w = np.abs(Q) * dth * rim
        grid = np.arange(60, 176); score = [min(w[(th_ > c) & (th_ < c + 30)].sum(), -w[(th_ < c) & (th_ > c - 30)].sum()) for c in grid]   # both sides must flow away
        c = float(grid[int(np.argmax(score))])
        kl = np.flatnonzero(rim & (th_ <= c) & (dth < 0)); kr = np.flatnonzero(rim & (th_ > c) & (dth > 0))
        kl, kr = kl[np.argmax(th_[kl])], kr[np.argmin(th_[kr])]                 # the two rim vessels flowing apart
        return 0.5 * (th_[kl] + th_[kr]), 0.5 * (mids[kl] + mids[kr]), dth
    from matplotlib.patches import FancyArrowPatch
    def arrow(ax, c, Q):
        """Fixed-length direction arrow at the middle of vessel c, pointing along its flow Q[c] (Q > 0: u → v)."""
        sg = segs[c]; u_ = np.array(nxy[edges[c][0]]); sg = sg if np.linalg.norm(sg[0] - u_) < np.linalg.norm(sg[-1] - u_) else sg[::-1]
        sg = sg if Q[c] > 0 else sg[::-1]
        k0 = len(sg) // 2; d_ = sg[min(k0 + 4, len(sg) - 1)] - sg[max(k0 - 4, 0)]; d_ = d_ / np.linalg.norm(d_)
        ax.add_patch(FancyArrowPatch(sg[k0] - 75 * d_, sg[k0] + 75 * d_, arrowstyle='-|>', mutation_scale=7, color=INKC, lw=1.0, zorder=7,
                                     shrinkA=0, shrinkB=0, path_effects=[pe.withStroke(linewidth=1.6, foreground='white')]))
    def mark_divide(ax, Q, name, ok, min_gap=0):
        c, xy_, dth = rim_divide(Q)
        for side, tgt in ((-1, c - 12), (1, c + 12)):                      # one flank arrow each side: flow leaving the divide
            cand = np.flatnonzero(rim & ok & (dth == side) & ((th_ - c) * side > 4) & (len_ok))
            if len(cand):
                k_ = cand[np.argmin(np.abs(th_[cand] - tgt))]
                if np.linalg.norm(mids[k_] - xy_) >= min_gap:              # arrows crowding the marker are dropped
                    arrow(ax, k_, Q)
        ax.plot(*xy_, 'D', ms=5.5, mfc='white', mec=INKC, mew=1.0, zorder=9)
        print(f"[fig7] posterior rim divide ({name}): θ = {c:.0f}° at (x {xy_[0]:.0f}, y {xy_[1]:.0f})")
        return xy_
    GREY_U, GREY_N = (0.45, 0.45, 0.45, 0.55), (0.78, 0.78, 0.78, 0.6)   # measured, direction undetermined | unmeasured (imputed)
    def greys(ax, show):
        """Grey under-layer: measured vessels below 2σ (mid grey) and unmeasured/imputed vessels (light grey)."""
        for sel, col in ((show & imaged, GREY_U), (show & ~imaged, GREY_N)):
            ax.add_collection(LineCollection([segs[c] for c in np.flatnonzero(sel)], colors=[col], linewidths=0.35, zorder=2))
    def field(ax, Q, mask_unresolved):
        und = ~resolved if mask_unresolved else np.zeros(len(Q), bool)
        greys(ax, und)
        keep = np.flatnonzero(~und)
        ax.add_collection(LineCollection([segs[c] for c in keep], colors=[cAB(nAB(min(max(abs(Q[c]), vmin), vmax))) for c in keep],
                                         linewidths=0.85, zorder=3))
    def totals(ax):
        """Net boundary fluxes of the reference field (A→V transport is reported only through these totals)."""
        f0 = D['fb']; inj_, sd = f0[dom['inj']], np.asarray(dom['side'])   # Q′'s own boundary fluxes (before the venous penalty)
        ax.text(-0.03, 0.995, 'Boundary flux (nL/s)', transform=ax.transAxes, ha='left', va='top', fontsize=8)
        ax.text(-0.03, 0.96, f"Arterial  L {inj_[sd == 'L'].sum():.1f} · R {inj_[sd == 'R'].sum():.1f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=8, color='#a00000')
        ax.text(-0.03, 0.925, f"Venous   V$_1$ {-f0[dom['out'][0]]:.1f} · V$_2$ {-f0[dom['out'][1]]:.1f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=8, color='#2b35a8')

    fig = plt.figure(figsize=(7.5, 6.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.8], hspace=0.16, left=0.06, right=0.9, top=0.935, bottom=0.08)
    gT = gs[0].subgridspec(1, 3, width_ratios=[1.0, 0.34, 1.0], wspace=0.04); gBm = gs[1].subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.8], wspace=0.45)
    LETTERS = {}
    def letter(ax, l, title, dx=-0.02):                                      # letter + title placed on a shared row line after layout (below)
        LETTERS[l] = (ax, title)

    # a
    axA = fig.add_subplot(gT[0]); bg(axA); field(axA, Qp, True); markers(axA)
    letter(axA, 'a', "Reference field Q′")
    totals(axA); div_a = mark_divide(axA, Qp, "Q′", resolved, min_gap=400)
    axA.plot([], [], color=GREY_U[:3], lw=1.2, label='measured, |Q′| < 2σ')
    axA.plot([], [], color=GREY_N[:3], lw=1.2, label='unmeasured')
    axA.plot([], [], 'D', ms=5, mfc='white', mec=INKC, mew=1.0, label='stagnation point')
    # middle column: the arterial cuts zoomed, each cut coloured by its inflow in Q′ (red log scale; white = net withdrawal)
    cxy_ = np.array([nxy[nodes[i_]] for i_ in dom['inj']]); pad_ = 120
    x0i, x1i = cxy_[:, 0].min() - pad_, cxy_[:, 0].max() + pad_; y0i, y1i = cxy_[:, 1].min() - pad_, cxy_[:, 1].max() + pad_
    axIn = fig.add_subplot(gT[1]); bg(axIn); field(axIn, Qp, True)
    axIn.set_xlim(x0i, x1i); axIn.set_ylim(y1i, y0i); axIn.set_aspect('equal')
    for s_ in axIn.spines.values():
        s_.set_visible(True); s_.set_color('0.3'); s_.set_linewidth(0.8)
    fcut_ = D['fb'][dom['inj']]; pos_ = fcut_ > 0
    nCut = mcolors.LogNorm(vmin=0.01, vmax=max(fcut_.max(), 1.0))
    cCut = mcolors.LinearSegmentedColormap.from_list('rt', plt.get_cmap('Reds')(np.linspace(0.25, 1.0, 256)))
    for i_, f_ in zip(dom['inj'], fcut_):
        if f_ > 0:
            axIn.scatter(*nxy[nodes[i_]], s=30, color=cCut(nCut(max(f_, 0.01))), edgecolor='black', linewidths=0.5, zorder=7)
        else:
            axIn.scatter(*nxy[nodes[i_]], s=24, facecolor='white', edgecolor='black', linewidths=0.7, zorder=7)
    axA.add_patch(plt.Rectangle((x0i, y0i), x1i - x0i, y1i - y0i, fill=False, ec='0.3', lw=0.8, zorder=7))
    from matplotlib.patches import ConnectionPatch
    for ya_ in (y0i, y1i):                                                   # box in a → zoomed panel
        fig.add_artist(ConnectionPatch(xyA=(x1i, ya_), coordsA=axA.transData, xyB=(x0i, ya_), coordsB=axIn.transData, color='0.5', lw=0.6, ls=(0, (2, 2))))
    axIn.set_title('Arterial cuts', fontsize=8, pad=2)
    cbCut = axIn.inset_axes([0.05, -0.1, 0.9, 0.035])
    cb_ = plt.colorbar(plt.cm.ScalarMappable(norm=nCut, cmap=cCut), cax=cbCut, orientation='horizontal', ticks=[0.01, 0.1, 1.0])
    cb_.ax.set_xticklabels(['0.01', '0.1', '1']); cb_.ax.minorticks_off(); cb_.ax.tick_params(labelsize=8, length=2, pad=1); cb_.outline.set_linewidth(0.4)
    cb_.set_label('inflow (nL/s)\n○ withdrawal', fontsize=8, labelpad=2, linespacing=1.1)
    _hs = []
    axA.legend(handles=[h_ for h_ in axA.get_legend_handles_labels()[0] if h_ not in _hs], loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=8, frameon=False, borderaxespad=0.2, scatterpoints=1,
               handlelength=1.4, columnspacing=1.2)
    # a handful of direction arrows on the strongest direction-resolved vessels, spread over the organ
    picked = []; bxy = np.array([nxy[nodes[i_]] for i_ in np.r_[dom['inj'], dom['out']]])
    far = lambda c, gap: all(np.linalg.norm(mids[c] - mids[q]) >= gap for q in picked) and np.linalg.norm(mids[c] - div_a) >= 400
    for c in np.argsort(-np.abs(Qp) * (resolved & imaged)):
        if len(picked) >= 14 or not resolved[c]:
            break
        if len_ok[c] and abs(Qp[c]) >= 4 * D['sigma'][c] and np.min(np.linalg.norm(bxy - mids[c], axis=1)) >= 250 and far(c, 420):
            picked.append(c); arrow(axA, c, Qp)
    # plus a few in regions the strongest-vessel pass leaves bare: the posterior plexus (below the arterial cuts) and the
    # upper-left lobe (left of the cuts, above them), and the top-left lobe tip (left of and above V1, where the rim vein runs round toward the heart; rim allowed); otherwise interior vessels only, |Q'| ≥ 3σ
    vxy = np.array([nxy[n] for n in G.graph['boundary']['V']])
    cxy = np.array([nxy[nodes[i_]] for i_ in dom['inj']]); y_cut, y_top, x_left = cxy[:, 1].max(), cxy[:, 1].min(), cxy[:, 0].min()
    regions = (('posterior plexus', lambda m_: m_[1] > y_cut + 250, 5, False, 450),
               ('upper-left lobe', lambda m_: m_[0] < x_left - 250 and m_[1] < y_top, 3, False, 450),
               ('top-left lobe tip', lambda m_: m_[0] < vxy[:, 0].min() - 300 and m_[1] < vxy[:, 1].min() - 350, 2, True, 300))   # rim allowed
    n_pl = 0
    for nm_, inside, n_max, rim_ok, gap in regions:
        n_r = 0
        for c in np.argsort(-np.abs(Qp) * resolved):
            if n_r >= n_max or not resolved[c]:
                break
            if len_ok[c] and (rim_ok or not near[c]) and inside(mids[c]) and abs(Qp[c]) >= 3 * D['sigma'][c] and far(c, gap):
                picked.append(c); arrow(axA, c, Qp); n_r += 1
        n_pl += n_r; print(f"[fig7] {nm_}: {n_r} extra arrows")
    print(f"[fig7] direction arrows: {len(picked) - n_pl} strongest + {n_pl} regional (+ divide flanks)")
    L_ = 1000 / 1.7; x0_, y0_ = 0.03 * Wm, 0.975 * Hm                  # 1 mm scale bar (mosaic 1.7 µm/px); shared by a–c
    axA.plot([x0_, x0_ + L_], [y0_, y0_], color=INKC, lw=2, solid_capstyle='butt', zorder=8)
    axA.text(x0_ + L_ / 2, y0_ - 0.012 * Hm, '1 mm', ha='center', va='bottom', fontsize=8, zorder=8)

    # b
    axB = fig.add_subplot(gT[2]); bg(axB); field(axB, Qs, False); markers(axB); mark_divide(axB, Qs, 'Q_sim', np.ones(len(Qs), bool))
    axB.add_patch(plt.Rectangle((x0i, y0i), x1i - x0i, y1i - y0i, fill=False, ec='0.3', lw=0.8, zorder=7))
    for ya_ in (y0i, y1i):                                                   # zoomed panel → same box in the passive prediction
        fig.add_artist(ConnectionPatch(xyA=(x1i, ya_), coordsA=axIn.transData, xyB=(x0i, ya_), coordsB=axB.transData, color='0.5', lw=0.6, ls=(0, (2, 2))))
    for c in picked:                                         # same vessels as a, in the passive model's own direction
        if abs(Qs[c]) >= 0.01:                               # no arrow where the model's flow is negligible (direction meaningless)
            arrow(axB, c, Qs)
    flip = [c for c in picked if np.sign(Qp[c]) != np.sign(Qs[c])]
    print(f"[fig7] arrows flipped between a and b: {len(flip)} of {len(picked)}: " +
          ", ".join(f"{edges[c]} at ({mids[c][0]:.0f}, {mids[c][1]:.0f}) Q′ {Qp[c]:+.2f} ({abs(Qp[c]) / D['sigma'][c]:.1f}σ) vs Q_sim {Qs[c]:+.2f}" for c in flip))     # same measurement-support mask as a
    inj = fb[dom['inj']]                                     # arterial cuts drawn red by markers(); net withdrawals not marked (caption)
    print(f"[fig7] arterial cuts: {int((inj < 0).sum())} net withdrawals, Σ {inj[inj < 0].sum():.2f} nL/s ({100 * -inj[inj < 0].sum() / inj[inj > 0].sum():.0f}% of inflow)")
    axB.plot([], [], 'o', ms=3.2, color='#d40000', mec='white', mew=0.3, label='arterial boundary')
    axB.plot([], [], 's', ms=4, color='#2b35a8', mec='white', mew=0.5, label='venous outlet')
    axB.plot([], [], 'D', ms=5, mfc='white', mec=INKC, mew=1.0, label='_nolegend_')
    axB.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False, borderaxespad=0.2,
               handletextpad=0.2, columnspacing=1.0)
    letter(axB, 'b', r'Passive prediction Q$_{\rm sim}$')
    pB = axB.get_position(); cax = fig.add_axes([pB.x1 + 0.012, pB.y0 + 0.1 * pB.height, 0.013, 0.8 * pB.height])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=nAB, cmap=cAB), cax=cax)
    cb.set_label(r'$|\bar Q|$ (nL/s)', fontsize=9, labelpad=2); cb.outline.set_linewidth(0.5)

    # c
    axC = fig.add_subplot(gBm[0]); bg(axC)
    m = imaged & (np.abs(Qp) > 1e-12) & (np.abs(Qs) > 1e-12)
    E = np.log10(np.abs(Qs) / np.maximum(np.abs(Qp), 1e-12))
    mr = m & (np.abs(Qp) > 2 * D['sigma'])
    greys(axC, ~mr)
    axC.add_collection(LineCollection([segs[c] for c in np.flatnonzero(mr)],
                                      colors=[cC(nC(E[c])) for c in np.flatnonzero(mr)], linewidths=0.8, zorder=3))
    q_floor = max(vmin * 3, 3e-3)
    # noise level = each vessel's calibrated measurement uncertainty σ_e = sqrt((c_e y_e)² + σ0²) (Fig 6b); a direction is only
    # judged where the reconciled flow is clearly above it, |Q'| > 2σ_e
    sig_ok = np.abs(Qp) > 2 * D['sigma']
    mism = m & sig_ok & (np.sign(Qp) != np.sign(Qs))
    for c in np.flatnonzero(mism):
        s = segs[c]; mid = s[len(s) // 2]
        axC.plot(*mid, 'o', color=MIS, ms=1.5, markeredgewidth=0, zorder=6, alpha=0.7)
    markers(axC)
    letter(axC, 'c', 'Local agreement')
    caxC = axC.inset_axes([0.15, -0.06, 0.7, 0.035])
    cbC = plt.colorbar(plt.cm.ScalarMappable(norm=nC, cmap=cC), cax=caxC, ticks=[-1, 0, 1], orientation='horizontal', extend='both')
    cbC.set_label(r'log$_{10}$(|Q$_{\rm sim}$| / |Q′|)', labelpad=1); cbC.outline.set_linewidth(0.5)
    axC.plot([], [], 'o', color=MIS, ms=3, label='direction mismatch')
    axC.legend(fontsize=8, loc='lower right', frameon=False, borderaxespad=0, handletextpad=0.1)

    # D
    axD = fig.add_subplot(gBm[1])
    md = m & sig_ok                                                          # direction-resolved vessels (|Q′| > 2σ), as in a
    lm, ls_ = np.log10(np.maximum(np.abs(Qp[md]), 1e-5)), np.log10(np.maximum(np.abs(Qs[md]), 1e-5))
    gcm = mcolors.LinearSegmentedColormap.from_list('g', [plt.cm.Greys(0.3), plt.cm.Greys(0.95)])   # lightest bin still visible
    axD.hexbin(lm, ls_, gridsize=45, cmap=gcm, mincnt=1, bins='log', linewidths=0, zorder=2)
    lo, hi = min(lm.min(), ls_.min()), max(lm.max(), ls_.max())
    axD.plot([lo, min(hi, 0.0)], [lo, min(hi, 0.0)], color=INKC, ls='--', lw=0.9, zorder=3)
    mm = mism[md]
    axD.scatter(lm[mm], ls_[mm], s=5, facecolors='none', edgecolors=MIS, linewidths=0.5, alpha=0.8,
                zorder=4, label='direction mismatch')
    r_log = float(np.corrcoef(np.log10(np.maximum(np.abs(Qp[m]), 1e-5)), np.log10(np.maximum(np.abs(Qs[m]), 1e-5)))[0, 1])
    consider = m & sig_ok
    r_log_res = float(np.corrcoef(np.log10(np.abs(Qp[consider])), np.log10(np.abs(Qs[consider])))[0, 1])
    p_agree = float(np.mean(np.sign(Qp[consider]) == np.sign(Qs[consider])))
    axD.text(0.03, 0.97, f'$r_{{\\log}}$ = {r_log_res:.2f}\ndirection {100 * p_agree:.0f}%',
             transform=axD.transAxes, va='top', ha='left', fontsize=8, linespacing=1.3, zorder=5,
             bbox=dict(fc='white', ec='none', pad=1.5))
    axD.set_xlabel(r"reconstructed log$_{10}$|Q′| (nL/s)", labelpad=1)
    axD.set_ylabel(r'predicted log$_{10}$|Q$_{\rm sim}$| (nL/s)', labelpad=1)
    letter(axD, 'd', 'Vessel-by-vessel (|Q′| > 2σ)', dx=-0.2)
    axD.set_aspect('equal')
    axD.spines[['top', 'right']].set_visible(False)
    axD.set_xlim(lo - 0.1, hi + 2.1); axD.set_ylim(lo - 0.1, hi + 2.1)       # headroom: statistics above the data

    # E  geometry controls
    ln = rec['length'][em]
    def r_of(Kx):
        q = passive(Kx, fb, dom); a, b = np.abs(Qp[m]), np.abs(q[m]); ok = (a > 1e-12) & (b > 1e-12)
        return float(np.corrcoef(np.log10(a[ok]), np.log10(b[ok]))[0, 1])
    rng = np.random.default_rng(0)
    r_uni = r_of(np.mean(r) ** 4 / ln)
    r_sh = np.array([r_of(rng.permutation(r) ** 4 / ln) for _ in range(N_SHUFFLE)])
    print(f'  r_log measured {r_log:.3f}, uniform {r_uni:.3f}, shuffled {r_sh.mean():.3f} ± {r_sh.std():.3f}')
    axE = fig.add_subplot(gBm[2])
    for yi, lab, rv, col, err in ((2, 'measured\nradii', r_log, '#2b70b4', 0), (1, 'uniform\nradii', r_uni, '#9e9e9e', 0),
                                  (0, 'shuffled\nradii', r_sh.mean(), '#e08214', r_sh.std())):
        axE.hlines(yi, 0, rv, color=col, lw=2, alpha=0.8)
        axE.errorbar(rv, yi, xerr=err if err else None, fmt='o', color=col, markersize=5, capsize=2, zorder=4)
        axE.text(rv + 0.03, yi + 0.28, f'{rv:.2f}', va='bottom', ha='center', fontsize=8, color=col)
    axE.set_yticks([2, 1, 0], ['measured radii', 'uniform radii', 'shuffled radii'])
    axE.set_xlim(0, 1.0); axE.set_ylim(-0.6, 2.9); axE.set_box_aspect(1.0)
    axE.set_xlabel(r"$r_{\log}$ (all measured)", labelpad=1)
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
    fig.canvas.draw(); rr_ = fig.canvas.get_renderer(); inv_ = fig.transFigure.inverted(); tb_ = lambda a_: inv_.transform(a_.get_tightbbox(rr_))
    # one baseline per row, just above the highest artist of that row (excluding text placed above the maps in a)
    yT_ = min(max(LETTERS[k][0].get_position().y1 for k in 'ab') + 0.012, 0.955)
    yB_ = max(tb_(LETTERS[k][0])[1, 1] for k in 'cde') + 0.01
    # Q′ and Q_sim are one comparison → one panel (a); bottom row b–d
    for k, shown in (('a', 'a'), ('b', None), ('c', 'b'), ('d', 'c'), ('e', 'd')):
        ax_, title_ = LETTERS[k]; y_ = yT_ if k in 'ab' else yB_
        xl_ = 0.005 if k in 'ac' else tb_(ax_)[0, 0] - 0.005
        if shown:
            fig.text(xl_, y_, shown, fontsize=12, fontweight='bold', va='baseline', ha='left')
        fig.text(xl_ + (0.028 if shown else 0.0), y_, title_, fontsize=9, va='baseline', ha='left')
    plos_style.save(fig, OUT_DIR / ('fig7' if np.isfinite(SIGMA_P) else 'fig7_no_vein_penalty')); plos_style.save(figF, OUT_DIR / 'fig7_S_boundary_distance')
    print(f"[fig7] r_log {r_log:.3f}, P(dir agree|filtered) {p_agree:.2f}, mismatches {int(mism.sum())}, "
          f"n {int(m.sum())}; wrote {OUT_DIR / 'fig7.png'}")


if __name__ == '__main__':
    main()
