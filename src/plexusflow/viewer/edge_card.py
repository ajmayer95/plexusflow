"""Edge flux card: for one vessel, the three levels of the DC reconstruction side by side.

  left   every tile recording (± σ = c|Q|, marker area ∝ its weight in the vessel average) → the single INPUT Q̄ the
         reconciliation sees (± σ) → the reconciled Q′ (± ensemble SD), all signed along the reconciled flow direction
  right  the two junctions at the vessel's ends: every carved vessel meeting there, coloured by input → reconciled flux
         (arrows = reconciled direction), and each junction's net flow from the inputs (measured vessels only) vs Q′

Reads the in-memory reconciliation attributes the viewer attaches (`_rec_Qm`, `_rec_c`, `_rec_tile_inputs`,
`_h_amp_DC_rec`, `_rec_Qp_sd`); `attach_reconciliation(G, D)` does the same for scripts.
"""
from __future__ import annotations

import numpy as np


def attach_reconciliation(G, D, gain=None):
    """Attach the reconciliation attributes the viewer uses (scripts; the viewer does this in _load_reconciliation)."""
    from ..reconcile import measurement_table
    from ..reconcile.reliability import c_of
    edges = [tuple(e) for e in D['edges']]
    Qm, cc, imd = np.asarray(D['Qm'], float), np.asarray(D['c'], float), np.asarray(D['imaged'], bool)
    sd = np.asarray(D.get('Qp_sd', np.full(len(edges), np.nan)), float)
    for i, e in enumerate(edges):
        d = G.edges[e]; d['_h_amp_DC_rec'] = float(D['Qp'][i]); d['_rec_Qp_sd'] = float(sd[i]); d['_rec_uv'] = e
        if imd[i]:
            d['_rec_Qm'] = float(Qm[i]); d['_rec_c'] = float(cc[i])
    eset = set(edges)
    for (t, e), m in measurement_table(G, gain=D.get('gain', gain)).items():
        if e in eset:
            G.edges[e].setdefault('_rec_tile_inputs', {})[int(t)] = (float(m['Q']), float(c_of(D['model'], m)))
    nodes = D['nodes']
    G.graph['_rec_boundary'] = {nodes[i] for i in np.r_[np.asarray(D['inj'], int), np.asarray(D['out'], int)]}


def _signed_into(G, e, n, q):
    """Flux q (along stored orientation e[0]→e[1]) expressed as flow INTO node n."""
    return q if e[1] == n else -q


def edge_flux_card(G, edge, mosaic_height=None, fig=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    u, v = edge
    if not G.has_edge(u, v):
        raise KeyError(edge)
    d = G.edges[u, v]
    e = tuple(d.get('_rec_uv') or (u, v))                         # reconciliation's orientation (signs refer to it)
    qp = d.get('_h_amp_DC_rec')
    if qp is None:
        raise ValueError(f"edge {edge} is not in the carved reconstruction network (open arterial region or excluded)")
    s = 1.0 if qp >= 0 else -1.0                                    # display sign: reconciled flow direction = +
    a_, b_ = (e[0], e[1]) if s > 0 else (e[1], e[0])
    fig = fig or plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([0.2, 0.13, 0.3, 0.72]); am = fig.add_axes([0.55, 0.04, 0.43, 0.86])

    # ---- left: recordings → input → reconciled
    tin = d.get('_rec_tile_inputs') or {}
    meas = {int(m['tile_id']): m for m in (d.get('measurements_piv') or [])}
    w = {t: 1 / max(c, 1e-12) ** 2 for t, (q, c) in tin.items()}; ws = sum(w.values()) or 1.0
    rows = []
    for t in sorted(meas):
        q_ = (meas[t].get('quality') or {})
        if t in tin:
            q, c = tin[t]
            rows.append((f"tile {t}  (align {q_.get('align', np.nan):.2f}, c {c:.2f})", s * q, abs(c * q), w[t] / ws, '#7f7f7f'))
        else:
            rows.append((f"tile {t}  (no flux / not used)", np.nan, np.nan, 0, '#bbbbbb'))
    qm, cm = d.get('_rec_Qm'), d.get('_rec_c')
    y = len(rows) + 1.0
    for k, (lab, x, err, wt, col) in enumerate(rows):
        yy = len(rows) - k + 1.0
        if np.isfinite(x):
            ax.errorbar(x, yy, xerr=err, fmt='o', color=col, ms=4 + 14 * np.sqrt(wt), capsize=3, lw=1)
            ax.text(x, yy + 0.22, f"{x:+.3f}  ({wt:.0%})", ha='center', fontsize=7.5, color='0.3')
        else:
            ax.text(0, yy, "no flux", ha='center', va='center', fontsize=7.5, color='0.6')
    ylab = [r[0] for r in rows]
    if qm is not None:
        ax.errorbar(s * qm, 1.0, xerr=abs(cm * qm), fmt='D', color='#1f77b4', ms=8, capsize=4, lw=1.4)
        ax.text(s * qm, 1.22, f"{s * qm:+.3f}", ha='center', fontsize=8, color='#1f77b4', fontweight='bold')
    else:
        ax.text(0, 1.0, "unmeasured — no input", ha='center', va='center', fontsize=8, color='#1f77b4')
    sd = d.get('_rec_Qp_sd')
    ax.errorbar(abs(qp), 0.0, xerr=sd if sd is not None and np.isfinite(sd) else None, fmt='*', color='#d62728', ms=14, capsize=4)
    ax.text(abs(qp), 0.25, f"{abs(qp):+.3f}", ha='center', fontsize=8, color='#d62728', fontweight='bold')
    ax.set_yticks([0, 1] + [len(rows) - k + 1.0 for k in range(len(rows))])
    ax.set_yticklabels(["reconciled Q′", "INPUT Q̄ (fit target)"] + ylab, fontsize=8)
    ax.axvline(0, color='k', lw=0.6); ax.set_ylim(-0.7, len(rows) + 1.7)
    ax.set_xlabel(f"flux along reconciled flow {a_}→{b_}  (nL/s; negative = opposite)")
    ttl = f"edge {e}   r {d.get('radius', np.nan):.1f} px"
    if qm:
        ttl += f"\nQ′/Q̄ = {qp / qm:+.2f}   ({(qp - qm) / abs(cm * qm):+.1f}σ)" + ("   DIRECTION REVERSED" if np.sign(qp) != np.sign(qm) else "")
    ax.set_title(ttl, fontsize=9, loc='left'); ax.spines[['top', 'right']].set_visible(False)

    # ---- right: the two end junctions
    H = mosaic_height or 0
    yy_ = (lambda yv: H - yv) if H else (lambda yv: -yv)
    def path(ed):
        p = G.edges[ed].get('path')
        P = np.asarray(p, float) if p is not None else np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in ed])
        return np.column_stack([P[:, 0], yy_(P[:, 1])])
    bnd = G.graph.get('_rec_boundary', set())
    inc = []
    for n in (e[0], e[1]):
        for nb in G.neighbors(n):
            dd_ = G.edges[n, nb]
            if '_h_amp_DC_rec' not in dd_:
                continue
            ed = tuple(dd_.get('_rec_uv') or (n, nb))
            if ed not in [x[0] for x in inc]:
                inc.append((ed, n))
    allp = np.vstack([path(ed) for ed, _ in inc])
    lo, hi = allp.min(0), allp.max(0); pad = 0.25 * max(hi - lo) + 10
    am.add_collection(LineCollection([path(ed) for ed, _ in inc], colors=['#d62728' if ed == e else '0.55' for ed, _ in inc],
                                     lw=[3 if ed == e else 1.6 for ed, _ in inc]))
    for ed, n in inc:
        dd = G.edges[ed]; P = path(ed); mid = P[len(P) // 2]; qpp = dd['_h_amp_DC_rec']; qmm = dd.get('_rec_Qm')
        a0, b0 = (ed[0], ed[1]) if qpp >= 0 else (ed[1], ed[0])                # reconciled direction
        pa = np.array([G.nodes[a0]['x'], yy_(G.nodes[a0]['y'])]); pb = np.array([G.nodes[b0]['x'], yy_(G.nodes[b0]['y'])])
        dv = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
        am.annotate('', xy=mid + 6 * dv, xytext=mid - 6 * dv, arrowprops=dict(arrowstyle='-|>', color='k', lw=1))
        same = qmm is None or np.sign(qmm) == np.sign(qpp)
        lab = (f"{abs(qmm):.3f}" + ("" if same else " (rev)") if qmm is not None else "unmeas.") + f" → {abs(qpp):.3f}"
        am.text(*(mid + np.array([4, 4])), lab, fontsize=7.5, color='#d62728' if ed == e else 'k',
                bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none', alpha=0.8))
    for n in (e[0], e[1]):
        xin = sum(_signed_into(G, ed, n, G.edges[ed].get('_rec_Qm') or 0.0) for ed, m in inc if n in ed)
        xp = sum(_signed_into(G, ed, n, G.edges[ed]['_h_amp_DC_rec']) for ed, m in inc if n in ed)
        nm = sum(1 for ed, m in inc if n in ed and G.edges[ed].get('_rec_Qm') is None)
        pn = np.array([G.nodes[n]['x'], yy_(G.nodes[n]['y'])])
        am.plot(*pn, 'o', ms=7, mfc='gold' if n in bnd else 'w', mec='k', zorder=5)
        am.text(*(pn + np.array([0, -10 if n == e[0] else 22])), f"node {n}{' (boundary: exchange)' if n in bnd else ''}\nnet in: inputs {xin:+.3f}"
                + (f" ({nm} unmeas.)" if nm else "") + f"  →  Q′ {xp:+.3f}", fontsize=7.5, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='#fff8dc', ec='0.6'))
    am.set_xlim(lo[0] - pad, hi[0] + pad); am.set_ylim(lo[1] - pad, hi[1] + pad); am.set_aspect('equal'); am.set_xticks([]); am.set_yticks([])
    am.set_title("end junctions: |input Q̄| → |reconciled Q′| (nL/s), arrows = reconciled direction\n"
                 "net in = flow into the node (interior nodes: Q′ sums to 0)", fontsize=8.5, loc='left')
    return fig
