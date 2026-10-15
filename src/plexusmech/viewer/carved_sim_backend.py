"""In-process backend for the viewer's Carved Sim tab.

Wraps the standalone scripts (``recompute_H1_phasor_direct.py``,
``fit_dc_porous_sources.py``, ``run_sim_carved_outlets.py``) into
callable functions so the viewer can drive the whole pipeline without
subprocess.

Designed to eventually generalise beyond "porous adjacent to A anchors" —
the outlet-detection is topology-based (any tagged region → any node
with an untagged edge leaving the tag becomes an outlet).  So masking
an interior region works with the same code path once we drop the
"exclude A/V anchors from outlet candidates" rule.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Set

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from scipy.optimize import nnls, minimize


N_HARM = 3
SNR_H1_DB_GATE = 10.0
SNR_H1_MIN_DB_FOR_PHI = 15.0
DEFAULT_D_AREAL = 3.5e-4
N_BINS_CF = 128
FRAME_DT = 1.0 / 250.0


# ── V2 field recompute (raw phasor + z_arg_H1 + CO_rel) ───────────
def recompute_v2_fields(G: nx.Graph, exclude_tiles: Optional[Set[int]] = None):
    """Recompute `_h_amp_H1_norm_v2` / `_h_phase_H1_norm_v2` /
    `_h_amp_DC_norm_v2` in-place on the graph.  Skips per-tile
    measurements whose tile_id is in `exclude_tiles`."""
    excl = exclude_tiles or set()
    tile_gauges = G.graph['tile_gauges']
    n_touched = 0
    for u, v, d in G.edges(data=True):
        H1_num = 0.0 + 0.0j; H1_den = 0.0
        DC_num = 0.0; DC_den = 0.0
        for m in (d.get('measurements_piv') or []):
            tid = m.get('tile_id')
            if tid is None: continue
            if int(tid) in excl: continue
            g = tile_gauges.get(int(tid)) or tile_gauges.get(str(tid))
            if g is None: continue
            harm = m.get('harm') or {}
            h1 = None
            for h in (harm.get('harmonics') or []):
                if h.get('k') == 1: h1 = h; break
            if h1 is None: continue
            amp = h1.get('amp'); phi = h1.get('phi'); snr = h1.get('snr_db')
            mQ = harm.get('mean_Q')
            if amp is None or phi is None: continue
            if snr is None or not np.isfinite(snr) or snr < SNR_H1_DB_GATE:
                continue
            CO_rel = float(g.get('CO_rel', 1.0))
            if CO_rel <= 0: continue
            z_arg_rad = np.radians(float(g.get('z_arg_deg', 0.0)))
            w = 10 ** (float(snr) / 10.0)
            sign = 1.0
            ff = m.get('flow_from'); ft = m.get('flow_to')
            if ff is not None and ft is not None:
                if int(ff) == int(v) and int(ft) == int(u):
                    sign = -1.0
            H1_i = sign * (float(amp) / CO_rel) * np.exp(1j * float(phi))
            H1_ref = H1_i * np.exp(-1j * z_arg_rad)
            H1_num += w * H1_ref
            H1_den += w
            if mQ is not None and np.isfinite(mQ):
                DC_num += w * (sign * float(mQ) / CO_rel)
                DC_den += w
        if H1_den > 0:
            H1_mean = H1_num / H1_den
            d['_h_amp_H1_norm_v2'] = float(abs(H1_mean))
            d['_h_phase_H1_norm_v2'] = float(np.angle(H1_mean))
        if DC_den > 0:
            d['_h_amp_DC_norm_v2'] = float(DC_num / DC_den)
        if H1_den > 0 or DC_den > 0:
            n_touched += 1
    return n_touched


# ── Porous zone + outlet detection ────────────────────────────────
def find_porous_and_outlets(G: nx.Graph):
    tagged = [(u, v) for u, v, d in G.edges(data=True)
              if d.get('radial_uncertain')]
    U = set()
    for u, v in tagged:
        U.add(u); U.add(v)
    A_a = {n for n, d in G.nodes(data=True)
           if d.get('boundary_type') == 'source'}
    V_a = {n for n, d in G.nodes(data=True)
           if d.get('boundary_type') == 'sink'}
    outlets = []
    for n in U:
        for m in G.neighbors(n):
            if m in U or m in A_a or m in V_a: continue
            if G.edges[n, m].get('radial_uncertain'): continue
            outlets.append(n); break
    outlets = sorted(set(outlets))
    H = nx.Graph(); H.add_edges_from(tagged)
    comps = sorted(nx.connected_components(H), key=lambda c: -len(c))
    node_to_comp = {n: ci for ci, c in enumerate(comps) for n in c}
    outlet_comp = np.array([node_to_comp.get(o, -1) for o in outlets])
    return U, outlets, outlet_comp, A_a, V_a, comps


# ── DC porous-source fit (α_k) ────────────────────────────────────
def fit_dc_alpha(G: nx.Graph, exclude_tiles: Optional[Set[int]] = None):
    """Constrained NNLS + SLSQP fit for the α_k values, using per-edge
    representative Q̄_rep (simple recipe).  Returns (outlets, α)."""
    U, outlets, outlet_comp, A_a, V_a, comps = find_porous_and_outlets(G)
    Nodes = (set(G.nodes()) - U - A_a) | set(outlets)
    Redges = [(u, v, d) for u, v, d in G.edges(data=True)
               if not d.get('radial_uncertain')
               and u in Nodes and v in Nodes]
    V_in = V_a & Nodes
    interior = sorted(Nodes - V_in)
    idx = {n: i for i, n in enumerate(interior)}
    N = len(interior)

    Ls = lil_matrix((N, N), dtype=np.float64)
    edge_meta = []; g_scale = 0.0
    for u, v, d in Redges:
        R_px = float(d['radius']); L_px = float(d['length'])
        g = (R_px ** 4) / max(L_px, 1e-6)
        iu = idx.get(u); iv = idx.get(v)
        if iu is not None:
            Ls[iu, iu] += g
            if iv is not None: Ls[iu, iv] -= g
        if iv is not None:
            Ls[iv, iv] += g
            if iu is not None: Ls[iv, iu] -= g
        edge_meta.append((iu, iv, g, u, v, d))
        g_scale += g
    g_scale /= max(len(Redges), 1)
    for k in range(N):
        Ls[k, k] += 1e-8 * g_scale
    solve = factorized(Ls.tocsc())
    A = np.zeros((len(edge_meta), len(outlets)))
    for k, o in enumerate(outlets):
        b = np.zeros(N); b[idx[o]] = 1.0
        p = solve(b)
        for e_i, (iu, iv, g, u, v, d) in enumerate(edge_meta):
            pu = p[iu] if iu is not None else 0.0
            pv = p[iv] if iv is not None else 0.0
            A[e_i, k] = g * (pu - pv)

    # Targets from Q̄_rep on each edge (simple recipe representative).
    Q_meas = np.full(len(edge_meta), np.nan)
    w = np.zeros(len(edge_meta))
    f0_map = G.graph.get('tile_f0_piv', {}) or {}
    tile_gauges = G.graph['tile_gauges']
    excl = exclude_tiles or set()
    for e_i, (iu, iv, g, u, v, d) in enumerate(edge_meta):
        Qbar, sigma = _repr_stats_one_edge(d, f0_map, tile_gauges,
                                             u=u, v=v, exclude_tiles=excl)
        if not (np.isfinite(Qbar) and np.isfinite(sigma) and sigma > 0):
            continue
        Q_meas[e_i] = Qbar
        w[e_i] = (Qbar / sigma) ** 2

    mask = np.isfinite(Q_meas) & (w > 0)
    if not mask.any():
        raise RuntimeError("no usable measured targets for DC fit")
    w = w / w[mask].max()

    A_m = A[mask]; Qm = Q_meas[mask]; W = w[mask]
    n_out = len(outlets)
    sqrtW = np.sqrt(W)
    Ah = A_m * sqrtW[:, None]
    bh = Qm * sqrtW
    AtWA = Ah.T @ Ah; AtWQ = Ah.T @ bh
    lam = 1e-3 * (np.trace(AtWA) / max(n_out, 1))
    M = AtWA + lam * np.eye(n_out); b_ = AtWQ

    # Warm start via augmented NNLS
    Ah_aug = np.vstack([Ah, np.sqrt(lam) * np.eye(n_out)])
    bh_aug = np.concatenate([bh, np.zeros(n_out)])
    q0, _ = nnls(Ah_aug, bh_aug, maxiter=10 * n_out)

    # Equality constraint Σα_zone0 = Σα_zone1
    c_eq = np.zeros(n_out)
    c_eq[outlet_comp == 0] = 1.0
    c_eq[outlet_comp == 1] = -1.0
    def _obj(q): return 0.5 * q @ M @ q - b_ @ q
    def _grad(q): return M @ q - b_
    res = minimize(_obj, q0, jac=_grad, method='SLSQP',
                    bounds=[(0.0, None)] * n_out,
                    constraints=[{'type': 'eq',
                                    'fun': lambda x: float(c_eq @ x),
                                    'jac': lambda x: c_eq}],
                    options={'ftol': 1e-10, 'maxiter': 500})
    return outlets, np.asarray(res.x)


def _repr_stats_one_edge(d, f0_map, tile_gauges, u, v, exclude_tiles):
    """Simple-recipe representative for one edge (see the standalone
    scripts).  Returns (Qbar, sigma_MAD).  Sign-aligned to (u, v)."""
    excl = exclude_tiles or set()
    ms = d.get('measurements_piv') or []
    num = np.zeros(N_BINS_CF); den = 0.0
    for m in ms:
        tid = m.get('tile_id')
        if tid is None or int(tid) in excl: continue
        Qt = m.get('Q_t')
        if Qt is None: continue
        Qt = np.asarray(Qt, float)
        if Qt.size < 30: continue
        f0 = f0_map.get(int(tid), f0_map.get(str(tid)))
        g = tile_gauges.get(int(tid)) or tile_gauges.get(str(tid))
        if f0 is None or g is None: continue
        CO_rel = float(g.get('CO_rel', 1.0))
        if CO_rel <= 0: continue
        z_arg_deg = float(g.get('z_arg_deg', 0.0))
        dphi_cycles = -z_arg_deg / 360.0
        Qt_scaled = Qt / CO_rel
        ff = m.get('flow_from'); ft = m.get('flow_to')
        if ff is not None and ft is not None:
            if int(ff) == int(v) and int(ft) == int(u):
                Qt_scaled = -Qt_scaled
            elif int(ff) != int(u) or int(ft) != int(v):
                continue
        snr = None
        for h in ((m.get('harm') or {}).get('harmonics') or []):
            if h.get('k') == 1: snr = h.get('snr_db'); break
        snr_val = float(snr) if snr is not None and np.isfinite(snr) else -np.inf
        if snr_val < SNR_H1_DB_GATE:
            continue
        w = 10 ** (snr_val / 10.0)
        t = np.arange(len(Qt_scaled)) * FRAME_DT
        phase = ((t * float(f0)) + dphi_cycles) % 1.0
        ib = np.clip((phase * N_BINS_CF).astype(int), 0, N_BINS_CF - 1)
        fold = np.zeros(N_BINS_CF); cnt = np.zeros(N_BINS_CF)
        for i, val in zip(ib, Qt_scaled):
            if np.isfinite(val):
                fold[i] += val; cnt[i] += 1
        ok = cnt > 0
        fold[ok] /= cnt[ok]
        num += w * fold; den += w
    if den <= 0:
        return float('nan'), float('nan')
    fold = num / den
    phi = (np.arange(N_BINS_CF) + 0.5) / N_BINS_CF
    X = [np.ones(N_BINS_CF)]
    for k in range(1, 4):
        X.append(np.cos(2 * np.pi * k * phi))
        X.append(np.sin(2 * np.pi * k * phi))
    X = np.stack(X, axis=1)
    coef, *_ = np.linalg.lstsq(X, fold, rcond=None)
    fit = X @ coef
    resid = fold - fit
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return float(coef[0]), sigma


# ── Wrapper: run the full carved sim end-to-end ───────────────────
def run_carved_sim_full(G: nx.Graph,
                          exclude_tiles: Optional[Set[int]] = None,
                          D: float = DEFAULT_D_AREAL,
                          verbose: bool = True):
    """Rebuild v2 fields, fit α, carve the graph, run LOCKED two-pass
    TL sim, and write `_h_*_sim_carved` fields back to G.  Returns a
    dict of stats and result."""
    # Load the standalone sim module (ported to plexusflow 2026-09-30)
    from plexusmech.tl.transmission_line import solve_transmission_line
    from importlib import import_module
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from run_sim_carved_outlets import (
        run_carved_outlets_sim, _extract_DA_harmonics,
        _phi_shift_per_outlet, _build_carved_graph,
    )

    if verbose:
        print(f"[carved-sim] recomputing v2 fields "
              f"(exclude={sorted(exclude_tiles) if exclude_tiles else 'none'})")
    n_v2 = recompute_v2_fields(G, exclude_tiles=exclude_tiles)
    if verbose:
        print("[carved-sim] fitting DC α ...")
    outlets, alpha = fit_dc_alpha(G, exclude_tiles=exclude_tiles)

    U, _, outlet_comp, A_a, V_a, _ = find_porous_and_outlets(G)
    if verbose:
        print("[carved-sim] extracting DA + φ_shift")
    H_DA = _extract_DA_harmonics(G, sorted(A_a))
    phi_shift = _phi_shift_per_outlet(G, outlets)
    Gc = _build_carved_graph(G, U, outlets, A_a)
    if verbose:
        print("[carved-sim] running LOCKED two-pass TL sim")
    result = run_carved_outlets_sim(Gc, alpha, outlets, phi_shift, H_DA,
                                      D=D, verbose=verbose)
    # Copy the sim fields from Gc back to G on shared edges.
    n_copied = 0
    for u, v, d in Gc.edges(data=True):
        if not G.has_edge(u, v): continue
        for k in list(d.keys()):
            if k.startswith('_h_') and k.endswith('_sim_carved'):
                G.edges[u, v][k] = d[k]
                n_copied += 1
    return dict(n_v2=n_v2, n_outlets=len(outlets), alpha_sum=float(alpha.sum()),
                 D=D, exclude_tiles=sorted(exclude_tiles) if exclude_tiles else [],
                 n_sim_fields=n_copied)
