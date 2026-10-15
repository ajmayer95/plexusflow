"""Conservative DC projection: Q̄_raw → Q̄' with B_I · Q̄' = 0.

Core idea (per expert feedback round, 2026-10-05):

    Instead of fitting α_j at carved outlets to match the measured field,
    first PROJECT the raw per-edge DC measurement onto the kernel of the
    interior-node incidence matrix B_I.  The emergent boundary fluxes
    then come straight from  f = B_bdry · Q̄'  — data-derived driving
    conditions, not free fit parameters.

Projection (closed-form, since the constraint is linear):

    Q̄' = Q̄_raw − B_I^T · (B_I B_I^T)^+ · (B_I · Q̄_raw)

where the pseudoinverse handles rank-deficient B_I (e.g. isolated interior
components reachable only through boundary nodes).

Public:
    build_signed_dc(G)             -> (edges_list, Q̄_raw, mask_valid)
    build_incidence(G, edges_list, boundary_nodes)
        -> (B_full, B_interior, B_boundary, interior_nodes, bdry_nodes)
    conservative_dc_projection(Q̄_raw, B_interior) -> Q̄'
    boundary_fluxes(Q̄_prime, B_boundary) -> {node: f_signed}
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ─── signed DC extraction ─────────────────────────────────────────────────
def _measured_direction(u, v, d) -> int:
    """+1 if u→v, -1 if v→u, 0 if unresolved (Q_kl-weighted vote)."""
    ms = d.get('measurements_piv') or []
    if not ms: return 0
    s = 0.0
    for m in ms:
        ff = m.get('flow_from'); ft = m.get('flow_to')
        if ff is None or ft is None: continue
        w = abs(float(m.get('Q_kl') or 0.0))
        if ff == u and ft == v: s += w
        elif ff == v and ft == u: s -= w
    return 1 if s > 0 else -1 if s < 0 else 0


def _edge_amp(d, prefer_norm=True) -> float:
    """Measured |Q̄| with normalized → raw fallback.  NaN if neither works."""
    if prefer_norm:
        q = d.get('_h_amp_DC_norm')
        if q is not None and np.isfinite(q) and q > 0: return float(q)
    q = d.get('_h_amp_DC')
    if q is not None and np.isfinite(q) and q > 0: return float(q)
    return float('nan')


def build_signed_dc(G, prefer_norm=True
                     ) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Return (edges_list, Q̄_raw signed, mask_valid).

    * `edges_list` — list of (u, v) tuples in iteration order
    * `Q̄_raw`     — signed DC flux in the canonical u→v direction (NaN if
                     direction is unresolved or amp is missing)
    * `mask_valid` — bool array True where Q̄_raw is finite
    """
    edges_list, Q, valid = [], [], []
    for u, v, d in G.edges(data=True):
        edges_list.append((u, v))
        amp = _edge_amp(d, prefer_norm=prefer_norm)
        sgn = _measured_direction(u, v, d)
        if sgn == 0 or not np.isfinite(amp):
            Q.append(np.nan); valid.append(False)
        else:
            Q.append(sgn * amp); valid.append(True)
    return edges_list, np.array(Q), np.array(valid)


# ─── incidence ────────────────────────────────────────────────────────────
def build_incidence(G, edges_list, boundary_nodes
                     ) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix,
                                 List[int], List[int]]:
    """Build the signed node–edge incidence matrix and partition rows into
    interior and boundary blocks.

    Convention: for edge e = (u, v), B[u, e] = -1 and B[v, e] = +1.  Then
    (B·Q)[n] = net inflow at node n (positive = flux ENTERS n).  KCL at
    an interior node is  (B·Q)[n] = 0.
    """
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edge_idx = {uv: j for j, uv in enumerate(edges_list)}
    rows, cols, data = [], [], []
    for (u, v), j in edge_idx.items():
        rows.append(node_idx[u]); cols.append(j); data.append(-1.0)
        rows.append(node_idx[v]); cols.append(j); data.append(+1.0)
    B_full = sp.csr_matrix((data, (rows, cols)),
                            shape=(len(nodes), len(edges_list)))
    bdry_set = set(int(n) for n in boundary_nodes)
    interior = [n for n in nodes if n not in bdry_set]
    bdry     = [n for n in nodes if n in bdry_set]
    I_rows = [node_idx[n] for n in interior]
    B_rows = [node_idx[n] for n in bdry]
    B_I = B_full[I_rows, :]
    B_B = B_full[B_rows, :]
    return B_full, B_I, B_B, interior, bdry


# ─── projection ───────────────────────────────────────────────────────────
def conservative_dc_projection(Q_raw: np.ndarray, B_I: sp.csr_matrix,
                                 valid_mask: np.ndarray = None,
                                 tol: float = 1e-8) -> np.ndarray:
    """Project Q_raw onto {Q : B_I·Q = 0} in least-squares sense.

    Edges with Q_raw = NaN (direction/magnitude unresolved) are kept
    unconstrained in the LS objective but still participate in KCL
    through B_I — they will be ASSIGNED a value by the projection step.

    Returns Q' with the SAME length as Q_raw; every edge gets a value.
    """
    # Treat missing entries as zero target but with zero weight — i.e.
    # minimize ||W (Q' - Q_raw)||² s.t. B_I Q' = 0.  Setting W_ii = 0 on
    # missing edges and 1 elsewhere gives the right behavior.
    n = len(Q_raw)
    Q_tgt = np.where(np.isnan(Q_raw), 0.0, Q_raw)
    if valid_mask is None:
        valid_mask = np.isfinite(Q_raw)
    w = valid_mask.astype(float)
    # Solve the KKT system:
    #     [ W  B_I^T ] [ Q'      ] = [ W Q_tgt ]
    #     [ B_I  0   ] [ λ       ]   [ 0       ]
    # Diagonal W; use sparse block solve.
    W = sp.diags(w, 0)
    # Regularize interior rows with no measured edges to avoid singular
    # system — add tiny Tikhonov on Q' (ridge).
    reg = 1e-10
    A11 = W + reg * sp.eye(n)
    nI = B_I.shape[0]
    KKT = sp.bmat([[A11, B_I.T],
                     [B_I, None]], format='csr')
    rhs = np.concatenate([w * Q_tgt, np.zeros(nI)])
    sol = spla.spsolve(KKT, rhs)
    return sol[:n]


def boundary_fluxes(Q_prime: np.ndarray, B_B: sp.csr_matrix,
                      bdry_nodes: Iterable[int]) -> Dict[int, float]:
    """Compute net flow (positive = INTO node) at each boundary node.

    By convention a positive value at an arterial outlet node means the
    projection says that node *receives* flow — i.e. inside the model's
    bookkeeping it should be a SINK; conversely, carved arterial outlets
    inject into the plexus, so we expect f < 0 at them under this
    convention.  Sign depends on how you partition the graph.
    """
    f = B_B @ Q_prime
    bdry_nodes = list(bdry_nodes)
    return {int(n): float(f[i]) for i, n in enumerate(bdry_nodes)}
