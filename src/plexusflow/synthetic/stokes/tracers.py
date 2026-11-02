"""Exact P2 velocity evaluation on a tet mesh, pseudo-time tracer advection, and the exposure-blurred tracer renderer.

FastP2: u(x) = Σ_i λ_i(2λ_i − 1) u_i + Σ_edges 4 λ_a λ_b u_ab in the containing tet (the same field as scikit-fem's probes).
advect_pseudotime: quasi-steady flow u(x, t) = Q0 s(t) û(x) ⇒ tracer paths are streamlines of û; integrate once in
τ (dτ = Q0 s dt), injecting at inflow caps at rate C·|q_cap| per unit τ, recording positions at the requested τ values.
splat: anisotropic Gaussians, σ² + L²/12 along the in-frame displacement (exposure = frame interval), σ² across.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class FastP2:
    def __init__(self, mesh, ub, u):
        P = mesh.p.T.copy(); T = mesh.t.T.copy(); self.P, self.T = P, T
        A = P[T[:, :3]] - P[T[:, 3]][:, None, :]
        self.Minv = np.linalg.inv(np.transpose(A, (0, 2, 1)))
        self.tree = cKDTree(P[T].mean(1))
        nd = ub.nodal_dofs; ed = ub.edge_dofs                 # (3, n_vert), (3, n_edge)
        self.uv = u[nd].T.copy()                               # (n_vert, 3)
        self.ue = u[ed].T.copy()                               # (n_edge, 3)
        E = mesh.edges.T; t2e = mesh.t2e.T                     # (n_edge, 2), (nT, 6)
        la = np.zeros((len(T), 6), np.int64); lb = np.zeros((len(T), 6), np.int64)
        for j in range(6):
            va, vb = E[t2e[:, j], 0], E[t2e[:, j], 1]
            la[:, j] = np.argmax(T == va[:, None], axis=1); lb[:, j] = np.argmax(T == vb[:, None], axis=1)
        self.t2e, self.la, self.lb = t2e, la, lb
        # tet neighbours (share a face) for cell tracking
        f2t = mesh.f2t.T; t2f = mesh.t2f.T
        nb = f2t[t2f]                                          # (nT, 4, 2)
        self.nbr = np.where(nb[:, :, 0] == np.arange(len(T))[:, None], nb[:, :, 1], nb[:, :, 0])

    def bary(self, X, c):
        lam = np.einsum("nij,nj->ni", self.Minv[c], X - self.P[self.T[c, 3]])
        return np.column_stack([lam, 1 - lam.sum(1)])

    def locate(self, X, hint=None, k=24):
        cell = np.full(len(X), -1)
        if hint is not None:
            h = hint.copy(); ok = h >= 0
            if ok.any():
                L = self.bary(X[ok], h[ok]); good = L.min(1) >= -1e-9; idx = np.flatnonzero(ok)
                cell[idx[good]] = h[ok][good]
                todo = idx[~good]
                for j in range(4):                              # one ring of face neighbours
                    if not len(todo):
                        break
                    cand = self.nbr[hint[todo], j]; v = cand >= 0
                    if v.any():
                        L = self.bary(X[todo[v]], cand[v]); g = L.min(1) >= -1e-9
                        cell[todo[v][g]] = cand[v][g]
                    todo = todo[cell[todo] < 0]
        rest = np.flatnonzero(cell < 0)
        if len(rest):
            _, cand = self.tree.query(X[rest], k=k)
            for j in range(k):
                r = rest[cell[rest] < 0]
                if not len(r):
                    break
                cc = cand[np.searchsorted(rest, r), j]; L = self.bary(X[r], cc); g = L.min(1) >= -1e-9
                cell[r[g]] = cc[g]
        return cell

    def __call__(self, X, cell=None):
        if cell is None:
            cell = self.locate(X)
        v = np.full((len(X), 3), np.nan); ok = cell >= 0
        if ok.any():
            c = cell[ok]; L = self.bary(X[ok], c)
            Vv = self.uv[self.T[c]]                                    # (n, 4, 3)
            out = np.einsum("ni,nij->nj", L * (2 * L - 1), Vv)
            Le = 4 * np.take_along_axis(L, self.la[c], 1) * np.take_along_axis(L, self.lb[c], 1)   # (n, 6)
            out += np.einsum("ni,nij->nj", Le, self.ue[self.t2e[c]])
            v[ok] = out
        return v, cell


def p2_field(p, t, u) -> FastP2:
    """FastP2 for a P2 vector field stored as (mesh points p, tets t, dof vector u) — the saved Stokes solutions."""
    from skfem import MeshTet, ElementVector, ElementTetP2, Basis
    mesh = MeshTet(p, t)
    return FastP2(mesh, Basis(mesh, ElementVector(ElementTetP2())), u)


def seed_tracers(F: FastP2, C: float, rng) -> np.ndarray:
    """Poisson(C · lumen volume) tracers uniformly inside the mesh at τ = 0."""
    P = F.P; vol = np.sum(np.abs(np.linalg.det(P[F.T[:, :3]] - P[F.T[:, 3]][:, None, :]))) / 6
    n0 = rng.poisson(C * vol); cand = rng.uniform(P.min(0), P.max(0), size=(n0 * 8, 3))
    return cand[F.locate(cand) >= 0][:n0]


def advect_pseudotime(F, caps, tau_needed, C, rng, x0, max_step_px=0.4):
    """Integrate tracers along û in τ.  caps: list of (centre, outward t, R, q) with q = outward unit flux (q < 0 inflow).
    Returns positions[k] (array of tracer ids, positions) at each τ in tau_needed (sorted)."""
    vmax = np.nanmax(np.linalg.norm(F(F.P[np.unique(F.T)][::5])[0], axis=1))
    dtau = max_step_px / max(vmax, 1e-12)
    X = x0.copy(); ids = np.arange(len(X)); next_id = len(X); cell = F.locate(X)
    out = {}; tau = 0.0; k_need = 0; tau_needed = np.asarray(tau_needed)
    while k_need < len(tau_needed):
        # inject
        new = []
        for c_, t_, R, q in caps:
            if q >= 0:
                continue
            n = rng.poisson(C * (-q) * dtau)
            e1 = np.cross(t_, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(t_, e1)
            while n > 0:
                a, b = rng.uniform(-R, R, 2)
                if a * a + b * b < R * R and rng.random() < 1 - (a * a + b * b) / R ** 2:
                    new.append(c_ - t_ * rng.uniform(0.05, max_step_px) + a * e1 + b * e2); n -= 1
        if new:
            nw = np.array(new); X = np.vstack([X, nw]); ids = np.r_[ids, np.arange(next_id, next_id + len(nw))]; next_id += len(nw)
            cell = np.r_[cell, F.locate(nw)]
        v1, cell = F(X, F.locate(X, cell))
        mid = X + 0.5 * dtau * np.nan_to_num(v1); v2, cm = F(mid, F.locate(mid, cell))
        Xn = X + dtau * np.nan_to_num(v2); cn = F.locate(Xn, cm)
        inside = cn >= 0
        # outside: exited through a cap (beyond its plane) → removed; otherwise grazing the wall → hold position
        exited = np.zeros(len(X), bool)
        for c_, t_, R, q in caps:
            exited |= (~inside) & ((Xn - c_) @ t_ > -1e-6) & (np.linalg.norm((Xn - c_) - np.outer((Xn - c_) @ t_, t_), axis=1) < 1.5 * R)
        hold = (~inside) & ~exited
        Xn[hold] = X[hold]; cn[hold] = cell[hold]
        tau_new = tau + dtau
        while k_need < len(tau_needed) and tau_needed[k_need] <= tau_new:
            w = (tau_needed[k_need] - tau) / dtau; keep = ~exited
            out[k_need] = (ids[keep], (X + w * (Xn - X))[keep].astype(np.float32)); k_need += 1
        keep = ~exited; X, cell, ids, tau = Xn[keep], cn[keep], ids[keep], tau_new
    return out, dtau


def _splat_py(img, rows, cols, dr, dc, sig, amp):
    H, W = img.shape
    for i in range(len(rows)):
        L2 = dr[i] * dr[i] + dc[i] * dc[i]; s2 = sig[i] * sig[i]; sl = s2 + L2 / 12.0
        if L2 > 1e-12:
            ur, uc = dr[i] / np.sqrt(L2), dc[i] / np.sqrt(L2)
        else:
            ur, uc = 1.0, 0.0
        k = int(np.ceil(3.0 * np.sqrt(sl))) + 1
        r0, c0 = rows[i], cols[i]; ri, ci = int(np.round(r0)), int(np.round(c0))
        norm = amp[i] * sig[i] / np.sqrt(sl)                 # same integrated brightness as the unblurred spot
        for r in range(max(ri - k, 0), min(ri + k + 1, H)):
            for c in range(max(ci - k, 0), min(ci + k + 1, W)):
                yr, yc = r - r0, c - c0; a = yr * ur + yc * uc; b = -yr * uc + yc * ur
                img[r, c] += norm * np.exp(-0.5 * (a * a / sl + b * b / s2))
    return img


try:                                                          # numba (optional `bench` extra) makes rendering ~100× faster
    import numba
    splat = numba.njit(cache=True)(_splat_py)
except ImportError:                                           # pragma: no cover
    splat = _splat_py
