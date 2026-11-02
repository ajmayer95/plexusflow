"""Regression tests for the DC network methods (plexusflow.reconcile.network) on a small synthetic network with known flow:
conservative projection, emergent boundary flux, passive Poiseuille flow and the venous-pressure penalty."""
import numpy as np
import scipy.sparse as sp

from plexusflow.reconcile.network import emergent, passive, project, venous_pressure_penalty


def grid_domain(nx=8, ny=6, seed=0):
    """nx × ny grid graph; inflow boundary = left column, two outflow (venous) nodes on the right edge."""
    rng = np.random.default_rng(seed)
    idx = lambda i, j: i * ny + j
    edges = [(idx(i, j), idx(i + 1, j)) for i in range(nx - 1) for j in range(ny)] + \
            [(idx(i, j), idx(i, j + 1)) for i in range(nx) for j in range(ny - 1)]
    m, n = len(edges), nx * ny
    B = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[[u for u, v in edges], [v for u, v in edges]],
                                                       np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
    inj = np.array([idx(0, j) for j in range(ny)]); out = np.array([idx(nx - 1, 1), idx(nx - 1, ny - 2)])
    dom = dict(edges=edges, nodes=list(range(n)), B=B, inj=inj, out=out, bnd=np.r_[inj, out],
               side=np.array(["L"] * ny), side_out=np.array(["L", "R"]))
    r = rng.uniform(8, 20, m); length = rng.uniform(30, 80, m)
    return dom, r, length


def true_flow(dom, r, length):
    fb = np.zeros(dom["B"].shape[0]); fb[dom["inj"]] = 1.0; fb[dom["out"]] = -len(dom["inj"]) / 2
    return passive(r ** 4 / length, fb, dom), fb


def interior(dom):
    return np.setdiff1d(np.arange(dom["B"].shape[0]), dom["bnd"])


def test_passive_flow_satisfies_the_imposed_boundary_flux():
    dom, r, length = grid_domain()
    Q, fb = true_flow(dom, r, length)
    net = dom["B"] @ Q
    assert np.allclose(net[interior(dom)], 0, atol=1e-10)
    assert np.allclose(net[dom["bnd"]], fb[dom["bnd"]], atol=1e-10)


def test_projection_conserves_mass_and_fixes_conservative_input():
    dom, r, length = grid_domain()
    Q, _ = true_flow(dom, r, length)
    w = np.ones_like(Q)
    assert np.allclose(project(Q, w, dom), Q, atol=1e-10)                     # already conservative → unchanged
    noisy = Q + np.random.default_rng(1).normal(0, 0.2, Q.size)
    Qp = project(noisy, w, dom)
    assert np.max(np.abs((dom["B"] @ Qp)[interior(dom)])) < 1e-10             # interior KCL exact
    assert np.linalg.norm(Qp - Q) < np.linalg.norm(noisy - Q)                 # projection removes noise


def test_projection_weights_trust_reliable_measurements():
    dom, r, length = grid_domain()
    Q, _ = true_flow(dom, r, length)
    rng = np.random.default_rng(2); noisy = Q.copy(); bad = rng.random(Q.size) < 0.5
    noisy[bad] += rng.normal(0, 0.5, bad.sum())
    w = np.where(bad, 1e-3, 1.0)                                              # low weight on the corrupted half
    err_w = np.linalg.norm(project(noisy, w, dom) - Q); err_u = np.linalg.norm(project(noisy, np.ones_like(w), dom) - Q)
    assert err_w < 0.75 * err_u                                              # ~35 loop constraints on 82 edges: clearly better, not perfect


def test_emergent_boundary_flux_balances():
    dom, r, length = grid_domain()
    Q, fb = true_flow(dom, r, length)
    f = emergent(Q, dom)
    assert np.allclose(f, fb, atol=1e-10)
    assert abs(f.sum()) < 1e-10


def test_venous_pressure_penalty():
    dom, r, length = grid_domain()
    Q, fb = true_flow(dom, r, length)
    fb = fb.copy(); fb[dom["out"]] = [-4.0, -2.0]                               # unequal split → unequal pressures
    sigma = np.full(Q.size, 0.1)
    same, info0 = venous_pressure_penalty(fb, dom, r, length, sigma, sigma_P=np.inf)
    assert np.array_equal(same, fb) and info0["shift"] == 0.0
    adj, info = venous_pressure_penalty(fb, dom, r, length, sigma, sigma_P=1e-6)
    assert abs(info["dP_after"]) < 1e-3 * abs(info["dP_before"])               # tight penalty equalises the pressures
    assert np.isclose(adj[dom["out"]].sum(), fb[dom["out"]].sum())              # total outflow unchanged
    assert np.array_equal(adj[dom["inj"]], fb[dom["inj"]])                     # arterial boundary untouched
