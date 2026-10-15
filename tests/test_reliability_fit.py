"""Reliability-model likelihood: exact gradient and a well-posed, reproducible fit on synthetic junctions."""
import numpy as np
from scipy.optimize import check_grad, minimize

from plexusflow.reconcile.reliability import _nll


def _synthetic(n_tiles=6, n_junctions=300, seed=0):
    rng = np.random.default_rng(seed)
    cT_true = rng.uniform(0.05, 0.3, n_tiles); cf, ca, s0 = 0.4, 1.0, 1e-3
    idx, dc, f, al, ti, net = [], [], [], [], [], []
    for k in range(n_junctions):
        t = k % n_tiles; q = rng.lognormal(-3, 1, 3); ff = rng.uniform(0.2, 1, 3); aa = rng.uniform(0.5, 1, 3)
        c = cT_true[t] + cf * (1 - ff) + ca * (1 - aa); v = np.sum((q * c) ** 2) + s0 ** 2
        idx += [k] * 3; dc += list(q); f += list(ff); al += list(aa); ti += [t] * 3; net.append(rng.normal(0, np.sqrt(v)))
    return tuple(np.array(x) for x in (idx, dc, f, al, ti, net)), n_tiles


def test_gradient_matches_finite_differences():
    d, nT = _synthetic()
    x = np.r_[np.full(nT, np.log(0.1)), np.log(0.5), np.log(0.5), np.log(1e-3)]
    err = check_grad(lambda th: _nll(th, d, nT, grad=False), lambda th: _nll(th, d, nT)[1], x)
    assert err / np.linalg.norm(_nll(x, d, nT)[1]) < 1e-5


def test_fit_independent_of_start():
    d, nT = _synthetic()
    sols = []
    for start in (np.log(0.05), np.log(0.2)):
        x0 = np.r_[np.full(nT, start), np.log(0.5), np.log(0.5), np.log(1e-3)]
        r = minimize(_nll, x0, args=(d, nT), jac=True, method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-9, maxiter=100000))
        sols.append(r.x)
    assert np.max(np.abs(np.exp(sols[0]) - np.exp(sols[1])) / np.exp(sols[0])) < 1e-4
