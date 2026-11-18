"""Reliability-model likelihood: exact gradient and a well-posed, reproducible fit on synthetic junctions."""
import numpy as np
from scipy.optimize import check_grad

from plexusflow.reconcile.reliability import _fit_theta, _nll


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
    a, b = _fit_theta(d, nT, jitters=(0.0,)), _fit_theta(d, nT, jitters=(0.7,))
    assert a.success and b.success
    assert np.max(np.abs(np.exp(a.x) - np.exp(b.x)) / np.exp(a.x)) < 1e-4


def test_single_c0_fit_recovers_constant():
    rng = np.random.default_rng(1); c0, cf, ca, s0 = 0.12, 0.3, 1.4, 1e-3; idx, dc, f, al, net = [], [], [], [], []
    for k in range(2000):
        q = rng.lognormal(-3, 1, 3); ff = rng.uniform(0.2, 1, 3); aa = rng.uniform(0.5, 1, 3)
        v = np.sum((q * (c0 + cf * (1 - ff) + ca * (1 - aa))) ** 2) + s0 ** 2
        idx += [k] * 3; dc += list(q); f += list(ff); al += list(aa); net.append(rng.normal(0, np.sqrt(v)))
    d = tuple(np.array(x) for x in (idx, dc, f, al, np.zeros(len(idx), int), net))
    r = _fit_theta(d, 1)
    assert r.success
    est = np.exp(r.x[:3])
    assert abs(est[1] / cf - 1) < 0.25 and abs(est[2] / ca - 1) < 0.25 and est[0] < 0.3


def test_hr_gain_recovers_exponent_and_normalises():
    import numpy as np
    from plexusflow.reconcile.gain import fit_hr_gain, apply_hr_gain
    rng = np.random.default_rng(0); f0 = {t: f for t, f in zip(range(1, 9), np.linspace(1.5, 3.0, 8))}; beta = 2.4; M = {}
    for k in range(300):
        q = np.exp(rng.normal(-2, 1)); e = (k, k + 1)
        for t in rng.choice(list(f0), 3, replace=False):
            M[(int(t), e)] = dict(tile=int(t), Q=q * (f0[t] / 2.25) ** beta * np.exp(rng.normal(0, 0.2)), align=0.9)
    fit = fit_hr_gain(M, f0)
    assert abs(fit["beta"] - beta) < 0.15
    Mc = apply_hr_gain(M, f0, fit)
    by = {}
    for (t, e), m in Mc.items(): by.setdefault(e, []).append(np.log(abs(m["Q"])))
    assert np.median([np.std(v) for v in by.values()]) < 0.3            # repeats agree after correction


def test_alignment_only_quality_model():
    from plexusflow.reconcile.reliability import c_of
    mod = dict(c_T={1: 0.2}, c0=0.2, profile_term="none", c_none=0.0, c_a=2.0, sigma0=1e-4)
    assert abs(c_of(mod, dict(tile=1, align=0.9, p=0.1, none=1.0)) - 0.4) < 1e-12          # parabolicity ignored
