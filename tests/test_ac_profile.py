"""AC profile fit: per-beat uncertainty, variance→κ mapping, vein forcing and known-truth recovery of D."""
import numpy as np
import scipy.sparse as sp

from plexusflow.reconcile.harmonics import beat_uncertainty, phasors, FS
from plexusmech.ac.likelihood import kappa_of_var
from plexusmech.ac.model import solve, vein_forcing, PX_M
from plexusmech.ac.profile import fit_profile, quadratic_min


def test_beat_uncertainty_predicts_phase_scatter():
    f0, t = 2.5, np.arange(int(4.0 * FS)) / FS; rng = np.random.default_rng(1); est, pred = [], []
    for _ in range(300):
        q = 1.0 + 0.4 * np.cos(2 * np.pi * f0 * t + 0.7) + 0.15 * np.cos(4 * np.pi * f0 * t - 0.3) + rng.normal(0, 0.3, t.size)
        est.append(np.angle(phasors(q, f0)[1][0])); pred.append(beat_uncertainty(q, f0)["var_phi1"])
    sd_emp = np.std(np.angle(np.exp(1j * (np.array(est) - np.angle(np.mean(np.exp(1j * np.array(est))))))))
    assert 0.7 < np.sqrt(np.median(pred)) / sd_emp < 1.4


def test_beat_uncertainty_needs_three_beats():
    q = np.ones(int(0.7 * FS))
    assert np.isnan(beat_uncertainty(q, 2.5)["var_phi1"])


def test_kappa_of_var_limits():
    assert abs(kappa_of_var(1e-3) * 1e-3 - 1) < 0.05            # small variance: κ ≈ 1/V
    assert kappa_of_var(20.0) < 0.01                              # uninformative phase: κ → 0


def _ladder(n=8, m=6, seed=0):
    """n × m grid network; carve = left column (top half L, bottom half R); two vein nodes on the right column."""
    rng = np.random.default_rng(seed); idx = lambda i, j: i * m + j; edges = []
    for i in range(n):
        for j in range(m):
            if j + 1 < m:
                edges.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < n:
                edges.append((idx(i, j), idx(i + 1, j)))
    nN, nE = n * m, len(edges); t = np.array([u for u, v in edges]); h = np.array([v for u, v in edges])
    B = sp.csr_matrix((np.r_[np.ones(nE), -np.ones(nE)], (np.r_[t, h], np.r_[np.arange(nE), np.arange(nE)])), shape=(nN, nE))
    inj = np.array([idx(i, 0) for i in range(n)]); out = np.array([idx(1, m - 1), idx(n - 2, m - 1)])
    dom = dict(B=B, inj=inj, out=out, bnd=np.r_[inj, out], side=np.array(["L" if i < n // 2 else "R" for i in range(n)]),
               side_out=np.array(["L", "R"]), nodes=list(range(nN)), edges=edges)
    Qp = rng.uniform(0.5, 1.5, nE)
    F = {(k, "flux"): np.zeros(nN, complex) for k in (1, 2)}
    for k in (1, 2):
        F[(k, "flux")][inj] = rng.uniform(0.3, 0.6, n) * np.exp(1j * k * rng.uniform(-0.3, 0.3, n))
        F[(k, "flux")][out] = -0.4 * np.exp(1j * k * 2.0)
    C = dict(f0=2.5, nN=nN, nE=nE, dom=dom, Qp=Qp, Qs_dc=Qp.copy(), elig=np.ones(nE, bool), F=F, tail=t, head=h,
             r_m=rng.uniform(6, 12, nE) * PX_M, l_m=rng.uniform(80, 160, nE) * PX_M)
    return C


def test_profile_recovers_known_D():
    C = _ladder(); D0 = 4.0e-4
    C["H"] = {k: solve(C, k, D0, 0.0, "flux") for k in (1, 2)}
    field = dict(tile=np.zeros(C["nE"], int), gauge_sd_deg={0: 0.0}, ref_tile=0,
                 var_phi=np.full((2, C["nE"]), 1e-3), var_logpi=np.full(C["nE"], 1e-3))
    grid = D0 * np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0])
    for w in ("waveform", "c"):
        R = fit_profile(C, np.full(C["nE"], 0.3), field, d_grid=grid, weights=w, n_jobs=1, progress=None)
        assert R["D"] == D0, (w, R["D"])
        assert abs(quadratic_min(grid, R["profile_D"]) / D0 - 1) < 0.1


def test_vein_forcing_mean_and_own():
    C = _ladder(); C["H"] = {k: solve(C, k, 4e-4, 0.0, "flux") for k in (1, 2)}
    field = dict(elig=np.ones(C["nE"], bool), H=np.array([C["H"][1], C["H"][2], C["H"][2]]), tile=np.zeros(C["nE"], int))
    Cm, info = vein_forcing(C, field, mode="mean"); out = C["dom"]["out"]; fdc = C["dom"]["B"] @ C["Qp"]
    r = [Cm["F"][(1, "flux")][n] / fdc[n] for n in out]
    assert np.allclose(r[0], r[1]) and np.isclose(r[0], info["mean"][1])
    Co, _ = vein_forcing(C, field, mode="own")
    assert np.isclose(Co["F"][(1, "flux")][out[0]] / fdc[out[0]], info["ratio"][1][int(out[0])])
