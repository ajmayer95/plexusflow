"""plexusflow.ac.pulse: direction rule and joint tile clocks on synthetic recordings."""
import numpy as np
from plexusflow.ac.pulse import direction_rule, tile_clocks, orient_to_majority
from plexusflow.reconcile.dc import residual_variance


def _recs(psi, n_ves=60, seed=0, flipped=()):
    rng = np.random.default_rng(seed); tiles = sorted(psi); out = {}
    for v in range(n_ves):
        a, b = rng.choice(len(tiles) - 1) + np.array([0, 1])                      # chain of overlaps: tile k with k + 1
        H = (0.2 + 0.3 * rng.random()) * np.exp(1j * rng.uniform(-np.pi, np.pi)); a0 = 1.0 + rng.random(); rs = []
        for t in (tiles[a], tiles[b]):
            s = -1.0 if (v, t) in flipped else 1.0
            rs.append(dict(t=t, a0=s * a0, H=s * np.array([H * np.exp(1j * psi[t]), 0.3 * H * np.exp(2j * psi[t])]), sigma2=1e-4 * (1 + rng.random()), rho=0.5, n=750))
        out[(v, v + 1)] = rs
    return out


def test_tile_clocks_recover_offsets():
    psi = {t: 0.0 if t == 3 else float(np.random.default_rng(t).uniform(-2.5, 2.5)) for t in range(1, 9)}
    est, info = tile_clocks(_recs(psi), ref_tile=3)
    assert len(info["components"]) == 1 and set(est) == set(psi)
    assert max(abs(np.angle(np.exp(1j * (est[t] - psi[t])))) for t in psi) < 1e-6


def test_direction_rule_drops_the_minority_recording():
    rs = [dict(t=1, a0=1.0, H=np.array([0.2 + 0j, 0.05]), sigma2=1e-4, rho=0.5, n=750), dict(t=2, a0=1.1, H=np.array([0.2 + 0j, 0.05]), sigma2=2e-4, rho=0.5, n=750),
          dict(t=3, a0=-0.3, H=np.array([-0.2 + 0j, -0.05]), sigma2=5e-3, rho=0.1, n=750)]
    kept, n_drop = direction_rule({(0, 1): rs})
    assert n_drop == 1 and [r["t"] for r in kept[(0, 1)]] == [1, 2]


def test_residual_variance_of_a_noisy_harmonic_trace():
    rng = np.random.default_rng(1); fs, f0, M = 250.0, 2.7, 750; t = np.arange(M) / fs
    q = 1.0 + 0.3 * np.cos(2 * np.pi * f0 * t + 0.4) + 0.1 * np.cos(4 * np.pi * f0 * t - 1.0) + 0.05 * rng.standard_normal(M)
    assert abs(residual_variance(q, f0) / (0.05 ** 2 / M) - 1) < 0.15
    assert not np.isfinite(residual_variance(q[:10], f0))


def test_orient_to_majority_negates_the_minority_recording_of_the_listed_edges_only():
    mk = lambda t, a0: dict(t=t, a0=a0, H=np.array([0.3 * np.sign(a0) + 0j, 0.1 * np.sign(a0)]), sigma2=1e-4, rho=0.8, n=750)
    recs = {(1, 2): [mk(14, 1.6), mk(21, 1.4), mk(20, -0.8)], (3, 4): [mk(1, 1.0), mk(2, -1.0)]}
    out, n = orient_to_majority(recs, [(1, 2)])
    assert n == 1 and [r["a0"] for r in out[(1, 2)]] == [1.6, 1.4, 0.8] and np.allclose(out[(1, 2)][2]["H"], [0.3, 0.1])
    assert out[(3, 4)] is recs[(3, 4)]
