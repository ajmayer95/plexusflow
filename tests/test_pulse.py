"""plexusflow.ac.pulse and ac.speed: tile clocks from the speed pulse of shared vessels, on synthetic data."""
import numpy as np
from plexusflow.ac.pulse import tile_clocks, flux_pairs, orient_to_majority
from plexusflow.ac.speed import speed_pairs, harmonic_fit, reverses, path_points
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
    est, info = tile_clocks(flux_pairs(_recs(psi)), ref_tile=3, n_boot=5)
    assert len(info["components"]) == 1 and set(est) == set(psi) and info["clock_sd_deg"][3] == 0.0
    assert max(abs(np.angle(np.exp(1j * (est[t] - psi[t])))) for t in psi) < 1e-6


def test_unlinked_tiles_get_no_clock():
    psi = {t: 0.1 * t for t in range(1, 7)}; P = flux_pairs(_recs(psi)); keep = ~(((P["a"] == 4) & (P["b"] == 5)) | ((P["a"] == 5) & (P["b"] == 4)))
    est, info = tile_clocks({k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in P.items()}, ref_tile=1)
    assert set(est) == {1, 2, 3, 4} and len(info["components"]) == 2


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


def _velocity(psi, f0, direction, noise, seed, T=700, n=30, mean=0.5, amp=0.2):
    """(valid path points, U (T, n)) of one vessel in one tile: along-vessel velocity = direction × (mean + pulse), plus noise."""
    rng = np.random.default_rng(seed); t = np.arange(T) / 250.0
    u = direction * (mean + amp * np.cos(2 * np.pi * f0 * t + psi + 0.7) + 0.08 * np.cos(4 * np.pi * f0 * t + 2 * psi))
    return np.arange(n), (u[:, None] + noise * rng.standard_normal((T, n))).astype(np.float16)


def test_speed_pairs_give_the_clock_difference_whatever_the_flow_direction():
    f0 = {1: 2.7, 2: 2.9, 3: 2.5}; psi = {1: 0.0, 2: 1.1, 3: -0.6}; V = {t: {} for t in f0}
    for v in range(40):                                         # vessel v is seen in two tiles; in half of them the flow runs the other way in one tile
        a, b = [(1, 2), (2, 3), (1, 3)][v % 3]
        V[a][(v, v + 1)] = _velocity(psi[a], f0[a], +1, 0.3, 2 * v); V[b][(v, v + 1)] = _velocity(psi[b], f0[b], -1 if v % 2 else +1, 0.3, 2 * v + 1)
    P = speed_pairs(V, f0); assert len(P["d"]) == 40 and not P["reversing"].any()
    err = np.degrees(np.angle(np.exp(1j * (P["d"] - np.array([psi[a] - psi[b] for a, b in zip(P["a"], P["b"])])))))
    assert np.abs(err).max() < 5.0 and np.all(P["var"] > 0)
    est, _ = tile_clocks(P, ref_tile=1)
    assert max(abs(np.degrees(np.angle(np.exp(1j * (est[t] - psi[t]))))) for t in psi) < 1.0


def test_a_velocity_that_crosses_zero_is_marked_reversing():
    t = np.arange(700) / 250.0; steady = 0.5 + 0.2 * np.cos(2 * np.pi * 2.7 * t); swinging = 0.05 + 0.3 * np.cos(2 * np.pi * 2.7 * t)
    assert not reverses(steady) and not reverses(-steady) and reverses(swinging)
    V = {1: {(0, 1): (np.arange(10), np.tile(steady[:, None], (1, 10)).astype(np.float16))}, 2: {(0, 1): (np.arange(10), np.tile(swinging[:, None], (1, 10)).astype(np.float16))}}
    assert speed_pairs(V, {1: 2.7, 2: 2.7})["reversing"].all()


def test_pairs_need_a_common_piece_of_vessel():
    u = (0.5 + 0.2 * np.cos(2 * np.pi * 2.7 * np.arange(700) / 250.0))[:, None]
    V = {1: {(0, 1): (np.arange(0, 10), np.tile(u, (1, 10)).astype(np.float16))}, 2: {(0, 1): (np.arange(7, 20), np.tile(u, (1, 13)).astype(np.float16))}}
    assert len(speed_pairs(V, {1: 2.7, 2: 2.7})["d"]) == 0                          # only 3 common path points
    assert path_points(np.array([[0.0, 0.0], [1.0, 0.0]])) is None and len(path_points(np.array([[0.0, 0.0], [20.0, 0.0]]))) == 10


def test_harmonic_fit_matches_the_flux_fit():
    rng = np.random.default_rng(1); t = np.arange(750) / 250.0; q = 1.0 + 0.3 * np.cos(2 * np.pi * 2.7 * t + 0.4) + 0.05 * rng.standard_normal(750)
    a0, H, s2 = harmonic_fit(q, 2.7)
    assert abs(a0 - 1.0) < 0.01 and abs(abs(H[0]) - 0.3) < 0.01 and abs(np.angle(H[0]) - 0.4) < 0.03 and abs(s2 / residual_variance(q, 2.7) - 1) < 1e-6
