"""plexusflow.ac.pulse: tile clocks from vessels recorded in two tiles, on synthetic data."""
import numpy as np
from plexusflow.ac.pulse import tile_clocks, flux_pairs, orient_to_majority
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
    est, info = tile_clocks(flux_pairs(_recs(psi), own_direction=False), ref_tile=3, n_boot=5)
    assert len(info["components"]) == 1 and set(est) == set(psi) and info["clock_sd_deg"][3] == 0.0
    assert max(abs(np.angle(np.exp(1j * (est[t] - psi[t])))) for t in psi) < 1e-6


def test_unlinked_tiles_get_no_clock():
    psi = {t: 0.1 * t for t in range(1, 7)}; P = flux_pairs(_recs(psi), own_direction=False); keep = ~(((P["a"] == 4) & (P["b"] == 5)) | ((P["a"] == 5) & (P["b"] == 4)))
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


def _noisy_recs(psi, links, n_ves, seed, sd_deg=6.0, flip_frac=0.0):
    """Vessels recorded in the two tiles of a random link, with phase noise; a fraction of the recordings read the flow backwards
    (mean and harmonics negated together)."""
    rng = np.random.default_rng(seed); out = {}
    for v in range(n_ves):
        a, b = links[rng.integers(len(links))]; H = 0.3 * np.exp(1j * rng.uniform(-np.pi, np.pi)); rs = []
        for t in (a, b):
            s = -1.0 if rng.random() < flip_frac else 1.0; ph = np.exp(1j * (psi[t] + rng.normal(0, np.radians(sd_deg) / np.sqrt(2))))
            rs.append(dict(t=t, a0=s * 1.0, H=s * np.array([H * ph, 0.3 * H * ph ** 2]), sigma2=np.radians(sd_deg) ** 2 / 2 * abs(H) ** 2 / 2, rho=0.5, n=750))
        out[(v, v + 1)] = rs
    return out


def test_own_direction_makes_backward_recordings_harmless_for_the_clocks():
    tiles = list(range(1, 9)); psi = {t: 0.0 if t == 3 else float(np.random.default_rng(t).uniform(-2.5, 2.5)) for t in tiles}
    recs = _noisy_recs(psi, [(t, t + 1) for t in tiles[:-1]], 1500, seed=2, flip_frac=0.25)      # a quarter of the recordings read the flow backwards
    err = lambda est: max(abs(np.degrees(np.angle(np.exp(1j * (est[t] - psi[t]))))) for t in psi)
    own, _ = tile_clocks(flux_pairs(recs, own_direction=True), ref_tile=3)
    P = flux_pairs(recs, own_direction=False); d = np.degrees(np.angle(np.exp(1j * (P["d"] - np.array([psi[a] - psi[b] for a, b in zip(P["a"], P["b"])])))))
    assert err(own) < 2.0 and np.mean(np.abs(d) > 90) > 0.3                           # in the fixed orientation a third of the pairs are half a cycle off
    Po = flux_pairs(recs); do = np.degrees(np.angle(np.exp(1j * (Po["d"] - np.array([psi[a] - psi[b] for a, b in zip(Po["a"], Po["b"])])))))
    assert np.mean(np.abs(do) > 90) < 0.01


def test_log_waveform_loss_recovers_distensibility(monkeypatch):
    """the log-waveform loss is minimal at the D that generated the harmonics, and symmetric in the amplitude ratio"""
    import plexusmech.ac.waveform as W
    n = 40; rng = np.random.default_rng(0); el = np.ones(n, bool); base = {k: rng.uniform(0.1, 0.5, n) * np.exp(1j * rng.uniform(-3, 3, n)) for k in (1, 2)}
    resp = lambda C, k, D, eta, gid, el_: (base[k] * np.exp(-1j * k * 1e3 * D) / (1 + 1e3 * D))[:, None]      # damping and delay grow with D
    monkeypatch.setattr(W, "group_response", resp); monkeypatch.setattr(W, "groups", lambda C: None)
    d_true = 4e-4; C = dict(elig=el, Qp=np.ones(n), Qs_dc=np.ones(n), H={k: resp(None, k, d_true, 0, None, el)[:, 0] for k in (1, 2)})
    R = W.fit_log_waveform(C, d_grid=np.logspace(np.log10(2.5e-4), np.log10(8e-4), 21), tile=np.arange(n) % 5, n_boot=20)
    assert abs(R["D_parabola"] / d_true - 1) < 0.03 and R["D_boot"][0] <= R["D_parabola"] <= R["D_boot"][2]
    C2 = dict(C, H={k: 2 * v for k, v in C["H"].items()}); C3 = dict(C, H={k: v / 2 for k, v in C["H"].items()})
    assert np.allclose(W.log_waveform_distance(C2, d_true), np.log(2) ** 2) and np.allclose(W.log_waveform_distance(C3, d_true), np.log(2) ** 2)
