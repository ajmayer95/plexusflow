"""Known-truth benchmark building blocks: exact P2 evaluation, tracer advection, renderer, background statistics."""
import numpy as np
import pytest

skfem = pytest.importorskip("skfem")

from plexusflow.synthetic.stokes.tracers import FastP2, advect_pseudotime, splat  # noqa: E402
from plexusflow.synthetic.stokes.bundle import Background, tile_noise, background_from_raw  # noqa: E402
from plexusflow.synthetic.stokes.render import camera, scale_to_contrast  # noqa: E402


def _box(n=3, L=6.0):
    from skfem import MeshTet, ElementVector, ElementTetP2, Basis
    mesh = MeshTet.init_tensor(*(np.linspace(0, L, n + 1),) * 3)
    return mesh, Basis(mesh, ElementVector(ElementTetP2()))


def test_fastp2_matches_scikit_fem_probes():
    mesh, ub = _box()
    u = ub.project(lambda x: np.stack([x[1] ** 2 - x[2], x[0] * x[1], np.ones_like(x[0])]))   # quadratic: exact in P2
    F = FastP2(mesh, ub, u)
    X = np.random.default_rng(0).uniform(0.1, 5.9, (200, 3))
    v, cell = F(X)
    assert (cell >= 0).all()
    ref = (ub.probes(X.T) @ u).reshape(3, -1).T
    assert np.allclose(v, ref, atol=1e-10)
    assert np.allclose(v[:, 0], X[:, 1] ** 2 - X[:, 2], atol=1e-8)


def test_advection_in_uniform_flow_moves_tracers_by_tau():
    mesh, ub = _box(L=20.0)
    u = ub.project(lambda x: np.stack([np.ones_like(x[0]), 0 * x[0], 0 * x[0]]))
    F = FastP2(mesh, ub, u)
    x0 = np.array([[2.0, 10.0, 10.0], [5.0, 4.0, 15.0]])
    out, _ = advect_pseudotime(F, caps=[], tau_needed=[0.0, 3.0, 7.5], C=0.0, rng=np.random.default_rng(0), x0=x0)
    for k, tau in enumerate((0.0, 3.0, 7.5)):
        ids, P = out[k]
        assert np.allclose(P, x0 + [tau, 0, 0], atol=1e-4)


def test_splat_conserves_spot_brightness_under_motion_blur():
    a, b = np.zeros((80, 80)), np.zeros((80, 80))
    args = lambda d: (np.array([40.0]), np.array([40.0]), np.array([0.0]), np.array([d]), np.array([1.2]), np.array([2.0]))
    splat(a, *args(0.0)); splat(b, *args(6.0))
    assert np.isclose(a.sum(), 2.0 * 2 * np.pi * 1.2 ** 2, rtol=1e-3)
    assert np.isclose(a.sum(), b.sum(), rtol=1e-3)                       # exposure blur spreads, does not add light
    assert b[40].argmax() == 40 and b[:, 40].max() < a[:, 40].max()


def test_background_statistics_and_crop(tmp_path):
    import tifffile
    rng = np.random.default_rng(1); gain = 25.0
    base = rng.uniform(40, 160, (150, 120))           # ≥ 50 px per brightness bin after ::3 subsampling
    files = []
    for i in range(220):
        f = tmp_path / f"f{i:04d}.tif"; tifffile.imwrite(f, np.clip(rng.poisson(base * gain) / gain, 0, 255).astype(np.uint8)); files.append(str(f))
    d = background_from_raw(files, nfs=(200,))
    B = Background(d["bg200"], d["sd200"], d["gain"], d["read"], "test", 1, 200)
    clip = np.stack([tifffile.imread(f) for f in files[:200]]).astype(np.float32)[:, 10:30, 5:25]
    bg, sd = B.crop(10, 30, 5, 25)
    assert np.array_equal(bg, np.percentile(clip, 10, axis=0)) and np.array_equal(sd, clip.std(0))
    assert isinstance(B.gain, np.float64)
    g, _ = tile_noise(tifffile.imread(f) for f in files[:200])
    assert 0.5 * gain < g < 2 * gain                                      # photon-transfer fit recovers the camera gain


def test_camera_contrast_scaling():
    rng = np.random.default_rng(2); bg = np.full((30, 30), 100.0, np.float32); mask = np.zeros((30, 30), bool); mask[10:20, 10:20] = True
    img = rng.uniform(0, 1, (300, 30, 30)) * mask
    s = scale_to_contrast(img, bg, np.float64(30.0), np.float64(0.0), 2.0, mask)
    vid = camera(img, bg, s, np.float64(30.0), np.float64(0.0), rng)
    noise = np.sqrt(100.0 / 30.0)
    assert np.isclose(np.median((s * img).std(0)[mask]), 2.0 * noise, rtol=1e-6)
    assert vid.dtype == np.float32 and 0 <= vid.min() and vid.max() <= 255
