import numpy as np
import pytest

from plexusflow.synthetic.pykitpiv_renderer import (
    DepthRenderConfig,
    _depth_edges,
    _validate_tracks,
)
from plexusflow.synthetic.synthetic_video import generate_vessel_video


def test_generator_can_return_renderer_neutral_particle_tracks():
    result = generate_vessel_video(
        T=5, W=20, vessel_radius_px=4, particle_density=2,
        v_mean=1, v_amplitude=0, v_harmonic=0,
        noise_sigma=0, poisson_noise=False, flicker_sigma=0,
        return_particle_tracks=True, seed=3,
    )
    tracks = result['particle_tracks']
    assert len(tracks) == 5
    assert all(frame.ndim == 2 and frame.shape[1] == 5 for frame in tracks)
    assert sum(map(len, tracks)) > 0


def test_track_validation_and_symmetric_depth_bins():
    tracks = [np.array([[2.0, 3.0, -2.0], [4.0, 5.0, 1.0]])]
    checked = _validate_tracks(tracks, (8, 9))
    edges = _depth_edges(checked, 4)
    np.testing.assert_allclose(edges, [-2, -1, 0, 1, 2])


def test_invalid_track_schema_is_rejected():
    with pytest.raises(ValueError, match='row, col, z'):
        _validate_tracks([np.ones((3, 2))], (8, 8))


def test_config_defaults_are_valid():
    cfg = DepthRenderConfig()
    assert cfg.n_depth_bins == 9
    assert cfg.bit_depth == 16
