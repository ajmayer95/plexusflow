import numpy as np
import pytest

from plexusflow.synthetic.synthetic_video import (
    _centerline_geometry,
    generate_vessel_video,
    make_centerline,
)


COMMON = dict(
    T=4,
    particle_density=0.0,
    v_mean=1.5,
    v_amplitude=0.0,
    v_harmonic=0.0,
    background=0.0,
    noise_sigma=0.0,
    poisson_noise=False,
    flicker_sigma=0.0,
    seed=2,
)


def test_explicit_radius_profile_conserves_q_along_vessel():
    radius = np.linspace(10.0, 6.0, 81)
    result = generate_vessel_video(
        W=80, vessel_radius_px=10.0, radius_profile=radius, **COMMON
    )
    assert result['R_of_s'][0] == pytest.approx(10.0)
    assert result['R_of_s'][-1] == pytest.approx(6.0)
    expected = np.broadcast_to(result['Q_t'][:, None], result['Q_st'].shape)
    np.testing.assert_allclose(result['Q_st'], expected, rtol=1e-12, atol=1e-12)


def test_callable_radius_profile_and_velocity_field_follow_curved_tangent():
    centerline = make_centerline(
        shape='sinusoid', length=50, vessel_radius_px=7,
        amplitude=5, n_periods=0.5,
    )
    result = generate_vessel_video(
        centerline=centerline,
        vessel_radius_px=7.0,
        radius_profile=lambda s: 7.0 - 1.5 * np.sin(np.pi * s / s[-1]) ** 2,
        return_velocity_field=True,
        **COMMON,
    )
    field = result['velocity_field_centerplane']
    assert field.shape == (COMMON['T'], *result['inside_mask'].shape, 2)
    assert np.all(field[:, ~result['inside_mask']] == 0)

    _, tangents, _ = _centerline_geometry(result['centerline'])
    mid = len(result['centerline']) // 2
    row, col = np.rint(result['centerline'][mid]).astype(int)
    vector = field[0, row, col]
    vector /= np.linalg.norm(vector)
    assert abs(float(np.dot(vector, tangents[mid]))) > 0.995


def test_radius_profile_is_interpolated_and_validated():
    result = generate_vessel_video(
        W=30, vessel_radius_px=6, radius_profile=[6, 4, 6], **COMMON
    )
    assert len(result['R_of_s']) == len(result['centerline'])
    assert np.min(result['R_of_s']) == pytest.approx(4.0, abs=0.02)

    with pytest.raises(ValueError, match='mutually exclusive'):
        generate_vessel_video(
            W=20, vessel_radius_px=5, radius_profile=[5, 4, 5],
            radius_variation=0.1, **COMMON,
        )
    with pytest.raises(ValueError, match='positive'):
        generate_vessel_video(
            W=20, vessel_radius_px=5, radius_profile=[5, 0, 5], **COMMON
        )
