"""Depth-resolved pykitPIV renderer for synthetic vessel trajectories.

PlexusFlow owns the vessel geometry, particle advection, and ground truth.
pykitPIV owns the Gaussian particle-image model and Fourier-optics PSF.  The
adapter groups particles by depth, renders each plane, convolves it with the
corresponding pykitPIV defocus PSF, and sums photon images before adding noise.

pykitPIV is deliberately an optional dependency.  Its source distribution is
currently installed from GitHub rather than PyPI::

    pip install "git+https://github.com/kamilazdybal/pykitPIV.git"

The installed pykitPIV version may also require a recent SciPy release.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DepthRenderConfig:
    """Minimal optical and detector settings for depth compositing."""

    n_depth_bins: int = 9
    particle_diameter_px: float = 2.0
    peak_photons: float = 500.0
    focal_z_px: float = 0.0
    defocus_waves_per_px: float = 0.15
    detection_sigma_z_px: Optional[float] = None
    psf_grid_size: int = 64
    pupil_radius: float = 18.0
    spherical_waves: float = 0.0
    background_photons: float = 20.0
    read_noise_sigma: float = 2.0
    bit_depth: int = 16
    shot_noise: bool = True
    seed: Optional[int] = 42


def _import_pykitpiv():
    # pykitPIV imports ``sph_harm_y`` at package import time even though the
    # renderer does not use spherical-harmonic flow fields.  That name is newer
    # than the last SciPy versions supporting Python 3.9.  Supplying the legacy
    # function lets this isolated rendering path run; users of pykitPIV's
    # spherical-harmonic flow generator should use Python >= 3.10 and a current
    # SciPy instead.
    try:
        from scipy import special
        if not hasattr(special, 'sph_harm_y') and hasattr(special, 'sph_harm'):
            special.sph_harm_y = special.sph_harm
    except Exception:
        pass
    try:
        from pykitPIV import Image, Particle, Postprocess
    except ModuleNotFoundError as exc:
        # pykitPIV 1.0 imports its ML module (and therefore torch) eagerly from
        # package __init__, although Particle/Image and the Fourier-PSF method
        # used here do not require torch. Load only that dependency-light
        # subset without modifying the installed package.
        if exc.name != 'torch':
            raise
        import importlib.util
        import sys
        import types
        from pathlib import Path

        spec = importlib.util.find_spec('pykitPIV')
        if spec is None or spec.origin is None:
            raise ImportError('pykitPIV is not installed') from exc
        package_dir = Path(spec.origin).parent
        package = types.ModuleType('pykitPIV')
        package.__path__ = [str(package_dir)]
        package.__package__ = 'pykitPIV'
        sys.modules['pykitPIV'] = package

        def load_module(name):
            module_name = f'pykitPIV.{name}'
            module_spec = importlib.util.spec_from_file_location(
                module_name, package_dir / f'{name}.py')
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
            return module

        load_module('checks')
        particle_module = load_module('particle')
        load_module('motion')
        load_module('flowfield')
        image_module = load_module('image')

        # postprocess imports torch at module scope but add_defocus does not use
        # it. A placeholder is sufficient for defining that class; torch-based
        # methods remain deliberately unavailable through this adapter.
        sys.modules.setdefault('torch', types.ModuleType('torch'))
        postprocess_module = load_module('postprocess')
        Particle = particle_module.Particle
        Image = image_module.Image
        Postprocess = postprocess_module.Postprocess
    except Exception as exc:  # optional dependency and binary/API compatibility
        raise ImportError(
            "The pykitPIV renderer requires a compatible pykitPIV installation."
        ) from exc
    return Particle, Image, Postprocess


def _validate_tracks(
    particle_tracks: Sequence[np.ndarray], shape: Tuple[int, int]
) -> List[np.ndarray]:
    H, W = map(int, shape)
    if H <= 0 or W <= 0:
        raise ValueError("shape must contain positive (height, width)")
    tracks = []
    for frame in particle_tracks:
        arr = np.asarray(frame, dtype=float)
        if arr.size == 0:
            arr = np.empty((0, 5), dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                "each particle_tracks entry must have columns [row, col, z] "
                "and may additionally contain intensity and diameter"
            )
        tracks.append(arr)
    if not tracks:
        raise ValueError("particle_tracks must contain at least one frame")
    return tracks


def _depth_edges(tracks: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_depth_bins must be >= 1")
    z_nonempty = [frame[:, 2] for frame in tracks if len(frame)]
    if not z_nonempty:
        return np.linspace(-0.5, 0.5, n_bins + 1)
    z = np.concatenate(z_nonempty)
    zmax = max(float(np.max(np.abs(z))), 0.5)
    return np.linspace(-zmax, zmax, n_bins + 1)


def _pykit_psf(Postprocess, shape, cfg, defocus_waves):
    """Ask pykitPIV to construct its pupil-derived PSF kernel."""
    H, W = shape
    probe = np.zeros((1, H, W), dtype=np.float32)
    probe[0, H // 2, W // 2] = 1.0
    pp = Postprocess(probe, random_seed=cfg.seed)
    kernel = pp.add_defocus(
        grid_size=int(cfg.psf_grid_size),
        pupil_radius=float(cfg.pupil_radius),
        defocus_waves=float(defocus_waves),
        spherical_waves=float(cfg.spherical_waves),
    )
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel /= max(float(kernel.sum()), np.finfo(float).eps)
    return kernel


def render_particle_tracks_pykitpiv(
    particle_tracks: Sequence[np.ndarray],
    shape: Tuple[int, int],
    config: Optional[DepthRenderConfig] = None,
) -> Dict[str, np.ndarray]:
    """Render persistent 3-D particle tracks using depth-resolved pykitPIV PSFs.

    Parameters
    ----------
    particle_tracks:
        One array per frame. Required columns are ``row, col, z``; optional
        columns are relative intensity and apparent diameter.  Tracks returned
        by :func:`generate_vessel_video(return_particle_tracks=True)` match this
        schema directly.
    shape:
        Output image ``(height, width)``.
    config:
        Optical/detector settings.

    Returns
    -------
    dict
        ``frames`` is a ``(T,H,W)`` integer camera stack. ``photon_expectation``
        is the noiseless floating-point stack and ``depth_centers_px`` records
        the planes used in the approximation.
    """
    cfg = config or DepthRenderConfig()
    tracks = _validate_tracks(particle_tracks, shape)
    Particle, Image, Postprocess = _import_pykitpiv()
    from scipy.signal import fftconvolve

    T, (H, W) = len(tracks), tuple(map(int, shape))
    edges = _depth_edges(tracks, int(cfg.n_depth_bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected = np.zeros((T, H, W), dtype=np.float64)

    for b, z_center in enumerate(centers):
        coords = []
        bin_scales = []
        for frame in tracks:
            # Include the upper edge in the final bin.
            in_bin = ((frame[:, 2] >= edges[b]) &
                      (frame[:, 2] < edges[b + 1]))
            if b == len(centers) - 1:
                in_bin |= frame[:, 2] == edges[b + 1]
            selected = frame[in_bin]
            coords.append((selected[:, 0], selected[:, 1]))
            bin_scales.append(float(np.mean(selected[:, 3]))
                              if len(selected) and selected.shape[1] > 3 else 1.0)

        particles = Particle(
            T, size=(H, W), size_buffer=0,
            diameters=float(cfg.particle_diameter_px), diameter_std=0.0,
            seeding_mode='user', dtype=np.float32, random_seed=cfg.seed,
        )
        particles.upload_particle_coordinates(coords)
        # pykitPIV 1.0 does not initialize this public attribute for
        # ``seeding_mode='user'`` although add_reflected_light accesses it even
        # when laser-plane weighting is disabled (upstream issue).  Depth is
        # handled explicitly by our bin/PSF compositor, so zeros are correct.
        particles.z_coordinates = [
            np.zeros(len(rc[0]), dtype=np.float32) for rc in coords
        ]
        image = Image(dtype=np.float32, random_seed=cfg.seed)
        image.add_particles(particles)
        image.add_reflected_light(
            exposures=1.0, maximum_intensity=int(round(cfg.peak_photons)),
            no_laser_plane=True, clip_intensities=False,
        )
        sharp = np.asarray(image.images_I1[:, 0], dtype=np.float64)

        dz = float(z_center - cfg.focal_z_px)
        psf = _pykit_psf(
            Postprocess, (H, W), cfg, dz * cfg.defocus_waves_per_px
        )
        if cfg.detection_sigma_z_px is None:
            detection_weight = 1.0
        else:
            sigma = max(float(cfg.detection_sigma_z_px), 1e-9)
            detection_weight = float(np.exp(-0.5 * (dz / sigma) ** 2))
        for t in range(T):
            if np.any(sharp[t]):
                expected[t] += (
                    detection_weight * bin_scales[t]
                    * fftconvolve(sharp[t], psf, mode='same')
                )

    expected += float(cfg.background_photons)
    np.clip(expected, 0.0, None, out=expected)
    rng = np.random.default_rng(cfg.seed)
    frames = rng.poisson(expected).astype(np.float64) if cfg.shot_noise else expected.copy()
    if cfg.read_noise_sigma > 0:
        frames += rng.normal(0.0, cfg.read_noise_sigma, frames.shape)
    max_count = float(2 ** int(cfg.bit_depth) - 1)
    frames = np.clip(np.rint(frames), 0, max_count)
    dtype = np.uint8 if cfg.bit_depth <= 8 else np.uint16
    return {
        'frames': frames.astype(dtype),
        'photon_expectation': expected,
        'depth_centers_px': centers,
        'depth_edges_px': edges,
    }
