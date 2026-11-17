"""Fan-beam filtered back-projection for an equiangular detector.

Implements Kak & Slaney (1988), Principles of Computerized Tomographic Imaging,
section 3.4.1 (equiangular rays). With ``D`` the source-isocentre distance,
``alpha`` the angular detector pitch and ``R_beta(gamma)`` the projections:

1. weight:   ``R'(n alpha) = R(n alpha) D cos(n alpha)``
2. filter:   ``Q(n alpha) = alpha * sum_m R'(m alpha) g((n - m) alpha)`` with
   ``g(n alpha) = 1/2 (n alpha / sin(n alpha))^2 h(n alpha)`` and the sampled ramp
   ``h(0) = 1 / (4 alpha^2)``, ``h(n) = 0`` (n even), ``-1 / (n pi alpha)^2`` (n odd)
3. back-project over the full rotation:
   ``f(x, y) = dbeta * sum_i Q_i(gamma') / L^2``, where ``L`` is the distance from
   the source of view ``i`` to the point and ``gamma'`` the fan angle of the ray
   through it.

The reconstruction kernel multiplies the ramp's frequency response by a window
(Kak & Slaney, section 3.3.3): none ("ramp"), a sinc ("shepp-logan"; Shepp &
Logan 1974, IEEE Trans. Nucl. Sci. 21:21-43) or a raised cosine ("hann").
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from neurosim.ct.config import DEFAULT_MAX_CHUNK_SAMPLES, FanBeamGeometry, Kernel


def _apodisation(kernel: Kernel, relative_frequency: np.ndarray) -> np.ndarray:
    """Window applied to the ramp, as a function of frequency / Nyquist frequency."""
    if kernel == "ramp":
        return np.ones_like(relative_frequency)
    if kernel == "shepp-logan":
        return np.sinc(0.5 * relative_frequency)
    if kernel == "hann":
        return 0.5 * (1.0 + np.cos(math.pi * relative_frequency))
    raise ValueError(f"unknown kernel {kernel!r}; expected 'ramp', 'shepp-logan' or 'hann'")


def filter_taps(geom: FanBeamGeometry, kernel: Kernel) -> np.ndarray:
    """Discrete convolution kernel ``g`` of the equiangular fan-beam filter.

    Args:
        geom: fan-beam geometry (number of detectors and angular pitch).
        kernel: apodisation of the ramp.

    Returns:
        Taps ``g(n alpha)`` for ``n = -(N - 1) .. (N - 1)`` [1/rad^2], shape (2 N - 1,).
    """
    alpha = geom.angular_pitch
    n = np.arange(-(geom.n_detectors - 1), geom.n_detectors)
    ramp = np.zeros(n.size)
    odd = n % 2 == 1
    ramp[odd] = -1.0 / (math.pi * alpha * n[odd]) ** 2
    ramp[n == 0] = 1.0 / (4.0 * alpha**2)
    if kernel != "ramp":
        # Window the band-limited ramp in the frequency domain; the padded
        # length keeps the periodic copies of the taps apart.
        padded = int(2 ** math.ceil(math.log2(4 * n.size)))
        response = np.fft.rfft(ramp, padded)
        relative_frequency = np.fft.rfftfreq(padded) / 0.5
        ramp = np.fft.irfft(response * _apodisation(kernel, relative_frequency), padded)[: n.size]
    angle = n * alpha
    geometric = np.ones(n.size)
    nonzero = n != 0
    geometric[nonzero] = (angle[nonzero] / np.sin(angle[nonzero])) ** 2
    return 0.5 * geometric * ramp


def filter_sinogram(sinogram: np.ndarray, geom: FanBeamGeometry, kernel: Kernel) -> np.ndarray:
    """Weight and filter fan-beam projections (steps 1 and 2 of the module docstring).

    Args:
        sinogram: line integrals [dimensionless], shape (n_views, n_detectors).
        geom: fan-beam geometry.
        kernel: reconstruction kernel.

    Returns:
        Filtered projections ``Q`` [m/rad], shape (n_views, n_detectors).
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    if sinogram.shape != (geom.n_views, geom.n_detectors):
        raise ValueError(
            f"sinogram shape {sinogram.shape} does not match the geometry "
            f"({geom.n_views}, {geom.n_detectors})"
        )
    weighted = sinogram * (geom.source_isocenter_distance * np.cos(geom.fan_angles()))
    taps = filter_taps(geom, kernel)
    return geom.angular_pitch * signal.fftconvolve(
        weighted, taps[np.newaxis, :], mode="same", axes=1
    )


def backproject(
    filtered: np.ndarray,
    geom: FanBeamGeometry,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_chunk_samples: int = DEFAULT_MAX_CHUNK_SAMPLES,
) -> np.ndarray:
    """Weighted back-projection of filtered projections onto arbitrary points (step 3).

    Args:
        filtered: filtered projections from `filter_sinogram`, shape (n_views, n_detectors).
        geom: fan-beam geometry.
        x, y: in-plane coordinates of the points relative to the isocentre [m], same shape.
        max_chunk_samples: largest number of (view, point) pairs handled at once.

    Returns:
        Reconstructed values at the points [1/m for dimensionless line integrals],
        float64, shape of ``x``. Rays that miss the detector contribute zero, so
        points outside ``geom.fov_radius`` are not reconstructed correctly.
    """
    x = np.asarray(x, dtype=np.float64)
    points_x = x.ravel()
    points_y = np.asarray(y, dtype=np.float64).ravel()
    source_distance = geom.source_isocenter_distance
    centre_index = 0.5 * (geom.n_detectors - 1)
    view_angles = geom.view_angles()
    result = np.zeros(points_x.size)
    views_per_chunk = max(1, max_chunk_samples // max(1, points_x.size))
    for start in range(0, geom.n_views, views_per_chunk):
        cos_beta = np.cos(view_angles[start : start + views_per_chunk, np.newaxis])
        sin_beta = np.sin(view_angles[start : start + views_per_chunk, np.newaxis])
        # Components of (point - source) along and across the central ray.
        along = source_distance - (points_x * cos_beta + points_y * sin_beta)
        across = points_x * sin_beta - points_y * cos_beta
        detector_index = np.arctan2(across, along) / geom.angular_pitch + centre_index
        lower = np.floor(detector_index).astype(np.intp)
        weight = detector_index - lower
        inside = (lower >= 0) & (lower < geom.n_detectors - 1)
        lower = np.clip(lower, 0, geom.n_detectors - 2)
        rows = np.arange(cos_beta.shape[0])[:, np.newaxis]
        block = filtered[start : start + views_per_chunk]
        value = (1.0 - weight) * block[rows, lower] + weight * block[rows, lower + 1]
        result += np.sum(np.where(inside, value, 0.0) / (along**2 + across**2), axis=0)
    return (2.0 * math.pi / geom.n_views * result).reshape(x.shape)


def pixel_coordinates(
    shape: tuple[int, int], grid_spacing: float, center_offset: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """In-plane coordinates of the pixel centres of a slice grid.

    Args:
        shape: number of pixels (nx, ny).
        grid_spacing: pixel size [m].
        center_offset: (x, y) of the grid centre relative to the isocentre [m].

    Returns:
        x, y [m] relative to the isocentre, each of shape (nx, ny), indexed [ix, iy].
    """
    x = center_offset[0] + (np.arange(shape[0]) - 0.5 * (shape[0] - 1)) * grid_spacing
    y = center_offset[1] + (np.arange(shape[1]) - 0.5 * (shape[1] - 1)) * grid_spacing
    return np.meshgrid(x, y, indexing="ij")


def fbp_fan_beam(
    sinogram: np.ndarray,
    geom: FanBeamGeometry,
    kernel: Kernel,
    out_shape: tuple[int, int],
    grid_spacing: float,
    *,
    center_offset: tuple[float, float] = (0.0, 0.0),
    max_chunk_samples: int = DEFAULT_MAX_CHUNK_SAMPLES,
) -> np.ndarray:
    """Reconstruct one slice from fan-beam projections by filtered back-projection.

    Args:
        sinogram: line integrals of the attenuation coefficient [dimensionless],
            shape (n_views, n_detectors), views over a full rotation.
        geom: fan-beam geometry.
        kernel: reconstruction kernel, "ramp", "shepp-logan" or "hann".
        out_shape: number of pixels (nx, ny) of the slice grid.
        grid_spacing: pixel size [m].
        center_offset: (x, y) of the grid centre relative to the isocentre [m].
        max_chunk_samples: largest number of (view, pixel) pairs handled at once.

    Returns:
        Linear attenuation coefficient mu [1/m], float64, shape (nx, ny), indexed
        [ix, iy]. Only pixels within ``geom.fov_radius`` of the isocentre are valid.
    """
    x, y = pixel_coordinates(out_shape, grid_spacing, center_offset)
    filtered = filter_sinogram(sinogram, geom, kernel)
    return backproject(filtered, geom, x, y, max_chunk_samples=max_chunk_samples)
