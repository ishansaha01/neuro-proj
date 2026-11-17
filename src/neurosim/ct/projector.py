"""Ray-driven fan-beam projector for one axial slice.

Each ray is sampled at equal steps and the image is interpolated bilinearly at
the sample points (the "sampled interpolation" projector; Kak & Slaney 1988,
section 7.4 discusses ray sums by interpolation). All rays of a block of views
are processed together with numpy; there is no Python loop over rays.

In-plane coordinates are head-frame (x, y) relative to the isocentre. The source
of view ``i`` is at ``R (cos beta_i, sin beta_i)`` and the ray to the detector
element at fan angle ``gamma`` leaves it along ``-(cos(beta_i + gamma),
sin(beta_i + gamma))``; ``gamma = 0`` is the central ray through the isocentre.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from neurosim.ct.config import (
    DEFAULT_MAX_CHUNK_SAMPLES,
    DEFAULT_RAY_STEP_FRACTION,
    FanBeamGeometry,
)


def enclosing_radius(
    shape: tuple[int, int], grid_spacing: float, center_offset: tuple[float, float]
) -> float:
    """Radius about the isocentre of a circle that contains the whole pixel grid [m].

    Args:
        shape: number of pixels (nx, ny).
        grid_spacing: pixel size [m].
        center_offset: (x, y) of the grid centre relative to the isocentre [m].
    """
    half_diagonal = 0.5 * grid_spacing * math.hypot(shape[0], shape[1])
    return math.hypot(*center_offset) + half_diagonal


def samples_per_ray(
    shape: tuple[int, int],
    grid_spacing: float,
    center_offset: tuple[float, float],
    ray_step_fraction: float,
) -> int:
    """Number of sample points on each ray for a grid of the given shape."""
    radius = enclosing_radius(shape, grid_spacing, center_offset)
    return max(1, math.ceil(2.0 * radius / (ray_step_fraction * grid_spacing)))


def fan_beam_project(
    image: np.ndarray,
    grid_spacing: float,
    geom: FanBeamGeometry,
    *,
    center_offset: tuple[float, float] = (0.0, 0.0),
    ray_step_fraction: float = DEFAULT_RAY_STEP_FRACTION,
    max_chunk_samples: int = DEFAULT_MAX_CHUNK_SAMPLES,
) -> np.ndarray:
    """Line integrals of one slice along every ray of a fan-beam scan.

    The image is zero outside its grid. Values between pixel centres are
    interpolated bilinearly, and the outermost pixels fade to zero over one
    pixel beyond their centres.

    Args:
        image: slice indexed [ix, iy], shape (nx, ny), or a stack of slices
            (n, nx, ny) sharing the grid (for example one per basis material).
        grid_spacing: pixel size [m].
        geom: fan-beam geometry.
        center_offset: (x, y) of the grid centre relative to the isocentre [m].
        ray_step_fraction: sampling step along a ray in units of the pixel size.
        max_chunk_samples: largest number of sample points handled at once.

    Returns:
        Line integrals [m x image units], float64, shape (n_views, n_detectors),
        or (n, n_views, n_detectors) for a stack.
    """
    stack = np.asarray(image, dtype=np.float64)
    if stack.ndim not in (2, 3):
        raise ValueError(f"image must have shape (nx, ny) or (n, nx, ny), got {stack.shape}")
    single = stack.ndim == 2
    if single:
        stack = stack[np.newaxis]
    nx, ny = stack.shape[1:]

    source_distance = geom.source_isocenter_distance
    radius = enclosing_radius((nx, ny), grid_spacing, center_offset)
    n_samples = samples_per_ray((nx, ny), grid_spacing, center_offset, ray_step_fraction)
    step = 2.0 * radius / n_samples
    # Distance from the source of each sample (midpoint rule), shape (n_samples,).
    distance = source_distance - radius + (np.arange(n_samples) + 0.5) * step

    view_angles = geom.view_angles()
    fan_angles = geom.fan_angles()
    # World (x, y) -> fractional pixel index: index = (coordinate - first_centre) / spacing.
    first_x = center_offset[0] - 0.5 * (nx - 1) * grid_spacing
    first_y = center_offset[1] - 0.5 * (ny - 1) * grid_spacing

    sinogram = np.empty((stack.shape[0], geom.n_views, geom.n_detectors))
    views_per_chunk = max(1, max_chunk_samples // (geom.n_detectors * n_samples))
    for start in range(0, geom.n_views, views_per_chunk):
        beta = view_angles[start : start + views_per_chunk, np.newaxis]
        ray_angle = beta + fan_angles[np.newaxis, :]
        # Fractional pixel index of every sample, shape (2, views, detectors, samples):
        # the source's index minus the ray direction times the distance, in pixels.
        index = np.empty((2, *ray_angle.shape, n_samples))
        for axis, first, trig in ((0, first_x, np.cos), (1, first_y, np.sin)):
            source_index = (source_distance * trig(beta) - first) / grid_spacing
            np.multiply.outer(trig(ray_angle) / grid_spacing, distance, out=index[axis])
            np.subtract(source_index[..., np.newaxis], index[axis], out=index[axis])
        for material, slice_image in enumerate(stack):
            values = ndimage.map_coordinates(
                slice_image, index, order=1, mode="constant", cval=0.0
            )
            sinogram[material, start : start + views_per_chunk] = values.sum(axis=-1) * step
    return sinogram[0] if single else sinogram
