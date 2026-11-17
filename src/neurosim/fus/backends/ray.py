"""Straight-ray backend: travel time, absorption and interface loss along element-to-point rays.

Model. For the straight ray from an element centre to a field point, of length L,

    B = S * T * exp(-A(f)) * exp(+i w (tau - L / c_ref))

    tau  = integral ds / c(s)                       travel time [s]
    A(f) = integral alpha(s, f) ds                  absorption [Np], power law alpha0 f^y
    T    = product over successive samples of 2 sqrt(Z1 Z2) / (Z1 + Z2),   Z = rho c
    S    = sqrt( rho_r c_e c_r / (rho_e c_ref c_mean) )

T is the pressure transmission coefficient `2 Z2 / (Z1 + Z2)` of every impedance
step along the ray, multiplied by `sqrt(Z1 / Z2)` so that it measures the
transmitted energy flux: the product of the plain pressure coefficients along a
smooth impedance ramp tends to `sqrt(Z_end / Z_start)`, which is no loss at all
(intensity p^2 / Z is conserved), whereas T tends to 1. A sharp step costs
`2 sqrt(Z1 Z2) / (Z1 + Z2)`; a ray that enters and leaves a slab costs
`4 Z1 Z2 / (Z1 + Z2)^2` (the same as the product of the two pressure
coefficients); an impedance-matched slab costs nothing. T is symmetric under
exchange of the two ends of the ray.

S turns the flux-normalised amplitude back into pressure for the source
normalisation of the bank (package docstring). `sqrt(rho_r / rho_e)` and
`sqrt(c_e c_r / (c_ref c_mean))`, with `c_mean` the path average of the sound
speed, are the far-field amplitude of a 2D line source seen through a layered
medium near normal incidence (the ray tube widens in proportion to the integral
of c along the path). S = 1 when both ends are in the reference medium and the
path-mean speed is c_ref; in a homogeneous medium (c1, rho1) the bank reduces
to `sqrt(c1 / c_ref) exp(i (k1 - k_ref) L)`, the far-field limit of the exact
ratio `H0(k1 L) / H0(k_ref L)`.

Approximations: straight rays (no refraction, so travel times are first-order
accurate in the sound-speed contrast and amplitudes ignore focusing off the
layer normal), normal-incidence interface coefficients, no multiple reflections
(no reverberation, `static_fmc` is None), no mode conversion, point-like
elements (the element directivity is that of the reference medium).

Sampling. Rays are sampled at the midpoints of equal segments no longer than
`RayConfig.step_fraction` grid cells. The integrands (slowness, alpha0) are read
with bilinear interpolation, so that tau and A vary continuously with the ray
end points instead of jumping as samples change cell. The impedance is read
with nearest-neighbour lookup, so that a sharp interface in the medium grid
stays one step whatever the sampling density; interpolating it would spread the
step over several samples and remove most of its loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurosim.common.config import to_dict
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D, sample_plane_map
from neurosim.common.units import db_cm_mhz_to_np_m
from neurosim.fus.backends.config import RayConfig
from neurosim.fus.backends.geometry import (
    check_geometry,
    check_medium,
    checked_frequencies,
    element_centres,
    inside_grid,
)
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

# Working memory per ray sample [bytes]: plane coordinates, fractional grid indices and the
# sampled slowness, alpha0, pixel number, sound speed and density, plus temporaries.
_BYTES_PER_SAMPLE = 96


@dataclass(frozen=True)
class RayIntegrals:
    """Frequency-independent quantities of every element-to-point ray.

    Every array has shape (n_elements, n_points).

    Attributes:
        path_length: ray length L [m].
        travel_time: integral of ds / c [s].
        alpha0_path: integral of the absorption prefactor alpha0 ds [dB/(cm MHz^y) * m].
        transmission: flux-normalised interface transmission T [dimensionless, <= 1].
        spreading: end-point and spreading factor S [dimensionless].
    """

    path_length: np.ndarray
    travel_time: np.ndarray
    alpha0_path: np.ndarray
    transmission: np.ndarray
    spreading: np.ndarray


def _integrate_rays(
    integrand_maps: np.ndarray,
    pixel_number: np.ndarray,
    sound_speed_of_pixel: np.ndarray,
    density_of_pixel: np.ndarray,
    grid: Grid2D,
    start: np.ndarray,
    ends: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate along the rays from one start point to several end points.

    Args:
        integrand_maps: slowness [s/m] and alpha0 [dB/(cm MHz^y)] on `grid`, shape (2, nz, nx).
        pixel_number: index of every pixel in the flattened grid, shape (nz, nx).
        sound_speed_of_pixel: sound speed [m/s] by pixel number, shape (nz * nx,).
        density_of_pixel: density [kg/m^3] by pixel number, shape (nz * nx,).
        grid: the medium grid.
        start: ray start (x, z) [m], shape (2,).
        ends: ray ends (x, z) [m], shape (n, 2).
        n_segments: equal segments per ray.

    Returns:
        mean slowness [s/m], mean alpha0 [dB/(cm MHz^y)], transmission T and spreading
        factor S (both dimensionless), each of shape (n,).
    """
    midpoints = (np.arange(n_segments) + 0.5) / n_segments
    fractions = np.concatenate(([0.0], midpoints, [1.0]))
    samples = start + fractions[None, :, None] * (ends - start)[:, None, :]

    slowness, alpha0 = sample_plane_map(integrand_maps, grid, samples[:, 1:-1], order=1)
    # One nearest-neighbour lookup of the pixel number serves both local maps.
    pixel = sample_plane_map(pixel_number, grid, samples, order=0)
    sound_speed = sound_speed_of_pixel[pixel]
    density = density_of_pixel[pixel]

    impedance = sound_speed * density
    before, after = impedance[:, :-1], impedance[:, 1:]
    transmission = np.prod(2.0 * np.sqrt(before * after) / (before + after), axis=1)
    mean_speed = sound_speed[:, 1:-1].mean(axis=1)
    spreading = np.sqrt(
        (density[:, -1] * sound_speed[:, 0] * sound_speed[:, -1])
        / (density[:, 0] * C_REF * mean_speed)
    )
    return slowness.mean(axis=1), alpha0.mean(axis=1), transmission, spreading


def trace_rays(
    medium: PlaneMedium, transducer: Transducer, points_xz: np.ndarray, cfg: RayConfig
) -> RayIntegrals:
    """Trace the straight ray from every element centre to every point.

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3], attenuation
            [dB/(cm MHz^y)] of shape (nz, nx).
        transducer: the array (element centres at z = 0).
        points_xz: field points (x, z) [m], shape (n_points, 2), inside the medium grid.
        cfg: ray sampling settings.

    Returns:
        `RayIntegrals` with arrays of shape (n_elements, n_points).
    """
    check_medium(medium)
    grid = medium.grid
    points = np.asarray(points_xz, dtype=np.float64)
    starts = element_centres(transducer)
    if not (np.all(inside_grid(grid, points)) and np.all(inside_grid(grid, starts))):
        raise ValueError(
            "every element centre and every field point must lie inside the medium grid"
        )
    integrand_maps = np.stack(
        (1.0 / medium.sound_speed.astype(np.float64), medium.attenuation.astype(np.float64))
    )
    pixel_number = np.arange(grid.shape[0] * grid.shape[1]).reshape(grid.shape)
    sound_speed_of_pixel = medium.sound_speed.astype(np.float64).ravel()
    density_of_pixel = medium.density.astype(np.float64).ravel()
    step = cfg.step_fraction * grid.spacing
    shape = (starts.shape[0], points.shape[0])
    path_length = np.hypot(
        points[None, :, 0] - starts[:, None, 0], points[None, :, 1] - starts[:, None, 1]
    )
    mean_slowness, mean_alpha0, transmission, spreading = (np.empty(shape) for _ in range(4))

    longest = max(1, math.ceil(path_length.max() / step))
    rays_per_chunk = max(1, cfg.max_chunk_bytes // (_BYTES_PER_SAMPLE * (longest + 2)))
    for element, start in enumerate(starts):
        # Short rays first: a chunk is sampled no more finely than its longest ray needs.
        order = np.argsort(path_length[element])
        for first in range(0, order.size, rays_per_chunk):
            chunk = order[first : first + rays_per_chunk]
            n_segments = max(1, math.ceil(path_length[element, chunk].max() / step))
            (
                mean_slowness[element, chunk],
                mean_alpha0[element, chunk],
                transmission[element, chunk],
                spreading[element, chunk],
            ) = _integrate_rays(
                integrand_maps,
                pixel_number,
                sound_speed_of_pixel,
                density_of_pixel,
                grid,
                start,
                points[chunk],
                n_segments,
            )

    return RayIntegrals(
        path_length=path_length,
        travel_time=path_length * mean_slowness,
        alpha0_path=path_length * mean_alpha0,
        transmission=transmission,
        spreading=spreading,
    )


def ray_travel_times(
    medium: PlaneMedium,
    transducer: Transducer,
    points_xz: np.ndarray,
    cfg: RayConfig = RayConfig(),
) -> np.ndarray:
    """One-way straight-ray travel time from every element centre to every point.

    Args:
        medium: plane medium (sound_speed [m/s] on its grid; z = 0 is the array face).
        transducer: the array.
        points_xz: field points (x, z) [m], shape (..., 2), inside the medium grid.
        cfg: ray sampling settings.

    Returns:
        travel times [s], float64, shape (n_elements, ...).
    """
    points = np.asarray(points_xz, dtype=np.float64)
    integrals = trace_rays(medium, transducer, points.reshape(-1, 2), cfg)
    return integrals.travel_time.reshape(transducer.n_elements, *points.shape[:-1])


def ray_bank(
    medium: PlaneMedium, transducer: Transducer, roi: Grid2D, freqs: np.ndarray, cfg: RayConfig
) -> GreensBank:
    """Green's bank of the straight-ray model (module docstring).

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3], attenuation
            [dB/(cm MHz^y)] of shape (nz, nx), and the power-law exponent alpha_power.
        transducer: the array.
        roi: grid the bank is stored on, inside the medium grid, z > 0.
        freqs: frequencies [Hz], shape (nf,), ascending and uniformly spaced.
        cfg: ray sampling settings.

    Returns:
        `GreensBank` with B complex64 of shape (nf, n_elements, nz_roi, nx_roi),
        `c_ref = C_REF` and `static_fmc = None`.
    """
    freqs = checked_frequencies(freqs)
    check_medium(medium)
    check_geometry(medium, transducer, roi)
    integrals = trace_rays(medium, transducer, roi.points().reshape(-1, 2), cfg)

    excess_delay = integrals.travel_time - integrals.path_length / C_REF  # [s]
    amplitude = integrals.transmission * integrals.spreading
    # Absorption [Np/m] at each frequency per unit alpha0; the power law is linear in alpha0.
    nepers_per_alpha0 = db_cm_mhz_to_np_m(1.0, freqs, medium.alpha_power)

    n_elements, n_points = amplitude.shape
    bank = np.empty((freqs.size, n_elements, n_points), dtype=np.complex64)
    for element in range(n_elements):
        exponent = (
            2j * np.pi * freqs[:, None] * excess_delay[element]
            - nepers_per_alpha0[:, None] * integrals.alpha0_path[element]
        )
        bank[:, element] = amplitude[element] * np.exp(exponent)

    return GreensBank(
        freqs=freqs,
        roi=roi,
        B=bank.reshape(freqs.size, n_elements, *roi.shape),
        c_ref=C_REF,
        transducer=transducer,
        backend="ray",
        static_fmc=None,
        meta={"config": to_dict(cfg), "ray_step_m": cfg.step_fraction * medium.grid.spacing},
    )


def estimate_ray_cost(
    medium: PlaneMedium, transducer: Transducer, roi: Grid2D, freqs: np.ndarray, cfg: RayConfig
) -> dict[str, float | int]:
    """Cost of `ray_bank`: no simulation, so `time_steps` and `n_simulations` are zero.

    Args: as for `ray_bank`.

    Returns:
        dict with `grid_points` (the medium grid), `time_steps`, `n_simulations`,
        `ray_samples` (medium lookups along all rays), `memory_bytes` (the bank plus one
        chunk of ray samples) and `run_time_s` (single-core estimate).
    """
    points = roi.points().reshape(-1, 2)
    starts = element_centres(transducer)
    path_length = np.hypot(
        points[None, :, 0] - starts[:, None, 0], points[None, :, 1] - starts[:, None, 1]
    )
    step = cfg.step_fraction * medium.grid.spacing
    ray_samples = int(np.sum(np.ceil(path_length / step) + 2))
    bank_values = np.asarray(freqs).size * path_length.size
    return {
        "grid_points": medium.grid.shape[0] * medium.grid.shape[1],
        "time_steps": 0,
        "n_simulations": 0,
        "ray_samples": ray_samples,
        "memory_bytes": 8 * bank_values + cfg.max_chunk_bytes + 5 * 8 * path_length.size,
        "run_time_s": cfg.seconds_per_sample * ray_samples
        + cfg.seconds_per_bank_value * bank_values,
    }
