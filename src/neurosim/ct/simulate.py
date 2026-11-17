"""CT scan of a head: basis-material fractions in, Hounsfield units out.

Per axial slice (a plane of constant ``iz``): project each basis material
(line integrals do not depend on energy), form the polyenergetic detector
signal, add photon noise, take the log, optionally apply the water
beam-hardening correction, reconstruct by fan-beam filtered back-projection on
the slice's own pixel grid and convert to HU with the water/air calibration of
the same scan settings. Slices are infinitely thin: there is no averaging along z.

The isocentre is on the head-frame z axis (x = y = 0).
"""

from __future__ import annotations

import numpy as np

from neurosim.common.config import to_dict
from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.common.units import hu_from_mu
from neurosim.ct.attenuation import basis_attenuation
from neurosim.ct.calibration import hu_calibration, process_signal
from neurosim.ct.config import CtConfig
from neurosim.ct.detector import detector_signal, flat_field_counts
from neurosim.ct.projector import fan_beam_project, samples_per_ray
from neurosim.ct.recon import fbp_fan_beam, pixel_coordinates
from neurosim.ct.types import CtVolume
from neurosim.head.types import HeadVolume

_AIR = BASIS_MATERIALS.index("air")

# Cost model of `estimate_cost`: seconds per elementary operation, measured
# single-threaded on the 2-core development machine (2026-11-17) and rounded up
# by about 20 %. Other machines differ; treat the time as an order of magnitude.
_SECONDS_PER_RAY_SAMPLE_COORDINATES = 3.0e-8  # position of one ray sample
_SECONDS_PER_RAY_SAMPLE_MATERIAL = 3.0e-8  # bilinear interpolation of one material at it
_SECONDS_PER_BACKPROJECTION = 5.0e-8  # one (view, pixel) pair
_SECONDS_PER_COUNT_BIN = 6.0e-8  # one (ray, energy bin): exponential and Poisson draw
_SECONDS_FIXED = 2.0  # spectrum model and calibration, once per configuration
# Upper bound on the float64-sized temporaries alive per chunk element (checked
# by tests/unit/ct/test_simulate.py::test_estimated_memory_bounds_the_run).
_ARRAYS_PER_CHUNK_ELEMENT = 12
_BYTES_PER_FLOAT64 = 8
_BYTES_PER_FLOAT32 = 4


def slice_indices(grid: Grid3D, cfg: CtConfig) -> range:
    """Axial slices (iz) scanned by a configuration on a grid.

    Returns the contiguous range of slices whose centres lie in
    ``cfg.slab_z_range`` (all slices if it is None). Raises ``ValueError`` if
    the slab contains no slice.
    """
    if cfg.slab_z_range is None:
        return range(grid.shape[2])
    z = grid.axes()[2]
    low, high = cfg.slab_z_range
    inside = np.flatnonzero((z >= low) & (z <= high))
    if inside.size == 0:
        raise ValueError(
            f"slab_z_range {cfg.slab_z_range} m contains no slice of the grid "
            f"(slice centres from {z[0]:.4f} to {z[-1]:.4f} m)"
        )
    return range(int(inside[0]), int(inside[-1]) + 1)


def _center_offset(grid: Grid3D) -> tuple[float, float]:
    """(x, y) of the centre of the slice grid relative to the isocentre [m]."""
    return (
        grid.origin[0] + 0.5 * (grid.shape[0] - 1) * grid.spacing,
        grid.origin[1] + 0.5 * (grid.shape[1] - 1) * grid.spacing,
    )


def estimate_cost(grid: Grid3D, cfg: CtConfig) -> dict[str, float]:
    """Estimate the size and cost of `simulate_ct_fractions` before running it.

    The estimate assumes all four basis materials are present in every slice
    (materials absent from a slice are skipped in the run) and photon noise on.
    The time is an upper estimate: ray samples that fall outside the grid cost
    less than the measured average, so a small or off-centre grid runs up to
    about three times faster than estimated.

    Returns a dict with:
        n_slices: slices scanned.
        rays: source-to-detector rays over all slices.
        samples: interpolated ray samples over all slices and materials.
        memory_bytes: estimated peak memory of the run, excluding the input volume.
        time_s: estimated single-threaded run time on the development machine [s].
    """
    geom = cfg.geometry
    n_slices = len(slice_indices(grid, cfg))
    nx, ny = grid.shape[:2]
    n_materials = len(BASIS_MATERIALS)
    n_energies = flat_field_counts(cfg)[0].size
    rays_per_slice = geom.n_views * geom.n_detectors
    per_ray = samples_per_ray((nx, ny), grid.spacing, _center_offset(grid), cfg.ray_step_fraction)
    ray_samples_per_slice = rays_per_slice * per_ray
    samples_per_slice = ray_samples_per_slice * n_materials
    backprojections_per_slice = geom.n_views * nx * ny
    count_bins_per_slice = rays_per_slice * n_energies

    # A chunk never splits a view, so one view's worth is the floor of the chunk size.
    chunk_elements = max(
        cfg.max_chunk_samples, geom.n_detectors * per_ray, nx * ny, geom.n_detectors * n_energies
    )
    sinogram_bytes = (n_materials + 3) * rays_per_slice * _BYTES_PER_FLOAT64
    slice_bytes = (n_materials + 4) * nx * ny * _BYTES_PER_FLOAT64
    output_bytes = n_slices * nx * ny * _BYTES_PER_FLOAT32
    memory_bytes = (
        _ARRAYS_PER_CHUNK_ELEMENT * chunk_elements * _BYTES_PER_FLOAT64
        + sinogram_bytes
        + slice_bytes
        + output_bytes
    )
    time_s = _SECONDS_FIXED + n_slices * (
        ray_samples_per_slice * _SECONDS_PER_RAY_SAMPLE_COORDINATES
        + samples_per_slice * _SECONDS_PER_RAY_SAMPLE_MATERIAL
        + backprojections_per_slice * _SECONDS_PER_BACKPROJECTION
        + count_bins_per_slice * _SECONDS_PER_COUNT_BIN
    )
    return {
        "n_slices": float(n_slices),
        "rays": float(n_slices * rays_per_slice),
        "samples": float(n_slices * samples_per_slice),
        "memory_bytes": float(memory_bytes),
        "time_s": float(time_s),
    }


def _measure_slice(
    path_lengths: np.ndarray,
    attenuation: np.ndarray,
    cfg: CtConfig,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Detector signal of one slice [keV], shape (n_views, n_detectors), in view chunks."""
    energies_kev, counts = flat_field_counts(cfg)
    n_views, n_detectors = path_lengths.shape[1:]
    signal = np.empty((n_views, n_detectors))
    views_per_chunk = max(1, cfg.max_chunk_samples // (n_detectors * energies_kev.size))
    for start in range(0, n_views, views_per_chunk):
        chunk = slice(start, start + views_per_chunk)
        signal[chunk] = detector_signal(
            path_lengths[:, chunk],
            attenuation,
            counts,
            energies_kev,
            rng=rng,
            noise_scale=cfg.noise_scale,
        )
    return signal


def simulate_ct_fractions(
    material_fractions: np.ndarray, grid: Grid3D, cfg: CtConfig, tree: SeedTree
) -> CtVolume:
    """Simulate a CT scan of a volume given as basis-material volume fractions.

    Photon noise of slice ``iz`` is drawn from the stream ``("ct_photon_noise",
    iz)``, so it does not depend on which other slices are scanned.

    Args:
        material_fractions: volume fractions of `BASIS_MATERIALS` [-], shape
            (4, nx, ny, nz), indexed [material, ix, iy, iz].
        grid: grid of the volume; its (x, y) origin places it relative to the
            isocentre at head-frame x = y = 0.
        cfg: scan settings.
        tree: seed tree of the run.

    Returns:
        `CtVolume` with ``hu`` float32 of shape (nx, ny, n_slices) on the slab
        of `slice_indices`. Pixels outside the scan field of view are set to
        -1000 HU. ``meta`` holds the scan settings, the HU calibration, the
        first slice index and the root seed.

    Raises:
        ValueError: if the shapes disagree, or a material other than air lies
            outside the scan field of view in a scanned slice (truncated
            projections cannot be reconstructed).
    """
    expected_shape = (len(BASIS_MATERIALS), *grid.shape)
    if material_fractions.shape != expected_shape:
        raise ValueError(
            f"material_fractions has shape {material_fractions.shape}, expected {expected_shape}"
        )
    geom = cfg.geometry
    slices = slice_indices(grid, cfg)
    nx, ny = grid.shape[:2]
    offset = _center_offset(grid)
    x, y = pixel_coordinates((nx, ny), grid.spacing, offset)
    in_fov = np.hypot(x, y) <= geom.fov_radius

    energies_kev, _ = flat_field_counts(cfg)
    attenuation = basis_attenuation(energies_kev)
    calibration = hu_calibration(cfg)

    hu = np.empty((nx, ny, len(slices)), dtype=np.float32)
    for k, iz in enumerate(slices):
        fractions = np.asarray(material_fractions[..., iz], dtype=np.float64)
        non_air = np.delete(fractions, _AIR, axis=0)
        if np.any(non_air[:, ~in_fov] > 0.0):
            raise ValueError(
                f"slice {iz}: material other than air lies outside the scan field of view "
                f"(radius {geom.fov_radius:.4f} m)"
            )
        present = np.flatnonzero(fractions.reshape(len(BASIS_MATERIALS), -1).any(axis=1))
        path_lengths = fan_beam_project(
            fractions[present],
            grid.spacing,
            geom,
            center_offset=offset,
            ray_step_fraction=cfg.ray_step_fraction,
            max_chunk_samples=cfg.max_chunk_samples,
        )
        rng = tree.rng("ct_photon_noise", iz) if cfg.photon_noise else None
        signal = _measure_slice(path_lengths, attenuation[present], cfg, rng)
        mu = fbp_fan_beam(
            process_signal(signal, cfg),
            geom,
            cfg.kernel,
            (nx, ny),
            grid.spacing,
            center_offset=offset,
            max_chunk_samples=cfg.max_chunk_samples,
        )
        mu[~in_fov] = calibration.mu_air
        hu[..., k] = hu_from_mu(mu, calibration.mu_water, calibration.mu_air)

    slab_grid = Grid3D(
        shape=(nx, ny, len(slices)),
        spacing=grid.spacing,
        origin=(grid.origin[0], grid.origin[1], grid.origin[2] + slices.start * grid.spacing),
    )
    meta = {
        "config": to_dict(cfg),
        "root_seed": tree.root_seed,
        "slice_start": slices.start,
        "mu_water": calibration.mu_water,
        "mu_air": calibration.mu_air,
    }
    return CtVolume(hu=hu, grid=slab_grid, meta=meta)


def simulate_ct(head: HeadVolume, cfg: CtConfig, tree: SeedTree) -> CtVolume:
    """Simulate a CT scan of a ground-truth head (see `simulate_ct_fractions`)."""
    return simulate_ct_fractions(head.material_fractions, grid=head.grid, cfg=cfg, tree=tree)
