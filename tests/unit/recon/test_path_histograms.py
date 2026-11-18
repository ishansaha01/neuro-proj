"""Path-length histograms of the bent-ray tracer (`fus.backends.bent_ray_paths.PathBins`)."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.bent_ray import trace_bent_rays
from neurosim.fus.backends.bent_ray_paths import PathBins
from neurosim.fus.backends.config import BentRayConfig
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

GRID = Grid2D(shape=(81, 61), spacing=0.25e-3, origin=(0.0, -7.5e-3))
ARRAY = Transducer(
    n_elements=5, pitch=1.5e-3, element_width=1.0e-3, center_frequency=1e6,
    fractional_bandwidth=0.6,
)
POINTS = np.array([[-3.0e-3, 15.0e-3], [0.0, 17.0e-3], [4.0e-3, 18.0e-3]])


@pytest.fixture(scope="module")
def medium():
    water = Material(sound_speed=1500.0, density=1000.0, attenuation=0.0)
    bone = Material(sound_speed=2600.0, density=1900.0, attenuation=4.0)
    return slab_medium(GRID, water, bone, z_top=5.0e-3, thickness=4.0e-3, alpha_power=1.1)


def _material_bins(medium) -> PathBins:
    """Bin 0: water nodes, bin 1: bone nodes; a second, empty entry per node."""
    is_bone = (medium.sound_speed.reshape(-1) > 2000.0).astype(np.int64)
    index = np.stack((is_bone, np.zeros_like(is_bone)), axis=1)
    weight = np.stack((np.ones(is_bone.shape), np.zeros(is_bone.shape)), axis=1)
    return PathBins(index, weight, 2)


def test_bins_do_not_change_the_other_integrals(medium):
    plain = trace_bent_rays(medium, ARRAY, POINTS, BentRayConfig())
    binned = trace_bent_rays(medium, ARRAY, POINTS, BentRayConfig(), _material_bins(medium))
    assert plain.bin_path is None
    for name in ("travel_time", "alpha0_path", "transmission", "spreading", "window"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(binned, name))


def test_histogram_reproduces_the_travel_time_and_the_absorption(medium):
    rays = trace_bent_rays(medium, ARRAY, POINTS, BentRayConfig(), _material_bins(medium))
    assert rays.bin_path.shape == (5, 3, 2)
    assert rays.bin_path.dtype == np.float32
    # Two materials: the slowness and alpha0 maps are exact combinations of the two bins.
    slowness = np.array([1.0 / 1500.0, 1.0 / 2600.0])
    np.testing.assert_allclose(rays.bin_path @ slowness, rays.travel_time, rtol=2e-6)
    np.testing.assert_allclose(rays.bin_path @ np.array([0.0, 4.0]), rays.alpha0_path, rtol=2e-6)
    # The path in bone is at least the slab thickness, and the whole path at least the chord.
    assert np.all(rays.bin_path[:, :, 1] >= 4.0e-3 * (1 - 1e-3))
    assert np.all(rays.bin_path.sum(axis=2) >= rays.path_length * (1 - 1e-6))


def test_bins_must_match_the_grid(medium):
    bins = PathBins(np.zeros((10, 1), dtype=np.int64), np.ones((10, 1)), 1)
    with pytest.raises(ValueError, match="nodes"):
        trace_bent_rays(medium, ARRAY, POINTS, BentRayConfig(), bins)
    with pytest.raises(ValueError, match="bin indices"):
        PathBins(np.full((4, 1), 3), np.ones((4, 1)), 2)
