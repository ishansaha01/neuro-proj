"""simulate_ct_fractions: shapes, slab selection, validation, cost estimate."""

from __future__ import annotations

import dataclasses
import tracemalloc

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.ct.config import CtConfig
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import (
    estimate_cost,
    simulate_ct,
    simulate_ct_fractions,
    slice_indices,
)
from neurosim.ct.types import CtVolume
from neurosim.head.types import HeadVolume


@pytest.fixture(scope="module")
def water_phantom(small_grid) -> np.ndarray:
    return layered_fractions(small_grid, [("water", disc_coverage(small_grid, 0.06))])


@pytest.fixture(scope="module")
def one_slice_cfg(small_cfg) -> CtConfig:
    return dataclasses.replace(small_cfg, slab_z_from_center=(0.0005, 0.0015))


@pytest.fixture(scope="module")
def noisy_scan(water_phantom, small_grid, one_slice_cfg) -> CtVolume:
    return simulate_ct_fractions(water_phantom, small_grid, one_slice_cfg, SeedTree(5))


def test_slice_indices_select_the_slab(small_grid, small_cfg):
    # Slice centres are at -7, -5, ..., 7 mm; the small config scans -4..4 mm.
    assert slice_indices(small_grid, small_cfg) == range(2, 6)
    whole = dataclasses.replace(small_cfg, slab_z_from_center=None)
    assert slice_indices(small_grid, whole) == range(8)
    with pytest.raises(ValueError, match="contains no slice"):
        slice_indices(small_grid, dataclasses.replace(small_cfg, slab_z_from_center=(0.1, 0.2)))


def test_slab_is_measured_from_the_centre_of_the_grid_wherever_the_grid_is(small_grid, small_cfg):
    # The same grid moved to z = 10 .. 24 mm, like a box-region head: the same slices.
    x0, y0, z0 = small_grid.origin
    moved = Grid3D(shape=small_grid.shape, spacing=small_grid.spacing, origin=(x0, y0, 0.010))
    assert moved.origin[2] != z0
    for slab in ((-0.004, 0.004), (0.0, 0.002), (0.0, 0.008), None):
        cfg = dataclasses.replace(small_cfg, slab_z_from_center=slab)
        assert slice_indices(moved, cfg) == slice_indices(small_grid, cfg)
    # The default slab on a grid shaped like the default head box (160 slices of 0.5 mm
    # from z = 10 mm): the 16 slices around its middle.
    box = Grid3D(shape=(4, 4, 160), spacing=0.5e-3, origin=(0.0, 0.0, 0.010))
    assert slice_indices(box, CtConfig()) == range(72, 88)


def test_output_volume_contract(noisy_scan, small_grid, one_slice_cfg):
    nx, ny, _ = small_grid.shape
    assert isinstance(noisy_scan, CtVolume)
    assert noisy_scan.hu.shape == (nx, ny, 1) and noisy_scan.hu.dtype == np.float32
    assert noisy_scan.grid.shape == (nx, ny, 1)
    assert noisy_scan.grid.spacing == small_grid.spacing
    assert noisy_scan.grid.origin[:2] == small_grid.origin[:2]
    assert noisy_scan.grid.origin[2] == pytest.approx(0.001)
    assert noisy_scan.meta["slice_start"] == 4
    assert noisy_scan.meta["root_seed"] == 5
    assert noisy_scan.meta["config"]["kvp"] == one_slice_cfg.kvp
    assert noisy_scan.meta["mu_water"] > noisy_scan.meta["mu_air"]


def test_hu_values_are_plausible(noisy_scan, small_grid):
    radius = radial_distance(small_grid)
    hu = noisy_scan.hu[..., 0]
    assert abs(hu[radius < 0.04].mean()) < 10.0
    assert hu[(radius > 0.075) & (radius < 0.1)].mean() == pytest.approx(-1000.0, abs=15.0)
    # Corners of the grid are outside the scan field of view.
    assert np.all(hu[radius > 0.112] == -1000.0)


def test_noise_switch_and_seed(water_phantom, small_grid, one_slice_cfg, noisy_scan):
    clean_cfg = dataclasses.replace(one_slice_cfg, photon_noise=False)
    clean_a = simulate_ct_fractions(water_phantom, small_grid, clean_cfg, SeedTree(5))
    clean_b = simulate_ct_fractions(water_phantom, small_grid, clean_cfg, SeedTree(6))
    np.testing.assert_array_equal(clean_a.hu, clean_b.hu)
    assert not np.array_equal(noisy_scan.hu, clean_a.hu)
    silent_cfg = dataclasses.replace(one_slice_cfg, noise_scale=0.0)
    silent = simulate_ct_fractions(water_phantom, small_grid, silent_cfg, SeedTree(5))
    np.testing.assert_allclose(silent.hu, clean_a.hu, atol=1e-3)


def test_wrong_shape_raises(small_grid, small_cfg):
    with pytest.raises(ValueError, match="expected"):
        simulate_ct_fractions(
            np.zeros((4, 3, 3, 3), np.float32), small_grid, small_cfg, SeedTree(0)
        )


def test_object_outside_the_field_of_view_raises(small_cfg):
    grid = Grid3D.centered((0.3, 0.3, 0.002), 0.002)
    fractions = layered_fractions(grid, [("water", disc_coverage(grid, 0.13))])
    cfg = dataclasses.replace(small_cfg, slab_z_from_center=None)
    with pytest.raises(ValueError, match="outside the scan field of view"):
        simulate_ct_fractions(fractions, grid, cfg, SeedTree(0))


def test_simulate_ct_reads_the_head_volume(water_phantom, small_grid, one_slice_cfg, noisy_scan):
    empty = np.zeros(small_grid.shape, np.float32)
    head = HeadVolume(
        grid=small_grid,
        sdf=empty,
        porosity=empty,
        hidden_c=empty,
        labels=empty,
        sound_speed=empty,
        density=empty,
        attenuation=empty,
        material_fractions=water_phantom,
        cbv_baseline=empty,
        activation=empty,
        blood_velocity=empty,
        alpha_power=1.1,
    )
    np.testing.assert_array_equal(simulate_ct(head, one_slice_cfg, SeedTree(5)).hu, noisy_scan.hu)


def test_estimate_cost_counts_rays_and_samples(small_grid, small_cfg):
    cost = estimate_cost(small_grid, small_cfg)
    geom = small_cfg.geometry
    assert cost["n_slices"] == 4
    assert cost["rays"] == 4 * geom.n_views * geom.n_detectors
    # 100 x 100 pixels of 2 mm: half-diagonal 141.4 mm, 1 mm steps, 4 materials.
    assert cost["samples"] == cost["rays"] * 283 * len(BASIS_MATERIALS)
    assert cost["time_s"] > 0.0
    doubled = estimate_cost(small_grid, dataclasses.replace(small_cfg, slab_z_from_center=None))
    assert doubled["rays"] == 2 * cost["rays"]


def test_estimated_memory_bounds_the_run(water_phantom, small_grid, one_slice_cfg):
    cost = estimate_cost(small_grid, one_slice_cfg)
    tracemalloc.start()
    simulate_ct_fractions(water_phantom, small_grid, one_slice_cfg, SeedTree(0))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < cost["memory_bytes"] < 2**30
    assert peak > 0.2 * cost["memory_bytes"]


def test_default_config_stays_under_one_gigabyte_on_a_head_grid():
    head_grid = Grid3D.centered((0.22, 0.22, 0.2), 0.0005)
    cost = estimate_cost(head_grid, CtConfig())
    assert cost["n_slices"] == 16
    assert cost["memory_bytes"] < 2**30


def test_off_centre_grid_reconstructs_objects_in_place(small_cfg):
    # A grid whose centre is 21 mm, -13 mm from the isocentre, holding one bone rod.
    grid = Grid3D(shape=(50, 60, 1), spacing=0.002, origin=(-0.028, -0.072, 0.0))
    rod_center = (0.03, -0.02)
    fractions = layered_fractions(
        grid,
        [
            ("water", disc_coverage(grid, 0.04, (0.02, -0.015))),
            ("cortical_bone", disc_coverage(grid, 0.006, rod_center)),
        ],
    )
    cfg = dataclasses.replace(small_cfg, slab_z_from_center=None, photon_noise=False)
    hu = simulate_ct_fractions(fractions, grid, cfg, SeedTree(0)).hu[..., 0]
    distance = radial_distance(grid, rod_center)
    assert hu[distance < 0.003].mean() > 1200.0
    assert abs(hu[(distance > 0.012) & (distance < 0.018)].mean()) < 40.0
    ix, iy = np.unravel_index(np.argmax(hu), hu.shape)
    assert distance[ix, iy] < 0.004
