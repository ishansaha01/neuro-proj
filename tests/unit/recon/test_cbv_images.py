import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import PlaneSpec, plane_points
from neurosim.common.seeding import SeedTree
from neurosim.head.plane import default_plane, sample_plane
from neurosim.recon.cbv_images import (
    PixelGridConfig,
    estimate_memory_bytes,
    estimate_runtime_seconds,
    is_heldout,
    lateral_shift_limits,
    pixel_grid,
    prior_head,
    rotated_plane,
    sample_cbv_image,
    sample_cbv_images,
)


def test_pixel_grid_matches_its_config():
    grid = pixel_grid(
        PixelGridConfig(n=32, spacing=1.0e-3, top_depth=12.0e-3, lateral_center=2.0e-3)
    )
    z, x = grid.axes()
    assert grid.shape == (32, 32)
    assert z[0] == pytest.approx(12.0e-3) and z[-1] == pytest.approx(43.0e-3)
    assert 0.5 * (x[0] + x[-1]) == pytest.approx(2.0e-3)


def test_dataset_arrays(small_cfg, small_data):
    cfg = small_cfg.data
    n_images, n = cfg.n_heads * cfg.planes_per_head, cfg.pixel_grid.n
    assert set(small_data) == {
        "images",
        "amplitude",
        "active",
        "head_index",
        "plane_index",
        "heldout",
    }
    images = small_data["images"]
    assert images.shape == (n_images, n, n) and images.dtype == np.float32
    assert small_data["amplitude"].shape == (n_images,)
    assert small_data["amplitude"].dtype == np.float32
    assert small_data["active"].dtype == bool and small_data["heldout"].dtype == bool
    np.testing.assert_array_equal(
        small_data["head_index"], np.repeat(np.arange(cfg.n_heads), cfg.planes_per_head)
    )
    np.testing.assert_array_equal(
        small_data["plane_index"], np.tile(np.arange(cfg.planes_per_head), cfg.n_heads)
    )


def test_images_are_cbv_fractions_with_brain_below_the_skull(small_data):
    images = small_data["images"]
    assert np.all(np.isfinite(images)) and images.min() >= 0.0 and images.max() < 1.0
    # The top row is above the brain (zero CBV); the bottom row is brain tissue.
    assert np.all(images[:, 0] == 0.0)
    assert np.all(images[:, -1].mean(axis=1) > 0.005)


def test_rest_and_active_amplitudes(small_cfg, small_data):
    low, high = small_cfg.data.amplitude_range
    active, amplitude = small_data["active"], small_data["amplitude"]
    assert np.all(amplitude[~active] == 0.0)
    assert np.all((amplitude[active] >= low) & (amplitude[active] <= high))
    assert 0.25 < active.mean() < 0.75


def test_activation_scales_the_resting_image(small_cfg, data_seed):
    all_rest = dataclasses.replace(small_cfg.data, n_heads=2, active_fraction=0.0)
    all_active = dataclasses.replace(all_rest, active_fraction=1.0)
    rest = sample_cbv_images(all_rest, SeedTree(data_seed))
    task = sample_cbv_images(all_active, SeedTree(data_seed))
    assert not rest["active"].any() and task["active"].all()
    gain = task["images"] - rest["images"]
    limit = task["amplitude"][:, None, None] * rest["images"]
    assert np.all(gain >= 0.0) and np.all(gain <= limit + 1e-7)
    # The activated cortex is inside the field of view of most planes.
    assert np.mean(gain.max(axis=(1, 2)) > 0.0) > 0.8


def test_dataset_is_deterministic_and_seed_dependent(small_cfg, small_data, data_seed):
    two_heads = dataclasses.replace(small_cfg.data, n_heads=2)
    again = sample_cbv_images(two_heads, SeedTree(data_seed))
    n_images = 2 * two_heads.planes_per_head
    for name, values in again.items():
        np.testing.assert_array_equal(values, small_data[name][:n_images], err_msg=name)
    other = sample_cbv_images(two_heads, SeedTree(data_seed + 1))
    assert not np.array_equal(other["images"], again["images"])


def test_single_image_can_be_regenerated_alone(small_cfg, small_data, data_seed):
    cfg = small_cfg.data
    head_index, plane_index = 4, 5
    image, amplitude, active = sample_cbv_image(cfg, SeedTree(data_seed), head_index, plane_index)
    entry = head_index * cfg.planes_per_head + plane_index
    np.testing.assert_array_equal(image, small_data["images"][entry])
    assert amplitude == pytest.approx(float(small_data["amplitude"][entry]))
    assert active == bool(small_data["active"][entry])


def test_split_keeps_every_head_on_one_side(small_cfg, small_data):
    heldout, head_index = small_data["heldout"], small_data["head_index"]
    assert set(head_index[heldout]).isdisjoint(head_index[~heldout])
    assert sorted(set(head_index[heldout])) == [2, 5]
    np.testing.assert_array_equal(
        is_heldout(np.arange(7), 3), [False, False, True, False, False, True, False]
    )


def test_rotated_plane():
    plane = PlaneSpec(
        origin=(0.01, 0.02, 0.03), u_lateral=(1.0, 0.0, 0.0), u_depth=(0.0, 0.0, -1.0)
    )
    assert rotated_plane(plane, 0.0) == plane
    quarter = rotated_plane(plane, np.pi / 2.0)
    np.testing.assert_allclose(
        quarter.u_lateral, np.cross(plane.u_depth, plane.u_lateral), atol=1e-12
    )
    np.testing.assert_allclose(rotated_plane(plane, np.pi).u_lateral, (-1.0, 0.0, 0.0), atol=1e-12)
    assert quarter.origin == plane.origin and quarter.u_depth == plane.u_depth


def _shifted(plane: PlaneSpec, shift: float) -> PlaneSpec:
    origin = np.asarray(plane.origin) + shift * np.asarray(plane.u_lateral)
    return PlaneSpec(origin=tuple(origin), u_lateral=plane.u_lateral, u_depth=plane.u_depth)


def test_shift_limits_are_the_edge_of_the_head_volume(small_cfg, data_seed):
    cfg = small_cfg.data
    head = prior_head(cfg, SeedTree(data_seed), 0)
    grid = pixel_grid(cfg.pixel_grid)
    upper_index = np.asarray(head.grid.shape) - 1
    for angle in (0.0, 0.7, 2.0, 4.1):
        plane = rotated_plane(default_plane(head, cfg.head), angle)
        lowest, highest = lateral_shift_limits(head.grid, plane, grid)
        assert lowest < 0.0 < highest
        for limit in (lowest, highest):
            index = head.grid.world_to_index(plane_points(_shifted(plane, limit), grid))
            assert index.min() >= -1e-9 and np.all(index <= upper_index + 1e-9)
            # At the limit a corner pixel sits on the outermost voxel centres.
            on_edge = np.isclose(index, 0.0, atol=1e-6) | np.isclose(index, upper_index, atol=1e-6)
            assert on_edge.any()
            sample_plane(head, _shifted(plane, limit), grid)
        with pytest.raises(ValueError, match="outside the head volume"):
            sample_plane(head, _shifted(plane, highest + head.grid.spacing), grid)


def test_shift_limits_with_an_axis_aligned_array(small_cfg, data_seed):
    cfg = small_cfg.data
    head = prior_head(cfg, SeedTree(data_seed), 0)
    centre = (
        np.asarray(head.grid.origin) + 0.5 * (np.asarray(head.grid.shape) - 1) * head.grid.spacing
    )
    half_box = 0.5 * (head.grid.shape[0] - 1) * head.grid.spacing
    grid = pixel_grid(PixelGridConfig(n=8, spacing=2.0e-3, top_depth=2.0e-3))
    plane = PlaneSpec(origin=tuple(centre), u_lateral=(1.0, 0.0, 0.0), u_depth=(0.0, 0.0, 1.0))
    lowest, highest = lateral_shift_limits(head.grid, plane, grid)
    assert highest == pytest.approx(half_box - 7.0e-3) and lowest == pytest.approx(-highest)
    outside = PlaneSpec(
        origin=tuple(centre + [0.0, 2.0 * half_box, 0.0]),
        u_lateral=plane.u_lateral,
        u_depth=plane.u_depth,
    )
    lowest, highest = lateral_shift_limits(head.grid, outside, grid)
    assert lowest > highest


def test_a_head_box_too_small_for_the_grid_is_reported(small_cfg, data_seed):
    cfg = small_cfg.data
    # Thin along head y: the probe axis is inside, a rotated 30 mm wide grid is not.
    small_box = dataclasses.replace(cfg.head.grid, box_size=(0.064, 0.02, 0.064))
    too_small = dataclasses.replace(
        cfg, n_heads=1, head=dataclasses.replace(cfg.head, grid=small_box)
    )
    with pytest.raises(ValueError, match="head box cannot hold the pixel grid"):
        sample_cbv_images(too_small, SeedTree(data_seed))


def test_amplitude_range_must_be_increasing(small_cfg):
    with pytest.raises(ValueError, match="amplitude_range"):
        dataclasses.replace(small_cfg.data, amplitude_range=(0.5, 0.1))


def test_estimates_scale_with_the_number_of_heads(small_cfg):
    cfg = small_cfg.data
    double = dataclasses.replace(cfg, n_heads=2 * cfg.n_heads)
    assert estimate_runtime_seconds(cfg) > 0.0
    assert estimate_runtime_seconds(double) == pytest.approx(2.0 * estimate_runtime_seconds(cfg))
    assert estimate_memory_bytes(double) > estimate_memory_bytes(cfg) > 0
