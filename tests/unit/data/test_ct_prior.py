"""data.ct_prior: slab geometry, plane-aligned scan and the CT-derived plane medium."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from neurosim.common.config import ConfigError, from_dict, load_yaml, to_dict
from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.grid import Grid2D, Grid3D, PlaneSpec, plane_to_world, sample_volume
from neurosim.common.seeding import SeedTree
from neurosim.ct.acoustics import BONE_THRESHOLD_HU, nominal_hu_mapping, sample_hu_mapping
from neurosim.ct.config import CtConfig
from neurosim.ct.io import load_ct, save_ct
from neurosim.ct.simulate import estimate_cost as estimate_ct_cost
from neurosim.ct.types import CtVolume
from neurosim.data.ct_prior import (
    CtPriorConfig,
    ct_plane_hu,
    ct_plane_medium,
    ct_prior_medium,
    estimate_cost,
    head_to_slab,
    plane_aligned_fractions,
    prior_hu_mapping,
    scan_plane,
    slab_frame,
    slab_grid,
    slab_to_head,
)
from neurosim.head.config import HeadConfig
from neurosim.head.generate import GeneratedHead, generate_head
from neurosim.head.materials import skull_mask
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import HeadVolume, PlaneMedium, Tissue

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

AIR, WATER, SOFT_TISSUE, BONE = (BASIS_MATERIALS.index(name) for name in BASIS_MATERIALS)

# Block phantom: a 48 mm cube of water at 1 mm with two balls centred in an
# oblique imaging plane. Neither ball is on a symmetry axis of the scanned
# region (x from -12 to 12 mm, z from 4 to 28 mm, centre (0, 16) mm), so a
# lateral flip, a depth flip or a transpose moves them by at least 12 mm.
BLOCK_SPACING = 1.0e-3
BLOCK_SIZE = 0.048
EXTENT_X = (-0.012, 0.012)
EXTENT_Z = (0.004, 0.028)
REGION_CENTER = (0.0, 0.016)
BONE_BALL_XZ = (0.006, 0.010)
BONE_BALL_RADIUS = 0.003
TISSUE_BALL_XZ = (-0.005, 0.022)
TISSUE_BALL_RADIUS = 0.004
WATER_POINT_XZ = (-0.009, 0.009)
FINE_SPACING = 0.5e-3


def _oblique_plane() -> PlaneSpec:
    """A plane tilted about all three head axes whose point (0, 16 mm) is the head centre."""
    u_depth = np.array([-0.5, 0.3, -0.81])
    u_depth /= np.linalg.norm(u_depth)
    u_lateral = np.array([1.0, 0.2, 0.0])
    u_lateral -= (u_lateral @ u_depth) * u_depth
    u_lateral /= np.linalg.norm(u_lateral)
    origin = -REGION_CENTER[1] * u_depth
    return PlaneSpec(origin=tuple(origin), u_lateral=tuple(u_lateral), u_depth=tuple(u_depth))


def _ball_coverage(grid: Grid3D, center: np.ndarray, radius: float) -> np.ndarray:
    """Fraction of each voxel inside a ball, with a one-voxel linear edge; shape of the grid."""
    distance = np.linalg.norm(grid.coords() - center, axis=-1)
    return np.clip((radius - distance) / grid.spacing + 0.5, 0.0, 1.0)


def _head_of_fractions(grid: Grid3D, fractions: np.ndarray) -> HeadVolume:
    """A `HeadVolume` that carries only a grid and material fractions."""
    empty = np.zeros(grid.shape, np.float32)
    return HeadVolume(
        grid=grid,
        sdf=empty,
        porosity=empty,
        hidden_c=empty,
        labels=empty.astype(np.uint8),
        sound_speed=empty,
        density=empty,
        attenuation=empty,
        material_fractions=fractions,
        cbv_baseline=empty,
        activation=empty,
        blood_velocity=empty,
        alpha_power=CtConfig().hu_mapping.alpha_power,
    )


@pytest.fixture(scope="module")
def plane() -> PlaneSpec:
    return _oblique_plane()


@pytest.fixture(scope="module")
def block_head(plane) -> HeadVolume:
    grid = Grid3D.centered((BLOCK_SIZE,) * 3, BLOCK_SPACING)
    fractions = np.zeros((len(BASIS_MATERIALS), *grid.shape))
    fractions[WATER] = 1.0
    for material, xz, radius in (
        (BONE, BONE_BALL_XZ, BONE_BALL_RADIUS),
        (SOFT_TISSUE, TISSUE_BALL_XZ, TISSUE_BALL_RADIUS),
    ):
        coverage = _ball_coverage(grid, plane_to_world(plane, xz), radius)
        fractions *= 1.0 - coverage
        fractions[material] += coverage
    return _head_of_fractions(grid, fractions.astype(np.float32))


@pytest.fixture(scope="module")
def ct_cfg() -> CtConfig:
    return from_dict(CtConfig, load_yaml(CONFIG_DIR / "ct_small.yaml"))


@pytest.fixture(scope="module")
def prior_cfg() -> CtPriorConfig:
    return CtPriorConfig(slab_half_slices=1, margin=2.0e-3)


@pytest.fixture(scope="module")
def fine_grid() -> Grid2D:
    return Grid2D.from_extent(EXTENT_X, EXTENT_Z, FINE_SPACING)


@pytest.fixture(scope="module")
def block_ct(block_head, plane, ct_cfg, prior_cfg) -> CtVolume:
    noise_free = dataclasses.replace(ct_cfg, photon_noise=False)
    return scan_plane(block_head, plane, noise_free, prior_cfg, SeedTree(0), EXTENT_X, EXTENT_Z)


@pytest.fixture(scope="module")
def block_medium(block_ct, fine_grid, plane, ct_cfg) -> PlaneMedium:
    hu_plane = ct_plane_hu(block_ct, fine_grid, plane)
    mapping = nominal_hu_mapping(ct_cfg)
    return ct_plane_medium(hu_plane, fine_grid, plane, mapping, ct_cfg.hu_mapping.alpha_power)


@pytest.fixture(scope="module")
def small_head_cfg() -> HeadConfig:
    return from_dict(HeadConfig, load_yaml(CONFIG_DIR / "head_small.yaml"))


@pytest.fixture(scope="module")
def small_head(small_head_cfg) -> GeneratedHead:
    return generate_head(small_head_cfg, SeedTree(3))


def _value_at(values: np.ndarray, grid: Grid2D, xz: tuple[float, float]) -> float:
    """Value of a plane map at the pixel nearest to a plane point."""
    z, x = grid.axes()
    return float(values[np.argmin(np.abs(z - xz[1])), np.argmin(np.abs(x - xz[0]))])


# --- configuration ---------------------------------------------------------------------


def test_default_yaml_equals_the_dataclass_defaults():
    data = load_yaml(CONFIG_DIR / "ct_prior_default.yaml")
    assert from_dict(CtPriorConfig, data) == CtPriorConfig()
    assert set(data) == set(to_dict(CtPriorConfig()))


@pytest.mark.parametrize(
    "data",
    [
        {"slab_half_slices": -1},
        {"slab_spacing": 0.0},
        {"margin": -1.0e-3},
        {"hu_mapping": "exact"},
        {"unknown_key": 1},
    ],
)
def test_invalid_config_values_are_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(CtPriorConfig, data)


# --- slab geometry ---------------------------------------------------------------------


def test_slab_frame_origin_is_the_centre_of_the_region(plane):
    frame = slab_frame(plane, EXTENT_X, EXTENT_Z)
    np.testing.assert_allclose(frame.origin, plane_to_world(plane, REGION_CENTER), atol=1e-12)
    assert frame.u_lateral == plane.u_lateral and frame.u_depth == plane.u_depth


def test_slab_axes_are_lateral_depth_and_the_right_handed_normal(plane):
    frame = slab_frame(plane, EXTENT_X, EXTENT_Z)
    origin = np.asarray(frame.origin)
    normal = np.cross(plane.u_lateral, plane.u_depth)
    np.testing.assert_allclose(slab_to_head(frame, (1.0, 0.0, 0.0)) - origin, plane.u_lateral)
    np.testing.assert_allclose(slab_to_head(frame, (0.0, 1.0, 0.0)) - origin, plane.u_depth)
    np.testing.assert_allclose(slab_to_head(frame, (0.0, 0.0, 1.0)) - origin, normal)


def test_head_to_slab_inverts_slab_to_head(plane):
    frame = slab_frame(plane, EXTENT_X, EXTENT_Z)
    local = SeedTree(0).rng("dataset_sweep").uniform(-0.02, 0.02, size=(5, 4, 3))
    np.testing.assert_allclose(head_to_slab(frame, slab_to_head(frame, local)), local, atol=1e-15)


def test_plane_points_map_to_offset_plane_coordinates_with_zero_normal(plane):
    frame = slab_frame(plane, EXTENT_X, EXTENT_Z)
    xz = np.array([[0.0, 0.0], BONE_BALL_XZ, TISSUE_BALL_XZ])
    local = head_to_slab(frame, plane_to_world(plane, xz))
    np.testing.assert_allclose(local[:, :2], xz - np.asarray(REGION_CENTER), atol=1e-15)
    np.testing.assert_allclose(local[:, 2], 0.0, atol=1e-15)


def test_slab_grid_is_odd_symmetric_and_covers_the_region_plus_margin(prior_cfg):
    grid = slab_grid(EXTENT_X, EXTENT_Z, BLOCK_SPACING, prior_cfg)
    # Half-width 12 mm + 2 mm margin at 1 mm: 14 voxels each side of the centre.
    assert grid.shape == (29, 29, 3)
    for axis in grid.axes():
        np.testing.assert_allclose(axis, -axis[::-1], atol=1e-15)
        assert axis[axis.size // 2] == 0.0
    assert grid.axes()[0][-1] == pytest.approx(0.014)


def test_slab_grid_rounds_a_partial_voxel_up():
    cfg = CtPriorConfig(slab_half_slices=0, margin=0.0)
    grid = slab_grid((0.0, 0.0101), (0.0, 0.0), 1.0e-3, cfg)
    assert grid.shape == (13, 1, 1)


def test_reversed_range_is_rejected(plane, prior_cfg):
    with pytest.raises(ValueError, match="extent_x"):
        slab_frame(plane, (0.01, -0.01), EXTENT_Z)
    with pytest.raises(ValueError, match="extent_z"):
        slab_grid(EXTENT_X, (0.02, 0.01), BLOCK_SPACING, prior_cfg)


# --- resampled fractions ---------------------------------------------------------------


def test_fractions_have_the_slab_shape_and_sum_to_one(block_head, plane, prior_cfg):
    fractions, grid = plane_aligned_fractions(block_head, plane, EXTENT_X, EXTENT_Z, prior_cfg)
    assert grid == slab_grid(EXTENT_X, EXTENT_Z, BLOCK_SPACING, prior_cfg)
    assert fractions.shape == (len(BASIS_MATERIALS), *grid.shape)
    assert fractions.dtype == np.float32
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1e-6)
    assert fractions.min() >= 0.0


def test_slab_voxels_hold_the_head_at_their_documented_plane_position(
    block_head, plane, prior_cfg
):
    fractions, grid = plane_aligned_fractions(block_head, plane, EXTENT_X, EXTENT_Z, prior_cfg)
    n_lateral, n_depth, n_normal = grid.shape
    lateral = REGION_CENTER[0] + (np.arange(n_lateral) - (n_lateral - 1) / 2) * grid.spacing
    depth = REGION_CENTER[1] + (np.arange(n_depth) - (n_depth - 1) / 2) * grid.spacing
    xz = np.stack(np.meshgrid(lateral, depth, indexing="ij"), axis=-1)
    normal = np.cross(plane.u_lateral, plane.u_depth)
    for i_normal in range(n_normal):
        offset = (i_normal - (n_normal - 1) / 2) * grid.spacing
        points = plane_to_world(plane, xz) + offset * normal
        expected = sample_volume(block_head.material_fractions, block_head.grid, points)
        np.testing.assert_allclose(fractions[..., i_normal], expected, atol=1e-6)


def test_slab_voxels_outside_the_head_volume_are_air(block_head, plane):
    wide = CtPriorConfig(slab_half_slices=0, margin=0.03)
    fractions, grid = plane_aligned_fractions(block_head, plane, EXTENT_X, EXTENT_Z, wide)
    points = slab_to_head(slab_frame(plane, EXTENT_X, EXTENT_Z), grid.coords())
    outside = np.any(np.abs(points) > 0.5 * BLOCK_SIZE, axis=-1)
    assert outside.any() and not outside.all()
    np.testing.assert_array_equal(fractions[AIR][outside], 1.0)
    np.testing.assert_array_equal(fractions[AIR][~outside], 0.0)
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1e-6)


def test_slab_spacing_overrides_the_head_spacing(block_head, plane):
    cfg = CtPriorConfig(slab_half_slices=0, slab_spacing=2.0e-3, margin=0.0)
    fractions, grid = plane_aligned_fractions(block_head, plane, EXTENT_X, EXTENT_Z, cfg)
    assert grid.spacing == 2.0e-3
    assert grid.shape == (13, 13, 1)
    assert fractions.shape[1:] == grid.shape


# --- scan ------------------------------------------------------------------------------


def test_scan_returns_the_slab_in_the_slab_frame(block_ct, plane, prior_cfg):
    grid = slab_grid(EXTENT_X, EXTENT_Z, BLOCK_SPACING, prior_cfg)
    assert block_ct.grid == grid
    assert block_ct.hu.shape == grid.shape and block_ct.hu.dtype == np.float32
    frame = slab_frame(plane, EXTENT_X, EXTENT_Z)
    assert block_ct.meta["slab_frame"]["origin"] == list(frame.origin)
    assert block_ct.meta["slab_frame"]["u_lateral"] == list(plane.u_lateral)
    assert block_ct.meta["slab_frame"]["u_depth"] == list(plane.u_depth)
    assert block_ct.meta["ct_prior"] == to_dict(prior_cfg)
    assert block_ct.meta["config"]["slab_z_range"] is None


def test_scan_ignores_the_axial_slab_range_of_the_scan_config(
    block_head, plane, ct_cfg, block_ct
):
    cfg = CtPriorConfig(slab_half_slices=0, margin=2.0e-3)
    elsewhere = dataclasses.replace(ct_cfg, photon_noise=False, slab_z_range=(0.5, 0.6))
    ct = scan_plane(block_head, plane, elsewhere, cfg, SeedTree(0), EXTENT_X, EXTENT_Z)
    assert ct.hu.shape == (*block_ct.hu.shape[:2], 1)
    # Slices are independent and noise is off: the single slice is the central slice.
    np.testing.assert_array_equal(ct.hu[..., 0], block_ct.hu[..., 1])


def test_scan_noise_is_seeded_by_the_tree(block_head, plane, ct_cfg):
    cfg = CtPriorConfig(slab_half_slices=0, margin=2.0e-3)
    first, again, other = (
        scan_plane(block_head, plane, ct_cfg, cfg, SeedTree(seed), EXTENT_X, EXTENT_Z).hu
        for seed in (4, 4, 5)
    )
    np.testing.assert_array_equal(first, again)
    assert np.any(first != other)


def test_scan_outside_the_field_of_view_is_rejected(block_head, plane, ct_cfg, prior_cfg):
    narrow_fan = dataclasses.replace(
        ct_cfg,
        geometry=dataclasses.replace(ct_cfg.geometry, n_detectors=8),
        calibration_phantom_diameter=0.004,
    )
    with pytest.raises(ValueError, match="field of view"):
        scan_plane(block_head, plane, narrow_fan, prior_cfg, SeedTree(0), EXTENT_X, EXTENT_Z)


# --- HU on the plane grid --------------------------------------------------------------


def _linear_ct(ct: CtVolume, gradient: tuple[float, float, float]) -> CtVolume:
    """The same slab with HU replaced by a linear function of slab coordinates (a, b, w)."""
    hu = (ct.grid.coords() @ np.asarray(gradient)).astype(np.float32)
    return dataclasses.replace(ct, hu=hu)


def test_plane_hu_follows_the_documented_coordinate_mapping(block_ct, fine_grid, plane):
    # HU = 1e5 a + 3e5 b + 7e5 w is reproduced exactly by trilinear interpolation,
    # so the plane image must be 1e5 (x - x_c) + 3e5 (z - z_c): any flip or
    # transpose of the mapping changes it.
    hu_plane = ct_plane_hu(_linear_ct(block_ct, (1.0e5, 3.0e5, 7.0e5)), fine_grid, plane)
    assert hu_plane.shape == fine_grid.shape and hu_plane.dtype == np.float32
    xz = fine_grid.points()
    expected = 1.0e5 * (xz[..., 0] - REGION_CENTER[0]) + 3.0e5 * (xz[..., 1] - REGION_CENTER[1])
    np.testing.assert_allclose(hu_plane, expected, atol=1e-2)


def test_plane_hu_blends_slices_for_a_plane_off_the_central_slice(block_ct, plane):
    shift = 0.4 * BLOCK_SPACING
    normal = np.cross(plane.u_lateral, plane.u_depth)
    shifted = dataclasses.replace(plane, origin=tuple(np.asarray(plane.origin) + shift * normal))
    grid = Grid2D.from_extent((-0.004, 0.004), (0.01, 0.02), FINE_SPACING)
    hu_plane = ct_plane_hu(_linear_ct(block_ct, (0.0, 0.0, 7.0e5)), grid, shifted)
    np.testing.assert_allclose(hu_plane, 7.0e5 * shift, rtol=1e-5)


def test_plane_hu_rejects_a_grid_that_leaves_the_slab(block_ct, plane):
    too_wide = Grid2D.from_extent((-0.02, 0.02), EXTENT_Z, FINE_SPACING)
    with pytest.raises(ValueError, match="outside the scanned slab"):
        ct_plane_hu(block_ct, too_wide, plane)


def test_plane_hu_rejects_a_ct_without_a_slab_frame(block_ct, fine_grid, plane):
    axial = CtVolume(hu=block_ct.hu, grid=block_ct.grid, meta={})
    with pytest.raises(ValueError, match="slab frame"):
        ct_plane_hu(axial, fine_grid, plane)


def test_plane_hu_survives_a_save_and_load(block_ct, fine_grid, plane, ct_cfg, tmp_path):
    save_ct(tmp_path / "slab.h5", block_ct, ct_cfg, root_seed=0)
    loaded = load_ct(tmp_path / "slab.h5")
    np.testing.assert_array_equal(
        ct_plane_hu(loaded, fine_grid, plane), ct_plane_hu(block_ct, fine_grid, plane)
    )


def test_asymmetric_bone_ball_lands_at_its_plane_position(block_ct, fine_grid, plane):
    hu_plane = ct_plane_hu(block_ct, fine_grid, plane).astype(np.float64)
    weight = np.clip(hu_plane - BONE_THRESHOLD_HU, 0.0, None)
    centroid = np.tensordot(weight, fine_grid.points(), axes=2) / weight.sum()
    # Within one slab voxel; a flip or a transpose would be off by 12 mm or more.
    assert np.abs(centroid - np.asarray(BONE_BALL_XZ)).max() < BLOCK_SPACING
    assert _value_at(hu_plane, fine_grid, BONE_BALL_XZ) > 1000.0
    assert abs(_value_at(hu_plane, fine_grid, WATER_POINT_XZ)) < 50.0


# --- CT-derived medium -----------------------------------------------------------------


def test_medium_maps_bone_soft_tissue_and_water_to_their_speeds(block_medium, fine_grid):
    speed = block_medium.sound_speed
    assert _value_at(speed, fine_grid, BONE_BALL_XZ) > 2700.0
    # Soft tissue (about +55 HU) and water map next to the water end point (1500 m/s).
    assert 1480.0 < _value_at(speed, fine_grid, TISSUE_BALL_XZ) < 1600.0
    assert _value_at(speed, fine_grid, WATER_POINT_XZ) == pytest.approx(1500.0, abs=40.0)
    assert _value_at(block_medium.labels, fine_grid, BONE_BALL_XZ) == Tissue.SKULL_OUTER
    assert _value_at(block_medium.labels, fine_grid, TISSUE_BALL_XZ) == Tissue.BACKGROUND
    assert _value_at(block_medium.labels, fine_grid, WATER_POINT_XZ) == Tissue.BACKGROUND


def test_medium_fields_follow_the_contract(ct_cfg, plane):
    grid = Grid2D.from_extent((0.0, 0.001), (0.0, 0.002), 1.0e-3)
    mapping = nominal_hu_mapping(ct_cfg)
    hu_plane = np.array(
        [
            [-1000.0, 0.0],
            [BONE_THRESHOLD_HU - 1.0, BONE_THRESHOLD_HU],
            [0.5 * mapping.hu_bone, 2.0 * mapping.hu_bone],
        ]
    )
    medium = ct_plane_medium(hu_plane, grid, plane, mapping, alpha_power=1.3)

    assert medium.grid == grid and medium.plane == plane and medium.alpha_power == 1.3
    bone = Tissue.SKULL_OUTER
    np.testing.assert_array_equal(medium.labels, [[0, 0], [0, bone], [bone, bone]])
    assert medium.labels.dtype == np.uint8
    threshold_porosity = 1.0 - BONE_THRESHOLD_HU / mapping.hu_bone
    np.testing.assert_allclose(
        medium.porosity, [[0.0, 0.0], [0.0, threshold_porosity], [0.5, 0.0]], atol=1e-6
    )
    middle = 0.5 * (mapping.c_water + mapping.c_bone)
    np.testing.assert_allclose(medium.sound_speed[0], mapping.c_water)
    np.testing.assert_allclose(medium.sound_speed[2], [middle, mapping.c_bone])
    np.testing.assert_allclose(medium.density[0], mapping.rho_water)
    assert not medium.attenuation[0].any() and medium.attenuation[1, 0] == 0.0
    assert np.all(medium.attenuation[[1, 2, 2], [1, 0, 1]] > 0.0)
    for name in ("porosity", "hidden_c", "sound_speed", "density", "attenuation"):
        values = getattr(medium, name)
        assert values.shape == grid.shape and values.dtype == np.float32
    for name in ("hidden_c", "cbv_baseline", "activation"):
        assert getattr(medium, name).shape == grid.shape and not getattr(medium, name).any()
    assert medium.blood_velocity.shape == (2, *grid.shape) and not medium.blood_velocity.any()


def test_medium_rejects_a_mismatched_image(ct_cfg, plane, fine_grid):
    with pytest.raises(ValueError, match="hu_plane"):
        ct_plane_medium(np.zeros((3, 3)), fine_grid, plane, nominal_hu_mapping(ct_cfg), 1.1)


def test_mapping_choice(ct_cfg):
    tree = SeedTree(7)
    nominal = prior_hu_mapping(ct_cfg, CtPriorConfig(hu_mapping="nominal"), tree)
    sampled = prior_hu_mapping(ct_cfg, CtPriorConfig(hu_mapping="sampled"), tree)
    assert nominal == nominal_hu_mapping(ct_cfg)
    assert sampled == sample_hu_mapping(tree, ct_cfg)
    assert sampled != nominal


# --- whole chain on a generated head ---------------------------------------------------


def test_prior_medium_of_a_generated_head(small_head, small_head_cfg, ct_cfg):
    plane = default_plane(small_head, small_head_cfg)
    grid = Grid2D.from_extent((-0.014, 0.014), (0.0, 0.032), 1.0e-3)
    cfg = CtPriorConfig(slab_half_slices=0)
    noise_free = dataclasses.replace(ct_cfg, photon_noise=False)
    ct, medium = ct_prior_medium(small_head, plane, grid, noise_free, cfg, SeedTree(3))
    truth = sample_plane(small_head, plane, grid)

    assert ct.grid.spacing == small_head.grid.spacing and ct.grid.shape[2] == 1
    assert medium.grid == grid and medium.alpha_power == small_head.alpha_power
    true_skull = skull_mask(truth.labels)
    ct_skull = skull_mask(medium.labels)
    # The 2 mm CT blurs a skull that is about three voxels thick, so its bone label
    # covers the true skull.
    assert ct_skull[true_skull].mean() > 0.95
    assert medium.sound_speed[true_skull].mean() > 2000.0
    brain = np.isin(truth.labels, (Tissue.GREY, Tissue.WHITE)) & ~ct_skull
    assert 1480.0 < np.median(medium.sound_speed[brain]) < 1600.0


def test_prior_medium_rejects_a_different_absorption_exponent(
    small_head, small_head_cfg, ct_cfg
):
    plane = default_plane(small_head, small_head_cfg)
    grid = Grid2D.from_extent((-0.004, 0.004), (0.0, 0.01), 1.0e-3)
    other = dataclasses.replace(
        ct_cfg, hu_mapping=dataclasses.replace(ct_cfg.hu_mapping, alpha_power=1.5)
    )
    with pytest.raises(ValueError, match="alpha_power"):
        ct_prior_medium(small_head, plane, grid, other, CtPriorConfig(), SeedTree(3))


# --- cost ------------------------------------------------------------------------------


def test_estimate_cost_describes_the_slab_scan(ct_cfg, prior_cfg):
    cost = estimate_cost(BLOCK_SPACING, ct_cfg, prior_cfg, EXTENT_X, EXTENT_Z)
    grid = slab_grid(EXTENT_X, EXTENT_Z, BLOCK_SPACING, prior_cfg)
    assert (cost["n_lateral"], cost["n_depth"], cost["n_slices"]) == grid.shape
    assert cost["spacing"] == BLOCK_SPACING
    reference = estimate_ct_cost(grid, dataclasses.replace(ct_cfg, slab_z_range=None))
    assert {key: cost[key] for key in reference} == reference
    one_slice = dataclasses.replace(prior_cfg, slab_half_slices=0)
    assert estimate_cost(BLOCK_SPACING, ct_cfg, one_slice, EXTENT_X, EXTENT_Z)["n_slices"] == 1
