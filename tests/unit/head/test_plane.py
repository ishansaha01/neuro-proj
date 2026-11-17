import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import Grid2D, PlaneSpec, plane_points, sample_volume
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.materials import acoustic_properties, brain_mask, skull_mask
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import HeadVolume, PlaneMedium, Tissue

SMALL_SEED = 3  # seed of the `small_head` fixture (conftest.py)
GRID = Grid2D.from_extent((-8.0e-3, 8.0e-3), (0.0, 30.0e-3), 0.25e-3)


@pytest.fixture(scope="module")
def plane(small_head, small_cfg) -> PlaneSpec:
    return default_plane(small_head, small_cfg)


@pytest.fixture(scope="module")
def medium(small_head, plane) -> PlaneMedium:
    return sample_plane(small_head, plane, GRID)


def test_default_plane_is_orthonormal_and_points_into_the_head(plane):
    u_lateral, u_depth, origin = (
        np.asarray(v) for v in (plane.u_lateral, plane.u_depth, plane.origin)
    )
    assert np.linalg.norm(u_lateral) == pytest.approx(1.0)
    assert np.linalg.norm(u_depth) == pytest.approx(1.0)
    assert u_lateral @ u_depth == pytest.approx(0.0, abs=1e-9)
    assert u_depth @ origin < 0.0
    assert u_lateral[1] == pytest.approx(0.0, abs=0.1)


def test_default_plane_is_deterministic(small_head, small_cfg, plane):
    assert default_plane(small_head, small_cfg) == plane


def test_array_sits_one_standoff_outside_the_scalp(medium, small_cfg):
    z, x = GRID.axes()
    column = medium.labels[:, np.argmin(np.abs(x))]
    assert column[0] == Tissue.BACKGROUND
    scalp_start = z[np.argmax(column >= Tissue.SCALP)]
    assert scalp_start == pytest.approx(small_cfg.probe.standoff, abs=1.5 * GRID.spacing)


def test_depth_axis_crosses_all_layers_in_order_and_hits_the_activation(medium):
    z, x = GRID.axes()
    centre = np.argmin(np.abs(x))
    column = medium.labels[:, centre]
    assert np.all(np.diff(column.astype(int)) >= 0)
    assert set(column) == set(Tissue)
    assert medium.activation[:, centre].max() > 0.8
    assert medium.activation[:, centre].max() == pytest.approx(medium.activation.max(), abs=0.1)


def test_plane_medium_shapes_and_dtypes(medium, plane, small_head):
    assert medium.grid == GRID and medium.plane == plane
    assert medium.labels.shape == GRID.shape and medium.labels.dtype == np.uint8
    for name in (
        "porosity",
        "hidden_c",
        "sound_speed",
        "density",
        "attenuation",
        "cbv_baseline",
        "activation",
    ):
        values = getattr(medium, name)
        assert values.shape == GRID.shape and values.dtype == np.float32, name
    assert medium.blood_velocity.shape == (2, *GRID.shape)
    assert medium.blood_velocity.dtype == np.float32
    assert medium.alpha_power == small_head.alpha_power


def test_acoustic_maps_come_from_the_sampled_latent_fields(medium, small_head):
    maps = acoustic_properties(
        medium.labels, medium.porosity, medium.hidden_c, small_head.materials
    )
    for derived, name in zip(maps, ("sound_speed", "density", "attenuation")):
        np.testing.assert_array_equal(getattr(medium, name), derived)
    in_skull = skull_mask(medium.labels)
    assert not medium.porosity[~in_skull].any()
    assert not medium.hidden_c[~in_skull].any()
    soft = medium.sound_speed[~in_skull]
    assert set(np.unique(soft)) <= set(
        np.unique(small_head.sound_speed[~skull_mask(small_head.labels)])
    )


def test_cbv_fields_are_zero_outside_the_brain_and_velocity_is_projected(medium, small_head, plane):
    in_brain = brain_mask(medium.labels)
    assert not medium.cbv_baseline[~in_brain].any()
    assert not medium.activation[~in_brain].any()
    assert not medium.blood_velocity[:, ~in_brain].any()
    assert medium.cbv_baseline[in_brain].max() > 0.0
    velocity = sample_volume(small_head.blood_velocity, small_head.grid, plane_points(plane, GRID))
    lateral = np.tensordot(np.asarray(plane.u_lateral), velocity.astype(np.float64), axes=1)
    depth = np.tensordot(np.asarray(plane.u_depth), velocity.astype(np.float64), axes=1)
    np.testing.assert_allclose(medium.blood_velocity[0][in_brain], lateral[in_brain], atol=1e-7)
    np.testing.assert_allclose(medium.blood_velocity[1][in_brain], depth[in_brain], atol=1e-7)


def test_plane_outside_the_volume_is_an_error(small_head, plane):
    too_deep = Grid2D.from_extent((-8.0e-3, 8.0e-3), (0.0, 0.2), 1.0e-3)
    with pytest.raises(ValueError, match="outside the head volume"):
        sample_plane(small_head, plane, too_deep)


def test_plain_head_volume_needs_explicit_material_parameters(small_head, plane, medium):
    plain = HeadVolume(
        **{f.name: getattr(small_head, f.name) for f in dataclasses.fields(HeadVolume)}
    )
    with pytest.raises(TypeError, match="pass `params`"):
        sample_plane(plain, plane, GRID)
    explicit = sample_plane(plain, plane, GRID, small_head.materials)
    np.testing.assert_array_equal(explicit.sound_speed, medium.sound_speed)


def test_default_plane_does_not_depend_on_the_head_grid(small_head, small_cfg, plane):
    full_cfg = dataclasses.replace(
        small_cfg, grid=dataclasses.replace(small_cfg.grid, region="full", spacing=4.0e-3)
    )
    full_plane = default_plane(generate_head(full_cfg, SeedTree(SMALL_SEED)), full_cfg)
    assert np.linalg.norm(np.subtract(full_plane.origin, plane.origin)) < 0.5e-3
    assert np.dot(full_plane.u_depth, plane.u_depth) > 0.999
    assert np.dot(full_plane.u_lateral, plane.u_lateral) > 0.999


def test_lateral_target_uses_the_fallback_array_orientation(small_cfg):
    cfg = dataclasses.replace(
        small_cfg, cbv=dataclasses.replace(small_cfg.cbv, target_direction=(1.0, 0.0, -0.1))
    )
    head = generate_head(cfg, SeedTree(SMALL_SEED))
    temporal_plane = default_plane(head, cfg)
    assert temporal_plane.u_depth[0] < -0.9
    assert abs(temporal_plane.u_lateral[1]) > 0.99
    medium = sample_plane(head, temporal_plane, GRID)
    assert medium.activation.max() > 0.8
