import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.bent_ray_paths import (
    integrate_paths,
    interface_transmission_squared,
    layer_normals,
    medium_maps,
)
from neurosim.fus.backends.eikonal import solve_eikonal
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium

SCALE, CONTRAST = 1.0e-3, 0.1


def paths_of(medium, sources, points, step_fraction=1.0):
    """Path integrals of the rays from `sources` to `points` in a plane medium."""
    grid = medium.grid
    maps = medium_maps(grid, medium.sound_speed, medium.density, medium.attenuation, SCALE)
    slowness = np.ascontiguousarray(maps.integrands[:, :, 0])
    field = solve_eikonal(slowness, grid, np.asarray(sources), 1.0e-6, 120)
    return integrate_paths(field, maps, np.asarray(points), step_fraction * grid.spacing, CONTRAST)


def fresnel_power(z1, c1, z2, c2, sine1):
    """Power transmission of a fluid-fluid interface from the reflection coefficient."""
    cosine1 = np.sqrt(1.0 - sine1**2)
    cosine2 = np.sqrt(1.0 - (sine1 * c2 / c1) ** 2)
    reflection = (z2 * cosine1 - z1 * cosine2) / (z2 * cosine1 + z1 * cosine2)
    return 1.0 - reflection**2


def test_layer_normals_of_a_flat_slab(grid, soft, bone, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    normals = layer_normals(1.0 / medium.sound_speed.astype(np.float64), spacing, SCALE)
    assert normals.shape == (*grid.shape, 2)
    near = normals[8:18]  # rows around and inside the slab
    np.testing.assert_allclose(np.abs(near[..., 1]), 1.0, atol=1e-9)
    np.testing.assert_allclose(near[..., 0], 0.0, atol=1e-9)
    # More than 3.7 scales (12.3 cells) from either face no gradient is near.
    assert not normals[-5:].any()


def test_layer_normals_follow_a_tilted_interface_despite_the_staircase():
    grid = Grid2D.from_extent((-6.0e-3, 6.0e-3), (0.0, 12.0e-3), 0.1e-3)
    z, x = grid.axes()
    tilt = np.deg2rad(17.0)
    below = z[:, None] > 6.0e-3 + np.tan(tilt) * x[None, :]
    slowness = np.where(below, 1.0 / 2800.0, 1.0 / 1500.0)
    normals = layer_normals(slowness, grid.spacing, SCALE)
    # Normal of the plane z = z0 + tan(tilt) x, up to its sign: (-sin, cos).
    centre = normals[60, 40:80]
    angle = np.rad2deg(np.arctan2(-centre[:, 0] * np.sign(centre[:, 1]), np.abs(centre[:, 1])))
    np.testing.assert_allclose(angle, 17.0, atol=1.5)


def test_layer_normals_of_a_homogeneous_medium_are_undefined(grid, spacing):
    assert not layer_normals(np.full(grid.shape, 1.0 / 1500.0), spacing, SCALE).any()


def test_interface_coefficient_is_the_fresnel_power_transmission():
    c1, c2 = np.array([1500.0]), np.array([2500.0])
    rho1, rho2 = np.array([1000.0]), np.array([1900.0])
    for sine1 in (0.0, 0.3, 0.55):
        oblique, at_normal = interface_transmission_squared(
            (c1, c2), (rho1, rho2), np.array([sine1 / 1500.0]), CONTRAST
        )
        expected = fresnel_power(1.5e6, 1500.0, 2500.0 * 1900.0, 2500.0, sine1)
        assert oblique[0] == pytest.approx(expected, rel=1e-12)
        assert at_normal[0] == pytest.approx(
            fresnel_power(1.5e6, 1500.0, 2500.0 * 1900.0, 2500.0, 0.0), rel=1e-12
        )
        # The same interface crossed the other way.
        reverse, _ = interface_transmission_squared(
            (c2, c1), (rho2, rho1), np.array([sine1 / 1500.0]), CONTRAST
        )
        assert reverse[0] == pytest.approx(oblique[0], rel=1e-12)


def test_interface_beyond_the_critical_angle_transmits_nothing():
    speed = (np.array([1500.0]), np.array([2500.0]))
    density = (np.array([1000.0]), np.array([1900.0]))
    sine1 = 1500.0 / 2500.0 + 0.01
    tangential = np.array([sine1 / 1500.0])
    oblique, _ = interface_transmission_squared(speed, density, tangential, CONTRAST)
    assert oblique[0] == 0.0


def test_weak_steps_keep_their_normal_incidence_coefficient():
    # 0.5 % faster: below the interface contrast, even for a ray at 85 degrees.
    speed = (np.array([2500.0]), np.array([2512.5]))
    density = (np.array([1900.0]), np.array([1900.0]))
    tangential = np.array([np.sin(np.deg2rad(85.0)) / 2500.0])
    oblique, at_normal = interface_transmission_squared(speed, density, tangential, CONTRAST)
    assert oblique[0] == pytest.approx(at_normal[0], abs=1e-4)
    assert at_normal[0] == pytest.approx(1.0, abs=1e-5)
    same, _ = interface_transmission_squared(
        (speed[0], speed[0]), density, tangential, CONTRAST
    )
    assert same[0] == 1.0


def test_homogeneous_medium_gives_straight_rays(grid):
    medium = homogeneous_medium(grid, Material(1480.0, 1000.0, attenuation=2.0), alpha_power=1.5)
    sources = np.array([[-1.2e-3, 0.0], [0.33e-3, 0.0]])
    points = np.array([[3.0e-3, 9.0e-3], [-2.2e-3, 4.1e-3], [0.33e-3, 11.0e-3]])
    paths = paths_of(medium, sources, points)
    distance = np.hypot(
        points[None, :, 0] - sources[:, None, 0], points[None, :, 1] - sources[:, None, 1]
    )
    assert paths.travel_time.shape == (2, 3)
    np.testing.assert_allclose(paths.travel_time, distance / 1480.0, rtol=1e-9)
    np.testing.assert_allclose(paths.alpha0_path, 2.0 * distance, rtol=1e-9)
    np.testing.assert_allclose(paths.speed_path, 1480.0 * distance, rtol=1e-9)
    np.testing.assert_allclose(paths.spread_path, paths.speed_path, rtol=1e-12)
    np.testing.assert_array_equal(paths.transmission, 1.0)
    np.testing.assert_array_equal(paths.end_cosines, 1.0)
    np.testing.assert_array_equal(paths.min_cosine, 1.0)
    assert not paths.forced.any()


def test_point_on_its_source_has_an_empty_path(grid):
    medium = homogeneous_medium(grid, Material(1480.0, 1000.0), alpha_power=1.5)
    paths = paths_of(medium, [[0.0, 0.0]], [[0.0, 0.0]])
    assert paths.travel_time[0, 0] == 0.0 and paths.speed_path[0, 0] == 0.0
    assert paths.transmission[0, 0] == 1.0


def test_oblique_ray_through_a_slab_follows_snell(soft, bone, spacing):
    grid = Grid2D.from_extent((-3.0e-3, 12.0e-3), (0.0, 15.0e-3), spacing)
    lossless = Material(bone.sound_speed, bone.density)
    top, thickness = 9.5 * spacing, 12 * spacing
    medium = slab_medium(grid, soft, lossless, top, thickness, alpha_power=1.5)
    offset, depth = 9.0e-3, 15.0e-3
    paths = paths_of(medium, [[0.0, 0.0]], [[offset, depth]])

    # Snell's law: sum of d_i tan(theta_i) = offset, sin(theta_i) = p c_i.
    layers = np.array([top, thickness, depth - top - thickness])
    speeds = np.array([soft.sound_speed, bone.sound_speed, soft.sound_speed])
    parameters = np.linspace(0.0, 0.999 / bone.sound_speed, 200001)
    cosines = np.sqrt(1.0 - np.outer(parameters, speeds) ** 2)
    reach = (layers * np.outer(parameters, speeds) / cosines).sum(axis=1)
    p = np.interp(offset, reach, parameters)
    cosine = np.sqrt(1.0 - (p * speeds) ** 2)
    # Midpoint rule with a step of one cell: at most spacing / 8 of path per interface.
    quadrature = 2 * spacing / 8 * (1.0 / soft.sound_speed - 1.0 / bone.sound_speed)
    assert paths.travel_time[0, 0] == pytest.approx(
        np.sum(layers / (speeds * cosine)), abs=quadrature
    )
    assert paths.travel_time[0, 0] < np.hypot(offset, depth) * np.sum(layers / speeds) / depth
    z1, z2 = soft.sound_speed * soft.density, bone.sound_speed * bone.density
    expected = fresnel_power(z1, soft.sound_speed, z2, bone.sound_speed, p * soft.sound_speed)
    assert paths.transmission[0, 0] == pytest.approx(expected, rel=0.02)
    assert paths.normal_transmission[0, 0] == pytest.approx(4 * z1 * z2 / (z1 + z2) ** 2, rel=1e-9)
    assert paths.end_cosines[0, 0] == pytest.approx(cosine[0] * cosine[2], rel=0.01)
    assert paths.min_cosine[0, 0] == pytest.approx(cosine[1], rel=0.03)
    assert paths.spread_path[0, 0] == pytest.approx(np.sum(layers * speeds / cosine**3), rel=0.02)


def test_single_sample_inclusion_is_not_an_interface(grid, soft, bone):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    column = grid.shape[1] // 2

    def with_bone_rows(rows):
        speed, density = medium.sound_speed.copy(), medium.density.copy()
        speed[rows, column - 2 : column + 3] = bone.sound_speed
        density[rows, column - 2 : column + 3] = bone.density
        return dataclasses.replace(medium, sound_speed=speed, density=density)

    below = [[0.0, 9.0e-3]]
    one = paths_of(with_bone_rows(slice(20, 21)), [[0.0, 0.0]], below)
    assert one.transmission[0, 0] == 1.0
    three = paths_of(with_bone_rows(slice(20, 23)), [[0.0, 0.0]], below)
    z1, z2 = soft.sound_speed * soft.density, bone.sound_speed * bone.density
    assert three.transmission[0, 0] == pytest.approx(4 * z1 * z2 / (z1 + z2) ** 2, rel=1e-9)


@pytest.mark.parametrize("step_fraction", [1.0, 0.5, 0.23])
def test_interface_loss_does_not_depend_on_the_ray_step(grid, soft, bone, spacing, step_fraction):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    paths = paths_of(medium, [[0.0, 0.0]], [[0.0, 9.0e-3]], step_fraction)
    z1, z2 = soft.sound_speed * soft.density, bone.sound_speed * bone.density
    assert paths.transmission[0, 0] == pytest.approx(4 * z1 * z2 / (z1 + z2) ** 2, rel=1e-9)
    assert paths.alpha0_path[0, 0] == pytest.approx(bone.attenuation * 6 * spacing, rel=1e-3)
