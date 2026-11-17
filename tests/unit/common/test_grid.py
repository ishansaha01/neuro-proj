import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import (
    Grid2D,
    Grid3D,
    PlaneSpec,
    plane_points,
    plane_to_world,
    sample_plane_map,
    sample_volume,
)
from neurosim.common.seeding import SeedTree

GRID = Grid3D(shape=(6, 5, 4), spacing=2.0e-3, origin=(-5.0e-3, 1.0e-3, 0.0))
PLANE_GRID = Grid2D(shape=(5, 7), spacing=1.0e-3, origin=(2.0e-3, -3.0e-3))


def linear_field(points: np.ndarray) -> np.ndarray:
    """A linear function of position [m]; trilinear interpolation reproduces it exactly."""
    return 3.0 + 200.0 * points[..., 0] - 150.0 * points[..., 1] + 70.0 * points[..., 2]


def interior_points(n_points: int) -> np.ndarray:
    """Random points inside the box spanned by the voxel centres of GRID [m]."""
    axes = GRID.axes()
    low = np.array([axis[0] for axis in axes])
    high = np.array([axis[-1] for axis in axes])
    return SeedTree(0).rng("anatomy").uniform(low, high, size=(n_points, 3))


def test_grid3d_axes_coords_and_size():
    x, y, z = GRID.axes()
    np.testing.assert_allclose(x, [-5e-3, -3e-3, -1e-3, 1e-3, 3e-3, 5e-3], atol=1e-15)
    assert (x.shape, y.shape, z.shape) == ((6,), (5,), (4,))
    coords = GRID.coords()
    assert coords.shape == (6, 5, 4, 3)
    np.testing.assert_allclose(coords[2, 3, 1], [x[2], y[3], z[1]])
    assert GRID.n_voxels == 120


def test_grid3d_index_world_round_trip():
    index = GRID.world_to_index(GRID.coords())
    expected = np.stack(np.meshgrid(np.arange(6), np.arange(5), np.arange(4), indexing="ij"), -1)
    np.testing.assert_allclose(index, expected, atol=1e-9)
    np.testing.assert_allclose(GRID.world_to_index([-4.0e-3, 1.0e-3, 3.0e-3]), [0.5, 0.0, 1.5])


def test_grid3d_centered_is_symmetric_about_zero():
    grid = Grid3D.centered((0.2, 0.15, 0.101), 5.0e-4)
    assert grid.shape == (400, 300, 202)
    for axis in grid.axes():
        np.testing.assert_allclose(axis[0], -axis[-1], rtol=1e-12)
    # Outer voxel faces are half a voxel beyond the outermost centres.
    np.testing.assert_allclose(grid.axes()[0][-1] + 0.5 * grid.spacing, 0.1, rtol=1e-12)


def test_grids_are_frozen_hashable_and_validated():
    grid = Grid3D(shape=np.array([2, 3, 4]), spacing=np.float32(0.5), origin=[0, 0, 0])
    assert grid.shape == (2, 3, 4) and all(type(n) is int for n in grid.shape)
    assert grid == Grid3D((2, 3, 4), 0.5, (0.0, 0.0, 0.0))
    assert hash(grid) == hash(Grid3D((2, 3, 4), 0.5, (0.0, 0.0, 0.0)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.spacing = 1.0
    with pytest.raises(ValueError, match="spacing"):
        Grid3D((2, 3, 4), 0.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        Grid3D((2, 0, 4), 1.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        Grid2D((2, 3, 4), 1.0, (0.0, 0.0))
    with pytest.raises(TypeError):
        Grid2D((2.5, 3), 1.0, (0.0, 0.0))


def test_grid2d_axes_and_points_order():
    z, x = PLANE_GRID.axes()
    np.testing.assert_allclose(z, [2e-3, 3e-3, 4e-3, 5e-3, 6e-3], atol=1e-15)
    np.testing.assert_allclose(x, np.arange(-3, 4) * 1e-3, atol=1e-15)
    points = PLANE_GRID.points()
    assert points.shape == (5, 7, 2)
    # Indexed [iz, ix]; the last axis is (x, z).
    np.testing.assert_allclose(points[4, 1], [x[1], z[4]])


def test_grid2d_from_extent_spans_the_pixel_centres():
    grid = Grid2D.from_extent(x_range=(-0.03, 0.03), z_range=(0.0, 0.1), spacing=1.0e-4)
    assert grid.shape == (1001, 601)
    assert grid.origin == (0.0, -0.03)
    z, x = grid.axes()
    np.testing.assert_allclose([z[0], z[-1], x[0], x[-1]], [0.0, 0.1, -0.03, 0.03], atol=1e-12)
    np.testing.assert_allclose(x[300], 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="increasing"):
        Grid2D.from_extent(x_range=(0.03, -0.03), z_range=(0.0, 0.1), spacing=1.0e-4)


def test_plane_spec_validates_unit_length_and_orthogonality():
    PlaneSpec(origin=(0, 0, 0.09), u_lateral=(1, 0, 0), u_depth=(0, 0, -1))
    PlaneSpec(origin=(0, 0, 0), u_lateral=(1.0 + 5e-7, 0, 0), u_depth=(0, 1, 0))
    with pytest.raises(ValueError, match="u_lateral must have unit length"):
        PlaneSpec(origin=(0, 0, 0), u_lateral=(1.01, 0, 0), u_depth=(0, 0, -1))
    with pytest.raises(ValueError, match="u_depth must have unit length"):
        PlaneSpec(origin=(0, 0, 0), u_lateral=(1, 0, 0), u_depth=(0, 0, -0.5))
    tilted = (np.sin(1e-3), 0.0, -np.cos(1e-3))
    with pytest.raises(ValueError, match="orthogonal"):
        PlaneSpec(origin=(0, 0, 0), u_lateral=(1, 0, 0), u_depth=tilted)


def test_plane_to_world_and_plane_points():
    plane = PlaneSpec(
        origin=(0.01, -0.02, 0.09),
        u_lateral=(0.6, 0.8, 0.0),
        u_depth=(0.0, 0.0, -1.0),
    )
    np.testing.assert_allclose(plane_to_world(plane, [0.0, 0.0]), plane.origin)
    # 5 mm along the array and 20 mm deep.
    np.testing.assert_allclose(
        plane_to_world(plane, [5e-3, 20e-3]), [0.01 + 3e-3, -0.02 + 4e-3, 0.07], atol=1e-15
    )
    points = plane_points(plane, PLANE_GRID)
    assert points.shape == (5, 7, 3)
    z, x = PLANE_GRID.axes()
    np.testing.assert_allclose(points[4, 1], plane_to_world(plane, [x[1], z[4]]))
    # The embedding is rigid: neighbouring pixels stay one spacing apart.
    np.testing.assert_allclose(
        np.linalg.norm(points[1:] - points[:-1], axis=-1), PLANE_GRID.spacing, rtol=1e-9
    )


def test_sample_volume_reproduces_a_linear_function():
    volume = linear_field(GRID.coords())
    points = interior_points(200)
    sampled = sample_volume(volume, GRID, points, order=1)
    assert sampled.shape == (200,)
    np.testing.assert_allclose(sampled, linear_field(points), rtol=1e-12, atol=1e-12)


def test_sample_volume_handles_channels_and_point_shapes():
    volume = np.stack([linear_field(GRID.coords()), -2.0 * linear_field(GRID.coords())])
    points = interior_points(24).reshape(2, 3, 4, 3)
    sampled = sample_volume(volume, GRID, points)
    assert sampled.shape == (2, 2, 3, 4)
    np.testing.assert_allclose(sampled[1], -2.0 * linear_field(points), rtol=1e-12, atol=1e-12)


def test_sample_volume_fills_outside_the_grid():
    volume = np.full(GRID.shape, 7.0, dtype=np.float32)
    x, y, z = GRID.axes()
    half = 0.5 * GRID.spacing
    points = np.array(
        [
            [x[0], y[0], z[0]],  # corner voxel centre
            [x[0] - 0.8 * half, y[0], z[0]],  # inside the outer face: edge value
            [x[0] - 1.2 * half, y[0], z[0]],  # beyond the outer face
            [x[-1], y[-1] + 1.2 * half, z[-1]],
            [1.0, 1.0, 1.0],
        ]
    )
    sampled = sample_volume(volume, GRID, points, fill=-1.0)
    assert sampled.dtype == np.float32
    np.testing.assert_array_equal(sampled, [7.0, 7.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(sample_volume(volume, GRID, points)[2:], 0.0)


def test_sample_volume_nearest_keeps_labels():
    labels = SeedTree(1).rng("anatomy").integers(0, 5, size=GRID.shape, dtype=np.uint8)
    offsets = SeedTree(1).rng("anatomy", 1).uniform(-0.4, 0.4, size=GRID.shape + (3,))
    sampled = sample_volume(labels, GRID, GRID.coords() + offsets * GRID.spacing, order=0)
    assert sampled.dtype == np.uint8
    np.testing.assert_array_equal(sampled, labels)
    masks = labels > 2
    np.testing.assert_array_equal(sample_volume(masks, GRID, GRID.coords(), order=0), masks)
    # Linear interpolation of integer data gives fractional values.
    assert sample_volume(labels, GRID, GRID.coords(), order=1).dtype.kind == "f"


def test_sample_volume_on_an_imaging_plane():
    plane = PlaneSpec(origin=(0.0, 5.0e-3, 5.5e-3), u_lateral=(1, 0, 0), u_depth=(0, 0, -1))
    plane_grid = Grid2D.from_extent(x_range=(-4e-3, 4e-3), z_range=(0.0, 5e-3), spacing=5e-4)
    points = plane_points(plane, plane_grid)
    sampled = sample_volume(linear_field(GRID.coords()), GRID, points)
    assert sampled.shape == plane_grid.shape
    np.testing.assert_allclose(sampled, linear_field(points), rtol=1e-12, atol=1e-12)


def test_sample_rejects_bad_arguments():
    volume = np.zeros(GRID.shape)
    with pytest.raises(ValueError, match="order"):
        sample_volume(volume, GRID, np.zeros((1, 3)), order=3)
    with pytest.raises(ValueError, match="do not match grid shape"):
        sample_volume(np.zeros((6, 5, 5)), GRID, np.zeros((1, 3)))
    with pytest.raises(ValueError, match=r"points must have shape \(..., 3\)"):
        sample_volume(volume, GRID, np.zeros((4, 2)))


def test_sample_plane_map_reproduces_a_complex_linear_function():
    z, x = PLANE_GRID.axes()
    points = PLANE_GRID.points()

    def field(xz: np.ndarray) -> np.ndarray:
        return (1.0 + 300.0 * xz[..., 0]) + 1j * (2.0 - 500.0 * xz[..., 1])

    values = field(points).astype(np.complex128)
    rng = SeedTree(2).rng("anatomy")
    xz = np.stack([rng.uniform(x[0], x[-1], size=50), rng.uniform(z[0], z[-1], size=50)], axis=-1)
    sampled = sample_plane_map(values, PLANE_GRID, xz)
    assert sampled.dtype == np.complex128
    np.testing.assert_allclose(sampled, field(xz), rtol=1e-12, atol=1e-12)

    stacked = sample_plane_map(np.stack([values, 2.0 * values]), PLANE_GRID, xz.reshape(5, 10, 2))
    assert stacked.shape == (2, 5, 10)
    np.testing.assert_allclose(stacked[1].ravel(), 2.0 * field(xz), rtol=1e-12, atol=1e-12)


def test_sample_plane_map_takes_points_as_x_then_z():
    values = np.zeros(PLANE_GRID.shape)
    values[4, 1] = 1.0
    z, x = PLANE_GRID.axes()
    assert sample_plane_map(values, PLANE_GRID, [x[1], z[4]]) == 1.0
    assert sample_plane_map(values, PLANE_GRID, [x[1], z[4]], order=0) == 1.0
    assert sample_plane_map(values, PLANE_GRID, [x[1], z[4] + 2e-3], fill=-5.0) == -5.0
