import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from neurosim.common.grid import Grid2D, sample_plane_map
from neurosim.fus.backends.eikonal import (
    bilinear_stencil,
    gather_bilinear,
    solve_eikonal,
)

TOLERANCE, MAX_SWEEPS = 1.0e-6, 120


def half_spaces(grid: Grid2D, interface_row: int, upper: float, lower: float) -> np.ndarray:
    """Slowness [s/m] of two half-spaces: speed `upper` in rows < interface_row, `lower` below."""
    rows = np.arange(grid.shape[0])[:, None]
    return np.where(rows < interface_row, 1.0 / upper, 1.0 / lower) * np.ones(grid.shape)


def fermat_two_layers(offset: float, d1: float, c1: float, d2: float, c2: float) -> float:
    """Least time through two layers of thickness d1, d2 for a lateral offset [s]."""

    def time(x: float) -> float:
        return np.hypot(x, d1) / c1 + np.hypot(offset - x, d2) / c2

    result = minimize_scalar(time, bounds=(0.0, offset), method="bounded", options={"xatol": 1e-12})
    return result.fun


def node_times(field, source: int) -> np.ndarray:
    """Travel time [s] of one source at every grid node, shape (nz, nx)."""
    points = field.grid.points().reshape(-1, 2)
    index = np.full(points.shape[0], source)
    return field.travel_time(index, points).reshape(field.grid.shape)


def test_homogeneous_medium_is_exact_for_sources_on_and_off_nodes(grid):
    sources = np.array([[0.0, 0.0], [1.37e-3, 0.0], [-2.05e-3, 0.0]])
    field = solve_eikonal(np.full(grid.shape, 1.0 / 1500.0), grid, sources, TOLERANCE, MAX_SWEEPS)
    assert field.excess_time.shape == (*grid.shape, 3)
    points = grid.points()
    for index, source in enumerate(sources):
        distance = np.hypot(points[..., 0] - source[0], points[..., 1] - source[1])
        np.testing.assert_allclose(node_times(field, index), distance / 1500.0, atol=1e-13)


def test_travel_time_through_an_interface_matches_fermat(grid, spacing):
    # Interface on the pixel face between rows 19 and 20, z = 19.5 cells.
    slowness = half_spaces(grid, 20, 1500.0, 2800.0)
    field = solve_eikonal(slowness, grid, np.array([[0.0, 0.0]]), TOLERANCE, MAX_SWEEPS)
    times = node_times(field, 0)
    z, x = grid.axes()
    for row, column in ((40, 20), (40, 30), (34, 38), (30, 4)):
        expected = fermat_two_layers(
            abs(x[column]), 19.5 * spacing, 1500.0, z[row] - 19.5 * spacing, 2800.0
        )
        # First order: a fraction of the time to cross one cell.
        assert times[row, column] == pytest.approx(expected, abs=0.1 * spacing / 1500.0)


def test_error_falls_with_the_grid_spacing():
    errors = []
    for spacing in (0.4e-3, 0.2e-3):
        grid = Grid2D.from_extent((-8.0e-3, 8.0e-3), (0.0, 12.0e-3), spacing)
        row = round(4.0e-3 / spacing)
        slowness = half_spaces(grid, row, 1500.0, 2800.0)
        field = solve_eikonal(slowness, grid, np.array([[-6.0e-3, 0.0]]), TOLERANCE, MAX_SWEEPS)
        time = field.travel_time(np.array([0]), np.array([[6.0e-3, 12.0e-3]]))[0]
        depth = (row - 0.5) * spacing
        expected = fermat_two_layers(12.0e-3, depth, 1500.0, 12.0e-3 - depth, 2800.0)
        errors.append(abs(time - expected))
    assert errors[1] < 0.75 * errors[0]


def test_head_wave_is_the_first_arrival_along_a_fast_half_space(spacing):
    grid = Grid2D.from_extent((-3.0e-3, 15.0e-3), (0.0, 6.0e-3), spacing)
    slowness = half_spaces(grid, 10, 1500.0, 3000.0)  # interface at z = 9.5 cells
    field = solve_eikonal(slowness, grid, np.array([[0.0, 0.0]]), TOLERANCE, MAX_SWEEPS)
    point = np.array([[12.0e-3, 1.5e-3]])
    depth = 9.5 * spacing
    cosine = np.sqrt(1.0 - (1500.0 / 3000.0) ** 2)
    head_wave = 12.0e-3 / 3000.0 + (2.0 * depth - 1.5e-3) * cosine / 1500.0
    direct = np.hypot(12.0e-3, 1.5e-3) / 1500.0
    assert head_wave < direct
    assert field.travel_time(np.array([0]), point)[0] == pytest.approx(head_wave, rel=5e-3)


def test_sources_of_a_batch_do_not_interact(grid):
    slowness = half_spaces(grid, 15, 1540.0, 2500.0)
    sources = np.array([[-1.2e-3, 0.0], [0.33e-3, 0.0]])
    together = solve_eikonal(slowness, grid, sources, TOLERANCE, MAX_SWEEPS)
    for index in range(2):
        alone = solve_eikonal(slowness, grid, sources[index : index + 1], TOLERANCE, MAX_SWEEPS)
        np.testing.assert_array_equal(together.excess_time[:, :, index], alone.excess_time[:, :, 0])


def test_same_input_gives_identical_output(grid):
    slowness = half_spaces(grid, 15, 1540.0, 2500.0)
    sources = np.array([[0.4e-3, 0.0]])
    first = solve_eikonal(slowness, grid, sources, TOLERANCE, MAX_SWEEPS)
    second = solve_eikonal(slowness, grid, sources, TOLERANCE, MAX_SWEEPS)
    np.testing.assert_array_equal(first.excess_time, second.excess_time)
    assert first.n_sweeps == second.n_sweeps >= 4


def test_too_few_sweeps_raise(grid):
    slowness = half_spaces(grid, 15, 1540.0, 2500.0)
    with pytest.raises(RuntimeError, match="max_sweeps"):
        solve_eikonal(slowness, grid, np.array([[0.0, 0.0]]), TOLERANCE, 4)


def test_slowness_must_match_the_grid(grid):
    with pytest.raises(ValueError, match="grid shape"):
        solve_eikonal(np.ones((3, 3)), grid, np.array([[0.0, 0.0]]), TOLERANCE, MAX_SWEEPS)


def test_bilinear_gather_matches_the_common_sampler(grid):
    z, x = grid.axes()
    values = np.sin(400.0 * z)[:, None] * np.cos(300.0 * x)[None, :] + 2.0
    # Inside the grid, on nodes, and between the outermost nodes and the outer faces.
    xz = np.array(
        [[0.11e-3, 3.33e-3], [x[5], z[7]], [x[0] - 0.1e-3, z[-1] + 0.1e-3], [x[-1], z[0]]]
    )
    stencil = bilinear_stencil(grid, xz)
    expected = sample_plane_map(values, grid, xz, order=1)
    np.testing.assert_allclose(gather_bilinear(values[:, :, None], stencil)[:, 0], expected)
    stacked = np.stack((values, 2.0 * values), axis=-1)
    channel = np.array([0, 1, 1, 0])
    np.testing.assert_allclose(
        gather_bilinear(stacked, stencil, channel), expected * (1 + channel)
    )
    pairs = np.stack((stacked, -stacked), axis=-1)  # (nz, nx, 2 channels, 2 components)
    both = gather_bilinear(pairs, stencil, channel)
    np.testing.assert_allclose(both[:, 0], expected * (1 + channel))
    np.testing.assert_allclose(both[:, 1], -expected * (1 + channel))
