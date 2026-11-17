import numpy as np
import pytest

from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.head.fields import SphereFields, grid_random_field

GRID = Grid3D(shape=(24, 20, 16), spacing=1.0e-3, origin=(0.0, 0.0, 0.0))


def test_grid_field_is_normalised_float32_and_reproducible():
    field = grid_random_field(SeedTree(1).rng("cbv"), GRID, 3.0e-3)
    assert field.shape == GRID.shape
    assert field.dtype == np.float32
    assert field.mean() == pytest.approx(0.0, abs=1e-5)
    assert field.std() == pytest.approx(1.0, abs=1e-5)
    again = grid_random_field(SeedTree(1).rng("cbv"), GRID, 3.0e-3)
    np.testing.assert_array_equal(field, again)


def test_grid_field_correlation_follows_the_correlation_length():
    grid = Grid3D(shape=(48, 48, 48), spacing=1.0e-3, origin=(0.0, 0.0, 0.0))
    length = 3.0e-3
    field = grid_random_field(SeedTree(2).rng("cbv"), grid, length).astype(np.float64)
    lag = 3
    correlation = np.mean(field * np.roll(field, lag, axis=0))
    expected = np.exp(-((lag * grid.spacing) ** 2) / (2.0 * length**2))
    assert correlation == pytest.approx(expected, abs=0.08)


def unit_vectors(n: int) -> np.ndarray:
    vectors = SeedTree(0).rng("anatomy", 99).standard_normal((n, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_sphere_fields_shape_dtype_and_unit_variance():
    fields = SphereFields.draw(SeedTree(0).rng("anatomy"), (0.5,) * 40, 32)
    values = fields.evaluate(unit_vectors(500))
    assert values.shape == (40, 500)
    assert values.dtype == np.float32
    assert values.var() == pytest.approx(1.0, abs=0.15)


def test_sphere_fields_are_smooth_and_independent_of_point_order():
    fields = SphereFields.draw(SeedTree(0).rng("anatomy"), (0.5, 0.5), 32)
    directions = unit_vectors(50)
    nearby = directions + 1.0e-3 * np.roll(directions, 1, axis=0)
    nearby /= np.linalg.norm(nearby, axis=1, keepdims=True)
    values = fields.evaluate(directions)
    assert np.max(np.abs(fields.evaluate(nearby) - values)) < 0.05
    np.testing.assert_allclose(fields.evaluate(directions[::-1]), values[:, ::-1], atol=1e-5)
