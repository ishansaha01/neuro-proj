"""Phantom builders: coverage maps and fraction volumes."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance


def test_radial_distance_is_measured_from_the_given_point(small_grid):
    distance = radial_distance(small_grid, (0.001, -0.003))
    assert distance.shape == small_grid.shape[:2]
    assert distance.min() == pytest.approx(0.0, abs=1e-12)


def test_disc_coverage_approximates_the_disc_area(small_grid):
    coverage = disc_coverage(small_grid, 0.05)
    assert coverage.min() == 0.0 and coverage.max() == 1.0
    assert coverage.sum() * small_grid.spacing**2 == pytest.approx(np.pi * 0.05**2, rel=0.005)


def test_layers_replace_what_is_underneath_and_fractions_sum_to_one(small_grid):
    water = disc_coverage(small_grid, 0.06)
    bone = disc_coverage(small_grid, 0.01, center=(0.02, 0.0))
    fractions = layered_fractions(small_grid, [("water", water), ("cortical_bone", bone)])
    assert fractions.shape == (len(BASIS_MATERIALS), *small_grid.shape)
    assert fractions.dtype == np.float32
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1e-6)
    ix, iy = np.unravel_index(np.argmax(bone), bone.shape)
    assert fractions[BASIS_MATERIALS.index("cortical_bone"), ix, iy, 0] == 1.0
    assert fractions[BASIS_MATERIALS.index("water"), ix, iy, 0] == 0.0
    assert fractions[BASIS_MATERIALS.index("air"), 0, 0, 0] == 1.0
    np.testing.assert_array_equal(fractions[..., 0], fractions[..., -1])
