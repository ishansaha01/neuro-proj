"""medium_greens and its bilinear interpolation of the bank."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.fus.greens import bilinear_weights, check_same_sampling, medium_greens
from neurosim.fus.reference import reference_greens


def _random_bank(water, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=water.B.shape) + 1j * rng.normal(size=water.B.shape)
    return replace(water, B=values.astype(np.complex64), backend="test")


def _points_inside(roi, n: int, seed: int = 1) -> np.ndarray:
    """Random points inside the hull of the ROI nodes, columns (x, z)."""
    rng = np.random.default_rng(seed)
    z, x = roi.axes()
    return np.stack((rng.uniform(x[0], x[-1], n), rng.uniform(z[0], z[-1], n)), axis=1)


def test_interpolation_matches_bank_interp(water):
    bank = _random_bank(water)
    xz = _points_inside(bank.roi, 40)
    nodes, weights = bilinear_weights(bank.roi, xz)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=1e-6)
    flat = bank.B.reshape(*bank.B.shape[:2], -1)
    interpolated = np.einsum("fecn,cn->fen", flat[:, :, nodes], weights)
    np.testing.assert_allclose(interpolated, bank.interp(xz), rtol=1e-4, atol=1e-5)


def test_nodes_are_reproduced_exactly(water):
    bank = _random_bank(water)
    nodes, weights = bilinear_weights(bank.roi, bank.roi.points().reshape(-1, 2))
    best = np.argmax(weights, axis=0)
    np.testing.assert_array_equal(nodes[best, np.arange(nodes.shape[1])], np.arange(nodes.shape[1]))
    np.testing.assert_allclose(weights.max(axis=0), 1.0)


def test_edge_value_is_used_up_to_one_spacing_outside(water):
    roi = water.roi
    z, x = roi.axes()
    outside = np.array([[x[0] - 0.9 * roi.spacing, z[3]], [x[2], z[-1] + 0.5 * roi.spacing]])
    on_edge = np.array([[x[0], z[3]], [x[2], z[-1]]])
    bank = _random_bank(water)
    flat = bank.B.reshape(*bank.B.shape[:2], -1)

    def sample(points):
        nodes, weights = bilinear_weights(roi, points)
        return np.einsum("fecn,cn->fen", flat[:, :, nodes], weights)

    np.testing.assert_allclose(sample(outside), sample(on_edge), rtol=1e-6)
    with pytest.raises(ValueError, match="outside the bank ROI"):
        bilinear_weights(roi, np.array([[x[0] - 1.1 * roi.spacing, z[3]]]))


def test_medium_greens_is_ratio_times_reference(water):
    bank = _random_bank(water)
    xz = _points_inside(bank.roi, 9)
    reference = reference_greens(bank.transducer, xz, bank.freqs, bank.c_ref)
    greens = medium_greens(bank, xz)
    assert greens.shape == reference.shape and greens.dtype == np.complex128
    np.testing.assert_allclose(greens, reference * bank.interp(xz), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(medium_greens(water, xz), reference, rtol=1e-6)


def test_frequency_slice_selects_frequencies(water):
    xz = _points_inside(water.roi, 5)
    bank = _random_bank(water)
    np.testing.assert_allclose(
        medium_greens(bank, xz, slice(4, 11)), medium_greens(bank, xz)[4:11], rtol=1e-9
    )


def test_banks_must_share_array_and_frequencies(water):
    check_same_sampling(water, _random_bank(water))
    other_freqs = replace(water, freqs=water.freqs + 1.0)
    with pytest.raises(ValueError, match="frequency grids"):
        check_same_sampling(water, other_freqs)
    other_array = replace(water, transducer=replace(water.transducer, center_frequency=3.0e6))
    with pytest.raises(ValueError, match="different arrays"):
        check_same_sampling(water, other_array)
