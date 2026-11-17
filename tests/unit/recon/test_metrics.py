"""Evaluation metrics on cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.recon.gauge import gauge_free_basis, remove_gauge
from neurosim.recon.metrics import (
    activation_cnr,
    activation_dice,
    cbv_correlation,
    decoding_accuracy,
    delay_error,
    dice_overlap,
    interval_coverage,
    stratified_folds,
    task_t_map,
)


def _rng(sub: int) -> np.random.Generator:
    return SeedTree(0).rng("recon_sampling", 9000, sub)


def _task_images(rng: np.random.Generator, n: int = 40, gain: float = 5.0):
    """Noise images with a response in the top-left 3 x 3 pixels of the task trials."""
    labels = np.tile([0, 1], n // 2)
    activation = np.zeros((8, 8))
    activation[:3, :3] = 1.0
    images = 10.0 + rng.standard_normal((n, 8, 8))
    images += gain * labels[:, None, None] * activation
    return images, labels, activation


def test_stratified_folds_balance_the_classes():
    labels = np.array([0] * 12 + [1] * 8)
    folds = stratified_folds(labels, 4, _rng(0).permutation(20))
    for fold in range(4):
        assert np.sum(labels[folds == fold] == 0) == 3
        assert np.sum(labels[folds == fold] == 1) == 2
    with pytest.raises(ValueError):
        stratified_folds(np.array([0, 0, 0, 1]), 2, np.arange(4))


def test_decoder_separates_separable_classes_and_not_shuffled_ones():
    rng = _rng(1)
    images, labels, _ = _task_images(rng, n=60)
    folds = stratified_folds(labels, 5, rng.permutation(60))
    assert decoding_accuracy(images, labels, folds, 10) == 1.0
    shuffled = rng.permutation(labels)
    accuracies = [
        decoding_accuracy(
            images,
            shuffled := rng.permutation(labels),
            stratified_folds(shuffled, 5, rng.permutation(60)),
            10,
        )
        for _ in range(20)
    ]
    assert abs(np.mean(accuracies) - 0.5) < 0.1
    assert shuffled.shape == labels.shape


def test_t_map_is_the_welch_statistic():
    images = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 10.0]).reshape(6, 1, 1)
    labels = np.array([0, 0, 0, 1, 1, 1])
    # rest: mean 2, var 1; task: mean 7, var 7; t = 5 / sqrt(1/3 + 7/3)
    np.testing.assert_allclose(task_t_map(images, labels), [[5.0 / np.sqrt(8.0 / 3.0)]])
    assert task_t_map(np.ones((4, 2, 2)), np.array([0, 0, 1, 1])).max() == 0.0


def test_dice_overlap_known_values():
    first = np.array([1, 1, 0, 0], dtype=bool)
    assert dice_overlap(first, first) == 1.0
    assert dice_overlap(first, ~first) == 0.0
    assert dice_overlap(first, np.array([1, 0, 1, 0], dtype=bool)) == 0.5
    assert dice_overlap(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)) == 0.0


def test_activation_dice_finds_a_strong_response():
    images, labels, activation = _task_images(_rng(2))
    mask = np.ones((8, 8), dtype=bool)
    assert activation_dice(images, labels, activation, mask, 5.0, 0.5) == 1.0
    assert activation_dice(images, 1 - labels, activation, mask, 5.0, 0.5) == 0.0


def test_cbv_correlation_is_scale_invariant_and_masked():
    rng = _rng(3)
    truth = rng.uniform(size=(3, 6, 6))
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    estimate = 3.0 * truth + 1.0
    estimate[:, ~mask] = rng.uniform(size=(3, int((~mask).sum())))
    np.testing.assert_allclose(cbv_correlation(estimate, truth, mask), 1.0)
    np.testing.assert_allclose(cbv_correlation(-estimate, truth, mask), -1.0)


def test_cnr_is_the_d_prime_of_the_region_ratio():
    labels = np.array([0, 0, 1, 1])
    activation = np.array([[1.0, 0.0]])
    mask = np.ones((1, 2), dtype=bool)
    # active / reference ratios: rest 1.0 and 1.2, task 1.9 and 2.1 -> (2.0 - 1.1) / sqrt(0.02)
    images = np.array([[[1.0, 1.0]], [[2.4, 2.0]], [[1.9, 1.0]], [[4.2, 2.0]]])
    np.testing.assert_allclose(
        activation_cnr(images, labels, activation, mask, 0.5, 0.01), 0.9 / np.sqrt(0.02)
    )
    with pytest.raises(ValueError):
        activation_cnr(images, labels, np.ones((1, 2)), mask, 0.5, 0.01)


def test_coverage_of_gaussian_posterior_samples_is_nominal():
    rng = _rng(4)
    n_values, n_samples = 40000, 4
    centre = rng.standard_normal(n_values)
    scale = rng.uniform(0.5, 2.0, n_values)
    truth = centre + scale * rng.standard_normal(n_values)
    samples = centre + scale * rng.standard_normal((n_samples, n_values))
    levels = np.array([0.5, 0.8, 0.9])
    coverage = interval_coverage(samples, truth, levels)
    np.testing.assert_allclose(coverage, levels, atol=0.01)  # binomial error at most 0.0025
    narrow = interval_coverage(centre + 0.5 * (samples - centre), truth, levels)
    assert np.all(narrow < levels - 0.1)
    with pytest.raises(ValueError):
        interval_coverage(samples[:1], truth, levels)


def test_delay_error_ignores_the_gauge_directions():
    element_x = np.linspace(-0.01, 0.01, 16)
    rng = _rng(5)
    truth = 1e-7 * rng.standard_normal(16)
    shifted = truth + 3e-7 + 2e-5 * element_x
    assert delay_error(shifted, truth, element_x, "mean_tilt") < 1e-20
    assert delay_error(shifted, truth, element_x, "mean") > 1e-8
    bump = np.zeros(16)
    bump[3] = 1e-7
    residual = remove_gauge(bump, element_x, "mean")
    np.testing.assert_allclose(
        delay_error(truth + bump, truth, element_x, "mean"), np.sqrt(np.mean(residual**2))
    )


@pytest.mark.parametrize("gauge, n_gauge", [("mean", 1), ("mean_tilt", 2)])
def test_gauge_free_basis_is_orthonormal_and_orthogonal_to_the_gauge(gauge, n_gauge):
    element_x = np.linspace(-0.01, 0.01, 16)
    basis = gauge_free_basis(element_x, gauge)
    assert basis.shape == (16, 16 - n_gauge)
    np.testing.assert_allclose(basis.T @ basis, np.eye(16 - n_gauge), atol=1e-12)
    np.testing.assert_allclose(basis.T @ np.ones(16), 0.0, atol=1e-12)
    if gauge == "mean_tilt":
        np.testing.assert_allclose(basis.T @ element_x, 0.0, atol=1e-12)
    # Smoothest first: mode k has k + n_gauge sign changes, give or take one.
    for k in range(8):
        changes = int(np.sum(np.diff(np.sign(basis[:, k])) != 0))
        assert abs(changes - (k + n_gauge)) <= 1
    with pytest.raises(ValueError):
        gauge_free_basis(element_x, "none")
