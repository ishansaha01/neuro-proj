"""Matched-filter decoder (`recon.decoding`)."""

from __future__ import annotations

import numpy as np

from neurosim.common.seeding import SeedTree
from neurosim.recon.decoding import (
    matched_filter_accuracy,
    matched_filter_predict,
    mean_difference_template,
    pooled_variance,
)
from neurosim.recon.metrics import stratified_folds


def _data(n: int, shift: float, seed: int = 0):
    rng = SeedTree(seed).rng("recon_sampling", 77)
    labels = np.arange(n) % 2
    images = rng.standard_normal((n, 3, 4))
    images[labels == 1, 1, 2] += shift
    return images, labels, rng


def test_template_and_variance():
    images, labels, _ = _data(400, 3.0)
    flat = images.reshape(400, -1)
    template = mean_difference_template(flat, labels)
    assert template.shape == (12,)
    assert np.argmax(template) == 6 and abs(template[6] - 3.0) < 0.4
    variance = pooled_variance(flat, labels)
    # The shift is a class effect: the within-state variance of that pixel stays near 1.
    np.testing.assert_allclose(variance, 1.0, atol=0.3)
    constant = np.concatenate((flat, np.zeros((400, 1))), axis=1)
    assert pooled_variance(constant, labels)[-1] > 0.0


def test_a_strong_shift_is_decoded_and_none_is_not():
    images, labels, rng = _data(200, 6.0)
    folds = stratified_folds(labels, 5, rng.permutation(200))
    assert matched_filter_accuracy(images, labels, folds) > 0.97
    null_images, null_labels, _ = _data(200, 0.0, seed=1)
    assert 0.3 < matched_filter_accuracy(null_images, null_labels, folds) < 0.7


def test_predict_uses_the_given_template_and_training_threshold():
    images, labels, _ = _data(300, 4.0)
    flat = images.reshape(300, -1)
    template = np.zeros(12)
    template[6] = 1.0
    predicted = matched_filter_predict(flat[:200], labels[:200], flat[200:], template)
    assert np.mean(predicted == labels[200:]) > 0.9
    # The opposite template inverts the decisions: the sign of the template matters.
    inverted = matched_filter_predict(flat[:200], labels[:200], flat[200:], -template)
    assert np.mean(inverted == labels[200:]) < 0.1


def test_custom_template_function_sees_training_folds_only():
    images, labels, rng = _data(60, 5.0)
    folds = stratified_folds(labels, 3, rng.permutation(60))
    sizes = []

    def template(train: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
        sizes.append(train.shape)
        return mean_difference_template(train, train_labels)

    accuracy = matched_filter_accuracy(images, labels, folds, template)
    assert sizes == [(40, 12)] * 3
    assert accuracy == matched_filter_accuracy(images, labels, folds)
