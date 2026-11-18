"""Matched-filter decoding of the task state from single power Doppler images.

One power Doppler image is close to a single speckle look (ROADMAP D-022), so
a decoder cannot afford to estimate many weights from a few tens of trials. The
matched filter uses one template, the expected task-minus-rest difference of
the images, as its weights:

    score_i = sum_p t_p (y_ip - m_p) / v_p

with `m` the mean and `v` the pooled within-state variance of the training
images per pixel. For Gaussian pixels with independent noise of variance `v_p`
and a task shift `t` this is the likelihood ratio, and its error rate is
`Phi(-d / 2)` with `d^2 = sum_p t_p^2 / v_p` (linear discriminant analysis with
a diagonal covariance and a given mean difference; Fisher 1936). The decision
threshold is midway between the mean training scores of the two states.

The template is estimated on the training folds only: either the difference of
the training condition means (`mean_difference_template`) or any function of
the training images and labels the caller supplies, for example the predicted
image difference of two reconstructions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Maps (training images (n, n_pix), training labels (n,)) to a template, shape (n_pix,).
TemplateFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
# Smallest pooled variance a pixel is given, as a fraction of the mean variance of the
# image: keeps a pixel without fluctuation in the training set from taking over the score.
_VARIANCE_FLOOR = 1.0e-6


def mean_difference_template(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Task mean minus rest mean of the images, shape (n_pix,) (labels: 1 = task, 0 = rest)."""
    labels = np.asarray(labels)
    return images[labels == 1].mean(axis=0) - images[labels == 0].mean(axis=0)


def pooled_variance(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Within-state variance per pixel, pooled over the two states; shape (n_pix,)."""
    labels = np.asarray(labels)
    task, rest = images[labels == 1], images[labels == 0]
    pooled = ((task - task.mean(axis=0)) ** 2).sum(axis=0) + ((rest - rest.mean(axis=0)) ** 2).sum(
        axis=0
    )
    variance = pooled / (images.shape[0] - 2)
    return np.maximum(variance, _VARIANCE_FLOOR * variance.mean())


def matched_filter_predict(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Classify test images with the matched filter of a template (module docstring).

    train, test: images as rows, shape (n, n_pix); train_labels: 1 = task, 0 = rest;
    template: expected task-minus-rest image, shape (n_pix,).

    Returns the predicted labels of the test images, shape (n_test,), 0 or 1.
    """
    train_labels = np.asarray(train_labels)
    weights = template / pooled_variance(train, train_labels)
    centre = train.mean(axis=0)
    train_scores = (train - centre) @ weights
    threshold = 0.5 * (
        train_scores[train_labels == 1].mean() + train_scores[train_labels == 0].mean()
    )
    return ((test - centre) @ weights > threshold).astype(train_labels.dtype)


def matched_filter_accuracy(
    images: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    template_function: TemplateFunction = mean_difference_template,
) -> float:
    """Cross-validated accuracy of the matched-filter decoder.

    For each fold the template, the variances and the threshold come from the
    other folds, and the fold's images are classified. Chance is 0.5 for balanced
    states.

    images: one image per trial, shape (n, ...); labels: 1 = task, 0 = rest, shape (n,);
    folds: fold index of each trial (`recon.metrics.stratified_folds`);
    template_function: see `TemplateFunction`.

    Returns the fraction of trials classified correctly when held out.
    """
    features = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    labels = np.asarray(labels)
    correct = 0
    for fold in np.unique(folds):
        held_out = folds == fold
        template = template_function(features[~held_out], labels[~held_out])
        predicted = matched_filter_predict(
            features[~held_out], labels[~held_out], features[held_out], template
        )
        correct += int(np.sum(predicted == labels[held_out]))
    return correct / labels.shape[0]
