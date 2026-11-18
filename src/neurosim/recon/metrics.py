"""Evaluation metrics of the reconstruction stage (README, stage 3). Pure functions on arrays.

All image metrics are invariant to a positive scaling of the images, so power
Doppler images (arbitrary units) and CBV reconstructions can be compared with the
same code. Run time is measured by `recon.experiment`, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.types import GreensBank
from neurosim.recon.gauge import remove_gauge

# Relative ridge on the within-class covariance of the decoder, as a fraction of its
# mean eigenvalue: keeps the discriminant finite when a principal component has (almost)
# no within-class variance. Numerical safeguard, far below any real variance.
_COVARIANCE_RIDGE = 1.0e-6


def stratified_folds(labels: np.ndarray, n_folds: int, order: np.ndarray) -> np.ndarray:
    """Assign every trial to a cross-validation fold, balancing the classes.

    labels: class of each trial (two values), shape (n,).
    n_folds: number of folds; every class needs at least this many trials.
    order: a permutation of ``0 .. n - 1`` (from a seeded generator); trials are dealt
        to the folds in this order, class by class.

    Returns the fold index of each trial, int, shape (n,).
    """
    labels = np.asarray(labels)
    folds = np.empty(labels.shape[0], dtype=np.int64)
    for value in np.unique(labels):
        members = np.asarray(order)[labels[order] == value]
        if members.shape[0] < n_folds:
            raise ValueError(
                f"class {value} has {members.shape[0]} trials, fewer than {n_folds} folds"
            )
        folds[members] = np.arange(members.shape[0]) % n_folds
    return folds


def _pca_lda_predict(
    train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, n_components: int
) -> np.ndarray:
    """Fit PCA + two-class LDA on the training trials and classify the test trials."""
    mean = train.mean(axis=0)
    _, _, components = np.linalg.svd(train - mean, full_matrices=False)
    components = components[: min(n_components, train.shape[0] - 2)]
    scores, test_scores = (train - mean) @ components.T, (test - mean) @ components.T
    classes = np.unique(train_labels)
    centroids = np.stack([scores[train_labels == value].mean(axis=0) for value in classes])
    residuals = scores - centroids[np.searchsorted(classes, train_labels)]
    covariance = residuals.T @ residuals / (scores.shape[0] - classes.shape[0])
    covariance += (
        _COVARIANCE_RIDGE * np.trace(covariance) / covariance.shape[0] * np.eye(covariance.shape[0])
    )
    direction = np.linalg.solve(covariance, centroids[1] - centroids[0])
    threshold = 0.5 * (centroids[0] + centroids[1]) @ direction
    return classes[(test_scores @ direction > threshold).astype(int)]


def decoding_accuracy(
    images: np.ndarray, labels: np.ndarray, folds: np.ndarray, n_components: int
) -> float:
    """Cross-validated accuracy of decoding the task label from single images.

    A linear decoder after Norman et al. (2021, Neuron, "Single-trial decoding of
    movement intentions using functional ultrasound neuroimaging"), who classify
    power Doppler images with PCA followed by linear discriminant analysis: the
    training images are centred and projected on their leading principal
    components, a two-class LDA with pooled covariance and equal priors is fitted,
    and the held-out images are classified. Chance is 0.5 for balanced classes.

    images: one image per trial, shape (n, ...), any units.
    labels: task state of each trial (two values), shape (n,).
    folds: fold index of each trial (`stratified_folds`), shape (n,).
    n_components: principal components kept (at most ``n_train - 2``).

    Returns the fraction of trials classified correctly when held out.
    """
    features = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    labels = np.asarray(labels)
    correct = 0
    for fold in np.unique(folds):
        held_out = folds == fold
        predicted = _pca_lda_predict(
            features[~held_out], labels[~held_out], features[held_out], n_components
        )
        correct += int(np.sum(predicted == labels[held_out]))
    return correct / labels.shape[0]


def task_t_map(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Welch t statistic of task against rest images, pixel by pixel.

    images: shape (n, nz, nx); labels: 1 = task, 0 = rest, shape (n,).
    Returns shape (nz, nx); positive where the task images are brighter. A pixel
    without variance in either state gets 0.
    """
    task, rest = images[np.asarray(labels) == 1], images[np.asarray(labels) == 0]
    variance = task.var(axis=0, ddof=1) / task.shape[0] + rest.var(axis=0, ddof=1) / rest.shape[0]
    difference = task.mean(axis=0) - rest.mean(axis=0)
    statistic = np.zeros_like(difference, dtype=np.float64)
    np.divide(difference, np.sqrt(variance), out=statistic, where=variance > 0)
    return statistic


def dice_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Dice coefficient ``2 |A and B| / (|A| + |B|)`` of two boolean masks; 0 if both are empty."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    total = first.sum() + second.sum()
    return float(2.0 * np.logical_and(first, second).sum() / total) if total > 0 else 0.0


def activation_dice(
    images: np.ndarray,
    labels: np.ndarray,
    activation: np.ndarray,
    brain_mask: np.ndarray,
    t_threshold: float,
    activation_threshold: float,
) -> float:
    """Overlap of the detected activation with the true one.

    Detected: brain pixels whose task-minus-rest Welch t statistic exceeds
    `t_threshold`. True: brain pixels whose activation pattern exceeds
    `activation_threshold`.

    images: shape (n, nz, nx); labels: shape (n,); activation: pattern in [0, 1],
    shape (nz, nx); brain_mask: bool, shape (nz, nx).
    """
    detected = (task_t_map(images, labels) > t_threshold) & brain_mask
    return dice_overlap(detected, (activation > activation_threshold) & brain_mask)


def cbv_correlation(estimate: np.ndarray, truth: np.ndarray, brain_mask: np.ndarray) -> float:
    """Mean over trials of the Pearson correlation of estimate and truth inside the brain.

    estimate, truth: shape (n, nz, nx); brain_mask: bool, shape (nz, nx).
    """
    correlations = [
        np.corrcoef(image[brain_mask], reference[brain_mask])[0, 1]
        for image, reference in zip(estimate, truth)
    ]
    return float(np.mean(correlations))


def activation_cnr(
    images: np.ndarray,
    labels: np.ndarray,
    activation: np.ndarray,
    brain_mask: np.ndarray,
    activation_threshold: float,
    reference_threshold: float,
) -> float:
    """Contrast-to-noise ratio of the task response in the truly activated region.

    Per image, the mean over the activated brain pixels (activation above
    `activation_threshold`) divided by the mean over the reference brain pixels
    (activation below `reference_threshold`), which removes global gain changes.
    The CNR is the task-minus-rest difference of that ratio over its pooled
    within-state standard deviation (a single-trial d-prime).

    Other arguments as for `activation_dice`. Returns a dimensionless number; 0
    means no detectable response.
    """
    active = (activation > activation_threshold) & brain_mask
    reference = (activation < reference_threshold) & brain_mask
    if not active.any() or not reference.any():
        raise ValueError("the activated or the reference region is empty")
    ratio = images[:, active].mean(axis=1) / images[:, reference].mean(axis=1)
    task, rest = ratio[np.asarray(labels) == 1], ratio[np.asarray(labels) == 0]
    pooled = np.sqrt(
        ((task.shape[0] - 1) * task.var(ddof=1) + (rest.shape[0] - 1) * rest.var(ddof=1))
        / (task.shape[0] + rest.shape[0] - 2)
    )
    return float((task.mean() - rest.mean()) / pooled)


def interval_coverage(samples: np.ndarray, truth: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fraction of true values inside central intervals built from posterior samples.

    For `K` samples with mean `m` and standard deviation `s` (ddof 1) the interval
    at nominal level `p` is ``m +- t_{K-1}((1 + p) / 2) * s * sqrt(1 + 1 / K)``: the
    Student prediction interval for one more draw from the same Gaussian. If the
    truth is such a draw (a calibrated Gaussian posterior) the coverage equals `p`
    exactly for any ``K >= 2``, so a handful of samples suffices; for non-Gaussian
    posteriors it is approximate, which is why the caller passes values in the
    prior's model space, where CBV is closest to Gaussian.

    samples: posterior samples, shape (K, ...), K >= 2.
    truth: true values, shape (...).
    levels: nominal levels in (0, 1), shape (n_levels,).

    Returns the empirical coverage at each level, shape (n_levels,).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n_samples = samples.shape[0]
    if n_samples < 2:
        raise ValueError("coverage needs at least two posterior samples")
    spread = samples.std(axis=0, ddof=1) * np.sqrt(1.0 + 1.0 / n_samples)
    deviation = np.abs(np.asarray(truth, dtype=np.float64) - samples.mean(axis=0))
    quantiles = student_t.ppf(0.5 * (1.0 + np.asarray(levels, dtype=np.float64)), n_samples - 1)
    return np.array([np.mean(deviation <= quantile * spread) for quantile in quantiles])


def delay_error(
    estimate: np.ndarray, truth: np.ndarray, element_x: np.ndarray, gauge: str
) -> float:
    """Rms difference of two delay profiles after removing the gauge directions [s].

    estimate, truth: per-element delays [s], shape (n_elements,).
    element_x: lateral element positions [m], shape (n_elements,).
    gauge: "mean" or "mean_tilt" (`recon.gauge`).
    """
    residual = remove_gauge(np.asarray(estimate) - np.asarray(truth), element_x, gauge)
    return float(np.sqrt(np.mean(residual**2)))


def map_correlation(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation of two maps over the pixels of a mask.

    estimate, truth: shape (nz, nx); mask: bool, shape (nz, nx). Invariant to a
    positive scaling and an offset of either map; 0 if either map is constant.
    """
    first, second = np.asarray(estimate)[mask], np.asarray(truth)[mask]
    if first.std() == 0.0 or second.std() == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def half_maximum_dice(
    estimate: np.ndarray, truth_region: np.ndarray, mask: np.ndarray, fraction: float
) -> float:
    """Dice overlap of the region where a map exceeds a fraction of its maximum with a true region.

    Detected: pixels of `mask` where `estimate > fraction * max(estimate over mask)`;
    nothing is detected if that maximum is not positive.

    estimate: shape (nz, nx); truth_region, mask: bool, shape (nz, nx); fraction in (0, 1).
    """
    mask = np.asarray(mask, dtype=bool)
    peak = float(np.max(np.asarray(estimate)[mask]))
    if peak <= 0.0:
        return 0.0
    detected = (np.asarray(estimate) > fraction * peak) & mask
    return dice_overlap(detected, np.asarray(truth_region, dtype=bool) & mask)


def map_cnr(estimate: np.ndarray, active: np.ndarray, reference: np.ndarray) -> float:
    """Spatial contrast-to-noise ratio of a response map.

    `(mean over the active region - mean over the reference region) / standard
    deviation over the reference region` (ddof 1): how far the true activation
    stands out of the map's own fluctuation where nothing is activated.

    estimate: shape (nz, nx); active, reference: bool regions, shape (nz, nx).
    """
    active, reference = np.asarray(active, dtype=bool), np.asarray(reference, dtype=bool)
    if not active.any() or reference.sum() < 2:
        raise ValueError("the activated region is empty or the reference region too small")
    values = np.asarray(estimate, dtype=np.float64)
    return float(
        (values[active].mean() - values[reference].mean()) / values[reference].std(ddof=1)
    )


def bank_delays(bank: GreensBank) -> tuple[np.ndarray, np.ndarray]:
    """Group delay and band-averaged amplitude of every element-node pair of a bank.

    For `B = a exp(+i 2 pi f eps)` the product of neighbouring frequency samples
    has the phase `2 pi df eps` whatever the carrier phase; the delay is the phase
    of the sum of these products over frequency divided by `2 pi df` (the estimate
    of `fus.screen.fit_phase_screen`, kept per pair instead of averaged over the
    points). The amplitude is the mean of `|B|` weighted by the two-way pulse spectrum.

    Returns `(excess_delay [s], amplitude)`, float64, shape (n_elements, n_nodes) each;
    the delay is relative to the reference medium and unambiguous within `+-1 / (2 df)`.
    """
    values = bank.B.reshape(*bank.B.shape[:2], -1)
    weights = pulse_spectrum(bank.transducer, bank.freqs)
    weights = weights / weights.sum()
    lag_products = np.zeros(values.shape[1:], dtype=np.complex128)
    amplitude = np.zeros(values.shape[1:])
    for f in range(values.shape[0]):
        amplitude += weights[f] * np.abs(values[f])
        if f + 1 < values.shape[0]:
            lag_products += values[f + 1].astype(np.complex128) * np.conj(values[f])
    frequency_step = (bank.freqs[-1] - bank.freqs[0]) / (bank.freqs.shape[0] - 1)
    return np.angle(lag_products) / (2.0 * np.pi * frequency_step), amplitude


def weighted_delay_rms(difference: np.ndarray, weight: np.ndarray, gauge: str) -> float:
    """Weighted rms of a delay difference over element-node pairs after removing a gauge [s].

    difference, weight: shape (n_elements, n_nodes). gauge: "common" removes the
    weighted mean over all pairs (a delay common to everything, which shifts the
    image in range and changes no focus); "node" removes the weighted mean over
    the elements of every node (a delay common to all elements at one field point,
    which moves that point's echo in range without defocusing it): what is left is
    the focusing error proper.
    """
    if gauge == "common":
        residual = difference - (weight * difference).sum() / weight.sum()
    elif gauge == "node":
        total = weight.sum(axis=0)
        mean = np.divide(
            (weight * difference).sum(axis=0), total, out=np.zeros_like(total), where=total > 0
        )
        residual = difference - mean
    else:
        raise ValueError(f"unknown gauge {gauge!r}")
    return float(np.sqrt((weight * residual**2).sum() / weight.sum()))


def travel_time_error(model_bank: GreensBank, true_bank: GreensBank, gauge: str) -> float:
    """Amplitude-weighted rms one-way travel-time error of a model bank [s].

    The difference of the two banks' group delays (`bank_delays`) over the
    element-node pairs to which both banks give amplitude, weighted by the true
    amplitude (pairs the true medium barely reaches carry no signal and no
    weight), after removing the gauge (`weighted_delay_rms`).
    """
    model_delay, model_amplitude = bank_delays(model_bank)
    true_delay, true_amplitude = bank_delays(true_bank)
    weight = true_amplitude * (model_amplitude > 0.0)
    return weighted_delay_rms(model_delay - true_delay, weight, gauge)


def non_finite_reason(arrays: dict[str, np.ndarray]) -> str | None:
    """Why a method's output cannot be scored: a description of its non-finite values, or None.

    arrays: the named arrays a method produced. Returns for example
    ``"images: 3072 of 122880 values not finite (3072 NaN, 0 infinite)"``, or None
    when every value of every array is finite.
    """
    reasons = []
    for name, values in arrays.items():
        values = np.asarray(values)
        n_nan = int(np.isnan(values).sum())
        n_infinite = int(np.isinf(values).sum())
        if n_nan + n_infinite:
            reasons.append(
                f"{name}: {n_nan + n_infinite} of {values.size} values not finite "
                f"({n_nan} NaN, {n_infinite} infinite)"
            )
    return "; ".join(reasons) or None
