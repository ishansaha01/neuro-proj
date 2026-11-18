"""E-RECON-015: the matched-filter decoder attains the analytic accuracy on Gaussian data.

For two classes of Gaussian vectors that differ by a mean shift `t` and share
independent noise of variance `v_p` per component, the likelihood-ratio
classifier is the matched filter `sum_p t_p y_p / v_p`, and its accuracy is
`Phi(d / 2)` with `d^2 = sum_p t_p^2 / v_p` (Fisher's linear discriminant with a
known diagonal covariance).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.decoding import matched_filter_accuracy, matched_filter_predict
from neurosim.recon.metrics import stratified_folds

EVAL_ID = "E-RECON-015"
STREAM_SUB = 1015
N_PIXELS = 200
N_TRIALS = 20_000
N_FOLDS = 5
SENSITIVITY = 2.0  # d: the Bayes accuracy is Phi(1) = 0.8413
MAX_DEVIATION = 0.015


def _data(rng: np.random.Generator, shift: np.ndarray, sigma: np.ndarray, n: int):
    labels = (np.arange(n) % 2).astype(np.int8)
    images = 5.0 + sigma * rng.standard_normal((n, shift.shape[0])) + labels[:, None] * shift
    return images, labels


@eval_case(
    id=EVAL_ID,
    claim="The matched-filter decoder (recon.decoding) classifies Gaussian data with a known "
    "mean shift at the analytic accuracy of the likelihood-ratio test, with a given template "
    "and with the template estimated by cross-validation, and at chance without a shift",
    reference="Analytic: accuracy = Phi(d / 2), d^2 = sum_p t_p^2 / v_p, for independent "
    "Gaussian components with variances v_p and mean shift t; 0.5 for t = 0",
    metric="largest absolute difference between measured and analytic accuracy over three "
    "cases with 200 components of unequal variance and 20000 trials: (a) the true shift "
    "(d = 2) as template, trained on one half and tested on the other; (b) 5-fold "
    "cross-validation with the template estimated from the training folds; (c) as (b) "
    "without a shift",
    threshold="< 0.015 (fixed in advance: the binomial standard deviation of an accuracy over "
    "10000 to 20000 trials is 0.0026 to 0.005, so 0.015 is three to five of them; estimating "
    "the template from 8000 trials per class lowers the analytic accuracy of case (b) by "
    "0.0015 (d^2 becomes d^4 / (d^2 + 2 P / n)), which is inside the bar; a decoder with a "
    "wrong sign, weighting or threshold is off by 0.05 to 0.35)",
    seeds=(0,),
)
def test_e_recon_015_matched_filter_decoder():
    rng = SeedTree(REGISTRY[EVAL_ID].seeds[0]).rng("recon_sampling", STREAM_SUB)
    sigma = np.exp(rng.uniform(-1.0, 1.0, size=N_PIXELS))
    direction = rng.standard_normal(N_PIXELS) * sigma
    shift = direction * SENSITIVITY / np.sqrt(np.sum((direction / sigma) ** 2))
    analytic = float(norm.cdf(0.5 * SENSITIVITY))

    images, labels = _data(rng, shift, sigma, N_TRIALS)
    half = N_TRIALS // 2
    predicted = matched_filter_predict(images[:half], labels[:half], images[half:], shift)
    given_template = float(np.mean(predicted == labels[half:]))

    folds = stratified_folds(labels, N_FOLDS, rng.permutation(N_TRIALS))
    cross_validated = matched_filter_accuracy(images, labels, folds)

    null_images, null_labels = _data(rng, np.zeros(N_PIXELS), sigma, N_TRIALS)
    null = matched_filter_accuracy(null_images, null_labels, folds)

    deviation = max(
        abs(given_template - analytic), abs(cross_validated - analytic), abs(null - 0.5)
    )
    report(
        EVAL_ID,
        {
            "largest_deviation": deviation,
            "analytic_accuracy": analytic,
            "accuracy_given_template": given_template,
            "accuracy_cross_validated": cross_validated,
            "accuracy_without_shift": null,
        },
        passed=deviation < MAX_DEVIATION,
    )
