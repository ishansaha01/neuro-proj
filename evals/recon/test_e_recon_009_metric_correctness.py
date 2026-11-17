"""E-RECON-009: the decoder and the coverage metric give their known answers on known data.

Decoder: on images whose labels were shuffled there is nothing to decode, so the
cross-validated accuracy must be chance; on images with a strong response in one
region it must be (nearly) perfect. Coverage: when the truth and the posterior
samples are independent draws from the same Gaussian, the Student prediction
interval covers the truth with exactly its nominal probability.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.metrics import decoding_accuracy, interval_coverage, stratified_folds

EVAL_ID = "E-RECON-009"
STREAM_SUB = 1009
N_TRIALS = 60
IMAGE_SHAPE = (16, 16)
N_SHUFFLES = 100
N_FOLDS = 5
N_COMPONENTS = 10
RESPONSE = 5.0  # in units of the pixel noise, over a 3 x 3 pixel region
N_VALUES = 200_000
N_POSTERIOR_SAMPLES = 4
LEVELS = (0.5, 0.8, 0.9)
MAX_CHANCE_ERROR = 0.05
MIN_SEPARABLE_ACCURACY = 0.98
MAX_COVERAGE_ERROR = 0.005


@eval_case(
    id=EVAL_ID,
    claim="The PCA + LDA decoder is at chance on label-shuffled images and near 100 % on "
    "trivially separable ones, and the coverage metric returns the nominal level for exact "
    "Gaussian posterior samples",
    reference="Analytic: chance 0.5 for balanced shuffled labels; an error rate below 1e-10 "
    "for a 15-sigma class separation; coverage equal to the nominal level by the Student "
    "prediction-interval theorem",
    metric="chance_error = |mean accuracy over 100 label shuffles - 0.5| (60 trials, 5 folds, "
    "10 components); separable_accuracy; coverage_error = max over levels 0.5, 0.8, 0.9 of "
    "|coverage - level| (200000 values, 4 samples each)",
    threshold="chance_error < 0.05, separable_accuracy > 0.98 and coverage_error < 0.005 (fixed "
    "in advance: one shuffle's accuracy has a standard deviation of about 0.1 with 60 "
    "dependent cross-validated trials, so the mean of 100 has 0.01 and 0.05 is five standard "
    "errors; the binomial standard error of a coverage over 200000 values is at most 0.0011, "
    "so 0.005 is over four standard errors, while an interval built with the normal instead "
    "of the Student quantile would miss by 0.07 or more)",
    seeds=(0,),
)
def test_e_recon_009_metric_correctness():
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    rng = tree.rng("recon_sampling", STREAM_SUB)

    labels = np.tile([0, 1], N_TRIALS // 2)
    noise = rng.standard_normal((N_TRIALS, *IMAGE_SHAPE))
    separable = noise.copy()
    separable[:, :3, :3] += RESPONSE * labels[:, None, None]
    folds = stratified_folds(labels, N_FOLDS, rng.permutation(N_TRIALS))
    separable_accuracy = decoding_accuracy(separable, labels, folds, N_COMPONENTS)
    shuffled_accuracies = []
    for _ in range(N_SHUFFLES):
        shuffled = rng.permutation(labels)
        shuffled_folds = stratified_folds(shuffled, N_FOLDS, rng.permutation(N_TRIALS))
        shuffled_accuracies.append(
            decoding_accuracy(separable, shuffled, shuffled_folds, N_COMPONENTS)
        )
    chance_error = abs(float(np.mean(shuffled_accuracies)) - 0.5)

    centre = rng.standard_normal(N_VALUES)
    scale = rng.uniform(0.5, 2.0, N_VALUES)
    truth = centre + scale * rng.standard_normal(N_VALUES)
    samples = centre + scale * rng.standard_normal((N_POSTERIOR_SAMPLES, N_VALUES))
    coverage = interval_coverage(samples, truth, np.asarray(LEVELS))
    coverage_error = float(np.abs(coverage - np.asarray(LEVELS)).max())

    report(
        EVAL_ID,
        {
            "chance_error": chance_error,
            "separable_accuracy": separable_accuracy,
            "coverage_error": coverage_error,
            **{f"coverage_{level:g}": float(value) for level, value in zip(LEVELS, coverage)},
        },
        passed=chance_error < MAX_CHANCE_ERROR
        and separable_accuracy > MIN_SEPARABLE_ACCURACY
        and coverage_error < MAX_COVERAGE_ERROR,
    )
