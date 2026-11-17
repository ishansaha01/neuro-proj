"""E-RECON-007: DPS on a linear-Gaussian problem moves toward the analytic posterior.

Prior ``x ~ N(mu, Sigma)`` (16 dimensions, the Gaussian of E-RECON-003 with its
exact denoiser), measurement ``y = A x + n`` with 8 rows and ``n ~ N(0, s^2 I)``.
The posterior is Gaussian with mean

    m = mu + Sigma A^T (A Sigma A^T + s^2 I)^-1 (y - A mu).

DPS is not exact even here: it scores ``p(y | x0_hat)`` and ignores the spread of
``x_0`` given ``x_t``, which over-weights the data. The bar is therefore on the
direction and rough size of the shift of the sample mean, not on equality.
"""

from __future__ import annotations

import numpy as np
import torch

from evals.recon.test_e_recon_003_sampler_gaussian import (
    IMAGE_SHAPE,
    _exact_denoiser,
    _gaussian_problem,
)
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.config import DpsConfig
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.dps import posterior_sample
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator

EVAL_ID = "E-RECON-007"
STREAM_SUB = 1007
N_SAMPLES = 4000
N_MEASUREMENTS = 8
NOISE_SIGMA = 0.3
UNINFORMATIVE_SIGMA = 1.0e4
FRACTION_RANGE = (0.5, 1.5)
MAX_RESIDUAL = 0.5
MAX_UNINFORMATIVE_SHIFT = 1.0e-3


@eval_case(
    id=EVAL_ID,
    claim="With an exact Gaussian prior and a linear measurement with Gaussian noise, the mean "
    "of DPS samples moves from the prior mean toward the analytic posterior mean, and for an "
    "uninformative measurement DPS returns the samples of the unguided sampler",
    reference="Analytic posterior mean of the linear-Gaussian model; the unguided sampler of "
    "E-RECON-003 run from the same generator",
    metric="fraction = <m_dps - mu, m - mu> / |m - mu|^2 and residual = |m_dps - m| / |m - mu| "
    "(4000 samples, default DpsConfig: 30 ancestral steps, noise sigma 0.3); "
    "uninformative_shift = max |x_dps - x_unguided| at noise sigma 1e4",
    threshold="0.5 <= fraction <= 1.5 and residual < 0.5 and uninformative_shift < 1e-3 (fixed "
    "in advance: DPS over-weights the data because p(y | x0_hat) ignores the posterior spread "
    "of x_0 given x_t, so an overshoot of some tens of percent is its known bias and equality "
    "is not expected; a wrong sign or weight of the guidance gives a fraction below 0 or far "
    "above 1. At sigma 1e4 the likelihood score is 1e-9 of that at sigma 0.3, so the samples "
    "must equal the unguided ones to float32 rounding, far below the prior spread of 0.2-1.4)",
    seeds=(0,),
)
def test_e_recon_007_dps_gaussian_posterior():
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    rng = tree.rng("recon_sampling", STREAM_SUB, 0)
    diffusion = GaussianDiffusion(DiffusionConfig())
    mean, eigenvalues, eigenvectors = _gaussian_problem(rng)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    dimension = mean.shape[0]
    operator = rng.standard_normal((N_MEASUREMENTS, dimension)) / np.sqrt(dimension)
    truth = mean + eigenvectors @ (np.sqrt(eigenvalues) * rng.standard_normal(dimension))
    measurement_noise = rng.standard_normal(N_MEASUREMENTS)
    model = _exact_denoiser(diffusion, mean, eigenvalues, eigenvectors)
    operator_t = torch.from_numpy(operator).to(torch.float32)
    cfg = DpsConfig()
    shape = (N_SAMPLES, *IMAGE_SHAPE)

    def sample(sigma: float | None) -> np.ndarray:
        log_likelihood = None
        if sigma is not None:
            measured = torch.from_numpy(operator @ truth + sigma * measurement_noise).to(
                torch.float32
            )

            def log_likelihood(x0_hat: torch.Tensor) -> torch.Tensor:
                residual = x0_hat.flatten(1) @ operator_t.T - measured
                return -0.5 * residual.square().sum(dim=1) / sigma**2

        generator = torch_generator(tree, SAMPLING_STREAM, STREAM_SUB, 1)
        samples = posterior_sample(model, diffusion, log_likelihood, shape, generator, cfg)
        return samples.flatten(1).numpy().astype(np.float64)

    measured = operator @ truth + NOISE_SIGMA * measurement_noise
    gain = (
        covariance
        @ operator.T
        @ np.linalg.inv(
            operator @ covariance @ operator.T + NOISE_SIGMA**2 * np.eye(N_MEASUREMENTS)
        )
    )
    posterior_mean = mean + gain @ (measured - operator @ mean)
    shift = posterior_mean - mean
    sample_mean = sample(NOISE_SIGMA).mean(axis=0)
    fraction = float((sample_mean - mean) @ shift / (shift @ shift))
    residual = float(np.linalg.norm(sample_mean - posterior_mean) / np.linalg.norm(shift))
    uninformative_shift = float(np.abs(sample(UNINFORMATIVE_SIGMA) - sample(None)).max())

    passed = (
        FRACTION_RANGE[0] <= fraction <= FRACTION_RANGE[1]
        and residual < MAX_RESIDUAL
        and uninformative_shift < MAX_UNINFORMATIVE_SHIFT
    )
    report(
        EVAL_ID,
        {"fraction": fraction, "residual": residual, "uninformative_shift": uninformative_shift},
        passed=passed,
    )
