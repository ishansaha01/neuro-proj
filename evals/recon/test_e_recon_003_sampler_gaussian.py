"""E-RECON-003: with the exact denoiser of Gaussian data both samplers return that Gaussian.

For data ``x_0 ~ N(mu, Sigma)`` the posterior mean is linear,

    E[x_0 | x_t] = mu + sqrt(ab) Sigma (ab Sigma + (1 - ab) I)^-1 (x_t - sqrt(ab) mu),

so the denoiser is known in closed form and the samplers can be checked without
a network. Both samplers run all T = 1000 steps. Even with the exact denoiser a
finite number of steps leaves a discretisation bias; iterating the scalar
recursion of the ideal samplers for one eigen-direction of unit variance gives a
variance bias of -0.5 % for ancestral sampling and -0.25 % for DDIM at 1000
steps (and -2.4 % for DDIM at 100 steps, -4.9 % at 50), well below the 2.1 %
sampling error of the covariance at the sample size used here.
"""

from __future__ import annotations

import numpy as np
import torch

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator

EVAL_ID = "E-RECON-003"
# First sub-index of the eval's draws in the `recon_sampling` stream.
STREAM_SUB = 1003
N_SAMPLES = 20_000
IMAGE_SHAPE = (1, 4, 4)
# Eigenvalues of the data covariance: log-spaced, from far below to above the unit
# variance of the noise the samplers start from.
EIGENVALUE_RANGE = (0.05, 2.0)
MEAN_SCALE = 0.5
MAX_ERROR_RATIO = 2.0


def _gaussian_problem(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean (d,), eigenvalues (d,) and orthonormal eigenvectors (d, d) of the data Gaussian."""
    dimension = int(np.prod(IMAGE_SHAPE))
    mean = MEAN_SCALE * rng.standard_normal(dimension)
    eigenvalues = np.geomspace(*EIGENVALUE_RANGE, dimension)
    eigenvectors, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return mean, eigenvalues, eigenvectors


def _exact_denoiser(
    diffusion: GaussianDiffusion,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
):
    """The v-prediction of the exact posterior mean of the Gaussian, as a sampler model."""
    mean_t = torch.from_numpy(mean).to(torch.float32)
    basis = torch.from_numpy(eigenvectors).to(torch.float32)
    alpha_bars = diffusion.alpha_bar.numpy()

    def model(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = float(alpha_bars[int(t[0])])
        gain = np.sqrt(alpha_bar) * eigenvalues / (alpha_bar * eigenvalues + 1.0 - alpha_bar)
        centred = x_t.flatten(1) - np.sqrt(alpha_bar) * mean_t
        x0_hat = mean_t + ((centred @ basis) * torch.from_numpy(gain).to(torch.float32)) @ basis.T
        v = (np.sqrt(alpha_bar) * x_t.flatten(1) - x0_hat) / np.sqrt(1.0 - alpha_bar)
        return v.reshape(x_t.shape)

    return model


@eval_case(
    id=EVAL_ID,
    claim="With the exact denoiser of a Gaussian data distribution, the ancestral and the "
    "DDIM sampler return samples with the data mean and covariance",
    reference="Analytic: the mean and covariance of the 16-dimensional Gaussian whose "
    "closed-form posterior mean is given to the samplers as the denoiser",
    metric="Largest of four error ratios (ancestral and DDIM, 1000 steps, 20000 samples): "
    "|sample mean - mu| / sqrt(tr(Sigma)/N) and |sample covariance - Sigma|_F / "
    "sqrt((tr(Sigma)^2 + |Sigma|_F^2)/N), i.e. error over its expected sampling error",
    threshold="< 2 (fixed in advance: an exact sampler gives ratios near 1; 2 allows the "
    "spread of the ratio and the 0.5 % discretisation bias of 1000 steps, and flags a "
    "covariance error above about 4 %)",
    seeds=(0,),
)
def test_e_recon_003_sampler_gaussian():
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    diffusion = GaussianDiffusion(DiffusionConfig())
    mean, eigenvalues, eigenvectors = _gaussian_problem(tree.rng("recon_sampling", STREAM_SUB, 0))
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    model = _exact_denoiser(diffusion, mean, eigenvalues, eigenvectors)

    mean_error = np.sqrt(eigenvalues.sum() / N_SAMPLES)
    covariance_error = np.sqrt((eigenvalues.sum() ** 2 + np.sum(eigenvalues**2)) / N_SAMPLES)
    samplers = {"ancestral": diffusion.ancestral_sample, "ddim": diffusion.ddim_sample}
    ratios: dict[str, float] = {}
    for index, (name, sampler) in enumerate(samplers.items()):
        samples = sampler(
            model,
            (N_SAMPLES, *IMAGE_SHAPE),
            torch_generator(tree, SAMPLING_STREAM, STREAM_SUB, 1 + index),
            n_steps=diffusion.n_timesteps,
        )
        samples = samples.flatten(1).numpy().astype(np.float64)
        ratios[f"{name}_mean"] = np.linalg.norm(samples.mean(axis=0) - mean) / mean_error
        ratios[f"{name}_covariance"] = (
            np.linalg.norm(np.cov(samples, rowvar=False) - covariance) / covariance_error
        )

    worst = max(ratios.values())
    report(EVAL_ID, {"max_ratio": worst, **ratios}, passed=worst < MAX_ERROR_RATIO)
