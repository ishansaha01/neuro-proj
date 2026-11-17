"""Guidance rule and posterior sampling."""

from __future__ import annotations

import math

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.recon.config import DpsConfig
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.dps import (
    cbv_log_likelihood,
    guidance_correction,
    next_timesteps,
    posterior_sample,
    prior_samples,
    reconstruct_trials,
)
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator
from neurosim.recon.transform import CbvTransform

from .scene_fixtures import tiny_prior, tiny_run  # noqa: F401


def _generator(*sub: int) -> torch.Generator:
    return torch_generator(SeedTree(3), SAMPLING_STREAM, 9100, *sub)


def test_next_timesteps_chain_ends_at_the_clean_image():
    diffusion = GaussianDiffusion(DiffusionConfig(n_timesteps=100))
    following = next_timesteps(diffusion, 5)
    assert list(following) == diffusion.sampling_timesteps(5)
    assert following[99] == diffusion.sampling_timesteps(5)[1]
    assert following[0] == -1


def test_weight_of_a_single_ancestral_step_is_the_score_step():
    diffusion = GaussianDiffusion(DiffusionConfig(n_timesteps=100))
    cfg = DpsConfig(n_steps=100, eta=1.0, max_step_ratio=1e9)
    generator = _generator(0)
    x_t = torch.randn((2, 1, 4, 4), generator=generator)
    x0_hat = torch.randn((2, 1, 4, 4), generator=generator)
    gradient = 1e-3 * torch.randn((2, 1, 4, 4), generator=generator)
    t = 40
    correction = guidance_correction(gradient, x_t, x0_hat, diffusion, t, t - 1, cfg)
    alpha = float(diffusion.alpha_bar[t] / diffusion.alpha_bar[t - 1])
    torch.testing.assert_close(
        correction, (1.0 - alpha) / math.sqrt(alpha) * gradient, rtol=1e-4, atol=0.0
    )


def test_trust_region_limits_the_correction_to_the_unguided_step():
    diffusion = GaussianDiffusion(DiffusionConfig(n_timesteps=100))
    cfg = DpsConfig(n_steps=10, eta=0.0, max_step_ratio=0.5)
    generator = _generator(1)
    x_t = torch.randn((3, 1, 4, 4), generator=generator)
    x0_hat = torch.randn((3, 1, 4, 4), generator=generator)
    gradient = 1e6 * torch.randn((3, 1, 4, 4), generator=generator)
    t, t_next = 55, 44
    correction = guidance_correction(gradient, x_t, x0_hat, diffusion, t, t_next, cfg)
    unguided = diffusion.step(x_t, x0_hat, t, t_next, 0.0, None) - x_t
    torch.testing.assert_close(
        correction.flatten(1).norm(dim=1),
        0.5 * unguided.flatten(1).norm(dim=1),
        rtol=1e-4,
        atol=0.0,
    )
    # The direction is kept.
    cosine = torch.nn.functional.cosine_similarity(correction.flatten(1), gradient.flatten(1))
    torch.testing.assert_close(cosine, torch.ones(3), rtol=1e-4, atol=0.0)


def test_flat_likelihood_and_zero_scale_return_the_unguided_samples(tiny_prior):
    cfg = DpsConfig(n_steps=4)
    shape = (2, 1, 16, 16)

    def flat(x0_hat: torch.Tensor) -> torch.Tensor:
        return 0.0 * x0_hat.flatten(1).sum(dim=1)

    def steep(x0_hat: torch.Tensor) -> torch.Tensor:
        return -1e3 * x0_hat.flatten(1).square().sum(dim=1)

    def sample(log_likelihood, config):
        return posterior_sample(
            tiny_prior.model, tiny_prior.diffusion, log_likelihood, shape, _generator(2), config
        )

    unguided = sample(None, cfg)
    torch.testing.assert_close(sample(flat, cfg), unguided)
    torch.testing.assert_close(sample(steep, DpsConfig(n_steps=4, guidance_scale=0.0)), unguided)
    assert not torch.allclose(sample(steep, cfg), unguided)


def test_cbv_log_likelihood_prefers_the_true_image():
    transform = CbvTransform()
    operator = torch.eye(4) + 0.1
    truth = torch.tensor([[0.02, 0.05, 0.1, 0.03]])
    floor = torch.full((4,), 1e-9)
    measured = truth * 1e-6 @ operator.T + floor
    log_likelihood = cbv_log_likelihood(transform, operator, 1e-6, floor, measured, 50.0)
    model_truth = transform.to_model(truth).reshape(1, 1, 2, 2)
    assert log_likelihood(model_truth) > log_likelihood(model_truth + 0.1)
    assert log_likelihood(model_truth) > log_likelihood(model_truth - 0.1)
    # Negative CBV is treated as zero: far below the floor of the model space nothing changes.
    torch.testing.assert_close(
        log_likelihood(torch.full((1, 1, 2, 2), -1.5)),
        log_likelihood(torch.full((1, 1, 2, 2), -3.0)),
    )


def test_reconstruct_trials_shape_seed_and_sub_index(tiny_prior, tiny_run):
    scene, trials = tiny_run
    operator = torch.from_numpy(scene.operators["skull_das"])
    floor = torch.from_numpy(scene.noise_floor["skull_das"])
    measured = trials.power_doppler["skull_das"][:3]
    cfg = DpsConfig(n_steps=3, n_samples=2)

    def run(seed: int, *sub: int) -> np.ndarray:
        return reconstruct_trials(
            tiny_prior,
            operator,
            floor,
            scene.cell_area,
            scene.effective_frames,
            measured,
            cfg,
            SeedTree(seed),
            *sub,
        )

    first = run(0, 1, 0)
    assert first.shape == (2, 3, 16, 16) and first.dtype == np.float32
    assert first.min() >= 0.0 and np.all(np.isfinite(first))
    np.testing.assert_array_equal(run(0, 1, 0), first)
    assert not np.array_equal(run(0, 1, 1), first)
    assert not np.array_equal(run(1, 1, 0), first)
    assert not np.array_equal(first[0], first[1])


def test_prior_samples_shape(tiny_prior):
    samples = prior_samples(tiny_prior, 5, DpsConfig(n_steps=3), SeedTree(0), 4, 0)
    assert samples.shape == (5, 16, 16) and samples.min() >= 0.0
