"""Gamma speckle likelihood."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import gamma

from neurosim.common.seeding import SeedTree
from neurosim.fus.acquire import statistical_power_doppler
from neurosim.recon.likelihood import (
    gamma_deviance,
    gamma_log_likelihood,
    power_doppler_deviance,
    power_doppler_log_likelihood,
    speckle_shape,
)


def test_log_likelihood_equals_the_scipy_gamma_density():
    measured = torch.tensor([[0.5, 2.0, 3.0], [1.0, 1.0, 4.0]], dtype=torch.float64)
    expected = torch.tensor([[1.0, 2.5, 2.0], [0.7, 1.2, 5.0]], dtype=torch.float64)
    shape = 7.0
    reference = gamma.logpdf(measured.numpy(), a=shape, scale=expected.numpy() / shape).sum(axis=1)
    np.testing.assert_allclose(
        gamma_log_likelihood(measured, expected, shape), reference, rtol=1e-12
    )


def test_likelihood_is_maximal_at_the_measured_image_and_deviance_is_zero_there():
    measured = torch.tensor([[0.5, 2.0, 3.0]], dtype=torch.float64)
    at_data = gamma_log_likelihood(measured, measured, 20.0)
    for factor in (0.8, 1.25):
        assert gamma_log_likelihood(measured, factor * measured, 20.0) < at_data
    assert gamma_deviance(measured, measured, 20.0).item() == 0.0
    gap = at_data - gamma_log_likelihood(measured, 1.25 * measured, 20.0)
    np.testing.assert_allclose(
        gamma_deviance(measured, 1.25 * measured, 20.0), 2.0 * gap, rtol=1e-12
    )


def test_deviance_per_pixel_is_about_one_for_data_from_the_model():
    shape, n_pix = 50.0, 20000
    rng = SeedTree(0).rng("speckle_statistical", 0)
    expected = rng.uniform(0.5, 2.0, size=(1, n_pix))
    measured = expected * rng.gamma(shape, 1.0 / shape, size=expected.shape)
    deviance = gamma_deviance(torch.from_numpy(measured), torch.from_numpy(expected), shape)
    # Mean 1 + 1 / (6 L) per pixel, standard error sqrt(2 / n_pix) = 0.01.
    assert abs(deviance.item() / n_pix - (1.0 + 1.0 / (6.0 * shape))) < 0.04


def _images() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measured and expected images (2 x 4 pixels) and a floor below every expected value."""
    measured = torch.tensor([[0.5, 2.0, 3.0, 1.1], [1.0, 1.0, 4.0, 0.2]], dtype=torch.float64)
    expected = torch.tensor([[1.0, 2.5, 2.0, 1.0], [0.7, 1.2, 5.0, 0.4]], dtype=torch.float64)
    floor = torch.tensor([0.0, 0.3, 1.5, 0.39], dtype=torch.float64)
    return measured, expected, floor


def test_speckle_shape_is_mean_squared_over_the_variance_of_the_generative_model():
    _, expected, floor = _images()
    blood_frames = torch.tensor([1.2, 3.0, 8.0, 2.0], dtype=torch.float64)
    noise_frames = 40.0
    blood = expected - floor
    variance = blood**2 / blood_frames + (2.0 * blood * floor + floor**2) / noise_frames
    shape = speckle_shape(expected, floor, blood_frames, noise_frames)
    torch.testing.assert_close(shape, expected**2 / variance)
    # Without a floor the shape is the blood count; a pixel at the floor has the noise count.
    torch.testing.assert_close(shape[:, 0], blood_frames[0].expand(2))
    at_floor = speckle_shape(floor[None, 1:], floor[1:], blood_frames[1:], noise_frames)
    torch.testing.assert_close(at_floor, torch.full((1, 3), noise_frames, dtype=torch.float64))
    assert torch.all(shape >= blood_frames.min()) and torch.all(shape <= noise_frames)


def test_per_pixel_likelihood_reduces_to_the_scalar_shape_when_all_counts_are_equal():
    measured, expected, floor = _images()
    common = 7.0
    for blood_frames in (common, torch.full((4,), common, dtype=torch.float64)):
        shape = speckle_shape(expected, floor, blood_frames, common)
        torch.testing.assert_close(shape, torch.full_like(expected, common))
        torch.testing.assert_close(
            power_doppler_log_likelihood(measured, expected, floor, blood_frames, common),
            gamma_log_likelihood(measured, expected, common),
        )
        torch.testing.assert_close(
            power_doppler_deviance(measured, expected, floor, blood_frames, common),
            gamma_deviance(measured, expected, common),
        )
    # A tensor of shapes is the per-pixel scipy density.
    shapes = torch.tensor([1.5, 4.0, 12.0, 30.0], dtype=torch.float64)
    reference = gamma.logpdf(
        measured.numpy(), a=shapes.numpy(), scale=expected.numpy() / shapes.numpy()
    ).sum(axis=1)
    np.testing.assert_allclose(
        gamma_log_likelihood(measured, expected, shapes), reference, rtol=1e-12
    )


def test_likelihood_gradient_is_finite_and_matches_finite_differences():
    measured, _, floor = _images()
    blood_frames = torch.tensor([1.0, 2.5, 6.0, 40.0], dtype=torch.float64)
    blood = torch.tensor(
        [[1.0, 2.2, 0.5, 0.01], [0.7, 0.9, 3.5, 0.0]], dtype=torch.float64, requires_grad=True
    )

    def log_likelihood(blood_expected: torch.Tensor) -> torch.Tensor:
        return power_doppler_log_likelihood(
            measured, blood_expected + floor, floor, blood_frames, 40.0
        ).sum()

    (gradient,) = torch.autograd.grad(log_likelihood(blood), blood)
    assert torch.all(torch.isfinite(gradient))
    assert torch.autograd.gradcheck(log_likelihood, (blood,), eps=1e-7, atol=1e-6)
    # Single precision, the dtype of the reconstruction, stays finite as well.
    single = blood.detach().to(torch.float32).requires_grad_(True)
    value = power_doppler_log_likelihood(
        measured.float(), single + floor.float(), floor.float(), blood_frames.float(), 40.0
    ).sum()
    (single_gradient,) = torch.autograd.grad(value, single)
    assert torch.isfinite(value) and torch.all(torch.isfinite(single_gradient))


def test_matched_gamma_has_the_moments_of_the_statistical_tier():
    """Draws of `statistical_power_doppler` have mean `mu` and variance `mu^2 / L`."""
    blood, floor, blood_frames, noise_frames = 1.0, 0.6, 2.5, 30.0
    n_draws = 200_000
    draws = statistical_power_doppler(
        np.full((n_draws, 1), blood),
        np.ones(1),
        floor,
        blood_frames,
        noise_frames,
        SeedTree(0).rng("speckle_statistical", 1),
    )
    expected = torch.tensor([[blood + floor]], dtype=torch.float64)
    shape = speckle_shape(expected, torch.tensor([floor]), blood_frames, noise_frames).item()
    assert draws.mean() == pytest.approx(blood + floor, rel=0.01)
    assert draws.mean() ** 2 / draws.var() == pytest.approx(shape, rel=0.03)
    # Deviance per pixel under the matched gamma is of order one for data from the model
    # (the matched density is not the exact one, so only the size is checked).
    measured = torch.from_numpy(draws[None, :])
    deviance = power_doppler_deviance(
        measured, expected.expand(1, n_draws), torch.tensor([floor]), blood_frames, noise_frames
    )
    assert 0.8 < deviance.item() / n_draws < 1.3
