"""Gamma speckle likelihood."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gamma

from neurosim.common.seeding import SeedTree
from neurosim.recon.likelihood import gamma_deviance, gamma_log_likelihood


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
