"""Baselines without a learned prior."""

from __future__ import annotations

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.types import BEAMFORMERS
from neurosim.recon.baselines import gamma_deconvolution, raw_power_doppler
from neurosim.recon.config import DeconvolutionConfig
from neurosim.recon.forward import expected_power_doppler
from neurosim.recon.likelihood import gamma_log_likelihood

from .scene_fixtures import tiny_run  # noqa: F401


def _problem():
    """A well-conditioned 3 x 3 blur on a 6 x 6 image, noiseless data with a floor."""
    rng = SeedTree(0).rng("recon_sampling", 9200)
    n = 36
    blur = np.eye(n)
    for index in range(n - 1):
        blur[index, index + 1] = blur[index + 1, index] = 0.3
    operator = torch.from_numpy(blur.astype(np.float32))
    truth = rng.uniform(0.01, 0.1, size=(2, 6, 6)).astype(np.float32)
    floor = torch.full((n,), 1.0e-9)
    cell_area = 1.0e-6
    measured = expected_power_doppler(
        operator, torch.from_numpy(truth).flatten(1), cell_area, floor
    )
    return operator, floor, cell_area, truth, measured.reshape(2, 6, 6).numpy()


def test_raw_power_doppler_returns_every_beamformer(tiny_run):
    raw = raw_power_doppler(tiny_run[1])
    assert tuple(raw) == BEAMFORMERS
    assert all(
        series.shape == (12, 16, 16) and series.dtype == np.float32 for series in raw.values()
    )


def test_deconvolution_recovers_noiseless_truth_and_stays_non_negative():
    operator, floor, cell_area, truth, measured = _problem()
    estimate = gamma_deconvolution(
        operator, floor, cell_area, measured, DeconvolutionConfig(n_iterations=3000)
    )
    assert estimate.shape == truth.shape and estimate.min() >= 0.0
    np.testing.assert_allclose(estimate, truth, rtol=0.02)


def test_updates_increase_the_likelihood_of_the_shared_definition():
    operator, floor, cell_area, _, measured = _problem()
    y = torch.from_numpy(measured).flatten(1)
    values = []
    for n_iterations in (1, 5, 25, 125):
        estimate = gamma_deconvolution(
            operator, floor, cell_area, measured, DeconvolutionConfig(n_iterations=n_iterations)
        )
        expected = expected_power_doppler(
            operator, torch.from_numpy(estimate).flatten(1), cell_area, floor
        )
        values.append(gamma_log_likelihood(y, expected, 10.0).sum().item())
    assert values == sorted(values)


def test_first_iterate_is_a_flat_image_matched_to_the_mean_power():
    operator, floor, cell_area, _, measured = _problem()
    flat = np.broadcast_to(measured.mean(axis=(1, 2), keepdims=True), measured.shape).copy()
    # For a flat measurement and an operator with (nearly) equal row sums the flat start is
    # (nearly) a fixed point: one update leaves the image flat within the edge effect.
    estimate = gamma_deconvolution(
        operator, floor, cell_area, flat, DeconvolutionConfig(n_iterations=1)
    )
    interior = estimate.reshape(2, -1)[:, 2:-2]
    assert np.ptp(interior, axis=1).max() < 0.02 * interior.mean()
