"""Torch beamformer weights of a family member and the scanner emulation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.scene import power_doppler_operator
from neurosim.data.types import SKULL_CT, SKULL_TRUE
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.recon import scanner
from neurosim.recon.rebeamform import model_weights, weight_model
from neurosim.recon.skull_model import delay_only, parameter_tensors

from .scene_fixtures import tiny_ct_family, tiny_ct_run  # noqa: F401


@pytest.fixture(scope="module")
def setup(tiny_ct_run):
    return imaging_setup(tiny_ct_run[0].imaging)


def _weights(scene, family, setup, pixels=None, member=None):
    model = weight_model(
        family, scene.pixel_grid, setup.sequence, scene.imaging.beamforming, pixels
    )
    member = member or family.nominal_parameters()
    return model_weights(model, *parameter_tensors(family, member))


def test_nominal_weights_equal_the_numpy_beamformer(tiny_ct_run, tiny_ct_family, setup):
    scene = tiny_ct_run[0]
    weights = _weights(scene, tiny_ct_family, setup)
    reference = make_beamformer(
        scene.pixel_grid, delay_only(scene.bank_ct), setup.sequence, scene.imaging.beamforming
    )
    np.testing.assert_allclose(weights.tx_phase.numpy(), reference.tx_weights, atol=2e-4)
    np.testing.assert_allclose(
        (weights.rx_phase / weights.unit_response).numpy(),
        reference.rx_weights,
        atol=2e-4 * np.abs(reference.rx_weights).max(),
    )
    np.testing.assert_allclose(
        (weights.squared_weights / weights.unit_response.square()).numpy(),
        reference.noise_gain.reshape(-1),
        rtol=1e-4,
    )


def test_pixel_subset_and_common_amplitude(tiny_ct_run, tiny_ct_family, setup):
    scene, family = tiny_ct_run[0], tiny_ct_family
    pixels = np.array([3, 17, 40])
    full = _weights(scene, family, setup)
    subset = _weights(scene, family, setup, pixels)
    np.testing.assert_allclose(
        subset.rx_phase.numpy(), full.rx_phase.numpy()[:, :, pixels], atol=1e-5
    )
    model = weight_model(
        family, scene.pixel_grid, setup.sequence, scene.imaging.beamforming, pixels
    )
    changes = parameter_tensors(family, family.nominal_parameters())
    scaled = model_weights(model, *changes, common_amplitude=0.5)
    # A common amplitude g scales the unit response by g^2 and leaves the phases alone.
    np.testing.assert_allclose(
        scaled.unit_response.numpy(), 0.25 * subset.unit_response.numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(scaled.rx_phase.numpy(), subset.rx_phase.numpy(), atol=1e-6)


def test_signal_to_floor_matches_the_stored_ct_beamformer(tiny_ct_run, tiny_ct_family, setup):
    scene, trials = tiny_ct_run
    record = scanner.record_channel_data(scene, trials, 1)
    with torch.no_grad():
        ratios = scanner.signal_to_floor(record, _weights(scene, tiny_ct_family, setup)).numpy()
    # The same ratio from an image series re-beamformed in numpy with the delay-only CT bank.
    series = scanner.rebeamform_trials(
        scene, trials, delay_only(scene.bank_ct), 1, SKULL_CT, SeedTree(1), 0
    )
    operator_ratio = (
        power_doppler_operator(
            scene.bank_skull, delay_only(scene.bank_ct), setup, scene.imaging.beamforming, 1
        ).astype(np.float64)
        @ (trials.cbv.mean(axis=0).reshape(-1) * scene.cell_area)
    ) / series.noise_floor
    np.testing.assert_allclose(ratios, operator_ratio, rtol=2e-3)


def test_signal_to_floor_is_differentiable_in_the_model(tiny_ct_run, tiny_ct_family, setup):
    scene, trials = tiny_ct_run
    family = tiny_ct_family
    record = scanner.record_channel_data(scene, trials, 1)
    model = weight_model(
        family, scene.pixel_grid, setup.sequence, scene.imaging.beamforming, np.arange(0, 64, 9)
    )
    knot_change, lateral = parameter_tensors(family, family.nominal_parameters())
    lateral = lateral.clone().requires_grad_(True)

    def total(profile: torch.Tensor) -> torch.Tensor:
        return scanner.signal_to_floor(record, model_weights(model, knot_change, profile)).sum()

    (gradient,) = torch.autograd.grad(total(lateral), lateral)
    direction = torch.from_numpy(family.lateral_modes[:, 0])
    step = 2.0e-7  # s/m
    with torch.no_grad():
        upper, lower = total(lateral + step * direction), total(lateral - step * direction)
        numeric = (upper - lower) / (2 * step)
    assert torch.isfinite(gradient).all()
    np.testing.assert_allclose(float(gradient @ direction), float(numeric), rtol=0.05)


def test_rebeamformed_series_is_seeded_and_uses_the_named_frame_counts(tiny_ct_run):
    scene, trials = tiny_ct_run
    bank = delay_only(scene.bank_ct)
    first = scanner.rebeamform_trials(scene, trials, bank, 1, SKULL_TRUE, SeedTree(3), 2)
    again = scanner.rebeamform_trials(scene, trials, bank, 1, SKULL_TRUE, SeedTree(3), 2)
    other = scanner.rebeamform_trials(scene, trials, bank, 1, SKULL_TRUE, SeedTree(3), 4)
    assert first.images.shape == trials.cbv.shape and first.images.dtype == np.float32
    np.testing.assert_array_equal(first.images, again.images)
    assert not np.array_equal(first.images, other.images)
    np.testing.assert_array_equal(first.effective_frames, scene.effective_frames[SKULL_TRUE])
    np.testing.assert_allclose(
        first.noise_floor, scanner.model_noise_floor(scene, bank), rtol=1e-6
    )


def test_oracles_and_noise_variance(tiny_ct_run):
    scene = tiny_ct_run[0]
    assert scanner.true_model_bank(scene) is scene.bank_skull
    assert scanner.true_model_operator(scene) is scene.operators[SKULL_TRUE]
    assert scanner.channel_noise_variance(scene) > 0.0
    silent = dataclasses.replace(
        scene,
        imaging=dataclasses.replace(
            scene.imaging,
            noise=dataclasses.replace(scene.imaging.noise, electronic_noise=False),
        ),
    )
    with pytest.raises(ValueError, match="signal-to-floor"):
        scanner.channel_noise_variance(silent)
