"""Condition-mean reconstruction on a model beamformer (`recon.medium_recon`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.types import SKULL_TRUE
from neurosim.recon.config import DeconvolutionConfig, DpsConfig
from neurosim.recon.medium_recon import (
    BeamformedSeries,
    condition_means,
    deconvolve_means,
    match_common_amplitude,
    model_operator,
    predicted_difference,
    sample_means,
)

from .scene_fixtures import tiny_ct_prior, tiny_ct_run  # noqa: F401


def _series(scene, trials) -> BeamformedSeries:
    return BeamformedSeries(
        images=trials.power_doppler[SKULL_TRUE],
        noise_floor=scene.noise_floor[SKULL_TRUE],
        effective_frames=scene.effective_frames[SKULL_TRUE],
        noise_frames=scene.noise_frames,
    )


def test_condition_means():
    images = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    labels = np.array([0, 1, 1, 0, 0, 1])
    means = condition_means(images, labels)
    assert (means.n_rest, means.n_task) == (3, 3)
    np.testing.assert_allclose(means.rest, images[[0, 3, 4]].mean(axis=0))
    np.testing.assert_allclose(means.task, images[[1, 2, 5]].mean(axis=0))
    with pytest.raises(ValueError, match="both states"):
        condition_means(images, np.zeros(6, dtype=int))


def test_model_operator_of_the_true_bank_is_the_stored_operator(tiny_ct_run):
    scene = tiny_ct_run[0]
    operator = model_operator(scene, scene.bank_skull, 1)
    np.testing.assert_allclose(operator, scene.operators[SKULL_TRUE], rtol=1e-5, atol=1e-12)


def test_amplitude_match_recovers_a_power_scale(tiny_ct_run):
    scene = tiny_ct_run[0]
    operator = scene.operators[SKULL_TRUE].astype(np.float64)
    mask = np.asarray(scene.brain_mask, dtype=bool)
    floor = scene.noise_floor[SKULL_TRUE].astype(np.float64)
    reference_cbv = 0.03
    uniform = mask.reshape(-1) * reference_cbv * scene.cell_area
    image = (7.0 * operator @ uniform + floor).reshape(mask.shape)
    scale = match_common_amplitude(image, floor, operator, mask, reference_cbv, scene.cell_area)
    assert scale == pytest.approx(7.0, rel=1e-6)
    with pytest.raises(ValueError, match="noise floor"):
        match_common_amplitude(
            0.5 * floor.reshape(mask.shape), floor, operator, mask, reference_cbv, scene.cell_area
        )


def test_deconvolution_and_sampling_of_the_means(tiny_ct_run, tiny_ct_prior):
    scene, trials = tiny_ct_run
    series = _series(scene, trials)
    means = condition_means(series.images, trials.label)
    operator = torch.from_numpy(np.ascontiguousarray(scene.operators[SKULL_TRUE]))
    rest, task = deconvolve_means(operator, series, means, scene.cell_area, DeconvolutionConfig())
    assert rest.shape == task.shape == scene.pixel_grid.shape
    assert np.all(rest >= 0.0) and np.isfinite(task).all()
    template = predicted_difference(operator, rest, task)
    assert template.shape == (64,) and template.dtype == np.float64

    cfg = DpsConfig(n_steps=3, n_samples=2)
    arguments = (tiny_ct_prior, operator, series, means, scene.cell_area, cfg, SeedTree(4), 9, 0)
    rest_samples, task_samples = sample_means(*arguments)
    assert rest_samples.shape == task_samples.shape == (2, *scene.pixel_grid.shape)
    again_rest, _ = sample_means(*arguments)
    np.testing.assert_array_equal(rest_samples, again_rest)
    # Equal measurements with the shared random numbers give equal samples.
    same = condition_means(
        np.stack([means.rest] * 4), np.array([0, 1, 0, 1])
    )
    same_rest, same_task = sample_means(
        tiny_ct_prior, operator, series, same, scene.cell_area, cfg, SeedTree(4), 9, 0
    )
    np.testing.assert_array_equal(same_rest, same_task)
