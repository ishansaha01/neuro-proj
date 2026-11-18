"""Focus calibration of the skull model (`recon.focus`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from neurosim.fus.config import imaging_setup
from neurosim.recon import scanner
from neurosim.recon.config import FocusConfig
from neurosim.recon.focus import (
    calibrate,
    criterion_and_gradient,
    focus_criterion,
    member_ratios,
    parameter_map,
    pixel_groups,
)
from neurosim.recon.rebeamform import weight_model

from .scene_fixtures import tiny_ct_family, tiny_ct_run  # noqa: F401

FOCUS = FocusConfig(n_pixels=24, pixels_per_pass=8, mode_stages=(0, 3), iterations_per_stage=3)


@pytest.fixture(scope="module")
def problem(tiny_ct_run, tiny_ct_family):
    scene, trials = tiny_ct_run
    record = scanner.record_channel_data(scene, trials, 1)
    sequence = imaging_setup(scene.imaging).sequence
    pixels = np.arange(4, 64, 3)[:24]
    weights = [
        weight_model(tiny_ct_family, scene.pixel_grid, sequence, scene.imaging.beamforming, group)
        for group in pixel_groups(pixels, FOCUS.pixels_per_pass)
    ]
    return record, weights


def test_pixel_groups_keep_every_pixel_once():
    groups = pixel_groups(np.arange(10), 4)
    assert [len(group) for group in groups] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(groups), np.arange(10))


def test_parameter_map_round_trip_and_bounds(tiny_ct_family):
    family = tiny_ct_family
    parameters = parameter_map(family)
    nominal = family.nominal_parameters()
    vector = parameters.to_vector(nominal)
    back = parameters.to_parameters(vector)
    assert back.c_bone == pytest.approx(nominal.c_bone)
    assert back.c_water == pytest.approx(nominal.c_water)
    assert back.hu_bone == pytest.approx(nominal.hu_bone)
    # Whatever the unconstrained vector, the constants stay inside the prior ranges.
    wild = parameters.to_parameters(torch.tensor([50.0, -50.0, 3.0, 0.0, 0.0, 0.0]))
    prior = family.cfg.mapping_prior
    assert prior.c_bone_range[0] <= wild.c_bone <= prior.c_bone_range[1]
    assert prior.c_water_range[0] <= wild.c_water <= prior.c_water_range[1]
    # The nominal member changes no slowness; the prior only sees the lateral coefficients.
    knot_change, lateral = parameters.slowness_changes(vector)
    assert float(knot_change.abs().max()) < 1e-15 and float(lateral.abs().max()) == 0.0
    assert float(parameters.log_prior(vector)) == 0.0
    shifted = vector.clone()
    shifted[3:] = torch.tensor([1.0, -2.0, 0.5])
    assert float(parameters.log_prior(shifted)) == pytest.approx(-0.5 * 5.25)
    outside = dataclasses.replace(nominal, c_bone=prior.c_bone_range[1] + 1.0)
    with pytest.raises(ValueError, match="prior ranges"):
        parameters.to_vector(outside)


def test_gradient_matches_finite_differences(problem, tiny_ct_family):
    record, weights = problem
    parameters = parameter_map(tiny_ct_family)
    vector = parameters.to_vector(tiny_ct_family.nominal_parameters())
    vector[3:] = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    value, gradient = criterion_and_gradient(record, weights, parameters, vector, 1.0)
    ratios = member_ratios(record, weights, parameters, vector)
    assert value == pytest.approx(float(focus_criterion(ratios, 1.0)), abs=1e-5)
    for index, step in ((0, 1e-2), (3, 1e-2), (5, 1e-2)):
        up, down = vector.clone(), vector.clone()
        up[index] += step
        down[index] -= step
        numeric = (
            criterion_and_gradient(record, weights, parameters, up, 1.0)[0]
            - criterion_and_gradient(record, weights, parameters, down, 1.0)[0]
        ) / (2 * step)
        assert float(gradient[index]) == pytest.approx(numeric, rel=0.1, abs=2e-4)


def test_grouping_does_not_change_the_criterion(problem, tiny_ct_run, tiny_ct_family):
    record, weights = problem
    scene = tiny_ct_run[0]
    parameters = parameter_map(tiny_ct_family)
    vector = parameters.to_vector(tiny_ct_family.nominal_parameters())
    pixels = np.arange(4, 64, 3)[:24]
    together = [
        weight_model(
            tiny_ct_family,
            scene.pixel_grid,
            imaging_setup(scene.imaging).sequence,
            scene.imaging.beamforming,
            pixels,
        )
    ]
    grouped = criterion_and_gradient(record, weights, parameters, vector, 2.0)
    single = criterion_and_gradient(record, together, parameters, vector, 2.0)
    assert grouped[0] == pytest.approx(single[0], abs=1e-5)
    np.testing.assert_allclose(grouped[1].numpy(), single[1].numpy(), rtol=1e-3, atol=1e-6)


def test_calibration_does_not_lower_the_criterion(problem, tiny_ct_family):
    record, weights = problem
    result = calibrate(record, tiny_ct_family, weights, FOCUS)
    assert len(result.stage_criteria) == 2
    assert result.criterion >= result.criterion_start - 1e-6
    assert result.criterion == result.stage_criteria[-1]
    assert result.n_evaluations > 0 and result.seconds > 0.0
    assert result.parameters.lateral.shape == (3,)
    assert result.parameters.common_amplitude == 1.0
    again = calibrate(record, tiny_ct_family, weights, FOCUS)
    assert again.criterion == result.criterion
    np.testing.assert_array_equal(again.parameters.lateral, result.parameters.lateral)


def test_too_many_modes_are_refused(problem, tiny_ct_family):
    record, weights = problem
    with pytest.raises(ValueError, match="mode_stages"):
        calibrate(record, tiny_ct_family, weights, FocusConfig(mode_stages=(0, 4)))
    with pytest.raises(ValueError, match="ascending"):
        FocusConfig(mode_stages=(4, 2))
