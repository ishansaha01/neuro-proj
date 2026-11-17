"""Screen calibration: schedule, gauge handling, determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurosim.common.seeding import SeedTree
from neurosim.fus.config import imaging_setup
from neurosim.recon.calibrate import calibrate_screen, match_total_power, update_schedule
from neurosim.recon.config import CalibrationConfig, DpsConfig
from neurosim.recon.forward import power_doppler_operator, scene_screen_model, screen_tensors
from neurosim.recon.gauge import gauge_directions

from .scene_fixtures import tiny_prior, tiny_run, tiny_scene  # noqa: F401

SMALL = CalibrationConfig(
    sampler=DpsConfig(n_steps=6, n_samples=2),
    n_candidates=2,
    n_updates=3,
    lbfgs_iterations=3,
    n_pixels=48,
    initial_spread=0.5,
)


def test_update_schedule_starts_with_the_amplitude_and_grows_the_modes():
    cfg = CalibrationConfig(
        sampler=DpsConfig(n_steps=21),
        n_updates=5,
        update_until=0.6,
        coarse_modes=2,
        max_delay_modes=16,
    )
    assert update_schedule(cfg) == {0: 0, 3: 2, 6: 4, 9: 8, 12: 16}
    assert update_schedule(CalibrationConfig(n_updates=1)) == {0: 0}


def test_config_rejects_inconsistent_settings():
    with pytest.raises(ValueError):
        CalibrationConfig(sampler=DpsConfig(n_steps=2), n_updates=3)
    with pytest.raises(ValueError):
        CalibrationConfig(coarse_modes=8, max_delay_modes=4)


@pytest.fixture(scope="module")
def calibration(tiny_prior, tiny_run):
    scene, trials = tiny_run
    setup = imaging_setup(scene.imaging)
    model = scene_screen_model(scene, 1)

    def run(seed: int, *sub: int):
        return calibrate_screen(
            tiny_prior,
            model,
            scene.screen_ct,
            setup.transducer.element_x(),
            setup.transducer.center_frequency,
            trials.power_doppler["skull_das"].mean(axis=0),
            scene.noise_floor["skull_das"],
            scene.cell_area,
            scene.effective_frames * len(trials.label),
            SMALL,
            SeedTree(seed),
            *sub,
        )

    return scene, setup.transducer.element_x(), run


def test_result_is_complete_and_selects_the_best_fit(calibration):
    scene, _, run = calibration
    result = run(0, 0, 0)
    assert len(result.candidates) == 2 and len(result.deviance_per_pixel) == 2
    assert result.selected == int(np.argmin(result.deviance_per_pixel))
    assert result.screen is result.candidates[result.selected]
    assert result.screen.delays.shape == scene.screen_ct.delays.shape
    assert np.all(np.isfinite(result.deviance_per_pixel)) and np.isfinite(
        result.start_deviance_per_pixel
    )


def test_gauge_directions_and_element_amplitude_ratios_stay_at_the_ct_values(calibration):
    scene, element_x, run = calibration
    result = run(0, 0, 0)
    for candidate in result.candidates:
        change = candidate.delays - scene.screen_ct.delays
        assert np.abs(change).max() > 0.0
        directions = gauge_directions(element_x, "mean_tilt")
        np.testing.assert_allclose(
            directions.T @ change, 0.0, atol=1e-12 * np.abs(directions).max()
        )
        ratio = candidate.amplitudes / scene.screen_ct.amplitudes
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] != 1.0


def test_same_seed_same_screen_other_sub_index_other_screen(calibration):
    _, _, run = calibration
    first = run(0, 0, 0)
    again = run(0, 0, 0)
    np.testing.assert_array_equal(again.screen.delays, first.screen.delays)
    np.testing.assert_array_equal(again.screen.amplitudes, first.screen.amplitudes)
    other = run(0, 0, 1)
    assert not np.array_equal(other.candidates[0].delays, first.candidates[0].delays)


def test_match_total_power_rescales_the_common_amplitude_exactly(tiny_scene):
    model = scene_screen_model(tiny_scene, 1)
    screen = tiny_scene.screen_ct
    operator = power_doppler_operator(model, *screen_tensors(screen)).to(torch.float32)
    target = 2.0 * float(operator.sum())
    rescaled, rescaled_operator = match_total_power(screen, operator, target)
    np.testing.assert_allclose(rescaled.amplitudes / screen.amplitudes, 2.0**0.25)
    np.testing.assert_array_equal(rescaled.delays, screen.delays)
    direct = power_doppler_operator(model, *screen_tensors(rescaled)).to(torch.float32)
    torch.testing.assert_close(
        rescaled_operator, direct, rtol=1e-4, atol=1e-6 * float(direct.max())
    )
