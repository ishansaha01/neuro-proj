"""Synthetic phase-screen scenes."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.sweep import load_dataset
from neurosim.data.types import BEAMFORMERS
from neurosim.recon.cbv_images import PriorDataConfig
from neurosim.recon.synthetic import (
    ScreenSkullConfig,
    SyntheticConfig,
    draw_screens,
    write_synthetic_dataset,
)

from .scene_fixtures import tiny_run  # noqa: F401


def test_screens_have_the_configured_statistics():
    skull = ScreenSkullConfig()
    true, ct = draw_screens(skull, 64, SeedTree(4))
    np.testing.assert_allclose(np.std(true.delays), skull.delay_rms, rtol=1e-9)
    np.testing.assert_allclose(np.exp(np.log(true.amplitudes).mean()), skull.mean_amplitude)
    np.testing.assert_allclose(np.std(np.log(true.amplitudes)), skull.log_amplitude_rms)
    np.testing.assert_allclose(np.std(ct.delays - true.delays), skull.ct_delay_error_rms, rtol=1e-9)
    log_error = np.log(ct.amplitudes) - np.log(true.amplitudes)
    np.testing.assert_allclose(log_error.mean(), skull.ct_log_amplitude_bias)
    np.testing.assert_allclose(log_error.std(), skull.ct_log_amplitude_error_rms)
    again, _ = draw_screens(skull, 64, SeedTree(4))
    np.testing.assert_array_equal(again.delays, true.delays)
    other, _ = draw_screens(skull, 64, SeedTree(5))
    assert not np.array_equal(other.delays, true.delays)
    flat, flat_ct = draw_screens(
        ScreenSkullConfig(delay_rms=0.0, ct_delay_error_rms=0.0), 8, SeedTree(4)
    )
    assert not flat.delays.any() and not flat_ct.delays.any()


def test_scene_follows_the_contract(tiny_run):
    scene, trials = tiny_run
    assert scene.meta["synthetic"] is True
    assert set(scene.operators) == set(BEAMFORMERS) == set(scene.noise_floor)
    for key in BEAMFORMERS:
        assert scene.operators[key].shape == (256, 256) and scene.operators[key].dtype == np.float32
        assert scene.noise_floor[key].shape == (256,) and scene.noise_floor[key].min() > 0.0
    assert scene.cbv_baseline.shape == (16, 16) and scene.brain_mask.any()
    # Independent blood frames per pixel and key, from the head model's blood velocity.
    n_frames = scene.imaging.sequence.n_frames
    assert set(scene.effective_frames) == set(BEAMFORMERS) and scene.noise_frames == n_frames
    for key in BEAMFORMERS:
        blood_frames = scene.effective_frames[key]
        assert blood_frames.shape == (256,) and blood_frames.dtype == np.float32
        assert blood_frames.min() >= 1.0 and blood_frames.max() <= n_frames
        assert np.ptp(blood_frames) > 0.0
    assert trials.cbv.shape == (12, 16, 16) and trials.fidelity == "statistical"
    assert set(np.unique(trials.label)) == {0, 1}
    # The clean operator is the homogeneous one: the skull only attenuates and blurs.
    assert scene.operators["clean_das"].sum() > scene.operators["skull_das"].sum()
    # The true-skull beamformer restores unit gain under its own model.
    np.testing.assert_allclose(np.sqrt(np.diag(scene.operators["skull_true"])), 1.0, atol=1e-3)


def test_dataset_round_trip(tmp_path, tiny_run):
    scene, _ = tiny_run
    run = RunConfig(
        head=PriorDataConfig().head,
        imaging=scene.imaging,
        source_refinement=1,
        trials=dataclasses.replace(RunConfig().trials, n_images=12, image_interval=10.0),
    )
    cfg = SyntheticConfig(run=run, n_scenes=2)
    write_synthetic_dataset(cfg, 5, tmp_path)
    loaded = list(load_dataset(tmp_path))
    assert len(loaded) == 2
    first_scene, first_trials = loaded[0]
    assert first_scene.meta["synthetic"] is True and first_scene.meta["sweep"]["index"] == 0
    assert first_trials.power_doppler["skull_das"].shape == (12, 16, 16)
    assert not np.array_equal(first_scene.screen_true.delays, loaded[1][0].screen_true.delays)
