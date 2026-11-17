"""data.trials: the image series at both fidelities."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.data.scene import build_scene_with_extras
from neurosim.data.trials import image_times, simulate_trials, task_amplitude
from neurosim.data.types import BEAMFORMERS, SKULL_DAS
from neurosim.fus.acquire import statistical_power_doppler
from neurosim.head.cbv import cbv_at, task_labels, task_timecourse

from .conftest import SMALL_SEED


def test_trial_set_matches_the_contract(small_cfg, small_run):
    scene, _, trials = small_run
    n, shape = small_cfg.trials.n_images, scene.pixel_grid.shape
    assert trials.fidelity == "statistical"
    assert trials.time.dtype == np.float64 and trials.time.shape == (n,)
    assert trials.amplitude.dtype == np.float32 and trials.amplitude.shape == (n,)
    assert trials.label.dtype == np.int8 and trials.label.shape == (n,)
    assert trials.cbv.dtype == np.float32 and trials.cbv.shape == (n, *shape)
    assert tuple(trials.power_doppler) == BEAMFORMERS
    for images in trials.power_doppler.values():
        assert images.dtype == np.float32 and images.shape == (n, *shape) and np.all(images > 0.0)


def test_series_follows_the_task_design_of_the_head(small_cfg, small_run):
    scene, _, trials = small_run
    task = small_cfg.head.task
    np.testing.assert_array_equal(trials.time, 10.0 * np.arange(12))
    np.testing.assert_array_equal(trials.time, image_times(small_cfg))
    np.testing.assert_array_equal(trials.label, task_labels(task, trials.time))
    assert set(trials.label.tolist()) == {0, 1}  # rest and task images
    np.testing.assert_allclose(trials.amplitude, task_timecourse(task, trials.time), atol=1e-7)
    np.testing.assert_array_equal(
        trials.cbv,
        cbv_at(scene.cbv_baseline, scene.activation, trials.amplitude.astype(np.float64)),
    )
    # CBV rises in the activated region of a task image and nowhere else.
    task_image = int(np.argmax(trials.amplitude))
    ratio = trials.cbv[task_image][scene.brain_mask] / scene.cbv_baseline[scene.brain_mask]
    assert ratio.max() == pytest.approx(1.0 + trials.amplitude.max() * scene.activation.max())
    assert ratio.min() == pytest.approx(1.0)


def test_any_statistical_image_can_be_regenerated_alone(small_cfg, small_run):
    scene, _, trials = small_run
    image, key_number = 7, BEAMFORMERS.index(SKULL_DAS)
    alone = statistical_power_doppler(
        scene.operators[SKULL_DAS].astype(np.float64),
        trials.cbv[image] * scene.cell_area,
        scene.noise_floor[SKULL_DAS],
        scene.effective_frames[SKULL_DAS],
        scene.noise_frames,
        SeedTree(SMALL_SEED).rng("speckle_statistical", image, key_number),
    )
    alone = alone.reshape(scene.pixel_grid.shape).astype(np.float32)
    np.testing.assert_array_equal(trials.power_doppler[SKULL_DAS][image], alone)
    # A shorter series is the start of the longer one.
    short = replace(small_cfg, trials=replace(small_cfg.trials, n_images=3))
    first = simulate_trials(scene, short, SeedTree(SMALL_SEED))
    for key in BEAMFORMERS:
        np.testing.assert_array_equal(first.power_doppler[key], trials.power_doppler[key][:3])


def test_statistical_images_scatter_about_the_expected_image(small_run):
    scene, _, trials = small_run
    for key in BEAMFORMERS:
        operator = scene.operators[key].astype(np.float64)
        cells = trials.cbv.reshape(trials.cbv.shape[0], -1).astype(np.float64) * scene.cell_area
        blood = cells @ operator.T
        floor = scene.noise_floor[key].astype(np.float64)
        expected = blood + floor
        ratio = trials.power_doppler[key].reshape(expected.shape) / expected
        # Blood counts as `effective_frames` independent frames per pixel (about one in
        # this 80 ms ensemble) and the noise as `noise_frames`: the variance of
        # `fus.acquire.statistical_power_doppler`, pooled over 12 x 256 pixels.
        blood_variance = blood**2 / scene.effective_frames[key]
        variance = blood_variance + (2.0 * blood + floor) * floor / scene.noise_frames
        assert ratio.mean() == pytest.approx(1.0, abs=0.05)
        assert ratio.var() == pytest.approx(np.mean(variance / expected**2), rel=0.2)


def test_keys_and_images_get_independent_speckle(small_run):
    scene, _, trials = small_run
    expected = {}
    for key in BEAMFORMERS:
        operator = scene.operators[key].astype(np.float64)
        cells = trials.cbv.reshape(trials.cbv.shape[0], -1).astype(np.float64) * scene.cell_area
        mean = cells @ operator.T + scene.noise_floor[key]
        expected[key] = (trials.power_doppler[key].reshape(mean.shape) / mean).ravel()
    correlations = np.corrcoef(np.stack(list(expected.values())))
    assert np.abs(correlations[~np.eye(len(BEAMFORMERS), dtype=bool)]).max() < 0.1


def test_amplitude_jitter_scales_each_block_by_its_own_gain(small_cfg):
    tree = SeedTree(SMALL_SEED)
    task = small_cfg.head.task
    times = np.asarray([onset + 15.0 for onset in task.onsets])  # near the peak of each block
    plain = task_amplitude(small_cfg, tree, times)
    np.testing.assert_allclose(plain, task_timecourse(task, times), rtol=1e-12)
    jittered_cfg = replace(small_cfg, trials=replace(small_cfg.trials, amplitude_jitter=0.3))
    jittered = task_amplitude(jittered_cfg, tree, times)
    gains = np.exp(0.3 * tree.rng("cbv", 0).standard_normal(len(task.onsets)))
    # 40 s between onsets: the neighbours' undershoot is a few percent of a block's peak.
    np.testing.assert_allclose(jittered / plain, gains, rtol=0.1)
    assert np.ptp(jittered / plain) > 0.05
    again = task_amplitude(jittered_cfg, SeedTree(SMALL_SEED), times)
    np.testing.assert_array_equal(jittered, again)


@pytest.fixture(scope="module")
def scatterer_run(quick_cfg):
    """Two images of four frames with a few hundred scatterers: the scatterer tier, tiny."""
    imaging = replace(
        quick_cfg.imaging,
        sequence=replace(quick_cfg.imaging.sequence, n_frames=4),
        doppler=replace(quick_cfg.imaging.doppler, n_discard=1),
        scatterers=replace(quick_cfg.imaging.scatterers, n_blood=150, n_tissue=150),
    )
    cfg = replace(
        quick_cfg,
        imaging=imaging,
        trials=replace(quick_cfg.trials, n_images=2, image_interval=20.0, fidelity="scatterer"),
    )
    scene, extras = build_scene_with_extras(cfg, SeedTree(SMALL_SEED))
    return cfg, scene, extras, simulate_trials(scene, cfg, SeedTree(SMALL_SEED), extras)


def test_scatterer_fidelity_gives_an_image_per_key(scatterer_run):
    cfg, scene, extras, trials = scatterer_run
    assert trials.fidelity == "scatterer"
    for key in BEAMFORMERS:
        images = trials.power_doppler[key]
        assert images.shape == (2, *scene.pixel_grid.shape) and images.dtype == np.float32
        assert np.all(np.isfinite(images)) and np.all(images > 0.0)


def test_scatterer_images_are_reproducible_and_independent_of_the_series(scatterer_run):
    cfg, scene, extras, trials = scatterer_run
    one = replace(cfg, trials=replace(cfg.trials, n_images=1))
    first = simulate_trials(scene, one, SeedTree(SMALL_SEED), extras)
    images = trials.power_doppler[SKULL_DAS]
    np.testing.assert_array_equal(first.power_doppler[SKULL_DAS][0], images[0])
    # The two images differ: other blood scatterers, other frames of noise and motion.
    assert not np.array_equal(images[0], images[1])


def test_scatterer_fidelity_needs_the_extras(scatterer_run):
    cfg, scene, _, _ = scatterer_run
    with pytest.raises(ValueError, match="extras"):
        simulate_trials(scene, cfg, SeedTree(SMALL_SEED))
