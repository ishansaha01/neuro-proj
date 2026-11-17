"""data.scene: the scene contract, its normalisations, probe placement and cost estimate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.config import config_hash
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.data.config import medium_grid
from neurosim.data.scene import (
    beamformer_banks,
    build_scene,
    estimate_cost,
    power_doppler_operator,
    probe_plane,
    skull_sound_speed_r2,
)
from neurosim.data.types import BEAMFORMERS, CLEAN_DAS, SKULL_CT, SKULL_DAS, SKULL_TRUE
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.fus.noise import electronic_noise_variance
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.head.generate import generate_head
from neurosim.head.materials import skull_mask
from neurosim.head.plane import default_plane

from .conftest import SMALL_SEED


def test_scene_matches_the_contract_shapes_and_dtypes(small_cfg, small_run):
    scene, _, _ = small_run
    setup = imaging_setup(small_cfg.imaging)
    shape = setup.pixel_grid.shape
    n_pix = shape[0] * shape[1]
    assert scene.imaging == small_cfg.imaging and scene.pixel_grid == setup.pixel_grid
    assert scene.cell_area == setup.pixel_grid.spacing**2
    maps = (("cbv_baseline", np.float32), ("activation", np.float32), ("brain_mask", bool))
    for name, dtype in maps:
        values = getattr(scene, name)
        assert values.shape == shape and values.dtype == dtype, name
    bank_shape = (setup.freqs.size, setup.transducer.n_elements, *setup.roi.shape)
    for bank in (scene.bank_skull, scene.bank_clean, scene.bank_ct):
        assert bank.B.shape == bank_shape and bank.B.dtype == np.complex64
        assert bank.roi == setup.roi and bank.backend == "ray"
    assert set(scene.operators) == set(scene.noise_floor) == set(BEAMFORMERS)
    for key in BEAMFORMERS:
        assert scene.operators[key].shape == (n_pix, n_pix)
        assert scene.operators[key].dtype == np.float32 and np.all(scene.operators[key] >= 0.0)
        assert scene.noise_floor[key].shape == (n_pix,)
        assert scene.noise_floor[key].dtype == np.float32 and np.all(scene.noise_floor[key] > 0.0)
    for screen in (scene.screen_ct, scene.screen_true):
        assert screen.delays.shape == screen.amplitudes.shape == (setup.transducer.n_elements,)
    assert scene.effective_frames == small_cfg.imaging.sequence.n_frames


def test_cbv_fields_live_in_the_brain(small_run):
    scene, _, _ = small_run
    assert scene.brain_mask.any() and not scene.brain_mask.all()
    assert np.all(scene.cbv_baseline[~scene.brain_mask] == 0.0)
    assert np.all(scene.cbv_baseline[scene.brain_mask] > 0.0)
    assert np.all(scene.activation[~scene.brain_mask] == 0.0)
    assert 0.5 < scene.activation.max() <= 1.0  # the activated patch is inside the image


def test_skull_delays_and_losses_show_in_the_screens(small_run):
    scene, _, _ = small_run
    for screen in (scene.screen_true, scene.screen_ct):
        # Bone is faster than the reference medium and lossy.
        assert np.all(screen.delays < 0.0) and np.all(screen.delays > -5.0e-6)
        assert np.all(screen.amplitudes > 0.0) and np.all(screen.amplitudes < 1.0)


def test_meta_records_provenance_grid_cost_and_ct_agreement(small_cfg, small_run):
    scene, extras, _ = small_run
    meta = scene.meta
    grid = medium_grid(small_cfg)
    assert meta["config_hash"] == config_hash(small_cfg) and meta["root_seed"] == SMALL_SEED
    assert meta["backend"] == "ray" and meta["skull_true_model"] == "ray"
    assert meta["effects"] == {"aberration": True, "attenuation": True, "reverberation": True}
    assert meta["medium_grid"] == {
        "shape": list(grid.shape),
        "spacing": grid.spacing,
        "origin": list(grid.origin),
    }
    assert meta["cost"] == estimate_cost(small_cfg)
    assert meta["ct"]["kvp"] == small_cfg.ct.kvp and meta["ct"]["kernel"] == "ramp"
    assert meta["ct"]["hu_mapping"]["c_bone"] > meta["ct"]["hu_mapping"]["c_water"]
    assert 0.0 <= meta["ct_sound_speed_r2"] <= 1.0
    # The run has no skull-effect switch off, so the stored medium is the plain truth.
    assert meta["ct_sound_speed_r2"] == skull_sound_speed_r2(extras.medium, extras.medium_ct)


def test_extras_carry_the_media_and_the_true_model(small_cfg, small_run):
    scene, extras, _ = small_run
    assert extras.medium.grid == extras.medium_ct.grid == medium_grid(small_cfg)
    assert extras.medium.plane == scene.plane
    assert skull_mask(extras.medium.labels).any()
    assert extras.model_true is scene.bank_skull  # ray backend: the model is the bank itself
    assert extras.blood_velocity.shape == (2, *scene.pixel_grid.shape)
    assert np.all(extras.blood_velocity[:, ~scene.brain_mask] == 0.0)


def test_build_scene_is_deterministic_and_equals_the_first_part_of_the_pair(
    quick_cfg, quick_scene
):
    scene = build_scene(quick_cfg, SeedTree(SMALL_SEED))
    again, _ = quick_scene
    assert scene.plane == again.plane and scene.meta == again.meta
    np.testing.assert_array_equal(scene.bank_ct.B, again.bank_ct.B)
    for key in BEAMFORMERS:
        np.testing.assert_array_equal(scene.operators[key], again.operators[key])


def test_beamformer_banks_pair_data_and_model(small_run):
    scene, extras, _ = small_run
    banks = beamformer_banks(scene.bank_skull, scene.bank_clean, scene.bank_ct, extras.model_true)
    assert tuple(banks) == BEAMFORMERS
    assert banks[SKULL_DAS][0] is banks[SKULL_CT][0] is banks[SKULL_TRUE][0] is scene.bank_skull
    assert banks[CLEAN_DAS][0] is scene.bank_clean
    assert banks[SKULL_CT][1] is scene.bank_ct and banks[SKULL_TRUE][1] is scene.bank_skull
    for key in (SKULL_DAS, CLEAN_DAS):
        model = banks[key][1]
        assert model.backend == "analytic" and np.all(model.B == 1.0)


def test_operator_and_noise_floor_follow_the_fus_normalisation(quick_cfg, quick_scene):
    """One sample per pixel: H = |A|^2 of fus.operators; floor = variance * noise gain."""
    scene, extras = quick_scene
    setup = imaging_setup(quick_cfg.imaging)
    banks = beamformer_banks(scene.bank_skull, scene.bank_clean, scene.bank_ct, extras.model_true)
    variance = electronic_noise_variance(scene.bank_skull, setup.sequence, quick_cfg.imaging.noise)
    for key, (bank, model) in banks.items():
        expected = pd_operator(
            psf_operator(bank, setup.pixel_grid, setup.pixel_grid, model, setup.sequence)
        )
        np.testing.assert_array_equal(scene.operators[key], expected)
        gain = make_beamformer(setup.pixel_grid, model, setup.sequence).noise_gain
        np.testing.assert_allclose(scene.noise_floor[key], variance * gain.ravel(), rtol=1e-6)
    # Unit scatterer gain under the true model: a scatterer on a pixel gives |IQ| = 1 there.
    np.testing.assert_allclose(np.diag(scene.operators[SKULL_TRUE]), 1.0, rtol=1e-3)
    # The skull costs signal: conventional DAS through it is far below the clean image.
    through_skull = np.diag(scene.operators[SKULL_DAS]).mean()
    assert through_skull < 0.1 * np.diag(scene.operators[CLEAN_DAS]).mean()


def test_refined_operator_is_the_mean_over_sub_cell_sources(small_cfg, small_run):
    scene, _, _ = small_run
    setup = imaging_setup(small_cfg.imaging)
    grid = setup.pixel_grid
    homogeneous = beamformer_banks(
        scene.bank_skull, scene.bank_clean, scene.bank_ct, scene.bank_skull
    )[CLEAN_DAS][1]
    assert small_cfg.source_refinement == 2
    expected = np.zeros_like(scene.operators[CLEAN_DAS])
    for dz in (-0.25, 0.25):
        for dx in (-0.25, 0.25):
            origin = (grid.origin[0] + dz * grid.spacing, grid.origin[1] + dx * grid.spacing)
            sources = Grid2D(shape=grid.shape, spacing=grid.spacing, origin=origin)
            expected += pd_operator(
                psf_operator(scene.bank_clean, grid, sources, homogeneous, setup.sequence)
            )
    np.testing.assert_allclose(scene.operators[CLEAN_DAS], expected / 4.0, rtol=1e-6)
    single = power_doppler_operator(
        scene.bank_clean, homogeneous, setup, small_cfg.imaging.beamforming, 1
    )
    # Sampling only the pixel centre, where the response peaks, overstates the diagonal.
    assert np.diag(single).mean() > 1.2 * np.diag(scene.operators[CLEAN_DAS]).mean()


def test_noise_switch_zeroes_the_floor_and_effect_switches_change_only_the_skull_bank(
    quick_cfg, quick_scene
):
    reference, _ = quick_scene
    noise = replace(quick_cfg.imaging.noise, electronic_noise=False)
    changed = replace(
        quick_cfg,
        imaging=replace(quick_cfg.imaging, noise=noise),
        effects=replace(quick_cfg.effects, attenuation=False),
    )
    scene = build_scene(changed, SeedTree(SMALL_SEED))
    for key in BEAMFORMERS:
        assert np.all(scene.noise_floor[key] == 0.0)
    np.testing.assert_array_equal(scene.bank_clean.B, reference.bank_clean.B)
    np.testing.assert_array_equal(scene.bank_ct.B, reference.bank_ct.B)
    np.testing.assert_array_equal(scene.cbv_baseline, reference.cbv_baseline)
    # Without bone absorption more sound gets through.
    assert np.abs(scene.bank_skull.B).mean() > 2.0 * np.abs(reference.bank_skull.B).mean()
    assert scene.meta["effects"]["attenuation"] is False


def test_probe_plane_applies_shift_and_tilt_to_the_default_plane(small_cfg):
    head = generate_head(small_cfg.head, SeedTree(SMALL_SEED))
    default = default_plane(head, small_cfg.head)
    assert probe_plane(head, small_cfg) == default

    pose = replace(small_cfg.probe, lateral_shift=2.0e-3, tilt_deg=10.0)
    plane = probe_plane(head, replace(small_cfg, probe=pose))
    u_lateral, u_depth = np.asarray(default.u_lateral), np.asarray(default.u_depth)
    np.testing.assert_allclose(
        np.asarray(plane.origin) - np.asarray(default.origin), 2.0e-3 * u_lateral, atol=1e-12
    )
    # The depth axis turns by 10 degrees towards +x and stays in the same plane.
    assert np.asarray(plane.u_depth) @ u_depth == pytest.approx(np.cos(np.deg2rad(10.0)))
    assert np.asarray(plane.u_depth) @ u_lateral == pytest.approx(np.sin(np.deg2rad(10.0)))
    normal = np.cross(u_lateral, u_depth)
    np.testing.assert_allclose(np.cross(plane.u_lateral, plane.u_depth), normal, atol=1e-12)


def test_an_array_pushed_into_the_scalp_is_refused(quick_cfg):
    tilted = replace(quick_cfg, probe=replace(quick_cfg.probe, tilt_deg=20.0))
    with pytest.raises(ValueError, match="array face touches the head"):
        build_scene(tilted, SeedTree(SMALL_SEED))


def test_skull_sound_speed_r2_is_a_squared_correlation(small_run):
    _, extras, _ = small_run
    truth = extras.medium
    assert skull_sound_speed_r2(truth, truth) == pytest.approx(1.0)
    rescaled = replace(truth, sound_speed=1000.0 - 0.5 * truth.sound_speed)
    assert skull_sound_speed_r2(truth, rescaled) == pytest.approx(1.0)


def test_cost_estimate_adds_up_the_package_estimates(small_cfg):
    cost = estimate_cost(small_cfg)
    stages = (
        cost["head"]["time_s"]
        + cost["ct_prior"]["time_s"]
        + 2 * cost["bank"]["run_time_s"]
        + cost["bank_ray"]["run_time_s"]
        + 16 * cost["operator"]["time_s"]
        + 4 * 12 * cost["trial_image"]["time_s"]
    )
    assert cost["time_s"] == pytest.approx(stages)
    assert cost["n_operator_passes"] == 16 and cost["n_images"] == 12
    scatterer = replace(small_cfg, trials=replace(small_cfg.trials, fidelity="scatterer"))
    assert estimate_cost(scatterer)["trial_image"]["n_frames"] == 40
    assert estimate_cost(scatterer)["time_s"] > cost["time_s"]


def test_cost_estimate_checks_the_full_wave_array_placement(small_cfg):
    kwave = replace(small_cfg, backend=replace(small_cfg.backend, backend="kwave"))
    with pytest.raises(ValueError, match="whole number of medium grid"):
        estimate_cost(replace(kwave, medium_spacing=1.0e-4))  # 0.385 mm pitch: 3.85 cells
    pitch = imaging_setup(small_cfg.imaging).transducer.pitch
    cost = estimate_cost(replace(kwave, medium_spacing=pitch / 3))
    assert cost["n_ray_banks"] == 2  # the CT bank and the ray model of the true skull
    assert cost["bank"]["n_simulations"] > 48 and cost["bank"]["time_steps"] > 0
