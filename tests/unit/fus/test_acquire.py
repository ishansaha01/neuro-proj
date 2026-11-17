"""Frame acquisition, the statistical tier and the cost estimate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.fus.acquire import (
    acquire_frames,
    acquire_power_doppler,
    estimate_cost,
    statistical_power_doppler,
)
from neurosim.fus.beamform import das
from neurosim.fus.config import ImagingConfig
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.fus.synthesis import static_clutter_spectra


def _quiet(cfg: ImagingConfig, **noise) -> ImagingConfig:
    """Config with electronic noise and tissue motion off unless overridden."""
    return replace(
        cfg,
        noise=replace(cfg.noise, **{"electronic_noise": False, "tissue_motion": False, **noise}),
    )


@pytest.fixture(scope="module")
def scene(tiny, tiny_cfg):
    tree = SeedTree(0)
    cbv = np.full(tiny.roi.shape, 0.03)
    velocity = np.full((2, *tiny.roi.shape), 0.02)
    blood = blood_scatterers(
        cbv, velocity, tiny.roi, tiny_cfg.scatterers.n_blood, tree.rng("blood_scatterers"), 1e-3
    )
    tissue = tissue_scatterers(
        tiny.roi,
        tiny_cfg.scatterers.n_tissue,
        tissue_amplitude(tiny.roi, tiny_cfg.scatterers.n_tissue, tiny_cfg.noise),
        tree.rng("tissue_scatterers"),
    )
    return tree, blood, tissue


def _acquire(tiny, water, scene, cfg, tissue="scene", bank=None, **kwargs):
    tree, blood, scene_tissue = scene
    return acquire_frames(
        bank or water,
        water,
        blood,
        scene_tissue if tissue == "scene" else tissue,
        tiny.sequence,
        tiny.pixel_grid,
        cfg,
        tree,
        **kwargs,
    )


def test_frames_shape_and_determinism(tiny, tiny_cfg, water, scene):
    frames = _acquire(tiny, water, scene, tiny_cfg)
    assert frames.shape == (6, 8, 8) and frames.dtype == np.complex128
    np.testing.assert_array_equal(_acquire(tiny, water, scene, tiny_cfg), frames)
    assert not np.array_equal(frames[0], frames[1])


def test_frame_offset_selects_absolute_frames(tiny, tiny_cfg, water, scene):
    tree, blood, tissue = scene
    frames = _acquire(tiny, water, scene, tiny_cfg)
    tail = acquire_frames(
        water,
        water,
        blood,
        tissue,
        replace(tiny.sequence, n_frames=2),
        tiny.pixel_grid,
        tiny_cfg,
        tree,
        4,
    )
    np.testing.assert_array_equal(tail, frames[4:])


def test_tissue_switch_equals_no_tissue(tiny, tiny_cfg, water, scene):
    off = _acquire(tiny, water, scene, _quiet(tiny_cfg, tissue_clutter=False))
    none = _acquire(tiny, water, scene, _quiet(tiny_cfg), tissue=None)
    np.testing.assert_array_equal(off, none)
    with_tissue = _acquire(tiny, water, scene, _quiet(tiny_cfg))
    clutter = with_tissue - none
    np.testing.assert_allclose(
        clutter, np.broadcast_to(clutter[0], clutter.shape), atol=1e-12 * np.abs(clutter).max()
    )
    assert np.abs(clutter).max() > 10 * np.abs(none).max()  # tissue is 40 dB above blood


def test_tissue_motion_makes_the_clutter_vary(tiny, tiny_cfg, water, scene):
    static = _acquire(tiny, water, scene, _quiet(tiny_cfg))
    moving = _acquire(
        tiny, water, scene, _quiet(tiny_cfg, tissue_motion=True, tissue_motion_rms=20e-6)
    )
    assert not np.allclose(moving, static, rtol=1e-3, atol=0.0)
    zero_rms = _acquire(
        tiny, water, scene, _quiet(tiny_cfg, tissue_motion=True, tissue_motion_rms=0.0)
    )
    np.testing.assert_array_equal(zero_rms, static)


def test_static_clutter_adds_the_same_image_to_every_frame(tiny, tiny_cfg, water, scene):
    rng = np.random.default_rng(2)
    fmc = (rng.normal(size=(28, 16, 16)) + 1j * rng.normal(size=(28, 16, 16))).astype(np.complex64)
    bank = replace(water, static_fmc=1e-6 * fmc)
    plain = _acquire(tiny, water, scene, _quiet(tiny_cfg))
    with_clutter = _acquire(
        tiny, water, scene, _quiet(tiny_cfg, static_clutter_scale=2.0), bank=bank
    )
    expected = 2.0 * das(
        static_clutter_spectra(bank, tiny.sequence), tiny.pixel_grid, water, tiny.sequence
    )
    difference = with_clutter - plain
    np.testing.assert_allclose(
        difference, np.broadcast_to(expected, difference.shape), atol=1e-5 * np.abs(expected).max()
    )
    switched_off = _acquire(tiny, water, scene, _quiet(tiny_cfg, static_clutter=False), bank=bank)
    np.testing.assert_array_equal(switched_off, plain)


def test_electronic_noise_is_added(tiny, tiny_cfg, water, scene):
    quiet = _acquire(tiny, water, scene, _quiet(tiny_cfg))
    noisy = _acquire(tiny, water, scene, _quiet(tiny_cfg, electronic_noise=True))
    assert not np.array_equal(noisy, quiet)


def test_unsuitable_banks_are_rejected(tiny, tiny_cfg, water, scene):
    tree, blood, tissue = scene
    coarse = replace(water, freqs=water.freqs[::4].copy(), B=water.B[::4].copy())
    arguments = (blood, tissue, tiny.sequence, tiny.pixel_grid, tiny_cfg, tree)
    with pytest.raises(ValueError, match="range aliasing"):
        acquire_frames(coarse, coarse, *arguments)
    with pytest.raises(ValueError, match="frequency grids"):
        acquire_frames(coarse, water, *arguments)


def test_power_doppler_image(tiny, tiny_cfg, water, scene):
    tree, blood, tissue = scene
    image = acquire_power_doppler(
        water, water, blood, tissue, tiny.sequence, tiny.pixel_grid, _quiet(tiny_cfg), tree
    )
    assert image.shape == (8, 8) and image.dtype == np.float64
    assert np.all(image >= 0.0) and image.max() > 0.0


def test_statistical_power_doppler_has_gamma_statistics():
    rng = np.random.default_rng(0)
    operator = rng.uniform(0.0, 1.0, size=(3, 5)).astype(np.float32)
    cbv_cells = rng.uniform(0.0, 1.0, size=5)
    floor = np.array([0.1, 0.2, 0.3])
    expected = operator.astype(np.float64) @ cbv_cells + floor
    n_effective = 12
    draws = np.stack(
        [
            statistical_power_doppler(operator, cbv_cells, floor, n_effective, rng)
            for _ in range(4000)
        ]
    )
    assert draws.shape == (4000, 3)
    np.testing.assert_allclose(draws.mean(axis=0), expected, rtol=0.03)
    np.testing.assert_allclose(draws.var(axis=0) / expected**2, 1.0 / n_effective, rtol=0.15)
    scalar_floor = statistical_power_doppler(operator, cbv_cells.reshape(5, 1), 0.5, 10**9, rng)
    np.testing.assert_allclose(
        scalar_floor, operator.astype(np.float64) @ cbv_cells + 0.5, rtol=1e-3
    )


def test_cost_estimate(tiny_cfg):
    cost = estimate_cost(tiny_cfg, 120)
    assert set(cost) == {"n_frames", "flops", "time_s", "memory_bytes"}
    assert cost["n_frames"] == 6 and all(value > 0 for value in cost.values())
    assert estimate_cost(tiny_cfg, 240)["time_s"] > cost["time_s"]
    statistical = estimate_cost(replace(tiny_cfg, fidelity="statistical"), 49)
    assert statistical["n_frames"] == 0 and statistical["time_s"] > 0
