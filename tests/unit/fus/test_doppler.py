"""SVD clutter filter, cutoff rules and power Doppler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.fus.doppler import choose_svd_cutoff, power_doppler, svd_clutter_filter

N_FRAMES, SHAPE, FRAME_RATE = 24, (5, 6), 500.0


def _blood(rng: np.random.Generator, doppler_hz: float = 80.0) -> np.ndarray:
    """Weak signal with an independent spatial pattern per frame and a mean Doppler shift."""
    noise = rng.normal(size=(N_FRAMES, *SHAPE)) + 1j * rng.normal(size=(N_FRAMES, *SHAPE))
    rotation = np.exp(2j * np.pi * doppler_hz * np.arange(N_FRAMES) / FRAME_RATE)
    pattern = rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)
    return 0.01 * (noise + 3.0 * rotation[:, None, None] * pattern)


def _static(rng: np.random.Generator) -> np.ndarray:
    image = rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)
    return np.broadcast_to(image, (N_FRAMES, *SHAPE)).copy()


def _cfg(tiny_cfg, **doppler):
    return replace(
        tiny_cfg,
        sequence=replace(tiny_cfg.sequence, frame_rate=FRAME_RATE, n_frames=N_FRAMES),
        doppler=replace(tiny_cfg.doppler, **doppler),
    )


def test_filter_removes_a_static_image():
    rng = np.random.default_rng(0)
    blood, static = _blood(rng), _static(rng)
    filtered = svd_clutter_filter(blood + static, 1)
    assert filtered.shape == blood.shape and filtered.dtype == np.complex128
    assert np.sum(np.abs(static) ** 2) > 100 * np.sum(np.abs(blood) ** 2)
    assert np.sum(np.abs(filtered - blood) ** 2) < 0.2 * np.sum(np.abs(blood) ** 2)


def test_filter_limits():
    frames = _blood(np.random.default_rng(1))
    np.testing.assert_allclose(svd_clutter_filter(frames, 0), frames, atol=1e-14)
    np.testing.assert_allclose(svd_clutter_filter(frames, N_FRAMES), 0.0, atol=1e-14)
    with pytest.raises(ValueError, match="n_discard"):
        svd_clutter_filter(frames, N_FRAMES + 1)
    with pytest.raises(ValueError, match="shape"):
        svd_clutter_filter(frames[0], 1)


def test_filter_removes_the_largest_components():
    frames = _blood(np.random.default_rng(2)) + _static(np.random.default_rng(3))
    values = np.linalg.svd(frames.reshape(N_FRAMES, -1), compute_uv=False)
    kept = np.linalg.svd(svd_clutter_filter(frames, 3).reshape(N_FRAMES, -1), compute_uv=False)
    np.testing.assert_allclose(kept[: N_FRAMES - 3], values[3:], rtol=1e-9)


def test_fixed_rule(tiny_cfg):
    frames = _blood(np.random.default_rng(4))
    assert choose_svd_cutoff(frames, _cfg(tiny_cfg, cutoff_rule="fixed", n_discard=5)) == 5


def test_energy_rule_counts_the_dominant_components(tiny_cfg):
    rng = np.random.default_rng(5)
    slow = np.cos(2 * np.pi * np.arange(N_FRAMES) / N_FRAMES)[:, None, None] * _static(rng)
    frames = _blood(rng) + _static(rng) + slow  # two strong components
    assert (
        choose_svd_cutoff(frames, _cfg(tiny_cfg, cutoff_rule="energy", energy_fraction=0.99)) == 2
    )


def test_frequency_rule_stops_at_the_first_fast_component(tiny_cfg):
    rng = np.random.default_rng(6)
    drift = np.exp(2j * np.pi * 2.0 * np.arange(N_FRAMES) / FRAME_RATE)[:, None, None]  # 2 Hz
    frames = _blood(rng) + _static(rng) + 0.3 * drift * _static(rng)
    cfg = _cfg(tiny_cfg, cutoff_rule="frequency", cutoff_frequency=20.0)
    assert choose_svd_cutoff(frames, cfg) == 2


def test_cutoff_never_removes_everything(tiny_cfg):
    frames = _blood(np.random.default_rng(7))
    cfg = _cfg(tiny_cfg, cutoff_rule="energy", energy_fraction=0.999999)
    assert choose_svd_cutoff(frames, cfg) == N_FRAMES - 1


def test_power_doppler_is_the_mean_power():
    frames = np.array([[[1 + 1j]], [[3 + 0j]]])
    np.testing.assert_allclose(power_doppler(frames), [[(2.0 + 9.0) / 2]])
    assert power_doppler(frames).dtype == np.float64
