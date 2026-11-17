"""Electronic noise and tissue micro-motion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.fus.noise import (
    electronic_noise,
    electronic_noise_variance,
    nominal_blood_echo_power,
    tissue_displacement,
)
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import Scatterers


def test_noise_has_the_requested_variance_and_is_circular():
    noise = electronic_noise(SeedTree(0), 3, (40, 5, 50), 2.5)
    assert noise.shape == (40, 5, 50) and noise.dtype == np.complex128
    assert np.mean(np.abs(noise) ** 2) == pytest.approx(2.5, rel=0.03)
    assert np.mean(noise.real**2) == pytest.approx(1.25, rel=0.05)
    assert abs(np.mean(noise**2)) < 0.1  # no preferred phase


def test_noise_depends_only_on_seed_and_frame():
    tree = SeedTree(5)
    first = electronic_noise(tree, 7, (4, 3, 8), 1.0)
    electronic_noise(tree, 2, (4, 3, 8), 1.0)  # drawing another frame changes nothing
    np.testing.assert_array_equal(electronic_noise(SeedTree(5), 7, (4, 3, 8), 1.0), first)
    assert not np.array_equal(electronic_noise(tree, 8, (4, 3, 8), 1.0), first)
    assert not np.array_equal(electronic_noise(SeedTree(6), 7, (4, 3, 8), 1.0), first)


def test_nominal_power_is_the_unit_scatterer_power_times_blood_area(tiny, water):
    centre = np.array([[0.0, 0.023]])
    unit = channel_spectra(water, Scatterers(xz=centre, amplitude=np.ones(1)), tiny.sequence)
    area = 49 * tiny.roi.spacing**2
    expected = 0.03 * area * np.mean(np.abs(unit) ** 2)
    assert nominal_blood_echo_power(water, tiny.sequence, 0.03) == pytest.approx(expected, rel=1e-9)


def test_noise_level_follows_the_snr_and_ignores_the_medium(tiny, tiny_cfg, water):
    variance = electronic_noise_variance(water, tiny.sequence, tiny_cfg.noise)
    quieter = replace(tiny_cfg.noise, electronic_snr_db=tiny_cfg.noise.electronic_snr_db + 10.0)
    assert electronic_noise_variance(water, tiny.sequence, quieter) == pytest.approx(
        variance / 10.0
    )
    attenuating = apply_phase_screen(water, np.zeros(16), np.full(16, 0.1))
    assert electronic_noise_variance(attenuating, tiny.sequence, tiny_cfg.noise) == variance


def test_tissue_displacement_rms_and_determinism(tiny_cfg):
    noise = replace(tiny_cfg.noise, tissue_motion_rms=30e-6, tissue_motion_frequencies=(0.25, 1.2))
    times = np.linspace(0.0, 400.0, 40001)  # many periods of both components
    displacement = tissue_displacement(SeedTree(1), noise, times)
    assert displacement.shape == (40001, 2)
    rms = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
    assert rms == pytest.approx(30e-6, rel=0.02)
    np.testing.assert_array_equal(
        tissue_displacement(SeedTree(1), noise, times[:5]), displacement[:5]
    )
    assert not np.array_equal(tissue_displacement(SeedTree(2), noise, times[:5]), displacement[:5])


def test_tissue_displacement_is_a_function_of_absolute_time(tiny_cfg):
    times = np.arange(10) / 500.0
    whole = tissue_displacement(SeedTree(3), tiny_cfg.noise, times)
    for index in (0, 4, 9):
        single = tissue_displacement(SeedTree(3), tiny_cfg.noise, times[index : index + 1])
        np.testing.assert_array_equal(single[0], whole[index])
