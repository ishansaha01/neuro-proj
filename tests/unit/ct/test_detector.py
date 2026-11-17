"""Detector model: Beer-Lambert signal, photon noise and the log transform."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.ct.attenuation import basis_attenuation
from neurosim.ct.detector import detector_signal, flat_field_counts, log_projection
from neurosim.ct.spectrum import xray_spectrum


@pytest.fixture(scope="module")
def beam(small_cfg):
    energies, counts = flat_field_counts(small_cfg)
    return energies, counts, basis_attenuation(energies)


def test_flat_field_counts_scale_with_mas_views_and_area(small_cfg):
    energies, counts = flat_field_counts(small_cfg)
    fluence = xray_spectrum(small_cfg)[1]
    geom = small_cfg.geometry
    np.testing.assert_allclose(
        counts, fluence * small_cfg.mas / geom.n_views * geom.detector_pitch * geom.detector_height
    )
    double = dataclasses.replace(small_cfg, mas=2.0 * small_cfg.mas)
    np.testing.assert_allclose(flat_field_counts(double)[1], 2.0 * counts)
    assert energies.shape == counts.shape


def test_no_object_gives_the_flat_field_signal(beam):
    energies, counts, attenuation = beam
    signal = detector_signal(np.zeros((4, 3, 5)), attenuation, counts, energies)
    assert signal.shape == (3, 5)
    np.testing.assert_allclose(signal, counts @ energies)


def test_signal_decreases_with_thickness_and_hardens(beam):
    energies, counts, attenuation = beam
    thickness = np.array([0.0, 0.1, 0.2])
    paths = np.zeros((4, 3))
    paths[1] = thickness  # water
    signal = detector_signal(paths, attenuation, counts, energies)
    projection = -np.log(signal / (counts @ energies))
    assert np.all(np.diff(projection) > 0.0)
    # Beam hardening: the second 10 cm of water attenuates less than the first.
    assert projection[2] - projection[1] < projection[1] - projection[0]


def test_noise_is_seeded_unbiased_and_scalable(beam):
    energies, counts, attenuation = beam
    paths = np.zeros((4, 2000))
    paths[1] = 0.15
    expected = detector_signal(paths, attenuation, counts, energies)
    first = detector_signal(
        paths, attenuation, counts, energies, rng=SeedTree(1).rng("ct_photon_noise", 0)
    )
    again = detector_signal(
        paths, attenuation, counts, energies, rng=SeedTree(1).rng("ct_photon_noise", 0)
    )
    half = detector_signal(
        paths,
        attenuation,
        counts,
        energies,
        rng=SeedTree(1).rng("ct_photon_noise", 0),
        noise_scale=0.5,
    )
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, expected)
    np.testing.assert_allclose(half - expected, 0.5 * (first - expected))
    relative = (first - expected) / expected
    assert abs(relative.mean()) < 5.0 * relative.std() / np.sqrt(relative.size)


def test_log_projection_handles_zero_and_negative_signal():
    projection = log_projection(np.array([100.0, 1.0, 0.0, -3.0]), 100.0, floor=0.5)
    assert np.all(np.isfinite(projection))
    assert projection[0] == 0.0
    assert projection[2] == projection[3] == pytest.approx(np.log(200.0))
