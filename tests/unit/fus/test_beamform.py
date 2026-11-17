"""Delay-and-sum beamforming."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.fus.beamform import beamform, das, make_beamformer, receive_apodization
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import Scatterers


def _pixel_scatterer(tiny, iz: int, ix: int, amplitude: complex = 1.0) -> Scatterers:
    return Scatterers(xz=tiny.pixel_grid.points()[iz, ix][None, :], amplitude=np.array([amplitude]))


def test_unit_gain_for_a_scatterer_on_a_pixel(tiny, water):
    """With the true medium as model, a scatterer of amplitude s on a pixel is imaged as s."""
    amplitude = 0.7 - 0.2j
    image = das(
        channel_spectra(water, _pixel_scatterer(tiny, 3, 4, amplitude), tiny.sequence),
        tiny.pixel_grid,
        water,
        tiny.sequence,
    )
    assert image.shape == tiny.pixel_grid.shape and image.dtype == np.complex128
    assert image[3, 4] == pytest.approx(amplitude, rel=1e-5)
    assert np.unravel_index(np.argmax(np.abs(image)), image.shape) == (3, 4)


def test_true_model_restores_unit_gain_behind_a_phase_screen(tiny, water):
    rng = np.random.default_rng(0)
    screen = apply_phase_screen(water, rng.normal(0.0, 1e-7, 16), rng.uniform(0.5, 1.0, 16))
    spectra = channel_spectra(screen, _pixel_scatterer(tiny, 5, 2), tiny.sequence)
    corrected = das(spectra, tiny.pixel_grid, screen, tiny.sequence)
    conventional = das(spectra, tiny.pixel_grid, water, tiny.sequence)
    assert corrected[5, 2] == pytest.approx(1.0, rel=1e-5)
    assert abs(conventional[5, 2]) < 0.9


def test_das_equals_precomputed_weights(tiny, water):
    spectra = channel_spectra(water, _pixel_scatterer(tiny, 1, 1), tiny.sequence)
    options = BeamformingConfig(f_number=2.0, window="hann")
    beamformer = make_beamformer(tiny.pixel_grid, water, tiny.sequence, options)
    np.testing.assert_array_equal(
        beamform(beamformer, spectra), das(spectra, tiny.pixel_grid, water, tiny.sequence, options)
    )


def test_beamforming_is_linear(tiny, water):
    beamformer = make_beamformer(tiny.pixel_grid, water, tiny.sequence)
    rng = np.random.default_rng(1)
    shape = (28, 3, 16)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    b = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    np.testing.assert_allclose(
        beamform(beamformer, a + 2j * b),
        beamform(beamformer, a) + 2j * beamform(beamformer, b),
        rtol=1e-4,
        atol=1e-6 * np.abs(beamform(beamformer, a)).max(),
    )


def test_noise_gain_is_the_sum_of_squared_weights(tiny, water):
    beamformer = make_beamformer(
        tiny.pixel_grid, water, tiny.sequence, BeamformingConfig(f_number=1.5, window="hann")
    )
    weights = beamformer.tx_weights[:, :, None, :] * beamformer.rx_weights[:, None, :, :]
    expected = np.sum(np.abs(weights.astype(np.complex128)) ** 2, axis=(0, 1, 2))
    np.testing.assert_allclose(beamformer.noise_gain.reshape(-1), expected, rtol=1e-5)


def test_weights_have_model_phase(tiny, water):
    beamformer = make_beamformer(tiny.pixel_grid, water, tiny.sequence)
    assert (
        beamformer.tx_weights.shape == (28, 3, 64) and beamformer.tx_weights.dtype == np.complex64
    )
    assert beamformer.rx_weights.shape == (28, 16, 64)
    np.testing.assert_allclose(np.abs(beamformer.tx_weights), 1.0, rtol=1e-5)


def test_receive_apodization(tiny):
    transducer, grid = tiny.transducer, tiny.pixel_grid
    full = receive_apodization(transducer, grid, BeamformingConfig())
    assert full.shape == (16, 64)
    np.testing.assert_array_equal(full, 1.0)

    f_number = 3.0
    boxcar = receive_apodization(transducer, grid, BeamformingConfig(f_number=f_number))
    points = grid.points().reshape(-1, 2)
    inside = np.abs(transducer.element_x()[:, None] - points[None, :, 0]) <= points[None, :, 1] / (
        2 * f_number
    )
    np.testing.assert_array_equal(boxcar, inside.astype(float))
    assert 0 < boxcar.sum() < boxcar.size

    hann = receive_apodization(
        transducer, grid, BeamformingConfig(f_number=f_number, window="hann")
    )
    assert np.all((hann >= 0.0) & (hann <= 1.0))
    assert np.all(hann[~inside] == 0.0)
    assert np.all(hann <= boxcar)


def test_wrong_spectra_shape_raises(tiny, water):
    beamformer = make_beamformer(tiny.pixel_grid, water, tiny.sequence)
    with pytest.raises(ValueError, match="spectra have shape"):
        beamform(beamformer, np.zeros((28, 16, 3), dtype=complex))
