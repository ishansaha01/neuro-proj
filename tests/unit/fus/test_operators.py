"""Point-spread and power Doppler operators."""

from __future__ import annotations

import numpy as np

from neurosim.fus import operators
from neurosim.fus.beamform import das
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import Scatterers


def _screen(water):
    rng = np.random.default_rng(0)
    return apply_phase_screen(water, rng.normal(0.0, 5e-8, 16), rng.uniform(0.7, 1.3, 16))


def test_columns_are_images_of_unit_scatterers(tiny, water):
    bank = _screen(water)
    options = BeamformingConfig(f_number=2.0)
    operator = psf_operator(bank, tiny.pixel_grid, tiny.roi, water, tiny.sequence, options)
    assert operator.shape == (64, 49) and operator.dtype == np.complex64
    sources = tiny.roi.points().reshape(-1, 2)
    for column in (0, 17, 48):
        scatterer = Scatterers(xz=sources[column : column + 1], amplitude=np.ones(1))
        image = das(
            channel_spectra(bank, scatterer, tiny.sequence),
            tiny.pixel_grid,
            water,
            tiny.sequence,
            options,
        )
        np.testing.assert_allclose(
            operator[:, column], image.reshape(-1), atol=2e-5 * np.abs(image).max()
        )


def test_result_does_not_depend_on_the_frequency_chunking(tiny, water, monkeypatch):
    bank = _screen(water)
    whole = psf_operator(bank, tiny.pixel_grid, tiny.roi, water, tiny.sequence)
    monkeypatch.setattr(operators, "_CHUNK_VALUES", 1)  # smallest passes: four frequencies each
    chunked = psf_operator(bank, tiny.pixel_grid, tiny.roi, water, tiny.sequence)
    np.testing.assert_allclose(chunked, whole, atol=2e-5 * np.abs(whole).max())


def test_true_model_gives_unit_diagonal(tiny, water):
    bank = _screen(water)
    operator = psf_operator(bank, tiny.pixel_grid, tiny.pixel_grid, bank, tiny.sequence)
    np.testing.assert_allclose(np.diag(operator), 1.0, atol=1e-4)


def test_pd_operator_is_the_squared_magnitude():
    operator = np.array([[3 + 4j, 0], [1j, 2]], dtype=np.complex64)
    power = pd_operator(operator)
    assert power.dtype == np.float32
    np.testing.assert_allclose(power, [[25.0, 0.0], [1.0, 4.0]])
