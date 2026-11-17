"""Phase-screen banks."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.fus.screen import apply_phase_screen


def test_screen_multiplies_each_element_by_amplitude_and_delay_phase(tiny, water):
    delays = np.linspace(-2e-7, 2e-7, 16)
    amplitudes = np.linspace(0.5, 1.5, 16)
    screen = apply_phase_screen(water, delays, amplitudes)
    assert screen.B.shape == water.B.shape and screen.B.dtype == np.complex64
    assert screen.backend == "phase_screen" and screen.roi == water.roi
    expected = amplitudes[3] * np.exp(2j * np.pi * tiny.freqs[10] * delays[3])  # exp(+i w tau)
    np.testing.assert_allclose(screen.B[10, 3], expected, rtol=1e-5)
    np.testing.assert_array_equal(water.B, 1.0)  # the input bank is not modified


def test_wrong_number_of_elements_raises(water):
    with pytest.raises(ValueError, match="shape"):
        apply_phase_screen(water, np.zeros(15), np.ones(16))
