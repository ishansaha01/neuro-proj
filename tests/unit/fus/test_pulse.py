"""Pulse spectrum, pulse length and the frequency grid."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.fus.pulse import band_frequencies, pulse_duration, pulse_spectrum
from neurosim.fus.types import Transducer

TRANSDUCER = Transducer(8, 3.85e-4, 3.4e-4, 2.0e6, 0.6)


def test_spectrum_peak_and_half_amplitude_width():
    f0, width = TRANSDUCER.center_frequency, 0.6 * 2.0e6
    spectrum = pulse_spectrum(TRANSDUCER, np.array([f0, f0 - width / 2, f0 + width / 2]))
    np.testing.assert_allclose(spectrum, [1.0, 0.5, 0.5], rtol=1e-12)
    assert spectrum.dtype == np.float64


def test_band_edges_are_at_the_truncation_level():
    freqs = band_frequencies(TRANSDUCER, 33, 20.0)
    assert freqs.shape == (33,)
    np.testing.assert_allclose(np.diff(freqs), freqs[1] - freqs[0], rtol=1e-12)
    np.testing.assert_allclose(freqs[16], TRANSDUCER.center_frequency)
    np.testing.assert_allclose(pulse_spectrum(TRANSDUCER, freqs[[0, -1]]), 0.1, rtol=1e-12)


def test_band_reaching_zero_frequency_raises():
    wide = Transducer(8, 3.85e-4, 3.4e-4, 2.0e6, 1.5)
    with pytest.raises(ValueError, match="band edge"):
        band_frequencies(wide, 16, 40.0)


def test_pulse_duration_matches_the_fourier_pair():
    """The envelope of the sampled spectrum falls to -20 dB at +- half the stated duration."""
    duration = pulse_duration(TRANSDUCER, 20.0)
    freqs = np.linspace(0.2e6, 3.8e6, 2001)
    spectrum = pulse_spectrum(TRANSDUCER, freqs)
    envelope_at_edge = abs(np.sum(spectrum * np.exp(-2j * np.pi * freqs * duration / 2)))
    # The frequency integral stops 3.5 standard deviations from the centre: 0.1 % error.
    np.testing.assert_allclose(envelope_at_edge / spectrum.sum(), 0.1, rtol=5e-3)
