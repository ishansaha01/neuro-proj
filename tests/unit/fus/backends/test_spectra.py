import numpy as np
import pytest

from neurosim.fus.backends.spectra import gaussian_pulse, spectra_at

DT = 20.0e-9  # [s]
FREQS = np.linspace(0.3e6, 0.7e6, 9)


def test_pulse_is_zero_mean_and_covers_the_band_at_the_requested_level():
    pulse = gaussian_pulse(FREQS, DT, edge_level_db=6.0, max_cycles=10.0)
    assert abs(pulse.sum()) < 1e-12 * np.abs(pulse).sum()
    assert pulse.size % 2 == 1 and pulse[pulse.size // 2] == 0.0
    assert abs(pulse[0]) < 1e-3 and abs(pulse[-1]) < 1e-3
    spectrum = np.abs(spectra_at(pulse[None, :], DT, FREQS)[:, 0])
    level_db = 20.0 * np.log10(spectrum / spectrum.max())
    assert np.argmax(spectrum) == 4
    np.testing.assert_allclose(level_db[[0, -1]], -6.0, atol=0.3)


def test_pulse_length_is_capped_for_a_single_frequency():
    pulse = gaussian_pulse(np.array([0.5e6]), DT, edge_level_db=6.0, max_cycles=2.0)
    # Envelope sigma = 2 periods = 4 us, cut at +-4 sigma.
    assert pulse.size * DT == pytest.approx(32.0e-6, rel=0.01)


def test_spectra_follow_the_exp_minus_i_omega_t_convention():
    t = np.arange(4000) * DT
    pulse = np.exp(-0.5 * ((t - 20.0e-6) / 2.0e-6) ** 2) * np.sin(2 * np.pi * 0.5e6 * (t - 20.0e-6))
    delay = 150 * DT
    delayed = np.roll(pulse, 150)
    spectra = spectra_at(np.stack((pulse, delayed)), DT, FREQS)
    assert spectra.shape == (9, 2) and spectra.dtype == np.complex128
    # A delay tau multiplies the spectrum by exp(+i w tau).
    np.testing.assert_allclose(
        spectra[:, 1] / spectra[:, 0], np.exp(2j * np.pi * FREQS * delay), rtol=1e-6
    )


def test_spectra_equal_the_direct_sum_for_any_frequency_and_dtype():
    signal = np.sin(np.arange(300) * 0.37) * np.linspace(0.0, 1.0, 300)
    freqs = np.array([0.123e6, 0.777e6])
    expected = DT * np.array(
        [np.sum(signal * np.exp(2j * np.pi * f * np.arange(300) * DT)) for f in freqs]
    )
    np.testing.assert_allclose(spectra_at(signal[None, :], DT, freqs)[:, 0], expected, rtol=1e-12)
    single = spectra_at(signal[None, :].astype(np.float32), DT, freqs)[:, 0]
    np.testing.assert_allclose(single, expected, rtol=1e-5)
