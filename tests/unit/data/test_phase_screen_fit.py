"""fus.screen.fit_phase_screen (kept with the data tests: `PhaseScreen` is a data contract type)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.config import from_dict
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.data.types import PhaseScreen
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.screen import apply_phase_screen, fit_phase_screen
from neurosim.fus.types import GreensBank

N_ELEMENTS = 16


@pytest.fixture(scope="module")
def water() -> GreensBank:
    cfg = from_dict(
        ImagingConfig,
        {
            "transducer": {"n_elements": N_ELEMENTS},
            "sequence": {"angles_deg": [0.0], "n_frames": 6},
            "frequencies": {"n_frequencies": 28},
            "roi": {"x_range": [-0.003, 0.003], "z_range": [0.020, 0.026], "spacing": 1.0e-3},
            "pixels": {"shape": [8, 8]},
            "doppler": {"n_discard": 1},
        },
    )
    setup = imaging_setup(cfg)
    return GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)


def test_fit_recovers_a_screen_whose_carrier_phase_wraps_many_times(water):
    rng = np.random.default_rng(0)
    # Up to 3 us is six periods at 2 MHz; the limit of the estimator is 1 / (2 df), about 6 us.
    delays = rng.uniform(-3.0e-6, 3.0e-6, N_ELEMENTS)
    amplitudes = rng.uniform(0.05, 1.5, N_ELEMENTS)
    assert np.abs(water.transducer.center_frequency * delays).max() > 5.0  # carrier periods
    fitted = fit_phase_screen(apply_phase_screen(water, delays, amplitudes))
    assert isinstance(fitted, PhaseScreen)
    assert fitted.delays.shape == fitted.amplitudes.shape == (N_ELEMENTS,)
    assert fitted.delays.dtype == np.float64
    np.testing.assert_allclose(fitted.delays, delays, atol=1e-12)  # [s]
    np.testing.assert_allclose(fitted.amplitudes, amplitudes, rtol=1e-6)


def test_homogeneous_bank_gives_the_identity_screen(water):
    fitted = fit_phase_screen(water)
    np.testing.assert_array_equal(fitted.delays, 0.0)
    np.testing.assert_allclose(fitted.amplitudes, 1.0, rtol=1e-12)


def test_region_selects_the_points_that_are_averaged(water):
    """A bank whose delay grows with depth: the fit is the mean delay over the region."""
    depth = water.roi.axes()[0]
    slope = 1.0e-4  # [s/m]: 0.1 us per mm of depth
    phase = 2j * np.pi * water.freqs[:, None, None, None] * slope * depth[None, None, :, None]
    bank = replace(water, B=(water.B * np.exp(phase)).astype(np.complex64))
    whole = fit_phase_screen(bank)
    np.testing.assert_allclose(whole.delays, slope * depth.mean(), rtol=1e-6)
    shallow = Grid2D.from_extent((-0.002, 0.002), (0.020, 0.022), 1.0e-3)
    np.testing.assert_allclose(fit_phase_screen(bank, shallow).delays, slope * 0.021, rtol=1e-6)


def test_amplitude_is_weighted_by_the_pulse_spectrum(water):
    """Loss that grows with frequency: the fitted amplitude is the pulse-weighted mean."""
    from neurosim.fus.pulse import pulse_spectrum

    loss = np.exp(-(water.freqs / water.freqs[-1]))
    bank = replace(water, B=(water.B * loss[:, None, None, None]).astype(np.complex64))
    weights = pulse_spectrum(water.transducer, water.freqs)
    expected = np.sum(weights * loss) / weights.sum()
    np.testing.assert_allclose(fit_phase_screen(bank).amplitudes, expected, rtol=1e-6)
    assert loss.mean() != pytest.approx(expected, rel=1e-3)


def test_a_single_frequency_is_refused(water):
    one = GreensBank.homogeneous(water.transducer, water.roi, water.freqs[:1], C_REF)
    with pytest.raises(ValueError, match="two frequencies"):
        fit_phase_screen(one)
