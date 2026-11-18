"""Metrics added for the second design (`recon.metrics`)."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.types import GreensBank, Transducer
from neurosim.recon.metrics import (
    bank_delays,
    half_maximum_dice,
    map_cnr,
    map_correlation,
    non_finite_reason,
    travel_time_error,
    weighted_delay_rms,
)


def _bank(delays: np.ndarray, amplitudes: np.ndarray) -> GreensBank:
    array = Transducer(8, 4e-4, 3e-4, 2e6, 0.6)
    roi = Grid2D(shape=(3, 4), spacing=1e-3, origin=(0.02, -0.0015))
    freqs = np.linspace(1.5e6, 2.5e6, 21)
    return apply_phase_screen(GreensBank.homogeneous(array, roi, freqs, C_REF), delays, amplitudes)


def test_map_metrics():
    rng = np.random.default_rng(3)  # noqa: NPY002 - fixed seed, test data only
    truth = rng.random((6, 6))
    mask = np.ones((6, 6), dtype=bool)
    mask[0] = False
    assert map_correlation(3.0 * truth + 1.0, truth, mask) == pytest.approx(1.0)
    assert map_correlation(np.ones((6, 6)), truth, mask) == 0.0
    estimate = np.zeros((6, 6))
    estimate[2:4, 2:4] = 1.0
    estimate[0, 0] = 50.0  # outside the mask: must not set the maximum
    region = np.zeros((6, 6), dtype=bool)
    region[2:4, 2:5] = True
    assert half_maximum_dice(estimate, region, mask, 0.5) == pytest.approx(2 * 4 / (4 + 6))
    assert half_maximum_dice(-np.ones((6, 6)), region, mask, 0.5) == 0.0
    reference = ~region & mask
    noisy = estimate + 0.1 * rng.standard_normal((6, 6))
    noisy[0] = 0.0
    assert map_cnr(noisy, estimate > 0.5, reference) > 3.0
    with pytest.raises(ValueError, match="empty"):
        map_cnr(noisy, np.zeros((6, 6), dtype=bool), reference)


def test_bank_delays_recover_a_phase_screen():
    delays = np.linspace(-80e-9, 120e-9, 8)
    amplitudes = np.linspace(0.2, 1.0, 8)
    delay, amplitude = bank_delays(_bank(delays, amplitudes))
    assert delay.shape == amplitude.shape == (8, 12)
    np.testing.assert_allclose(delay, delays[:, None] * np.ones((1, 12)), atol=1e-12)
    np.testing.assert_allclose(amplitude, amplitudes[:, None] * np.ones((1, 12)), rtol=1e-6)


def test_travel_time_error_removes_the_gauge():
    truth_delays = np.linspace(-50e-9, 50e-9, 8)
    amplitudes = np.ones(8)
    truth = _bank(truth_delays, amplitudes)
    shifted = _bank(truth_delays + 300e-9, amplitudes)
    assert travel_time_error(shifted, truth, "common") == pytest.approx(0.0, abs=1e-12)
    error = np.array([10, -10, 10, -10, 10, -10, 10, -10]) * 1e-9
    model = _bank(truth_delays + error, amplitudes)
    assert travel_time_error(model, truth, "common") == pytest.approx(10e-9, rel=1e-6)
    assert travel_time_error(model, truth, "node") == pytest.approx(10e-9, rel=1e-6)
    # Pairs to which the model gives no amplitude do not count.
    dark = np.array(model.B)
    dark[:, 0] = 0.0
    partial = GreensBank(model.freqs, model.roi, dark, model.c_ref, model.transducer, "test")
    assert travel_time_error(partial, truth, "common") < 10.5e-9


def test_weighted_delay_rms_gauges():
    difference = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]]) * 1e-9
    weight = np.ones((3, 2))
    # A delay common to the elements of each node is no focusing error.
    assert weighted_delay_rms(difference, weight, "node") == pytest.approx(
        np.sqrt(((np.array([-2.0, 0.0, 2.0]) * 1e-9) ** 2).mean() / 2)
    )
    assert weighted_delay_rms(difference, weight, "common") > weighted_delay_rms(
        difference, weight, "node"
    )
    with pytest.raises(ValueError, match="gauge"):
        weighted_delay_rms(difference, weight, "tilt")


def test_non_finite_reason():
    assert non_finite_reason({"images": np.ones(4)}) is None
    values = np.array([1.0, np.nan, np.inf, np.nan])
    reason = non_finite_reason({"good": np.ones(3), "images": values})
    assert reason == "images: 3 of 4 values not finite (2 NaN, 1 infinite)"
