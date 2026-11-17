"""Unambiguous time window of the frequency grid."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.pulse import band_frequencies
from neurosim.fus.types import PlaneWaveSequence, Transducer
from neurosim.fus.window import check_time_window, round_trip_span, unambiguous_window

TRANSDUCER = Transducer(1, 3.85e-4, 3.4e-4, 2.0e6, 0.6)
ON_AXIS = PlaneWaveSequence(angles_deg=(0.0,), frame_rate=500.0, n_frames=1)


def test_unambiguous_window_is_one_over_spacing():
    assert unambiguous_window(np.linspace(1e6, 3e6, 41)) == pytest.approx(1.0 / 50e3)
    with pytest.raises(ValueError):
        unambiguous_window(np.array([2e6]))


def test_round_trip_span_of_a_column_under_one_element():
    """On the axis of a single element the round trip is 2 z / c, so the span is 2 * extent / c."""
    column = Grid2D(shape=(11, 1), spacing=1e-3, origin=(0.02, 0.0))
    # 11 cells of 1 mm: 11 mm between the outer faces; the half-cell lateral extent adds < 1e-3.
    span = round_trip_span(TRANSDUCER, ON_AXIS, column, C_REF)
    assert span == pytest.approx(2 * 0.011 / C_REF, rel=1e-3)


def test_span_grows_with_steering_and_aperture():
    grid = Grid2D(shape=(11, 11), spacing=1e-3, origin=(0.02, -0.005))
    array = Transducer(32, 3.85e-4, 3.4e-4, 2.0e6, 0.6)
    steered = PlaneWaveSequence(angles_deg=(-10.0, 0.0, 10.0), frame_rate=500.0, n_frames=1)
    assert round_trip_span(array, steered, grid, C_REF) > round_trip_span(
        array, ON_AXIS, grid, C_REF
    )


def test_check_accepts_enough_frequencies_and_rejects_too_few():
    grid = Grid2D(shape=(11, 11), spacing=1e-3, origin=(0.02, -0.005))
    array = Transducer(32, 3.85e-4, 3.4e-4, 2.0e6, 0.6)
    check_time_window(band_frequencies(array, 64, 20.0), array, ON_AXIS, grid, C_REF)
    with pytest.raises(ValueError, match="range aliasing"):
        check_time_window(band_frequencies(array, 16, 20.0), array, ON_AXIS, grid, C_REF)
