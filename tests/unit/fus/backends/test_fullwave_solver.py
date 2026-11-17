import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.fus.backends.fullwave_solver import check_absorption_model, pml_sizes, simulate

SPACING = 0.3e-3  # [m]
DT = 0.3 * SPACING / C_REF  # [s]


def test_pml_sizes_stay_in_the_configured_range(fast_kwave):
    sizes = pml_sizes((41, 57), fast_kwave)
    assert len(sizes) == 2 and all(isinstance(size, int) and 8 <= size <= 12 for size in sizes)
    # The padded sizes are FFT-friendly: no prime factor above 7.
    for n_points, size in zip((41, 57), sizes):
        padded = n_points + 2 * size
        for prime in (2, 3, 5, 7):
            while padded % prime == 0:
                padded //= prime
        assert padded == 1
    # A short axis limits the layer: k-Wave needs the grid longer than twice the layer.
    assert all(8 <= size <= 9 for size in pml_sizes((19, 20), fast_kwave))
    assert pml_sizes((17, 17), fast_kwave) == (8, 8)
    with pytest.raises(ValueError, match="too short"):
        pml_sizes((16, 41), fast_kwave)


def test_absorption_model_checks(fast_kwave):
    lossless, lossy = np.zeros((4, 4)), np.full((4, 4), 2.0)
    dispersive = dataclasses.replace(fast_kwave, model_dispersion=True)
    check_absorption_model(lossless, 1.0, dispersive)  # the exponent is unused without absorption
    check_absorption_model(lossy, 1.5, dispersive)
    check_absorption_model(lossy, 2.0, dispersive)
    with pytest.raises(ValueError, match="0 <= alpha_power < 3"):
        check_absorption_model(lossy, 3.0, fast_kwave)
    with pytest.raises(ValueError, match="singular"):
        check_absorption_model(lossy, 1.02, dispersive)
    assert not fast_kwave.model_dispersion
    check_absorption_model(lossy, 1.0, fast_kwave)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_simulate_orders_sensors_by_row_then_column_and_keeps_the_orientation(fast_kwave, dtype):
    shape = (48, 36)  # (nz, nx), deliberately not square
    source_mask = np.zeros(shape, dtype=bool)
    source_mask[5, 10] = True
    sensors = [(5, 28), (11, 10), (41, 10)]  # 18 cells along x, 6 and 36 cells along z
    sensor_mask = np.zeros(shape, dtype=bool)
    for row, column in sensors[::-1]:  # the order of filling must not matter
        sensor_mask[row, column] = True
    n_steps = 560
    t = np.arange(120) * DT
    centred = t - 60 * DT
    pulse = np.exp(-0.5 * (centred / (15 * DT)) ** 2) * np.sin(2 * np.pi * centred / (40 * DT))

    recording = simulate(
        sound_speed=np.full(shape, C_REF),
        density=np.full(shape, 1000.0),
        attenuation=np.zeros(shape),
        alpha_power=1.5,
        spacing=SPACING,
        dt=DT,
        n_steps=n_steps,
        source_mask=source_mask,
        source_signals=pulse[None, :],
        sensor_mask=sensor_mask,
        pml=(8, 10),
        cfg=dataclasses.replace(fast_kwave, dtype=dtype),
    )

    assert recording.shape == (3, n_steps) and recording.dtype == np.dtype(dtype)
    arrival_steps = np.argmax(np.abs(recording), axis=1)
    distances = np.array([18, 6, 36]) * SPACING  # in the order (5, 28), (11, 10), (41, 10)
    np.testing.assert_allclose(arrival_steps * DT, 60 * DT + distances / C_REF, atol=12 * DT)
    # Cylindrical spreading: amplitude falls roughly as 1 / sqrt(distance).
    peaks = np.abs(recording).max(axis=1)
    assert peaks[1] > peaks[0] > peaks[2]
