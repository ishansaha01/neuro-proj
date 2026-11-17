"""Near-field phase screen: per-element delay and amplitude applied to a bank.

A thin aberrating layer on the array face delays and scales what each element
sends and receives, whatever the field point:

    B_screen[f, e, r] = B[f, e, r] * a_e * exp(+i 2 pi f tau_e)

This is the simplest model of skull aberration, and the low-dimensional skull
correction (per-element delays and amplitudes) that `recon` estimates.
`apply_phase_screen` applies a screen to a bank; `fit_phase_screen` reduces a
bank to the screen that describes it best.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.types import GreensBank, PhaseScreen


def apply_phase_screen(bank: GreensBank, delays: np.ndarray, amplitudes: np.ndarray) -> GreensBank:
    """Return a copy of a bank with a per-element delay and amplitude applied.

    Args:
        bank: the medium behind the screen (`GreensBank.homogeneous` for a screen in water).
        delays: extra one-way delay of each element [s], shape (n_elements,).
        amplitudes: one-way amplitude factor of each element, shape (n_elements,).

    Returns:
        a bank with backend "phase_screen"; `static_fmc` is carried over unchanged.
    """
    delays = np.asarray(delays, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    expected = (bank.transducer.n_elements,)
    if delays.shape != expected or amplitudes.shape != expected:
        raise ValueError(
            f"delays {delays.shape} and amplitudes {amplitudes.shape} must have shape {expected}"
        )
    screen = amplitudes[None, :] * np.exp(2j * np.pi * bank.freqs[:, None] * delays[None, :])
    return replace(
        bank,
        B=(bank.B * screen[:, :, None, None]).astype(np.complex64),
        backend="phase_screen",
    )


def fit_phase_screen(bank: GreensBank, region: Grid2D | None = None) -> PhaseScreen:
    """Reduce a bank to one delay and one amplitude per element.

    Delay. For `B = a exp(+i 2 pi f tau)` the product of neighbouring frequency
    samples, `B[f + 1] conj(B[f])`, has the phase `2 pi df tau` whatever the
    carrier phase `2 pi f tau`, which wraps many times. The delay of an
    (element, point) pair is the phase of the sum of these products over
    frequency (each weighted by its own magnitude) divided by `2 pi df`; it is
    unambiguous for `|tau| < 1 / (2 df)`, half the unambiguous time window of
    the frequency grid (`fus.window`). The element's delay is the mean over the
    points of the region.

    Amplitude. The mean of `|B|` over the points of the region and over
    frequency, the frequencies weighted by the two-way pulse spectrum
    (`fus.pulse.pulse_spectrum`), so that the band edges count as little as
    they do in an image.

    A bank made by `apply_phase_screen` on the homogeneous bank is recovered
    exactly. For a skull the screen is the isoplanatic approximation: what
    remains of `B` after the fit is its variation over the region.

    Args:
        bank: the medium; needs at least two frequencies.
        region: grid of points over which to average, inside `bank.roi`; None =
            every node of `bank.roi`.

    Returns:
        the fitted `PhaseScreen` (delays [s] and amplitudes, float64, shape (n_elements,)).
    """
    n_freqs = bank.freqs.shape[0]
    if n_freqs < 2:
        raise ValueError("fitting a delay needs at least two frequencies")
    if region is None:
        values = bank.B.reshape(*bank.B.shape[:2], -1)
    else:
        values = bank.interp(region.points().reshape(-1, 2))
    weights = pulse_spectrum(bank.transducer, bank.freqs)
    weights = weights / weights.sum()
    # Accumulated one frequency at a time, so that no copy of the whole bank is made.
    lag_products = np.zeros(values.shape[1:], dtype=np.complex128)  # (n_elements, n_points)
    amplitudes = np.zeros(values.shape[1])
    for f in range(n_freqs):
        amplitudes += weights[f] * np.abs(values[f]).mean(axis=1, dtype=np.float64)
        if f + 1 < n_freqs:
            lag_products += values[f + 1].astype(np.complex128) * np.conj(values[f])
    frequency_step = (bank.freqs[-1] - bank.freqs[0]) / (n_freqs - 1)
    delays = np.angle(lag_products).mean(axis=1) / (2.0 * np.pi * frequency_step)
    return PhaseScreen(delays=delays, amplitudes=amplitudes)
