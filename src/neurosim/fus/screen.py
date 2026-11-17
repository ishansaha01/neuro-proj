"""Near-field phase screen: per-element delay and amplitude applied to a bank.

A thin aberrating layer on the array face delays and scales what each element
sends and receives, whatever the field point:

    B_screen[f, e, r] = B[f, e, r] * a_e * exp(+i 2 pi f tau_e)

This is the simplest model of skull aberration, and the low-dimensional skull
correction (per-element delays and amplitudes) that `recon` estimates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from neurosim.fus.types import GreensBank


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
