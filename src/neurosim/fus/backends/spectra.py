"""Drive pulse and time-to-frequency conversion of the full-wave backend.

Spectra follow the project convention exp(-i w t): the spectrum of a real
signal p(t) is `P(f) = sum_n p(t_n) exp(+i 2 pi f t_n) dt`, so that a delay tau
multiplies it by exp(+i 2 pi f tau) (docs/ARCHITECTURE.md, section 1).
"""

from __future__ import annotations

import math

import numpy as np

from neurosim.common.units import np_to_db

# The Gaussian envelope of the drive pulse is cut at this many standard deviations on
# each side, where it has fallen to exp(-8) = -69 dB of its peak.
PULSE_TRUNCATION_SIGMAS = 4.0

# Largest block of recording samples converted in one matrix product (bounds temporaries).
_BLOCK_SAMPLES = 4_000_000


def gaussian_pulse(
    freqs: np.ndarray, dt: float, edge_level_db: float, max_cycles: float
) -> np.ndarray:
    """Zero-mean Gaussian-modulated sine whose spectrum covers the requested frequencies.

    The carrier is the mid-band frequency. The Gaussian amplitude spectrum is
    `edge_level_db` below its peak at the lowest and highest frequency, unless
    that would make the envelope longer than `max_cycles` carrier periods
    (standard deviation), in which case the envelope is shortened to that.

    Args:
        freqs: frequencies to cover [Hz], shape (nf,), ascending.
        dt: sampling period [s].
        edge_level_db: spectrum level at the band edges below the peak [dB], > 0.
        max_cycles: upper bound on the envelope standard deviation [carrier periods].

    Returns:
        pulse samples (dimensionless source amplitude) at t = n dt, shape (n_pulse,).
    """
    carrier = 0.5 * (freqs[0] + freqs[-1])
    half_band = 0.5 * (freqs[-1] - freqs[0])
    # Gaussian spectrum exp(-(f - carrier)^2 / (2 sigma_f^2)): level in Np at half_band.
    sigma_f = half_band / math.sqrt(2.0 * edge_level_db / float(np_to_db(1.0)))
    sigma_t_cap = max_cycles / carrier
    sigma_t = min(1.0 / (2.0 * math.pi * sigma_f), sigma_t_cap) if sigma_f > 0.0 else sigma_t_cap
    # Samples symmetric about the centre sample make the pulse exactly odd, hence zero-mean:
    # a mean value would leave the slowly decaying wake of a 2D line source in the recordings.
    half_samples = math.ceil(PULSE_TRUNCATION_SIGMAS * sigma_t / dt)
    t = np.arange(-half_samples, half_samples + 1) * dt
    return np.exp(-0.5 * (t / sigma_t) ** 2) * np.sin(2.0 * np.pi * carrier * t)


def spectra_at(recordings: np.ndarray, dt: float, freqs: np.ndarray) -> np.ndarray:
    """Spectra of real time series at exactly the requested frequencies (direct DFT sum).

    Args:
        recordings: time series sampled at t = n dt, shape (n_series, n_steps).
        dt: sampling period [s].
        freqs: frequencies [Hz], shape (nf,).

    Returns:
        complex128 spectra, shape (nf, n_series), in units of `recordings` times seconds,
        with the exp(-i w t) convention (module docstring).
    """
    n_series, n_steps = recordings.shape
    phase = 2.0 * np.pi * np.outer(np.arange(n_steps) * dt, freqs)  # (n_steps, nf)
    cosine, sine = np.cos(phase), np.sin(phase)
    spectra = np.empty((freqs.size, n_series), dtype=np.complex128)
    block = max(1, _BLOCK_SAMPLES // n_steps)
    for first in range(0, n_series, block):
        rows = recordings[first : first + block].astype(np.float64, copy=False)
        spectra[:, first : first + block] = dt * ((rows @ cosine) + 1j * (rows @ sine)).T
    return spectra
