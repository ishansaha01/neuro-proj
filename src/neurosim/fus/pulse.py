"""Two-way pulse spectrum of the transducer and the frequency grid that samples it."""

from __future__ import annotations

import numpy as np

from neurosim.common.units import db_to_np
from neurosim.fus.types import Transducer


def _spectral_sigma(transducer: Transducer) -> float:
    """Standard deviation of the Gaussian two-way amplitude spectrum [Hz].

    `Transducer.fractional_bandwidth` is the full width at half amplitude (-6 dB)
    as a fraction of the centre frequency; a Gaussian `exp(-(f - f0)^2 / (2 s^2))`
    has full width at half maximum `2 sqrt(2 ln 2) s`.
    """
    full_width = transducer.fractional_bandwidth * transducer.center_frequency
    return full_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _half_width_in_sigmas(level_db: float) -> float:
    """Half width of a Gaussian at `level_db` below its peak [standard deviations]."""
    return float(np.sqrt(2.0 * db_to_np(level_db)))


def pulse_spectrum(transducer: Transducer, freqs: np.ndarray) -> np.ndarray:
    """Two-way (transmit times receive) pulse spectrum `P(f)`.

    A Gaussian centred on the centre frequency with the -6 dB width given by
    `transducer.fractional_bandwidth`, real and positive (a zero-phase pulse
    centred on t = 0), peak value 1.

    Args:
        transducer: the array.
        freqs: frequencies [Hz], shape (nf,).

    Returns:
        float64 array, shape (nf,) [dimensionless].
    """
    deviation = np.asarray(freqs, dtype=np.float64) - transducer.center_frequency
    return np.exp(-0.5 * (deviation / _spectral_sigma(transducer)) ** 2)


def pulse_duration(transducer: Transducer, level_db: float) -> float:
    """Length of the two-way pulse envelope between the points `level_db` below its peak [s].

    The envelope of a Gaussian spectrum with standard deviation `s` [Hz] is a
    Gaussian in time with standard deviation `1 / (2 pi s)`.
    """
    sigma_t = 1.0 / (2.0 * np.pi * _spectral_sigma(transducer))
    return 2.0 * _half_width_in_sigmas(level_db) * sigma_t


def band_frequencies(
    transducer: Transducer, n_frequencies: int, truncation_db: float
) -> np.ndarray:
    """Uniform frequency grid over the band where the pulse spectrum is significant.

    The band is centred on the centre frequency and ends where the two-way
    amplitude spectrum is `truncation_db` below its peak.

    Args:
        transducer: the array.
        n_frequencies: number of bins (>= 2).
        truncation_db: level of the band edges below the spectral peak [dB], > 0.

    Returns:
        float64 array, shape (n_frequencies,) [Hz], ascending and uniformly spaced.

    Raises:
        ValueError: if the band would reach zero frequency.
    """
    half_band = _half_width_in_sigmas(truncation_db) * _spectral_sigma(transducer)
    lowest = transducer.center_frequency - half_band
    if lowest <= 0.0:
        raise ValueError(
            f"band edge {lowest:.3g} Hz is not positive: reduce fractional_bandwidth "
            f"({transducer.fractional_bandwidth}) or truncation_db ({truncation_db})"
        )
    return np.linspace(lowest, transducer.center_frequency + half_band, n_frequencies)
