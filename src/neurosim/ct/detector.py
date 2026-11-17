"""Polyenergetic Beer-Lambert detection, photon noise and the log transform.

The detector is energy-integrating, as the scintillator detectors of clinical
scanners are: a photon of energy ``E`` contributes a signal proportional to
``E``. The signal is therefore kept in keV: ``signal = sum_E E * N_E`` with
``N_E`` the photons absorbed in bin ``E``. Every photon that reaches an element
is absorbed (unit quantum efficiency; assumed, to verify) and there is no
electronic noise, scatter or crosstalk.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from neurosim.ct.config import CtConfig
from neurosim.ct.spectrum import xray_spectrum


def flat_field_counts(cfg: CtConfig) -> tuple[np.ndarray, np.ndarray]:
    """Expected photons per detector element per view with nothing in the beam.

    The tube output ``cfg.mas`` is shared equally by the views of one rotation.

    Returns:
        energies_kev: bin-centre energies [keV], shape (n_energies,).
        counts: expected photons per bin [-], shape (n_energies,).
    """
    energies_kev, fluence = xray_spectrum(cfg)
    mas_per_view = cfg.mas / cfg.geometry.n_views
    return energies_kev, fluence * mas_per_view * cfg.geometry.detector_area


def detector_signal(
    path_lengths: np.ndarray,
    attenuation: np.ndarray,
    counts: np.ndarray,
    energies_kev: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Energy-integrated detector signal behind the object.

    ``signal = sum_E E * N_E`` with ``N_E`` either its expectation
    ``counts_E * exp(-sum_m mu_m(E) L_m)`` (``rng`` None) or a Poisson draw with
    that mean, independent per ray and energy bin. Beam hardening needs no extra
    model: it follows from the sum over energies.

    Args:
        path_lengths: line integral of each material's volume fraction [m],
            shape (n_materials, ...).
        attenuation: linear attenuation of the pure materials [1/m],
            shape (n_materials, n_energies).
        counts: flat-field photons per element, view and bin [-], shape (n_energies,).
        energies_kev: bin-centre energies [keV], shape (n_energies,).
        rng: generator for the photon noise; None gives the expected signal.
        noise_scale: multiplier on the fluctuation about the expected signal [-].

    Returns:
        Signal [keV], float64, shape ``path_lengths.shape[1:]``.
    """
    optical_depth = np.tensordot(path_lengths, attenuation, axes=(0, 0))
    expected_counts = counts * np.exp(-optical_depth)
    expected = expected_counts @ energies_kev
    if rng is None:
        return expected
    noisy = rng.poisson(expected_counts) @ energies_kev
    return expected + noise_scale * (noisy - expected)


def log_projection(signal: np.ndarray, flat_signal: float, floor: float) -> np.ndarray:
    """Convert detector signals to polyenergetic projections ``-ln(I / I0)``.

    Args:
        signal: detector signal behind the object [keV], any shape.
        flat_signal: signal with nothing in the beam (air scan) [keV].
        floor: smallest signal passed to the logarithm [keV]; signals at or
            below it (photon starvation, zero counts) are set to it.

    Returns:
        Projection values [dimensionless], shape of ``signal``.
    """
    return -np.log(np.maximum(signal, floor) / flat_signal)
