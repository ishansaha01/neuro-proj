"""Analytic backend: the bank of the reference medium itself, B = 1."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

# Largest relative deviation of the sound speed from C_REF, or relative spread of the
# density, that still counts as "the reference medium": the resolution of float32 maps.
_HOMOGENEITY_TOLERANCE = 1e-6


def analytic_bank(
    medium: PlaneMedium, transducer: Transducer, roi: Grid2D, freqs: np.ndarray
) -> GreensBank:
    """Bank of a homogeneous lossless medium at the reference sound speed: B = 1.

    Valid only when `medium.sound_speed` equals `C_REF` everywhere, the
    attenuation is zero and the density is uniform. The value of the uniform
    density does not matter: a homogeneous density drops out of the pressure
    field of a given source (see the package docstring for the normalisation).

    Args:
        medium: the plane medium; sound_speed [m/s], density [kg/m^3] and
            attenuation [dB/(cm MHz^y)] maps of shape (nz, nx).
        transducer: the array.
        roi: grid the bank is stored on.
        freqs: frequencies [Hz], shape (nf,).

    Returns:
        `GreensBank.homogeneous(...)`, B of shape (nf, n_elements, nz_roi, nx_roi).

    Raises:
        ValueError: if the medium is not the reference medium.
    """
    speed_error = np.max(np.abs(medium.sound_speed / C_REF - 1.0))
    if speed_error > _HOMOGENEITY_TOLERANCE:
        raise ValueError(
            f"the analytic backend needs sound_speed == C_REF = {C_REF} m/s everywhere; "
            f"largest relative deviation is {speed_error:.3g}"
        )
    if np.any(medium.attenuation != 0.0):
        raise ValueError("the analytic backend needs a lossless medium (attenuation == 0)")
    density_spread = np.ptp(medium.density) / np.mean(medium.density)
    if density_spread > _HOMOGENEITY_TOLERANCE:
        raise ValueError(
            f"the analytic backend needs a uniform density; relative spread is {density_spread:.3g}"
        )
    return GreensBank.homogeneous(transducer, roi, freqs, C_REF)
