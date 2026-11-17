"""Unit conversions. Every conversion in the project goes through this module.

All functions are vectorised: they accept scalars or arrays and broadcast
their arguments. Python scalars in give a Python scalar out, and a Python
scalar argument does not change the dtype of an array argument (float32
fields stay float32).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

# Amplitude ratio: 1 Np = 20 / ln(10) dB = 8.686 dB (definition of the units).
_DB_PER_NP = 20.0 / math.log(10.0)
_CM_PER_M = 100.0
_HZ_PER_MHZ = 1.0e6
_HU_PER_UNIT_CONTRAST = 1000.0
# Binary prefix: 1 GiB = 2**30 bytes (IEC 80000-13).
_BYTES_PER_GIB = 2**30

# What a conversion returns: a Python or numpy scalar, or an array.
Value = float | np.ndarray


def _operand(x: ArrayLike) -> Value:
    """Return ``x`` ready for arithmetic: Python scalars as they are, anything else as an array.

    A Python scalar adopts the dtype of the array it meets; wrapped in
    ``np.asarray`` it would be float64 and promote a float32 array.
    """
    return x if isinstance(x, (int, float)) else np.asarray(x)


def db_to_np(x: ArrayLike) -> Value:
    """Convert an amplitude level from decibels to nepers.

    x: amplitude level or attenuation [dB] -> [Np].
    """
    return _operand(x) / _DB_PER_NP


def np_to_db(x: ArrayLike) -> Value:
    """Convert an amplitude level from nepers to decibels.

    x: amplitude level or attenuation [Np] -> [dB].
    """
    return _operand(x) * _DB_PER_NP


def db_cm_mhz_to_np_m(alpha0_db_cm_mhz: ArrayLike, f_hz: ArrayLike, y: ArrayLike) -> Value:
    """Evaluate a power-law attenuation ``alpha0 * (f / 1 MHz)**y`` in Np/m.

    alpha0_db_cm_mhz: power-law prefactor [dB/(cm MHz^y)].
    f_hz: frequency [Hz].
    y: power-law exponent [dimensionless].
    Returns the amplitude attenuation coefficient at ``f_hz`` [Np/m].
    """
    alpha_db_cm = _operand(alpha0_db_cm_mhz) * (_operand(f_hz) / _HZ_PER_MHZ) ** _operand(y)
    return db_to_np(alpha_db_cm) * _CM_PER_M


def np_m_to_db_cm(alpha_np_m: ArrayLike) -> Value:
    """Convert an amplitude attenuation coefficient from Np/m to dB/cm.

    alpha_np_m: attenuation coefficient [Np/m] -> [dB/cm].
    """
    return np_to_db(alpha_np_m) / _CM_PER_M


def hu_from_mu(mu: ArrayLike, mu_water: ArrayLike, mu_air: ArrayLike) -> Value:
    """Convert linear attenuation to Hounsfield units.

    ``HU = 1000 * (mu - mu_water) / (mu_water - mu_air)``, so water is 0 HU and
    air is -1000 HU.

    mu: linear attenuation coefficient [1/m] (any unit shared by all three arguments).
    mu_water: linear attenuation coefficient of water, same unit as ``mu``.
    mu_air: linear attenuation coefficient of air, same unit as ``mu``.
    Returns CT number [HU].
    """
    mu_water = _operand(mu_water)
    return _HU_PER_UNIT_CONTRAST * (_operand(mu) - mu_water) / (mu_water - _operand(mu_air))


def mu_from_hu(hu: ArrayLike, mu_water: ArrayLike, mu_air: ArrayLike) -> Value:
    """Convert Hounsfield units to linear attenuation (inverse of `hu_from_mu`).

    hu: CT number [HU].
    mu_water: linear attenuation coefficient of water [1/m] (or any other unit).
    mu_air: linear attenuation coefficient of air, same unit as ``mu_water``.
    Returns linear attenuation coefficient, same unit as ``mu_water``.
    """
    mu_water = _operand(mu_water)
    return mu_water + _operand(hu) * (mu_water - _operand(mu_air)) / _HU_PER_UNIT_CONTRAST


def bytes_to_gib(n_bytes: ArrayLike) -> Value:
    """Convert a memory or file size from bytes to gibibytes (1 GiB = 2**30 bytes).

    n_bytes: size [bytes] -> [GiB].
    """
    return _operand(n_bytes) / _BYTES_PER_GIB


def wavelength(c: ArrayLike, f: ArrayLike) -> Value:
    """Return the wavelength of a wave.

    c: phase speed [m/s].
    f: frequency [Hz].
    Returns wavelength [m].
    """
    return _operand(c) / _operand(f)
