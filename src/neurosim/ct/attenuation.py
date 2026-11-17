"""X-ray attenuation of the CT basis materials.

Elemental mass attenuation coefficients come from ``xraydb.mu_elam``, i.e. the
tabulation of Elam, Ravel & Sieber (2002), Radiat. Phys. Chem. 63:121-128
(total cross-section: photoelectric + coherent + incoherent). They are combined
by the mass fractions of the standard compositions in NIST X-Ray Mass
Attenuation Coefficients (Hubbell & Seltzer), Table 2,
https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html. Eval E-CT-002
checks the result against the NIST tables of the same compounds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xraydb
from numpy.typing import ArrayLike

from neurosim.common.constants import BASIS_MATERIALS, RHO_AIR, RHO_WATER

# Composition by weight, atomic number -> mass fraction. NIST Table 2 (URL above),
# rows "Air, Dry (near sea level)", "Water, Liquid", "Tissue, Soft (ICRU-44)" and
# "Bone, Cortical (ICRU-44)".
MASS_FRACTIONS: Mapping[str, Mapping[int, float]] = {
    "air": {6: 0.000124, 7: 0.755268, 8: 0.231781, 18: 0.012827},
    "water": {1: 0.111898, 8: 0.888102},
    "soft_tissue": {
        1: 0.102,
        6: 0.143,
        7: 0.034,
        8: 0.708,
        11: 0.002,
        15: 0.003,
        16: 0.003,
        17: 0.002,
        19: 0.003,
    },
    "cortical_bone": {
        1: 0.034,
        6: 0.155,
        7: 0.042,
        8: 0.435,
        11: 0.001,
        12: 0.002,
        15: 0.103,
        16: 0.003,
        20: 0.225,
    },
}

# Mass density of each pure basis material [kg/m^3]. Air and water: the project's
# reference state (common.constants). Soft tissue 1.060 g/cm^3 and cortical bone
# 1.920 g/cm^3: NIST Table 2, same rows as above.
DENSITIES: Mapping[str, float] = {
    "air": RHO_AIR,
    "water": RHO_WATER,
    "soft_tissue": 1060.0,
    "cortical_bone": 1920.0,
}

# xraydb takes energies in eV and returns cm^2/g. Used only at this boundary.
_EV_PER_KEV = 1.0e3
_M2_KG_PER_CM2_G = 0.1


def mass_attenuation(material: str, energies_kev: ArrayLike) -> np.ndarray:
    """Mass attenuation coefficient of one basis material.

    Args:
        material: one of `neurosim.common.constants.BASIS_MATERIALS`.
        energies_kev: photon energies [keV], shape (n_energies,).

    Returns:
        mu/rho [m^2/kg], shape (n_energies,).
    """
    energies_ev = np.atleast_1d(np.asarray(energies_kev, dtype=np.float64)) * _EV_PER_KEV
    total = np.zeros_like(energies_ev)
    for atomic_number, mass_fraction in MASS_FRACTIONS[material].items():
        total += mass_fraction * xraydb.mu_elam(atomic_number, energies_ev, kind="total")
    return total * _M2_KG_PER_CM2_G


def basis_attenuation(energies_kev: ArrayLike) -> np.ndarray:
    """Linear attenuation coefficient of each pure basis material.

    A voxel with volume fractions ``f_m`` has ``mu(E) = sum_m f_m * mu_m(E)``.

    Args:
        energies_kev: photon energies [keV], shape (n_energies,).

    Returns:
        mu [1/m], shape (4, n_energies), rows ordered as `BASIS_MATERIALS`.
    """
    return np.stack(
        [DENSITIES[name] * mass_attenuation(name, energies_kev) for name in BASIS_MATERIALS]
    )
