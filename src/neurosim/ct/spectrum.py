"""Polyenergetic X-ray source spectrum from SpekPy.

SpekPy (Poludniowski et al. 2021, Med. Phys. 48(7):3630-3641) models the
bremsstrahlung and characteristic emission of a tungsten reflection target and
its attenuation by the tube filtration.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import spekpy

from neurosim.ct.config import CtConfig

# SpekPy works in centimetres; the rest of the code is SI. Used only at this boundary.
_CM_PER_M = 100.0


def xray_spectrum(cfg: CtConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the source spectrum of a scan.

    The fluence is the number of photons per square metre and per mAs of tube
    output that cross a surface normal to the central ray at the detector
    (distance ``cfg.geometry.source_detector_distance`` from the focus) with
    nothing but the tube filtration in the beam, split into energy bins. The
    same value is used for every detector element (no heel effect, no bow-tie
    filter) and nothing outside the scanned grid attenuates the beam.

    Expected photons in one element in one view of an air scan:
    ``fluence * (cfg.mas / n_views) * geometry.detector_area``.

    With ``cfg.monoenergetic_kev`` set, all photons are put into one line at
    that energy (same total fluence).

    Args:
        cfg: scan settings (kVp, anode angle, filtration, energy bin width, geometry).

    Returns:
        energies_kev: bin-centre photon energies [keV], shape (n_energies,).
        fluence: photons per bin [1/(m^2 mAs)], shape (n_energies,); bins with
            zero fluence are dropped.
    """
    energies_kev, fluence = _spekpy_spectrum(
        cfg.kvp,
        cfg.anode_angle_deg,
        cfg.filtration_al_mm,
        cfg.energy_bin_kev,
        cfg.geometry.source_detector_distance,
    )
    if cfg.monoenergetic_kev is not None:
        return np.array([cfg.monoenergetic_kev]), np.array([fluence.sum()])
    return energies_kev.copy(), fluence.copy()


@lru_cache(maxsize=32)
def _spekpy_spectrum(
    kvp: float,
    anode_angle_deg: float,
    filtration_al_mm: float,
    energy_bin_kev: float,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SpekPy spectrum on the central axis at `distance` [m], per mAs and per m^2."""
    model = spekpy.Spek(
        kvp=kvp, th=anode_angle_deg, dk=energy_bin_kev, z=distance * _CM_PER_M, mas=1.0
    )
    model.filter("Al", filtration_al_mm)
    # diff=False: photons per bin, not per keV [photons / cm^2 at 1 mAs].
    energies_kev, fluence_per_cm2 = model.get_spectrum(edges=False, flu=True, diff=False)
    energies_kev = np.asarray(energies_kev, dtype=np.float64)
    fluence = np.asarray(fluence_per_cm2, dtype=np.float64) * _CM_PER_M**2
    emitted = fluence > 0.0
    return energies_kev[emitted], fluence[emitted]
