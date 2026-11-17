"""Scanner calibration: water beam-hardening correction and the HU scale.

Both use a centred water cylinder in air scanned at the settings of the scan
itself. Its line integrals are analytic chords, so the calibration does not
depend on the voxel projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.units import hu_from_mu
from neurosim.ct.attenuation import basis_attenuation
from neurosim.ct.config import CtConfig
from neurosim.ct.detector import detector_signal, flat_field_counts, log_projection
from neurosim.ct.recon import backproject, filter_sinogram

_AIR = BASIS_MATERIALS.index("air")
_WATER = BASIS_MATERIALS.index("water")

# Water thicknesses tabulated for the beam-hardening correction: 0 to 1 m (beyond
# any path through a head) in 1 mm steps. Numerical choice.
_TABLE_MAX_THICKNESS = 1.0
_TABLE_STEP = 1.0e-3
# Calibration regions of interest. Water: the central disc out to this fraction of
# the phantom radius. Air: the middle half of the gap between the phantom and the
# edge of the field of view. Numerical choice.
_WATER_ROI_FRACTION = 0.3
_AIR_ROI_FRACTIONS = (0.25, 0.75)
_ROI_RADII = 8
_ROI_ANGLES = 12


@dataclass(frozen=True)
class WaterCorrection:
    """Lookup table that linearises polyenergetic projections of water.

    Attributes:
        thickness: water thicknesses [m], increasing, shape (n,).
        projection: polyenergetic projection ``-ln(I / I0)`` of each thickness
            [dimensionless], increasing, shape (n,).
        mu_reference: attenuation coefficient the corrected projections are
            expressed in [1/m]: ``corrected = mu_reference * thickness``.
    """

    thickness: np.ndarray
    projection: np.ndarray
    mu_reference: float


@dataclass(frozen=True)
class HuCalibration:
    """Reconstructed attenuation of water and air that defines the HU scale.

    Attributes:
        mu_water: mean reconstructed mu in the water region of the calibration phantom [1/m].
        mu_air: mean reconstructed mu in the air around it [1/m].
    """

    mu_water: float
    mu_air: float


def _water_projection(cfg: CtConfig, thickness: np.ndarray) -> np.ndarray:
    """Noise-free polyenergetic projection of water slabs of the given thicknesses [m]."""
    energies_kev, counts = flat_field_counts(cfg)
    mu_water = basis_attenuation(energies_kev)[_WATER : _WATER + 1]
    signal = detector_signal(thickness[np.newaxis], mu_water, counts, energies_kev)
    return -np.log(signal / (counts @ energies_kev))


@lru_cache(maxsize=32)
def water_correction_table(cfg: CtConfig) -> WaterCorrection:
    """Build the water beam-hardening correction of a scan configuration.

    The reference coefficient is the mean attenuation over the diameter of the
    calibration phantom, ``p(d) / d``, so a ray through the phantom centre is
    left unchanged.
    """
    thickness = np.arange(0.0, _TABLE_MAX_THICKNESS + 0.5 * _TABLE_STEP, _TABLE_STEP)
    projection = _water_projection(cfg, thickness)
    diameter = np.array([cfg.calibration_phantom_diameter])
    mu_reference = float(_water_projection(cfg, diameter)[0] / diameter[0])
    return WaterCorrection(thickness, projection, mu_reference)


def apply_water_correction(projection: np.ndarray, table: WaterCorrection) -> np.ndarray:
    """First-order beam-hardening correction: map projections to equivalent water.

    Each projection value is replaced by ``mu_reference`` times the water
    thickness that would produce it, which makes the projections of a water
    object linear in thickness (the water correction clinical scanners apply
    before reconstruction; Kak & Slaney 1988, ch. 4, describe the polychromaticity
    artefact it removes). Values beyond the table follow its last slope.

    Args:
        projection: polyenergetic projections [dimensionless], any shape.
        table: correction from `water_correction_table`.

    Returns:
        Corrected projections [dimensionless], shape of ``projection``.
    """
    thickness = np.interp(projection, table.projection, table.thickness)
    last_slope = (table.thickness[-1] - table.thickness[-2]) / (
        table.projection[-1] - table.projection[-2]
    )
    beyond = np.maximum(projection - table.projection[-1], 0.0)
    return table.mu_reference * (thickness + last_slope * beyond)


def process_signal(signal: np.ndarray, cfg: CtConfig) -> np.ndarray:
    """Turn detector signals into the projections handed to the reconstruction.

    Applies the log transform with the configured signal floor and, if enabled,
    the water beam-hardening correction.

    Args:
        signal: detector signal [keV], any shape.
        cfg: scan settings.

    Returns:
        Projections [dimensionless], shape of ``signal``.
    """
    energies_kev, counts = flat_field_counts(cfg)
    flat_signal = float(counts @ energies_kev)
    floor = cfg.log_floor_photons * flat_signal / float(counts.sum())
    projection = log_projection(signal, flat_signal, floor)
    if cfg.water_correction:
        projection = apply_water_correction(projection, water_correction_table(cfg))
    return projection


def _chord(radius: float, impact_parameter: np.ndarray) -> np.ndarray:
    """Chord length of a line through a disc [m]; zero if the line misses it."""
    return 2.0 * np.sqrt(np.maximum(radius**2 - impact_parameter**2, 0.0))


def _ring_points(inner_radius: float, outer_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over an annulus (a disc if ``inner_radius`` is 0), as (x, y) [m]."""
    radius = inner_radius + (np.arange(_ROI_RADII) + 0.5) / _ROI_RADII * (
        outer_radius - inner_radius
    )
    angle = 2.0 * np.pi * (np.arange(_ROI_ANGLES) + 0.5) / _ROI_ANGLES
    return np.outer(radius, np.cos(angle)), np.outer(radius, np.sin(angle))


@lru_cache(maxsize=32)
def hu_calibration(cfg: CtConfig) -> HuCalibration:
    """Scan and reconstruct the calibration phantom, noise-free, at the scan settings.

    The phantom is a water cylinder of ``cfg.calibration_phantom_diameter``
    centred on the isocentre, in air that fills the scan field of view. The
    projections go through the same processing and the same reconstruction
    kernel as a scan. The result is cached per configuration.

    Returns:
        The mean reconstructed attenuation of water and of air [1/m].
    """
    geom = cfg.geometry
    energies_kev, counts = flat_field_counts(cfg)
    attenuation = basis_attenuation(energies_kev)[[_AIR, _WATER]]
    impact_parameter = geom.source_isocenter_distance * np.sin(geom.fan_angles())
    water_radius = 0.5 * cfg.calibration_phantom_diameter
    water_chord = _chord(water_radius, impact_parameter)
    air_chord = _chord(geom.fov_radius, impact_parameter) - water_chord
    signal = detector_signal(np.stack((air_chord, water_chord)), attenuation, counts, energies_kev)
    # The phantom is rotationally symmetric: every view measures the same row.
    sinogram = np.tile(process_signal(signal, cfg), (geom.n_views, 1))
    filtered = filter_sinogram(sinogram, geom, cfg.kernel)

    gap = geom.fov_radius - water_radius
    water_x, water_y = _ring_points(0.0, _WATER_ROI_FRACTION * water_radius)
    air_x, air_y = _ring_points(
        water_radius + _AIR_ROI_FRACTIONS[0] * gap, water_radius + _AIR_ROI_FRACTIONS[1] * gap
    )
    mu_water = backproject(filtered, geom, water_x, water_y).mean()
    mu_air = backproject(filtered, geom, air_x, air_y).mean()
    return HuCalibration(mu_water=float(mu_water), mu_air=float(mu_air))


@lru_cache(maxsize=32)
def _effective_hu(cfg: CtConfig) -> tuple[float, ...]:
    energies_kev, counts = flat_field_counts(cfg)
    attenuation = basis_attenuation(energies_kev)
    transmitted = (
        counts * energies_kev * np.exp(-attenuation[_WATER] * cfg.calibration_phantom_diameter)
    )
    mu_effective = attenuation @ transmitted / transmitted.sum()
    return tuple(hu_from_mu(mu_effective, mu_effective[_WATER], mu_effective[_AIR]).tolist())


def effective_hu(cfg: CtConfig) -> np.ndarray:
    """Expected HU of a small region of each pure basis material inside a head.

    Small-signal model: a thin insert of material ``m`` at the centre of the
    calibration phantom changes the projection by ``mu_eff_m`` per unit length,
    where ``mu_eff_m`` is ``mu_m(E)`` averaged over the detector-signal spectrum
    transmitted through the phantom diameter (and rescaled by the water
    correction if it is on, which cancels in the HU ratio). This predicts the
    HU a large scan would give without running one; it ignores the object's own
    beam hardening and the reconstruction's blur. Cached per configuration.

    Returns:
        HU of each basis material, shape (4,), ordered as `BASIS_MATERIALS`.
    """
    return np.array(_effective_hu(cfg))
