"""Typed configuration of the CT scanner model (docs/ARCHITECTURE.md, section 6).

Every default that is a physical or modelling choice names its source in the
field doc, or is marked ``assumed, to verify``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from neurosim.common.config import field
from neurosim.common.constants import (
    ALPHA_POWER_DEFAULT,
    AUBRY_2003_ABSORPTION_EXPONENT,
    AUBRY_2003_C_MAX,
    AUBRY_2003_C_MIN,
    C_REF,
)

Kernel = Literal["ramp", "shepp-logan", "hann"]

# Numerical defaults shared by `CtConfig` and the keyword defaults of the projector
# and the reconstruction.
DEFAULT_RAY_STEP_FRACTION = 0.5
DEFAULT_MAX_CHUNK_SAMPLES = 4_000_000


@dataclass(frozen=True)
class FanBeamGeometry:
    """Third-generation fan-beam geometry with an equiangular (curved) detector.

    The source moves on a circle of radius `source_isocenter_distance` about
    the isocentre; view ``i`` has the source at angle ``beta_i = 2 pi i /
    n_views`` from the head-frame +x axis. The detector is an arc centred on
    the source, so its elements are equally spaced in the fan angle ``gamma``
    (Kak & Slaney 1988, section 3.4.1, "equiangular rays"). The fan is
    symmetric about the central ray (no quarter-detector offset).
    """

    source_isocenter_distance: float = field(
        0.595,
        gt=0.0,
        doc="Focus to isocentre distance [m]. 595 mm as quoted for Siemens Somatom "
        "Definition-class scanners; assumed, to verify.",
    )
    source_detector_distance: float = field(
        1.0856,
        gt=0.0,
        doc="Focus to detector-arc distance [m]. 1085.6 mm, same scanner class; "
        "assumed, to verify.",
    )
    n_views: int = field(
        720,
        ge=4,
        doc="Projections per 360 degree rotation. Clinical scanners take about "
        "1000; 720 is a cost compromise; assumed, to verify.",
    )
    n_detectors: int = field(
        736,
        ge=8,
        doc="Detector elements along the arc. 736 channels, same scanner class; "
        "assumed, to verify.",
    )
    detector_pitch: float = field(
        1.2858e-3,
        gt=0.0,
        doc="Arc length between neighbouring element centres on the detector [m]. "
        "Chosen so 736 elements span a 49.9 degree fan (50 cm field of view); "
        "assumed, to verify.",
    )
    detector_height: float = field(
        1.1e-3,
        gt=0.0,
        doc="Element extent along the rotation axis [m]; with the pitch it sets the "
        "photon-collecting area. 0.6 mm at the isocentre projected to the "
        "detector; assumed, to verify.",
    )

    def __post_init__(self) -> None:
        if self.source_detector_distance <= self.source_isocenter_distance:
            raise ValueError(
                "source_detector_distance must exceed source_isocenter_distance, got "
                f"{self.source_detector_distance} <= {self.source_isocenter_distance}"
            )
        if self.half_fan_angle >= 0.5 * math.pi:
            raise ValueError(
                f"the fan half-angle must be below 90 degrees, got {self.half_fan_angle} rad"
            )

    @property
    def angular_pitch(self) -> float:
        """Fan angle between neighbouring detector elements [rad]."""
        return self.detector_pitch / self.source_detector_distance

    @property
    def half_fan_angle(self) -> float:
        """Fan angle from the central ray to the outer edge of the last element [rad]."""
        return 0.5 * self.n_detectors * self.angular_pitch

    @property
    def fov_radius(self) -> float:
        """Radius of the scan field of view, the disc seen by every view [m]."""
        return self.source_isocenter_distance * math.sin(self.half_fan_angle)

    @property
    def detector_area(self) -> float:
        """Photon-collecting area of one detector element [m^2]."""
        return self.detector_pitch * self.detector_height

    def fan_angles(self) -> np.ndarray:
        """Fan angle of every detector element centre [rad], shape (n_detectors,)."""
        return (np.arange(self.n_detectors) - 0.5 * (self.n_detectors - 1)) * self.angular_pitch

    def view_angles(self) -> np.ndarray:
        """Source angle of every view [rad], shape (n_views,)."""
        return 2.0 * math.pi * np.arange(self.n_views) / self.n_views


@dataclass(frozen=True)
class HuMappingConfig:
    """Constants of the clinical HU -> acoustics map, and the ranges they are drawn from.

    The nominal values are those of Aubry et al. (2003), J. Acoust. Soc. Am.
    113(1):84-93, Eqs. 3-6. Each ``*_range`` is the ``(low, high)`` interval
    `neurosim.ct.acoustics.sample_hu_mapping` draws uniformly from.
    """

    hu_bone: Optional[float] = field(
        None,
        gt=0.0,
        doc="HU of fully dense cortical bone [HU]. None: computed for the scan "
        "settings (ct.calibration.effective_hu). Aubry 2003 Eq. 1 normalises its "
        "scale so that this value is 1000.",
    )
    hu_bone_relative_range: tuple[float, float] = field(
        (0.85, 1.15),
        gt=0.0,
        doc="Range of the factor applied to hu_bone [-]. Webb et al. 2018 (IEEE TUFFC "
        "65:1111), Tables I-II: at 120 kVp the velocity-vs-HU slope changes by a "
        "factor 0.72 to 1.31 between soft-tissue and bone kernels; +-15 % on hu_bone "
        "is our split of that spread between hu_bone and c_bone; assumed, to verify.",
    )
    c_water: float = field(
        AUBRY_2003_C_MIN,
        gt=0.0,
        doc="Sound speed at porosity 1 [m/s]. Aubry 2003 Eq. 5, c_min.",
    )
    c_water_range: tuple[float, float] = field(
        (1480.0, C_REF),
        gt=0.0,
        doc="[m/s]. From water at 20 degC (1482 m/s) to the soft-tissue convention "
        "(1540 m/s); assumed, to verify.",
    )
    c_bone: float = field(
        AUBRY_2003_C_MAX,
        gt=0.0,
        doc="Sound speed at porosity 0 [m/s]. Aubry 2003 Eq. 5, c_max.",
    )
    c_bone_range: tuple[float, float] = field(
        (2700.0, 3100.0),
        gt=0.0,
        doc="[m/s]. +-200 m/s: Webb 2018 report a mean 95 % confidence interval of "
        "+-198 m/s on velocity predicted from HU; the upper end is the 3100 m/s used "
        "by later CT-based models (Marquet 2009 as quoted by secondary sources; "
        "assumed, to verify) and close to Webb's largest measured velocity, 3114 m/s.",
    )
    rho_water: float = field(
        1000.0, gt=0.0, doc="Density at porosity 1 [kg/m^3]. Aubry 2003 Eq. 4."
    )
    rho_bone: float = field(
        2100.0, gt=0.0, doc="Density at porosity 0 [kg/m^3]. Aubry 2003 Eq. 4."
    )
    rho_bone_range: tuple[float, float] = field(
        (1900.0, 2200.0),
        gt=0.0,
        doc="[kg/m^3]. Lower end: 1900, the inner-table mean of Fry & Barger quoted "
        "by Aubry 2003 and used by later CT-based models; upper end assumed, to verify.",
    )
    absorption_frequency: float = field(
        1.5e6,
        gt=0.0,
        doc="Frequency at which the absorption end points hold [Hz]. Centre frequency "
        "of the array in Aubry 2003 (the paper does not state it for Eq. 6); "
        "assumed, to verify.",
    )
    absorption_min: float = field(
        2.0,
        ge=0.0,
        doc="Absorption of dense bone at absorption_frequency [dB/cm]. Aubry 2003 "
        "Eq. 6: 0.2 dB/mm.",
    )
    absorption_min_range: tuple[float, float] = field(
        (1.0, 3.0), ge=0.0, doc="[dB/cm]. +-50 %; assumed, to verify."
    )
    absorption_max: float = field(
        80.0,
        ge=0.0,
        doc="Absorption at porosity 1 at absorption_frequency [dB/cm]. Aubry 2003 "
        "Eq. 6: 8 dB/mm.",
    )
    absorption_max_range: tuple[float, float] = field(
        (13.0, 80.0),
        ge=0.0,
        doc="[dB/cm]. Upper end Aubry 2003; lower end 8.7 dB/(cm MHz) scaled to "
        "1.5 MHz, the largest skull value of Yaakub et al. 2023 (arXiv:2209.12031, "
        "Table 1); assumed, to verify.",
    )
    absorption_exponent: float = field(
        AUBRY_2003_ABSORPTION_EXPONENT,
        gt=0.0,
        doc="Exponent beta of porosity in the absorption law [-]. Aubry 2003 Eq. 6.",
    )
    absorption_exponent_range: tuple[float, float] = field(
        (0.3, 0.7), gt=0.0, doc="[-]. Range tested by Aubry 2003."
    )
    alpha_power: float = field(
        ALPHA_POWER_DEFAULT,
        gt=0.0,
        doc="Global power-law exponent y of the acoustic model [-], used to express "
        "the absorption as alpha0 [dB/(cm MHz^y)]. Must equal HeadVolume.alpha_power "
        "of the head being imaged; the default is the one of head.materials.MaterialParams "
        "(common.constants.ALPHA_POWER_DEFAULT).",
    )
    bone_threshold_hu: float = field(
        300.0,
        gt=0.0,
        doc="CT number from which a voxel counts as skull [HU]: voxels at or above it get "
        "the bone absorption law and the bone label of the CT-derived medium, voxels below "
        "it are soft tissue or water. 300 HU is the skull threshold of Yaakub et al. 2023 "
        "(arXiv:2209.12031); assumed, to verify.",
    )

    def __post_init__(self) -> None:
        for name in (
            "hu_bone_relative_range",
            "c_water_range",
            "c_bone_range",
            "rho_bone_range",
            "absorption_min_range",
            "absorption_max_range",
            "absorption_exponent_range",
        ):
            low, high = getattr(self, name)
            if high < low:
                raise ValueError(f"{name} must be (low, high) with low <= high, got {(low, high)}")


@dataclass(frozen=True)
class CtConfig:
    """Settings of one simulated CT scan."""

    kvp: float = field(
        120.0,
        ge=30.0,
        le=300.0,
        doc="Tube potential [kV]. 120 kVp is the standard head protocol and the "
        "setting of Aubry 2003; the bounds are SpekPy's range for a tungsten target.",
    )
    anode_angle_deg: float = field(
        12.0,
        gt=0.0,
        le=45.0,
        doc="Tungsten anode angle [degrees]. SpekPy default; CT tubes use 7-12 degrees; "
        "assumed, to verify.",
    )
    filtration_al_mm: float = field(
        6.0,
        ge=0.0,
        doc="Total filtration in aluminium [mm], inherent plus flat filter; no bow-tie "
        "filter is modelled. Typical CT total filtration is 5-7 mm Al equivalent; "
        "assumed, to verify.",
    )
    mas: float = field(
        200.0,
        gt=0.0,
        doc="Tube current-time product of one rotation [mAs], shared equally by the "
        "views. Typical adult head scans use 200-350 mAs; assumed, to verify.",
    )
    energy_bin_kev: float = field(
        2.0, gt=0.0, le=10.0, doc="Width of the spectrum's energy bins [keV]. Modelling choice."
    )
    monoenergetic_kev: Optional[float] = field(
        None,
        gt=0.0,
        doc="If set, replace the spectrum by one line at this energy [keV] carrying "
        "the same photon fluence: an ablation that removes beam hardening.",
    )
    geometry: FanBeamGeometry = field(default_factory=FanBeamGeometry, doc="Fan-beam geometry.")
    kernel: Kernel = field("ramp", doc="Reconstruction kernel (apodisation of the ramp filter).")
    slab_z_from_center: Optional[tuple[float, float]] = field(
        (-0.004, 0.004),
        doc="Axial extent of the scan: a z interval (low, high) measured from the centre "
        "of the z extent of the scanned grid [m]. The slices whose centres lie in it are "
        "scanned. It is relative to the grid, not a head-frame coordinate, so one setting "
        "serves a whole head and a box region placed anywhere. None scans every slice of "
        "the grid. Default: an 8 mm slab around the middle of the volume; modelling choice.",
    )
    photon_noise: bool = field(True, doc="Poisson photon noise on or off (ROADMAP D-005).")
    noise_scale: float = field(
        1.0,
        ge=0.0,
        doc="Multiplier on the fluctuation of the detector signal about its "
        "expectation [-]; 1 is the physical Poisson noise.",
    )
    water_correction: bool = field(
        True,
        doc="First-order (water) beam-hardening correction of the projections, as "
        "clinical scanners apply.",
    )
    calibration_phantom_diameter: float = field(
        0.2,
        gt=0.0,
        doc="Diameter of the water cylinder used for the HU calibration and as the "
        "reference thickness of the water correction [m]. 20 cm head-size water "
        "phantom; assumed, to verify.",
    )
    ray_step_fraction: float = field(
        DEFAULT_RAY_STEP_FRACTION,
        gt=0.0,
        le=1.0,
        doc="Sampling step along a ray as a fraction of the voxel size [-]. Numerical choice.",
    )
    log_floor_photons: float = field(
        1.0,
        gt=0.0,
        doc="Smallest detector signal passed to the logarithm, as a number of photons "
        "of the mean flat-field energy [-]; guards against photon starvation. "
        "Numerical choice.",
    )
    max_chunk_samples: int = field(
        DEFAULT_MAX_CHUNK_SAMPLES,
        ge=1000,
        doc="Largest number of array elements processed at once (ray samples, "
        "view-pixel pairs, count bins); bounds memory at about 60 bytes per element.",
    )
    hu_mapping: HuMappingConfig = field(
        default_factory=HuMappingConfig, doc="Constants of the HU -> acoustics map."
    )

    def __post_init__(self) -> None:
        if self.monoenergetic_kev is not None and self.monoenergetic_kev >= self.kvp:
            raise ValueError(
                f"monoenergetic_kev must be below kvp, got {self.monoenergetic_kev} >= {self.kvp}"
            )
        slab = self.slab_z_from_center
        if slab is not None and slab[1] < slab[0]:
            raise ValueError(f"slab_z_from_center must be (low, high), got {slab}")
        if self.calibration_phantom_diameter >= 2.0 * self.geometry.fov_radius:
            raise ValueError(
                "calibration_phantom_diameter must be smaller than the scan field of view "
                f"({2.0 * self.geometry.fov_radius:.4f} m), got {self.calibration_phantom_diameter}"
            )
