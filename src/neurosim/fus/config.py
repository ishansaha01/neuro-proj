"""Typed configuration of the fUS imaging chain and the objects derived from it.

Units are SI throughout. The reference sound speed is `common.constants.C_REF`
and is not a config value: it is the convention every `GreensBank` is built on.

Normalisation of scattering strengths (used by `scatterers`, `noise`, `acquire`
and the statistical tier alike):

- Blood has unit backscatter per unit blood area: scatterers drawn for a CBV map
  carry `sum_k |s_k|^2 = sum(cbv * cell_area)`, whatever their number. The
  expected blood power Doppler image is then `H @ (cbv * cell_area)`.
- Tissue backscatter per unit area is `tissue_to_blood_db` above that of blood
  at the volume fraction `cbv_reference`.
- Electronic noise is set against the *nominal blood echo*: the incoherent
  channel power of blood at `cbv_reference` filling the ROI (see `fus.noise`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from neurosim.common.config import field
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.units import wavelength
from neurosim.fus.pulse import band_frequencies
from neurosim.fus.types import PlaneWaveSequence, Transducer
from neurosim.fus.window import check_time_window


@dataclass(frozen=True)
class TransducerConfig:
    """Linear array; pitch and element width are set from the wavelength at the centre frequency."""

    n_elements: int = field(128, ge=2, doc="number of elements (ROADMAP phase 2: 128)")
    center_frequency: float = field(
        2.0e6, gt=0.0, doc="pulse centre frequency [Hz] (ROADMAP phase 2: 2 MHz, sweep 1-3 MHz)"
    )
    fractional_bandwidth: float = field(
        0.6,
        gt=0.0,
        lt=2.0,
        doc="-6 dB two-way bandwidth / centre frequency; 0.6 assumed, to verify (typical probe)",
    )
    pitch_wavelengths: float = field(
        0.5,
        gt=0.0,
        doc="element pitch in wavelengths at the centre frequency and C_REF; 0.5 (no grating "
        "lobes at any steering angle) assumed, to verify against a real 2 MHz probe",
    )
    element_width_fraction: float = field(
        0.9, gt=0.0, le=1.0, doc="element width / pitch (kerf 10 %); assumed, to verify"
    )


@dataclass(frozen=True)
class SequenceConfig:
    """Plane-wave compounding sequence."""

    angles_deg: tuple[float, ...] = field(
        (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0),
        gt=-90.0,
        lt=90.0,
        doc="steering angles [degrees] (ROADMAP phase 2: 7-11 plane waves over +-10 degrees)",
    )
    frame_rate: float = field(
        500.0, gt=0.0, doc="compounded frame rate [Hz]; 500 Hz assumed, to verify"
    )
    n_frames: int = field(
        200,
        ge=1,
        doc="compounded frames per power Doppler image; 200 assumed, to verify",
    )


@dataclass(frozen=True)
class FrequencyGridConfig:
    """Uniform frequency grid on which channel data are synthesised (see `fus.window`)."""

    n_frequencies: int = field(
        128, ge=2, doc="number of frequency bins; sets the unambiguous time window 1/df"
    )
    truncation_db: float = field(
        20.0,
        gt=0.0,
        doc="band edges lie where the two-way pulse spectrum is this far below its peak [dB]; "
        "modelling choice (band truncation gives axial sidelobes at roughly this level below "
        "the edge value)",
    )


@dataclass(frozen=True)
class RoiConfig:
    """Region of the imaging plane that holds every scatterer and pixel.

    Also the coarse grid on which a `GreensBank` stores `B`.
    """

    x_range: tuple[float, float] = field(
        (-0.015, 0.015), doc="lateral coordinates of the first and last ROI nodes [m]; assumed"
    )
    z_range: tuple[float, float] = field(
        (0.020, 0.050),
        gt=0.0,
        doc="depths of the first and last ROI nodes [m]; assumed (cortex below scalp and skull)",
    )
    spacing: float = field(1.0e-3, gt=0.0, doc="ROI node spacing [m]; numerical choice")


@dataclass(frozen=True)
class PixelGridConfig:
    """Beamforming grid, centred on the ROI."""

    shape: tuple[int, int] = field((64, 64), ge=1, doc="pixels (nz, nx)")
    spacing: Optional[float] = field(
        None, gt=0.0, doc="pixel size [m]; None = the largest that keeps the grid inside the ROI"
    )


@dataclass(frozen=True)
class BeamformingConfig:
    """Receive aperture of the delay-and-sum beamformer."""

    f_number: float = field(
        0.0,
        ge=0.0,
        doc="receive f-number: elements within depth / (2 f_number) of the pixel's lateral "
        "position are used; 0 = the whole array",
    )
    window: Literal["boxcar", "hann"] = field(
        "boxcar", doc="receive apodisation over the active aperture"
    )


@dataclass(frozen=True)
class DopplerConfig:
    """SVD clutter filter (Demene et al. 2015, IEEE TMI 34:2271) and power Doppler."""

    cutoff_rule: Literal["fixed", "energy", "frequency"] = field(
        "fixed", doc="how the number of discarded singular components is chosen (fus.doppler)"
    )
    n_discard: int = field(
        30,
        ge=0,
        doc="'fixed' rule: number of largest singular components removed; 30 of 200 assumed, "
        "to verify",
    )
    energy_fraction: float = field(
        0.99,
        gt=0.0,
        lt=1.0,
        doc="'energy' rule: remove the fewest leading components holding at least this "
        "fraction of the total energy; assumed, to verify",
    )
    cutoff_frequency: float = field(
        10.0,
        ge=0.0,
        doc="'frequency' rule: remove leading components whose temporal singular vector has "
        "a mean Doppler frequency below this [Hz]; assumed, to verify",
    )
    effective_frames: Optional[int] = field(
        None,
        ge=1,
        doc="statistical tier: number of independent speckle realisations averaged in one "
        "power Doppler image; None = sequence.n_frames (fully decorrelated blood: an upper bound)",
    )


@dataclass(frozen=True)
class NoiseConfig:
    """Clutter and noise sources; each one can be switched off and scaled (ROADMAP D-005)."""

    electronic_noise: bool = field(True, doc="add electronic noise to the channel spectra")
    electronic_snr_db: float = field(
        -10.0,
        doc="nominal blood echo power / noise power per channel sample [dB] (fus.noise); "
        "assumed, to verify",
    )
    tissue_clutter: bool = field(True, doc="include the tissue scatterers")
    tissue_to_blood_db: float = field(
        40.0,
        doc="tissue / blood backscatter power per unit area at cbv_reference [dB]. 40 dB "
        "assumed, to verify: clutter-to-blood ratios span tens of dB and 'may in some cases "
        "exceed 100 dB' (Torp 1997, IEEE TUFFC 44(2):417-424); no brain-specific value found",
    )
    tissue_motion: bool = field(True, doc="rigid micro-motion of everything in the ROI")
    tissue_motion_rms: float = field(
        20.0e-6,
        ge=0.0,
        doc="rms displacement magnitude [m]. Brain pulse displacement of 58-109 um peak to "
        "peak (hippocampus means, Jurkonis et al. 2020, Diagnostics 10(2):57) is 20-39 um rms "
        "for a sinusoid",
    )
    tissue_motion_frequencies: tuple[float, ...] = field(
        (0.25, 1.2),
        gt=0.0,
        doc="frequencies of the motion components [Hz]; breathing and heartbeat, assumed",
    )
    static_clutter: bool = field(
        True, doc="add the static echoes of the medium (bank.static_fmc) when the bank has them"
    )
    static_clutter_scale: float = field(
        1.0, ge=0.0, doc="amplitude factor on the static echoes (1 = as simulated)"
    )
    cbv_reference: float = field(
        0.03,
        gt=0.0,
        le=1.0,
        doc="blood volume fraction that the tissue and noise levels are referred to; "
        "0.03 assumed, to verify (grey-matter CBV is a few percent)",
    )


@dataclass(frozen=True)
class ScattererConfig:
    """Discretisation of blood and tissue into point scatterers (numerical choices)."""

    n_blood: int = field(
        8000, ge=1, doc="blood scatterers; carries CBV through their density, not their strength"
    )
    n_tissue: int = field(4000, ge=1, doc="tissue scatterers")
    blood_transit_length: float = field(
        1.0e-3,
        gt=0.0,
        doc="length of the straight path a blood scatterer travels before it re-enters at the "
        "path start [m]; about one resolution cell at 2 MHz, assumed, to verify",
    )


@dataclass(frozen=True)
class ImagingConfig:
    """Everything the imaging chain needs besides the medium (a `GreensBank`) and the CBV map."""

    transducer: TransducerConfig = field(default_factory=TransducerConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    frequencies: FrequencyGridConfig = field(default_factory=FrequencyGridConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    pixels: PixelGridConfig = field(default_factory=PixelGridConfig)
    beamforming: BeamformingConfig = field(default_factory=BeamformingConfig)
    doppler: DopplerConfig = field(default_factory=DopplerConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    scatterers: ScattererConfig = field(default_factory=ScattererConfig)
    fidelity: Literal["scatterer", "statistical"] = field(
        "scatterer", doc="explicit moving scatterers, or H @ cbv with gamma speckle"
    )

    def __post_init__(self) -> None:
        for name, (first, last) in (("x_range", self.roi.x_range), ("z_range", self.roi.z_range)):
            if not last > first:
                raise ValueError(f"roi.{name} must be increasing, got {(first, last)}")
        if self.doppler.cutoff_rule == "fixed" and self.doppler.n_discard >= self.sequence.n_frames:
            raise ValueError(
                f"doppler.n_discard ({self.doppler.n_discard}) must be below "
                f"sequence.n_frames ({self.sequence.n_frames})"
            )
        pixel_grid_from_config(self)
        check_time_window(
            frequencies_from_config(self),
            transducer_from_config(self),
            sequence_from_config(self),
            roi_grid_from_config(self),
            C_REF,
        )


@dataclass(frozen=True)
class ImagingSetup:
    """The objects every stage of the chain needs, derived once from an `ImagingConfig`.

    Attributes:
        transducer: the array.
        sequence: the plane-wave sequence.
        freqs: frequency grid [Hz], float64, shape (nf,).
        roi: ROI grid (scatterer region and bank storage grid).
        pixel_grid: beamforming grid.
    """

    transducer: Transducer
    sequence: PlaneWaveSequence
    freqs: np.ndarray
    roi: Grid2D
    pixel_grid: Grid2D


def imaging_setup(cfg: ImagingConfig) -> ImagingSetup:
    """Derive the array, sequence, frequency grid and both grids from a config."""
    return ImagingSetup(
        transducer=transducer_from_config(cfg),
        sequence=sequence_from_config(cfg),
        freqs=frequencies_from_config(cfg),
        roi=roi_grid_from_config(cfg),
        pixel_grid=pixel_grid_from_config(cfg),
    )


def transducer_from_config(cfg: ImagingConfig) -> Transducer:
    """Build the array: pitch and element width follow from the wavelength at `C_REF`."""
    spec = cfg.transducer
    pitch = spec.pitch_wavelengths * float(wavelength(C_REF, spec.center_frequency))
    return Transducer(
        n_elements=spec.n_elements,
        pitch=pitch,
        element_width=spec.element_width_fraction * pitch,
        center_frequency=spec.center_frequency,
        fractional_bandwidth=spec.fractional_bandwidth,
    )


def sequence_from_config(cfg: ImagingConfig) -> PlaneWaveSequence:
    """Build the plane-wave sequence."""
    spec = cfg.sequence
    return PlaneWaveSequence(
        angles_deg=spec.angles_deg, frame_rate=spec.frame_rate, n_frames=spec.n_frames
    )


def frequencies_from_config(cfg: ImagingConfig) -> np.ndarray:
    """Frequency grid [Hz], float64, shape (n_frequencies,), ascending and uniformly spaced."""
    return band_frequencies(
        transducer_from_config(cfg), cfg.frequencies.n_frequencies, cfg.frequencies.truncation_db
    )


def roi_grid_from_config(cfg: ImagingConfig) -> Grid2D:
    """ROI grid: scatterers live inside its cells, banks store `B` on its nodes."""
    return Grid2D.from_extent(cfg.roi.x_range, cfg.roi.z_range, cfg.roi.spacing)


def pixel_grid_from_config(cfg: ImagingConfig) -> Grid2D:
    """Beamforming grid centred on the ROI.

    Raises:
        ValueError: if the grid with the configured spacing does not fit in the ROI.
    """
    nz, nx = cfg.pixels.shape
    (x_first, x_last), (z_first, z_last) = cfg.roi.x_range, cfg.roi.z_range
    spacing = cfg.pixels.spacing
    if spacing is None:
        extents = ((z_last - z_first, nz), (x_last - x_first, nx))
        fits = [extent / (n - 1) for extent, n in extents if n > 1]
        spacing = min(fits) if fits else cfg.roi.spacing
    half_z = 0.5 * (nz - 1) * spacing
    half_x = 0.5 * (nx - 1) * spacing
    tolerance = 1e-9 * spacing
    if 2 * half_z > z_last - z_first + tolerance or 2 * half_x > x_last - x_first + tolerance:
        raise ValueError(
            f"pixel grid {cfg.pixels.shape} with spacing {spacing} m does not fit in the ROI"
        )
    origin = (0.5 * (z_first + z_last) - half_z, 0.5 * (x_first + x_last) - half_x)
    return Grid2D(shape=(nz, nx), spacing=spacing, origin=origin)
