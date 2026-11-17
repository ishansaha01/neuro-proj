"""Typed configuration of one run: head -> plane -> banks -> fUS data -> CT prior.

A `RunConfig` holds everything about a run except the seed (docs/ARCHITECTURE.md,
section 7): one config and one root seed give one reproducible run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from neurosim.common.config import field
from neurosim.common.grid import Grid2D
from neurosim.ct.config import CtConfig
from neurosim.data.ct_prior import CtPriorConfig
from neurosim.fus.backends import BackendConfig
from neurosim.fus.config import ImagingConfig, roi_grid_from_config, transducer_from_config
from neurosim.head.config import HeadConfig
from neurosim.head.types import SkullEffects

Fidelity = Literal["scatterer", "statistical"]

# Slack, in grid cells, when counting the cells needed to cover a length, so that a
# length that is a whole number of cells up to rounding does not gain one.
_CELL_COUNT_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class ProbePoseConfig:
    """Offsets of the probe from `head.plane.default_plane` (array centred over the target)."""

    lateral_shift: float = field(
        0.0,
        doc="Shift of the array centre along the array axis [m]. The activation centre then "
        "lies at x = -lateral_shift in the image. 0 = the default plane.",
    )
    tilt_deg: float = field(
        0.0,
        gt=-45.0,
        lt=45.0,
        doc="Rotation of the array in the imaging plane about its (shifted) centre [degrees]; "
        "positive turns the depth axis towards +x. 0 = the default plane.",
    )


@dataclass(frozen=True)
class TrialConfig:
    """The power Doppler image series of one scene."""

    n_images: int = field(
        120, ge=1, doc="Number of power Doppler images; image i is taken at i * image_interval."
    )
    image_interval: float = field(
        1.0,
        gt=0.0,
        doc="Time between the starts of successive images [s]. 1 s assumed, to verify (fUS "
        "systems give about one power Doppler image per second); it must hold one Doppler "
        "ensemble, n_frames / frame_rate.",
    )
    fidelity: Optional[Fidelity] = field(
        None, doc="Data fidelity of the series; None = `imaging.fidelity`."
    )
    amplitude_jitter: float = field(
        0.0,
        ge=0.0,
        doc="Log standard deviation of the response amplitude from task block to task block: "
        "the response to block b is scaled by exp(amplitude_jitter * n_b), n_b standard "
        "normal. 0 = every block responds alike. Modelling choice; assumed, to verify.",
    )


@dataclass(frozen=True)
class RunConfig:
    """Seed-independent description of one run (docs/ARCHITECTURE.md, section 7)."""

    head: HeadConfig = field(default_factory=HeadConfig, doc="Ground-truth head.")
    imaging: ImagingConfig = field(
        default_factory=ImagingConfig,
        doc="fUS array, sequence, frequency grid, ROI, pixel grid, noise and fidelity.",
    )
    backend: BackendConfig = field(
        default_factory=BackendConfig,
        doc="Backend of the true-skull and no-skull banks. The CT-derived bank and the "
        "true-skull correction model always use its `ray` settings.",
    )
    ct: CtConfig = field(default_factory=CtConfig, doc="CT scan and HU -> acoustics constants.")
    ct_prior: CtPriorConfig = field(
        default_factory=CtPriorConfig, doc="Plane-aligned CT slab and choice of HU mapping."
    )
    effects: SkullEffects = field(
        default_factory=SkullEffects, doc="Skull-effect switches (ROADMAP D-005); all on = real."
    )
    medium_spacing: float = field(
        1.0e-4,
        gt=0.0,
        doc="Pixel size of the fine acoustic grid in the imaging plane [m]. Numerical choice: "
        "0.1 mm puts ten points across a 1 mm skull table for the ray backend; the full-wave "
        "backend needs the element pitch to be a whole number of these cells (ROADMAP D-003 "
        "quotes 0.08 mm at 2 MHz).",
    )
    medium_margin: float = field(
        1.0e-3,
        ge=0.0,
        doc="Width of medium kept beyond the aperture and the ROI laterally and below the "
        "ROI [m]. Numerical choice.",
    )
    source_refinement: int = field(
        1,
        ge=1,
        doc="Sub-samples per pixel side over which |A|^2 is averaged when the power Doppler "
        "operator is built. 1 = the scatterer response is sampled at the pixel centre only, "
        "which overstates the blood signal when a pixel is wider than the point-spread "
        "function. Numerical choice; choose it so that pixel spacing / source_refinement is "
        "below about half a wavelength.",
    )
    probe: ProbePoseConfig = field(
        default_factory=ProbePoseConfig, doc="Probe offsets from the default plane."
    )
    trials: TrialConfig = field(default_factory=TrialConfig, doc="Image series.")

    def __post_init__(self) -> None:
        if self.ct.hu_mapping.alpha_power != self.head.materials.alpha_power:
            raise ValueError(
                f"ct.hu_mapping.alpha_power ({self.ct.hu_mapping.alpha_power}) must equal "
                f"head.materials.alpha_power ({self.head.materials.alpha_power}): the true and "
                "the CT-derived medium share one absorption power law"
            )
        ensemble = self.imaging.sequence.n_frames / self.imaging.sequence.frame_rate
        if self.trials.image_interval < ensemble:
            raise ValueError(
                f"trials.image_interval ({self.trials.image_interval} s) is shorter than one "
                f"Doppler ensemble ({ensemble} s = n_frames / frame_rate)"
            )
        if self.medium_spacing > self.imaging.roi.spacing:
            raise ValueError(
                f"medium_spacing ({self.medium_spacing} m) must not exceed the ROI node "
                f"spacing ({self.imaging.roi.spacing} m)"
            )


def trial_fidelity(cfg: RunConfig) -> Fidelity:
    """Data fidelity of the image series: the trial override, else `imaging.fidelity`."""
    return cfg.trials.fidelity or cfg.imaging.fidelity


def medium_grid(cfg: RunConfig) -> Grid2D:
    """Fine acoustic grid of the imaging plane.

    Rows run from the array face (z = 0, the first row) to `medium_margin` below
    the deepest ROI node. Columns are symmetric about x = 0, with a node on
    x = 0, and reach `medium_margin` beyond the wider of the aperture and the
    ROI. This is the placement the full-wave backend asks for (z = 0 on a row;
    element centres on nodes or half-way between nodes once the pitch is a
    whole number of cells).
    """
    spacing = cfg.medium_spacing
    transducer = transducer_from_config(cfg.imaging)
    roi_z, roi_x = roi_grid_from_config(cfg.imaging).axes()
    half_width = max(0.5 * transducer.aperture, abs(roi_x[0]), abs(roi_x[-1])) + cfg.medium_margin
    half_columns = math.ceil(half_width / spacing - _CELL_COUNT_TOLERANCE)
    last_row = math.ceil((roi_z[-1] + cfg.medium_margin) / spacing - _CELL_COUNT_TOLERANCE)
    return Grid2D(
        shape=(last_row + 1, 2 * half_columns + 1),
        spacing=spacing,
        origin=(0.0, -half_columns * spacing),
    )
