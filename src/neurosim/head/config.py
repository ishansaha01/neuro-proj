"""Typed configuration of the ground-truth head (docs/ARCHITECTURE.md, section 4).

Every default that is a physical or modelling choice names its source in the
field doc, or is marked ``assumed, to verify``. Sources used below:

- de Boer et al. (2016), Int. J. Legal Med. 130(5):1371-1377, Table 4: cranial
  vault thickness in 994 adults. Frontal 6.15 (SD 1.91) mm in men and 7.13
  (2.30) mm in women; lateral (euryon) 4.33 (1.22) and 4.41 (1.15) mm;
  occipital 7.33 (2.01) and 6.89 (2.00) mm.
- Rowbotham et al. (2023), Int. J. Legal Med. 137:195-213: site means of the
  neurocranium start at 2.11 mm, in the temporal squama.
- Aubry et al. (2022), J. Acoust. Soc. Am. 152(2):1003 (transcranial benchmark):
  4 mm skin over a 6.5 mm parietal skull made of 1.5 mm outer table, 4 mm
  diploe and 1 mm inner table (averages from Alexander et al. 2019 and Hori et
  al. 1972); Table I gives 2800 m/s for cortical and 2300 m/s for trabecular bone.
- Jing & Lindsey (2021), Ultrasound Med. Biol. 47(9):2734: a micro-CT parietal
  sample with 26.9 % porosity over its full thickness.
- Adanty et al. (2021), Bone, "Cortical and trabecular morphometric properties
  of the human calvarium": the inner table is more porous than the outer table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neurosim.common.config import field
from neurosim.head.materials import MaterialParams


@dataclass(frozen=True)
class GridConfig:
    """Voxel grid of the head volume."""

    spacing: float = field(
        0.5e-3, gt=0.0, doc="Isotropic voxel size [m]. ROADMAP Phase 1 target: 0.5 mm."
    )
    region: Literal["full", "box"] = field(
        "box",
        doc="'full': the whole head (about 7e7 voxels and 8 GiB at 0.5 mm). "
        "'box': only a box around the target and the probe.",
    )
    box_size: tuple[float, float, float] = field(
        (0.08, 0.08, 0.08),
        gt=0.0,
        doc="Full widths of the box along head x, y, z [m] (region 'box').",
    )
    box_center_depth: float = field(
        0.025,
        doc="Depth of the box centre below the scalp along the target direction [m] "
        "(region 'box'). With 25 mm an 80 mm box holds an imaging plane about 50 mm wide "
        "and 55 mm deep under the default probe, whose axis is tilted about 30 degrees "
        "from the head z axis.",
    )
    full_margin: float = field(
        4.0e-3,
        ge=0.0,
        doc="Coupling medium kept around the largest possible scalp [m] (region 'full').",
    )


@dataclass(frozen=True)
class AnatomyConfig:
    """Nested head surfaces: a deformed ellipsoid scalp and smooth thickness maps.

    A thickness is ``median * exp(variation * g)`` with ``g`` a smooth random
    field on the sphere squashed to (-3, 3), so it is strictly positive and
    bounded. Thicknesses are measured along the radial direction.
    """

    semi_axes: tuple[float, float, float] = field(
        (0.076, 0.096, 0.086),
        gt=0.0,
        doc="Median scalp semi-axes along head x, y, z [m]: half of a 152 mm head "
        "breadth and a 192 mm head length; the vertical semi-axis closes the vault as "
        "an ellipsoid (no face or neck). Adult anthropometry; assumed, to verify.",
    )
    size_variation: float = field(
        0.03,
        ge=0.0,
        le=0.2,
        doc="Standard deviation of the log scale factor of each semi-axis between "
        "heads (truncated at 2.5 sigma); assumed, to verify.",
    )
    shape_variation: float = field(
        0.02,
        ge=0.0,
        le=0.1,
        doc="Standard deviation of the smooth log-radius deformation of the scalp; "
        "assumed, to verify.",
    )
    shape_correlation: float = field(
        0.7,
        gt=0.0,
        doc="Correlation length of the scalp deformation on the unit sphere [rad]; "
        "assumed, to verify.",
    )
    thickness_correlation: float = field(
        0.4,
        gt=0.0,
        doc="Correlation length of every thickness map on the unit sphere [rad] "
        "(about 30 mm on the scalp); assumed, to verify.",
    )
    scalp_thickness: float = field(
        5.0e-3,
        gt=0.0,
        doc="Median scalp thickness [m]. Aubry 2022 uses 4 mm (Hori et al. 1972); "
        "5 mm with the variation below spans about 3.5-7 mm; assumed, to verify.",
    )
    scalp_variation: float = field(
        0.12, ge=0.0, le=0.5, doc="Log standard deviation of scalp thickness; assumed, to verify."
    )
    skull_thickness_vault: float = field(
        6.5e-3,
        gt=0.0,
        doc="Median skull thickness away from the temporal window [m]. 6.5 mm parietal "
        "(Aubry 2022); frontal and occipital means 6.2-7.3 mm (de Boer 2016).",
    )
    skull_thickness_temporal: float = field(
        3.0e-3,
        gt=0.0,
        doc="Median skull thickness at the centre of the temporal window [m]: between "
        "the 2.11 mm temporal squama (Rowbotham 2023) and the 4.4 mm euryon (de Boer 2016).",
    )
    temporal_window_direction: tuple[float, float, float] = field(
        (1.0, 0.0, -0.2),
        doc="Direction of the right temporal window from the head centre (normalised "
        "in the code; the left window is its mirror image in x): lateral and slightly "
        "inferior; assumed, to verify.",
    )
    temporal_window_width: float = field(
        0.45,
        gt=0.0,
        doc="Angular standard deviation of the thin temporal region [rad]; assumed, to verify.",
    )
    skull_variation: float = field(
        0.15,
        ge=0.0,
        le=0.5,
        doc="Log standard deviation of skull thickness within and between heads. "
        "de Boer 2016 reports SD/mean of about 0.3 across people at one site; a smaller "
        "value is used because the sites of one skull are correlated; assumed, to verify.",
    )
    outer_table_thickness: float = field(
        1.5e-3, gt=0.0, doc="Median outer-table thickness in thick bone [m]. Aubry 2022."
    )
    inner_table_thickness: float = field(
        1.0e-3, gt=0.0, doc="Median inner-table thickness in thick bone [m]. Aubry 2022."
    )
    table_variation: float = field(
        0.12, ge=0.0, le=0.5, doc="Log standard deviation of table thickness; assumed, to verify."
    )
    outer_table_max_fraction: float = field(
        0.4,
        gt=0.0,
        lt=1.0,
        doc="Largest share of the local skull thickness the outer table may take (thin "
        "bone is mostly cortical); assumed, to verify.",
    )
    inner_table_max_fraction: float = field(
        0.3,
        gt=0.0,
        lt=1.0,
        doc="Largest share of the local skull thickness the inner table may take; "
        "assumed, to verify.",
    )
    csf_thickness: float = field(
        3.0e-3,
        gt=0.0,
        doc="Median thickness of the CSF layer between skull and cortex [m]; assumed, to verify.",
    )
    csf_variation: float = field(
        0.2, ge=0.0, le=0.5, doc="Log standard deviation of CSF thickness; assumed, to verify."
    )
    grey_thickness: float = field(
        2.5e-3,
        gt=0.0,
        doc="Median cortical grey-matter thickness [m]. Fischl & Dale (2000), PNAS "
        "97(20):11050: 1-4.5 mm, about 2.5 mm on average (quoted from memory: assumed, "
        "to verify). The cortex is a smooth shell here, without gyri and sulci.",
    )
    grey_variation: float = field(
        0.15, ge=0.0, le=0.5, doc="Log standard deviation of cortex thickness; assumed, to verify."
    )

    def __post_init__(self) -> None:
        if self.outer_table_max_fraction + self.inner_table_max_fraction >= 1.0:
            raise ValueError("table max fractions must sum to less than 1 (no room for diploe)")
        if all(component == 0.0 for component in self.temporal_window_direction):
            raise ValueError("temporal_window_direction must not be the zero vector")


@dataclass(frozen=True)
class MicrostructureConfig:
    """Skull porosity and the sound-speed component hidden from X-rays."""

    porosity_outer_table: float = field(
        0.06,
        ge=0.0,
        le=1.0,
        doc="Mean porosity of the outer table. Under the Aubry 2003 law the benchmark "
        "cortical speed of 2800 m/s (Aubry 2022) is a porosity of 0.07; assumed, to verify.",
    )
    porosity_diploe: float = field(
        0.40,
        ge=0.0,
        le=1.0,
        doc="Mean porosity of the diploe. Under the Aubry 2003 law the benchmark "
        "trabecular speed of 2300 m/s is a porosity of 0.43; with the table values and "
        "1.5/4/1 mm layers the full-thickness mean is 0.28, next to the 26.9 % of Jing & "
        "Lindsey 2021; assumed, to verify.",
    )
    porosity_inner_table: float = field(
        0.10,
        ge=0.0,
        le=1.0,
        doc="Mean porosity of the inner table, above the outer table (Adanty 2021, "
        "qualitative); assumed, to verify.",
    )
    layer_transition: float = field(
        0.1e-3,
        gt=0.0,
        doc="Width of the smooth transition between table and diploe porosity [m]; "
        "assumed, to verify.",
    )
    texture_variation: float = field(
        0.3,
        ge=0.0,
        le=1.0,
        doc="Log standard deviation of the porosity texture that multiplies the layer "
        "profile; assumed, to verify.",
    )
    texture_correlation: float = field(
        1.5e-3,
        gt=0.0,
        doc="Correlation length of the porosity texture [m] (order of the trabecular "
        "spacing; porosity is a local volume average, pores are not resolved); "
        "assumed, to verify.",
    )
    porosity_r2: float = field(
        0.5,
        gt=0.0,
        le=1.0,
        doc="Fraction of skull sound-speed variance explained by porosity (R^2 of a "
        "linear fit over skull voxels). 0.5 so that the CT measurement, which degrades "
        "it further, lands in the 23-53 % of Webb et al. (2018), IEEE TUFFC "
        "65(7):1111-1124.",
    )
    hidden_correlation: float = field(
        3.0e-3,
        gt=0.0,
        doc="Correlation length of the hidden sound-speed field [m]; assumed, to verify.",
    )
    sound_speed_max: float = field(
        3500.0,
        gt=0.0,
        doc="Upper bound of skull sound speed [m/s]; the lower bound is the pore-fluid "
        "speed of the mixing law. Clement & Hynynen (2002) report 2209-3507 m/s over "
        "their density range (as quoted by Angla et al. 2023, Med. Phys., "
        "'Transcranial ultrasound simulations: a review').",
    )


@dataclass(frozen=True)
class CbvConfig:
    """Spatial ground truth of cerebral blood volume and blood velocity.

    The vascular texture (a smooth log-normal field plus tubular structures along
    the common zero lines of two smooth random fields) is a modelling assumption,
    not an anatomical vascular tree.
    """

    cbv_grey: float = field(
        0.04,
        ge=0.0,
        le=1.0,
        doc="Median blood volume fraction of grey matter. Reported values run from "
        "3.5 ml/100 g (Wikipedia 'Cerebral blood volume') to 5.2 ml/100 ml (Leenders "
        "et al. 1990, Brain 113:27, from memory); assumed, to verify.",
    )
    cbv_white: float = field(
        0.02,
        ge=0.0,
        le=1.0,
        doc="Median blood volume fraction of white matter. Reported values run from "
        "1.7 ml/100 g to 2.7 ml/100 ml (same sources as `cbv_grey`); assumed, to verify.",
    )
    texture_variation: float = field(
        0.3, ge=0.0, le=1.0, doc="Log standard deviation of the smooth CBV texture; assumed."
    )
    texture_correlation: float = field(
        2.0e-3, gt=0.0, doc="Correlation length of the smooth CBV texture [m]; assumed."
    )
    vessel_correlation: float = field(
        4.0e-3,
        gt=0.0,
        doc="Correlation length of the two fields whose common zero lines are the vessel "
        "axes [m]; it sets the spacing and tortuosity of the vessels; assumed.",
    )
    vessel_radius: float = field(
        0.5e-3, gt=0.0, doc="Gaussian radius of the vessel cross-section [m]; assumed."
    )
    vessel_cbv: float = field(
        0.25,
        ge=0.0,
        le=1.0,
        doc="Blood volume fraction added on a vessel axis (vessels are not resolved, so "
        "it is below 1); assumed.",
    )
    target_direction: tuple[float, float, float] = field(
        (0.5, 0.0, 0.866),
        doc="Direction from the head centre to the activated cortical site (normalised in "
        "the code). Default: right superior-lateral convexity, 30 degrees from the vertex, "
        "analogous to the parietal and motor areas of Norman et al. (2021); assumed.",
    )
    activation_radius: float = field(
        5.0e-3,
        gt=0.0,
        doc="Gaussian radius of the activated patch [m]: the pattern is "
        "exp(-d^2 / (2 radius^2)) inside grey matter; assumed.",
    )
    speed_tissue: float = field(
        1.0e-3,
        ge=0.0,
        doc="Median blood speed away from vessels [m/s]. fUS detects flow down to about "
        "1 mm/s (Iconeus, 'What is functional ultrasound?', citing Mace et al. 2011); "
        "assumed, to verify.",
    )
    speed_vessel: float = field(
        20.0e-3,
        ge=0.0,
        doc="Blood speed on a vessel axis [m/s]; arterioles and venules, order of "
        "10-30 mm/s; assumed, to verify.",
    )

    def __post_init__(self) -> None:
        if all(component == 0.0 for component in self.target_direction):
            raise ValueError("target_direction must not be the zero vector")
        if self.speed_vessel < self.speed_tissue:
            raise ValueError("speed_vessel must be >= speed_tissue")


@dataclass(frozen=True)
class TaskConfig:
    """Block design of the task that drives the activation."""

    onsets: tuple[float, ...] = field(
        (10.0, 50.0, 90.0), ge=0.0, doc="Start time of each task block [s]; assumed."
    )
    block_duration: float = field(20.0, gt=0.0, doc="Length of every task block [s]; assumed.")
    peak_fractional_change: float = field(
        0.2,
        ge=0.0,
        doc="Largest fractional CBV change, reached by a long block at the centre of the "
        "activated patch. fUS reports 10-50 % in an activated region (Iconeus, 'What is "
        "functional ultrasound?'); assumed, to verify.",
    )

    def __post_init__(self) -> None:
        for previous, onset in zip(self.onsets, self.onsets[1:]):
            if onset < previous + self.block_duration:
                raise ValueError("task blocks must be in ascending order and must not overlap")


@dataclass(frozen=True)
class ProbeConfig:
    """Default placement of the imaging plane."""

    standoff: float = field(
        2.0e-3,
        ge=0.0,
        doc="Distance of the array centre from the scalp along the outward normal [m] "
        "(coupling gel); assumed.",
    )


@dataclass(frozen=True)
class HeadConfig:
    """Configuration of one ground-truth head."""

    grid: GridConfig = field(default_factory=GridConfig, doc="Voxel grid.")
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig, doc="Surfaces.")
    microstructure: MicrostructureConfig = field(
        default_factory=MicrostructureConfig, doc="Skull porosity and hidden sound speed."
    )
    materials: MaterialParams = field(default_factory=MaterialParams, doc="Material mapping.")
    cbv: CbvConfig = field(default_factory=CbvConfig, doc="CBV ground truth.")
    task: TaskConfig = field(default_factory=TaskConfig, doc="Task design.")
    probe: ProbeConfig = field(default_factory=ProbeConfig, doc="Default imaging plane.")
