"""Material mapping: acoustic and X-ray properties from the latent microstructure.

Everything here is a pure function of ``(labels, porosity, hidden_c)`` (ROADMAP
D-002, docs/ARCHITECTURE.md section 4 rule 1). ``hidden_c`` enters the sound
speed only; the X-ray view (`material_fractions`) never sees it.

Skull: the porosity mixing law of Aubry et al. (2003), J. Acoust. Soc. Am.
113(1):84-93, Eqs. 4-6::

    density      rho = phi * rho_fluid + (1 - phi) * rho_bone
    sound speed  c   = c_fluid + (c_bone - c_fluid) * (1 - phi)      [+ hidden_c]
    absorption   a   = a_min + (a_max - a_min) * phi**beta

Soft tissues: one row of sourced values per tissue (`MaterialParams`).

Absorption convention: the contract stores the power-law prefactor ``alpha0`` in
dB/(cm MHz^y) with one global exponent ``y`` (`MaterialParams.alpha_power`).
Published values are quoted in dB/cm at one frequency, so each one is stored
here with that frequency and converted with ``alpha0 = alpha(f_ref) /
(f_ref / 1 MHz)**y``: the model then reproduces the published value at its own
frequency whatever ``y`` is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.config import field
from neurosim.common.constants import (
    ALPHA_POWER_DEFAULT,
    AUBRY_2003_ABSORPTION_EXPONENT,
    AUBRY_2003_C_MAX,
    AUBRY_2003_C_MIN,
    BASIS_MATERIALS,
    C_WATER,
    RHO_CORTICAL_BONE_ICRU44,
    RHO_SOFT_TISSUE_ICRU44,
    RHO_WATER,
)
from neurosim.common.units import db_cm_mhz_to_np_m, np_m_to_db_cm
from neurosim.head.types import BRAIN_LABELS, SKULL_LABELS, Tissue

# Aubry et al. (2022), "Benchmark problems for transcranial ultrasound simulation:
# Intercomparison of compressional wave models", J. Acoust. Soc. Am. 152(2):1003,
# Table I, quotes every absorption value at this frequency [Hz].
_BENCHMARK_FREQUENCY = 0.5e6


@dataclass(frozen=True)
class TissueAcoustics:
    """Acoustic properties of one homogeneous soft tissue or fluid."""

    sound_speed: float = field(gt=0.0, doc="Compressional sound speed [m/s].")
    density: float = field(gt=0.0, doc="Mass density [kg/m^3].")
    absorption: float = field(
        ge=0.0, doc="Absorption coefficient at `absorption_frequency` [dB/cm]."
    )
    absorption_frequency: float = field(
        gt=0.0, doc="Frequency at which `absorption` is quoted [Hz]."
    )


@dataclass(frozen=True)
class SkullMixing:
    """Constants of the porosity mixing law for skull bone (Aubry et al. 2003, Eqs. 4-6)."""

    c_fluid: float = field(
        AUBRY_2003_C_MIN,
        gt=0.0,
        doc="Sound speed of the pore fluid [m/s]. Aubry 2003: c_min = 1.5 mm/us "
        "(water-filled pores).",
    )
    c_bone: float = field(
        AUBRY_2003_C_MAX,
        gt=0.0,
        doc="Sound speed of pore-free cortical bone [m/s]. Aubry 2003: c_max = 2.9 mm/us, "
        "the value of Fry & Barger (1978) for both tables. Later CT-based work used "
        "3100 m/s (Marquet et al. 2009, Phys. Med. Biol. 54:2597; value not checked "
        "against the paper: assumed, to verify).",
    )
    rho_fluid: float = field(
        RHO_SOFT_TISSUE_ICRU44,
        gt=0.0,
        doc="Mass density of the pore content [kg/m^3]. 1.060 g/cm^3, 'Tissue, Soft "
        "(ICRU-44)' in NIST X-Ray Mass Attenuation Coefficients, Table 2: the basis "
        "material that fills the pores in `material_fractions`, so the acoustic density "
        "equals the mass density the CT model sees (eval E-HEAD-007). Aubry 2003 used "
        "1000 (water-filled skulls in vitro).",
    )
    rho_bone: float = field(
        RHO_CORTICAL_BONE_ICRU44,
        gt=0.0,
        doc="Mass density of pore-free cortical bone [kg/m^3]. 1.920 g/cm^3, 'Bone, "
        "Cortical (ICRU-44)' in NIST Table 2, the CT basis material. Aubry 2003 used "
        "2100, tuned so that the CT-apparent inner table averaged the 1900 of Fry & "
        "Barger; with 1920 and a table porosity of 0.06 the tables come out at about "
        "1870, next to the 1850 of the Aubry 2022 benchmark.",
    )
    absorption_min: float = field(
        1.4,
        ge=0.0,
        doc="Absorption of pore-free bone at `absorption_frequency` [dB/cm]. Fitted "
        "together with `absorption_max` so that the law gives the Aubry 2022 benchmark "
        "values, 4 dB/cm in cortical bone (porosity 0.07) and 8 dB/cm in trabecular "
        "bone (porosity 0.45) at 0.5 MHz. Aubry 2003 fitted 0.2 dB/mm (2 dB/cm) to one "
        "skull with 1.5 MHz transducers.",
    )
    absorption_max: float = field(
        11.2,
        ge=0.0,
        doc="Absorption at porosity 1 at `absorption_frequency` [dB/cm]; see "
        "`absorption_min`. Aubry 2003 fitted 8 dB/mm (80 dB/cm) with 1.5 MHz "
        "transducers, which gives about twice the skull loss measured later "
        "(Pinton et al. 2012, Med. Phys. 39:299: 13.3 dB/cm total at 1 MHz; "
        "assumed, to verify).",
    )
    absorption_exponent: float = field(
        AUBRY_2003_ABSORPTION_EXPONENT,
        gt=0.0,
        doc="Exponent beta of porosity in the absorption law. Aubry 2003: best match "
        "0.5, plausible range 0.3-0.7.",
    )
    absorption_frequency: float = field(
        _BENCHMARK_FREQUENCY,
        gt=0.0,
        doc="Frequency at which the two absorption end points are quoted [Hz] "
        "(Aubry 2022, Table I).",
    )

    def __post_init__(self) -> None:
        if self.c_bone <= self.c_fluid:
            raise ValueError("c_bone must exceed c_fluid")
        if self.rho_bone <= self.rho_fluid:
            raise ValueError("rho_bone must exceed rho_fluid")
        if self.absorption_max < self.absorption_min:
            raise ValueError("absorption_max must be >= absorption_min")


def _water() -> TissueAcoustics:
    # Sound speed and density: the project's reference water (common.constants).
    # Absorption 0.0022 dB/cm at 1 MHz: the value used throughout the k-Wave
    # documentation and examples for water (alpha_coeff = 0.0022 dB/(MHz^y cm));
    # assumed, to verify.
    return TissueAcoustics(
        sound_speed=C_WATER, density=RHO_WATER, absorption=0.0022, absorption_frequency=1.0e6
    )


def _skin() -> TissueAcoustics:
    # Aubry et al. (2022), Table I, row "Skin".
    return TissueAcoustics(
        sound_speed=1610.0,
        density=1090.0,
        absorption=0.2,
        absorption_frequency=_BENCHMARK_FREQUENCY,
    )


def _brain() -> TissueAcoustics:
    # Aubry et al. (2022), Table I, row "Brain".
    return TissueAcoustics(
        sound_speed=1560.0,
        density=1040.0,
        absorption=0.3,
        absorption_frequency=_BENCHMARK_FREQUENCY,
    )


@dataclass(frozen=True)
class MaterialParams:
    """Parameters of the material mapping."""

    alpha_power: float = field(
        ALPHA_POWER_DEFAULT,
        gt=0.0,
        lt=3.0,
        doc="Global power-law exponent y of absorption, alpha0 * f**y. Soft tissue is "
        "close to 1, but k-Wave's power-law absorption has a dispersion term "
        "proportional to tan(pi y / 2), which is singular at y = 1, so 1 is rejected "
        "and 1.1 is the default; assumed, to verify.",
    )
    background: TissueAcoustics = field(
        default_factory=_water, doc="Coupling medium outside the head (water)."
    )
    scalp: TissueAcoustics = field(default_factory=_skin, doc="Scalp. Aubry 2022, Table I, 'Skin'.")
    csf: TissueAcoustics = field(
        default_factory=_water,
        doc="Cerebrospinal fluid, taken as water; assumed, to verify.",
    )
    grey: TissueAcoustics = field(
        default_factory=_brain, doc="Grey matter. Aubry 2022, Table I, 'Brain'."
    )
    white: TissueAcoustics = field(
        default_factory=_brain,
        doc="White matter. Aubry 2022, Table I, 'Brain' (one value for both brain "
        "tissues; a separate white-matter value is assumed, to verify).",
    )
    skull: SkullMixing = field(default_factory=SkullMixing, doc="Skull mixing law.")

    def __post_init__(self) -> None:
        if self.alpha_power == 1.0:
            raise ValueError("alpha_power must not be exactly 1 (singular in k-Wave)")


def power_law_prefactor(absorption: np.ndarray | float, frequency: float, y: float) -> np.ndarray:
    """Convert an absorption quoted at one frequency to the power-law prefactor.

    absorption: absorption coefficient at `frequency` [dB/cm].
    frequency: frequency at which it is quoted [Hz].
    y: power-law exponent.
    Returns alpha0 [dB/(cm MHz^y)] such that ``alpha0 * (frequency / 1 MHz)**y``
    equals `absorption`.
    """
    unit_prefactor_at_frequency = np_m_to_db_cm(db_cm_mhz_to_np_m(1.0, frequency, y))
    return np.asarray(absorption) / unit_prefactor_at_frequency


def skull_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of the three skull layers, same shape as `labels`."""
    return np.isin(labels, SKULL_LABELS)


def brain_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of grey and white matter, same shape as `labels`."""
    return np.isin(labels, BRAIN_LABELS)


def skull_sound_speed_from_porosity(porosity: np.ndarray, skull: SkullMixing) -> np.ndarray:
    """Porosity-driven part of the skull sound speed (Aubry 2003, Eq. 5).

    porosity: pore volume fraction in [0, 1], any shape.
    Returns sound speed [m/s], same shape and dtype; `hidden_c` is not included.
    """
    return skull.c_fluid + (skull.c_bone - skull.c_fluid) * (1.0 - porosity)


def _tissue_rows(params: MaterialParams) -> dict[Tissue, TissueAcoustics]:
    return {
        Tissue.BACKGROUND: params.background,
        Tissue.SCALP: params.scalp,
        Tissue.CSF: params.csf,
        Tissue.GREY: params.grey,
        Tissue.WHITE: params.white,
    }


def acoustic_properties(
    labels: np.ndarray, porosity: np.ndarray, hidden_c: np.ndarray, params: MaterialParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acoustic maps from the latent fields.

    labels: `Tissue` values, uint8, any shape.
    porosity: skull pore volume fraction in [0, 1], same shape; used in the skull only.
    hidden_c: sound-speed component invisible to X-rays [m/s], same shape; added
        to the sound speed in the skull only.
    params: material parameters.

    Returns ``(sound_speed [m/s], density [kg/m^3], attenuation [dB/(cm MHz^y)])``,
    float32 arrays of the shape of `labels`, with ``y = params.alpha_power``.
    """
    labels = np.asarray(labels)
    if porosity.shape != labels.shape or hidden_c.shape != labels.shape:
        raise ValueError(
            f"labels {labels.shape}, porosity {porosity.shape} and hidden_c "
            f"{hidden_c.shape} must have the same shape"
        )
    y = params.alpha_power
    lookup = np.zeros((3, len(Tissue)), dtype=np.float32)
    for tissue, row in _tissue_rows(params).items():
        lookup[:, tissue] = (
            row.sound_speed,
            row.density,
            power_law_prefactor(row.absorption, row.absorption_frequency, y),
        )
    sound_speed, density, attenuation = (table[labels] for table in lookup)

    in_skull = skull_mask(labels)
    skull = params.skull
    phi = porosity[in_skull].astype(np.float32)
    sound_speed[in_skull] = skull_sound_speed_from_porosity(phi, skull) + hidden_c[in_skull]
    density[in_skull] = phi * skull.rho_fluid + (1.0 - phi) * skull.rho_bone
    absorption = skull.absorption_min + (skull.absorption_max - skull.absorption_min) * (
        phi**skull.absorption_exponent
    )
    attenuation[in_skull] = power_law_prefactor(absorption, skull.absorption_frequency, y)
    return sound_speed, density, attenuation


def material_fractions(labels: np.ndarray, porosity: np.ndarray) -> np.ndarray:
    """Volume fractions of the CT basis materials (`common.constants.BASIS_MATERIALS`).

    Background and CSF are water; scalp, grey and white matter are soft tissue;
    a skull voxel is ``1 - porosity`` cortical bone and ``porosity`` soft tissue
    (the marrow and vessels in the pores are approximated as soft tissue). No
    voxel contains air.

    labels: `Tissue` values, uint8, any shape.
    porosity: skull pore volume fraction in [0, 1], same shape.

    Returns float32 fractions of shape ``(4, *labels.shape)`` that sum to 1.
    """
    labels = np.asarray(labels)
    if porosity.shape != labels.shape:
        raise ValueError(f"labels {labels.shape} and porosity {porosity.shape} differ in shape")
    fractions = np.zeros((len(BASIS_MATERIALS), *labels.shape), dtype=np.float32)
    water = fractions[BASIS_MATERIALS.index("water")]
    soft_tissue = fractions[BASIS_MATERIALS.index("soft_tissue")]
    cortical_bone = fractions[BASIS_MATERIALS.index("cortical_bone")]

    is_water = (labels == Tissue.BACKGROUND) | (labels == Tissue.CSF)
    in_skull = skull_mask(labels)
    water[is_water] = 1.0
    soft_tissue[~is_water & ~in_skull] = 1.0
    phi = porosity[in_skull].astype(np.float32)
    soft_tissue[in_skull] = phi
    cortical_bone[in_skull] = 1.0 - phi
    return fractions
