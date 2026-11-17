"""Clinical map from CT Hounsfield units to acoustic properties.

Aubry et al. (2003), J. Acoust. Soc. Am. 113(1):84-93, Eqs. 3-6, as used for
CT-based transcranial focusing by Marquet et al. (2009), Phys. Med. Biol.
54(9):2597-2613:

    porosity     phi = 1 - HU / HU_bone, clipped to [0, 1]
    density      rho = phi rho_water + (1 - phi) rho_bone
    sound speed  c   = c_water + (c_bone - c_water) (1 - phi)
    absorption   a   = a_min + (a_max - a_min) phi^beta      (bone voxels only)

The constants and the ranges they are drawn from live in
`neurosim.ct.config.HuMappingConfig`. The map is the reconstruction's skull
prior; it is deliberately not the ground-truth material model of the head.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.seeding import SeedTree
from neurosim.ct.calibration import effective_hu
from neurosim.ct.config import CtConfig, HuMappingConfig
from neurosim.ct.types import HuMapping
from neurosim.head.materials import power_law_prefactor

# Absorption prefactor given to voxels below the bone threshold
# (`HuMappingConfig.bone_threshold_hu`) [dB/(cm MHz^y)]: water-like, alpha_water = 0
# in Yaakub et al. 2023, Table 1.
SOFT_TISSUE_ALPHA0 = 0.0


def _alpha0(absorption_db_cm: float, mapping_cfg: HuMappingConfig) -> float:
    """Express an absorption [dB/cm] at the mapping's frequency as alpha0 [dB/(cm MHz^y)]."""
    return float(
        power_law_prefactor(
            absorption_db_cm, mapping_cfg.absorption_frequency, mapping_cfg.alpha_power
        )
    )


def _nominal_hu_bone(cfg: CtConfig) -> float:
    """HU of fully dense cortical bone: the configured value, or the scanner model's."""
    if cfg.hu_mapping.hu_bone is not None:
        return cfg.hu_mapping.hu_bone
    return float(effective_hu(cfg)[BASIS_MATERIALS.index("cortical_bone")])


def nominal_hu_mapping(cfg: CtConfig) -> HuMapping:
    """HU -> acoustics map with the nominal (published) constants.

    ``hu_bone`` is taken from ``cfg.hu_mapping.hu_bone`` or, if that is None,
    computed for the scan settings (kVp, filtration) with
    `neurosim.ct.calibration.effective_hu`.
    """
    mapping_cfg = cfg.hu_mapping
    return HuMapping(
        hu_bone=_nominal_hu_bone(cfg),
        c_water=mapping_cfg.c_water,
        c_bone=mapping_cfg.c_bone,
        rho_water=mapping_cfg.rho_water,
        rho_bone=mapping_cfg.rho_bone,
        alpha_min=_alpha0(mapping_cfg.absorption_min, mapping_cfg),
        alpha_max=_alpha0(mapping_cfg.absorption_max, mapping_cfg),
        alpha_exponent=mapping_cfg.absorption_exponent,
    )


def sample_hu_mapping(tree: SeedTree, cfg: CtConfig, *sub: int) -> HuMapping:
    """HU -> acoustics map with constants drawn within their published uncertainty.

    Each constant is drawn uniformly from its ``*_range`` in
    ``cfg.hu_mapping`` (``hu_bone``: the nominal value times a factor from
    ``hu_bone_relative_range``), in a fixed order, from the stream
    ``("ct_hu_mapping", *sub)``. ``rho_water`` is not randomised.

    Args:
        tree: seed tree of the run.
        cfg: scan settings and mapping ranges.
        *sub: sub-indices of the stream, for example one per posterior sample.
    """
    mapping_cfg = cfg.hu_mapping
    rng = tree.rng("ct_hu_mapping", *sub)
    hu_bone = _nominal_hu_bone(cfg) * rng.uniform(*mapping_cfg.hu_bone_relative_range)
    c_water = rng.uniform(*mapping_cfg.c_water_range)
    c_bone = rng.uniform(*mapping_cfg.c_bone_range)
    rho_bone = rng.uniform(*mapping_cfg.rho_bone_range)
    absorption_min = rng.uniform(*mapping_cfg.absorption_min_range)
    absorption_max = rng.uniform(*mapping_cfg.absorption_max_range)
    absorption_exponent = rng.uniform(*mapping_cfg.absorption_exponent_range)
    return HuMapping(
        hu_bone=float(hu_bone),
        c_water=float(c_water),
        c_bone=float(c_bone),
        rho_water=mapping_cfg.rho_water,
        rho_bone=float(rho_bone),
        alpha_min=_alpha0(absorption_min, mapping_cfg),
        alpha_max=_alpha0(absorption_max, mapping_cfg),
        alpha_exponent=float(absorption_exponent),
    )


def hu_to_acoustic(
    hu: np.ndarray, mapping: HuMapping, bone_threshold_hu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a CT image to acoustic property maps (module docstring equations).

    Sound speed and density follow the porosity everywhere, so HU <= 0 (soft
    tissue, water, air) gives the water end points. The absorption law applies
    to bone voxels (HU >= ``bone_threshold_hu``); other voxels get
    `SOFT_TISSUE_ALPHA0`.

    Args:
        hu: CT numbers [HU], any shape.
        mapping: constants of the map.
        bone_threshold_hu: CT number from which a voxel is bone [HU]
            (``CtConfig.hu_mapping.bone_threshold_hu``).

    Returns:
        sound_speed [m/s], density [kg/m^3] and attenuation, the power-law
        prefactor alpha0 [dB/(cm MHz^y)]; float32 arrays of the shape of ``hu``.
    """
    hu = np.asarray(hu, dtype=np.float64)
    porosity = np.clip(1.0 - hu / mapping.hu_bone, 0.0, 1.0)
    solid = 1.0 - porosity
    sound_speed = mapping.c_water + (mapping.c_bone - mapping.c_water) * solid
    density = porosity * mapping.rho_water + solid * mapping.rho_bone
    absorption_span = mapping.alpha_max - mapping.alpha_min
    bone_absorption = mapping.alpha_min + absorption_span * porosity**mapping.alpha_exponent
    attenuation = np.where(hu >= bone_threshold_hu, bone_absorption, SOFT_TISSUE_ALPHA0)
    return (
        sound_speed.astype(np.float32),
        density.astype(np.float32),
        attenuation.astype(np.float32),
    )
