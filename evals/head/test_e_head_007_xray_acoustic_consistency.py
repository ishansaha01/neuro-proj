"""E-HEAD-007: the X-ray and acoustic views describe the same skull mass."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.types import SKULL_LABELS

EVAL_ID = "E-HEAD-007"

# Mass density of the pure basis materials [kg/m^3]: NIST X-Ray Mass Attenuation
# Coefficients (Hubbell & Seltzer), Table 2: "Air, Dry" 1.205e-3 g/cm^3, "Water,
# Liquid" 1.000, "Tissue, Soft (ICRU-44)" 1.060, "Bone, Cortical (ICRU-44)" 1.920.
NIST_DENSITY = {"air": 1.205, "water": 1000.0, "soft_tissue": 1060.0, "cortical_bone": 1920.0}

SUM_TOLERANCE = 1.0e-6
DENSITY_TOLERANCE = 0.01
SPACING = 1.0e-3


@eval_case(
    id=EVAL_ID,
    claim="material_fractions sum to 1 in every voxel, and the mass density they imply in "
    "the skull equals the acoustic density map",
    reference="Mass conservation: rho = sum_i f_i rho_i with the NIST Table 2 densities of "
    "the basis materials (soft tissue ICRU-44 1060 kg/m^3, cortical bone ICRU-44 1920 kg/m^3)",
    metric="Largest |sum of fractions - 1| over all voxels; largest relative difference "
    "between the implied and the acoustic density over skull voxels",
    threshold="sum error < 1e-6 (float32 rounding) and density difference < 1 %: both "
    "views are functions of one porosity field, so only rounding should remain, and 1 % is "
    "well below the 9 % by which the Aubry 2003 bone density (2100) would disagree with "
    "the CT bone density (1920). Fixed before the first run",
    seeds=(0, 1, 2),
)
def test_e_head_007_xray_acoustic_consistency(small_cfg):
    cfg = dataclasses.replace(small_cfg, grid=dataclasses.replace(small_cfg.grid, spacing=SPACING))
    densities = np.array([NIST_DENSITY[name] for name in BASIS_MATERIALS])
    sum_error = 0.0
    density_error = 0.0
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(cfg, SeedTree(seed))
        fractions = head.material_fractions.astype(np.float64)
        sum_error = max(sum_error, float(np.abs(fractions.sum(axis=0) - 1.0).max()))
        in_skull = np.isin(head.labels, SKULL_LABELS)
        implied = np.tensordot(densities, fractions, axes=1)[in_skull]
        acoustic = head.density[in_skull].astype(np.float64)
        density_error = max(density_error, float(np.abs(implied / acoustic - 1.0).max()))
    report(
        EVAL_ID,
        {"max_fraction_sum_error": sum_error, "max_relative_density_difference": density_error},
        passed=sum_error < SUM_TOLERANCE and density_error < DENSITY_TOLERANCE,
    )
