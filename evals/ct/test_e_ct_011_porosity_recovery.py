"""E-CT-011: the nominal HU map recovers the bone volume fraction of bone-water mixtures."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.ct.acoustics import nominal_hu_mapping
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-011"
MAX_POROSITY_ERROR = 0.1

BODY_RADIUS = 0.07
INSERT_RADIUS = 0.008
ROI_RADIUS = 0.004
RING_RADIUS = 0.04
BONE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@eval_case(
    id=EVAL_ID,
    claim="phi = 1 - HU / HU_bone with the nominal HU_bone of the scan settings recovers "
    "the true porosity of bone-water mixtures (the premise of the Aubry 2003 map holds "
    "in the simulated scanner)",
    reference="Known composition of the phantom: 16 mm rods with cortical-bone volume "
    "fractions 0.25, 0.5, 0.75 and 1 (pores filled with water) in a 140 mm water cylinder",
    metric="Maximum absolute porosity error over the four rods [-]",
    threshold="< 0.1. Fixed in advance: 0.1 in porosity is 140 m/s in the Aubry map, "
    "just below the 160 m/s standard deviation of HU-predicted velocity in Webb et al. "
    "2018; a scanner model worse than that would dominate the CT prior's error budget.",
    seeds=(0,),
)
def test_e_ct_011_porosity_recovery(small_cfg, small_grid):
    cfg = dataclasses.replace(small_cfg, photon_noise=False, slab_z_range=(0.0, 0.002))
    angles = 2.0 * np.pi * np.arange(len(BONE_FRACTIONS)) / len(BONE_FRACTIONS)
    centers = [(RING_RADIUS * np.cos(a), RING_RADIUS * np.sin(a)) for a in angles]
    layers = [("water", disc_coverage(small_grid, BODY_RADIUS))]
    layers += [
        ("cortical_bone", fraction * disc_coverage(small_grid, INSERT_RADIUS, center))
        for fraction, center in zip(BONE_FRACTIONS, centers)
    ]
    fractions = layered_fractions(small_grid, layers)
    hu = simulate_ct_fractions(fractions, small_grid, cfg, SeedTree(0)).hu[..., 0]
    hu_bone = nominal_hu_mapping(cfg).hu_bone

    errors = {}
    for fraction, center in zip(BONE_FRACTIONS, centers):
        roi = radial_distance(small_grid, center) < ROI_RADIUS
        porosity = float(np.clip(1.0 - hu[roi].mean() / hu_bone, 0.0, 1.0))
        errors[f"bone_fraction_{fraction:.2f}"] = abs(porosity - (1.0 - fraction))
    worst = max(errors.values())
    report(EVAL_ID, {"max": worst, "hu_bone": hu_bone, **errors}, worst < MAX_POROSITY_ERROR)
