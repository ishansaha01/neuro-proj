"""E-CT-006: bone HU falls as the tube voltage rises (Webb et al. 2018)."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import eval_case, report

EVAL_ID = "E-CT-006"
KVPS = (80.0, 100.0, 120.0, 140.0)
RATIO_RANGE = (1.2, 2.0)


@eval_case(
    id=EVAL_ID,
    claim="The HU of cortical bone decreases monotonically from 80 to 140 kVp, by the "
    "factor implied by the velocity-vs-HU slopes of Webb et al. 2018",
    reference="Webb et al. 2018, IEEE TUFFC 65(7):1111, Tables I-II: the slope of "
    "velocity vs HU (inversely proportional to the HU of a given bone) rises from 0.57 to "
    "0.81 m/s/HU between 80 and 140 kVp (GE, bone kernel; factor 1.42) and from 0.57 to "
    "0.87 between 70 and 140 kVp (Siemens, bone kernel; factor 1.53)",
    metric="HU of a cortical-bone rod at 80, 100, 120, 140 kVp and the ratio HU(80)/HU(140) [-]",
    threshold="Strictly decreasing with kVp and ratio in 1.2 to 2.0. Fixed in advance: "
    "the measured factors are 1.4 to 1.5; the band allows for our different filtration, "
    "detector and bone thickness, and for the intercept changes in Webb's fits, while "
    "still rejecting no dependence (1.0) and a monoenergetic-like swing (above 2).",
    seeds=(0,),
)
def test_e_ct_006_energy_dependence(small_cfg, bone_rod_hu):
    hu = [bone_rod_hu(dataclasses.replace(small_cfg, kvp=kvp)) for kvp in KVPS]
    ratio = hu[0] / hu[-1]
    decreasing = bool(np.all(np.diff(hu) < 0.0))
    values = {f"hu_{kvp:.0f}kvp": value for kvp, value in zip(KVPS, hu)}
    report(
        EVAL_ID,
        {"ratio_80_to_140": ratio, **values},
        decreasing and RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1],
    )
