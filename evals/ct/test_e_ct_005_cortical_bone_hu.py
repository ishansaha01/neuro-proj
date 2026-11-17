"""E-CT-005: dense cortical bone at 120 kVp has a HU in the published range."""

from __future__ import annotations

from neurosim.common.evalkit import eval_case, report

EVAL_ID = "E-CT-005"
HU_RANGE = (1300.0, 2400.0)


@eval_case(
    id=EVAL_ID,
    claim="Fully dense ICRU-44 cortical bone (1.92 g/cm^3) reconstructs to a HU in the "
    "published range for dense cortical bone at 120 kVp",
    reference="Upper limits used for skull cortical bone in CT-based acoustic models: "
    "2000 HU (Yaakub et al. 2023, arXiv:2209.12031) and 2400 HU (Marsac et al. 2017, as "
    "quoted in arXiv:1805.10007); the general cortical range tops out near 1900 HU "
    "(Wikipedia 'Hounsfield scale': +500 to +1900). Lower limit from NIST tables: "
    "1000 (mu_bone / mu_water - 1) = 1330 HU at 80 keV, above the effective energy of a "
    "120 kVp head scan (1935 HU at 60 keV).",
    metric="Mean HU in a 20 mm cortical-bone rod inside a 140 mm water cylinder [HU]",
    threshold="1300 to 2400 HU. Fixed in advance: pore-free bone must sit at the top of "
    "the published cortical range; below 1300 HU the beam would be harder than any "
    "120 kVp spectrum, above 2400 HU softer than models of real skulls allow.",
    seeds=(0,),
)
def test_e_ct_005_cortical_bone_hu(small_cfg, bone_rod_hu):
    bone_hu = bone_rod_hu(small_cfg)
    report(EVAL_ID, bone_hu, HU_RANGE[0] <= bone_hu <= HU_RANGE[1])
