"""E-CT-001: water reconstructs to 0 HU and air to -1000 HU."""

from __future__ import annotations

import dataclasses

from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-001"
# A 140 mm water cylinder: deliberately not the 160 mm calibration phantom.
CYLINDER_RADIUS = 0.07
WATER_ROI_RADIUS = 0.05
AIR_ROI = (0.08, 0.105)
WATER_TOLERANCE_HU = 5.0
AIR_TOLERANCE_HU = 10.0


@eval_case(
    id=EVAL_ID,
    claim="A water cylinder reconstructs to 0 HU and the surrounding air to -1000 HU "
    "(noise off, water correction on)",
    reference="Definition of the Hounsfield scale: water = 0 HU, air = -1000 HU",
    metric="Mean HU in a water ROI and in an air ROI of a 140 mm water cylinder [HU]",
    threshold="|water| < 5 HU and |air + 1000| < 10 HU. Fixed in advance: 5 HU is the "
    "usual clinical water-calibration tolerance (ACR CT accreditation accepts 0 +- 5 to "
    "7 HU; assumed, to verify); air gets 10 HU because the ROI sits next to the "
    "cylinder edge on a 2 mm grid; a noise-free simulator has no excuse for more.",
    seeds=(0,),
)
def test_e_ct_001_hu_calibration(small_cfg, small_grid):
    cfg = dataclasses.replace(small_cfg, photon_noise=False)
    fractions = layered_fractions(
        small_grid, [("water", disc_coverage(small_grid, CYLINDER_RADIUS))]
    )
    hu = simulate_ct_fractions(fractions, small_grid, cfg, SeedTree(0)).hu
    radius = radial_distance(small_grid)
    water_hu = float(hu[radius < WATER_ROI_RADIUS].mean())
    air_hu = float(hu[(radius > AIR_ROI[0]) & (radius < AIR_ROI[1])].mean())
    passed = abs(water_hu) < WATER_TOLERANCE_HU and abs(air_hu + 1000.0) < AIR_TOLERANCE_HU
    report(EVAL_ID, {"water_hu": water_hu, "air_hu": air_hu}, passed)
