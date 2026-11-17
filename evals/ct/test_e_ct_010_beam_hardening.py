"""E-CT-010: beam hardening cups a water cylinder; the water correction removes it."""

from __future__ import annotations

import dataclasses

from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-010"
CUPPING_RANGE_HU = (10.0, 100.0)
CORRECTED_LIMIT_HU = 5.0
MIN_REDUCTION = 5.0

# Head-size water cylinder (180 mm), larger than the 160 mm calibration phantom.
BODY_RADIUS = 0.09
CENTER_ROI_RADIUS = 0.2 * BODY_RADIUS
EDGE_ROI = (0.8 * BODY_RADIUS, 0.9 * BODY_RADIUS)


def _cupping(cfg, grid) -> float:
    """Edge-minus-centre HU of the water cylinder [HU]."""
    fractions = layered_fractions(grid, [("water", disc_coverage(grid, BODY_RADIUS))])
    one_slice = dataclasses.replace(cfg, photon_noise=False, slab_z_from_center=(0.0, 0.002))
    hu = simulate_ct_fractions(fractions, grid, one_slice, SeedTree(0)).hu[..., 0]
    radius = radial_distance(grid)
    edge = hu[(radius > EDGE_ROI[0]) & (radius < EDGE_ROI[1])].mean()
    return float(edge - hu[radius < CENTER_ROI_RADIUS].mean())


@eval_case(
    id=EVAL_ID,
    claim="Without correction a 120 kVp scan of a head-size water cylinder shows cupping "
    "(centre darker than edge) of a few percent of mu_water; the water correction removes it",
    reference="Polyenergetic Beer-Lambert: the effective attenuation of water falls with "
    "path length, so the centre (long paths in every view) reconstructs lower than the "
    "edge (Brooks & Di Chiro 1976, Phys. Med. Biol. 21:390; Kak & Slaney 1988, ch. 4). "
    "Magnitude of 'a few percent' for 20 cm of water at 120 kVp: assumed, to verify",
    metric="Edge-minus-centre HU of a 180 mm water cylinder, uncorrected and corrected [HU]",
    threshold="Uncorrected +10 to +100 HU (1 to 10 % of mu_water, positive sign); "
    "corrected |cupping| < 5 HU and at least 5 times smaller. Fixed in advance: the "
    "band brackets 'a few percent'; 5 HU is the water tolerance of E-CT-001.",
    seeds=(0,),
)
def test_e_ct_010_beam_hardening(small_cfg, small_grid):
    uncorrected = _cupping(dataclasses.replace(small_cfg, water_correction=False), small_grid)
    corrected = _cupping(dataclasses.replace(small_cfg, water_correction=True), small_grid)
    passed = (
        CUPPING_RANGE_HU[0] <= uncorrected <= CUPPING_RANGE_HU[1]
        and abs(corrected) < CORRECTED_LIMIT_HU
        and abs(corrected) * MIN_REDUCTION < uncorrected
    )
    report(EVAL_ID, {"uncorrected_hu": uncorrected, "corrected_hu": corrected}, passed)
