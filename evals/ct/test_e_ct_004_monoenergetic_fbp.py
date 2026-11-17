"""E-CT-004: with a single-energy beam, FBP recovers the known mu of inserts."""

from __future__ import annotations

import dataclasses

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.common.units import mu_from_hu
from neurosim.ct.attenuation import basis_attenuation
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-004"
MAX_ERROR = 0.02

ENERGY_KEV = 60.0
BODY_RADIUS = 0.07
INSERT_RADIUS = 0.012
ROI_RADIUS = 0.006
# Insert material -> head-frame (x, y) of its centre [m]. The water ROI is a
# place in the body away from the inserts.
INSERTS = {
    "cortical_bone": (0.03, 0.0),
    "soft_tissue": (-0.03, 0.0),
    "air": (0.0, 0.03),
}
WATER_ROI_CENTER = (0.0, -0.03)


@eval_case(
    id=EVAL_ID,
    claim="With a monoenergetic 60 keV beam (no beam hardening), fan-beam FBP recovers "
    "the linear attenuation coefficient of bone, soft-tissue and air inserts in water",
    reference="The attenuation coefficients given to the simulation at 60 keV "
    "(E-CT-002 ties them to NIST); Beer-Lambert is exactly linear for one energy",
    metric="Maximum over materials of |mu_recon - mu_true| / max(mu_true, mu_water) [-]",
    threshold="< 2 %. Fixed in advance: the polyenergetic effects this eval is separated "
    "from are several times larger (dense bone shifts by about 10 %), and FBP with 240 "
    "views and linear detector interpolation on 2 mm pixels should be accurate to about "
    "1 % inside a 24 mm insert.",
    seeds=(0,),
)
def test_e_ct_004_monoenergetic_fbp(small_cfg, small_grid):
    cfg = dataclasses.replace(
        small_cfg, monoenergetic_kev=ENERGY_KEV, photon_noise=False, slab_z_range=(0.0, 0.002)
    )
    layers = [("water", disc_coverage(small_grid, BODY_RADIUS))]
    layers += [
        (material, disc_coverage(small_grid, INSERT_RADIUS, center))
        for material, center in INSERTS.items()
    ]
    ct = simulate_ct_fractions(layered_fractions(small_grid, layers), small_grid, cfg, SeedTree(0))
    mu = mu_from_hu(ct.hu[..., 0], ct.meta["mu_water"], ct.meta["mu_air"])

    mu_true = dict(zip(BASIS_MATERIALS, basis_attenuation([ENERGY_KEV])[:, 0]))
    rois = {**INSERTS, "water": WATER_ROI_CENTER}
    errors = {}
    for material, center in rois.items():
        roi = radial_distance(small_grid, center) < ROI_RADIUS
        scale = max(mu_true[material], mu_true["water"])
        errors[material] = float(abs(mu[roi].mean() - mu_true[material]) / scale)
    worst = max(errors.values())
    report(EVAL_ID, {"max": worst, **errors}, worst < MAX_ERROR)
