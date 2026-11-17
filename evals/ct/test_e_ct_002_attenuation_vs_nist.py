"""E-CT-002: basis-material mass attenuation coefficients match the NIST tables."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.ct.attenuation import mass_attenuation

EVAL_ID = "E-CT-002"
MAX_RELATIVE_DEVIATION = 0.03

ENERGIES_KEV = (30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
# mu/rho [cm^2/g] at ENERGIES_KEV from NIST X-Ray Mass Attenuation Coefficients
# (Hubbell & Seltzer), Table 4, fetched 2026-11-17:
#   https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
#   https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/tissue.html  (Tissue, Soft (ICRU-44))
#   https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html    (Bone, Cortical (ICRU-44))
#   https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html     (Air, Dry)
NIST_MU_RHO_CM2_G = {
    "air": (0.3538, 0.2485, 0.2080, 0.1875, 0.1662, 0.1541),
    "water": (0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707),
    "soft_tissue": (0.3790, 0.2688, 0.2264, 0.2048, 0.1823, 0.1693),
    "cortical_bone": (1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855),
}
# 1 cm^2/g = 0.1 m^2/kg (the eval states its reference in the unit of the NIST tables).
CM2_G_PER_M2_KG = 10.0


@eval_case(
    id=EVAL_ID,
    claim="Mass attenuation coefficients of air, water, ICRU-44 soft tissue and ICRU-44 "
    "cortical bone built from xraydb (Elam) elemental data match the NIST tables",
    reference="NIST X-Ray Mass Attenuation Coefficients, Table 4 (XCOM-based), at 30, 40, "
    "50, 60, 80 and 100 keV; https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html "
    "(and tissue.html, bone.html, air.html in the same directory)",
    metric="Maximum relative deviation over materials and energies [-]",
    threshold="< 3 %. Fixed in advance: independent photon cross-section tabulations "
    "(Elam vs XCOM) agree to 1-2 % in the diagnostic range away from absorption edges, "
    "and 3 % of bone attenuation is about 50 HU, below the HU shifts the model studies.",
    seeds=(0,),
)
def test_e_ct_002_attenuation_vs_nist():
    deviations = {}
    for material, nist in NIST_MU_RHO_CM2_G.items():
        ours = mass_attenuation(material, ENERGIES_KEV) * CM2_G_PER_M2_KG
        deviations[material] = float(np.max(np.abs(ours / np.array(nist) - 1.0)))
    worst = max(deviations.values())
    report(EVAL_ID, {"max": worst, **deviations}, worst < MAX_RELATIVE_DEVIATION)
