"""E-CT-007: image noise scales as (tube output)^(-1/2)."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-007"
EXPECTED_EXPONENT = -0.5
EXPONENT_TOLERANCE = 0.05

MAS_VALUES = (25.0, 100.0, 400.0)
BODY_RADIUS = 0.07
ROI_RADIUS = 0.04
SEED = 0


@eval_case(
    id=EVAL_ID,
    claim="The standard deviation of image noise in uniform water scales as "
    "(tube output)^(-1/2)",
    reference="Poisson statistics: projection variance is inversely proportional to the "
    "detected photon number, and FBP is linear (Kak & Slaney 1988, section 5.2)",
    metric="Exponent of a power-law fit of noise standard deviation vs mAs "
    "(25, 100, 400 mAs) [-]",
    threshold="-0.5 +- 0.05. Fixed in advance: with about 5000 ROI pixels over four "
    "slices each standard deviation is known to a few percent, which over a factor 16 "
    "in mAs gives about +-0.02 on the exponent; 0.05 leaves room for the weak "
    "nonlinearity of the logarithm at the lowest dose.",
    seeds=(SEED,),
)
def test_e_ct_007_noise_scaling(small_cfg, small_grid):
    fractions = layered_fractions(small_grid, [("water", disc_coverage(small_grid, BODY_RADIUS))])
    roi = radial_distance(small_grid) < ROI_RADIUS
    clean_cfg = dataclasses.replace(small_cfg, photon_noise=False)
    clean = simulate_ct_fractions(fractions, small_grid, clean_cfg, SeedTree(SEED)).hu
    noise_std = []
    for mas in MAS_VALUES:
        cfg = dataclasses.replace(small_cfg, mas=mas)
        noisy = simulate_ct_fractions(fractions, small_grid, cfg, SeedTree(SEED)).hu
        noise_std.append(float((noisy - clean)[roi].std()))
    exponent = float(np.polyfit(np.log(MAS_VALUES), np.log(noise_std), 1)[0])
    values = {f"std_hu_{mas:.0f}mas": std for mas, std in zip(MAS_VALUES, noise_std)}
    report(
        EVAL_ID,
        {"exponent": exponent, **values},
        abs(exponent - EXPECTED_EXPONENT) < EXPONENT_TOLERANCE,
    )
