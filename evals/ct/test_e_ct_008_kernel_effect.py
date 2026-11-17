"""E-CT-008: a smoother kernel lowers noise and the HU of sub-millimetre bone."""

from __future__ import annotations

import dataclasses

from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

EVAL_ID = "E-CT-008"
MAX_RATIO = 0.95

# A fine grid so that the sub-millimetre shell is resolved in the object; the
# scanner (1.15 mm ray spacing at the isocentre in ct_small) is what blurs it.
SPACING = 0.00025
BODY_RADIUS = 0.04
SHELL_RADIUS = 0.02
SHELL_THICKNESS = 0.00075
NOISE_ROI_RADIUS = 0.012
SEED = 0


@eval_case(
    id=EVAL_ID,
    claim="Relative to the ramp kernel, the Hann kernel lowers image noise and lowers "
    "the reconstructed HU of a 0.75 mm bone shell (the mechanism by which the kernel "
    "shifts the HU-to-sound-speed relation)",
    reference="Webb et al. 2018, IEEE TUFFC 65(7):1111: bone and soft-tissue kernels "
    "give different HU for the same skull fragments (limits of agreement up to several "
    "hundred HU, growing as fragments get thinner); linear-systems theory: a low-pass "
    "window reduces both noise power and the peak of an unresolved structure",
    metric="Hann / ramp ratio of (a) noise standard deviation in water and (b) mean HU "
    "on the centre line of the shell [-]",
    threshold="Both ratios < 0.95. Fixed in advance: the Hann window removes most of the "
    "upper half of the band, so both effects should be tens of percent; 0.95 demands a "
    "clear effect without tuning to the result.",
    seeds=(SEED,),
)
def test_e_ct_008_kernel_effect(small_cfg):
    grid = Grid3D.centered((0.1, 0.1, SPACING), SPACING)
    shell = disc_coverage(grid, SHELL_RADIUS + 0.5 * SHELL_THICKNESS) - disc_coverage(
        grid, SHELL_RADIUS - 0.5 * SHELL_THICKNESS
    )
    fractions = layered_fractions(
        grid, [("water", disc_coverage(grid, BODY_RADIUS)), ("cortical_bone", shell)]
    )
    radius = radial_distance(grid)
    on_shell = abs(radius - SHELL_RADIUS) < 0.5 * SPACING
    in_water = radius < NOISE_ROI_RADIUS

    shell_hu, noise_std = {}, {}
    for kernel in ("ramp", "hann"):
        cfg = dataclasses.replace(small_cfg, kernel=kernel, slab_z_from_center=None)
        clean_cfg = dataclasses.replace(cfg, photon_noise=False)
        clean = simulate_ct_fractions(fractions, grid, clean_cfg, SeedTree(SEED)).hu[..., 0]
        noisy = simulate_ct_fractions(fractions, grid, cfg, SeedTree(SEED)).hu[..., 0]
        shell_hu[kernel] = float(clean[on_shell].mean())
        noise_std[kernel] = float((noisy - clean)[in_water].std())
    contrast_ratio = shell_hu["hann"] / shell_hu["ramp"]
    noise_ratio = noise_std["hann"] / noise_std["ramp"]
    report(
        EVAL_ID,
        {
            "noise_ratio": noise_ratio,
            "shell_hu_ratio": contrast_ratio,
            "shell_hu_ramp": shell_hu["ramp"],
            "shell_hu_hann": shell_hu["hann"],
            "noise_std_ramp_hu": noise_std["ramp"],
            "noise_std_hann_hu": noise_std["hann"],
        },
        noise_ratio < MAX_RATIO and contrast_ratio < MAX_RATIO,
    )
