"""E-CT-009: same seed gives identical HU; a slice's noise ignores the other slices."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.ct.phantoms import disc_coverage, layered_fractions
from neurosim.ct.simulate import simulate_ct_fractions, slice_indices

EVAL_ID = "E-CT-009"
SEED = 7
OTHER_SEED = 8
SLAB_A = (-0.004, 0.004)  # slices 2..5 of the small grid
SLAB_B = (0.0, 0.008)  # slices 4..7


@eval_case(
    id=EVAL_ID,
    claim="A CT scan is a deterministic function of (volume, config, seed), and the "
    "photon noise of a slice does not depend on which other slices are scanned",
    reference="ROADMAP D-004: a stream depends only on (root seed, name, sub-index); "
    "the noise of slice iz uses ('ct_photon_noise', iz)",
    metric="Maximum absolute HU difference (a) between two runs with the same seed and "
    "(b) on the shared slices of two overlapping slabs; and (c) between two seeds [HU]",
    threshold="(a) = 0 and (b) = 0 exactly, (c) > 0. Fixed in advance: determinism is "
    "bitwise or it is not determinism.",
    seeds=(SEED, OTHER_SEED),
)
def test_e_ct_009_determinism(small_cfg, small_grid):
    fractions = layered_fractions(small_grid, [("water", disc_coverage(small_grid, 0.06))])
    cfg_a = dataclasses.replace(small_cfg, slab_z_range=SLAB_A)
    cfg_b = dataclasses.replace(small_cfg, slab_z_range=SLAB_B)
    first = simulate_ct_fractions(fractions, small_grid, cfg_a, SeedTree(SEED)).hu
    again = simulate_ct_fractions(fractions, small_grid, cfg_a, SeedTree(SEED)).hu
    other_slab = simulate_ct_fractions(fractions, small_grid, cfg_b, SeedTree(SEED)).hu
    other_seed = simulate_ct_fractions(fractions, small_grid, cfg_a, SeedTree(OTHER_SEED)).hu

    slices_a, slices_b = slice_indices(small_grid, cfg_a), slice_indices(small_grid, cfg_b)
    shared = sorted(set(slices_a) & set(slices_b))
    assert shared, "the two slabs must overlap"
    in_a = [iz - slices_a.start for iz in shared]
    in_b = [iz - slices_b.start for iz in shared]

    repeat_difference = float(np.abs(first - again).max())
    shared_difference = float(np.abs(first[..., in_a] - other_slab[..., in_b]).max())
    seed_difference = float(np.abs(first - other_seed).max())
    report(
        EVAL_ID,
        {
            "same_seed": repeat_difference,
            "shared_slices": shared_difference,
            "different_seed": seed_difference,
        },
        repeat_difference == 0.0 and shared_difference == 0.0 and seed_difference > 0.0,
    )
