"""E-FUS-007: the k-Wave backend is reciprocal between array elements."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-007"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
BONE = Material(sound_speed=2800.0, density=1900.0, attenuation=4.0)  # bone-like, absorbing
MAX_ASYMMETRY = 0.01


@eval_case(
    id=EVAL_ID,
    claim="In the k-Wave backend the transfer function from element a to the position of element b "
    "equals that from b to a, in a heterogeneous, absorbing, laterally asymmetric medium",
    reference="Acoustic reciprocity: the wave operator div(grad p / rho) + w^2 p / (rho c^2) of a "
    "medium at rest is symmetric, so source and receiver are interchangeable; the medium at "
    "the array face is uniform, so no density factor appears",
    metric="max |static_fmc[f, a, b] - static_fmc[f, b, a]| / max |static_fmc| over frequencies "
    "and element pairs [dimensionless]; static_fmc is the medium-minus-reference transfer "
    "function between elements and the reference part is symmetric by construction",
    threshold="< 0.01 (fixed in advance: reciprocity is exact for the wave equation; the "
    "discrete scheme, the absorbing layer and the finite record may break it slightly, and 1 % "
    "of the echo amplitude is -40 dB, below every other error of the bank)",
)
def test_e_fus_007_kwave_reciprocity():
    grid = Grid2D.from_extent((-12.6e-3, 12.6e-3), (0.0, 25.2e-3), SPACING)
    x = grid.axes()[1]
    # A slab whose thickness doubles to one side of the array: no mirror symmetry.
    thickness = 3.0e-3 + 3.0e-3 / (1.0 + np.exp(-(x - 1.5e-3) / 1.0e-3))
    medium = slab_medium(
        grid, Material(C_REF, RHO_WATER), BONE, 13.5 * SPACING, thickness, alpha_power=1.5
    )
    transducer = Transducer(
        n_elements=3,
        pitch=6 * SPACING,
        element_width=3 * SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (18.0e-3, 18.0e-3), 3.0e-3)
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.3, reverberation_tail=15.0e-6))

    fmc = compute_bank(medium, transducer, roi, FREQS, cfg).static_fmc
    scale = float(np.max(np.abs(fmc)))
    asymmetry = float(np.max(np.abs(fmc - fmc.transpose(0, 2, 1))) / scale)
    # The medium must actually distinguish the two outer elements, or the test is empty.
    mirror_contrast = float(np.max(np.abs(fmc[:, 0, 0] - fmc[:, 2, 2])) / scale)
    assert mirror_contrast > 10.0 * MAX_ASYMMETRY
    report(
        EVAL_ID,
        {"reciprocity_error": asymmetry, "mirror_contrast": mirror_contrast},
        passed=asymmetry < MAX_ASYMMETRY,
    )
