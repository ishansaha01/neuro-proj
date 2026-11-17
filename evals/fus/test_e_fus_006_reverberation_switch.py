"""E-FUS-006: matching the slab impedance removes the static echo (reverberation switch)."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-006"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
BONE = Material(sound_speed=2800.0, density=1900.0)  # bone-like, lossless
SLAB_FIRST_ROW, SLAB_ROWS = 20, 20  # the slab fills grid rows 20..39: 6 mm thick
MAX_ECHO_RATIO_DB = -20.0


@eval_case(
    id=EVAL_ID,
    claim="An impedance-matched slab (bone sound speed, density lowered so that rho c equals the "
    "background's) returns far less static echo energy to the array than the unmatched slab",
    reference="Normal-incidence reflection coefficient R = (Z2 - Z1) / (Z2 + Z1): zero for a "
    "matched slab, 0.55 for the bone-like slab",
    metric="10 log10 of the ratio of sum |static_fmc|^2 (all frequencies and element pairs), "
    "matched over unmatched [dB]",
    threshold="< -20 dB (fixed in advance: the switch must remove at least 99 % of the echo "
    "energy; a density-matched slab still reflects off the normal, "
    "R ~ theta^2 ((c2/c1)^2 - 1) / 4, about 0.02 at the 10 degree specular angles of this "
    "3-element array against 0.55, i.e. -28 dB, so -20 dB leaves room for that residue but not "
    "for a failed switch)",
)
def test_e_fus_006_reverberation_switch():
    grid = Grid2D.from_extent((-12.6e-3, 12.6e-3), (0.0, 25.2e-3), SPACING)
    background = Material(C_REF, RHO_WATER)
    transducer = Transducer(
        n_elements=3,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (18.0e-3, 18.0e-3), 3.0e-3)
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.3, reverberation_tail=15.0e-6))

    def echo_energy(slab: Material) -> float:
        medium = slab_medium(
            grid,
            background,
            slab,
            (SLAB_FIRST_ROW - 0.5) * SPACING,
            SLAB_ROWS * SPACING,
            alpha_power=1.5,
        )
        bank = compute_bank(medium, transducer, roi, FREQS, cfg)
        return float(np.sum(np.abs(bank.static_fmc) ** 2))

    unmatched = echo_energy(BONE)
    matched = echo_energy(background.impedance_matched(BONE.sound_speed))
    ratio_db = 10.0 * np.log10(matched / unmatched)
    report(
        EVAL_ID,
        {
            "echo_energy_ratio_db": float(ratio_db),
            "unmatched_echo_energy": unmatched,
            "matched_echo_energy": matched,
        },
        passed=ratio_db < MAX_ECHO_RATIO_DB,
    )
