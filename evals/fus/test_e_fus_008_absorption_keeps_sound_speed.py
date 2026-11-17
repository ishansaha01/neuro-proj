"""E-FUS-008: absorption does not move the k-Wave phase speed off the sound-speed map."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import ALPHA_POWER_DEFAULT, C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-008"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
SLAB_SPEED = 2800.0  # bone-like sound speed [m/s]
SLAB_ALPHA0 = 8.0  # bone-like absorption prefactor [dB/(cm MHz^y)]
SLAB_FIRST_ROW, SLAB_ROWS = 20, 20  # the slab fills grid rows 20..39: 6 mm thick
MAX_RELATIVE_ERROR = 0.02


@eval_case(
    id=EVAL_ID,
    claim="In the k-Wave backend with its default settings, an absorbing bone-like slab delays "
    "the wave by its sound_speed map, as the same slab without absorption does: the absorption "
    "model does not shift the phase speed",
    reference="Analytic: dt = d * (1/c_slab - 1/c_background), with c_slab the value of the "
    "sound_speed map",
    metric="|group delay - dt| / |dt| on the slab normal [dimensionless]; group delay = "
    "least-squares slope of the unwrapped phase of B against angular frequency, 0.3-0.7 MHz",
    threshold="< 0.02 (fixed in advance: the same 2 % of dt as E-FUS-002, of which this is the "
    "absorbing counterpart at the project's default exponent ALPHA_POWER_DEFAULT; the "
    "dispersion term of k-Wave's power-law absorption, reported alongside, fails it by an "
    "order of magnitude)",
)
def test_e_fus_008_absorption_keeps_sound_speed():
    grid = Grid2D.from_extent((-9.6e-3, 9.6e-3), (0.0, 30.0e-3), SPACING)
    background = Material(C_REF, RHO_WATER)
    slab = dataclasses.replace(background.impedance_matched(SLAB_SPEED), attenuation=SLAB_ALPHA0)
    thickness = SLAB_ROWS * SPACING
    medium = slab_medium(
        grid, background, slab, (SLAB_FIRST_ROW - 0.5) * SPACING, thickness, ALPHA_POWER_DEFAULT
    )
    transducer = Transducer(
        n_elements=1,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (27.0e-3, 27.0e-3), 3.0e-3)
    analytic = thickness * (1.0 / SLAB_SPEED - 1.0 / C_REF)  # [s]

    def relative_error(settings: KWaveConfig) -> float:
        bank = compute_bank(
            medium, transducer, roi, FREQS, BackendConfig(backend="kwave", kwave=settings)
        )
        phase = np.unwrap(np.angle(bank.B[:, 0, 0, 0]))
        group_delay = np.polyfit(2.0 * np.pi * FREQS, phase, 1)[0]  # [s]
        return float(abs(group_delay - analytic) / abs(analytic))

    # cfl = 0.1: the phase error of the k-space scheme in the background scales with cfl^2.
    settings = KWaveConfig(cfl=0.1, reverberation_tail=5.0e-6)
    error = relative_error(settings)
    # Diagnostic only, not part of the verdict: the same run with k-Wave's dispersion term.
    with_dispersion = relative_error(dataclasses.replace(settings, model_dispersion=True))

    report(
        EVAL_ID,
        {
            "relative_error": error,
            "relative_error_with_dispersion_term": with_dispersion,
            "analytic_delay_s": analytic,
        },
        passed=error < MAX_RELATIVE_ERROR,
    )
