"""E-FUS-003: one-way attenuation through an absorbing slab follows the power law."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-003"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
ALPHA0 = 8.0  # absorption prefactor of the slab [dB/(cm MHz^y)], bone-like
ALPHA_POWER = 1.5
SLAB_FIRST_ROW, SLAB_ROWS = 20, 30  # the slab fills grid rows 20..49: 9 mm thick
MAX_RELATIVE_ERROR = {"ray": 0.01, "kwave": 0.05}


@eval_case(
    id=EVAL_ID,
    claim="The one-way attenuation through an impedance-matched absorbing slab, for the element "
    "and point on the slab normal, follows the power law at every frequency in the ray and the "
    "k-Wave backend",
    reference="Analytic: loss = alpha0 * (f / 1 MHz)^y * d [dB], alpha0 in dB/(cm MHz^y), d in cm",
    metric="Largest |(-20 log10 |B|) - loss| / loss over 9 frequencies from 0.3 to 0.7 MHz, per "
    "backend [dimensionless]",
    threshold="ray < 0.01 and k-Wave < 0.05 (fixed in advance: the ray integral is exact up to "
    "sampling, so 1 %; k-Wave's fractional-Laplacian absorption follows the power law only "
    "approximately and with dispersion, and 5 % of a 10 dB skull loss is 0.5 dB, well inside "
    "the spread of published skull attenuation values). The backends run with their default "
    "settings; the k-Wave default has no dispersion term (KWaveConfig.model_dispersion)",
)
def test_e_fus_003_slab_attenuation():
    grid = Grid2D.from_extent((-9.6e-3, 9.6e-3), (0.0, 30.0e-3), SPACING)
    background = Material(C_REF, RHO_WATER)
    # Same sound speed and density as the background: absorption is the only effect.
    slab = Material(C_REF, RHO_WATER, attenuation=ALPHA0)
    thickness = SLAB_ROWS * SPACING
    medium = slab_medium(
        grid, background, slab, (SLAB_FIRST_ROW - 0.5) * SPACING, thickness, ALPHA_POWER
    )
    transducer = Transducer(
        n_elements=1,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (24.0e-3, 27.0e-3), 3.0e-3)
    analytic_db = ALPHA0 * (FREQS / 1.0e6) ** ALPHA_POWER * (thickness / 1.0e-2)  # (nf,)

    def relative_error(cfg: BackendConfig) -> float:
        bank = compute_bank(medium, transducer, roi, FREQS, cfg)
        loss_db = -20.0 * np.log10(np.abs(bank.B[:, 0, :, 0]))  # (nf, nz_roi)
        return float(np.max(np.abs(loss_db / analytic_db[:, None] - 1.0)))

    kwave = KWaveConfig(cfl=0.3, reverberation_tail=5.0e-6)
    errors = {
        "ray": relative_error(BackendConfig(backend="ray")),
        "kwave": relative_error(BackendConfig(backend="kwave", kwave=kwave)),
    }
    # Diagnostic only, not part of the verdict: the same run with k-Wave's dispersion term.
    with_dispersion = relative_error(
        BackendConfig(backend="kwave", kwave=dataclasses.replace(kwave, model_dispersion=True))
    )

    report(
        EVAL_ID,
        {
            "ray_relative_error": errors["ray"],
            "kwave_relative_error": errors["kwave"],
            "kwave_relative_error_with_dispersion_term": with_dispersion,
            "analytic_loss_db_min": float(analytic_db.min()),
            "analytic_loss_db_max": float(analytic_db.max()),
        },
        passed=all(errors[name] < MAX_RELATIVE_ERROR[name] for name in errors),
        note="With k-Wave's dispersion term switched on (not the default) the loss falls short "
        "of the power law by an amount that grows with frequency. This is a property of "
        "k-Wave's lossy wave equation, not of the wiring: its dispersion relation k^2 = "
        "(w/c0)^2 + tau (-i w) k^y + eta k^(y+1) gives Im(k) = 0.964, 0.954, 0.946 times "
        "alpha0 w^y at 0.3, 0.5, 0.7 MHz for this slab (8 dB/(cm MHz^1.5)); the power law "
        "holds only to first order in the absorption.",
    )
