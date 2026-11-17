"""E-FUS-001: the k-Wave bank of a homogeneous medium equals the analytic Green's-function ratio."""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1

from neurosim.common.constants import C_REF, C_WATER, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, homogeneous_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-001"
SPACING = 0.3e-3  # medium grid spacing [m]: 7 points per wavelength at 0.7 MHz in water
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
MAX_RELATIVE_ERROR = 0.01


@eval_case(
    id=EVAL_ID,
    claim="The k-Wave bank of a homogeneous lossless medium with sound speed c1 != c_ref equals "
    "the ratio of the 2D free-space Green's functions, in magnitude and phase",
    reference="Analytic: B = H0^(1)(k1 r) / H0^(1)(k_ref r) for a line source "
    "(Morse & Ingard 1968, eq. 7.3.14), exp(-i w t) convention",
    metric="Maximum of |B / B_analytic - 1| over frequencies, elements and ROI points "
    "[dimensionless]",
    threshold="< 0.01 (fixed in advance: the pseudospectral solver is spectrally accurate in a "
    "homogeneous medium, so only boundary-layer reflections and the DFT remain; 1 % in complex "
    "amplitude is 0.01 rad of phase, far below the lambda/14 budget of aberration correction, "
    "and any error of sign convention, grid orientation or source scaling is of order 1)",
)
def test_e_fus_001_kwave_homogeneous_greens_ratio():
    grid = Grid2D.from_extent((-12.0e-3, 12.0e-3), (0.0, 30.0e-3), SPACING)
    medium = homogeneous_medium(grid, Material(C_WATER, RHO_WATER), alpha_power=1.5)
    # Elements one grid node wide, so that each is the line source of the analytic solution.
    transducer = Transducer(
        n_elements=2,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((-9.0e-3, 9.0e-3), (6.0e-3, 27.0e-3), 3.0e-3)
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.3, reverberation_tail=5.0e-6))

    bank = compute_bank(medium, transducer, roi, FREQS, cfg)

    points = roi.points()
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )  # (n_elements, nz_roi, nx_roi) [m]
    wavenumber = 2.0 * np.pi * FREQS[:, None, None, None]
    analytic = hankel1(0, wavenumber / C_WATER * distance) / hankel1(
        0, wavenumber / C_REF * distance
    )
    error = float(np.max(np.abs(bank.B / analytic - 1.0)))

    assert bank.meta["roi_snap_distance_m"] < 1e-9
    report(
        EVAL_ID,
        {
            "max_relative_error": error,
            "max_phase_error_rad": float(np.max(np.abs(np.angle(bank.B / analytic)))),
            "max_magnitude_error": float(np.max(np.abs(np.abs(bank.B / analytic) - 1.0))),
        },
        passed=error < MAX_RELATIVE_ERROR,
    )
