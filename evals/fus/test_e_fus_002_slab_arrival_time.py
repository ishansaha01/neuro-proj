"""E-FUS-002: the extra arrival time through a flat slab matches the analytic delay."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank, ray_travel_times
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-002"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
SLAB_SPEED = 2800.0  # bone-like sound speed [m/s]
SLAB_FIRST_ROW, SLAB_ROWS = 20, 20  # the slab fills grid rows 20..39: 6 mm thick
MAX_RELATIVE_ERROR = 0.02


@eval_case(
    id=EVAL_ID,
    claim="The extra one-way arrival time through a homogeneous flat slab, for the element and "
    "point on the slab normal, matches the analytic delay in the ray and the k-Wave backend",
    reference="Analytic: dt = d * (1/c_slab - 1/c_background)",
    metric="Largest |delay - dt| / |dt| over the two backends and the ROI points [dimensionless]; "
    "ray: difference of ray_travel_times with and without the slab; k-Wave: group delay, the "
    "least-squares slope of the unwrapped phase of B against angular frequency",
    threshold="< 0.02 (fixed in advance: the 2 % of dt of the CLAUDE.md example; for the 6 mm "
    "bone-like slab it is 28 ns, 1/70 of the 2 us period at the centre frequency)",
)
def test_e_fus_002_slab_arrival_time():
    grid = Grid2D.from_extent((-9.6e-3, 9.6e-3), (0.0, 30.0e-3), SPACING)
    background = Material(C_REF, RHO_WATER)
    # Impedance-matched slab: a pure delay, with no reverberation to bend the phase of B.
    slab = background.impedance_matched(SLAB_SPEED)
    thickness = SLAB_ROWS * SPACING
    medium = slab_medium(
        grid, background, slab, (SLAB_FIRST_ROW - 0.5) * SPACING, thickness, alpha_power=1.5
    )
    transducer = Transducer(
        n_elements=1,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (21.0e-3, 27.0e-3), 3.0e-3)
    analytic = thickness * (1.0 / SLAB_SPEED - 1.0 / C_REF)  # [s], negative: the slab is fast

    water = homogeneous_medium(grid, background, alpha_power=1.5)
    ray_delay = ray_travel_times(medium, transducer, roi.points()) - ray_travel_times(
        water, transducer, roi.points()
    )
    ray_error = float(np.max(np.abs(ray_delay - analytic)) / abs(analytic))

    # cfl = 0.1: the phase error of the k-space scheme in the background scales with cfl^2.
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.1, reverberation_tail=5.0e-6))
    bank = compute_bank(medium, transducer, roi, FREQS, cfg)
    phase = np.unwrap(np.angle(bank.B[:, 0, :, 0]), axis=0)  # (nf, nz_roi)
    group_delay = np.polyfit(2.0 * np.pi * FREQS, phase, 1)[0]  # [s], one per ROI point
    kwave_error = float(np.max(np.abs(group_delay - analytic)) / abs(analytic))

    report(
        EVAL_ID,
        {
            "ray_relative_error": ray_error,
            "kwave_relative_error": kwave_error,
            "analytic_delay_s": analytic,
            "kwave_phase_error_bound_rad": bank.meta["cost_estimate"]["phase_error_bound_rad"],
        },
        passed=max(ray_error, kwave_error) < MAX_RELATIVE_ERROR,
    )
