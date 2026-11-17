"""E-FUS-005: ray and k-Wave banks agree in phase on a smooth, weakly heterogeneous phantom."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-005"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
LAYER = Material(sound_speed=2200.0, density=1300.0)  # between soft tissue and bone
BASE_THICKNESS, BUMP_HEIGHT, BUMP_WIDTH = 4.2e-3, 2.1e-3, 6.0e-3  # [m]
MAX_RMS_PHASE = 2.0 * np.pi / 14.0  # [rad]


@eval_case(
    id=EVAL_ID,
    claim="On a smooth, weakly heterogeneous phantom (a layer with a gentle thickness bump) the "
    "straight-ray bank reproduces the phase of the full-wave bank, i.e. its aberration delays",
    reference="Full-wave solution (k-Wave) of the same medium; the two backends share no code "
    "beyond the input checks",
    metric="RMS over frequencies, elements and ROI points of arg(B_ray / B_kwave) [rad]",
    threshold="< 2 pi / 14 = 0.449 rad (fixed in advance: the Marechal criterion, an RMS "
    "wavefront error of lambda/14 keeps the focal intensity above 80 %; the aberration "
    "phase of the phantom itself is several times larger and is reported alongside)",
)
def test_e_fus_005_ray_kwave_phase_agreement():
    grid = Grid2D.from_extent((-15.0e-3, 15.0e-3), (0.0, 30.0e-3), SPACING)
    x = grid.axes()[1]
    thickness = BASE_THICKNESS + BUMP_HEIGHT * np.exp(-0.5 * ((x - 2.4e-3) / BUMP_WIDTH) ** 2)
    medium = slab_medium(
        grid, Material(C_REF, RHO_WATER), LAYER, 19.5 * SPACING, thickness, alpha_power=1.5
    )
    transducer = Transducer(
        n_elements=5,
        pitch=10 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((-6.0e-3, 6.0e-3), (21.0e-3, 27.0e-3), 6.0e-3)

    ray = compute_bank(medium, transducer, roi, FREQS, BackendConfig(backend="ray"))
    # cfl = 0.1: the phase error of the k-space scheme in the background scales with cfl^2.
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.1, reverberation_tail=5.0e-6))
    full = compute_bank(medium, transducer, roi, FREQS, cfg)

    difference = np.angle(ray.B / full.B)
    rms = float(np.sqrt(np.mean(difference**2)))
    report(
        EVAL_ID,
        {
            "rms_phase_difference_rad": rms,
            "max_phase_difference_rad": float(np.max(np.abs(difference))),
            # Unwrapped by construction: the ray phase is w * (excess delay).
            "rms_aberration_phase_rad": float(
                np.sqrt(np.mean(np.unwrap(np.angle(ray.B), axis=0) ** 2))
            ),
            "kwave_phase_error_bound_rad": full.meta["cost_estimate"]["phase_error_bound_rad"],
        },
        passed=rms < MAX_RMS_PHASE,
    )
