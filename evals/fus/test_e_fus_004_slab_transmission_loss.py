"""E-FUS-004: transmission loss through a non-absorbing bone-like slab at normal incidence."""

from __future__ import annotations

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.types import GreensBank, Transducer

EVAL_ID = "E-FUS-004"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.3e6, 0.7e6, 21)  # [Hz]; 20 kHz steps give a 50 us unambiguous delay range
BONE = Material(sound_speed=2800.0, density=1900.0)  # bone-like, lossless
SLAB_FIRST_ROW, SLAB_ROWS = 20, 40  # the slab fills grid rows 20..59: 12 mm thick
ARRIVAL_SEARCH = 2.0e-6  # half-width of the delay window searched for the first arrival [s]
MAX_RELATIVE_ERROR = {"ray": 0.01, "kwave": 0.05}


def first_arrival_amplitude(bank: GreensBank, expected_delay: float) -> float:
    """Amplitude of the first arrival in the band-limited impulse response of B (pulse compression).

    The Hann-weighted inverse Fourier sum of B over the band has one peak per
    arrival; the first is searched within `ARRIVAL_SEARCH` of `expected_delay` [s].
    """
    window = np.hanning(FREQS.size + 2)[1:-1]
    delays = expected_delay + np.linspace(-ARRIVAL_SEARCH, ARRIVAL_SEARCH, 401)
    kernel = np.exp(-2j * np.pi * np.outer(delays, FREQS))  # undo exp(+i w tau)
    response = kernel @ (window * bank.B[:, 0, 0, 0]) / window.sum()
    return float(np.max(np.abs(response)))


@eval_case(
    id=EVAL_ID,
    claim="A one-way path through a non-absorbing bone-like slab at normal incidence loses the "
    "product of the two interface transmission coefficients, in the ray backend and in the "
    "first arrival of the k-Wave backend",
    reference="Analytic: T = [2 Z2 / (Z1 + Z2)] * [2 Z1 / (Z1 + Z2)] = 4 Z1 Z2 / (Z1 + Z2)^2, "
    "Z = rho c",
    metric="|A_slab / A_matched - T| / T per backend [dimensionless], where A is the amplitude of "
    "B on the slab normal (k-Wave: of its first arrival, by pulse compression over 0.3-0.7 MHz) "
    "and 'matched' is a slab of the same sound speed whose density makes Z2 = Z1, which removes "
    "everything but the interface loss (delay, spreading)",
    threshold="ray < 0.01 and k-Wave < 0.05 (fixed in advance: the ray model is the analytic "
    "product up to float32 medium maps; for k-Wave the 12 mm slab puts the first internal echo "
    "8.6 us after the first arrival, beyond the 5 us half-width of the Hann-weighted 0.4 MHz "
    "band, leaving about 1 % of sidelobe leakage, and 5 % is 0.4 dB)",
)
def test_e_fus_004_slab_transmission_loss():
    grid = Grid2D.from_extent((-9.6e-3, 9.6e-3), (0.0, 33.0e-3), SPACING)
    background = Material(C_REF, RHO_WATER)
    thickness = SLAB_ROWS * SPACING
    z_top = (SLAB_FIRST_ROW - 0.5) * SPACING
    media = {
        "slab": slab_medium(grid, background, BONE, z_top, thickness, alpha_power=1.5),
        "matched": slab_medium(
            grid,
            background,
            background.impedance_matched(BONE.sound_speed),
            z_top,
            thickness,
            alpha_power=1.5,
        ),
    }
    transducer = Transducer(
        n_elements=1,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, 0.0), (30.0e-3, 30.0e-3), 3.0e-3)
    z_background = background.sound_speed * background.density
    z_bone = BONE.sound_speed * BONE.density
    analytic = 4.0 * z_background * z_bone / (z_background + z_bone) ** 2
    slab_delay = thickness * (1.0 / BONE.sound_speed - 1.0 / C_REF)  # [s]

    ray = {
        name: compute_bank(medium, transducer, roi, FREQS, BackendConfig(backend="ray"))
        for name, medium in media.items()
    }
    ray_ratio = float(np.mean(np.abs(ray["slab"].B / ray["matched"].B)))

    # The tail holds two round trips of the slab; later echoes do not touch the first arrival.
    cfg = BackendConfig(backend="kwave", kwave=KWaveConfig(cfl=0.3, reverberation_tail=20.0e-6))
    amplitude = {
        name: first_arrival_amplitude(compute_bank(medium, transducer, roi, FREQS, cfg), slab_delay)
        for name, medium in media.items()
    }
    kwave_ratio = amplitude["slab"] / amplitude["matched"]

    errors = {"ray": abs(ray_ratio / analytic - 1.0), "kwave": abs(kwave_ratio / analytic - 1.0)}
    report(
        EVAL_ID,
        {
            "ray_relative_error": errors["ray"],
            "kwave_relative_error": errors["kwave"],
            "analytic_transmission": analytic,
            "ray_transmission": ray_ratio,
            "kwave_transmission": kwave_ratio,
            "kwave_matched_slab_amplitude": amplitude["matched"],
        },
        passed=all(errors[name] < MAX_RELATIVE_ERROR[name] for name in errors),
    )
