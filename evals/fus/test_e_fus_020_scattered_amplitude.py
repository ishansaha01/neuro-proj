"""E-FUS-020: `static_fmc` and the bank share one normalisation, density factor included.

`GreensBank.static_fmc` is documented as the echo at the array "per unit source
spectrum (same normalisation as G_medium)", and the backends state that the
same B serves transmit and receive "up to the constant density ratio between
the array face and the ROI". Both statements fix the absolute size of an echo,
which E-FUS-019 leaves free. Here it is checked against an analytic value.

Reference. `G` solves `rho div(grad G / rho) + k^2 G = -delta(r - r_e)`. Raise
the sound speed at one node from `c` to `c'`: `k^2` changes there by
`dk2 = w^2 (1/c'^2 - 1/c^2)` over the node area `A = dx^2`. The first term of
the Lippmann-Schwinger series (first Born approximation) for the echo at
element `j` when element `i` fires is

    S[f, i, j] = dk2 A G(r_j | r_s) G(r_s | r_i)
               = dk2 A [rho(array face) / rho(r_s)] g[i] g[j],   g[i] = G(r_s | r_i),

because `G(r | r') rho(r')` is symmetric (the operator `div(grad . / rho) +
k^2 / rho` is self-adjoint). `g` is what the bank stores (`G = B G_ref`).

Phantom. The array sits directly on a bone-like layer and the scatterer in
brain-like tissue behind it. This is not anatomy: it makes the two factors a
wrong normalisation would miss both far from 1. The density ratio is 1.83, and
the sound speed at the array face is 1.82 times the reference speed (the
backend scales the drive by that ratio to inject the same mass source as in the
reference run). The direct wave between the elements also differs from the
reference there and is part of `static_fmc`; it cancels in the difference of
the two runs.

Expected deviation from the first Born term at a contrast of 2 %: the node
re-scatters its own field, a relative correction `dk2 A G(r_s | r_s)` with
`|G(r_s | r_s)|` about 0.35 on this grid ((1/2 pi) ln(pi / (k dx)) + i/4), i.e.
1 % at 0.7 MHz; and the time stepping of the k-space scheme changes the
effective `dk2` in a medium slower than the fastest one by about
`(pi f dt)^2 (2 c_max^2 / c^2 - 1) / 3`, 0.9 % at 0.7 MHz and cfl = 0.3
(from the numerical wavenumber in `fullwave_plan.phase_error_bound`).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.greens import medium_greens
from neurosim.fus.types import Transducer
from neurosim.head.materials import MaterialParams

EVAL_ID = "E-FUS-020"
SPACING = 0.3e-3  # medium grid spacing [m]: 7 points per wavelength at 0.7 MHz
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]
FACE_LAYER = Material(sound_speed=2800.0, density=1900.0)  # bone-like, lossless (as E-FUS-006)
FACE_LAYER_ROWS = 20  # the layer fills grid rows 0..19: from the array face to 5.85 mm
SCATTERER_NODE = (50, 35)  # medium-grid (row, column): 15 mm deep, 0.9 mm off the array axis
SOUND_SPEED_CONTRAST = 0.02  # relative sound-speed increase of the scatterer node
# The echo's round trip must fit the record: one more one-way time (10 us) after the
# backend's own one-way allowance; the default tail of 20 us covers it here.
REVERBERATION_TAIL = 20.0e-6  # [s]
MAX_AMPLITUDE_ERROR = 0.05
# The amplitude is defined only if the echo has the pattern g g^T (the claim of E-FUS-019).
MIN_CORRELATION = 0.999


@eval_case(
    id=EVAL_ID,
    claim="static_fmc is in the units of the bank's Green's function, and transmit and receive "
    "differ by the density ratio between the array face and the scatterer: the full-wave echo of "
    "a known weak sound-speed perturbation at r_s is dk2 A [rho(face) / rho(r_s)] "
    "G(r_s, i, f) G(r_s, j, f) with no free constant",
    reference="Analytic: first Born term of the Lippmann-Schwinger series for "
    "rho div(grad p / rho) + (w/c)^2 p, with dk2 = w^2 (1/c'^2 - 1/c^2) and A the area of one "
    "grid cell (derivation in the module docstring)",
    metric="max over the 9 frequencies of |c(f) / c_Born(f) - 1| [dimensionless], c(f) the "
    "least-squares complex amplitude of the scattered full matrix on g g^T",
    threshold="< 0.05 (fixed in advance: at a contrast of 2 % the first Born term is off by "
    "the node's self-interaction, about 1 % at 0.7 MHz, and the solver's time stepping shifts "
    "the effective contrast by about 0.9 % at 0.7 MHz and cfl 0.3, so about 2 % is expected; "
    "5 % leaves a factor 2.5, while a missing density factor or source scaling is an error of "
    "more than 45 % in this phantom and a wrong power of k a factor 2.3 across the band)",
)
def test_e_fus_020_scattered_amplitude():
    grid = Grid2D.from_extent((-9.6e-3, 9.6e-3), (0.0, 21.0e-3), SPACING)
    brain = MaterialParams().grey
    static = slab_medium(
        grid,
        Material(brain.sound_speed, brain.density),
        FACE_LAYER,
        z_top=-0.5 * SPACING,
        thickness=FACE_LAYER_ROWS * SPACING,
        alpha_power=1.5,
    )
    sound_speed = static.sound_speed.copy()
    sound_speed[SCATTERER_NODE] *= 1.0 + SOUND_SPEED_CONTRAST
    perturbed = dataclasses.replace(static, sound_speed=sound_speed)
    transducer = Transducer(
        n_elements=2,
        pitch=10 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )
    scatterer_xz = grid.points()[SCATTERER_NODE]
    # The bank is stored on the scatterer node alone, so G(r_s) is read without interpolation.
    roi = Grid2D(shape=(1, 1), spacing=SPACING, origin=(scatterer_xz[1], scatterer_xz[0]))
    cfg = BackendConfig(
        backend="kwave", kwave=KWaveConfig(cfl=0.3, reverberation_tail=REVERBERATION_TAIL)
    )

    bank = compute_bank(static, transducer, roi, FREQS, cfg)
    bank_with_scatterer = compute_bank(perturbed, transducer, roi, FREQS, cfg)
    assert bank.meta["roi_snap_distance_m"] < 1e-9
    # The difference is the scattered field only if both runs share the time axis.
    for key in ("time_step_s", "time_steps"):
        assert bank.meta[key] == bank_with_scatterer.meta[key]
    scattered = bank_with_scatterer.static_fmc.astype(np.complex128) - bank.static_fmc

    greens = medium_greens(bank, scatterer_xz[None, :])[:, :, 0]  # g[f, i]
    model = greens[:, :, None] * greens[:, None, :]
    inner = np.sum(np.conj(model) * scattered, axis=(1, 2))
    correlations = np.abs(inner) / (
        np.linalg.norm(scattered, axis=(1, 2)) * np.linalg.norm(model, axis=(1, 2))
    )
    assert correlations.min() > MIN_CORRELATION
    amplitude = inner / np.sum(np.abs(model) ** 2, axis=(1, 2))  # c(f)

    slowness_squared_change = (
        1.0 / float(perturbed.sound_speed[SCATTERER_NODE]) ** 2
        - 1.0 / float(static.sound_speed[SCATTERER_NODE]) ** 2
    )
    born_without_density = (2.0 * np.pi * FREQS) ** 2 * slowness_squared_change * SPACING**2
    # Row 0 is the array face; the layer under it is uniform along the array.
    face_density = float(static.density[0, grid.shape[1] // 2])
    density_ratio = face_density / float(static.density[SCATTERER_NODE])
    ratio = amplitude / (born_without_density * density_ratio)
    error = float(np.max(np.abs(ratio - 1.0)))

    report(
        EVAL_ID,
        {
            "max_amplitude_error": error,
            "amplitude_ratio_lowest_frequency_real": float(ratio[0].real),
            "amplitude_ratio_lowest_frequency_imag": float(ratio[0].imag),
            "amplitude_ratio_highest_frequency_real": float(ratio[-1].real),
            "amplitude_ratio_highest_frequency_imag": float(ratio[-1].imag),
            "density_ratio": density_ratio,
            "max_amplitude_error_without_density_ratio": float(
                np.max(np.abs(amplitude / born_without_density - 1.0))
            ),
            "min_correlation": float(correlations.min()),
            "scattered_over_static_echo": float(
                np.abs(scattered).max() / np.abs(bank.static_fmc).max()
            ),
        },
        passed=error < MAX_AMPLITUDE_ERROR,
    )
