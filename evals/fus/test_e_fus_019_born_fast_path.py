"""E-FUS-019: the fast path reproduces the full-wave echo of a weak scatterer behind a bone slab.

The fast path never simulates a scatterer. It takes the bank of the static
medium and writes the echo of a point scatterer at `r_s`, transmit element `i`,
receive element `j`, as `s G(r_s, i, f) G(r_s, j, f)` with `G = B G_ref`
(`fus.greens.medium_greens`, used by `fus.synthesis`).

Here the scatterer is put into the full-wave simulation instead: the sound
speed of one grid node behind an absorbing, reverberating, laterally varying
bone-like slab is raised by a few percent, and the bank is computed with and
without it. Both runs share the grid, the time step and the record length, so
the difference of their `static_fmc` is the full-wave scattered field
`S[f, i, j]`; everything static (slab echoes, the reference field subtracted by
the backend) cancels.

What the comparison can and cannot see. With `L = rho div(grad . / rho) + k^2`
and `L G = -delta`, a change `dk2 = w^2 (1/c'^2 - 1/c^2)` over the node area `A`
gives, to first order, `S = dk2 A [rho(array) / rho(r_s)] g g^T`, `g[i] =
G(r_s, i, f)`; the density ratio is the reciprocity factor of this
normalisation and is 1 here. A single sub-wavelength node re-scatters only
through the same `g`, so the product form `c(f) g g^T` holds at any strength
and the Born error shows up in `c(f)` alone (measured while designing: against
the first-order value `c(f)` is 1.8 % off at a contrast of 5 %, 7 % at 20 %,
12 % at 50 %, while the correlation does not move). The eval therefore tests
the bank's Green's function, reciprocity and the consistency of `static_fmc`
with the bank, not the size of `c(f)` (E-FUS-020) and not the mutual scattering
of several scatterers.

Choice of the contrast. Lower limit: `static_fmc` is stored as complex64, so the
difference of two banks carries rounding noise of about 3e-9 against a scattered
field of 4e-4 times the contrast; the worst pair's phase error from that noise
was 0.26 rad at a contrast of 1e-4, 0.026 rad at 1e-3 and 0.003 rad at 1e-2.
Upper limit: none for the metric (see above); the sound speed must stay between
the reference and the bone speed so that the time step and record length do not
change. 5 % sits a factor 50 above the noise limit of the phase threshold.

Recording window. The backend records for (pulse) + (one-way time to the
farthest ROI point at the slowest speed) + `reverberation_tail`. The scattered
echo needs the round trip, so the tail must cover a second one-way time (12.7 us
here) plus the reverberation of the slab. Measured while designing: with a tail
of 5 us the correlation is 0.72-0.78 and with 10 us the phase error is 0.39 rad;
20 us and 30 us agree to 0.002 rad. 30 us is used.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.greens import medium_greens
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

EVAL_ID = "E-FUS-019"
SPACING = 0.3e-3  # medium grid spacing [m]: 7 points per wavelength at 0.7 MHz
FREQS = np.linspace(0.3e6, 0.7e6, 9)  # [Hz]; the drive pulse is 6 dB down at both ends
BONE = Material(sound_speed=2800.0, density=1900.0, attenuation=4.0)  # bone-like, absorbing
ALPHA_POWER = 1.5  # absorption exponent of the phantom, as in E-FUS-007
SLAB_TOP = 13.5 * SPACING  # depth of the slab's upper face [m], on a pixel face
SLAB_THIN, SLAB_STEP = 3.0e-3, 3.0e-3  # slab thickness far left, and its increase to the right [m]
SLAB_STEP_X, SLAB_STEP_WIDTH = 1.5e-3, 1.0e-3  # centre and scale of the thickness step [m]
SCATTERER_NODE = (60, 47)  # medium-grid (row, column): 18 mm deep, 1.5 mm off the array axis
SOUND_SPEED_CONTRAST = 0.05  # relative sound-speed increase of the scatterer node
REVERBERATION_TAIL = 30.0e-6  # [s]; see "Recording window" in the module docstring
MIN_CORRELATION = 0.999
MAX_PHASE_ERROR = 0.1  # [rad]
# The eval is only meaningful if ignoring the slab is clearly visible in the same metric.
MAX_HOMOGENEOUS_CORRELATION = 0.9


def with_faster_node(medium: PlaneMedium, node: tuple[int, int], contrast: float) -> PlaneMedium:
    """The medium with the sound speed of one grid node raised by the factor `1 + contrast`."""
    sound_speed = medium.sound_speed.copy()
    sound_speed[node] *= 1.0 + contrast
    return dataclasses.replace(medium, sound_speed=sound_speed)


def pair_products(greens: np.ndarray) -> np.ndarray:
    """`g[f, i] g[f, j]` for every element pair, shape (nf, n_elements, n_elements)."""
    return greens[:, :, None] * greens[:, None, :]


def correlation(scattered: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Normalised complex correlation over element pairs at each frequency, shape (nf,)."""
    inner = np.abs(np.sum(np.conj(model) * scattered, axis=(1, 2)))
    return inner / (np.linalg.norm(scattered, axis=(1, 2)) * np.linalg.norm(model, axis=(1, 2)))


@eval_case(
    id=EVAL_ID,
    claim="The echo of a weak point scatterer behind an aberrating, absorbing, reverberating "
    "bone-like slab, as synthesised by the fast path from the bank of the static medium "
    "(G(r_s, i, f) G(r_s, j, f) with G = B G_ref), has the element-pair pattern of the echo in a "
    "full-wave simulation that contains the scatterer",
    reference="Full-wave simulation (k-Wave) of the same medium with the sound speed of one grid "
    "node raised by 5 %: the difference of the two full-matrix recordings, S[f, i, j]. The "
    "prediction uses only the bank of the unperturbed medium",
    metric="(a) min over the 9 frequencies of |<g g^T, S>| / (|g g^T| |S|) over all 25 element "
    "pairs [dimensionless]; (b) max over frequencies and pairs of |arg(S_ij / (c(f) g_i g_j))| "
    "[rad], c(f) the least-squares complex amplitude at each frequency",
    threshold="(a) > 0.999 and (b) < 0.1 rad (fixed in advance). One sub-wavelength node "
    "re-scatters only through the same g, so the Born error enters c(f) alone; the scatterer is "
    "on a bank node, so no interpolation enters; the static_fmc normalisation is one factor per "
    "frequency. What remains is the solver's reciprocity error (bounded at 1 % by E-FUS-007), "
    "the element model of G_ref (< 0.2 %), the truncated reverberation tail (designed below "
    "1 %) and complex64 storage (0.1 %): about 2 % of relative residual. 0.999 allows 4.5 % "
    "(1 - correlation = residual^2 / 2), and 0.1 rad is 5 % on a pair of half the rms "
    "amplitude, a quarter of the lambda/14 budget of E-FUS-005",
)
def test_e_fus_019_born_fast_path():
    grid = Grid2D.from_extent((-12.6e-3, 12.6e-3), (0.0, 25.2e-3), SPACING)
    x = grid.axes()[1]
    # Thickness doubles across the aperture: every element sees a different delay and loss.
    thickness = SLAB_THIN + SLAB_STEP / (1.0 + np.exp(-(x - SLAB_STEP_X) / SLAB_STEP_WIDTH))
    static = slab_medium(
        grid, Material(C_REF, RHO_WATER), BONE, SLAB_TOP, thickness, alpha_power=ALPHA_POWER
    )
    perturbed = with_faster_node(static, SCATTERER_NODE, SOUND_SPEED_CONTRAST)
    transducer = Transducer(
        n_elements=5,
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

    def prediction(model_bank: GreensBank) -> np.ndarray:
        return pair_products(medium_greens(model_bank, scatterer_xz[None, :])[:, :, 0])

    model = prediction(bank)
    correlations = correlation(scattered, model)
    amplitude = np.sum(np.conj(model) * scattered, axis=(1, 2)) / np.sum(
        np.abs(model) ** 2, axis=(1, 2)
    )  # c(f)
    phase_error = np.abs(np.angle(scattered / (amplitude[:, None, None] * model)))
    # Diagnostic: the same metric when the slab is ignored (the no-skull fast path).
    homogeneous = correlation(
        scattered, prediction(GreensBank.homogeneous(transducer, roi, FREQS, C_REF))
    )
    assert homogeneous.max() < MAX_HOMOGENEOUS_CORRELATION
    # Diagnostic: a sound-speed (compressibility) contrast scatters in proportion to k^2.
    amplitude_per_k2 = np.abs(amplitude) / FREQS**2

    report(
        EVAL_ID,
        {
            "min_correlation": float(correlations.min()),
            "max_one_minus_correlation": float(np.max(1.0 - correlations)),
            "max_phase_error_rad": float(phase_error.max()),
            "homogeneous_correlation_min": float(homogeneous.min()),
            "homogeneous_correlation_max": float(homogeneous.max()),
            "amplitude_frequency_exponent": float(
                np.polyfit(np.log(FREQS), np.log(np.abs(amplitude)), 1)[0]
            ),
            "amplitude_over_k2_spread": float(np.ptp(amplitude_per_k2) / amplitude_per_k2.mean()),
            "scattered_over_static_echo": float(
                np.abs(scattered).max() / np.abs(bank.static_fmc).max()
            ),
            "tail_level_db": float(bank_with_scatterer.meta["tail_level_db"]),
        },
        passed=bool(correlations.min() > MIN_CORRELATION and phase_error.max() < MAX_PHASE_ERROR),
    )
