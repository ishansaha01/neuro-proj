"""E-FUS-024: bent-ray and k-Wave banks agree in first-arrival phase at oblique incidence.

E-FUS-005 checks the straight-ray bank against k-Wave on a weak layer at small
angles. This eval checks the `bent_ray` bank where refraction matters: through
a bone-like slab (2500 m/s, 1900 kg/m^3, 12 mm) at element-to-target angles up
to 54 degrees, well beyond the 30 degrees at which straight rays have already
left the lambda/14 budget.

Two media on a 0.3 mm grid, 0.25 to 0.75 MHz (6.8 grid points per wavelength at
the top frequency), lossless:

- flat: the slab between 5.85 and 17.85 mm depth; one element at x = 0 and
  targets at 30 and 33 mm depth, 0 to 42 mm to the side (straight-line angles 0
  to 54.5 degrees; the refracted ray meets the slab at up to 34.8 degrees, the
  critical angle being 38.0 degrees);
- curved: the same slab with its upper face on a circle of 80 mm radius whose
  apex is 6 mm off the array centre (the faces tilt by up to 17 degrees); two
  elements at -15 and +15 mm and targets 0 to 42 mm to either side of them.

A bone-like slab reverberates, and the k-Wave bank holds every arrival, so the
phase of `B` itself is not the phase of the first arrival. The comparison is
therefore made on the first arrival, by pulse compression: with Hann weights
`w` over the band,

    h = sum_f w(f) B_kwave(f) conj(B_model(f)) / sum_f w(f)

is the band-limited impulse response of the k-Wave bank at the delay of the
model's (single) arrival, and `arg(h)` is the phase of k-Wave's first arrival
relative to the model at the centre frequency. The compressed pulse of the
Hann-weighted 0.5 MHz band has its first null 4 us from its peak. The first
internal echo of the slab follows the first arrival by `2 d cos(theta_slab) /
c_slab`: 9.6 us at normal incidence, and 3.6 us for the most oblique pair,
where the window passes 5 % of an echo that is itself about half the first
arrival, so about 3 % of it.

Cost, from `estimate_cost` before the runs (logged by `compute_bank`): about
40 000 grid points and 2 800 steps per simulation at CFL 0.15, five simulations
in all, about 70 s. The recording window is the one-way time plus a 10 us tail
(`KWaveConfig.reverberation_tail`); the tail-level warning of the backend is
expected here and harmless, since only the first arrival is used.

Reported besides the verdict: the same phase for the straight-ray bank, and the
delay of each model against the delay of the k-Wave first arrival (the position
of the peak of the compressed pulse), in periods of the centre frequency.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

EVAL_ID = "E-FUS-024"
SPACING = 0.3e-3  # medium grid spacing [m]
FREQS = np.linspace(0.25e6, 0.75e6, 21)  # [Hz]; 25 kHz steps give a 40 us unambiguous range
BONE = Material(sound_speed=2500.0, density=1900.0)  # bone-like, lossless
SLAB_FIRST_ROW, SLAB_ROWS = 20, 40  # flat slab: grid rows 20..59, 12 mm thick
CURVATURE_RADIUS = 80.0e-3  # radius of the curved slab's upper face [m]
APEX_X = 6.0e-3  # lateral position of the curved face's shallowest point [m]
ELEMENT_OFFSET = 15.0e-3  # the two elements of the curved case sit at -/+ this [m]
TARGET_DEPTHS = (30.0e-3, 33.0e-3)  # [m]
TARGET_STEP = 3.0e-3  # lateral spacing of the targets [m]
MAX_TARGET_OFFSET = 42.0e-3  # largest lateral element-to-target distance [m]
GRID_MARGIN = 3.0e-3  # medium kept beyond the outermost element and target [m]
DELAY_SEARCH = 1.5e-6  # half-width of the window searched for the k-Wave first arrival [s]
DELAY_STEP = 5.0e-9  # resolution of that search [s]
MIN_STRAIGHT_ANGLE_DEG = 30.0  # the test must reach at least this element-to-target angle
MAX_RMS_PHASE = 2.0 * np.pi / 14.0  # [rad]
KWAVE = KWaveConfig(cfl=0.15, reverberation_tail=10.0e-6)


def _flat_case() -> tuple[PlaneMedium, Transducer, Grid2D]:
    """Flat slab, one element at x = 0, targets 0 to 42 mm to one side."""
    grid = Grid2D.from_extent(
        (-GRID_MARGIN, MAX_TARGET_OFFSET + GRID_MARGIN),
        (0.0, TARGET_DEPTHS[-1] + GRID_MARGIN),
        SPACING,
    )
    medium = slab_medium(
        grid,
        Material(C_REF, RHO_WATER),
        BONE,
        (SLAB_FIRST_ROW - 0.5) * SPACING,
        SLAB_ROWS * SPACING,
        alpha_power=1.5,
    )
    transducer = Transducer(
        n_elements=1,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=float(FREQS.mean()),
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent((0.0, MAX_TARGET_OFFSET), TARGET_DEPTHS, TARGET_STEP)
    return medium, transducer, roi


def _curved_case() -> tuple[PlaneMedium, Transducer, Grid2D]:
    """Slab with a circular upper face, two elements, targets to either side of them."""
    x_first = -ELEMENT_OFFSET - GRID_MARGIN
    x_last = MAX_TARGET_OFFSET - ELEMENT_OFFSET + GRID_MARGIN
    grid = Grid2D.from_extent((x_first, x_last), (0.0, TARGET_DEPTHS[-1] + GRID_MARGIN), SPACING)
    z, x = grid.axes()
    sag = CURVATURE_RADIUS - np.sqrt(CURVATURE_RADIUS**2 - (x - APEX_X) ** 2)
    top = (SLAB_FIRST_ROW - 0.5) * SPACING + sag
    in_slab = (z[:, None] >= top[None, :]) & (z[:, None] < top[None, :] + SLAB_ROWS * SPACING)
    medium = dataclasses.replace(
        homogeneous_medium(grid, Material(C_REF, RHO_WATER), alpha_power=1.5),
        sound_speed=np.where(in_slab, BONE.sound_speed, C_REF).astype(np.float32),
        density=np.where(in_slab, BONE.density, RHO_WATER).astype(np.float32),
    )
    transducer = Transducer(
        n_elements=2,
        pitch=2.0 * ELEMENT_OFFSET,
        element_width=SPACING,
        center_frequency=float(FREQS.mean()),
        fractional_bandwidth=0.8,
    )
    roi = Grid2D.from_extent(
        (-ELEMENT_OFFSET, MAX_TARGET_OFFSET - ELEMENT_OFFSET), TARGET_DEPTHS, TARGET_STEP
    )
    return medium, transducer, roi


def _band_average(values: np.ndarray) -> np.ndarray:
    """Hann-weighted average over the frequency axis (axis 0) of an array of shape (nf, ...)."""
    weights = np.hanning(FREQS.size + 2)[1:-1]
    return np.tensordot(weights / weights.sum(), values, axes=(0, 0))


def _delay(bank_values: np.ndarray) -> np.ndarray:
    """Delay of a single-arrival bank [s]: the slope of its phase, shape (n_pairs,)."""
    phase = np.unwrap(np.angle(bank_values), axis=0)
    return np.polyfit(2.0 * np.pi * FREQS, phase, 1)[0]


def _case_results(
    medium: PlaneMedium, transducer: Transducer, roi: Grid2D
) -> dict[str, np.ndarray]:
    """Phase and delay of both ray models against the k-Wave first arrival, per pair."""
    banks: dict[str, GreensBank] = {
        name: compute_bank(medium, transducer, roi, FREQS, cfg)
        for name, cfg in (
            ("kwave", BackendConfig(backend="kwave", kwave=KWAVE)),
            ("bent_ray", BackendConfig(backend="bent_ray")),
            ("ray", BackendConfig(backend="ray")),
        )
    }
    values = {
        name: bank.B.reshape(FREQS.size, -1).astype(np.complex128) for name, bank in banks.items()
    }
    points = roi.points()
    lateral = np.abs(points[None, ..., 0] - transducer.element_x()[:, None, None])
    angle = np.rad2deg(np.arctan2(lateral, points[None, ..., 1])).reshape(-1)
    lit = np.abs(values["bent_ray"]).min(axis=0) > 0.0
    kwave = values["kwave"][:, lit]
    # Delay of the k-Wave first arrival: the peak of the compressed pulse near the bent-ray delay.
    offsets = np.arange(-DELAY_SEARCH, DELAY_SEARCH + 0.5 * DELAY_STEP, DELAY_STEP)
    searched = _delay(values["bent_ray"][:, lit])[None, :] + offsets[:, None]
    shifted = kwave[:, None, :] * np.exp(-2.0j * np.pi * FREQS[:, None, None] * searched[None])
    peak = np.argmax(np.abs(_band_average(shifted)), axis=0)
    kwave_delay = searched[peak, np.arange(peak.size)]
    result = {
        "angle_deg": angle[lit],
        "n_dark": np.array(np.count_nonzero(~lit)),
        "phase_error_bound": np.array(
            banks["kwave"].meta["cost_estimate"]["phase_error_bound_rad"]
        ),
    }
    for name in ("bent_ray", "ray"):
        model = values[name][:, lit]
        result[f"{name}_phase"] = np.angle(_band_average(kwave * np.conj(model)))
        result[f"{name}_delay_error_periods"] = (_delay(model) - kwave_delay) * FREQS.mean()
    return result


@eval_case(
    id=EVAL_ID,
    claim=(
        "through a bone-like slab (2500 m/s, 12 mm; flat, and gently curved with 80 mm "
        "radius) at element-to-target angles from 0 to 54 degrees, the bent_ray bank "
        "reproduces the phase of the first arrival of the full-wave bank"
    ),
    reference=(
        "Full-wave solution (k-Wave) of the same medium on the same 0.3 mm grid, 0.25 to "
        "0.75 MHz; its first arrival is isolated by pulse compression over the band. The two "
        "backends share no code beyond the input checks"
    ),
    metric=(
        "RMS over all element-target pairs of both media of arg(sum_f w B_kwave "
        "conj(B_bent_ray)), w a Hann window over the band [rad at the centre frequency, 0.5 MHz]"
    ),
    threshold=(
        "< 2 pi / 14 = 0.449 rad (fixed in advance: the Marechal lambda/14 budget that "
        "E-FUS-005 applies to the straight-ray bank on a weak layer at small angles, here "
        "applied at angles up to 54 degrees; it includes the phase error of the k-Wave "
        "reference itself, whose bound is reported)"
    ),
)
def test_e_fus_024_bent_ray_kwave_phase() -> None:
    cases = {"flat": _flat_case(), "curved": _curved_case()}
    results = {name: _case_results(*case) for name, case in cases.items()}

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2)))

    bent_phase = np.concatenate([result["bent_ray_phase"] for result in results.values()])
    values = {
        "rms_phase_rad": rms(bent_phase),
        "max_abs_phase_rad": float(np.abs(bent_phase).max()),
    }
    for name, result in results.items():
        oblique = result["angle_deg"] >= MIN_STRAIGHT_ANGLE_DEG
        values.update(
            {
                f"{name}_pairs": float(result["angle_deg"].size),
                f"{name}_pairs_without_bent_ray_amplitude": float(result["n_dark"]),
                f"{name}_max_straight_angle_deg": float(result["angle_deg"].max()),
                f"{name}_rms_phase_rad": rms(result["bent_ray_phase"]),
                f"{name}_rms_phase_rad_from_30_deg": rms(result["bent_ray_phase"][oblique]),
                f"{name}_max_abs_delay_error_periods": float(
                    np.abs(result["bent_ray_delay_error_periods"]).max()
                ),
                f"{name}_straight_ray_rms_phase_rad": rms(result["ray_phase"]),
                f"{name}_straight_ray_rms_phase_rad_from_30_deg": rms(
                    result["ray_phase"][oblique]
                ),
                f"{name}_straight_ray_max_abs_delay_error_periods": float(
                    np.abs(result["ray_delay_error_periods"]).max()
                ),
                f"{name}_kwave_phase_error_bound_rad": float(result["phase_error_bound"]),
            }
        )
    reaches = all(
        values[f"{name}_max_straight_angle_deg"] >= MIN_STRAIGHT_ANGLE_DEG for name in results
    )
    report(
        EVAL_ID,
        values,
        passed=reaches and values["rms_phase_rad"] < MAX_RMS_PHASE,
        note=(
            "phases are of the k-Wave first arrival relative to each model at 0.5 MHz, wrapped "
            "to +-pi (the straight-ray phase wraps beyond about 45 degrees, so its delay error "
            "in periods, measured against the peak of the compressed k-Wave pulse, is the "
            "better measure of its error); a period is 2 us"
        ),
    )
