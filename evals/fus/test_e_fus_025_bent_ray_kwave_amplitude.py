"""E-FUS-025: bent-ray and k-Wave banks agree in first-arrival amplitude at oblique incidence.

The amplitude of the `bent_ray` bank is ray theory in a locally plane-layered
medium: the interface coefficients with the angle of incidence, and the
spreading of the ray tube (`fus.backends.bent_ray`). This eval checks it
against the full-wave first arrival on the two media of E-FUS-024 (a bone-like
slab, 2500 m/s, 1900 kg/m^3, 12 mm; flat, and with its upper face on a circle
of 80 mm radius), 0.25 to 0.75 MHz on a 0.3 mm grid, lossless, at
element-to-target angles from 0 to 54.5 degrees.

First arrival. A bone-like slab reverberates, so the k-Wave amplitude is taken
from the compressed pulse, as in E-FUS-004: with Hann weights `w` over the band,
`|sum_f w B_kwave(f) exp(-i w tau)| / sum_f w` has one peak per arrival, and the
first is searched within 1.5 us of the bent-ray delay. The `bent_ray` and `ray`
banks hold one arrival of frequency-independent amplitude in a lossless
medium, so their amplitude is `|B|`.

Which pairs. The claim is about pairs below the critical angle. In the flat
medium the refracted ray of every pair meets the slab below the critical angle
of 38.0 degrees (at up to 34.8 degrees; computed here from Snell's law). In the
curved medium there is no closed form, and the pairs are those the backend
itself does not taper as grazing (`BentRayIntegrals.window` >= 1/2). Both
counts are reported.

Beyond the critical angle. The brief of the backend expected the amplitude to
be small where the straight line from element to target is clearly beyond the
critical angle. Reported as a diagnostic, for the flat medium and straight-line
angles of at least 48 degrees (10 degrees beyond critical): the k-Wave
first-arrival amplitude relative to normal incidence, and the bent-ray
amplitude relative to k-Wave there. A bar of 0.3 for the first was written
down before the run; whether it is met is reported, and is not part of the
verdict: the backend does not cut the amplitude at the straight-line critical
angle but lets it fall with the refracted angle, and that is what the second
number tests.

Cost, from `estimate_cost` before the runs (logged by `compute_bank`): about
40 000 grid points and 1 400 steps per simulation at CFL 0.3, five simulations
in all, about 35 s.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import brentq

from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.units import np_to_db
from neurosim.fus.backends import BackendConfig, BentRayConfig, KWaveConfig, compute_bank
from neurosim.fus.backends.bent_ray import trace_bent_rays
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.types import Transducer
from neurosim.head.types import PlaneMedium

EVAL_ID = "E-FUS-025"
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
BEYOND_CRITICAL_DEG = 10.0  # "clearly beyond": straight-line angle this far past critical
MAX_BEYOND_CRITICAL_RATIO = 0.3  # bar of the diagnostic (module docstring)
MAX_AMPLITUDE_ERROR_DB = 2.0
KWAVE = KWaveConfig(cfl=0.3, reverberation_tail=10.0e-6)


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


def _first_arrival_amplitude(kwave: np.ndarray, expected_delay: np.ndarray) -> np.ndarray:
    """Amplitude of the first arrival of k-Wave bank values, shape (n_pairs,).

    Args:
        kwave: B of shape (nf, n_pairs).
        expected_delay: delay near which the first arrival is searched [s], shape (n_pairs,).
    """
    offsets = np.arange(-DELAY_SEARCH, DELAY_SEARCH + 0.5 * DELAY_STEP, DELAY_STEP)
    searched = expected_delay[None, :] + offsets[:, None]
    shifted = kwave[:, None, :] * np.exp(-2.0j * np.pi * FREQS[:, None, None] * searched[None])
    return np.abs(_band_average(shifted)).max(axis=0)


def _incidence_on_flat_slab(offset: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Angle at which the refracted ray meets the flat slab [deg], from Snell's law.

    Args:
        offset: lateral element-to-target distance [m], shape (n,).
        depth: target depth [m], shape (n,).
    """
    speeds = np.array([C_REF, BONE.sound_speed, C_REF])
    top, thickness = (SLAB_FIRST_ROW - 0.5) * SPACING, SLAB_ROWS * SPACING
    angles = np.zeros(offset.shape)
    for index, (lateral, z) in enumerate(zip(offset, depth)):
        layers = np.array([top, thickness, z - top - thickness])

        def miss(ray_parameter: float) -> float:
            sines = ray_parameter * speeds
            return float(np.sum(layers * sines / np.sqrt(1.0 - sines**2)) - lateral)

        if lateral > 0.0:
            ray_parameter = brentq(miss, 0.0, (1.0 - 1.0e-12) / BONE.sound_speed)
            angles[index] = np.rad2deg(np.arcsin(ray_parameter * C_REF))
    return angles


def _case_results(
    medium: PlaneMedium, transducer: Transducer, roi: Grid2D
) -> dict[str, np.ndarray]:
    """First-arrival amplitudes of the three backends and the pair geometry, per pair."""
    banks = {
        name: compute_bank(medium, transducer, roi, FREQS, cfg).B.reshape(FREQS.size, -1)
        for name, cfg in (
            ("kwave", BackendConfig(backend="kwave", kwave=KWAVE)),
            ("bent_ray", BackendConfig(backend="bent_ray")),
            ("ray", BackendConfig(backend="ray")),
        )
    }
    points = roi.points().reshape(-1, 2)
    integrals = trace_bent_rays(medium, transducer, points, BentRayConfig())
    offset = np.abs(points[None, :, 0] - transducer.element_x()[:, None]).reshape(-1)
    depth = np.broadcast_to(points[None, :, 1], (transducer.n_elements, points.shape[0]))
    expected_delay = (integrals.travel_time - integrals.path_length / C_REF).reshape(-1)
    return {
        "offset": offset,
        "depth": depth.reshape(-1),
        "angle_deg": np.rad2deg(np.arctan2(offset, depth.reshape(-1))),
        "window": integrals.window.reshape(-1),
        "kwave": _first_arrival_amplitude(banks["kwave"].astype(np.complex128), expected_delay),
        "bent_ray": np.abs(banks["bent_ray"]).mean(axis=0),
        "ray": np.abs(banks["ray"]).mean(axis=0),
    }


def _decibels(ratio: np.ndarray) -> np.ndarray:
    return np_to_db(np.log(ratio))


@eval_case(
    id=EVAL_ID,
    claim=(
        "through a bone-like slab (2500 m/s, 1900 kg/m^3, 12 mm; flat, and gently curved with "
        "80 mm radius) the amplitude of the bent_ray bank equals the amplitude of the first "
        "arrival of the full-wave bank for every element-target pair below the critical angle, "
        "at element-to-target angles from 0 to 54 degrees"
    ),
    reference=(
        "Full-wave solution (k-Wave) of the same medium on the same 0.3 mm grid, 0.25 to "
        "0.75 MHz; the amplitude of its first arrival is the first peak of the pulse "
        "compressed over the band (as in E-FUS-004)"
    ),
    metric=(
        "max over the pairs below the critical angle of both media of "
        "|20 log10(|B_bent_ray| / first-arrival amplitude of B_kwave)| [dB]"
    ),
    threshold=(
        "< 2 dB (fixed in advance from the approximations: ray theory at 0.5 MHz, where the "
        "first Fresnel zone spans roughly +-8 degrees of incidence angle, so that the steep "
        "fall of the transmission towards the critical angle is smoothed; plane parallel "
        "layers assumed along each ray, against a slab curved at 80 mm radius; the 5 % that "
        "E-FUS-004 allows the k-Wave amplitude at normal incidence; and the leakage of the "
        "slab's first internal echo into the compressed first arrival)"
    ),
)
def test_e_fus_025_bent_ray_kwave_amplitude() -> None:
    results = {"flat": _case_results(*_flat_case()), "curved": _case_results(*_curved_case())}
    flat = results["flat"]
    critical = float(np.rad2deg(np.arcsin(C_REF / BONE.sound_speed)))
    incidence = _incidence_on_flat_slab(flat["offset"], flat["depth"])
    below = {"flat": incidence < critical, "curved": results["curved"]["window"] >= 0.5}

    values: dict[str, float] = {}
    worst = 0.0
    for name, result in results.items():
        chosen = below[name]
        error_db = _decibels(result["bent_ray"][chosen] / result["kwave"][chosen])
        ray_error_db = _decibels(result["ray"][chosen] / result["kwave"][chosen])
        worst = max(worst, float(np.abs(error_db).max()))
        values.update(
            {
                f"{name}_pairs": float(chosen.size),
                f"{name}_pairs_below_critical": float(chosen.sum()),
                f"{name}_max_straight_angle_deg": float(result["angle_deg"][chosen].max()),
                f"{name}_max_abs_error_db": float(np.abs(error_db).max()),
                f"{name}_rms_error_db": float(np.sqrt(np.mean(error_db**2))),
                f"{name}_straight_ray_max_abs_error_db": float(np.abs(ray_error_db).max()),
            }
        )
    normal = flat["kwave"][flat["offset"] == 0.0].mean()
    beyond = flat["angle_deg"] >= critical + BEYOND_CRITICAL_DEG
    kwave_ratio = flat["kwave"][beyond] / normal
    values.update(
        {
            "max_abs_error_db": worst,
            "critical_angle_deg": critical,
            "flat_max_refracted_incidence_deg": float(incidence.max()),
            "flat_pairs_beyond_critical_straight_line": float(beyond.sum()),
            "flat_kwave_ratio_to_normal_beyond_critical_min": float(kwave_ratio.min()),
            "flat_kwave_ratio_to_normal_beyond_critical_max": float(kwave_ratio.max()),
            "flat_kwave_ratio_beyond_critical_within_bar": float(
                kwave_ratio.max() < MAX_BEYOND_CRITICAL_RATIO
            ),
            "flat_bent_ray_to_kwave_beyond_critical_min": float(
                (flat["bent_ray"][beyond] / flat["kwave"][beyond]).min()
            ),
            "flat_bent_ray_to_kwave_beyond_critical_max": float(
                (flat["bent_ray"][beyond] / flat["kwave"][beyond]).max()
            ),
            "flat_straight_ray_to_kwave_beyond_critical_max": float(
                (flat["ray"][beyond] / flat["kwave"][beyond]).max()
            ),
        }
    )
    report(
        EVAL_ID,
        values,
        passed=worst < MAX_AMPLITUDE_ERROR_DB,
        note=(
            "'beyond critical' is the diagnostic of the module docstring: flat medium, "
            f"straight-line angle at least {BEYOND_CRITICAL_DEG:.0f} degrees past the critical "
            "angle; the k-Wave ratio is its first-arrival amplitude there over that at normal "
            f"incidence, with a bar of {MAX_BEYOND_CRITICAL_RATIO} written before the run "
            "(within_bar: 1 = met, 0 = not met; not part of the verdict). In these media no "
            "pair comes within the grazing taper of the backend, so the taper window itself "
            "is not exercised here"
        ),
    )
