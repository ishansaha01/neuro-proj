"""E-FUS-023: bent-ray travel times against refracted first arrivals across the full aperture.

E-FUS-021 measures how far the straight-ray travel times of the `ray` backend
are from the refracted first arrival through a flat bone-like slab, and fails:
0.66 period at the edge of the default aperture against a budget of 1/14. This
eval puts the `bent_ray` backend (eikonal travel times, `fus.backends.bent_ray`)
through the same geometry, with the same budget and the same reference.

Set-up, as in E-FUS-021:

- Array: `configs/fus_default.yaml` (128 elements at 2 MHz, pitch half a wavelength
  at C_REF).
- Medium: soft tissue / slab / soft tissue, built with `phantoms.slab_medium` on the
  production medium grid (`RunConfig.medium_spacing`, 0.1 mm), with the default
  `BentRayConfig`.
- Soft tissue: the 'Brain' row of the head material table (`MaterialParams.grey`,
  1560 m/s; Aubry 2022, Table I).
- Slab: the median vault thickness of the head model
  (`AnatomyConfig.skull_thickness_vault`, 6.5 mm), upper face 7 mm below the array,
  at 2500 m/s (the case of E-FUS-021) and, as a second case, at 2800 m/s (a faster,
  cortical-like bone with a smaller critical angle, 33.9 degrees).
- Target: on the array axis at 30 mm depth.
- The slab has the soft-tissue density and no absorption: neither enters a travel time.

Reference. The first arrival through three homogeneous layers is the fastest
path made of three straight segments, element -> (x1, slab top) -> (x2, slab
bottom) -> target. Its time is found by minimising the path time over (x1, x2)
with scipy (BFGS, numerical gradient, so Snell's law is not put in by hand).
The path time is a convex function of (x1, x2), so the minimum is unique and
global. The minimiser is checked against Snell's law, and gives the angle of
incidence of the refracted ray on the slab; the claim is restricted to elements
for which that angle is below the critical angle `asin(c_soft / c_slab)` (in
this geometry a transmitted path exists for every element, so that is all 128).

Reported besides the verdict: the same comparison for the first-order grid
value of the eikonal solver (the backend integrates along the traced ray
instead), both on the production grid and on a grid of twice the spacing, with
the observed order of convergence of each; and the straight-ray error of the
`ray` backend on the same medium.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.data.config import RunConfig
from neurosim.fus.backends.bent_ray import trace_bent_rays
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.backends.ray import ray_travel_times
from neurosim.fus.config import ImagingConfig, transducer_from_config
from neurosim.fus.types import Transducer
from neurosim.head.types import PlaneMedium

EVAL_ID = "E-FUS-023"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_default.yaml"
SLAB_SOUND_SPEEDS = (2500.0, 2800.0)  # [m/s]: the case of E-FUS-021 and a faster bone
SLAB_TOP = 7.0e-3  # nominal depth of the slab's upper face [m]
TARGET_DEPTH = 30.0e-3  # [m]
TABLE_ELEMENT_STEP = 13  # every 13th element (5 mm) enters the table of the result note
COARSE_FACTOR = 2  # the convergence check doubles the grid spacing
MAX_ERROR_PERIODS = 1.0 / 14.0
# Largest allowed mismatch of sin(theta) / c across an interface, times c_soft. The
# path time is stationary at the minimum, so this bounds its error to about 1e-12 of itself.
MAX_SNELL_RESIDUAL = 1.0e-6


def _segment_lengths(element_x: float, crossings: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """Lengths of the three segments element -> crossings -> on-axis target [m], shape (3,).

    Args:
        element_x: lateral position of the element [m] (at depth 0).
        crossings: lateral positions where the path crosses the two slab faces [m], shape (2,).
        layers: thickness of the three layers along depth [m], shape (3,).
    """
    lateral = np.diff(np.concatenate(([element_x], crossings, [0.0])))
    return np.hypot(lateral, layers)


def _fermat_path(
    element_x: float, layers: np.ndarray, speeds: np.ndarray
) -> tuple[float, np.ndarray]:
    """Path of least time from an element to the on-axis target through three layers.

    Args:
        element_x: lateral position of the element [m].
        layers: thickness of the three layers along depth [m], shape (3,).
        speeds: sound speed of each layer [m/s], shape (3,).

    Returns:
        (travel time [s], crossing positions [m] of shape (2,)).
    """
    depth = layers.sum()

    def scaled_time(scaled_crossings: np.ndarray) -> float:
        lengths = _segment_lengths(element_x, scaled_crossings * depth, layers)
        return float(np.sum(lengths / speeds) * speeds[0] / depth)

    # Start from the straight line, in units of the target depth.
    straight = element_x * (1.0 - np.cumsum(layers)[:2] / depth) / depth
    result = minimize(scaled_time, straight, method="BFGS", options={"gtol": 1.0e-8})
    crossings = result.x * depth
    return float(np.sum(_segment_lengths(element_x, crossings, layers) / speeds)), crossings


def _snell_residual(
    element_x: float, crossings: np.ndarray, layers: np.ndarray, speeds: np.ndarray
) -> float:
    """Largest mismatch of `sin(theta) / c` across the two interfaces, times `speeds[0]`."""
    lateral = np.diff(np.concatenate(([element_x], crossings, [0.0])))
    ray_parameter = lateral / _segment_lengths(element_x, crossings, layers) / speeds
    return float(np.max(np.abs(np.diff(ray_parameter))) * speeds[0])


def _first_arrivals(
    element_x: np.ndarray, layers: np.ndarray, speeds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First-arrival time and incidence angle on the slab for every element.

    Args:
        element_x: lateral positions of the elements [m], shape (n,).
        layers: thickness of the three layers along depth [m], shape (3,).
        speeds: sound speed of each layer [m/s], shape (3,).

    Returns:
        (travel times [s], incidence angles of the refracted paths on the slab [rad]),
        each of shape (n,).

    Raises:
        AssertionError: if a minimised path does not satisfy Snell's law.
    """
    times = np.empty(element_x.shape)
    incidence = np.empty(element_x.shape)
    for index, x in enumerate(element_x):
        times[index], crossings = _fermat_path(x, layers, speeds)
        assert _snell_residual(x, crossings, layers, speeds) < MAX_SNELL_RESIDUAL
        incidence[index] = np.arctan2(abs(crossings[0] - x), layers[0])
    return times, incidence


def _slab_case(
    run: RunConfig, transducer: Transducer, slab_speed: float, spacing: float
) -> tuple[PlaneMedium, np.ndarray, np.ndarray]:
    """Slab phantom of one case on a grid of the given spacing.

    Returns:
        (medium, layer thicknesses [m] of shape (3,), layer sound speeds [m/s] of shape (3,)).
        The slab faces lie on pixel faces, where the bilinear slowness map puts the interface.
    """
    soft = run.head.materials.grey
    slab_top = (round(SLAB_TOP / spacing) - 0.5) * spacing
    thickness = round(run.head.anatomy.skull_thickness_vault / spacing) * spacing
    layers = np.array([slab_top, thickness, TARGET_DEPTH - slab_top - thickness])
    speeds = np.array([soft.sound_speed, slab_speed, soft.sound_speed])
    half_width = np.ceil((0.5 * transducer.aperture + run.medium_margin) / spacing) * spacing
    grid = Grid2D.from_extent(
        (-half_width, half_width), (0.0, TARGET_DEPTH + run.medium_margin), spacing
    )
    medium = slab_medium(
        grid,
        Material(soft.sound_speed, soft.density),
        Material(slab_speed, soft.density),
        slab_top,
        thickness,
        run.head.materials.alpha_power,
    )
    return medium, layers, speeds


@eval_case(
    id=EVAL_ID,
    claim=(
        "for a flat bone-like slab (6.5 mm, 7 mm below the array, at 2500 m/s and at 2800 m/s) "
        "the bent_ray backend's one-way travel time from every element of the default "
        "128-element 2 MHz array whose refracted ray meets the slab below the critical angle "
        "to a target on the array axis at 30 mm depth equals the refracted first-arrival time "
        "to within lambda/14"
    ),
    reference=(
        "Fermat's principle in a three-layer medium: the least time over all paths of three "
        "straight segments, minimised numerically over the two interface crossing points and "
        "checked against Snell's law (the reference of E-FUS-021)"
    ),
    metric=(
        "max over the two slabs and the elements below the critical angle of |bent_ray travel "
        "time - first-arrival time| * centre frequency [periods], on the 0.1 mm production grid"
    ),
    threshold=(
        "< 1/14 = 0.071 period (fixed in advance: the budget, geometry and reference of "
        "E-FUS-021, which the straight-ray backend fails; applied to the worst element because "
        "a travel-time error that grows across the aperture does not average out)"
    ),
)
def test_e_fus_023_bent_ray_travel_time() -> None:
    run = RunConfig()
    transducer = transducer_from_config(from_dict(ImagingConfig, load_yaml(CONFIG)))
    target = np.array([[0.0, TARGET_DEPTH]])
    frequency = transducer.center_frequency
    element_x = transducer.element_x()
    table = np.arange(transducer.n_elements // 2, transducer.n_elements, TABLE_ELEMENT_STEP)
    table = np.append(table, transducer.n_elements - 1)

    values: dict[str, float] = {}
    notes = []
    worst = 0.0
    for slab_speed in SLAB_SOUND_SPEEDS:
        name = f"{slab_speed:.0f}"
        errors = {}
        for spacing in (run.medium_spacing, COARSE_FACTOR * run.medium_spacing):
            medium, layers, speeds = _slab_case(run, transducer, slab_speed, spacing)
            first_arrival, incidence = _first_arrivals(element_x, layers, speeds)
            below_critical = incidence < np.arcsin(speeds[0] / speeds[1])
            integrals = trace_bent_rays(medium, transducer, target, run.backend.bent_ray)
            errors[spacing] = {
                "path": (integrals.travel_time[:, 0] - first_arrival) * frequency,
                "grid": (integrals.eikonal_time[:, 0] - first_arrival) * frequency,
            }
            if spacing == run.medium_spacing:
                straight = ray_travel_times(medium, transducer, target[0], run.backend.ray)
                straight_error = (straight - first_arrival) * frequency
                counted = below_critical
                edge_incidence = float(np.rad2deg(incidence.max()))
        fine, coarse = errors[run.medium_spacing], errors[COARSE_FACTOR * run.medium_spacing]
        case_worst = float(np.max(np.abs(fine["path"][counted])))
        worst = max(worst, case_worst)
        values.update(
            {
                f"max_abs_error_periods_{name}": case_worst,
                f"rms_error_periods_{name}": float(np.sqrt(np.mean(fine["path"][counted] ** 2))),
                f"elements_below_critical_{name}": int(counted.sum()),
                f"critical_angle_deg_{name}": float(
                    np.rad2deg(np.arcsin(run.head.materials.grey.sound_speed / slab_speed))
                ),
                f"edge_refracted_incidence_deg_{name}": edge_incidence,
                f"max_abs_grid_value_error_periods_{name}": float(np.max(np.abs(fine["grid"]))),
                f"max_abs_error_periods_coarse_grid_{name}": float(
                    np.max(np.abs(coarse["path"]))
                ),
                f"max_abs_grid_value_error_periods_coarse_grid_{name}": float(
                    np.max(np.abs(coarse["grid"]))
                ),
                f"observed_order_grid_value_{name}": float(
                    np.log2(np.max(np.abs(coarse["grid"])) / np.max(np.abs(fine["grid"])))
                ),
                f"observed_order_path_integral_rms_{name}": float(
                    np.log2(np.sqrt(np.mean(coarse["path"] ** 2) / np.mean(fine["path"] ** 2)))
                ),
                f"straight_ray_max_error_periods_{name}": float(np.max(straight_error)),
            }
        )
        rows = ", ".join(
            f"{1.0e3 * element_x[e]:.2f} mm: {fine['path'][e]:+.4f} / {coarse['path'][e]:+.4f}"
            f" (grid value {fine['grid'][e]:+.4f} / {coarse['grid'][e]:+.4f})"
            for e in table
        )
        notes.append(f"slab at {name} m/s, error [periods] at 0.1 mm / 0.2 mm by offset: {rows}")
    values["max_error_periods"] = worst
    note = (
        "; ".join(notes)
        + "; 'grid value' is the first-order eikonal solution read at the target, the backend's "
        "travel time is the slowness integral along the ray traced through it; observed_order "
        "is log2 of the error ratio between 0.2 mm and 0.1 mm spacing (maximum for the grid "
        "value, rms over the aperture for the path integral, whose error is the midpoint-rule "
        "noise at the two interfaces)"
    )
    report(EVAL_ID, values, passed=worst < MAX_ERROR_PERIODS, note=note)
