"""E-FUS-021: straight-ray travel times against refracted first arrivals across the full aperture.

The `ray` backend integrates slowness along the straight line from an element
to a point (`fus.backends.ray`). Through a fast layer the first arrival follows
the refracted path of least time (Fermat), which is never slower than the
straight ray, and the gap grows with obliquity. The backend's checks against
k-Wave (E-FUS-005, E-DATA-011) use small apertures at low frequency; this eval
measures the gap over the production aperture at the production frequency.

Set-up (a flat slab, the simplest case that isolates the claim):

- Array: `configs/fus_default.yaml` (128 elements at 2 MHz, pitch half a wavelength
  at C_REF).
- Medium: soft tissue / slab / soft tissue, built with `phantoms.slab_medium` on the
  production medium grid (`RunConfig.medium_spacing`, 0.1 mm) and traced with the
  production ray settings (`RunConfig.backend.ray`).
- Soft tissue: the 'Brain' row of the head material table (`MaterialParams.grey`,
  1560 m/s; Aubry 2022, Table I).
- Slab thickness: the median vault thickness of the head model
  (`AnatomyConfig.skull_thickness_vault`, 6.5 mm). Slab top and sound speed are
  round values near those of the default head: in the plane medium of the demo run
  (`configs/run_demo.yaml`, seed 0; measured once offline, not recomputed here) the
  skull starts 6.7 mm below the array centre and its mean sound speed is 2554 m/s.
- Target: on the array axis at 30 mm depth (mid-ROI of the default configuration).
- The slab is given the soft-tissue density and no absorption: neither enters a
  travel time.

Reference. The first arrival through three homogeneous layers is the fastest
path made of three straight segments, element -> (x1, slab top) -> (x2, slab
bottom) -> target. Its time is found by minimising the path time over (x1, x2)
with scipy (BFGS, numerical gradient, so Snell's law is not put in by hand).
The path time is a convex function of (x1, x2), so the minimum is unique and
global; any other route through the slab is slower. The minimiser is checked
against Snell's law, `sin(theta) / c` equal on both sides of each interface.
A transmitted path exists for every element, at an incidence angle below the
critical angle `asin(c_soft / c_slab)`.

The backend's sampling error is reported separately, as its difference from the
exact time along the straight line (a fraction `thickness / depth` of any
straight element-to-target line lies in the slab).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.data.config import RunConfig
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.backends.ray import ray_travel_times
from neurosim.fus.config import ImagingConfig, transducer_from_config
from neurosim.fus.types import Transducer

EVAL_ID = "E-FUS-021"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_default.yaml"
SLAB_SOUND_SPEED = 2500.0  # [m/s] near the mean skull speed of the default head (docstring)
SLAB_TOP = 7.0e-3  # nominal depth of the slab's upper face [m] (docstring)
TARGET_DEPTH = 30.0e-3  # [m]
TABLE_PITCH = 5.0e-3  # spacing of the offsets listed in the result note [m]
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


@eval_case(
    id=EVAL_ID,
    claim=(
        "for a flat bone-like slab (6.5 mm at 2500 m/s, 7 mm below the array) the ray backend's "
        "one-way travel time from every element of the default 128-element 2 MHz array to a "
        "target on the array axis at 30 mm depth equals the refracted first-arrival time to "
        "within lambda/14"
    ),
    reference=(
        "Fermat's principle in a three-layer medium: the least time over all paths of three "
        "straight segments, minimised numerically over the two interface crossing points and "
        "checked against Snell's law"
    ),
    metric=(
        "max over the 128 elements of (ray-backend travel time - first-arrival time) * centre "
        "frequency [periods]"
    ),
    threshold=(
        "< 1/14 = 0.071 period (fixed in advance: the lambda/14 Marechal-type budget that "
        "E-FUS-005 already applies to the ray backend's phase, 2 pi / 14 = 0.449 rad at the "
        "centre frequency; applied to the worst element because the refraction error is "
        "systematic, growing towards the aperture edge, so it does not average out)"
    ),
)
def test_e_fus_021_ray_refraction_error() -> None:
    run = RunConfig()
    transducer = transducer_from_config(from_dict(ImagingConfig, load_yaml(CONFIG)))
    soft = run.head.materials.grey
    spacing = run.medium_spacing

    # Slab faces on pixel faces, where the backend's bilinear slowness map puts the interface.
    slab_top = (round(SLAB_TOP / spacing) - 0.5) * spacing
    thickness = round(run.head.anatomy.skull_thickness_vault / spacing) * spacing
    layers = np.array([slab_top, thickness, TARGET_DEPTH - slab_top - thickness])
    speeds = np.array([soft.sound_speed, SLAB_SOUND_SPEED, soft.sound_speed])

    half_width = np.ceil((0.5 * transducer.aperture + run.medium_margin) / spacing) * spacing
    grid = Grid2D.from_extent(
        (-half_width, half_width), (0.0, TARGET_DEPTH + run.medium_margin), spacing
    )
    background = Material(soft.sound_speed, soft.density)
    medium = slab_medium(
        grid,
        background,
        Material(SLAB_SOUND_SPEED, soft.density),
        slab_top,
        thickness,
        run.head.materials.alpha_power,
    )
    target = np.array([0.0, TARGET_DEPTH])
    frequency = transducer.center_frequency

    element_x = transducer.element_x()
    traced = ray_travel_times(medium, transducer, target, run.backend.ray)
    first_arrival, incidence = _first_arrivals(element_x, layers, speeds)
    error = (traced - first_arrival) * frequency  # [periods]
    exact_straight = np.hypot(element_x, TARGET_DEPTH) * np.sum(layers / speeds) / TARGET_DEPTH

    # The same comparison at round offsets, for the table in the result note.
    table_array = Transducer(
        n_elements=2 * int(0.5 * transducer.aperture / TABLE_PITCH) + 1,
        pitch=TABLE_PITCH,
        element_width=transducer.element_width,
        center_frequency=frequency,
        fractional_bandwidth=transducer.fractional_bandwidth,
    )
    table_x = table_array.element_x()
    table_error = (
        ray_travel_times(medium, table_array, target, run.backend.ray)
        - _first_arrivals(table_x, layers, speeds)[0]
    ) * frequency

    offsets = np.abs(element_x)
    within = error <= MAX_ERROR_PERIODS
    # The error grows with the offset, so the usable half aperture ends at the innermost
    # element that exceeds the budget.
    limit = offsets[~within].min() if not within.all() else np.inf
    usable_offset = float(offsets[offsets < limit].max())
    edge = int(np.argmax(offsets))
    rows = [(x, e) for x, e in zip(table_x, table_error) if x >= 0.0]
    values = {
        "max_error_periods": float(error.max()),
        "fraction_of_aperture_within_budget": float(within.mean()),
        "largest_offset_within_budget_mm": 1.0e3 * usable_offset,
        "f_number_within_budget": TARGET_DEPTH / (2.0 * usable_offset),
        "critical_angle_deg": float(np.rad2deg(np.arcsin(speeds[0] / speeds[1]))),
        "edge_offset_mm": 1.0e3 * float(offsets[edge]),
        "edge_straight_ray_incidence_deg": float(
            np.rad2deg(np.arctan2(offsets[edge], TARGET_DEPTH))
        ),
        "edge_refracted_incidence_deg": float(np.rad2deg(incidence[edge])),
        "max_abs_sampling_error_periods": float(
            np.max(np.abs(traced - exact_straight)) * frequency
        ),
        **{f"error_periods_at_{1.0e3 * x:.0f}mm": float(e) for x, e in rows},
        "error_periods_at_edge": float(error[edge]),
    }
    rows.append((offsets[edge], error[edge]))
    note = (
        "error [periods] by element offset: "
        + ", ".join(f"{1.0e3 * x:.2f} mm: {e:.4f}" for x, e in rows)
        + f"; slab {1.0e3 * slab_top:.2f} to {1.0e3 * (slab_top + thickness):.2f} mm at "
        f"{SLAB_SOUND_SPEED:.0f} m/s in {soft.sound_speed:.0f} m/s; the receive f-number is "
        "target depth / (2 * largest offset within budget); max_abs_sampling_error is the "
        "backend against the exact time along the straight line"
    )
    report(EVAL_ID, values, passed=values["max_error_periods"] < MAX_ERROR_PERIODS, note=note)
