import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, estimate_cost
from neurosim.fus.backends.fullwave_plan import (
    build_plan,
    element_footprints,
    phase_error_bound,
    reference_shifts,
    roi_nodes,
)
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.types import Transducer

FREQS = np.linspace(0.3e6, 0.7e6, 5)


def array(n_elements: int, pitch_cells: float, width_cells: float, spacing: float) -> Transducer:
    return Transducer(n_elements, pitch_cells * spacing, width_cells * spacing, 0.5e6, 0.8)


def test_footprints_centred_on_nodes(grid, spacing):
    row, columns, pitch_cells = element_footprints(grid, array(3, 4, 3, spacing))
    assert row == 0 and pitch_cells == 4
    np.testing.assert_array_equal(columns, [[15, 16, 17], [19, 20, 21], [23, 24, 25]])


def test_footprints_centred_between_nodes(grid, spacing):
    # Two elements 3 cells apart: centres at columns 18.5 and 21.5.
    _, columns, _ = element_footprints(grid, array(2, 3, 2, spacing))
    np.testing.assert_array_equal(columns, [[18, 19], [21, 22]])
    # An element narrower than a cell still drives the two nearest nodes.
    _, columns, _ = element_footprints(grid, array(2, 3, 0.5, spacing))
    np.testing.assert_array_equal(columns, [[18, 19], [21, 22]])


def test_footprint_requirements(grid, spacing):
    with pytest.raises(ValueError, match="whole number of medium grid cells"):
        element_footprints(grid, array(3, 3.5, 1, spacing))
    shifted = Grid2D(grid.shape, spacing, (0.0, grid.origin[1] + 0.3 * spacing))
    with pytest.raises(ValueError, match="half-way between two nodes"):
        element_footprints(shifted, array(3, 4, 1, spacing))
    lowered = Grid2D(grid.shape, spacing, (0.4 * spacing, grid.origin[1]))
    with pytest.raises(ValueError, match="must be a row"):
        element_footprints(lowered, array(3, 4, 1, spacing))
    with pytest.raises(ValueError, match="outside the medium grid"):
        element_footprints(grid, array(11, 4, 3, spacing))


def test_array_row_below_the_top_of_the_grid(spacing):
    grid = Grid2D.from_extent((-6.0e-3, 6.0e-3), (-5 * spacing, 12.0e-3), spacing)
    row, _, _ = element_footprints(grid, array(3, 4, 1, spacing))
    assert row == 5


def test_roi_snaps_to_the_nearest_nodes(grid, spacing):
    roi = Grid2D.from_extent(
        (-3.0e-3 + 0.1 * spacing, 3.0e-3), (6.0e-3 - 0.2 * spacing, 9.0e-3), 3.0e-3
    )
    rows, columns, snap = roi_nodes(grid, roi)
    np.testing.assert_array_equal(rows, [20, 30])
    np.testing.assert_array_equal(columns, [10, 20, 30])
    assert snap == pytest.approx(spacing * np.hypot(0.2, 0.1))
    with pytest.raises(ValueError, match="finer than the medium grid"):
        roi_nodes(grid, Grid2D.from_extent((0.0, 1.0e-3), (6.0e-3, 9.0e-3), 0.4 * spacing))


def test_reference_shifts_mirror_about_the_element_centre():
    footprint = np.array([10, 11, 12])  # centre 11
    np.testing.assert_array_equal(
        reference_shifts(np.array([11, 12, 15, 10, 7]), footprint), [1, 2, 5, 2, 5]
    )
    footprint = np.array([10, 11])  # centre 10.5
    np.testing.assert_array_equal(
        reference_shifts(np.array([11, 14, 10, 7]), footprint), [1, 4, 1, 4]
    )


def test_plan_time_stepping_and_reference_layout(
    grid, soft, bone, transducer, roi, fast_kwave, spacing
):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    plan = build_plan(medium, transducer, roi, FREQS, fast_kwave)
    assert plan.dt == pytest.approx(0.3 * spacing / 2800.0)
    assert plan.points_per_wavelength == pytest.approx(C_REF / (0.7e6 * spacing))
    assert plan.longest_path == pytest.approx(spacing * np.hypot(14, 30))
    expected_duration = plan.pulse.size * plan.dt + plan.longest_path / C_REF + 2.0e-6
    assert plan.n_steps == int(np.ceil(expected_duration / plan.dt))
    assert plan.n_sensor_points == 15 + 3
    assert all(8 <= size <= 12 for size in plan.pml + plan.reference.pml)

    layout = plan.reference
    # ROI columns 10, 15, ..., 30 seen from element columns 16, 20, 24 (mirrored): 0 to 14 cells.
    np.testing.assert_array_equal(layout.roi_shifts, [0, 1, 4, 5, 6, 9, 10, 14])
    np.testing.assert_array_equal(layout.array_shifts, [0, 4, 8])
    assert layout.first_column == 6
    assert layout.shape[1] >= 6 + 14 + 1 + 6 and layout.shape[0] <= grid.shape[0]
    assert layout.row_groups == ((0, 20, 25, 30),)


def test_reference_recording_is_split_to_fit_the_memory_budget(
    grid, soft, transducer, roi, fast_kwave
):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    plan = build_plan(medium, transducer, roi, FREQS, fast_kwave)
    row_bytes = plan.reference.roi_shifts.size * plan.n_steps * 8
    medium_bytes = plan.n_sensor_points * plan.n_steps * 8
    tight = dataclasses.replace(fast_kwave, max_recording_bytes=medium_bytes)
    groups = build_plan(medium, transducer, roi, FREQS, tight).reference.row_groups
    assert len(groups) > 1 and sorted(sum(groups, ())) == [0, 20, 25, 30]
    with pytest.raises(ValueError, match="max_recording_bytes"):
        too_tight = dataclasses.replace(fast_kwave, max_recording_bytes=row_bytes - 1)
        build_plan(medium, transducer, roi, FREQS, too_tight)


def test_plan_refuses_unusable_inputs(grid, soft, transducer, roi, fast_kwave, spacing):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    with pytest.raises(ValueError, match="points per wavelength"):
        build_plan(medium, transducer, roi, np.linspace(1.0e6, 2.0e6, 3), fast_kwave)
    with pytest.raises(ValueError, match="too short for an absorbing layer"):
        thick = dataclasses.replace(fast_kwave, pml_size=30, pml_search=12)
        build_plan(medium, transducer, roi, FREQS, thick)
    lossy = homogeneous_medium(
        grid, Material(C_REF, 1000.0, attenuation=1.0), alpha_power=1.0
    )
    build_plan(lossy, transducer, roi, FREQS, fast_kwave)
    with pytest.raises(ValueError, match="model_dispersion"):
        with_dispersion = dataclasses.replace(fast_kwave, model_dispersion=True)
        build_plan(lossy, transducer, roi, FREQS, with_dispersion)


def test_phase_error_bound_vanishes_when_homogeneous_and_scales_with_cfl_squared(
    grid, soft, bone, transducer, roi, fast_kwave, spacing
):
    water = homogeneous_medium(grid, Material(1480.0, 1000.0), alpha_power=1.5)
    water_plan = build_plan(water, transducer, roi, FREQS, fast_kwave)
    assert phase_error_bound(water_plan, 0.7e6) == pytest.approx(0.0, abs=1e-9)
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    coarse = phase_error_bound(build_plan(medium, transducer, roi, FREQS, fast_kwave), 0.7e6)
    fine_cfg = dataclasses.replace(fast_kwave, cfl=0.1)
    fine = phase_error_bound(build_plan(medium, transducer, roi, FREQS, fine_cfg), 0.7e6)
    assert coarse > 0.0 and coarse / fine == pytest.approx(9.0, rel=0.02)


def test_cost_estimate(grid, soft, bone, transducer, roi, fast_kwave, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    cfg = BackendConfig(backend="kwave", kwave=fast_kwave)
    cost = estimate_cost(medium, transducer, roi, FREQS, cfg)
    plan = build_plan(medium, transducer, roi, FREQS, fast_kwave)
    assert cost["grid_points"] == (41 + 2 * plan.pml[0]) * (41 + 2 * plan.pml[1])
    assert cost["time_steps"] == plan.n_steps
    assert cost["n_simulations"] == 3 + 1
    assert cost["memory_bytes"] > cost["grid_points"] * 8 * 30
    assert cost["run_time_s"] > 0.0 and cost["phase_error_bound_rad"] > 0.0
    two_workers = BackendConfig(backend="kwave", kwave=dataclasses.replace(fast_kwave, n_workers=2))
    faster = estimate_cost(medium, transducer, roi, FREQS, two_workers)
    assert faster["run_time_s"] < cost["run_time_s"]
    assert faster["memory_bytes"] > cost["memory_bytes"]
