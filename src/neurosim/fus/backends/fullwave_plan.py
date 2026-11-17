"""Geometry, time stepping and cost of a full-wave bank computation, fixed before it runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.backends.config import KWaveConfig
from neurosim.fus.backends.fullwave_solver import check_absorption_model, pml_sizes
from neurosim.fus.backends.geometry import check_geometry, check_medium, checked_frequencies
from neurosim.fus.backends.spectra import gaussian_pulse
from neurosim.fus.types import Transducer
from neurosim.head.types import PlaneMedium

# Distance from a grid node (or half-way point) within which a position counts as on it [cells].
_NODE_TOLERANCE = 1e-6

# Working memory of one simulation, in float arrays of the padded grid size. Measured as peak
# resident memory on a 547 000-point grid (float64): 49 lossless, 54 with absorption. Counting
# the arrays held by kwave/solvers/kspace_solver.py gives 37 (19 real fields and coefficients,
# 9 complex operators and FFT temporaries); the rest is transient.
_STATE_ARRAYS_LOSSLESS = 49
_STATE_ARRAYS_ABSORPTION = 5
# FFTs per time step, counted from the same source: 11 lossless (1 + 2 for the pressure
# gradient, 4 for the divergence, 4 for the k-space source correction), plus 2 for the
# absorption term and 2 for the dispersion term. The run-time constant in the config is for
# the lossless count.
_FFTS_LOSSLESS = 11
_FFTS_PER_LOSS_TERM = 2


@dataclass(frozen=True)
class ReferenceLayout:
    """Sensors of the single-element simulation in the homogeneous reference medium.

    The reference medium is translation invariant and the element is mirror
    symmetric, so the field of any element at any medium-grid node equals the
    reference field at the same row and at a column `first_column + shift`,
    with `shift` from `reference_shifts`.

    Attributes:
        shape: reference grid (nz, nx) [grid points].
        first_column: first column of the reference element's footprint.
        roi_shifts: sorted distinct shifts needed on the ROI rows, shape (n_roi_shifts,).
        array_shifts: sorted distinct shifts needed on the array row, shape (n_array_shifts,).
        row_groups: sensor rows of each reference simulation (the array row and the ROI
            rows, split so that one recording stays within the memory budget).
        pml: absorbing-layer thickness along (z, x) [grid points].
    """

    shape: tuple[int, int]
    first_column: int
    roi_shifts: np.ndarray
    array_shifts: np.ndarray
    row_groups: tuple[tuple[int, ...], ...]
    pml: tuple[int, int]


@dataclass(frozen=True)
class FullWavePlan:
    """Everything the full-wave backend fixes before simulating.

    Attributes:
        spacing: grid spacing [m].
        medium_shape: medium grid (nz, nx) [grid points].
        array_row: grid row of the array face (z = 0).
        element_columns: grid columns driven by each element, shape (n_elements, n_footprint).
        pitch_cells: element pitch [grid cells].
        roi_rows: medium-grid row of every ROI row, shape (nz_roi,).
        roi_columns: medium-grid column of every ROI column, shape (nx_roi,).
        roi_snap_distance: largest distance between a ROI point and the node used for it [m].
        dt: time step [s].
        n_steps: number of time steps.
        pulse: drive pulse samples, shape (n_pulse,).
        pml: absorbing-layer thickness of the medium simulations along (z, x) [grid points].
        reference: layout of the reference simulation.
        top_frequency: highest frequency of the bank [Hz].
        points_per_wavelength: grid points per shortest wavelength [dimensionless].
        longest_path: largest element-to-ROI-point distance [m].
        slowest_speed, fastest_speed: extreme sound speeds of medium and reference [m/s].
        medium_speed_range: (lowest, highest) sound speed of the medium alone [m/s].
    """

    spacing: float
    medium_shape: tuple[int, int]
    array_row: int
    element_columns: np.ndarray
    pitch_cells: int
    roi_rows: np.ndarray
    roi_columns: np.ndarray
    roi_snap_distance: float
    dt: float
    n_steps: int
    pulse: np.ndarray
    pml: tuple[int, int]
    reference: ReferenceLayout
    top_frequency: float
    points_per_wavelength: float
    longest_path: float
    slowest_speed: float
    fastest_speed: float
    medium_speed_range: tuple[float, float]

    @property
    def n_sensor_points(self) -> int:
        """Recorded points of one medium simulation: the ROI nodes and every element's footprint."""
        return self.roi_rows.size * self.roi_columns.size + self.element_columns.size


def element_footprints(grid: Grid2D, transducer: Transducer) -> tuple[int, np.ndarray, int]:
    """Place the array on the medium grid.

    Each element drives the nodes of the z = 0 row within half an element width
    of its centre (at least the nearest ones). The reference simulation is
    shifted and mirrored to serve every element, which needs the same symmetric
    footprint for all of them; hence the requirements below.

    Args:
        grid: medium grid.
        transducer: the array.

    Returns:
        (array_row, element_columns of shape (n_elements, n_footprint), pitch in grid cells).

    Raises:
        ValueError: if z = 0 is not a grid row, the pitch is not a whole number of
            grid cells, the element centres are neither on nodes nor half-way
            between nodes, or a footprint leaves the grid.
    """
    z0, x0 = grid.origin
    row = -z0 / grid.spacing
    array_row = round(row)
    if abs(row - array_row) > _NODE_TOLERANCE or not 0 <= array_row < grid.shape[0]:
        raise ValueError(
            f"the array face z = 0 must be a row of the medium grid; it falls at row {row:.6g}"
        )
    pitch = transducer.pitch / grid.spacing
    pitch_cells = round(pitch)
    if pitch_cells < 1 or abs(pitch - pitch_cells) > _NODE_TOLERANCE:
        raise ValueError(
            f"the element pitch ({transducer.pitch:.6g} m) must be a whole number of medium grid "
            f"cells ({grid.spacing:.6g} m) so that one reference simulation serves every element; "
            f"it is {pitch:.6g} cells"
        )
    centres = (transducer.element_x() - x0) / grid.spacing  # fractional columns
    doubled = 2.0 * centres[0]
    if abs(doubled - round(doubled)) > 2.0 * _NODE_TOLERANCE:
        raise ValueError(
            "element centres must lie on medium grid nodes or half-way between two nodes "
            f"(symmetric footprint); the first centre falls at column {centres[0]:.6g}"
        )
    on_node = round(doubled) % 2 == 0
    half_width = max(0.5 * transducer.element_width / grid.spacing, 0.0 if on_node else 0.5)
    first = np.ceil(centres - half_width - _NODE_TOLERANCE).astype(int)
    last = np.floor(centres + half_width + _NODE_TOLERANCE).astype(int)
    n_footprint = int(last[0] - first[0] + 1)
    columns = first[:, None] + np.arange(n_footprint)[None, :]
    if columns.min() < 0 or columns.max() >= grid.shape[1]:
        raise ValueError("an element footprint falls outside the medium grid")
    return array_row, columns, pitch_cells


def roi_nodes(grid: Grid2D, roi: Grid2D) -> tuple[np.ndarray, np.ndarray, float]:
    """Snap the ROI grid to the nearest medium-grid nodes.

    The bank is computed at these nodes and stored on `roi`. B varies slowly in
    space (that is why it is stored on a coarse grid), so the shift of at most
    half a medium cell per axis is harmless; it is returned so it can be recorded.

    Returns:
        (rows of shape (nz_roi,), columns of shape (nx_roi,), largest snap distance [m]).

    Raises:
        ValueError: if two ROI rows or columns snap to the same node.
    """
    z0, x0 = grid.origin
    z, x = roi.axes()
    exact_rows, exact_columns = (z - z0) / grid.spacing, (x - x0) / grid.spacing
    rows, columns = np.rint(exact_rows).astype(int), np.rint(exact_columns).astype(int)
    if np.unique(rows).size != rows.size or np.unique(columns).size != columns.size:
        raise ValueError("the ROI grid must not be finer than the medium grid")
    snap = grid.spacing * math.hypot(
        np.abs(exact_rows - rows).max(), np.abs(exact_columns - columns).max()
    )
    return rows, columns, snap


def reference_shifts(columns: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Column offsets in the reference simulation that give an element's field at `columns`.

    The field of the element at a column equals the reference field at the
    reference element's first column plus the returned shift; columns on the
    far side of the element centre are mirrored onto the near side.

    Args:
        columns: medium-grid columns, any shape.
        footprint: the element's driven columns, ascending, shape (n_footprint,).

    Returns:
        non-negative integer shifts, same shape as `columns`.
    """
    right_of_centre = 2 * columns >= footprint[0] + footprint[-1]
    return np.where(right_of_centre, columns - footprint[0], footprint[-1] - columns)


def _row_groups(
    array_row: int,
    roi_rows: np.ndarray,
    n_array_points: int,
    n_roi_points: int,
    bytes_per_point: int,
    budget: int,
) -> tuple[tuple[int, ...], ...]:
    """Split the reference sensor rows into simulations whose recordings fit the memory budget."""
    sizes = [(array_row, n_array_points)] + [(int(row), n_roi_points) for row in roi_rows]
    groups: list[list[int]] = [[]]
    used = 0
    for row, n_points in sizes:
        needed = n_points * bytes_per_point
        if needed > budget:
            raise ValueError(
                f"one reference sensor row needs {needed} bytes of recording, more than "
                f"kwave.max_recording_bytes = {budget}"
            )
        if groups[-1] and used + needed > budget:
            groups.append([])
            used = 0
        groups[-1].append(row)
        used += needed
    return tuple(tuple(group) for group in groups)


def build_plan(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: KWaveConfig,
) -> FullWavePlan:
    """Validate the inputs of a full-wave bank computation and fix its discretisation.

    Args:
        medium: plane medium on a grid whose z = 0 row is the array face.
        transducer: the array.
        roi: grid the bank is stored on.
        freqs: frequencies [Hz], shape (nf,).
        cfg: backend settings.

    Raises:
        ValueError: for a medium grid too coarse for the top frequency or too small for
            the absorbing layer, an array or ROI that does not fit the grid (see
            `element_footprints`, `roi_nodes`), an absorption model the solver cannot run,
            or a recording over the memory budget.
    """
    freqs = checked_frequencies(freqs)
    check_medium(medium)
    check_geometry(medium, transducer, roi)
    check_absorption_model(medium.attenuation, medium.alpha_power, cfg)
    grid = medium.grid
    spacing = grid.spacing

    medium_speed_range = (float(medium.sound_speed.min()), float(medium.sound_speed.max()))
    slowest = min(medium_speed_range[0], C_REF)
    fastest = max(medium_speed_range[1], C_REF)
    points_per_wavelength = float(slowest / (freqs[-1] * spacing))
    if points_per_wavelength < cfg.min_points_per_wavelength:
        raise ValueError(
            f"the medium grid ({spacing:.3g} m) gives {points_per_wavelength:.2f} points per "
            f"wavelength at {freqs[-1]:.3g} Hz and {slowest:.0f} m/s; "
            f"kwave.min_points_per_wavelength = {cfg.min_points_per_wavelength}"
        )

    medium_pml = pml_sizes(grid.shape, cfg)

    array_row, element_columns, pitch_cells = element_footprints(grid, transducer)
    roi_rows, roi_columns, snap = roi_nodes(grid, roi)
    if roi_rows.min() <= array_row:
        raise ValueError("every ROI row must snap to a grid row in front of the array face")

    dt = cfg.cfl * spacing / fastest
    pulse = gaussian_pulse(freqs, dt, cfg.pulse_edge_level_db, cfg.pulse_max_cycles)
    centre_columns = 0.5 * (element_columns[:, 0] + element_columns[:, -1])
    lateral = np.abs(roi_columns[None, :] - centre_columns[:, None]).max()
    longest_path = spacing * math.hypot(lateral, roi_rows.max() - array_row)
    duration = pulse.size * dt + longest_path / slowest + cfg.reverberation_tail
    n_steps = math.ceil(duration / dt)

    bytes_per_point = n_steps * np.dtype(cfg.dtype).itemsize
    n_footprint = element_columns.shape[1]
    roi_shifts = np.unique(
        np.stack([reference_shifts(roi_columns, footprint) for footprint in element_columns])
    )
    array_shifts = np.unique(
        pitch_cells * np.arange(transducer.n_elements)[:, None] + np.arange(n_footprint)[None, :]
    )
    reference_columns = (
        2 * cfg.reference_margin + int(max(roi_shifts.max(), array_shifts.max())) + 1
    )
    # k-Wave needs every axis longer than twice the absorbing layer.
    smallest_axis = 2 * cfg.pml_size + 1
    reference_shape = (
        min(grid.shape[0], max(int(roi_rows.max()) + 1 + cfg.reference_margin, smallest_axis)),
        max(reference_columns, smallest_axis),
    )
    reference = ReferenceLayout(
        shape=reference_shape,
        first_column=cfg.reference_margin,
        roi_shifts=roi_shifts,
        array_shifts=array_shifts,
        row_groups=_row_groups(
            array_row,
            roi_rows,
            array_shifts.size,
            roi_shifts.size,
            bytes_per_point,
            cfg.max_recording_bytes,
        ),
        pml=pml_sizes(reference_shape, cfg),
    )

    plan = FullWavePlan(
        spacing=spacing,
        medium_shape=grid.shape,
        array_row=array_row,
        element_columns=element_columns,
        pitch_cells=pitch_cells,
        roi_rows=roi_rows,
        roi_columns=roi_columns,
        roi_snap_distance=snap,
        dt=dt,
        n_steps=n_steps,
        pulse=pulse,
        pml=medium_pml,
        reference=reference,
        top_frequency=float(freqs[-1]),
        points_per_wavelength=points_per_wavelength,
        longest_path=longest_path,
        slowest_speed=slowest,
        fastest_speed=fastest,
        medium_speed_range=medium_speed_range,
    )
    if plan.n_sensor_points * bytes_per_point > cfg.max_recording_bytes:
        raise ValueError(
            f"one medium simulation records {plan.n_sensor_points} points x {n_steps} steps "
            f"({plan.n_sensor_points * bytes_per_point} bytes), more than "
            f"kwave.max_recording_bytes = {cfg.max_recording_bytes}; coarsen the ROI or raise it"
        )
    return plan


def phase_error_bound(plan: FullWavePlan, frequency: float) -> float:
    """Upper bound on the numerical phase error of the medium simulations [rad].

    The k-space scheme is exact where the sound speed equals the largest speed
    of the simulated medium, c_max. Elsewhere its leapfrog update with the factor
    `sinc(c_max k dt / 2)` gives the numerical wavenumber

        k_num = 2 / (c_max dt) * arcsin( (c_max / c0) sin(w dt / 2) )

    instead of w / c0 (derived from the update equations in
    `kwave/solvers/kspace_solver.py` and confirmed against a 1D simulation to five
    digits; the exactness at c0 = c_max is stated in Treeby et al. 2012, J. Acoust.
    Soc. Am. 131(6), sec. V A). The bound is the excess phase over the longest
    element-to-ROI path if all of it were at the slowest speed of the medium. It
    scales with cfl^2 and vanishes for a homogeneous medium; the reference
    simulation is homogeneous and therefore exact.

    Args:
        plan: the simulation plan.
        frequency: frequency [Hz].
    """
    slowest, fastest = plan.medium_speed_range
    half_step = math.pi * frequency * plan.dt  # w dt / 2
    sine = min(1.0, fastest / slowest * math.sin(half_step))
    numerical = 2.0 / (fastest * plan.dt) * math.asin(sine)
    exact = 2.0 * math.pi * frequency / slowest
    return (numerical - exact) * plan.longest_path


def _padded_points(shape: tuple[int, int], pml: tuple[int, int]) -> int:
    return (shape[0] + 2 * pml[0]) * (shape[1] + 2 * pml[1])


def estimate_cost(
    plan: FullWavePlan, n_freqs: int, n_elements: int, absorbing: bool, cfg: KWaveConfig
) -> dict[str, float | int]:
    """Cost of a full-wave bank computation.

    Args:
        plan: the simulation plan.
        n_freqs: number of frequencies of the bank.
        n_elements: number of array elements (one medium simulation each).
        absorbing: whether the medium has absorption (more work per time step).
        cfg: backend settings.

    Returns:
        dict with `grid_points` (one medium simulation, absorbing layer included),
        `reference_grid_points`, `time_steps`, `n_simulations` (medium plus reference),
        `memory_bytes` (peak estimate), `run_time_s` (wall-clock estimate) and
        `phase_error_bound_rad` (see `phase_error_bound`, at the top frequency).
    """
    itemsize = np.dtype(cfg.dtype).itemsize
    grid_points = _padded_points(plan.medium_shape, plan.pml)
    reference_points = _padded_points(plan.reference.shape, plan.reference.pml)
    n_reference = len(plan.reference.row_groups)

    state_arrays = _STATE_ARRAYS_LOSSLESS + (_STATE_ARRAYS_ABSORPTION if absorbing else 0)
    spectra_bytes = 16 * n_freqs * (plan.n_sensor_points + 2 * plan.n_steps)
    worker_bytes = (
        state_arrays * grid_points * itemsize
        + plan.n_sensor_points * plan.n_steps * itemsize
        + spectra_bytes
    )
    layout = plan.reference
    reference_recording = max(
        sum(
            layout.array_shifts.size if row == plan.array_row else layout.roi_shifts.size
            for row in group
        )
        for group in layout.row_groups
    )
    reference_bytes = (
        _STATE_ARRAYS_LOSSLESS * reference_points * itemsize
        + reference_recording * plan.n_steps * itemsize
    )
    n_roi = plan.roi_rows.size * plan.roi_columns.size
    result_bytes = 8 * n_freqs * n_elements * n_roi + 16 * n_freqs * (
        plan.roi_rows.size * plan.reference.roi_shifts.size + n_elements * n_elements
    )
    workers = min(cfg.n_workers, n_elements)
    memory = result_bytes + max(reference_bytes, workers * worker_bytes)

    def simulation_seconds(points: int, ffts: int) -> float:
        per_step = cfg.seconds_per_point_step * points * math.log2(points) * ffts / _FFTS_LOSSLESS
        return per_step * plan.n_steps

    loss_terms = (2 if cfg.model_dispersion else 1) if absorbing else 0
    medium_seconds = simulation_seconds(
        grid_points, _FFTS_LOSSLESS + loss_terms * _FFTS_PER_LOSS_TERM
    )
    run_time = (
        n_reference * simulation_seconds(reference_points, _FFTS_LOSSLESS)
        + n_elements * medium_seconds / workers
    )
    return {
        "grid_points": grid_points,
        "reference_grid_points": reference_points,
        "time_steps": plan.n_steps,
        "n_simulations": n_elements + n_reference,
        "memory_bytes": int(memory),
        "run_time_s": float(run_time),
        "phase_error_bound_rad": phase_error_bound(plan, plan.top_frequency),
    }
