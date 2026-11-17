"""First-arrival travel times of point sources on a plane grid: a factored eikonal solver.

The eikonal equation `|grad T| = 1 / c` is solved for a batch of point sources
at once, in plain numpy, for the `bent_ray` backend (`bent_ray.py`).

Factoring. Near a point source `T` has a distance-like singularity, which a
plain first-order scheme resolves poorly over hundreds of cells. The solver
therefore works with the excess `u = T - T0` over the analytic field of a
homogeneous medium, `T0 = s0 r` (`s0` the slowness at the source, `r` the
distance from it): `u` is smooth, and it vanishes wherever the straight path
from the source is homogeneous, so the scheme is exact there (Fomel, Luo &
Zhao 2009, J. Comput. Phys. 228:6440-6455, additive form as in Luo & Qian
2012, J. Comput. Phys. 231:4742-4755).

Update. A node C is updated from one edge of the grid line behind it: the
axis neighbour A and a diagonal neighbour B (a triangle of the eight-neighbour
stencil). On the triangle `T` is linear, and the eikonal equation gives

    T_C = T~_A + sqrt((s h)^2 - d^2),    d = T~_A - T~_B,    0 <= d <= s h / sqrt(2),

the first-order semi-Lagrangian update (Huygens' principle with a straight ray
from the edge A-B to C). Outside that range the ray comes from a vertex:
`T~_A + s_edge h` or `T~_B + sqrt(2) s h`. `T~` are the neighbour values
corrected for the curvature of `T0`,

    T~_A = T_A + T0_C - T0_A - grad T0(C) . (x_C - x_A),

which is exactly what turns the update of `T` into the update of the factored
unknown `u`. The slowness `s` of the triangle is the mean of the four nodes of
its cell, and `s_edge` the mean of the two nodes of the edge A-C, so the
scheme integrates the same bilinear slowness map as the ray backends.

Ordering. The grid is swept line by line in four directions (+z, +x, -x, -z).
The update of a line uses only the line behind it, so one line is one
vectorised operation over its nodes and over all sources of the batch. A
sweep along +z settles every ray that stays within 45 degrees of +z, and so
on; rays that turn from one cone to another need further sweeps. Values only
decrease, starting from the upper bound `r * max(slowness)` (a straight path
through the slowest material), and the iteration stops when four consecutive
sweeps (one of each kind) leave every node unchanged.

Accuracy: first order in the grid spacing (measured in E-FUS-023 against the
analytic refracted path through a slab). Nodes within 1.5 cells of a source
are set analytically and kept fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid2D

# Nodes within this many cells of a source are initialised from the straight path and
# kept fixed: every triangle that has the source inside or on its boundary has all three
# vertices within sqrt(2) cells of it.
_FROZEN_RADIUS_CELLS = 1.5
# Stands in for a zero distance from the source when dividing [m].
_TINY_DISTANCE = 1.0e-300
# The four sweeps of one cycle: (axis the sweep marches along, direction).
_SWEEPS = ((0, 1), (1, 1), (1, -1), (0, -1))


@dataclass(frozen=True)
class EikonalField:
    """First-arrival times of a batch of point sources on one grid.

    Attributes:
        grid: the grid of the slowness map.
        sources_xz: source positions (x, z) [m], shape (n_sources, 2).
        source_slowness: slowness at each source [s/m], shape (n_sources,).
        excess_time: `T - s0 r` [s], float64, shape (nz, nx, n_sources).
        n_sweeps: line sweeps run, the final four idle ones included.
    """

    grid: Grid2D
    sources_xz: np.ndarray
    source_slowness: np.ndarray
    excess_time: np.ndarray
    n_sweeps: int

    def travel_time(self, source: np.ndarray, xz: np.ndarray) -> np.ndarray:
        """First-arrival time from source `source[i]` to point `xz[i]`.

        Args:
            source: index of the source each point refers to, int, shape (n,).
            xz: points (x, z) [m], shape (n, 2), inside the grid.

        Returns:
            travel times [s], shape (n,): `s0 r` plus the bilinear interpolant of `excess_time`.
        """
        offset = xz - self.sources_xz[source]
        reference = self.source_slowness[source] * np.hypot(offset[:, 0], offset[:, 1])
        stencil = bilinear_stencil(self.grid, xz)
        return reference + gather_bilinear(self.excess_time, stencil, source)


@dataclass(frozen=True)
class BilinearStencil:
    """Cell and weights of the bilinear interpolation at a set of points.

    Attributes:
        corner: flattened (row-major) grid index of the cell corner with the
            lowest z and x, int64, shape (n,).
        weight_z, weight_x: fractional position inside the cell, in [0, 1], shape (n,).
        row_stride: number of grid columns (flat-index step of one row).
    """

    corner: np.ndarray
    weight_z: np.ndarray
    weight_x: np.ndarray
    row_stride: int


def bilinear_stencil(grid: Grid2D, xz: np.ndarray) -> BilinearStencil:
    """Bilinear stencil of plane points on `grid`.

    Follows `common.grid.sample_plane_map`: the fractional index of a point is
    `(z - z0) / spacing, (x - x0) / spacing`, and beyond the outermost node
    centres the edge value is used.

    Args:
        grid: the grid (at least 2 x 2 nodes).
        xz: points (x, z) [m], shape (n, 2).
    """
    nz, nx = grid.shape
    z0, x0 = grid.origin
    row = np.clip((xz[:, 1] - z0) / grid.spacing, 0.0, nz - 1.0)
    column = np.clip((xz[:, 0] - x0) / grid.spacing, 0.0, nx - 1.0)
    row0 = np.minimum(row.astype(np.int64), nz - 2)
    column0 = np.minimum(column.astype(np.int64), nx - 2)
    return BilinearStencil(row0 * nx + column0, row - row0, column - column0, nx)


def gather_bilinear(
    values: np.ndarray, stencil: BilinearStencil, channel: np.ndarray | None = None
) -> np.ndarray:
    """Bilinear interpolation of node values at the points of a stencil.

    Args:
        values: shape (nz, nx, n_channels) or (nz, nx, n_channels, k).
        stencil: stencil of n points.
        channel: channel read at each point, int, shape (n,); None reads every
            channel of a (nz, nx, k) array at every point.

    Returns:
        shape (n,), (n, k) for 4D values, or (n, k) for `channel=None`.
    """
    if channel is None:
        table = values.reshape(-1, values.shape[-1])
        index = stencil.corner
        step_x, step_z = 1, stencil.row_stride
    else:
        n_channels = values.shape[2]
        table = values.reshape(-1, *values.shape[3:])
        index = stencil.corner * n_channels + channel
        step_x, step_z = n_channels, stencil.row_stride * n_channels
    weight_z, weight_x = stencil.weight_z, stencil.weight_x
    if table.ndim == 2:
        weight_z, weight_x = weight_z[:, None], weight_x[:, None]
    upper = table.take(index, axis=0)
    upper += weight_x * (table.take(index + step_x, axis=0) - upper)
    lower = table.take(index + step_z, axis=0)
    lower += weight_x * (table.take(index + step_z + step_x, axis=0) - lower)
    return upper + weight_z * (lower - upper)


@dataclass(frozen=True)
class _SweepAxis:
    """Geometry of the sweeps that march along one grid axis.

    "Line" is the grid line updated at once; "march" the axis the sweep moves
    along. Shapes have `n_sources` last.

    Attributes:
        march_offset: node minus source coordinate along the marching axis [m],
            (n_march, n_sources).
        line_offset_squared: square of the node minus source coordinate along the line
            [m^2], (n_line, n_sources).
        reference_gradient_line: `h s0` times the offset along the line [s m], (n_line,
            n_sources); divided by r it is h times the line component of grad T0.
        cell_step: `h *` slowness of the cells between consecutive lines [s],
            (n_march - 1, n_line - 1, 1).
        edge_step: `h *` slowness of the edges between consecutive lines [s],
            (n_march - 1, n_line, 1).
    """

    march_offset: np.ndarray
    line_offset_squared: np.ndarray
    reference_gradient_line: np.ndarray
    cell_step: np.ndarray
    edge_step: np.ndarray


def _sweep(
    excess_time: np.ndarray,
    frozen: np.ndarray,
    axis: _SweepAxis,
    direction: int,
    spacing: float,
    source_slowness: np.ndarray,
    min_improvement: float,
) -> bool:
    """One line sweep (module docstring); `excess_time` is updated in place.

    Args:
        excess_time, frozen: fields with the marching axis first, shape
            (n_march, n_line, n_sources).
        axis: geometry of the marching axis.
        direction: +1 or -1 along the marching axis.
        spacing: grid spacing h [m].
        source_slowness: s0 [s/m], shape (n_sources,).
        min_improvement: smallest decrease of a travel time that counts as an update [s].

    Returns:
        True if any node was updated.
    """
    n_march = excess_time.shape[0]
    lines = range(1, n_march) if direction > 0 else range(n_march - 2, -1, -1)
    frozen_lines = frozen.any(axis=(1, 2))
    lower, upper = slice(None, -1), slice(1, None)
    march_gradient_scale = spacing * direction * source_slowness
    updated = False
    for line in lines:
        behind = line - direction
        between = min(line, behind)
        march = axis.march_offset[line]
        inverse_distance = 1.0 / np.maximum(
            np.sqrt(axis.line_offset_squared + march * march), _TINY_DISTANCE
        )
        # h * grad T0 at the nodes of this line: component along the line, and along the march.
        gradient_line = axis.reference_gradient_line * inverse_distance
        gradient_march = march_gradient_scale * march * inverse_distance
        time_behind = excess_time[behind]

        # Ray from the axis neighbour A (along the edge A-C).
        corrected = time_behind - gradient_march
        candidate = corrected + axis.edge_step[between]
        cell_step = axis.cell_step[between]
        cell_step_squared = cell_step * cell_step
        diagonal_limit = cell_step * math.sqrt(0.5)
        # Rays from the edge between A and the diagonal neighbour B on either side.
        for side, here, there in ((1, upper, lower), (-1, lower, upper)):
            difference = time_behind[here] - time_behind[there] + side * gradient_line[here]
            clipped = np.minimum(np.maximum(difference, 0.0), diagonal_limit)
            through_edge = (
                corrected[here] + np.sqrt(cell_step_squared - clipped * clipped) + clipped
            ) - difference
            # For difference <= 0 the ray comes from A along the edge A-C, already a candidate.
            take = (difference > 0.0) & (through_edge < candidate[here])
            np.copyto(candidate[here], through_edge, where=take)

        better = candidate < excess_time[line] - min_improvement
        if frozen_lines[line]:
            better &= ~frozen[line]
        if better.any():
            np.copyto(excess_time[line], candidate, where=better)
            updated = True
    return updated


def solve_eikonal(
    slowness: np.ndarray,
    grid: Grid2D,
    sources_xz: np.ndarray,
    sweep_tolerance: float,
    max_sweeps: int,
) -> EikonalField:
    """Solve the eikonal equation for a batch of point sources (module docstring).

    Args:
        slowness: 1 / sound speed at the grid nodes [s/m], float64, shape (nz, nx),
            positive; at least 2 x 2 nodes.
        grid: the grid of `slowness`.
        sources_xz: source positions (x, z) [m], shape (n_sources, 2), inside the grid.
        sweep_tolerance: smallest decrease of a node's travel time that counts as
            an update, as a fraction of the time to cross one cell at the highest
            sound speed [dimensionless].
        max_sweeps: upper bound on the number of line sweeps.

    Returns:
        the converged `EikonalField`.

    Raises:
        RuntimeError: if the sweeps have not converged after `max_sweeps`.
    """
    nz, nx = grid.shape
    if slowness.shape != (nz, nx) or min(nz, nx) < 2:
        raise ValueError(f"slowness must have the grid shape {grid.shape}, at least 2 x 2")
    spacing = grid.spacing
    sources = np.asarray(sources_xz, dtype=np.float64)
    z, x = grid.axes()
    offset_z = z[:, None] - sources[None, :, 1]  # (nz, n_sources)
    offset_x = x[:, None] - sources[None, :, 0]  # (nx, n_sources)
    source_stencil = bilinear_stencil(grid, sources)
    source_slowness = gather_bilinear(slowness[:, :, None], source_stencil)[:, 0]

    distance = np.sqrt(offset_z[:, None, :] ** 2 + offset_x[None, :, :] ** 2)
    frozen = distance <= _FROZEN_RADIUS_CELLS * spacing
    # Straight path: through the slowest material (an upper bound) far from the source,
    # the trapezoid rule on the bilinear slowness map next to it.
    excess_time = distance * (slowness.max() - source_slowness)
    near_source = 0.5 * distance * (slowness[:, :, None] - source_slowness)
    np.copyto(excess_time, near_source, where=frozen)
    del distance, near_source

    cell = 0.25 * (slowness[1:, 1:] + slowness[1:, :-1] + slowness[:-1, 1:] + slowness[:-1, :-1])
    edge_z = 0.5 * (slowness[1:] + slowness[:-1])
    edge_x = 0.5 * (slowness[:, 1:] + slowness[:, :-1])
    axes = (
        _SweepAxis(
            march_offset=offset_z,
            line_offset_squared=offset_x * offset_x,
            reference_gradient_line=spacing * source_slowness * offset_x,
            cell_step=spacing * cell[:, :, None],
            edge_step=spacing * edge_z[:, :, None],
        ),
        _SweepAxis(
            march_offset=offset_x,
            line_offset_squared=offset_z * offset_z,
            reference_gradient_line=spacing * source_slowness * offset_z,
            cell_step=spacing * cell.T[:, :, None],
            edge_step=spacing * edge_x.T[:, :, None],
        ),
    )
    # Views with the marching axis first.
    views = ((excess_time, frozen), (excess_time.swapaxes(0, 1), frozen.swapaxes(0, 1)))
    min_improvement = sweep_tolerance * spacing * float(slowness.min())

    idle = 0
    n_sweeps = 0
    while idle < len(_SWEEPS):
        if n_sweeps == max_sweeps:
            raise RuntimeError(
                f"the eikonal sweeps did not converge in {max_sweeps} sweeps; "
                "raise BentRayConfig.max_sweeps"
            )
        march_axis, direction = _SWEEPS[n_sweeps % len(_SWEEPS)]
        updated = _sweep(
            *views[march_axis],
            axes[march_axis],
            direction,
            spacing,
            source_slowness,
            min_improvement,
        )
        idle = 0 if updated else idle + 1
        n_sweeps += 1

    return EikonalField(
        grid=grid,
        sources_xz=sources,
        source_slowness=source_slowness,
        excess_time=excess_time,
        n_sweeps=n_sweeps,
    )
