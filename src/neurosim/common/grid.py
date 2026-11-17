"""Grids, the imaging-plane embedding and interpolation, shared by head, fus and ct.

Frames (docs/ARCHITECTURE.md section 1):

- Head frame (3D): right-handed, metres, volumes indexed ``[ix, iy, iz]``.
- Imaging-plane frame (2D): ``x`` lateral along the array, ``z`` depth into the
  head, metres. Plane maps are indexed ``[iz, ix]``, but plane *points* are
  written ``(x, z)``.

Grid coordinates are voxel (pixel) centres.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage

# Tolerance on unit length and orthogonality of the plane vectors [dimensionless].
_PLANE_TOLERANCE = 1e-6


def _float_tuple(values: ArrayLike, length: int, what: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != length:
        raise ValueError(f"{what} must have {length} entries, got {len(result)}")
    return result


def _shape_tuple(values: ArrayLike, length: int) -> tuple[int, ...]:
    result = tuple(operator.index(v) for v in values)
    if len(result) != length or min(result) < 1:
        raise ValueError(f"shape must be {length} positive integers, got {values!r}")
    return result


def _points_array(points: ArrayLike, n_coordinates: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != n_coordinates:
        raise ValueError(f"points must have shape (..., {n_coordinates}), got {points.shape}")
    return points


def _checked_spacing(spacing: float) -> float:
    spacing = float(spacing)
    if not spacing > 0.0:
        raise ValueError(f"spacing must be > 0 m, got {spacing}")
    return spacing


@dataclass(frozen=True)
class Grid3D:
    """Regular isotropic voxel grid in the head frame.

    shape: number of voxels ``(nx, ny, nz)``.
    spacing: voxel size [m].
    origin: head-frame coordinate of the centre of voxel ``[0, 0, 0]`` [m].
    """

    shape: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape_tuple(self.shape, 3))
        object.__setattr__(self, "spacing", _checked_spacing(self.spacing))
        object.__setattr__(self, "origin", _float_tuple(self.origin, 3, "origin"))

    @classmethod
    def centered(cls, extent: tuple[float, float, float], spacing: float) -> Grid3D:
        """Build a grid that is symmetric about the head-frame origin.

        extent: full size of the gridded box along x, y, z [m], measured between
            the outer faces of the outermost voxels. Each axis gets
            ``round(extent / spacing)`` voxels (at least one).
        spacing: voxel size [m].
        """
        spacing = _checked_spacing(spacing)
        shape = tuple(max(1, round(float(size) / spacing)) for size in extent)
        origin = tuple(-0.5 * (n - 1) * spacing for n in shape)
        return cls(shape=shape, spacing=spacing, origin=origin)

    @property
    def n_voxels(self) -> int:
        """Total number of voxels."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the voxel-centre coordinates along x, y and z [m], as 1D float64 arrays."""
        x, y, z = (o + self.spacing * np.arange(n) for o, n in zip(self.origin, self.shape))
        return x, y, z

    def coords(self) -> np.ndarray:
        """Return all voxel-centre coordinates [m], shape ``(nx, ny, nz, 3)``."""
        return np.stack(np.meshgrid(*self.axes(), indexing="ij"), axis=-1)

    def world_to_index(self, points: ArrayLike) -> np.ndarray:
        """Convert head-frame points to fractional voxel indices.

        points: head-frame coordinates [m], shape ``(..., 3)``.
        Returns fractional indices ``(ix, iy, iz)``, shape ``(..., 3)``; a voxel
        centre maps to its integer index.
        """
        return (_points_array(points, 3) - np.asarray(self.origin)) / self.spacing


@dataclass(frozen=True)
class Grid2D:
    """Regular isotropic pixel grid in the imaging-plane frame.

    shape: number of pixels ``(nz, nx)``; rows are depth.
    spacing: pixel size [m].
    origin: plane coordinate ``(z0, x0)`` of the centre of pixel ``[0, 0]`` [m].
    """

    shape: tuple[int, int]
    spacing: float
    origin: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape_tuple(self.shape, 2))
        object.__setattr__(self, "spacing", _checked_spacing(self.spacing))
        object.__setattr__(self, "origin", _float_tuple(self.origin, 2, "origin"))

    @classmethod
    def from_extent(
        cls, x_range: tuple[float, float], z_range: tuple[float, float], spacing: float
    ) -> Grid2D:
        """Build a grid whose pixel centres span the given ranges.

        x_range: lateral coordinates ``(x_first, x_last)`` of the first and last
            pixel centres [m].
        z_range: depth coordinates ``(z_first, z_last)`` of the first and last
            pixel centres [m].
        spacing: pixel size [m].

        Each axis gets ``round((last - first) / spacing) + 1`` pixels starting
        exactly at ``first``; the last centre equals ``last`` when the range is
        a whole number of spacings.
        """
        spacing = _checked_spacing(spacing)
        (x_first, x_last), (z_first, z_last) = x_range, z_range
        if x_last < x_first or z_last < z_first:
            raise ValueError(f"ranges must be increasing, got x {x_range}, z {z_range}")
        shape = (
            round((z_last - z_first) / spacing) + 1,
            round((x_last - x_first) / spacing) + 1,
        )
        return cls(shape=shape, spacing=spacing, origin=(z_first, x_first))

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the pixel-centre coordinates ``(z, x)`` [m], as 1D float64 arrays."""
        z, x = (o + self.spacing * np.arange(n) for o, n in zip(self.origin, self.shape))
        return z, x

    def points(self) -> np.ndarray:
        """Return all pixel-centre points [m], shape ``(nz, nx, 2)``, last axis ``(x, z)``."""
        z, x = self.axes()
        z_map, x_map = np.meshgrid(z, x, indexing="ij")
        return np.stack((x_map, z_map), axis=-1)


@dataclass(frozen=True)
class PlaneSpec:
    """An imaging plane embedded in the head frame.

    origin: head-frame position of the array centre (plane ``x = 0, z = 0``) [m].
    u_lateral: unit vector along the array (plane ``+x``) in the head frame.
    u_depth: unit vector into the head (plane ``+z``) in the head frame.

    The two vectors must have unit length and be orthogonal, each within 1e-6.
    """

    origin: tuple[float, float, float]
    u_lateral: tuple[float, float, float]
    u_depth: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("origin", "u_lateral", "u_depth"):
            object.__setattr__(self, name, _float_tuple(getattr(self, name), 3, name))
        for name in ("u_lateral", "u_depth"):
            norm = float(np.linalg.norm(getattr(self, name)))
            if abs(norm - 1.0) > _PLANE_TOLERANCE:
                raise ValueError(f"{name} must have unit length, got norm {norm}")
        dot = float(np.dot(self.u_lateral, self.u_depth))
        if abs(dot) > _PLANE_TOLERANCE:
            raise ValueError(f"u_lateral and u_depth must be orthogonal, got dot product {dot}")


def plane_to_world(plane: PlaneSpec, xz: ArrayLike) -> np.ndarray:
    """Map imaging-plane points to the head frame.

    xz: plane points ``(x, z)`` [m], shape ``(..., 2)``.
    Returns head-frame points [m], shape ``(..., 3)``.
    """
    xz = _points_array(xz, 2)
    return (
        np.asarray(plane.origin)
        + xz[..., :1] * np.asarray(plane.u_lateral)
        + xz[..., 1:] * np.asarray(plane.u_depth)
    )


def plane_points(plane: PlaneSpec, grid: Grid2D) -> np.ndarray:
    """Return the head-frame position of every pixel of a plane grid [m], shape ``(nz, nx, 3)``."""
    return plane_to_world(plane, grid.points())


def _sample(
    values: np.ndarray, grid_shape: tuple[int, ...], index: np.ndarray, order: int, fill: float
) -> np.ndarray:
    """Interpolate ``values`` at fractional indices ``index`` (shape ``(..., ndim)``).

    ``values`` has shape ``grid_shape`` or ``(c, *grid_shape)``. A point is
    inside the grid when it lies within the outer faces of the outermost
    voxels (index in ``[-0.5, n - 0.5]``); between the outermost centres and
    those faces the edge value is used.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 (nearest) or 1 (linear), got {order}")
    ndim = len(grid_shape)
    values = np.asarray(values)
    if values.ndim not in (ndim, ndim + 1) or values.shape[-ndim:] != tuple(grid_shape):
        raise ValueError(
            f"values of shape {values.shape} do not match grid shape {tuple(grid_shape)}"
        )
    # map_coordinates needs at least one point axis; a single point gets one for the call.
    coordinates = np.moveaxis(np.atleast_2d(index), -1, 0)
    upper = np.asarray(grid_shape).reshape((ndim,) + (1,) * (coordinates.ndim - 1)) - 0.5
    outside = np.any((coordinates < -0.5) | (coordinates > upper), axis=0)
    # Linear interpolation of integer data (for example labels) is fractional.
    dtype = values.dtype if order == 0 else np.result_type(values.dtype, np.float32)
    channels = values if values.ndim == ndim + 1 else values[np.newaxis]
    sampled = np.empty((channels.shape[0],) + coordinates.shape[1:], dtype=dtype)
    for channel, out in zip(channels, sampled):
        ndimage.map_coordinates(channel, coordinates, output=out, order=order, mode="nearest")
        out[outside] = fill
    sampled = sampled.reshape((channels.shape[0],) + index.shape[:-1])
    return sampled if values.ndim == ndim + 1 else sampled[0]


def sample_volume(
    volume: np.ndarray,
    grid: Grid3D,
    points: ArrayLike,
    *,
    order: int = 1,
    fill: float = 0.0,
) -> np.ndarray:
    """Sample a head-frame volume at arbitrary points.

    volume: values on ``grid``, shape ``(nx, ny, nz)`` or ``(c, nx, ny, nz)``.
    grid: the grid the volume lives on.
    points: head-frame points [m], shape ``(..., 3)``.
    order: 0 for nearest neighbour (keeps the dtype, use for labels), 1 for
        trilinear.
    fill: value returned for points outside the grid (beyond the outer faces
        of the outermost voxels), in the unit of ``volume``.

    Returns the sampled values, shape ``(...)`` or ``(c, ...)``, in the unit of
    ``volume``.
    """
    return _sample(volume, grid.shape, grid.world_to_index(points), order, fill)


def sample_plane_map(
    values: np.ndarray,
    grid: Grid2D,
    xz: ArrayLike,
    *,
    order: int = 1,
    fill: float = 0.0,
) -> np.ndarray:
    """Sample an imaging-plane map at arbitrary plane points.

    values: real or complex values on ``grid``, shape ``(nz, nx)`` or ``(c, nz, nx)``.
    grid: the grid the map lives on.
    xz: plane points ``(x, z)`` [m], shape ``(..., 2)``.
    order: 0 for nearest neighbour, 1 for bilinear.
    fill: value returned for points outside the grid, in the unit of ``values``.

    Returns the sampled values, shape ``(...)`` or ``(c, ...)``, in the unit of
    ``values``.
    """
    xz = _points_array(xz, 2)
    z0, x0 = grid.origin
    index = np.stack(((xz[..., 1] - z0) / grid.spacing, (xz[..., 0] - x0) / grid.spacing), axis=-1)
    return _sample(values, grid.shape, index, order, fill)
