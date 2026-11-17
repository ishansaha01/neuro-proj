"""Procedural head anatomy: nested closed surfaces around the head centre.

Every surface of `types.SURFACES` is star-shaped about the head-frame origin and
described by its radius as a function of direction. The scalp is a deformed
ellipsoid; each inner surface is the previous one minus a strictly positive,
smooth thickness map, so the nesting cannot break.

The signed-distance fields built here are **radial**: ``sdf[k](p) = |p| -
R_k(p / |p|)``, the distance to surface ``k`` along the ray from the head
centre, negative inside. They are not exact Euclidean distances; the two differ
by the cosine of the angle between the ray and the surface normal (a few
percent for these shapes). Likewise a "thickness" is measured along the ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid3D
from neurosim.head.config import AnatomyConfig
from neurosim.head.fields import SphereFields
from neurosim.head.types import SURFACES

# Random plane waves per sphere field. Numerical: 32 modes make the sum close to
# Gaussian while keeping the cost per voxel low.
_N_MODES = 32
# The smooth random fields are squashed to (-_FIELD_BOUND, _FIELD_BOUND) standard
# deviations, so thicknesses and the head size are bounded.
_FIELD_BOUND = 3.0
# Truncation of the per-axis size factor, in standard deviations.
_SIZE_BOUND = 2.5
# Voxels processed at once when filling a volume (bounds the temporary memory).
_CHUNK_VOXELS = 20_000

# Order of the fields in `HeadGeometry.fields`.
_SHAPE, _SCALP, _SKULL, _OUTER_TABLE, _INNER_TABLE, _CSF, _GREY = range(7)


@dataclass(frozen=True)
class HeadGeometry:
    """One drawn head shape: everything needed to evaluate the surfaces.

    semi_axes: scalp semi-axes of this head along x, y, z [m].
    fields: seven smooth random fields on the sphere (scalp shape and six thicknesses).
    cfg: the anatomy configuration.
    """

    semi_axes: tuple[float, float, float]
    fields: SphereFields
    cfg: AnatomyConfig


def unit_vector(direction: tuple[float, float, float]) -> np.ndarray:
    """Return `direction` scaled to unit length, float64, shape (3,)."""
    vector = np.asarray(direction, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def max_scalp_radius(cfg: AnatomyConfig) -> tuple[float, float, float]:
    """Upper bound of the scalp extent along x, y, z for any seed [m]."""
    bound = np.exp(_SIZE_BOUND * cfg.size_variation + _FIELD_BOUND * cfg.shape_variation)
    return tuple(float(axis * bound) for axis in cfg.semi_axes)


def draw_geometry(cfg: AnatomyConfig, rng: np.random.Generator) -> HeadGeometry:
    """Draw the random shape of one head from the `anatomy` stream."""
    size_factors = np.exp(
        cfg.size_variation * np.clip(rng.standard_normal(3), -_SIZE_BOUND, _SIZE_BOUND)
    )
    semi_axes = tuple(float(a * s) for a, s in zip(cfg.semi_axes, size_factors))
    correlations = (cfg.shape_correlation,) + (cfg.thickness_correlation,) * 6
    geometry = HeadGeometry(semi_axes, SphereFields.draw(rng, correlations, _N_MODES), cfg)

    largest_thickness = (
        cfg.scalp_thickness * np.exp(_FIELD_BOUND * cfg.scalp_variation)
        + max(cfg.skull_thickness_vault, cfg.skull_thickness_temporal)
        * np.exp(_FIELD_BOUND * cfg.skull_variation)
        + cfg.csf_thickness * np.exp(_FIELD_BOUND * cfg.csf_variation)
        + cfg.grey_thickness * np.exp(_FIELD_BOUND * cfg.grey_variation)
    )
    smallest_radius = min(semi_axes) * np.exp(-_FIELD_BOUND * cfg.shape_variation)
    if largest_thickness >= smallest_radius:
        raise ValueError(
            f"tissue layers (up to {largest_thickness:.4f} m) do not fit inside the "
            f"smallest scalp radius ({smallest_radius:.4f} m)"
        )
    return geometry


def _bounded(field: np.ndarray) -> np.ndarray:
    """Squash a unit-variance field smoothly into (-_FIELD_BOUND, _FIELD_BOUND)."""
    return _FIELD_BOUND * np.tanh(field / _FIELD_BOUND)


def _saturate(x: np.ndarray) -> np.ndarray:
    """Smooth version of min(x, 1) for x >= 0: close to x below 1, tends to 1 above."""
    return x / (1.0 + x**4) ** 0.25


def surface_radii(geometry: HeadGeometry, directions: np.ndarray) -> np.ndarray:
    """Radius of every surface along given directions.

    directions: unit vectors from the head centre, shape (n, 3).
    Returns radii [m], float32, shape (len(SURFACES), n), strictly decreasing
    along the first axis (outermost surface first).
    """
    cfg = geometry.cfg
    directions = np.asarray(directions, dtype=np.float32)
    g = _bounded(geometry.fields.evaluate(directions))

    inverse_axes = (1.0 / np.asarray(geometry.semi_axes)).astype(np.float32)
    ellipsoid_radius = 1.0 / np.linalg.norm(directions * inverse_axes, axis=1)
    scalp_outer = ellipsoid_radius * np.exp(cfg.shape_variation * g[_SHAPE])

    # Thin region around the two temporal windows (mirror images in x). Each
    # lobe exp(-(1 - cos angle) / width^2) is exp(-angle^2 / (2 width^2)) for
    # small angles and smooth over the whole sphere; the two are combined as a
    # smooth union so that the weight stays in [0, 1].
    window = unit_vector(cfg.temporal_window_direction).astype(np.float32)
    mirrored = window * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    right, left = (
        np.exp((directions @ centre - 1.0) / cfg.temporal_window_width**2)
        for centre in (window, mirrored)
    )
    temporal_weight = 1.0 - (1.0 - right) * (1.0 - left)
    skull_median = (
        cfg.skull_thickness_vault
        + (cfg.skull_thickness_temporal - cfg.skull_thickness_vault) * temporal_weight
    )

    scalp = cfg.scalp_thickness * np.exp(cfg.scalp_variation * g[_SCALP])
    skull = skull_median * np.exp(cfg.skull_variation * g[_SKULL])
    outer_cap = cfg.outer_table_max_fraction * skull
    inner_cap = cfg.inner_table_max_fraction * skull
    outer_table = outer_cap * _saturate(
        cfg.outer_table_thickness * np.exp(cfg.table_variation * g[_OUTER_TABLE]) / outer_cap
    )
    inner_table = inner_cap * _saturate(
        cfg.inner_table_thickness * np.exp(cfg.table_variation * g[_INNER_TABLE]) / inner_cap
    )
    diploe = skull - outer_table - inner_table
    csf = cfg.csf_thickness * np.exp(cfg.csf_variation * g[_CSF])
    grey = cfg.grey_thickness * np.exp(cfg.grey_variation * g[_GREY])

    thicknesses = np.stack((scalp, outer_table, diploe, inner_table, csf, grey))
    radii = np.empty((len(SURFACES), directions.shape[0]), dtype=np.float32)
    radii[0] = scalp_outer
    radii[1:] = scalp_outer - np.cumsum(thicknesses, axis=0)
    return radii


def radial_sdf(geometry: HeadGeometry, grid: Grid3D) -> np.ndarray:
    """Radial signed distance to every surface on a grid.

    Returns float32 distances [m] of shape ``(len(SURFACES), nx, ny, nz)``,
    negative inside the surface (see the module docstring for "radial").
    """
    x_axis, y_axis, z_axis = grid.axes()
    nx, ny, nz = grid.shape
    sdf = np.empty((len(SURFACES), nx, ny, nz), dtype=np.float32)
    y_plane, z_plane = np.meshgrid(y_axis, z_axis, indexing="ij")
    slab = max(1, _CHUNK_VOXELS // (ny * nz))
    for start in range(0, nx, slab):
        stop = min(start + slab, nx)
        points = np.empty((stop - start, ny, nz, 3), dtype=np.float32)
        points[..., 0] = x_axis[start:stop, None, None]
        points[..., 1] = y_plane
        points[..., 2] = z_plane
        points = points.reshape(-1, 3)
        radius = np.linalg.norm(points, axis=1)
        # The direction is undefined at the head centre; any direction gives the
        # same (deep white matter) result there.
        centre = radius < np.finfo(np.float32).tiny
        directions = points / np.where(centre, 1.0, radius)[:, None]
        directions[centre] = (0.0, 0.0, 1.0)
        sdf[:, start:stop] = (radius - surface_radii(geometry, directions)).reshape(
            len(SURFACES), stop - start, ny, nz
        )
    return sdf
