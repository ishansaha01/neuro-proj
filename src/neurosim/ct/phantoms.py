"""Simple test objects as basis-material fraction volumes.

Cylinders along z with partial-volume (soft) edges, used by the CT unit tests
and evals. Pure array builders; nothing here is used by a scan of a head.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.grid import Grid3D


def radial_distance(grid: Grid3D, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """In-plane distance of every pixel centre from a point.

    Args:
        grid: grid whose (x, y) sampling is used.
        center: head-frame (x, y) of the point [m].

    Returns:
        Distance [m], float64, shape (nx, ny).
    """
    x, y, _ = grid.axes()
    return np.hypot(x[:, np.newaxis] - center[0], y[np.newaxis, :] - center[1])


def disc_coverage(
    grid: Grid3D, radius: float, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Fraction of each in-plane pixel covered by a disc.

    The edge is a linear ramp one pixel wide centred on the disc boundary, which
    approximates the covered area of a boundary pixel.

    Args:
        grid: grid whose (x, y) sampling is used.
        radius: disc radius [m].
        center: head-frame (x, y) of the disc centre [m].

    Returns:
        Coverage in [0, 1], float64, shape (nx, ny).
    """
    return np.clip((radius - radial_distance(grid, center)) / grid.spacing + 0.5, 0.0, 1.0)


def layered_fractions(grid: Grid3D, layers: Sequence[tuple[str, np.ndarray]]) -> np.ndarray:
    """Paint in-plane coverage maps over an air background, the same in every slice.

    Each layer replaces what lies under it in proportion to its coverage, so
    the fractions always sum to 1.

    Args:
        grid: grid of the volume.
        layers: (material name from `BASIS_MATERIALS`, coverage of shape (nx, ny))
            pairs, painted in order.

    Returns:
        Volume fractions, float32, shape (4, nx, ny, nz).
    """
    fractions = np.zeros((len(BASIS_MATERIALS), *grid.shape[:2]))
    fractions[BASIS_MATERIALS.index("air")] = 1.0
    for material, coverage in layers:
        fractions *= 1.0 - coverage
        fractions[BASIS_MATERIALS.index(material)] += coverage
    volume = np.repeat(fractions[..., np.newaxis], grid.shape[2], axis=-1)
    return volume.astype(np.float32)
