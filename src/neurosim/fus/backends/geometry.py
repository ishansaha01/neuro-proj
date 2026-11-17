"""Input checks shared by the Green's-bank backends."""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.types import Transducer
from neurosim.head.types import PlaneMedium

# Relative tolerance on the uniform spacing of the frequency list [dimensionless].
_FREQUENCY_SPACING_TOLERANCE = 1e-6


def element_centres(transducer: Transducer) -> np.ndarray:
    """Element centre points [m], shape (n_elements, 2), columns (x, z); z = 0 is the array face."""
    x = transducer.element_x()
    return np.stack((x, np.zeros_like(x)), axis=-1)


def checked_frequencies(freqs: np.ndarray) -> np.ndarray:
    """Return the frequency list [Hz] as float64, shape (nf,), after checking the bank contract.

    `GreensBank.freqs` must be positive, ascending and uniformly spaced.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError(f"freqs must be a non-empty 1D array, got shape {freqs.shape}")
    if not np.all(freqs > 0.0):
        raise ValueError("freqs must be positive [Hz]")
    steps = np.diff(freqs)
    if steps.size and not np.all(steps > 0.0):
        raise ValueError("freqs must be strictly ascending")
    if steps.size > 1 and np.ptp(steps) > _FREQUENCY_SPACING_TOLERANCE * steps.mean():
        raise ValueError("freqs must be uniformly spaced")
    return freqs


def inside_grid(grid: Grid2D, xz: np.ndarray) -> np.ndarray:
    """True where plane points (x, z) [m] lie within the outer faces of the outermost pixels."""
    z0, x0 = grid.origin
    iz = (xz[..., 1] - z0) / grid.spacing
    ix = (xz[..., 0] - x0) / grid.spacing
    return (iz >= -0.5) & (iz <= grid.shape[0] - 0.5) & (ix >= -0.5) & (ix <= grid.shape[1] - 0.5)


def check_medium(medium: PlaneMedium) -> None:
    """Check that the acoustic maps of a plane medium are well formed.

    sound_speed [m/s] and density [kg/m^3] must be positive, attenuation
    [dB/(cm MHz^y)] non-negative, all of shape `medium.grid.shape`.
    """
    for name in ("sound_speed", "density", "attenuation"):
        values = getattr(medium, name)
        if values.shape != medium.grid.shape:
            raise ValueError(
                f"medium.{name} has shape {values.shape}, expected {medium.grid.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"medium.{name} must be finite")
    if not (np.all(medium.sound_speed > 0.0) and np.all(medium.density > 0.0)):
        raise ValueError("medium.sound_speed and medium.density must be positive")
    if np.any(medium.attenuation < 0.0):
        raise ValueError("medium.attenuation must be non-negative")


def check_geometry(medium: PlaneMedium, transducer: Transducer, roi: Grid2D) -> None:
    """Check that the array face and the ROI lie inside the medium grid.

    Every element centre (z = 0) and every ROI point must be inside the medium
    grid, and the ROI must lie strictly in front of the array (z > 0).
    """
    if not np.all(inside_grid(medium.grid, element_centres(transducer))):
        raise ValueError("every array element (z = 0) must lie inside the medium grid")
    roi_points = roi.points()
    if not np.all(inside_grid(medium.grid, roi_points)):
        raise ValueError("the ROI grid must lie inside the medium grid")
    if not np.all(roi_points[..., 1] > 0.0):
        raise ValueError("the ROI grid must lie in front of the array face (z > 0)")
