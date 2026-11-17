"""Contract types for the head model (see docs/ARCHITECTURE.md, section 4).

These types are the interface between `head` and the packages that consume it
(`fus`, `ct`, `data`). Change them only together with docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from neurosim.common.grid import Grid2D, Grid3D, PlaneSpec


class Tissue(IntEnum):
    """Tissue label stored in `labels` arrays.

    The value equals the number of nested surfaces (see `SURFACES`) that enclose
    the point, so labels follow from the signed-distance fields by counting.
    """

    BACKGROUND = 0  # coupling medium outside the head (water / gel)
    SCALP = 1
    SKULL_OUTER = 2  # outer table (cortical bone)
    DIPLOE = 3  # trabecular layer
    SKULL_INNER = 4  # inner table (cortical bone)
    CSF = 5
    GREY = 6
    WHITE = 7


SKULL_LABELS: tuple[int, ...] = (Tissue.SKULL_OUTER, Tissue.DIPLOE, Tissue.SKULL_INNER)
BRAIN_LABELS: tuple[int, ...] = (Tissue.GREY, Tissue.WHITE)

# Nested closed surfaces, outermost first. `sdf[k]` is the signed distance [m] to
# SURFACES[k], negative inside. A point's label is the count of surfaces with sdf < 0.
SURFACES: tuple[str, ...] = (
    "scalp_outer",
    "skull_outer",
    "outer_table_inner",
    "diploe_inner",
    "skull_inner",
    "brain_outer",
    "white_outer",
)


def labels_from_sdf(sdf: np.ndarray) -> np.ndarray:
    """Tissue labels from nested signed-distance fields.

    Args:
        sdf: signed distances [m], shape (len(SURFACES), ...), negative inside.

    Returns:
        uint8 labels with the trailing shape of `sdf`, values in `Tissue`.
    """
    return np.count_nonzero(sdf < 0.0, axis=0).astype(np.uint8)


@dataclass(frozen=True)
class SkullEffects:
    """Switches for the three skull effects (ROADMAP D-005). All on = real skull."""

    aberration: bool = True  # False: skull sound speed and density set to their means
    attenuation: bool = True  # False: bone absorption set to zero
    reverberation: bool = True  # False: skull density set so impedance matches soft tissue


@dataclass(frozen=True)
class HeadVolume:
    """Ground-truth head on a 3D grid. All arrays are indexed [ix, iy, iz].

    Latent fields (everything else is derived from these by `head.materials`):
        sdf: signed distance to each surface in SURFACES [m], float32, (7, nx, ny, nz).
        porosity: skull pore volume fraction in [0, 1], float32; 0 outside the skull.
        hidden_c: sound-speed component invisible to X-rays [m/s], float32; 0 outside the skull.

    Derived fields:
        labels: `Tissue` values, uint8.
        sound_speed: [m/s], float32.
        density: [kg/m^3], float32.
        attenuation: power-law prefactor alpha0 [dB/(cm MHz^y)], float32.
        material_fractions: volume fractions of `common.constants.BASIS_MATERIALS`,
            float32, (4, nx, ny, nz), summing to 1 in every voxel. Input to the CT model.

    CBV model:
        cbv_baseline: resting blood volume fraction (0..1), float32; 0 outside the brain.
        activation: spatial pattern of the task response in [0, 1], float32.
        blood_velocity: blood velocity vector in the head frame [m/s], float32, (3, nx, ny, nz).

    alpha_power: global power-law exponent y of the absorption model.
    """

    grid: Grid3D
    sdf: np.ndarray
    porosity: np.ndarray
    hidden_c: np.ndarray
    labels: np.ndarray
    sound_speed: np.ndarray
    density: np.ndarray
    attenuation: np.ndarray
    material_fractions: np.ndarray
    cbv_baseline: np.ndarray
    activation: np.ndarray
    blood_velocity: np.ndarray
    alpha_power: float


@dataclass(frozen=True)
class PlaneMedium:
    """Head properties on a fine 2D grid in one imaging plane. Arrays are [iz, ix].

    Fields have the same meaning and units as in `HeadVolume`. `blood_velocity`
    is the in-plane projection, float32, shape (2, nz, nx), components (vx, vz).
    """

    grid: Grid2D
    plane: PlaneSpec
    labels: np.ndarray
    porosity: np.ndarray
    hidden_c: np.ndarray
    sound_speed: np.ndarray
    density: np.ndarray
    attenuation: np.ndarray
    cbv_baseline: np.ndarray
    activation: np.ndarray
    blood_velocity: np.ndarray
    alpha_power: float
