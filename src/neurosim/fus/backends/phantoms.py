"""Layered plane media without anatomy, used to validate the backends (unit tests and evals).

Real media come from `neurosim.head`; these phantoms only fill the acoustic
maps of a `PlaneMedium` and leave every other field at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid2D, PlaneSpec
from neurosim.head.types import PlaneMedium

# The phantoms are not embedded in a head; any valid plane serves.
_PLANE = PlaneSpec(origin=(0.0, 0.0, 0.0), u_lateral=(1.0, 0.0, 0.0), u_depth=(0.0, 0.0, -1.0))


@dataclass(frozen=True)
class Material:
    """Acoustic properties of one homogeneous material.

    Attributes:
        sound_speed: [m/s].
        density: [kg/m^3].
        attenuation: power-law prefactor alpha0 [dB/(cm MHz^y)].
    """

    sound_speed: float
    density: float
    attenuation: float = 0.0

    def impedance_matched(self, sound_speed: float) -> "Material":
        """Lossless material with the given sound speed [m/s] and this material's rho c."""
        return Material(sound_speed, self.density * self.sound_speed / sound_speed)


def slab_medium(
    grid: Grid2D,
    background: Material,
    slab: Material,
    z_top: float,
    thickness: float | np.ndarray,
    alpha_power: float,
) -> PlaneMedium:
    """A slab of one material in a homogeneous background.

    A pixel belongs to the slab when its centre depth z satisfies
    `z_top <= z < z_top + thickness`. With `z_top` and `thickness` on pixel
    faces the slab is exactly `thickness` thick.

    Args:
        grid: medium grid; the row at z = 0 is the array face.
        background: material outside the slab.
        slab: material of the slab.
        z_top: depth of the slab's upper face [m].
        thickness: slab thickness [m]; a scalar (flat slab) or one value per
            grid column, shape (nx,) (a slab with a thickness profile). Zero
            gives the homogeneous background.
        alpha_power: power-law exponent y of the absorption model.

    Returns:
        `PlaneMedium` with float32 maps of shape (nz, nx).
    """
    z, x = grid.axes()
    thickness = np.broadcast_to(np.asarray(thickness, dtype=np.float64), x.shape)
    in_slab = (z[:, None] >= z_top) & (z[:, None] < z_top + thickness[None, :])

    def acoustic_map(name: str) -> np.ndarray:
        return np.where(in_slab, getattr(slab, name), getattr(background, name)).astype(np.float32)

    zeros = np.zeros(grid.shape, dtype=np.float32)
    return PlaneMedium(
        grid=grid,
        plane=_PLANE,
        labels=np.zeros(grid.shape, dtype=np.uint8),
        porosity=zeros,
        hidden_c=zeros,
        sound_speed=acoustic_map("sound_speed"),
        density=acoustic_map("density"),
        attenuation=acoustic_map("attenuation"),
        cbv_baseline=zeros,
        activation=zeros,
        blood_velocity=np.zeros((2, *grid.shape), dtype=np.float32),
        alpha_power=alpha_power,
    )


def homogeneous_medium(grid: Grid2D, material: Material, alpha_power: float) -> PlaneMedium:
    """A homogeneous medium of one material on `grid` (a slab phantom of zero thickness)."""
    return slab_medium(grid, material, material, z_top=0.0, thickness=0.0, alpha_power=alpha_power)
