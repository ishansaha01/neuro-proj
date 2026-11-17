"""Contract types for the CT model (see docs/ARCHITECTURE.md, section 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurosim.common.grid import Grid3D


@dataclass(frozen=True)
class CtVolume:
    """Reconstructed CT volume.

    Attributes:
        hu: Hounsfield units, float32, indexed [ix, iy, iz] on `grid` (head frame).
        grid: grid of the reconstructed volume; may be a slab of the head grid.
        meta: scan settings used (kVp, filtration, mAs, kernel, geometry) and provenance.
    """

    hu: np.ndarray
    grid: Grid3D
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HuMapping:
    """Constants of the clinical linear HU -> acoustics map (Aubry 2003, Marquet 2009).

    Bone porosity is estimated as `phi = 1 - hu / hu_bone` (clipped to [0, 1]); sound
    speed and density are linear in `1 - phi`, absorption follows a power of `phi`.

    Attributes:
        hu_bone: HU of fully dense cortical bone at the scan settings.
        c_water, c_bone: sound-speed end points [m/s].
        rho_water, rho_bone: density end points [kg/m^3].
        alpha_min, alpha_max: absorption end points [dB/(cm MHz^y)].
        alpha_exponent: exponent of porosity in the absorption law.
    """

    hu_bone: float
    c_water: float
    c_bone: float
    rho_water: float
    rho_bone: float
    alpha_min: float
    alpha_max: float
    alpha_exponent: float
