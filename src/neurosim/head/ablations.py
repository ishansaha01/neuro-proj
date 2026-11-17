"""Skull-effect switches: pure transforms of a `PlaneMedium` (ROADMAP D-005).

Each transform changes only the acoustic maps (`sound_speed`, `density`,
`attenuation`) inside the skull labels and returns a new `PlaneMedium`. Labels
and the latent fields (`porosity`, `hidden_c`) are kept, so the true geometry
and microstructure stay known.

The soft-tissue values a transform needs (scalp impedance, grey-matter
properties) are read from the medium itself: every soft tissue has one value
per label, so any pixel of that label gives it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from neurosim.head.materials import skull_mask
from neurosim.head.types import PlaneMedium, SkullEffects, Tissue


def _skull_pixels(pm: PlaneMedium) -> np.ndarray:
    in_skull = skull_mask(pm.labels)
    if not in_skull.any():
        raise ValueError("the plane medium contains no skull pixel")
    return in_skull


def _tissue_value(pm: PlaneMedium, values: np.ndarray, tissue: Tissue) -> np.float32:
    """Value of a per-label constant map in one soft tissue."""
    pixels = values[pm.labels == tissue]
    if pixels.size == 0:
        raise ValueError(f"the plane medium contains no {tissue.name} pixel")
    return pixels[0]


def homogenize_skull(pm: PlaneMedium) -> PlaneMedium:
    """Replace skull sound speed and density by their means over the skull pixels.

    Removes the aberration caused by skull heterogeneity (layers, porosity
    texture, hidden sound speed); the skull keeps its shape, its mean
    properties and its absorption map.
    """
    in_skull = _skull_pixels(pm)
    sound_speed = pm.sound_speed.copy()
    density = pm.density.copy()
    sound_speed[in_skull] = pm.sound_speed[in_skull].mean(dtype=np.float64)
    density[in_skull] = pm.density[in_skull].mean(dtype=np.float64)
    return replace(pm, sound_speed=sound_speed, density=density)


def zero_skull_absorption(pm: PlaneMedium) -> PlaneMedium:
    """Set the absorption prefactor to zero in the skull."""
    attenuation = pm.attenuation.copy()
    attenuation[_skull_pixels(pm)] = 0.0
    return replace(pm, attenuation=attenuation)


def match_skull_impedance(pm: PlaneMedium) -> PlaneMedium:
    """Set skull density so that its impedance equals that of the scalp.

    ``density = Z_scalp / sound_speed`` pixel by pixel, with ``Z_scalp`` the
    impedance [kg/(m^2 s)] of the scalp, the soft tissue the sound enters the
    skull from. The scalp/skull interface and all interfaces inside the skull
    then have a zero normal-incidence reflection coefficient; sound speed, and
    so travel time and refraction, are unchanged. The skull/CSF interface keeps
    the small scalp-to-CSF impedance step.
    """
    in_skull = _skull_pixels(pm)
    scalp_impedance = np.float64(_tissue_value(pm, pm.density, Tissue.SCALP)) * np.float64(
        _tissue_value(pm, pm.sound_speed, Tissue.SCALP)
    )
    density = pm.density.copy()
    density[in_skull] = scalp_impedance / pm.sound_speed[in_skull].astype(np.float64)
    return replace(pm, density=density)


def remove_skull(pm: PlaneMedium) -> PlaneMedium:
    """Give the skull pixels the acoustic properties of grey matter.

    This is the medium of the paired clean run. Labels are kept, so the skull
    geometry is still known.
    """
    in_skull = _skull_pixels(pm)
    maps = {}
    for name in ("sound_speed", "density", "attenuation"):
        values = getattr(pm, name).copy()
        values[in_skull] = _tissue_value(pm, values, Tissue.GREY)
        maps[name] = values
    return replace(pm, **maps)


def apply_skull_effects(pm: PlaneMedium, effects: SkullEffects) -> PlaneMedium:
    """Switch skull effects off as requested by `effects`.

    Order: homogenise (aberration off), zero the absorption (attenuation off),
    then match the impedance (reverberation off), so that the matched density
    follows the final sound speed. All switches on returns `pm` unchanged.
    """
    if not effects.aberration:
        pm = homogenize_skull(pm)
    if not effects.attenuation:
        pm = zero_skull_absorption(pm)
    if not effects.reverberation:
        pm = match_skull_impedance(pm)
    return pm
