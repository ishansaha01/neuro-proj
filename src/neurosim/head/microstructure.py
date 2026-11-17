"""Skull microstructure: the porosity field and the sound speed hidden from X-rays."""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from neurosim.head.config import MicrostructureConfig
from neurosim.head.materials import SkullMixing, skull_mask, skull_sound_speed_from_porosity
from neurosim.head.types import SURFACES

_OUTER_TABLE_INNER = SURFACES.index("outer_table_inner")
_DIPLOE_INNER = SURFACES.index("diploe_inner")

# Search range of the hidden standard deviation, as multiples of the closed-form
# value sqrt(var * (1 - r2) / r2) that holds without bounds or chance correlation.
_HIDDEN_STD_BRACKET = (1.0e-3, 50.0)
# The sound-speed bounds are pulled inward by this much [m/s] so that they still
# hold after the maps are rounded to float32 (spacing 2.4e-4 m/s near 3000 m/s).
_ROUNDING_MARGIN = 2.0e-3


def skull_porosity(
    sdf: np.ndarray, labels: np.ndarray, texture: np.ndarray, cfg: MicrostructureConfig
) -> np.ndarray:
    """Porosity of the skull: a three-layer profile times a random texture.

    The profile is low in the outer table, high in the diploe and low in the
    inner table, with smooth transitions of width `cfg.layer_transition` at the
    two table/diploe surfaces. It is multiplied by a log-normal texture of mean
    1, clipped to [0, 1] and set to zero outside the skull.

    sdf: radial signed distances [m], shape (len(SURFACES), ...).
    labels: `Tissue` labels, shape (...).
    texture: zero-mean, unit-variance random field, shape (...).

    Returns the pore volume fraction, float32, shape (...).
    """
    past_outer_table = special.expit(-sdf[_OUTER_TABLE_INNER] / cfg.layer_transition)
    past_diploe = special.expit(-sdf[_DIPLOE_INNER] / cfg.layer_transition)
    profile = (
        cfg.porosity_outer_table
        + (cfg.porosity_diploe - cfg.porosity_outer_table) * past_outer_table
        + (cfg.porosity_inner_table - cfg.porosity_diploe) * past_diploe
    )
    sigma = cfg.texture_variation
    porosity = profile * np.exp(sigma * texture - 0.5 * sigma**2)
    np.clip(porosity, 0.0, 1.0, out=porosity)
    porosity[~skull_mask(labels)] = 0.0
    return porosity.astype(np.float32)


def explained_variance(sound_speed: np.ndarray, porosity: np.ndarray) -> float:
    """R^2 of a linear fit of sound speed on porosity (squared Pearson correlation).

    sound_speed: [m/s], 1D. porosity: 1D, same length.
    """
    return float(
        np.corrcoef(sound_speed.astype(np.float64), porosity.astype(np.float64))[0, 1] ** 2
    )


def hidden_sound_speed(
    porosity: np.ndarray,
    labels: np.ndarray,
    texture: np.ndarray,
    cfg: MicrostructureConfig,
    skull: SkullMixing,
) -> np.ndarray:
    """Sound-speed component that X-rays cannot see, confined to the skull.

    ``hidden_c = clip(std * texture, c_fluid - c_phi, cfg.sound_speed_max -
    c_phi)`` in the skull, with ``c_phi`` the porosity-driven sound speed. The
    clip keeps the total skull sound speed between the pore-fluid speed and
    `cfg.sound_speed_max`. `std` is solved so that porosity explains exactly
    the fraction `cfg.porosity_r2` of the variance of the total skull sound
    speed in this volume. Without the clip and without chance correlation this
    is ``std**2 = var(c_phi) * (1 - r2) / r2``. The field is zero-mean before
    the clip, which touches well under 1 % of the voxels with the default config.

    porosity: pore volume fraction, shape (...).
    labels: `Tissue` labels, shape (...).
    texture: zero-mean, unit-variance random field, shape (...).

    Returns hidden_c [m/s], float32, shape (...), zero outside the skull.
    """
    hidden = np.zeros(porosity.shape, dtype=np.float32)
    in_skull = skull_mask(labels)
    phi = porosity[in_skull].astype(np.float64)
    c_phi = skull_sound_speed_from_porosity(phi, skull)
    if cfg.porosity_r2 == 1.0 or phi.size < 2 or np.var(c_phi) == 0.0:
        return hidden

    pattern = texture[in_skull].astype(np.float64)
    pattern = (pattern - pattern.mean()) / pattern.std()
    lower = np.minimum(skull.c_fluid + _ROUNDING_MARGIN - c_phi, 0.0)
    upper = np.maximum(cfg.sound_speed_max - _ROUNDING_MARGIN - c_phi, 0.0)

    def hidden_values(std: float) -> np.ndarray:
        return np.clip(std * pattern, lower, upper)

    def r2_excess(std: float) -> float:
        return explained_variance(c_phi + hidden_values(std), phi) - cfg.porosity_r2

    closed_form = np.sqrt(np.var(c_phi) * (1.0 - cfg.porosity_r2) / cfg.porosity_r2)
    low, high = (factor * closed_form for factor in _HIDDEN_STD_BRACKET)
    if r2_excess(low) < 0.0 or r2_excess(high) > 0.0:
        raise ValueError(
            f"porosity_r2 = {cfg.porosity_r2} cannot be reached with skull sound speed "
            f"bounded to [{skull.c_fluid}, {cfg.sound_speed_max}] m/s"
        )
    std = optimize.brentq(r2_excess, low, high, xtol=1.0e-6 * closed_form)
    hidden[in_skull] = hidden_values(std)
    return hidden
