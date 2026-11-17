"""Spatial CBV ground truth: baseline blood volume, activation pattern, blood velocity.

The vascular texture is procedural and is a modelling assumption, not an
anatomical vascular tree. Vessel axes are the curves where two independent
smooth random fields ``g1`` and ``g2`` both vanish; ``vesselness = exp(-(g1^2 +
g2^2) / (2 w^2))`` is then a set of smooth tubes around random space curves, and
``grad g1 x grad g2`` points along them.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid3D
from neurosim.head.config import CbvConfig
from neurosim.head.fields import grid_random_field
from neurosim.head.materials import brain_mask
from neurosim.head.types import Tissue


def _vessel_fields(
    rng: np.random.Generator, grid: Grid3D, cfg: CbvConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Vesselness in [0, 1], shape grid.shape, and unit tangent vectors, shape (3, *grid.shape)."""
    g1 = grid_random_field(rng, grid, cfg.vessel_correlation)
    g2 = grid_random_field(rng, grid, cfg.vessel_correlation)
    # A unit-variance field with squared-exponential covariance has an rms slope
    # of 1 / correlation length per axis, so |g| < w spans a distance of about
    # w * correlation length around a zero line.
    width = cfg.vessel_radius / cfg.vessel_correlation
    vesselness = np.exp(-(g1**2 + g2**2) / (2.0 * width**2))

    gradient_1 = np.gradient(g1, grid.spacing)
    gradient_2 = np.gradient(g2, grid.spacing)
    tangent = np.stack(
        (
            gradient_1[1] * gradient_2[2] - gradient_1[2] * gradient_2[1],
            gradient_1[2] * gradient_2[0] - gradient_1[0] * gradient_2[2],
            gradient_1[0] * gradient_2[1] - gradient_1[1] * gradient_2[0],
        )
    )
    norm = np.linalg.norm(tangent, axis=0)
    tangent /= np.maximum(norm, np.finfo(np.float32).tiny)
    return vesselness, tangent


def cbv_fields(
    rng: np.random.Generator,
    grid: Grid3D,
    labels: np.ndarray,
    activation_center: np.ndarray,
    cfg: CbvConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the CBV ground truth on a grid.

    rng: generator of the `cbv` stream.
    grid: head grid.
    labels: `Tissue` labels, shape grid.shape.
    activation_center: head-frame position of the centre of the activated patch [m], shape (3,).

    Returns float32 arrays:
        cbv_baseline: resting blood volume fraction in [0, 1], shape grid.shape, zero
            outside the brain. Tissue median (grey or white) times a smooth log-normal
            texture, plus `cfg.vessel_cbv` times the vesselness.
        activation: task-response pattern in [0, 1], shape grid.shape:
            ``exp(-d^2 / (2 radius^2))`` of the distance d to `activation_center`,
            inside grey matter only.
        blood_velocity: head-frame velocity [m/s], shape (3, *grid.shape), zero outside
            the brain. The speed rises from about `cfg.speed_tissue` to
            `cfg.speed_vessel` with the vesselness; the direction follows the vessels.
    """
    in_brain = brain_mask(labels)
    texture = grid_random_field(rng, grid, cfg.texture_correlation)
    sigma = cfg.texture_variation
    texture = np.exp(sigma * texture - 0.5 * sigma**2)
    vesselness, tangent = _vessel_fields(rng, grid, cfg)

    tissue_cbv = np.zeros(len(Tissue), dtype=np.float32)
    tissue_cbv[Tissue.GREY] = cfg.cbv_grey
    tissue_cbv[Tissue.WHITE] = cfg.cbv_white
    cbv_baseline = tissue_cbv[labels] * texture + cfg.vessel_cbv * vesselness
    np.clip(cbv_baseline, 0.0, 1.0, out=cbv_baseline)
    cbv_baseline[~in_brain] = 0.0

    speed = cfg.speed_tissue * texture + (cfg.speed_vessel - cfg.speed_tissue) * vesselness
    speed[~in_brain] = 0.0
    blood_velocity = tangent
    blood_velocity *= speed

    squared_distance = sum(
        (axis.astype(np.float32) - np.float32(center)) ** 2
        for axis, center in zip(np.ix_(*grid.axes()), activation_center)
    )
    activation = np.exp(-squared_distance / (2.0 * cfg.activation_radius**2))
    activation[labels != Tissue.GREY] = 0.0
    return (
        cbv_baseline.astype(np.float32),
        activation.astype(np.float32),
        blood_velocity.astype(np.float32),
    )
