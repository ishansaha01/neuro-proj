"""E-CT-003: projector line integrals match analytic chords of an ellipse."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import eval_case, report
from neurosim.ct.projector import fan_beam_project

EVAL_ID = "E-CT-003"
MAX_RELATIVE_ERROR = 0.01

# Pixel size of the production head grid (0.5 mm). History: the eval was first
# written on the 2 mm test grid and failed there (1.85 %). The error did not
# depend on the ray step or on the supersampling of the phantom and fell with the
# pixel size (1.85 % at 2 mm, 0.78 % at 1 mm, 0.39 % at 0.5 mm): it is the error of
# representing a sharp edge on pixels, seen by the most grazing rays, not an
# error of the projector. Threshold and ray selection are unchanged.
SPACING = 0.0005
SHAPE = (400, 400)
SEMI_AXES = (0.06, 0.04)
CENTER = (0.01, -0.005)
# Rays with a chord of at least half the major diameter; shorter chords graze
# the boundary.
MIN_CHORD = 0.06
SUBSAMPLES = 8
PIXEL_SIZE_NOTE = (
    "Pixel size 0.5 mm (production head grid). On the 2 mm test grid the same metric "
    "is 1.85 % (fails): the error is first-order in the pixel size for grazing rays "
    "through a sharp edge (0.78 % at 1 mm)."
)


def _ellipse_image() -> np.ndarray:
    """Area fraction of each pixel inside the ellipse, by 8 x 8 supersampling."""
    fine = SPACING / SUBSAMPLES
    x = (np.arange(SHAPE[0] * SUBSAMPLES) - 0.5 * (SHAPE[0] * SUBSAMPLES - 1)) * fine
    y = (np.arange(SHAPE[1] * SUBSAMPLES) - 0.5 * (SHAPE[1] * SUBSAMPLES - 1)) * fine
    inside = ((x[:, None] - CENTER[0]) / SEMI_AXES[0]) ** 2 + (
        (y[None, :] - CENTER[1]) / SEMI_AXES[1]
    ) ** 2 <= 1.0
    blocks = inside.reshape(SHAPE[0], SUBSAMPLES, SHAPE[1], SUBSAMPLES)
    return blocks.mean(axis=(1, 3))


def _analytic_chords(geom) -> np.ndarray:
    """Chord length of every ray through the ellipse [m], shape (n_views, n_detectors)."""
    beta = geom.view_angles()[:, None]
    ray_angle = beta + geom.fan_angles()[None, :]
    # Ray: source + t * direction, in coordinates scaled so the ellipse is a unit circle.
    source_x = (geom.source_isocenter_distance * np.cos(beta) - CENTER[0]) / SEMI_AXES[0]
    source_y = (geom.source_isocenter_distance * np.sin(beta) - CENTER[1]) / SEMI_AXES[1]
    direction_x = -np.cos(ray_angle) / SEMI_AXES[0]
    direction_y = -np.sin(ray_angle) / SEMI_AXES[1]
    a = direction_x**2 + direction_y**2
    b = 2.0 * (source_x * direction_x + source_y * direction_y)
    c = source_x**2 + source_y**2 - 1.0
    return np.sqrt(np.maximum(b**2 - 4.0 * a * c, 0.0)) / a


@eval_case(
    id=EVAL_ID,
    claim="Fan-beam line integrals through a uniform off-centre ellipse equal the "
    "analytic chord lengths",
    reference="Analytic: chord of the line source + t d through "
    "((x - x0) / a)^2 + ((y - y0) / b)^2 = 1",
    metric="Maximum relative chord error over rays with chord >= 60 mm [-]",
    threshold="< 1 %. Fixed in advance: water must reconstruct within 5 HU (0.5 % of "
    "mu_water), which needs line integrals good to better than 1 %. Near-tangent rays "
    "are excluded because their chords change by many pixels within one pixel of impact "
    "parameter. The threshold was set for a 2 mm grid, where the eval fails (1.85 %); "
    "it is kept, and the eval now runs at the 0.5 mm pixel size of the head grid "
    "(reason in the eval file and in the result note).",
    seeds=(0,),
)
def test_e_ct_003_projector_chords(small_cfg):
    geom = small_cfg.geometry
    measured = fan_beam_project(
        _ellipse_image(), SPACING, geom, ray_step_fraction=small_cfg.ray_step_fraction
    )
    analytic = _analytic_chords(geom)
    long_rays = analytic >= MIN_CHORD
    relative_error = np.abs(measured[long_rays] / analytic[long_rays] - 1.0)
    worst = float(relative_error.max())
    mean_abs_error = float(np.abs(measured - analytic).mean())
    report(
        EVAL_ID,
        {"max_relative_error": worst, "mean_abs_error_all_rays_m": mean_abs_error},
        worst < MAX_RELATIVE_ERROR,
        note=PIXEL_SIZE_NOTE,
    )
