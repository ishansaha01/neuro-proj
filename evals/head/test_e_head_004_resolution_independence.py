"""E-HEAD-004: plane sampling gives the same skull at two pixel sizes."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import SKULL_LABELS, PlaneMedium, Tissue

EVAL_ID = "E-HEAD-004"
COARSE = 0.2e-3
FINE = 0.1e-3
X_RANGE = (-8.0e-3, 8.0e-3)
Z_RANGE = (0.0, 30.0e-3)
# An interface located on a pixel grid is known to half a pixel, so two grids can
# disagree by at most half the sum of their pixel sizes (plus float rounding).
POSITION_TOLERANCE = 0.5 * (COARSE + FINE) + 1.0e-9
MEAN_TOLERANCE = 0.01


def interface_depths(medium: PlaneMedium, first_label: int) -> np.ndarray:
    """Depth [m] per column of the upper edge of the first pixel with label >= first_label."""
    z, _ = medium.grid.axes()
    first = np.argmax(medium.labels >= first_label, axis=0)
    return z[first] - 0.5 * medium.grid.spacing


@eval_case(
    id=EVAL_ID,
    claim="sample_plane is resolution independent: two pixel sizes give the same skull "
    "interface positions and the same mean skull properties",
    reference="Geometry: an interface located on a pixel grid of size h is known to +-h/2, "
    "so grids of 0.2 mm and 0.1 mm may differ by at most 0.15 mm; a mean over a 6 mm thick "
    "layer changes by about (pixel / thickness) x (contrast to the mean), i.e. below 1 %",
    metric="Largest interface position difference [mm] over the outer and inner skull "
    "surface and all shared columns; largest relative difference of the skull means of "
    "sound speed and density",
    threshold="position difference <= 0.15 mm (the half-pixel bound) and mean difference "
    "< 1 % (boundary-pixel bound above). Fixed before the first run",
    seeds=(0, 1, 2),
)
def test_e_head_004_resolution_independence(small_cfg):
    position_error = 0.0
    mean_error = 0.0
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(small_cfg, SeedTree(seed))
        plane = default_plane(head, small_cfg)
        coarse = sample_plane(head, plane, Grid2D.from_extent(X_RANGE, Z_RANGE, COARSE))
        fine = sample_plane(head, plane, Grid2D.from_extent(X_RANGE, Z_RANGE, FINE))
        shared = slice(None, None, round(COARSE / FINE))
        for first_label in (Tissue.SKULL_OUTER, Tissue.CSF):
            difference = (
                interface_depths(coarse, first_label) - interface_depths(fine, first_label)[shared]
            )
            position_error = max(position_error, float(np.max(np.abs(difference))))
        for name in ("sound_speed", "density"):
            means = [
                getattr(medium, name)[np.isin(medium.labels, SKULL_LABELS)].mean(dtype=np.float64)
                for medium in (coarse, fine)
            ]
            mean_error = max(mean_error, abs(means[0] - means[1]) / means[1])
    report(
        EVAL_ID,
        {
            "max_position_difference_mm": position_error * 1.0e3,
            "max_relative_mean_difference": mean_error,
        },
        passed=position_error <= POSITION_TOLERANCE and mean_error < MEAN_TOLERANCE,
    )
