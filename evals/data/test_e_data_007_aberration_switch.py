"""E-DATA-007: homogenising the skull sharpens the uncorrected point-spread function.

The image-domain point-spread function of conventional DAS through the skull is
a column of the power Doppler operator `operators[skull_das]` built with one
sample per pixel: the power image of a point scatterer at one pixel centre.
Its extent is measured by the participation area

    area = (sum_p h_p)^2 / sum_p h_p^2 * pixel area,

the area a uniform patch of the same total power and the same mean square power
would cover (one pixel for a single bright pixel, the whole image for a flat
one). It needs no peak finding, so it is not thrown off by the displacement of
the image through the fast skull.

Status (2026-11-17): failing, and left failing. The ratio is 1.33 and 1.29 on
seeds 1 and 2 but 0.97 on seed 0, where the heterogeneous skull gives the
slightly smaller area (2.33 against 2.39 mm^2). Diagnosis: the straight-ray
delay error across the aperture has a standard deviation of 0.30 to 0.47
periods at 2 MHz with the heterogeneous skull against 0.07 to 0.09 homogenised,
but most of it is tilt and curvature, which move or defocus the image; what is
left after a quadratic fit is 0.10 to 0.18 periods against 0.03 to 0.04. Even
the homogenised skull leaves the uncorrected PSF several times wider than the
diffraction limit (a curved 6.5 mm layer about 1000 m/s faster than the
beamformer assumes), so the irregular part is a modest addition and the
low-order part of the heterogeneity can add to or cancel the defocus of the
layer. With absorption switched off as well the ratios are 0.88, 1.56 and 1.48:
the exception is not an absorption effect.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene
from neurosim.data.types import SKULL_DAS, Scene

EVAL_ID = "E-DATA-007"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
HEAD_SPACING = 1.0e-3  # [m]: resolves the porosity (1.5 mm) and hidden-speed (3 mm) textures
PIXELS = (32, 32)
PIXEL_SPACING = 0.25e-3  # [m]: a third of a wavelength at 2 MHz
# Point sources: brain pixels in the lower half of the 8 mm image and within 2 mm of its
# axis, 1 mm apart, so that the image of each, displaced upwards by the skull, stays inside.
SOURCE_ROWS = slice(18, 27, 4)
SOURCE_COLUMNS = slice(8, 25, 4)


def _mean_psf_area(scene: Scene) -> float:
    """Mean participation area of the skull_das point-spread function [m^2]."""
    shape = scene.pixel_grid.shape
    index = np.arange(shape[0] * shape[1]).reshape(shape)
    sources = index[SOURCE_ROWS, SOURCE_COLUMNS][scene.brain_mask[SOURCE_ROWS, SOURCE_COLUMNS]]
    assert sources.size >= 5, "too few point sources in the brain"
    columns = scene.operators[SKULL_DAS][:, sources].astype(np.float64)
    areas = columns.sum(axis=0) ** 2 / (columns**2).sum(axis=0) * scene.cell_area
    return float(areas.mean())


@eval_case(
    id=EVAL_ID,
    claim=(
        "with the aberration switch off (skull sound speed and density replaced by their "
        "means) the point-spread function of conventional DAS through the skull is sharper "
        "than through the heterogeneous skull"
    ),
    reference=(
        "Ablation (ROADMAP D-005): the same head, plane and array with and without skull "
        "heterogeneity; a homogeneous layer keeps the bulk delay and the absorption but "
        "removes the irregular element-to-element delays"
    ),
    metric=(
        "area_ratio_seed_<s>: mean participation area of the PSF with the heterogeneous skull "
        "divided by that with the homogenised skull (15 point sources in the brain); "
        "area_mm2_* are the two areas"
    ),
    threshold=(
        "area_ratio > 1 for every seed (fixed in advance: the claim is an ordering of two "
        "noise-free, deterministic numbers, so any ratio above 1 on all seeds supports it; the "
        "hidden sound-speed field of about 230 m/s rms over a 6.5 mm skull is expected to give "
        "delay errors near a quarter period at 2 MHz, enough to spread the PSF visibly)"
    ),
    seeds=(0, 1, 2),
)
def test_e_data_007_aberration_switch() -> None:
    base = from_dict(RunConfig, load_yaml(CONFIG))
    cfg = replace(
        base,
        head=replace(base.head, grid=replace(base.head.grid, spacing=HEAD_SPACING)),
        imaging=replace(
            base.imaging, pixels=replace(base.imaging.pixels, shape=PIXELS, spacing=PIXEL_SPACING)
        ),
        source_refinement=1,
    )
    homogenised = replace(cfg, effects=replace(cfg.effects, aberration=False))
    values = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        heterogeneous_area = _mean_psf_area(build_scene(cfg, SeedTree(seed)))
        homogenised_area = _mean_psf_area(build_scene(homogenised, SeedTree(seed)))
        values[f"area_ratio_seed_{seed}"] = heterogeneous_area / homogenised_area
        values[f"area_mm2_heterogeneous_seed_{seed}"] = heterogeneous_area * 1.0e6
        values[f"area_mm2_homogenised_seed_{seed}"] = homogenised_area * 1.0e6
    ratios = [values[f"area_ratio_seed_{seed}"] for seed in REGISTRY[EVAL_ID].seeds]
    report(EVAL_ID, values, passed=min(ratios) > 1.0)
