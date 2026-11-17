"""E-HEAD-001: regional skull thickness and three-layer proportions match the literature.

Thickness is measured on the generated volume, not read from the generator:
the SDF of each skull surface is sampled along rays from the head centre and
its zero crossing located, at vault sites (vertex, both parietal convexities)
and at both lateral poles (the temporal region).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import sample_volume
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.types import SURFACES, HeadVolume

EVAL_ID = "E-HEAD-001"
MM = 1.0e-3

VAULT_SITES = ((0.0, 0.0, 1.0), (0.5, 0.0, 0.866), (-0.5, 0.0, 0.866))
LATERAL_SITES = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

# Acceptance ranges, fixed before the first run.
VAULT_SITE_RANGE = (3.0 * MM, 11.0 * MM)
LATERAL_SITE_RANGE = (1.5 * MM, 6.8 * MM)
VAULT_MEAN_RANGE = (5.5 * MM, 8.0 * MM)
LATERAL_MEAN_RANGE = (2.1 * MM, 4.4 * MM)
LAYER_FRACTION_RANGES = {
    "outer_table": (0.13, 0.33),
    "diploe": (0.52, 0.72),
    "inner_table": (0.05, 0.25),
}

# Voxel size of the full-head volume the measurement runs on [m], and the step
# along a ray [m]. The SDFs are smooth, so their zero crossings are located far
# below the voxel size.
SPACING = 3.0 * MM
RAY_STEP = 0.25 * MM


def crossing_radii(head: HeadVolume, direction: tuple[float, float, float]) -> np.ndarray:
    """Radius [m] at which each of the four skull surfaces crosses a ray from the head centre."""
    radii = np.arange(0.03, 0.125, RAY_STEP)
    points = radii[:, None] * np.asarray(direction) / np.linalg.norm(direction)
    crossings = []
    for surface in ("skull_outer", "outer_table_inner", "diploe_inner", "skull_inner"):
        values = sample_volume(head.sdf[SURFACES.index(surface)], head.grid, points)
        # Radial SDFs rise monotonically with radius, as np.interp requires.
        crossings.append(np.interp(0.0, values, radii))
    return np.array(crossings)


def inside(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


@eval_case(
    id=EVAL_ID,
    claim="Skull thickness is in the adult range at the vault, thinner at the temporal "
    "region, and split into outer table, diploe and inner table in adult proportions",
    reference="de Boer et al. 2016 (Int J Legal Med 130:1371, Table 4): adult frontal 6.15 "
    "(SD 1.91) to 7.13 (2.30) mm, occipital 6.89 (2.00) to 7.33 (2.01) mm, lateral (euryon) "
    "4.33 (1.22) to 4.41 (1.15) mm. Rowbotham et al. 2023 (Int J Legal Med 137:195): "
    "temporal squama 2.11 mm. Aubry et al. 2022 (JASA 152:1003): parietal 6.5 mm = 1.5 mm "
    "outer table + 4 mm diploe + 1 mm inner table (fractions 0.23 / 0.62 / 0.15)",
    metric="Number of violated range checks over all seeds and sites (the measured "
    "thicknesses [mm] and layer fractions are reported alongside)",
    threshold="0 violations of: every vault site in 3.0-11.0 mm (about +-2 SD of de Boer's "
    "frontal and occipital data) and every lateral site in 1.5-6.8 mm (from below the "
    "squama mean to the euryon mean + 2 SD); in every head the lateral mean below the vault "
    "mean; over seeds, vault mean in 5.5-8.0 mm and lateral mean in 2.1-4.4 mm (the spans "
    "of the published means); mean layer fractions at the vault within +-0.10 of "
    "0.23 / 0.62 / 0.15 (no published spread at hand, so a tolerance that still tells the "
    "three layers apart). Fixed before the first run",
    seeds=(0, 1, 2, 3, 4),
)
def test_e_head_001_skull_thickness(small_cfg):
    cfg = dataclasses.replace(
        small_cfg, grid=dataclasses.replace(small_cfg.grid, region="full", spacing=SPACING)
    )
    violations = 0
    vault, lateral, fractions = [], [], []
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(cfg, SeedTree(seed))
        vault_radii = np.array([crossing_radii(head, site) for site in VAULT_SITES])
        lateral_radii = np.array([crossing_radii(head, site) for site in LATERAL_SITES])
        vault_thickness = vault_radii[:, 0] - vault_radii[:, 3]
        lateral_thickness = lateral_radii[:, 0] - lateral_radii[:, 3]
        violations += sum(not inside(t, VAULT_SITE_RANGE) for t in vault_thickness)
        violations += sum(not inside(t, LATERAL_SITE_RANGE) for t in lateral_thickness)
        violations += int(lateral_thickness.mean() >= vault_thickness.mean())
        vault.append(vault_thickness)
        lateral.append(lateral_thickness)
        fractions.append(-np.diff(vault_radii, axis=1) / vault_thickness[:, None])

    vault_mean = float(np.mean(vault))
    lateral_mean = float(np.mean(lateral))
    layer_fractions = dict(zip(LAYER_FRACTION_RANGES, np.mean(fractions, axis=(0, 1))))
    violations += int(not inside(vault_mean, VAULT_MEAN_RANGE))
    violations += int(not inside(lateral_mean, LATERAL_MEAN_RANGE))
    violations += sum(
        not inside(layer_fractions[name], bounds) for name, bounds in LAYER_FRACTION_RANGES.items()
    )
    report(
        EVAL_ID,
        {
            "violations": violations,
            "vault_mean_mm": vault_mean / MM,
            "vault_min_mm": float(np.min(vault)) / MM,
            "vault_max_mm": float(np.max(vault)) / MM,
            "lateral_mean_mm": lateral_mean / MM,
            "lateral_min_mm": float(np.min(lateral)) / MM,
            "lateral_max_mm": float(np.max(lateral)) / MM,
            **{f"{name}_fraction": float(value) for name, value in layer_fractions.items()},
        },
        passed=violations == 0,
    )
