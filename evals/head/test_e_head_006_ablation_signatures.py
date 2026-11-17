"""E-HEAD-006: each skull-effect switch removes its signature from the medium."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.common.units import db_cm_mhz_to_np_m, np_m_to_db_cm
from neurosim.head.ablations import (
    homogenize_skull,
    match_skull_impedance,
    remove_skull,
    zero_skull_absorption,
)
from neurosim.head.generate import generate_head
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import SKULL_LABELS, PlaneMedium, Tissue

EVAL_ID = "E-HEAD-006"
GRID = Grid2D.from_extent((-8.0e-3, 8.0e-3), (0.0, 30.0e-3), 0.2e-3)

# Tolerances and soft-tissue ceilings, fixed before the first run.
SPEED_SPREAD_MAX = 1.0e-3  # m/s
REFLECTION_MAX = 1.0e-6
SOFT_SPEED_MAX = 1700.0  # m/s
SOFT_DENSITY_MAX = 1200.0  # kg/m^3
SOFT_ABSORPTION_MAX = 1.0  # dB/cm at 0.5 MHz


def scalp_skull_reflection(medium: PlaneMedium) -> float:
    """Largest |R| = |Z2 - Z1| / (Z2 + Z1) over adjacent scalp/skull pixel pairs."""
    impedance = medium.density.astype(np.float64) * medium.sound_speed.astype(np.float64)
    scalp = medium.labels == Tissue.SCALP
    skull = np.isin(medium.labels, SKULL_LABELS)
    largest = 0.0
    for axis in (0, 1):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        pairs = (scalp[a] & skull[b]) | (skull[a] & scalp[b])
        reflection = np.abs(impedance[a] - impedance[b]) / (impedance[a] + impedance[b])
        largest = max(largest, float(reflection[pairs].max()))
    return largest


@eval_case(
    id=EVAL_ID,
    claim="Each skull-effect switch removes its signature: a homogenised skull has no "
    "sound-speed spread, a matched skull has no reflection at the scalp/skull interface, "
    "zeroed absorption is zero, and a removed skull leaves no pixel with bone properties",
    reference="Definitions: R = (Z2 - Z1) / (Z2 + Z1) at normal incidence, Z = rho c; bone "
    "is told from soft tissue by the Aubry et al. 2022 Table I values (soft tissue at most "
    "1610 m/s, 1090 kg/m^3, 0.3 dB/cm at 0.5 MHz; bone at least 2300, 1700, 4)",
    metric="Number of failed signatures over seeds (the four measured values of the first "
    "seed and the unablated reflection coefficient are reported alongside)",
    threshold="0 failed: skull sound-speed spread < 1e-3 m/s and |R| < 1e-6 (float32 "
    "rounding of an exact identity), skull absorption exactly 0, and after removal every "
    "pixel below 1700 m/s, 1200 kg/m^3 and 1 dB/cm at 0.5 MHz (between the soft-tissue and "
    "bone values above). Fixed before the first run",
    seeds=(0, 1, 2),
)
def test_e_head_006_ablation_signatures(small_cfg):
    failed = 0
    values: dict[str, float] = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(small_cfg, SeedTree(seed))
        medium = sample_plane(head, default_plane(head, small_cfg), GRID)
        skull = np.isin(medium.labels, SKULL_LABELS)

        speed_spread = float(np.ptp(homogenize_skull(medium).sound_speed[skull]))
        reflection = scalp_skull_reflection(match_skull_impedance(medium))
        absorption = float(np.abs(zero_skull_absorption(medium).attenuation[skull]).max())
        removed = remove_skull(medium)
        removed_absorption = np_m_to_db_cm(
            db_cm_mhz_to_np_m(removed.attenuation, 0.5e6, removed.alpha_power)
        )
        bone_left = (
            removed.sound_speed.max() >= SOFT_SPEED_MAX
            or removed.density.max() >= SOFT_DENSITY_MAX
            or removed_absorption.max() >= SOFT_ABSORPTION_MAX
        )
        failed += int(speed_spread >= SPEED_SPREAD_MAX)
        failed += int(reflection >= REFLECTION_MAX)
        failed += int(absorption != 0.0)
        failed += int(bone_left)
        if not values:
            values = {
                "homogenized_speed_spread_m_s": speed_spread,
                "matched_reflection": reflection,
                "unablated_reflection": scalp_skull_reflection(medium),
                "zeroed_absorption_max": absorption,
                "removed_max_speed_m_s": float(removed.sound_speed.max()),
                "removed_max_density_kg_m3": float(removed.density.max()),
            }
    report(EVAL_ID, {"failed": failed, **values}, passed=failed == 0)
