"""E-HEAD-002: every derived acoustic property and the layer porosities lie in published ranges."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.common.units import db_cm_mhz_to_np_m, np_m_to_db_cm
from neurosim.head.generate import generate_head
from neurosim.head.types import Tissue

EVAL_ID = "E-HEAD-002"

# Aubry et al. (2022), Table I: sound speed [m/s], density [kg/m^3], absorption [dB/cm at 0.5 MHz].
BENCHMARK_FREQUENCY = 0.5e6
SKIN = (1610.0, 1090.0, 0.2)
BRAIN = (1560.0, 1040.0, 0.3)
CORTICAL = (2800.0, 1850.0, 4.0)
TRABECULAR = (2300.0, 1700.0, 8.0)

# Acceptance bands, fixed before the first run.
SOFT_TOLERANCE = 0.05  # relative, sound speed and density of soft tissue
BONE_TOLERANCE = 0.15  # relative, layer means of sound speed and density
ABSORPTION_FACTOR = 2.0  # absorption within a factor of 2 of the benchmark value
WATER_SPEED = (1480.0, 1530.0)
WATER_DENSITY = (990.0, 1010.0)
WATER_ABSORPTION_MAX = 0.01  # dB/cm at 1 MHz
SKULL_SPEED = (1500.0, 3507.0)
TABLE_POROSITY = (0.0, 0.2)
DIPLOE_POROSITY = (0.25, 0.60)
SKULL_POROSITY = (0.135, 0.40)

# Voxel size of the volume the check runs on [m]: 1 mm resolves the tables.
SPACING = 1.0e-3


def absorption_at(alpha0: np.ndarray, frequency: float, y: float) -> np.ndarray:
    """Absorption [dB/cm] at `frequency` of a power law with prefactor alpha0 [dB/(cm MHz^y)]."""
    return np_m_to_db_cm(db_cm_mhz_to_np_m(alpha0, frequency, y))


def relative_band(value: float, tolerance: float) -> tuple[float, float]:
    return value * (1.0 - tolerance), value * (1.0 + tolerance)


def factor_band(value: float, factor: float) -> tuple[float, float]:
    return value / factor, value * factor


@eval_case(
    id=EVAL_ID,
    claim="Sound speed, density and absorption of every tissue, and the porosity of every "
    "skull layer, lie in published ranges",
    reference="Aubry et al. 2022 (JASA 152:1003), Table I: skin 1610 m/s, 1090 kg/m^3, "
    "0.2 dB/cm; brain 1560, 1040, 0.3; cortical bone 2800, 1850, 4; trabecular bone 2300, "
    "1700, 8 (absorption at 0.5 MHz); water 1500, 1000. Skull sound speed between water "
    "(Aubry et al. 2003) and 3507 m/s (Clement & Hynynen 2002, upper end, as quoted by "
    "Angla et al. 2023). Porosity: 0.07 and 0.43 follow from the benchmark speeds under the "
    "Aubry 2003 law; 26.9 % full-thickness porosity of a parietal sample (Jing & Lindsey 2021)",
    metric="Number of quantities outside their band, over all seeds (every checked value "
    "is reported for the first seed)",
    threshold="0 outside: soft-tissue sound speed and density within 5 % of the benchmark "
    "(they are constants, 5 % covers rounding between sources); CSF and coupling medium "
    "1480-1530 m/s, 990-1010 kg/m^3, absorption < 0.01 dB/cm at 1 MHz (water from 20 to "
    "37 degC); every skull voxel 1500-3507 m/s; table and diploe means of sound speed and "
    "density within 15 % of the cortical and trabecular benchmark values (the spread "
    "between the constants of Aubry 2003 and Aubry 2022); absorption at 0.5 MHz within a "
    "factor 2 of the benchmark values (published skull absorption differs by about that); "
    "table mean porosity <= 0.2, diploe mean 0.25-0.60, skull mean 0.135-0.40 (26.9 % +- "
    "half). Fixed before the first run",
    seeds=(0, 1, 2),
)
def test_e_head_002_property_ranges(small_cfg):
    cfg = dataclasses.replace(small_cfg, grid=dataclasses.replace(small_cfg.grid, spacing=SPACING))
    outside = 0
    first_seed_values: dict[str, float] = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(cfg, SeedTree(seed))
        y = head.alpha_power
        at_benchmark = absorption_at(head.attenuation, BENCHMARK_FREQUENCY, y)
        at_1mhz = absorption_at(head.attenuation, 1.0e6, y)
        checks: list[tuple[str, float, tuple[float, float]]] = []

        def add_range(name: str, values: np.ndarray, band: tuple[float, float]) -> None:
            checks.append((f"{name}_min", float(values.min()), band))
            checks.append((f"{name}_max", float(values.max()), band))

        for name, tissues, reference in (
            ("scalp", (Tissue.SCALP,), SKIN),
            ("grey", (Tissue.GREY,), BRAIN),
            ("white", (Tissue.WHITE,), BRAIN),
        ):
            mask = np.isin(head.labels, tissues)
            add_range(
                f"{name}_c", head.sound_speed[mask], relative_band(reference[0], SOFT_TOLERANCE)
            )
            add_range(
                f"{name}_rho", head.density[mask], relative_band(reference[1], SOFT_TOLERANCE)
            )
            add_range(
                f"{name}_abs", at_benchmark[mask], factor_band(reference[2], ABSORPTION_FACTOR)
            )
        for name, tissue in (("csf", Tissue.CSF), ("background", Tissue.BACKGROUND)):
            mask = head.labels == tissue
            add_range(f"{name}_c", head.sound_speed[mask], WATER_SPEED)
            add_range(f"{name}_rho", head.density[mask], WATER_DENSITY)
            checks.append(
                (f"{name}_abs_max", float(at_1mhz[mask].max()), (0.0, WATER_ABSORPTION_MAX))
            )

        tables = np.isin(head.labels, (Tissue.SKULL_OUTER, Tissue.SKULL_INNER))
        diploe = head.labels == Tissue.DIPLOE
        skull = tables | diploe
        add_range("skull_c", head.sound_speed[skull], SKULL_SPEED)
        for name, mask, reference in (("table", tables, CORTICAL), ("diploe", diploe, TRABECULAR)):
            checks.append(
                (
                    f"{name}_c_mean",
                    float(head.sound_speed[mask].mean()),
                    relative_band(reference[0], BONE_TOLERANCE),
                )
            )
            checks.append(
                (
                    f"{name}_rho_mean",
                    float(head.density[mask].mean()),
                    relative_band(reference[1], BONE_TOLERANCE),
                )
            )
            checks.append(
                (
                    f"{name}_abs_mean",
                    float(at_benchmark[mask].mean()),
                    factor_band(reference[2], ABSORPTION_FACTOR),
                )
            )
        checks.append(("table_porosity_mean", float(head.porosity[tables].mean()), TABLE_POROSITY))
        checks.append(
            ("diploe_porosity_mean", float(head.porosity[diploe].mean()), DIPLOE_POROSITY)
        )
        checks.append(("skull_porosity_mean", float(head.porosity[skull].mean()), SKULL_POROSITY))

        outside += sum(not band[0] <= value <= band[1] for _, value, band in checks)
        if not first_seed_values:
            first_seed_values = {name: value for name, value, _ in checks}

    report(EVAL_ID, {"outside": outside, **first_seed_values}, passed=outside == 0)
