"""E-RECON-011: fixed-path (Fermat) travel times against re-tracing the bent rays.

`recon.skull_model` predicts the travel times of any member of the CT skull-model
family from path-length histograms of the rays traced once through the nominal
CT medium. The reference re-traces the refracted rays through the member's own
medium with the backend (`fus.backends.bent_ray.trace_bent_rays`), which knows
nothing of the histograms. By Fermat's principle the fixed-path time is an upper
bound on the first arrival, and its excess is second order in the perturbation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.fus.backends.bent_ray import trace_bent_rays
from neurosim.recon.config import SkullModelConfig
from neurosim.recon.family_scene import bent_ray_case, draw_member
from neurosim.recon.skull_model import (
    SkullFamily,
    SkullParameters,
    fermat_travel_time,
    model_medium,
)

EVAL_ID = "E-RECON-011"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
# The small scene's medium is 21 mm wide: 8 lateral modes instead of the default 16.
SKULL_MODEL = SkullModelConfig(n_lateral_modes=8)
LATERAL_SPEED_SIGMA = 40.0  # m/s per lateral mode of the drawn members
N_DRAWS = 6
FREQUENCY = 2.0e6  # Hz: the period the errors are quoted in
MAX_RMS_PERIODS = 1.0 / 14.0
# Fractions of a drawn member's distance from the nominal member at which the breakdown
# of the shortcut is reported.
REPORT_SCALES = (0.5, 2.0)


def _weighted_rms(difference: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt((weight * difference**2).sum() / weight.sum()))


def _errors(family: SkullFamily, member: SkullParameters) -> tuple[float, float, float]:
    """`(rms error, largest error, rms change)` of the fixed-path times of a member [s].

    The error is the fixed-path travel time minus the re-traced one; the change is
    the re-traced time minus the nominal one. Weights: the re-traced ray amplitude
    `W T S` on the pairs to which the nominal trace gives amplitude (the pairs a
    beamformer of the family uses).
    """
    rays = trace_bent_rays(
        model_medium(family, member),
        family.transducer,
        family.roi.points().reshape(-1, 2),
        family.backend.bent_ray,
    )
    weight = rays.window * rays.transmission * rays.spreading * (family.amplitude > 0.0)
    error = fermat_travel_time(family, member) - rays.travel_time
    used = weight > 0.0
    return (
        _weighted_rms(error, weight),
        float(np.abs(error[used]).max()),
        _weighted_rms(rays.travel_time - family.travel_time, weight),
    )


def _scaled(family: SkullFamily, member: SkullParameters, scale: float) -> SkullParameters:
    """The member a fraction `scale` of the way from the nominal member to `member`."""
    nominal = family.nominal_parameters()
    return SkullParameters(
        **{
            name: getattr(nominal, name) + scale * (getattr(member, name) - getattr(nominal, name))
            for name in ("c_bone", "c_water", "hu_bone")
        },
        lateral=scale * member.lateral,
    )


@eval_case(
    id=EVAL_ID,
    claim="The travel times predicted from the path-length histograms of the nominal rays "
    "(recon.skull_model.fermat_travel_time) for skull-model members drawn across the prior "
    "agree with the travel times of rays re-traced through the member's medium",
    reference="fus.backends.bent_ray.trace_bent_rays through recon.skull_model.model_medium of "
    "the member: the backend's eikonal solve and ray integral, independent of the histograms",
    metric="amplitude-weighted rms over element-point pairs of (fixed-path - re-traced) one-way "
    "travel time, in periods at 2 MHz; the largest over 6 members drawn from the prior (HU-map "
    "constants uniform in their ranges, 8 lateral modes of 40 m/s each) on the small "
    "bent-ray scene (48 elements, 13 x 13 ROI nodes, 0.2 mm medium grid)",
    threshold="< 1/14 period (0.0714) for every member (fixed in advance: lambda/14 is the "
    "project's travel-time budget, the Marechal criterion used in E-FUS-021 and E-FUS-023; a "
    "model whose delays the shortcut misplaces by more cannot be calibrated to a "
    "diffraction-limited focus through it)",
    seeds=(3,),
)
def test_e_recon_011_fermat_path_histograms():
    seed = REGISTRY[EVAL_ID].seeds[0]
    _, _, family = bent_ray_case(from_dict(RunConfig, load_yaml(CONFIG)), seed, SKULL_MODEL)
    tree = SeedTree(seed)
    period = 1.0 / FREQUENCY
    nominal_rms, nominal_max, _ = _errors(family, family.nominal_parameters())
    draws = []
    for draw in range(N_DRAWS):
        member = draw_member(family, tree, LATERAL_SPEED_SIGMA, 11, draw)
        rms, largest, change = _errors(family, member)
        record = {
            "rms_periods": rms / period,
            "max_periods": largest / period,
            "change_rms_periods": change / period,
            "c_bone": member.c_bone,
            "c_water": member.c_water,
            "hu_bone_factor": member.hu_bone / family.nominal.hu_bone,
        }
        for scale in REPORT_SCALES:
            scaled_rms, _, _ = _errors(family, _scaled(family, member, scale))
            record[f"rms_periods_at_{scale:g}x"] = scaled_rms / period
        draws.append(record)
    worst = max(record["rms_periods"] for record in draws)
    values = {
        "worst_rms_periods": worst,
        "nominal_member_rms_periods": nominal_rms / period,
        "nominal_member_max_periods": nominal_max / period,
    }
    for draw, record in enumerate(draws):
        values.update({f"draw{draw}_{name}": value for name, value in record.items()})
    report(
        EVAL_ID,
        values,
        passed=worst < MAX_RMS_PERIODS,
        note="change_rms_periods is how far the member's re-traced times are from the nominal "
        "ones (what the shortcut has to predict); rms_periods_at_0.5x and _at_2x are the "
        "errors for the member half as far and twice as far from the nominal one (twice as "
        "far can leave the prior ranges): the excess grows faster than the perturbation, as "
        "a second-order term does. The nominal member has zero error by construction of "
        "the family only if tracing is deterministic; its row checks that.",
    )
