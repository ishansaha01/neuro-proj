"""E-HEAD-003: porosity explains the configured fraction of skull sound-speed variance."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.types import SKULL_LABELS

EVAL_ID = "E-HEAD-003"
TARGETS = (0.5, 0.3)
TOLERANCE = 0.005


@eval_case(
    id=EVAL_ID,
    claim="The fraction of skull sound-speed variance explained by porosity equals the "
    "configured porosity_r2",
    reference="The configured target (default 0.5, chosen above the 23-53 % of Webb et al. "
    "2018, IEEE TUFFC 65:1111, which the CT measurement then degrades), against R^2 of an "
    "ordinary least-squares line of sound speed on porosity over the skull voxels of the "
    "generated volume",
    metric="Largest |R^2 - porosity_r2| over seeds and the targets 0.5 and 0.3",
    threshold="< 0.005: the hidden amplitude is solved for the target in each volume, so "
    "only float32 storage and the solver tolerance remain; 0.005 is far below the 0.3-wide "
    "band the target has to hit. Fixed before the first run",
    seeds=(0, 1, 2, 3, 4),
)
def test_e_head_003_porosity_r2(small_cfg):
    errors = {}
    for target in TARGETS:
        cfg = dataclasses.replace(
            small_cfg,
            microstructure=dataclasses.replace(small_cfg.microstructure, porosity_r2=target),
        )
        for seed in REGISTRY[EVAL_ID].seeds:
            head = generate_head(cfg, SeedTree(seed))
            in_skull = np.isin(head.labels, SKULL_LABELS)
            porosity = head.porosity[in_skull].astype(np.float64)
            sound_speed = head.sound_speed[in_skull].astype(np.float64)
            slope, intercept = np.polyfit(porosity, sound_speed, 1)
            residual = sound_speed - (slope * porosity + intercept)
            r2 = 1.0 - residual.var() / sound_speed.var()
            errors[f"r2_target_{target}_seed_{seed}"] = r2
    worst = max(
        abs(r2 - target)
        for target in TARGETS
        for name, r2 in errors.items()
        if name.startswith(f"r2_target_{target}_")
    )
    report(EVAL_ID, {"max_abs_error": worst, **errors}, passed=worst < TOLERANCE)
