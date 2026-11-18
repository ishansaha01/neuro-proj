"""E-RECON-013: the focus criterion is largest for the true model when the truth is in the family.

A matched filter maximises the signal-to-noise ratio, so the mean ratio of blood
power to the beamformer's noise floor should be largest when the beamforming
model is the medium that made the data. Here the "true" skull of a small real
scene is replaced by a drawn member of the scene's own CT skull-model family
(`recon.family_scene`), traced by the real backend, and the criterion
(`recon.focus`, through the scanner emulation) is evaluated for the true member,
for the nominal member and for random perturbations of the true member.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.fus.config import imaging_setup
from neurosim.recon import scanner
from neurosim.recon.config import SkullModelConfig
from neurosim.recon.family_scene import bent_ray_case, draw_member, with_member_skull
from neurosim.recon.focus import focus_criterion
from neurosim.recon.rebeamform import model_weights, weight_model
from neurosim.recon.skull_model import SkullParameters, parameter_tensors

EVAL_ID = "E-RECON-013"
STREAM_SUB = 1013
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
# The small scene's medium is 21 mm wide: 8 lateral modes instead of the default 16.
SKULL_MODEL = SkullModelConfig(n_lateral_modes=8)
LATERAL_SPEED_SIGMA = 40.0  # m/s per lateral mode of the drawn true member
PERTURBATION_SPEED_SIGMA = 20.0  # m/s per lateral mode of a perturbation
PERTURBATION_RANGE_FRACTION = 0.25  # of each constant's prior range, uniform, either side
N_PERTURBATIONS = 8
N_PIXELS = 64


def _perturbed(family, member: SkullParameters, rng: np.random.Generator) -> SkullParameters:
    prior = family.cfg.mapping_prior
    widths = {
        "c_bone": prior.c_bone_range[1] - prior.c_bone_range[0],
        "c_water": prior.c_water_range[1] - prior.c_water_range[0],
        "hu_bone": family.nominal.hu_bone
        * (prior.hu_bone_relative_range[1] - prior.hu_bone_relative_range[0]),
    }
    shifts = {
        name: PERTURBATION_RANGE_FRACTION * width * rng.uniform(-1.0, 1.0)
        for name, width in widths.items()
    }
    sigma = PERTURBATION_SPEED_SIGMA / family.cfg.lateral_reference_speed**2
    return SkullParameters(
        c_bone=member.c_bone + shifts["c_bone"],
        c_water=member.c_water + shifts["c_water"],
        hu_bone=member.hu_bone + shifts["hu_bone"],
        lateral=member.lateral + sigma * rng.standard_normal(member.lateral.shape),
    )


@eval_case(
    id=EVAL_ID,
    claim="When the true skull is a member of the CT skull-model family, the focus criterion "
    "(log of the mean ratio of time-averaged blood power to the beamformer's noise floor over "
    "the calibration pixels) is larger for the true member than for the nominal CT member and "
    "than for random perturbations of the true member",
    reference="Matched filtering (Cauchy-Schwarz): the signal-to-noise ratio of a linear filter "
    "is largest when the filter matches the signal; the true member is known because the "
    "true bank is generated from it (recon.family_scene.with_member_skull, real backend)",
    metric="smallest margin, over 3 scene seeds, of the criterion of the true member over the "
    "largest criterion among the nominal member and 8 perturbed members (lateral modes "
    "perturbed by 20 m/s each, HU-map constants by up to a quarter of their prior range); "
    "small bent-ray scene, 64 calibration pixels, 8 lateral modes",
    threshold="> 0 on every seed: the true member wins all 27 comparisons (fixed in advance: "
    "the claim is an ordering; the perturbations move the travel times by tens of "
    "nanoseconds, several times the 10 to 30 ns by which the fixed-path weights of the true "
    "member differ from its re-traced bank in development runs on this scene)",
    seeds=(3, 4, 5),
)
def test_e_recon_013_focus_criterion_maximum():
    cfg = from_dict(RunConfig, load_yaml(CONFIG))
    values, margins = {}, []
    for seed in REGISTRY[EVAL_ID].seeds:
        scene, trials, family = bent_ray_case(cfg, seed, SKULL_MODEL)
        tree = SeedTree(seed)
        truth = draw_member(family, tree, LATERAL_SPEED_SIGMA, STREAM_SUB)
        scene = with_member_skull(scene, family, truth)
        rng = tree.rng("recon_sampling", STREAM_SUB)
        brain = np.flatnonzero(np.asarray(scene.brain_mask).reshape(-1))
        pixels = np.sort(rng.choice(brain, size=min(N_PIXELS, brain.shape[0]), replace=False))
        record = scanner.record_channel_data(scene, trials, 1)
        weights = weight_model(
            family,
            scene.pixel_grid,
            imaging_setup(scene.imaging).sequence,
            scene.imaging.beamforming,
            pixels,
        )

        def criterion(member: SkullParameters) -> float:
            with torch.no_grad():
                ratios = scanner.signal_to_floor(
                    record, model_weights(weights, *parameter_tensors(family, member))
                )
                return float(focus_criterion(ratios, 1.0))

        at_truth = criterion(truth)
        at_nominal = criterion(family.nominal_parameters())
        perturbed = [criterion(_perturbed(family, truth, rng)) for _ in range(N_PERTURBATIONS)]
        margin = at_truth - max(at_nominal, *perturbed)
        margins.append(margin)
        values.update(
            {
                f"seed{seed}_criterion_true": at_truth,
                f"seed{seed}_criterion_nominal": at_nominal,
                f"seed{seed}_criterion_best_perturbation": max(perturbed),
                f"seed{seed}_criterion_mean_perturbation": float(np.mean(perturbed)),
                f"seed{seed}_margin": margin,
            }
        )
    values["smallest_margin"] = min(margins)
    report(
        EVAL_ID,
        values,
        passed=min(margins) > 0.0,
        note="criteria are natural logarithms of a mean power ratio: a margin of 0.1 is 10 % "
        "more mean blood power over the floor",
    )
