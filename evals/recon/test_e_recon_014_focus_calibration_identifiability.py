"""E-RECON-014: without model mismatch, focus calibration brings the skull model to the truth.

The analogue of E-RECON-008 for the second design. The true skull of a small
real scene is a drawn member of the scene's own CT skull-model family, traced
by the real backend (`recon.family_scene`), so the family contains the truth.
Starting from the nominal CT member, `recon.focus.calibrate` maximises the focus
criterion; the calibrated member is re-traced and its travel times are compared
with those of the true bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.fus.config import imaging_setup
from neurosim.recon import scanner
from neurosim.recon.config import FocusConfig, SkullModelConfig
from neurosim.recon.family_scene import bent_ray_case, draw_member, with_member_skull
from neurosim.recon.focus import calibrate, pixel_groups
from neurosim.recon.metrics import travel_time_error
from neurosim.recon.rebeamform import weight_model
from neurosim.recon.skull_model import model_bank

EVAL_ID = "E-RECON-014"
STREAM_SUB = 1014
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
# The small scene's medium is 21 mm wide: 8 lateral modes instead of the default 16, and
# the stages of the default calibration cut at 8.
SKULL_MODEL = SkullModelConfig(n_lateral_modes=8)
FOCUS = FocusConfig(n_pixels=64, mode_stages=(0, 2, 4, 8))
LATERAL_SPEED_SIGMA = 40.0  # m/s per lateral mode of the drawn true member
MAX_MEDIAN_RATIO = 0.5
MAX_RATIO = 1.0


@eval_case(
    id=EVAL_ID,
    claim="On data whose true skull is a member of the CT skull-model family, calibration by "
    "maximising the focus criterion from the nominal CT member reduces the amplitude-weighted "
    "rms one-way travel-time error of the model against the true bank",
    reference="The member that generated the true bank (recon.family_scene: HU-map constants "
    "uniform in their prior ranges, 8 lateral modes of 40 m/s each; real backend); no model "
    "mismatch other than the calibration's own fixed-path approximation",
    metric="ratio = travel-time error of the re-traced calibrated member / travel-time error of "
    "the nominal CT bank (recon.metrics.travel_time_error, common delay removed), per scene; "
    "median and maximum over five scene seeds (small bent-ray scene, 64 calibration pixels, "
    "stages of 0, 2, 4 and 8 lateral modes, 8 L-BFGS iterations each)",
    threshold="median ratio < 0.5 and every ratio < 1.0 (fixed before this eval was run: the "
    "nominal errors of the demo scenes are about a third of a period at 2 MHz, where the "
    "coherent gain exp(-sigma_phase^2) is a few percent; halving the error brings it to "
    "about 0.4, out of the regime in which the focus is lost; a calibration that makes any "
    "scene worse than its starting point is not usable unattended)",
    seeds=(3, 4, 5, 6, 7),
)
def test_e_recon_014_focus_calibration_identifiability():
    cfg = from_dict(RunConfig, load_yaml(CONFIG))
    values, ratios = {}, []
    for seed in REGISTRY[EVAL_ID].seeds:
        scene, trials, family = bent_ray_case(cfg, seed, SKULL_MODEL)
        tree = SeedTree(seed)
        truth = draw_member(family, tree, LATERAL_SPEED_SIGMA, STREAM_SUB)
        scene = with_member_skull(scene, family, truth)
        rng = tree.rng("recon_sampling", STREAM_SUB)
        brain = np.flatnonzero(np.asarray(scene.brain_mask).reshape(-1))
        pixels = np.sort(
            rng.choice(brain, size=min(FOCUS.n_pixels, brain.shape[0]), replace=False)
        )
        record = scanner.record_channel_data(scene, trials, FOCUS.source_refinement)
        sequence = imaging_setup(scene.imaging).sequence
        weights = [
            weight_model(family, scene.pixel_grid, sequence, scene.imaging.beamforming, group)
            for group in pixel_groups(pixels, FOCUS.pixels_per_pass)
        ]
        result = calibrate(record, family, weights, FOCUS)
        calibrated = model_bank(family, result.parameters)
        errors = {
            gauge: (
                travel_time_error(scene.bank_ct, scene.bank_skull, gauge),
                travel_time_error(calibrated, scene.bank_skull, gauge),
            )
            for gauge in ("common", "node")
        }
        ratio = errors["common"][1] / errors["common"][0]
        ratios.append(ratio)
        values.update(
            {
                f"seed{seed}_ratio": ratio,
                f"seed{seed}_nominal_error_ns": 1.0e9 * errors["common"][0],
                f"seed{seed}_calibrated_error_ns": 1.0e9 * errors["common"][1],
                f"seed{seed}_focusing_ratio": errors["node"][1] / errors["node"][0],
                f"seed{seed}_criterion_gain": result.criterion - result.criterion_start,
            }
        )
    values["median_ratio"] = float(np.median(ratios))
    values["max_ratio"] = float(np.max(ratios))
    report(
        EVAL_ID,
        values,
        passed=values["median_ratio"] < MAX_MEDIAN_RATIO and values["max_ratio"] < MAX_RATIO,
        note="focusing_ratio is the same ratio with the delay common to the elements of each "
        "field point removed as well (the focusing error proper, which the criterion sees); "
        "criterion_gain is the rise of the criterion (natural log of a mean power ratio)",
    )
