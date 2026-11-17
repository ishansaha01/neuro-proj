"""E-RECON-010: a reconstruction is a function of the seed and its `recon_sampling` sub-index.

Calibration and per-trial posterior sampling are run twice from the same seed on
one synthetic scene with the trained prior, and once more with another
sub-index of the `recon_sampling` stream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.config import imaging_setup
from neurosim.recon.calibrate import calibrate_screen
from neurosim.recon.checkpoint import DEFAULT_PRIOR_PATH, load_prior
from neurosim.recon.config import ReconConfig
from neurosim.recon.dps import reconstruct_trials
from neurosim.recon.forward import power_doppler_operator, scene_screen_model, screen_tensors
from neurosim.recon.seeding import configure_torch
from neurosim.recon.synthetic import synthetic_run

EVAL_ID = "E-RECON-010"
STREAM_SUB = 1010
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recon_small.yaml"
N_TRIALS = 4


@eval_case(
    id=EVAL_ID,
    claim="The same seed reproduces a blind reconstruction (calibrated screen and posterior "
    "samples) bit for bit, and another recon_sampling sub-index changes it",
    reference="Determinism: two runs of the same code with the same seed, thread count and inputs",
    metric="differing_values = array elements (calibrated delays and amplitudes, posterior "
    "CBV samples of 4 trials) that differ between two runs with the same seed; "
    "changed_fraction = fraction of posterior-sample values that differ when only the "
    "sub-index changes",
    threshold="differing_values == 0 and changed_fraction > 0.5 (fixed in advance: every draw "
    "comes from a generator seeded by (root seed, recon_sampling, sub-indices) and torch runs "
    "single-threaded with deterministic algorithms, so equality must be exact; another "
    "sub-index gives independent noise, which changes essentially every brain pixel)",
    seeds=(0,),
)
def test_e_recon_010_reconstruction_reproducibility():
    if not DEFAULT_PRIOR_PATH.is_file():
        pytest.fail(
            f"{EVAL_ID} needs the trained prior at {DEFAULT_PRIOR_PATH}; create it with "
            "scripts/make_prior_dataset.py and scripts/train_prior.py (configs/prior_default.yaml)"
        )
    seed = REGISTRY[EVAL_ID].seeds[0]
    cfg = from_dict(ReconConfig, load_yaml(CONFIG))
    configure_torch(cfg.num_threads)
    prior = load_prior(DEFAULT_PRIOR_PATH)
    scene, trials = synthetic_run(cfg.synthetic, SeedTree(seed))
    transducer = imaging_setup(scene.imaging).transducer
    model = scene_screen_model(scene, cfg.calibration.source_refinement)
    measured = trials.power_doppler["skull_das"]
    floor = torch.from_numpy(scene.noise_floor["skull_das"])

    def reconstruct(sub: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree = SeedTree(seed)
        calibration = calibrate_screen(
            prior,
            model,
            scene.screen_ct,
            transducer.element_x(),
            transducer.center_frequency,
            measured.mean(axis=0),
            scene.noise_floor["skull_das"],
            scene.cell_area,
            scene.effective_frames["skull_das"] * len(trials.label),
            scene.noise_frames * len(trials.label),
            cfg.calibration,
            tree,
            STREAM_SUB,
            sub,
        )
        with torch.no_grad():
            operator = power_doppler_operator(model, *screen_tensors(calibration.screen))
        samples = reconstruct_trials(
            prior,
            operator.to(torch.float32),
            floor,
            scene.cell_area,
            torch.from_numpy(scene.effective_frames["skull_das"]),
            scene.noise_frames,
            measured[:N_TRIALS],
            cfg.dps,
            tree,
            STREAM_SUB,
            sub,
        )
        return calibration.screen.delays, calibration.screen.amplitudes, samples

    first, again, other = reconstruct(0), reconstruct(0), reconstruct(1)
    differing = int(sum(np.sum(a != b) for a, b in zip(first, again)))
    changed_fraction = float(np.mean(first[2] != other[2]))
    report(
        EVAL_ID,
        {"differing_values": differing, "changed_fraction": changed_fraction},
        passed=differing == 0 and changed_fraction > 0.5,
    )
