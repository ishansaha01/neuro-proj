"""E-DATA-008: beamforming with the true skull model recovers more of the clean image.

Compared are the expected blood images `operators[key] @ (cbv * cell_area)` of
the resting CBV map (no speckle, no noise floor), over all pixels of the image.
The similarity of an image to the paired clean image is their Pearson
correlation, which ignores the overall scale: the corrected beamformer is
normalised to unit scatterer gain under its model, so its image differs from
conventional DAS by many dB in level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene
from neurosim.data.types import CLEAN_DAS, SKULL_CT, SKULL_DAS, SKULL_TRUE

EVAL_ID = "E-DATA-008"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"


@eval_case(
    id=EVAL_ID,
    claim=(
        "through the real skull, the image beamformed with the true skull model (skull_true) "
        "is more similar to the paired no-skull image than the conventional DAS image "
        "(skull_das)"
    ),
    reference=(
        "The paired clean image of the same scene (ROADMAP D-010) and the ordering of the "
        "baselines of README stage 3: true-skull correction above uncorrected beamforming"
    ),
    metric=(
        "correlation_<key>_seed_<s>: Pearson correlation over the pixels between the expected "
        "blood image of the key and that of clean_das; the CT-corrected value is reported as "
        "a diagnostic"
    ),
    threshold=(
        "correlation of skull_true > correlation of skull_das for every seed (fixed in "
        "advance: an ordering of noise-free, deterministic numbers; the skull is about 1000 m/s "
        "faster than the reference over 6.5 mm, which displaces the uncorrected image by about "
        "2.5 mm, three pixels, while the true model removes the displacement)"
    ),
    seeds=(0, 1, 2, 3, 4),
)
def test_e_data_008_correction_ordering() -> None:
    cfg = from_dict(RunConfig, load_yaml(CONFIG))
    values = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        scene = build_scene(cfg, SeedTree(seed))
        cells = scene.cbv_baseline.ravel().astype(np.float64) * scene.cell_area
        images = {key: scene.operators[key].astype(np.float64) @ cells for key in scene.operators}
        for key in (SKULL_DAS, SKULL_CT, SKULL_TRUE):
            values[f"correlation_{key}_seed_{seed}"] = float(
                np.corrcoef(images[key], images[CLEAN_DAS])[0, 1]
            )
    passed = all(
        values[f"correlation_{SKULL_TRUE}_seed_{seed}"]
        > values[f"correlation_{SKULL_DAS}_seed_{seed}"]
        for seed in REGISTRY[EVAL_ID].seeds
    )
    report(EVAL_ID, values, passed=passed)
