"""E-DATA-009: through a real skull bank, the scatterer tier averages to the statistical prediction.

Extends E-FUS-014 (water, a synthetic CBV blob, blood only) to the pipeline: the
banks of a generated head (ray backend), its CBV map, all four beamformers and
electronic noise. The image series is simulated at scatterer fidelity
(`fus.acquire.acquire_power_doppler` through `data.trials.simulate_trials`) and
its mean over images and pixels is compared with the image the scene predicts,
`operators[key] @ (cbv * cell_area) + noise_floor[key]`.

To isolate the claim, what the statistical tier does not model is switched off:
tissue clutter and motion, and the SVD filter (no component removed). The noise
level is set so that blood and floor both matter: 25 dB puts the uncorrected
blood signal near the floor, the corrected ones a few dB above it, and the
clean image far above it.

History: the first version used 24 images of 2 frames and failed for skull_ct
(-8.2 % against the 6 % bar; the other keys were within 3.6 %). The scatter of
the per-image sums showed why: the standard error of the skull_ct sum is twice
that of the other keys (about 5 % at 24 images, not the 2.5 % assumed), because
the CT model's unit-gain normalisation concentrates the sum in fewer pixels and
its aberrated speckle cells are larger. The simulator and the bar were not
changed; the same number of frames is now spent on 96 images of one frame, and
the measured standard error of every key is reported next to its error.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene_with_extras
from neurosim.data.trials import simulate_trials
from neurosim.data.types import BEAMFORMERS

EVAL_ID = "E-DATA-009"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
PIXELS = (16, 16)
PIXEL_SPACING = 0.5e-3  # [m]: an 8 mm x 8 mm image
SOURCE_REFINEMENT = 3  # 0.17 mm sub-cells, a fifth of a wavelength
N_IMAGES = 96  # independent blood realisations
N_FRAMES = 1  # frames per image
N_BLOOD = 800  # about 12 per mm^2 of image
ELECTRONIC_SNR_DB = 25.0
MAX_ERROR = 0.06


@eval_case(
    id=EVAL_ID,
    claim=(
        "for the banks and CBV map of a generated head, the mean scatterer-fidelity power "
        "Doppler image of every beamformer equals the statistical prediction "
        "H @ (cbv * cell_area) + noise floor"
    ),
    reference=(
        "Expected power of independent random-sign scatterers (the sum of their powers) plus "
        "white noise through the beamformer; the explicit frame-by-frame simulation is the "
        "reference for the operator and floor the reconstruction will use"
    ),
    metric=(
        "error_<key>: sum over 96 images and all pixels of the scatterer-tier power Doppler / "
        "sum of the prediction, minus 1; diagnostics: standard_error_<key>, the standard "
        "error of that ratio from the scatter of the per-image sums, and blood_fraction_<key>, "
        "the share of blood in the predicted sum"
    ),
    threshold=(
        "|error| < 0.06 for every key (fixed in advance, before the first run, as two "
        "standard errors of 2.5 % plus about 1 % for the 3 x 3 sub-cell quadrature of the "
        "operator and the bilinear bank interpolation; with 96 images the measured standard "
        "error is 1 to 2.5 %, so the bar is at least 2.4 standard errors for every key)"
    ),
    seeds=(0,),
)
def test_e_data_009_tier_agreement_through_skull() -> None:
    base = from_dict(RunConfig, load_yaml(CONFIG))
    imaging = replace(
        base.imaging,
        pixels=replace(base.imaging.pixels, shape=PIXELS, spacing=PIXEL_SPACING),
        sequence=replace(base.imaging.sequence, n_frames=N_FRAMES),
        doppler=replace(base.imaging.doppler, cutoff_rule="fixed", n_discard=0),
        noise=replace(
            base.imaging.noise,
            electronic_snr_db=ELECTRONIC_SNR_DB,
            tissue_clutter=False,
            tissue_motion=False,
        ),
        scatterers=replace(base.imaging.scatterers, n_blood=N_BLOOD),
    )
    cfg = replace(
        base,
        imaging=imaging,
        source_refinement=SOURCE_REFINEMENT,
        trials=replace(base.trials, n_images=N_IMAGES, image_interval=5.0, fidelity="scatterer"),
    )
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    scene, extras = build_scene_with_extras(cfg, tree)
    trials = simulate_trials(scene, cfg, tree, extras)

    cells = trials.cbv.reshape(N_IMAGES, -1).astype(np.float64) * scene.cell_area
    values = {}
    for key in BEAMFORMERS:
        blood = cells @ scene.operators[key].astype(np.float64).T
        floor = scene.noise_floor[key].astype(np.float64)
        predicted = blood.sum() + N_IMAGES * floor.sum()
        measured = trials.power_doppler[key].sum(dtype=np.float64)
        values[f"error_{key}"] = float(measured / predicted - 1.0)
        per_image = trials.power_doppler[key].reshape(N_IMAGES, -1).sum(axis=1, dtype=np.float64)
        per_image /= blood.sum(axis=1) + floor.sum()
        values[f"standard_error_{key}"] = float(per_image.std(ddof=1) / np.sqrt(N_IMAGES))
        values[f"blood_fraction_{key}"] = float(blood.sum() / predicted)
    errors = [abs(values[f"error_{key}"]) for key in BEAMFORMERS]
    report(EVAL_ID, values, passed=max(errors) < MAX_ERROR)
