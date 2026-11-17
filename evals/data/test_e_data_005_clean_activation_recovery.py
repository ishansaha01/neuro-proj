"""E-DATA-005: without a skull, the power Doppler time course follows the ground-truth a(t).

The whole pipeline is run (head, plane, banks, CT prior, operators, statistical
image series); only the paired clean series (`clean_das`) is looked at. The
regional signal is the activation-weighted sum of the floor-subtracted power
Doppler image.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.data.config import RunConfig
from neurosim.data.pipeline import run
from neurosim.data.types import CLEAN_DAS

EVAL_ID = "E-DATA-005"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
N_IMAGES = 60
IMAGE_INTERVAL = 2.0  # [s]: 0 to 118 s, three task blocks
N_FRAMES = 200  # frames per image (the fus default ensemble)
ELECTRONIC_SNR_DB = 20.0  # the noise level of configs/run_demo.yaml
MIN_CORRELATION = 0.9


@eval_case(
    id=EVAL_ID,
    claim=(
        "through the whole pipeline without a skull, the power Doppler signal of the "
        "activated region (activation-weighted, noise floor subtracted) tracks the "
        "ground-truth fractional CBV change a(t) of the task design"
    ),
    reference=(
        "The ground truth put in: a(t) = block design convolved with the hemodynamic response "
        "(head.cbv.task_timecourse, checked against numerical convolution by E-HEAD-009)"
    ),
    metric=(
        "correlation_seed_<s>: Pearson correlation over the 60 images between a(t) and the "
        "regional clean power Doppler signal; slope_seed_<s> (diagnostic, not in the verdict): "
        "least-squares slope of the regional fractional change against that of the CBV truth"
    ),
    threshold=(
        "correlation >= 0.9 for every seed (fixed in advance: with 200 frames a pixel "
        "fluctuates by 7 %, the regional sum over the roughly 25 activated pixels by about "
        "1.5 %, against a response whose standard deviation over the series is about 5 % of "
        "the regional signal; that predicts a correlation near 0.96, and 0.9 leaves room for "
        "small activated regions)"
    ),
    seeds=(0, 1, 2),
)
def test_e_data_005_clean_activation_recovery() -> None:
    base = from_dict(RunConfig, load_yaml(CONFIG))
    imaging = replace(
        base.imaging,
        sequence=replace(base.imaging.sequence, n_frames=N_FRAMES),
        noise=replace(base.imaging.noise, electronic_snr_db=ELECTRONIC_SNR_DB),
    )
    cfg = replace(
        base,
        imaging=imaging,
        trials=replace(base.trials, n_images=N_IMAGES, image_interval=IMAGE_INTERVAL),
    )
    values = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        scene, trials = run(cfg, seed)
        weights = scene.activation.astype(np.float64)
        floor = scene.noise_floor[CLEAN_DAS].reshape(scene.pixel_grid.shape)
        blood = trials.power_doppler[CLEAN_DAS].astype(np.float64) - floor
        regional = np.tensordot(blood, weights, axes=2)
        truth = np.tensordot(trials.cbv.astype(np.float64), weights, axes=2)
        amplitude = trials.amplitude.astype(np.float64)
        values[f"correlation_seed_{seed}"] = float(np.corrcoef(amplitude, regional)[0, 1])
        rest = np.abs(amplitude) < 0.05 * np.abs(amplitude).max()
        measured_change = regional / regional[rest].mean() - 1.0
        true_change = truth / truth[rest].mean() - 1.0
        values[f"slope_seed_{seed}"] = float(np.polyfit(true_change, measured_change, 1)[0])
    correlations = [values[f"correlation_seed_{seed}"] for seed in REGISTRY[EVAL_ID].seeds]
    report(EVAL_ID, values, passed=min(correlations) >= MIN_CORRELATION)
