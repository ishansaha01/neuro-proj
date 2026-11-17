"""The reconstruction experiment: every method on every scene, every metric, one table.

Per scene (`run_scene`):

1. Calibrate the phase screen once from the time-averaged `skull_das` image
   (`recon.calibrate`), starting from `scene.screen_ct`.
2. Assemble the phase-screen operators `H(screen_ct)`, `H(phi_hat)` and, as a
   measure of model mismatch only, `H(screen_true)` (`recon.forward`); if the
   calibration used a coarser forward model, correct the common amplitude of
   `phi_hat` for it (`recon.calibrate.match_total_power`).
3. Reconstruct every trial with every method (`METHODS`).
4. Compute the metrics of `recon.metrics` and time every method.

Which method "wins" is a result of the experiment, not something the code
assumes; the table is written as measured.

Seeding: every random draw comes from the `recon_sampling` stream through
`recon.seeding.torch_generator`, with sub-indices ``(Draw, scene index, ...)``;
within a DPS method the last sub-index is the posterior-sample number
(`recon.dps.reconstruct_trials`). The same seed gives the same results.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from collections.abc import Iterable, Iterator
from enum import IntEnum
from typing import Any

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.types import BEAMFORMERS, SKULL_DAS, Scene, TrialSet
from neurosim.fus.config import imaging_setup
from neurosim.recon.baselines import gamma_deconvolution, raw_power_doppler
from neurosim.recon.calibrate import CalibrationResult, calibrate_screen, match_total_power
from neurosim.recon.checkpoint import Prior
from neurosim.recon.config import ReconConfig
from neurosim.recon.dps import prior_samples, reconstruct_trials
from neurosim.recon.forward import (
    expected_power_doppler,
    power_doppler_operator,
    reference_bank,
    scene_screen_model,
    scene_screen_operators,
    screen_tensors,
)
from neurosim.recon.likelihood import power_doppler_deviance
from neurosim.recon.metrics import (
    activation_cnr,
    activation_dice,
    cbv_correlation,
    decoding_accuracy,
    delay_error,
    interval_coverage,
    stratified_folds,
)
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator

# Methods in the order of the results table. "pd_*" are the raw power Doppler images.
RAW_METHODS: tuple[str, ...] = tuple(f"pd_{key}" for key in BEAMFORMERS)
DECONVOLUTION_METHODS: tuple[str, ...] = ("deconv_true_h", "deconv_ct", "deconv_calibrated")
DPS_METHODS: tuple[str, ...] = ("dps_true_h", "dps_ct", "dps_calibrated")
PRIOR_ONLY = "prior_only"
METHODS: tuple[str, ...] = (*RAW_METHODS, *DECONVOLUTION_METHODS, *DPS_METHODS, PRIOR_ONLY)
IMAGE_METRICS: tuple[str, ...] = (
    "decoding_accuracy",
    "activation_dice",
    "cbv_correlation",
    "activation_cnr",
)

# Measured single-thread costs on the 2-core development machine (2026-11-17), used only
# by `estimate_seconds`. assumed, to verify on the machine that runs the experiment.
_SECONDS_PER_GUIDED_IMAGE_STEP = 9.0e-3  # U-Net forward and backward, batches of 30-60
_SECONDS_PER_UNGUIDED_IMAGE_STEP = 3.0e-3  # U-Net forward only
_SECONDS_PER_OPERATOR_VALUE = 1.2e-8  # per (frequency, pixel, source) of one assembly
_GRADIENT_COST_FACTOR = 4.0  # an operator gradient relative to an assembly
_LBFGS_EVALUATIONS_PER_ITERATION = 1.25  # function evaluations per L-BFGS iteration
_SECONDS_PER_FACTOR_VALUE = 9.0e-8  # per (frequency, element, source or pixel) of the factors


# Largest difference between the scene's and the prior's pixel coordinates that counts as
# rounding, in pixel spacings.
_GRID_TOLERANCE = 1.0e-6


class Draw(IntEnum):
    """First sub-index of the `recon_sampling` stream for each independent use."""

    CALIBRATION = 0
    DPS_TRUE_H = 1
    DPS_CT = 2
    DPS_CALIBRATED = 3
    PRIOR_ONLY = 4
    DECODER_FOLDS = 5


_DPS_DRAWS = {
    "dps_true_h": Draw.DPS_TRUE_H,
    "dps_ct": Draw.DPS_CT,
    "dps_calibrated": Draw.DPS_CALIBRATED,
}


def without_banks(scene: Scene) -> Scene:
    """A copy of a scene whose three Green's banks are replaced by a small placeholder.

    The experiment uses the stored operators and the array, frequency grid and
    reference speed of a bank, never `B` itself; at 128 elements and 136
    frequencies the three banks hold about 0.5 GB. The placeholder is
    `recon.forward.reference_bank`.
    """
    placeholder = reference_bank(scene)
    return dataclasses.replace(
        scene, bank_skull=placeholder, bank_clean=placeholder, bank_ct=placeholder
    )


def check_scene(scene: Scene, prior: Prior, cfg: ReconConfig) -> None:
    """Raise if a scene cannot be reconstructed with this prior and configuration.

    The pixel grid must be the prior's (up to rounding of the coordinates). `Scene`
    does not carry the source refinement its operators were built with; where its
    `meta` allows it to be inferred (the pipeline's cost record, or the synthetic
    helper's entry) it must equal `cfg.source_refinement`.
    """
    tolerance = _GRID_TOLERANCE * prior.pixel_grid.spacing
    same_grid = (
        scene.pixel_grid.shape == prior.pixel_grid.shape
        and abs(scene.pixel_grid.spacing - prior.pixel_grid.spacing) <= tolerance
        and np.allclose(scene.pixel_grid.origin, prior.pixel_grid.origin, rtol=0.0, atol=tolerance)
    )
    if not same_grid:
        raise ValueError(
            f"the scene's pixel grid {scene.pixel_grid} is not the prior's {prior.pixel_grid}"
        )
    refinement = scene.meta.get("source_refinement")
    if refinement is None and "cost" in scene.meta:
        refinement = round((scene.meta["cost"]["n_operator_passes"] / len(BEAMFORMERS)) ** 0.5)
    if refinement is not None and refinement != cfg.source_refinement:
        raise ValueError(
            f"the scene's operators use source_refinement = {refinement}, the reconstruction "
            f"config {cfg.source_refinement}"
        )


def _relative_error(operator: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.norm(operator - reference) / torch.linalg.norm(reference))


def _mean_deviance_per_pixel(
    operator: torch.Tensor, scene: Scene, trials: TrialSet, floor: torch.Tensor
) -> float:
    """Gamma deviance per pixel of the measured images under an operator and the true CBV."""
    cbv = torch.from_numpy(np.ascontiguousarray(trials.cbv, dtype=np.float32)).flatten(1)
    measured = torch.from_numpy(
        np.ascontiguousarray(trials.power_doppler[SKULL_DAS], dtype=np.float32)
    ).flatten(1)
    expected = expected_power_doppler(operator, cbv, scene.cell_area, floor)
    blood_frames = torch.from_numpy(np.ascontiguousarray(scene.effective_frames[SKULL_DAS]))
    deviance = power_doppler_deviance(measured, expected, floor, blood_frames, scene.noise_frames)
    return float(deviance.mean()) / measured.shape[1]


def reconstruct_scene(
    scene: Scene, trials: TrialSet, prior: Prior, cfg: ReconConfig, tree: SeedTree, scene_index: int
) -> dict[str, Any]:
    """Run every method on one scene.

    scene, trials: the scene (see `without_banks`) and its image series.
    prior: the trained prior; its pixel grid must be the scene's.
    cfg: reconstruction settings.
    tree: seed tree of the experiment.
    scene_index: index of the scene in the dataset (a seed sub-index).

    Returns a dict with
        ``images``: method -> point estimate per trial, float32, (n_trials, nz, nx)
            (for the DPS methods the mean of the posterior samples);
        ``samples``: DPS method -> posterior samples, (n_samples, n_trials, nz, nx);
        ``seconds``: method -> run time [s] (operator assembly included where a method
            needs its own; the calibration is counted in `dps_calibrated` and
            `deconv_calibrated`);
        ``calibration``: the `CalibrationResult`;
        ``diagnostics``: model-mismatch numbers (relative Frobenius error of each
            phase-screen operator against the true `skull_das` operator, and the gamma
            deviance per pixel of the measured images under each operator with the true CBV).
    """
    check_scene(scene, prior, cfg)
    setup = imaging_setup(scene.imaging)
    measured = np.asarray(trials.power_doppler[SKULL_DAS], dtype=np.float32)
    n_trials = measured.shape[0]
    floor = torch.from_numpy(np.ascontiguousarray(scene.noise_floor[SKULL_DAS], dtype=np.float32))
    true_operator = torch.from_numpy(np.ascontiguousarray(scene.operators[SKULL_DAS]))
    blood_frames = np.ascontiguousarray(scene.effective_frames[SKULL_DAS], dtype=np.float32)
    seconds: dict[str, float] = {}

    start = time.perf_counter()
    calibration_model = scene_screen_model(scene, cfg.calibration.source_refinement)
    calibration = calibrate_screen(
        prior,
        calibration_model,
        scene.screen_ct,
        setup.transducer.element_x(),
        setup.transducer.center_frequency,
        measured.mean(axis=0),
        scene.noise_floor[SKULL_DAS],
        scene.cell_area,
        blood_frames * n_trials,
        scene.noise_frames * n_trials,
        cfg.calibration,
        tree,
        Draw.CALIBRATION,
        scene_index,
    )
    with torch.no_grad():
        calibration_total = float(
            power_doppler_operator(calibration_model, *screen_tensors(calibration.screen)).sum()
        )
    del calibration_model
    # The calibration's closures form reference cycles that keep its forward model alive.
    gc.collect()
    calibration_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ct_operator, calibrated_operator, screen_true_operator = scene_screen_operators(
        scene, (scene.screen_ct, calibration.screen, scene.screen_true), cfg.source_refinement
    )
    # The third operator is a diagnostic; each method is charged one assembly.
    assembly = (time.perf_counter() - start) / 3.0
    calibrated_screen, calibrated_operator = match_total_power(
        calibration.screen, calibrated_operator, calibration_total
    )
    calibration = dataclasses.replace(calibration, screen=calibrated_screen)
    operators = {"true_h": true_operator, "ct": ct_operator, "calibrated": calibrated_operator}
    assembly_seconds = {"true_h": 0.0, "ct": assembly, "calibrated": assembly + calibration_seconds}

    images: dict[str, np.ndarray] = {}
    for key, series in raw_power_doppler(trials).items():
        images[f"pd_{key}"] = series
        seconds[f"pd_{key}"] = 0.0
    samples: dict[str, np.ndarray] = {}
    for name, operator in operators.items():
        start = time.perf_counter()
        images[f"deconv_{name}"] = gamma_deconvolution(
            operator,
            floor,
            scene.cell_area,
            torch.from_numpy(blood_frames),
            scene.noise_frames,
            measured,
            cfg.deconvolution,
        )
        seconds[f"deconv_{name}"] = time.perf_counter() - start + assembly_seconds[name]
        method = f"dps_{name}"
        start = time.perf_counter()
        samples[method] = reconstruct_trials(
            prior,
            operator,
            floor,
            scene.cell_area,
            torch.from_numpy(blood_frames),
            scene.noise_frames,
            measured,
            cfg.dps,
            tree,
            _DPS_DRAWS[method],
            scene_index,
        )
        images[method] = samples[method].mean(axis=0)
        seconds[method] = time.perf_counter() - start + assembly_seconds[name]
    start = time.perf_counter()
    images[PRIOR_ONLY] = prior_samples(prior, n_trials, cfg.dps, tree, Draw.PRIOR_ONLY, scene_index)
    seconds[PRIOR_ONLY] = time.perf_counter() - start

    diagnostics = {
        f"operator_relative_error_{name}": _relative_error(operator, true_operator)
        for name, operator in (
            ("screen_true", screen_true_operator),
            ("screen_ct", operators["ct"]),
            ("screen_calibrated", operators["calibrated"]),
        )
    }
    for name, operator in (
        ("true_h", true_operator),
        ("screen_true", screen_true_operator),
        ("screen_ct", operators["ct"]),
        ("screen_calibrated", operators["calibrated"]),
    ):
        diagnostics[f"deviance_per_pixel_{name}"] = _mean_deviance_per_pixel(
            operator, scene, trials, floor
        )
    return {
        "images": images,
        "samples": samples,
        "seconds": seconds,
        "calibration": calibration,
        "calibration_seconds": calibration_seconds,
        "diagnostics": diagnostics,
    }


def scene_metrics(
    scene: Scene,
    trials: TrialSet,
    prior: Prior,
    outputs: dict[str, Any],
    cfg: ReconConfig,
    tree: SeedTree,
    scene_index: int,
) -> dict[str, Any]:
    """Every metric of every method for one scene, as plain data.

    Returns ``{"methods": {method: {metric: value}}, "screens": {...},
    "calibration": {...}, "diagnostics": {...}}``. Per method: the four
    `IMAGE_METRICS`, ``seconds``, and for the DPS methods ``coverage_<level>`` per
    nominal level of `cfg.metrics.coverage_levels` (intervals in the prior's model
    space, brain pixels). Screens: gauge-removed rms delay error [ns] of the CT
    and of the calibrated screen against `scene.screen_true`, and the rms error of
    their log amplitudes [Np].
    """
    metric_cfg = cfg.metrics
    labels = np.asarray(trials.label)
    order = torch.randperm(
        labels.shape[0],
        generator=torch_generator(tree, SAMPLING_STREAM, Draw.DECODER_FOLDS, scene_index),
    ).numpy()
    folds = stratified_folds(labels, metric_cfg.n_folds, order)
    mask = np.asarray(scene.brain_mask, dtype=bool)
    truth_model = prior.transform.to_model(np.asarray(trials.cbv, dtype=np.float64))
    levels = np.asarray(metric_cfg.coverage_levels)

    methods: dict[str, dict[str, float]] = {}
    for method in METHODS:
        series = outputs["images"][method]
        values = {
            "decoding_accuracy": decoding_accuracy(series, labels, folds, metric_cfg.n_components),
            "activation_dice": activation_dice(
                series,
                labels,
                scene.activation,
                mask,
                metric_cfg.t_threshold,
                metric_cfg.activation_threshold,
            ),
            "cbv_correlation": cbv_correlation(series, trials.cbv, mask),
            "activation_cnr": activation_cnr(
                series,
                labels,
                scene.activation,
                mask,
                metric_cfg.activation_threshold,
                metric_cfg.reference_threshold,
            ),
            "seconds": outputs["seconds"][method],
        }
        if method in outputs["samples"]:
            sample_model = prior.transform.to_model(outputs["samples"][method].astype(np.float64))
            coverage = interval_coverage(sample_model[:, :, mask], truth_model[:, mask], levels)
            values.update(
                {f"coverage_{level:g}": float(value) for level, value in zip(levels, coverage)}
            )
        methods[method] = values

    element_x = imaging_setup(scene.imaging).transducer.element_x()
    calibration: CalibrationResult = outputs["calibration"]
    true_log_amplitudes = np.log(scene.screen_true.amplitudes)
    screens = {}
    for name, screen in (("ct", scene.screen_ct), ("calibrated", calibration.screen)):
        log_error = np.log(screen.amplitudes) - true_log_amplitudes
        screens[f"delay_error_{name}_ns"] = 1.0e9 * delay_error(
            screen.delays, scene.screen_true.delays, element_x, cfg.calibration.gauge
        )
        screens[f"log_amplitude_error_{name}_mean"] = float(log_error.mean())
        screens[f"log_amplitude_error_{name}_rms"] = float(np.sqrt(np.mean(log_error**2)))
    return {
        "methods": methods,
        "screens": screens,
        "calibration": {
            "selected": calibration.selected,
            "deviance_per_pixel": list(calibration.deviance_per_pixel),
            "start_deviance_per_pixel": calibration.start_deviance_per_pixel,
            "seconds": outputs["calibration_seconds"],
        },
        "diagnostics": outputs["diagnostics"],
    }


def example_images(trials: TrialSet, outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Truth and every method's image for the first rest and the first task trial of a scene.

    Returns arrays of shape (2, nz, nx) (rest, task) keyed ``truth`` and by method.
    """
    labels = np.asarray(trials.label)
    chosen = [int(np.flatnonzero(labels == state)[0]) for state in (0, 1)]
    examples = {"truth": np.asarray(trials.cbv)[chosen]}
    examples.update({method: outputs["images"][method][chosen] for method in METHODS})
    return examples


def summarise(scenes: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and spread over scenes of every metric of every method.

    Returns ``{method: {metric: {"mean", "std", "n"}}}`` (std with ddof 1; 0 for one
    scene), plus the pseudo-method ``"screens"`` for the screen errors.
    """
    summary: dict[str, dict[str, dict[str, float]]] = {}
    groups = {method: [scene["methods"][method] for scene in scenes] for method in METHODS}
    groups["screens"] = [scene["screens"] for scene in scenes]
    for name, records in groups.items():
        summary[name] = {}
        for metric in records[0]:
            values = np.array([record[metric] for record in records], dtype=np.float64)
            summary[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                "n": int(values.shape[0]),
            }
    return summary


def markdown_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    """The results table: one row per method, mean +- standard deviation over scenes."""
    metrics = [*IMAGE_METRICS]
    coverage = sorted(
        {
            metric
            for method in DPS_METHODS
            for metric in summary[method]
            if metric.startswith("coverage_")
        }
    )
    columns = [*metrics, *coverage, "seconds"]
    lines = [
        "| method | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for method in METHODS:
        cells = []
        for column in columns:
            entry = summary[method].get(column)
            cells.append("" if entry is None else f"{entry['mean']:.3f} ± {entry['std']:.3f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| screen | " + " | ".join(summary["screens"]) + " |")
    lines.append("|---|" + "---|" * len(summary["screens"]))
    screen_cells = [
        f"{entry['mean']:.3f} ± {entry['std']:.3f}" for entry in summary["screens"].values()
    ]
    lines.append("| mean ± std | " + " | ".join(screen_cells) + " |")
    return "\n".join(lines) + "\n"


def estimate_seconds(
    scene: Scene, n_trials: int, n_scenes: int, cfg: ReconConfig
) -> dict[str, float]:
    """Rough single-thread run time of the experiment, from one scene's sizes.

    Returns seconds per stage (``factors``, ``calibration``, ``assembly``, ``dps``,
    ``prior_only``) summed over `n_scenes`, and their ``total``.
    """
    reference = scene.bank_skull
    nf, n_elements = reference.freqs.shape[0], reference.transducer.n_elements
    n_pix = scene.pixel_grid.shape[0] * scene.pixel_grid.shape[1]
    calibration = cfg.calibration

    def assembly(n_selected: int, refinement: int) -> float:
        return _SECONDS_PER_OPERATOR_VALUE * nf * n_selected * n_pix * refinement**2

    refinements = {cfg.source_refinement, calibration.source_refinement}
    factors = sum(
        _SECONDS_PER_FACTOR_VALUE * nf * n_elements * n_pix * (1 + refinement**2)
        for refinement in refinements
    )
    gradients = (
        calibration.n_updates * calibration.lbfgs_iterations * _LBFGS_EVALUATIONS_PER_ITERATION
    )
    per_candidate = (
        gradients
        * _GRADIENT_COST_FACTOR
        * assembly(min(calibration.n_pixels, n_pix), calibration.source_refinement)
        + (calibration.n_updates + 1) * assembly(n_pix, calibration.source_refinement)
        + calibration.sampler.n_steps
        * calibration.sampler.n_samples
        * _SECONDS_PER_GUIDED_IMAGE_STEP
    )
    image_steps = n_trials * cfg.dps.n_samples * cfg.dps.n_steps
    stages = {
        "factors": factors,
        "calibration": calibration.n_candidates * per_candidate,
        "assembly": 3 * assembly(n_pix, cfg.source_refinement),
        "dps": len(DPS_METHODS) * image_steps * _SECONDS_PER_GUIDED_IMAGE_STEP,
        "prior_only": n_trials * cfg.dps.n_steps * _SECONDS_PER_UNGUIDED_IMAGE_STEP,
    }
    stages = {name: n_scenes * value for name, value in stages.items()}
    stages["total"] = sum(stages.values())
    return stages


def run_experiment(
    dataset: Iterable[tuple[Scene, TrialSet]],
    prior: Prior,
    cfg: ReconConfig,
    root_seed: int,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Run the experiment scene by scene.

    dataset: `(scene, trials)` pairs, e.g. `data.sweep.load_dataset(directory)`.
    prior: the trained prior.
    cfg: reconstruction settings.
    root_seed: root seed of the experiment's `SeedTree`.

    Yields, per scene, the metrics record of `scene_metrics` (with ``"index"`` and
    ``"n_trials"``) and the `example_images` of the scene. Call
    `recon.seeding.configure_torch(cfg.num_threads)` first.
    """
    tree = SeedTree(root_seed)
    scene_index = 0
    # No `enumerate`: it keeps the last yielded pair, and with it the scene's banks, alive.
    for scene, trials in dataset:
        scene = without_banks(scene)
        outputs = reconstruct_scene(scene, trials, prior, cfg, tree, scene_index)
        record = scene_metrics(scene, trials, prior, outputs, cfg, tree, scene_index)
        record.update(index=scene_index, n_trials=int(np.asarray(trials.label).shape[0]))
        yield record, example_images(trials, outputs)
        scene_index += 1
