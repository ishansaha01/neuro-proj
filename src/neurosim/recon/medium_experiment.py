"""The experiment of the second design: a medium-space skull model calibrated by re-beamforming.

Question: does calibrating the CT-derived skull model move the CT-based result
toward the result with the true model? Per scene (`reconstruct_scene`):

1. Build the skull-model family from the CT (`recon.skull_model`), calibrate
   it on the focus criterion (`recon.focus`, through `recon.scanner`), and
   re-trace the calibrated member once with the real backend.
2. Seven beamformed image series. Stored in the dataset: no correction
   (`skull_das`), the nominal CT model with its own amplitudes (`ct_nominal`,
   the dataset's `skull_ct`), the true model (`true`, the dataset's
   `skull_true`) and no skull (`clean_das`). Re-beamformed by `recon.scanner`
   with a model's delays only (`recon.skull_model.delay_only`): the nominal CT
   model (`ct`), the calibrated model (`calibrated`) and, as the oracle of a
   delay-only correction, the true bank (`true_delays`).
3. On the series of `ct`, `calibrated`, `true_delays` and `true`: gamma
   deconvolution and diffusion posterior sampling of the rest mean and the task
   mean with the operator of that model (`recon.medium_recon`): for the three
   delay-only models the operator `|A(M; w(M))|^2` of the delay-only bank with
   one matched common amplitude, for `true` the dataset's exact operator.
4. Metrics (`scene_metrics`).

The comparison that isolates the calibration is `ct` -> `calibrated` ->
`true_delays`: one pipeline, three sets of delays. `true` adds exact amplitudes;
`ct_nominal` shows what the CT model's amplitudes do to a normalised image.

Rows and what they are scored on. Baseline CBV: the row's rest map (mean rest
image minus the beamformer's noise floor, or the reconstruction of the mean
rest image; DPS: mean of the posterior samples). Activation: the row's task map
minus its rest map. Decoding of single trials: (i) PCA + LDA
(`recon.metrics.decoding_accuracy`) on the row's single images (deconvolution
rows: every image deconvolved; DPS rows: not computed, because sampling 120
single-look images per row and scene does not fit the run-time budget and is
the wrong estimator for single looks); (ii) the cross-validated matched filter
of `recon.decoding` on the single beamformed images, with the template
estimated on the training folds: the difference of their condition means, or
for a reconstruction row the image difference its two training-fold
reconstructions predict. The prior-only control is three unconditional samples:
one as "rest map", the difference of the other two as "activation map".

Ground truth is read here for the metrics only (`_Truth`, `skull_diagnostics`).

Failures. A row whose maps are not all finite for a scene, or whose model
cannot be amplitude-matched because the mean image does not rise above the
noise floor, is recorded as failed for that scene with the reason and is left
out of that row's statistics; it never stops the run and is never replaced by
other numbers (`failures`).

Seeding: `recon_sampling` sub-indices `(MediumDraw, scene index, ...)`; the
speckle of a re-beamformed series comes from `speckle_statistical` with
sub-indices `(scene index, model number, image)` of the experiment's own seed tree.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.types import CLEAN_DAS, SKULL_CT, SKULL_DAS, SKULL_TRUE, Scene, TrialSet
from neurosim.fus.config import imaging_setup
from neurosim.fus.types import GreensBank
from neurosim.recon import scanner
from neurosim.recon.baselines import gamma_deconvolution
from neurosim.recon.checkpoint import Prior
from neurosim.recon.config import ReconConfig
from neurosim.recon.decoding import matched_filter_accuracy
from neurosim.recon.dps import prior_samples
from neurosim.recon.experiment import check_scene
from neurosim.recon.focus import FocusResult, calibrate, pixel_groups
from neurosim.recon.forward import reference_bank
from neurosim.recon.medium_recon import (
    BeamformedSeries,
    condition_means,
    deconvolve_means,
    match_common_amplitude,
    model_operator,
    predicted_difference,
    sample_means,
)
from neurosim.recon.metrics import (
    bank_delays,
    decoding_accuracy,
    half_maximum_dice,
    map_cnr,
    map_correlation,
    non_finite_reason,
    stratified_folds,
    travel_time_error,
    weighted_delay_rms,
)
from neurosim.recon.rebeamform import weight_model
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator
from neurosim.recon.skull_model import (
    SkullFamily,
    build_family,
    delay_only,
    fermat_bank,
    model_bank,
)

# Skull models with an inversion on top (module docstring); the first three are delay-only
# models whose series is re-beamformed, the last is the dataset's true-model beamformer.
REBEAMFORMED_MODELS: tuple[str, ...] = ("ct", "calibrated", "true_delays")
MODELS: tuple[str, ...] = (*REBEAMFORMED_MODELS, "true")
BEAMFORMED_METHODS: tuple[str, ...] = (
    "pd_skull_das",
    "pd_ct_nominal",
    "pd_ct",
    "pd_calibrated",
    "pd_true_delays",
    "pd_true",
    "pd_clean_das",
)
DECONVOLUTION_METHODS: tuple[str, ...] = tuple(f"deconv_{model}" for model in MODELS)
DPS_METHODS: tuple[str, ...] = tuple(f"dps_{model}" for model in MODELS)
PRIOR_ONLY = "prior_only"
METHODS: tuple[str, ...] = (
    *BEAMFORMED_METHODS,
    *DECONVOLUTION_METHODS,
    *DPS_METHODS,
    PRIOR_ONLY,
)
METRICS: tuple[str, ...] = (
    "baseline_correlation",
    "activation_correlation",
    "activation_dice",
    "activation_cnr",
    "decoding_pca_lda",
    "decoding_matched_filter",
    "signal_to_floor",
    "seconds",
)

# Measured single-thread costs on the 2-core development machine (2026-11-17, one other
# job running), used only by `estimate_seconds`. assumed, to verify on the machine that
# runs the experiment.
_SECONDS_PER_TRACE_VALUE = 1.7e-6  # bent-ray trace, per (element, medium grid node)
_SECONDS_PER_HISTOGRAM_SAMPLE = 7.5e-8  # path histograms, per (element, ROI node, grid row)
_SECONDS_PER_CRITERION_VALUE = 7.0e-10  # criterion gradient, per (f, pixel, element, source)
_LBFGS_EVALUATIONS_PER_ITERATION = 1.3
_SECONDS_PER_OPERATOR_VALUE = 2.6e-10  # operator pass, per (f, pixel, element, source)
_SECONDS_PER_GUIDED_IMAGE_STEP = 1.8e-2  # U-Net forward and backward, batch of one
_SECONDS_PER_UNGUIDED_IMAGE_STEP = 3.0e-3
_SECONDS_PER_DECONVOLVED_IMAGE = 1.5e-2  # single trials and the fold templates


class MediumDraw(IntEnum):
    """First sub-index of the `recon_sampling` stream for each independent use."""

    FOCUS_PIXELS = 10
    DECODER_FOLDS = 11
    PRIOR_ONLY = 12
    DPS_CT = 13
    DPS_CALIBRATED = 14
    DPS_TRUE = 15
    DPS_TRUE_DELAYS = 16


_DPS_DRAWS = {
    "ct": MediumDraw.DPS_CT,
    "calibrated": MediumDraw.DPS_CALIBRATED,
    "true_delays": MediumDraw.DPS_TRUE_DELAYS,
    "true": MediumDraw.DPS_TRUE,
}
# Stored beamformer whose independent-frame counts a re-beamformed series is given
# (`recon.scanner`): the one with the same or the closest delays.
_FRAMES_LIKE = {"ct": SKULL_CT, "calibrated": SKULL_TRUE, "true_delays": SKULL_TRUE}
# Last sub-index of the DPS streams: the full condition means, or those of a training fold.
_FULL_DATA, _TRAINING_FOLD = 0, 1


@dataclass(frozen=True)
class SkullCalibration:
    """The calibrated skull model of a scene and how it was obtained.

    Attributes:
        family: the skull-model family.
        focus: outcome of the focus calibration.
        bank: the calibrated member's bank from the real backend (common amplitude 1).
        seconds: run time of each step [s]: `family`, `focus`, `retrace`.
    """

    family: SkullFamily
    focus: FocusResult
    bank: GreensBank
    seconds: dict[str, float]


def calibrate_skull(
    scene: Scene, trials: TrialSet, cfg: ReconConfig, tree: SeedTree, scene_index: int
) -> SkullCalibration:
    """Build the family, calibrate it by re-beamforming and re-trace the result (step 1)."""
    medium = cfg.medium
    start = time.perf_counter()
    family = build_family(scene, medium.skull_model)
    seconds = {"family": time.perf_counter() - start}

    start = time.perf_counter()
    brain = np.flatnonzero(np.asarray(scene.brain_mask, dtype=bool).reshape(-1))
    rng = tree.rng(SAMPLING_STREAM, MediumDraw.FOCUS_PIXELS, scene_index)
    n_pixels = min(medium.focus.n_pixels, brain.shape[0])
    pixels = np.sort(rng.choice(brain, size=n_pixels, replace=False))
    record = scanner.record_channel_data(scene, trials, medium.focus.source_refinement)
    sequence = imaging_setup(scene.imaging).sequence
    weights = [
        weight_model(family, scene.pixel_grid, sequence, scene.imaging.beamforming, group)
        for group in pixel_groups(pixels, medium.focus.pixels_per_pass)
    ]
    focus = calibrate(record, family, weights, medium.focus)
    del record, weights
    # The optimiser's closures form reference cycles that keep the channel record alive.
    gc.collect()
    seconds["focus"] = time.perf_counter() - start

    start = time.perf_counter()
    bank = model_bank(family, focus.parameters)
    seconds["retrace"] = time.perf_counter() - start
    return SkullCalibration(family=family, focus=focus, bank=bank, seconds=seconds)


def _stored_series(scene: Scene, trials: TrialSet, key: str) -> BeamformedSeries:
    return BeamformedSeries(
        images=np.asarray(trials.power_doppler[key], dtype=np.float32),
        noise_floor=np.asarray(scene.noise_floor[key], dtype=np.float32),
        effective_frames=np.asarray(scene.effective_frames[key], dtype=np.float32),
        noise_frames=scene.noise_frames,
    )


def _signal_to_floor(series: BeamformedSeries, mask: np.ndarray) -> float:
    """Mean over the brain pixels of (time-averaged image - floor) / floor, from the data."""
    mean_image = series.images.mean(axis=0, dtype=np.float64).reshape(-1)
    floor = series.noise_floor.astype(np.float64)
    return float(np.mean(((mean_image - floor) / floor)[mask.reshape(-1)]))


def reconstruct_scene(
    scene: Scene, trials: TrialSet, prior: Prior, cfg: ReconConfig, tree: SeedTree, scene_index: int
) -> dict[str, Any]:
    """Run every row of the experiment on one scene (module docstring).

    Returns a dict with
        ``calibration``: the `SkullCalibration`;
        ``series``: beamformed method -> `BeamformedSeries`;
        ``rest``, ``task``: method -> map, float32, (nz, nx) (CBV for the inversion
            rows, power Doppler above the noise floor for the beamformed rows);
        ``trial_images``: method -> single-trial images for the PCA + LDA decoder,
            (n, nz, nx), for the beamformed and the deconvolution rows;
        ``decoding_matched_filter``: method -> cross-validated accuracy;
        ``amplitude_scale``: model -> power scale of its operator (1 for `true`);
        ``seconds``: method -> run time [s];
        ``folds``: fold index of every trial;
        ``failed``: method -> reason, for the rows that could not be computed
            (module docstring); such a row has no other entry than ``seconds``.
    """
    check_scene(scene, prior, cfg)
    medium = cfg.medium
    labels = np.asarray(trials.label)
    mask = np.asarray(scene.brain_mask, dtype=bool)
    order = torch.randperm(
        labels.shape[0],
        generator=torch_generator(tree, SAMPLING_STREAM, MediumDraw.DECODER_FOLDS, scene_index),
    ).numpy()
    folds = stratified_folds(labels, cfg.metrics.n_folds, order)
    reference_cbv = medium.reference_cbv or scene.imaging.noise.cbv_reference

    calibration = calibrate_skull(scene, trials, cfg, tree, scene_index)
    calibration_seconds = sum(calibration.seconds.values())

    series = {
        "pd_skull_das": _stored_series(scene, trials, SKULL_DAS),
        "pd_ct_nominal": _stored_series(scene, trials, SKULL_CT),
        "pd_true": _stored_series(scene, trials, SKULL_TRUE),
        "pd_clean_das": _stored_series(scene, trials, CLEAN_DAS),
    }
    seconds = {method: 0.0 for method in series}
    operators: dict[str, np.ndarray] = {}
    assembly_seconds = {"true": 0.0}
    # One delay-only bank at a time: each holds as much memory as a bank of the scene.
    for number, model in enumerate(REBEAMFORMED_MODELS):
        start = time.perf_counter()
        full_bank = {
            "ct": scene.bank_ct,
            "calibrated": calibration.bank,
            "true_delays": scanner.true_model_bank(scene),
        }[model]
        bank = delay_only(full_bank)
        rebeamformed = scanner.rebeamform_trials(
            scene, trials, bank, cfg.source_refinement, _FRAMES_LIKE[model],
            tree, scene_index, number,
        )
        series[f"pd_{model}"] = BeamformedSeries(
            rebeamformed.images,
            rebeamformed.noise_floor,
            rebeamformed.effective_frames,
            scene.noise_frames,
        )
        seconds[f"pd_{model}"] = time.perf_counter() - start
        start = time.perf_counter()
        operators[model] = model_operator(scene, bank, cfg.source_refinement)
        assembly_seconds[model] = time.perf_counter() - start
        del bank, full_bank
    seconds["pd_calibrated"] += calibration_seconds

    rest, task, trial_images, matched, failed = {}, {}, {}, {}, {}

    def keep(method: str, rest_map: np.ndarray, task_map: np.ndarray) -> bool:
        """Store a row's two maps, or record why they cannot be scored."""
        reason = non_finite_reason({"rest map": rest_map, "task map": task_map})
        if reason is None:
            rest[method], task[method] = rest_map, task_map
        else:
            failed[method] = reason
        return reason is None

    for method in BEAMFORMED_METHODS:
        data = series[method]
        means = condition_means(data.images, labels)
        floor_map = data.noise_floor.reshape(means.rest.shape)
        if keep(method, means.rest - floor_map, means.task - floor_map):
            trial_images[method] = data.images
            matched[method] = matched_filter_accuracy(data.images, labels, folds)

    amplitude_scale = {}
    for model in MODELS:
        data = series[f"pd_{model}"]
        model_seconds = assembly_seconds[model] + (
            calibration_seconds if model == "calibrated" else 0.0
        )
        methods = (f"deconv_{model}", f"dps_{model}")
        if f"pd_{model}" in failed:
            for method in methods:
                failed[method] = f"its image series failed ({failed[f'pd_{model}']})"
                seconds[method] = model_seconds
            continue
        means = condition_means(data.images, labels)
        if model == "true":
            operator = torch.from_numpy(np.ascontiguousarray(scanner.true_model_operator(scene)))
            amplitude_scale[model] = 1.0
        else:
            signal = float(np.sum((means.rest.reshape(-1) - data.noise_floor)[mask.reshape(-1)]))
            if not signal > 0.0:
                for method in methods:
                    failed[method] = (
                        "no amplitude match: the mean rest image is not above the noise floor "
                        f"inside the brain (summed excess {signal:.3g})"
                    )
                    seconds[method] = model_seconds
                continue
            amplitude_scale[model] = match_common_amplitude(
                means.rest, data.noise_floor, operators[model], mask, reference_cbv,
                scene.cell_area,
            )
            operator = torch.from_numpy(operators[model]) * amplitude_scale[model]
        floor = torch.from_numpy(np.ascontiguousarray(data.noise_floor))
        frames = torch.from_numpy(np.ascontiguousarray(data.effective_frames))
        features = data.images.reshape(len(labels), -1).astype(np.float64)

        method = f"deconv_{model}"
        start = time.perf_counter()
        single_trials = gamma_deconvolution(
            operator, floor, scene.cell_area, frames, data.noise_frames, data.images,
            cfg.deconvolution,
        )

        def deconvolution_template(train: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
            fold_means = condition_means(
                train.reshape(-1, *scene.pixel_grid.shape).astype(np.float32), train_labels
            )
            return predicted_difference(
                operator,
                *deconvolve_means(operator, data, fold_means, scene.cell_area, cfg.deconvolution),
            )

        if keep(
            method, *deconvolve_means(operator, data, means, scene.cell_area, cfg.deconvolution)
        ):
            matched[method] = matched_filter_accuracy(
                features, labels, folds, deconvolution_template
            )
            reason = non_finite_reason({"single-trial images": single_trials})
            if reason is None:
                trial_images[method] = single_trials
            else:
                failed[f"{method} (PCA + LDA decoding only)"] = reason
        seconds[method] = time.perf_counter() - start + model_seconds

        method = f"dps_{model}"
        draw = _DPS_DRAWS[model]
        start = time.perf_counter()
        rest_samples, task_samples = sample_means(
            prior, operator, data, means, scene.cell_area, medium.dps, tree,
            draw, scene_index, _FULL_DATA,
        )
        fold_cfg = dataclasses.replace(medium.dps, n_samples=medium.fold_samples)

        def dps_template(train: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
            fold_means = condition_means(
                train.reshape(-1, *scene.pixel_grid.shape).astype(np.float32), train_labels
            )
            fold_rest, fold_task = sample_means(
                prior, operator, data, fold_means, scene.cell_area, fold_cfg, tree,
                draw, scene_index, _TRAINING_FOLD,
            )
            return predicted_difference(operator, fold_rest.mean(axis=0), fold_task.mean(axis=0))

        if keep(method, rest_samples.mean(axis=0), task_samples.mean(axis=0)):
            matched[method] = matched_filter_accuracy(features, labels, folds, dps_template)
        seconds[method] = time.perf_counter() - start + model_seconds

    start = time.perf_counter()
    control = prior_samples(prior, 3, medium.dps, tree, MediumDraw.PRIOR_ONLY, scene_index)
    keep(PRIOR_ONLY, control[0], control[0] + control[2] - control[1])
    seconds[PRIOR_ONLY] = time.perf_counter() - start
    return {
        "calibration": calibration,
        "series": series,
        "rest": rest,
        "task": task,
        "trial_images": trial_images,
        "decoding_matched_filter": matched,
        "amplitude_scale": amplitude_scale,
        "seconds": seconds,
        "folds": folds,
        "failed": failed,
    }


@dataclass(frozen=True)
class _Truth:
    """Ground truth of a scene, for the metrics only (shapes (nz, nx))."""

    rest: np.ndarray  # mean true CBV of the rest images
    difference: np.ndarray  # mean true CBV of the task images minus that of the rest images
    active: np.ndarray  # brain pixels whose activation pattern exceeds the threshold
    reference: np.ndarray  # brain pixels with (almost) no activation


def _truth(scene: Scene, trials: TrialSet, cfg: ReconConfig) -> _Truth:
    labels = np.asarray(trials.label)
    cbv = np.asarray(trials.cbv, dtype=np.float64)
    mask = np.asarray(scene.brain_mask, dtype=bool)
    rest = cbv[labels == 0].mean(axis=0)
    return _Truth(
        rest=rest,
        difference=cbv[labels == 1].mean(axis=0) - rest,
        active=(scene.activation > cfg.metrics.activation_threshold) & mask,
        reference=(scene.activation < cfg.metrics.reference_threshold) & mask,
    )


def family_limit(family: SkullFamily, true_bank: GreensBank, with_hu_curve: bool) -> float:
    """DIAGNOSTIC (uses the truth): the travel-time error the family cannot remove [s].

    Weighted least-squares fit of the family's travel-time changes to the true
    bank's delays: the lateral modes (and a common delay), and with
    `with_hu_curve` also a free slowness at every HU knot, which contains every
    choice of the HU-map constants. Returns the amplitude-weighted rms of what is
    left (common gauge, as `recon.metrics.travel_time_error`). It is a property
    of the family and of the fixed-path approximation, not of the calibration.
    """
    true_delay, true_amplitude = bank_delays(true_bank)
    nominal_delay = family.travel_time - family.path_length / true_bank.c_ref
    weight = (true_amplitude * (family.amplitude > 0.0)).reshape(-1)
    target = (true_delay - nominal_delay).reshape(-1)
    n_pairs = target.shape[0]
    columns = [
        family.lateral_path.reshape(n_pairs, -1).astype(np.float64) @ family.lateral_modes,
        np.ones((n_pairs, 1)),
    ]
    if with_hu_curve:
        columns.append(family.hu_path.reshape(n_pairs, -1).astype(np.float64))
    design = np.concatenate(columns, axis=1)
    root = np.sqrt(weight)
    solution, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    residual = (target - design @ solution).reshape(true_delay.shape)
    return weighted_delay_rms(residual, weight.reshape(true_delay.shape), "common")


def skull_diagnostics(scene: Scene, calibration: SkullCalibration) -> dict[str, float]:
    """Skull-model errors of a scene against the true bank, and the calibration's own numbers.

    Travel-time errors are amplitude-weighted rms values in nanoseconds
    (`recon.metrics.travel_time_error`): ``*_common`` with the delay common to all
    pairs removed, ``*_node`` with the delay common to the elements of each field
    point removed (the focusing error). ``fermat_vs_retrace_ns`` compares the
    fixed-path prediction of the calibrated bank with the re-traced one.
    """
    family, focus = calibration.family, calibration.focus
    truth = scene.bank_skull
    values = {}
    for name, bank in (("nominal", scene.bank_ct), ("calibrated", calibration.bank)):
        for gauge in ("common", "node"):
            values[f"travel_time_error_{name}_{gauge}_ns"] = 1.0e9 * travel_time_error(
                bank, truth, gauge
            )
    values["family_limit_lateral_ns"] = 1.0e9 * family_limit(family, truth, False)
    values["family_limit_lateral_hu_ns"] = 1.0e9 * family_limit(family, truth, True)
    values["fermat_vs_retrace_ns"] = 1.0e9 * travel_time_error(
        fermat_bank(family, focus.parameters), calibration.bank, "common"
    )
    values.update(
        criterion_start=focus.criterion_start,
        criterion_calibrated=focus.criterion,
        criterion_evaluations=float(focus.n_evaluations),
        c_bone=focus.parameters.c_bone,
        c_water=focus.parameters.c_water,
        hu_bone_factor=focus.parameters.hu_bone / family.nominal.hu_bone,
        lateral_rms_m_per_s=float(
            np.sqrt(np.mean(focus.parameters.lateral**2))
            * family.cfg.lateral_reference_speed**2
        ),
        seconds_family=calibration.seconds["family"],
        seconds_focus=calibration.seconds["focus"],
        seconds_retrace=calibration.seconds["retrace"],
    )
    return values


def scene_metrics(
    scene: Scene, trials: TrialSet, outputs: dict[str, Any], cfg: ReconConfig
) -> dict[str, Any]:
    """Every metric of every row for one scene, as plain data.

    Returns ``{"methods": {method: {metric: value or None}}, "skull": {...},
    "amplitude_scale": {...}, "failed": {...}}`` with the metrics of `METRICS`; a
    metric a row does not have is None, and a row that failed on this scene has
    ``{"failed": reason, "seconds": ...}`` only. A single metric that cannot be
    computed for a row (a singular decoder, a non-finite value) is None and is
    listed in ``"failed"`` under the row and the metric.
    """
    truth = _truth(scene, trials, cfg)
    mask = np.asarray(scene.brain_mask, dtype=bool)
    labels = np.asarray(trials.label)
    # Without an activated or a reference region in the image the contrast is undefined.
    has_regions = bool(truth.active.any()) and int(truth.reference.sum()) >= 2
    failed = dict(outputs["failed"])
    methods: dict[str, dict[str, float | None]] = {}
    for method in METHODS:
        if method in outputs["failed"]:
            methods[method] = {
                "failed": outputs["failed"][method],
                "seconds": outputs["seconds"][method],
            }
            continue
        rest_map = outputs["rest"][method].astype(np.float64)
        difference = outputs["task"][method].astype(np.float64) - rest_map
        values: dict[str, float | None] = {
            "baseline_correlation": map_correlation(rest_map, truth.rest, mask),
            "activation_correlation": map_correlation(difference, truth.difference, mask),
            "activation_dice": half_maximum_dice(
                difference, truth.active, mask, cfg.medium.dice_fraction
            ),
            "activation_cnr": (
                map_cnr(difference, truth.active, truth.reference) if has_regions else None
            ),
            "decoding_pca_lda": None,
            "decoding_matched_filter": outputs["decoding_matched_filter"].get(method),
            "signal_to_floor": None,
            "seconds": outputs["seconds"][method],
        }
        if method in outputs["trial_images"]:
            try:
                values["decoding_pca_lda"] = decoding_accuracy(
                    outputs["trial_images"][method],
                    labels,
                    outputs["folds"],
                    cfg.metrics.n_components,
                )
            except np.linalg.LinAlgError as error:
                # Degenerate single-trial images (for example identical in every trial).
                failed[f"{method} (decoding_pca_lda only)"] = f"LinAlgError: {error}"
        if method in outputs["series"]:
            values["signal_to_floor"] = _signal_to_floor(outputs["series"][method], mask)
        for metric, value in values.items():
            if value is not None and not np.isfinite(value):
                failed[f"{method} ({metric} only)"] = f"the metric is not finite ({value})"
                values[metric] = None
        methods[method] = values
    return {
        "methods": methods,
        "skull": skull_diagnostics(scene, outputs["calibration"]),
        "amplitude_scale": outputs["amplitude_scale"],
        "failed": failed,
    }


def example_maps(
    scene: Scene, trials: TrialSet, outputs: dict[str, Any], cfg: ReconConfig
) -> dict[str, np.ndarray]:
    """Truth and every row's rest map and activation map of a scene, for plotting.

    Returns arrays of shape (2, nz, nx) (rest map, task-minus-rest map) keyed
    ``truth`` and by method; a row that failed on the scene is absent.
    """
    truth = _truth(scene, trials, cfg)
    examples = {"truth": np.stack((truth.rest, truth.difference))}
    for method in METHODS:
        if method in outputs["failed"]:
            continue
        rest_map = outputs["rest"][method].astype(np.float64)
        examples[method] = np.stack((rest_map, outputs["task"][method] - rest_map))
    return examples


def summarise(scenes: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and spread over scenes of every metric of every row and of the skull numbers.

    Returns ``{row: {metric: {"mean", "std", "n"}}}`` (std with ddof 1; 0 for one
    scene) with the rows of `METHODS` plus ``"skull"``. A metric is averaged over
    the scenes on which the row has it, so ``n`` is smaller than the number of
    scenes where the row failed (`failures`); metrics a row never has are left out.
    """
    groups = {method: [scene["methods"][method] for scene in scenes] for method in METHODS}
    groups["skull"] = [scene["skull"] for scene in scenes]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, records in groups.items():
        summary[name] = {}
        metrics = dict.fromkeys(key for record in records for key in record if key != "failed")
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            summary[name][metric] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if array.shape[0] > 1 else 0.0,
                "n": int(array.shape[0]),
            }
    return summary


def failures(scenes: list[dict[str, Any]]) -> list[str]:
    """One line per (scene, row) that could not be computed or scored, with the reason."""
    return [
        f"scene {scene['index']}, {method}: {reason}"
        for scene in scenes
        for method, reason in scene["failed"].items()
    ]


def markdown_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    """The results table: one row per method, mean +- standard deviation over scenes."""

    def cell(entry: dict[str, float] | None) -> str:
        return "" if entry is None else f"{entry['mean']:.3f} ± {entry['std']:.3f}"

    columns = (*METRICS, "scenes scored")
    lines = ["| method | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for method in METHODS:
        cells = [cell(summary[method].get(metric)) for metric in METRICS]
        scored = summary[method].get("baseline_correlation", {"n": 0})["n"]
        lines.append(f"| {method} | " + " | ".join([*cells, str(scored)]) + " |")
    lines += ["", "| skull model | mean ± std over scenes |", "|---|---|"]
    lines += [f"| {name} | {cell(entry)} |" for name, entry in summary["skull"].items()]
    return "\n".join(lines) + "\n"


def scene_table(scenes: list[dict[str, Any]]) -> str:
    """Per-scene skull-model errors and signal-to-floor ratios, as a markdown table."""
    columns = (
        "travel_time_error_nominal_common_ns",
        "travel_time_error_calibrated_common_ns",
        "travel_time_error_nominal_node_ns",
        "travel_time_error_calibrated_node_ns",
        "family_limit_lateral_ns",
        "fermat_vs_retrace_ns",
    )
    header = ["scene", *columns, *(f"signal_to_floor {m}" for m in BEAMFORMED_METHODS)]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---|" * (len(header) - 1)]
    for scene in scenes:
        cells = [f"{scene['skull'][column]:.1f}" for column in columns]
        ratios = [scene["methods"][method].get("signal_to_floor") for method in BEAMFORMED_METHODS]
        cells += ["failed" if ratio is None else f"{ratio:.3g}" for ratio in ratios]
        lines.append(f"| {scene['index']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def estimate_seconds(
    scene: Scene, n_trials: int, n_scenes: int, cfg: ReconConfig
) -> dict[str, float]:
    """Rough single-thread run time of the experiment, from one scene's sizes.

    Returns seconds per stage (``family``, ``focus``, ``retrace``, ``operators``,
    ``deconvolution``, ``dps``, ``prior_only``) summed over `n_scenes`, and their ``total``;
    ``operators`` holds the re-beamforming and the model operator of each delay-only model.
    """
    setup = imaging_setup(scene.imaging)
    nf, n_elements = setup.freqs.shape[0], setup.transducer.n_elements
    n_pix = scene.pixel_grid.shape[0] * scene.pixel_grid.shape[1]
    n_nodes = setup.roi.shape[0] * setup.roi.shape[1]
    medium_nodes = scene.medium_grid.shape[0] * scene.medium_grid.shape[1]
    medium, focus = cfg.medium, cfg.medium.focus
    trace = _SECONDS_PER_TRACE_VALUE * n_elements * medium_nodes
    histograms = (
        _SECONDS_PER_HISTOGRAM_SAMPLE * n_elements * n_nodes * scene.medium_grid.shape[0]
    )
    evaluations = (
        len(focus.mode_stages) * focus.iterations_per_stage * _LBFGS_EVALUATIONS_PER_ITERATION
    )
    criterion = (
        _SECONDS_PER_CRITERION_VALUE
        * nf
        * min(focus.n_pixels, n_pix)
        * n_elements
        * n_pix
        * focus.source_refinement**2
    )
    operator = _SECONDS_PER_OPERATOR_VALUE * nf * n_pix * n_elements * n_pix
    n_folds = cfg.metrics.n_folds
    image_steps = 2 * medium.dps.n_steps * (medium.dps.n_samples + n_folds * medium.fold_samples)
    stages = {
        "family": trace + histograms,
        "focus": evaluations * criterion,
        "retrace": trace,
        "operators": 2 * len(REBEAMFORMED_MODELS) * cfg.source_refinement**2 * operator,
        "deconvolution": len(MODELS) * n_trials * _SECONDS_PER_DECONVOLVED_IMAGE,
        "dps": len(MODELS) * image_steps * _SECONDS_PER_GUIDED_IMAGE_STEP,
        "prior_only": 3 * medium.dps.n_steps * _SECONDS_PER_UNGUIDED_IMAGE_STEP,
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
    cfg: reconstruction settings (`cfg.medium` and the shared sections).
    root_seed: root seed of the experiment's `SeedTree`.

    Yields, per scene, the record of `scene_metrics` (with ``"index"`` and
    ``"n_trials"``) and the `example_maps` of the scene. Call
    `recon.seeding.configure_torch(cfg.num_threads)` first.
    """
    tree = SeedTree(root_seed)
    scene_index = 0
    # No `enumerate`: it keeps the last yielded pair, and with it the scene's banks, alive.
    for scene, trials in dataset:
        # The no-skull bank is not used; dropping it frees a third of the scene's memory.
        scene = dataclasses.replace(scene, bank_clean=reference_bank(scene))
        outputs = reconstruct_scene(scene, trials, prior, cfg, tree, scene_index)
        record = scene_metrics(scene, trials, outputs, cfg)
        record.update(index=scene_index, n_trials=int(np.asarray(trials.label).shape[0]))
        examples = example_maps(scene, trials, outputs, cfg)
        # Release the scene (its banks hold about 0.5 GB) before the next one is loaded.
        del scene, trials, outputs
        gc.collect()
        yield record, examples
        scene_index += 1
