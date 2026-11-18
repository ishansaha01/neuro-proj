"""The second design's experiment on a tiny real scene with an untrained prior."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.recon import experiment as screen_design
from neurosim.recon import medium_experiment as design
from neurosim.recon.config import (
    DpsConfig,
    FocusConfig,
    MediumReconConfig,
    MetricsConfig,
    ReconConfig,
    SkullModelConfig,
)

from .conftest import CONFIG_DIR
from .scene_fixtures import tiny_ct_prior, tiny_ct_run  # noqa: F401

SMALL = ReconConfig(
    source_refinement=1,
    metrics=MetricsConfig(n_folds=3, n_components=4),
    medium=MediumReconConfig(
        skull_model=SkullModelConfig(n_lateral_modes=3),
        focus=FocusConfig(
            n_pixels=16, pixels_per_pass=8, mode_stages=(0, 3), iterations_per_stage=2
        ),
        dps=DpsConfig(n_steps=3, n_samples=2),
    ),
)


def test_default_config_has_the_medium_section():
    cfg = from_dict(ReconConfig, load_yaml(CONFIG_DIR / "recon_default.yaml"))
    assert cfg.medium.focus.mode_stages[-1] <= cfg.medium.skull_model.n_lateral_modes
    assert cfg.medium.dps.n_samples >= 1


@pytest.fixture(scope="module")
def outputs(tiny_ct_run, tiny_ct_prior):
    scene, trials = tiny_ct_run
    return design.reconstruct_scene(scene, trials, tiny_ct_prior, SMALL, SeedTree(7), 0)


@pytest.fixture(scope="module")
def record(tiny_ct_run, outputs):
    scene, trials = tiny_ct_run
    record = design.scene_metrics(scene, trials, outputs, SMALL)
    record.update(index=0, n_trials=12)
    return record


def test_every_row_is_reconstructed(tiny_ct_run, outputs):
    shape = tiny_ct_run[0].pixel_grid.shape
    assert outputs["failed"] == {}
    assert set(outputs["rest"]) == set(outputs["task"]) == set(design.METHODS)
    assert all(image.shape == shape for image in outputs["rest"].values())
    assert set(outputs["series"]) == set(design.BEAMFORMED_METHODS)
    assert set(outputs["amplitude_scale"]) == set(design.MODELS)
    assert outputs["amplitude_scale"]["true"] == 1.0
    assert set(outputs["seconds"]) == set(design.METHODS)
    # The calibration is charged to the calibrated rows only.
    assert outputs["seconds"]["pd_calibrated"] > sum(outputs["calibration"].seconds.values())
    assert outputs["seconds"]["pd_true"] == 0.0


def test_metrics_record(record):
    for method in design.METHODS:
        values = record["methods"][method]
        assert set(values) == set(design.METRICS)
        assert -1.0 <= values["baseline_correlation"] <= 1.0
    assert record["methods"]["pd_ct"]["signal_to_floor"] > 0.0
    assert record["methods"]["deconv_ct"]["signal_to_floor"] is None
    assert record["methods"]["dps_ct"]["decoding_pca_lda"] is None
    assert 0.0 <= record["methods"]["dps_ct"]["decoding_matched_filter"] <= 1.0
    assert record["methods"]["prior_only"]["decoding_matched_filter"] is None
    skull = record["skull"]
    assert skull["travel_time_error_nominal_common_ns"] > 0.0
    assert skull["family_limit_lateral_hu_ns"] <= skull["family_limit_lateral_ns"] + 1e-9
    assert skull["family_limit_lateral_ns"] <= skull["travel_time_error_nominal_common_ns"] + 1e-6
    assert skull["criterion_calibrated"] >= skull["criterion_start"] - 1e-6


def test_same_seed_same_results(tiny_ct_run, tiny_ct_prior, outputs):
    scene, trials = tiny_ct_run
    again = design.reconstruct_scene(scene, trials, tiny_ct_prior, SMALL, SeedTree(7), 0)
    for method in design.METHODS:
        np.testing.assert_array_equal(again["task"][method], outputs["task"][method])
    other = design.reconstruct_scene(scene, trials, tiny_ct_prior, SMALL, SeedTree(8), 0)
    assert not np.array_equal(other["task"]["dps_true"], outputs["task"]["dps_true"])


def test_summary_tables_and_examples(tiny_ct_run, outputs, record):
    scene, trials = tiny_ct_run
    summary = design.summarise([record, record])
    assert summary["pd_ct"]["baseline_correlation"]["n"] == 2
    assert "decoding_pca_lda" not in summary["dps_ct"]
    table = design.markdown_table(summary)
    assert all(f"| {method} |" in table for method in design.METHODS)
    assert "travel_time_error_calibrated_common_ns" in table
    assert "| 0 |" in design.scene_table([record])
    examples = design.example_maps(scene, trials, outputs, SMALL)
    assert set(examples) == {"truth", *design.METHODS}
    assert all(maps.shape == (2, *scene.pixel_grid.shape) for maps in examples.values())
    assert design.failures([record]) == []


def test_a_non_finite_row_is_recorded_as_failed_not_scored(tiny_ct_run, outputs):
    scene, trials = tiny_ct_run
    broken = {**outputs, "failed": {"dps_ct": "rest map: 3 of 64 values not finite"}}
    record = design.scene_metrics(scene, trials, broken, SMALL)
    record.update(index=5, n_trials=12)
    assert set(record["methods"]["dps_ct"]) == {"failed", "seconds"}
    assert design.failures([record]) == ["scene 5, dps_ct: rest map: 3 of 64 values not finite"]
    good = design.scene_metrics(scene, trials, outputs, SMALL)
    good.update(index=6, n_trials=12)
    summary = design.summarise([record, good])
    assert summary["dps_ct"]["baseline_correlation"]["n"] == 1
    assert summary["dps_true"]["baseline_correlation"]["n"] == 2
    assert "| dps_ct |" in design.markdown_table(summary)
    assert "dps_ct" not in design.example_maps(scene, trials, broken, SMALL)


def test_a_series_below_the_floor_fails_its_inversions_only(
    tiny_ct_run, tiny_ct_prior, monkeypatch
):
    scene, trials = tiny_ct_run
    real = design.scanner.rebeamform_trials

    def below_the_floor(*arguments, **keywords):
        series = real(*arguments, **keywords)
        return dataclasses.replace(series, images=1.0e-6 * series.images)

    monkeypatch.setattr(design.scanner, "rebeamform_trials", below_the_floor)
    outputs = design.reconstruct_scene(scene, trials, tiny_ct_prior, SMALL, SeedTree(7), 0)
    failed = outputs["failed"]
    inversions = {
        f"{kind}_{model}" for kind in ("deconv", "dps") for model in design.REBEAMFORMED_MODELS
    }
    assert set(failed) == inversions
    assert all("noise floor" in reason for reason in failed.values())
    record = design.scene_metrics(scene, trials, outputs, SMALL)
    assert "baseline_correlation" in record["methods"]["deconv_true"]
    assert set(record["methods"]["deconv_ct"]) == {"failed", "seconds"}


def test_a_metric_that_cannot_be_computed_is_recorded(tiny_ct_run, outputs):
    scene, trials = tiny_ct_run
    constant = np.ones_like(outputs["trial_images"]["pd_ct"])
    broken = {**outputs, "trial_images": {**outputs["trial_images"], "pd_ct": constant}}
    record = design.scene_metrics(scene, trials, broken, SMALL)
    assert record["methods"]["pd_ct"]["decoding_pca_lda"] is None
    assert record["methods"]["pd_ct"]["baseline_correlation"] is not None
    assert any(key.startswith("pd_ct (decoding_pca_lda") for key in record["failed"])


def test_estimate_is_positive_and_adds_up(tiny_ct_run):
    estimate = design.estimate_seconds(tiny_ct_run[0], 12, 3, SMALL)
    assert estimate["total"] > 0.0
    np.testing.assert_allclose(
        estimate["total"], sum(value for name, value in estimate.items() if name != "total")
    )


def test_first_design_summary_skips_failed_methods():
    scored = {"decoding_accuracy": 0.6, "seconds": 2.0}
    scenes = [
        {
            "index": index,
            "methods": {method: dict(scored) for method in screen_design.METHODS},
            "screens": {"delay_error_ct_ns": 50.0},
        }
        for index in range(2)
    ]
    scenes[1]["methods"]["dps_calibrated"] = {
        "failed": "images: 8 of 8 values not finite",
        "seconds": 9.0,
    }
    summary = screen_design.summarise(scenes)
    assert summary["dps_calibrated"]["decoding_accuracy"]["n"] == 1
    assert summary["dps_calibrated"]["seconds"]["n"] == 2
    assert summary["dps_ct"]["decoding_accuracy"]["n"] == 2
    assert screen_design.failures(scenes) == [
        "scene 1, dps_calibrated: images: 8 of 8 values not finite"
    ]
