"""The experiment on a tiny synthetic scene with an untrained prior."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.recon.config import CalibrationConfig, DpsConfig, MetricsConfig, ReconConfig
from neurosim.recon.experiment import (
    DPS_METHODS,
    IMAGE_METRICS,
    METHODS,
    check_scene,
    estimate_seconds,
    markdown_table,
    run_experiment,
    summarise,
    without_banks,
)

from .conftest import CONFIG_DIR
from .scene_fixtures import tiny_prior, tiny_run, tiny_scene  # noqa: F401

SMALL = ReconConfig(
    source_refinement=1,
    dps=DpsConfig(n_steps=3, n_samples=2),
    calibration=CalibrationConfig(
        sampler=DpsConfig(n_steps=4, n_samples=2),
        n_candidates=1,
        n_updates=2,
        lbfgs_iterations=2,
        n_pixels=32,
    ),
    metrics=MetricsConfig(n_folds=3, n_components=4),
)


@pytest.mark.parametrize("name", ["recon_default.yaml", "recon_small.yaml"])
def test_shipped_configs_load(name):
    cfg = from_dict(ReconConfig, load_yaml(CONFIG_DIR / name))
    assert cfg.dps.n_samples >= 2


def test_without_banks_keeps_what_reconstruction_needs(tiny_scene):
    slim = without_banks(tiny_scene)
    assert slim.bank_skull.B.size < tiny_scene.bank_skull.B.size
    assert slim.bank_skull.transducer == tiny_scene.bank_skull.transducer
    np.testing.assert_array_equal(slim.bank_skull.freqs, tiny_scene.bank_skull.freqs)
    assert slim.operators is tiny_scene.operators


def test_check_scene_rejects_wrong_refinement_and_grid(tiny_scene, tiny_prior):
    check_scene(tiny_scene, tiny_prior, SMALL)
    with pytest.raises(ValueError, match="source_refinement"):
        check_scene(tiny_scene, tiny_prior, dataclasses.replace(SMALL, source_refinement=2))
    shifted = dataclasses.replace(
        tiny_prior, pixel_grid=dataclasses.replace(tiny_prior.pixel_grid, origin=(0.02, -0.0075))
    )
    with pytest.raises(ValueError, match="pixel grid"):
        check_scene(tiny_scene, shifted, SMALL)
    pipeline_meta = {"cost": {"n_operator_passes": 16}}
    with pytest.raises(ValueError, match="source_refinement = 2"):
        check_scene(dataclasses.replace(tiny_scene, meta=pipeline_meta), tiny_prior, SMALL)


def test_estimate_is_positive_and_adds_up(tiny_scene):
    estimate = estimate_seconds(tiny_scene, 12, 3, SMALL)
    assert estimate["total"] > 0.0
    np.testing.assert_allclose(
        estimate["total"], sum(value for name, value in estimate.items() if name != "total")
    )


@pytest.fixture(scope="module")
def records(tiny_run, tiny_prior):
    return list(run_experiment([tiny_run], tiny_prior, SMALL, 7))


def test_every_method_gets_every_metric(records, tiny_run):
    ((record, examples),) = records
    assert record["index"] == 0 and record["n_trials"] == 12
    assert set(record["methods"]) == set(METHODS)
    for method, values in record["methods"].items():
        assert set(IMAGE_METRICS) <= set(values) and values["seconds"] >= 0.0
        assert all(np.isfinite(values[metric]) for metric in IMAGE_METRICS), method
        assert ("coverage_0.5" in values) == (method in DPS_METHODS)
    assert set(examples) == {"truth", *METHODS}
    assert all(image.shape == (2, 16, 16) for image in examples.values())
    assert record["screens"]["delay_error_ct_ns"] > 0.0
    # The synthetic skull is a phase screen: the model reproduces the true operator.
    assert record["diagnostics"]["operator_relative_error_screen_true"] < 1e-4
    assert record["diagnostics"]["operator_relative_error_screen_ct"] > 1e-2


def test_same_seed_same_metrics(records, tiny_run, tiny_prior):
    again = list(run_experiment([tiny_run], tiny_prior, SMALL, 7))
    for method in METHODS:
        for metric in IMAGE_METRICS:
            assert (
                again[0][0]["methods"][method][metric] == records[0][0]["methods"][method][metric]
            )
    np.testing.assert_array_equal(again[0][1]["dps_calibrated"], records[0][1]["dps_calibrated"])


def test_summary_and_table(records):
    scenes = [records[0][0], records[0][0]]
    summary = summarise(scenes)
    entry = summary["pd_clean_das"]["cbv_correlation"]
    assert entry["n"] == 2 and entry["std"] == 0.0
    assert entry["mean"] == pytest.approx(
        records[0][0]["methods"]["pd_clean_das"]["cbv_correlation"]
    )
    table = markdown_table(summary)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert [row.split("|")[1].strip() for row in rows[1 : 1 + len(METHODS)]] == list(METHODS)
    assert "coverage_0.9" in rows[0] and "delay_error_calibrated_ns" in table
