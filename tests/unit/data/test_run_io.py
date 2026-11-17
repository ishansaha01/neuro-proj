"""data.io: one run in one bundle, and data.pipeline.run."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.config import config_hash, dump_yaml
from neurosim.common.io import load_bundle
from neurosim.data.io import load_run, media_paths, save_run
from neurosim.data.pipeline import run
from neurosim.data.types import BEAMFORMERS, Scene, TrialSet
from neurosim.fus.types import GreensBank

from .conftest import SMALL_SEED


def _assert_equal(loaded: object, original: object, name: str) -> None:
    if isinstance(original, np.ndarray):
        assert loaded.dtype == original.dtype, name
        np.testing.assert_array_equal(loaded, original, err_msg=name)
    elif isinstance(original, dict) and all(isinstance(v, np.ndarray) for v in original.values()):
        assert tuple(loaded) == tuple(original) == BEAMFORMERS, name
        for key in original:
            _assert_equal(loaded[key], original[key], f"{name}[{key}]")
    elif dataclasses.is_dataclass(original) and not isinstance(original, type):
        for spec in dataclasses.fields(original):
            _assert_equal(getattr(loaded, spec.name), getattr(original, spec.name), spec.name)
    else:
        assert loaded == original, name


def test_round_trip_restores_every_field(tmp_path, small_cfg, small_run):
    scene, _, trials = small_run
    path = tmp_path / "run.h5"
    save_run(path, scene, trials, small_cfg, SMALL_SEED)
    loaded_scene, loaded_trials = load_run(path)
    assert isinstance(loaded_scene, Scene) and isinstance(loaded_trials, TrialSet)
    assert isinstance(loaded_scene.bank_ct, GreensBank)
    _assert_equal(loaded_scene, scene, "scene")
    _assert_equal(loaded_trials, trials, "trials")


def test_bundle_carries_provenance_and_is_byte_deterministic(tmp_path, small_cfg, small_run):
    scene, _, trials = small_run
    first, second, resaved = (tmp_path / name for name in ("a.h5", "b.h5", "c.h5"))
    save_run(first, scene, trials, small_cfg, SMALL_SEED)
    save_run(second, scene, trials, small_cfg, SMALL_SEED)
    assert first.read_bytes() == second.read_bytes()
    save_run(resaved, *load_run(first), small_cfg, SMALL_SEED)
    assert resaved.read_bytes() == first.read_bytes()

    arrays, attrs = load_bundle(first)
    assert attrs["root_seed"] == SMALL_SEED and attrs["config_hash"] == config_hash(small_cfg)
    assert attrs["config_yaml"] == dump_yaml(small_cfg)
    assert {"git_commit", "neurosim_version"} <= set(attrs)
    assert attrs["fidelity"] == "statistical" and attrs["meta"]["root_seed"] == SMALL_SEED
    assert "scene/bank_ct/B" in arrays and "trials/power_doppler/clean_das" in arrays
    # The estimated bundle size is the size of the file, to within its small overhead.
    assert scene.meta["cost"]["bundle_bytes"] == pytest.approx(first.stat().st_size, rel=0.05)


def test_media_paths_sit_next_to_the_run(tmp_path):
    true_path, ct_path = media_paths(tmp_path / "run.h5")
    assert true_path == tmp_path / "run_medium_true.h5"
    assert ct_path == tmp_path / "run_medium_ct.h5"


def test_run_is_the_scene_and_trials_of_the_seed_tree(quick_cfg, quick_scene):
    scene, trials = run(quick_cfg, SMALL_SEED)
    reference, _ = quick_scene
    assert scene.meta == reference.meta and scene.plane == reference.plane
    for key in BEAMFORMERS:
        np.testing.assert_array_equal(scene.operators[key], reference.operators[key])
    assert trials.time.shape == (quick_cfg.trials.n_images,)
