"""data.sweep: scene configs, the index, resumable generation and the dataset reader."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from neurosim.common import io as common_io
from neurosim.common.config import ConfigError, config_hash, from_dict, load_yaml
from neurosim.common.io import git_commit, load_attrs
from neurosim.common.seeding import SeedTree
from neurosim.data.sweep import (
    INDEX_NAME,
    SweepConfig,
    load_dataset,
    run_sweep,
    scene_config,
    scene_file,
    sweep_index,
)
from neurosim.head.types import SkullEffects

from .conftest import CONFIG_DIR

ROOT_SEED = 5


@pytest.fixture(scope="module")
def sweep_cfg(quick_cfg) -> SweepConfig:
    """Two quick scenes with every kind of variation switched on."""
    small = from_dict(SweepConfig, load_yaml(CONFIG_DIR / "sweep_small.yaml"))
    return replace(small, base=quick_cfg, n_scenes=2)


@pytest.fixture(scope="module")
def dataset(sweep_cfg, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("sweep")
    return out_dir, run_sweep(sweep_cfg, ROOT_SEED, out_dir)


def test_scene_config_draws_every_varied_parameter_within_its_range(sweep_cfg):
    tree = SeedTree(ROOT_SEED)
    base = sweep_cfg.base
    for index in range(6):
        run, varied = scene_config(sweep_cfg, tree, index)
        assert set(varied) == {
            "center_frequency",
            "probe",
            "target_direction",
            "effects",
            "electronic_snr_db",
        }
        assert run.imaging.transducer.center_frequency in sweep_cfg.center_frequencies
        assert abs(run.probe.lateral_shift) <= sweep_cfg.probe_shift_jitter
        assert abs(run.probe.tilt_deg) <= sweep_cfg.probe_tilt_jitter_deg
        direction = np.asarray(run.head.cbv.target_direction)
        reference = np.asarray(base.head.cbv.target_direction)
        cosine = direction @ reference / np.linalg.norm(reference)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        assert np.degrees(np.arccos(cosine)) <= sweep_cfg.target_direction_jitter_deg + 1e-9
        assert run.effects in sweep_cfg.effects
        assert run.imaging.noise.electronic_snr_db in sweep_cfg.electronic_snr_db
        # Nothing else moved.
        assert run.trials == base.trials and run.ct == base.ct and run.backend == base.backend


def test_scene_config_is_a_function_of_seed_and_index(sweep_cfg):
    first = scene_config(sweep_cfg, SeedTree(ROOT_SEED), 1)
    assert first == scene_config(sweep_cfg, SeedTree(ROOT_SEED), 1)
    assert first != scene_config(sweep_cfg, SeedTree(ROOT_SEED), 0)
    assert first != scene_config(sweep_cfg, SeedTree(ROOT_SEED + 1), 1)


def test_switching_one_variation_off_keeps_the_other_draws(sweep_cfg):
    tree = SeedTree(ROOT_SEED)
    _, everything = scene_config(sweep_cfg, tree, 0)
    _, without_frequency = scene_config(replace(sweep_cfg, center_frequencies=()), tree, 0)
    assert "center_frequency" not in without_frequency
    assert without_frequency == {
        key: value for key, value in everything.items() if key != "center_frequency"
    }


def test_no_variation_gives_the_base_config(quick_cfg):
    cfg = SweepConfig(base=quick_cfg, n_scenes=2)
    run, varied = scene_config(cfg, SeedTree(ROOT_SEED), 1)
    assert run == quick_cfg and varied == {}


def test_a_frequency_the_imaging_setup_cannot_hold_is_rejected_at_load(quick_cfg):
    # At 6 MHz the band is three times wider, so 56 frequencies no longer span the ROI.
    with pytest.raises(ValueError, match="range aliasing"):
        SweepConfig(base=quick_cfg, n_scenes=1, center_frequencies=(6.0e6,))
    data = load_yaml(CONFIG_DIR / "sweep_small.yaml")
    data["center_frequencies"] = [6.0e6]
    with pytest.raises(ConfigError, match="range aliasing"):
        from_dict(SweepConfig, data)


def test_index_lists_every_scene_and_is_written_to_the_directory(sweep_cfg, dataset):
    out_dir, index = dataset
    # The index of the sweep, with the commit that produced each bundle filled in.
    planned = sweep_index(sweep_cfg, ROOT_SEED)
    assert [entry["git_commit"] for entry in planned["scenes"]] == [None, None]
    for entry in planned["scenes"]:
        entry["git_commit"] = git_commit()
    assert index == planned and index["git_commit"] == git_commit()
    written = (out_dir / INDEX_NAME).read_bytes()
    assert json.loads(written.decode("utf-8")) == index
    # The same bytes on every operating system.
    assert written.endswith(b"}\n") and b"\r" not in written
    assert index["root_seed"] == ROOT_SEED and index["config_hash"] == config_hash(sweep_cfg)
    assert index["n_scenes"] == 2 and [entry["index"] for entry in index["scenes"]] == [0, 1]
    tree = SeedTree(ROOT_SEED)
    for entry in index["scenes"]:
        run, varied = scene_config(sweep_cfg, tree, entry["index"])
        assert entry["file"] == scene_file(entry["index"]) and (out_dir / entry["file"]).exists()
        assert entry["seed_path"] == [entry["index"]] and entry["root_seed"] == ROOT_SEED
        assert entry["config_hash"] == config_hash(run) and entry["varied"] == varied
        assert load_attrs(out_dir / entry["file"])["git_commit"] == entry["git_commit"]


def test_load_dataset_yields_the_scenes_in_order(sweep_cfg, dataset):
    out_dir, index = dataset
    runs = list(load_dataset(out_dir))
    assert len(runs) == 2
    for entry, (scene, trials) in zip(index["scenes"], runs):
        assert scene.meta["sweep"] == {
            "index": entry["index"],
            "seed_path": entry["seed_path"],
            "varied": entry["varied"],
        }
        assert scene.meta["config_hash"] == entry["config_hash"]
        assert scene.imaging.transducer.center_frequency == entry["varied"]["center_frequency"]
        assert SkullEffects(**scene.meta["effects"]) == SkullEffects(**entry["varied"]["effects"])
        assert trials.time.shape == (sweep_cfg.base.trials.n_images,)
    # Different skulls: the two scenes do not share their CBV map.
    assert not np.array_equal(runs[0][0].cbv_baseline, runs[1][0].cbv_baseline)


def test_a_scene_alone_and_a_resumed_sweep_reproduce_the_files(sweep_cfg, dataset, tmp_path):
    out_dir, _ = dataset
    lone = run_sweep(sweep_cfg, ROOT_SEED, tmp_path, scenes=(1,))
    assert not (tmp_path / scene_file(0)).exists()
    assert (tmp_path / scene_file(1)).read_bytes() == (out_dir / scene_file(1)).read_bytes()
    # A scene without a bundle has no producing commit yet.
    assert [entry["git_commit"] for entry in lone["scenes"]] == [None, git_commit()]
    assert json.loads((tmp_path / INDEX_NAME).read_text(encoding="utf-8")) == lone
    with pytest.raises(FileNotFoundError, match="scene 0"):
        list(load_dataset(tmp_path))

    # A bundle cut short (an interrupted write) is generated again; a good one is kept.
    kept = (tmp_path / scene_file(1)).stat().st_mtime_ns
    complete = (out_dir / scene_file(0)).read_bytes()
    (tmp_path / scene_file(0)).write_bytes(complete[: len(complete) // 2])
    run_sweep(sweep_cfg, ROOT_SEED, tmp_path)
    assert (tmp_path / scene_file(1)).stat().st_mtime_ns == kept
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        path.name for path in out_dir.iterdir()
    )
    for path in out_dir.iterdir():
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()


def test_bundles_of_another_commit_are_regenerated_unless_kept(
    sweep_cfg, dataset, tmp_path, monkeypatch
):
    out_dir, original = dataset
    for path in out_dir.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    produced_by = original["git_commit"]
    stamps = [(tmp_path / scene_file(index)).stat().st_mtime_ns for index in range(2)]
    # The code moves on to another commit.
    current = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(common_io, "git_commit", lambda: current)

    # Explicitly kept: nothing is generated, and the index names the commit of every scene.
    kept = run_sweep(sweep_cfg, ROOT_SEED, tmp_path, keep_existing=True)
    assert [(tmp_path / scene_file(index)).stat().st_mtime_ns for index in range(2)] == stamps
    assert kept["git_commit"] == current
    assert [entry["git_commit"] for entry in kept["scenes"]] == [produced_by, produced_by]
    assert json.loads((tmp_path / INDEX_NAME).read_text(encoding="utf-8")) == kept

    # By default the requested scene is generated again with the current code; the scene
    # that was not requested stays, and stays listed with the commit that produced it.
    resumed = run_sweep(sweep_cfg, ROOT_SEED, tmp_path, scenes=(0,))
    assert (tmp_path / scene_file(0)).stat().st_mtime_ns != stamps[0]
    assert (tmp_path / scene_file(1)).stat().st_mtime_ns == stamps[1]
    assert [entry["git_commit"] for entry in resumed["scenes"]] == [current, produced_by]
    assert load_attrs(tmp_path / scene_file(0))["git_commit"] == current
    assert json.loads((tmp_path / INDEX_NAME).read_text(encoding="utf-8")) == resumed
    # Only the recorded commit differs: the data of the scene are reproduced.
    for first, second in zip(*(load_dataset(directory) for directory in (out_dir, tmp_path))):
        np.testing.assert_array_equal(first[1].cbv, second[1].cbv)
        assert first[0].meta == second[0].meta


def test_a_directory_of_another_sweep_is_refused(sweep_cfg, dataset):
    out_dir, _ = dataset
    with pytest.raises(ValueError, match="another sweep"):
        run_sweep(sweep_cfg, ROOT_SEED + 1, out_dir, scenes=(0,))
