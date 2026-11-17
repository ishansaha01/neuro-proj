"""scripts/run_pipeline.py, plot_run.py and make_dataset.py: smoke tests on the quick run."""

from __future__ import annotations

import yaml

from neurosim.common.config import dump_yaml
from neurosim.data.io import load_run, media_paths
from neurosim.data.sweep import INDEX_NAME, scene_file
from neurosim.head.io import load_plane_medium
from scripts import make_dataset, plot_run, run_pipeline

from .conftest import SMALL_SEED


def test_run_pipeline_estimates_then_saves_run_and_media_and_plot_run_draws_them(
    tmp_path, quick_cfg, quick_scene, capsys
):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(dump_yaml(quick_cfg), encoding="utf-8")
    out = tmp_path / "out" / "run.h5"
    common = ["--config", str(config_path), "--seed", str(SMALL_SEED)]

    assert run_pipeline.main([*common, "--estimate-only"]) == 0
    printed = capsys.readouterr().out
    assert "ray backend, statistical fidelity" in printed and "estimated cost: about" in printed
    assert not out.parent.exists()

    assert run_pipeline.main([*common, "--out", str(out), "--save-media"]) == 0
    printed = capsys.readouterr().out
    assert printed.index("estimated cost") < printed.index("saved")
    scene, trials = load_run(out)
    assert scene.meta == quick_scene[0].meta and trials.time.size == quick_cfg.trials.n_images
    for path in media_paths(out):
        assert load_plane_medium(path).grid.shape == tuple(scene.meta["medium_grid"]["shape"])

    figure = tmp_path / "run.png"
    assert plot_run.main(["--run", str(out), "--out", str(figure)]) == 0
    assert figure.read_bytes().startswith(b"\x89PNG") and figure.stat().st_size > 20_000


def test_make_dataset_estimates_then_generates_the_requested_scenes(tmp_path, quick_cfg, capsys):
    base = yaml.safe_load(dump_yaml(quick_cfg))
    sweep = {"base": base, "n_scenes": 2, "probe_shift_jitter": 1.0e-3}
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(sweep), encoding="utf-8")
    out_dir = tmp_path / "dataset"
    common = ["--config", str(config_path), "--seed", "9"]

    assert make_dataset.main([*common, "--estimate-only"]) == 0
    assert "sweep of 2 scenes, seed 9: about" in capsys.readouterr().out
    assert not out_dir.exists()

    assert make_dataset.main([*common, "--out", str(out_dir), "--scenes", "1"]) == 0
    assert "1 of 2 scenes" in capsys.readouterr().out
    assert sorted(path.name for path in out_dir.iterdir()) == [INDEX_NAME, scene_file(1)]
    scene, _ = load_run(out_dir / scene_file(1))
    assert scene.meta["sweep"]["index"] == 1 and "probe" in scene.meta["sweep"]["varied"]
