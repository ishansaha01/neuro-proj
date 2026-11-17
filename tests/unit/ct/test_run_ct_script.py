"""scripts/run_ct.py: head file in, CT file with provenance out."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np

from neurosim.common.config import dump_yaml, from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.ct.io import load_ct
from neurosim.head.config import HeadConfig
from neurosim.head.generate import generate_head
from neurosim.head.io import load_head, save_head

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load_script(name: str = "run_ct"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_scans_a_saved_head(tmp_path, config_dir, small_cfg, capsys):
    head_cfg = from_dict(HeadConfig, load_yaml(config_dir / "head_small.yaml"))
    head = generate_head(head_cfg, SeedTree(1))
    save_head(tmp_path / "head.h5", head, head_cfg, 1)
    # One slice in the middle of the head's own z range: with an even number of slices the
    # centre of the z extent lies half a voxel below slice nz // 2.
    z = head.grid.axes()[2]
    middle = float(z[z.size // 2])
    cfg = dataclasses.replace(small_cfg, slab_z_from_center=(0.0, head.grid.spacing))
    (tmp_path / "ct.yaml").write_text(dump_yaml(cfg), encoding="utf-8")

    exit_code = _load_script().main(
        [
            "--head", str(tmp_path / "head.h5"),
            "--config", str(tmp_path / "ct.yaml"),
            "--seed", "1",
            "--out", str(tmp_path / "ct.h5"),
        ]
    )  # fmt: skip
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "CT scan: 1 slices" in printed and "saved" in printed

    ct = load_ct(tmp_path / "ct.h5")
    assert ct.hu.shape == (*head.grid.shape[:2], 1)
    assert ct.grid.origin[2] == middle
    assert ct.meta["provenance"]["root_seed"] == 1
    assert np.isfinite(ct.hu).all()
    # Bone is bright, soft tissue is near water.
    assert ct.hu.max() > 500.0


def test_default_slab_setting_scans_a_box_region_head(tmp_path, config_dir, capsys):
    """generate_head -> run_ct with small configs built like the two default configs.

    configs/head_default.yaml is a box region whose z range does not contain z = 0 and
    configs/ct_default.yaml scans `slab_z_from_center: [-0.004, 0.004]`; the small configs
    have the same structure, so the scripts must run one after the other unchanged.
    """
    head_data = load_yaml(config_dir / "head_small.yaml")
    ct_data = load_yaml(config_dir / "ct_small.yaml")
    default_head = load_yaml(config_dir / "head_default.yaml")["grid"]
    assert head_data["grid"]["region"] == default_head["region"] == "box"
    assert set(head_data["grid"]) == set(default_head)
    default_slab = load_yaml(config_dir / "ct_default.yaml")["slab_z_from_center"]
    assert ct_data["slab_z_from_center"] == default_slab == [-0.004, 0.004]

    head_path, ct_path = tmp_path / "head.h5", tmp_path / "ct.h5"
    head_arguments = ["--config", str(config_dir / "head_small.yaml"), "--seed", "1"]
    assert _load_script("generate_head").main([*head_arguments, "--out", str(head_path)]) == 0
    ct_arguments = ["--head", str(head_path), "--config", str(config_dir / "ct_small.yaml")]
    assert _load_script().main([*ct_arguments, "--seed", "1", "--out", str(ct_path)]) == 0
    assert "CT scan: 4 slices" in capsys.readouterr().out

    head = load_head(head_path)
    z = head.grid.axes()[2]
    # The box lies away from the head-frame origin: no slice is within 4 mm of z = 0.
    assert np.abs(z).min() > 0.004
    ct = load_ct(ct_path)
    # The four 2 mm slices within 4 mm of the middle of the box.
    assert ct.hu.shape == (*head.grid.shape[:2], 4)
    assert ct.grid.origin[2] == z[z.size // 2 - 2]
    assert np.isfinite(ct.hu).all() and ct.hu.max() > 500.0
