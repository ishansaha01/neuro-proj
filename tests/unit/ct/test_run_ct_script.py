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
from neurosim.head.io import save_head

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_ct.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_ct", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_scans_a_saved_head(tmp_path, config_dir, small_cfg, capsys):
    head_cfg = from_dict(HeadConfig, load_yaml(config_dir / "head_small.yaml"))
    head = generate_head(head_cfg, SeedTree(1))
    save_head(tmp_path / "head.h5", head, head_cfg, 1)
    # One slice in the middle of the head's own z range.
    z = head.grid.axes()[2]
    middle = float(z[z.size // 2])
    cfg = dataclasses.replace(small_cfg, slab_z_range=(middle - 1e-4, middle + 1e-4))
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
