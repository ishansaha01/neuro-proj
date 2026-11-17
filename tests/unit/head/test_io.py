import dataclasses
import hashlib

import numpy as np

from neurosim.common.config import config_hash
from neurosim.common.io import load_bundle
from neurosim.head.generate import GeneratedHead
from neurosim.head.io import load_head, save_head
from neurosim.head.types import HeadVolume


def test_round_trip_restores_every_field(tmp_path, small_head, small_cfg):
    path = tmp_path / "head.h5"
    save_head(path, small_head, small_cfg, root_seed=3)
    loaded = load_head(path)
    assert isinstance(loaded, GeneratedHead)
    assert loaded.grid == small_head.grid
    assert loaded.alpha_power == small_head.alpha_power
    assert loaded.materials == small_cfg.materials
    for spec in dataclasses.fields(HeadVolume):
        if spec.name in ("grid", "alpha_power"):
            continue
        stored, original = getattr(loaded, spec.name), getattr(small_head, spec.name)
        assert stored.dtype == original.dtype, spec.name
        np.testing.assert_array_equal(stored, original)


def test_bundle_carries_provenance(tmp_path, small_head, small_cfg):
    path = tmp_path / "head.h5"
    save_head(path, small_head, small_cfg, root_seed=3)
    _, attrs = load_bundle(path)
    assert attrs["root_seed"] == 3
    assert attrs["config_hash"] == config_hash(small_cfg)
    assert "spacing: 0.002" in attrs["config_yaml"]
    assert {"git_commit", "neurosim_version"} <= set(attrs)


def test_saved_file_is_byte_deterministic(tmp_path, small_head, small_cfg):
    digests = []
    for name in ("a.h5", "b.h5"):
        save_head(tmp_path / name, small_head, small_cfg, root_seed=3)
        digests.append(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
    assert digests[0] == digests[1]
