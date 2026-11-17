"""save_ct / load_ct round trip."""

from __future__ import annotations

import numpy as np

from neurosim.common.config import config_hash
from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.ct.io import load_ct, save_ct
from neurosim.ct.types import CtVolume


def test_round_trip_keeps_volume_grid_meta_and_provenance(tmp_path, small_cfg):
    grid = Grid3D(shape=(5, 4, 3), spacing=0.002, origin=(-0.004, -0.003, 0.001))
    hu = SeedTree(2).rng("ct_photon_noise").normal(0.0, 500.0, size=grid.shape).astype(np.float32)
    ct = CtVolume(hu=hu, grid=grid, meta={"slice_start": 7, "mu_water": 20.1, "root_seed": 2})
    path = tmp_path / "ct.h5"
    save_ct(path, ct, small_cfg, root_seed=2)
    loaded = load_ct(path)
    np.testing.assert_array_equal(loaded.hu, hu)
    assert loaded.hu.dtype == np.float32
    assert loaded.grid == grid
    assert loaded.meta["slice_start"] == 7 and loaded.meta["mu_water"] == 20.1
    provenance = loaded.meta["provenance"]
    assert provenance["root_seed"] == 2
    assert provenance["config_hash"] == config_hash(small_cfg)
    assert "git_commit" in provenance and "config_yaml" in provenance


def test_saving_twice_gives_identical_bytes(tmp_path, small_cfg):
    grid = Grid3D(shape=(3, 3, 2), spacing=0.001, origin=(0.0, 0.0, 0.0))
    ct = CtVolume(hu=np.zeros(grid.shape, np.float32), grid=grid, meta={"slice_start": 0})
    save_ct(tmp_path / "a.h5", ct, small_cfg, root_seed=0)
    save_ct(tmp_path / "b.h5", ct, small_cfg, root_seed=0)
    assert (tmp_path / "a.h5").read_bytes() == (tmp_path / "b.h5").read_bytes()
