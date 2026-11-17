"""Save and load CT volumes as HDF5 bundles with provenance."""

from __future__ import annotations

import os

import numpy as np

from neurosim.common.grid import Grid3D
from neurosim.common.io import load_bundle, provenance, save_bundle
from neurosim.ct.config import CtConfig
from neurosim.ct.types import CtVolume


def save_ct(path: str | os.PathLike[str], ct: CtVolume, cfg: CtConfig, root_seed: int) -> None:
    """Write a CT volume, its grid, its scan metadata and the run provenance.

    Args:
        path: output HDF5 file.
        ct: reconstructed volume.
        cfg: scan configuration that produced it.
        root_seed: root seed of the run's seed tree.
    """
    attrs = {
        "grid_shape": list(ct.grid.shape),
        "grid_spacing": ct.grid.spacing,
        "grid_origin": list(ct.grid.origin),
        "meta": dict(ct.meta),
        **provenance(cfg, root_seed),
    }
    save_bundle(path, {"hu": ct.hu}, attrs)


def load_ct(path: str | os.PathLike[str]) -> CtVolume:
    """Read a CT volume written by `save_ct`.

    The returned ``meta`` holds the stored scan metadata plus a ``provenance``
    entry (config YAML and hash, root seed, git commit, package version).
    """
    arrays, attrs = load_bundle(path)
    grid = Grid3D(
        shape=tuple(attrs.pop("grid_shape")),
        spacing=attrs.pop("grid_spacing"),
        origin=tuple(attrs.pop("grid_origin")),
    )
    meta = dict(attrs.pop("meta"))
    meta["provenance"] = attrs
    return CtVolume(hu=np.asarray(arrays["hu"], dtype=np.float32), grid=grid, meta=meta)
