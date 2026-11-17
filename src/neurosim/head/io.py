"""Save and load a head volume as one HDF5 bundle with provenance."""

from __future__ import annotations

import dataclasses
import os

from neurosim.common.config import from_dict, to_dict
from neurosim.common.grid import Grid3D
from neurosim.common.io import load_bundle, provenance, save_bundle
from neurosim.head.config import HeadConfig
from neurosim.head.generate import GeneratedHead
from neurosim.head.materials import MaterialParams
from neurosim.head.types import HeadVolume

# Fields of `HeadVolume` stored as datasets (everything except the grid and scalars).
_ARRAY_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(HeadVolume) if f.name not in ("grid", "alpha_power")
)


def save_head(
    path: str | os.PathLike[str], head: HeadVolume, cfg: HeadConfig, root_seed: int
) -> None:
    """Write a head to an HDF5 bundle.

    path: output file.
    head: the head to store.
    cfg: the config that generated it (stored as provenance; its material
        parameters are stored for `load_head`).
    root_seed: root seed of the `SeedTree` that generated it.
    """
    attrs = provenance(cfg, root_seed)
    attrs.update(
        grid_shape=list(head.grid.shape),
        grid_spacing=head.grid.spacing,
        grid_origin=list(head.grid.origin),
        alpha_power=head.alpha_power,
        materials=to_dict(cfg.materials),
    )
    save_bundle(path, {name: getattr(head, name) for name in _ARRAY_FIELDS}, attrs)


def load_head(path: str | os.PathLike[str]) -> GeneratedHead:
    """Read a head written by `save_head`.

    Returns a `GeneratedHead`: a `HeadVolume` plus the stored material parameters.
    """
    arrays, attrs = load_bundle(path)
    grid = Grid3D(
        shape=tuple(attrs["grid_shape"]),
        spacing=attrs["grid_spacing"],
        origin=tuple(attrs["grid_origin"]),
    )
    return GeneratedHead(
        grid=grid,
        alpha_power=attrs["alpha_power"],
        materials=from_dict(MaterialParams, attrs["materials"]),
        **{name: arrays[name] for name in _ARRAY_FIELDS},
    )
