"""Save and load a head volume or a plane medium as one HDF5 bundle with provenance."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from neurosim.common.config import from_dict, to_dict
from neurosim.common.grid import Grid2D, Grid3D, PlaneSpec
from neurosim.common.io import load_bundle, provenance, save_bundle
from neurosim.head.config import HeadConfig
from neurosim.head.generate import GeneratedHead
from neurosim.head.materials import MaterialParams
from neurosim.head.types import HeadVolume, PlaneMedium

# Fields of `HeadVolume` stored as datasets (everything except the grid and scalars).
_ARRAY_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(HeadVolume) if f.name not in ("grid", "alpha_power")
)
# Fields of `PlaneMedium` stored as datasets (everything except the grid, the plane and scalars).
_PLANE_ARRAY_FIELDS: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(PlaneMedium)
    if f.name not in ("grid", "plane", "alpha_power")
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


def plane_medium_to_bundle(medium: PlaneMedium) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arrays and attributes that hold a plane medium.

    One dataset per array field of `PlaneMedium`; attributes ``grid_shape``,
    ``grid_spacing``, ``grid_origin``, ``alpha_power``, ``plane_origin``,
    ``plane_u_lateral`` and ``plane_u_depth``.
    """
    attrs = {
        "grid_shape": list(medium.grid.shape),
        "grid_spacing": medium.grid.spacing,
        "grid_origin": list(medium.grid.origin),
        "alpha_power": medium.alpha_power,
        "plane_origin": list(medium.plane.origin),
        "plane_u_lateral": list(medium.plane.u_lateral),
        "plane_u_depth": list(medium.plane.u_depth),
    }
    return {name: getattr(medium, name) for name in _PLANE_ARRAY_FIELDS}, attrs


def plane_medium_from_bundle(
    arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]
) -> PlaneMedium:
    """Rebuild a plane medium from the arrays and attributes of `plane_medium_to_bundle`."""
    return PlaneMedium(
        grid=Grid2D(
            shape=tuple(attrs["grid_shape"]),
            spacing=attrs["grid_spacing"],
            origin=tuple(attrs["grid_origin"]),
        ),
        plane=PlaneSpec(
            origin=tuple(attrs["plane_origin"]),
            u_lateral=tuple(attrs["plane_u_lateral"]),
            u_depth=tuple(attrs["plane_u_depth"]),
        ),
        alpha_power=attrs["alpha_power"],
        **{name: arrays[name] for name in _PLANE_ARRAY_FIELDS},
    )


def save_plane_medium(
    path: str | os.PathLike[str], medium: PlaneMedium, provenance: Mapping[str, Any]
) -> None:
    """Write a plane medium to an HDF5 bundle.

    path: output file.
    medium: the plane medium to store.
    provenance: attributes that say where the medium comes from
        (`common.io.provenance` of the config and seed that produced it).
    """
    arrays, attrs = plane_medium_to_bundle(medium)
    save_bundle(path, arrays, {**provenance, **attrs})


def load_plane_medium(path: str | os.PathLike[str]) -> PlaneMedium:
    """Read a plane medium written by `save_plane_medium`."""
    return plane_medium_from_bundle(*load_bundle(path))
