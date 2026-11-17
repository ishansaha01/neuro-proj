"""Generate one ground-truth head from a config and a seed tree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.head.anatomy import (
    HeadGeometry,
    draw_geometry,
    max_scalp_radius,
    radial_sdf,
    surface_radii,
    unit_vector,
)
from neurosim.head.config import GridConfig, HeadConfig
from neurosim.head.fields import grid_random_field
from neurosim.head.materials import MaterialParams, acoustic_properties, material_fractions
from neurosim.head.microstructure import hidden_sound_speed, skull_porosity
from neurosim.head.types import SURFACES, HeadVolume, labels_from_sdf
from neurosim.head.vasculature import cbv_fields

_BRAIN_OUTER = SURFACES.index("brain_outer")
_WHITE_OUTER = SURFACES.index("white_outer")

# Bytes per voxel of the fields stored in a `HeadVolume`: sdf 7 x 4, porosity 4,
# hidden_c 4, labels 1, three acoustic maps 12, material fractions 16,
# cbv_baseline 4, activation 4, blood_velocity 12.
_STORED_BYTES_PER_VOXEL = 85
# Largest temporary memory per voxel during generation, on top of the stored
# fields. Measured with tracemalloc on a 160^3 grid (2026-11-17): 41 bytes per
# voxel, reached in the vessel step (two fields and their six gradients).
_WORKING_BYTES_PER_VOXEL = 48
# Wall time per voxel, single-threaded on the 2-core development machine
# (2026-11-17): 1.0e-6 s measured on a 160^3 grid.
_SECONDS_PER_VOXEL = 1.3e-6


@dataclass(frozen=True)
class GeneratedHead(HeadVolume):
    """A `HeadVolume` together with the material parameters that derived its maps.

    `HeadVolume` itself does not carry them, yet `plane.sample_plane` needs them
    to apply the same material mapping on the fine grid.
    """

    materials: MaterialParams


def _full_extent(cfg: HeadConfig) -> tuple[float, float, float]:
    """Extent of the full-head grid along x, y, z [m]; it holds the scalp of any seed."""
    return tuple(2.0 * (radius + cfg.grid.full_margin) for radius in max_scalp_radius(cfg.anatomy))


def _box_grid(grid_cfg: GridConfig, center: np.ndarray) -> Grid3D:
    shape = tuple(max(1, round(size / grid_cfg.spacing)) for size in grid_cfg.box_size)
    origin = tuple(float(c - 0.5 * (n - 1) * grid_cfg.spacing) for c, n in zip(center, shape))
    return Grid3D(shape=shape, spacing=grid_cfg.spacing, origin=origin)


def head_grid(cfg: HeadConfig, geometry: HeadGeometry) -> Grid3D:
    """Voxel grid of a head.

    Region 'full': centred on the head centre and large enough for the scalp of
    any seed, so it depends on the config only. Region 'box': centred
    `cfg.grid.box_center_depth` below the scalp of this head along the target
    direction.
    """
    if cfg.grid.region == "full":
        return Grid3D.centered(_full_extent(cfg), cfg.grid.spacing)
    target = unit_vector(cfg.cbv.target_direction)
    scalp_radius = float(surface_radii(geometry, target[None])[0, 0])
    return _box_grid(cfg.grid, target * (scalp_radius - cfg.grid.box_center_depth))


def _n_voxels(cfg: HeadConfig) -> int:
    if cfg.grid.region == "full":
        return Grid3D.centered(_full_extent(cfg), cfg.grid.spacing).n_voxels
    return _box_grid(cfg.grid, np.zeros(3)).n_voxels


def estimate_memory_bytes(cfg: HeadConfig) -> int:
    """Estimated peak memory of `generate_head` for a config [bytes].

    Stored fields (85 bytes per voxel) plus the largest temporaries.
    """
    return _n_voxels(cfg) * (_STORED_BYTES_PER_VOXEL + _WORKING_BYTES_PER_VOXEL)


def estimate_runtime_seconds(cfg: HeadConfig) -> float:
    """Rough single-threaded run time of `generate_head` for a config [s]."""
    return _n_voxels(cfg) * _SECONDS_PER_VOXEL


def activation_center(cfg: HeadConfig, geometry: HeadGeometry) -> np.ndarray:
    """Head-frame centre of the activated patch [m], shape (3,).

    The point half-way through the cortex along `cfg.cbv.target_direction`.
    """
    target = unit_vector(cfg.cbv.target_direction)
    radii = surface_radii(geometry, target[None])[:, 0]
    return target * 0.5 * float(radii[_BRAIN_OUTER] + radii[_WHITE_OUTER])


def generate_head(cfg: HeadConfig, tree: SeedTree) -> GeneratedHead:
    """Generate the ground-truth head of one seed.

    Streams used: `anatomy` (surfaces), `skull_microstructure` (porosity
    texture), `hidden_soundspeed` (hidden sound-speed pattern) and `cbv`
    (vascular texture). The acoustic maps and material fractions are derived
    from the latent fields by `head.materials`.

    Returns a `GeneratedHead` (a `HeadVolume` plus its material parameters).
    """
    geometry = draw_geometry(cfg.anatomy, tree.rng("anatomy"))
    grid = head_grid(cfg, geometry)
    sdf = radial_sdf(geometry, grid)
    labels = labels_from_sdf(sdf)

    micro = cfg.microstructure
    porosity = skull_porosity(
        sdf,
        labels,
        grid_random_field(tree.rng("skull_microstructure"), grid, micro.texture_correlation),
        micro,
    )
    hidden_c = hidden_sound_speed(
        porosity,
        labels,
        grid_random_field(tree.rng("hidden_soundspeed"), grid, micro.hidden_correlation),
        micro,
        cfg.materials.skull,
    )
    sound_speed, density, attenuation = acoustic_properties(
        labels, porosity, hidden_c, cfg.materials
    )
    cbv_baseline, activation, blood_velocity = cbv_fields(
        tree.rng("cbv"), grid, labels, activation_center(cfg, geometry), cfg.cbv
    )
    return GeneratedHead(
        grid=grid,
        sdf=sdf,
        porosity=porosity,
        hidden_c=hidden_c,
        labels=labels,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        material_fractions=material_fractions(labels, porosity),
        cbv_baseline=cbv_baseline,
        activation=activation,
        blood_velocity=blood_velocity,
        alpha_power=cfg.materials.alpha_power,
        materials=cfg.materials,
    )
