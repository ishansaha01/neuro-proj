"""The CT-derived skull in the imaging plane: the imperfect prior of the reconstruction.

The ultrasound imaging plane is oblique to the head axes, while the CT model
scans axial slices (constant ``iz``) of whatever volume it is given. Instead of
scanning every axial slice the plane crosses, the head is resampled onto a
thin slab aligned with the imaging plane and that slab is scanned: physically
a tilted-gantry acquisition whose central slice *is* the imaging plane.

Frames
------
- Head frame: points ``p`` [m]; volumes indexed ``[ix, iy, iz]``.
- Imaging-plane frame: ``(x, z)`` = (lateral, depth) [m], with
  ``p = plane.origin + x * u_lateral + z * u_depth``; maps indexed ``[iz, ix]``.
- Slab frame (the CT's local frame), right-handed, coordinates ``(a, b, w)`` [m]:

      axes    (u_lateral, u_depth, n)   with n = u_lateral x u_depth
      origin  c = plane.origin + x_c * u_lateral + z_c * u_depth

  where ``(x_c, z_c)`` are the mid-points of the plane region to cover. So

      a = x - x_c   (lateral),   b = z - z_c   (depth),   w = distance out of the plane,

  and ``w = 0`` is the imaging plane. The frame is stored as a `PlaneSpec` whose
  origin is ``c`` (`slab_frame`); `slab_to_head` and `head_to_slab` are the only
  two functions that convert between it and the head frame.

The slab grid (`slab_grid`) is a `Grid3D` in slab coordinates, indexed
``[ia, ib, iw]``, symmetric about the slab origin with an odd number of voxels
on every axis: voxel ``[(n_a - 1) / 2, (n_b - 1) / 2, (n_w - 1) / 2]`` is centred
on ``c`` and slice ``iw = (n_w - 1) / 2`` is the imaging plane. Voxel
``[ia, ib, iw]`` is at plane coordinates

    x = x_c + (ia - (n_a - 1) / 2) * spacing,    z = z_c + (ib - (n_b - 1) / 2) * spacing.

A CT slice is therefore indexed ``[lateral, depth]`` while plane maps are
``[depth, lateral]``. `ct_plane_hu` never transposes arrays: it converts the
pixel positions of the plane grid to slab coordinates and interpolates there.

What the CT code (`neurosim.ct.simulate`) assumes about the grid it is given,
and how the slab meets it:

- The isocentre is at grid ``x = y = 0``: here the slab origin ``c``, the centre
  of the scanned region, which also gives the shortest rays (lowest cost).
- Slices are planes of constant grid ``z``, selected by ``CtConfig.slab_z_range``
  in the grid's own z coordinate. In the slab frame that coordinate is ``w``,
  so the ``slab_z_range`` of the scan config (a head-frame axial interval) does
  not apply: `scan_plane` scans every slice of the slab grid and the slab
  thickness is set by `CtPriorConfig.slab_half_slices`.
- Every material other than air must lie inside the scan field of view (a disc
  of radius ``FanBeamGeometry.fov_radius`` about the isocentre);
  `simulate_ct_fractions` raises ``ValueError`` otherwise.
- Nothing exists outside the grid: the slab is scanned as an isolated block.
  Slab voxels outside the head volume are air. The block is smaller than a head
  (and a box-region head is itself only part of one), so the beam is hardened
  less and attenuated less than in a clinical head scan.
- Slices are infinitely thin and independent; the photon noise of slab slice
  ``iw`` comes from the stream ``("ct_photon_noise", iw)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike

from neurosim.common.config import field, to_dict
from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.grid import (
    Grid2D,
    Grid3D,
    PlaneSpec,
    plane_points,
    plane_to_world,
    sample_volume,
)
from neurosim.common.seeding import SeedTree
from neurosim.ct.acoustics import (
    BONE_THRESHOLD_HU,
    hu_to_acoustic,
    nominal_hu_mapping,
    sample_hu_mapping,
)
from neurosim.ct.config import CtConfig
from neurosim.ct.simulate import estimate_cost as estimate_ct_cost
from neurosim.ct.simulate import simulate_ct_fractions
from neurosim.ct.types import CtVolume, HuMapping
from neurosim.head.types import HeadVolume, PlaneMedium, Tissue

_AIR = BASIS_MATERIALS.index("air")

# Slack, in voxels, when counting the voxels needed to cover a length, so that a
# length that is a whole number of voxels up to rounding does not gain one.
_VOXEL_COUNT_TOLERANCE = 1.0e-9

# Keys of `CtVolume.meta` written by `scan_plane`.
_META_FRAME = "slab_frame"
_META_PRIOR_CONFIG = "ct_prior"

Range = tuple[float, float]


@dataclass(frozen=True)
class CtPriorConfig:
    """How the CT-derived skull of an imaging plane is acquired and mapped."""

    slab_half_slices: int = field(
        1,
        ge=0,
        doc="Slices scanned on each side of the central slice (the imaging plane) [-]; "
        "the slab has 2 * slab_half_slices + 1 slices. CT slices are independent and "
        "infinitely thin, so only the central slice enters the CT-derived medium of the "
        "plane itself; the neighbours describe the skull just out of the plane and cost "
        "one scan each. 0 is the cheapest valid setting. Modelling choice.",
    )
    slab_spacing: Optional[float] = field(
        None,
        gt=0.0,
        doc="Voxel size of the slab, in-plane pixel size and slice spacing [m]. None: "
        "the voxel size of the head grid (0.5 mm by default, the in-plane pixel size "
        "of a clinical head CT with a 25 cm field of view on a 512 matrix).",
    )
    margin: float = field(
        5.0e-3,
        ge=0.0,
        doc="Width of head scanned around the requested plane region on each in-plane "
        "side [m], so that the cut faces of the block (tissue ending in nothing) are "
        "not at the border of the region. 5 mm is ten default voxels; assumed, to verify.",
    )
    hu_mapping: Literal["nominal", "sampled"] = field(
        "nominal",
        doc="Constants of the HU -> acoustics map: 'nominal' (published values, "
        "ct.acoustics.nominal_hu_mapping) or 'sampled' (drawn within their published "
        "uncertainty from the 'ct_hu_mapping' stream, ct.acoustics.sample_hu_mapping).",
    )


def _checked_range(values: Range, name: str) -> Range:
    first, last = (float(value) for value in values)
    if last < first:
        raise ValueError(f"{name} must be (first, last) with first <= last, got {values}")
    return first, last


def slab_frame(plane: PlaneSpec, extent_x: Range, extent_z: Range) -> PlaneSpec:
    """The slab frame of a plane region (module docstring, "Frames").

    plane: the imaging plane in the head frame.
    extent_x: lateral range ``(first, last)`` of the plane region [m].
    extent_z: depth range ``(first, last)`` of the plane region [m].

    Returns a `PlaneSpec` with the axes of ``plane`` whose origin is the centre
    of the region, the CT isocentre, in the head frame [m].
    """
    x_first, x_last = _checked_range(extent_x, "extent_x")
    z_first, z_last = _checked_range(extent_z, "extent_z")
    center = plane_to_world(plane, (0.5 * (x_first + x_last), 0.5 * (z_first + z_last)))
    return PlaneSpec(origin=tuple(center), u_lateral=plane.u_lateral, u_depth=plane.u_depth)


def _axes(frame: PlaneSpec) -> np.ndarray:
    """Unit vectors of the slab axes (a, b, w) in the head frame, as rows; shape (3, 3)."""
    return np.array([frame.u_lateral, frame.u_depth, np.cross(frame.u_lateral, frame.u_depth)])


def slab_to_head(frame: PlaneSpec, local: ArrayLike) -> np.ndarray:
    """Convert slab coordinates to the head frame.

    frame: slab frame from `slab_frame`.
    local: slab coordinates ``(a, b, w)`` [m], shape ``(..., 3)``.
    Returns head-frame points [m], shape ``(..., 3)``.
    """
    return np.asarray(frame.origin) + np.asarray(local, dtype=np.float64) @ _axes(frame)


def head_to_slab(frame: PlaneSpec, points: ArrayLike) -> np.ndarray:
    """Convert head-frame points to slab coordinates (inverse of `slab_to_head`).

    frame: slab frame from `slab_frame`.
    points: head-frame points [m], shape ``(..., 3)``.
    Returns slab coordinates ``(a, b, w)`` [m], shape ``(..., 3)``.
    """
    offset = np.asarray(points, dtype=np.float64) - np.asarray(frame.origin)
    return offset @ _axes(frame).T


def slab_grid(extent_x: Range, extent_z: Range, spacing: float, cfg: CtPriorConfig) -> Grid3D:
    """The voxel grid of the slab, in slab coordinates (module docstring).

    extent_x: lateral range ``(first, last)`` of the plane region [m].
    extent_z: depth range ``(first, last)`` of the plane region [m].
    spacing: voxel size of the slab [m].
    cfg: slab settings (margin and number of slices).

    Returns a `Grid3D` of shape ``(n_lateral, n_depth, n_normal)``, all odd,
    symmetric about the slab origin. Its in-plane voxel centres reach at least
    ``cfg.margin`` beyond both ends of each range.
    """
    in_plane = []
    for name, values in (("extent_x", extent_x), ("extent_z", extent_z)):
        first, last = _checked_range(values, name)
        half_width = 0.5 * (last - first) + cfg.margin
        in_plane.append(2 * math.ceil(half_width / spacing - _VOXEL_COUNT_TOLERANCE) + 1)
    shape = (*in_plane, 2 * cfg.slab_half_slices + 1)
    return Grid3D.centered(tuple(n * spacing for n in shape), spacing)


def _slab_spacing(head_spacing: float, cfg: CtPriorConfig) -> float:
    return head_spacing if cfg.slab_spacing is None else cfg.slab_spacing


def plane_aligned_fractions(
    head: HeadVolume, plane: PlaneSpec, extent_x: Range, extent_z: Range, cfg: CtPriorConfig
) -> tuple[np.ndarray, Grid3D]:
    """Resample the head's CT basis-material fractions onto the plane-aligned slab.

    The fractions are interpolated trilinearly at the slab voxel centres and
    renormalised to sum to 1. Slab voxels outside the head volume are air.

    head: the 3D head (only ``grid`` and ``material_fractions`` are used).
    plane: the imaging plane in the head frame.
    extent_x: lateral range ``(first, last)`` of the plane region to cover [m].
    extent_z: depth range ``(first, last)`` of the plane region to cover [m].
    cfg: slab settings.

    Returns ``(fractions, grid)``: volume fractions of `BASIS_MATERIALS` [-],
    float32, shape ``(4, n_lateral, n_depth, n_normal)``, and the slab grid
    they live on (slab coordinates, see `slab_grid`).
    """
    frame = slab_frame(plane, extent_x, extent_z)
    grid = slab_grid(extent_x, extent_z, _slab_spacing(head.grid.spacing, cfg), cfg)
    points = slab_to_head(frame, grid.coords())
    fractions = sample_volume(head.material_fractions, head.grid, points, order=1, fill=np.nan)
    outside = np.isnan(fractions[0])
    fractions[:, outside] = 0.0
    fractions[_AIR, outside] = 1.0
    fractions /= fractions.sum(axis=0)
    return fractions, grid


def _slab_scan_config(ct_cfg: CtConfig) -> CtConfig:
    """The scan settings with every slice of the slab grid selected."""
    return dataclasses.replace(ct_cfg, slab_z_range=None)


def scan_plane(
    head: HeadVolume,
    plane: PlaneSpec,
    ct_cfg: CtConfig,
    cfg: CtPriorConfig,
    tree: SeedTree,
    extent_x: Range,
    extent_z: Range,
) -> CtVolume:
    """CT scan of the slab aligned with an imaging plane.

    head: the 3D head.
    plane: the imaging plane in the head frame.
    ct_cfg: scan settings. ``ct_cfg.slab_z_range`` is ignored: every slice of
        the slab is scanned (module docstring).
    cfg: slab settings.
    tree: seed tree of the run; slab slice ``iw`` draws its photon noise from
        ``("ct_photon_noise", iw)``.
    extent_x: lateral range ``(first, last)`` of the plane region to cover [m].
    extent_z: depth range ``(first, last)`` of the plane region to cover [m].

    Returns a `CtVolume` in the slab frame, not the head frame: ``hu`` [HU],
    float32, indexed ``[ia, ib, iw]`` = (lateral, depth, normal) on ``grid``
    (slab coordinates). ``meta`` holds what `simulate_ct_fractions` records
    plus ``slab_frame`` (origin and axes of the slab frame in the head frame)
    and ``ct_prior`` (the slab settings).

    Raises ``ValueError`` if the slab reaches outside the scan field of view.
    """
    fractions, grid = plane_aligned_fractions(head, plane, extent_x, extent_z, cfg)
    ct = simulate_ct_fractions(fractions, grid, _slab_scan_config(ct_cfg), tree)
    frame = slab_frame(plane, extent_x, extent_z)
    meta = {
        **ct.meta,
        _META_FRAME: {
            "origin": list(frame.origin),
            "u_lateral": list(frame.u_lateral),
            "u_depth": list(frame.u_depth),
        },
        _META_PRIOR_CONFIG: to_dict(cfg),
    }
    return CtVolume(hu=ct.hu, grid=ct.grid, meta=meta)


def ct_plane_hu(ct: CtVolume, grid: Grid2D, plane: PlaneSpec) -> np.ndarray:
    """Hounsfield units of a plane-aligned CT on a fine imaging-plane grid.

    Every pixel of ``grid`` is converted to slab coordinates and the CT is
    interpolated trilinearly there. For the plane the slab was scanned for,
    all pixels have ``w = 0`` and the interpolation is bilinear within the
    central slice; for any other plane it blends the slices around each pixel.

    ct: CT volume from `scan_plane` (its ``meta`` carries the slab frame).
    grid: pixel grid in the imaging plane.
    plane: the imaging plane in the head frame.

    Returns HU, float32, shape ``(nz, nx)``, indexed ``[iz, ix]``.

    Raises ``ValueError`` if ``ct`` does not come from `scan_plane` or a pixel
    of ``grid`` lies outside the slab.
    """
    if _META_FRAME not in ct.meta:
        raise ValueError("this CtVolume carries no slab frame; it does not come from scan_plane")
    stored = ct.meta[_META_FRAME]
    frame = PlaneSpec(
        origin=tuple(stored["origin"]),
        u_lateral=tuple(stored["u_lateral"]),
        u_depth=tuple(stored["u_depth"]),
    )
    local = head_to_slab(frame, plane_points(plane, grid))
    hu = sample_volume(ct.hu, ct.grid, local, order=1, fill=np.nan)
    if np.any(np.isnan(hu)):
        raise ValueError(
            "the plane grid reaches outside the scanned slab; scan a larger region "
            "or use a smaller plane grid"
        )
    return hu


def prior_hu_mapping(ct_cfg: CtConfig, cfg: CtPriorConfig, tree: SeedTree) -> HuMapping:
    """The HU -> acoustics map selected by ``cfg.hu_mapping``.

    'nominal': `ct.acoustics.nominal_hu_mapping`. 'sampled': one draw of
    `ct.acoustics.sample_hu_mapping` from the stream ``"ct_hu_mapping"`` of ``tree``.
    """
    if cfg.hu_mapping == "nominal":
        return nominal_hu_mapping(ct_cfg)
    return sample_hu_mapping(tree, ct_cfg)


def ct_plane_medium(
    hu_plane: np.ndarray, grid: Grid2D, plane: PlaneSpec, mapping: HuMapping, alpha_power: float
) -> PlaneMedium:
    """Acoustic estimate of the imaging plane from the CT alone.

    Sound speed, density and absorption are `ct.acoustics.hu_to_acoustic`
    applied pixel by pixel on the fine grid. The fields a CT cannot know are
    filled as follows:

    - ``labels``: `Tissue.SKULL_OUTER` where ``HU >= BONE_THRESHOLD_HU`` (the
      bone threshold of `ct.acoustics`), `Tissue.BACKGROUND` elsewhere. The CT
      separates neither the skull layers nor the soft tissues, so
      ``SKULL_OUTER`` stands for "bone" (it makes `head.materials.skull_mask`
      work on this medium) and ``BACKGROUND`` for everything else.
    - ``porosity``: the CT-estimated porosity of the map in bone pixels, 0
      elsewhere. It is recovered from the mapped sound speed through the map's
      linear law, ``phi = (c_bone - c) / (c_bone - c_water)``, so it is exactly
      the porosity `hu_to_acoustic` used.
    - ``hidden_c``, ``cbv_baseline``, ``activation``, ``blood_velocity``: 0.

    hu_plane: CT numbers on ``grid`` [HU], shape ``(nz, nx)``, indexed ``[iz, ix]``.
    grid: pixel grid in the imaging plane.
    plane: the imaging plane in the head frame.
    mapping: constants of the HU -> acoustics map.
    alpha_power: power-law exponent ``y`` the absorption prefactors of
        ``mapping`` are expressed for (``CtConfig.hu_mapping.alpha_power``) [-].

    Returns the CT-derived `PlaneMedium` (arrays ``[iz, ix]``, float32; labels uint8).
    """
    hu_plane = np.asarray(hu_plane)
    if hu_plane.shape != grid.shape:
        raise ValueError(f"hu_plane has shape {hu_plane.shape}, expected {grid.shape}")
    sound_speed, density, attenuation = hu_to_acoustic(hu_plane, mapping)
    is_bone = hu_plane >= BONE_THRESHOLD_HU
    labels = np.where(is_bone, Tissue.SKULL_OUTER, Tissue.BACKGROUND).astype(np.uint8)
    porosity = (mapping.c_bone - sound_speed) / (mapping.c_bone - mapping.c_water)
    porosity = np.where(is_bone, np.clip(porosity, 0.0, 1.0), 0.0).astype(np.float32)
    zeros = np.zeros(grid.shape, dtype=np.float32)
    return PlaneMedium(
        grid=grid,
        plane=plane,
        labels=labels,
        porosity=porosity,
        hidden_c=zeros.copy(),
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        cbv_baseline=zeros.copy(),
        activation=zeros.copy(),
        blood_velocity=np.zeros((2, *grid.shape), dtype=np.float32),
        alpha_power=float(alpha_power),
    )


def ct_prior_medium(
    head: HeadVolume,
    plane: PlaneSpec,
    grid: Grid2D,
    ct_cfg: CtConfig,
    cfg: CtPriorConfig,
    tree: SeedTree,
) -> tuple[CtVolume, PlaneMedium]:
    """Scan the slab of an imaging plane and derive the plane's CT-based medium.

    Chains `scan_plane` (over the region spanned by ``grid``), `ct_plane_hu`,
    `prior_hu_mapping` and `ct_plane_medium`.

    head: the 3D head.
    plane: the imaging plane in the head frame.
    grid: fine pixel grid of the acoustic medium in the imaging plane.
    ct_cfg: scan settings and HU-mapping constants.
    cfg: slab settings and choice of HU mapping.
    tree: seed tree of the run.

    Returns ``(ct, medium)``: the slab CT (slab frame, see `scan_plane`) and
    the CT-derived `PlaneMedium` on ``grid``.

    Raises ``ValueError`` if the absorption exponent of the HU mapping differs
    from that of the head, since both media must share one power law.
    """
    alpha_power = ct_cfg.hu_mapping.alpha_power
    if alpha_power != head.alpha_power:
        raise ValueError(
            f"ct_cfg.hu_mapping.alpha_power ({alpha_power}) must equal the head's "
            f"alpha_power ({head.alpha_power})"
        )
    z, x = grid.axes()
    ct = scan_plane(head, plane, ct_cfg, cfg, tree, (x[0], x[-1]), (z[0], z[-1]))
    hu_plane = ct_plane_hu(ct, grid, plane)
    mapping = prior_hu_mapping(ct_cfg, cfg, tree)
    return ct, ct_plane_medium(hu_plane, grid, plane, mapping, alpha_power)


def estimate_cost(
    head_spacing: float, ct_cfg: CtConfig, cfg: CtPriorConfig, extent_x: Range, extent_z: Range
) -> dict[str, float]:
    """Estimate the size and cost of `scan_plane` before running it.

    head_spacing: voxel size of the head grid [m] (used when
        ``cfg.slab_spacing`` is None).
    ct_cfg: scan settings.
    cfg: slab settings.
    extent_x: lateral range ``(first, last)`` of the plane region to cover [m].
    extent_z: depth range ``(first, last)`` of the plane region to cover [m].

    Returns the dict of `ct.simulate.estimate_cost` for the slab grid (slices,
    rays, samples, ``memory_bytes`` excluding the head, ``time_s`` as an upper
    estimate on the development machine) plus the slab shape ``n_lateral``,
    ``n_depth`` and its voxel size ``spacing`` [m].
    """
    grid = slab_grid(extent_x, extent_z, _slab_spacing(head_spacing, cfg), cfg)
    cost = estimate_ct_cost(grid, _slab_scan_config(ct_cfg))
    return {
        **cost,
        "n_lateral": float(grid.shape[0]),
        "n_depth": float(grid.shape[1]),
        "spacing": grid.spacing,
    }
