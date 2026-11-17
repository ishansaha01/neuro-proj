"""Imaging planes through the head: default placement and fine-grid sampling."""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D, PlaneSpec, plane_points, sample_volume
from neurosim.head.anatomy import unit_vector
from neurosim.head.config import HeadConfig
from neurosim.head.generate import GeneratedHead
from neurosim.head.materials import MaterialParams, acoustic_properties, brain_mask, skull_mask
from neurosim.head.types import SURFACES, HeadVolume, PlaneMedium, labels_from_sdf

_SCALP_OUTER = SURFACES.index("scalp_outer")
_BRAIN_OUTER = SURFACES.index("brain_outer")
_WHITE_OUTER = SURFACES.index("white_outer")

# Head axis used to orient the array when it is not nearly parallel to the depth
# axis, and the fallback axis; see `default_plane`.
_LATERAL_REFERENCE = np.array([1.0, 0.0, 0.0])
_LATERAL_FALLBACK = np.array([0.0, 1.0, 0.0])
_PARALLEL_LIMIT = 0.9


def _zero_crossing(
    head: HeadVolume, surface: int, start: np.ndarray, direction: np.ndarray
) -> float:
    """Distance from `start` along `direction` at which a surface is crossed [m].

    The SDF of the surface is sampled along the ray every half voxel inside the
    head grid, and the first sign change is located by linear interpolation.
    """
    step = 0.5 * head.grid.spacing
    reach = float(np.linalg.norm(np.asarray(head.grid.shape) * head.grid.spacing)) + float(
        np.linalg.norm(np.asarray(head.grid.origin) - start)
    )
    distances = np.arange(0.0, reach, step)
    values = sample_volume(
        head.sdf[surface], head.grid, start + distances[:, None] * direction, fill=np.nan
    )
    crossing = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
    if crossing.size == 0:
        raise ValueError(
            f"surface '{SURFACES[surface]}' is not crossed inside the head grid along the ray"
        )
    i = crossing[0]
    fraction = values[i] / (values[i] - values[i + 1]) if values[i] != values[i + 1] else 0.0
    return float(distances[i] + fraction * step)


def _sdf_gradient(head: HeadVolume, surface: int, point: np.ndarray) -> np.ndarray:
    """Central-difference gradient of one SDF at a head-frame point, shape (3,)."""
    offsets = head.grid.spacing * np.eye(3)
    forward = sample_volume(head.sdf[surface], head.grid, point + offsets)
    backward = sample_volume(head.sdf[surface], head.grid, point - offsets)
    return (forward - backward).astype(np.float64)


def default_plane(head: HeadVolume, cfg: HeadConfig) -> PlaneSpec:
    """Imaging plane of a probe on the scalp over the activated region.

    The activation centre is the point half-way through the cortex along
    `cfg.cbv.target_direction`. The array centre sits `cfg.probe.standoff`
    outside the scalp on the line through the activation centre along the
    outward scalp normal; `u_depth` points inward along that line, so the
    activation centre lies on the depth axis. `u_lateral` is the head x axis
    made perpendicular to `u_depth` (a coronal-like plane); if `u_depth` is
    within about 26 degrees of the x axis the head y axis is used instead.
    """
    target = unit_vector(cfg.cbv.target_direction)
    origin = np.zeros(3)
    cortex_radius = 0.5 * (
        _zero_crossing(head, _BRAIN_OUTER, origin, target)
        + _zero_crossing(head, _WHITE_OUTER, origin, target)
    )
    center = cortex_radius * target
    scalp_point = _zero_crossing(head, _SCALP_OUTER, origin, target) * target
    normal = _sdf_gradient(head, _SCALP_OUTER, scalp_point)
    normal /= np.linalg.norm(normal)

    scalp_distance = _zero_crossing(head, _SCALP_OUTER, center, normal)
    array_center = center + (scalp_distance + cfg.probe.standoff) * normal
    u_depth = -normal
    reference = (
        _LATERAL_REFERENCE
        if abs(u_depth @ _LATERAL_REFERENCE) < _PARALLEL_LIMIT
        else _LATERAL_FALLBACK
    )
    u_lateral = reference - (reference @ u_depth) * u_depth
    u_lateral /= np.linalg.norm(u_lateral)
    return PlaneSpec(origin=tuple(array_center), u_lateral=tuple(u_lateral), u_depth=tuple(u_depth))


def sample_plane(
    head: HeadVolume, plane: PlaneSpec, grid: Grid2D, params: MaterialParams | None = None
) -> PlaneMedium:
    """Sample a head on a fine 2D grid in one imaging plane.

    Follows docs/ARCHITECTURE.md section 4, rule 2: the latent fields (`sdf`,
    `porosity`, `hidden_c`) are interpolated trilinearly, labels come from the
    interpolated SDFs, porosity and hidden_c are zeroed outside the skull
    labels, and the acoustic maps are derived by the same
    `materials.acoustic_properties` as in 3D. Sound speed is never
    interpolated. The CBV fields are interpolated trilinearly and zeroed
    outside the brain labels; `blood_velocity` is projected on the plane axes.

    head: the 3D head.
    plane: plane embedded in the head frame.
    grid: pixel grid in the plane; every pixel must lie inside the head grid.
    params: material parameters. Default: those of `head` when it comes from
        `generate_head` or `load_head` (`GeneratedHead.materials`); required
        for any other `HeadVolume`.

    Returns the `PlaneMedium` (arrays indexed [iz, ix], float32).
    """
    if params is None:
        if not isinstance(head, GeneratedHead):
            raise TypeError("this HeadVolume carries no material parameters; pass `params`")
        params = head.materials

    points = plane_points(plane, grid)
    index = head.grid.world_to_index(points)
    if np.any(index < -0.5) or np.any(index > np.asarray(head.grid.shape) - 0.5):
        raise ValueError(
            "the plane grid reaches outside the head volume; use a smaller plane grid "
            "or a larger head region"
        )

    labels = labels_from_sdf(sample_volume(head.sdf, head.grid, points))
    in_skull = skull_mask(labels)
    in_brain = brain_mask(labels)
    porosity = sample_volume(head.porosity, head.grid, points)
    hidden_c = sample_volume(head.hidden_c, head.grid, points)
    porosity[~in_skull] = 0.0
    hidden_c[~in_skull] = 0.0
    sound_speed, density, attenuation = acoustic_properties(labels, porosity, hidden_c, params)

    cbv_baseline = sample_volume(head.cbv_baseline, head.grid, points)
    activation = sample_volume(head.activation, head.grid, points)
    velocity = sample_volume(head.blood_velocity, head.grid, points)
    in_plane_axes = np.array([plane.u_lateral, plane.u_depth], dtype=np.float32)
    blood_velocity = np.einsum("ac,czx->azx", in_plane_axes, velocity)
    for values in (cbv_baseline, activation, blood_velocity):
        values[..., ~in_brain] = 0.0

    return PlaneMedium(
        grid=grid,
        plane=plane,
        labels=labels,
        porosity=porosity,
        hidden_c=hidden_c,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        cbv_baseline=cbv_baseline,
        activation=activation,
        blood_velocity=blood_velocity,
        alpha_power=head.alpha_power,
    )
