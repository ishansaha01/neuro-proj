"""Clean CBV images for the diffusion prior, drawn from the head model.

The head model is the single source of truth: every image is
``head.cbv.cbv_at(cbv_baseline, activation, a)`` sampled by
``head.plane.sample_plane`` on the reconstruction pixel grid of one imaging
plane. There is no separate 2D image generator.

Randomness (ROADMAP D-004). Head ``h`` is generated from the subtree
``tree.child(h)``; the plane, the task state and the amplitude of image
``(h, p)`` come from the stream ``("dataset_sweep", h, p)``. Any image can
therefore be regenerated alone with `sample_cbv_image`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.config import field
from neurosim.common.grid import Grid2D, Grid3D, PlaneSpec, plane_to_world
from neurosim.common.seeding import SeedTree
from neurosim.head import generate as head_generate
from neurosim.head.cbv import cbv_at
from neurosim.head.config import GridConfig, HeadConfig
from neurosim.head.generate import GeneratedHead, generate_head
from neurosim.head.plane import default_plane, sample_plane

# Wall time of one `sample_plane` call on a 32 x 32 pixel grid, single-threaded on the
# 2-core development machine (2026-11-17): 2.4-3.2 ms measured; scaled by the pixel count.
_SECONDS_PER_PIXEL = 4.0e-6
# Bytes held per image: the float32 image plus amplitude, flags and indices.
_BYTES_PER_IMAGE_OVERHEAD = 16


def _coarse_head() -> HeadConfig:
    """Default head of the prior data set: a 60 mm box at 1 mm (2.2e5 voxels, about 0.4 s).

    The box is centred 25.5 mm below the scalp, the depth of the centre of the default
    pixel grid (12 mm top row + 15.5 mm, minus the 2 mm probe standoff), so the grid can
    be rotated freely about the depth axis inside it.
    """
    return HeadConfig(
        grid=GridConfig(
            spacing=1.0e-3, region="box", box_size=(0.06, 0.06, 0.06), box_center_depth=0.0255
        )
    )


@dataclass(frozen=True)
class PixelGridConfig:
    """Square reconstruction pixel grid in the imaging plane (rows are depth)."""

    n: int = field(32, ge=2, doc="Pixels per side; the image is n x n.")
    spacing: float = field(
        1.0e-3,
        gt=0.0,
        doc="Pixel size [m]. 1 mm is about 1.3 wavelengths at 2 MHz and matches the "
        "voxel size of the coarse head; assumed, to verify against the fUS pixel grid.",
    )
    top_depth: float = field(
        12.0e-3,
        gt=0.0,
        doc="Depth of the top pixel row below the array face [m]. With a 2 mm standoff, "
        "a 5 mm scalp, a 6.5 mm skull and 3 mm of CSF (head defaults) the cortex starts "
        "near 15 mm, so 12 mm keeps the brain surface inside the image for thin and "
        "thick skulls; assumed.",
    )
    lateral_center: float = field(
        0.0, doc="Lateral coordinate of the image centre in the plane frame [m] (0 = array centre)."
    )


@dataclass(frozen=True)
class PriorDataConfig:
    """Data set of clean CBV images for the diffusion prior."""

    head: HeadConfig = field(
        default_factory=_coarse_head,
        doc="Head model. A coarse box keeps one head well under a second.",
    )
    pixel_grid: PixelGridConfig = field(
        default_factory=PixelGridConfig, doc="Pixel grid of every image."
    )
    n_heads: int = field(480, ge=1, doc="Number of heads (head indices 0 .. n_heads - 1).")
    planes_per_head: int = field(10, ge=1, doc="Imaging planes (images) drawn per head.")
    active_fraction: float = field(
        0.5,
        ge=0.0,
        le=1.0,
        doc="Probability that an image is in the task state (a > 0); the rest are at rest.",
    )
    amplitude_range: tuple[float, float] = field(
        (0.05, 0.5),
        ge=0.0,
        doc="Range of the fractional CBV change `a` of an active image (uniform). fUS "
        "reports 10-50 % in an activated region (Iconeus, quoted in head.config.TaskConfig); "
        "the lower end is extended to 5 % to cover the rise and fall of the response; "
        "assumed, to verify.",
    )
    lateral_jitter: float = field(
        6.0e-3,
        ge=0.0,
        doc="Largest lateral shift of the probe along the array axis [m] (uniform, "
        "narrowed where the head box requires it); assumed.",
    )
    heldout_every: int = field(
        10,
        ge=2,
        doc="One head in every `heldout_every` is held out: those with "
        "index % heldout_every == heldout_every - 1.",
    )

    def __post_init__(self) -> None:
        low, high = self.amplitude_range
        if high < low:
            raise ValueError(f"amplitude_range must be increasing, got {self.amplitude_range}")


def pixel_grid(cfg: PixelGridConfig) -> Grid2D:
    """The pixel grid of the images in the imaging-plane frame."""
    half_width = 0.5 * (cfg.n - 1) * cfg.spacing
    return Grid2D(
        shape=(cfg.n, cfg.n),
        spacing=cfg.spacing,
        origin=(cfg.top_depth, cfg.lateral_center - half_width),
    )


def is_heldout(head_index: np.ndarray, heldout_every: int) -> np.ndarray:
    """Deterministic split by head: True for images of held-out heads.

    head_index: head index of each image, integer array of any shape.
    heldout_every: `PriorDataConfig.heldout_every`.

    A head is on one side only, so no anatomy or vessel texture is shared between
    the training and the held-out images.
    """
    return np.asarray(head_index) % heldout_every == heldout_every - 1


def rotated_plane(plane: PlaneSpec, angle: float) -> PlaneSpec:
    """Rotate the array about the depth axis of a plane.

    plane: the starting plane.
    angle: rotation of `u_lateral` about `u_depth` [rad].
    """
    u_lateral = np.asarray(plane.u_lateral)
    u_depth = np.asarray(plane.u_depth)
    rotated = np.cos(angle) * u_lateral + np.sin(angle) * np.cross(u_depth, u_lateral)
    rotated /= np.linalg.norm(rotated)
    return PlaneSpec(origin=plane.origin, u_lateral=tuple(rotated), u_depth=plane.u_depth)


def lateral_shift_limits(head_grid: Grid3D, plane: PlaneSpec, grid: Grid2D) -> tuple[float, float]:
    """Lateral probe shifts that keep every pixel inside the head volume.

    head_grid: voxel grid of the head.
    plane: imaging plane before the shift.
    grid: pixel grid in the plane.

    Returns ``(lowest, highest)`` shift of the plane origin along `u_lateral` [m] for
    which all pixel centres lie between the outermost voxel centres of `head_grid`
    (so `sample_plane` interpolates and never meets its out-of-volume error). The
    pixel grid is a rectangle and the head grid a box, so the four corner pixels
    decide. The interval is empty (``lowest > highest``) when the box cannot hold
    the grid at any shift.
    """
    z, x = grid.axes()
    corners = plane_to_world(plane, [(x[0], z[0]), (x[0], z[-1]), (x[-1], z[0]), (x[-1], z[-1])])
    box_low = np.asarray(head_grid.origin)
    box_high = box_low + (np.asarray(head_grid.shape) - 1) * head_grid.spacing
    lowest, highest = -np.inf, np.inf
    for axis, direction in enumerate(plane.u_lateral):
        room_low = box_low[axis] - corners[:, axis]
        room_high = box_high[axis] - corners[:, axis]
        if direction == 0.0:
            if np.any(room_low > 0.0) or np.any(room_high < 0.0):
                return np.inf, -np.inf
            continue
        bounds = np.sort(np.stack((room_low, room_high)) / direction, axis=0)
        lowest = max(lowest, float(bounds[0].max()))
        highest = min(highest, float(bounds[1].min()))
    return lowest, highest


def prior_head(cfg: PriorDataConfig, tree: SeedTree, head_index: int) -> GeneratedHead:
    """Head number `head_index` of the data set, generated from ``tree.child(head_index)``."""
    return generate_head(cfg.head, tree.child(head_index))


def _draw_image(
    cfg: PriorDataConfig, head: GeneratedHead, base_plane: PlaneSpec, rng: np.random.Generator
) -> tuple[np.ndarray, float, bool]:
    """One image of a head: ``(image (n, n) float32, amplitude, active)``.

    The four draws (angle, shift, task state, amplitude) are always made in this
    order, so the stream does not depend on the outcome of any of them.
    """
    grid = pixel_grid(cfg.pixel_grid)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    shift_fraction = rng.uniform()
    active = bool(rng.uniform() < cfg.active_fraction)
    amplitude = float(rng.uniform(*cfg.amplitude_range)) if active else 0.0

    plane = rotated_plane(base_plane, angle)
    lowest, highest = lateral_shift_limits(head.grid, plane, grid)
    lowest = max(lowest, -cfg.lateral_jitter)
    highest = min(highest, cfg.lateral_jitter)
    if lowest > highest:
        raise ValueError(
            "the head box cannot hold the pixel grid at this probe orientation; enlarge "
            "head.grid.box_size or adjust head.grid.box_center_depth"
        )
    shift = lowest + shift_fraction * (highest - lowest)
    origin = np.asarray(plane.origin) + shift * np.asarray(plane.u_lateral)
    plane = PlaneSpec(origin=tuple(origin), u_lateral=plane.u_lateral, u_depth=plane.u_depth)

    medium = sample_plane(head, plane, grid)
    return cbv_at(medium.cbv_baseline, medium.activation, amplitude), amplitude, active


def _head_images(
    cfg: PriorDataConfig, tree: SeedTree, head_index: int, plane_indices: range
) -> list[tuple[np.ndarray, float, bool]]:
    head = prior_head(cfg, tree, head_index)
    base_plane = default_plane(head, cfg.head)
    return [
        _draw_image(cfg, head, base_plane, tree.rng("dataset_sweep", head_index, plane_index))
        for plane_index in plane_indices
    ]


def sample_cbv_image(
    cfg: PriorDataConfig, tree: SeedTree, head_index: int, plane_index: int
) -> tuple[np.ndarray, float, bool]:
    """Regenerate one image of the data set alone.

    Returns ``(image, amplitude, active)``: the CBV fraction on the pixel grid,
    float32, shape (n, n), indexed [iz, ix]; the fractional activation `a`
    (0 at rest); and the task state. Identical to entry
    ``head_index * cfg.planes_per_head + plane_index`` of `sample_cbv_images`.
    """
    return _head_images(cfg, tree, head_index, range(plane_index, plane_index + 1))[0]


def sample_cbv_images(cfg: PriorDataConfig, tree: SeedTree) -> dict[str, np.ndarray]:
    """Generate the whole data set, head by head.

    Streams used: the head streams of ``tree.child(h)`` and ``("dataset_sweep", h, p)``.

    Returns a dict of arrays with ``N = cfg.n_heads * cfg.planes_per_head`` entries,
    ordered by head, then plane:
        images: CBV fraction (0..1) on the pixel grid, float32, (N, n, n), indexed [iz, ix].
        amplitude: fractional CBV change `a` at the activation centre, float32, (N,);
            0 for rest images.
        active: task state, bool, (N,).
        head_index: head of each image, int32, (N,).
        plane_index: plane of each image within its head, int32, (N,).
        heldout: `is_heldout(head_index, cfg.heldout_every)`, bool, (N,).
    """
    images, amplitudes, actives = [], [], []
    for head_index in range(cfg.n_heads):
        for image, amplitude, active in _head_images(
            cfg, tree, head_index, range(cfg.planes_per_head)
        ):
            images.append(image)
            amplitudes.append(amplitude)
            actives.append(active)
    head_index = np.repeat(np.arange(cfg.n_heads, dtype=np.int32), cfg.planes_per_head)
    return {
        "images": np.stack(images).astype(np.float32),
        "amplitude": np.asarray(amplitudes, dtype=np.float32),
        "active": np.asarray(actives, dtype=bool),
        "head_index": head_index,
        "plane_index": np.tile(np.arange(cfg.planes_per_head, dtype=np.int32), cfg.n_heads),
        "heldout": is_heldout(head_index, cfg.heldout_every),
    }


def estimate_runtime_seconds(cfg: PriorDataConfig) -> float:
    """Rough single-threaded run time of `sample_cbv_images` [s]."""
    per_head = head_generate.estimate_runtime_seconds(cfg.head)
    per_plane = _SECONDS_PER_PIXEL * cfg.pixel_grid.n**2
    return cfg.n_heads * (per_head + cfg.planes_per_head * per_plane)


def estimate_memory_bytes(cfg: PriorDataConfig) -> int:
    """Estimated peak memory of `sample_cbv_images` [bytes]: one head plus all images."""
    n_images = cfg.n_heads * cfg.planes_per_head
    per_image = 4 * cfg.pixel_grid.n**2 + _BYTES_PER_IMAGE_OVERHEAD
    # The list of images and the stacked array exist together at the end.
    return head_generate.estimate_memory_bytes(cfg.head) + 2 * n_images * per_image
