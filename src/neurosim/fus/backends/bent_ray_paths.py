"""Refracted ray paths of the `bent_ray` backend, and the integrals taken along them.

A first-arrival ray is the curve that follows `-grad T` from the field point
back to the source. `integrate_paths` traces every (source, point) ray of an
`EikonalField` in lock step and accumulates along it

    travel time     integral ds / c                     (bilinear slowness map)
    absorption      integral alpha0 ds                   (bilinear alpha0 map)
    transmission    product of interface coefficients   (nearest-neighbour impedance)
    spreading       integral c ds / cos(theta)^2        (nearest-neighbour sound speed)

Travel time. The traced curve joins the two end points, so by Fermat's
principle its time is an upper bound on the first arrival, and its excess is
second order in the distance between the curve and the true ray. The integral
uses the bilinear slowness map of the straight-ray backend; the grid value of
`T` it was traced from is first order, and the two are compared in `bank.meta`.

Tracing. With step `l` (`BentRayConfig.step_fraction` cells) the ray moves by
`-l g(m)`, `g` the unit vector along `grad T = grad T0 + grad u` (analytic
plus central differences of the factored unknown, bilinear in between) at the
predicted midpoint `m` of the step. All integrands are read at `m` (midpoint
rule). Within `_CAPTURE_CELLS` cells of the source the ray goes straight to
it, and stops there. A ray that has not arrived after the longest path its
travel time allows is also sent straight to the source; it remains a valid
path, and such rays are counted.

Obliquity. The amplitude model treats the medium along the ray as locally
plane-layered: at every sample the layers are normal to `n`, and `theta` is
the angle between the ray and `n`. `n` is the orientation of the strongest
slowness gradients nearby, the principal axis of the structure tensor of the
slowness map (the smoothed outer product of its gradient; Bigun & Granlund
1987, Proc. 1st Int. Conf. Computer Vision, 433-438) taken over
`BentRayConfig.layer_normal_scale`. The scalp-bone and bone-brain surfaces
dominate it throughout the skull, so `n` follows the curved skull and ignores
both the pixel staircase and the texture inside the bone. Where no gradient is
near (`_NORMAL_ENERGY_FRACTION`), a sample keeps the normal of the last
interface region its ray passed, and the stretch between the field point and
the first such region takes the normal met there. At a sample the cosine is
`|g . n|`.

Interface transmission. As in the straight-ray backend the impedance is read
with nearest-neighbour lookup and every change between successive samples is
one interface, with the flux-normalised coefficient

    t = 2 sqrt(Y1 Y2) / (Y1 + Y2),    Y = rho c / cos(theta).

This is the fluid-fluid pressure coefficient `2 Y2 / (Y1 + Y2)` times
`sqrt(Y1 / Y2)`, so that `t^2` is the transmitted fraction of the energy flux
through the interface; at normal incidence it is the coefficient of `ray.py`.
The two cosines follow from Snell's law: the slowness component along the
layers, `p_t = sin(theta) / c`, is the same on both sides, so `cos(theta_i) =
sqrt(1 - (p_t c_i)^2)` with each side's sound speed, and beyond the critical
angle the cosine of the fast side is 0 and `t = 0`. `p_t` is read from the
direction of the traced ray `_ANGLE_LAG_CELLS` cells before and after the
interface, and the smaller of the two values is used: within about two cells
of an interface the direction of the numerical `grad T` is a blend of the two
sides, and near the critical angle a 1 % error of `p_t` decides between a
small coefficient and none, so the interface is only taken as totally
reflecting when the ray on both sides says so. This holds for an
interface that follows the layering. The gradual variation of the sound speed
inside the bone does not: it has no orientation of its own, and near-grazing
rays would be reflected by changes of a few per cent along the layers. The
refracted cosine therefore enters with a weight that rises smoothly from 0 to
1 as the relative sound-speed step between the two samples goes from 0 to
`BentRayConfig.interface_contrast`; a step with weight 0 has equal cosines on
both sides, that is, its normal-incidence coefficient. An oblique ray can clip
a corner of a staircase interface and see slow-fast-slow-fast where there is
one interface; the sound speed and density sequences therefore pass a
three-sample median along the ray, which removes single-sample excursions and
leaves a step a step (at normal incidence through layers of two samples or
more the product equals that of `ray.py`).

Spreading. Through plane layers the ray tube of a line source has, at the
field point, the width per unit take-off angle

    J = cos(theta_e) cos(theta_r) / c_e * integral c ds / cos(theta)^2

(from `x(p) = sum d_i tan(theta_i)` with `sin(theta_i) = p c_i`; indices e, r:
element and field point). `integrate_paths` returns the integral and the two
end cosines.

Path-length histograms (optional, `PathBins`). Every node of the grid may carry
weights on a few bins (for example the two neighbouring knots of its CT number,
or of its lateral position). The tracer then also returns, per ray and bin,

    bin_path[b] = integral of (bilinear interpolant of the node map W_b) ds

with `W_b` the node weights on bin `b`, taken with the same midpoint samples as
the travel time. For any node map that is a combination `sum_b v_b W_b` of the
bins, its integral along the traced ray is `sum_b v_b bin_path[b]`: the path
integral of a whole family of slowness or absorption maps from one trace (the
first-order travel-time perturbation of Fermat's principle, with the ray fixed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.eikonal import (
    BilinearStencil,
    EikonalField,
    bilinear_stencil,
    gather_bilinear,
)

# Radius around a source inside which a ray is closed with a straight segment [grid cells]:
# just outside the nodes the eikonal solver sets analytically (1.5 cells).
_CAPTURE_CELLS = 2.0
# Rays are given this multiple of the longest path their travel time allows (travel time
# times the highest sound speed) before they are sent straight to the source.
_PATH_LENGTH_MARGIN = 1.25
# Parked rays are removed from the working set once they make up this fraction of it.
_COMPRESS_FRACTION = 0.25
# Standard deviation of the Gaussian applied to the slowness map before its gradient
# enters the structure tensor [grid cells]: the usual one-cell derivative scale.
_GRADIENT_SMOOTHING_CELLS = 1.0
# A pixel has a layer normal of its own where the trace of the structure tensor exceeds
# this fraction of its largest value on the grid: within about 3.7 integration scales of
# the strongest interface.
_NORMAL_ENERGY_FRACTION = 1.0e-3
# Distance from an interface, along the ray, at which the angle of incidence is read
# [grid cells]: beyond the two cells over which central differences and bilinear
# interpolation blend the ray directions of the two sides.
_ANGLE_LAG_CELLS = 4.0
# Stands in for zero when dividing.
_TINY = 1.0e-300


@dataclass(frozen=True)
class PathBins:
    """Sparse node weights of the path-length histograms (module docstring).

    Node `n` (row-major index of the medium grid) contributes `weight[n, m]` to
    bin `index[n, m]`, for each of its `m` entries; an entry with weight 0 is
    skipped.

    Attributes:
        index: bin of every entry, int64, shape (nz * nx, n_entries), values in [0, n_bins).
        weight: weight of every entry [dimensionless], float64, shape (nz * nx, n_entries).
        n_bins: number of bins.
    """

    index: np.ndarray
    weight: np.ndarray
    n_bins: int

    def __post_init__(self) -> None:
        if self.index.shape != self.weight.shape or self.index.ndim != 2:
            raise ValueError(
                f"index {self.index.shape} and weight {self.weight.shape} must share one "
                "shape (n_nodes, n_entries)"
            )
        if self.index.size and (self.index.min() < 0 or self.index.max() >= self.n_bins):
            raise ValueError(f"bin indices must lie in [0, {self.n_bins})")


@dataclass(frozen=True)
class MediumMaps:
    """What the path integrals read from a plane medium.

    Attributes:
        grid: the medium grid.
        integrands: slowness [s/m] and alpha0 [dB/(cm MHz^y)] at the nodes,
            float64, shape (nz, nx, 2); read with bilinear interpolation.
        pixels: per pixel (row-major): sound speed [m/s], density [kg/m^3] and the
            (x, z) components of the unit layer normal (zero where the pixel has
            none), float64, shape (nz * nx, 4); read with nearest-neighbour lookup.
        max_sound_speed: highest sound speed of the medium [m/s].
        bins: node weights of the path-length histograms, or None for none.
    """

    grid: Grid2D
    integrands: np.ndarray
    pixels: np.ndarray
    max_sound_speed: float
    bins: PathBins | None = None


def layer_normals(slowness: np.ndarray, spacing: float, scale: float) -> np.ndarray:
    """Orientation of the layering of a slowness map (module docstring).

    Args:
        slowness: [s/m], shape (nz, nx).
        spacing: grid spacing [m].
        scale: integration scale of the structure tensor [m].

    Returns:
        (x, z) components of a unit vector normal to the layers (defined up to
        its sign), zero where no slowness gradient is near; shape (nz, nx, 2).
    """
    smooth = ndimage.gaussian_filter(slowness, _GRADIENT_SMOOTHING_CELLS, mode="nearest")
    gradient_z, gradient_x = np.gradient(smooth, spacing)
    sigma = scale / spacing
    tensor_xx = ndimage.gaussian_filter(gradient_x * gradient_x, sigma, mode="nearest")
    tensor_xz = ndimage.gaussian_filter(gradient_x * gradient_z, sigma, mode="nearest")
    tensor_zz = ndimage.gaussian_filter(gradient_z * gradient_z, sigma, mode="nearest")
    angle = 0.5 * np.arctan2(2.0 * tensor_xz, tensor_xx - tensor_zz)  # principal axis, from +x
    energy = tensor_xx + tensor_zz
    defined = energy > _NORMAL_ENERGY_FRACTION * energy.max()
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1) * defined[:, :, None]


def medium_maps(
    grid: Grid2D,
    sound_speed: np.ndarray,
    density: np.ndarray,
    attenuation: np.ndarray,
    layer_normal_scale: float,
    bins: PathBins | None = None,
) -> MediumMaps:
    """Prepare the lookup tables of a plane medium.

    Args:
        grid: the medium grid.
        sound_speed: [m/s], shape (nz, nx).
        density: [kg/m^3], shape (nz, nx).
        attenuation: alpha0 [dB/(cm MHz^y)], shape (nz, nx).
        layer_normal_scale: integration scale of the layer normals [m].
        bins: node weights of the path-length histograms (one row per node of
            `grid`, row-major), or None.
    """
    if bins is not None and bins.index.shape[0] != grid.shape[0] * grid.shape[1]:
        raise ValueError(
            f"bins describe {bins.index.shape[0]} nodes, the grid has "
            f"{grid.shape[0] * grid.shape[1]}"
        )
    speed = sound_speed.astype(np.float64)
    slowness = 1.0 / speed
    normals = layer_normals(slowness, grid.spacing, layer_normal_scale)
    pixels = np.concatenate(
        (np.stack((speed, density.astype(np.float64)), axis=-1), normals), axis=-1
    ).reshape(-1, 4)
    integrands = np.stack((slowness, attenuation.astype(np.float64)), axis=-1)
    return MediumMaps(grid, integrands, pixels, float(speed.max()), bins)


@dataclass(frozen=True)
class PathIntegrals:
    """Integrals along the refracted ray of every (source, point) pair.

    Every array has shape (n_sources, n_points).

    Attributes:
        travel_time: integral of ds / c along the traced ray [s].
        alpha0_path: integral of alpha0 ds [dB/(cm MHz^y) * m].
        transmission: product of the interface coefficients with the angle of
            incidence [dimensionless, <= 1]; 0 where an interface is met beyond
            its critical angle.
        normal_transmission: the same product with every interface taken at normal
            incidence (the model of `ray.py`), for diagnostics.
        speed_path: integral of c ds [m^2/s]; the spreading integral at normal incidence.
        spread_path: integral of c ds / cos(theta)^2 [m^2/s]; very large where the ray
            runs along the layers.
        end_cosines: cos(theta) at the source times cos(theta) at the field point.
        min_cosine: smallest cos(theta) along the ray.
        forced: True where the ray was sent straight to the source before it
            came within the capture radius.
        bin_path: path-length histograms [m], float32, shape (n_sources, n_points,
            n_bins) (module docstring); None when the maps carry no bins.
    """

    travel_time: np.ndarray
    alpha0_path: np.ndarray
    transmission: np.ndarray
    normal_transmission: np.ndarray
    speed_path: np.ndarray
    spread_path: np.ndarray
    end_cosines: np.ndarray
    min_cosine: np.ndarray
    forced: np.ndarray
    bin_path: np.ndarray | None = None


def _nearest_pixel(stencil: BilinearStencil) -> np.ndarray:
    """Flat index of the pixel whose centre is nearest to each point of a stencil."""
    return (
        stencil.corner
        + (stencil.weight_z >= 0.5) * stencil.row_stride
        + (stencil.weight_x >= 0.5)
    )


def _median(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    """Element-wise median of three arrays."""
    return np.maximum(np.minimum(first, second), np.minimum(np.maximum(first, second), third))


def _smoothstep(fraction: np.ndarray) -> np.ndarray:
    """`3 x^2 - 2 x^3` of the argument clipped to [0, 1]."""
    clipped = np.clip(fraction, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def interface_transmission_squared(
    speed: tuple[np.ndarray, np.ndarray],
    density: tuple[np.ndarray, np.ndarray],
    tangential_slowness: np.ndarray,
    interface_contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flux transmission of the interface between two successive samples of a ray.

    Args:
        speed: sound speed c [m/s] of the two samples, each of shape (n,).
        density: [kg/m^3] of the two samples, each of shape (n,).
        tangential_slowness: slowness component along the layers, `p_t` [s/m], shape (n,).
        interface_contrast: relative sound-speed step at and above which the
            refracted angle follows Snell's law in full [dimensionless].

    Returns:
        (`t^2` with the angle of incidence, `t^2` at normal incidence), each of
        shape (n,) and equal to 1 where the two samples have the same properties.
    """
    slow, fast = np.minimum(*speed), np.maximum(*speed)
    cosine_slow = np.sqrt(np.maximum(1.0 - (tangential_slowness * slow) ** 2, 0.0))
    refracted = np.sqrt(np.maximum(1.0 - (tangential_slowness * fast) ** 2, 0.0))
    weight = _smoothstep((fast - slow) / (slow * interface_contrast))
    cosine_fast = cosine_slow + weight * (refracted - cosine_slow)
    first_is_slow = speed[0] <= speed[1]
    cosine = (
        np.where(first_is_slow, cosine_slow, cosine_fast),
        np.where(first_is_slow, cosine_fast, cosine_slow),
    )
    impedance = [c * rho for c, rho in zip(speed, density)]
    crossed = (impedance[1] * cosine[0], impedance[0] * cosine[1])
    denominator = (crossed[0] + crossed[1]) ** 2
    oblique = np.where(
        denominator > 0.0, 4.0 * crossed[0] * crossed[1] / np.maximum(denominator, _TINY), 1.0
    )
    at_normal = 4.0 * impedance[0] * impedance[1] / (impedance[0] + impedance[1]) ** 2
    return oblique, at_normal


class _RayState:
    """Working set of the rays that are still being traced (arrays of equal length)."""

    def __init__(self, **arrays: np.ndarray) -> None:
        self.__dict__.update(arrays)

    def compress(self, keep: np.ndarray) -> None:
        for name, values in self.__dict__.items():
            self.__dict__[name] = values[keep]


def _accumulate_sample(
    state: _RayState,
    length: np.ndarray,
    integrands: np.ndarray,
    pixel: np.ndarray,
    sample: int,
    lag: int,
    interface_contrast: float,
) -> None:
    """Add one sample of every ray to its integrals and to its transmission product.

    The interface coefficients run `lag + 1` samples behind: the three-sample
    median of sound speed and density is known one sample late, and the
    interface between two filtered samples needs the direction of the ray
    `lag` samples before and after it. The histories are ring buffers over the
    sample number.

    Args:
        state: the rays; `state.direction` is the unit vector along grad T at the sample.
        length: path length the sample stands for [m], shape (n,).
        integrands: slowness [s/m] and alpha0 at the sample, shape (n, 2).
        pixel: row of `MediumMaps.pixels` nearest to the sample, shape (n, 4).
        sample: number of this sample along the ray (0 for the first step).
        lag: samples between an interface and the samples that give its `p_t`.
        interface_contrast: see `interface_transmission_squared`.
    """
    speed, density = pixel[:, 0], pixel[:, 1]
    own_normal = (pixel[:, 2] != 0.0) | (pixel[:, 3] != 0.0)
    normal = np.where(own_normal[:, None], pixel[:, 2:], state.normal)
    sine = np.abs(state.direction[:, 0] * normal[:, 1] - state.direction[:, 1] * normal[:, 0])
    cosine_squared = np.maximum(1.0 - sine * sine, 0.0)
    cosine = np.sqrt(cosine_squared)

    state.travel_time += length * integrands[:, 0]
    state.alpha0_path += length * integrands[:, 1]
    state.speed_path += length * speed
    # Until a ray meets its first layer normal, its stretch waits for that normal.
    first = own_normal & ~state.has_normal
    state.has_normal |= own_normal
    weight = 1.0 / np.maximum(cosine_squared, _TINY)
    state.spread_path += np.where(first, state.pending_path * weight, 0.0)
    state.pending_path = np.where(state.has_normal, 0.0, state.pending_path + length * speed)
    state.spread_path += np.where(state.has_normal, length * speed * weight, 0.0)
    state.cosine_at_point = np.where(first, cosine, state.cosine_at_point)
    state.cosine_last = cosine
    state.min_cosine = np.minimum(state.min_cosine, cosine)
    state.normal = normal

    ring = state.tangential_history.shape[1]
    state.tangential_history[:, sample % ring] = sine / speed
    state.speed_history[:, (sample - 1) % ring] = _median(
        state.speed_before, state.speed_last, speed
    )
    state.density_history[:, (sample - 1) % ring] = _median(
        state.density_before, state.density_last, density
    )
    state.speed_before, state.density_before = state.speed_last, state.density_last
    state.speed_last, state.density_last = speed, density
    # Interface between the filtered samples `second - 1` and `second`.
    second = sample - 1 - lag
    speeds = tuple(state.speed_history[:, (second + k) % ring] for k in (-1, 0))
    densities = tuple(state.density_history[:, (second + k) % ring] for k in (-1, 0))
    tangential = np.minimum(
        state.tangential_history[:, (second - 1 - lag) % ring],
        state.tangential_history[:, (second + lag) % ring],
    )
    oblique, at_normal = interface_transmission_squared(
        speeds, densities, tangential, interface_contrast
    )
    state.transmission_squared *= oblique
    state.normal_transmission_squared *= at_normal


# An entry column of `PathBins` in which more than this fraction of the nodes has a weight
# is added for every ray; a sparser one only for the rays whose node has a weight.
_DENSE_ENTRY_FRACTION = 0.5


def _bin_columns(bins: PathBins) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Contiguous `(index, weight, dense)` columns of the entries of `bins`."""
    return [
        (
            np.ascontiguousarray(bins.index[:, entry]),
            np.ascontiguousarray(bins.weight[:, entry]),
            bool(np.count_nonzero(bins.weight[:, entry]) > _DENSE_ENTRY_FRACTION * len(bins.index)),
        )
        for entry in range(bins.index.shape[1])
    ]


def _accumulate_bins(
    histogram: np.ndarray,
    n_bins: int,
    ray: np.ndarray,
    length: np.ndarray,
    stencil: BilinearStencil,
    columns: list[tuple[np.ndarray, np.ndarray, bool]],
) -> None:
    """Add one sample of every ray to its path-length histograms.

    The bilinear interpolant of the node map of bin `b` at the sample is the sum
    over the four corners of the cell of `corner weight * node weight on b`.

    Args:
        histogram: flat view of the (n_rays, n_bins) histograms [m], updated in place.
        n_bins: number of bins.
        ray: number of each ray of the working set, shape (n,).
        length: path length the sample stands for [m], shape (n,).
        stencil: bilinear stencil of the samples.
        columns: node weights, from `_bin_columns`.
    """
    row = ray * n_bins
    weight_z, weight_x = stencil.weight_z, stencil.weight_x
    corners = (
        (0, (1.0 - weight_z) * (1.0 - weight_x)),
        (1, (1.0 - weight_z) * weight_x),
        (stencil.row_stride, weight_z * (1.0 - weight_x)),
        (stencil.row_stride + 1, weight_z * weight_x),
    )
    # Every statement below adds at most one value per ray, so its flat indices are unique.
    for offset, corner_weight in corners:
        node = stencil.corner + offset
        share = length * corner_weight
        for index, weight, dense in columns:
            if dense:
                histogram[row + index.take(node)] += share * weight.take(node)
            else:
                node_weight = weight.take(node)
                used = np.flatnonzero(node_weight)
                histogram[row[used] + index.take(node[used])] += share[used] * node_weight[used]


def integrate_paths(
    field: EikonalField,
    maps: MediumMaps,
    points_xz: np.ndarray,
    step: float,
    interface_contrast: float,
) -> PathIntegrals:
    """Trace every (source, point) ray of an eikonal field and integrate along it.

    Args:
        field: converged eikonal field of a batch of sources on `maps.grid`.
        maps: lookup tables of the medium.
        points_xz: field points (x, z) [m], shape (n_points, 2), inside the grid.
        step: ray step [m].
        interface_contrast: relative sound-speed step between successive samples at
            and above which an interface refracts in full (module docstring).

    Returns:
        `PathIntegrals` with arrays of shape (n_sources, n_points).
    """
    grid = maps.grid
    n_sources, n_points = field.sources_xz.shape[0], points_xz.shape[0]
    n_rays = n_sources * n_points
    z, x = grid.axes()
    lowest = np.array([x[0], z[0]])
    highest = np.array([x[-1], z[-1]])
    capture = _CAPTURE_CELLS * grid.spacing
    gradient_z, gradient_x = np.gradient(field.excess_time, grid.spacing, axis=(0, 1))
    excess_gradient = np.stack((gradient_x, gradient_z), axis=-1)  # (nz, nx, n_sources, 2)
    del gradient_z, gradient_x

    def read(
        source: np.ndarray, xz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, BilinearStencil]:
        """Unit vector along grad T, the bilinear integrands, the nearest pixel and the
        bilinear stencil at points."""
        stencil = bilinear_stencil(grid, xz)
        gradient = gather_bilinear(excess_gradient, stencil, source)
        offset = xz - field.sources_xz[source]
        distance = np.maximum(np.hypot(offset[:, 0], offset[:, 1]), _TINY)
        gradient += (field.source_slowness[source] / distance)[:, None] * offset
        norm = np.maximum(np.hypot(gradient[:, 0], gradient[:, 1]), _TINY)
        return (
            gradient / norm[:, None],
            gather_bilinear(maps.integrands, stencil),
            maps.pixels.take(_nearest_pixel(stencil), axis=0),
            stencil,
        )

    source = np.repeat(np.arange(n_sources), n_points)
    position = np.tile(np.asarray(points_xz, dtype=np.float64), (n_sources, 1))
    direction, integrands, pixel, _ = read(source, position)
    histogram = None if maps.bins is None else np.zeros(n_rays * maps.bins.n_bins)
    bin_columns = [] if maps.bins is None else _bin_columns(maps.bins)
    longest_path = _PATH_LENGTH_MARGIN * maps.max_sound_speed * field.travel_time(source, position)
    # Slowness along the layers at the field point: sin(theta) / c where its pixel has a normal.
    sine = np.abs(direction[:, 0] * pixel[:, 3] - direction[:, 1] * pixel[:, 2])
    lag = math.ceil(_ANGLE_LAG_CELLS * grid.spacing / step)
    ring = 2 * lag + 3
    state = _RayState(
        ray=np.arange(n_rays),
        source=source,
        position=position,
        direction=direction,
        steps_left=np.ceil(longest_path / step).astype(np.int64),
        parked_steps=np.zeros(n_rays, dtype=np.int64),
        forced=np.zeros(n_rays, dtype=bool),
        travel_time=np.zeros(n_rays),
        alpha0_path=np.zeros(n_rays),
        speed_path=np.zeros(n_rays),
        spread_path=np.zeros(n_rays),
        pending_path=np.zeros(n_rays),
        has_normal=np.zeros(n_rays, dtype=bool),
        normal=np.zeros((n_rays, 2)),
        cosine_at_point=np.ones(n_rays),
        cosine_last=np.ones(n_rays),
        min_cosine=np.ones(n_rays),
        transmission_squared=np.ones(n_rays),
        normal_transmission_squared=np.ones(n_rays),
        # The histories start from the pixel of the field point itself.
        speed_before=pixel[:, 0],
        density_before=pixel[:, 1],
        speed_last=pixel[:, 0],
        density_last=pixel[:, 1],
        speed_history=np.repeat(pixel[:, :1], ring, axis=1),
        density_history=np.repeat(pixel[:, 1:2], ring, axis=1),
        tangential_history=np.repeat((sine / pixel[:, 0])[:, None], ring, axis=1),
    )
    names = (
        "travel_time",
        "alpha0_path",
        "speed_path",
        "spread_path",
        "pending_path",
        "cosine_at_point",
        "cosine_last",
        "min_cosine",
        "transmission_squared",
        "normal_transmission_squared",
        "forced",
    )
    results = {name: np.empty(n_rays, dtype=getattr(state, name).dtype) for name in names}

    n_steps = 0
    while state.ray.size:
        offset = state.position - field.sources_xz[state.source]
        distance = np.hypot(offset[:, 0], offset[:, 1])
        state.forced |= (state.steps_left <= 0) & (distance > capture)
        homing = (distance <= capture) | state.forced
        parked = distance == 0.0
        # Homing rays go straight to the source, by the remaining distance at most.
        length = np.where(homing, np.minimum(step, distance), step)
        towards = offset / np.maximum(distance, _TINY)[:, None]
        lead = np.where((homing & ~parked)[:, None], towards, state.direction)
        midpoint = np.clip(state.position - (0.5 * length)[:, None] * lead, lowest, highest)
        along_gradient, integrands, pixel, stencil = read(state.source, midpoint)
        if histogram is not None:
            _accumulate_bins(
                histogram, maps.bins.n_bins, state.ray, length, stencil, bin_columns
            )
        state.direction = np.where(homing[:, None], lead, along_gradient)
        arrived = homing & (distance <= step)
        advanced = np.clip(state.position - length[:, None] * state.direction, lowest, highest)
        state.position = np.where(arrived[:, None], field.sources_xz[state.source], advanced)
        _accumulate_sample(
            state, length, integrands, pixel, n_steps, lag, interface_contrast
        )
        n_steps += 1
        state.steps_left -= 1
        state.parked_steps += parked

        # A parked ray repeats the sample of its source until the delayed interface
        # coefficients have passed the end of its path.
        done = state.parked_steps >= lag + 2
        if done.sum() >= _COMPRESS_FRACTION * state.ray.size:
            for name, values in results.items():
                values[state.ray[done]] = getattr(state, name)[done]
            state.compress(~done)

    shape = (n_sources, n_points)
    return PathIntegrals(
        travel_time=results["travel_time"].reshape(shape),
        alpha0_path=results["alpha0_path"].reshape(shape),
        transmission=np.sqrt(results["transmission_squared"]).reshape(shape),
        normal_transmission=np.sqrt(results["normal_transmission_squared"]).reshape(shape),
        speed_path=results["speed_path"].reshape(shape),
        spread_path=(results["spread_path"] + results["pending_path"]).reshape(shape),
        end_cosines=(results["cosine_at_point"] * results["cosine_last"]).reshape(shape),
        min_cosine=results["min_cosine"].reshape(shape),
        forced=results["forced"].reshape(shape),
        bin_path=(
            None
            if histogram is None
            else histogram.reshape(*shape, maps.bins.n_bins).astype(np.float32)
        ),
    )
