"""Bent-ray backend: refracted first arrivals from the eikonal equation, with ray amplitudes.

Model. For the first-arrival (refracted) ray from an element centre to a field
point, with L the straight-line distance between them (the path of the
reference medium),

    B = W * S * T * exp(-A(f)) * exp(+i w (tau - L / c_ref))

    tau  = integral ds / c(s)  along the refracted ray       travel time [s]
    A(f) = integral alpha(s, f) ds  along the refracted ray  absorption [Np]
    T    = product over interfaces of 2 sqrt(Y1 Y2) / (Y1 + Y2),  Y = rho c / cos(theta)
    S    = sqrt( rho_r c_r L / (rho_e c_ref J) )
    J    = cos(theta_e) cos(theta_r) / c_e * integral c ds / cos(theta)^2
    W    = smoothstep( (90 deg - theta_max) / taper )

The ray is found in two steps: the eikonal equation `|grad T| = 1 / c` is
solved on the medium grid for every element (`eikonal.py`: factored, first
order, pure numpy), and the ray is traced back from the field point along
`-grad T` (`bent_ray_paths.py`), where tau, A, T and the integral of J are
accumulated. By reciprocity the same ray serves receive.

Amplitude: a locally plane-layered medium. `theta` is the angle between the
refracted ray and the local normal of the layering (the skull surfaces; see
`bent_ray_paths.py`), so the layers may be curved and tilted, but along one ray
they are treated as plane and parallel.

T is the flux-normalised transmission coefficient of `ray.py` with the angle
of incidence from Snell's law; an interface met beyond its critical angle
gives T = 0. The angle enters in full for sound-speed steps of at least
`BentRayConfig.interface_contrast` between successive ray samples and fades
out below, because the gradual variation inside the bone is not layered.

S is the amplitude of a 2D line source seen through the ray tube. `J = dw /
d(phi)` is the width `w` of the tube at the field point per unit take-off
angle `phi` at the element [m/rad]; the energy flux in the tube is conserved,
which gives `|B|^2 = (rho_r c_r / rho_e) T^2 L / (c_ref J)` for the source
normalisation of the package docstring (indices e, r: element, field point).
The expression for J is exact for plane parallel layers. At normal incidence
it is `integral c ds / c_e`, S is the paraxial factor of `ray.py`, and the two
backends agree; in a homogeneous medium `J = L` at any angle. `rho_r / rho_e`
keeps the reciprocity relation of the package docstring.

Beyond the critical angle. A first arrival never crosses an interface beyond
its critical angle. Where the straight line from element to point would, the
refracted ray enters the fast layer nearer the element, below the critical
angle, and runs obliquely through it. As it approaches grazing in the fast
layer, `cos(theta) -> 0` there: T falls as `sqrt(cos)` per interface and `1 /
J` as `cos^3`, so the amplitude falls continuously, as `cos^2.5` through a
fast slab, and not at a cut-off angle (through the 12 mm slab of E-FUS-025 it
is still 0.29 of its normal-incidence value 16 degrees beyond the straight-line
critical angle, in k-Wave and in this model alike). The end of that fall is a
ray that runs along the layers: the grazing limit through a slab, or a head or
creeping wave along an interface of a curved skull. Such rays carry no
amplitude in ray theory, and the window W makes that exact: it is 1 while the
ray stays more than `BentRayConfig.grazing_taper_deg` away from grazing
everywhere (`theta_max` is the largest angle between ray and layer normal
along the path) and falls smoothly to 0 at grazing, which absorbs the scatter
of the numerical ray directions. `bank.meta["fraction_post_critical"]` is the
fraction of pairs with W < 1/2.

Approximations: geometric optics of the first arrival only. Later arrivals
are dropped. This matters behind a curved, heterogeneous skull: where a wave
that ran along the fast bone overtakes the directly transmitted one, the model
follows the earlier, weak arrival and gives the pair little or no amplitude,
although the transmitted wave still carries energy for some distance beyond
that point; the amplitude then ends at an edge instead of fading. Energy that
tunnels through a thin fast layer beyond the critical angle and head waves
are dropped too. The spreading and the interface angles assume plane parallel
layers along each ray: focusing and defocusing by the curvature of the skull
and by lateral variations of the sound speed are not modelled (nor are they in
`ray.py`), and a fast inclusion on the path counts as a layer. No multiple
reflections (`static_fmc` is None), no mode conversion or shear waves, no
caustic phase shifts, point-like elements. Each change of impedance between
successive ray samples is one sharp interface whatever the wavelength.
E-FUS-023 checks the travel times against the analytic refracted path, and
E-FUS-024 and E-FUS-025 the phase and amplitude against k-Wave through a flat
and a gently curved slab; the grazing window W is not exercised by any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.config import to_dict
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.units import db_cm_mhz_to_np_m
from neurosim.fus.backends.bent_ray_paths import integrate_paths, medium_maps
from neurosim.fus.backends.config import BentRayConfig
from neurosim.fus.backends.eikonal import solve_eikonal
from neurosim.fus.backends.geometry import (
    check_geometry,
    check_medium,
    checked_frequencies,
    element_centres,
    inside_grid,
)
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

SOLVER = "factored first-order eikonal, alternating line sweeps (numpy)"
# Working memory per grid node and source of one batch [bytes]: the field of the solver
# and its frozen mask, the two components of the gradient that the rays follow, and the
# temporaries of their construction.
_BYTES_PER_NODE_SOURCE = 48
# Working memory per ray of one batch [bytes]: the state of the tracer and its temporaries.
_BYTES_PER_RAY = 1024


@dataclass(frozen=True)
class BentRayIntegrals:
    """Frequency-independent quantities of every element-to-point refracted ray.

    Every array has shape (n_elements, n_points).

    Attributes:
        path_length: straight-line distance L [m].
        travel_time: integral of ds / c along the refracted ray [s].
        eikonal_time: first-arrival time read from the eikonal grid [s] (first-order
            accurate; kept as a check on `travel_time`).
        alpha0_path: integral of the absorption prefactor alpha0 ds [dB/(cm MHz^y) * m].
        transmission: interface transmission T with the angle of incidence [dimensionless].
        spreading: end-point and ray-tube factor S [dimensionless].
        window: grazing window W [dimensionless, 0 to 1].
        obliquity: `W T S` divided by `T S` of the normal-incidence, paraxial model of
            `ray.py` on the same path [dimensionless]; 1 at normal incidence.
        forced: True where a traced ray had to be closed with a straight segment.
        n_sweeps: line sweeps of the eikonal solver, the largest over the batches.
    """

    path_length: np.ndarray
    travel_time: np.ndarray
    eikonal_time: np.ndarray
    alpha0_path: np.ndarray
    transmission: np.ndarray
    spreading: np.ndarray
    window: np.ndarray
    obliquity: np.ndarray
    forced: np.ndarray
    n_sweeps: int


def grazing_window(min_cosine: np.ndarray, taper_deg: float) -> np.ndarray:
    """Window W that removes rays running along the layers (module docstring).

    Args:
        min_cosine: smallest cosine of the angle between ray and layer normal
            along the path, any shape.
        taper_deg: width of the taper, in degrees from grazing.

    Returns:
        `3 x^2 - 2 x^3` of `x = (angle from grazing) / taper_deg` clipped to [0, 1].
    """
    from_grazing = np.rad2deg(np.arcsin(np.clip(min_cosine, 0.0, 1.0)))
    fraction = np.clip(from_grazing / taper_deg, 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _batch_size(n_nodes: int, n_points: int, n_sources: int, cfg: BentRayConfig) -> int:
    """Sources solved and traced together within `cfg.max_batch_bytes`."""
    per_source = _BYTES_PER_NODE_SOURCE * n_nodes + _BYTES_PER_RAY * n_points
    return int(np.clip(cfg.max_batch_bytes // per_source, 1, n_sources))


def _nearest(values: np.ndarray, grid: Grid2D, xz: np.ndarray) -> np.ndarray:
    """Value of the pixel nearest to each plane point (x, z) [m]; float64, shape (n,)."""
    z0, x0 = grid.origin
    row = np.clip(np.rint((xz[:, 1] - z0) / grid.spacing), 0, grid.shape[0] - 1).astype(int)
    column = np.clip(np.rint((xz[:, 0] - x0) / grid.spacing), 0, grid.shape[1] - 1).astype(int)
    return values[row, column].astype(np.float64)


def trace_bent_rays(
    medium: PlaneMedium, transducer: Transducer, points_xz: np.ndarray, cfg: BentRayConfig
) -> BentRayIntegrals:
    """Find the refracted ray from every element centre to every point.

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3], attenuation
            [dB/(cm MHz^y)] of shape (nz, nx).
        transducer: the array (element centres at z = 0).
        points_xz: field points (x, z) [m], shape (n_points, 2), inside the medium grid.
        cfg: settings of the backend.

    Returns:
        `BentRayIntegrals` with arrays of shape (n_elements, n_points).

    Raises:
        ValueError: if an element centre or a field point lies outside the medium grid.
        RuntimeError: if the eikonal sweeps do not converge (`cfg.max_sweeps`).
    """
    check_medium(medium)
    grid = medium.grid
    points = np.asarray(points_xz, dtype=np.float64)
    sources = element_centres(transducer)
    if not (np.all(inside_grid(grid, points)) and np.all(inside_grid(grid, sources))):
        raise ValueError(
            "every element centre and every field point must lie inside the medium grid"
        )
    maps = medium_maps(
        grid, medium.sound_speed, medium.density, medium.attenuation, cfg.layer_normal_scale
    )
    slowness = np.ascontiguousarray(maps.integrands[:, :, 0])
    step = cfg.step_fraction * grid.spacing
    n_elements, n_points = sources.shape[0], points.shape[0]
    shape = (n_elements, n_points)
    names = (
        "travel_time",
        "alpha0_path",
        "transmission",
        "normal_transmission",
        "speed_path",
        "spread_path",
        "end_cosines",
        "min_cosine",
    )
    out = {name: np.empty(shape) for name in names}
    eikonal_time = np.empty(shape)
    forced = np.empty(shape, dtype=bool)
    n_sweeps = 0

    batch = _batch_size(grid.shape[0] * grid.shape[1], n_points, n_elements, cfg)
    for first in range(0, n_elements, batch):
        chunk = slice(first, first + batch)
        field = solve_eikonal(slowness, grid, sources[chunk], cfg.sweep_tolerance, cfg.max_sweeps)
        n_sweeps = max(n_sweeps, field.n_sweeps)
        n_sources = field.sources_xz.shape[0]
        source = np.repeat(np.arange(n_sources), n_points)
        at_points = field.travel_time(source, np.tile(points, (n_sources, 1)))
        eikonal_time[chunk] = at_points.reshape(n_sources, n_points)
        paths = integrate_paths(field, maps, points, step, cfg.interface_contrast)
        for name in names:
            out[name][chunk] = getattr(paths, name)
        forced[chunk] = paths.forced

    path_length = np.hypot(
        points[None, :, 0] - sources[:, None, 0], points[None, :, 1] - sources[:, None, 1]
    )
    density_ratio = (
        _nearest(medium.density, grid, points)[None, :]
        / _nearest(medium.density, grid, sources)[:, None]
    )
    speed_at_points = _nearest(medium.sound_speed, grid, points)[None, :]
    speed_at_sources = _nearest(medium.sound_speed, grid, sources)[:, None]
    # c_e / J, with J of the layered medium and of normal incidence (`ray.py`); both are
    # zero for a point that coincides with its source, where the path is empty.
    layered = _safe_ratio(np.ones(shape), out["end_cosines"] * out["spread_path"])
    paraxial = _safe_ratio(np.ones(shape), out["speed_path"])
    end_points = density_ratio * speed_at_points * speed_at_sources * path_length / C_REF
    spreading = np.sqrt(end_points * layered)
    window = grazing_window(out["min_cosine"], cfg.grazing_taper_deg)
    reference = out["normal_transmission"] * np.sqrt(end_points * paraxial)
    obliquity = _safe_ratio(window * out["transmission"] * spreading, reference)
    return BentRayIntegrals(
        path_length=path_length,
        travel_time=out["travel_time"],
        eikonal_time=eikonal_time,
        alpha0_path=out["alpha0_path"],
        transmission=out["transmission"],
        spreading=spreading,
        window=window,
        obliquity=obliquity,
        forced=forced,
        n_sweeps=n_sweeps,
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """`numerator / denominator`, 0 where the denominator is not positive."""
    return np.divide(
        numerator, denominator, out=np.zeros(denominator.shape), where=denominator > 0.0
    )


def bent_ray_travel_times(
    medium: PlaneMedium,
    transducer: Transducer,
    points_xz: np.ndarray,
    cfg: BentRayConfig = BentRayConfig(),
) -> np.ndarray:
    """One-way first-arrival (refracted) travel time from every element centre to every point.

    Args:
        medium: plane medium (sound_speed [m/s] on its grid; z = 0 is the array face).
        transducer: the array.
        points_xz: field points (x, z) [m], shape (..., 2), inside the medium grid.
        cfg: settings of the backend.

    Returns:
        travel times [s], float64, shape (n_elements, ...).
    """
    points = np.asarray(points_xz, dtype=np.float64)
    integrals = trace_bent_rays(medium, transducer, points.reshape(-1, 2), cfg)
    return integrals.travel_time.reshape(transducer.n_elements, *points.shape[:-1])


def bent_ray_bank(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: BentRayConfig,
) -> GreensBank:
    """Green's bank of the bent-ray model (module docstring).

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3], attenuation
            [dB/(cm MHz^y)] of shape (nz, nx), and the power-law exponent alpha_power.
        transducer: the array.
        roi: grid the bank is stored on, inside the medium grid, z > 0.
        freqs: frequencies [Hz], shape (nf,), ascending and uniformly spaced.
        cfg: settings of the backend.

    Returns:
        `GreensBank` with B complex64 of shape (nf, n_elements, nz_roi, nx_roi),
        `c_ref = C_REF` and `static_fmc = None`. `meta` holds the solver, the grid
        spacing and ray step [m], the number of eikonal sweeps, the fraction of
        element-point pairs tapered as post-critical (W < 1/2), the fraction whose
        amplitude the obliquity has reduced by more than 20 dB against the
        normal-incidence model of `ray.py` on the same path, the fraction of rays
        closed by force, and the difference between the traced and the grid travel
        times [s] as an accuracy estimate.
    """
    freqs = checked_frequencies(freqs)
    check_medium(medium)
    check_geometry(medium, transducer, roi)
    integrals = trace_bent_rays(medium, transducer, roi.points().reshape(-1, 2), cfg)

    excess_delay = integrals.travel_time - integrals.path_length / C_REF  # [s]
    amplitude = integrals.window * integrals.transmission * integrals.spreading
    # Absorption [Np/m] at each frequency per unit alpha0; the power law is linear in alpha0.
    nepers_per_alpha0 = db_cm_mhz_to_np_m(1.0, freqs, medium.alpha_power)

    n_elements, n_points = amplitude.shape
    bank = np.empty((freqs.size, n_elements, n_points), dtype=np.complex64)
    for element in range(n_elements):
        exponent = (
            2j * np.pi * freqs[:, None] * excess_delay[element]
            - nepers_per_alpha0[:, None] * integrals.alpha0_path[element]
        )
        bank[:, element] = amplitude[element] * np.exp(exponent)

    time_difference = integrals.travel_time - integrals.eikonal_time
    meta = {
        "config": to_dict(cfg),
        "solver": SOLVER,
        "grid_spacing_m": medium.grid.spacing,
        "ray_step_m": cfg.step_fraction * medium.grid.spacing,
        "eikonal_sweeps": integrals.n_sweeps,
        "fraction_post_critical": float(np.mean(integrals.window < 0.5)),
        "fraction_below_minus_20_db": float(np.mean(integrals.obliquity < 0.1)),
        "fraction_forced_closure": float(np.mean(integrals.forced)),
        "travel_time_accuracy": {
            "estimate": "refracted-path integral minus first-order eikonal grid value",
            "rms_s": float(np.sqrt(np.mean(time_difference**2))),
            "max_abs_s": float(np.max(np.abs(time_difference))),
        },
    }
    return GreensBank(
        freqs=freqs,
        roi=roi,
        B=bank.reshape(freqs.size, n_elements, *roi.shape),
        c_ref=C_REF,
        transducer=transducer,
        backend="bent_ray",
        static_fmc=None,
        meta=meta,
    )


def estimate_bent_ray_cost(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: BentRayConfig,
) -> dict[str, float | int]:
    """Cost of `bent_ray_bank`: one eikonal solve per element, no time stepping.

    Args: as for `bent_ray_bank`.

    Returns:
        dict with `grid_points` (the medium grid), `time_steps` (0),
        `n_simulations` (eikonal solves, one per element), `batch_elements`
        (elements solved together), `ray_samples` (upper estimate of the medium
        lookups along all rays), `memory_bytes` (the bank plus the working set of
        one batch) and `run_time_s` (single-core estimate).
    """
    points = roi.points().reshape(-1, 2)
    sources = element_centres(transducer)
    n_nodes = medium.grid.shape[0] * medium.grid.shape[1]
    n_elements, n_points = sources.shape[0], points.shape[0]
    path_length = np.hypot(
        points[None, :, 0] - sources[:, None, 0], points[None, :, 1] - sources[:, None, 1]
    )
    step = cfg.step_fraction * medium.grid.spacing
    # A refracted path is at most a few per cent longer than the straight line.
    ray_samples = int(np.sum(np.ceil(path_length / step) + 2))
    bank_values = np.asarray(freqs).size * path_length.size
    batch = _batch_size(n_nodes, n_points, n_elements, cfg)
    return {
        "grid_points": n_nodes,
        "time_steps": 0,
        "n_simulations": n_elements,
        "batch_elements": batch,
        "ray_samples": ray_samples,
        "memory_bytes": 8 * bank_values
        + batch * (_BYTES_PER_NODE_SOURCE * n_nodes + _BYTES_PER_RAY * n_points)
        + 9 * 8 * path_length.size,
        "run_time_s": cfg.seconds_per_node_solve * n_nodes * n_elements
        + cfg.seconds_per_sample * ray_samples
        + cfg.seconds_per_bank_value * bank_values,
    }
