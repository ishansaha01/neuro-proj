"""The uncertain skull as a family of media built from the CT image of the plane.

A member of the family is a plane medium: the CT image `scene.ct_hu` mapped to
sound speed, density and absorption with the clinical HU map
(`ct.acoustics.hu_to_acoustic`), plus a smooth lateral profile of extra slowness
inside the CT bone (the part of the skull's sound speed a CT cannot see). Its
Green's bank comes from the scene's correction-model backend (`bent_ray`).
Unknown parameters (`SkullParameters`):

- `c_bone`, `c_water`, `hu_bone`: the constants of the HU map that set the sound
  speed, each within the range `ct.config.HuMappingConfig` draws it from (a
  uniform prior; `c_water` is the speed of every pixel at or below 0 HU, that is
  of the coupling water, and the low end of the soft-tissue ramp);
- `lateral`: coefficients of `n_lateral_modes` cosine modes of extra slowness
  [s/m] across the plane, applied to pixels at or above the bone threshold
  (Gaussian prior, `SkullModelConfig.lateral_sound_speed_sigma`);
- `common_amplitude`: one factor on the bank (a transmission loss common to all
  paths). The focus criterion does not depend on it; it is set by a power match.

Delays only. The reconstruction uses a member's travel times and not its
amplitudes: `delay_only` replaces the values of a bank by unit phasors on the
element-point pairs to which the model gives any amplitude. Reason (measured on
scene 0 of the demo dataset, `recon.medium_experiment`): the amplitudes of a bent-ray bank
through the CT medium (interface transmission, spreading, grazing window and the
nominal absorption, about twice the truth, ROADMAP D-014c) predict a two-way
sensitivity that differs from the true one by more than a decade rms across the
brain, and a beamformer normalised by them, or an inversion operator built from
them, is useless whatever the delays; the true skull's own sensitivity varies by
a factor of about 1.5. The density and absorption constants of the HU map are
therefore not parameters, and one common amplitude stands for the skull's loss.

Cheap evaluation (Fermat's principle). Tracing a bent-ray bank takes tens of
seconds, so the rays are traced once, through the nominal CT medium, and the
tracer also returns path-length histograms (`fus.backends.bent_ray_paths.PathBins`):
per element-node ray, the length spent at each knot of a CT-number axis and at
each knot of a lateral axis inside bone. The backend integrates the bilinear
interpolant of the slowness map along the ray; for a map that is a pointwise
function `s(HU)` plus a lateral profile in bone, that integral is

    tau(theta) = tau_nominal + sum_k hu_path[k] (s_theta(h_k) - s_nominal(h_k))
                             + sum_j lateral_path[j] ds_theta(x_j)

exactly for the fixed ray, up to the linear interpolation of `s` between the HU
knots (`SkullModelConfig.hu_knot_spacing`; under 1 ns). What it leaves out is
second order in the perturbation: the ray itself moves (eval E-RECON-011). The
amplitude factors that depend on the ray geometry (interface transmission,
spreading, grazing window, and with them which element-point pairs carry any
amplitude) and the absorption along the path stay at their nominal values.
`model_bank` re-traces a member with the real backend; `fermat_bank` is the
fixed-path prediction of the same bank.

This module reads from a `Scene` only what a scanner has: `ct_hu`,
`medium_grid`, `plane`, `imaging` and the CT and backend settings in `meta`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.config import from_dict
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D, PlaneSpec
from neurosim.common.units import db_cm_mhz_to_np_m
from neurosim.ct.types import HuMapping
from neurosim.data.ct_prior import ct_plane_medium
from neurosim.data.types import Scene
from neurosim.fus.backends import BackendConfig, compute_bank
from neurosim.fus.backends.bent_ray import trace_bent_rays
from neurosim.fus.backends.bent_ray_paths import PathBins
from neurosim.fus.config import imaging_setup
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium
from neurosim.recon.config import SkullModelConfig


@dataclass(frozen=True)
class SkullParameters:
    """One member of the family (module docstring).

    Attributes:
        c_bone: sound speed of dense bone [m/s].
        c_water: sound speed at and below 0 HU [m/s].
        hu_bone: CT number of dense bone [HU].
        lateral: coefficients of the lateral modes of extra slowness in bone [s/m],
            float64, shape (n_lateral_modes,).
        common_amplitude: one-way amplitude factor common to all element-point pairs.
    """

    c_bone: float
    c_water: float
    hu_bone: float
    lateral: np.ndarray
    common_amplitude: float = 1.0


@dataclass(frozen=True)
class SkullFamily:
    """Everything needed to evaluate members of the family for one scene.

    Arrays over element-node pairs have shape (n_elements, n_nodes), nodes being
    the nodes of `roi` in C order.

    Attributes:
        transducer, roi, freqs: array, bank grid and frequency grid [Hz] of the scene.
        grid, plane: fine grid of the CT image and the imaging plane.
        ct_hu: CT image [HU], float32, shape `grid.shape`.
        nominal: the nominal HU map of the scene's CT.
        alpha_power: power-law exponent of the absorption.
        bone_threshold_hu: CT number from which a pixel is bone [HU].
        backend: the correction-model backend (`bent_ray`).
        cfg: settings of the family (knots, modes, prior).
        hu_knots: CT number of every HU knot [HU], shape (n_hu,). The knots below
            the bone threshold come first; the threshold itself is a knot of both
            groups, because the absorption law is discontinuous there.
        lateral_knots: lateral position of every lateral knot [m], shape (n_lateral,).
        lateral_modes: value of every mode at the lateral knots [dimensionless, unit
            rms], shape (n_lateral, n_lateral_modes).
        hu_path: length of every ray at each HU knot [m], float32, (n_elements, n_nodes, n_hu).
        lateral_path: length of every ray in bone at each lateral knot [m], float32,
            (n_elements, n_nodes, n_lateral).
        path_length: straight-line distance L [m].
        travel_time: travel time along the ray through the nominal medium [s].
        alpha0_path: absorption integral of the nominal medium [dB/(cm MHz^y) m].
        amplitude: nominal geometric amplitude `W T S` of `fus.backends.bent_ray`.
    """

    transducer: Transducer
    roi: Grid2D
    freqs: np.ndarray
    grid: Grid2D
    plane: PlaneSpec
    ct_hu: np.ndarray
    nominal: HuMapping
    alpha_power: float
    bone_threshold_hu: float
    backend: BackendConfig
    cfg: SkullModelConfig
    hu_knots: np.ndarray
    lateral_knots: np.ndarray
    lateral_modes: np.ndarray
    hu_path: np.ndarray
    lateral_path: np.ndarray
    path_length: np.ndarray
    travel_time: np.ndarray
    alpha0_path: np.ndarray
    amplitude: np.ndarray

    @property
    def n_lateral_modes(self) -> int:
        """Number of lateral modes."""
        return self.lateral_modes.shape[1]

    def nominal_parameters(self) -> SkullParameters:
        """The member the CT gives without calibration: the nominal map, no lateral profile."""
        return SkullParameters(
            c_bone=self.nominal.c_bone,
            c_water=self.nominal.c_water,
            hu_bone=self.nominal.hu_bone,
            lateral=np.zeros(self.n_lateral_modes),
        )


def hu_knots(hu_max: float, bone_threshold_hu: float, spacing: float) -> np.ndarray:
    """Knots of the CT-number axis [HU]: 0 to the threshold, then the threshold to `hu_max`.

    Both groups are evenly spaced with at most `spacing` between knots, and the
    threshold is the last knot of the first group and the first of the second.
    """
    n_soft = int(np.ceil(bone_threshold_hu / spacing)) + 1
    top = max(hu_max, bone_threshold_hu + spacing)
    n_bone = int(np.ceil((top - bone_threshold_hu) / spacing)) + 1
    return np.concatenate(
        (np.linspace(0.0, bone_threshold_hu, n_soft), np.linspace(bone_threshold_hu, top, n_bone))
    )


def _hat_weights(position: np.ndarray, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower knot and weight of the upper knot for fractional knot positions in [0, n - 1]."""
    lower = np.minimum(np.floor(position).astype(np.int64), n_knots - 2)
    return lower, position - lower


def path_bins(
    ct_hu: np.ndarray,
    grid: Grid2D,
    knots: np.ndarray,
    bone_threshold_hu: float,
    lateral_knots: np.ndarray,
) -> PathBins:
    """Node weights of the HU and the lateral histograms of a CT image.

    Bins `0 .. n_hu - 1` are the HU knots: a pixel below the threshold (its CT
    number clipped to [0, threshold], which the HU map does as well) weighs on the
    two neighbouring knots of the first group, a bone pixel on those of the
    second. Bins `n_hu ..` are the lateral knots, on which only bone pixels weigh.

    Args:
        ct_hu: CT image [HU], shape `grid.shape`.
        grid: its grid.
        knots: HU knots from `hu_knots`.
        bone_threshold_hu: CT number from which a pixel is bone [HU].
        lateral_knots: evenly spaced lateral knots [m] that span the grid.
    """
    hu = np.asarray(ct_hu, dtype=np.float64).reshape(-1)
    is_bone = hu >= bone_threshold_hu
    n_soft = int(np.searchsorted(knots, bone_threshold_hu, side="left")) + 1
    n_bone = knots.shape[0] - n_soft
    soft_step = knots[1] - knots[0]
    bone_step = knots[n_soft + 1] - knots[n_soft]
    soft_lower, soft_upper = _hat_weights(
        np.clip(hu, 0.0, bone_threshold_hu) / soft_step, n_soft
    )
    bone_lower, bone_upper = _hat_weights(
        np.clip((hu - bone_threshold_hu) / bone_step, 0.0, n_bone - 1.0), n_bone
    )
    lower = np.where(is_bone, n_soft + bone_lower, soft_lower)
    upper_weight = np.where(is_bone, bone_upper, soft_upper)

    _, x = grid.axes()
    lateral_step = lateral_knots[1] - lateral_knots[0]
    column_lower, column_upper = _hat_weights(
        np.clip((x - lateral_knots[0]) / lateral_step, 0.0, lateral_knots.shape[0] - 1.0),
        lateral_knots.shape[0],
    )
    lateral_lower = np.tile(column_lower, grid.shape[0]) + knots.shape[0]
    lateral_upper = np.tile(column_upper, grid.shape[0])
    index = np.stack((lower, lower + 1, lateral_lower, lateral_lower + 1), axis=1)
    weight = np.stack(
        (
            1.0 - upper_weight,
            upper_weight,
            np.where(is_bone, 1.0 - lateral_upper, 0.0),
            np.where(is_bone, lateral_upper, 0.0),
        ),
        axis=1,
    )
    return PathBins(index, weight, knots.shape[0] + lateral_knots.shape[0])


def cosine_modes(n_knots: int, n_modes: int) -> np.ndarray:
    """Cosine modes `sqrt(2) cos(pi (j + 1/2) k / n)`, k = 1 .. n_modes, at n knots.

    Smoothest first, unit rms over the knots, mutually orthogonal. The constant
    (k = 0) is left out: a constant extra slowness in bone is what `c_bone` does.

    Returns float64, shape (n_knots, n_modes).
    """
    if n_modes >= n_knots:
        raise ValueError(f"{n_modes} lateral modes need more than {n_knots} lateral knots")
    knot = np.arange(n_knots)[:, None] + 0.5
    order = np.arange(1, n_modes + 1)[None, :]
    return np.sqrt(2.0) * np.cos(np.pi * knot * order / n_knots)


def build_family(scene: Scene, cfg: SkullModelConfig) -> SkullFamily:
    """Trace the nominal CT medium of a scene once and collect the path histograms.

    Cost: one bent-ray trace of the scene's bank (the eikonal solves and the ray
    tracing; the histograms add about a third to the tracing).

    Raises:
        ValueError: if the scene has no CT image, or its correction-model backend
            is not `bent_ray` (the histograms come from that tracer).
    """
    if scene.ct_hu is None or scene.medium_grid is None:
        raise ValueError("the scene carries no CT image (Scene.ct_hu); it cannot have a CT model")
    backend = from_dict(BackendConfig, scene.meta["model_backend"])
    if backend.backend != "bent_ray":
        raise ValueError(
            f"the skull-model family needs the bent_ray backend, the scene uses {backend.backend!r}"
        )
    ct = scene.meta["ct"]
    nominal = HuMapping(**ct["hu_mapping"])
    setup = imaging_setup(scene.imaging)
    grid = scene.medium_grid
    threshold = float(ct["bone_threshold_hu"])
    knots = hu_knots(float(np.max(scene.ct_hu)), threshold, cfg.hu_knot_spacing)
    _, x = grid.axes()
    n_lateral = max(int(np.ceil((x[-1] - x[0]) / cfg.lateral_knot_spacing)) + 1, 2)
    lateral_knots = np.linspace(x[0], x[-1], n_lateral)
    bins = path_bins(scene.ct_hu, grid, knots, threshold, lateral_knots)
    medium = ct_plane_medium(
        scene.ct_hu, grid, scene.plane, nominal, float(ct["alpha_power"]), threshold
    )
    rays = trace_bent_rays(
        medium, setup.transducer, setup.roi.points().reshape(-1, 2), backend.bent_ray, bins
    )
    return SkullFamily(
        transducer=setup.transducer,
        roi=setup.roi,
        freqs=setup.freqs,
        grid=grid,
        plane=scene.plane,
        ct_hu=scene.ct_hu,
        nominal=nominal,
        alpha_power=float(ct["alpha_power"]),
        bone_threshold_hu=threshold,
        backend=backend,
        cfg=cfg,
        hu_knots=knots,
        lateral_knots=lateral_knots,
        lateral_modes=cosine_modes(n_lateral, cfg.n_lateral_modes),
        hu_path=rays.bin_path[:, :, : knots.shape[0]],
        lateral_path=rays.bin_path[:, :, knots.shape[0] :],
        path_length=rays.path_length,
        travel_time=rays.travel_time,
        alpha0_path=rays.alpha0_path,
        amplitude=rays.window * rays.transmission * rays.spreading,
    )


def knot_slowness(
    knots: torch.Tensor, c_bone: torch.Tensor, c_water: torch.Tensor, hu_bone: torch.Tensor
) -> torch.Tensor:
    """Slowness the HU map gives each HU knot [s/m] (`ct.acoustics.hu_to_acoustic`).

    knots: CT numbers [HU], shape (n,); the constants are scalars (tensors).
    """
    solid = torch.clamp(knots / hu_bone, 0.0, 1.0)
    return 1.0 / (c_water + (c_bone - c_water) * solid)


def nominal_knot_slowness(family: SkullFamily) -> torch.Tensor:
    """`knot_slowness` of the family's nominal map, float64, shape (n_hu,)."""
    nominal = family.nominal
    return knot_slowness(
        torch.from_numpy(family.hu_knots),
        torch.tensor(nominal.c_bone, dtype=torch.float64),
        torch.tensor(nominal.c_water, dtype=torch.float64),
        torch.tensor(nominal.hu_bone, dtype=torch.float64),
    )


def travel_time_change(
    hu_path: torch.Tensor,
    lateral_path: torch.Tensor,
    knot_slowness_change: torch.Tensor,
    lateral_slowness: torch.Tensor,
) -> torch.Tensor:
    """Fixed-path travel-time change of a member against the nominal medium [s].

    hu_path: (..., n_hu) [m]; lateral_path: (..., n_lateral) [m];
    knot_slowness_change: slowness of the member minus the nominal one at the HU
    knots [s/m], (n_hu,); lateral_slowness: extra slowness in bone at the lateral
    knots [s/m], (n_lateral,). Returns float64, shape (...).
    """
    return (
        hu_path.to(torch.float64) @ knot_slowness_change
        + lateral_path.to(torch.float64) @ lateral_slowness
    )


def parameter_tensors(
    family: SkullFamily, parameters: SkullParameters
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(knot_slowness_change, lateral_slowness)` of a member, float64 (`travel_time_change`)."""

    def as_tensor(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)

    slowness = knot_slowness(
        torch.from_numpy(family.hu_knots),
        as_tensor(parameters.c_bone),
        as_tensor(parameters.c_water),
        as_tensor(parameters.hu_bone),
    )
    lateral = torch.from_numpy(family.lateral_modes @ np.asarray(parameters.lateral, np.float64))
    return slowness - nominal_knot_slowness(family), lateral


def fermat_travel_time(family: SkullFamily, parameters: SkullParameters) -> np.ndarray:
    """Fixed-path travel time of a member for every element-node pair [s], (n_elements, n_nodes)."""
    change = travel_time_change(
        torch.from_numpy(family.hu_path),
        torch.from_numpy(family.lateral_path),
        *parameter_tensors(family, parameters),
    )
    return family.travel_time + change.numpy()


def bank_from_travel_time(
    family: SkullFamily, travel_time: np.ndarray, common_amplitude: float
) -> GreensBank:
    """Bank with the nominal amplitudes and absorption and the given travel times.

    `B = g W T S exp(-A(f)) exp(+i w (tau - L / c_ref))`, the expression of
    `fus.backends.bent_ray.bent_ray_bank`.

    travel_time: [s], shape (n_elements, n_nodes).
    """
    freqs = family.freqs
    excess_delay = travel_time - family.path_length / C_REF
    nepers_per_alpha0 = db_cm_mhz_to_np_m(1.0, freqs, family.alpha_power)
    n_elements, n_nodes = travel_time.shape
    values = np.empty((freqs.size, n_elements, n_nodes), dtype=np.complex64)
    for element in range(n_elements):
        exponent = (
            2j * np.pi * freqs[:, None] * excess_delay[element]
            - nepers_per_alpha0[:, None] * family.alpha0_path[element]
        )
        values[:, element] = common_amplitude * family.amplitude[element] * np.exp(exponent)
    return GreensBank(
        freqs=freqs,
        roi=family.roi,
        B=values.reshape(freqs.size, n_elements, *family.roi.shape),
        c_ref=C_REF,
        transducer=family.transducer,
        backend="bent_ray_fixed_path",
        meta={"model": "recon.skull_model.fermat_bank"},
    )


def fermat_bank(family: SkullFamily, parameters: SkullParameters) -> GreensBank:
    """Fixed-path prediction of a member's bank (module docstring)."""
    return bank_from_travel_time(
        family, fermat_travel_time(family, parameters), parameters.common_amplitude
    )


def lateral_slowness_map(family: SkullFamily, lateral: np.ndarray) -> np.ndarray:
    """Extra slowness of a member on the CT grid [s/m], float64, shape `grid.shape`.

    The lateral profile (modes times coefficients at the knots, linear between
    them) in bone pixels, zero elsewhere: the map whose path integral
    `travel_time_change` takes.
    """
    _, x = family.grid.axes()
    profile = np.interp(x, family.lateral_knots, family.lateral_modes @ np.asarray(lateral))
    return np.where(family.ct_hu >= family.bone_threshold_hu, profile[None, :], 0.0)


def model_medium(family: SkullFamily, parameters: SkullParameters) -> PlaneMedium:
    """The plane medium of a member: the re-mapped CT plus the lateral profile in bone."""
    mapping = dataclasses.replace(
        family.nominal,
        c_bone=float(parameters.c_bone),
        c_water=float(parameters.c_water),
        hu_bone=float(parameters.hu_bone),
    )
    medium = ct_plane_medium(
        family.ct_hu, family.grid, family.plane, mapping, family.alpha_power,
        family.bone_threshold_hu,
    )
    slowness = 1.0 / medium.sound_speed.astype(np.float64)
    slowness += lateral_slowness_map(family, parameters.lateral)
    return dataclasses.replace(medium, sound_speed=(1.0 / slowness).astype(np.float32))


def model_bank(family: SkullFamily, parameters: SkullParameters) -> GreensBank:
    """Bank of a member from the real backend (one bent-ray trace; tens of seconds)."""
    bank = compute_bank(
        model_medium(family, parameters), family.transducer, family.roi, family.freqs,
        family.backend,
    )
    if parameters.common_amplitude == 1.0:
        return bank
    return dataclasses.replace(
        bank, B=(bank.B * np.float32(parameters.common_amplitude)).astype(np.complex64)
    )


def delay_only(bank: GreensBank) -> GreensBank:
    """A bank with the phases of `bank` and unit amplitude wherever `bank` has amplitude.

    `B / |B|`, and 0 where `B` is 0 (pairs the model does not reach, such as rays
    beyond grazing): the model's delays without its amplitudes (module docstring).
    """
    magnitude = np.abs(bank.B)
    values = np.zeros_like(bank.B)
    np.divide(bank.B, magnitude, out=values, where=magnitude > 0.0)
    return dataclasses.replace(bank, B=values, backend=f"{bank.backend}_delay_only")

