"""Calibrate the skull model by re-beamforming: maximise a focus criterion over the family.

Principle. A beamformer is a linear filter on the channel data, and for a
signal `g` in white noise the output signal-to-noise ratio `|w^H g|^2 / (v w^H w)`
is largest for the matched filter `w = g` (Cauchy-Schwarz). Blood is a random
signal, but each scatterer's echo is such a `g`, fixed by the true medium: the
ratio of the time-averaged blood power to the beamformer's own noise floor,
`r_p = (PD_p - floor_p) / floor_p`, rises when the model that sets the weights
approaches the true medium, and it does not depend on how the weights are
normalised (ROADMAP D-011). Estimating aberration by maximising such a quantity
computed from the data alone is classical: speckle brightness (Nock, Trahey &
Smith 1989, J. Acoust. Soc. Am. 85:1819) and image sharpness (Muller &
Buffington 1974, J. Opt. Soc. Am. 64:1200).

Criterion. With `r_p(theta)` the ratio at calibration pixel `p` for the weights
of the family member `theta` (`recon.scanner.signal_to_floor`),

    J(theta) = (1 / g) log mean_p r_p(theta)^g  +  prior_weight * log prior(theta)

The default `g = 1` is the mean ratio. Why the mean of the ratio and not of its
logarithm: (i) for a power Doppler image, which is an incoherent image of the
blood (`PD = |A|^2 @ cbv`), aberration moves power between pixels, and a concave
function such as the logarithm rewards spreading it evenly (Jensen), that is,
blurring, while the mean rewards only what focusing gains: the coherent two-way
gain of transmit and receive weights that match the same medium; (ii) the mean is
defined for measured images whose pixels dip below the floor, the logarithm is
not; (iii) it weights pixels by their blood power, so pixels without blood,
which carry no information about focus, do not count. `g = 2` (sharpness)
additionally rewards concentration and is dominated by the brightest vessels.
The logarithm outside the mean makes the criterion a relative quantity: its
changes are the same at every noise level.

Parameters and prior. The three HU-map constants live in the ranges of the CT
package (a uniform prior), enforced by a logistic map of an unconstrained
variable, so the optimiser cannot leave the box and the prior adds nothing
inside it. The lateral coefficients are in units of their prior standard
deviation, with the Gaussian log prior `-sum v_j^2 / 2`.

Degeneracies and limits. A common amplitude of the bank changes the weights'
normalisation only, hence not `r_p`: it is not optimised here but set afterwards
by a power match (`recon.medium_recon.match_common_amplitude`); the absorption
constants are not optimised for the same reason. A delay common to all elements
of a field point changes no focus, but it is not neutral for this criterion: it
moves the pixel's range gate, and with it the blood the pixel sees. More
generally `r_p` is a brightness, not a normalised coherence: besides focusing,
it rises when a model steers pixels toward blood that is brighter or lies where
the skull transmits better. The family has no free gauge directions (its
members are media with constants in bounded ranges), which limits this, but
does not remove it. Measured: on demo scene 0, where the true two-way
sensitivity varies by a factor of about 1.5 over the brain, the criterion peaks
within a few percent of the best fit of the family to the true travel times; on
the small scenes of evals E-RECON-013 and E-RECON-014, where it varies by one to
two decades, the criterion is not maximal at the true member, the HU-map
constants run to the bounds of their ranges and calibration increases the
travel-time error (both evals fail). A criterion normalised by the incoherent
power of the same pixel (a coherence factor) would not have this weakness; it
is not built.

Optimisation: L-BFGS with a strong-Wolfe line search from the nominal CT
member, in stages that add lateral modes from coarse to fine (`FocusConfig`):
the criterion is a coherent sum and has side maxima once delay errors approach
half a period, and the smooth modes carry the largest errors.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.recon.config import FocusConfig
from neurosim.recon.rebeamform import WeightModel, model_weights
from neurosim.recon.scanner import ChannelRecord, signal_to_floor
from neurosim.recon.skull_model import (
    SkullFamily,
    SkullParameters,
    knot_slowness,
    nominal_knot_slowness,
)

# Number of HU-map constants in the parameter vector: c_bone, c_water, hu_bone.
_N_CONSTANTS = 3


@dataclass(frozen=True)
class FocusResult:
    """Outcome of one calibration.

    Attributes:
        parameters: the calibrated member (its `common_amplitude` is left at 1).
        criterion_start: criterion `(1 / g) log mean r^g` of the nominal member, prior excluded.
        criterion: the same for the calibrated member.
        stage_criteria: criterion after each stage.
        n_evaluations: criterion evaluations (each with its gradient).
        seconds: run time [s].
    """

    parameters: SkullParameters
    criterion_start: float
    criterion: float
    stage_criteria: tuple[float, ...]
    n_evaluations: int
    seconds: float


@dataclass(frozen=True)
class ParameterMap:
    """Map between the optimiser's unconstrained vector and the family's parameters.

    The vector is `(a_c_bone, a_c_water, a_hu_bone, v_1 .. v_n)`: each constant is
    `low + (high - low) * logistic(a)`, each lateral coefficient `v_j * sigma` [s/m].

    Attributes:
        low, high: bounds of the three constants, float64 tensors, shape (3,).
        lateral_sigma: prior standard deviation of a lateral coefficient [s/m].
        hu_knots: HU knots of the family, float64 tensor.
        lateral_modes: modes at the lateral knots, float64 tensor, (n_lateral, n_modes).
        nominal_slowness: slowness of the nominal map at the HU knots, float64 tensor.
    """

    low: torch.Tensor
    high: torch.Tensor
    lateral_sigma: float
    hu_knots: torch.Tensor
    lateral_modes: torch.Tensor
    nominal_slowness: torch.Tensor

    def constants(self, vector: torch.Tensor) -> torch.Tensor:
        """`(c_bone [m/s], c_water [m/s], hu_bone [HU])` of a parameter vector."""
        return self.low + (self.high - self.low) * torch.sigmoid(vector[:_N_CONSTANTS])

    def slowness_changes(self, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(knot_slowness_change, lateral_slowness)` of a parameter vector [s/m]."""
        c_bone, c_water, hu_bone = self.constants(vector)
        slowness = knot_slowness(self.hu_knots, c_bone, c_water, hu_bone)
        lateral = self.lateral_modes @ (self.lateral_sigma * vector[_N_CONSTANTS:])
        return slowness - self.nominal_slowness, lateral

    def log_prior(self, vector: torch.Tensor) -> torch.Tensor:
        """Log prior density of a parameter vector up to a constant (module docstring)."""
        return -0.5 * vector[_N_CONSTANTS:].square().sum()

    def to_vector(self, parameters: SkullParameters) -> torch.Tensor:
        """Parameter vector of a member whose constants lie strictly inside their ranges."""
        values = torch.tensor(
            [parameters.c_bone, parameters.c_water, parameters.hu_bone], dtype=torch.float64
        )
        fraction = (values - self.low) / (self.high - self.low)
        if bool(((fraction <= 0.0) | (fraction >= 1.0)).any()):
            raise ValueError(
                f"the constants {values.tolist()} do not lie inside the prior ranges "
                f"{self.low.tolist()} to {self.high.tolist()}"
            )
        lateral = torch.from_numpy(np.asarray(parameters.lateral, dtype=np.float64))
        return torch.cat((torch.logit(fraction), lateral / self.lateral_sigma))

    def to_parameters(self, vector: torch.Tensor) -> SkullParameters:
        """The member of a parameter vector."""
        c_bone, c_water, hu_bone = (float(value) for value in self.constants(vector))
        lateral = (self.lateral_sigma * vector[_N_CONSTANTS:]).detach().numpy().copy()
        return SkullParameters(c_bone=c_bone, c_water=c_water, hu_bone=hu_bone, lateral=lateral)


def parameter_map(family: SkullFamily) -> ParameterMap:
    """The parameter map of a family, with the prior of its `SkullModelConfig`."""
    cfg, prior = family.cfg, family.cfg.mapping_prior
    hu_low, hu_high = (factor * family.nominal.hu_bone for factor in prior.hu_bone_relative_range)
    low = torch.tensor([prior.c_bone_range[0], prior.c_water_range[0], hu_low])
    high = torch.tensor([prior.c_bone_range[1], prior.c_water_range[1], hu_high])
    return ParameterMap(
        low=low.to(torch.float64),
        high=high.to(torch.float64),
        lateral_sigma=cfg.lateral_sound_speed_sigma / cfg.lateral_reference_speed**2,
        hu_knots=torch.from_numpy(family.hu_knots),
        lateral_modes=torch.from_numpy(family.lateral_modes),
        nominal_slowness=nominal_knot_slowness(family),
    )


def focus_criterion(ratios: torch.Tensor, power: float) -> torch.Tensor:
    """`(1 / g) log mean_p r_p^g` of the pixel signal-to-floor ratios (module docstring)."""
    return torch.log(ratios.pow(power).mean()) / power


def member_ratios(
    record: ChannelRecord,
    weights: Sequence[WeightModel],
    parameters: ParameterMap,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Signal-to-floor ratios of the member of a parameter vector at all calibration pixels.

    weights: the family's weight models, one per group of calibration pixels.
    Returns a real tensor, shape (n_pixels,), without a graph.
    """
    with torch.no_grad():
        changes = parameters.slowness_changes(vector)
        return torch.cat(
            [signal_to_floor(record, model_weights(group, *changes)) for group in weights]
        )


def criterion_and_gradient(
    record: ChannelRecord,
    weights: Sequence[WeightModel],
    parameters: ParameterMap,
    vector: torch.Tensor,
    power: float,
) -> tuple[float, torch.Tensor]:
    """Criterion (prior excluded) of a parameter vector and its gradient.

    With `U = sum_p r_p^g`, the criterion is `log(U / n) / g` and its gradient
    `grad U / (g U)`: `U` is a sum over pixels, so its gradient is accumulated one
    group of pixels at a time and only one group's graph is alive at once.

    Returns `(criterion, gradient)`, the gradient float64 of the shape of `vector`.
    """
    total = 0.0
    gradient = torch.zeros_like(vector)
    n_pixels = 0
    for group in weights:
        leaf = vector.detach().clone().requires_grad_(True)
        ratios = signal_to_floor(record, model_weights(group, *parameters.slowness_changes(leaf)))
        powers = ratios.pow(power).sum()
        (group_gradient,) = torch.autograd.grad(powers, leaf)
        total += float(powers.detach())
        gradient += group_gradient
        n_pixels += ratios.shape[0]
    return math.log(total / n_pixels) / power, gradient / (power * total)


def calibrate(
    record: ChannelRecord, family: SkullFamily, weights: Sequence[WeightModel], cfg: FocusConfig
) -> FocusResult:
    """Maximise the focus criterion over the family, from the nominal member.

    record: the scanner's channel record (`recon.scanner.record_channel_data`).
    family: the skull-model family of the scene.
    weights: its weight models at the calibration pixels, one per group of pixels
        (`recon.rebeamform.weight_model`; `pixel_groups` splits the pixels).
    cfg: stages, iterations and prior weight.

    Raises:
        ValueError: if a stage asks for more lateral modes than the family has.
    """
    if cfg.mode_stages[-1] > family.n_lateral_modes:
        raise ValueError(
            f"mode_stages ends at {cfg.mode_stages[-1]} modes, the family has "
            f"{family.n_lateral_modes}"
        )
    start = time.perf_counter()
    parameters = parameter_map(family)
    vector = parameters.to_vector(family.nominal_parameters())
    n_evaluations = 0

    def criterion(at: torch.Tensor) -> float:
        ratios = member_ratios(record, weights, parameters, at)
        return float(focus_criterion(ratios, cfg.criterion_power))

    criterion_start = criterion(vector)
    stage_criteria = []
    for n_modes in cfg.mode_stages:
        n_active = _N_CONSTANTS + n_modes
        active = vector[:n_active].clone().requires_grad_(True)
        frozen = vector[n_active:]
        optimiser = torch.optim.LBFGS(
            [active],
            max_iter=cfg.iterations_per_stage,
            max_eval=math.ceil(2.5 * cfg.iterations_per_stage),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal n_evaluations
            optimiser.zero_grad()
            value, gradient = criterion_and_gradient(
                record, weights, parameters, torch.cat((active.detach(), frozen)),
                cfg.criterion_power,
            )
            log_prior = cfg.prior_weight * parameters.log_prior(torch.cat((active, frozen)))
            (-log_prior).backward()
            active.grad -= gradient[:n_active]
            n_evaluations += 1
            return -(value + log_prior.detach())

        optimiser.step(closure)
        vector = torch.cat((active.detach(), frozen))
        stage_criteria.append(criterion(vector))
    return FocusResult(
        parameters=parameters.to_parameters(vector),
        criterion_start=criterion_start,
        criterion=stage_criteria[-1],
        stage_criteria=tuple(stage_criteria),
        n_evaluations=n_evaluations,
        seconds=time.perf_counter() - start,
    )


def pixel_groups(pixels: np.ndarray, pixels_per_pass: int) -> list[np.ndarray]:
    """Split the calibration pixels into groups of at most `pixels_per_pass` (in order)."""
    return [
        pixels[first : first + pixels_per_pass]
        for first in range(0, pixels.shape[0], pixels_per_pass)
    ]
