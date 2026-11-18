"""Beamformer weights of a skull-model member, in torch, differentiable in the model.

A scanner that keeps its channel data can beamform them again with any model
of the medium. `fus.beamform.make_beamformer` builds the weights of one model
bank in numpy; this module builds the same weights for a member of the
`recon.skull_model` family from the arrays `fus.operators.weight_factors`
exports and the family's fixed-path travel times, so that an image-quality
criterion can be differentiated with respect to the skull parameters. There is
no second physics implementation: eval E-RECON-012 checks the weights, applied
to the true medium's factors, against `fus.operators.psf_operator` with the
member's bank as the model bank.

The member enters through its delays only (`recon.skull_model.delay_only`): the
bank is a unit phasor on the element-node pairs the nominal trace reaches and
0 elsewhere. The model's transfer function to pixel `p` is

    M[f, e, p] = reference[f, e, p] * sum_c w_c[p] support[e, n_c] exp(i 2 pi f eps[e, n_c])
    eps = tau(theta) - L / c_ref

over the four ROI nodes `n_c` around the pixel, and the weights follow
`fus.operators.WeightFactors`. A common amplitude of the bank changes
`unit_response` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.operators import weight_factors
from neurosim.fus.types import GreensBank, PlaneWaveSequence
from neurosim.recon.skull_model import SkullFamily, travel_time_change


@dataclass(frozen=True)
class WeightModel:
    """Tensors from which the weights of any family member follow, for a set of pixels.

    `n_nodes` counts the distinct ROI nodes around the selected pixels.

    Attributes:
        freqs: frequencies [Hz], float64, shape (nf,).
        pulse: two-way pulse spectrum, float32, shape (nf,).
        tx_phasors: firing phasors, complex64, shape (nf, n_angles, n_elements).
        reference: reference Green's function at the pixels, complex64, (nf, n_elements, n_sel).
        apodization: receive apodisation, float32, shape (n_elements, n_sel).
        corner: for each pixel, positions of its four ROI nodes in the node list, int64, (4, n_sel).
        corner_weight: their bilinear weights, float32, shape (4, n_sel).
        support: 1 where the nominal trace gives the element-node pair amplitude, 0
            elsewhere, float32, shape (n_elements, n_nodes).
        nominal_excess_delay: `tau_nominal - L / c_ref` at the nodes [s], float64,
            shape (n_elements, n_nodes).
        hu_path, lateral_path: path-length histograms of the rays to the nodes [m],
            float32, shapes (n_elements, n_nodes, n_hu) and (n_elements, n_nodes, n_lateral).
    """

    freqs: torch.Tensor
    pulse: torch.Tensor
    tx_phasors: torch.Tensor
    reference: torch.Tensor
    apodization: torch.Tensor
    corner: torch.Tensor
    corner_weight: torch.Tensor
    support: torch.Tensor
    nominal_excess_delay: torch.Tensor
    hu_path: torch.Tensor
    lateral_path: torch.Tensor


@dataclass(frozen=True)
class Weights:
    """Beamformer weights of one model at the selected pixels, before normalisation.

    The normalised weights of `fus.beamform.Beamformer` are `tx_phase` and
    `rx_phase / unit_response`; its noise gain is `squared_weights / unit_response^2`.

    Attributes:
        tx_phase: conjugate unit phasors of the model transmit field, complex64,
            shape (nf, n_angles, n_sel).
        rx_phase: apodisation times conjugate unit phasors of the model receive
            Green's function, complex64, shape (nf, n_elements, n_sel).
        unit_response: response of the un-normalised sum to a unit scatterer at the
            pixel in the model medium, float32, shape (n_sel,).
        squared_weights: sum over (f, angle, element) of the squared un-normalised
            weights, float32, shape (n_sel,).
    """

    tx_phase: torch.Tensor
    rx_phase: torch.Tensor
    unit_response: torch.Tensor
    squared_weights: torch.Tensor


def weight_model(
    family: SkullFamily,
    pixel_grid: Grid2D,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig,
    pixel_index: np.ndarray | None = None,
) -> WeightModel:
    """Collect the tensors of `WeightModel` for a family and a set of pixels.

    family: the skull-model family of the scene.
    pixel_grid: image grid; sequence, beamforming: as for `fus.beamform.make_beamformer`.
    pixel_index: flat indices of the pixels to beamform; None = all pixels.
    """
    grid_bank = GreensBank(
        freqs=family.freqs,
        roi=family.roi,
        B=np.broadcast_to(
            np.complex64(1.0), (family.freqs.size, family.transducer.n_elements, *family.roi.shape)
        ),
        c_ref=C_REF,
        transducer=family.transducer,
        backend="analytic",
    )
    factors = weight_factors(grid_bank, pixel_grid, sequence, beamforming, pixel_index)
    nodes, corner = np.unique(factors.nodes, return_inverse=True)
    return WeightModel(
        freqs=torch.from_numpy(factors.freqs),
        pulse=torch.from_numpy(factors.pulse.astype(np.float32)),
        tx_phasors=torch.from_numpy(factors.tx_phasors),
        reference=torch.from_numpy(factors.reference),
        apodization=torch.from_numpy(factors.apodization.astype(np.float32)),
        corner=torch.from_numpy(corner.reshape(factors.nodes.shape)),
        corner_weight=torch.from_numpy(factors.node_weights),
        support=torch.from_numpy((family.amplitude[:, nodes] > 0.0).astype(np.float32)),
        nominal_excess_delay=torch.from_numpy(
            (family.travel_time - family.path_length / C_REF)[:, nodes]
        ),
        hu_path=torch.from_numpy(np.ascontiguousarray(family.hu_path[:, nodes])),
        lateral_path=torch.from_numpy(np.ascontiguousarray(family.lateral_path[:, nodes])),
    )


def _unit_phasor(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(conj(v) / |v|, |v|)`, the phasor being 0 where `v` is 0."""
    magnitude = values.abs()
    nonzero = magnitude > 0.0
    safe = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    return torch.where(nonzero, 1.0, 0.0) * values.conj() / safe, magnitude


def model_weights(
    model: WeightModel,
    knot_slowness_change: torch.Tensor,
    lateral_slowness: torch.Tensor,
    common_amplitude: float = 1.0,
) -> Weights:
    """Weights of the family member with the given slowness changes (module docstring).

    model: the tensors of the family at the selected pixels.
    knot_slowness_change: slowness of the member minus the nominal one at the HU
        knots [s/m], float64, shape (n_hu,).
    lateral_slowness: extra slowness in bone at the lateral knots [s/m], float64,
        shape (n_lateral,).
    common_amplitude: common one-way amplitude factor of the member's bank.
    """
    excess_delay = model.nominal_excess_delay + travel_time_change(
        model.hu_path, model.lateral_path, knot_slowness_change, lateral_slowness
    )
    # The phase is formed in float64 (f * delay spans several cycles) and reduced to one
    # turn before it is rounded to float32.
    turns = model.freqs[:, None, None] * excess_delay[None, :, :]
    phase = (2.0 * math.pi * (turns - torch.round(turns))).to(torch.float32)
    at_nodes = torch.polar((common_amplitude * model.support).expand_as(phase), phase)
    bank = sum(
        model.corner_weight[c] * at_nodes[:, :, model.corner[c]] for c in range(4)
    )
    receive = model.reference * bank
    transmit = model.tx_phasors @ receive
    rx_unit, rx_magnitude = _unit_phasor(receive)
    tx_phase, tx_magnitude = _unit_phasor(transmit)
    rx_phase = rx_unit * model.apodization
    unit_response = torch.einsum(
        "f,fp,fp->p",
        model.pulse,
        tx_magnitude.sum(dim=1),
        (rx_magnitude * model.apodization).sum(dim=1),
    )
    squared_weights = (
        tx_phase.abs().square().sum(dim=1) * rx_phase.abs().square().sum(dim=1)
    ).sum(dim=0)
    return Weights(tx_phase, rx_phase, unit_response, squared_weights)
