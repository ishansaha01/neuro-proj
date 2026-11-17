"""Differentiable forward model of the power Doppler image for an uncertain skull.

The skull is modelled by a phase screen on the homogeneous reference medium
(`data.types.PhaseScreen`, `fus.screen`): element `e` carries the complex factor

    u[f, e] = amplitude_e * exp(+i 2 pi f delay_e)

on transmit and on receive. The image is beamformed conventionally (homogeneous
model), as `skull_das` is. There is no second physics implementation here:
`fus.operators.operator_factors` computes the per-frequency arrays once, and
this module only contracts them in torch,

    A(u)[p, s] = sum_f P[f] * T[f][p, s] * R[f][p, s]
    T[f] = tx_w[f]^T ((D[f] * u[f]) G[f]),   R[f] = (rx_w[f]^T * u[f]) G[f]
    H(u)[p, c] = mean over the sub-cell sources s of cell c of |A(u)[p, s]|^2

so that `H` can be differentiated with respect to the delays and the log
amplitudes. Eval E-RECON-005 checks the assembly against `fus.operators.psf_operator`
on a phase-screen bank.

The expected image is `H @ (cbv * cell_area) + noise_floor` (`Scene` docstring);
`expected_power_doppler` is shared by the known-skull and the uncertain-skull case.

Cost of one assembly: nf * n_selected_pixels * (n_elements + n_angles) * n_src
complex multiply-adds; with gradients about three times that.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from neurosim.common.grid import Grid2D
from neurosim.data.types import PhaseScreen, Scene
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.fus.operators import OperatorFactors, operator_factors
from neurosim.fus.types import GreensBank

# Largest number of complex intermediate values kept for a backward pass at once;
# larger problems are assembled in frequency chunks that are recomputed in the
# backward pass. 1.2e7 complex64 values are 96 MB.
_CHUNK_VALUES = 12_000_000
# (n_selected, n_src) arrays autograd keeps per frequency: transmit, receive, their product.
_STORED_PER_FREQUENCY = 3


@dataclass(frozen=True)
class ScreenModel:
    """Tensors of `fus.operators.OperatorFactors`, arranged for batched matrix products.

    Attributes:
        freqs: frequencies [Hz], float64 tensor, shape (nf,).
        pulse: two-way pulse spectrum, real tensor, shape (nf,).
        tx_weights: complex tensor, shape (nf, n_angles, n_pix).
        tx_phasors: complex tensor, shape (nf, n_angles, n_elements).
        rx_weights: complex tensor, shape (nf, n_elements, n_pix).
        greens: complex tensor, shape (nf, n_elements, n_src), `n_src = sources_per_cell * n_cells`,
            ordered sub-cell offset first, then cell.
        sources_per_cell: sub-cell source points per CBV cell.
    """

    freqs: torch.Tensor
    pulse: torch.Tensor
    tx_weights: torch.Tensor
    tx_phasors: torch.Tensor
    rx_weights: torch.Tensor
    greens: torch.Tensor
    sources_per_cell: int

    @property
    def n_elements(self) -> int:
        """Number of array elements."""
        return self.greens.shape[1]

    @property
    def n_pixels(self) -> int:
        """Number of image pixels."""
        return self.rx_weights.shape[2]


def refined_source_points(pixel_grid: Grid2D, refinement: int) -> np.ndarray:
    """Source points of every CBV cell: `refinement` x `refinement` points per cell.

    The midpoint rule of `data.scene.power_doppler_operator` (points spread evenly
    over the cell, depth offset in the outer loop), which is how the operators of a
    `Scene` are built.

    Returns:
        float64 array, shape (refinement**2 * n_pix, 2), columns (x, z) [m]; all cells
        at the first offset, then all cells at the second, and so on.
    """
    if refinement < 1:
        raise ValueError(f"refinement must be >= 1, got {refinement}")
    centres = pixel_grid.points().reshape(-1, 2)
    fractions = (np.arange(refinement) + 0.5) / refinement - 0.5
    offsets = [
        (lateral * pixel_grid.spacing, depth * pixel_grid.spacing)
        for depth in fractions
        for lateral in fractions
    ]
    return np.concatenate([centres + np.asarray(offset) for offset in offsets])


def screen_model_from_factors(
    factors: OperatorFactors, sources_per_cell: int, dtype: torch.dtype = torch.complex64
) -> ScreenModel:
    """Wrap exported operator factors as tensors (without copying them for complex64).

    factors: from `fus.operators.operator_factors`.
    sources_per_cell: sub-cell source points per CBV cell (`refined_source_points`).
    dtype: `torch.complex64` (fast) or `torch.complex128` (for gradient checks).
    """
    if factors.greens.shape[2] % sources_per_cell != 0:
        raise ValueError(
            f"{factors.greens.shape[2]} sources are not a multiple of {sources_per_cell} per cell"
        )
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    return ScreenModel(
        freqs=torch.from_numpy(factors.freqs),
        pulse=torch.from_numpy(factors.pulse).to(real),
        tx_weights=torch.from_numpy(factors.tx_weights).to(dtype),
        tx_phasors=torch.from_numpy(factors.tx_phasors).to(dtype),
        rx_weights=torch.from_numpy(factors.rx_weights).to(dtype),
        greens=torch.from_numpy(factors.greens).to(dtype),
        sources_per_cell=sources_per_cell,
    )


def reference_bank(scene: Scene) -> GreensBank:
    """The homogeneous reference bank of a scene's array and frequency grid.

    `B = 1` everywhere, so it is stored on the smallest grid that holds the pixel
    cells: 2 x 2 nodes spanning the pixel grid (a bank on the scene's ROI grid
    would hold nf * n_elements * n_roi ones).
    """
    reference = scene.bank_skull
    nz, nx = scene.pixel_grid.shape
    roi = Grid2D(
        shape=(2, 2),
        spacing=max(nz, nx) * scene.pixel_grid.spacing,
        origin=scene.pixel_grid.origin,
    )
    return GreensBank.homogeneous(reference.transducer, roi, reference.freqs, reference.c_ref)


def scene_screen_model(
    scene: Scene, source_refinement: int, dtype: torch.dtype = torch.complex64
) -> ScreenModel:
    """Phase-screen forward model of a scene's `skull_das` image, all source points at once.

    The screen sits on the homogeneous reference medium of the scene's array and
    frequency grid; the beamformer is the conventional one. Memory grows with
    `source_refinement**2` (nf * n_elements * n_src values); `scene_screen_operators`
    assembles operators one sub-cell offset at a time instead.

    scene: the scene.
    source_refinement: sub-cell source points per pixel side; to reproduce the scene's
        operators it must equal the value they were built with
        (`data.config.RunConfig.source_refinement`).
    """
    homogeneous = reference_bank(scene)
    factors = operator_factors(
        homogeneous,
        scene.pixel_grid,
        refined_source_points(scene.pixel_grid, source_refinement),
        homogeneous,
        imaging_setup(scene.imaging).sequence,
        scene.imaging.beamforming,
    )
    return screen_model_from_factors(factors, source_refinement**2, dtype)


def scene_screen_operators(
    scene: Scene, screens: Sequence[PhaseScreen], source_refinement: int
) -> list[torch.Tensor]:
    """Power Doppler operators `H(phi)` of several screens for one scene, without gradients.

    Equal to `power_doppler_operator(scene_screen_model(scene, source_refinement), ...)`
    for each screen, but the sub-cell source offsets are processed one at a time
    (sharing the beamformer weights), so memory stays at that of one source point
    per pixel.

    Returns one float32 tensor of shape (n_pix, n_pix) per screen.
    """
    homogeneous = reference_bank(scene)
    sequence = imaging_setup(scene.imaging).sequence
    beamformer = make_beamformer(scene.pixel_grid, homogeneous, sequence, scene.imaging.beamforming)
    n_pix = scene.pixel_grid.shape[0] * scene.pixel_grid.shape[1]
    sources = refined_source_points(scene.pixel_grid, source_refinement).reshape(-1, n_pix, 2)
    operators = [torch.zeros((n_pix, n_pix), dtype=torch.float32) for _ in screens]
    weights = None
    for offset_sources in sources:
        factors = operator_factors(
            homogeneous,
            scene.pixel_grid,
            offset_sources,
            homogeneous,
            sequence,
            scene.imaging.beamforming,
            beamformer,
        )
        if weights is None:
            weights = screen_model_from_factors(factors, 1)
        model = dataclasses.replace(weights, greens=torch.from_numpy(factors.greens))
        with torch.no_grad():
            for operator, screen in zip(operators, screens):
                operator += power_doppler_operator(model, *screen_tensors(screen))
    return [operator / len(sources) for operator in operators]


def element_factors(
    freqs: torch.Tensor, delays: torch.Tensor, log_amplitudes: torch.Tensor
) -> torch.Tensor:
    """Per-element complex factor `u[f, e] = exp(log_amplitude_e + i 2 pi f delay_e)`.

    freqs: frequencies [Hz], float64, shape (nf,).
    delays: extra one-way delay per element [s], shape (n_elements,).
    log_amplitudes: natural log of the one-way amplitude factor, shape (n_elements,).

    Returns a complex128 tensor, shape (nf, n_elements). The phase is formed in
    float64: `f * delay` is of order 1 and has to be resolved far below one cycle.
    """
    phase = 2.0 * math.pi * freqs[:, None] * delays.to(torch.float64)[None, :]
    return torch.polar(torch.exp(log_amplitudes.to(torch.float64))[None, :].expand_as(phase), phase)


def _psf_chunk(
    model: ScreenModel,
    factors: torch.Tensor,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Sum over the frequencies `start:stop` of the terms of `A(u)`; shape (n_selected, n_src).

    One frequency at a time, so the temporaries are (n_selected, n_src), not
    (frequencies, n_selected, n_src).
    """
    total = None
    for f in range(start, stop):
        u = factors[f].to(model.greens.dtype)
        greens = model.greens[f]
        transmit = tx_weights[f].T @ ((model.tx_phasors[f] * (u * model.pulse[f])) @ greens)
        receive = (rx_weights[f].T * u) @ greens
        if total is None:
            total = transmit * receive
        elif total.requires_grad:
            total = total + transmit * receive
        else:
            total.addcmul_(transmit, receive)
    return total


def psf_matrix(
    model: ScreenModel,
    delays: torch.Tensor,
    log_amplitudes: torch.Tensor,
    pixel_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Point-spread operator `A(u)` of a phase screen, differentiable in the screen.

    model: the factor tensors.
    delays: [s], shape (n_elements,).
    log_amplitudes: shape (n_elements,).
    pixel_index: optional integer tensor of pixel indices (rows of `A`) to assemble;
        None = all pixels.

    Returns a complex tensor, shape (n_selected, n_src).
    """
    factors = element_factors(model.freqs, delays, log_amplitudes)
    tx_weights, rx_weights = model.tx_weights, model.rx_weights
    if pixel_index is not None:
        tx_weights, rx_weights = tx_weights[:, :, pixel_index], rx_weights[:, :, pixel_index]
    nf, n_src = model.greens.shape[0], model.greens.shape[2]
    if not factors.requires_grad:
        return _psf_chunk(model, factors, tx_weights, rx_weights, 0, nf)
    # With gradients every frequency keeps three (n_selected, n_src) arrays for the
    # backward pass; beyond the memory budget the frequencies are split into chunks
    # that are recomputed in the backward pass instead.
    per_chunk = max(1, _CHUNK_VALUES // (_STORED_PER_FREQUENCY * rx_weights.shape[2] * n_src))
    if per_chunk >= nf:
        return _psf_chunk(model, factors, tx_weights, rx_weights, 0, nf)
    total = torch.zeros((rx_weights.shape[2], n_src), dtype=model.greens.dtype)
    for start in range(0, nf, per_chunk):
        total = total + checkpoint(
            _psf_chunk,
            model,
            factors,
            tx_weights,
            rx_weights,
            start,
            min(start + per_chunk, nf),
            use_reentrant=False,
        )
    return total


def power_doppler_operator(
    model: ScreenModel,
    delays: torch.Tensor,
    log_amplitudes: torch.Tensor,
    pixel_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Power Doppler operator `H(u)` from CBV cells to pixels.

    `|A(u)|^2` averaged over the sub-cell sources of each cell. Arguments as for
    `psf_matrix`.

    Returns a real tensor, shape (n_selected, n_cells).
    """
    psf = psf_matrix(model, delays, log_amplitudes, pixel_index)
    power = psf.real.square() + psf.imag.square()
    return power.reshape(power.shape[0], model.sources_per_cell, -1).mean(dim=1)


def screen_tensors(screen: PhaseScreen) -> tuple[torch.Tensor, torch.Tensor]:
    """Delays [s] and log amplitudes of a `PhaseScreen` as float64 tensors, shape (n_elements,)."""
    return (
        torch.from_numpy(np.asarray(screen.delays, dtype=np.float64)),
        torch.from_numpy(np.log(np.asarray(screen.amplitudes, dtype=np.float64))),
    )


def expected_power_doppler(
    operator: torch.Tensor, cbv: torch.Tensor, cell_area: float, noise_floor: torch.Tensor
) -> torch.Tensor:
    """Expected power Doppler image `H @ (cbv * cell_area) + noise_floor`.

    operator: power Doppler operator, shape (n_selected, n_cells).
    cbv: blood volume fraction per cell, shape (batch, n_cells).
    cell_area: area of one cell [m^2].
    noise_floor: electronic-noise floor of the selected pixels, shape (n_selected,).

    Returns shape (batch, n_selected), in the units of the measured image.
    """
    return (cbv * cell_area) @ operator.T + noise_floor
