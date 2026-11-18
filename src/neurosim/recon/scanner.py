"""Emulation of a scanner that re-beamforms its recorded channel data. THE TRUTH BOUNDARY.

A real scanner records channel data and can beamform them again with any model
of the medium. The datasets store images of four fixed beamformers, not channel
data, so re-beamforming is emulated here: the power Doppler image a scanner
would get from the recorded data with model weights `w` has the expectation

    |A(bank_true; w)|^2 @ (cbv * cell_area) + noise_floor(w)

(`fus.operators`, `Scene` docstring), with speckle about it.

This is the only module of the reconstruction that may read the ground truth
of a scene (`scene.bank_skull`, `scene.bank_clean`, `scene.operators`,
`scene.screen_true`, `scene.cbv_baseline`, `scene.activation`, `trials.cbv`)
for anything other than final metrics; `tests/unit/recon/test_truth_boundary.py`
greps the other modules. What it hands out is what a scanner has:

- `signal_to_floor`: for requested beamformer weights, the ratio of the
  time-averaged blood power to the beamformer's own noise floor at selected
  pixels, as a torch tensor that can be differentiated with respect to the
  weights. It is computed from the *expected* time-averaged image, not from a
  speckled one: over the 120 images of a scene and tens of pixels the speckle of
  the average is well under a percent of the ratio. A scanner would measure the
  floor of a beamformer with the transmitters off, or compute it from the
  channel noise level as here.
- `rebeamform_trials`: the image series for a requested model bank, with
  speckle drawn by `fus.acquire.statistical_power_doppler`, together with that
  beamformer's noise floor and the independent-frame counts (an oracle noise
  model, as for the stored beamformers).
- `true_model_bank` and `true_model_operator`: the oracles of the experiment's
  upper-bound rows.

Approximations of the emulation: the blood of the criterion sits at
`source_refinement`^2 points per pixel (the stored operators use the dataset's
refinement); the independent-frame counts of a re-beamformed image are those of
a stored beamformer the caller names, `skull_ct` or `skull_true` (`Scene` carries
no blood velocity, from which `fus.speckle` would derive them; the counts depend
on the beamformer only through the mix of source cells a pixel sees); the
speckle of a re-beamformed series is drawn independently of the stored series,
although a scanner would beamform the same channel data.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.scene import power_doppler_operator
from neurosim.data.types import SKULL_TRUE, Scene, TrialSet
from neurosim.fus.acquire import statistical_power_doppler
from neurosim.fus.beamform import Beamformer, make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.fus.noise import electronic_noise_variance
from neurosim.fus.operators import operator_factors
from neurosim.fus.types import GreensBank
from neurosim.recon.forward import ScreenModel, refined_source_points, screen_model_from_factors
from neurosim.recon.forward import power_doppler_operator as assemble_power_doppler_operator
from neurosim.recon.rebeamform import Weights

# Stream of the speckle of re-beamformed image series (`common.seeding.STREAMS`).
_SPECKLE_STREAM = "speckle_statistical"


@dataclass(frozen=True)
class ChannelRecord:
    """Stand-in for the recorded channel data of a scene: what re-beamforming needs.

    Attributes:
        medium: the factors of the true medium at the blood source points
            (`fus.operators.OperatorFactors` as tensors; its weights are placeholders).
        backscatter: time-averaged backscatter of every CBV cell, `mean_t cbv * cell_area`
            [m^2], float32, shape (n_cells,).
        noise_variance: variance of the electronic noise per channel sample.
    """

    medium: ScreenModel
    backscatter: torch.Tensor
    noise_variance: float


def channel_noise_variance(scene: Scene) -> float:
    """Variance of the electronic noise per channel-spectrum sample (`fus.noise`).

    A property of the electronics and the sequence, not of the medium: the bank
    passed on gives the array, the frequency grid and the ROI only.

    Raises:
        ValueError: if the scene was simulated without electronic noise; a
            signal-to-floor ratio is then undefined.
    """
    if not scene.imaging.noise.electronic_noise:
        raise ValueError("the scene has no electronic noise: a signal-to-floor ratio is undefined")
    return electronic_noise_variance(
        scene.bank_ct, imaging_setup(scene.imaging).sequence, scene.imaging.noise
    )


def record_channel_data(scene: Scene, trials: TrialSet, source_refinement: int) -> ChannelRecord:
    """Prepare the emulation of re-beamforming the scene's channel data (module docstring).

    scene, trials: the scene and its image series (the true bank and the true CBV
        are read here and stay inside the record).
    source_refinement: blood source points per pixel side of the emulation.
    """
    sequence = imaging_setup(scene.imaging).sequence
    bank = scene.bank_skull
    # Only the medium-side factors are used: the weights are replaced on every call of
    # `signal_to_floor`, so the export gets a beamformer without pixels.
    no_weights = Beamformer(
        pixel_grid=scene.pixel_grid,
        tx_weights=np.zeros((bank.freqs.size, len(sequence.angles_deg), 0), dtype=np.complex64),
        rx_weights=np.zeros((bank.freqs.size, bank.transducer.n_elements, 0), dtype=np.complex64),
        noise_gain=np.zeros(0),
    )
    factors = operator_factors(
        bank,
        scene.pixel_grid,
        refined_source_points(scene.pixel_grid, source_refinement),
        bank,
        sequence,
        scene.imaging.beamforming,
        no_weights,
    )
    mean_cbv = np.asarray(trials.cbv, dtype=np.float64).mean(axis=0).reshape(-1)
    return ChannelRecord(
        medium=screen_model_from_factors(factors, source_refinement**2),
        backscatter=torch.from_numpy((mean_cbv * scene.cell_area).astype(np.float32)),
        noise_variance=channel_noise_variance(scene),
    )


def mean_blood_power(record: ChannelRecord, weights: Weights) -> torch.Tensor:
    """Expected time-averaged blood power of the un-normalised sum at the weights' pixels.

    `sum_s q_s |sum_f P(f) (tx_phase^T D G_true) (rx_phase^T G_true)|^2`, the
    contraction of `recon.forward` with the true medium's factors.

    Returns a real tensor, shape (n_sel,), differentiable with respect to the weights.
    """
    medium = dataclasses.replace(
        record.medium, tx_weights=weights.tx_phase, rx_weights=weights.rx_phase
    )
    no_screen = torch.zeros(medium.n_elements, dtype=torch.float64)
    operator = assemble_power_doppler_operator(medium, no_screen, no_screen)
    return operator @ record.backscatter


def signal_to_floor(record: ChannelRecord, weights: Weights) -> torch.Tensor:
    """Time-averaged blood power over the beamformer's noise floor, per selected pixel.

    The ratio does not depend on how the weights are normalised: both the blood
    power and the floor carry `1 / unit_response^2`.

    Returns a real tensor, shape (n_sel,), differentiable with respect to the weights.
    """
    return mean_blood_power(record, weights) / (record.noise_variance * weights.squared_weights)


@dataclass(frozen=True)
class RebeamformedSeries:
    """The image series of a scene beamformed with a requested model, as a scanner gives it.

    Attributes:
        images: power Doppler images, float32, shape (n_images, nz, nx).
        noise_floor: electronic-noise floor of the beamformer, float32, shape (n_pix,).
        effective_frames: independent blood frames per pixel (oracle noise model;
            module docstring), float32, shape (n_pix,).
    """

    images: np.ndarray
    noise_floor: np.ndarray
    effective_frames: np.ndarray


def model_noise_floor(scene: Scene, model_bank: GreensBank) -> np.ndarray:
    """Noise floor of the beamformer of a model bank, float32, shape (n_pix,).

    `channel noise variance * Beamformer.noise_gain`, as `data.scene` computes the
    floors of the stored beamformers. Uses no ground truth.
    """
    beamformer = make_beamformer(
        scene.pixel_grid, model_bank, imaging_setup(scene.imaging).sequence,
        scene.imaging.beamforming,
    )
    return (channel_noise_variance(scene) * beamformer.noise_gain).reshape(-1).astype(np.float32)


def rebeamform_trials(
    scene: Scene,
    trials: TrialSet,
    model_bank: GreensBank,
    source_refinement: int,
    frames_like: str,
    tree: SeedTree,
    *sub: int,
) -> RebeamformedSeries:
    """The scene's image series beamformed with the weights of `model_bank` (module docstring).

    scene, trials: the scene and its series.
    model_bank: the medium the beamformer assumes.
    source_refinement: blood source points per pixel side; the dataset's value
        makes the series comparable with the stored ones.
    frames_like: key of the stored beamformer whose independent-frame counts the
        series is given (module docstring): the one whose delays are closest to
        the model's.
    tree: seed tree of the experiment; image `i` draws its speckle from the stream
        `("speckle_statistical", *sub, i)`.

    Cost: `source_refinement`^2 assemblies of the point-spread operator.
    """
    setup = imaging_setup(scene.imaging)
    operator = power_doppler_operator(
        scene.bank_skull, model_bank, setup, scene.imaging.beamforming, source_refinement
    ).astype(np.float64)
    noise_floor = model_noise_floor(scene, model_bank)
    frames = np.asarray(scene.effective_frames[frames_like], dtype=np.float32)
    shape = scene.pixel_grid.shape
    images = np.stack(
        [
            statistical_power_doppler(
                operator,
                image_cbv * scene.cell_area,
                noise_floor,
                frames,
                scene.noise_frames,
                tree.rng(_SPECKLE_STREAM, *sub, image),
            ).reshape(shape)
            for image, image_cbv in enumerate(np.asarray(trials.cbv))
        ]
    ).astype(np.float32)
    return RebeamformedSeries(images=images, noise_floor=noise_floor, effective_frames=frames)


def true_model_operator(scene: Scene) -> np.ndarray:
    """ORACLE: the power Doppler operator of the true model, `|A(true; w(true))|^2`.

    The inversion operator of the experiment's upper-bound row: what a model
    operator would be if the skull model equalled the true medium. float32, (n_pix, n_pix).
    """
    return scene.operators[SKULL_TRUE]


def true_model_bank(scene: Scene) -> GreensBank:
    """ORACLE: the bank of the true medium, as a skull model.

    The model of the experiment's upper-bound rows: what a skull model would be if
    its delays (and, where the caller keeps them, its amplitudes) were exact.
    """
    return scene.bank_skull
