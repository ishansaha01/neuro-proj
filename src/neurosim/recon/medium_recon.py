"""Reconstruction from condition averages on top of a model beamformer.

One power Doppler image is close to a single speckle look (ROADMAP D-022): the
information is in averages. This module therefore reconstructs two images per
scene, the mean of the rest images and the mean of the task images, each an
average of tens of images:

- baseline CBV map: the reconstruction of the rest mean;
- activation map: the reconstruction of the task mean minus that of the rest
  mean. For a two-level block design this is the pixelwise general linear model:
  the least-squares slope against the 0/1 task regressor is the difference of
  the condition means. A regressor shaped by a haemodynamic response would use
  the transitions better, but `Scene` and `TrialSet` carry the task labels only.

Measurement and operator. For a model bank `M` the measurement is the image
series beamformed with the weights of `M` (stored in the dataset for the
nominal CT model, re-beamformed by `recon.scanner` otherwise), and the inversion
operator is the one a scanner can compute: the power Doppler operator of a
medium that equals the model, `|A(M; w(M))|^2` (`model_operator`). The mean of
`n` images is given the likelihood of `recon.likelihood` with both frame counts
multiplied by `n` (its docstring: first two moments matched for a small task
response).

Common amplitude. The image of a beamformer normalised under `M` scales with
`(a_true / a_model)^4` if the model's one-way amplitudes are off by a common
factor, and no focus criterion can see that (`recon.focus`). It is fixed by one
scalar per model, `match_common_amplitude`: the mean blood power of the rest
image inside the brain equals what the model operator gives for a brain filled
with the reference blood volume fraction. Only a prior on the CBV scale can set
this number; the same match is applied to the nominal and the calibrated model.

Nothing here reads ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.scene import power_doppler_operator
from neurosim.data.types import Scene
from neurosim.fus.config import imaging_setup
from neurosim.fus.types import GreensBank
from neurosim.recon.baselines import gamma_deconvolution
from neurosim.recon.checkpoint import Prior
from neurosim.recon.config import DeconvolutionConfig, DpsConfig
from neurosim.recon.dps import reconstruct_trials


@dataclass(frozen=True)
class BeamformedSeries:
    """An image series of one beamformer with its noise model.

    Attributes:
        images: power Doppler images, float32, shape (n_images, nz, nx).
        noise_floor: electronic-noise floor of the beamformer, float32, shape (n_pix,).
        effective_frames: independent blood frames of every pixel of one image
            (oracle noise model, `Scene.effective_frames`), float32, shape (n_pix,).
        noise_frames: independent noise frames of one image.
    """

    images: np.ndarray
    noise_floor: np.ndarray
    effective_frames: np.ndarray
    noise_frames: float


@dataclass(frozen=True)
class ConditionMeans:
    """Rest and task averages of an image series.

    Attributes:
        rest, task: mean images, float32, shape (nz, nx).
        n_rest, n_task: number of images averaged.
    """

    rest: np.ndarray
    task: np.ndarray
    n_rest: int
    n_task: int


def condition_means(images: np.ndarray, labels: np.ndarray) -> ConditionMeans:
    """Average the images of each task state (labels: 1 = task, 0 = rest).

    images: shape (n, nz, nx); labels: shape (n,).

    Raises:
        ValueError: if either state has no image.
    """
    labels = np.asarray(labels)
    n_task = int(np.sum(labels == 1))
    n_rest = int(np.sum(labels == 0))
    if n_task == 0 or n_rest == 0:
        raise ValueError(f"need images of both states, got {n_rest} rest and {n_task} task")
    return ConditionMeans(
        rest=images[labels == 0].mean(axis=0, dtype=np.float64).astype(np.float32),
        task=images[labels == 1].mean(axis=0, dtype=np.float64).astype(np.float32),
        n_rest=n_rest,
        n_task=n_task,
    )


def model_operator(scene: Scene, model_bank: GreensBank, source_refinement: int) -> np.ndarray:
    """Power Doppler operator of a medium that equals the model, `|A(M; w(M))|^2`.

    Built by `data.scene.power_doppler_operator` with the model bank as medium and
    as beamforming model, exactly as the dataset's operators are built from the
    true bank. float32, shape (n_pix, n_pix).
    """
    return power_doppler_operator(
        model_bank,
        model_bank,
        imaging_setup(scene.imaging),
        scene.imaging.beamforming,
        source_refinement,
    )


def match_common_amplitude(
    rest_mean: np.ndarray,
    noise_floor: np.ndarray,
    operator: np.ndarray,
    brain_mask: np.ndarray,
    reference_cbv: float,
    cell_area: float,
) -> float:
    """Power scale of a model operator from the mean blood power in the brain (module docstring).

    rest_mean: mean rest image of the model's beamformer, shape (nz, nx).
    noise_floor: its floor, shape (n_pix,).
    operator: model operator, shape (n_pix, n_pix).
    brain_mask: bool, shape (nz, nx); reference_cbv: blood volume fraction assumed
    for the brain on average; cell_area: area of one CBV cell [m^2].

    Returns the factor `k` such that `k * operator` reproduces the measured mean
    blood power of the brain pixels for a brain of uniform `reference_cbv`; the
    one-way amplitude of the model is off by `k^(1/4)`.

    Raises:
        ValueError: if the mean image does not rise above the floor inside the brain.
    """
    mask = np.asarray(brain_mask, dtype=bool).reshape(-1)
    measured = float(np.sum((rest_mean.reshape(-1) - noise_floor)[mask], dtype=np.float64))
    predicted = float(
        np.sum(operator[mask] @ (mask * reference_cbv * cell_area), dtype=np.float64)
    )
    if measured <= 0.0:
        raise ValueError("the mean image is not above the noise floor inside the brain")
    return measured / predicted


def _mean_frames(series: BeamformedSeries, n_images: int) -> tuple[torch.Tensor, float]:
    """Frame counts of the mean of `n_images` images (`recon.likelihood`)."""
    frames = torch.from_numpy(np.ascontiguousarray(series.effective_frames, dtype=np.float32))
    return frames * n_images, series.noise_frames * n_images


def deconvolve_means(
    operator: torch.Tensor,
    series: BeamformedSeries,
    means: ConditionMeans,
    cell_area: float,
    cfg: DeconvolutionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma deconvolution of the rest and the task mean (`recon.baselines`).

    operator: (amplitude-matched) model operator, float32 tensor, (n_pix, n_pix).
    Returns `(rest, task)` CBV maps, float32, shape (nz, nx) each.
    """
    floor = torch.from_numpy(np.ascontiguousarray(series.noise_floor, dtype=np.float32))
    maps = []
    for image, n_images in ((means.rest, means.n_rest), (means.task, means.n_task)):
        frames, noise_frames = _mean_frames(series, n_images)
        maps.append(
            gamma_deconvolution(
                operator, floor, cell_area, frames, noise_frames, image[None], cfg
            )[0]
        )
    return maps[0], maps[1]


def sample_means(
    prior: Prior,
    operator: torch.Tensor,
    series: BeamformedSeries,
    means: ConditionMeans,
    cell_area: float,
    cfg: DpsConfig,
    tree: SeedTree,
    *sub: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion posterior samples of the rest and the task CBV map (`recon.dps`).

    The two conditions are sampled with the same random numbers (the same
    `recon_sampling` sub-indices), so the randomness of the sampler is common to
    both and largely cancels in their difference, the activation map.

    Returns `(rest, task)` samples, float32, shape (n_samples, nz, nx) each.
    """
    floor = torch.from_numpy(np.ascontiguousarray(series.noise_floor, dtype=np.float32))
    samples = []
    for image, n_images in ((means.rest, means.n_rest), (means.task, means.n_task)):
        frames, noise_frames = _mean_frames(series, n_images)
        samples.append(
            reconstruct_trials(
                prior, operator, floor, cell_area, frames, noise_frames, image[None], cfg,
                tree, *sub,
            )[:, 0]
        )
    return samples[0], samples[1]


def predicted_difference(operator: torch.Tensor, rest: np.ndarray, task: np.ndarray) -> np.ndarray:
    """Image-space task-minus-rest difference a pair of CBV maps predicts, `H (task - rest)`.

    The matched-filter template of a reconstruction (`recon.decoding`); the cell
    area is left out, since a template's scale does not matter. float64, shape (n_pix,).
    """
    difference = torch.from_numpy(np.ascontiguousarray((task - rest).reshape(-1), np.float32))
    return (operator @ difference).numpy().astype(np.float64)
