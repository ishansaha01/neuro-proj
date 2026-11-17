"""Diffusion posterior sampling (DPS; Chung et al. 2023, ICLR, "Diffusion posterior
sampling for general noisy inverse problems") on top of the prior's samplers.

DPS approximates the intractable likelihood of a noisy iterate by the likelihood
of its denoised estimate, ``p(y | x_t) ~ p(y | x0_hat(x_t))``, and adds the
gradient of its logarithm with respect to ``x_t`` (taken through the denoiser) to
every reverse step. Here the gradient enters through the guidance hook of
`recon.diffusion.GaussianDiffusion.sample`.

Step-size rule (`likelihood_guidance`). The posterior score is the prior score
plus ``g = grad_{x_t} log p(y | x0_hat(x_t))``. By Tweedie's formula adding ``g`` to
the score shifts the denoised estimate by ``(1 - ab_t) / sqrt(ab_t) * g``, and
the reverse step of `GaussianDiffusion.step` carries a shift of its ``x0_hat``
into the next iterate with the factor ``sqrt(ab_next) - c sqrt(ab_t) /
sqrt(1 - ab_t)``, ``c = sqrt(1 - ab_next - sigma^2)``. The correction is therefore

    weight(t) * g,   weight = sqrt(ab_next) (1 - ab_t) / sqrt(ab_t) - c sqrt(1 - ab_t)

(for single ancestral steps this is ``beta_t / sqrt(alpha_t)``, the usual score
step), times `guidance_scale` (1 = the weight the DPS approximation prescribes).
Because ``p(y | x0_hat)`` ignores the spread of ``x_0`` given ``x_t``, this weight
is far too large at high noise levels when the measurement is informative, so the
correction of each image is shortened to at most `max_step_ratio` times the
expected length of the unguided step (a trust region). Chung et al. instead
normalise by the residual norm; that rule does not vanish when the measurement
is uninformative, the rule here does (eval E-RECON-007).

Expected bias. (1) The DPS approximation itself: samples are not exact posterior
samples even with a perfect prior, and their spread is typically too small.
(2) Few steps: the discretisation bias of the sampler (E-RECON-003 quotes -2.4 %
of the variance at 100 DDIM steps and -4.9 % at 50 for a Gaussian) grows, and the
guidance has fewer steps to act. (3) The trust region under-weights the data at
high noise levels, which pulls coarse structure toward the prior. All three are
measured together by the calibration metric of `recon.metrics`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.recon.checkpoint import Prior
from neurosim.recon.config import DpsConfig
from neurosim.recon.diffusion import GaussianDiffusion, Guidance, Model
from neurosim.recon.forward import expected_power_doppler
from neurosim.recon.likelihood import power_doppler_log_likelihood
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator
from neurosim.recon.transform import CbvTransform

# Maps a batch of denoised images in model space, shape (batch, ...), to the log-likelihood
# of each image's measurement, shape (batch,).
LogLikelihood = Callable[[torch.Tensor], torch.Tensor]


def _step_coefficients(
    diffusion: GaussianDiffusion, t: int, t_next: int, eta: float
) -> tuple[float, float, float, float]:
    """``(sqrt(ab_t), sqrt(ab_next), c, sigma)`` of the reverse step from `t` to `t_next`."""
    alpha_bar = float(diffusion.alpha_bar[t])
    alpha_bar_next = float(diffusion.alpha_bar[t_next]) if t_next >= 0 else 1.0
    sigma = (
        eta
        * math.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar))
        * math.sqrt(1.0 - alpha_bar / alpha_bar_next)
    )
    direction = math.sqrt(max(1.0 - alpha_bar_next - sigma**2, 0.0))
    return math.sqrt(alpha_bar), math.sqrt(alpha_bar_next), direction, sigma


def next_timesteps(diffusion: GaussianDiffusion, n_steps: int) -> dict[int, int]:
    """Map from every timestep a sampler with `n_steps` steps visits to the one that follows it."""
    timesteps = diffusion.sampling_timesteps(n_steps)
    return dict(zip(timesteps, [*timesteps[1:], -1]))


def guidance_correction(
    gradient: torch.Tensor,
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    diffusion: GaussianDiffusion,
    t: int,
    t_next: int,
    cfg: DpsConfig,
) -> torch.Tensor:
    """Correction of the next iterate for a likelihood score (the rule of the module docstring).

    gradient: ``grad_{x_t} log p(y | x0_hat(x_t))``, shape of `x_t` (batch, ...).
    x_t, x0_hat: current iterate and its denoised estimate.
    t, t_next: the reverse step being taken.
    cfg: `eta`, `guidance_scale` and `max_step_ratio`.
    """
    root_ab, root_ab_next, direction, sigma = _step_coefficients(diffusion, t, t_next, cfg.eta)
    noise_scale = math.sqrt(1.0 - root_ab**2)
    weight = root_ab_next * noise_scale**2 / root_ab - direction * noise_scale
    correction = cfg.guidance_scale * weight * gradient
    with torch.no_grad():
        noise_hat = (x_t - root_ab * x0_hat) / noise_scale
        unguided = root_ab_next * x0_hat + direction * noise_hat - x_t
        step_length = torch.sqrt(
            unguided.flatten(1).square().sum(dim=1) + sigma**2 * x_t[0].numel()
        )
        length = correction.flatten(1).norm(dim=1).clamp_min(torch.finfo(x_t.dtype).tiny)
        shrink = torch.clamp(cfg.max_step_ratio * step_length / length, max=1.0)
    return correction * shrink.reshape(-1, *([1] * (x_t.ndim - 1)))


def likelihood_guidance(
    log_likelihood: LogLikelihood, diffusion: GaussianDiffusion, cfg: DpsConfig
) -> Guidance:
    """Guidance hook that adds the likelihood score of the denoised estimate.

    log_likelihood: log-likelihood of each image's measurement given the denoised image.
    diffusion: the schedule the sampler runs on.
    cfg: `n_steps` and `eta` must be those passed to the sampler.
    """
    following = next_timesteps(diffusion, cfg.n_steps)

    def guidance(x_t: torch.Tensor, x0_hat: torch.Tensor, t: int) -> torch.Tensor:
        (gradient,) = torch.autograd.grad(log_likelihood(x0_hat).sum(), x_t)
        return guidance_correction(gradient, x_t, x0_hat, diffusion, t, following[t], cfg)

    return guidance


def posterior_sample(
    model: Model,
    diffusion: GaussianDiffusion,
    log_likelihood: LogLikelihood | None,
    shape: tuple[int, ...],
    generator: torch.Generator,
    cfg: DpsConfig,
    x0_clip: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Draw one batch of (approximate) posterior samples in model space.

    model: the denoiser of the prior (v-prediction).
    diffusion: its schedule.
    log_likelihood: data term (see `LogLikelihood`); None samples the prior alone
        with the same sampler settings.
    shape: (batch, ...) of the result.
    generator: from `recon.seeding.torch_generator` on the `recon_sampling` stream.
    cfg: sampler and guidance settings.
    x0_clip: clamp of the denoised estimate inside the unguided step.

    Returns the samples, detached, shape `shape`.
    """
    guidance = None
    if log_likelihood is not None:
        guidance = likelihood_guidance(log_likelihood, diffusion, cfg)
    return diffusion.sample(
        model,
        shape,
        generator,
        n_steps=cfg.n_steps,
        eta=cfg.eta,
        guidance=guidance,
        x0_clip=x0_clip,
    )


def cbv_log_likelihood(
    transform: CbvTransform,
    operator: torch.Tensor,
    cell_area: float,
    noise_floor: torch.Tensor,
    measured: torch.Tensor,
    blood_frames: torch.Tensor,
    noise_frames: float,
) -> LogLikelihood:
    """Data term of power Doppler images for denoised CBV images in model space.

    The denoised image is mapped to physical CBV (`transform.to_physical`, negative
    values set to zero), pushed through the forward model
    (`recon.forward.expected_power_doppler`) and scored with
    `recon.likelihood.power_doppler_log_likelihood`.

    transform: the prior's map between model space and CBV.
    operator: power Doppler operator, shape (n_selected, n_cells).
    cell_area: area of one CBV cell [m^2].
    noise_floor: floor of the selected pixels, shape (n_selected,).
    measured: measured images, shape (batch, n_selected).
    blood_frames: independent blood frames of the selected pixels
        (`Scene.effective_frames` of the measured image), shape (n_selected,).
    noise_frames: independent noise frames (`Scene.noise_frames`).
    """

    def log_likelihood(x0_hat: torch.Tensor) -> torch.Tensor:
        cbv = transform.to_physical(x0_hat).clamp_min(0.0).flatten(1)
        expected = expected_power_doppler(operator, cbv, cell_area, noise_floor)
        return power_doppler_log_likelihood(
            measured, expected, noise_floor, blood_frames, noise_frames
        )

    return log_likelihood


def _to_cbv(prior: Prior, samples: torch.Tensor) -> np.ndarray:
    """Model-space samples (batch, 1, nz, nx) -> non-negative CBV, float32, (batch, nz, nx)."""
    return prior.transform.to_physical(samples[:, 0]).clamp_min(0.0).numpy().astype(np.float32)


def reconstruct_trials(
    prior: Prior,
    operator: torch.Tensor,
    noise_floor: torch.Tensor,
    cell_area: float,
    blood_frames: torch.Tensor,
    noise_frames: float,
    measured: np.ndarray,
    cfg: DpsConfig,
    tree: SeedTree,
    *sub: int,
) -> np.ndarray:
    """Posterior samples of the CBV image of every trial for a fixed forward operator.

    All trials are sampled in one batch; sample number `k` of every trial uses the
    generator ``(recon_sampling, *sub, k)``, so the result depends on the seed,
    `sub` and the set of trials only.

    prior: the trained prior (`recon.checkpoint.load_prior`).
    operator: power Doppler operator `H`, float32, shape (n_pix, n_pix).
    noise_floor: float32, shape (n_pix,).
    cell_area: area of one pixel [m^2].
    blood_frames: independent blood frames of every pixel of the measured image
        (`Scene.effective_frames` of its beamformer key), float32, shape (n_pix,).
    noise_frames: independent noise frames (`Scene.noise_frames`).
    measured: power Doppler images, shape (n_trials, nz, nx).
    cfg: sampler settings; `cfg.n_samples` samples per trial.
    tree: seed tree of the reconstruction.
    *sub: sub-indices of the `recon_sampling` stream (method, scene, ...).

    Returns CBV fraction, float32, shape (n_samples, n_trials, nz, nx).
    """
    n_trials = measured.shape[0]
    shape = (n_trials, 1, *measured.shape[1:])
    log_likelihood = cbv_log_likelihood(
        prior.transform,
        operator,
        cell_area,
        noise_floor,
        torch.from_numpy(np.ascontiguousarray(measured, dtype=np.float32)).flatten(1),
        blood_frames,
        noise_frames,
    )
    clip = (prior.transform.model_floor, math.inf)
    samples = [
        _to_cbv(
            prior,
            posterior_sample(
                prior.model,
                prior.diffusion,
                log_likelihood,
                shape,
                torch_generator(tree, SAMPLING_STREAM, *sub, k),
                cfg,
                clip,
            ),
        )
        for k in range(cfg.n_samples)
    ]
    return np.stack(samples)


def prior_samples(
    prior: Prior, n_images: int, cfg: DpsConfig, tree: SeedTree, *sub: int
) -> np.ndarray:
    """Samples of the prior alone, with the sampler settings of the reconstruction.

    The control for hallucination: these images have seen no measurement.

    Returns CBV fraction, float32, shape (n_images, nz, nx) on the prior's pixel grid.
    """
    shape = (n_images, 1, *prior.pixel_grid.shape)
    clip = (prior.transform.model_floor, math.inf)
    generator = torch_generator(tree, SAMPLING_STREAM, *sub)
    return _to_cbv(
        prior, posterior_sample(prior.model, prior.diffusion, None, shape, generator, cfg, clip)
    )
