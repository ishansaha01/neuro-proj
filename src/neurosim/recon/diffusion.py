"""Variance-preserving diffusion: cosine schedule, v-prediction loss, Tweedie estimate, samplers.

Forward process (Ho et al. 2020, "Denoising diffusion probabilistic models", eq. 4):

    q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

with the cosine schedule of Nichol & Dhariwal (2021), "Improved denoising
diffusion probabilistic models", eq. 17. Timesteps are indices ``t = 0 .. T-1``;
index ``t`` is step ``t + 1`` of the papers, so ``alpha_bar[0]`` is just below 1
and ``alpha_bar[T-1]`` is about 0.

Parameterisation: the network predicts ``v = sqrt(alpha_bar_t) eps -
sqrt(1 - alpha_bar_t) x_0`` (Salimans & Ho 2022, "Progressive distillation for
fast sampling of diffusion models", section 4 and appendix D), and the training
loss is the mean squared error of v. Unlike the epsilon parameterisation, the x_0
estimate ``sqrt(alpha_bar_t) x_t - sqrt(1 - alpha_bar_t) v`` stays well
conditioned as alpha_bar goes to 0, which matters here because posterior
sampling differentiates that estimate at every noise level.

Everything works on tensors of shape (batch, ...) in model space
(`recon.transform`); a "model" is any callable ``(x_t, t) -> v`` with ``t`` an
integer tensor of shape (batch,).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.config import field

Model = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# Per-step hook of the samplers: (x_t, x0_hat, t) -> correction added to the next iterate.
Guidance = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class DiffusionConfig:
    """Noise schedule."""

    n_timesteps: int = field(
        1000, ge=2, doc="Number of diffusion steps T (Ho et al. 2020 use 1000)."
    )
    cosine_offset: float = field(
        0.008, gt=0.0, doc="Offset s of the cosine schedule (Nichol & Dhariwal 2021, eq. 17)."
    )
    max_beta: float = field(
        0.999,
        gt=0.0,
        lt=1.0,
        doc="Upper clip of beta_t (Nichol & Dhariwal 2021, section 3.2); it only acts on "
        "the last step.",
    )


def cosine_alpha_bar(cfg: DiffusionConfig) -> np.ndarray:
    """Cumulative signal fraction ``alpha_bar`` of every timestep, float64, shape (T,).

    ``alpha_bar_t = f(t) / f(0)``, ``f(t) = cos^2(((t / T + s) / (1 + s)) pi / 2)``,
    built as the cumulative product of ``1 - beta_t`` with ``beta_t`` clipped at
    `cfg.max_beta`.
    """
    steps = np.arange(cfg.n_timesteps + 1, dtype=np.float64) / cfg.n_timesteps
    f = np.cos((steps + cfg.cosine_offset) / (1.0 + cfg.cosine_offset) * np.pi / 2.0) ** 2
    betas = np.minimum(1.0 - f[1:] / f[:-1], cfg.max_beta)
    return np.cumprod(1.0 - betas)


class GaussianDiffusion:
    """The forward process of one schedule and everything derived from it.

    Attributes:
        config: the schedule configuration.
        alpha_bar: cumulative signal fraction per timestep, float64 tensor, shape (T,).
    """

    def __init__(self, cfg: DiffusionConfig):
        self.config = cfg
        self.alpha_bar = torch.from_numpy(cosine_alpha_bar(cfg))

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps T."""
        return self.config.n_timesteps

    def timestep_for_alpha_bar(self, alpha_bar: float) -> int:
        """Index of the timestep whose alpha_bar is closest to the given value."""
        return int(torch.argmin(torch.abs(self.alpha_bar - alpha_bar)))

    def _scales(
        self, t: int | torch.Tensor, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``sqrt(alpha_bar_t)`` and ``sqrt(1 - alpha_bar_t)``, broadcastable against `like`."""
        alpha_bar = self.alpha_bar[torch.as_tensor(t)]
        alpha_bar = alpha_bar.reshape(alpha_bar.shape + (1,) * (like.ndim - alpha_bar.ndim))
        return torch.sqrt(alpha_bar).to(like.dtype), torch.sqrt(1.0 - alpha_bar).to(like.dtype)

    def q_sample(
        self, x0: torch.Tensor, t: int | torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Draw ``x_t ~ q(x_t | x_0)`` given standard normal `noise`.

        x0: clean images, shape (batch, ...).
        t: timestep index, an int or an integer tensor of shape (batch,).
        noise: standard normal, same shape as `x0`.
        """
        signal_scale, noise_scale = self._scales(t, x0)
        return signal_scale * x0 + noise_scale * noise

    def v_target(
        self, x0: torch.Tensor, t: int | torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Regression target ``v = sqrt(alpha_bar_t) noise - sqrt(1 - alpha_bar_t) x0``."""
        signal_scale, noise_scale = self._scales(t, x0)
        return signal_scale * noise - noise_scale * x0

    def v_loss(
        self, model: Model, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Training loss per image: mean squared error of the v-prediction, shape (batch,).

        x0: clean images in model space, shape (batch, ...).
        t: timestep indices, integer tensor, shape (batch,).
        noise: standard normal, same shape as `x0`.
        """
        prediction = model(self.q_sample(x0, t, noise), t)
        return (prediction - self.v_target(x0, t, noise)).square().flatten(1).mean(dim=1)

    def predict_x0(self, model: Model, x_t: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Tweedie estimate ``E[x_0 | x_t]`` of the model, differentiable in `x_t`.

        model: the denoiser (v-prediction).
        x_t: noisy images, shape (batch, ...). Gradients flow from the result back
            to `x_t` (and to the model parameters if they require grad); wrap the
            call in ``torch.no_grad()`` when they are not needed.
        t: timestep index, an int or an integer tensor of shape (batch,).

        Returns ``sqrt(alpha_bar_t) x_t - sqrt(1 - alpha_bar_t) v(x_t, t)``, shape of `x_t`.
        """
        t_batch = torch.as_tensor(t).expand(x_t.shape[0])
        signal_scale, noise_scale = self._scales(t, x_t)
        return signal_scale * x_t - noise_scale * model(x_t, t_batch)

    def sampling_timesteps(self, n_steps: int) -> list[int]:
        """Timesteps visited by a sampler with `n_steps` steps: descending, T-1 down to 0."""
        if not 1 <= n_steps <= self.n_timesteps:
            raise ValueError(f"n_steps must be in 1..{self.n_timesteps}, got {n_steps}")
        if n_steps == 1:
            return [self.n_timesteps - 1]
        spaced = np.linspace(self.n_timesteps - 1, 0, n_steps)
        return [int(t) for t in np.round(spaced)]

    def step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        t_next: int,
        eta: float,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """One reverse step from timestep `t` to the earlier timestep `t_next`.

        Draws from the non-Markovian reverse kernel of Song, Meng & Ermon (2021),
        "Denoising diffusion implicit models", eq. 12, with the noise level of
        their eq. 16:

            sigma = eta sqrt((1 - ab_next) / (1 - ab_t)) sqrt(1 - ab_t / ab_next)
            x_next = sqrt(ab_next) x0_hat + sqrt(1 - ab_next - sigma^2) eps_hat + sigma noise

        where ``eps_hat = (x_t - sqrt(ab_t) x0_hat) / sqrt(1 - ab_t)``. ``eta = 1`` is the
        exact posterior ``q(x_next | x_t, x_0 = x0_hat)`` (ancestral sampling, Ho et
        al. 2020, eq. 6 and 7, for any spacing of the steps); ``eta = 0`` is
        deterministic DDIM.

        x_t: current iterate, shape (batch, ...).
        x0_hat: estimate of the clean image, same shape.
        t: current timestep index.
        t_next: next timestep index, ``-1 <= t_next < t``; -1 is the clean image
            (alpha_bar = 1), for which the step returns `x0_hat`.
        eta: noise scale in [0, 1].
        noise: standard normal, same shape; may be None when the step adds no noise
            (``eta == 0`` or ``t_next == -1``).
        """
        if not -1 <= t_next < t:
            raise ValueError(f"need -1 <= t_next < t, got t = {t}, t_next = {t_next}")
        alpha_bar = float(self.alpha_bar[t])
        alpha_bar_next = float(self.alpha_bar[t_next]) if t_next >= 0 else 1.0
        sigma = (
            eta
            * math.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar))
            * math.sqrt(1.0 - alpha_bar / alpha_bar_next)
        )
        noise_hat = (x_t - math.sqrt(alpha_bar) * x0_hat) / math.sqrt(1.0 - alpha_bar)
        direction_scale = math.sqrt(max(1.0 - alpha_bar_next - sigma**2, 0.0))
        x_next = math.sqrt(alpha_bar_next) * x0_hat + direction_scale * noise_hat
        if sigma > 0.0:
            if noise is None:
                raise ValueError("this step adds noise; pass `noise`")
            x_next = x_next + sigma * noise
        return x_next

    def sample(
        self,
        model: Model,
        shape: tuple[int, ...],
        generator: torch.Generator,
        *,
        n_steps: int,
        eta: float,
        guidance: Guidance | None = None,
        x0_clip: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Run the reverse process from pure noise.

        model: the denoiser (v-prediction).
        shape: shape of the result, (batch, ...).
        generator: source of the initial noise and of the noise of every step.
        n_steps: number of denoiser evaluations; the timesteps are
            `sampling_timesteps(n_steps)`.
        eta: 1 for ancestral sampling, 0 for DDIM (see `step`).
        guidance: optional hook ``guidance(x_t, x0_hat, t) -> correction`` called once
            per step. `x_t` requires grad and `x0_hat = predict_x0(model, x_t, t)`
            is attached to it (gradients are enabled for the call whatever the
            caller's grad mode), so the hook can differentiate a data misfit of
            `x0_hat` with respect to `x_t`; `t` is the current timestep index. The
            returned tensor (shape of `x_t`) is added to the next iterate after the
            unguided step, as in algorithm 1 of Chung et al. (2023): return
            ``-zeta * grad``. The hook sees the unclipped `x0_hat`.
        x0_clip: optional ``(low, high)`` clamp of `x0_hat` inside the unguided step
            (use ``(transform.model_floor, inf)`` to keep CBV non-negative).

        Returns the final iterate (timestep -1, the clean image), detached, shape `shape`.
        """
        timesteps = self.sampling_timesteps(n_steps)
        x = torch.randn(shape, generator=generator)
        for t, t_next in zip(timesteps, [*timesteps[1:], -1]):
            x = x.detach().requires_grad_(guidance is not None)
            with torch.set_grad_enabled(guidance is not None):
                x0_hat = self.predict_x0(model, x, t)
                correction = guidance(x, x0_hat, t) if guidance is not None else None
            x0_hat = x0_hat.detach()
            if x0_clip is not None:
                x0_hat = x0_hat.clamp(*x0_clip)
            adds_noise = eta > 0.0 and t_next >= 0
            noise = torch.randn(shape, generator=generator) if adds_noise else None
            x = self.step(x.detach(), x0_hat, t, t_next, eta, noise)
            if correction is not None:
                x = x + correction.detach()
        return x

    def ancestral_sample(
        self,
        model: Model,
        shape: tuple[int, ...],
        generator: torch.Generator,
        *,
        n_steps: int,
        guidance: Guidance | None = None,
        x0_clip: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Ancestral (DDPM) sampler: `sample` with ``eta = 1``."""
        return self.sample(
            model, shape, generator, n_steps=n_steps, eta=1.0, guidance=guidance, x0_clip=x0_clip
        )

    def ddim_sample(
        self,
        model: Model,
        shape: tuple[int, ...],
        generator: torch.Generator,
        *,
        n_steps: int,
        guidance: Guidance | None = None,
        x0_clip: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampler: `sample` with ``eta = 0``."""
        return self.sample(
            model, shape, generator, n_steps=n_steps, eta=0.0, guidance=guidance, x0_clip=x0_clip
        )
