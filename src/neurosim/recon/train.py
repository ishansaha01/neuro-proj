"""Train the diffusion prior on clean CBV images (AdamW, EMA of the weights).

`train_prior` is pure: arrays, a config and a seed tree in; trained weights and
a loss history out. File I/O lives in `recon.checkpoint` and `scripts/train_prior.py`.

Reproducibility: every random draw uses a generator from `recon.seeding`, and
the step count is fixed by the config, so for a fixed seed and thread count the
result is bit-identical on CPU. The optional wall-time limit trades this for a
bounded run time: when it stops a run early, the number of completed steps
depends on the speed of the machine.

Shared machines: with more than one thread, start Python with the environment
variable ``OMP_WAIT_POLICY=passive``. Idle OpenMP threads otherwise spin, and on
the 2-core development machine with another job running a 2-thread step took
5.5 s instead of 0.5 s (measured 2026-11-17). The setting does not change results.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neurosim.common.config import field
from neurosim.common.seeding import SeedTree
from neurosim.recon.cbv_images import PriorDataConfig
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.seeding import TRAINING_STREAM, TrainingDraw, configure_torch, torch_generator
from neurosim.recon.transform import CbvTransform
from neurosim.recon.unet import UNet, UNetConfig, build_unet

# The exponential moving average ramps up as (1 + n) / (10 + n) until it reaches
# its configured decay, so early averages are not dominated by the initial weights.
_EMA_WARMUP_OFFSET = 10.0
_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation of the prior. The values are numerical choices, not physics."""

    n_steps: int = field(2400, ge=1, doc="Number of optimiser steps.")
    batch_size: int = field(32, ge=1, doc="Images per step (drawn with replacement).")
    learning_rate: float = field(
        1.0e-3, gt=0.0, doc="AdamW learning rate after the warm-up; constant afterwards."
    )
    warmup_steps: int = field(100, ge=0, doc="Steps of linear learning-rate warm-up.")
    weight_decay: float = field(0.01, ge=0.0, doc="AdamW weight decay.")
    grad_clip_norm: float = field(1.0, gt=0.0, doc="Largest gradient norm (clipped above).")
    ema_decay: float = field(
        0.998,
        ge=0.0,
        lt=1.0,
        doc="Decay of the exponential moving average of the weights per step (an "
        "averaging horizon of 1 / (1 - decay) steps).",
    )
    eval_every: int = field(300, ge=1, doc="Steps between held-out loss evaluations.")
    eval_alpha_bars: tuple[float, ...] = field(
        (0.9, 0.5, 0.1),
        gt=0.0,
        lt=1.0,
        doc="Noise levels (alpha_bar) at which the held-out loss is logged.",
    )
    eval_images: int = field(
        256, ge=1, doc="Largest number of held-out images used per evaluation."
    )
    num_threads: int = field(
        1, ge=1, doc="Torch intra-op threads. Part of the reproducibility contract."
    )
    max_wall_minutes: float | None = field(
        None,
        gt=0.0,
        doc="Stop after this much training wall time [min] even if n_steps is not "
        "reached (the run is then no longer reproducible in its step count). None = no limit.",
    )


@dataclass(frozen=True)
class PriorConfig:
    """One prior: its training data, normalisation, network, schedule and optimisation."""

    data: PriorDataConfig = field(default_factory=PriorDataConfig, doc="Training images.")
    transform: CbvTransform = field(default_factory=CbvTransform, doc="CBV <-> model space.")
    unet: UNetConfig = field(default_factory=UNetConfig, doc="Denoiser network.")
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig, doc="Noise schedule.")
    training: TrainConfig = field(default_factory=TrainConfig, doc="Optimisation.")

    def __post_init__(self) -> None:
        if self.data.pixel_grid.n % self.unet.downsampling:
            raise ValueError(
                f"data.pixel_grid.n = {self.data.pixel_grid.n} must be divisible by the "
                f"U-Net downsampling factor {self.unet.downsampling}"
            )


@dataclass(frozen=True)
class TrainResult:
    """Outcome of `train_prior`.

    model: the network with the final raw weights (eval mode).
    ema_model: the network with the averaged weights (eval mode); the one to use.
    history: one record per held-out evaluation, see `train_prior`.
    steps_completed: optimiser steps done (below `n_steps` only if the wall-time limit hit).
    wall_seconds: training wall time [s].
    """

    model: UNet
    ema_model: UNet
    history: list[dict[str, Any]]
    steps_completed: int
    wall_seconds: float


def to_model_tensor(images: np.ndarray, transform: CbvTransform) -> torch.Tensor:
    """Physical CBV images (N, n, n) -> model-space tensor, float32, (N, 1, n, n)."""
    return torch.from_numpy(transform.to_model(np.asarray(images, dtype=np.float32)))[:, None]


def _optimiser_step(
    model: UNet,
    optimiser: torch.optim.Optimizer,
    diffusion: GaussianDiffusion,
    cfg: TrainConfig,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    optimiser.zero_grad(set_to_none=True)
    loss = diffusion.v_loss(model, x0, t, noise).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    optimiser.step()
    return float(loss.detach())


def _update_ema(ema_model: UNet, model: UNet, decay: float, step: int) -> None:
    """Move the averaged weights towards the current ones; `step` counts from 1."""
    decay = min(decay, (1.0 + step) / (_EMA_WARMUP_OFFSET + step))
    with torch.no_grad():
        for averaged, current in zip(ema_model.parameters(), model.parameters()):
            averaged.lerp_(current, 1.0 - decay)


def heldout_losses(
    model: UNet,
    diffusion: GaussianDiffusion,
    x0: torch.Tensor,
    noise: torch.Tensor,
    alpha_bars: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    """Held-out losses of a model at fixed noise levels, with fixed noise.

    x0: held-out images in model space, (N, 1, n, n).
    noise: standard normal, same shape; reused at every level.
    alpha_bars: noise levels; each is mapped to its nearest timestep.

    Returns ``{"<alpha_bar>": {"timestep", "v_loss", "x0_mse"}}``: the mean
    squared error of the v-prediction and of the Tweedie estimate of x0 (model
    space, mean over pixels and images). The two differ by the factor
    ``1 - alpha_bar_t``, because ``x0_hat - x0 = -sqrt(1 - alpha_bar_t) (v_hat - v)``.
    """
    losses = {}
    with torch.no_grad():
        for alpha_bar in alpha_bars:
            timestep = diffusion.timestep_for_alpha_bar(alpha_bar)
            t = torch.full((x0.shape[0],), timestep)
            v_loss = float(diffusion.v_loss(model, x0, t, noise).mean())
            losses[f"{alpha_bar:g}"] = {
                "timestep": timestep,
                "v_loss": v_loss,
                "x0_mse": (1.0 - float(diffusion.alpha_bar[timestep])) * v_loss,
            }
    return losses


def measure_step_seconds(cfg: PriorConfig, tree: SeedTree, n_steps: int = 3) -> float:
    """Wall time of one optimiser step [s], measured on a throwaway model.

    Runs one untimed and `n_steps` timed steps on a fixed non-random batch of the
    configured size, so it needs no data and draws from no random stream.
    """
    configure_torch(cfg.training.num_threads)
    model = build_unet(cfg.unet, torch_generator(tree, TRAINING_STREAM, TrainingDraw.WEIGHT_INIT))
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    diffusion = GaussianDiffusion(cfg.diffusion)
    batch, n = cfg.training.batch_size, cfg.data.pixel_grid.n
    # A ramp over [-1, 1]: all-zero inputs would run faster than real images.
    x0 = torch.linspace(-1.0, 1.0, batch * n * n).reshape(batch, 1, n, n)
    noise = x0.flip(0)
    t = torch.arange(batch) % diffusion.n_timesteps
    _optimiser_step(model, optimiser, diffusion, cfg.training, x0, t, noise)
    start = time.perf_counter()
    for _ in range(n_steps):
        _optimiser_step(model, optimiser, diffusion, cfg.training, x0, t, noise)
    return (time.perf_counter() - start) / n_steps


def train_prior(
    cfg: PriorConfig,
    train_images: np.ndarray,
    heldout_images: np.ndarray,
    tree: SeedTree,
    on_eval: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train the denoiser.

    cfg: the prior configuration.
    train_images: physical CBV fraction, float32, (N_train, n, n).
    heldout_images: physical CBV fraction of held-out heads, float32, (N_heldout, n, n);
        used for logging only.
    tree: seed tree; stream `diffusion_training` with the sub-indices of
        `recon.seeding.TrainingDraw`.
    on_eval: optional callback given every history record as it is produced.

    A history record is ``{"step", "train_loss", "heldout"}``: the number of steps
    done, the mean training loss since the previous record, and
    `heldout_losses` of the averaged (EMA) weights.
    """
    train_cfg = cfg.training
    configure_torch(train_cfg.num_threads)
    diffusion = GaussianDiffusion(cfg.diffusion)
    x_train = to_model_tensor(train_images, cfg.transform)
    x_heldout = to_model_tensor(heldout_images[: train_cfg.eval_images], cfg.transform)
    heldout_noise = torch.randn(
        x_heldout.shape,
        generator=torch_generator(tree, TRAINING_STREAM, TrainingDraw.HELDOUT_NOISE),
    )

    model = build_unet(cfg.unet, torch_generator(tree, TRAINING_STREAM, TrainingDraw.WEIGHT_INIT))
    ema_model = copy.deepcopy(model).requires_grad_(False)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    batches = torch_generator(tree, TRAINING_STREAM, TrainingDraw.BATCHES)
    wall_limit = (
        float("inf")
        if train_cfg.max_wall_minutes is None
        else train_cfg.max_wall_minutes * _SECONDS_PER_MINUTE
    )

    history: list[dict[str, Any]] = []
    recent_losses: list[float] = []
    steps_completed = 0
    start = time.perf_counter()
    model.train()
    for step in range(1, train_cfg.n_steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = train_cfg.learning_rate * min(1.0, step / (train_cfg.warmup_steps + 1))
        index = torch.randint(0, x_train.shape[0], (train_cfg.batch_size,), generator=batches)
        t = torch.randint(0, diffusion.n_timesteps, (train_cfg.batch_size,), generator=batches)
        x0 = x_train[index]
        noise = torch.randn(x0.shape, generator=batches)
        recent_losses.append(_optimiser_step(model, optimiser, diffusion, train_cfg, x0, t, noise))
        _update_ema(ema_model, model, train_cfg.ema_decay, step)
        steps_completed = step

        out_of_time = time.perf_counter() - start > wall_limit
        if step % train_cfg.eval_every == 0 or step == train_cfg.n_steps or out_of_time:
            record = {
                "step": step,
                "train_loss": float(np.mean(recent_losses)),
                "heldout": heldout_losses(
                    ema_model, diffusion, x_heldout, heldout_noise, train_cfg.eval_alpha_bars
                ),
            }
            history.append(record)
            recent_losses = []
            if on_eval is not None:
                on_eval(record)
        if out_of_time:
            break

    return TrainResult(
        model=model.eval().requires_grad_(False),
        ema_model=ema_model.eval(),
        history=history,
        steps_completed=steps_completed,
        wall_seconds=time.perf_counter() - start,
    )
