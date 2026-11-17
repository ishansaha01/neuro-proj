"""E-RECON-002: the forward process q(x_t | x_0) has the mean and variance of the schedule."""

from __future__ import annotations

import numpy as np
import torch

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.seeding import TRAINING_STREAM, torch_generator

EVAL_ID = "E-RECON-002"
# First sub-index of the eval's draws in the `diffusion_training` stream (training
# itself uses the single sub-indices of `recon.seeding.TrainingDraw`).
STREAM_SUB = 1002
N_SAMPLES = 20_000
IMAGE_SHAPE = (1, 16, 16)
# Timestep indices spread over the schedule. The last index is left out: its beta is
# clipped (Nichol & Dhariwal 2021, section 3.2), so the closed form does not hold there.
TIMESTEPS = (0, 99, 399, 699, 949)
MAX_ABS_Z = 4.0


def _closed_form_alpha_bar(t: int, cfg: DiffusionConfig) -> float:
    """alpha_bar of timestep index t (step t + 1 of the paper): f(t + 1) / f(0), eq. 17."""

    def f(step: int) -> float:
        fraction = step / cfg.n_timesteps
        return np.cos((fraction + cfg.cosine_offset) / (1.0 + cfg.cosine_offset) * np.pi / 2.0) ** 2

    return f(t + 1) / f(0)


@eval_case(
    id=EVAL_ID,
    claim="Samples of the forward process q(x_t | x_0) have mean sqrt(alpha_bar_t) x_0 and "
    "variance 1 - alpha_bar_t, with alpha_bar_t of the cosine schedule",
    reference="Analytic: alpha_bar_t = f(t)/f(0), f(t) = cos^2(((t/T + s)/(1 + s)) pi/2), "
    "s = 0.008 (Nichol & Dhariwal 2021, eq. 17), evaluated in closed form in the eval",
    metric="Largest |z| over 5 timesteps of three statistics of 20000 samples of a 16 x 16 "
    "image: regression slope of x_t on x_0 against sqrt(alpha_bar_t), mean residual against "
    "0, residual variance against 1 - alpha_bar_t, each in units of its standard error",
    threshold="< 4 standard errors (fixed in advance: with 15 statistics an exact "
    "implementation exceeds 4 with probability 1e-3, while the standard errors, 0.06 % of "
    "the variance and at most 0.05 % of the slope, expose any wrong coefficient)",
    seeds=(0,),
)
def test_e_recon_002_forward_process():
    cfg = DiffusionConfig()
    diffusion = GaussianDiffusion(cfg)
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    x0 = torch.randn(IMAGE_SHAPE, generator=torch_generator(tree, TRAINING_STREAM, STREAM_SUB, 0))
    x0_flat = x0.flatten().to(torch.float64)
    n_values = N_SAMPLES * x0_flat.numel()

    z_scores: dict[str, float] = {}
    for index, t in enumerate(TIMESTEPS):
        noise = torch.randn(
            (N_SAMPLES, *IMAGE_SHAPE),
            generator=torch_generator(tree, TRAINING_STREAM, STREAM_SUB, 1 + index),
        )
        x_t = diffusion.q_sample(x0.expand(N_SAMPLES, *IMAGE_SHAPE), t, noise)
        x_t = x_t.flatten(1).to(torch.float64)

        alpha_bar = _closed_form_alpha_bar(t, cfg)
        noise_variance = 1.0 - alpha_bar
        slope = float((x_t @ x0_flat).mean() / (x0_flat @ x0_flat))
        slope_error = np.sqrt(noise_variance / (N_SAMPLES * float(x0_flat @ x0_flat)))
        residual = x_t - np.sqrt(alpha_bar) * x0_flat
        z_scores[f"slope_t{t}"] = (slope - np.sqrt(alpha_bar)) / slope_error
        z_scores[f"offset_t{t}"] = float(residual.mean()) / np.sqrt(noise_variance / n_values)
        z_scores[f"variance_t{t}"] = (float(residual.square().mean()) / noise_variance - 1.0) / (
            np.sqrt(2.0 / n_values)
        )

    worst = max(abs(z) for z in z_scores.values())
    report(EVAL_ID, {"max_abs_z": worst, **z_scores}, passed=worst < MAX_ABS_Z)
