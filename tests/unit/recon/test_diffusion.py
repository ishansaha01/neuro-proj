import itertools
import math

import numpy as np
import pytest
import torch

from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion, cosine_alpha_bar

CFG = DiffusionConfig(n_timesteps=100)
DIFFUSION = GaussianDiffusion(CFG)
SHAPE = (4, 1, 4, 4)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _linear_model(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A v-prediction that depends on both arguments (not a meaningful denoiser)."""
    return 0.3 * x_t + 0.001 * t.reshape(-1, 1, 1, 1)


def test_cosine_schedule_matches_the_closed_form():
    alpha_bar = cosine_alpha_bar(CFG)
    steps = np.arange(CFG.n_timesteps + 1) / CFG.n_timesteps
    s = CFG.cosine_offset
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    assert alpha_bar.shape == (CFG.n_timesteps,) and alpha_bar.dtype == np.float64
    # Only the last beta is clipped, so all but the last entry telescope to f(t) / f(0).
    np.testing.assert_allclose(alpha_bar[:-1], f[1:-1] / f[0], rtol=1e-12)
    assert alpha_bar[-1] == pytest.approx(alpha_bar[-2] * (1.0 - CFG.max_beta))
    assert np.all(np.diff(alpha_bar) < 0.0)
    assert 0.99 < alpha_bar[0] < 1.0 and 0.0 < alpha_bar[-1] < 1e-4


def test_timestep_for_alpha_bar():
    assert DIFFUSION.n_timesteps == 100
    t = DIFFUSION.timestep_for_alpha_bar(0.5)
    assert abs(float(DIFFUSION.alpha_bar[t]) - 0.5) < 0.02
    assert DIFFUSION.timestep_for_alpha_bar(1.0) == 0
    assert DIFFUSION.timestep_for_alpha_bar(0.0) == 99


def test_q_sample_accepts_int_and_per_image_timesteps():
    x0 = torch.randn(SHAPE, generator=_generator())
    noise = torch.randn(SHAPE, generator=_generator(1))
    t = torch.tensor([3, 40, 40, 99])
    x_t = DIFFUSION.q_sample(x0, t, noise)
    assert x_t.shape == SHAPE and x_t.dtype == torch.float32
    alpha_bar = float(DIFFUSION.alpha_bar[40])
    expected = math.sqrt(alpha_bar) * x0[1] + math.sqrt(1 - alpha_bar) * noise[1]
    assert torch.allclose(x_t[1], expected, atol=1e-6)
    assert torch.equal(DIFFUSION.q_sample(x0, 40, noise)[1], x_t[1])


def test_true_v_recovers_x0_and_gives_zero_loss():
    x0 = torch.randn(SHAPE, generator=_generator())
    noise = torch.randn(SHAPE, generator=_generator(1))
    t = torch.tensor([0, 20, 60, 99])
    x_t = DIFFUSION.q_sample(x0, t, noise)
    true_v = DIFFUSION.v_target(x0, t, noise)

    def oracle(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        assert torch.equal(x, x_t) and torch.equal(steps, t)
        return true_v

    assert torch.allclose(DIFFUSION.predict_x0(oracle, x_t, t), x0, atol=1e-5)
    loss = DIFFUSION.v_loss(oracle, x0, t, noise)
    assert loss.shape == (4,) and torch.allclose(loss, torch.zeros(4), atol=1e-12)
    zero_loss = DIFFUSION.v_loss(lambda x, steps: torch.zeros_like(x), x0, t, noise)
    assert torch.allclose(zero_loss, true_v.square().flatten(1).mean(dim=1))


def test_predict_x0_is_differentiable_in_x_t():
    x_t = torch.randn(SHAPE, generator=_generator(), dtype=torch.float64, requires_grad=True)
    t = 30
    x0_hat = DIFFUSION.predict_x0(_linear_model, x_t, t)
    (gradient,) = torch.autograd.grad(x0_hat.sum(), x_t)
    alpha_bar = float(DIFFUSION.alpha_bar[t])
    slope = math.sqrt(alpha_bar) - 0.3 * math.sqrt(1.0 - alpha_bar)
    assert torch.allclose(gradient, torch.full(SHAPE, slope, dtype=torch.float64), rtol=1e-6)


def test_sampling_timesteps():
    assert DIFFUSION.sampling_timesteps(100) == list(range(99, -1, -1))
    strided = DIFFUSION.sampling_timesteps(7)
    assert len(strided) == 7 and strided[0] == 99 and strided[-1] == 0
    assert all(later < earlier for earlier, later in itertools.pairwise(strided))
    assert DIFFUSION.sampling_timesteps(1) == [99]
    for bad in (0, 101):
        with pytest.raises(ValueError, match="n_steps"):
            DIFFUSION.sampling_timesteps(bad)


def test_ancestral_step_is_the_ddpm_posterior():
    """eta = 1 between neighbouring steps equals Ho et al. (2020), eq. 6 and 7."""
    x_t = torch.randn(SHAPE, generator=_generator(), dtype=torch.float64)
    x0 = torch.randn(SHAPE, generator=_generator(1), dtype=torch.float64)
    t = 37
    alpha_bar, alpha_bar_prev = (float(DIFFUSION.alpha_bar[i]) for i in (t, t - 1))
    beta = 1.0 - alpha_bar / alpha_bar_prev
    mean = (
        math.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar) * x0
        + math.sqrt(1.0 - beta) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * x_t
    )
    variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta
    assert torch.allclose(DIFFUSION.step(x_t, x0, t, t - 1, 1.0, torch.zeros_like(x_t)), mean)
    with_noise = DIFFUSION.step(x_t, x0, t, t - 1, 1.0, torch.ones_like(x_t))
    assert torch.allclose(with_noise - mean, torch.full_like(mean, math.sqrt(variance)))


def test_ddim_step_keeps_the_predicted_noise():
    x0 = torch.randn(SHAPE, generator=_generator(), dtype=torch.float64)
    noise = torch.randn(SHAPE, generator=_generator(1), dtype=torch.float64)
    x_t = DIFFUSION.q_sample(x0, 80, noise)
    x_next = DIFFUSION.step(x_t, x0, 80, 25, 0.0, None)
    assert torch.allclose(x_next, DIFFUSION.q_sample(x0, 25, noise))


def test_last_step_returns_the_estimate_and_bad_steps_are_rejected():
    x_t = torch.randn(SHAPE, generator=_generator())
    x0 = torch.randn(SHAPE, generator=_generator(1))
    assert torch.allclose(DIFFUSION.step(x_t, x0, 0, -1, 1.0, None), x0, atol=1e-6)
    with pytest.raises(ValueError, match="t_next"):
        DIFFUSION.step(x_t, x0, 5, 5, 1.0, None)
    with pytest.raises(ValueError, match="t_next"):
        DIFFUSION.step(x_t, x0, 5, -2, 1.0, None)
    with pytest.raises(ValueError, match="pass `noise`"):
        DIFFUSION.step(x_t, x0, 5, 4, 1.0, None)


def test_samplers_are_reproducible_and_named_by_eta():
    def run(sampler, seed, **kwargs):
        return sampler(_linear_model, SHAPE, _generator(seed), n_steps=12, **kwargs)

    ancestral = run(DIFFUSION.ancestral_sample, 3)
    ddim = run(DIFFUSION.ddim_sample, 3)
    assert ancestral.shape == SHAPE and not ancestral.requires_grad
    assert torch.equal(ancestral, run(DIFFUSION.ancestral_sample, 3))
    assert torch.equal(ancestral, run(DIFFUSION.sample, 3, eta=1.0))
    assert torch.equal(ddim, run(DIFFUSION.sample, 3, eta=0.0))
    assert not torch.equal(ancestral, ddim)
    assert not torch.equal(ancestral, run(DIFFUSION.ancestral_sample, 4))


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_guidance_hook_sees_a_differentiable_estimate_and_shifts_the_iterate(eta):
    calls = []
    offset = 0.25

    def guidance(x_t: torch.Tensor, x0_hat: torch.Tensor, t: int) -> torch.Tensor:
        assert x_t.requires_grad and x0_hat.requires_grad
        (gradient,) = torch.autograd.grad(x0_hat.sum(), x_t)
        alpha_bar = float(DIFFUSION.alpha_bar[t])
        slope = math.sqrt(alpha_bar) - 0.3 * math.sqrt(1.0 - alpha_bar)
        assert torch.allclose(gradient, torch.full(SHAPE, slope), atol=1e-6)
        calls.append(t)
        # Shift only the final iterate, so the effect on the result is known exactly.
        return torch.full(SHAPE, offset if t == 0 else 0.0)

    def run(hook):
        return DIFFUSION.sample(
            _linear_model, SHAPE, _generator(5), n_steps=9, eta=eta, guidance=hook
        )

    plain = run(None)
    with torch.no_grad():  # the sampler enables gradients for the hook by itself
        guided = run(guidance)
    assert calls == DIFFUSION.sampling_timesteps(9)
    assert not guided.requires_grad
    assert torch.allclose(guided - plain, torch.full(SHAPE, offset), atol=1e-6)


def test_x0_clip_bounds_the_result():
    def run(clip):
        return DIFFUSION.ddim_sample(
            lambda x, t: -3.0 * x, SHAPE, _generator(6), n_steps=5, x0_clip=clip
        )

    assert run(None).abs().max() > 0.5
    clipped = run((-0.5, 0.5))
    assert clipped.min() >= -0.5 and clipped.max() <= 0.5
