"""Calibrate the skull phase screen once per scene, from the whole trial set.

The skull is static, so the screen `phi` (per-element delay and amplitude) is
estimated once from the time-averaged `skull_das` image and then held fixed
while every trial is reconstructed (`recon.dps.reconstruct_trials` with
`H(phi_hat)`).

Method: blind diffusion posterior sampling in the sense of Chung et al. (2023,
CVPR, "Parallel diffusion models of operator and image for blind inverse
problems", BlindDPS): within one reverse process the image iterate and the
operator parameters are both moved by the same data term,
``log p(y_mean | x0_hat(x_t), phi)``. Differences from that paper, stated plainly:

- `phi` has no diffusion prior of its own but a Gaussian prior about the CT
  screen, so there is no second reverse process. It is updated at a few reverse
  steps by L-BFGS on ``log-likelihood + log prior`` with the denoised images held
  fixed (alternating updates). Plain gradient or Adam steps did not work: with
  the true image given, 200 Adam steps left the delay error unchanged where 60
  L-BFGS evaluations reduced it five-fold (the problem is badly conditioned).
- Several image chains share one `phi`, which averages its objective over the
  image uncertainty.
- The image is guided by all pixels; the screen objective uses a random subset
  of pixels, for run time.
- The delay updates use a gamma shape of at most
  `CalibrationConfig.delay_fit_shape`, far below that of the time-averaged
  image: the denoised image the delays are fitted to is uncertain in its fine
  structure, and with the full shape the Gaussian prior had no weight and single
  scenes ended several times further from the true delays than they started.
  The amplitude update keeps the full shape: it fits one global level.

After CryoBoltz (Raghu et al. 2025): *coarse-to-fine guidance*: the delays are
expanded in aperture modes ordered from smooth to rough (a DCT across the
elements, gauge directions removed). The first update, at the first reverse
step where the denoised image is the prior mean, moves the common amplitude only
(the prior knows the level of CBV); the following updates add delay modes from
the smoothest on, and the screen is frozen for the later part of the reverse
process (`CalibrationConfig.update_until`). Per-element amplitudes and delay
modes beyond `max_delay_modes` stay at the CT values. *Best-of-N
selection*: N independent runs, of which the one whose final images and screen
fit the data best (lowest gamma deviance over all pixels) is kept.

What limits it: a wrong screen can be absorbed by a wrong image. Alternating
"sample images given `phi`, fit `phi` given the images" after the reverse process
does not move `phi` toward the truth for that reason, and updates late in the
reverse process, when the image iterate has adapted to the current screen,
increased the delay error again on synthetic development scenes. Early in the
reverse process the prior still holds the image and the residual goes to `phi`,
but there the denoised image is a blurred posterior mean, which biases the fit.
How much of the delay error is recovered therefore depends on how much sharp
structure the prior asserts (eval E-RECON-008 measures it without model mismatch).

Gauge freedoms: directions of `phi` that the power Doppler image cannot
separate from a change of the unknown CBV image.

1. *Common delay.* The same delay on all elements shifts every echo in time,
   i.e. moves the image in depth by ``c * delay`` (0.15 mm per 100 ns): the same
   data as a shifted CBV image.
2. *Linear tilt* of the delays across the aperture steers transmit and receive
   alike, i.e. rotates the image about the array centre: the same data as a
   rotated CBV image.
3. *Common amplitude.* A factor ``a`` on all elements scales `H` by ``a^4``: the
   same data as the CBV image divided by ``a^4``.
4. A delay of a whole number of carrier periods is nearly invisible for a
   narrow-band pulse (cycle skipping); it is a discrete ambiguity, not a gauge.

1 and 2 are exact gauges only for an image prior without preferred position,
and their effect is below a pixel for the delays in question, so the data cannot
fix them: they are held at the CT values by construction
(`recon.gauge.gauge_free_basis`; the choice between 1 and 1 + 2 is
`CalibrationConfig.gauge`), and `recon.gauge.remove_gauge` takes them out before
two screens are compared. 3 is not fixed:
the diffusion prior knows the absolute level of CBV, so the common amplitude is
estimated, with its own (wide) prior; what is identified is only as good as the
prior's knowledge of that level. 4 is kept out by the prior width (100 ns against
a 500 ns period at 2 MHz).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from neurosim.common.seeding import SeedTree
from neurosim.data.types import PhaseScreen
from neurosim.recon.checkpoint import Prior
from neurosim.recon.config import CalibrationConfig
from neurosim.recon.dps import guidance_correction, next_timesteps
from neurosim.recon.forward import (
    ScreenModel,
    expected_power_doppler,
    power_doppler_operator,
    screen_tensors,
)
from neurosim.recon.gauge import gauge_free_basis
from neurosim.recon.likelihood import gamma_deviance, gamma_log_likelihood
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator

# A screen as tensors: delays [s] and log amplitudes, each of shape (n_elements,).
Screen = tuple[torch.Tensor, torch.Tensor]
# Sub-indices of a calibration's draws within its `recon_sampling` sub-stream.
_PIXEL_DRAW = 0
_FIRST_CANDIDATE_DRAW = 1


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of `calibrate_screen`.

    Attributes:
        screen: the selected screen estimate `phi_hat`.
        candidates: the final screen of every candidate run.
        deviance_per_pixel: final data fit of every candidate: gamma deviance of the
            time-averaged image over all pixels, per pixel, averaged over the candidate's
            image chains (about 1 for a fit at the speckle level).
        start_deviance_per_pixel: the same statistic for the first candidate's final images
            with the starting (CT) screen, for comparison.
        selected: index of the selected candidate.
    """

    screen: PhaseScreen
    candidates: tuple[PhaseScreen, ...]
    deviance_per_pixel: tuple[float, ...]
    start_deviance_per_pixel: float
    selected: int


def update_schedule(cfg: CalibrationConfig) -> dict[int, int]:
    """Reverse steps (0-based) at which the screen is updated, with what each update moves.

    The updates are evenly spaced from step 0 to the fraction `cfg.update_until`
    of the run. The value is the number of delay modes the update may move; 0
    stands for the first update (step 0, where the denoised image is the prior
    mean), which moves the common amplitude instead. From the second update on the
    delay modes grow geometrically from `cfg.coarse_modes` to
    `cfg.max_delay_modes`, reached at the last update.
    """
    last_step = cfg.update_until * (cfg.sampler.n_steps - 1)
    steps = sorted({int(round(step)) for step in np.linspace(0.0, last_step, cfg.n_updates)})
    schedule = {steps[0]: 0}
    for update, step in enumerate(steps[1:]):
        progress = update / max(1, len(steps) - 2)
        schedule[step] = int(
            round(cfg.coarse_modes * (cfg.max_delay_modes / cfg.coarse_modes) ** progress)
        )
    return schedule


class _ScreenEstimate:
    """The screen of one candidate run: delay-mode coefficients and the common log amplitude."""

    def __init__(
        self,
        start: PhaseScreen,
        element_x: np.ndarray,
        center_frequency: float,
        cfg: CalibrationConfig,
        generator: torch.Generator | None,
    ):
        self.start_delays, self.start_log_amplitudes = screen_tensors(start)
        # Delay coefficients are phases at the centre frequency [rad].
        self.radians_to_seconds = 1.0 / (2.0 * math.pi * center_frequency)
        self.delay_basis = torch.from_numpy(
            gauge_free_basis(element_x, cfg.gauge)[:, : cfg.max_delay_modes]
        )
        self.delay_sigma = cfg.delay_prior_sigma / self.radians_to_seconds
        self.amplitude_sigma = cfg.common_log_amplitude_prior_sigma
        self.delay_coefficients = torch.zeros(self.delay_basis.shape[1], dtype=torch.float64)
        self.common_log_amplitude = torch.zeros(1, dtype=torch.float64)
        if generator is not None and cfg.initial_spread > 0.0:
            self.delay_coefficients += (
                cfg.initial_spread
                * self.delay_sigma
                * torch.randn(
                    self.delay_coefficients.shape, generator=generator, dtype=torch.float64
                )
            )
        self.delay_coefficients.requires_grad_(True)
        self.common_log_amplitude.requires_grad_(True)

    def screen(self) -> Screen:
        """Delays [s] and log amplitudes, differentiable in the coefficients."""
        delays = self.start_delays + self.radians_to_seconds * (
            self.delay_basis @ self.delay_coefficients
        )
        return delays, self.start_log_amplitudes + self.common_log_amplitude

    def detached_screen(self) -> Screen:
        """The current delays [s] and log amplitudes without a graph."""
        delays, log_amplitudes = self.screen()
        return delays.detach(), log_amplitudes.detach()

    def log_prior(self) -> torch.Tensor:
        """Gaussian log prior of the coefficients about the starting screen (up to a constant).

        In an orthonormal basis independent delays of standard deviation `s` per
        element are independent coefficients of standard deviation `s`.
        """
        return -0.5 * (
            (self.delay_coefficients / self.delay_sigma).square().sum()
            + (self.common_log_amplitude / self.amplitude_sigma).square().sum()
        )

    def fit(
        self,
        log_likelihood: Callable[[Screen], torch.Tensor],
        n_delay_modes: int,
        n_iterations: int,
    ) -> None:
        """Maximise ``log_likelihood(screen) + log prior`` by L-BFGS.

        log_likelihood: scalar log-likelihood of the data for a screen (delays, log amplitudes).
        n_delay_modes: 0 moves the common amplitude only; otherwise the first
            `n_delay_modes` delay modes move and the amplitude stays.
        n_iterations: L-BFGS iterations (strong Wolfe line search).
        """
        optimiser = torch.optim.LBFGS(
            [self.delay_coefficients, self.common_log_amplitude],
            max_iter=n_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = -(log_likelihood(self.screen()) + self.log_prior())
            loss.backward()
            self.delay_coefficients.grad[n_delay_modes:] = 0.0
            if n_delay_modes > 0:
                self.common_log_amplitude.grad.zero_()
            return loss

        optimiser.step(closure)

    def phase_screen(self) -> PhaseScreen:
        """The current estimate as a `PhaseScreen`."""
        delays, log_amplitudes = (value.numpy() for value in self.detached_screen())
        return PhaseScreen(delays=delays.copy(), amplitudes=np.exp(log_amplitudes))


def calibrate_screen(
    prior: Prior,
    model: ScreenModel,
    start: PhaseScreen,
    element_x: np.ndarray,
    center_frequency: float,
    mean_image: np.ndarray,
    noise_floor: np.ndarray,
    cell_area: float,
    shape: float,
    cfg: CalibrationConfig,
    tree: SeedTree,
    *sub: int,
) -> CalibrationResult:
    """Estimate the phase screen of a scene from its time-averaged image (module docstring).

    prior: the trained image prior.
    model: phase-screen forward model of the scene (`recon.forward.scene_screen_model`
        with `cfg.source_refinement`).
    start: the CT-derived screen: starting point and centre of the Gaussian prior.
    element_x: lateral element positions [m], shape (n_elements,).
    center_frequency: pulse centre frequency [Hz] (scales the delay coefficients).
    mean_image: `skull_das` power Doppler averaged over the trials, shape (nz, nx).
    noise_floor: floor of `skull_das`, shape (n_pix,).
    cell_area: area of one pixel [m^2].
    shape: gamma shape of the averaged image, `effective_frames * n_trials`
        (`recon.likelihood` on the approximation).
    cfg: calibration settings.
    tree: seed tree of the reconstruction.
    *sub: sub-indices of the `recon_sampling` stream for this calibration.

    Returns the selected screen with the diagnostics of every candidate.
    """
    n_pix = model.n_pixels
    pixel_generator = torch_generator(tree, SAMPLING_STREAM, *sub, _PIXEL_DRAW)
    order = torch.randperm(n_pix, generator=pixel_generator)
    pixel_index = torch.sort(order[: min(cfg.n_pixels, n_pix)]).values
    measured = torch.from_numpy(np.ascontiguousarray(mean_image, dtype=np.float32)).reshape(-1)
    floor = torch.from_numpy(np.ascontiguousarray(noise_floor, dtype=np.float32))
    sampler = cfg.sampler
    diffusion, transform = prior.diffusion, prior.transform
    following = next_timesteps(diffusion, sampler.n_steps)
    timesteps = diffusion.sampling_timesteps(sampler.n_steps)
    step_number = {t: number for number, t in enumerate(timesteps)}
    schedule = update_schedule(cfg)
    delay_fit_shape = min(shape, cfg.delay_fit_shape)
    image_shape = (sampler.n_samples, 1, *prior.pixel_grid.shape)

    def cbv_cells(samples: torch.Tensor) -> torch.Tensor:
        return transform.to_physical(samples).clamp_min(0.0).flatten(1)

    def full_operator(screen: Screen) -> torch.Tensor:
        with torch.no_grad():
            return power_doppler_operator(model, *screen).to(torch.float32)

    def deviance_per_pixel(samples: torch.Tensor, operator: torch.Tensor) -> float:
        with torch.no_grad():
            expected = expected_power_doppler(operator, cbv_cells(samples), cell_area, floor)
            return float(gamma_deviance(measured, expected, shape).mean()) / n_pix

    candidates, fits = [], []
    start_fit = math.nan
    for candidate in range(cfg.n_candidates):
        generator = torch_generator(tree, SAMPLING_STREAM, *sub, _FIRST_CANDIDATE_DRAW + candidate)
        estimate = _ScreenEstimate(
            start, element_x, center_frequency, cfg, generator if candidate > 0 else None
        )
        operator = [full_operator(estimate.detached_screen())]

        def guidance(x_t: torch.Tensor, x0_hat: torch.Tensor, t: int) -> torch.Tensor:
            step = step_number[t]
            if step in schedule:
                cells = cbv_cells(x0_hat.detach())
                # The amplitude update (no delay modes) keeps the full shape; see the config.
                fit_shape = shape if schedule[step] == 0 else delay_fit_shape

                def screen_log_likelihood(screen: Screen) -> torch.Tensor:
                    subset = power_doppler_operator(model, *screen, pixel_index).to(torch.float32)
                    expected = expected_power_doppler(subset, cells, cell_area, floor[pixel_index])
                    return gamma_log_likelihood(measured[pixel_index], expected, fit_shape).mean()

                estimate.fit(screen_log_likelihood, schedule[step], cfg.lbfgs_iterations)
                operator[0] = full_operator(estimate.detached_screen())
            expected = expected_power_doppler(operator[0], cbv_cells(x0_hat), cell_area, floor)
            log_likelihood = gamma_log_likelihood(measured, expected, shape)
            (gradient,) = torch.autograd.grad(log_likelihood.sum(), x_t)
            return guidance_correction(gradient, x_t, x0_hat, diffusion, t, following[t], sampler)

        samples = diffusion.sample(
            prior.model,
            image_shape,
            generator,
            n_steps=sampler.n_steps,
            eta=sampler.eta,
            guidance=guidance,
            x0_clip=(transform.model_floor, math.inf),
        )
        candidates.append(estimate.phase_screen())
        fits.append(deviance_per_pixel(samples, operator[0]))
        if candidate == 0:
            start_fit = deviance_per_pixel(samples, full_operator(screen_tensors(start)))
    selected = int(np.argmin(fits))
    return CalibrationResult(
        screen=candidates[selected],
        candidates=tuple(candidates),
        deviance_per_pixel=tuple(fits),
        start_deviance_per_pixel=start_fit,
        selected=selected,
    )


def match_total_power(
    screen: PhaseScreen, operator: torch.Tensor, target_total: float
) -> tuple[PhaseScreen, torch.Tensor]:
    """Rescale a screen's common amplitude so that its operator has a given total power.

    Calibration may use a coarser forward model than the final operator
    (`CalibrationConfig.source_refinement`); one source point per pixel overstates
    the blood signal, so the common amplitude it finds is too small for the finer
    operator. `H` is exactly proportional to the fourth power of the common
    amplitude, so the amplitude is corrected in closed form: the operator of the
    returned screen has the same sum over all pixels and cells (the image power of
    a uniform CBV map) as the calibration operator of the estimated screen.

    screen: the estimated screen.
    operator: its operator in the final forward model, shape (n_pix, n_cells).
    target_total: sum of its operator in the calibration forward model.

    Returns the rescaled screen and its operator.
    """
    gain = target_total / float(operator.sum())
    rescaled = PhaseScreen(delays=screen.delays, amplitudes=screen.amplitudes * gain**0.25)
    return rescaled, operator * gain
