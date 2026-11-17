"""E-RECON-006: autograd gradient of the log-likelihood in the screen matches finite differences.

The log-likelihood of a measured image given the true CBV, as a function of the
phase screen, is differentiated by torch through the operator assembly
(`recon.forward`) and the gamma likelihood (`recon.likelihood`), in double
precision. The reference is the central difference of the same scalar function
in every one of the 2 * n_elements parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.config import imaging_setup
from neurosim.recon.config import ReconConfig
from neurosim.recon.forward import (
    expected_power_doppler,
    power_doppler_operator,
    scene_screen_model,
    screen_tensors,
)
from neurosim.recon.likelihood import gamma_log_likelihood
from neurosim.recon.synthetic import synthetic_run

EVAL_ID = "E-RECON-006"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recon_small.yaml"
# Finite-difference steps: 1 mrad of phase at the centre frequency, and 1e-4 Np.
PHASE_STEP = 1.0e-3
LOG_AMPLITUDE_STEP = 1.0e-4
MAX_RELATIVE_ERROR = 1.0e-5
N_PIXELS = 96


@eval_case(
    id=EVAL_ID,
    claim="The autograd gradient of the gamma log-likelihood of a power Doppler image with "
    "respect to the per-element delays and log amplitudes of the phase screen equals its "
    "central finite-difference gradient",
    reference="Central finite differences of the same scalar function in each of the "
    "2 * n_elements parameters (steps: 1 mrad of phase at the centre frequency; 1e-4 Np)",
    metric="|g_autograd - g_fd| / |g_fd| (Euclidean norms), separately for delays and log "
    "amplitudes, the larger of the two; complex128, 48 elements, 64 frequencies, 96 of "
    "1024 pixels, evaluated at the CT screen of a synthetic scene",
    threshold="< 1e-5 (fixed in advance: the truncation error of a central difference is of "
    "the order of the squared step relative to the scale of variation, about 1e-6 here, and "
    "double-precision rounding is smaller; a missing transmit or receive term halves the "
    "gradient and any wrong term gives an error of order 1)",
    seeds=(0,),
)
def test_e_recon_006_screen_gradient():
    seed = REGISTRY[EVAL_ID].seeds[0]
    synthetic = from_dict(ReconConfig, load_yaml(CONFIG)).synthetic
    scene, trials = synthetic_run(synthetic, SeedTree(seed))
    model = scene_screen_model(scene, synthetic.run.source_refinement, torch.complex128)
    order = SeedTree(seed).rng("recon_sampling", 1006).permutation(model.n_pixels)
    pixel_index = torch.from_numpy(np.sort(order[:N_PIXELS]))
    cbv = torch.from_numpy(trials.cbv[:1].astype(np.float64)).flatten(1)
    measured = torch.from_numpy(trials.power_doppler["skull_das"][:1].astype(np.float64))
    measured = measured.flatten(1)[:, pixel_index]
    floor = torch.from_numpy(scene.noise_floor["skull_das"].astype(np.float64))[pixel_index]

    def log_likelihood(delays: torch.Tensor, log_amplitudes: torch.Tensor) -> torch.Tensor:
        operator = power_doppler_operator(model, delays, log_amplitudes, pixel_index)
        expected = expected_power_doppler(operator, cbv, scene.cell_area, floor)
        return gamma_log_likelihood(measured, expected, scene.effective_frames).sum()

    delays, log_amplitudes = (
        value.clone().requires_grad_(True) for value in screen_tensors(scene.screen_ct)
    )
    autograd = torch.autograd.grad(log_likelihood(delays, log_amplitudes), [delays, log_amplitudes])

    center_frequency = imaging_setup(scene.imaging).transducer.center_frequency
    steps = (PHASE_STEP / (2.0 * np.pi * center_frequency), LOG_AMPLITUDE_STEP)
    point = [value.detach() for value in (delays, log_amplitudes)]
    errors = {}
    with torch.no_grad():
        for which, (name, step) in enumerate(zip(("delays", "log_amplitudes"), steps)):
            finite = torch.zeros_like(point[which])
            for element in range(finite.shape[0]):
                shifted = [[value.clone() for value in point] for _ in range(2)]
                shifted[0][which][element] += step
                shifted[1][which][element] -= step
                finite[element] = (log_likelihood(*shifted[0]) - log_likelihood(*shifted[1])) / (
                    2.0 * step
                )
            errors[name] = float(
                torch.linalg.norm(autograd[which] - finite) / torch.linalg.norm(finite)
            )

    worst = max(errors.values())
    report(EVAL_ID, {"max_relative_error": worst, **errors}, passed=worst < MAX_RELATIVE_ERROR)
