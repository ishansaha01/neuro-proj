"""E-RECON-004: unconditional samples of the trained prior match held-out images in distribution.

Two simple statistics, fixed before the prior was trained, both in model space:
quantiles of the pooled pixel values, and the radially averaged power spectrum
of the mean-removed images. The reference images come from held-out heads of
the head model, never seen in training.

Needs the trained prior; see `neurosim.recon.checkpoint.load_prior_with_data`.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.checkpoint import load_prior_with_data
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator
from neurosim.recon.train import to_model_tensor

EVAL_ID = "E-RECON-004"
# Sub-index of the eval's draws in the `recon_sampling` stream.
STREAM_SUB = 1004
N_SAMPLES = 128
N_STEPS = 100
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MAX_QUANTILE_DIFFERENCE = 0.1
MAX_LOG10_POWER_RATIO = 0.3


def _radial_power_spectrum(images: np.ndarray) -> np.ndarray:
    """Mean power of mean-removed images per radial frequency.

    images: (N, n, n). Returns shape (n // 2,): the power averaged over images and
    over the Fourier coefficients whose radial frequency rounds to 1 .. n/2 cycles
    per image width.
    """
    n = images.shape[-1]
    centred = images - images.mean(axis=(1, 2), keepdims=True)
    power = np.mean(np.abs(np.fft.fft2(centred)) ** 2, axis=0)
    frequency = np.fft.fftfreq(n, d=1.0 / n)
    radius = np.rint(np.hypot(frequency[:, None], frequency[None, :])).astype(int)
    return np.array([power[radius == r].mean() for r in range(1, n // 2 + 1)])


def _statistic_differences(images: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Largest quantile difference and largest |log10 power ratio| between two image sets."""
    quantile_difference = np.abs(
        np.quantile(images, QUANTILES) - np.quantile(reference, QUANTILES)
    ).max()
    log_ratio = np.log10(_radial_power_spectrum(images) / _radial_power_spectrum(reference))
    return float(quantile_difference), float(np.abs(log_ratio).max())


@eval_case(
    id=EVAL_ID,
    claim="Unconditional samples of the trained prior have the pixel-value distribution and "
    "the spatial power spectrum of held-out CBV images",
    reference="Images of held-out heads of the head model (never seen in training), in the "
    "model space of the prior",
    metric="Two numbers for 128 DDIM samples (100 steps) against all held-out images: the "
    "largest absolute difference of the pooled pixel quantiles 5, 25, 50, 75, 95, 99 % "
    "(model units) and the largest |log10| ratio of the radially averaged power spectra "
    "over radial frequencies 1 to n/2",
    threshold="quantile difference < 0.1 and |log10 power ratio| < 0.3 (fixed in advance: "
    "0.1 is 5 % of the nominal model range [-1, 1], about the model-space effect of a 30 % "
    "CBV change in grey matter; 0.3 is a factor of 2 in power, 3 dB, at every spatial "
    "scale; the same statistics between training and held-out images are reported as "
    "the noise floor)",
    seeds=(0,),
)
def test_e_recon_004_sample_statistics():
    prior, data = load_prior_with_data()
    heldout = to_model_tensor(data["images"][data["heldout"]], prior.transform).numpy()[:, 0]
    train = to_model_tensor(data["images"][~data["heldout"]], prior.transform).numpy()[:, 0]

    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    samples = prior.diffusion.ddim_sample(
        prior.model,
        (N_SAMPLES, 1, *prior.pixel_grid.shape),
        torch_generator(tree, SAMPLING_STREAM, STREAM_SUB),
        n_steps=N_STEPS,
        x0_clip=(prior.transform.model_floor, float("inf")),
    ).numpy()[:, 0]

    quantile_difference, log_power_ratio = _statistic_differences(samples, heldout)
    # The same comparison for real images: N_SAMPLES training images spread over all heads.
    spread = train[:: train.shape[0] // N_SAMPLES][:N_SAMPLES]
    floor_quantile, floor_power = _statistic_differences(spread, heldout)
    report(
        EVAL_ID,
        {
            "quantile_difference": quantile_difference,
            "log10_power_ratio": log_power_ratio,
            "real_quantile_difference": floor_quantile,
            "real_log10_power_ratio": floor_power,
        },
        passed=quantile_difference < MAX_QUANTILE_DIFFERENCE
        and log_power_ratio < MAX_LOG10_POWER_RATIO,
        note=f"prior trained for {prior.steps_completed} steps; real_* compare "
        f"{N_SAMPLES} training images with the held-out images",
    )
