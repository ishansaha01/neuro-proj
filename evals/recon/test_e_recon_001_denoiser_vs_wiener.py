"""E-RECON-001: the trained denoiser beats the best linear denoiser on held-out images.

The reference is the Wiener estimator of a Gaussian with the mean and covariance
of the training images (model space, 1024 dimensions):

    x0_hat = mu + sqrt(ab) Sigma (ab Sigma + (1 - ab) I)^-1 (x_t - sqrt(ab) mu).

It is the best estimator that is linear in x_t, and it involves no network. A
denoiser with a lower error must use structure a Gaussian cannot express (sparse
vessels, the sharp brain boundary).

Needs the trained prior; see `neurosim.recon.checkpoint.load_prior_with_data`.
"""

from __future__ import annotations

import numpy as np
import torch

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.recon.checkpoint import load_prior_with_data
from neurosim.recon.seeding import TRAINING_STREAM, torch_generator
from neurosim.recon.train import to_model_tensor

EVAL_ID = "E-RECON-001"
# First sub-index of the eval's draws in the `diffusion_training` stream.
STREAM_SUB = 1001
ALPHA_BARS = (0.9, 0.7, 0.5, 0.3)
MAX_MSE_RATIO = 0.9
BATCH_SIZE = 128


def _wiener_mse(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    alpha_bar: float,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    basis: np.ndarray,
) -> float:
    """x_0 mean squared error of the Wiener estimator.

    noisy, clean: x_t and x_0, shape (N, 1, n, n). The estimator is the posterior
    mean of x_0 under ``N(mean, basis diag(eigenvalues) basis^T)``.
    """
    x_t = noisy.flatten(1).numpy().astype(np.float64)
    gain = np.sqrt(alpha_bar) * eigenvalues / (alpha_bar * eigenvalues + 1.0 - alpha_bar)
    estimate = mean + (((x_t - np.sqrt(alpha_bar) * mean) @ basis) * gain) @ basis.T
    return float(np.mean((estimate - clean.flatten(1).numpy()) ** 2))


@eval_case(
    id=EVAL_ID,
    claim="On held-out images the trained denoiser estimates x_0 with a lower mean squared "
    "error than the optimal linear (Wiener) denoiser, so it has learned non-Gaussian structure",
    reference="Wiener estimator from the mean and covariance of the training images "
    "(closed form, independent of the network), applied to the same noisy held-out images",
    metric="Largest ratio, over alpha_bar in (0.9, 0.7, 0.5, 0.3), of the network's x_0 mean "
    "squared error to the Wiener error on the held-out images (model space)",
    threshold="< 0.9, i.e. at least 10 % below the Wiener error at every level (fixed in "
    "advance: 10 % is far above the Monte-Carlo error of a paired ratio over 512 images x "
    "1024 pixels; levels below alpha_bar 0.3 are left out because there the estimate is "
    "dominated by the prior mean, linear estimation is close to optimal and no margin can "
    "be demanded)",
    seeds=(0,),
)
def test_e_recon_001_denoiser_vs_wiener():
    prior, data = load_prior_with_data()
    diffusion = prior.diffusion
    train = to_model_tensor(data["images"][~data["heldout"]], prior.transform)
    heldout = to_model_tensor(data["images"][data["heldout"]], prior.transform)
    n_heldout = heldout.shape[0]

    train_flat = train.flatten(1).numpy().astype(np.float64)
    mean = train_flat.mean(axis=0)
    eigenvalues, basis = np.linalg.eigh(np.cov(train_flat, rowvar=False))
    gaussian = (mean, np.maximum(eigenvalues, 0.0), basis)
    train_subset = train[:n_heldout]

    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    values: dict[str, float] = {}
    ratios = []
    for index, level in enumerate(ALPHA_BARS):
        t = diffusion.timestep_for_alpha_bar(level)
        alpha_bar = float(diffusion.alpha_bar[t])
        noise = torch.randn(
            heldout.shape, generator=torch_generator(tree, TRAINING_STREAM, STREAM_SUB, index)
        )
        x_t = diffusion.q_sample(heldout, t, noise)
        with torch.no_grad():
            network = torch.cat(
                [diffusion.predict_x0(prior.model, batch, t) for batch in x_t.split(BATCH_SIZE)]
            )
        network_mse = float((network - heldout).square().mean())
        wiener_mse = _wiener_mse(x_t, heldout, alpha_bar, *gaussian)
        ratios.append(network_mse / wiener_mse)
        values[f"ratio_{level:g}"] = ratios[-1]
        values[f"network_mse_{level:g}"] = network_mse
        values[f"wiener_mse_{level:g}"] = wiener_mse
        # In-sample error of the fitted Wiener filter: optimistic for any linear estimator.
        values[f"wiener_train_mse_{level:g}"] = _wiener_mse(
            diffusion.q_sample(train_subset, t, noise), train_subset, alpha_bar, *gaussian
        )

    worst = max(ratios)
    report(
        EVAL_ID,
        {"max_ratio": worst, **values},
        passed=worst < MAX_MSE_RATIO,
        note=f"prior trained for {prior.steps_completed} steps; wiener_train_mse is the "
        "in-sample error of the Wiener filter on training images, for comparison",
    )
