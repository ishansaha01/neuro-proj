"""Save and load a trained prior.

A checkpoint is one ``torch.save`` file of plain containers and tensors (it
loads with ``weights_only=True``): raw and averaged weights, the full
`PriorConfig`, provenance of the training run and of its data set, the loss
history and the timing of the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurosim.common.config import from_dict, to_dict
from neurosim.common.grid import Grid2D
from neurosim.common.io import load_bundle, provenance
from neurosim.recon.cbv_images import pixel_grid
from neurosim.recon.diffusion import GaussianDiffusion
from neurosim.recon.train import PriorConfig, TrainResult
from neurosim.recon.transform import CbvTransform
from neurosim.recon.unet import UNet

_FORMAT_VERSION = 1

# Where the default prior and its data set are kept (the folder is git-ignored);
# this file is <repo>/src/neurosim/recon/checkpoint.py.
_PRIOR_DIR = Path(__file__).resolve().parents[3] / "outputs" / "prior"
DEFAULT_PRIOR_PATH: Path = _PRIOR_DIR / "prior.pt"
DEFAULT_DATASET_PATH: Path = _PRIOR_DIR / "dataset.h5"
# Keys of a data set bundle that identify it (see `common.io.provenance`).
DATASET_PROVENANCE_KEYS: tuple[str, ...] = (
    "config_hash",
    "root_seed",
    "git_commit",
    "neurosim_version",
)
_HOW_TO_BUILD = (
    "create them with\n"
    "  python scripts/make_prior_dataset.py --config configs/prior_default.yaml --seed 0 "
    "--out outputs/prior/dataset.h5\n"
    "  python scripts/train_prior.py --config configs/prior_default.yaml "
    "--data outputs/prior/dataset.h5 --seed 0 --out outputs/prior/prior.pt"
)


@dataclass(frozen=True)
class Prior:
    """A trained prior, ready for sampling and posterior sampling.

    model: the denoiser ``(x_t, t) -> v`` in eval mode with frozen parameters, so
        gradients taken through it reach the input only. Input and output are in
        model space, float32, shape (batch, 1, n, n); ``t`` is an integer tensor (batch,).
    diffusion: the noise schedule with `predict_x0`, `step` and the samplers.
    transform: the differentiable map between model space and physical CBV.
    pixel_grid: the pixel grid the images live on (imaging-plane frame, [iz, ix]).
    config: the full configuration the prior was trained with.
    provenance: config hash, seed and code version of the training run, plus the
        provenance of its data set under ``"dataset"``.
    history: held-out loss records of the training run (`recon.train.train_prior`).
    steps_completed: optimiser steps of the training run.
    """

    model: UNet
    diffusion: GaussianDiffusion
    transform: CbvTransform
    pixel_grid: Grid2D
    config: PriorConfig
    provenance: dict[str, Any]
    history: list[dict[str, Any]]
    steps_completed: int


def save_prior(
    path: str | os.PathLike[str],
    result: TrainResult,
    cfg: PriorConfig,
    root_seed: int,
    dataset_provenance: dict[str, Any],
) -> None:
    """Write a training result to a checkpoint file.

    path: output file; parent directories are created.
    result: outcome of `recon.train.train_prior`.
    cfg: the configuration it was trained with.
    root_seed: root seed of the training `SeedTree`.
    dataset_provenance: provenance attributes of the data set bundle it was trained on.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": _FORMAT_VERSION,
            "model": result.model.state_dict(),
            "ema_model": result.ema_model.state_dict(),
            "config": to_dict(cfg),
            "provenance": {**provenance(cfg, root_seed), "dataset": dict(dataset_provenance)},
            "history": result.history,
            "steps_completed": result.steps_completed,
            "wall_seconds": result.wall_seconds,
        },
        path,
    )


def load_prior(path: str | os.PathLike[str], *, ema: bool = True) -> Prior:
    """Read a checkpoint written by `save_prior`.

    path: checkpoint file.
    ema: use the averaged weights (default) or the raw final weights.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != _FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {checkpoint['format_version']} is not supported "
            f"(expected {_FORMAT_VERSION})"
        )
    cfg = from_dict(PriorConfig, checkpoint["config"])
    model = UNet(cfg.unet)
    model.load_state_dict(checkpoint["ema_model" if ema else "model"])
    return Prior(
        model=model.eval().requires_grad_(False),
        diffusion=GaussianDiffusion(cfg.diffusion),
        transform=cfg.transform,
        pixel_grid=pixel_grid(cfg.data.pixel_grid),
        config=cfg,
        provenance=checkpoint["provenance"],
        history=checkpoint["history"],
        steps_completed=checkpoint["steps_completed"],
    )


def load_prior_with_data(
    prior_path: str | os.PathLike[str] = DEFAULT_PRIOR_PATH,
    dataset_path: str | os.PathLike[str] = DEFAULT_DATASET_PATH,
) -> tuple[Prior, dict[str, np.ndarray]]:
    """Load a trained prior together with the data set it was trained on.

    prior_path: checkpoint written by ``scripts/train_prior.py``.
    dataset_path: bundle written by ``scripts/make_prior_dataset.py``.

    Returns the prior and the arrays of the data set (see
    `recon.cbv_images.sample_cbv_images`; ``heldout`` marks the images the prior
    never saw).

    Raises `FileNotFoundError`, naming the two scripts to run, when either file
    is missing, and `ValueError` when the data set is not the one the prior was
    trained on (config hash or seed differ).
    """
    missing = [str(path) for path in (prior_path, dataset_path) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"the trained prior is not available (missing: {', '.join(missing)}); {_HOW_TO_BUILD}"
        )
    prior = load_prior(prior_path)
    arrays, attrs = load_bundle(dataset_path)
    trained_on = prior.provenance["dataset"]
    for key in ("config_hash", "root_seed"):
        if attrs[key] != trained_on[key]:
            raise ValueError(
                f"{dataset_path} is not the data set {prior_path} was trained on "
                f"({key}: {attrs[key]!r} != {trained_on[key]!r}); {_HOW_TO_BUILD}"
            )
    return prior, arrays
