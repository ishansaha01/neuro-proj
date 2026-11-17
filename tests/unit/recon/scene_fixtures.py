"""Fixtures of the reconstruction tests: a tiny synthetic run and an untrained prior.

Imported by the test modules that need them (`from .scene_fixtures import ...`).
"""

from __future__ import annotations

import pytest

from neurosim.common.config import from_dict, to_dict
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.recon.cbv_images import PriorDataConfig
from neurosim.recon.checkpoint import Prior
from neurosim.recon.diffusion import DiffusionConfig, GaussianDiffusion
from neurosim.recon.seeding import TRAINING_STREAM, TrainingDraw, torch_generator
from neurosim.recon.synthetic import SyntheticConfig, synthetic_run
from neurosim.recon.train import PriorConfig
from neurosim.recon.transform import CbvTransform
from neurosim.recon.unet import UNetConfig, build_unet

DATA_SEED = 5


def _tiny_synthetic(refinement: int):
    """A 16 x 16 pixel synthetic phase-screen run: 16 elements, 32 frequencies, 12 trials."""
    run = from_dict(
        RunConfig,
        {
            "head": to_dict(PriorDataConfig().head),
            "imaging": {
                "transducer": {"n_elements": 16, "fractional_bandwidth": 0.4},
                "sequence": {"angles_deg": [-8.0, 0.0, 8.0], "n_frames": 50},
                "frequencies": {"n_frequencies": 32, "truncation_db": 12.0},
                "roi": {
                    "x_range": [-0.0075, 0.0075],
                    "z_range": [0.014, 0.029],
                    "spacing": 1.0e-3,
                },
                "pixels": {"shape": [16, 16], "spacing": 1.0e-3},
                "doppler": {"n_discard": 2},
                "noise": {"electronic_snr_db": 20.0},
                "fidelity": "statistical",
            },
            "source_refinement": refinement,
            "trials": {"n_images": 12, "image_interval": 10.0},
        },
    )
    return synthetic_run(SyntheticConfig(run=run), SeedTree(DATA_SEED))


@pytest.fixture(scope="session")
def tiny_run():
    """`(scene, trials)` of the tiny synthetic run with one source point per pixel."""
    return _tiny_synthetic(1)


@pytest.fixture(scope="session")
def tiny_scene(tiny_run):
    """The scene of `tiny_run`."""
    return tiny_run[0]


@pytest.fixture(scope="session")
def tiny_scene_refined():
    """The tiny synthetic scene with 2 x 2 source points per pixel."""
    return _tiny_synthetic(2)[0]


@pytest.fixture(scope="session")
def tiny_prior():
    """An untrained prior on the 16 x 16 grid of `tiny_scene`: fast, statistically meaningless."""
    unet_cfg = UNetConfig(
        base_channels=8,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        norm_groups=4,
        attention=False,
    )
    generator = torch_generator(SeedTree(DATA_SEED), TRAINING_STREAM, TrainingDraw.WEIGHT_INIT)
    return Prior(
        model=build_unet(unet_cfg, generator).eval().requires_grad_(False),
        diffusion=GaussianDiffusion(DiffusionConfig()),
        transform=CbvTransform(),
        pixel_grid=Grid2D(shape=(16, 16), spacing=1.0e-3, origin=(0.014, -0.0075)),
        config=PriorConfig(),
        provenance={},
        history=[],
        steps_completed=0,
    )
