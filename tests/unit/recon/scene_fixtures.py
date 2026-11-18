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


def _tiny_ct_config() -> RunConfig:
    """A tiny real run with a CT and the bent-ray backend: 16 elements, 8 x 8 pixels, 12 trials."""
    return from_dict(
        RunConfig,
        {
            "head": {
                "grid": {
                    "spacing": 2.0e-3,
                    "region": "box",
                    "box_size": [0.064, 0.064, 0.064],
                    "box_center_depth": 0.02,
                }
            },
            "imaging": {
                "transducer": {"n_elements": 16},
                "sequence": {"angles_deg": [-8.0, 0.0, 8.0], "frame_rate": 500.0, "n_frames": 40},
                "frequencies": {"n_frequencies": 28, "truncation_db": 12.0},
                "roi": {"x_range": [-0.004, 0.004], "z_range": [0.016, 0.024], "spacing": 1.0e-3},
                "pixels": {"shape": [8, 8], "spacing": 1.0e-3},
                "doppler": {"cutoff_rule": "fixed", "n_discard": 2},
                "noise": {"electronic_snr_db": 60.0},
                "fidelity": "statistical",
            },
            "backend": {"backend": "bent_ray"},
            "ct": {
                "geometry": {"n_views": 240, "n_detectors": 192, "detector_pitch": 2.1e-3},
                "calibration_phantom_diameter": 0.16,
                "max_chunk_samples": 2000000,
            },
            "ct_prior": {"slab_half_slices": 0},
            "medium_spacing": 0.4e-3,
            "source_refinement": 1,
            "trials": {"n_images": 12, "image_interval": 10.0},
        },
    )


@pytest.fixture(scope="session")
def tiny_ct_run():
    """`(scene, trials)` of a tiny real run that carries its CT image (about 1 s)."""
    from neurosim.data.scene import build_scene
    from neurosim.data.trials import simulate_trials

    cfg = _tiny_ct_config()
    tree = SeedTree(DATA_SEED)
    scene = build_scene(cfg, tree)
    return scene, simulate_trials(scene, cfg, tree)


@pytest.fixture(scope="session")
def tiny_ct_family(tiny_ct_run):
    """The skull-model family of `tiny_ct_run` with three lateral modes."""
    from neurosim.recon.config import SkullModelConfig
    from neurosim.recon.skull_model import build_family

    return build_family(tiny_ct_run[0], SkullModelConfig(n_lateral_modes=3))


@pytest.fixture(scope="session")
def tiny_ct_prior(tiny_ct_run):
    """An untrained prior on the 8 x 8 pixel grid of `tiny_ct_run`."""
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
        pixel_grid=tiny_ct_run[0].pixel_grid,
        config=PriorConfig(),
        provenance={},
        history=[],
        steps_completed=0,
    )
