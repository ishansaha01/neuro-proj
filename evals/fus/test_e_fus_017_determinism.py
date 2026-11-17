"""E-FUS-017: acquisition is deterministic and the electronic-noise stream is isolated."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.acquire import acquire_frames
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.noise import electronic_noise, electronic_noise_variance
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.fus.types import GreensBank

EVAL_ID = "E-FUS-017"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
N_FRAMES = 4
N_SCATTERERS = 200  # blood and tissue each
NOISE_STREAM = "us_electronic_noise"
OTHER_NOISE_ROOT = 12345  # root seed of the replacement noise stream
MAX_RESIDUAL = 1e-10


class _NoiseSwappedTree(SeedTree):
    """A seed tree whose electronic-noise stream comes from another root seed."""

    def __init__(self, root_seed: int, noise_root_seed: int):
        super().__init__(root_seed)
        self._noise_tree = SeedTree(noise_root_seed)

    def rng(self, name: str, *sub: int) -> np.random.Generator:
        if name == NOISE_STREAM:
            return self._noise_tree.rng(name, *sub)
        return super().rng(name, *sub)


@eval_case(
    id=EVAL_ID,
    claim=(
        "the same seed gives bit-identical frames, a frame does not depend on which other "
        "frames are simulated, and changing only the electronic-noise stream changes only the "
        "noise part of the frames"
    ),
    reference=(
        "Bitwise comparison of repeated runs; the noise-free frames of the same run (noise "
        "switched off) plus the separately beamformed noise of each stream"
    ),
    metric=(
        "repeat_difference: max |difference| between two runs with the same seed; "
        "offset_difference: max |difference| between frames 2-3 of a 4-frame run and a 2-frame "
        "run starting at frame 2; noise_free_residual: max |frames - noise-free frames - "
        "beamformed noise| / max |noise-free frames|, over both noise streams"
    ),
    threshold=(
        "repeat_difference = 0, offset_difference = 0 and noise_free_residual < 1e-10 (fixed "
        "in advance: determinism is exact; the noise is added to the noise-free image by one "
        "float64 addition, whose rounding error is about 1e-16 of the larger term)"
    ),
    seeds=(7,),
)
def test_e_fus_017_determinism() -> None:
    base = from_dict(ImagingConfig, load_yaml(CONFIG))
    cfg = replace(
        base,
        sequence=replace(base.sequence, n_frames=N_FRAMES),
        doppler=replace(base.doppler, n_discard=1),  # must stay below the frame counts used here
    )
    setup = imaging_setup(cfg)
    seed = REGISTRY[EVAL_ID].seeds[0]
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    cbv = np.full(setup.roi.shape, cfg.noise.cbv_reference)
    velocity = np.full((2, *setup.roi.shape), 0.01)

    def acquire(tree: SeedTree, config: ImagingConfig, frame_offset: int = 0) -> np.ndarray:
        blood = blood_scatterers(
            cbv,
            velocity,
            setup.roi,
            N_SCATTERERS,
            tree.rng("blood_scatterers"),
            config.scatterers.blood_transit_length,
        )
        tissue = tissue_scatterers(
            setup.roi,
            N_SCATTERERS,
            tissue_amplitude(setup.roi, N_SCATTERERS, config.noise),
            tree.rng("tissue_scatterers"),
        )
        sequence = imaging_setup(config).sequence
        return acquire_frames(
            bank, bank, blood, tissue, sequence, setup.pixel_grid, config, tree, frame_offset
        )

    frames = acquire(SeedTree(seed), cfg)
    repeat_difference = float(np.abs(frames - acquire(SeedTree(seed), cfg)).max())

    tail_cfg = replace(cfg, sequence=replace(cfg.sequence, n_frames=2))
    tail = acquire(SeedTree(seed), tail_cfg, frame_offset=2)
    offset_difference = float(np.abs(frames[2:] - tail).max())

    quiet_cfg = replace(cfg, noise=replace(cfg.noise, electronic_noise=False))
    noise_free = acquire(SeedTree(seed), quiet_cfg)
    beamformer = make_beamformer(setup.pixel_grid, bank, setup.sequence, cfg.beamforming)
    variance = electronic_noise_variance(bank, setup.sequence, cfg.noise)
    shape = (setup.freqs.shape[0], len(setup.sequence.angles_deg), setup.transducer.n_elements)
    residual = 0.0
    swapped = _NoiseSwappedTree(seed, OTHER_NOISE_ROOT)
    for tree, noisy in ((SeedTree(seed), frames), (swapped, acquire(swapped, cfg))):
        for n in range(N_FRAMES):
            noise_image = beamform(beamformer, electronic_noise(tree, n, shape, variance))
            residual = max(residual, float(np.abs(noisy[n] - noise_free[n] - noise_image).max()))
    noise_free_residual = residual / float(np.abs(noise_free).max())

    report(
        EVAL_ID,
        {
            "repeat_difference": repeat_difference,
            "offset_difference": offset_difference,
            "noise_free_residual": noise_free_residual,
        },
        passed=repeat_difference == 0.0
        and offset_difference == 0.0
        and noise_free_residual < MAX_RESIDUAL,
    )
