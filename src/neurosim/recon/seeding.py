"""The single place that seeds and configures torch (docs/ARCHITECTURE.md section 1).

Every torch random draw in `recon` takes an explicit `torch.Generator` built
here from a `SeedTree` stream, so nothing depends on torch's global generator.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from neurosim.common.seeding import SeedTree

TRAINING_STREAM = "diffusion_training"
SAMPLING_STREAM = "recon_sampling"
_TORCH_STREAMS = (TRAINING_STREAM, SAMPLING_STREAM)


class TrainingDraw(IntEnum):
    """Sub-indices of the `diffusion_training` stream, one per independent use."""

    WEIGHT_INIT = 0  # initial network weights
    BATCHES = 1  # image indices, timesteps and noise of every training step
    HELDOUT_NOISE = 2  # the fixed noise of the held-out loss


def torch_generator(tree: SeedTree, stream: str, *sub: int) -> torch.Generator:
    """Return a CPU generator seeded from ``tree.seed_int(stream, *sub)``.

    tree: seed tree of the run.
    stream: `TRAINING_STREAM` or `SAMPLING_STREAM` (``ValueError`` otherwise).
    *sub: sub-indices of the stream, e.g. a `TrainingDraw` or a sample number.
    """
    if stream not in _TORCH_STREAMS:
        raise ValueError(f"torch is seeded from {_TORCH_STREAMS} only, got {stream!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(tree.seed_int(stream, *sub))
    return generator


def configure_torch(num_threads: int) -> None:
    """Fix what CPU results depend on besides the seed.

    num_threads: intra-op threads. Floating-point sums are split differently for
        different thread counts, so results are bit-reproducible only for a fixed
        seed *and* a fixed thread count.

    Also makes torch raise on any operation without a deterministic implementation.
    """
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
