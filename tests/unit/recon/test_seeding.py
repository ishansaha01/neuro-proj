import pytest
import torch

from neurosim.common.seeding import SeedTree
from neurosim.recon.seeding import (
    SAMPLING_STREAM,
    TRAINING_STREAM,
    TrainingDraw,
    configure_torch,
    torch_generator,
)


def _draw(generator: torch.Generator) -> torch.Tensor:
    return torch.randn(8, generator=generator)


def test_generator_is_seeded_from_the_tree():
    tree = SeedTree(11)
    generator = torch_generator(tree, TRAINING_STREAM, TrainingDraw.BATCHES)
    reference = torch.Generator().manual_seed(tree.seed_int("diffusion_training", 1))
    assert torch.equal(_draw(generator), _draw(reference))


def test_same_stream_repeats_and_other_streams_differ():
    tree = SeedTree(11)
    first = _draw(torch_generator(tree, SAMPLING_STREAM, 3))
    assert torch.equal(first, _draw(torch_generator(tree, SAMPLING_STREAM, 3)))
    assert not torch.equal(first, _draw(torch_generator(tree, SAMPLING_STREAM, 4)))
    assert not torch.equal(first, _draw(torch_generator(tree, TRAINING_STREAM, 3)))
    assert not torch.equal(first, _draw(torch_generator(SeedTree(12), SAMPLING_STREAM, 3)))


def test_generator_does_not_touch_the_global_generator():
    state = torch.random.get_rng_state()
    _draw(torch_generator(SeedTree(0), TRAINING_STREAM))
    assert torch.equal(state, torch.random.get_rng_state())


def test_only_the_two_torch_streams_are_allowed():
    with pytest.raises(ValueError, match="torch is seeded from"):
        torch_generator(SeedTree(0), "cbv")


def test_training_draws_are_distinct():
    assert len({int(draw) for draw in TrainingDraw}) == len(TrainingDraw)


def test_configure_torch_sets_threads_and_determinism():
    configure_torch(1)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError, match="num_threads"):
        configure_torch(0)
