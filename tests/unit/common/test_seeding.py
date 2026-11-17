import numpy as np
import pytest

from neurosim.common.seeding import STREAMS, SeedTree


def first_draws(rng: np.random.Generator) -> list[int]:
    return rng.integers(0, 2**63, size=4).tolist()


def test_documented_streams_keep_their_positions():
    # Append-only: the index of a name is part of every stream's identity.
    assert STREAMS[:14] == (
        "anatomy",
        "skull_microstructure",
        "hidden_soundspeed",
        "cbv",
        "tissue_scatterers",
        "blood_scatterers",
        "us_electronic_noise",
        "ct_photon_noise",
        "tissue_motion",
        "ct_hu_mapping",
        "dataset_sweep",
        "speckle_statistical",
        "diffusion_training",
        "recon_sampling",
    )
    assert len(set(STREAMS)) == len(STREAMS)


def test_stream_matches_documented_seed_sequence_construction():
    sequence = np.random.SeedSequence(entropy=42, spawn_key=(STREAMS.index("cbv"), 3, 1))
    expected = first_draws(np.random.default_rng(sequence))
    assert first_draws(SeedTree(42).rng("cbv", 3, 1)) == expected


def test_stream_does_not_depend_on_draw_order_or_other_streams():
    forward = SeedTree(5)
    forward_draws = {name: first_draws(forward.rng(name)) for name in STREAMS}
    backward = SeedTree(5)
    for name in reversed(STREAMS):
        backward.rng("anatomy", 9).normal(size=100)
        assert first_draws(backward.rng(name)) == forward_draws[name]


def test_each_call_returns_a_fresh_generator_at_the_stream_start():
    tree = SeedTree(1)
    first = tree.rng("cbv")
    first.normal(size=50)
    assert first_draws(tree.rng("cbv")) == first_draws(SeedTree(1).rng("cbv"))
    assert tree.rng("cbv") is not tree.rng("cbv")


def test_names_sub_indices_and_roots_give_different_streams():
    tree = SeedTree(0)
    draws = [
        first_draws(tree.rng("anatomy")),
        first_draws(tree.rng("cbv")),
        first_draws(tree.rng("anatomy", 0)),
        first_draws(tree.rng("anatomy", 1)),
        first_draws(tree.rng("anatomy", 0, 0)),
        first_draws(SeedTree(1).rng("anatomy")),
    ]
    assert len({tuple(d) for d in draws}) == len(draws)


def test_seed_int_is_a_reproducible_63_bit_integer():
    tree = SeedTree(3)
    seed = tree.seed_int("diffusion_training", 2)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**63
    assert seed == SeedTree(3).seed_int("diffusion_training", 2)
    assert seed != tree.seed_int("diffusion_training", 3)
    assert seed != tree.seed_int("recon_sampling", 2)


def test_child_subtrees_are_reproducible_and_independent():
    tree = SeedTree(7)
    child = tree.child(4)
    assert child.root_seed == 7
    assert first_draws(child.rng("cbv")) == first_draws(SeedTree(7).child(4).rng("cbv"))
    draws = [
        first_draws(tree.rng("cbv")),
        first_draws(tree.rng("cbv", 4)),
        first_draws(child.rng("cbv")),
        first_draws(tree.child(5).rng("cbv")),
        first_draws(tree.child(4, 0).rng("cbv")),
        first_draws(child.child(0).rng("cbv")),
        first_draws(SeedTree(8).child(4).rng("cbv")),
    ]
    assert len({tuple(d) for d in draws}) == len(draws)


def test_child_keys_cannot_collide_with_stream_sub_indices():
    tree = SeedTree(7)
    via_sub = first_draws(tree.child(1).rng(STREAMS[0], 5))
    via_child = first_draws(tree.child(1, 0).rng(STREAMS[5]))
    assert via_sub != via_child


def test_unknown_stream_raises_key_error():
    tree = SeedTree(0)
    with pytest.raises(KeyError, match="no_such_stream"):
        tree.rng("no_such_stream")
    with pytest.raises(KeyError, match="no_such_stream"):
        tree.seed_int("no_such_stream")


@pytest.mark.parametrize("root_seed", [-1, 2**63])
def test_root_seed_out_of_range_is_rejected(root_seed):
    with pytest.raises(ValueError, match="root_seed"):
        SeedTree(root_seed)


@pytest.mark.parametrize("root_seed", [True, 1.5, "7"])
def test_root_seed_must_be_an_integer(root_seed):
    with pytest.raises(TypeError, match="root_seed"):
        SeedTree(root_seed)


def test_sub_indices_are_validated():
    tree = SeedTree(0)
    # 2**32 would be split into the words (0, 1) and collide with sub-index (0, 1).
    with pytest.raises(ValueError, match="sub-index"):
        tree.rng("cbv", 2**32)
    with pytest.raises(ValueError, match="sub-index"):
        tree.rng("cbv", -1)
    with pytest.raises(ValueError, match="child index"):
        tree.child(2**32)
    with pytest.raises(ValueError, match="at least one"):
        tree.child()


def test_numpy_integers_are_accepted_as_indices():
    tree = SeedTree(np.int64(3))
    assert tree.root_seed == 3 and type(tree.root_seed) is int
    assert first_draws(tree.rng("cbv", np.int32(2))) == first_draws(SeedTree(3).rng("cbv", 2))
