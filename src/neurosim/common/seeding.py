"""Seed tree: the only source of randomness in the project (ROADMAP D-004).

One root seed per run is split into named streams. A stream is identified by
``(root_seed, name, sub)`` alone, so drawing from one stream, or changing the
order in which streams are drawn, never changes another stream.

Streams are built on ``numpy.random.SeedSequence``: the root seed is the
entropy and the stream identity is the spawn key (the mechanism behind
``SeedSequence.spawn``; see the numpy documentation, "Parallel random number
generation").
"""

from __future__ import annotations

import operator

import numpy as np

# Append only: a stream's index is part of its identity, so reordering or
# removing names would change every existing output.
STREAMS: tuple[str, ...] = (
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

# SeedSequence consumes spawn-key entries as 32-bit words and splits larger
# integers into several words, so (2**32,) would collide with (0, 1).
_WORD_LIMIT = 2**32
# First spawn-key word of a child subtree. Stream indices are far below it, so
# a child key can never be mistaken for a stream key.
_CHILD_MARKER = _WORD_LIMIT - 1
# Root seeds are stored as HDF5 int64 attributes and may come from `seed_int`.
_ROOT_SEED_LIMIT = 2**63


def _as_index(value: int, what: str, limit: int) -> int:
    """Return ``value`` as a plain int after checking ``0 <= value < limit``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got bool")
    try:
        index = operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {type(value).__name__}") from None
    if not 0 <= index < limit:
        raise ValueError(f"{what} must satisfy 0 <= value < {limit}, got {index}")
    return index


class SeedTree:
    """Named, order-independent random streams derived from one root seed.

    Parameters
    ----------
    root_seed:
        Root seed of the run, ``0 <= root_seed < 2**63``.

    Examples
    --------
    >>> tree = SeedTree(7)
    >>> noise = tree.rng("us_electronic_noise")            # one stream
    >>> frame_rng = tree.rng("blood_scatterers", 12)       # sub-stream for frame 12
    >>> sample_tree = tree.child(3)                        # subtree for dataset sample 3
    """

    def __init__(self, root_seed: int):
        self.root_seed: int = _as_index(root_seed, "root_seed", _ROOT_SEED_LIMIT)
        self._prefix: tuple[int, ...] = ()

    def __repr__(self) -> str:
        return f"SeedTree(root_seed={self.root_seed}, prefix={self._prefix})"

    def _sequence(self, name: str, sub: tuple[int, ...]) -> np.random.SeedSequence:
        if name not in STREAMS:
            raise KeyError(f"unknown seed stream {name!r}; add it to the end of STREAMS")
        key = (STREAMS.index(name), *(_as_index(s, "sub-index", _WORD_LIMIT) for s in sub))
        return np.random.SeedSequence(entropy=self.root_seed, spawn_key=(*self._prefix, *key))

    def rng(self, name: str, *sub: int) -> np.random.Generator:
        """Return a fresh generator positioned at the start of a stream.

        Parameters
        ----------
        name:
            Stream name; must be in ``STREAMS`` (``KeyError`` otherwise).
        *sub:
            Optional sub-indices (frame number, element number, ...), each
            ``0 <= s < 2**32``. ``rng(name)``, ``rng(name, 0)`` and
            ``rng(name, 0, 0)`` are three different streams.
        """
        return np.random.default_rng(self._sequence(name, sub))

    def seed_int(self, name: str, *sub: int) -> int:
        """Return a 63-bit non-negative integer seed for a stream.

        For libraries that take an integer seed (for example torch). Arguments
        are as for `rng`.
        """
        state = self._sequence(name, sub).generate_state(1, dtype=np.uint64)
        return int(state[0] >> np.uint64(1))

    def child(self, *sub: int) -> SeedTree:
        """Return an independent subtree, for example one per dataset sample.

        The subtree has the same stream names as its parent, but every stream
        differs from the parent's and from every other child's. Children nest:
        ``tree.child(1).child(2)`` and ``tree.child(1, 2)`` are different
        subtrees.

        Parameters
        ----------
        *sub:
            One or more indices, each ``0 <= s < 2**32``.
        """
        if not sub:
            raise ValueError("child() needs at least one index")
        indices = tuple(_as_index(s, "child index", _WORD_LIMIT) for s in sub)
        tree = SeedTree(self.root_seed)
        # The length word keeps the key uniquely decodable: without it
        # child(1).rng(STREAMS[0], 5) and child(1, 0).rng(STREAMS[5]) would
        # share the key (marker, 1, 0, 5).
        tree._prefix = (*self._prefix, _CHILD_MARKER, len(indices), *indices)
        return tree
