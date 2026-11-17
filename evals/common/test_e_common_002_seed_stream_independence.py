"""E-COMMON-002: a seed stream depends only on (root seed, name, sub-index)."""

from __future__ import annotations

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import STREAMS, SeedTree

EVAL_ID = "E-COMMON-002"
N_DRAWS = 8
SUB_INDICES: tuple[tuple[int, ...], ...] = ((), (0,), (1,), (0, 0), (3, 7))


def _draws(tree: SeedTree, name: str, sub: tuple[int, ...]) -> tuple[int, ...]:
    """Return the stream's integer seed followed by its first ``N_DRAWS`` 63-bit draws."""
    values = tree.rng(name, *sub).integers(0, 2**63, size=N_DRAWS)
    return (tree.seed_int(name, *sub), *(int(value) for value in values))


def _disturb(tree: SeedTree, step: int) -> None:
    """Draw a varying amount from two other streams, to expose any shared state."""
    for offset in (1, 2):
        tree.rng(STREAMS[(step + offset) % len(STREAMS)], step).normal(size=3 + step % 5)


@eval_case(
    id=EVAL_ID,
    claim=(
        "a seed stream depends only on (root seed, name, sub-index): draw order and other "
        "streams do not change it, and different names/roots give different draws"
    ),
    reference=(
        "Each stream drawn alone from a fresh SeedTree (bitwise comparison), and the "
        "distinct-stream guarantee of numpy.random.SeedSequence spawn keys"
    ),
    metric=(
        "order_dependent_streams: streams whose draws change when drawn in reverse order "
        "between draws from other streams; duplicate_streams: streams sharing their seed "
        "and first 8 draws with an earlier stream, over all names, sub-indices and roots"
    ),
    threshold=(
        "0 and 0: both properties are exact, and a chance match of a 63-bit seed plus eight "
        "63-bit draws has probability about 2^-567 per pair"
    ),
    seeds=(0, 1, 20261117),
)
def test_e_common_002_seed_stream_independence() -> None:
    keys = [(name, sub) for name in STREAMS for sub in SUB_INDICES]
    order_dependent = 0
    all_draws = []
    for root in REGISTRY[EVAL_ID].seeds:
        alone = {key: _draws(SeedTree(root), *key) for key in keys}
        shared_tree = SeedTree(root)
        for step, key in enumerate(reversed(keys)):
            _disturb(shared_tree, step)
            order_dependent += _draws(shared_tree, *key) != alone[key]
        all_draws.extend(alone.values())

    duplicates = len(all_draws) - len(set(all_draws))
    report(
        EVAL_ID,
        {"order_dependent_streams": order_dependent, "duplicate_streams": duplicates},
        passed=order_dependent == 0 and duplicates == 0,
    )
