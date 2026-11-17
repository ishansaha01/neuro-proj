"""E-HEAD-005: a head is reproducible, and the cbv stream changes only the CBV fields."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.types import HeadVolume

EVAL_ID = "E-HEAD-005"
ARRAY_FIELDS = tuple(
    f.name for f in dataclasses.fields(HeadVolume) if f.name not in ("grid", "alpha_power")
)
# Fields fed by the cbv stream. `activation` is not among them: it is set by the
# anatomy and the configured target direction alone.
CBV_STREAM_FIELDS = ("cbv_baseline", "blood_velocity")


class CbvPerturbedTree(SeedTree):
    """A seed tree whose `cbv` stream, and only that stream, is replaced by another one."""

    def rng(self, name: str, *sub: int) -> np.random.Generator:
        if name == "cbv":
            return super().rng(name, *sub, 1)
        return super().rng(name, *sub)


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@eval_case(
    id=EVAL_ID,
    claim="Same config and seed give a byte-identical head, and replacing the cbv stream "
    "changes the CBV fields while every other field stays byte-identical",
    reference="Determinism requirement of ROADMAP D-004 (independent named streams)",
    metric="Number of fields that break the expectation, over seeds: any field differing "
    "between two runs, any non-CBV field changed by the cbv stream, any cbv-stream field "
    "left unchanged by it",
    threshold="0: byte equality has no tolerance",
    seeds=(0, 1),
)
def test_e_head_005_determinism(small_cfg):
    broken = 0
    for seed in REGISTRY[EVAL_ID].seeds:
        first = generate_head(small_cfg, SeedTree(seed))
        again = generate_head(small_cfg, SeedTree(seed))
        perturbed = generate_head(small_cfg, CbvPerturbedTree(seed))
        broken += int(first.grid != again.grid) + int(first.grid != perturbed.grid)
        for name in ARRAY_FIELDS:
            broken += int(not same_bytes(getattr(first, name), getattr(again, name)))
            unchanged = same_bytes(getattr(first, name), getattr(perturbed, name))
            broken += int(unchanged == (name in CBV_STREAM_FIELDS))
    report(EVAL_ID, broken, passed=broken == 0)
