"""E-HEAD-008: baseline CBV is higher in grey than in white matter, in reported ranges."""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.head.generate import generate_head
from neurosim.head.types import BRAIN_LABELS, Tissue

EVAL_ID = "E-HEAD-008"
GREY_RANGE = (0.035, 0.052)
WHITE_RANGE = (0.017, 0.027)
SPACING = 1.0e-3


@eval_case(
    id=EVAL_ID,
    claim="Mean baseline CBV is higher in grey than in white matter, both within reported "
    "blood volume fractions, and CBV is zero outside the brain",
    reference="Grey 3.5 +- 0.4 and white 1.7 +- 0.4 ml/100 g (Wikipedia, 'Cerebral blood "
    "volume') up to grey 5.2 and white 2.7 ml/100 ml (Leenders et al. 1990, Brain 113:27; "
    "quoted from memory, to verify)",
    metric="Number of failed checks over seeds (grey and white means of the first seed "
    "are reported alongside)",
    threshold="0 failed: grey mean in 0.035-0.052, white mean in 0.017-0.027 (the spans of "
    "the reported values), grey above white, and no nonzero CBV outside the brain. Fixed "
    "before the first run",
    seeds=(0, 1, 2),
)
def test_e_head_008_cbv_fractions(small_cfg):
    cfg = dataclasses.replace(small_cfg, grid=dataclasses.replace(small_cfg.grid, spacing=SPACING))
    failed = 0
    values: dict[str, float] = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        head = generate_head(cfg, SeedTree(seed))
        grey = float(head.cbv_baseline[head.labels == Tissue.GREY].mean())
        white = float(head.cbv_baseline[head.labels == Tissue.WHITE].mean())
        outside_brain = ~np.isin(head.labels, BRAIN_LABELS)
        failed += int(not GREY_RANGE[0] <= grey <= GREY_RANGE[1])
        failed += int(not WHITE_RANGE[0] <= white <= WHITE_RANGE[1])
        failed += int(grey <= white)
        failed += int(head.cbv_baseline[outside_brain].any())
        if not values:
            values = {"grey_mean": grey, "white_mean": white}
    report(EVAL_ID, {"failed": failed, **values}, passed=failed == 0)
