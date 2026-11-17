"""Eval harness: register one claim per eval and record its result.

An eval is a plain pytest test in ``evals/<area>/test_e_<area>_<nnn>_<slug>.py``::

    @eval_case(
        id="E-FUS-003",
        claim="Plane-wave arrival time through a homogeneous slab matches the analytic delay",
        reference="Analytic: dt = d * (1/c_slab - 1/c_water)",
        metric="Maximum absolute error in arrival time [samples]",
        threshold="< 1 sample: the delay is resolved to the sampling period",
        seeds=(0,),
    )
    def test_e_fus_003_slab_delay():
        error = ...
        report("E-FUS-003", error, passed=error < 1.0)

The threshold text states the bar and why, and is fixed before the result is
looked at (CLAUDE.md section 1). ``scripts/build_registry.py`` turns the
registered specs and the latest results into ``evals/REGISTRY.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from neurosim.common.io import git_commit

F = TypeVar("F", bound=Callable[..., object])

AREAS: tuple[str, ...] = ("HEAD", "FUS", "CT", "DATA", "RECON", "COMMON")
_ID_PATTERN = re.compile(rf"E-({'|'.join(AREAS)})-\d{{3}}")

# Result files live in <repo>/evals/results/ (this file is <repo>/src/neurosim/common/evalkit.py).
RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / "evals" / "results"


@dataclass(frozen=True)
class EvalSpec:
    """Registry entry of one eval. ``area`` is the lower-case area of the id (``"fus"``)."""

    id: str
    claim: str
    reference: str
    metric: str
    threshold: str
    seeds: tuple[int, ...]
    area: str


REGISTRY: dict[str, EvalSpec] = {}
# Where each id was registered, for the duplicate-id error message.
_ORIGINS: dict[str, str] = {}


def eval_case(
    *,
    id: str,
    claim: str,
    reference: str,
    metric: str,
    threshold: str,
    seeds: tuple[int, ...] = (0,),
) -> Callable[[F], F]:
    """Register a pytest test function as an eval and mark it ``pytest.mark.eval``.

    Parameters
    ----------
    id:
        ``E-<AREA>-<nnn>`` with AREA in `AREAS`; must be unique.
    claim:
        The single claim the eval checks.
    reference:
        What the output is compared with, independent of our code (analytic
        solution, published measurement, conservation law).
    metric:
        The number that is reported, with its unit.
    threshold:
        The bar and why, in words; fixed before looking at the result.
    seeds:
        The fixed root seeds the eval runs with.

    The test function itself is returned unchanged apart from the mark.
    """
    match = _ID_PATTERN.fullmatch(id)
    if match is None:
        raise ValueError(f"eval id {id!r} does not match E-({'|'.join(AREAS)})-<3 digits>")
    for name, text in (
        ("claim", claim),
        ("reference", reference),
        ("metric", metric),
        ("threshold", threshold),
    ):
        if not text.strip():
            raise ValueError(f"eval {id}: {name} must not be empty")
    seeds = tuple(seeds)
    if not seeds or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError(f"eval {id}: seeds must be a non-empty tuple of ints, got {seeds!r}")
    spec = EvalSpec(
        id=id,
        claim=claim,
        reference=reference,
        metric=metric,
        threshold=threshold,
        seeds=seeds,
        area=match.group(1).lower(),
    )

    def decorate(function: F) -> F:
        origin = f"{function.__module__}.{function.__qualname__}"
        if id in REGISTRY:
            raise ValueError(f"duplicate eval id {id}: {_ORIGINS[id]} and {origin}")
        REGISTRY[id] = spec
        _ORIGINS[id] = origin
        return pytest.mark.eval(function)

    return decorate


def report(eval_id: str, value: float | Mapping[str, float], passed: bool, note: str = "") -> None:
    """Record the result of an eval, then fail the test if it did not pass.

    Writes ``RESULTS_DIR/<eval_id>.json`` with the value(s), the verdict, the
    note and the git commit. Call it once per eval, as its last statement;
    report several numbers (for example one per seed) as a mapping.

    Parameters
    ----------
    eval_id:
        Id of a registered eval (``KeyError`` otherwise).
    value:
        The metric, or a name -> number mapping, in the unit stated by the
        eval's ``metric``.
    passed:
        Whether the value meets the registered threshold.
    note:
        Optional free text stored with the result.
    """
    spec = REGISTRY[eval_id]
    if isinstance(value, Mapping):
        stored: float | dict[str, float] = {str(key): float(item) for key, item in value.items()}
    else:
        stored = float(value)
    result = {
        "id": eval_id,
        "value": stored,
        "passed": bool(passed),
        "note": note,
        "git_commit": git_commit(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{eval_id}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise AssertionError(
            f"{eval_id} failed: {spec.metric} = {stored}; threshold: {spec.threshold}"
            + (f"; {note}" if note else "")
        )
