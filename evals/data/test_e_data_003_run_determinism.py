"""E-DATA-003: one config and one seed give a byte-identical run bundle in separate processes."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from neurosim.common.evalkit import REGISTRY, eval_case, report

EVAL_ID = "E-DATA-003"
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "run_small.yaml"
SCRIPT = REPO / "scripts" / "run_pipeline.py"


def _run_in_new_process(seed: int, out: Path, hash_seed: str) -> str:
    """Run the pipeline script in a fresh interpreter; return the SHA-256 of its bundle."""
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--seed", str(seed)]
        + ["--out", str(out)],
        check=True,
        capture_output=True,
        env=environment,
    )
    return hashlib.sha256(out.read_bytes()).hexdigest()


@eval_case(
    id=EVAL_ID,
    claim=(
        "the whole pipeline (head, plane, banks, CT prior, operators, image series, bundle) is "
        "deterministic: the same config and root seed give the same run bundle, byte for byte, "
        "in two separate processes, and another seed gives another bundle"
    ),
    reference=(
        "Bitwise comparison (SHA-256) of the bundles written by scripts/run_pipeline.py in "
        "independent interpreters started with different PYTHONHASHSEED values"
    ),
    metric=(
        "repeat_differs: 1 if the two bundles of the same seed differ, else 0; "
        "other_seed_identical: 1 if the bundle of the next seed equals them, else 0"
    ),
    threshold=(
        "repeat_differs = 0 and other_seed_identical = 0 (fixed in advance: determinism is "
        "exact, there is no tolerance; the second number guards against a bundle that does "
        "not depend on the seed at all)"
    ),
    seeds=(0,),
)
def test_e_data_003_run_determinism(tmp_path) -> None:
    seed = REGISTRY[EVAL_ID].seeds[0]
    first = _run_in_new_process(seed, tmp_path / "first.h5", hash_seed="1")
    second = _run_in_new_process(seed, tmp_path / "second.h5", hash_seed="2")
    other = _run_in_new_process(seed + 1, tmp_path / "other.h5", hash_seed="1")
    values = {
        "repeat_differs": float(first != second),
        "other_seed_identical": float(other == first),
        "bundle_bytes": float((tmp_path / "first.h5").stat().st_size),
    }
    report(
        EVAL_ID,
        values,
        passed=values["repeat_differs"] == 0.0 and values["other_seed_identical"] == 0.0,
    )
