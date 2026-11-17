"""E-COMMON-001: save_bundle is byte-deterministic across processes.

Run as a script (``python <this file> <path> <seed>``) it writes the test
bundle to ``path``; the eval uses that to write the same bundle from two
fresh interpreter processes.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import neurosim
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.io import save_bundle
from neurosim.common.seeding import SeedTree

EVAL_ID = "E-COMMON-001"
# HDF5 timestamps have a resolution of 1 s, so the two processes must write in
# different seconds for a stored timestamp to show up as a difference.
SECONDS_BETWEEN_WRITES = 1.1


def make_bundle(seed: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return arrays and attributes covering every dtype and attribute kind a bundle holds."""
    rng = SeedTree(seed).rng("anatomy")
    arrays = {
        "head/sound_speed": rng.normal(1540.0, 50.0, size=(6, 5, 4)).astype(np.float32),
        "head/labels": rng.integers(0, 5, size=(6, 5, 4), dtype=np.uint8),
        "fus/iq/frames": (rng.normal(size=(3, 8)) + 1j * rng.normal(size=(3, 8))).astype(
            np.complex64
        ),
        "mask": rng.random(size=(7,)) > 0.5,
        "duration": np.float64(0.25),
    }
    attrs = {
        "root_seed": seed,
        "spacing": 5.0e-4,
        "name": "determinism bundle",
        "noise_on": True,
        "config": {"fus": {"f0": 2.0e6, "angles_deg": [-10.0, 0.0, 10.0]}, "ct": {"kvp": 120}},
    }
    return arrays, attrs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_from_new_process(path: Path, seed: int, hash_seed: int) -> str:
    """Write the bundle from a fresh interpreter and return the SHA-256 of the file."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONPATH"] = str(Path(neurosim.__file__).resolve().parents[1])
    subprocess.run([sys.executable, __file__, str(path), str(seed)], check=True, env=env)
    return _sha256(path)


@eval_case(
    id=EVAL_ID,
    claim=(
        "save_bundle is byte-deterministic: identical inputs give identical file bytes "
        "across processes"
    ),
    reference=(
        "SHA-256 (hashlib) of the files written by two fresh interpreter processes, started "
        "more than 1 s apart with different PYTHONHASHSEED, and by the test process with the "
        "mappings in reversed order"
    ),
    metric="Number of differing SHA-256 hashes (distinct hashes minus one)",
    threshold="0: byte identity is exact, so there is no tolerance to choose",
    seeds=(0,),
)
def test_e_common_001_bundle_determinism(tmp_path: Path) -> None:
    (seed,) = REGISTRY[EVAL_ID].seeds
    hashes = [_hash_from_new_process(tmp_path / "first.h5", seed, hash_seed=1)]
    time.sleep(SECONDS_BETWEEN_WRITES)
    hashes.append(_hash_from_new_process(tmp_path / "second.h5", seed, hash_seed=2))

    arrays, attrs = make_bundle(seed)
    save_bundle(
        tmp_path / "reversed.h5", dict(reversed(arrays.items())), dict(reversed(attrs.items()))
    )
    hashes.append(_sha256(tmp_path / "reversed.h5"))

    n_differing = len(set(hashes)) - 1
    report(EVAL_ID, n_differing, passed=n_differing == 0)


if __name__ == "__main__":
    save_bundle(sys.argv[1], *make_bundle(int(sys.argv[2])))
