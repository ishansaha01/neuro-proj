"""Run the evals with quiet output.

Usage: ``python scripts/run_evals.py [area ...]`` where an area is one of the
directories under ``evals/`` (``common``, ``head``, ``fus``, ...). Without
arguments every eval runs. Results go to ``evals/results/`` (or the directory
named by the environment variable ``NEUROSIM_EVAL_RESULTS``); run
``scripts/build_registry.py`` afterwards to refresh ``evals/REGISTRY.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"


def main(areas: list[str]) -> int:
    """Run pytest on the given eval areas (all of ``evals/`` if none); return its exit code."""
    targets = [EVALS_DIR / area.lower() for area in areas] or [EVALS_DIR]
    missing = [str(target) for target in targets if not target.is_dir()]
    if missing:
        print(f"no such eval area: {', '.join(missing)}", file=sys.stderr)
        return 2
    return int(pytest.main(["-q", *(str(target) for target in targets)]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
