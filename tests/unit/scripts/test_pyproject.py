"""pyproject.toml: the Python floor and the runtime dependencies match the code."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Import name of each distribution whose name differs from it.
IMPORT_NAMES = {"pyyaml": "yaml", "k-wave-python": "kwave"}


def _project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_python_floor_is_the_one_the_pinned_dependencies_need():
    # torch 2.15, xraydb 4.6.1 (and tomllib, used above) need Python 3.11.
    assert _project()["requires-python"] == ">=3.11"


def test_every_runtime_dependency_is_imported_by_the_package_or_the_scripts():
    sources = [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").glob("*.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    distributions = [requirement.split("==")[0] for requirement in _project()["dependencies"]]
    assert "numpy" in distributions and "scikit-image" not in distributions
    for distribution in distributions:
        module = IMPORT_NAMES.get(distribution, distribution)
        pattern = rf"^\s*(import|from) {re.escape(module)}\b"
        assert re.search(pattern, text, flags=re.MULTILINE), f"{distribution} is never imported"
