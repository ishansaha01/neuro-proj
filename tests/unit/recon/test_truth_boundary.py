"""The reconstruction reads ground truth only where it is allowed to.

`recon.scanner` is the one module that may read the true skull bank, the true
operators and the true CBV for the reconstruction itself. This test parses every
module of `neurosim.recon` and fails if an attribute named like a ground-truth
field of `Scene` or `TrialSet` is read anywhere else, except in the places
listed in `ALLOWED` with their reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import neurosim.recon

# Attribute names that are ground truth of a scene or a trial set: the true-skull and
# no-skull banks, the true phase screen, the true CBV maps and series, and the operators
# (every one of them contains the true medium).
TRUTH_FIELDS = frozenset(
    {"bank_skull", "bank_clean", "screen_true", "cbv_baseline", "activation", "cbv", "operators"}
)
# module -> functions that may read truth fields (None: the whole module), with the reason.
ALLOWED: dict[str, tuple[frozenset[str] | None, str]] = {
    "scanner.py": (None, "the emulation of the scanner: the truth boundary itself"),
    "synthetic.py": (None, "creates the synthetic phase-screen scenes, truth included"),
    "family_scene.py": (None, "creates stand-in scenes whose true skull is a family member"),
    "cbv_images.py": (None, "samples CBV images of simulated heads as training data of the prior"),
    "medium_experiment.py": (
        frozenset({"_truth", "skull_diagnostics"}),
        "final metrics of the second design",
    ),
    "experiment.py": (
        frozenset(
            {"_mean_deviance_per_pixel", "reconstruct_scene", "scene_metrics", "example_images"}
        ),
        "first design: its oracle rows (the true operator of the uncorrected image), "
        "model-mismatch diagnostics and final metrics",
    ),
}


def _truth_reads(path: Path) -> list[tuple[str, int, str]]:
    """`(enclosing top-level function, line, attribute)` of every truth-field read in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reads = []
    for top in tree.body:
        owner = getattr(top, "name", "<module>")
        for node in ast.walk(top):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in TRUTH_FIELDS
            ):
                reads.append((owner, node.lineno, node.attr))
    return reads


def test_only_listed_places_read_ground_truth():
    package = Path(neurosim.recon.__file__).parent
    violations = []
    for path in sorted(package.glob("*.py")):
        functions, _ = ALLOWED.get(path.name, (frozenset(), ""))
        if functions is None:
            continue
        violations += [
            f"{path.name}:{line} reads .{attribute} in {owner}"
            for owner, line, attribute in _truth_reads(path)
            if owner not in functions
        ]
    assert not violations, "ground truth read outside recon.scanner:\n" + "\n".join(violations)


def test_the_check_sees_a_truth_read(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("def reconstruct(scene):\n    return scene.bank_skull.B\n", encoding="utf-8")
    assert _truth_reads(probe) == [("reconstruct", 2, "bank_skull")]


def test_every_exception_names_an_existing_module():
    package = Path(neurosim.recon.__file__).parent
    assert all((package / name).is_file() for name in ALLOWED)
