"""One config and one seed give one run: the scene and its image series.

docs/ARCHITECTURE.md, section 7. Saving is `data.io.save_run`; a run is
reproducible from the config, the root seed and the code version stored there.
"""

from __future__ import annotations

from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene_with_extras
from neurosim.data.trials import simulate_trials
from neurosim.data.types import Scene, TrialSet


def run_tree(cfg: RunConfig, tree: SeedTree) -> tuple[Scene, TrialSet]:
    """Run the pipeline with a given seed tree (a root tree, or the subtree of a sweep scene)."""
    scene, extras = build_scene_with_extras(cfg, tree)
    return scene, simulate_trials(scene, cfg, tree, extras)


def run(cfg: RunConfig, root_seed: int) -> tuple[Scene, TrialSet]:
    """Run the pipeline: head -> plane -> banks -> CT prior -> operators -> image series.

    Args:
        cfg: the run configuration.
        root_seed: root seed of the run's `SeedTree`.

    Returns:
        the `Scene` (static part) and the `TrialSet` (CBV and power Doppler series).
    """
    return run_tree(cfg, SeedTree(root_seed))
