"""E-DATA-010: a sweep scene does not depend on which other scenes are generated, or when."""

from __future__ import annotations

from pathlib import Path

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.data.sweep import INDEX_NAME, SweepConfig, run_sweep, scene_file

EVAL_ID = "E-DATA-010"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sweep_small.yaml"
LONE_SCENE = 1


def _differing_files(first: Path, second: Path, names: list[str]) -> int:
    return sum((first / name).read_bytes() != (second / name).read_bytes() for name in names)


@eval_case(
    id=EVAL_ID,
    claim=(
        "scene i of a sweep is byte-identical whether it is generated alone or as part of the "
        "full sweep, and a sweep resumed after an interruption (one scene done, one bundle cut "
        "short) equals an uninterrupted one, index included"
    ),
    reference=(
        "Bitwise comparison of the bundles and of index.json between three dataset "
        "directories of the same sweep config and root seed (ROADMAP D-004: scene i draws "
        "only from the seed subtree child(i))"
    ),
    metric=(
        "lone_scene_differs: 1 if scene 1 generated alone differs from scene 1 of the full "
        "sweep, else 0; resumed_files_differing: number of files (3 bundles and the index) "
        "that differ between the resumed and the uninterrupted sweep; "
        "scenes_identical_to_each_other: number of equal pairs among the 3 bundles"
    ),
    threshold=(
        "all three numbers = 0 (fixed in advance: reproducibility is exact; the third number "
        "guards against scenes that ignore their index)"
    ),
    seeds=(11,),
)
def test_e_data_010_sweep_reproducibility(tmp_path) -> None:
    cfg = from_dict(SweepConfig, load_yaml(CONFIG))
    seed = REGISTRY[EVAL_ID].seeds[0]
    bundles = [scene_file(index) for index in range(cfg.n_scenes)]
    full, lone, resumed = (tmp_path / name for name in ("full", "lone", "resumed"))

    run_sweep(cfg, seed, full)
    run_sweep(cfg, seed, lone, scenes=(LONE_SCENE,))
    # An interrupted sweep: scene 0 complete, the bundle of scene 1 cut short, scene 2 missing.
    run_sweep(cfg, seed, resumed, scenes=(0,))
    cut_short = (full / bundles[1]).read_bytes()[:100_000]
    (resumed / bundles[1]).write_bytes(cut_short)
    run_sweep(cfg, seed, resumed)

    contents = [(full / name).read_bytes() for name in bundles]
    values = {
        "lone_scene_differs": float(_differing_files(full, lone, [bundles[LONE_SCENE]])),
        "resumed_files_differing": float(_differing_files(full, resumed, [*bundles, INDEX_NAME])),
        "scenes_identical_to_each_other": float(
            sum(contents[i] == contents[j] for i in range(len(contents)) for j in range(i))
        ),
    }
    report(EVAL_ID, values, passed=all(value == 0.0 for value in values.values()))
