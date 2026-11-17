"""E-DATA-002: the CT-derived skull lies where the true skull is, within one CT voxel."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.ct.config import HuMappingConfig

EVAL_ID = "E-DATA-002"
# Bone threshold of the CT-derived medium [HU]: the evals scan with configs/ct_default.yaml,
# which keeps the default of the HU mapping.
BONE_THRESHOLD_HU = HuMappingConfig().bone_threshold_hu
# One voxel of the CT slab at the default settings [m].
MAX_INTERFACE_ERROR = 0.5e-3


def _crossings(depth: np.ndarray, profile: np.ndarray, level: float) -> np.ndarray:
    """Depths at which a sampled profile crosses a level, by linear interpolation [m]."""
    above = profile >= level
    before = np.flatnonzero(above[:-1] != above[1:])
    fraction = (level - profile[before]) / (profile[before + 1] - profile[before])
    return depth[before] + fraction * (depth[before + 1] - depth[before])


@eval_case(
    id=EVAL_ID,
    claim="The CT-derived skull is geometrically registered to the truth: along the "
    "central depth line (x = 0) the outer and inner skull interfaces of the CT-derived "
    "medium are at the depths of the true interfaces",
    reference="The true outer and inner skull surfaces of the head: zero crossings of "
    "their signed-distance fields along the same line, which do not pass through the CT "
    "or the slab resampling",
    metric="Largest |z_CT - z_true| over seeds and both interfaces [m]; z_CT is where "
    "the CT image first rises above (outer) and last falls below (inner) the bone "
    "threshold of ct.acoustics, i.e. the edge of the medium's bone label",
    threshold="<= 0.5e-3 m, one voxel of the CT slab: a CT cannot localise an edge "
    "better than its sampling, while a flip, transpose or offset of the slab geometry "
    "would be off by millimetres. Fixed before the first run",
    seeds=(0, 1, 2),
)
def test_e_data_002_ct_skull_registration(plane_prior):
    values = {}
    errors = []
    for seed in REGISTRY[EVAL_ID].seeds:
        prior = plane_prior(seed)
        depth, lateral = prior.truth.grid.axes()
        center_column = int(np.argmin(np.abs(lateral)))
        bone_edges = _crossings(depth, prior.hu_plane[:, center_column], BONE_THRESHOLD_HU)
        true_outer, true_inner = (
            _crossings(depth, sdf, 0.0)[0] for sdf in prior.skull_sdf_center_line
        )
        # No bone on the line at all is a failure, reported as NaN.
        ct_outer, ct_inner = (bone_edges[0], bone_edges[-1]) if bone_edges.size else (np.nan,) * 2
        outer_error = ct_outer - true_outer
        inner_error = ct_inner - true_inner
        errors += [abs(outer_error), abs(inner_error)]
        values[f"outer_error_seed_{seed}"] = outer_error
        values[f"inner_error_seed_{seed}"] = inner_error
        # Diagnostics (not part of the verdict): a registration error moves both
        # interfaces the same way (centre), edge blur moves them apart (thickness).
        values[f"center_error_seed_{seed}"] = 0.5 * (outer_error + inner_error)
        values[f"thickness_error_seed_{seed}"] = inner_error - outer_error
    values["max_abs_error"] = float(np.max(errors))
    report(EVAL_ID, values, passed=bool(np.all(np.asarray(errors) <= MAX_INTERFACE_ERROR)))
