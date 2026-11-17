import dataclasses
import time
import tracemalloc

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.seeding import SeedTree
from neurosim.head.anatomy import draw_geometry
from neurosim.head.config import HeadConfig
from neurosim.head.generate import (
    GeneratedHead,
    activation_center,
    estimate_memory_bytes,
    estimate_runtime_seconds,
    generate_head,
    head_grid,
)
from neurosim.head.materials import (
    acoustic_properties,
    brain_mask,
    material_fractions,
    skull_mask,
)
from neurosim.head.types import SURFACES, HeadVolume, Tissue, labels_from_sdf

ARRAY_FIELDS = [
    f.name for f in dataclasses.fields(HeadVolume) if f.name not in ("grid", "alpha_power")
]


def test_fields_have_the_contract_shapes_and_dtypes(small_head, small_cfg):
    shape = small_head.grid.shape
    assert isinstance(small_head, HeadVolume)
    assert shape == (32, 32, 32)
    assert small_head.grid.spacing == small_cfg.grid.spacing
    assert small_head.sdf.shape == (len(SURFACES), *shape)
    assert small_head.material_fractions.shape == (len(BASIS_MATERIALS), *shape)
    assert small_head.blood_velocity.shape == (3, *shape)
    assert small_head.labels.dtype == np.uint8
    for name in ARRAY_FIELDS:
        values = getattr(small_head, name)
        assert np.all(np.isfinite(values)), name
        if name != "labels":
            assert values.dtype == np.float32, name
            assert values.shape[-3:] == shape, name
    assert small_head.alpha_power == small_cfg.materials.alpha_power
    assert small_head.materials == small_cfg.materials


def test_every_tissue_is_present_in_the_small_box(small_head):
    assert set(np.unique(small_head.labels)) == set(Tissue)


def test_derived_fields_follow_from_the_latent_fields(small_head, small_cfg):
    np.testing.assert_array_equal(small_head.labels, labels_from_sdf(small_head.sdf))
    maps = acoustic_properties(
        small_head.labels, small_head.porosity, small_head.hidden_c, small_cfg.materials
    )
    for derived, name in zip(maps, ("sound_speed", "density", "attenuation")):
        np.testing.assert_array_equal(getattr(small_head, name), derived)
    np.testing.assert_array_equal(
        small_head.material_fractions, material_fractions(small_head.labels, small_head.porosity)
    )


def test_latent_and_cbv_fields_are_confined_to_their_tissues(small_head, small_cfg):
    in_skull = skull_mask(small_head.labels)
    in_brain = brain_mask(small_head.labels)
    assert not small_head.porosity[~in_skull].any()
    assert not small_head.hidden_c[~in_skull].any()
    assert small_head.porosity.min() >= 0.0 and small_head.porosity.max() <= 1.0
    assert not small_head.cbv_baseline[~in_brain].any()
    assert not small_head.blood_velocity[:, ~in_brain].any()
    assert small_head.cbv_baseline[in_brain].min() > 0.0
    assert small_head.cbv_baseline.max() <= 1.0
    assert not small_head.activation[small_head.labels != Tissue.GREY].any()
    assert 0.5 < small_head.activation.max() <= 1.0
    speed = np.linalg.norm(small_head.blood_velocity, axis=0)[in_brain]
    assert speed.max() <= 2.0 * small_cfg.cbv.speed_vessel
    assert np.median(speed) == pytest.approx(small_cfg.cbv.speed_tissue, rel=0.5)


def test_activation_peaks_at_the_activation_centre(small_head, small_cfg):
    geometry = draw_geometry(small_cfg.anatomy, SeedTree(3).rng("anatomy"))
    center = activation_center(small_cfg, geometry)
    peak = np.unravel_index(np.argmax(small_head.activation), small_head.grid.shape)
    peak_position = np.array([axis[i] for axis, i in zip(small_head.grid.axes(), peak)])
    assert np.linalg.norm(peak_position - center) <= 1.5 * small_head.grid.spacing


def test_seeds_give_different_heads(small_head, small_cfg):
    other = generate_head(small_cfg, SeedTree(4))
    assert np.max(np.abs(other.sdf - small_head.sdf)) > 1.0e-4 or other.grid != small_head.grid
    assert not np.array_equal(other.porosity, small_head.porosity)


def test_small_config_is_fast(small_cfg):
    start = time.perf_counter()
    generate_head(small_cfg, SeedTree(0))
    assert time.perf_counter() - start < 3.0


def test_full_region_holds_the_whole_scalp_for_any_seed():
    cfg = HeadConfig(grid=dataclasses.replace(HeadConfig().grid, region="full", spacing=6.0e-3))
    grids = set()
    for seed in (0, 1):
        head = generate_head(cfg, SeedTree(seed))
        grids.add(head.grid)
        outer_faces = [head.labels[0], head.labels[-1], head.labels[:, 0], head.labels[:, -1]]
        outer_faces += [head.labels[:, :, 0], head.labels[:, :, -1]]
        assert all(np.all(face == Tissue.BACKGROUND) for face in outer_faces)
        assert head.labels[tuple(n // 2 for n in head.grid.shape)] == Tissue.WHITE
    assert len(grids) == 1


def test_box_is_centred_below_the_scalp_along_the_target(small_cfg):
    geometry = draw_geometry(small_cfg.anatomy, SeedTree(3).rng("anatomy"))
    grid = head_grid(small_cfg, geometry)
    center = np.mean([[axis[0], axis[-1]] for axis in grid.axes()], axis=1)
    target = np.asarray(small_cfg.cbv.target_direction)
    target = target / np.linalg.norm(target)
    assert np.linalg.norm(np.cross(center, target)) < 1.0e-9
    cortex = activation_center(small_cfg, geometry)
    assert 0.0 < np.linalg.norm(cortex) - np.linalg.norm(center) < small_cfg.grid.box_center_depth


def test_memory_estimate_bounds_the_measured_peak():
    cfg = HeadConfig(grid=dataclasses.replace(HeadConfig().grid, spacing=1.0e-3))
    tracemalloc.start()
    head = generate_head(cfg, SeedTree(0))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    stored = sum(getattr(head, name).nbytes for name in ARRAY_FIELDS)
    estimate = estimate_memory_bytes(cfg)
    assert stored == 85 * head.grid.n_voxels
    assert stored < peak <= estimate
    assert estimate < 2.0 * peak


def test_estimates_scale_with_the_voxel_count():
    coarse = HeadConfig(grid=dataclasses.replace(HeadConfig().grid, spacing=1.0e-3))
    fine = HeadConfig()
    assert estimate_memory_bytes(fine) == 8 * estimate_memory_bytes(coarse)
    assert estimate_runtime_seconds(fine) == pytest.approx(8 * estimate_runtime_seconds(coarse))
    full = HeadConfig(grid=dataclasses.replace(HeadConfig().grid, region="full"))
    assert estimate_memory_bytes(full) > 2**32


def test_generated_head_is_a_head_volume(small_head):
    assert isinstance(small_head, GeneratedHead)
    plain = HeadVolume(
        **{f.name: getattr(small_head, f.name) for f in dataclasses.fields(HeadVolume)}
    )
    assert plain.grid == small_head.grid
