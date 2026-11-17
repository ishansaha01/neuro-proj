import dataclasses

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.head.config import MicrostructureConfig
from neurosim.head.materials import SkullMixing, skull_mask, skull_sound_speed_from_porosity
from neurosim.head.microstructure import explained_variance, hidden_sound_speed, skull_porosity
from neurosim.head.types import SURFACES, Tissue, labels_from_sdf

CFG = MicrostructureConfig()
SKULL = SkullMixing()


def slab_sdf(n: int = 400) -> np.ndarray:
    """SDFs of flat layers: 5 mm scalp, 1.5 / 4 / 1 mm skull, 3 mm CSF, 2.5 mm grey matter."""
    depth = np.linspace(-2.0e-3, 22.0e-3, n, dtype=np.float32)
    interfaces = np.cumsum([0.0, 5.0e-3, 1.5e-3, 4.0e-3, 1.0e-3, 3.0e-3, 2.5e-3])
    sdf = (interfaces[:, None] - depth[None, :]).astype(np.float32)
    assert sdf.shape[0] == len(SURFACES)
    return np.broadcast_to(sdf[:, :, None], (len(SURFACES), n, 50)).copy()


def test_porosity_profile_is_low_high_low_and_zero_outside_the_skull():
    sdf = slab_sdf()
    labels = labels_from_sdf(sdf)
    no_texture = dataclasses.replace(CFG, texture_variation=0.0)
    porosity = skull_porosity(sdf, labels, np.zeros(labels.shape, dtype=np.float32), no_texture)
    assert porosity.dtype == np.float32
    assert np.all(porosity[~skull_mask(labels)] == 0.0)
    outer = np.median(porosity[labels == Tissue.SKULL_OUTER])
    diploe = np.median(porosity[labels == Tissue.DIPLOE])
    inner = np.median(porosity[labels == Tissue.SKULL_INNER])
    assert outer == pytest.approx(CFG.porosity_outer_table, abs=0.01)
    assert diploe == pytest.approx(CFG.porosity_diploe, abs=0.01)
    assert inner == pytest.approx(CFG.porosity_inner_table, abs=0.02)


def test_porosity_texture_stays_in_range():
    sdf = slab_sdf()
    labels = labels_from_sdf(sdf)
    texture = SeedTree(0).rng("skull_microstructure").standard_normal(labels.shape) * 4.0
    porosity = skull_porosity(sdf, labels, texture.astype(np.float32), CFG)
    assert porosity.min() >= 0.0
    assert porosity.max() <= 1.0
    assert porosity[labels == Tissue.DIPLOE].std() > 0.05


def hidden_inputs():
    sdf = slab_sdf()
    labels = labels_from_sdf(sdf)
    rng = SeedTree(0).rng("skull_microstructure")
    porosity = skull_porosity(
        sdf, labels, rng.standard_normal(labels.shape).astype(np.float32), CFG
    )
    texture = SeedTree(0).rng("hidden_soundspeed").standard_normal(labels.shape)
    return labels, porosity, texture.astype(np.float32)


@pytest.mark.parametrize("r2", [0.3, 0.5, 0.8])
def test_hidden_sound_speed_meets_the_r2_target_within_bounds(r2):
    labels, porosity, texture = hidden_inputs()
    cfg = dataclasses.replace(CFG, porosity_r2=r2)
    hidden = hidden_sound_speed(porosity, labels, texture, cfg, SKULL)
    in_skull = skull_mask(labels)
    assert hidden.dtype == np.float32
    assert np.all(hidden[~in_skull] == 0.0)
    total = skull_sound_speed_from_porosity(porosity[in_skull], SKULL) + hidden[in_skull]
    assert explained_variance(total, porosity[in_skull]) == pytest.approx(r2, abs=1e-4)
    assert total.min() >= SKULL.c_fluid - 0.01
    assert total.max() <= cfg.sound_speed_max + 0.01
    assert abs(hidden[in_skull].mean()) < 0.05 * hidden[in_skull].std()


def test_r2_of_one_means_no_hidden_component():
    labels, porosity, texture = hidden_inputs()
    cfg = dataclasses.replace(CFG, porosity_r2=1.0)
    assert not hidden_sound_speed(porosity, labels, texture, cfg, SKULL).any()


def test_no_skull_means_no_hidden_component():
    labels = np.full((4, 4), Tissue.WHITE, dtype=np.uint8)
    zeros = np.zeros(labels.shape, dtype=np.float32)
    assert not hidden_sound_speed(zeros, labels, zeros, CFG, SKULL).any()


def test_unreachable_r2_is_an_error():
    labels, porosity, _ = hidden_inputs()
    cfg = dataclasses.replace(CFG, porosity_r2=0.05)
    # A hidden pattern collinear with porosity cannot lower the explained variance.
    with pytest.raises(ValueError, match="cannot be reached"):
        hidden_sound_speed(porosity, labels, -porosity, cfg, SKULL)
