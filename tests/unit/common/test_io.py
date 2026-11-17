import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pytest

import neurosim
from neurosim.common.config import config_hash, dump_yaml, field
from neurosim.common.io import git_commit, load_bundle, provenance, save_bundle
from neurosim.common.seeding import SeedTree


@dataclass(frozen=True)
class DemoConfig:
    f0: float = field(2.0e6, gt=0.0, doc="centre frequency [Hz]")
    n_angles: int = field(7, gt=0)


def demo_arrays() -> dict[str, np.ndarray]:
    rng = SeedTree(0).rng("anatomy")
    return {
        "head/sound_speed": rng.normal(1540.0, 30.0, size=(4, 3, 2)).astype(np.float32),
        "head/labels": rng.integers(0, 5, size=(4, 3, 2), dtype=np.uint8),
        "fus/iq/frames": (rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))).astype(
            np.complex64
        ),
        "mask": np.array([True, False, True]),
        "counts": np.arange(6, dtype=np.int64).reshape(2, 3),
        "duration": np.float64(0.25),
    }


DEMO_ATTRS = {
    "name": "demo",
    "root_seed": 12,
    "spacing": 5.0e-4,
    "noise_on": False,
    "config": {"fus": {"f0": 2.0e6, "angles_deg": [-10.0, 0.0, 10.0], "probe": None}, "n": 3},
    "shape": (4, 3, 2),
}


def test_round_trip_of_arrays_and_attributes(tmp_path):
    path = tmp_path / "nested" / "dir" / "bundle.h5"
    arrays = demo_arrays()
    save_bundle(path, arrays, DEMO_ATTRS)
    loaded_arrays, loaded_attrs = load_bundle(path)

    assert sorted(loaded_arrays) == sorted(arrays)
    for name, array in arrays.items():
        assert isinstance(loaded_arrays[name], np.ndarray)
        assert loaded_arrays[name].dtype == array.dtype
        np.testing.assert_array_equal(loaded_arrays[name], array)

    assert loaded_attrs == {**DEMO_ATTRS, "shape": [4, 3, 2]}
    assert type(loaded_attrs["root_seed"]) is int
    assert type(loaded_attrs["spacing"]) is float
    assert type(loaded_attrs["noise_on"]) is bool
    assert type(loaded_attrs["name"]) is str


def test_attrs_are_optional_and_numpy_scalars_are_accepted(tmp_path):
    save_bundle(tmp_path / "bare.h5", {"x": np.zeros(3)})
    assert load_bundle(tmp_path / "bare.h5")[1] == {}

    save_bundle(tmp_path / "numpy.h5", {}, {"n": np.int32(4), "gain": np.float32(0.5)})
    arrays, attrs = load_bundle(tmp_path / "numpy.h5")
    assert arrays == {}
    assert attrs == {"n": 4, "gain": 0.5}


def test_identical_inputs_give_identical_bytes(tmp_path):
    arrays = demo_arrays()
    save_bundle(tmp_path / "a.h5", arrays, DEMO_ATTRS)
    save_bundle(
        tmp_path / "b.h5", dict(reversed(arrays.items())), dict(reversed(DEMO_ATTRS.items()))
    )
    digests = {
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("a.h5", "b.h5")
    }
    assert len(digests) == 1

    arrays["mask"] = np.array([True, True, True])
    save_bundle(tmp_path / "c.h5", arrays, DEMO_ATTRS)
    assert hashlib.sha256((tmp_path / "c.h5").read_bytes()).hexdigest() not in digests


def test_unsupported_inputs_are_rejected(tmp_path):
    path = tmp_path / "bad.h5"
    with pytest.raises(TypeError, match="array 'names': unsupported dtype"):
        save_bundle(path, {"names": np.array(["a", "b"])})
    with pytest.raises(TypeError, match="attribute 'when': unsupported type"):
        save_bundle(path, {}, {"when": object()})
    with pytest.raises(TypeError, match=r"attribute 'config.grid\[1\]': unsupported type"):
        save_bundle(path, {}, {"config": {"grid": [1, {2}]}})
    with pytest.raises(TypeError, match="attribute 'probe': None"):
        save_bundle(path, {}, {"probe": None})
    with pytest.raises(ValueError, match="reserved"):
        save_bundle(path, {}, {"__json_attrs__": "x"})
    assert not path.exists()


def test_git_commit_is_a_hash_or_unknown():
    commit = git_commit()
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_provenance_fields_and_storage(tmp_path):
    config = DemoConfig(n_angles=9)
    record = provenance(config, root_seed=31)
    assert record == {
        "config_yaml": dump_yaml(config),
        "config_hash": config_hash(config),
        "root_seed": 31,
        "git_commit": git_commit(),
        "neurosim_version": neurosim.__version__,
    }
    save_bundle(tmp_path / "run.h5", {"x": np.ones(2)}, record)
    assert load_bundle(tmp_path / "run.h5")[1] == record
