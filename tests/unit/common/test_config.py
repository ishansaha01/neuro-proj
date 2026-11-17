from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Literal, Optional

import pytest

from neurosim.common.config import (
    ConfigError,
    config_hash,
    dump_yaml,
    field,
    from_dict,
    load_yaml,
    to_dict,
)


@dataclass(frozen=True)
class Transducer:
    n_elements: int = field(gt=0, doc="number of elements")
    pitch: float = field(gt=0.0, le=1.0e-2, doc="element pitch [m]")
    apodization: Literal["none", "hann"] = field("none")


@dataclass(frozen=True)
class Run:
    name: str = field(doc="run name")
    transducer: Transducer = field()
    angles_deg: tuple[float, ...] = field((0.0,), ge=-45.0, le=45.0, doc="steering angles [deg]")
    extent: tuple[float, float, float] = field((0.1, 0.1, 0.1), gt=0.0, doc="box size [m]")
    kvp: Optional[float] = field(None, gt=0.0, lt=200.0, doc="tube voltage [kV]")
    depth: float | None = field(None, doc="imaging depth [m]")
    noise_on: bool = field(True)
    mode: str = field("fast", choices=("fast", "full"))
    frames: list[int] = field(default_factory=tuple, ge=0)


@dataclass(frozen=True)
class Band:
    f_min: float = field(gt=0.0, doc="lower band edge [Hz]")
    f_max: float = field(gt=0.0, doc="upper band edge [Hz]")

    def __post_init__(self):
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be above f_min")


@dataclass(frozen=True)
class Sensor:
    band: Band = field()


@dataclass(frozen=True)
class Unsupported:
    table: dict = field(default_factory=dict)


FULL = {
    "name": "demo",
    "transducer": {"n_elements": 128, "pitch": 3.0e-4, "apodization": "hann"},
    "angles_deg": [-10.0, 0.0, 10.0],
    "extent": [0.2, 0.22, 0.18],
    "kvp": 120.0,
    "depth": None,
    "noise_on": False,
    "mode": "full",
    "frames": [0, 5, 9],
}


def changed(path: tuple, value) -> dict:
    """Return a copy of FULL with the entry at ``path`` replaced (or added)."""
    data = copy.deepcopy(FULL)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


def removed(path: tuple) -> dict:
    data = copy.deepcopy(FULL)
    target = data
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return data


def test_round_trip_through_dict():
    run = from_dict(Run, FULL)
    assert to_dict(run) == FULL
    assert from_dict(Run, to_dict(run)) == run


def test_values_are_typed_and_sequences_become_tuples():
    run = from_dict(Run, FULL)
    assert run.transducer == Transducer(n_elements=128, pitch=3.0e-4, apodization="hann")
    assert run.angles_deg == (-10.0, 0.0, 10.0)
    assert run.extent == (0.2, 0.22, 0.18)
    assert run.frames == (0, 5, 9)
    assert hash(run) == hash(from_dict(Run, FULL))
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.name = "other"


def test_defaults_apply_to_omitted_keys():
    run = from_dict(Run, {"name": "minimal", "transducer": {"n_elements": 4, "pitch": 1e-3}})
    assert run.angles_deg == (0.0,)
    assert run.kvp is None
    assert run.noise_on is True
    assert run.transducer.apodization == "none"
    assert run.frames == ()


def test_int_is_accepted_for_float_and_stored_as_float():
    run = from_dict(Run, changed(("angles_deg",), [-10, 0, 10]))
    assert run.angles_deg == (-10.0, 0.0, 10.0)
    assert all(type(angle) is float for angle in run.angles_deg)


def test_constraints_are_stored_in_field_metadata():
    metadata = {f.name: f.metadata for f in dataclasses.fields(Transducer)}
    assert metadata["pitch"] == {"doc": "element pitch [m]", "gt": 0.0, "le": 1.0e-2}
    assert dataclasses.fields(Run)[7].metadata["choices"] == ("fast", "full")


@pytest.mark.parametrize(
    ("data", "key_path", "reason"),
    [
        (changed(("bogus",), 1), "bogus", "unknown key"),
        (changed(("transducer", "bogus"), 1), "transducer.bogus", "unknown key"),
        (removed(("name",)), "name", "missing required key"),
        (removed(("transducer", "pitch")), "transducer.pitch", "missing required key"),
        (changed(("transducer", "n_elements"), True), "transducer.n_elements", "expected int"),
        (changed(("transducer", "n_elements"), 128.0), "transducer.n_elements", "expected int"),
        (changed(("transducer", "pitch"), "3e-4 m"), "transducer.pitch", "expected float"),
        (changed(("transducer", "pitch"), False), "transducer.pitch", "expected float"),
        (changed(("name",), 7), "name", "expected str"),
        (changed(("noise_on",), 1), "noise_on", "expected bool"),
        (changed(("transducer",), [1, 2]), "transducer", "expected a mapping"),
        (changed(("angles_deg",), 5.0), "angles_deg", "expected a list"),
        (changed(("angles_deg",), [0.0, "ten"]), "angles_deg[1]", "expected float"),
        (changed(("extent",), [0.2, 0.2]), "extent", "expected 3 items"),
        (changed(("kvp",), "high"), "kvp", "expected float"),
        (changed(("transducer", "apodization"), "tukey"), "transducer.apodization", "one of"),
        (changed(("transducer", "n_elements"), 0), "transducer.n_elements", "must be > 0"),
        (changed(("transducer", "pitch"), 0.02), "transducer.pitch", "must be <= 0.01"),
        (changed(("angles_deg",), [0.0, 10.0, -50.0]), "angles_deg[2]", "must be >= -45.0"),
        (changed(("kvp",), 200.0), "kvp", "must be < 200.0"),
        (changed(("kvp",), math.nan), "kvp", "must be > 0.0"),
        (changed(("mode",), "slow"), "mode", "must be one of"),
        (changed(("frames",), [0, -1]), "frames[1]", "must be >= 0"),
    ],
)
def test_invalid_config_raises_with_key_path(data, key_path, reason):
    with pytest.raises(ConfigError) as error:
        from_dict(Run, data)
    message = str(error.value)
    assert message.startswith(f"{key_path}: ")
    assert reason in message


def test_root_must_be_a_mapping():
    with pytest.raises(ConfigError, match="<root>: expected a mapping"):
        from_dict(Run, [1, 2, 3])


def test_post_init_value_error_gets_the_key_path():
    assert from_dict(Sensor, {"band": {"f_min": 1e6, "f_max": 3e6}}).band.f_max == 3e6
    with pytest.raises(ConfigError, match=r"^band: f_max must be above f_min$"):
        from_dict(Sensor, {"band": {"f_min": 3e6, "f_max": 1e6}})


def test_schema_mistakes_are_type_errors_not_config_errors():
    with pytest.raises(TypeError, match="unsupported annotation"):
        from_dict(Unsupported, {"table": {}})
    with pytest.raises(TypeError, match="dataclass type"):
        from_dict(dict, {})
    with pytest.raises(TypeError, match="dataclass instance"):
        to_dict({"a": 1})


def test_load_yaml_reads_exponent_floats_and_round_trips(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: demo\n"
        "transducer: {n_elements: 128, pitch: 3e-4}\n"
        "angles_deg: [-1e1, 0, 1.0e+1]\n"
        "kvp: 1.2e2\n",
        encoding="utf-8",
    )
    data = load_yaml(path)
    assert data["transducer"]["pitch"] == 3.0e-4
    assert data["angles_deg"] == [-10.0, 0, 10.0]
    run = from_dict(Run, data)
    assert run.kvp == 120.0

    path.write_text(dump_yaml(run), encoding="utf-8")
    assert from_dict(Run, load_yaml(path)) == run


def test_load_yaml_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_yaml(path)


def test_dump_yaml_is_canonical():
    run = from_dict(Run, FULL)
    same_run = from_dict(Run, dict(reversed(FULL.items())))
    text = dump_yaml(run)
    assert text == dump_yaml(same_run)
    top_level_keys = [line.split(":")[0] for line in text.splitlines() if line[0].isalpha()]
    assert top_level_keys == sorted(FULL)


def test_config_hash_identifies_the_config():
    run = from_dict(Run, FULL)
    digest = config_hash(run)
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert digest == config_hash(from_dict(Run, FULL))
    assert digest != config_hash(from_dict(Run, changed(("kvp",), 100.0)))
