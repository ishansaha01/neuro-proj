"""Typed, validated configuration built on frozen dataclasses.

A config class is a frozen dataclass whose fields are declared with `field`,
which records value constraints in the field metadata. `from_dict` builds the
class from plain data (for example a loaded YAML file) and rejects anything
that does not match the schema, naming the offending key by its dotted path.

Supported annotations: ``bool``, ``int``, ``float``, ``str``, ``Literal[...]``,
``Optional[X]`` / ``X | None``, ``tuple[X, ...]``, ``tuple[X, Y, Z]``,
``list[X]`` and nested dataclasses. Sequences are always stored as tuples so
that config objects stay hashable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import operator
import os
import re
import types
import typing
from collections.abc import Mapping
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_BOUND_CHECKS = (
    ("ge", operator.ge, ">="),
    ("gt", operator.gt, ">"),
    ("le", operator.le, "<="),
    ("lt", operator.lt, "<"),
)


class ConfigError(ValueError):
    """A config value does not match its schema. The message starts with the key path."""


def field(
    default: Any = ...,
    *,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
    choices: typing.Iterable[Any] | None = None,
    doc: str = "",
    default_factory: Any = ...,
) -> dataclasses.Field:
    """Declare a config field with constraints checked by `from_dict`.

    Parameters
    ----------
    default, default_factory:
        As in ``dataclasses.field``. Leave both unset (``...``) for a required key.
    ge, gt, le, lt:
        Inclusive / exclusive lower and upper bounds, in the unit of the field.
        For a sequence field the bounds apply to every element.
    choices:
        Allowed values (every element, for a sequence field).
    doc:
        One-line description of the field, including its unit.
    """
    metadata: dict[str, Any] = {"doc": doc}
    for name, bound in (("ge", ge), ("gt", gt), ("le", le), ("lt", lt)):
        if bound is not None:
            metadata[name] = bound
    if choices is not None:
        metadata["choices"] = tuple(choices)
    return dataclasses.field(
        default=dataclasses.MISSING if default is ... else default,
        default_factory=dataclasses.MISSING if default_factory is ... else default_factory,
        metadata=metadata,
    )


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-3`` and ``2.5e6`` as floats.

    PyYAML follows YAML 1.1, where a float needs a dot and a signed exponent,
    so it would load those two as strings.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*\.?[0-9_]*|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def load_yaml(path: str | os.PathLike[str]) -> dict:
    """Load a YAML config file whose top level is a mapping."""
    with open(path, encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=_Loader)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _join(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def _describe(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _convert_sequence(tp: Any, value: Any, path: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected a list, got {_describe(value)}")
    args = typing.get_args(tp)
    if typing.get_origin(tp) is list or (len(args) == 2 and args[1] is ...):
        item_types = (args[0],) * len(value)
    elif len(value) == len(args):
        item_types = args
    else:
        raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
    return tuple(
        _convert(item_type, item, f"{path}[{i}]")
        for i, (item_type, item) in enumerate(zip(item_types, value))
    )


def _convert(tp: Any, value: Any, path: str) -> Any:
    """Check ``value`` against annotation ``tp`` and return it in canonical form."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        inner = [arg for arg in args if arg is not type(None)]
        if len(args) != 2 or len(inner) != 1:
            raise TypeError(f"{path}: unsupported annotation {tp!r}; only Optional[X] unions")
        return None if value is None else _convert(inner[0], value, path)
    if origin is typing.Literal:
        options = typing.get_args(tp)
        # Compare types too, so that True does not pass for the literal 1.
        if any(type(option) is type(value) and option == value for option in options):
            return value
        raise ConfigError(f"{path}: expected one of {list(options)}, got {_describe(value)}")
    if origin in (tuple, list):
        return _convert_sequence(tp, value, path)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    else:
        raise TypeError(f"{path}: unsupported annotation {tp!r}")
    raise ConfigError(f"{path}: expected {tp.__name__}, got {_describe(value)}")


def _check_constraints(metadata: Mapping[str, Any], value: Any, path: str) -> None:
    if value is None:
        return
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            _check_constraints(metadata, item, f"{path}[{i}]")
        return
    for name, satisfied, symbol in _BOUND_CHECKS:
        # Written as "not satisfied" so that NaN fails every bound.
        if name in metadata and not satisfied(value, metadata[name]):
            raise ConfigError(f"{path}: must be {symbol} {metadata[name]!r}, got {value!r}")
    if "choices" in metadata and value not in metadata["choices"]:
        raise ConfigError(
            f"{path}: must be one of {list(metadata['choices'])}, got {value!r}"
        )


def _build(cls: type[T], data: Any, path: str) -> T:
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path or '<root>'}: expected a mapping for {cls.__name__}, got {_describe(data)}"
        )
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    for key in data:
        if key not in fields:
            raise ConfigError(f"{_join(path, key)}: unknown key; allowed keys: {sorted(fields)}")
    kwargs = {}
    for name, spec in fields.items():
        key_path = _join(path, name)
        if name in data:
            value = _convert(hints[name], data[name], key_path)
            _check_constraints(spec.metadata, value, key_path)
            kwargs[name] = value
        elif spec.default is dataclasses.MISSING and spec.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{key_path}: missing required key")
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except ValueError as error:
        # A cross-field check in the class's __post_init__; add where it happened.
        raise ConfigError(f"{path or '<root>'}: {error}") from error


def from_dict(cls: type[T], data: Mapping) -> T:
    """Build the dataclass ``cls`` from plain data, strictly.

    Raises `ConfigError` naming the full dotted key path (``fus.transducer.pitch``,
    ``angles[2]``) for an unknown key, a missing required key, a value of the
    wrong type (a bool is not an int; an int is accepted for a float) or a
    violated constraint. Keys that have a default may be omitted. Lists become
    tuples.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"from_dict needs a dataclass type, got {cls!r}")
    return _build(cls, data, "")


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_dict(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def to_dict(obj: Any) -> dict:
    """Convert a config dataclass instance to plain data (dicts, lists, scalars).

    Inverse of `from_dict`: ``from_dict(type(obj), to_dict(obj)) == obj``.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"to_dict needs a dataclass instance, got {type(obj).__name__}")
    return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}


def dump_yaml(obj: Any) -> str:
    """Return the canonical YAML text of a config: sorted keys, block style.

    The text loads back to the same config through `load_yaml` and `from_dict`.
    """
    return yaml.safe_dump(
        to_dict(obj), sort_keys=True, default_flow_style=False, allow_unicode=True, indent=2
    )


def config_hash(obj: Any) -> str:
    """Return the SHA-256 hex digest of ``dump_yaml(obj)``."""
    return hashlib.sha256(dump_yaml(obj).encode("utf-8")).hexdigest()
