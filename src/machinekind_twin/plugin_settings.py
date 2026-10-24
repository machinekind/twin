"""Plugin classes from installed packages and the settings the operator gives them.

Strategies (`strategies.py`) and detectors share this code. A package registers a class under an
entry point group, named like the plugin. The class declares what the operator sets for one run
as a dataclass in its `settings` class attribute and takes an instance in its constructor. The
twin never reads the values. It builds the dataclass strictly from raw values, the panel's form,
which the config file's section named like the plugin prefills: an unknown key, a missing one or
a value of another type is an error, as in `robot.yaml`. Then it creates the plugin as
`cls(settings)`, or `cls()` when the class has no `settings`.

Settings fields are `str`, `int`, `float` or `bool`: the panel draws a text field, a number or a
switch for each. String fields may declare dropdown `choices` in dataclass metadata, optionally
keyed by an earlier field named by `choices_from`. A field without a default must be given.

The twin checks a plugin class by attribute only, never by `isinstance` against the contract: a
class without the contract's base class loads when it has the method.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, ClassVar, get_type_hints

from .config import read_config_data

#: Settings field types the panel can draw: a text field, a number, a switch.
FIELD_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class SettingField:
    """One field of a plugin's settings: what the panel's form draws for it."""

    name: str
    #: str, int, float or bool.
    kind: type
    #: `dataclasses.MISSING` = no default: the field must be given.
    default: Any
    #: UI choices only: explicit config values remain accepted by the plugin's own validation.
    choices: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None
    choices_from: str | None = None

    @property
    def required(self) -> bool:
        return self.default is dataclasses.MISSING

    def convert(self, value: Any, where: str) -> Any:
        """`value` as this field's type, or ValueError."""
        return _as_kind(self.kind, value, f"{where}.{self.name}")

    def options(self, values: Mapping[str, Any]) -> tuple[str, ...]:
        """Dropdown choices for the current values of the form."""
        if isinstance(self.choices, dict):
            assert self.choices_from is not None
            selector = values.get(self.choices_from)
            return self.choices.get(selector, ()) if isinstance(selector, str) else ()
        return self.choices or ()


@dataclass(frozen=True)
class PluginClass:
    """A plugin class from an installed package, imported and checked: a strategy or a detector."""

    name: str
    cls: type
    #: The settings dataclass; None = the plugin has no settings.
    settings: type | None
    fields: tuple[SettingField, ...]
    #: The package that registered it and the entry point's value,
    #: e.g. "machinekind-twin-swing 0.1.0 (swing:Swing)".
    source: str
    #: The first line of the class docstring.
    summary: str
    #: The Polish word the panel's messages start with: "Strategia", "Detektor".
    kind: ClassVar[str] = "Wtyczka"

    def check(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Raw values converted to their fields' types. An unknown key or a wrong type fails;
        a missing key does not (`build_settings` does that)."""
        if self.settings is None and raw:
            raise ValueError(f"{self.kind} {self.name} nie ma ustawien, a dostala klucze {sorted(raw)}")

        known = {field.name: field for field in self.fields}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"Nieznane klucze w sekcji {self.name}: {unknown}")

        values = {}
        for name, value in raw.items():
            values[name] = known[name].convert(value, self.name)
        return values

    def build_settings(self, raw: Mapping[str, Any]) -> Any:
        """The settings dataclass from raw values, strictly; None for a plugin without settings."""
        values = self.check(raw)
        if self.settings is None:
            return None

        missing = [field.name for field in self.fields if field.required and field.name not in values]
        if missing:
            raise ValueError(f"Brak kluczy w sekcji {self.name}: {missing}")

        return self.settings(**values)

    def create(self, raw: Mapping[str, Any]) -> Any:
        """The plugin: `cls(settings)` with settings built from raw values, or `cls()` without settings."""
        settings = self.build_settings(raw)
        if self.settings is None:
            return self.cls()

        return self.cls(settings)


def registered(group: str, clash: str) -> dict[str, EntryPoint]:
    """Entry points of `group` by lower-cased name, not loaded yet: loading imports the package.

    Two packages registering one name with different values fail; `clash` starts the error,
    e.g. "Dwie strategie".
    """
    found: dict[str, EntryPoint] = {}

    for ep in entry_points(group=group):
        name = ep.name.lower()
        if name in found and found[name].value != ep.value:
            raise ValueError(f"{clash} o nazwie {name!r}: {found[name].value} i {ep.value}")
        found[name] = ep

    return found


def load_class(ep: EntryPoint, method: str, where: str, contract: str) -> type:
    """`ep` loaded: a class with a callable attribute named like `method` up to "(", or TypeError.

    The check is by attribute only, never `isinstance` against the contract, so a class without
    the contract's base class loads. `method` and `contract` name the requirement in the error,
    e.g. "act(obs)" and "Strategy".
    """
    attribute = method.split("(", 1)[0]

    cls = ep.load()
    if not isinstance(cls, type) or not callable(getattr(cls, attribute, None)):
        raise TypeError(f"{where}: potrzeba klasy z `{method}` - patrz kontrakt {contract}")

    return cls


def config_section(name: str) -> dict[str, Any]:
    """The config file's section named like the plugin, raw; `{}` without a file or a section.

    The file is the one `twin --config` gives (`config.read_config_data`).
    """
    section = read_config_data().get(name.lower())
    if section is None:
        return {}

    if not isinstance(section, Mapping):
        raise ValueError(f"Sekcja konfiguracji {name}: potrzeba mapowania, jest {type(section).__name__}")
    return dict(section)


def _fields(settings: Any, where: str) -> tuple[SettingField, ...]:
    if settings is None:
        return ()

    if not isinstance(settings, type) or not dataclasses.is_dataclass(settings):
        raise TypeError(f"{where}: `settings` musi byc klasa dataclass, jest {settings!r}")

    # Resolves the strings that `from __future__ import annotations` leaves in a package's dataclass.
    hints = get_type_hints(settings)
    found = []

    for field in dataclasses.fields(settings):
        if not field.init:
            continue

        kind = hints[field.name]
        if kind not in FIELD_TYPES:
            raise TypeError(f"{where}: ustawienie {field.name} ma typ {kind} - panel umie tylko str, int, "
                            "float i bool")

        choices, source = _choices(field, kind, found, where)
        found.append(SettingField(field.name, kind, _default(field, kind, where), choices, source))

    return tuple(found)


def _choices(field: dataclasses.Field, kind: type, earlier: list[SettingField], where: str
             ) -> tuple[tuple[str, ...] | dict[str, tuple[str, ...]] | None, str | None]:
    """Check dropdown metadata while loading the plugin, before drawing its form."""
    choices = field.metadata.get("choices")
    source = field.metadata.get("choices_from")
    if choices is None and source is None:
        return None, None
    problem = f"{where}.{field.name}: niepoprawne choices / choices_from"
    if kind is not str:
        raise TypeError(f"{problem} - lista wyboru wymaga str")

    def items(raw: Any) -> tuple[str, ...]:
        if not isinstance(raw, (tuple, list)) or not raw or not all(isinstance(v, str) for v in raw):
            raise TypeError(f"{problem} - choices wymaga niepustej listy napisow")
        return tuple(dict.fromkeys(raw))

    if source is None:
        return items(choices), None
    selector = next((f for f in earlier if f.name == source), None)
    if not isinstance(source, str) or selector is None or not isinstance(selector.choices, tuple):
        raise TypeError(f"{problem} - choices_from wymaga wczesniejszej listy wyboru")
    if not isinstance(choices, Mapping) or set(choices) != set(selector.choices):
        raise TypeError(f"{problem} - choices musi opisywac kazda opcje {source}")
    return {key: items(value) for key, value in choices.items()}, source


def _default(field: dataclasses.Field, kind: type, where: str) -> Any:
    """The field's default as the field's type, or `dataclasses.MISSING`.

    `speed: float = 1` must give 1.0: the panel's number field keeps the type of its first value,
    and an int one would cut what the operator types to whole numbers.
    """
    if field.default is not dataclasses.MISSING:
        value = field.default
    elif field.default_factory is not dataclasses.MISSING:
        value = field.default_factory()
    else:
        return dataclasses.MISSING

    return _as_kind(kind, value, f"{where}.{field.name}")


def _as_kind(kind: type, value: Any, where: str) -> Any:
    """`value` as `kind`, or ValueError. A whole float counts as an int: the panel's number field gives floats."""
    number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if kind is bool and isinstance(value, bool):
        return value
    if kind is str and isinstance(value, str):
        return value
    if kind is int and number and float(value).is_integer():
        return int(value)
    if kind is float and number and math.isfinite(value):
        return float(value)

    raise ValueError(f"{where}: potrzeba {kind.__name__}, jest {value!r}")


def _source(ep: EntryPoint) -> str:
    if ep.dist is None:
        return ep.value

    return f"{ep.dist.name} {ep.dist.version} ({ep.value})"


def _summary(cls: type) -> str:
    lines = (cls.__doc__ or "").strip().splitlines()
    return lines[0] if lines else ""
