"""Strategies from installed packages, and the settings the operator gives them.

A package registers a strategy class under the `lerobot_mp.strategies` entry point group
(`lerobot_mp_api.entry_points.STRATEGIES`), named like the strategy:

    [project.entry-points."lerobot_mp.strategies"]
    pi05 = "vla_strategies.pi05:Pi05"

The class subclasses `strategy.Strategy`, which the package imports from `lerobot_mp_api`. What the
operator sets for one run (a VLA's instruction, a policy file, task options) stays out of the
`Observation`, which holds only what the twin measures. The strategy declares it as a dataclass in
its `settings` class attribute and takes an instance in its constructor:

    @dataclass
    class VlaSettings:
        instruction: str
        server: str = "localhost:8000"

    class Pi05(Strategy[VlaSettings]):
        hz = 50
        settings = VlaSettings

        def __init__(self, settings: VlaSettings): ...
        def act(self, obs): ...

The package decides which settings exist, their types and their defaults, with nothing but
`dataclasses` from the standard library. The twin never reads the values. It builds the dataclass
strictly from raw values, the panel's form, which the config file's section named like the
strategy (`pi05:`) prefills: an unknown key, a missing one or a value of another type is an error,
as in `robot.yaml`. Then it creates the strategy as `cls(settings)`, or `cls()` when the class has
no `settings`.

Settings fields are `str`, `int`, `float` or `bool`: the panel draws a text field, a number or a
switch for each. A field without a default must be given.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, get_type_hints

from lerobot_mp_api.entry_points import STRATEGIES

from .config import read_config_data
from .strategy import Strategy

ENTRY_POINT_GROUP = STRATEGIES

#: Settings field types the panel can draw: a text field, a number, a switch.
FIELD_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class SettingField:
    """One field of a strategy's settings: what the panel's form draws for it."""

    name: str
    #: str, int, float or bool.
    kind: type
    #: `dataclasses.MISSING` = no default: the field must be given.
    default: Any

    @property
    def required(self) -> bool:
        return self.default is dataclasses.MISSING

    def convert(self, value: Any, where: str) -> Any:
        """`value` as this field's type, or ValueError."""
        return _as_kind(self.kind, value, f"{where}.{self.name}")


@dataclass(frozen=True)
class StrategyPlugin:
    """A strategy class from an installed package, imported and checked."""

    name: str
    cls: type
    #: The settings dataclass; None = the strategy has no settings.
    settings: type | None
    fields: tuple[SettingField, ...]
    #: The package that registered it and the entry point's value, e.g. "lerobot-mp-swing 0.1.0 (swing:Swing)".
    source: str
    #: The first line of the class docstring.
    summary: str

    def check(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Raw values converted to their fields' types. An unknown key or a wrong type fails;
        a missing key does not (`build_settings` does that)."""
        if self.settings is None and raw:
            raise ValueError(f"Strategia {self.name} nie ma ustawien, a dostala klucze {sorted(raw)}")

        known = {field.name: field for field in self.fields}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"Nieznane klucze w sekcji {self.name}: {unknown}")

        values = {}
        for name, value in raw.items():
            values[name] = known[name].convert(value, self.name)
        return values

    def build_settings(self, raw: Mapping[str, Any]) -> Any:
        """The settings dataclass from raw values, strictly; None for a strategy without settings."""
        values = self.check(raw)
        if self.settings is None:
            return None

        missing = [field.name for field in self.fields if field.required and field.name not in values]
        if missing:
            raise ValueError(f"Brak kluczy w sekcji {self.name}: {missing}")

        return self.settings(**values)

    def create(self, raw: Mapping[str, Any]) -> Strategy[Any]:
        """The strategy: `cls(settings)` with settings built from raw values, or `cls()` without settings."""
        settings = self.build_settings(raw)
        if self.settings is None:
            return self.cls()

        return self.cls(settings)


def strategies() -> dict[str, EntryPoint]:
    """Registered strategies by name, not loaded yet: loading imports the strategy's package."""
    found: dict[str, EntryPoint] = {}

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        name = ep.name.lower()
        if name in found and found[name].value != ep.value:
            raise ValueError(f"Dwie strategie o nazwie {name!r}: {found[name].value} i {ep.value}")
        found[name] = ep

    return found


def load_strategy(name: str) -> StrategyPlugin:
    """The registered strategy `name`, imported: a class with `act`, and settings the panel can draw."""
    ep = _entry_point(name)
    where = f"strategia {ep.name} ({ep.value})"

    cls = ep.load()
    if not isinstance(cls, type) or not callable(getattr(cls, "act", None)):
        raise TypeError(f"{where}: potrzeba klasy z `act(obs)` - patrz kontrakt Strategy")

    settings = getattr(cls, "settings", None)

    return StrategyPlugin(
        name=ep.name.lower(),
        cls=cls,
        settings=settings,
        fields=_fields(settings, where),
        source=_source(ep),
        summary=_summary(cls),
    )


def config_section(name: str) -> dict[str, Any]:
    """The config file's section named like the strategy, raw; `{}` without a file or a section.

    The file is the one `lerobot-twin --config` gives (`config.read_config_data`).
    """
    section = read_config_data().get(name.lower())
    if section is None:
        return {}

    if not isinstance(section, Mapping):
        raise ValueError(f"Sekcja konfiguracji {name}: potrzeba mapowania, jest {type(section).__name__}")
    return dict(section)


def _entry_point(name: str) -> EntryPoint:
    registered = strategies()

    try:
        return registered[name.lower()]
    except KeyError as exc:
        # Entry points live in the installed package's metadata: a package added to the
        # environment without `pip install` registers nothing.
        known = ", ".join(sorted(registered)) or "zadne - zainstaluj pakiet ze strategia (pip install)"
        raise KeyError(f"Nieznana strategia {name!r}. Zainstalowane: {known}") from exc


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

        found.append(SettingField(field.name, kind, _default(field, kind, where)))

    return tuple(found)


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
