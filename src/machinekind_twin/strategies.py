"""Strategies from installed packages, and the settings the operator gives them.

A package registers a strategy class under the `machinekind_twin.strategies` entry point group
(`machinekind_twin_api.entry_points.STRATEGIES`), named like the strategy:

    [project.entry-points."machinekind_twin.strategies"]
    pi05 = "vla_strategies.pi05:Pi05"

The class subclasses `strategy.Strategy`, which the package imports from `machinekind_twin_api`. What the
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

The settings code is shared with detectors and lives in `plugin_settings`; this module keeps its
public names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

from machinekind_twin_api.entry_points import STRATEGIES

from . import plugin_settings
from .plugin_settings import (
    FIELD_TYPES,
    PluginClass,
    SettingField,
    _fields,
    _source,
    _summary,
    load_class,
    registered,
)
from .strategy import Strategy

__all__ = [
    "ENTRY_POINT_GROUP",
    "FIELD_TYPES",
    "SettingField",
    "StrategyPlugin",
    "config_section",
    "load_strategy",
    "strategies",
]

ENTRY_POINT_GROUP = STRATEGIES


@dataclass(frozen=True)
class StrategyPlugin(PluginClass):
    """A strategy class from an installed package, imported and checked."""

    kind: ClassVar[str] = "Strategia"

    def create(self, raw: Mapping[str, Any]) -> Strategy[Any]:
        """The strategy: `cls(settings)` with settings built from raw values, or `cls()` without settings."""
        return super().create(raw)


def strategies() -> dict[str, EntryPoint]:
    """Registered strategies by name, not loaded yet: loading imports the strategy's package."""
    return registered(ENTRY_POINT_GROUP, "Dwie strategie")


def load_strategy(name: str) -> StrategyPlugin:
    """The registered strategy `name`, imported: a class with `act`, and settings the panel can draw."""
    ep = _entry_point(name)
    where = f"strategia {ep.name} ({ep.value})"

    cls = load_class(ep, "act(obs)", where, "Strategy")
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

    The file is the one `twin --config` gives (`config.read_config_data`).
    """
    return plugin_settings.config_section(name)


def _entry_point(name: str) -> EntryPoint:
    found = strategies()

    try:
        return found[name.lower()]
    except KeyError as exc:
        # Entry points live in the installed package's metadata: a package added to the
        # environment without `pip install` registers nothing.
        known = ", ".join(sorted(found)) or "zadne - zainstaluj pakiet ze strategia (uv pip install)"
        raise KeyError(f"Nieznana strategia {name!r}. Zainstalowane: {known}") from exc
