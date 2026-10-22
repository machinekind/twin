"""Detectors from installed packages, and the settings the operator gives them.

A package registers a detector class under the `machinekind_twin.detectors` entry point group
(`machinekind_twin_api.entry_points.DETECTORS`), named like the detector:

    [project.entry-points."machinekind_twin.detectors"]
    cube = "cube_detector:ColourCube"

The class subclasses `Detector`, which the package imports from `machinekind_twin_api`. Its
settings work exactly like a strategy's (`plugin_settings`): a dataclass in the `settings` class
attribute, built strictly from the panel's form, which the config file's section named like the
detector (`cube:`) prefills. The twin creates the detector as `cls(settings)`, or `cls()` when the
class has no `settings`, and checks it by attribute only: a class with `detect` loads.

The config namespace is flat: a detector, a strategy and a backend that share a name share one
config section. `shared_section` tells the panel so it can warn.

Loader errors are Polish, like the rest of the panel. A plugin's own exception texts stay as the
plugin wrote them (English) and reach the panel after a Polish label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

from machinekind_twin_api.detector import Detector
from machinekind_twin_api.entry_points import DETECTORS

from . import plugin_settings
from .plugin_settings import PluginClass, _fields, _source, _summary, load_class, registered

__all__ = [
    "ENTRY_POINT_GROUP",
    "DetectorPlugin",
    "config_section",
    "detectors",
    "load_detector",
    "shared_section",
]

ENTRY_POINT_GROUP = DETECTORS


@dataclass(frozen=True)
class DetectorPlugin(PluginClass):
    """A detector class from an installed package, imported and checked."""

    kind: ClassVar[str] = "Detektor"

    def create(self, raw: Mapping[str, Any]) -> Detector[Any]:
        """The detector: `cls(settings)` with settings built from raw values, or `cls()` without settings."""
        return super().create(raw)


def detectors() -> dict[str, EntryPoint]:
    """Registered detectors by name, not loaded yet: loading imports the detector's package."""
    return registered(ENTRY_POINT_GROUP, "Dwa detektory")


def load_detector(name: str) -> DetectorPlugin:
    """The registered detector `name`, imported: a class with `detect`, and settings the panel can draw."""
    ep = _entry_point(name)
    where = f"detektor {ep.name} ({ep.value})"

    cls = load_class(ep, "detect(request)", where, "Detector")
    settings = getattr(cls, "settings", None)

    return DetectorPlugin(
        name=ep.name.lower(),
        cls=cls,
        settings=settings,
        fields=_fields(settings, where),
        source=_source(ep),
        summary=_summary(cls),
    )


def config_section(name: str) -> dict[str, Any]:
    """The config file's section named like the detector, raw; `{}` without a file or a section.

    The file is the one `twin --config` gives (`config.read_config_data`).
    """
    return plugin_settings.config_section(name)


def shared_section(name: str) -> str:
    """The other kind that registers `name` too, so both read one config section: "strategia" or
    "backend"; "" when the name is the detector's alone.

    It only feeds a warning: a registry that cannot be read (two packages clash on a name) counts
    as not sharing.
    """
    # Imported here: the backend factory and the strategies pull in more than discovery needs.
    from .backends.factory import backends
    from .strategies import strategies

    key = name.lower()
    for kind, registry in (("strategia", strategies), ("backend", backends)):
        try:
            if key in registry():
                return kind
        except ValueError:
            continue

    return ""


def _entry_point(name: str) -> EntryPoint:
    found = detectors()

    try:
        return found[name.lower()]
    except KeyError as exc:
        # Entry points live in the installed package's metadata: a package added to the
        # environment without `pip install` registers nothing.
        known = ", ".join(sorted(found)) or "zadne - zainstaluj pakiet z detektorem (uv pip install)"
        raise KeyError(f"Nieznany detektor {name!r}. Zainstalowane: {known}") from exc
