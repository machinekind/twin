"""Backend plugins: found through entry points, built from the twin's configuration.

A package registers a backend factory under the `machinekind_twin.backends` entry point group
(`machinekind_twin_api.entry_points.BACKENDS`):

    [project.entry-points."machinekind_twin.backends"]
    testbot = "testbot.backend:create"

`create(context: BackendContext)` returns the backend, a `RobotBackend` or any object following
`BackendPlugin`. The backend's settings are the config section named like the backend
(`testbot:`), handed over raw in `BackendContext.settings`. Every backend comes from an installed
package; the twin registers none of its own.

The twin's simulated arm (`sim`) is not a plugin: `runtime.SceneBackend` needs the twin's
scene, so `Twin.connect` builds it and never calls `create_backend` for it.

`open_read_only` opens a robot the twin only watches: a factory marked `read_only_capable` gets
`BackendContext.read_only`, and the twin holds the result as a `ReadOnlyArm`, which has no way to
send anything.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any

from machinekind_twin_api.backend import supports_read_only
from machinekind_twin_api.entry_points import BACKENDS

from ..config import TwinConfig
from .base import BackendContext, RobotBackend, RobotInfo

if TYPE_CHECKING:
    from ..robots import RobotSpec

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = BACKENDS

_REQUIRED = ("connect", "disconnect", "read_joints", "send_joints", "is_connected")


def backends() -> dict[str, EntryPoint]:
    """Registered backend plugins by name, not loaded yet (loading imports the plugin)."""
    found: dict[str, EntryPoint] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        name = ep.name.lower()
        if name in found and found[name].value != ep.value:
            raise ValueError(f"Dwa backendy o nazwie {name!r}: {found[name].value} i {ep.value}")
        found[name] = ep
    return found


def backend_context(cfg: TwinConfig, spec: RobotSpec, backend: str | None = None) -> BackendContext:
    """What the backend `backend` (default: the configured one) gets for this robot."""
    name = (backend or cfg.robot.backend).lower()
    return BackendContext(
        robot=spec.name,
        joints=tuple(spec.joints),
        gripper=spec.gripper,
        limits={n: (cfg.joint(n).min, cfg.joint(n).max) for n in spec.joints},
        max_vel={n: cfg.joint(n).max_vel for n in spec.joints},
        port=cfg.robot.port or None,
        settings=copy.deepcopy(cfg.sections.get(name, {})),
    )


def _factory(name: str, given: str) -> Any:
    """The registered factory of backend `name` (`given` is the name as configured, for the message)."""
    registered = backends()
    if name not in registered:
        # Entry points live in the installed package's metadata: a checkout updated without
        # `pip install -e .` has no backends registered at all.
        known = (", ".join(sorted(registered))
                 or "zadne - po aktualizacji repozytorium uruchom ponownie uv sync")
        raise ValueError(f"Nieznany backend robota: {given!r}. Zainstalowane: {known} "
                         "(albo `sim` - symulacja w blizniaku)")
    return registered[name].load()


def create_backend(cfg: TwinConfig, spec: RobotSpec) -> RobotBackend:
    """The real-arm backend `cfg.robot.backend` for this robot, from its plugin."""
    name = cfg.robot.backend.lower()
    if name == "sim":
        raise ValueError(
            "Backend `sim` to fizyka blizniaka - polacz przez Twin.connect('sim') "
            "(panel: Ramie = symulacja); create_backend tworzy tylko prawdziwe ramiona."
        )
    factory = _factory(name, cfg.robot.backend)
    backend = factory(backend_context(cfg, spec, name))
    return backend if isinstance(backend, RobotBackend) else PluginBackend(backend, name)


def open_read_only(cfg: TwinConfig, spec: RobotSpec, backend: str, port: str | None) -> ReadOnlyArm:
    """Backend `backend` of robot `spec` on `port`, created read-only; not connected yet.

    Refused, before the plugin is even created, unless its factory is marked `read_only_capable`.
    """
    name = backend.lower()
    if name == "sim":
        raise ValueError("`sim` nie ma czego czytac - wybierz stala poze albo backend sprzetu")
    factory = _factory(name, backend)
    if not supports_read_only(factory):
        raise ValueError(f"backend {name!r} nie umie czytac ramienia bez zapisu (read_only) - "
                         "zaktualizuj jego pakiet; do tego czasu robot stoi w stalej pozie")
    ctx = dataclasses.replace(backend_context(cfg, spec, name), port=port or None, read_only=True)
    return ReadOnlyArm(factory(ctx), name)


class ReadOnlyArm:
    """A backend the twin only reads: connect, read, faults and disconnect; nothing that sends."""

    def __init__(self, plugin: Any, name: str):
        missing = [m for m in ("connect", "disconnect", "read_joints") if not hasattr(plugin, m)]
        if missing:
            raise TypeError(f"Backend {name!r} nie ma {missing} - patrz kontrakt BackendPlugin")
        self._plugin = plugin
        self.name = name

    def connect(self) -> None:
        self._plugin.connect()

    def disconnect(self) -> None:
        self._plugin.disconnect()

    def read_joints(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._plugin.read_joints().items()}

    @property
    def link_silent(self) -> bool:
        return bool(getattr(self._plugin, "link_silent", getattr(self._plugin, "_link_silent", False)))

    def faults(self) -> list[str]:
        fn = getattr(self._plugin, "faults", None)
        try:
            return list(fn()) if fn is not None else []
        except Exception as exc:  # noqa: BLE001 - never raises, like `RobotBackend.faults`
            return [f"{type(exc).__name__}: {exc}"]


class PluginBackend(RobotBackend):
    """A `BackendPlugin` from another package as a `RobotBackend`.

    Required calls go straight to the plugin; optional ones fall back to the base class defaults
    when the plugin does not have them. Anything else (`enable`, `gripper_measured`, ...) is read
    from the plugin, so the twin's look-ups by name keep working.
    """

    def __init__(self, plugin: Any, name: str):
        missing = [m for m in _REQUIRED if not hasattr(plugin, m)]
        if missing:
            raise TypeError(f"Backend {name!r} nie ma {missing} - patrz kontrakt BackendPlugin")
        self.plugin = plugin
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        if attr in ("plugin", "name") or attr.startswith("__"):
            raise AttributeError(attr)
        return getattr(self.plugin, attr)

    @property
    def info(self) -> RobotInfo:
        info = getattr(self.plugin, "info", None)
        return RobotInfo(name=str(getattr(info, "name", self.name)),
                         description=str(getattr(info, "description", "")),
                         simulated=bool(getattr(info, "simulated", False)))

    def connect(self) -> None:
        self.plugin.connect()

    def disconnect(self) -> None:
        self.plugin.disconnect()

    def read_joints(self) -> dict[str, float]:
        return self.plugin.read_joints()

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        return self.plugin.send_joints(targets)

    @property
    def is_connected(self) -> bool:
        return bool(self.plugin.is_connected)

    @property
    def link_silent(self) -> bool:
        # `_link_silent` is what the A1X backend and the soarm backends set.
        return bool(getattr(self.plugin, "link_silent", getattr(self.plugin, "_link_silent", False)))

    def faults(self) -> list[str]:
        return self._call("faults", super().faults)

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return self._call("joint_limits", super().joint_limits)

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        fn = getattr(self.plugin, "set_joint_limits", None)
        if fn is not None:
            fn(limits)

    def gripper_range_q(self) -> tuple[float, float] | None:
        return self._call("gripper_range_q", super().gripper_range_q)

    def calibration_warnings(self) -> list[str]:
        return self._call("calibration_warnings", super().calibration_warnings)

    def efforts(self) -> dict[str, float] | None:
        return self._call("efforts", super().efforts)

    def velocities(self) -> dict[str, float] | None:
        return self._call("velocities", super().velocities)

    def set_grip_strength(self, strength: int | None) -> bool:
        fn = getattr(self.plugin, "set_grip_strength", None)
        return bool(fn(strength)) if fn is not None else False

    def step(self, dt: float) -> None:
        fn = getattr(self.plugin, "step", None)
        if fn is not None:
            fn(dt)

    def _call(self, method: str, default: Any) -> Any:
        fn = getattr(self.plugin, method, None)
        return fn() if fn is not None else default()
