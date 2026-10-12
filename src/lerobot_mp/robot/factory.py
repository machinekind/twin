"""Backend plugins: found through entry points, built from the twin's configuration.

A package registers a backend factory under the `lerobot_mp.backends` entry point group:

    [project.entry-points."lerobot_mp.backends"]
    feetech = "lerobot_mp.robot.feetech:create"

`create(context: BackendContext)` returns the backend, a `RobotBackend` or any object following
`BackendPlugin`. The backend's settings are the config section named like the backend
(`feetech:`, `a1x:`), handed over raw in `BackendContext.settings`.

The twin's simulated arm (`sim`) is not a plugin: `twin.runtime.SceneBackend` needs the twin's
scene, so `Twin.connect` builds it and never calls `create_backend` for it.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any

from ..twin.config import TwinConfig
from .base import BackendContext, RobotBackend, RobotInfo

if TYPE_CHECKING:
    from ..twin.robots import RobotSpec

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "lerobot_mp.backends"

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


def create_backend(cfg: TwinConfig, spec: RobotSpec) -> RobotBackend:
    """The real-arm backend `cfg.robot.backend` for this robot, from its plugin."""
    name = cfg.robot.backend.lower()
    if name == "sim":
        raise ValueError(
            "Backend `sim` to fizyka blizniaka - polacz przez Twin.connect('sim') "
            "(panel: Ramie = symulacja); create_backend tworzy tylko prawdziwe ramiona."
        )
    registered = backends()
    if name not in registered:
        # Entry points live in the installed package's metadata: a checkout updated without
        # `pip install -e .` has no backends registered at all.
        known = (", ".join(sorted(registered))
                 or "zadne - po aktualizacji repozytorium uruchom ponownie pip install -e .")
        raise ValueError(f"Nieznany backend robota: {cfg.robot.backend!r}. Zainstalowane: {known} "
                         "(albo `sim` - symulacja w blizniaku)")
    factory = registered[name].load()
    backend = factory(backend_context(cfg, spec, name))
    return backend if isinstance(backend, RobotBackend) else PluginBackend(backend, name)


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
        # The backends written in this repository set `_link_silent`; they keep it when they move out.
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

    def step(self, dt: float) -> None:
        fn = getattr(self.plugin, "step", None)
        if fn is not None:
            fn(dt)

    def _call(self, method: str, default: Any) -> Any:
        fn = getattr(self.plugin, method, None)
        return fn() if fn is not None else default()
