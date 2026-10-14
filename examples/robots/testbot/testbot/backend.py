"""An arm in memory: the test robot's backend.

A perfect position servo: every command, clamped to the joint limits, is the new position at
once. It never imports the twin, only the contracts (`lerobot-mp-api`); everything it knows comes
from the `BackendContext` the twin passes to `create`: joints, limits, port and the `testbot:`
section of the twin's config.

Settings (`testbot:` section):

    start: {j1: 10.0}   # pose at connect [app units]; joints not given start at 0
"""

from __future__ import annotations

from collections.abc import Mapping

from lerobot_mp_api import BackendContext, RobotInfo

SETTINGS = ("start",)


def create(ctx: BackendContext) -> MemoryArm:
    """Entry point of the `testbot` backend plugin."""
    return MemoryArm(ctx)


class MemoryArm:
    """A `BackendPlugin`: the five required members and `set_joint_limits`."""

    def __init__(self, ctx: BackendContext):
        unknown = set(ctx.settings) - set(SETTINGS)
        if unknown:
            raise ValueError(f"testbot: nieznane klucze {sorted(unknown)} (znane: {', '.join(SETTINGS)})")
        start = dict(ctx.settings.get("start", {}))
        foreign = set(start) - set(ctx.joints)
        if foreign:
            raise ValueError(f"testbot.start: stawy spoza ramienia {sorted(foreign)}")

        self.info = RobotInfo(name="testbot", description="ramie w pamieci", simulated=False)
        self.joints = tuple(ctx.joints)
        self.limits = {name: tuple(map(float, lim)) for name, lim in ctx.limits.items()}
        self.positions = {name: float(start.get(name, 0.0)) for name in self.joints}
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("testbot: ramie nie jest polaczone")
        return dict(self.positions)

    def send_joints(self, targets: Mapping[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("testbot: ramie nie jest polaczone")
        sent = {}
        for name in self.joints:
            if name in targets:
                lo, hi = self.limits[name]
                sent[name] = self.positions[name] = min(max(float(targets[name]), lo), hi)
        return sent

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        for name, (lo, hi) in limits.items():
            if name in self.limits:
                self.limits[name] = (float(lo), float(hi))
