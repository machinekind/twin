"""The Galaxea A1X from galaxeo in the twin: found through its entry points, driven on a fake CAN bus.

The backend's own tests live in galaxeo; these check the twin's side of the plugin contract with the
real backend. Without galaxeo installed next to the twin the module skips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("galaxeo.robot")
pytest.importorskip("galaxeo.arm.fake")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.arm.fake import FakeA1XBus  # noqa: E402
from galaxeo.robot.backend import A1XArm  # noqa: E402

from lerobot_mp.robot import backend_context, create_backend  # noqa: E402
from lerobot_mp.robot.factory import PluginBackend  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.config import TwinConfig  # noqa: E402
from lerobot_mp.twin.robots import RobotSpec, get_spec, robot_config  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
STEP = 0.005                                   # one stream tick of the backend and the fake bus (200 Hz)
DT = 0.02                                      # one twin loop tick (50 Hz)
LSB_DEG = math.degrees(1.0 / P.POS_SCALE)


@dataclass
class Rig:
    """A twin on galaxeo's A1X backend over a fake bus, all on the bus's virtual clock."""

    twin: rt.Twin
    bus: FakeA1XBus
    arm: A1XArm | None = None

    def create_backend(self, cfg: TwinConfig, spec: RobotSpec) -> PluginBackend:
        """`create_backend` with the fake bus handed in: the same context and wrapper as the plugin path."""
        self.arm = A1XArm(backend_context(cfg, spec), bus=self.bus, clock=self.bus.now, threaded=False,
                          sleep=self.bus.sleep, connect_timeout_s=0.05)
        return PluginBackend(self.arm, "a1x")

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            for _ in range(round(DT / STEP)):
                self.bus.sleep(STEP)
                self.arm.io_step(self.bus.now())
            self.twin.step(DT)


def test_the_a1x_backend_comes_from_galaxeo_through_its_entry_point():
    spec = get_spec("a1x")
    cfg = robot_config(spec, overrides={"robot": {"backend": "a1x", "port": "can0"}})

    backend = create_backend(cfg, spec)

    assert isinstance(backend, PluginBackend)
    assert isinstance(backend.plugin, A1XArm)
    assert not backend.is_connected


def test_twin_connects_to_a_fake_a1x_holds_pose_and_follows_a_target(monkeypatch, tmp_path):
    spec = get_spec("a1x")
    bus = FakeA1XBus(start_deg=[spec.home.get(name, 0.0) for name in ARM])
    bus.tick(n=60)
    rig = Rig(twin=rt.Twin(Workspace(robot="a1x", path=tmp_path / "twin.json")), bus=bus)
    monkeypatch.setattr(rt, "create_backend", rig.create_backend)
    twin = rig.twin

    try:
        twin.connect("a1x", port="fake", threaded=False)
        rig.run(1.0)
        assert set(twin.status.measured) == set(spec.joints)
        assert bus.tx == []                                                      # nothing before a command

        twin.set_engaged(True)
        j1 = twin.status.measured["arm_joint1"]
        twin.set_target({"arm_joint1": j1 + 5.0})
        rig.run(1.5)
        streamed = [math.degrees(FakeA1XBus.pdes(data)[0]) for data in bus.frames(P.CMD_ID)]
        assert streamed
        assert max(abs(b - a) for a, b in zip(streamed, streamed[1:])) <= 90.0 * STEP + 2 * LSB_DEG
        assert twin.status.measured["arm_joint1"] == pytest.approx(j1 + 5.0, abs=2.0)

        bus.silent = True
        rig.run(0.5)
        assert twin.status.faults

        sent = len(bus.tx)
        rig.run(0.3)
        assert len(bus.tx) == sent                                     # a silent link stops the stream
    finally:
        twin.close()
