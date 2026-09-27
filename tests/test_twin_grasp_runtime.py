"""What the grasp driver needs from the twin: finite targets only, the effective speed limits,
the scene's gripper state, and holding with an explicit jaw opening.

The sim tests run the MuJoCo twin. The jaw tests run the twin on the fake A1X bus from
`test_a1x.py` (`threaded=False`, one virtual clock), where the measured opening is the last
command, so a hold must take the streamed opening or a closing jaw finishes its stroke.
"""

from __future__ import annotations

import math
import os
import struct
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
DT = 0.02
NON_FINITE = (math.nan, math.inf, -math.inf)


@pytest.fixture
def sim():
    tw = Twin(Workspace(robot="a1x"))
    tw.connect("sim", threaded=False)
    try:
        yield tw
    finally:
        tw.close()


def run(twin: Twin, seconds: float) -> None:
    for _ in range(int(round(seconds / DT))):
        twin.step(DT)


class A1XRig:
    """The twin on the fake A1X bus: each twin tick is four 5 ms bus and IO steps."""

    def __init__(self, monkeypatch, max_step_deg_s: float | None = None):
        pytest.importorskip("galaxeo")
        from galaxeo import protocol as P
        from test_a1x import STEP, Clock, FakeA1XBus

        from lerobot_mp.robot.a1x import A1XArm
        from lerobot_mp.twin import robots

        home = robots.REGISTRY["a1x"].home
        self.P, self.io_step = P, STEP
        self.clock = Clock()
        self.bus = FakeA1XBus([home.get(n, 0.0) for n in ARM])
        self.bus.tick(n=60)
        self.arm: A1XArm | None = None

        def sleep(secs):
            self.clock.t += secs
            self.bus.tick(secs)

        def create(cfg):
            if max_step_deg_s is not None:
                cfg.a1x.max_step_deg_s = max_step_deg_s
            self.arm = A1XArm(cfg, bus=self.bus, clock=self.clock, threaded=False, sleep=sleep,
                              connect_timeout_s=0.05)
            return self.arm

        monkeypatch.setattr(rt, "create_backend", create)
        self.twin = Twin(Workspace(robot="a1x"))
        self.twin.connect("a1x", port="fake", threaded=False)

    def run(self, seconds: float, until=None) -> bool:
        for _ in range(int(round(seconds / DT))):
            if until is not None and until():
                return True
            for _ in range(int(round(DT / self.io_step))):
                self.clock.t += self.io_step
                self.bus.tick(self.io_step)
                self.arm.io_step(self.clock.t)
            self.twin.step(DT)
        return until is not None and bool(until())

    def payloads(self, can_id: int, since: int = 0) -> list[bytes]:
        return [d for i, d, _ in self.bus.tx[since:] if i == can_id]

    def openings(self, since: int = 0) -> list[float]:
        """Opening 0..100 of every 0x051 frame sent after `since`."""
        a = self.arm.acfg
        span = a.gripper_open_pdes - a.gripper_closed_pdes
        return [(struct.unpack_from(">h", d, 0)[0] / self.P.POS_SCALE - a.gripper_closed_pdes) * 100.0 / span
                for d in self.payloads(self.P.GRIP_ID, since)]

    def streamed(self) -> float:
        gs = self.twin.backend.gripper_state()
        assert gs is not None and gs.streamed is not None
        return gs.streamed

    def close_jaw_to(self, opening: float) -> float:
        """Closes the jaw from its reported 100 to 0; returns the streamed opening once <= `opening`."""
        tw = self.twin
        assert tw.status.measured["gripper"] == 100.0
        tw.set_engaged(True)
        tw.set_target({"gripper": 0.0})

        def reached() -> bool:
            s = tw.backend.gripper_state().streamed
            return s is not None and s <= opening

        assert self.run(2.0, reached)
        return self.streamed()

    def close(self) -> None:
        self.twin.close()


@pytest.fixture
def a1x(monkeypatch):
    rig = A1XRig(monkeypatch)
    try:
        yield rig
    finally:
        rig.close()


# ------------------------------------------------------------------ finite targets
def test_a_non_finite_target_raises_and_changes_neither_the_target_nor_the_sim_arm(sim):
    sim.set_engaged(True)
    m = sim.status.measured
    goal = {**{k: m[k] for k in ARM}, "arm_joint1": m["arm_joint1"] + 5.0, "gripper": 40.0}
    sim.set_target(goal)
    run(sim, 1.5)
    target, command, joints = dict(sim._target), dict(sim.status.command), sim.joints()
    ctrl = sim.scene.data.ctrl.copy()
    for bad in NON_FINITE:
        for name in ("arm_joint1", "arm_joint6", "gripper"):
            with pytest.raises(ValueError, match="nie jest liczba"):
                sim.set_target({**goal, name: bad})
    with pytest.raises(ValueError, match="nie jest liczba"):      # before the ownership check
        sim.set_target({"arm_joint1": math.nan}, owner="ktos inny")
    assert sim._target == target
    run(sim, 0.5)
    assert sim.status.command == command
    assert (sim.scene.data.ctrl == ctrl).all()
    # The joints themselves still settle a little under physics after the arm has arrived.
    after = sim.joints()
    assert all(math.isfinite(v) for v in after.values())
    assert max(abs(after[k] - joints[k]) for k in joints) < 0.1


def test_a_non_finite_target_raises_and_moves_no_frame_on_the_a1x_bus(a1x):
    tw, P = a1x.twin, a1x.P
    tw.set_engaged(True)
    m = tw.status.measured
    goal = {**{k: m[k] for k in ARM}, "arm_joint1": m["arm_joint1"] + 5.0, "gripper": 50.0}
    tw.set_target(goal)
    a1x.run(2.0)
    n = len(a1x.bus.tx)
    a1x.run(0.2)
    arm_held, grip_held = set(a1x.payloads(P.CMD_ID, n)), set(a1x.payloads(P.GRIP_ID, n))
    assert len(arm_held) == 1 and len(grip_held) == 1                    # settled: one frame repeated
    for bad in NON_FINITE:
        for name in ("arm_joint1", "gripper"):
            with pytest.raises(ValueError, match="nie jest liczba"):
                tw.set_target({**goal, name: bad})
    n = len(a1x.bus.tx)
    a1x.run(1.0)
    assert set(a1x.payloads(P.CMD_ID, n)) == arm_held
    assert set(a1x.payloads(P.GRIP_ID, n)) == grip_held
    assert tw.safety_state.value != "ESTOP" and not tw.status.faults


# ---------------------------------------------------------- the scene's gripper state
def test_scene_gripper_state_follows_the_scene_and_changes_nothing(sim):
    b, s = sim.backend, sim.scene
    sim.set_engaged(True)
    sim.set_target({"gripper": 20.0})
    run(sim, 0.1)
    gs = b.gripper_state()
    assert gs.position == sim.joints()["gripper"]                          # the slide, not the ctrl
    assert gs.position - gs.commanded > 2.0
    run(sim, 0.9)
    qpos, ctrl = s.data.qpos.copy(), s.data.ctrl.copy()
    for _ in range(100):
        gs = b.gripper_state()
    assert (s.data.qpos == qpos).all() and (s.data.ctrl == ctrl).all()
    assert gs.measured and not gs.frozen
    assert gs.position == pytest.approx(20.0, abs=2.0)
    assert gs.commanded == gs.target == gs.streamed == pytest.approx(20.0, abs=1e-6)
    assert gs.effort == s.data.actuator_force[s.model.actuator(rt.sc.PREFIX + "gripper").id]
    assert gs.t == pytest.approx(1.0) and gs.age_s == 0.0
    assert gs.rate is None and gs.force_threshold is None
    sim.disconnect()
    assert b.gripper_state() is None


def test_scene_gripper_state_is_stamped_with_the_monotonic_clock_when_threaded():
    tw = Twin(Workspace(robot="a1x"))
    try:
        tw.connect("sim")
        t0 = time.monotonic()
        gs = tw.backend.gripper_state()
        assert t0 <= gs.t <= time.monotonic()
    finally:
        tw.close()


# ------------------------------------------------------------------ speed limits
def test_speed_limits_are_the_supervisors_and_follow_the_speed_cap(sim):
    lim = sim.speed_limits()
    assert set(lim) == set(sim.spec.joints)
    assert all(lim[k] == 90.0 for k in ARM) and lim["gripper"] == 300.0
    sim.set_speed(20.0)
    lim = sim.speed_limits()
    assert all(lim[k] == 20.0 for k in ARM) and lim["gripper"] == 300.0
    sim.set_speed(None)
    assert all(sim.speed_limits()[k] == 90.0 for k in ARM)


def test_speed_limits_are_empty_when_not_connected():
    tw = Twin(Workspace(robot="a1x"))
    try:
        assert tw.speed_limits() == {}
        tw.connect("sim", threaded=False)
        assert tw.speed_limits()
        tw.disconnect()
        assert tw.speed_limits() == {}
    finally:
        tw.close()


@pytest.mark.parametrize("slew, arm", [(None, 90.0), (45.0, 45.0)])
def test_speed_limits_fold_in_the_a1x_backend_slew(monkeypatch, slew, arm):
    rig = A1XRig(monkeypatch, max_step_deg_s=slew)
    try:
        lim = rig.twin.speed_limits()
        assert all(lim[k] == arm for k in ARM) and lim["gripper"] == 300.0
        rig.twin.set_speed(20.0)
        assert all(rig.twin.speed_limits()[k] == 20.0 for k in ARM)
    finally:
        rig.close()


# ------------------------------------------------------ holding with a jaw opening
def test_holding_with_the_streamed_opening_stops_a_closing_a1x_jaw(a1x):
    g = a1x.close_jaw_to(63.5)
    assert g == pytest.approx(63.0, abs=2.0)
    n = len(a1x.bus.tx)
    a1x.twin.hold_measured(gripper=g)
    a1x.run(1.5)
    opens = a1x.openings(n)
    assert len(opens) >= 250
    assert max(abs(o - g) for o in opens) <= 5.0
    assert a1x.streamed() == pytest.approx(g, abs=5.0)


def test_holding_with_the_streamed_opening_after_estop_stops_the_jaw_and_keeps_the_stop(a1x):
    a1x.close_jaw_to(63.5)
    a1x.twin.estop("STOP w tescie")
    g = a1x.streamed()
    n = len(a1x.bus.tx)
    a1x.twin.hold_measured(gripper=g)
    a1x.run(1.5)
    opens = a1x.openings(n)
    assert len(opens) >= 250
    assert max(abs(o - g) for o in opens) <= 5.0
    assert a1x.twin.safety_state.value == "ESTOP"


@pytest.mark.parametrize("how", ["hold", "stop"])
def test_a_hold_or_stop_without_the_keyword_stops_a_closing_a1x_jaw_at_the_stream(a1x, how):
    # The A1X's "measured" opening is its last command (0): held as is, the jaw closed fully.
    g = a1x.close_jaw_to(63.5)
    n = len(a1x.bus.tx)
    if how == "hold":
        a1x.twin.hold_measured()
    else:
        a1x.twin.estop("STOP w tescie")
    a1x.run(1.5)
    opens = a1x.openings(n)
    assert len(opens) >= 250
    assert max(abs(o - g) for o in opens) <= 8.0
    assert a1x.streamed() == pytest.approx(g, abs=8.0)


def test_hold_with_a_gripper_value_changes_only_the_gripper_entry(sim):
    sup = sim._supervisor
    sim.hold_measured()
    plain = sup.command
    sim.hold_measured(gripper=None)
    assert sup.command == plain
    sim.hold_measured(gripper=30.0)
    held = sup.command
    assert {k: held[k] for k in ARM} == {k: plain[k] for k in ARM}
    assert held["gripper"] == 30.0
    reading = dict(sim.status.measured, arm_joint1=sim.status.measured["arm_joint1"] + 1.0)
    sim.hold_measured(reading, gripper=25.0)
    held = sup.command
    assert {k: held[k] for k in ARM} == pytest.approx({k: reading[k] for k in ARM})
    assert held["gripper"] == 25.0


def test_hold_clips_the_gripper_value_and_refuses_a_non_finite_one(sim):
    sim.hold_measured(gripper=150.0)
    sim.step(DT)
    assert sim.status.command["gripper"] == 100.0
    sim.hold_measured(gripper=-5.0)
    sim.step(DT)
    assert sim.status.command["gripper"] == 0.0
    command = dict(sim._supervisor.command)
    for bad in NON_FINITE:
        with pytest.raises(ValueError, match="nie jest liczba"):
            sim.hold_measured(gripper=bad)
    assert sim._supervisor.command == command
    sim.disconnect()
    sim.hold_measured(gripper=50.0)                                        # not connected: nothing to hold
    with pytest.raises(ValueError, match="nie jest liczba"):
        sim.hold_measured(gripper=math.nan)
