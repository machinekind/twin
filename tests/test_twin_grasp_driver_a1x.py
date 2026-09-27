"""`GraspDriver` on the `a1x` backend over a fake CAN bus (`FakeA1XBus`), no hardware.

The twin runs with `threaded=False`. The driver runs on the test's thread with the twin's own
clock (`twin._clock`) and a `sleep` that ticks the bus and steps the twin until that much twin
time has passed, so every run is deterministic and needs no wall-clock time. Frames are fake,
stamped with the same clock.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("galaxeo")

from galaxeo import protocol as P  # noqa: E402
from galaxeo.bus import Frame as CanFrame  # noqa: E402
from test_a1x import STEP, Clock, FakeA1XBus  # noqa: E402
from test_twin_grasp_driver import Scripted, segments, spying, targets  # noqa: E402

from lerobot_mp.control.safety import SafetyState  # noqa: E402
from lerobot_mp.robot.a1x import A1XArm  # noqa: E402
from lerobot_mp.twin import robots  # noqa: E402
from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin.grasp import driver as driver_mod  # noqa: E402
from lerobot_mp.twin.grasp import motion  # noqa: E402
from lerobot_mp.twin.grasp.camera import Frame, WristCamera  # noqa: E402
from lerobot_mp.twin.grasp.contract import FB_AGE_END_S, TOUCH_EXPECTED_Z  # noqa: E402
from lerobot_mp.twin.grasp.driver import GraspDriver, TickRecord  # noqa: E402
from lerobot_mp.twin.grasp.hover import Hover, from_scene  # noqa: E402
from lerobot_mp.twin.grasp.rig import Rig as GraspRig  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
DT = 0.02                                      # twin loop period (4 bus ticks)
SPEC = robots.REGISTRY["a1x"]
CLOSE = [0.0, 0.0, 0.0, 0.0, -1.0]
STILL = [0.0, 0.0, 0.0, 0.0, 1.0]


def hover() -> Hover:
    return from_scene(RobotKinematics(SPEC), np.array([0.0, 0.33]), 0.0, touch_z=TOUCH_EXPECTED_Z,
                      limits=SPEC.limits)


class WallBus(FakeA1XBus):
    """Joint `wall[0]` cannot be commanded past `wall[1]` [rad] in the direction `wall[2]`."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.wall: tuple[int, float, float] | None = None

    def send(self, can_id: int, data: bytes, *, fd: bool = True) -> None:
        super().send(can_id, data, fd=fd)
        if self.wall is not None and self.p_des is not None:
            j, lim, sign = self.wall
            if sign * (self.p_des[j] - lim) > 0:
                self.p_des[j] = lim


class Frames:
    """Black policy frames stamped with the twin's clock."""

    def __init__(self, twin: rt.Twin) -> None:
        self.twin = twin
        self.seq = 0

    def latest(self) -> Frame:
        self.seq += 1
        return Frame(np.zeros((120, 160, 3), np.uint8), self.twin._clock, self.seq)

    def close(self) -> None:
        pass


class Rig:
    """The twin on the fake A1X, started at the hover; `sleep` is the driver's clock."""

    def __init__(self, monkeypatch, *, stale_ms: float | None = None,
                 max_step_deg_s: float | None = None) -> None:
        self.hover = hover()
        self.clock = Clock()
        self.bus = WallBus([self.hover.joints[j] for j in ARM])
        self.bus.tick(n=60)
        self.arm: A1XArm | None = None
        #: Called after every twin step.
        self.hooks: list[Callable[[], None]] = []
        self.faults: list[str] = []

        def backend_sleep(secs: float) -> None:
            self.clock.t += secs
            self.bus.tick(secs)

        def create(cfg):
            if stale_ms is not None:
                cfg.a1x.stale_ms = stale_ms
            if max_step_deg_s is not None:
                cfg.a1x.max_step_deg_s = max_step_deg_s
            self.arm = A1XArm(cfg, bus=self.bus, clock=self.clock, threaded=False, sleep=backend_sleep,
                              connect_timeout_s=0.05)
            return self.arm

        monkeypatch.setattr(rt, "create_backend", create)
        self.twin = rt.Twin(Workspace(robot="a1x"))
        self.twin.connect("a1x", port="fake", threaded=False)
        self.pump(0.3)

    def step(self) -> None:
        assert self.arm is not None
        for _ in range(int(round(DT / STEP))):
            self.clock.t += STEP
            self.bus.tick(STEP)
            self.arm.io_step(self.clock.t)
        self.twin.step(DT)
        self.faults.extend(self.twin.status.faults)
        for hook in list(self.hooks):
            hook()

    def sleep(self, secs: float) -> None:
        """At least one loop step, then until `secs` of twin time have passed."""
        t_end = self.twin._clock + secs
        self.step()
        while self.twin._clock < t_end - 1e-9:
            self.step()

    def pump(self, secs: float) -> None:
        for _ in range(int(round(secs / DT))):
            self.step()

    def driver(self, script, *, mode: str = "real", **kw) -> GraspDriver:
        kw.setdefault("sleep", self.sleep)
        return GraspDriver(self.twin, Scripted(script), Frames(self.twin), WristCamera(), GraspRig.for_sim(),
                           mode=mode, real_ok=True, clock=lambda: self.twin._clock, **kw)

    def streamed(self) -> float:
        assert self.arm is not None
        gs = self.arm.gripper_state()
        assert gs is not None and gs.streamed is not None
        return gs.streamed

    def openings(self, since: int = 0) -> list[float]:
        """Openings (0..100) of every 0x051 frame sent since `since`."""
        a = self.twin._supervisor.cfg.a1x if self.twin._supervisor is not None else None
        closed, opened = (a.gripper_closed_pdes, a.gripper_open_pdes) if a is not None else (0.6, -2.0)
        out = []
        for can_id, data, _ in self.bus.tx[since:]:
            if can_id == P.GRIP_ID:
                p = struct.unpack_from(">h", data, 0)[0] / P.POS_SCALE
                out.append((p - closed) * 100.0 / (opened - closed))
        return out

    def close(self) -> None:
        self.twin.close()


@pytest.fixture
def rig(monkeypatch):
    r = Rig(monkeypatch)
    yield r
    r.close()


def run(rig: Rig, drv: GraspDriver) -> None:
    assert drv.check() is None, drv.detail
    assert drv.take() is None, drv.detail
    assert drv.preposition(rig.hover) is None, drv.detail


LSB_UNITS = 100.0 / 2.6 / P.POS_SCALE          # one p_des step of the jaw, in opening units


def test_dry_mode_transmits_no_frame_at_all(rig):
    recs: list[TickRecord] = []
    drv = rig.driver(segments((1, STILL)), mode="dry")
    try:
        assert drv.check() is None
        for _ in range(10):
            rec, inp = drv.dry_tick(rig.hover)
            recs.append(rec)
            rig.sleep(0.05)
    finally:
        drv.close()
    rig.pump(0.3)
    assert rig.bus.tx == []
    assert rig.twin.owner is None
    assert recs[-1].kind == "policy", recs[-1]


def test_shadow_never_sends_an_opening_below_99(rig):
    recs: list[TickRecord] = []
    script = segments((8, [0.0, 0.0, -1.0, 0.0, 1.0]), (40, [0.0, 0.0, -1.0, 0.0, -1.0]))
    drv = rig.driver(script, mode="shadow", max_ticks=40, on_tick=recs.append)
    try:
        run(rig, drv)
        rep = drv.episode(rig.hover)
    finally:
        drv.close()
    rig.pump(0.5)
    assert rep.ending == "tick cap", rep
    assert min(r.g_cmd for r in recs) < 0.4           # the policy closed; the jaw was never told to
    openings = rig.openings()
    assert openings and min(openings) >= 99.0 - LSB_UNITS


def close_mid_way(rig: Rig, at_record: int, action: Callable[[], None], **kw) -> tuple[GraspDriver, object]:
    """An episode that closes the jaw at the hover; `action` runs when record `at_record` is written."""
    n = [0]

    def on_tick(rec: TickRecord) -> None:
        n[0] += 1
        if n[0] == at_record:
            action()

    drv = rig.driver(segments((60, CLOSE)), max_ticks=60, on_tick=on_tick, **kw)
    run(rig, drv)
    return drv, drv.episode(rig.hover)


def test_a_guard_ending_mid_close_stops_the_jaw_stream_within_5_units(rig):
    def spike() -> None:
        rig.bus.eff[0] = 60.0                          # joint 1 pushes hard: effort rule (12, guessed)

    drv, rep = close_mid_way(rig, 6, spike)
    try:
        at_end = rig.streamed()
        rig.bus.eff[0] = 0.0
        rig.pump(1.0)
        after = rig.streamed()
    finally:
        drv.close()
    assert rep.ending == "impact effort arm_joint1", rep
    assert 5.0 < at_end < 95.0, at_end                 # it was mid-close
    assert abs(after - at_end) <= 5.0, (at_end, after)


def test_stop_now_mid_close_stops_the_jaw_stream_within_5_units_with_the_state_stop(rig):
    seen: dict[str, float] = {}

    def enter() -> None:
        seen["stream"] = rig.streamed()
        motion.stop_now(rig.twin, "STOP z klawiatury (chwyt)")

    drv, rep = close_mid_way(rig, 6, enter)
    try:
        rig.pump(1.0)
        after = rig.streamed()
    finally:
        drv.close()
    assert rep.ending == "stop", rep
    assert rig.twin.safety_state == SafetyState.ESTOP
    assert 5.0 < seen["stream"] < 95.0
    assert abs(after - seen["stream"]) <= 5.0, (seen, after)


def test_a_twin_side_stop_mid_close_stops_the_jaw_within_8_units(rig):
    seen: dict[str, float] = {}

    def watch() -> None:
        if "stream" not in seen and rig.twin.safety_state == SafetyState.ESTOP:
            seen["stream"] = rig.streamed()

    def foreign() -> None:
        # A command frame this backend never sent: another sender on the link, the twin STOPs.
        rig.bus.rx.append(CanFrame(P.CMD_ID, P.encode_arm([0.0] * 6, 20.0, 1.0)))
        rig.hooks.append(watch)

    drv, rep = close_mid_way(rig, 6, foreign)
    try:
        rig.pump(1.0)
        after = rig.streamed()
    finally:
        drv.close()
    assert rep.ending == "stop", rep
    assert "stream" in seen and 5.0 < seen["stream"] < 95.0
    assert abs(after - seen["stream"]) <= 8.0, (seen, after)


def test_after_tick_cap_with_the_jaw_closed_the_jaw_command_stays_0(rig):
    drv = rig.driver(segments((60, CLOSE)), max_ticks=40)
    try:
        run(rig, drv)
        rep = drv.episode(rig.hover)
        rig.pump(1.0)
        gs = rig.arm.gripper_state()
    finally:
        drv.close()
    assert rep.ending == "tick cap", rep
    assert gs.commanded == 0.0 and gs.streamed <= 1.0
    assert rig.twin.status.command["gripper"] == 0.0


def test_the_deliberate_open_sends_99_then_100(rig):
    with spying(rig.twin) as calls:
        drv = rig.driver(segments((1, STILL)))
        try:
            run(rig, drv)
        finally:
            drv.close()
    grips = [t["gripper"] for t in targets(calls)]
    assert grips[0] == 99.0 and grips[-1] == 100.0
    assert grips.index(100.0) > grips.index(99.0)
    openings = rig.openings()
    k = int(np.argmin(openings))
    assert abs(openings[k] - 99.0) <= LSB_UNITS
    assert openings[-1] >= 100.0 - LSB_UNITS and k < len(openings) - 1


def test_a_stalled_joint_ends_impact_lag_before_the_backends_deaf_fault(rig):
    # Joint 1 stands at 90 deg; a wall 1 deg below it. Moving the TCP towards +x turns joint 1 down.
    j1 = math.radians(rig.hover.joints["arm_joint1"])
    rig.bus.wall = (0, j1 - math.radians(1.0), -1.0)
    drv = rig.driver(segments((60, [1.0, 0.0, 0.0, 0.0, 1.0])), max_ticks=60)
    try:
        run(rig, drv)
        rep = drv.episode(rig.hover)
    finally:
        drv.close()
    assert rep.ending == "impact lag arm_joint1", rep
    assert not any("gluche" in f for f in rig.faults), rig.faults


def test_feedback_frames_0_2_s_old_skip_with_stale_feedback_although_the_twin_samples_are_fresh(monkeypatch):
    rig = Rig(monkeypatch, stale_ms=1000.0)           # the backend keeps its link "alive" for a second
    recs: list[TickRecord] = []
    silent: dict[str, float] = {}

    def resume() -> None:
        if rig.bus.silent and rig.twin._clock - silent["t"] >= 0.22:
            rig.bus.silent = False

    def on_tick(rec: TickRecord) -> None:
        recs.append(rec)
        if len(recs) == 5:
            rig.bus.silent = True
            silent["t"] = rig.twin._clock
            rig.hooks.append(resume)

    drv = rig.driver(segments((40, STILL)), max_ticks=25, on_tick=on_tick)
    try:
        run(rig, drv)
        rep = drv.episode(rig.hover)
    finally:
        drv.close()
        rig.close()
    assert rep.ending == "tick cap", rep
    stale = [r for r in recs if r.skip == "stale feedback"]
    assert stale, [(r.skip, r.fb_age_s) for r in recs]
    assert all(r.fb_age_s > 0.1 and r.joint_age_s <= 0.1 for r in stale)


def test_feedback_that_stays_silent_ends_stale_feedback_at_0_3_s_without_a_target_after(monkeypatch):
    rig = Rig(monkeypatch, stale_ms=1000.0)
    monkeypatch.setattr(driver_mod, "SKIPS_END", 100)  # only the hard guard can end it
    recs: list[TickRecord] = []
    n_sent: list[int] = []

    def on_tick(rec: TickRecord) -> None:
        recs.append(rec)
        n_sent.append(len(calls["set_target"]))
        if len(recs) == 5:
            rig.bus.silent = True

    with spying(rig.twin) as calls:
        drv = rig.driver(segments((60, STILL)), max_ticks=60, on_tick=on_tick)
        try:
            run(rig, drv)
            rep = drv.episode(rig.hover)
        finally:
            drv.close()
            rig.close()
    assert rep.ending == "stale feedback", rep
    assert recs[-1].skip == "stale feedback" and recs[-1].fb_age_s > FB_AGE_END_S, recs[-1]
    assert all(r.fb_age_s <= FB_AGE_END_S for r in recs[:-1])
    assert n_sent[-1] == n_sent[-2]                    # the ending tick sent nothing


def test_a_frozen_twin_loop_skips_with_stale_feedback_and_ends_after_five(rig, monkeypatch):
    recs: list[TickRecord] = []
    drv = rig.driver(segments((40, STILL)), max_ticks=40, on_tick=recs.append)
    try:
        run(rig, drv)
        monkeypatch.setattr(rig.twin, "_frozen", lambda: True)
        rep = drv.episode(rig.hover)
    finally:
        drv.close()
    assert rep.ending == "stale feedback", rep
    assert [r.skip for r in recs] == ["stale feedback"] * driver_mod.SKIPS_END


def test_a_backend_slew_under_90_refuses_in_take_before_any_target(monkeypatch):
    rig = Rig(monkeypatch, max_step_deg_s=45.0)
    try:
        with spying(rig.twin) as calls:
            drv = rig.driver(segments((1, STILL)))
            try:
                assert drv.check() is None
                assert drv.take() == "speed limited"
                assert "45 st./s" in drv.detail
            finally:
                drv.close()
        assert calls["set_target"] == []
        assert rig.twin.owner is None
    finally:
        rig.close()


def test_an_episode_without_take_writes_nothing_while_the_jaw_is_in_mid_stroke(rig):
    tw = rig.twin
    tw.claim("test")
    assert tw.set_engaged(True, owner="test")
    tw.set_target({**rig.hover.joints, "gripper": 0.0}, owner="test")
    rig.pump(0.5)
    tw.release("test")
    assert 5.0 < rig.streamed() < 95.0
    with spying(tw) as calls:
        drv = rig.driver(segments((1, STILL)))
        try:
            rep = drv.episode(rig.hover)
        finally:
            drv.close()
    assert rep.ending == "error", rep
    assert calls["order"] == []


def test_without_arm_efforts_every_joint_is_listed_in_no_effort_rule(rig):
    drv = rig.driver(segments((1, STILL)))
    try:
        run(rig, drv)
        assert drv.arm.no_effort_rule == []
        rig.arm.efforts = lambda: None                 # one stale read at the moment the guard is armed
        drv.arm.guard(lag_deg=3.0)
        assert drv.arm.no_effort_rule == list(ARM)
    finally:
        drv.close()


def test_a_foreign_0x050_frame_during_the_listen_makes_check_return_busy(rig):
    injected = [False]

    def listen(secs: float) -> None:
        if not injected[0]:
            injected[0] = True
            rig.bus.rx.append(CanFrame(P.CMD_ID, P.encode_arm([0.1] * 6, 20.0, 1.0)))
        rig.sleep(secs)

    drv = rig.driver(segments((1, STILL)), sleep=listen)
    try:
        assert drv.check() == "busy"
        assert drv.detail == "inny nadawca na laczu CAN"
        assert injected[0]
    finally:
        drv.close()
