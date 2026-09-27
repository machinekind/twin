"""Bench measurements of the grasp student, without hardware.

The jaw log runs the twin on the fake A1X bus from `test_a1x.py` with a jaw that works (effort
while it moves, a cube that blocks a close); teaching and latency run the MuJoCo twin. Both twins
use `connect(threaded=False)`: the tools get the twin's clock and a `sleep` that steps the loop
(and the bus), so nothing waits on the wall clock. Keys are scripted (`FakeKeys`). Every file is
written under `tmp_path`.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin import runtime as rt  # noqa: E402
from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.grasp import bench  # noqa: E402
from lerobot_mp.twin.grasp.contract import (  # noqa: E402
    HOVER,
    OWNER,
    STREAM_STROKE_S,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    wrap,
)
from lerobot_mp.twin.grasp.hover import solve_hover  # noqa: E402
from lerobot_mp.twin.grasp.motion import ArmSession  # noqa: E402
from lerobot_mp.twin.grasp.policy import find_checkpoint  # noqa: E402
from lerobot_mp.twin.grasp.rig import Rig  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
CKPT = find_checkpoint()
ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
DT = 0.02                                      # twin loop period
#: p_des of a jaw closed on a 3 cm cube: gap 30 mm = opening 27, closed 0.6, open -2.0.
CUBE_P = 0.6 + 27.0 / 100.0 * (-2.6)
TEACH_XY = (0.0, 0.33)
START_Z = 0.10                                 # the twin's arm sags 2.8 mm radially here, inside arrival


def ns(**kw) -> argparse.Namespace:
    base = dict(action="gripper", backend="sim", port=None, i_know=False, rig=None, workspace=None,
                checkpoint=None, xy=None, yaw_deg=0.0, start_z=START_Z, gap_mm=None, set_freeze=None,
                error_px=None, force=False, seconds=None, object_height=0.030, camera=None)
    base.update(kw)
    return argparse.Namespace(**base)


class FakeKeys:
    """`Keys` with scripted answers; a bare Enter once armed sets `stop`, as the real reader does."""

    def __init__(self, lines=(), *, confirm: bool = True, proceed: bool = True, on_proceed=None) -> None:
        self.stop = threading.Event()
        self.armed = False
        self.interactive = False
        self.prompts: list[str] = []
        self._lines = list(lines)
        self._confirm, self._proceed, self._on_proceed = confirm, proceed, on_proceed
        self._lock = threading.Lock()

    def start(self) -> None:
        pass

    def arm(self) -> None:
        self.armed = True

    def note_end(self) -> None:
        pass

    def confirm(self, prompt: str, timeout_s: float | None = None) -> bool:
        self.prompts.append(prompt)
        return self._confirm

    def proceed(self, prompt: str, word: str = "d", timeout_s: float | None = None) -> bool:
        self.prompts.append(prompt)
        if self._on_proceed is not None:
            self._on_proceed()
        if not self._proceed:
            self.stop.set()
        return self._proceed

    def readline(self, timeout_s: float | None = None) -> str | None:
        with self._lock:
            if not self._lines:
                return None
            text = self._lines.pop(0)
        if isinstance(text, BaseException):
            raise text
        if text == "" and self.armed:
            self.stop.set()
        return text


def close_twin(twin: Twin) -> None:
    """The tools close the twin they were given; a second `close` finds the render thread stopped."""
    try:
        twin.close()
    except RuntimeError:
        pass


# ------------------------------------------------------------------ the MuJoCo twin, stepped
class SimBench:
    """The simulated bench stepped by the test: `sleep` advances a virtual clock and ticks the loop."""

    def __init__(self) -> None:
        self.twin = Twin(Workspace(robot="a1x"))
        self.twin.connect("sim", threaded=False)
        bench.configure_sim(self.twin)
        self.kin = RobotKinematics(self.twin.spec)
        self.t = self.twin._clock
        self.hooks: list = []

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(float(dt), 0.0)
        while self.twin._clock + DT <= self.t + 1e-12 and self.twin.connected:
            self.twin.step(DT)
            for hook in self.hooks:
                hook()

    def bench(self, rig_path: Path) -> bench.Bench:
        return bench.Bench(self.twin, Rig.load(rig_path), RobotKinematics(self.twin.spec), True,
                           clock=self.clock, sleep=self.sleep, may_write=True, rig_path=rig_path)

    def session(self) -> ArmSession:
        return ArmSession(self.twin, RobotKinematics(self.twin.spec), Rig.for_sim(), clock=self.clock,
                          sleep=self.sleep)

    def hover(self, z: float = START_Z):
        return solve_hover(self.kin, np.array([TEACH_XY[0], TEACH_XY[1], z - HOVER]), 0.0,
                           limits=self.twin.joint_limits())

    def close(self) -> None:
        close_twin(self.twin)


@pytest.fixture
def sim():
    s = SimBench()
    try:
        yield s
    finally:
        s.close()


def table_contact(twin: Twin) -> str | None:
    """Name of a robot body touching the table in the scene, else None."""
    s = twin.scene
    m, d = s.model, s.data
    table = m.geom("table").id
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if table in (g1, g2):
            name = m.body(m.geom_bodyid[g2 if g1 == table else g1]).name
            if name.startswith("robot/"):
                return name
    return None


# ------------------------------------------------------------------ the fake A1X bus, stepped
class A1XBench:
    """The twin on the fake A1X bus with a working jaw; `sleep` interleaves 5 ms bus steps and twin ticks."""

    def __init__(self, monkeypatch, *, gripper_force: float = 100.0, start_deg=None) -> None:
        pytest.importorskip("galaxeo")
        from galaxeo import protocol as P
        from galaxeo.bus import Frame
        from test_a1x import START_DEG, STEP, Clock, FakeA1XBus

        from lerobot_mp.robot.a1x import A1XArm

        class JawBus(FakeA1XBus):
            """A jaw with effort while it moves, blocked by a cube at `cube_p` when closing past it."""

            moving_effort = 0.3
            loaded_effort = 5.0

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.cube_p: float | None = None

            def tick(self, dt: float = STEP, n: int = 1) -> None:
                for _ in range(n):
                    if self.p_des is not None:
                        for j in range(6):
                            d = max(-self.rate * dt, min(self.rate * dt, self.p_des[j] - self.q[j]))
                            self.q[j] += d
                            self.v[j] = d / dt
                    else:
                        self.v[:6] = [0.0] * 6
                    effort = 0.0
                    if self.g_des is not None:
                        goal = self.g_des
                        if self.cube_p is not None and goal > self.cube_p:     # closing is towards +p_des
                            goal = max(self.cube_p, min(goal, self.q[6]))
                        step = max(-self.grip_rate * dt, min(self.grip_rate * dt, goal - self.q[6]))
                        self.q[6] += step
                        pressing = self.cube_p is not None and self.g_des > self.cube_p
                        if pressing and self.q[6] >= self.cube_p - 1e-9:
                            effort = self.loaded_effort
                        elif abs(step) > 1e-12:
                            effort = self.moving_effort
                    self.eff[6] = effort
                    self._n += 1
                    if self.silent:
                        continue
                    eff = list(self.eff)
                    eff[0] += 0.002 * (self._n % 2)
                    self.rx.append(Frame(P.FB_ID, P.encode_feedback(self.q, self.v, eff)))

        self.P, self.io = P, STEP
        self.bus_clock = Clock()
        self.bus = JawBus(list(start_deg or START_DEG))
        self.bus.tick(n=60)
        self.arm: A1XArm | None = None

        def bus_sleep(secs):
            self.bus_clock.t += secs
            self.bus.tick(secs)

        def create(cfg):
            cfg.a1x.gripper_force = gripper_force
            self.arm = A1XArm(cfg, bus=self.bus, clock=self.bus_clock, threaded=False, sleep=bus_sleep,
                              connect_timeout_s=0.05)
            return self.arm

        monkeypatch.setattr(rt, "create_backend", create)
        self.twin = Twin(Workspace(robot="a1x"))
        self.twin.connect("a1x", port="fake", threaded=False)
        self.kin = RobotKinematics(self.twin.spec)
        self.t = self.twin._clock
        self.t_io = self.t

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(float(dt), 0.0)
        while True:
            next_io = self.t_io + self.io
            next_tw = self.twin._clock + DT if self.twin.connected else math.inf
            if min(next_io, next_tw) > self.t + 1e-12:
                return
            if next_io <= next_tw:
                self.t_io = next_io
                self.bus_clock.t += self.io
                self.bus.tick(self.io)
                self.arm.io_step(self.bus_clock.t)
            else:
                self.twin.step(DT)

    def bench(self, rig_path: Path) -> bench.Bench:
        return bench.Bench(self.twin, Rig.load(rig_path), self.kin, False, clock=self.clock, sleep=self.sleep,
                           may_write=True, rig_path=rig_path)

    def command_frames(self) -> list[bytes]:
        return [d for i, d, _ in self.bus.tx if i in (self.P.CMD_ID, self.P.GRIP_ID)]

    def close(self) -> None:
        close_twin(self.twin)


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    out = tmp_path / "bench"
    monkeypatch.setattr(bench, "BENCH_DIR", str(out))
    return out


def take_jaw(a1x: A1XBench) -> dict[str, float]:
    tw = a1x.twin
    tw.claim(OWNER)
    assert tw.set_engaged(True, owner=OWNER)
    return {k: float(tw.status.measured[k]) for k in ARM}


# ================================================================== jaw
def test_gripper_log_on_the_fake_bus_samples_at_200_hz_finds_the_stream_stroke_and_tells_a_loaded_close(
        monkeypatch):
    a1x = A1XBench(monkeypatch)
    try:
        arm = take_jaw(a1x)
        free = bench.log_gripper(a1x.twin, [99.0, 100.0, 0.0, 100.0], arm_joints=arm, clock=a1x.clock,
                                 sleep=a1x.sleep)
        a1x.bus.cube_p = CUBE_P
        loaded = bench.log_gripper(a1x.twin, [0.0, 100.0], arm_joints=arm, clock=a1x.clock, sleep=a1x.sleep)
    finally:
        a1x.close()
    assert len(free) == 4 and len(loaded) == 2
    for log in free + loaded:
        assert log.rate_hz >= 150.0, log.rate_hz
        assert log.valid
        assert np.any(log.t < 0) and np.all(np.diff(log.t) > 0)
    close, opened = bench.stroke_from_log(free[2]), bench.stroke_from_log(free[3])
    assert close["delta"] < -95 and opened["delta"] > 95
    assert abs(close["stream_s"] - STREAM_STROKE_S) < 0.1, close
    assert abs(opened["stream_s"] - STREAM_STROKE_S) < 0.1, opened
    # The jaw works while it moves: the effort settles when the stroke ends.
    assert abs(close["effort_s"] - STREAM_STROKE_S) < 0.15, close
    assert bench.stroke_from_log(free[0])["stream_s"] is None               # 100 -> 99 is no stroke

    summary = bench.gripper_summary(free, loaded)
    assert summary["free_effort_peak"] == pytest.approx(0.3, abs=0.01)
    assert summary["loaded_effort_peak"] == pytest.approx(5.0, abs=0.01)
    assert summary["separable"] is True
    assert 0.3 < summary["suggested_threshold"] < 5.0
    assert "freeze_threshold" not in summary


def test_gripper_tool_on_the_fake_bus_writes_the_valid_values_and_never_the_freeze_threshold(
        monkeypatch, tmp_path, out_dir):
    a1x = A1XBench(monkeypatch)
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: a1x.bench(rig_path))
    keys = FakeKeys(on_proceed=lambda: setattr(a1x.bus, "cube_p", CUBE_P))
    try:
        code = bench.bench_gripper(ns(action="gripper", backend="a1x", i_know=True, rig=str(rig_path)), keys)
    finally:
        a1x.close()
    assert code == 0
    assert keys.prompts and "Enter" in keys.prompts[0]                        # confirmed before any motion
    g = Rig.load(rig_path).gripper
    assert g["separable"] is True
    assert g["loaded_effort_peak"] > 1.5 * g["free_effort_peak"]
    assert abs(g["stroke_close_s_effort"] - STREAM_STROKE_S) < 0.15
    assert abs(g["stroke_open_s_effort"] - STREAM_STROKE_S) < 0.15
    assert "freeze_threshold" not in g
    logs = list(out_dir.glob("gripper_*.npz"))
    assert len(logs) == 1
    data = np.load(logs[0])
    assert list(data["target"]) == [99.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
    assert data["valid"].all()


def test_gripper_tool_refuses_with_freeze_active_when_the_backend_threshold_is_1_2(monkeypatch, tmp_path,
                                                                                    capsys):
    a1x = A1XBench(monkeypatch, gripper_force=1.2)
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: a1x.bench(rig_path))
    keys = FakeKeys()
    try:
        code = bench.bench_gripper(ns(action="gripper", backend="a1x", i_know=True, rig=str(rig_path)), keys)
        assert a1x.twin.owner is None
    finally:
        a1x.close()
    assert code == 2
    assert "freeze active" in capsys.readouterr().err
    assert a1x.command_frames() == []
    assert not keys.prompts and not rig_path.exists()


def test_gripper_tool_refuses_with_arm_outside_limits_and_sends_no_frame(monkeypatch, tmp_path, capsys):
    # Folded rest: J2 stands under its 3 deg limit.
    a1x = A1XBench(monkeypatch, start_deg=(10.0, -0.2, -60.0, 5.0, 0.0, -20.0))
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: a1x.bench(rig_path))
    try:
        assert not a1x.twin.joint_limits()["arm_joint2"][0] <= a1x.twin.status.measured["arm_joint2"]
        code = bench.bench_gripper(ns(action="gripper", backend="a1x", i_know=True, rig=str(rig_path)),
                                   FakeKeys())
    finally:
        a1x.close()
    assert code == 2
    err = capsys.readouterr().err
    assert "arm outside limits" in err and "arm_joint2" in err
    assert a1x.command_frames() == []


def test_a_move_in_which_the_freeze_fired_is_invalid_and_none_of_its_numbers_are_written(monkeypatch):
    a1x = A1XBench(monkeypatch, gripper_force=1.2)
    try:
        arm = take_jaw(a1x)
        free = bench.log_gripper(a1x.twin, [99.0, 100.0, 0.0, 100.0], arm_joints=arm, clock=a1x.clock,
                                 sleep=a1x.sleep)
        a1x.bus.cube_p = CUBE_P
        loaded = bench.log_gripper(a1x.twin, [0.0, 100.0], arm_joints=arm, clock=a1x.clock, sleep=a1x.sleep)
    finally:
        a1x.close()
    assert all(log.valid for log in free)                                    # 0.3 while moving: no freeze
    assert not loaded[0].valid and loaded[0].frozen.any()
    # The open after it starts with the flag still set and clears it: the freeze did not fire there.
    assert loaded[1].frozen[loaded[1].t < 0].all() and loaded[1].valid
    summary = bench.gripper_summary(free, loaded)
    assert "loaded_effort_peak" not in summary and "separable" not in summary
    assert "suggested_threshold" not in summary
    # A frozen free close cuts the stroke short: its stroke is not written either.
    short = bench.EffortLog(t=free[2].t, effort=free[2].effort, streamed=free[2].streamed,
                            commanded=free[2].commanded, frozen=np.ones_like(free[2].frozen), rate_hz=200.0,
                            valid=False, target=0.0, operator_s=0.4)
    summary = bench.gripper_summary([free[0], free[1], short, free[3]], [])
    assert "stroke_close_s_effort" not in summary and "stroke_close_s_operator" not in summary
    assert "free_effort_peak" not in summary
    assert "stroke_open_s_effort" in summary


def test_the_gripper_tool_writes_nothing_of_a_close_in_which_the_freeze_fired(monkeypatch, tmp_path, out_dir,
                                                                             capsys):
    a1x = A1XBench(monkeypatch)
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: a1x.bench(rig_path))

    def cube_and_threshold() -> None:
        # The tool starts only with the freeze off; the effort field cannot reach 100, so the freeze
        # fires in a measurement only when the backend's threshold changes under the running tool.
        a1x.bus.cube_p = CUBE_P
        a1x.arm.acfg.gripper_force = 1.2
    try:
        code = bench.bench_gripper(ns(action="gripper", backend="a1x", i_know=True, rig=str(rig_path)),
                                   FakeKeys(on_proceed=cube_and_threshold))
    finally:
        a1x.close()
    assert code == 0
    data = np.load(next(out_dir.glob("gripper_*.npz")))
    assert list(data["valid"]) == [True] * 6 + [False, True]
    assert data["frozen_6"].any()
    g = Rig.load(rig_path).gripper
    assert not {"loaded_effort_peak", "separable", "suggested_threshold", "freeze_threshold"} & set(g)
    assert abs(g["stroke_close_s_effort"] - STREAM_STROKE_S) < 0.15          # the free moves still count
    assert g["free_effort_peak"] == pytest.approx(0.3, abs=0.01)
    assert "NIE (zamrozenie)" in capsys.readouterr().out


def test_a_bare_enter_during_the_jaw_log_is_a_stop_through_stop_now(monkeypatch, tmp_path, out_dir):
    a1x = A1XBench(monkeypatch)
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: a1x.bench(rig_path))
    calls: list[str] = []
    real = bench.stop_now

    def spy(twin, reason):
        calls.append(reason)
        real(twin, reason)
    monkeypatch.setattr(bench, "stop_now", spy)
    keys = FakeKeys([""])
    try:
        code = bench.bench_gripper(ns(action="gripper", backend="a1x", i_know=True, rig=str(rig_path)), keys)
        assert a1x.twin.owner is None
    finally:
        a1x.close()
    assert code == 1 and keys.stop.is_set()
    assert calls == [bench.STOP_REASON]
    assert not rig_path.exists()


def test_stroke_of_a_synthetic_close_and_the_operators_time_go_into_the_summary():
    t = np.arange(-0.5, 2.0, 0.005)
    streamed = np.clip(100.0 - np.maximum(t - 0.02, 0.0) * 100.0 / 1.3, 0.0, 100.0)
    effort = np.where((t > 0.02) & (streamed > 0.0), 0.8, 0.01)
    log = bench.EffortLog(t=t, effort=effort, streamed=streamed, commanded=np.where(t >= 0, 0.0, 100.0),
                          frozen=np.zeros(t.shape, bool), rate_hz=200.0, valid=True, target=0.0,
                          operator_s=1.45)
    s = bench.stroke_from_log(log)
    assert s["stream_s"] == pytest.approx(1.3, abs=0.01)
    assert s["effort_s"] == pytest.approx(1.32, abs=0.01)
    assert s["peak"] == pytest.approx(0.8)
    summary = bench.gripper_summary([log], [])
    assert summary["stroke_close_s_operator"] == pytest.approx(1.45)
    assert summary["stroke_close_s_effort"] == pytest.approx(1.32, abs=0.01)


def test_gap_readings_and_the_freeze_decision_are_recorded_without_moving(tmp_path, capsys):
    rig_path = tmp_path / "grasp.json"
    a = ns(action="gripper", rig=str(rig_path), gap_mm=["0:3", "50:60"], set_freeze=100.0)
    assert bench.bench_gripper(a, FakeKeys()) == 0
    g = Rig.load(rig_path).gripper
    assert g["gap_mm"] == {"0": 3.0, "50": 60.0}
    assert g["freeze_threshold"] == 100.0
    assert "roznica +7.0 mm" in capsys.readouterr().out                    # 60 mm against 53 mm modelled
    assert bench.bench_gripper(ns(action="gripper", rig=str(rig_path), gap_mm=["50"]), FakeKeys()) == 2
    assert bench.bench_gripper(ns(action="gripper", rig=str(rig_path), gap_mm=["120:3"]), FakeKeys()) == 2


def test_a_moving_tool_on_a_real_backend_needs_i_know_before_anything_connects(monkeypatch):
    monkeypatch.setattr(rt.Twin, "connect", lambda *a, **k: pytest.fail("connected without --i-know"))
    assert bench._open(ns(action="table", backend="a1x", i_know=False), moves=True) == 2


# ================================================================== teaching
def test_jog_steps_down_in_1_mm_until_contact_and_the_table_touch_it_saves_is_the_measured_height(sim,
                                                                                                  tmp_path):
    first: dict[str, float] = {}
    lowest = [math.inf]

    def watch() -> None:
        tw = sim.twin
        lowest[0] = min(lowest[0], float(sim.kin.tcp(tw.status.command)[2, 3]))
        if not first and table_contact(tw) is not None:
            first["measured"] = float(sim.kin.tcp(tw.status.measured)[2, 3])
            first["command"] = float(sim.kin.tcp(tw.status.command)[2, 3])

    session = sim.session()
    jog = bench.Jog(session)
    assert jog.start(sim.hover()) is None, session.detail
    sim.hooks.append(watch)
    # The steps' own targets, up to the contact: the hold and the backoff start at the measured pose.
    sent = [math.inf]
    stepping = [True]
    command, contact = session.command, jog._contact

    def spy(target) -> None:
        if stepping[0]:
            sent[0] = min(sent[0], float(sim.kin.tcp(target)[2, 3]))
        command(target)

    def on_contact() -> str:
        stepping[0] = False
        return contact()
    session.command, jog._contact = spy, on_contact
    while jog.p[2] > 0.0605:
        assert jog.step(dz=-0.005) is None, jog.note
        assert not jog.clipped
    assert not first, "the fingers touched the table above 6 cm"
    results = []
    for _ in range(60):
        before = float(jog.p[2])
        results.append(jog.step(dz=-0.005))
        if results[-1] == "contact":
            break
        assert jog.clipped and before - jog.p[2] == pytest.approx(0.001, abs=1e-9)   # 1 mm below 6 cm
    assert results[-1] == "contact", results
    assert "wysokosc" in jog.note or "opoznienie" in jog.note or "straznik" in jog.note
    assert first, "contact reported, but no finger touched the table"
    assert lowest[0] >= first["measured"] - 0.003, (lowest[0], first)
    assert sent[0] >= first["command"] - 0.003, (sent[0], first)
    assert abs(jog.contact_z - TOUCH_EXPECTED_Z) < 0.002, jog.contact_z
    assert table_contact(sim.twin) is None                                  # backed off: the tips are free

    rig_path = tmp_path / "grasp.json"
    code = bench.teach_lines(jog, FakeKeys(["d 1", "d 1", "ok"]), "table", rig_path=rig_path)
    assert code == 0
    measured, _yaw, _joints = jog.measured_tcp()
    rig = Rig.load(rig_path)
    touch = rig.touch_tcp_z()
    assert abs(measured[2] - TOUCH_EXPECTED_Z) < 0.002, measured
    assert touch == pytest.approx(measured[2], abs=3e-4)
    assert lowest[0] >= first["measured"] - 0.003
    # The measured TCP is stored; the twin's arm sags about 7 mm towards its base at table height.
    assert rig.table["tcp"][:2] == pytest.approx(list(TEACH_XY), abs=0.01)
    assert rig.table["tcp"] == pytest.approx(list(measured), abs=3e-4)
    assert datetime.fromisoformat(rig.table["taught_at"]).utcoffset().total_seconds() == 0
    assert set(rig.table["joints"]) >= set(ARM)
    jog.close()
    assert sim.twin.owner is None


def test_an_impact_during_a_step_holds_the_measured_pose_and_backs_off_2_mm(sim):
    session = sim.session()
    jog = bench.Jog(session)
    assert jog.start(sim.hover()) is None, session.detail
    pending = [True]

    def hit() -> None:
        if pending[0]:
            pending[0] = False
            session.interrupt("impact")                                    # what the guard does on a hit
    sim.hooks.append(hit)
    assert jog.step(dx=0.005) == "contact"
    assert "impact" in jog.note
    assert session.cancel.is_set() is False and not pending[0]
    measured, _yaw, _joints = jog.measured_tcp()
    assert measured[2] - jog.contact_z == pytest.approx(bench.JOG_BACKOFF, abs=6e-4)
    assert jog.step(dz=0.001) is None                                       # the jog goes on from there
    jog.close()


def test_a_blocked_rotation_ends_with_contact_by_the_lag_rule_and_backs_off_at_the_yaw_the_tool_holds(
        sim, monkeypatch):
    session = sim.session()
    jog = bench.Jog(session)
    assert jog.start(sim.hover()) is None, session.detail
    yaw0 = jog.measured_tcp()[1]
    # Joint 6 blocked in the scene: the rotation cannot happen, whatever the command.
    m = sim.twin.scene.model
    j6 = m.joint(sc.PREFIX + "arm_joint6").id
    q6 = float(sim.twin.scene.data.qpos[m.jnt_qposadr[j6]])
    m.jnt_range[j6] = (q6 - 0.004, q6 + 0.004)
    m.jnt_limited[j6] = 1
    # Without the impact guard the lag rule is the one that sees it (the guard's 3 deg fire first).
    monkeypatch.setattr(session, "guard", lambda **kw: None)
    assert jog.step(dyaw=math.radians(5.0)) == "contact"
    assert "opoznienie" in jog.note and "arm_joint6" in jog.note and "cofniecie" not in jog.note
    measured, yaw, _joints = jog.measured_tcp()
    assert abs(math.degrees(wrap(yaw - yaw0))) < 1.0
    assert abs(math.degrees(wrap(jog.yaw - yaw))) < 0.5
    assert measured[2] - jog.contact_z == pytest.approx(bench.JOG_BACKOFF, abs=6e-4)
    jog.close()


def test_with_the_impact_guard_a_blocked_rotation_still_backs_off_cleanly(sim):
    session = sim.session()
    jog = bench.Jog(session)
    assert jog.start(sim.hover()) is None, session.detail
    m = sim.twin.scene.model
    j6 = m.joint(sc.PREFIX + "arm_joint6").id
    q6 = float(sim.twin.scene.data.qpos[m.jnt_qposadr[j6]])
    m.jnt_range[j6] = (q6 - 0.004, q6 + 0.004)
    m.jnt_limited[j6] = 1
    for dyaw in (5.0, 5.0, -5.0):
        assert jog.step(dyaw=math.radians(dyaw)) == "contact"
        assert "straznik" in jog.note and "cofniecie" not in jog.note, jog.note
        assert not session.cancel.is_set()
    assert jog.step(dz=0.002) is None                                       # a translation still goes
    jog.close()
    assert sim.twin.owner is None


def teach(sim: SimBench, monkeypatch, tmp_path: Path, what: str, lines, **kw):
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: sim.bench(rig_path))
    keys = FakeKeys(lines)
    code = bench.bench_teach(ns(action=what, rig=str(rig_path), xy=list(TEACH_XY), **kw), keys, what)
    return code, rig_path, keys


def test_a_table_touch_20_mm_off_is_not_saved_without_force(sim, monkeypatch, tmp_path, capsys):
    # From 0.10 m eight 5 mm steps leave the measured TCP about 21 mm above the expected touch.
    code, rig_path, _ = teach(sim, monkeypatch, tmp_path, "table", ["d 5"] * 8 + ["ok", "q"])
    assert code == 2
    out = capsys.readouterr()
    assert "Nie zapisano" in out.err
    assert Rig.load(rig_path).touch_tcp_z() is None


def test_force_saves_a_table_touch_20_mm_off_and_notes_the_override(sim, monkeypatch, tmp_path):
    code, rig_path, _ = teach(sim, monkeypatch, tmp_path, "table", ["d 5"] * 8 + ["ok"], force=True)
    assert code == 0
    rig = Rig.load(rig_path)
    assert 0.015 < rig.touch_tcp_z() - TOUCH_EXPECTED_Z < 0.03
    assert "--force" in rig.overrides["table"]
    assert "table touch implausible" not in rig.missing("dry", simulated=False)


def test_point_stores_the_object_height_and_the_joint_offsets_of_the_running_backend(sim, monkeypatch,
                                                                                     tmp_path, capsys):
    rig_path = tmp_path / "grasp.json"
    Rig(path=rig_path, table={"touch_tcp_z": TOUCH_EXPECTED_Z, "tcp": [0.0, 0.33, TOUCH_EXPECTED_Z]}).save()
    offsets = (0.0, 0.0, 1.49, 0.0, 0.0, 0.0)
    sim.twin.backend.acfg = SimpleNamespace(joint_offsets_deg=offsets)
    code, rig_path, _ = teach(sim, monkeypatch, tmp_path, "point", ["d 5", "ok"], object_height=0.025)
    assert code == 0
    rig = Rig.load(rig_path)
    g = rig.grasp
    assert g["object_height"] == 0.025
    assert g["joint_offsets_deg"] == list(offsets)
    assert g["tcp"][:2] == pytest.approx(list(TEACH_XY), abs=0.006)
    assert math.isfinite(g["yaw"])
    datetime.fromisoformat(g["taught_at"])
    assert rig.believed()[2] == pytest.approx(TOUCH_EXPECTED_Z + TIP_CLEAR)
    assert rig.table["touch_tcp_z"] == TOUCH_EXPECTED_Z                    # the other section is kept
    out = capsys.readouterr().out
    assert "Chwyt dla polityki" in out and "obszarze treningu" in out


def test_point_on_the_sim_backend_stores_zero_joint_offsets(sim, monkeypatch, tmp_path):
    code, rig_path, _ = teach(sim, monkeypatch, tmp_path, "point", ["ok"])
    assert code == 0
    g = Rig.load(rig_path).grasp
    assert g["joint_offsets_deg"] == [0.0] * 6 and g["object_height"] == 0.030


def test_a_bare_enter_while_jogging_is_a_stop_through_stop_now(sim, monkeypatch, tmp_path):
    calls: list[str] = []
    real = bench.stop_now

    def spy(twin, reason):
        calls.append(reason)
        real(twin, reason)
    monkeypatch.setattr(bench, "stop_now", spy)
    code, rig_path, keys = teach(sim, monkeypatch, tmp_path, "table", ["d 5", "", "ok"])
    assert code == 1 and keys.stop.is_set()
    assert calls == [bench.STOP_REASON]
    assert not rig_path.exists()


def test_ctrl_c_while_jogging_is_a_stop_through_stop_now(sim, monkeypatch, tmp_path):
    calls: list[str] = []
    real = bench.stop_now

    def spy(twin, reason):
        calls.append(reason)
        real(twin, reason)
    monkeypatch.setattr(bench, "stop_now", spy)
    code, _, _ = teach(sim, monkeypatch, tmp_path, "table", ["d 5", KeyboardInterrupt()])
    assert code == 1
    assert calls == [bench.CTRL_C_REASON]


# ================================================================== latency
@pytest.mark.render
def test_the_latency_tool_on_the_sim_backend_stores_finite_numbers(sim, monkeypatch, tmp_path):
    if CKPT is None:
        monkeypatch.setattr(bench, "_checkpoint", lambda a: Path("student.pt"))
        monkeypatch.setattr(bench, "_inference_p95_s", lambda a: 0.012)
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: sim.bench(rig_path))
    code = bench.bench_latency(ns(action="latency", rig=str(rig_path), xy=list(TEACH_XY)), FakeKeys())
    assert code == 0
    lat = Rig.load(rig_path).latency
    for key in ("camera_s", "actuation_s", "infer_p95_s", "total_s"):
        assert math.isfinite(lat[key]) and lat[key] >= 0.0, (key, lat)
    parts = lat["camera_s"] + lat["infer_p95_s"] + lat["actuation_s"]
    assert lat["total_s"] == pytest.approx(parts, abs=1e-3)
    assert lat["actuation_s"] < 0.2
    # The sim stamps a frame before rendering it: the stamp is the exposure, so no camera latency.
    assert lat["camera_s"] <= DT + 1e-9


def test_a_silent_wrist_camera_ends_the_latency_tool_before_the_arm_is_taken(sim, monkeypatch, tmp_path,
                                                                             capsys):
    class Silent:
        closed = False

        def latest(self):
            return None

        def close(self) -> None:
            Silent.closed = True
    monkeypatch.setattr(bench, "_checkpoint", lambda a: Path("student.pt"))
    monkeypatch.setattr(bench, "_inference_p95_s", lambda a: pytest.fail("timed the policy without a camera"))
    monkeypatch.setattr(bench, "_open_frames", lambda b, a: Silent())
    monkeypatch.setattr(ArmSession, "take", lambda self: pytest.fail("took the arm without a camera"))
    rig_path = tmp_path / "grasp.json"
    monkeypatch.setattr(bench, "_open", lambda a, moves: sim.bench(rig_path))
    code = bench.bench_latency(ns(action="latency", rig=str(rig_path), xy=list(TEACH_XY)), FakeKeys())
    assert code == 1
    assert "wrist camera silent" in capsys.readouterr().err
    assert Silent.closed and not rig_path.exists()


# ================================================================== dot and show
def test_dot_records_the_operators_error_and_refuses_without_a_value(tmp_path, capsys):
    rig_path = tmp_path / "grasp.json"
    assert bench.bench_dot(ns(action="dot", rig=str(rig_path))) == 2
    assert bench.bench_dot(ns(action="dot", rig=str(rig_path), error_px=-1.0)) == 2
    assert not rig_path.exists()
    assert bench.bench_dot(ns(action="dot", rig=str(rig_path), error_px=3.5)) == 0
    assert Rig.load(rig_path).wrist["dot_error_px"] == 3.5
    assert bench.bench_dot(ns(action="dot", rig=str(rig_path), error_px=9.0)) == 0
    assert "dot off" in capsys.readouterr().out


def test_show_lists_what_each_mode_misses(tmp_path, capsys):
    rig_path = tmp_path / "grasp.json"
    Rig(path=rig_path, table={"touch_tcp_z": 0.034, "tcp": [0.0, 0.33, 0.034]},
        wrist={"source": "0"}).save()
    assert bench.bench_show(ns(action="show", rig=str(rig_path))) == 0
    out = capsys.readouterr().out
    rig = Rig.load(rig_path)
    for mode in ("dry", "shadow", "real"):
        missing = rig.missing(mode, simulated=False, now=datetime.now().timestamp())
        assert missing
        assert f"{mode:<7s} " + ", ".join(missing) in out
    assert "grasp point not taught" in out and "no shadow episodes" in out
    assert "teaching old" in out                                            # a touch without its stamp
    assert bench.bench_show(ns(action="show", rig=str(tmp_path / "none.json"))) == 0
    assert "table not taught" in capsys.readouterr().out


def test_the_latency_tool_without_a_checkpoint_refuses_before_connecting(monkeypatch, tmp_path, capsys):
    from lerobot_mp.twin.grasp import policy

    monkeypatch.delenv(policy.CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(policy, "DEFAULT_CHECKPOINT", tmp_path / "absent" / "student.pt")
    monkeypatch.setattr(bench, "_open", lambda a, moves: pytest.fail("connected without a checkpoint"))
    code = bench.bench_latency(ns(action="latency", checkpoint=str(tmp_path / "missing.pt")), FakeKeys())
    assert code == 2
    assert "grasp-model fetch" in capsys.readouterr().err


def test_main_bench_refuses_an_unknown_tool():
    assert bench.main_bench(ns(action="nothing")) == 2
