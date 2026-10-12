"""The strategy contract: `StrategyRunner` with the test strategy and with plugin-style strategies.

Everything in simulated time: the twin's loop without a thread (`connect(threaded=False)` +
`Twin.step`) and the runner without a thread (`start(threaded=False)` + `step_once`).
"""

from __future__ import annotations

import ast
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.strategy import Action, ObjectPose, StrategyRunner  # noqa: E402
from lerobot_mp.twin.waypoints import Waypoints  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

from test_twin_runtime import FakeArm, connect_fake  # noqa: E402

TWIN = Path(__file__).resolve().parents[1] / "src" / "lerobot_mp" / "twin"


@pytest.fixture
def twin():
    tw = Twin(Workspace(), clock=VirtualClock(t0=0.0))
    try:
        yield tw
    finally:
        tw.close()


def drive(twin, runner, seconds: float, t0: float = 0.0) -> float:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock (10 ms steps)."""
    every = max(1, round(1.0 / runner.strategy.hz / 0.01))
    n = int(round(seconds / 0.01))
    for k in range(1, n + 1):
        if k % 2 == 0:
            twin.step(0.02)
        if k % every == 0 and runner.status.running:
            runner.step_once(t0 + k * 0.01)
    return t0 + n * 0.01


class Plugin:
    """A strategy as another package writes it: no twin types, plain values out."""

    hz = 20.0

    def __init__(self, answer):
        self.answer = answer
        self.seen = []
        self.started = None
        self.stopped = None

    def start(self, run, obs):
        self.started = run

    def act(self, obs):
        self.seen.append(obs)
        return self.answer(obs)

    def stop(self, reason):
        self.stopped = reason


# ------------------------------------------------------------ the test strategy
def test_the_test_strategy_drives_the_simulated_twin_through_its_poses(twin):
    twin.connect("sim", threaded=False)
    home = dict(twin.status.measured)
    a, b = dict(home, shoulder_pan=20.0), dict(home, shoulder_pan=-15.0, elbow_flex=30.0)
    runner = StrategyRunner(twin, Waypoints([a, b]))
    runner.start(threaded=False)
    assert twin.owner == "strategia"
    drive(twin, runner, 5.0)

    st = runner.status
    assert (st.running, st.done, st.success, st.stopped_because) == (False, True, True, "zadanie wykonane")
    assert twin.owner is None and not twin._engaged                     # the arm is given back
    for name, want in b.items():
        assert twin.status.measured[name] == pytest.approx(want, abs=1.0)


def test_the_same_strategy_drives_hardware_through_the_supervisor(twin, monkeypatch):
    """Same code path on a (fake) real arm: every command is the supervisor's, rate-limited."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    target = dict(twin.status.measured, shoulder_pan=40.0)
    runner = StrategyRunner(twin, Waypoints([target], tolerance=0.5))
    runner.start(threaded=False)
    drive(twin, runner, 4.0)

    assert runner.status.success
    pans = [s["shoulder_pan"] for s in arm.sent if "shoulder_pan" in s]
    assert pans[-1] == pytest.approx(40.0, abs=0.5)
    max_step = twin.workspace.spec().max_vel["shoulder_pan"] * 0.02         # one 20 ms tick of the supervisor
    assert np.abs(np.diff(pans)).max() <= max_step * 1.05 + 1e-6


def test_a_waypoint_on_a_joint_the_robot_does_not_have_refuses_the_run(twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Waypoints([{"arm_joint1": 10.0}]))
    with pytest.raises(ValueError, match="arm_joint1"):
        runner.start(threaded=False)
    assert twin.owner is None and not runner.status.running


# ------------------------------------------------------------ plugin strategies
def test_a_plugin_strategy_returns_plain_values_and_sees_the_whole_observation(twin):
    twin.connect("sim", threaded=False)
    cube = ObjectPose(np.array([0.2, 0.0, 0.02]), np.eye(3), source="symulacja")

    class Result:                                   # the plugin's own action type, same attributes
        def __init__(self, target, done=False):
            self.target, self.done, self.success, self.error, self.note = target, done, False, 0.1, "koniec"

    def answer(obs):
        if obs.step < 3:
            return {"shoulder_pan": 5.0}             # a plain mapping is a target
        return Result({}, done=True)

    plugin = Plugin(answer)
    runner = StrategyRunner(twin, plugin, goal=lambda: (0.25, 0.05, 0.1), objects=lambda: {"cube": cube})
    runner.start(threaded=False)
    drive(twin, runner, 1.0)

    spec = twin.workspace.spec()
    assert plugin.started.robot == "so101" and plugin.started.simulated
    assert plugin.started.joints == spec.joints and plugin.started.folder == spec.root
    assert plugin.started.max_vel == spec.max_vel
    obs = plugin.seen[-1]
    assert obs.step == 3 and obs.t == pytest.approx(0.15)
    assert np.allclose(obs.goal, (0.25, 0.05, 0.1)) and obs.objects["cube"].source == "symulacja"
    assert set(obs.joints) == set(obs.command) == set(plugin.started.joints)
    st = runner.status
    assert (st.done, st.success, st.error, st.note) == (True, False, 0.1, "koniec")
    assert st.stopped_because == "strategia skonczyla" and plugin.stopped == "strategia skonczyla"


def test_camera_frames_are_rendered_only_when_the_strategy_reads_them(twin):
    twin.connect("sim", threaded=False)
    calls = []

    def frame(name):
        calls.append(name)
        return np.zeros((4, 4, 3), np.uint8)

    blind = StrategyRunner(twin, Plugin(lambda obs: {}), cameras=("a", "b"), frame=frame)
    blind.start(threaded=False)
    drive(twin, blind, 0.5)
    blind.stop()
    assert calls == []

    def look(obs):
        assert set(obs.frames) == {"a", "b"}
        obs.frames["a"], obs.frames["a"]             # twice in a tick: one render
        return {}

    seeing = StrategyRunner(twin, Plugin(look), cameras=("a", "b"), frame=frame)
    seeing.start(threaded=False)
    drive(twin, seeing, 0.5)
    assert calls and set(calls) == {"a"} and len(calls) == seeing.status.step


def test_the_runner_and_the_test_strategy_import_nothing_from_rl():
    for name in ("strategy.py", "waypoints.py"):
        tree = ast.parse((TWIN / name).read_text(encoding="utf-8"))
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m == "rl" or m.startswith("rl.") or ".rl" in m for m in mods), (name, mods)


# ------------------------------------------------------------ guards
@pytest.mark.parametrize("bad, why", [
    ({"shoulder_pan": math.nan}, "shoulder_pan"),
    ({"gripper": math.inf}, "gripper"),
    ({"arm_joint1": 10.0}, "arm_joint1"),
    ({"shoulder_pan": True}, "nie liczba"),
    ({"shoulder_pan": "5"}, "nie liczba"),
])
def test_a_bad_target_stops_the_run_and_holds_the_measured_pose(twin, monkeypatch, bad, why):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    held = []
    monkeypatch.setattr(twin, "hold_measured", lambda *a, **k: held.append(True))
    runner = StrategyRunner(twin, Plugin(lambda obs: bad))
    runner.start(threaded=False)
    drive(twin, runner, 0.5)

    assert not runner.status.running and runner.status.stopped_because.startswith("zly cel strategii")
    assert why in runner.status.stopped_because and held == [True]
    assert all(math.isfinite(v) for s in arm.sent for v in s.values())
    assert twin.owner is None


def test_home_takes_the_arm_and_the_strategy_hears_why(twin):
    twin.connect("sim", threaded=False)
    plugin = Plugin(lambda obs: {"shoulder_pan": 30.0})
    runner = StrategyRunner(twin, plugin)
    runner.start(threaded=False)
    drive(twin, runner, 0.3)
    twin.home()
    assert not runner.status.running
    assert runner.status.stopped_because.startswith("przerwana")
    assert plugin.stopped == runner.status.stopped_because
    with pytest.raises(RuntimeError):
        twin.set_target({"shoulder_pan": 0.0}, owner=runner.owner)


def test_the_time_limit_ends_the_run(twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(lambda obs: {}), time_limit_s=0.5)
    runner.start(threaded=False)
    drive(twin, runner, 1.0)
    assert runner.status.stopped_because == "limit czasu" and runner.status.step == 10


def test_a_strategy_that_refuses_to_start_leaves_the_arm_alone(twin):
    twin.connect("sim", threaded=False)

    class Picky(Plugin):
        def start(self, run, obs):
            raise RuntimeError(f"uczona na a1x, stanowisko ma {run.robot}")

    runner = StrategyRunner(twin, Picky(lambda obs: {"shoulder_pan": 30.0}))
    with pytest.raises(RuntimeError, match="uczona na a1x"):
        runner.start(threaded=False)
    assert twin.owner is None and not runner.status.running


def test_a_disconnected_arm_ends_the_run(twin, monkeypatch):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    runner = StrategyRunner(twin, Plugin(lambda obs: {}))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)
    twin.disconnect()
    assert not runner.step_once(10.0)
    assert not runner.status.running


def test_an_object_without_hz_or_act_is_not_a_strategy(twin):
    class NoRate:
        def act(self, obs):
            return {}

    with pytest.raises(TypeError, match="hz"):
        StrategyRunner(twin, NoRate())
    with pytest.raises(ValueError, match="hz"):
        StrategyRunner(twin, Waypoints([{"shoulder_pan": 0.0}], hz=0.0))


def test_an_action_object_is_used_as_is(twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(lambda obs: Action(target={"shoulder_pan": 3.0}, note="jade")))
    runner.start(threaded=False)
    drive(twin, runner, 0.3)
    assert runner.status.running and runner.status.note == "jade"
    runner.stop()
    assert runner.status.stopped_because == "zatrzymana" and twin.owner is None


# ------------------------------------------------------------ races the review found
def test_a_slow_strategy_that_outlives_stop_cannot_drive_the_next_run(twin, monkeypatch):
    """`act` slower than the join (VLA inference) used to come back and send its stale target, under
    the same owner name, into the run started after it."""
    twin.connect("sim", threaded=False)
    sent = []
    real = twin.set_target
    def spy(target, owner=None):
        sent.append(dict(target))
        return real(target, owner)

    monkeypatch.setattr(twin, "set_target", spy)
    release, inside = threading.Event(), threading.Event()

    def slow(obs):
        inside.set()
        release.wait(5.0)
        return {"shoulder_pan": 40.0}

    first = StrategyRunner(twin, Plugin(slow))
    first.join_timeout_s = 0.05
    first.start()
    assert inside.wait(5.0)
    first.stop()
    assert not first.status.running

    first.strategy.answer = lambda obs: {}
    first.start()                                    # the same runner again, while the old thread still waits
    release.set()
    deadline = time.monotonic() + 2.0
    while first._thread is not None and time.monotonic() < deadline and not sent:
        time.sleep(0.01)
    time.sleep(0.2)
    first.stop()
    assert sent == [] and first.status.stopped_because == "zatrzymana"


def test_the_arm_taken_during_a_tick_reports_the_preempt_not_a_crash(twin, monkeypatch, caplog):
    twin.connect("sim", threaded=False)

    def taken(target, owner=None):
        twin.preempt("Dom")
        raise RuntimeError("ramie odebrane (Dom)")

    plugin = Plugin(lambda obs: {"shoulder_pan": 5.0})
    runner = StrategyRunner(twin, plugin)
    runner.start(threaded=False)
    monkeypatch.setattr(twin, "set_target", taken)
    with caplog.at_level(logging.ERROR, logger="lerobot_mp.twin.strategy"):
        assert not runner.step_once(0.0)
    assert runner.status.stopped_because == "przerwana: Dom" and plugin.stopped == "przerwana: Dom"
    assert not caplog.records


def test_the_strategy_hears_stop_once(twin):
    twin.connect("sim", threaded=False)
    calls = []

    class Counting(Plugin):
        def stop(self, reason):
            calls.append(reason)

    runner = StrategyRunner(twin, Counting(lambda obs: Action(done=True, success=True)))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)
    runner.stop()
    runner.stop("jeszcze raz")
    assert calls == ["zadanie wykonane"]


def test_the_arm_taken_while_the_strategy_starts_still_gets_it_a_stop(twin):
    twin.connect("sim", threaded=False)

    class HomeOnStart(Plugin):
        def start(self, run, obs):
            twin.home()

    plugin = HomeOnStart(lambda obs: {"shoulder_pan": 30.0})
    runner = StrategyRunner(twin, plugin)
    with pytest.raises(RuntimeError, match="ramie odebrane"):
        runner.start(threaded=False)
    assert plugin.stopped and plugin.stopped.startswith("przerwana")
    assert twin.owner is None and not runner.status.running


def test_cameras_on_hardware_need_a_frame_source(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    runner = StrategyRunner(twin, Plugin(lambda obs: {}), cameras=("a",))
    with pytest.raises(ValueError, match="frame"):
        runner.start(threaded=False)
    assert twin.owner is None
