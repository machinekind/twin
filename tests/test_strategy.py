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
from dataclasses import dataclass, field
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

TWIN_SOURCES = Path(__file__).resolve().parents[1] / "src" / "lerobot_mp" / "twin"

#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02


@pytest.fixture
def twin():
    twin = Twin(Workspace(), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def drive(twin: Twin, runner: StrategyRunner, seconds: float) -> None:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock."""
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)

    for k in range(1, round(seconds / CLOCK_STEP) + 1):
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(k * CLOCK_STEP)


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


def answer(target):
    """A plugin answer that always returns the same target."""
    return lambda obs: target


# --------------------------------------------------------------------------- the test strategy


def test_the_test_strategy_drives_the_simulated_twin_through_its_poses(twin):
    twin.connect("sim", threaded=False)
    home = dict(twin.status.measured)
    first = dict(home, shoulder_pan=20.0)
    last = dict(home, shoulder_pan=-15.0, elbow_flex=30.0)

    runner = StrategyRunner(twin, Waypoints([first, last]))
    runner.start(threaded=False)
    assert twin.owner == "strategia"

    drive(twin, runner, 5.0)

    assert not runner.status.running
    assert runner.status.done
    assert runner.status.success
    assert runner.status.stopped_because == "zadanie wykonane"
    assert twin.owner is None
    assert not twin._engaged
    for joint, want in last.items():
        assert twin.status.measured[joint] == pytest.approx(want, abs=1.0)


def test_the_same_strategy_drives_hardware_through_the_supervisor(twin, monkeypatch):
    """Same code path on a (fake) real arm: every command is the supervisor's, rate-limited."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    target = dict(twin.status.measured, shoulder_pan=40.0)

    runner = StrategyRunner(twin, Waypoints([target], tolerance=0.5))
    runner.start(threaded=False)
    drive(twin, runner, 4.0)

    assert runner.status.success

    pans = [sent["shoulder_pan"] for sent in arm.sent if "shoulder_pan" in sent]
    one_tick = twin.workspace.spec().max_vel["shoulder_pan"] * TWIN_PERIOD
    assert pans[-1] == pytest.approx(40.0, abs=0.5)
    assert np.abs(np.diff(pans)).max() <= one_tick * 1.05


def test_a_waypoint_on_a_joint_the_robot_does_not_have_refuses_the_run(twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Waypoints([{"arm_joint1": 10.0}]))

    with pytest.raises(ValueError, match="arm_joint1"):
        runner.start(threaded=False)

    assert twin.owner is None
    assert not runner.status.running


# --------------------------------------------------------------------------- plugin strategies


@dataclass
class PluginAction:
    """A plugin's own action type: not the twin's `Action`, but the same attributes."""

    target: dict = field(default_factory=dict)
    done: bool = False
    success: bool = False
    error: float = 0.1
    note: str = "koniec"


def test_a_plugin_strategy_returns_plain_values_and_sees_the_whole_observation(twin):
    twin.connect("sim", threaded=False)
    cube = ObjectPose(np.array([0.2, 0.0, 0.02]), np.eye(3), source="symulacja")

    def three_steps_then_done(obs):
        if obs.step < 3:
            return {"shoulder_pan": 5.0}
        return PluginAction(done=True)

    plugin = Plugin(three_steps_then_done)
    runner = StrategyRunner(twin, plugin, goal=lambda: (0.25, 0.05, 0.1), objects=lambda: {"cube": cube})
    runner.start(threaded=False)
    drive(twin, runner, 1.0)

    spec = twin.workspace.spec()
    run = plugin.started
    assert run.robot == "so101"
    assert run.simulated
    assert run.joints == spec.joints
    assert run.folder == spec.root
    assert run.max_vel == spec.max_vel

    obs = plugin.seen[-1]
    assert obs.step == 3
    assert obs.t == pytest.approx(0.15)
    assert np.allclose(obs.goal, (0.25, 0.05, 0.1))
    assert obs.objects["cube"].source == "symulacja"
    assert set(obs.joints) == set(run.joints)
    assert set(obs.command) == set(run.joints)

    status = runner.status
    assert status.done
    assert status.success is False
    assert status.error == 0.1
    assert status.note == "koniec"
    assert status.stopped_because == "strategia skonczyla"
    assert plugin.stopped == "strategia skonczyla"


def test_camera_frames_are_rendered_only_when_the_strategy_reads_them(twin):
    twin.connect("sim", threaded=False)
    rendered = []

    def render(camera):
        rendered.append(camera)
        return np.zeros((4, 4, 3), np.uint8)

    blind = StrategyRunner(twin, Plugin(answer({})), cameras=("a", "b"), frame=render)
    blind.start(threaded=False)
    drive(twin, blind, 0.5)
    blind.stop()

    assert rendered == []

    def look_at_a_twice(obs):
        assert set(obs.frames) == {"a", "b"}
        obs.frames["a"]
        obs.frames["a"]
        return {}

    seeing = StrategyRunner(twin, Plugin(look_at_a_twice), cameras=("a", "b"), frame=render)
    seeing.start(threaded=False)
    drive(twin, seeing, 0.5)

    assert rendered == ["a"] * seeing.status.step


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)

    return modules


@pytest.mark.parametrize("source", ["strategy.py", "waypoints.py"])
def test_the_runner_and_the_test_strategy_import_nothing_from_rl(source):
    for module in imported_modules(TWIN_SOURCES / source):
        assert module != "rl"
        assert not module.startswith("rl.")
        assert ".rl" not in module


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize("bad_target, named", [
    ({"shoulder_pan": math.nan}, "shoulder_pan"),
    ({"gripper": math.inf}, "gripper"),
    ({"arm_joint1": 10.0}, "arm_joint1"),
    ({"shoulder_pan": True}, "nie liczba"),
    ({"shoulder_pan": "5"}, "nie liczba"),
])
def test_a_bad_target_stops_the_run_and_holds_the_measured_pose(twin, monkeypatch, bad_target, named):
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    held = []
    monkeypatch.setattr(twin, "hold_measured", lambda *args, **kwargs: held.append(True))

    runner = StrategyRunner(twin, Plugin(answer(bad_target)))
    runner.start(threaded=False)
    drive(twin, runner, 0.5)

    assert not runner.status.running
    assert runner.status.stopped_because.startswith("zly cel strategii")
    assert named in runner.status.stopped_because
    assert held == [True]
    assert all(math.isfinite(value) for sent in arm.sent for value in sent.values())
    assert twin.owner is None


def test_home_takes_the_arm_and_the_strategy_hears_why(twin):
    twin.connect("sim", threaded=False)
    plugin = Plugin(answer({"shoulder_pan": 30.0}))
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
    runner = StrategyRunner(twin, Plugin(answer({})), time_limit_s=0.5)
    runner.start(threaded=False)

    drive(twin, runner, 1.0)

    assert runner.status.stopped_because == "limit czasu"
    assert runner.status.step == 10


def test_a_strategy_that_refuses_to_start_leaves_the_arm_alone(twin):
    twin.connect("sim", threaded=False)

    class TrainedOnAnotherRobot(Plugin):
        def start(self, run, obs):
            raise RuntimeError(f"uczona na a1x, stanowisko ma {run.robot}")

    runner = StrategyRunner(twin, TrainedOnAnotherRobot(answer({"shoulder_pan": 30.0})))

    with pytest.raises(RuntimeError, match="uczona na a1x"):
        runner.start(threaded=False)

    assert twin.owner is None
    assert not runner.status.running


def test_a_disconnected_arm_ends_the_run(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    runner = StrategyRunner(twin, Plugin(answer({})))
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
    runner = StrategyRunner(twin, Plugin(answer(Action(target={"shoulder_pan": 3.0}, note="jade"))))
    runner.start(threaded=False)
    drive(twin, runner, 0.3)

    assert runner.status.running
    assert runner.status.note == "jade"

    runner.stop()

    assert runner.status.stopped_because == "zatrzymana"
    assert twin.owner is None


# --------------------------------------------------------------------------- races the review found


def test_a_slow_strategy_that_outlives_stop_cannot_drive_the_next_run(twin, monkeypatch):
    """An `act` slower than the join (VLA inference) used to come back and send its stale target,
    under the same owner name, into the run started after it."""
    twin.connect("sim", threaded=False)
    sent = []
    send = twin.set_target

    def recording_send(target, owner=None):
        sent.append(dict(target))
        return send(target, owner)

    monkeypatch.setattr(twin, "set_target", recording_send)

    inside_act, let_act_return = threading.Event(), threading.Event()

    def slow(obs):
        inside_act.set()
        let_act_return.wait(5.0)
        return {"shoulder_pan": 40.0}

    runner = StrategyRunner(twin, Plugin(slow))
    runner.join_timeout_s = 0.05
    runner.start()
    assert inside_act.wait(5.0)

    runner.stop()
    assert not runner.status.running

    # The same runner again, while the old thread still waits inside `act`.
    runner.strategy.answer = answer({})
    runner.start()
    let_act_return.set()

    deadline = time.monotonic() + 1.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()

    assert sent == []
    assert runner.status.stopped_because == "zatrzymana"


def test_the_arm_taken_during_a_tick_reports_the_preempt_not_a_crash(twin, monkeypatch, caplog):
    twin.connect("sim", threaded=False)
    plugin = Plugin(answer({"shoulder_pan": 5.0}))
    runner = StrategyRunner(twin, plugin)
    runner.start(threaded=False)

    def taken_while_sending(target, owner=None):
        twin.preempt("Dom")
        raise RuntimeError("ramie odebrane (Dom)")

    monkeypatch.setattr(twin, "set_target", taken_while_sending)
    with caplog.at_level(logging.ERROR, logger="lerobot_mp.twin.strategy"):
        assert not runner.step_once(0.0)

    assert runner.status.stopped_because == "przerwana: Dom"
    assert plugin.stopped == "przerwana: Dom"
    assert not caplog.records


def test_the_strategy_hears_stop_once(twin):
    twin.connect("sim", threaded=False)
    reasons = []

    class Counting(Plugin):
        def stop(self, reason):
            reasons.append(reason)

    runner = StrategyRunner(twin, Counting(answer(Action(done=True, success=True))))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)

    runner.stop()
    runner.stop("jeszcze raz")

    assert reasons == ["zadanie wykonane"]


def test_the_arm_taken_while_the_strategy_starts_still_gets_it_a_stop(twin):
    twin.connect("sim", threaded=False)

    class HomePressedDuringStart(Plugin):
        def start(self, run, obs):
            twin.home()

    plugin = HomePressedDuringStart(answer({"shoulder_pan": 30.0}))
    runner = StrategyRunner(twin, plugin)

    with pytest.raises(RuntimeError, match="ramie odebrane"):
        runner.start(threaded=False)

    assert plugin.stopped.startswith("przerwana")
    assert twin.owner is None
    assert not runner.status.running


def test_cameras_on_hardware_need_a_frame_source(twin, monkeypatch):
    connect_fake(twin, monkeypatch, FakeArm())
    runner = StrategyRunner(twin, Plugin(answer({})), cameras=("a",))

    with pytest.raises(ValueError, match="frame"):
        runner.start(threaded=False)

    assert twin.owner is None
