"""The episode log: its writer and loader, every `StrategyRunner` run writing one, and the observation mode.

Everything in simulated time: the twin's loop without a thread (`connect(threaded=False)` +
`Twin.step`) and the runner without a thread (`start(threaded=False)` + `step_once`), as in
`test_strategy`.
"""

from __future__ import annotations

import collections
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT
from test_strategy import Plugin, answer, drive
from test_twin_runtime import FakeArm, connect_fake

from machinekind_twin.cli import main as twin_cli
from machinekind_twin.clock import VirtualClock
from machinekind_twin.episode import (
    FORMAT,
    EpisodeEnd,
    EpisodeLog,
    EpisodeOptions,
    EpisodeWriter,
    StepRecord,
    episodes_dir,
    twin_header,
)
from machinekind_twin.episode_report import DelayFit, report
from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.strategy import Action, Decision, Frame, ObjectPose, StrategyRunner
from machinekind_twin.waypoints import Waypoints
from machinekind_twin.workspace import Workspace

TWIN_PERIOD = 0.02


@pytest.fixture
def twin():
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def observer(twin: Twin, folder: Path, **kw) -> EpisodeWriter:
    """A log without a runner, as an observer (#175) opens one."""
    header = {**twin_header(twin), "mode": "observation", "strategy": None, "meta": {"seed": 3}}
    return EpisodeWriter.create(folder, header, twin, **kw)


def record(writer: EpisodeWriter, twin: Twin, step: int, **kw) -> StepRecord:
    """A step as an observer writes it: the twin's readings and command now."""
    status = twin.status
    values = {
        "step": step, "run_t": None, "obs_t": writer.now(), "t": writer.now(),
        "joints": dict(status.measured), "command": dict(status.command), "intent": {}, "goal": None,
        "objects": {}, "frames": {},
    }
    return StepRecord(**{**values, **kw})


def end(code: str = "stopped", steps: int = 0, **kw) -> EpisodeEnd:
    values = {"code": code, "reason": "zatrzymana", "t": 1.0, "done": False, "success": None, "error": None,
              "note": "", "steps": steps, "began": True}
    return EpisodeEnd(**{**values, **kw})


def run_ticks(twin: Twin, count: int) -> None:
    for _ in range(count):
        twin.step(TWIN_PERIOD)


def lines(path: Path) -> list[dict]:
    """The log's records; only "\\n" ends a line, as in `EpisodeLog.load`."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


# --------------------------------------------------------------------------- the file


def test_a_log_round_trips_through_its_loader(twin, tmp_path):
    twin.connect("sim", threaded=False)
    joints = twin.workspace.spec().joints
    writer = observer(twin, tmp_path, name="run")

    run_ticks(twin, 5)
    cube = ObjectPose(np.array([0.2, 0.0, 0.015]), np.eye(3), "kamery", 0.1)
    writer.step(record(writer, twin, 0, intent={"j1": 20.0}, goal=np.array([0.2, 0.0, 0.1]),
                       objects={"cube": cube}, error=0.5, note="poza 1/1"))
    run_ticks(twin, 3)
    writer.close(end(steps=1, error=0.5, note="poza 1/1"))

    log = EpisodeLog.load(writer.path)

    assert writer.path == tmp_path / "run.jsonl"
    assert log.header["format"] == FORMAT
    assert log.header["robot"] == BOT
    assert log.header["mode"] == "observation"
    assert log.header["meta"] == {"seed": 3}
    assert log.header["clock"] == "virtual"
    assert log.joints == tuple(joints)
    assert log.loop_t.shape == (8,)
    assert log.measured.shape == log.sent.shape == log.at_limit.shape == (8, len(joints))
    assert np.all(np.diff(log.loop_t) > 0)

    t, readings = log.readings()
    assert len(t) == 8
    assert np.isfinite(readings).all()

    step = log.steps[0]
    assert step["intent"] == {"j1": 20.0}
    assert step["goal"] == [0.2, 0.0, 0.1]
    assert step["objects"]["cube"]["source"] == "kamery"
    assert step["objects"]["cube"]["rotation"] == np.eye(3).tolist()

    intent_t, intents = log.intents()
    assert intent_t.tolist() == [step["t"]]
    assert intents[0, joints.index("j1")] == 20.0
    assert np.isnan(np.delete(intents[0], joints.index("j1"))).all()

    assert log.end.code == "stopped"
    assert log.end.steps == 1
    assert log.end.error == 0.5


def test_every_line_is_strict_json_with_constant_keys(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    image = np.zeros((480, 640, 3), np.uint8)

    run_ticks(twin, 2)
    decision = Decision(question="gdzie kostka?", answer="tu", state={"image": image, "gap": math.nan},
                        confidence=math.inf)
    writer.step(record(writer, twin, 0, error=math.nan, decisions=[decision]))
    writer.step(record(writer, twin, 1))
    writer.close(end(steps=2, error=math.nan))

    def refuse(constant):
        raise ValueError(constant)

    parsed = [json.loads(line, parse_constant=refuse) for line in writer.path.read_text().splitlines()]
    steps = [line for line in parsed if line["kind"] == "step"]

    assert steps[0]["error"] is None
    assert steps[0]["decisions"][0]["state"] == {"image": "<ndarray uint8 (480, 640, 3)>", "gap": None}
    assert steps[0]["decisions"][0]["confidence"] is None
    assert steps[1]["decisions"] == []
    assert set(steps[0]) == set(steps[1])
    assert parsed[-1]["kind"] == "end"


def test_a_cut_off_log_loads_as_unfinished(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    for step in range(3):
        run_ticks(twin, 2)
        writer.step(record(writer, twin, step, note=f"krok {step}"))
    writer.close(end(code="done", steps=3))

    text = writer.path.read_text(encoding="utf-8").splitlines()
    cut = tmp_path / "cut.jsonl"
    cut.write_text("\n".join(text[:-2] + [text[-2][: len(text[-2]) // 2]]), encoding="utf-8")

    log = EpisodeLog.load(cut)

    assert log.end.code == "unfinished"
    assert log.end.t is None
    assert len(log.steps) == 2
    assert log.end.steps == 2
    assert log.end.note == "krok 1"


def test_a_log_cut_inside_a_character_loads_as_unfinished(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    for step, note in enumerate(("chwytam kostkę", "chwytak złapał")):
        run_ticks(twin, 2)
        writer.step(record(writer, twin, step, note=note))
    writer.close(end(code="done", steps=2))

    data = writer.path.read_bytes()
    cut = tmp_path / "cut.jsonl"
    cut.write_bytes(data[: data.rindex("ł".encode()) + 1])

    log = EpisodeLog.load(cut)

    assert log.end.code == "unfinished"
    assert [step["note"] for step in log.steps] == ["chwytam kostkę"]
    assert EpisodeLog.summary(cut).end.code == "unfinished"


def test_line_separators_inside_text_stay_inside_their_line(twin, tmp_path):
    """JSON keeps U+2028, U+2029 and U+0085 in a string as they are: only "\\n" ends a line."""
    twin.connect("sim", threaded=False)
    odd = "lewo\u2028prawo\u2029dalej\x85koniec"
    writer = EpisodeWriter.create(tmp_path, {**twin_header(twin), "meta": {"ustawienia": odd}}, twin)
    run_ticks(twin, 2)
    writer.step(record(writer, twin, 0, note=odd, decisions=[Decision(question="gdzie kostka?", answer=odd)]))
    writer.close(end(code="done", steps=1, success=True, reason=odd, note=odd))

    log = EpisodeLog.load(writer.path)
    summary = EpisodeLog.summary(writer.path)

    assert log.header["meta"] == {"ustawienia": odd}
    assert log.steps[0]["note"] == odd
    assert log.steps[0]["decisions"][0]["answer"] == odd
    assert log.end.code == "done"
    assert log.end.reason == odd
    assert summary.header["meta"] == {"ustawienia": odd}
    assert summary.end == log.end


def test_a_broken_line_a_newer_version_or_another_file_are_refused(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    run_ticks(twin, 3)
    writer.close(end())
    text = writer.path.read_text(encoding="utf-8").splitlines()

    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join([text[0], "{nie json", *text[1:]]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken.jsonl:2: uszkodzona linia"):
        EpisodeLog.load(broken)

    newer = tmp_path / "newer.jsonl"
    newer.write_text("\n".join([json.dumps({**json.loads(text[0]), "version": 2}), *text[1:]]), encoding="utf-8")
    with pytest.raises(ValueError, match="w wersji 2"):
        EpisodeLog.load(newer)

    other = tmp_path / "other.jsonl"
    other.write_text('{"kind": "header", "format": "cos innego"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="to nie jest zapis przebiegu"):
        EpisodeLog.load(other)


def test_two_logs_with_one_name_get_their_own_files(twin, tmp_path):
    twin.connect("sim", threaded=False)
    first = observer(twin, tmp_path, name="same")
    second = observer(twin, tmp_path, name="same")
    first.close(end())
    second.close(end())

    assert first.path.name == "same.jsonl"
    assert second.path.name == "same-2.jsonl"
    assert EpisodeLog.load(first.path).header["name"] == "same"
    assert EpisodeLog.load(second.path).header["name"] == "same-2"


def test_a_closed_log_stops_growing(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    run_ticks(twin, 3)
    writer.close(end())
    size = writer.path.stat().st_size

    run_ticks(twin, 10)
    writer.close(end(code="error"))
    writer.flush()

    assert writer.path.stat().st_size == size
    assert EpisodeLog.load(writer.path).end.code == "stopped"


def test_the_summary_reads_the_header_and_the_end(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    for step in range(3):
        run_ticks(twin, 2)
        writer.step(record(writer, twin, step, error=0.25))
    writer.close(end(code="done", steps=3, success=True, error=0.25))

    summary = EpisodeLog.summary(writer.path)

    assert summary.header["robot"] == BOT
    assert summary.end == EpisodeLog.load(writer.path).end
    assert summary.end.succeeded

    unfinished = tmp_path / "unfinished.jsonl"
    unfinished.write_text("\n".join(writer.path.read_text(encoding="utf-8").splitlines()[:-1]), encoding="utf-8")

    assert EpisodeLog.summary(unfinished).end.code == "unfinished"
    assert EpisodeLog.summary(unfinished).end.steps == 3


def test_kept_frames_are_jpegs_next_to_the_log(twin, tmp_path):
    cv2 = pytest.importorskip("cv2")
    twin.connect("sim", threaded=False)
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :, 0] = 200

    every = observer(twin, tmp_path, name="every", keep_frames_s=0)
    every.step(record(every, twin, 0, frames={"a": Frame(image, 0.01), "b": None}))
    every.close(end(steps=1))

    frames = EpisodeLog.load(every.path).steps[0]["frames"]
    kept = tmp_path / frames[0]["file"]
    assert frames[0]["camera"] == "a"
    assert frames[0]["shape"] == [48, 64, 3]
    assert frames[1] == {"camera": "b", "age_s": None, "shape": None, "file": None}
    assert kept.parent == tmp_path / "every.frames"
    assert cv2.imread(str(kept)).shape == (48, 64, 3)
    assert cv2.imread(str(kept))[0, 0, 2] > 150

    sparse = observer(twin, tmp_path, name="sparse", keep_frames_s=1.0)
    for step, t in enumerate((0.0, 0.5, 0.95, 1.0, 1.5)):
        sparse.step(record(sparse, twin, step, t=t, frames={"a": Frame(image, 0.01)}))
    sparse.close(end(steps=5))

    files = [step["frames"][0]["file"] for step in EpisodeLog.load(sparse.path).steps]
    assert [file is not None for file in files] == [True, False, False, True, False]

    plain = observer(twin, tmp_path, name="plain")
    plain.step(record(plain, twin, 0, frames={"a": Frame(image, 0.01)}))
    plain.close(end(steps=1))

    assert EpisodeLog.load(plain.path).steps[0]["frames"][0]["file"] is None
    assert not (tmp_path / "plain.frames").exists()


def test_logs_go_next_to_the_workspace_file(tmp_path):
    """Without a workspace file, the default `workspace/` (in tests: the test's folder, conftest)."""
    assert episodes_dir(tmp_path / "twin.json") == tmp_path / "episodes"
    assert episodes_dir(None) == tmp_path / "workspace" / "episodes"


# --------------------------------------------------------------------------- every run leaves a log

#: Where the camera-seen gripper sits against the readings' TCP [m]: readings minus cameras.
OFFSET = np.array([0.010, -0.005, 0.0])


def tour(home: dict[str, float]) -> list[dict[str, float]]:
    """Poses that move every joint of the test robot, then home."""
    return [dict(home, j1=20.0, j2=0.0, gripper=60.0), dict(home, j1=-10.0, j2=35.0, gripper=10.0), dict(home)]


def test_a_run_of_the_test_strategy_leaves_a_log_whose_report_shows_latency_on_every_joint_and_gripper_drift(
        twin, tmp_path):
    """#151 done: `EpisodeLog.load` reads the test strategy's run, and its report shows the latency of
    every joint and the gripper's drift against the cameras. The cameras here see the gripper
    10 mm and -5 mm off the readings' TCP, by construction."""
    twin.connect("sim", threaded=False)
    kinematics = RobotKinematics(get_spec(BOT))

    def gripper_seen() -> dict[str, ObjectPose]:
        T = kinematics.tcp(twin.status.measured)
        return {"chwytak": ObjectPose(T[:3, 3] - OFFSET, T[:3, :3], "kamery", 0.0)}

    runner = StrategyRunner(twin, Waypoints(tour(dict(twin.status.measured))), objects=gripper_seen,
                            episode=EpisodeOptions(folder=tmp_path, gripper="chwytak"))
    runner.start(threaded=False)
    drive(twin, runner, 10.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    result = report(log)

    assert log.end.code == "done"
    assert log.end.succeeded
    assert len(log.steps) == runner.status.step
    assert [joint.joint for joint in result.latency] == list(log.joints)
    for joint in result.latency:
        for fit in (joint.total, joint.supervisor, joint.actuator):
            assert isinstance(fit, DelayFit), (joint.joint, fit)
            assert 0.0 < fit.lag_s < 1.0

    drift = result.drift
    assert drift is not None
    assert drift.samples == len(log.steps)
    assert drift.mean_m == pytest.approx(OFFSET, abs=1e-4)
    assert drift.angle_max_deg is not None and drift.angle_max_deg < 0.1

    text = result.lines()
    for joint in log.joints:
        assert any(line.split()[:1] == [joint] for line in text), joint
    assert any("odczyty - kamery" in line for line in text)


def test_a_fake_arms_delay_shows_as_servo_latency(twin, monkeypatch, tmp_path):
    """A (fake) real arm whose readings show the pose from 60 ms ago: the servo column measures it."""
    connect_fake(twin, monkeypatch, FakeArm(delay=0.06))
    runner = StrategyRunner(twin, Waypoints(tour(dict(twin.status.measured))),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 10.0)

    assert runner.status.stopped_code == "done"
    assert runner.status.episode is not None
    result = report(EpisodeLog.load(runner.status.episode))

    for joint in result.latency:
        assert isinstance(joint.actuator, DelayFit), (joint.joint, joint.actuator)
        assert 0.05 <= joint.actuator.lag_s <= 0.09, joint.joint


def test_the_log_holds_every_accepted_tick(twin, tmp_path):
    twin.connect("sim", threaded=False)
    poses = tour(dict(twin.status.measured))
    runner = StrategyRunner(twin, Waypoints(poses), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 10.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    kinds = [line["kind"] for line in lines(runner.status.episode)]

    assert [step["step"] for step in log.steps] == list(range(runner.status.step))
    assert {json.dumps(step["intent"], sort_keys=True) for step in log.steps} == {
        json.dumps(pose, sort_keys=True) for pose in poses}
    assert [step["run_t"] for step in log.steps] == pytest.approx([0.05 * k for k in range(len(log.steps))])
    assert len(log.loop_t) > len(log.steps)
    assert kinds[0] == "header"
    assert kinds[-1] == "end"
    assert kinds.count("end") == 1
    assert log.end.error == pytest.approx(runner.status.error, abs=1e-6)
    assert log.end.steps == runner.status.step
    assert log.header["strategy"] == {"class": "machinekind_twin.waypoints:Waypoints", "hz": 20.0,
                                      "time_limit_s": None}
    assert log.header["owner"] == "strategia"
    assert log.header["mode"] == "strategy"


class Refusing(Plugin):
    def start(self, run, obs):
        raise RuntimeError("uczona na innym ramieniu")


def raising(obs):
    raise RuntimeError("zepsuta")


def ends(code: str, twin: Twin, runner_for) -> StrategyRunner:
    """Runs a strategy until it ends with `code`, the way that end happens."""
    if code == "refused":
        runner = runner_for(Refusing(answer({})))
        with pytest.raises(RuntimeError, match="uczona na innym"):
            runner.start(threaded=False)
        return runner

    answers = {
        "done": answer(Action(target={"j1": 3.0}, done=True, success=True, error=0.2)),
        "gave_up": answer(Action(target={"j1": 3.0}, done=True, success=False)),
        "bad_target": answer({"j1": math.nan}),
        "error": raising,
    }
    runner = runner_for(Plugin(answers.get(code, answer({"j1": 3.0}))),
                        time_limit_s=0.2 if code == "time_limit" else None)
    runner.start(threaded=False)
    drive(twin, runner, 0.4)

    if code == "stopped":
        runner.stop()
    elif code == "preempted":
        twin.estop()
    return runner


@pytest.mark.parametrize("code, reason", [
    ("done", "zadanie wykonane"),
    ("gave_up", "strategia skonczyla"),
    ("time_limit", "limit czasu"),
    ("stopped", "zatrzymana"),
    ("preempted", "przerwana: STOP awaryjny"),
    ("bad_target", "zly cel strategii"),
    ("error", "RuntimeError: zepsuta"),
    ("refused", "strategia odmowila: uczona na innym ramieniu"),
])
def test_every_end_leaves_a_closed_log_with_its_code(twin, tmp_path, code, reason):
    twin.connect("sim", threaded=False)

    def runner_for(strategy, **kw) -> StrategyRunner:
        return StrategyRunner(twin, strategy, episode=EpisodeOptions(folder=tmp_path), **kw)

    runner = ends(code, twin, runner_for)
    status = runner.status

    assert not status.running
    assert status.stopped_code == code
    assert status.stopped_because.startswith(reason)
    assert status.episode is not None
    log = EpisodeLog.load(status.episode)

    assert lines(status.episode)[-1]["kind"] == "end"
    assert log.end.code == code
    assert log.end.reason == status.stopped_because
    assert log.end.steps == len(log.steps) == status.step
    assert log.end.began == (code != "refused")
    assert status.episode_error == ""


def test_a_failed_claim_leaves_no_file(twin, tmp_path):
    twin.connect("sim", threaded=False)
    twin.estop()
    runner = StrategyRunner(twin, Plugin(answer({})), episode=EpisodeOptions(folder=tmp_path / "runs"))

    with pytest.raises(RuntimeError, match="aktywny STOP"):
        runner.start(threaded=False)

    assert runner.status.episode is None
    assert not (tmp_path / "runs").exists()


def test_a_late_act_writes_no_step(twin, tmp_path):
    """An `act` that returns after its run ended (a slow model past `stop`) is refused, and so is its line."""
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(answer({})), episode=EpisodeOptions(folder=tmp_path))

    def stopped_while_thinking(obs):
        if obs.step == 3:
            runner.stop("x")
        return {"j1": 5.0}

    runner.strategy.answer = stopped_while_thinking
    runner.start(threaded=False)
    drive(twin, runner, 0.5)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)

    assert runner.status.step == 3
    assert len(log.steps) == 3
    assert lines(runner.status.episode)[-1]["kind"] == "end"
    assert log.end.code == "stopped"
    assert log.end.reason == "x"


def test_the_log_stops_growing_once_the_run_ends(twin, tmp_path):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(answer({"j1": 3.0})), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 0.3)
    runner.stop()

    assert runner.status.episode is not None
    size = runner.status.episode.stat().st_size
    run_ticks(twin, 20)

    assert runner.status.episode.stat().st_size == size


def test_frames_the_strategy_read_are_referenced_not_fetched(twin, tmp_path):
    cv2 = pytest.importorskip("cv2")
    twin.connect("sim", threaded=False)
    image = np.full((48, 64, 3), 90, np.uint8)
    fetched = []

    def camera(name):
        fetched.append(name)
        return image, time.monotonic()

    def look_at_a(obs):
        obs.frames["a"]
        return {}

    def run(keep_frames_s, name) -> EpisodeLog:
        fetched.clear()
        runner = StrategyRunner(twin, Plugin(look_at_a), cameras=("a", "b"), frame=camera,
                                episode=EpisodeOptions(folder=tmp_path, name=name, keep_frames_s=keep_frames_s))
        runner.start(threaded=False)
        drive(twin, runner, 2.5)
        runner.stop()
        assert fetched == ["a"] * runner.status.step
        assert runner.status.episode is not None
        return EpisodeLog.load(runner.status.episode)

    log = run(None, "references")
    assert all([frame["camera"] for frame in step["frames"]] == ["a"] for step in log.steps)
    assert all(step["frames"][0]["file"] is None for step in log.steps)
    assert all(step["frames"][0]["shape"] == [48, 64, 3] for step in log.steps)

    log = run(0, "every")
    files = [step["frames"][0]["file"] for step in log.steps]
    assert all(files)
    assert cv2.imread(str(tmp_path / files[0])).shape == (48, 64, 3)

    log = run(1.0, "sparse")
    kept = [step["t"] for step in log.steps if step["frames"][0]["file"]]
    assert len(kept) == 3
    assert np.all(np.diff(kept) >= 1.0 - 1e-6)
    assert len(list((tmp_path / "sparse.frames").iterdir())) == 3


@dataclass
class Move:
    """A strategy's own answer object with `Action`'s attributes, decisions among them."""

    target: dict = field(default_factory=dict)
    decisions: tuple = ()


def test_a_decision_goes_into_the_log_of_its_tick(twin, tmp_path):
    twin.connect("sim", threaded=False)
    asked = Decision(question="gdzie jest kostka?", answer='{"x": 0.2}', state={"joints": {"j1": 1.0}},
                     confidence=0.8, response_s=1.25)
    own = Decision(question="czy chwycic?", answer="tak")
    last = Decision(question="koniec?", answer="tak", state={"image": np.zeros((480, 640, 3)), "gap": math.nan})

    def thinking(obs):
        if obs.step == 0:
            return {"j1": 1.0}
        if obs.step == 1:
            return Action(target={"j1": 2.0}, decisions=[asked])
        if obs.step == 2:
            return Move(target={"j1": 3.0}, decisions=(own,))
        if obs.step == 3:
            return Move(target={"j1": 3.0}, decisions=own)
        if obs.step == 4:
            return Action(target={"j1": 3.0}, decisions=collections.deque([own, asked]))
        return Action(target={"j1": 3.0}, done=True, success=True, error=math.nan, decisions=(last,))

    runner = StrategyRunner(twin, Plugin(thinking), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 0.5)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    decisions = [step["decisions"] for step in log.steps]

    assert decisions[0] == []
    assert decisions[1] == [{"question": "gdzie jest kostka?", "answer": '{"x": 0.2}', "state": {"joints": {"j1": 1.0}},
                             "confidence": 0.8, "response_s": 1.25}]
    assert decisions[2][0]["question"] == "czy chwycic?"
    assert decisions[3] == []
    assert [decision["question"] for decision in decisions[4]] == ["czy chwycic?", "gdzie jest kostka?"]
    assert decisions[5][0]["state"] == {"image": "<ndarray float64 (480, 640, 3)>", "gap": None}
    assert log.steps[5]["done"]
    assert log.end.code == "done"
    assert report(log).decisions.count == 5


def test_the_log_is_strict_json(twin, tmp_path):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(answer(Action(target={"j1": 1.0}, error=math.nan))),
                            episode=EpisodeOptions(folder=tmp_path, meta={"limit": math.inf}))
    runner.start(threaded=False)
    drive(twin, runner, 0.3)
    runner.stop()

    def refuse(constant):
        raise ValueError(constant)

    assert runner.status.episode is not None
    parsed = [json.loads(line, parse_constant=refuse) for line in runner.status.episode.read_text().splitlines()]

    assert parsed[0]["meta"] == {"limit": None}
    assert all(line["error"] is None for line in parsed if line["kind"] in ("step", "end"))


def test_two_runs_with_one_name_get_their_own_files(twin, tmp_path):
    twin.connect("sim", threaded=False)
    paths = []
    for _ in range(2):
        runner = StrategyRunner(twin, Plugin(answer({})), episode=EpisodeOptions(folder=tmp_path, name="same"))
        runner.start(threaded=False)
        drive(twin, runner, 0.2)
        runner.stop()
        paths.append(runner.status.episode)

    assert paths == [tmp_path / "same.jsonl", tmp_path / "same-2.jsonl"]
    assert [EpisodeLog.load(path).end.code for path in paths] == ["stopped", "stopped"]


def test_a_run_goes_next_to_its_workspace(twin, tmp_path):
    placed = Twin(Workspace(robot=BOT, path=tmp_path / "station" / "twin.json"), clock=VirtualClock(t0=0.0))
    try:
        for each in (placed, twin):
            each.connect("sim", threaded=False)
            runner = StrategyRunner(each, Plugin(answer({})))
            runner.start(threaded=False)
            runner.stop()
            assert runner.status.episode is not None
            assert runner.status.episode.parent == episodes_dir(each.workspace.path)
    finally:
        placed.close()

    assert episodes_dir(placed.workspace.path) == tmp_path / "station" / "episodes"
    assert episodes_dir(twin.workspace.path) == tmp_path / "workspace" / "episodes"


def test_a_log_that_cannot_open_does_not_stop_the_run(twin, tmp_path):
    twin.connect("sim", threaded=False)
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    runner = StrategyRunner(twin, Waypoints([dict(twin.status.measured, j1=5.0)]),
                            episode=EpisodeOptions(folder=blocker))
    runner.start(threaded=False)
    drive(twin, runner, 3.0)

    assert runner.status.stopped_code == "done"
    assert runner.status.episode is None
    assert runner.status.episode_error.startswith("nie zapisze przebiegu")


def test_a_log_that_breaks_mid_run_does_not_stop_the_run(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Waypoints([dict(twin.status.measured, j1=40.0)]),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)
    assert runner.status.running

    def full_disk(self, line):
        raise OSError("brak miejsca na dysku")

    monkeypatch.setattr(EpisodeWriter, "_write", full_disk)
    drive(twin, runner, 3.0)

    assert runner.status.stopped_code == "done"
    assert "brak miejsca na dysku" in runner.status.episode_error
    assert runner.status.episode is not None
    monkeypatch.undo()
    assert EpisodeLog.load(runner.status.episode).end.code == "unfinished"


class Unreadable:
    """An object whose `repr` raises, as a model client's may."""

    def __repr__(self) -> str:
        raise RuntimeError("repr nie dziala")


class Changing(dict):
    """A dict another thread changes while the log reads it."""

    def items(self):
        raise RuntimeError("dictionary changed size during iteration")


def test_a_value_the_log_cannot_read_costs_only_that_value(twin, tmp_path):
    twin.connect("sim", threaded=False)
    state = {"client": Unreadable(), "live": Changing(a=1), "ok": 1}

    def asking(obs):
        return Action(target={"j1": 2.0}, done=obs.step == 3, success=True,
                      decisions=(Decision(question="gdzie?", answer="tu", state=state),))

    runner = StrategyRunner(twin, Plugin(asking),
                            episode=EpisodeOptions(folder=tmp_path, meta={"client": Unreadable()}))
    runner.start(threaded=False)
    drive(twin, runner, 0.5)

    assert runner.status.stopped_code == "done"
    assert runner.status.episode_error == ""
    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)

    assert log.header["meta"] == {"client": "<Unreadable>"}
    assert log.end.code == "done"
    assert len(log.steps) == runner.status.step == 4
    assert log.steps[0]["decisions"][0]["state"] == {"client": "<Unreadable>", "live": "<Changing>", "ok": 1}


def test_a_log_whose_header_cannot_be_written_leaves_no_file_and_does_not_stop_the_run(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)

    def broken(self, line):
        raise ValueError("zly naglowek")

    monkeypatch.setattr(EpisodeWriter, "_write", broken)
    runner = StrategyRunner(twin, Waypoints([dict(twin.status.measured, j1=5.0)]),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 3.0)

    assert runner.status.stopped_code == "done"
    assert runner.status.episode is None
    assert "zly naglowek" in runner.status.episode_error
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- observation mode (#175)


def test_observation_without_commands_does_not_count_latency(twin, monkeypatch, tmp_path):
    """An arm driven by a foreign controller: the twin only reads it, and nobody tells it the command."""
    connect_fake(twin, monkeypatch, FakeArm())
    writer = observer(twin, tmp_path)
    for step in range(20):
        run_ticks(twin, 5)
        writer.step(record(writer, twin, step))
    writer.close(end(steps=20))

    log = EpisodeLog.load(writer.path)
    result = report(log)

    assert np.isnan(log.sent).all()
    assert np.isfinite(log.readings()[1]).all()
    assert result.latency == ()
    assert result.latency_note
    assert result.latency_note in result.lines()


def test_observation_with_a_foreign_controllers_commands_counts_latency(twin, tmp_path):
    """The foreign controller's command goes into the intent: the latency counts against it."""
    twin.connect("sim", threaded=False)
    twin.claim("obcy")
    assert twin.set_engaged(True, owner="obcy")
    start = twin.status.measured["j1"]

    writer = observer(twin, tmp_path)
    for step in range(40):
        command = {"j1": start + min(step, 20) * 1.5}
        twin.set_target(command, owner="obcy")
        writer.step(record(writer, twin, step, intent=command))
        run_ticks(twin, 3)
    writer.close(end(steps=40))

    total = report(EpisodeLog.load(writer.path)).latency[0].total

    assert isinstance(total, DelayFit)
    assert 0.0 < total.lag_s < 0.5


# --------------------------------------------------------------------------- twin episode


def finished_run(twin: Twin, folder: Path | None, name: str, code: str = "done") -> Path:
    """A short run of the test strategy that ends with `code`; its log's path."""
    pose = dict(twin.status.measured, j1=5.0 if code == "done" else 60.0)
    runner = StrategyRunner(twin, Waypoints([pose]), episode=EpisodeOptions(folder=folder, name=name))
    runner.start(threaded=False)
    drive(twin, runner, 0.5 if code == "stopped" else 3.0)
    runner.stop()
    assert runner.status.stopped_code == code
    assert runner.status.episode is not None
    return runner.status.episode


def test_twin_episode_reports_a_file(twin, tmp_path, capsys):
    twin.connect("sim", threaded=False)
    path = finished_run(twin, tmp_path, "run")

    assert twin_cli(["episode", str(path), "--gripper", "chwytak", "--offset-mm", "0", "0", "10"]) == 0

    out = capsys.readouterr().out
    assert "[done]" in out
    for joint in twin.workspace.spec().joints:
        assert f"\n  {joint} " in out
    assert "chwytak" in out


def test_twin_episode_lists_a_folder_newest_first(twin, tmp_path, capsys):
    twin.connect("sim", threaded=False)
    for name, code in (("run-2", "done"), ("run-10", "stopped")):
        finished_run(twin, tmp_path, name, code)
    (tmp_path / "broken.jsonl").write_text("nie zapis\n", encoding="utf-8")

    assert twin_cli(["episode", str(tmp_path)]) == 0

    rows = capsys.readouterr().out.splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["run-10", "run-2", "broken"]
    assert "stopped" in rows[0]
    assert "done" in rows[1] and "sukces tak" in rows[1]
    assert "nieczytelny" in rows[2]


def test_twin_episode_without_a_path_reports_the_newest_log_next_to_the_workspace(twin, tmp_path, capsys):
    twin.connect("sim", threaded=False)
    finished_run(twin, None, "a")
    finished_run(twin, None, "b", "stopped")

    assert twin_cli(["episode"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Przebieg b ")
    assert str(tmp_path / "workspace" / "episodes" / "b.jsonl") in out


def test_twin_episode_says_what_is_missing(tmp_path, capsys):
    assert twin_cli(["episode", str(tmp_path / "nie-ma.jsonl")]) == 1
    assert "nie ma" in capsys.readouterr().err

    assert twin_cli(["episode", "--path", str(tmp_path / "twin.json")]) == 1
    assert "nie ma" in capsys.readouterr().err

    (tmp_path / "episodes").mkdir()
    assert twin_cli(["episode", "--path", str(tmp_path / "twin.json")]) == 1
    assert "brak zapisow przebiegow" in capsys.readouterr().err

    (tmp_path / "other.jsonl").write_text('{"kind": "header"}\n', encoding="utf-8")
    assert twin_cli(["episode", str(tmp_path / "other.jsonl")]) == 1
    assert "to nie jest zapis przebiegu" in capsys.readouterr().err
