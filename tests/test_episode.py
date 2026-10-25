"""The episode log: its writer and loader, opt-in strategy recording, and the observation mode.

Everything in simulated time: the twin's loop without a thread (`connect(threaded=False)` +
`Twin.step`) and the runner without a thread (`start(threaded=False)` + `step_once`), as in
`test_strategy`. The simulated clock starts far from 0, as `time.monotonic` does in the
application, so every time in a log must count from the writer's start.
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

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
from machinekind_twin.episode_report import AT_LIMIT_FLAG, NEITHER, DelayFit, report
from machinekind_twin.kinematics import RobotKinematics
from machinekind_twin.robots import get_spec
from machinekind_twin.runtime import Twin
from machinekind_twin.strategy import Action, Decision, Frame, ObjectPose, StrategyRunner
from machinekind_twin.waypoints import Waypoints
from machinekind_twin.workspace import Workspace

TWIN_PERIOD = 0.02
#: Where the simulated clock starts [s].
CLOCK_T0 = 12345.678


@pytest.fixture
def twin():
    twin = Twin(Workspace(robot=BOT), clock=VirtualClock(t0=CLOCK_T0))
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


def log_bytes(path: Path) -> bytes:
    if path.suffix == ".zst":
        with path.open("rb") as file, zstd.ZstdDecompressor().stream_reader(file) as reader:
            return reader.read()
    return path.read_bytes()


def log_text(path: Path) -> str:
    return log_bytes(path).decode("utf-8")


def lines(path: Path) -> list[dict]:
    """The log's records; only "\\n" ends a line, as in `EpisodeLog.load`."""
    return [json.loads(line) for line in log_text(path).split("\n") if line]


# --------------------------------------------------------------------------- the file


def test_a_log_round_trips_through_its_loader(twin, tmp_path):
    twin.connect("sim", threaded=False)
    joints = twin.workspace.spec().joints
    writer = observer(twin, tmp_path, name="run")

    run_ticks(twin, 5)
    measured = [twin.status.measured[joint] for joint in joints]
    command = [twin.status.command[joint] for joint in joints]
    cube = ObjectPose(np.array([0.2, 0.0, 0.015]), np.eye(3), "kamery", 0.1)
    writer.step(record(writer, twin, 0, run_t=0.05, obs_t=0.07, t=0.09, intent={"j1": 20.0},
                       goal=np.array([0.2, 0.0, 0.1]), objects={"cube": cube},
                       frames={"a": Frame(np.zeros((4, 6, 3), np.uint8), 0.02)}, success=False, error=0.5,
                       note="poza 1/1"))
    run_ticks(twin, 3)
    assert writer.now() == pytest.approx(8 * TWIN_PERIOD)
    writer.close(end(t=1.0, steps=1, success=True, error=0.5, note="poza 1/1"), wait=True)

    log = EpisodeLog.load(writer.path)

    assert writer.path == tmp_path / "run.jsonl.zst"
    assert log.header["format"] == FORMAT
    assert log.header["robot"] == BOT
    assert log.header["mode"] == "observation"
    assert log.header["meta"] == {"seed": 3}
    assert log.header["clock"] == "virtual"
    assert log.header["t0"] == pytest.approx(CLOCK_T0)
    assert log.joints == tuple(joints)
    assert log.loop_t == pytest.approx(TWIN_PERIOD * np.arange(1, 9))
    assert log.measured.shape == log.sent.shape == log.at_limit.shape == (8, len(joints))

    t, readings = log.readings()
    assert t == pytest.approx(TWIN_PERIOD * np.arange(1, 9))
    assert np.isfinite(readings).all()

    step = log.steps[0]
    assert (step["run_t"], step["obs_t"], step["t"]) == (0.05, 0.07, 0.09)
    assert step["joints"] == pytest.approx(measured, abs=1e-4)
    assert step["command"] == pytest.approx(command, abs=1e-4)
    assert step["intent"] == {"j1": 20.0}
    assert step["goal"] == [0.2, 0.0, 0.1]
    assert step["objects"] == {"cube": {"position": [0.2, 0.0, 0.015], "rotation": np.eye(3).tolist(),
                                        "source": "kamery", "age_s": 0.1}}
    assert step["frames"] == [{"camera": "a", "age_s": 0.02, "shape": [4, 6, 3], "file": None}]
    assert (step["done"], step["success"], step["error"], step["note"]) == (False, False, 0.5, "poza 1/1")

    intent_t, intents = log.intents()
    assert intent_t.tolist() == [0.09]
    assert intents[0, joints.index("j1")] == 20.0
    assert np.isnan(np.delete(intents[0], joints.index("j1"))).all()

    assert log.end == EpisodeEnd(code="stopped", reason="zatrzymana", t=1.0, done=False, success=True, error=0.5,
                                 note="poza 1/1", steps=1, began=True)


def test_every_line_is_strict_json_with_constant_keys(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    image = np.zeros((480, 640, 3), np.uint8)

    run_ticks(twin, 2)
    decision = Decision(question="gdzie kostka?", answer="tu", state={"image": image, "gap": math.nan},
                        confidence=math.inf)
    writer.step(record(writer, twin, 0, error=math.nan, decisions=[decision]))
    writer.step(record(writer, twin, 1))
    writer.close(end(steps=2, error=math.nan), wait=True)

    def refuse(constant):
        raise ValueError(constant)

    parsed = [json.loads(line, parse_constant=refuse) for line in log_text(writer.path).splitlines()]
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
    writer.close(end(code="done", steps=3), wait=True)

    text = log_text(writer.path).splitlines()
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
    writer.close(end(code="done", steps=2), wait=True)

    data = log_bytes(writer.path)
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
    writer.close(end(code="done", steps=1, success=True, reason=odd, note=odd), wait=True)

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
    writer.close(end(), wait=True)
    text = log_text(writer.path).splitlines()

    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join([text[0], "{nie json", *text[1:]]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken.jsonl:2: corrupt record"):
        EpisodeLog.load(broken)

    newer = tmp_path / "newer.jsonl"
    newer.write_text("\n".join([json.dumps({**json.loads(text[0]), "version": 3}), *text[1:]]), encoding="utf-8")
    with pytest.raises(ValueError, match="version 3"):
        EpisodeLog.load(newer)

    other = tmp_path / "other.jsonl"
    other.write_text('{"kind": "header", "format": "cos innego"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not an episode log"):
        EpisodeLog.load(other)


def test_two_logs_with_one_name_get_their_own_files(twin, tmp_path):
    twin.connect("sim", threaded=False)
    first = observer(twin, tmp_path, name="same")
    second = observer(twin, tmp_path, name="same")
    first.close(end(), wait=True)
    second.close(end(), wait=True)

    assert {first.path.name, second.path.name} == {"same.jsonl.zst", "same-2.jsonl.zst"}
    assert EpisodeLog.load(first.path).header["name"] == first.path.name.removesuffix(".jsonl.zst")
    assert EpisodeLog.load(second.path).header["name"] == second.path.name.removesuffix(".jsonl.zst")


def test_a_closed_log_stops_growing(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    run_ticks(twin, 3)
    writer.close(end(), wait=True)
    size = writer.path.stat().st_size

    run_ticks(twin, 10)
    writer.close(end(code="error"), wait=True)
    writer.flush()

    assert writer.path.stat().st_size == size
    assert EpisodeLog.load(writer.path).end.code == "stopped"


def test_the_summary_reads_the_header_and_the_end(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    for step in range(3):
        run_ticks(twin, 2)
        writer.step(record(writer, twin, step, error=0.25))
    writer.close(end(code="done", steps=3, success=True, error=0.25), wait=True)

    summary = EpisodeLog.summary(writer.path)

    assert summary.header["robot"] == BOT
    assert summary.end == EpisodeLog.load(writer.path).end
    assert summary.end.succeeded

    unfinished = tmp_path / "unfinished.jsonl"
    unfinished.write_text("\n".join(log_text(writer.path).splitlines()[:-1]), encoding="utf-8")

    assert EpisodeLog.summary(unfinished).end.code == "unfinished"
    assert EpisodeLog.summary(unfinished).end.steps == 3


@pytest.mark.parametrize("frame_format, extension", [("webp", ".webp"), ("jpeg", ".jpg")])
def test_kept_frames_round_trip_in_both_formats(twin, tmp_path, frame_format, extension):
    cv2 = pytest.importorskip("cv2")
    twin.connect("sim", threaded=False)
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :, 0] = 200

    every = observer(twin, tmp_path, name="every", keep_frames_s=0, frame_format=frame_format)
    every.step(record(every, twin, 0, frames={"a": Frame(image, 0.01), "b": None}))
    every.close(end(steps=1), wait=True)

    frames = EpisodeLog.load(every.path).steps[0]["frames"]
    kept = tmp_path / frames[0]["file"]
    assert frames[0]["camera"] == "a"
    assert frames[0]["shape"] == [48, 64, 3]
    assert frames[1] == {"camera": "b", "age_s": None, "shape": None, "file": None,
                         "drop_reason": "camera frame unavailable"}
    assert kept.parent == tmp_path / "every.frames"
    assert kept.suffix == extension
    assert cv2.imread(str(kept)).shape == (48, 64, 3)
    assert cv2.imread(str(kept))[0, 0, 2] > 150

    sparse = observer(twin, tmp_path, name="sparse", keep_frames_s=1.0)
    for step, t in enumerate((0.0, 0.5, 0.95, 1.0, 1.5)):
        sparse.step(record(sparse, twin, step, t=t, frames={"a": Frame(image, 0.01)}))
    sparse.close(end(steps=5), wait=True)

    files = [step["frames"][0]["file"] for step in EpisodeLog.load(sparse.path).steps]
    assert [file is not None for file in files] == [True, False, False, True, False]

    plain = observer(twin, tmp_path, name="plain")
    plain.step(record(plain, twin, 0, frames={"a": Frame(image, 0.01)}))
    plain.close(end(steps=1), wait=True)

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

    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    result = report(log)

    assert log.end.code == "done"
    assert log.end.succeeded
    assert len(log.steps) == runner.status.step
    assert log.header["t0"] == pytest.approx(CLOCK_T0)
    assert 0.0 <= log.loop_t[0] < 0.1
    assert [joint.joint for joint in result.latency] == list(log.joints)
    for joint in result.latency:
        total, supervisor, actuator = joint.total, joint.supervisor, joint.actuator
        assert isinstance(total, DelayFit), (joint.joint, total)
        assert isinstance(supervisor, DelayFit), (joint.joint, supervisor)
        assert isinstance(actuator, DelayFit), (joint.joint, actuator)
        assert 0.0 < actuator.lag_s < supervisor.lag_s < total.lag_s < 1.0, joint

    drift = result.drift
    assert drift is not None
    assert drift.samples == len(log.steps)
    assert drift.mean_m == pytest.approx(OFFSET, abs=1e-4)
    assert drift.angle_max_deg is not None and drift.angle_max_deg < 0.1

    text = result.lines()
    for joint in log.joints:
        assert any(line.split()[:1] == [joint] for line in text), joint
    assert any("measured - cameras" in line for line in text)


def test_a_fake_arms_delay_shows_as_servo_latency(twin, monkeypatch, tmp_path):
    """A (fake) real arm whose readings show the pose from 60 ms ago: the servo column measures it.

    The twin reads a real arm every 40 ms. On a simulated clock whose rounding puts each read one
    loop tick late, it reads every 60 ms and this column gives 85 ms, so the test checks the read
    period first.
    """
    connect_fake(twin, monkeypatch, FakeArm(delay=0.06))
    runner = StrategyRunner(twin, Waypoints(tour(dict(twin.status.measured))),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 10.0)

    assert runner.status.stopped_code == "done"
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    result = report(log)

    assert np.median(np.diff(log.readings()[0])) == pytest.approx(0.04)

    for joint in result.latency:
        assert isinstance(joint.actuator, DelayFit), (joint.joint, joint.actuator)
        assert 0.06 <= joint.actuator.lag_s <= 0.08, joint.joint


def test_a_single_move_has_its_latency(twin, tmp_path):
    """One waypoint: the intent never changes, and the move is from where the arm stood."""
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Waypoints([dict(twin.status.measured, j1=20.0)]),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 3.0)

    assert runner.status.stopped_code == "done"
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    j1 = next(joint for joint in report(EpisodeLog.load(runner.status.episode)).latency if joint.joint == "j1")

    assert isinstance(j1.total, DelayFit), j1.total
    assert 0.0 < j1.total.lag_s < 1.0


def test_readings_at_a_limit_are_logged_and_flagged(twin, tmp_path):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(answer({"j1": 1000.0})), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 3.0)
    runner.stop()

    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    result = report(log)
    column = log.joints.index("j1")
    j1 = result.latency[column]

    assert log.at_limit[:, column].any()
    assert not np.delete(log.at_limit, column, axis=1).any()
    assert isinstance(j1.actuator, DelayFit), j1.actuator
    assert j1.actuator.at_limit_share > AT_LIMIT_FLAG
    assert any(line.startswith("  j1 ") and "samples at a joint limit" in line for line in result.lines())


def test_joints_whose_target_copies_the_readings_get_no_delay(twin, tmp_path):
    """A strategy that moves j1 and keeps the rest where they are by copying the readings into its
    target. Those joints creep a little in simulation, and the target follows them: no delay."""
    twin.connect("sim", threaded=False)

    def swing_j1_keep_the_rest(obs):
        return dict(obs.joints, j1=20.0 if obs.step // 20 % 2 == 0 else -20.0)

    runner = StrategyRunner(twin, Plugin(swing_j1_keep_the_rest), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 6.0)
    runner.stop()

    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    latency = {joint.joint: joint for joint in report(EpisodeLog.load(runner.status.episode)).latency}

    assert all(isinstance(fit, DelayFit) for fit in (latency["j1"].total, latency["j1"].actuator))
    for name in set(latency) - {"j1"}:
        assert not isinstance(latency[name].total, DelayFit), latency[name]
        assert not isinstance(latency[name].actuator, DelayFit), latency[name]


def test_the_log_holds_every_accepted_tick(twin, tmp_path):
    twin.connect("sim", threaded=False)
    poses = tour(dict(twin.status.measured))
    runner = StrategyRunner(twin, Waypoints(poses), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 10.0)

    assert runner.wait_for_episode(timeout=5.0)
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


def ends(code: str, twin: Twin, runner_for, monkeypatch) -> StrategyRunner:
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
    elif code == "disconnected":
        # The loop died on another thread, and its preempt has not reached the runner before its tick.
        monkeypatch.setattr(Twin, "connected", property(lambda self: False))
        runner.step_once(0.45)
        monkeypatch.undo()
    return runner


@pytest.mark.parametrize("code, reason", [
    ("done", "zadanie wykonane"),
    ("gave_up", "strategia skonczyla"),
    ("time_limit", "limit czasu"),
    ("disconnected", "ramie rozlaczone"),
    ("stopped", "zatrzymana"),
    ("preempted", "przerwana: STOP awaryjny"),
    ("bad_target", "zly cel strategii"),
    ("error", "RuntimeError: zepsuta"),
    ("refused", "strategia odmowila: uczona na innym ramieniu"),
])
def test_every_end_leaves_a_closed_log_with_its_code(twin, tmp_path, monkeypatch, code, reason):
    twin.connect("sim", threaded=False)

    def runner_for(strategy, **kw) -> StrategyRunner:
        return StrategyRunner(twin, strategy, episode=EpisodeOptions(folder=tmp_path), **kw)

    runner = ends(code, twin, runner_for, monkeypatch)
    assert runner.wait_for_episode(timeout=5.0)
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

    assert runner.wait_for_episode(timeout=5.0)
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

    assert runner.wait_for_episode(timeout=5.0)
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

    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    size = runner.status.episode.stat().st_size
    run_ticks(twin, 20)

    assert runner.wait_for_episode(timeout=5.0)
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
        assert runner.wait_for_episode(timeout=5.0)
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

    assert runner.wait_for_episode(timeout=5.0)
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

    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is not None
    parsed = [json.loads(line, parse_constant=refuse) for line in log_text(runner.status.episode).splitlines()]

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
        assert runner.wait_for_episode(timeout=5.0)
        paths.append(runner.status.episode)

    assert paths == [tmp_path / "same.jsonl.zst", tmp_path / "same-2.jsonl.zst"]
    assert [EpisodeLog.load(path).end.code for path in paths] == ["stopped", "stopped"]


def test_a_run_goes_next_to_its_workspace(twin, tmp_path):
    placed = Twin(Workspace(robot=BOT, path=tmp_path / "station" / "twin.json"), clock=VirtualClock(t0=0.0))
    try:
        for each in (placed, twin):
            each.connect("sim", threaded=False)
            runner = StrategyRunner(each, Plugin(answer({})), episode=EpisodeOptions())
            runner.start(threaded=False)
            runner.stop()
            assert runner.wait_for_episode(timeout=5.0)
            assert runner.status.episode is not None
            assert runner.wait_for_episode(timeout=5.0)
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
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is None
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode_error.startswith("episode recording failed")


def test_a_log_that_breaks_mid_run_does_not_stop_the_run(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Waypoints([dict(twin.status.measured, j1=40.0)]),
                            episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)
    assert runner.status.running

    assert runner._run is not None and runner._run.log is not None
    assert runner._run.log.flush(wait=True, timeout=5.0)

    def full_disk(self, line):
        raise OSError("brak miejsca na dysku")

    monkeypatch.setattr(EpisodeWriter, "_write", full_disk)
    drive(twin, runner, 3.0)

    assert runner.status.stopped_code == "done"
    assert runner.wait_for_episode(timeout=5.0)
    assert "brak miejsca na dysku" in runner.status.episode_error
    assert runner.wait_for_episode(timeout=5.0)
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
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode_error == ""
    assert runner.wait_for_episode(timeout=5.0)
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
    assert runner.wait_for_episode(timeout=5.0)
    assert runner.status.episode is None
    assert runner.wait_for_episode(timeout=5.0)
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
    writer.close(end(steps=20), wait=True)

    log = EpisodeLog.load(writer.path)
    result = report(log)

    assert np.isnan(log.sent).all()
    assert np.isfinite(log.readings()[1]).all()
    assert result.latency == ()
    assert result.latency_note == NEITHER
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
    writer.close(end(steps=40), wait=True)

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
    assert runner.wait_for_episode(timeout=5.0)
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
    assert "'chwytak'" in out


def test_twin_episode_lists_a_folder_newest_first(twin, tmp_path, capsys):
    twin.connect("sim", threaded=False)
    for name, code in (("run-2", "done"), ("run-10", "stopped")):
        finished_run(twin, tmp_path, name, code)
    (tmp_path / "broken.jsonl").write_text("nie zapis\n", encoding="utf-8")

    assert twin_cli(["episode", str(tmp_path)]) == 0

    rows = capsys.readouterr().out.splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["run-10", "run-2", "broken"]
    assert "stopped" in rows[0]
    assert "done" in rows[1] and "success yes" in rows[1]
    assert "unreadable" in rows[2]


def test_twin_episode_without_a_path_reports_the_newest_log_next_to_the_workspace(twin, tmp_path, capsys):
    twin.connect("sim", threaded=False)
    finished_run(twin, None, "a")
    finished_run(twin, None, "b", "stopped")

    assert twin_cli(["episode"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Episode b ")
    assert str(tmp_path / "workspace" / "episodes" / "b.jsonl.zst") in out


def test_twin_episode_says_what_is_missing(tmp_path, capsys):
    assert twin_cli(["episode", str(tmp_path / "nie-ma.jsonl")]) == 1
    assert "no such path" in capsys.readouterr().err

    assert twin_cli(["episode", "--path", str(tmp_path / "twin.json")]) == 1
    assert "no such path" in capsys.readouterr().err

    (tmp_path / "episodes").mkdir()
    assert twin_cli(["episode", "--path", str(tmp_path / "twin.json")]) == 1
    assert "no episode logs" in capsys.readouterr().err

    (tmp_path / "other.jsonl").write_text('{"kind": "header"}\n', encoding="utf-8")
    assert twin_cli(["episode", str(tmp_path / "other.jsonl")]) == 1
    assert "not an episode log" in capsys.readouterr().err


def test_recording_is_disabled_unless_requested(twin, tmp_path):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, Plugin(answer({})))
    runner.start(threaded=False)
    drive(twin, runner, 0.2)
    runner.stop()

    assert runner.wait_for_episode(timeout=1.0)
    assert runner.status.episode is None
    assert runner.status.episode_error == ""
    assert not (tmp_path / "workspace" / "episodes").exists()


def test_disk_writes_do_not_block_steps_or_stopping(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    writing, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def slow_write(self, line):
        assert threading.current_thread() is not threading.main_thread()
        if line["kind"] == "step":
            writing.set()
            assert release.wait(5.0)
        original(self, line)

    monkeypatch.setattr(EpisodeWriter, "_write", slow_write)
    runner = StrategyRunner(twin, Plugin(answer({"j1": 5.0})), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    try:
        assert runner.step_once(0.0)
        assert writing.wait(5.0)
        assert runner.step_once(0.05)

        def stop():
            runner.stop()
            stopped.set()

        stopper = threading.Thread(target=stop)
        stopper.start()
        assert stopped.wait(1.0), "stop waited for the blocked disk writer"
        stopper.join()
        assert not runner.status.running
        assert not runner.wait_for_episode(timeout=0.01)
    finally:
        release.set()
        runner.stop()
        assert runner.wait_for_episode(timeout=5.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    assert len(log.steps) == 2
    assert log.end.code == "stopped"


def test_telemetry_overflow_is_marked_and_collection_resumes(twin, tmp_path, monkeypatch):
    from machinekind_twin import episode

    twin.connect("sim", threaded=False)
    writing, release, resumed = threading.Event(), threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def paused(self, line):
        if line["kind"] == "header":
            writing.set()
            assert release.wait(5.0)
        original(self, line)
        if line["kind"] == "loop":
            resumed.set()

    monkeypatch.setattr(episode, "MAX_PENDING", 4)
    monkeypatch.setattr(EpisodeWriter, "_write", paused)
    runner = StrategyRunner(twin, Plugin(answer({"j1": 5.0})), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    try:
        assert writing.wait(5.0)
        run_ticks(twin, 20)
        assert runner.step_once(0.0)
        assert runner.status.running
        assert runner.status.episode_error == ""
        assert "INCOMPLETE TELEMETRY" in runner.status.episode_warning
        release.set()
        assert resumed.wait(5.0)
        writer = runner._episode_writer
        assert writer is not None
        assert writer.flush(wait=True, timeout=5.0)
        assert runner.step_once(0.05)
        assert writer.flush(wait=True, timeout=5.0)
    finally:
        release.set()
        runner.stop()
        assert runner.wait_for_episode(timeout=5.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    assert log.end.code == "stopped"
    assert log.recording["loop_dropped"] == 16
    assert log.recording["step_dropped"] == 1
    assert not log.recording["telemetry_complete"]
    assert [step["step"] for step in log.steps] == [1]
    assert report(log).latency == ()
    assert "Telemetry is incomplete" in report(log).latency_note
    assert twin_cli(["episode", str(runner.status.episode)]) == 2


def test_zstandard_truncation_recovers_records_and_checksum_corruption_is_reported(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path, name="complete")
    writer.step(record(writer, twin, 0, intent={"j1": 5.0}))
    assert writer.flush(wait=True, timeout=5.0)
    writer.close(end(code="done", steps=1), wait=True)
    data = writer.path.read_bytes()

    truncated = tmp_path / "cut.jsonl.zst"
    truncated.write_bytes(data[:-4])
    log = EpisodeLog.load(truncated)
    assert log.end.code == "unfinished"
    assert len(log.steps) == 1
    assert EpisodeLog.summary(truncated).end.code == "unfinished"

    corrupt = tmp_path / "corrupt.jsonl.zst"
    corrupt.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="corrupt Zstandard log"):
        EpisodeLog.load(corrupt)
    with pytest.raises(ValueError, match="corrupt Zstandard log"):
        EpisodeLog.summary(corrupt)


def test_queued_images_and_decision_state_are_snapshots(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    writing, release = threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def paused(self, line):
        if line["kind"] == "header":
            writing.set()
            assert release.wait(5.0)
        original(self, line)

    monkeypatch.setattr(EpisodeWriter, "_write", paused)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :, 0] = 200
    state = {"joint": 1}
    try:
        assert writing.wait(5.0)
        writer.step(record(writer, twin, 0, frames={"a": Frame(image, 0.0)},
                           decisions=(Decision("reach", "yes", state=state),)))
        image[:] = 0
        state["joint"] = 99
    finally:
        release.set()
        writer.close(end(steps=1), wait=True)

    import cv2

    log = EpisodeLog.load(writer.path)
    saved = cv2.imread(str(tmp_path / log.steps[0]["frames"][0]["file"]))
    assert saved[0, 0, 2] > 150
    assert log.steps[0]["decisions"][0]["state"] == {"joint": 1}


def test_image_queue_has_a_byte_limit(twin, tmp_path, monkeypatch):
    from machinekind_twin import episode

    twin.connect("sim", threaded=False)
    monkeypatch.setattr(episode, "MAX_PENDING_IMAGE_BYTES", 100)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    writer.step(record(writer, twin, 0, frames={"a": Frame(np.zeros((20, 20, 3), np.uint8), 0.0)}))
    writer.close(end(steps=1), wait=True)

    assert writer.error == ""
    log = EpisodeLog.load(writer.path)
    assert log.end.code == "stopped"
    assert log.recording["telemetry_complete"]
    assert log.recording["frames_skipped"] == 1
    assert log.steps[0]["frames"][0]["drop_reason"] == "image memory budget exhausted"


@pytest.mark.parametrize("options", [
    {"keep_frames_s": -1}, {"keep_frames_s": math.inf},
    {"frame_quality": 0}, {"frame_quality": 101}, {"frame_quality": True},
    {"frame_format": "png"}, {"queue_capacity": 0}, {"queue_capacity": True},
    {"frame_buffer_bytes": -1},
])
def test_invalid_recording_options_are_rejected(twin, tmp_path, options):
    with pytest.raises(ValueError):
        observer(twin, tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_failed_writer_releases_flush_waiters(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    assert writer.flush(wait=True, timeout=5.0)

    def broken(self, line):
        raise OSError("disk disconnected")

    monkeypatch.setattr(EpisodeWriter, "_write", broken)
    writer.step(record(writer, twin, 0))
    returned = threading.Event()
    results = []

    def flush():
        results.append(writer.flush(wait=True))
        returned.set()

    waiter = threading.Thread(target=flush, daemon=True)
    waiter.start()
    assert returned.wait(1.0)
    assert results == [False]
    assert writer.wait(timeout=5.0)


@pytest.mark.parametrize("stage", ["create", "serialize", "encode", "flush", "sync", "close"])
def test_runner_does_not_wait_for_any_storage_stage(twin, tmp_path, monkeypatch, stage):
    import cv2

    from machinekind_twin import episode

    twin.connect("sim", threaded=False)
    entered, release = threading.Event(), threading.Event()

    def pause():
        assert threading.current_thread() is not threading.main_thread()
        entered.set()
        assert release.wait(5.0)

    name = {"create": "_open_new", "serialize": "_write", "encode": "imencode",
            "flush": "_flush", "sync": "fsync"}.get(stage)
    target = {"create": episode, "serialize": EpisodeWriter, "encode": cv2,
              "flush": EpisodeWriter, "sync": episode.os}.get(stage)
    if stage == "close":
        original = episode._open_new

        class SlowClose:
            def __init__(self, file):
                self.file = file

            def __getattr__(self, key):
                return getattr(self.file, key)

            def close(self):
                pause()
                return self.file.close()

        def open_new(folder, stem):
            path, file = original(folder, stem)
            return path, SlowClose(file)

        monkeypatch.setattr(episode, "_open_new", open_new)
    else:
        original = getattr(target, name)

        def blocked(*args, **kwargs):
            pause()
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, blocked)

    image = np.zeros((24, 32, 3), np.uint8)

    def look(obs):
        obs.frames["camera"]
        return {"j1": float(obs.step)}

    runner = StrategyRunner(twin, Plugin(look), cameras=("camera",),
                            frame=lambda _: (image, time.monotonic()),
                            episode=EpisodeOptions(folder=tmp_path, keep_frames_s=0))
    runner.start(threaded=False)
    try:
        assert runner.step_once(0.0)
        if stage in {"sync", "close"}:
            runner.stop()
        assert entered.wait(2.0)
        if runner.status.running:
            assert runner.step_once(0.05)
            run_ticks(twin, 3)
        runner.stop()
        assert not runner.status.running
        assert not runner.wait_for_episode(timeout=0.01)
        assert runner._episode_writer is not None
        assert not runner._episode_writer.close(end(), wait=True, timeout=0.01)
    finally:
        release.set()
        runner.stop()
        assert runner.wait_for_episode(timeout=5.0)

    assert runner.status.episode_error == ""
    assert runner.status.episode is not None
    assert EpisodeLog.load(runner.status.episode).end.code == "stopped"


def test_close_succeeds_with_a_full_queue_without_waiting_for_capacity(twin, tmp_path, monkeypatch):
    from machinekind_twin import episode

    twin.connect("sim", threaded=False)
    writing, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def paused(self, line):
        if line["kind"] == "header":
            writing.set()
            assert release.wait(5.0)
        original(self, line)

    monkeypatch.setattr(episode, "MAX_PENDING", 4)
    monkeypatch.setattr(EpisodeWriter, "_write", paused)
    writer = observer(twin, tmp_path)
    try:
        assert writing.wait(5.0)
        run_ticks(twin, 4)

        def close():
            writer.close(end())
            closed.set()

        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        assert closed.wait(1.0)
        closer.join(timeout=1.0)
        assert writer.error == ""
    finally:
        release.set()
        assert writer.wait(timeout=5.0)

    log = EpisodeLog.load(writer.path)
    assert len(log.loop_t) == 4
    assert log.end.code == "stopped"



def test_concurrent_close_writes_exactly_one_end_record(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    writing, release = threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def paused(self, line):
        if line["kind"] == "header":
            writing.set()
            assert release.wait(5.0)
        original(self, line)

    monkeypatch.setattr(EpisodeWriter, "_write", paused)
    writer = observer(twin, tmp_path)
    barrier = threading.Barrier(8)

    def close(number):
        barrier.wait(timeout=2.0)
        writer.close(end(reason=f"caller {number}"))

    closers = [threading.Thread(target=close, args=(i,), daemon=True) for i in range(8)]
    try:
        assert writing.wait(5.0)
        for closer in closers:
            closer.start()
        for closer in closers:
            closer.join(timeout=2.0)
            assert not closer.is_alive()
    finally:
        release.set()
        writer.close(end(reason="cleanup"))
        assert writer.wait(timeout=5.0)

    ends = [line for line in lines(writer.path) if line["kind"] == "end"]
    assert len(ends) == 1
    assert ends[0]["reason"] in {f"caller {i}" for i in range(8)}


def test_stuck_storage_cannot_hold_process_exit(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(root / "src"), str(root / "packages" / "machinekind_twin_api"),
    ]))
    script = textwrap.dedent('''
        import threading
        from types import SimpleNamespace
        from machinekind_twin import episode

        entered = threading.Event()

        class Twin:
            clock = SimpleNamespace(now=lambda: 0.0, virtual=True)
            def add_tick_listener(self, fn): return 1
            def remove_tick_listener(self, handle): pass

        def blocked(*args, **kwargs):
            entered.set()
            threading.Event().wait()

        episode._open_new = blocked
        writer = episode.EpisodeWriter.create(FOLDER, {"joints": []}, Twin())
        assert entered.wait(2.0)
        writer.close(episode.EpisodeEnd("stopped", "exit", 0.0, False, None, None, "", 0, True))
        assert not writer.wait(timeout=0.01)
        print("runner exited", flush=True)
    ''').replace("FOLDER", repr(str(tmp_path)))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                            timeout=5.0, check=True)
    assert result.stdout.strip() == "runner exited"


def test_diagnostic_snapshots_do_not_call_plugin_hooks(twin, tmp_path):
    twin.connect("sim", threaded=False)
    called = []

    class Client:
        def __repr__(self):
            called.append("repr")
            raise AssertionError("must not run on the producer")

    class CustomMapping(dict):
        def items(self):
            called.append("items")
            raise AssertionError("must not run on the producer")

    class CustomDecision:
        @property
        def question(self):
            called.append("property")
            raise AssertionError("must not run on the producer")

    writer = EpisodeWriter.create(tmp_path, {**twin_header(twin), "meta": {"client": Client()}}, twin)
    writer.step(record(writer, twin, 0, decisions=(
        Decision("reach", "yes", state={"client": Client(), "mapping": CustomMapping(x=1)}),
        CustomDecision(),
    )))
    assert writer.close(end(steps=1), wait=True)

    assert called == []
    log = EpisodeLog.load(writer.path)
    assert log.header["meta"]["client"] == "<Client>"
    assert log.steps[0]["decisions"][0]["state"] == {"client": "<Client>", "mapping": "<CustomMapping>"}


def test_thread_start_failure_disables_recording_without_failing_the_run(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    listeners = len(twin._listeners)
    original = threading.Thread.start

    def start(thread):
        if thread.name == "episode-writer":
            raise RuntimeError("thread unavailable")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    runner = StrategyRunner(twin, Plugin(answer({"j1": 5.0})), episode=EpisodeOptions(folder=tmp_path))
    runner.start(threaded=False)
    assert runner.step_once(0.0)
    assert runner.status.running
    assert runner.status.episode is None
    assert "thread unavailable" in runner.status.episode_error
    assert len(twin._listeners) == listeners
    runner.stop()


@pytest.mark.parametrize("timeout", [None, math.inf, math.nan, -1])
def test_explicit_waits_reject_unbounded_or_invalid_timeouts(twin, tmp_path, timeout):
    writer = observer(twin, tmp_path)
    try:
        with pytest.raises(ValueError, match="finite, non-negative"):
            writer.wait(timeout)
        with pytest.raises(ValueError, match="finite, non-negative"):
            writer.flush(wait=True, timeout=timeout)
    finally:
        assert writer.close(end(), wait=True)


def test_threaded_runner_keeps_acting_while_storage_is_blocked(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    writing, release, progressed = threading.Event(), threading.Event(), threading.Event()
    original = EpisodeWriter._write

    def blocked(self, line):
        if line["kind"] == "step":
            writing.set()
            assert release.wait(5.0)
        original(self, line)

    def act(obs):
        if obs.step >= 4:
            progressed.set()
        return {"j1": float(obs.step)}

    monkeypatch.setattr(EpisodeWriter, "_write", blocked)
    runner = StrategyRunner(twin, Plugin(act), episode=EpisodeOptions(folder=tmp_path))
    runner.start()
    try:
        assert writing.wait(2.0)
        assert progressed.wait(2.0), "the strategy stopped ticking while the writer was blocked"
        runner.stop()
        assert runner.status.step >= 5
        assert runner.status.episode_error == ""
    finally:
        release.set()
        runner.stop()
        assert runner.wait_for_episode(timeout=5.0)

    assert runner.status.episode is not None
    assert EpisodeLog.load(runner.status.episode).end.steps == runner.status.step


def test_image_backlog_cannot_spoil_two_minutes_of_complete_telemetry(twin, tmp_path, monkeypatch):
    blocked, release = threading.Event(), threading.Event()
    original = EpisodeWriter._keep

    def slow(self, step, camera, image):
        blocked.set()
        assert release.wait(15.0)
        return original(self, step, camera, image)

    monkeypatch.setattr(EpisodeWriter, "_keep", slow)
    twin.connect("sim", threaded=False)
    image = np.zeros((48, 64, 3), np.uint8)

    def look(obs):
        obs.frames["camera"]
        return {}

    runner = StrategyRunner(twin, Plugin(look), cameras=("camera",),
                            frame=lambda _: (image, time.monotonic()),
                            episode=EpisodeOptions(folder=tmp_path, keep_frames_s=0))
    runner.start(threaded=False)
    try:
        assert runner.step_once(0.0)
        assert blocked.wait(2.0)
        drive(twin, runner, 120.0)
        writer = runner._episode_writer
        assert writer is not None
        assert writer.flush(wait=True, timeout=10.0), "image I/O blocked the telemetry writer"
        assert writer.recording["step_written"] == runner.status.step == 2401
        assert writer.recording["loop_written"] == 6000
        assert writer.recording["step_dropped"] == writer.recording["loop_dropped"] == 0
        assert writer.recording["frames_skipped"] > 0
        assert runner.status.running
        assert runner.status.episode_error == ""
        release.set()
        drained = threading.Event()

        def drain_images():
            writer._images.join()
            drained.set()

        threading.Thread(target=drain_images, daemon=True).start()
        assert drained.wait(5.0)
        twin.step(TWIN_PERIOD)
        assert runner.step_once(120.05)
    finally:
        release.set()
        runner.stop()
        assert runner.wait_for_episode(timeout=10.0)

    assert runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    assert log.recording["telemetry_complete"]
    assert len(log.steps) == 2402
    assert len(log.loop_t) == 6001
    assert log.recording["frames_requested"] == 2402
    assert log.steps[-1]["frames"][0]["file"], "image capture did not recover after its backlog drained"
    assert log.recording["frames_pending"] == 0
    assert sum(bool(frame.get("drop_reason")) for step in log.steps for frame in step["frames"]) == (
        log.recording["frames_skipped"] + log.recording["frames_failed"])


def test_an_image_encoding_failure_does_not_stop_later_frames_or_telemetry(twin, tmp_path, monkeypatch):
    twin.connect("sim", threaded=False)
    original = EpisodeWriter._keep

    def fails_once(self, step, camera, image):
        if step == 0:
            raise OSError("temporary image failure")
        return original(self, step, camera, image)

    monkeypatch.setattr(EpisodeWriter, "_keep", fails_once)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    frame = Frame(np.zeros((48, 64, 3), np.uint8), 0.0)
    writer.step(record(writer, twin, 0, frames={"camera": frame}))
    writer.step(record(writer, twin, 1, frames={"camera": frame}))
    assert writer.close(end(steps=2), wait=True)

    log = EpisodeLog.load(writer.path)
    assert writer.error == ""
    assert log.recording["telemetry_complete"]
    assert log.recording["frames_failed"] == 1
    assert log.recording["frames_saved"] == 1
    assert log.steps[0]["frames"][0]["file"] is None
    assert "temporary image failure" in log.steps[0]["frames"][0]["drop_reason"]
    assert (tmp_path / log.steps[1]["frames"][0]["file"]).exists()
    assert EpisodeLog.summary(writer.path).recording == log.recording


def test_stuck_image_cannot_prevent_a_complete_telemetry_end(twin, tmp_path, monkeypatch):
    from machinekind_twin import episode

    twin.connect("sim", threaded=False)
    blocked, release = threading.Event(), threading.Event()
    original = EpisodeWriter._keep

    def stuck(self, step, camera, image):
        blocked.set()
        assert release.wait(5.0)
        return original(self, step, camera, image)

    monkeypatch.setattr(EpisodeWriter, "_keep", stuck)
    monkeypatch.setattr(episode, "FRAME_CLOSE_TIMEOUT_S", 0.05)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    try:
        writer.step(record(writer, twin, 0, frames={"camera": Frame(np.zeros((12, 16, 3), np.uint8), 0.0)}))
        assert blocked.wait(2.0)
        writer.close(end(steps=1))
        assert writer.wait(timeout=2.0)
        log = EpisodeLog.load(writer.path)
        assert log.end.code == "stopped"
        assert log.recording["telemetry_complete"]
        assert log.recording["frames_pending"] == 1
        assert not log.recording["frames_complete"]
        assert log.steps[0]["frames"][0]["file"] is None
        assert "did not finish" in log.steps[0]["frames"][0]["drop_reason"]
        assert "telemetry complete" in "\n".join(report(log).lines())
    finally:
        release.set()
        assert writer._images_finished.wait(5.0)


def test_concurrent_producers_do_not_abort_recording(twin, tmp_path):
    from types import SimpleNamespace

    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    gate = threading.Barrier(4)

    def produce():
        gate.wait(timeout=2.0)
        for number in range(200):
            writer.on_tick(SimpleNamespace(t=twin.clock.now(), sent={}, measured={}, fresh=False, measured_t=0.0))

    producers = [threading.Thread(target=produce) for _ in range(4)]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join(timeout=3.0)
        assert not producer.is_alive()
    assert writer.close(end(), wait=True)

    log = EpisodeLog.load(writer.path)
    assert len(log.loop_t) == 800
    assert log.recording["loop_dropped"] == 0
    assert log.recording["telemetry_complete"]
    assert writer.error == ""


def test_camera_names_cannot_overwrite_each_others_frames(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    writer.step(record(writer, twin, 0, frames={
        name: Frame(np.full((12, 16, 3), value, np.uint8), 0.0)
        for name, value in (("a/b", 20), ("a_b", 50), ("A", 100), ("a", 200))
    }))
    assert writer.close(end(steps=1), wait=True)
    log = EpisodeLog.load(writer.path)
    files = [frame["file"] for frame in log.steps[0]["frames"]]
    assert len({file.lower() for file in files}) == 4
    assert all((tmp_path / file).exists() for file in files)
    assert log.recording["frames_saved"] == 4


def test_version_one_logs_remain_readable(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path)
    writer.step(record(writer, twin, 0))
    assert writer.close(end(steps=1), wait=True)
    records = lines(writer.path)
    records[0]["version"] = 1
    records = [record for record in records if record["kind"] not in ("frame", "recording")]
    records[-1].pop("recording", None)
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    assert len(EpisodeLog.load(path).steps) == 1
    assert EpisodeLog.load(path).recording == {}


def test_missing_camera_frames_are_explicit_and_later_images_are_saved(twin, tmp_path):
    twin.connect("sim", threaded=False)
    writer = observer(twin, tmp_path, keep_frames_s=0)
    writer.step(record(writer, twin, 0, frames={"camera": None}))
    writer.step(record(writer, twin, 1, frames={"camera": Frame(np.zeros((12, 16, 3), np.uint8), 0.0)}))
    assert writer.close(end(steps=2), wait=True)
    log = EpisodeLog.load(writer.path)
    assert log.recording["telemetry_complete"]
    assert log.recording["frames_requested"] == 2
    assert log.recording["frames_skipped"] == 1
    assert log.recording["frames_saved"] == 1
    assert log.steps[0]["frames"][0]["drop_reason"] == "camera frame unavailable"
    assert log.steps[1]["frames"][0]["file"]
