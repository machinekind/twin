"""The episode log: its writer and loader on the simulated twin.

Everything in simulated time: the twin's loop without a thread (`connect(threaded=False)` +
`Twin.step`), as in `test_strategy`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT

from machinekind_twin.clock import VirtualClock
from machinekind_twin.episode import (
    FORMAT,
    EpisodeEnd,
    EpisodeLog,
    EpisodeWriter,
    StepRecord,
    episodes_dir,
    twin_header,
)
from machinekind_twin.paths import data_path
from machinekind_twin.runtime import Twin
from machinekind_twin.strategy import Decision, Frame, ObjectPose
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    assert episodes_dir(tmp_path / "twin.json") == tmp_path / "episodes"
    assert episodes_dir(None) == data_path(Path("workspace") / "twin.json").parent / "episodes"
