"""The episode log: one JSON Lines file per run, written while the run goes on, and its loader.

    log = EpisodeLog.load(path)
    log.end.code, log.end.success, log.end.error      # how the run ended
    t, readings = log.readings()                        # (R,), (R, n): the arm's readings
    print("\\n".join(report(log).lines()))             # `episode_report`: latency and drift

Every run of `strategy.StrategyRunner` writes one file, `<folder>/<name>.jsonl`; the folder is
`episodes/` next to the workspace file (`episodes_dir`). Each line is one JSON object with a
`kind`: the `header` first, then a `loop` line for every tick of the twin's loop and a `step` line
for every tick the runner accepted, and the `end` last. A file without an `end` line is a run that
was cut off. Frames the strategy read can be kept as JPEG files in `<name>.frames/`.

The writer takes loop ticks through `Twin.add_tick_listener`. `EpisodeWriter.on_tick` runs on the
twin's loop thread and only appends to a deque; `step`, `flush` and `close` drain it into the file
from one thread at a time, which the runner guarantees with its lock. A log that cannot be written
stops writing and says why in `EpisodeWriter.error`: it never ends a run.

Every time in the file except `run_t` and the ages is in seconds on the twin's clock (`Twin.clock`)
since the writer opened, `header.t0` being that moment. Intent, command and readings thus share one
axis. `run_t` is `Observation.t`, the runner's own time, and `age_s` is a duration, as `Frame` and
`ObjectPose` give it.

Joints are in app units (degrees, the gripper 0..100), in `header.joints` order, with null for a
missing joint. Positions are metres and rotations 3x3 rows, in the robot's base frame. The format
needs nothing but the standard library and numpy to read.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Literal, TextIO

import machinekind_twin_api
import numpy as np
from machinekind_twin_api.strategy import Frame, ObjectPose

from . import __version__ as twin_version
from .paths import data_path

__all__ = [
    "FORMAT", "VERSION", "EndCode", "EpisodeEnd", "EpisodeLog", "EpisodeOptions", "EpisodeSummary",
    "EpisodeWriter", "StepRecord", "episodes_dir", "twin_header",
]

logger = logging.getLogger(__name__)

#: The header's `format`: what tells an episode log from any other JSON Lines file.
FORMAT = "machinekind-twin-episode"
#: The version this twin writes, and the newest it reads.
VERSION = 1

#: The workspace file's default place, as `workspace.DEFAULT_REL` (not imported: `workspace` pulls in MuJoCo).
DEFAULT_WORKSPACE = Path("workspace") / "twin.json"

#: How a run ended, next to its Polish reason. A log without an end line loads as "unfinished".
EndCode = Literal[
    "done", "gave_up", "time_limit", "disconnected", "stopped", "preempted", "bad_target", "error", "refused",
]

#: Quality of the kept JPEG frames: about 17 KB for 640x480.
JPEG_QUALITY = 85
#: An array with more elements than this is logged by its type and shape, not its values.
ARRAY_LIMIT = 64

JOINT_DIGITS = 4
SECOND_DIGITS = 6
METRE_DIGITS = 6


def episodes_dir(workspace_path: str | Path | None = None) -> Path:
    """The folder of the logs: `episodes/` next to the workspace file, the default one when None."""
    return Path(workspace_path or data_path(DEFAULT_WORKSPACE)).parent / "episodes"


def twin_header(twin: Any) -> dict[str, Any]:
    """The header fields the twin knows: the robot, its limits, the backend and the workspace file."""
    spec = twin.workspace.spec()
    workspace = twin.workspace

    return {
        "robot": spec.name,
        "folder": str(spec.root),
        "joints": list(spec.joints),
        "gripper": spec.gripper,
        "limits": {name: list(limits) for name, limits in twin.joint_limits().items()},
        "max_vel": dict(twin.speed_limits()),
        "simulated": bool(twin.status.simulated),
        "backend": str(twin.status.backend),
        "loop_hz": float(twin.loop_hz),
        "workspace": {
            "path": None if workspace.path is None else str(workspace.path),
            "calib_preset": workspace.calib_preset,
        },
    }


@dataclass(frozen=True)
class EpisodeOptions:
    """How a run's log is written."""

    #: Where the file goes; None = `episodes_dir` of the twin's workspace file.
    folder: Path | None = None
    #: The file's name without `.jsonl`; None = the start time, `20261024-153012`.
    name: str | None = None
    #: The caller's JSON values for the header: the plugin and its settings, a run number, a seed.
    meta: Mapping[str, Any] = field(default_factory=dict)
    #: None keeps references to frames only. A number keeps at most one JPEG per camera per that
    #: many seconds; 0 keeps every frame the strategy read.
    keep_frames_s: float | None = None
    #: The label of the camera-seen object that marks the gripper, for the drift report.
    gripper: str | None = None


@dataclass(frozen=True)
class StepRecord:
    """One tick the runner accepted, or one tick of an observer. Times are on the log's clock."""

    step: int
    #: `Observation.t`; None without a runner.
    run_t: float | None
    #: Just before the observation was built.
    obs_t: float
    #: Right after the target was handed to the twin.
    t: float
    #: The readings and the supervisor's command the strategy saw.
    joints: Mapping[str, float]
    command: Mapping[str, float]
    #: The strategy's checked target, or a foreign controller's command; empty = keep driving.
    intent: Mapping[str, float]
    goal: np.ndarray | None
    objects: Mapping[str, ObjectPose]
    #: Only the cameras the strategy read in this tick; None = that camera had no frame.
    frames: Mapping[str, Frame | None]
    done: bool = False
    success: bool | None = None
    error: float | None = None
    note: str = ""
    #: `Decision`s, or any objects with their attributes.
    decisions: Sequence[Any] = ()


@dataclass(frozen=True)
class EpisodeEnd:
    """How a run ended."""

    #: An `EndCode`, or "unfinished" for a log without an end line.
    code: str
    #: `RunnerStatus.stopped_because`, in Polish.
    reason: str
    #: The log's clock at the end, which is the run's length; None when unknown.
    t: float | None
    done: bool
    success: bool | None
    error: float | None
    note: str
    steps: int
    #: `strategy.start` returned.
    began: bool

    @property
    def succeeded(self) -> bool:
        return self.code == "done"


class EpisodeWriter:
    """Writes one log. `create` opens it; the runner calls `step` per accepted tick and `close` once."""

    def __init__(self, path: Path, file: TextIO, twin: Any, t0: float, joints: Sequence[str],
                 keep_frames_s: float | None):
        #: The file.
        self.path = path
        #: Why the log stopped being written; "" while writing works.
        self.error = ""

        self._file = file
        self._twin = twin
        self._clock = twin.clock
        self._t0 = t0
        self._joints = tuple(joints)
        self._keep_frames_s = keep_frames_s
        self._ticks: collections.deque[tuple[float, dict[str, float], dict[str, float] | None, float,
                                             tuple[str, ...]]] = collections.deque()
        self._handle: int | None = None
        self._closed = False
        self._kept_at: dict[str, float] = {}

    @classmethod
    def create(cls, folder: str | Path, header: Mapping[str, Any], twin: Any, *, name: str | None = None,
               keep_frames_s: float | None = None) -> EpisodeWriter:
        """Opens a new file in `folder`, writes the header and starts taking loop ticks. Raises OSError only.

        `header` holds the fields `twin_header` gives plus the run's own (`mode`, `owner`,
        `strategy`, `cameras`, `gripper_label`, `meta`); this adds the format, the name, the start
        time, the clock and the versions. A name taken already gets `-2`, `-3`, ...: nothing is
        overwritten.
        """
        local = time.localtime()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path, file = _open_new(folder, _safe(name) if name else time.strftime("%Y%m%d-%H%M%S", local))

        t0 = float(twin.clock.now())
        joints = [str(joint) for joint in header.get("joints", ())]
        writer = cls(path, file, twin, t0, joints, keep_frames_s)

        own = {
            "kind": "header", "format": FORMAT, "version": VERSION, "name": path.stem,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", local),
            "clock": "virtual" if getattr(twin.clock, "virtual", False) else "wall", "t0": t0,
        }
        line = {**own, **_HEADER_DEFAULTS, **header, **own, "joints": joints, "keep_frames_s": keep_frames_s,
                "versions": {"twin": twin_version, "api": machinekind_twin_api.__version__}}
        try:
            writer._write(line)
            file.flush()
        except OSError:
            file.close()
            raise

        writer._handle = twin.add_tick_listener(writer.on_tick)
        return writer

    def now(self) -> float:
        """The log's clock: seconds on the twin's clock since the writer opened."""
        return self._clock.now() - self._t0

    def since(self, t: float) -> float:
        """A time on the twin's clock, on the log's clock."""
        return t - self._t0

    def on_tick(self, sample: Any) -> None:
        """The tick listener: runs on the twin's loop thread, so it only appends."""
        self._ticks.append((sample.t, sample.sent, sample.measured if sample.fresh else None, sample.measured_t,
                            tuple(self._twin.status.at_limit)))

    def step(self, record: StepRecord) -> None:
        """Writes the loop ticks gathered so far, the frames due and the step's line."""
        if self._closed or self.error:
            return

        try:
            self._drain()
            frames = [self._frame(record, camera, frame) for camera, frame in record.frames.items()]
            self._write({
                "kind": "step",
                "step": int(record.step),
                "run_t": _rounded(record.run_t, SECOND_DIGITS),
                "obs_t": _rounded(record.obs_t, SECOND_DIGITS),
                "t": _rounded(record.t, SECOND_DIGITS),
                "joints": self._vector(record.joints),
                "command": self._vector(record.command),
                "intent": {str(name): _rounded(value, JOINT_DIGITS) for name, value in record.intent.items()},
                "goal": _rounded(record.goal, METRE_DIGITS),
                "objects": {str(label): _pose(pose) for label, pose in record.objects.items()},
                "frames": frames,
                "done": bool(record.done),
                "success": None if record.success is None else bool(record.success),
                "error": _rounded(record.error, SECOND_DIGITS),
                "note": str(record.note),
                "decisions": [_decision(decision) for decision in record.decisions],
            })
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 - a broken log must never end a run or stop the arm
            self._broken(exc)

    def flush(self) -> None:
        """Writes the loop ticks gathered so far: for an observer that writes no steps for a while."""
        if self._closed or self.error:
            return

        try:
            self._drain()
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 - a broken log must never end a run or stop the arm
            self._broken(exc)

    def close(self, end: EpisodeEnd) -> None:
        """Stops taking ticks, writes the rest and the end line, and closes the file. Once."""
        if self._closed:
            return
        self._closed = True
        self._detach()

        try:
            if not self.error:
                self._drain()
                self._write({
                    "kind": "end",
                    "t": _rounded(end.t, SECOND_DIGITS),
                    "code": end.code,
                    "reason": end.reason,
                    "done": bool(end.done),
                    "success": None if end.success is None else bool(end.success),
                    "error": _rounded(end.error, SECOND_DIGITS),
                    "note": end.note,
                    "steps": int(end.steps),
                    "began": bool(end.began),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                self._file.flush()
                os.fsync(self._file.fileno())
        except Exception as exc:  # noqa: BLE001 - a broken log must never end a run or stop the arm
            self._broken(exc)
        finally:
            try:
                self._file.close()
            except OSError as exc:
                self._broken(exc)

    # ------------------------------------------------------------------ inside

    def _drain(self) -> None:
        while self._ticks:
            t, sent, measured, measured_t, at_limit = self._ticks.popleft()
            self._write({
                "kind": "loop",
                "t": _rounded(t - self._t0, SECOND_DIGITS),
                "sent": self._vector(sent) if sent else None,
                "measured": None if measured is None else self._vector(measured),
                "measured_t": None if measured is None else _rounded(measured_t - self._t0, SECOND_DIGITS),
                "at_limit": list(at_limit),
            })

    def _vector(self, values: Mapping[str, float]) -> list[float | None]:
        return [_rounded(values.get(joint), JOINT_DIGITS) for joint in self._joints]

    def _frame(self, record: StepRecord, camera: str, frame: Frame | None) -> dict[str, Any]:
        if frame is None:
            return {"camera": camera, "age_s": None, "shape": None, "file": None}

        image = np.asarray(frame.image)
        return {
            "camera": camera,
            "age_s": _rounded(frame.age_s, SECOND_DIGITS),
            "shape": list(image.shape),
            "file": self._keep(record, camera, image),
        }

    def _keep(self, record: StepRecord, camera: str, image: np.ndarray) -> str | None:
        """Writes the frame as a JPEG when one is due for this camera; its path next to the log."""
        every = self._keep_frames_s
        if every is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return None

        kept_at = self._kept_at.get(camera)
        if kept_at is not None and record.t - kept_at < every:
            return None

        import cv2

        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        encoded, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not encoded:
            return None

        folder = self.path.with_suffix(".frames")
        folder.mkdir(exist_ok=True)
        name = f"{record.step:06d}-{_safe(camera)}.jpg"
        (folder / name).write_bytes(jpeg.tobytes())
        self._kept_at[camera] = record.t

        return f"{folder.name}/{name}"

    def _write(self, line: Mapping[str, Any]) -> None:
        text = json.dumps(_clean(line), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._file.write(text + "\n")

    def _broken(self, exc: BaseException) -> None:
        if not self.error:
            logger.warning("Zapis przebiegu %s przerwany: %s", self.path, exc)
            self.error = f"zapis przebiegu przerwany: {exc}"
        self._detach()
        self._ticks.clear()

    def _detach(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._twin.remove_tick_listener(handle)


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class EpisodeSummary:
    """A log's header and end, read without the lines between them."""

    path: Path
    header: dict[str, Any]
    end: EpisodeEnd


@dataclass
class EpisodeLog:
    """A log read back. The loop's columns follow `joints`; NaN marks a value the tick did not have."""

    path: Path
    header: dict[str, Any]
    joints: tuple[str, ...]
    #: Loop ticks: (L,) times, (L, n) what went to the servos (NaN where nothing was sent), (L, n)
    #: readings and (L,) their times (NaN where the tick had no fresh reading), (L, n) joints at a limit.
    loop_t: np.ndarray
    sent: np.ndarray
    measured: np.ndarray
    measured_t: np.ndarray
    at_limit: np.ndarray
    #: Step lines as written.
    steps: list[dict[str, Any]]
    end: EpisodeEnd

    @staticmethod
    def load(path: str | Path) -> EpisodeLog:
        """Reads a log; ValueError when it is not one, is newer than this twin, or is broken mid-file.

        A last line cut off mid-write is skipped, and a missing end line gives the code "unfinished".
        """
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        records = _parse(path, lines)
        header = _header(path, records[0] if records else None)

        joints = tuple(header["joints"])
        loops = [record for record in records if record.get("kind") == "loop"]
        steps = [record for record in records if record.get("kind") == "step"]
        ends = [record for record in records if record.get("kind") == "end"]

        width = len(joints)
        loop_t = np.array([_number(record.get("t")) for record in loops], float)
        sent = _rows([record.get("sent") for record in loops], width)
        measured = _rows([record.get("measured") for record in loops], width)
        measured_t = np.array([_number(record.get("measured_t")) for record in loops], float)
        at_limit = np.array([[joint in (record.get("at_limit") or ()) for joint in joints] for record in loops],
                            bool).reshape(len(loops), width)

        return EpisodeLog(path=path, header=header, joints=joints, loop_t=loop_t, sent=sent, measured=measured,
                          measured_t=measured_t, at_limit=at_limit, steps=steps,
                          end=_end(ends[-1] if ends else None, steps))

    @staticmethod
    def summary(path: str | Path, tail_bytes: int = 65536) -> EpisodeSummary:
        """The header and the end of a log, from its first line and its last `tail_bytes`."""
        path = Path(path)
        with path.open("rb") as file:
            first = file.readline().decode("utf-8", errors="replace")
            size = file.seek(0, os.SEEK_END)
            file.seek(max(0, size - tail_bytes))
            tail = file.read().decode("utf-8", errors="replace").splitlines()

        header = _header(path, _loads(first))
        end_line = None
        last_step = []
        for line in reversed(tail):
            record = _loads(line) or {}
            if record.get("kind") == "end":
                end_line = record
                break
            if record.get("kind") == "step":
                last_step = [record]
                break

        return EpisodeSummary(path, header, _end(end_line, last_step))

    def readings(self) -> tuple[np.ndarray, np.ndarray]:
        """The fresh readings: (R,) their times and (R, n) their values, each reading once."""
        rows = self._reading_rows()
        return self.measured_t[rows], self.measured[rows]

    def commands(self) -> tuple[np.ndarray, np.ndarray]:
        """What went to the servos: (C,) loop times and (C, n) values, only ticks that sent something."""
        rows = self._command_rows()
        return self.loop_t[rows], self.sent[rows]

    def intents(self) -> tuple[np.ndarray, np.ndarray]:
        """The intents: (E,) step times and (E, n) values, NaN where the step did not name the joint."""
        t = np.array([_number(step.get("t")) for step in self.steps], float)
        values = np.array([[_number((step.get("intent") or {}).get(joint)) for joint in self.joints]
                           for step in self.steps], float).reshape(len(self.steps), len(self.joints))
        return t, values

    def _reading_rows(self) -> np.ndarray:
        fresh = np.isfinite(self.measured_t) & np.isfinite(self.measured).any(axis=1)
        rows = np.flatnonzero(fresh)
        _, first = np.unique(self.measured_t[rows], return_index=True)
        return rows[np.sort(first)]

    def _command_rows(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.sent).any(axis=1))


# --------------------------------------------------------------------------- helpers


#: Header keys every log has, whoever writes it: a missing value is null, never a missing key.
_HEADER_DEFAULTS: dict[str, Any] = {
    "mode": "strategy", "robot": "", "folder": None, "joints": [], "gripper": None, "limits": {},
    "max_vel": {}, "simulated": None, "backend": "", "loop_hz": None, "owner": "", "strategy": None,
    "cameras": [], "gripper_label": None, "workspace": {"path": None, "calib_preset": None}, "meta": {},
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(name: str) -> str:
    """A name for a file: characters outside letters, digits, `.`, `_` and `-` become `_`."""
    return _UNSAFE.sub("_", name)


def _open_new(folder: Path, stem: str) -> tuple[Path, TextIO]:
    for number in itertools.count(1):
        path = folder / (f"{stem}.jsonl" if number == 1 else f"{stem}-{number}.jsonl")
        try:
            return path, path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def _rounded(value: Any, digits: int) -> Any:
    """A number, or an array of numbers as nested lists, rounded; None for a missing or non-finite one."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        number = float(array)
        return round(number, digits) if math.isfinite(number) else None
    return _clean(np.round(array, digits).tolist())


def _pose(pose: Any) -> dict[str, Any]:
    return {
        "position": _rounded(getattr(pose, "position", None), METRE_DIGITS),
        "rotation": _rounded(getattr(pose, "rotation", None), METRE_DIGITS),
        "source": str(getattr(pose, "source", "")),
        "age_s": _rounded(getattr(pose, "age_s", None), SECOND_DIGITS),
    }


def _decision(decision: Any) -> dict[str, Any]:
    """By attributes: a plugin's own decision class with `Decision`'s attribute names works too."""
    state = getattr(decision, "state", None)
    return {
        "question": str(getattr(decision, "question", "")),
        "answer": str(getattr(decision, "answer", "")),
        "state": _clean(state) if state is not None else {},
        "confidence": _rounded(getattr(decision, "confidence", None), SECOND_DIGITS),
        "response_s": _rounded(getattr(decision, "response_s", None), SECOND_DIGITS),
    }


def _clean(value: Any, depth: int = 0) -> Any:
    """`value` as strict JSON values. It never raises: a non-finite number becomes null, a large
    array its type and shape, and anything JSON has no type for its `repr`."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if depth > 32:
        return repr(value)
    if isinstance(value, np.ndarray):
        if value.size > ARRAY_LIMIT:
            return f"<ndarray {value.dtype} {value.shape}>"
        return _clean(value.tolist(), depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _clean(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    return repr(value)


def _loads(line: str) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _parse(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    numbered = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    records = []

    for index, (number, line) in enumerate(numbered):
        record = _loads(line)
        if record is not None:
            records.append(record)
        elif index == 0:
            raise ValueError(f"{path}: to nie jest zapis przebiegu blizniaka")
        elif index < len(numbered) - 1:
            raise ValueError(f"{path}:{number}: uszkodzona linia")

    return records


def _header(path: Path, record: dict[str, Any] | None) -> dict[str, Any]:
    record = record or {}
    version = record.get("version")
    if record.get("kind") != "header" or record.get("format") != FORMAT or not isinstance(version, int):
        raise ValueError(f"{path}: to nie jest zapis przebiegu blizniaka")
    if version > VERSION:
        raise ValueError(f"{path}: zapis w wersji {version} - ten blizniak czyta do {VERSION}")

    return {**_HEADER_DEFAULTS, **record, "joints": [str(joint) for joint in record.get("joints") or ()]}


def _end(record: dict[str, Any] | None, steps: list[dict[str, Any]]) -> EpisodeEnd:
    if record is None:
        last = steps[-1] if steps else {}
        return EpisodeEnd(code="unfinished", reason="", t=None, done=bool(last.get("done", False)),
                          success=last.get("success"), error=_optional(last.get("error")),
                          note=str(last.get("note", "")), steps=int(last.get("step", -1)) + 1, began=bool(steps))

    return EpisodeEnd(code=str(record.get("code", "")), reason=str(record.get("reason", "")),
                      t=_optional(record.get("t")), done=bool(record.get("done", False)),
                      success=record.get("success"), error=_optional(record.get("error")),
                      note=str(record.get("note", "")), steps=int(record.get("steps", 0)),
                      began=bool(record.get("began", False)))


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan


def _optional(value: Any) -> float | None:
    number = _number(value)
    return number if math.isfinite(number) else None


def _rows(vectors: list[Any], width: int) -> np.ndarray:
    """(L, width) floats; NaN for null, and a whole row of NaN for a missing or misshapen vector."""
    rows = np.full((len(vectors), width), math.nan)
    for index, vector in enumerate(vectors):
        if isinstance(vector, list) and len(vector) == width:
            rows[index] = [_number(value) for value in vector]
    return rows
