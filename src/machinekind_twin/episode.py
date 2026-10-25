"""Asynchronous episode recording and loaders for JSON Lines and Zstandard logs.

See docs/EPISODES.md for recording options, the record format and report limitations.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, BinaryIO, Literal

import machinekind_twin_api
import numpy as np
import zstandard as zstd
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

#: Stable termination codes. A log without a complete end record loads as "unfinished".
EndCode = Literal[
    "done", "gave_up", "time_limit", "disconnected", "stopped", "preempted", "bad_target", "error", "refused",
]

#: Default quality for saved camera images.
FRAME_QUALITY = 85
#: Maximum queued events per writer, including loop samples and strategy steps.
MAX_PENDING = 1024
MAX_PENDING_IMAGE_BYTES = 64 * 1024 * 1024
FLUSH_INTERVAL_S = 0.25
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
    #: The file's name without `.jsonl.zst`; None = the start time, `20261024-153012`.
    name: str | None = None
    #: The caller's JSON values for the header: the plugin and its settings, a run number, a seed.
    meta: Mapping[str, Any] = field(default_factory=dict)
    #: None saves camera metadata only. A number saves at most one image per camera per that
    #: many seconds; 0 keeps every frame the strategy read.
    keep_frames_s: float | None = None
    frame_format: Literal["webp", "jpeg"] = "jpeg"
    frame_quality: int = FRAME_QUALITY
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
    #: Free-form termination detail from `RunnerStatus.stopped_because`.
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
    """Records snapshots on a bounded queue; one worker owns compression and filesystem I/O.

    `close` queues the end record. Call `wait` when the completed file is needed immediately.
    Errors disable recording without stopping the strategy. Queue overflow leaves an unfinished log.
    """

    def __init__(self, folder: Path, header: Mapping[str, Any], twin: Any, name: str,
                 keep_frames_s: float | None, frame_format: Literal["webp", "jpeg"], frame_quality: int,
                 on_update: Callable[[EpisodeWriter], None] | None):
        self.path = folder / f"{name}.jsonl.zst"
        self.error = ""
        self.opened = False
        self._twin = twin
        self._clock = twin.clock
        self._t0 = float(self._clock.now())
        self._joints = tuple(str(joint) for joint in header.get("joints", ()))
        self._keep_frames_s = keep_frames_s
        self._frame_format = frame_format
        self._frame_quality = frame_quality
        self._kept_at: dict[str, float] = {}
        self._on_update = on_update
        self._condition = threading.Condition()
        self._pending: collections.deque[tuple[str, Any]] = collections.deque()
        self._pending_image_bytes = 0
        self._flush_waiters: set[threading.Event] = set()
        self._closed = False
        self._end: EpisodeEnd | None = None
        self._file: BinaryIO | None = None
        self._compressor: Any = None
        self._handle: int | None = twin.add_tick_listener(self.on_tick)
        local = time.localtime()
        own = {
            "kind": "header", "format": FORMAT, "version": VERSION, "name": name,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", local),
            "clock": "virtual" if getattr(twin.clock, "virtual", False) else "wall", "t0": self._t0,
        }
        self._header = {**_HEADER_DEFAULTS, **_clean(header), **own, "joints": list(self._joints),
                        "keep_frames_s": keep_frames_s, "frame_format": frame_format, "frame_quality": frame_quality,
                        "versions": {"twin": twin_version, "api": machinekind_twin_api.__version__}}
        self._thread = threading.Thread(target=self._work, name="episode-writer", daemon=True)
        self._thread.start()

    @classmethod
    def create(cls, folder: str | Path, header: Mapping[str, Any], twin: Any, *, name: str | None = None,
               keep_frames_s: float | None = None, frame_format: Literal["webp", "jpeg"] = "jpeg",
               frame_quality: int = FRAME_QUALITY,
               on_update: Callable[[EpisodeWriter], None] | None = None) -> EpisodeWriter:
        """Start recording without waiting for disk access. `path` is final once `opened` is true."""
        if keep_frames_s is not None and (not math.isfinite(keep_frames_s) or keep_frames_s < 0):
            raise ValueError("keep_frames_s must be finite and non-negative")
        if frame_format not in ("webp", "jpeg"):
            raise ValueError("frame_format must be webp or jpeg")
        if isinstance(frame_quality, bool) or not isinstance(frame_quality, int) or not 1 <= frame_quality <= 100:
            raise ValueError("frame_quality must be an integer between 1 and 100")
        return cls(Path(folder), header, twin, _safe(name) if name else time.strftime("%Y%m%d-%H%M%S"),
                   keep_frames_s, frame_format, frame_quality, on_update)

    def now(self) -> float:
        """Seconds on the twin's clock since recording was requested."""
        return self._clock.now() - self._t0

    def on_tick(self, sample: Any) -> None:
        """Capture a loop sample without serializing data or accessing the filesystem."""
        self._put("loop", (sample.t, dict(sample.sent), dict(sample.measured) if sample.fresh else None,
                           sample.measured_t, tuple(self._twin.status.at_limit)))

    def step(self, record: StepRecord) -> None:
        """Snapshot an accepted action. Only images due for retention are copied."""
        if self._closed or self.error:
            return
        try:
            frames, images = [], []
            for camera, frame in record.frames.items():
                item: dict[str, Any] = {"camera": camera, "age_s": None, "shape": None, "file": None}
                if frame is not None:
                    image = np.asarray(frame.image)
                    item.update(age_s=_rounded(frame.age_s, SECOND_DIGITS), shape=list(image.shape))
                    every = self._keep_frames_s
                    previous = self._kept_at.get(camera)
                    if (every is not None and (previous is None or record.t - previous >= every)
                            and image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8):
                        images.append((len(frames), camera, image.copy()))
                        self._kept_at[camera] = record.t
                frames.append(item)
            line = {
                "kind": "step", "step": int(record.step), "run_t": _rounded(record.run_t, SECOND_DIGITS),
                "obs_t": _rounded(record.obs_t, SECOND_DIGITS), "t": _rounded(record.t, SECOND_DIGITS),
                "joints": self._vector(record.joints), "command": self._vector(record.command),
                "intent": {str(name): _rounded(value, JOINT_DIGITS) for name, value in record.intent.items()},
                "goal": _rounded(record.goal, METRE_DIGITS),
                "objects": {str(label): _pose(pose) for label, pose in record.objects.items()},
                "frames": frames, "done": bool(record.done),
                "success": None if record.success is None else bool(record.success),
                "error": _rounded(record.error, SECOND_DIGITS), "note": str(record.note),
                "decisions": [_decision(decision) for decision in record.decisions],
            }
            self._put("step", (line, images))
        except Exception as exc:  # noqa: BLE001 - recording errors must not interrupt control
            self._broken(exc)

    def flush(self, *, wait: bool = False, timeout: float | None = None) -> bool:
        """Request a flush; optionally wait for all previously queued events to reach the file."""
        completed = threading.Event()
        with self._condition:
            self._flush_waiters.add(completed)
        if not self._put("flush", completed):
            with self._condition:
                self._flush_waiters.discard(completed)
            return False
        if not wait:
            return True
        finished = completed.wait(timeout)
        with self._condition:
            self._flush_waiters.discard(completed)
        return finished and not self.error

    def close(self, end: EpisodeEnd, *, wait: bool = True) -> None:
        """Stop accepting events and queue the end record; optionally wait for durable completion."""
        with self._condition:
            if not self._closed:
                self._closed = True
                self._end = end
                self._condition.notify()
        self._detach()
        if wait:
            self.wait()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a closed or failed writer. False means the worker is still running."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _put(self, kind: str, value: Any) -> bool:
        with self._condition:
            if self._closed or self.error:
                return False
            image_bytes = sum(image.nbytes for _, _, image in value[1]) if kind == "step" else 0
            if (len(self._pending) < MAX_PENDING
                    and self._pending_image_bytes + image_bytes <= MAX_PENDING_IMAGE_BYTES):
                self._pending_image_bytes += image_bytes
                self._pending.append((kind, value))
                self._condition.notify()
                return True
        self._broken(RuntimeError("episode queue capacity exceeded"))
        return False

    def _work(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path, self._file = _open_new(self.path.parent, self._header["name"])
            self._header["name"] = self.path.name.removesuffix(".jsonl.zst")
            self._compressor = zstd.ZstdCompressor(level=1, write_checksum=True).compressobj()
            try:
                self._write(self._header)
                self._flush()
            except Exception:
                self._file.close()
                self.path.unlink(missing_ok=True)
                raise
            self.opened = True
            self._notify()
            flushed = time.monotonic()
            while True:
                with self._condition:
                    if not self._pending and not self._closed and not self.error:
                        self._condition.wait(FLUSH_INTERVAL_S)
                    if self.error:
                        break
                    event = self._pending.popleft() if self._pending else None
                    if event is not None and event[0] == "step":
                        self._pending_image_bytes -= sum(image.nbytes for _, _, image in event[1][1])
                    if event is None and self._closed:
                        break
                if event is not None:
                    kind, value = event
                    if kind == "loop":
                        t, sent, measured, measured_t, at_limit = value
                        self._write({
                            "kind": "loop", "t": _rounded(t - self._t0, SECOND_DIGITS),
                            "sent": self._vector(sent) if sent else None,
                            "measured": None if measured is None else self._vector(measured),
                            "measured_t": None if measured is None else _rounded(measured_t - self._t0,
                                                                                 SECOND_DIGITS),
                            "at_limit": list(at_limit),
                        })
                    elif kind == "step":
                        line, images = value
                        for index, camera, image in images:
                            line["frames"][index]["file"] = self._keep(line["step"], camera, image)
                        self._write(line)
                    elif kind == "flush":
                        self._flush()
                        with self._condition:
                            self._flush_waiters.discard(value)
                        value.set()
                if time.monotonic() - flushed >= FLUSH_INTERVAL_S:
                    self._flush()
                    flushed = time.monotonic()
            end = self._end
            if not self.error and end is not None:
                self._write({
                    "kind": "end", "t": _rounded(end.t, SECOND_DIGITS), "code": end.code,
                    "reason": end.reason, "done": bool(end.done),
                    "success": None if end.success is None else bool(end.success),
                    "error": _rounded(end.error, SECOND_DIGITS), "note": end.note,
                    "steps": int(end.steps), "began": bool(end.began),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            self._file.write(self._compressor.flush())
            self._file.flush()
            os.fsync(self._file.fileno())
        except Exception as exc:  # noqa: BLE001 - recording errors must not interrupt control
            self._broken(exc)
        finally:
            self._detach()
            if self._file is not None:
                try:
                    self._file.close()
                except OSError as exc:
                    self._broken(exc)
            with self._condition:
                self._pending.clear()
                self._pending_image_bytes = 0
                waiters = tuple(self._flush_waiters)
                self._flush_waiters.clear()
            for completed in waiters:
                completed.set()
            if self.error:
                logger.warning("Episode recording %s failed: %s", self.path, self.error)
            self._notify()

    def _flush(self) -> None:
        assert self._file is not None
        self._file.write(self._compressor.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK))
        self._file.flush()

    def _vector(self, values: Mapping[str, float]) -> list[float | None]:
        return [_rounded(values.get(joint), JOINT_DIGITS) for joint in self._joints]

    def _keep(self, step: int, camera: str, image: np.ndarray) -> str | None:
        import cv2

        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        extension = ".webp" if self._frame_format == "webp" else ".jpg"
        quality_flag = cv2.IMWRITE_WEBP_QUALITY if self._frame_format == "webp" else cv2.IMWRITE_JPEG_QUALITY
        encoded, data = cv2.imencode(extension, bgr, [int(quality_flag), self._frame_quality])
        if not encoded:
            return None
        folder = self.path.parent / (self.path.name.removesuffix(".jsonl.zst") + ".frames")
        folder.mkdir(exist_ok=True)
        name = f"{step:06d}-{_safe(camera)}{extension}"
        (folder / name).write_bytes(data.tobytes())
        return f"{folder.name}/{name}"

    def _write(self, line: Mapping[str, Any]) -> None:
        assert self._file is not None
        data = (json.dumps(_clean(line), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n").encode("utf-8")
        self._file.write(self._compressor.compress(data))

    def _broken(self, exc: BaseException) -> None:
        with self._condition:
            if not self.error:
                self.error = f"episode recording failed: {exc}"
            self._condition.notify()
        self._notify()

    def _notify(self) -> None:
        if self._on_update is not None:
            self._on_update(self)

    def _detach(self) -> None:
        with self._condition:
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

        A last line cut off mid-write is skipped, even inside a character, and a missing end line
        gives the code "unfinished".
        """
        path = Path(path)
        reader = _LogReader(path)
        records = _parse(path, list(reader))
        if not reader.complete:
            records = [record for record in records if record.get("kind") != "end"]
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
    def summary(path: str | Path) -> EpisodeSummary:
        """Stream the header and final status without retaining loop and step records."""
        path = Path(path)
        reader = _LogReader(path)
        header = None
        last = None
        last_step = []
        previous_bad = False
        for number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            if previous_bad:
                raise ValueError(f"{path}:{number - 1}: corrupt record")
            record = _loads(line)
            if header is None:
                header = _header(path, record)
            elif record is not None and record.get("kind") == "end":
                last = record
            if record is not None and record.get("kind") == "step":
                last_step = [record]
            previous_bad = record is None
        return EpisodeSummary(path=path, header=_header(path, header),
                              end=_end(last if reader.complete else None, last_step))

    def readings(self) -> tuple[np.ndarray, np.ndarray]:
        """The fresh readings: (R,) their times and (R, n) their values, each reading once."""
        rows = self.reading_rows()
        return self.measured_t[rows], self.measured[rows]

    def commands(self) -> tuple[np.ndarray, np.ndarray]:
        """What went to the servos: (C,) loop times and (C, n) values, only ticks that sent something."""
        rows = self.command_rows()
        return self.loop_t[rows], self.sent[rows]

    def intents(self) -> tuple[np.ndarray, np.ndarray]:
        """The intents: (E,) step times and (E, n) values, NaN where the step did not name the joint."""
        t = np.array([_number(step.get("t")) for step in self.steps], float)
        values = np.array([[_number((step.get("intent") or {}).get(joint)) for joint in self.joints]
                           for step in self.steps], float).reshape(len(self.steps), len(self.joints))
        return t, values

    def reading_rows(self) -> np.ndarray:
        """The loop rows of `readings`: the first row of each fresh reading."""
        fresh = np.isfinite(self.measured_t) & np.isfinite(self.measured).any(axis=1)
        rows = np.flatnonzero(fresh)
        _, first = np.unique(self.measured_t[rows], return_index=True)
        return rows[np.sort(first)]

    def command_rows(self) -> np.ndarray:
        """The loop rows of `commands`."""
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


def _open_new(folder: Path, stem: str) -> tuple[Path, BinaryIO]:
    for number in itertools.count(1):
        path = folder / (f"{stem}.jsonl.zst" if number == 1 else f"{stem}-{number}.jsonl.zst")
        try:
            return path, path.open("xb")
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
    """By attributes: a plugin's own decision class with `Decision`'s attribute names works too.

    A decision whose attributes cannot be read is logged by its type, and the run's log goes on.
    """
    try:
        state = getattr(decision, "state", None)
        return {
            "question": str(getattr(decision, "question", "")),
            "answer": str(getattr(decision, "answer", "")),
            "state": _clean(state) if state is not None else {},
            "confidence": _rounded(getattr(decision, "confidence", None), SECOND_DIGITS),
            "response_s": _rounded(getattr(decision, "response_s", None), SECOND_DIGITS),
        }
    except Exception:  # noqa: BLE001 - a decision the log cannot read costs only that decision
        return {"question": _placeholder(decision), "answer": "", "state": {}, "confidence": None,
                "response_s": None}


def _clean(value: Any, depth: int = 0) -> Any:
    """`value` as strict JSON values. It never raises: a non-finite number becomes null, a large
    array its type and shape, and anything JSON has no type for its `repr`. A value that cannot be
    read, such as a mapping another thread changes or an object whose `repr` raises, becomes its
    type in angle brackets, and only that value is lost."""
    try:
        return _cleaned(value, depth)
    except Exception:  # noqa: BLE001 - a value the log cannot read costs only that value
        return _placeholder(value)


def _cleaned(value: Any, depth: int) -> Any:
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
        return {str(key): _clean(item, depth + 1) for key, item in list(value.items())}
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in list(value)]
    if isinstance(value, PurePath):
        return str(value)
    return repr(value)


def _placeholder(value: Any) -> str:
    return f"<{type(value).__name__}>"


class _LogReader:
    """Stream complete JSON lines; track whether a compressed file has its final checksum."""

    def __init__(self, path: Path):
        self.path = path
        self.complete = True

    def __iter__(self) -> Iterator[bytes]:
        decoder = zstd.ZstdDecompressor().decompressobj() if self.path.suffix == ".zst" else None
        pending = b""
        with self.path.open("rb") as file:
            while data := file.read(65536):
                try:
                    if decoder is not None:
                        if decoder.eof:
                            raise ValueError(f"{self.path}: unexpected data after Zstandard frame")
                        data = decoder.decompress(data)
                        if decoder.unused_data:
                            raise ValueError(f"{self.path}: unexpected data after Zstandard frame")
                except zstd.ZstdError as exc:
                    raise ValueError(f"{self.path}: corrupt Zstandard log: {exc}") from exc
                parts = (pending + data).split(b"\n")
                pending = parts.pop()
                yield from parts
        self.complete = decoder is None or decoder.eof
        if pending:
            yield pending


def _loads(line: bytes) -> dict[str, Any] | None:
    """One line as a record; None for a line that is not a JSON object, or not UTF-8.

    Lines are split on "\\n" only: JSON keeps U+2028, U+2029 and U+0085 inside a string as they
    are, and `str.splitlines` would split there.
    """
    try:
        record = json.loads(line.decode("utf-8"))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _parse(path: Path, lines: list[bytes]) -> list[dict[str, Any]]:
    numbered = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    records = []

    for index, (number, line) in enumerate(numbered):
        record = _loads(line)
        if record is not None:
            records.append(record)
        elif index == 0:
            raise ValueError(f"{path}: not an episode log")
        elif index < len(numbered) - 1:
            raise ValueError(f"{path}:{number}: corrupt record")

    return records


def _header(path: Path, record: dict[str, Any] | None) -> dict[str, Any]:
    record = record or {}
    version = record.get("version")
    if record.get("kind") != "header" or record.get("format") != FORMAT or not isinstance(version, int):
        raise ValueError(f"{path}: not an episode log")
    if version > VERSION:
        raise ValueError(f"{path}: log version {version} is newer than supported version {VERSION}")

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
