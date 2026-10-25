"""Asynchronous episode recording and loaders for JSON Lines and Zstandard logs.

See docs/EPISODES.md for recording options, the record format and report limitations.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from queue import Empty, Full, Queue
from types import MemberDescriptorType
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

#: The header's `format`: what tells an episode log from any other JSON Lines file.
FORMAT = "machinekind-twin-episode"
#: The version this twin writes, and the newest it reads.
VERSION = 2

#: The workspace file's default place, as `workspace.DEFAULT_REL` (not imported: `workspace` pulls in MuJoCo).
DEFAULT_WORKSPACE = Path("workspace") / "twin.json"

#: Stable termination codes. A log without a complete end record loads as "unfinished".
EndCode = Literal[
    "done", "gave_up", "time_limit", "disconnected", "stopped", "preempted", "bad_target", "error", "refused",
]

#: Default quality for saved camera images.
FRAME_QUALITY = 85
#: Maximum queued events per writer, including loop samples and strategy steps.
MAX_PENDING = 8192
MAX_PENDING_IMAGE_BYTES = 64 * 1024 * 1024
FLUSH_INTERVAL_S = 0.25
WAIT_TIMEOUT_S = 5.0
IMAGE_BUDGET_UNIT = 64 * 1024
MAX_FRAME_JOBS = 128
FRAME_CLOSE_TIMEOUT_S = 3.0
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
    #: Telemetry records buffered during storage stalls; None uses 8192.
    queue_capacity: int | None = None
    #: Raw image memory budget, reserved before copying; None uses 64 MiB.
    frame_buffer_bytes: int | None = None


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


@dataclass
class _Write:
    kind: str
    value: Any
    resources: ExitStack = field(default_factory=ExitStack)
    completed: threading.Event = field(default_factory=threading.Event)
    success: bool = False

    def finish(self, success: bool) -> None:
        if self.kind == "images" and self.value is not None:
            self.value[1].clear()
            self.value = None
        self.resources.close()
        self.success = success
        self.completed.set()


class EpisodeWriter:
    """Independent telemetry and image workers with bounded queues and visible loss accounting.

    Producers never wait for storage or queue capacity. Frame overload creates marked gaps;
    telemetry overload marks the recording incomplete and collection resumes as capacity returns.
    """

    def __init__(self, folder: Path, header: Mapping[str, Any], twin: Any, name: str,
                 keep_frames_s: float | None, frame_format: Literal["webp", "jpeg"], frame_quality: int,
                 queue_capacity: int, frame_buffer_bytes: int):
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
        self._attempted_at: dict[str, float] = {}
        self._pending: Queue[_Write] = Queue(maxsize=queue_capacity)
        self._ending: Queue[EpisodeEnd] = Queue(maxsize=1)
        self._images: Queue[_Write] = Queue(maxsize=MAX_FRAME_JOBS)
        self._image_results: Queue[dict[str, Any]] = Queue(maxsize=MAX_FRAME_JOBS)
        self._closed = threading.Event()
        self._finished = threading.Event()
        self._ready = threading.Event()
        self._images_finished = threading.Event()
        self._cancel_images = threading.Event()
        self._image_error = ""
        self._close_requested_at: float | None = None
        # This lock protects only counters and queue handoffs: never copies, encoding, disk I/O or waits.
        self._state = threading.Lock()
        self._counts = {key: 0 for key in (
            "loop_enqueued", "loop_written", "loop_dropped", "step_enqueued", "step_written", "step_dropped",
            "frames_requested", "frames_saved", "frames_skipped", "frames_failed",
        )}
        self._image_budget = threading.BoundedSemaphore(frame_buffer_bytes // IMAGE_BUDGET_UNIT)
        self._file: BinaryIO | None = None
        self._compressor: Any = None
        self._handle: int | None = None
        self._last_recording: dict[str, Any] = {}
        own = {
            "kind": "header", "format": FORMAT, "version": VERSION, "name": name,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "clock": "virtual" if getattr(twin.clock, "virtual", False) else "wall", "t0": self._t0,
        }
        self._header = {**_HEADER_DEFAULTS, **_clean(header), **own, "joints": list(self._joints),
                        "keep_frames_s": keep_frames_s, "frame_format": frame_format, "frame_quality": frame_quality,
                        "queue_capacity": queue_capacity, "frame_buffer_bytes": frame_buffer_bytes,
                        "versions": {"twin": twin_version, "api": machinekind_twin_api.__version__}}
        self._handle = twin.add_tick_listener(self.on_tick)
        # No executor: a permanently blocked disk must not make interpreter exit join these workers.
        if keep_frames_s is None:
            self._images_finished.set()
        else:
            try:
                threading.Thread(target=self._work_images, name="episode-frames", daemon=True).start()
            except Exception as exc:  # noqa: BLE001 - losing image capture must not lose telemetry
                self._image_error = f"image worker unavailable: {exc}"
                self._images_finished.set()
        try:
            threading.Thread(target=self._work, name="episode-writer", daemon=True).start()
        except Exception:
            self._cancel_images.set()
            self._closed.set()
            self._detach()
            raise

    @classmethod
    def create(cls, folder: str | Path, header: Mapping[str, Any], twin: Any, *, name: str | None = None,
               keep_frames_s: float | None = None, frame_format: Literal["webp", "jpeg"] = "jpeg",
               frame_quality: int = FRAME_QUALITY, queue_capacity: int | None = None,
               frame_buffer_bytes: int | None = None) -> EpisodeWriter:
        """Start background collection. `path` is final once `opened` is true."""
        if keep_frames_s is not None and (not math.isfinite(keep_frames_s) or keep_frames_s < 0):
            raise ValueError("keep_frames_s must be finite and non-negative")
        if frame_format not in ("webp", "jpeg"):
            raise ValueError("frame_format must be webp or jpeg")
        if isinstance(frame_quality, bool) or not isinstance(frame_quality, int) or not 1 <= frame_quality <= 100:
            raise ValueError("frame_quality must be an integer between 1 and 100")
        capacity = MAX_PENDING if queue_capacity is None else queue_capacity
        image_bytes = MAX_PENDING_IMAGE_BYTES if frame_buffer_bytes is None else frame_buffer_bytes
        for key, value in (("queue_capacity", capacity), ("frame_buffer_bytes", image_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        return cls(Path(folder), header, twin, _safe(name) if name else time.strftime("%Y%m%d-%H%M%S"),
                   keep_frames_s, frame_format, frame_quality, capacity, image_bytes)

    def now(self) -> float:
        return self._clock.now() - self._t0

    @property
    def recording(self) -> dict[str, Any]:
        with self._state:
            counts = dict(self._counts)
        counts["frames_pending"] = max(0, counts["frames_requested"] - counts["frames_saved"]
                                        - counts["frames_skipped"] - counts["frames_failed"])
        counts["telemetry_complete"] = (not self.error and not counts["loop_dropped"] and not counts["step_dropped"]
                                        and counts["loop_enqueued"] == counts["loop_written"]
                                        and counts["step_enqueued"] == counts["step_written"])
        counts["frames_complete"] = not (counts["frames_skipped"] or counts["frames_failed"]
                                         or counts["frames_pending"])
        return counts

    @property
    def warning(self) -> str:
        counts = self.recording
        notes = []
        if counts["loop_dropped"] or counts["step_dropped"]:
            notes.append(f"INCOMPLETE TELEMETRY: {counts['loop_dropped']} loop samples and "
                         f"{counts['step_dropped']} steps lost")
        if counts["frames_skipped"] or counts["frames_failed"]:
            notes.append(f"Frame gaps: {counts['frames_skipped']} skipped, {counts['frames_failed']} failed")
        if self._finished.is_set() and counts["frames_pending"]:
            notes.append(f"Frame gaps: {counts['frames_pending']} saves did not finish")
        if self._image_error:
            notes.append(self._image_error)
        if self._pending.qsize() >= self._pending.maxsize * 0.8:
            notes.append("Telemetry buffer above 80% capacity")
        return "; ".join(notes)

    def _count(self, **values: int) -> None:
        with self._state:
            for key, value in values.items():
                self._counts[key] += value

    def on_tick(self, sample: Any) -> None:
        if not self._closed.is_set() and not self.error:
            self._put(_Write("loop", (sample.t, dict(sample.sent), dict(sample.measured) if sample.fresh else None,
                                     sample.measured_t, tuple(self._twin.status.at_limit))))

    def step(self, record: StepRecord) -> None:
        """Snapshot telemetry; image pressure skips only the requested images, never the step."""
        if self._closed.is_set() or self.error:
            return
        images = _Write("images", (record.step, []))
        published = False
        try:
            frames = []
            for camera, frame in record.frames.items():
                item: dict[str, Any] = {"camera": camera, "age_s": None, "shape": None, "file": None}
                image = None
                if frame is not None:
                    image = np.asarray(frame.image)
                    item.update(age_s=_rounded(frame.age_s, SECOND_DIGITS), shape=list(image.shape))
                every, previous = self._keep_frames_s, self._attempted_at.get(camera)
                if every is not None and (previous is None or record.t - previous >= every):
                    self._attempted_at[camera] = record.t
                    self._count(frames_requested=1)
                    reason = "camera frame unavailable" if image is None else self._capture_image(images, camera, image)
                    if reason:
                        item["drop_reason"] = reason
                        self._count(frames_skipped=1)
                    else:
                        item["pending"] = True
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
            if images.value[1]:
                try:
                    with self._state:
                        if self._closed.is_set() or self.error or self._cancel_images.is_set():
                            raise Full
                        self._images.put_nowait(images)
                        published = True
                except Full:
                    images.finish(False)
                    for item in frames:
                        if item.pop("pending", False):
                            item["drop_reason"] = "image buffer full"
                            self._count(frames_skipped=1)
            self._put(_Write("step", line))
        except Exception:  # noqa: BLE001 - a malformed diagnostic packet cannot end collection or motion
            if not published:
                self._count(frames_skipped=len(images.value[1]))
                images.finish(False)
            self._count(step_dropped=1)

    def _capture_image(self, job: _Write, camera: str, image: np.ndarray) -> str:
        if self._image_error:
            return self._image_error
        if self._cancel_images.is_set() or self._images_finished.is_set():
            return "image worker unavailable"
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return "unsupported image shape or type"
        if self._images.full():
            return "image buffer full"
        if self._pending.qsize() >= max(1, self._pending.maxsize // 2):
            return "telemetry backlog"
        with ExitStack() as reserved:
            for _ in range(math.ceil(image.nbytes / IMAGE_BUDGET_UNIT)):
                if not self._image_budget.acquire(blocking=False):
                    return "image memory budget exhausted"
                reserved.callback(self._image_budget.release)
            copied = image.copy()
            job.value[1].append((camera, copied))
            job.resources.callback(reserved.pop_all().close)
        return ""

    def flush(self, *, wait: bool = False, timeout: float = WAIT_TIMEOUT_S) -> bool:
        """Flush telemetry already queued, independently of image encoding and saving."""
        if wait:
            _check_timeout(timeout)
        job = _Write("flush", None)
        if not self._put(job):
            return False
        return (job.completed.wait(timeout) and job.success) if wait else True

    def close(self, end: EpisodeEnd, *, wait: bool = False, timeout: float = WAIT_TIMEOUT_S) -> bool:
        with self._state:
            if not self._closed.is_set():
                self._ending.put_nowait(end)
                self._close_requested_at = time.monotonic()
                self._closed.set()
        return self.wait(timeout) if wait else True

    def wait(self, timeout: float = WAIT_TIMEOUT_S) -> bool:
        """Wait outside control paths; an image timeout becomes explicit frame gaps."""
        _check_timeout(timeout)
        return self._finished.wait(timeout)

    def _put(self, job: _Write) -> bool:
        accepted = False
        with self._state:
            if not self._closed.is_set() and not self.error:
                try:
                    self._pending.put_nowait(job)
                    accepted = True
                    if job.kind in ("loop", "step"):
                        self._counts[f"{job.kind}_enqueued"] += 1
                except Full:
                    if job.kind in ("loop", "step"):
                        self._counts[f"{job.kind}_dropped"] += 1
        if not accepted:
            job.finish(False)
        return accepted

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
            self._ready.set()
            flushed = time.monotonic()
            end = None
            while not self.error:
                self._drain_image_results()
                try:
                    job = self._pending.get(timeout=FLUSH_INTERVAL_S)
                except Empty:
                    if self._closed.is_set():
                        expired = (self._close_requested_at is not None
                                   and time.monotonic() - self._close_requested_at >= FRAME_CLOSE_TIMEOUT_S)
                        if self._images_finished.is_set() or expired:
                            self._cancel_images.set()
                            self._drain_image_results(all_pending=True)
                            end = self._ending.get_nowait()
                            break
                    self._write_recording()
                    self._flush()
                    flushed = time.monotonic()
                    continue
                success = False
                try:
                    if job.kind == "loop":
                        t, sent, measured, measured_t, at_limit = job.value
                        self._write({
                            "kind": "loop", "t": _rounded(t - self._t0, SECOND_DIGITS),
                            "sent": self._vector(sent) if sent else None,
                            "measured": None if measured is None else self._vector(measured),
                            "measured_t": None if measured is None else _rounded(measured_t - self._t0,
                                                                                 SECOND_DIGITS),
                            "at_limit": list(at_limit),
                        })
                        self._count(loop_written=1)
                    elif job.kind == "step":
                        line = job.value
                        self._write(line)
                        self._count(step_written=1)
                    elif job.kind == "flush":
                        self._write_recording()
                        self._flush()
                    if time.monotonic() - flushed >= FLUSH_INTERVAL_S:
                        self._write_recording()
                        self._flush()
                        flushed = time.monotonic()
                    success = True
                finally:
                    job.finish(success)
                    self._pending.task_done()
            if not self.error and end is not None:
                self._write({
                    "kind": "end", "t": _rounded(end.t, SECOND_DIGITS), "code": end.code,
                    "reason": end.reason, "done": bool(end.done),
                    "success": None if end.success is None else bool(end.success),
                    "error": _rounded(end.error, SECOND_DIGITS), "note": end.note,
                    "steps": int(end.steps), "began": bool(end.began),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "recording": self.recording,
                })
            self._file.write(self._compressor.flush())
            self._file.flush()
            os.fsync(self._file.fileno())
        except Exception as exc:  # noqa: BLE001 - storage failure cannot affect motion
            self._broken(exc)
        finally:
            with self._state:
                self._closed.set()
            self._cancel_images.set()
            try:
                if self._file is not None:
                    self._file.close()
            except Exception as exc:  # noqa: BLE001 - cleanup errors remain diagnostics
                self._broken(exc)
            finally:
                while True:
                    try:
                        job = self._pending.get_nowait()
                    except Empty:
                        break
                    job.finish(False)
                    self._pending.task_done()
                try:
                    self._detach()
                except Exception as exc:  # noqa: BLE001 - always signal completion
                    self._broken(exc)
                finally:
                    self._finished.set()

    def _write_recording(self) -> None:
        current = self.recording
        if current != self._last_recording:
            self._write({"kind": "recording", "t": _rounded(self.now(), SECOND_DIGITS), **current})
            self._last_recording = current

    def _drain_image_results(self, *, all_pending: bool = False) -> None:
        for _ in range(MAX_FRAME_JOBS if all_pending else 32):
            try:
                result = self._image_results.get_nowait()
            except Empty:
                break
            try:
                self._write(result)
                self._count(**{f"frames_{result['status']}": 1})
            finally:
                self._image_results.task_done()

    def _work_images(self) -> None:
        try:
            while not self._ready.wait(FLUSH_INTERVAL_S):
                if self._cancel_images.is_set():
                    return
            while not self._cancel_images.is_set():
                try:
                    job = self._images.get(timeout=FLUSH_INTERVAL_S)
                except Empty:
                    if self._closed.is_set():
                        break
                    continue
                try:
                    step, images = job.value
                    for camera, image in images:
                        result: dict[str, Any] = {"kind": "frame", "step": step, "camera": camera,
                                                   "file": None, "drop_reason": "", "status": "failed"}
                        if self._cancel_images.is_set():
                            break
                        try:
                            if self._pending.qsize() >= max(1, self._pending.maxsize // 2):
                                result["drop_reason"] = "telemetry backlog"
                                result["status"] = "skipped"
                            else:
                                result["file"] = self._keep(step, camera, image)
                                if not result["file"]:
                                    result["drop_reason"] = "image encoder returned no data"
                                else:
                                    result["status"] = "saved"
                        except Exception as exc:  # noqa: BLE001 - one failed image must not end collection
                            result["drop_reason"] = f"{type(exc).__name__}: {exc}"
                        while not self._cancel_images.is_set():
                            try:
                                self._image_results.put(result, timeout=FLUSH_INTERVAL_S)
                                break
                            except Full:
                                continue  # Only the image worker waits, never telemetry or control.
                finally:
                    image = None  # Release the last loop variable before returning its memory reservation.
                    job.finish(True)
                    self._images.task_done()
        except Exception as exc:  # noqa: BLE001 - unexpected image-worker errors stay separate from telemetry
            self._image_error = f"image worker failed: {exc}"
        finally:
            while True:
                try:
                    job = self._images.get_nowait()
                except Empty:
                    break
                job.finish(False)
                self._images.task_done()
            self._images_finished.set()

    def _flush(self) -> None:
        assert self._file is not None
        self._file.write(self._compressor.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK))
        self._file.flush()

    def _vector(self, values: Mapping[str, float]) -> list[float | None]:
        return [_rounded(values.get(joint), JOINT_DIGITS) for joint in self._joints]

    def _frame_name(self, step: int, camera: str) -> tuple[Path, str]:
        extension = ".webp" if self._frame_format == "webp" else ".jpg"
        folder = self.path.parent / (self.path.name.removesuffix(".jsonl.zst") + ".frames")
        camera_id = hashlib.sha256(camera.encode("utf-8")).hexdigest()[:20]
        name = f"{step:06d}-{_safe(camera)[:64]}-{camera_id}{extension}"
        return folder / name, f"{folder.name}/{name}"

    def _keep(self, step: int, camera: str, image: np.ndarray) -> str | None:
        import cv2

        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        extension = ".webp" if self._frame_format == "webp" else ".jpg"
        quality_flag = cv2.IMWRITE_WEBP_QUALITY if self._frame_format == "webp" else cv2.IMWRITE_JPEG_QUALITY
        encoded, data = cv2.imencode(extension, bgr, [int(quality_flag), self._frame_quality])
        if not encoded:
            return None
        path, relative = self._frame_name(step, camera)
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as file:
            file.write(data.tobytes())
            file.flush()
            os.fsync(file.fileno())
        return relative

    def _write(self, line: Mapping[str, Any]) -> None:
        assert self._file is not None
        data = (json.dumps(_clean(line), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n").encode("utf-8")
        self._file.write(self._compressor.compress(data))

    def _broken(self, exc: BaseException) -> None:
        reason = f"episode recording failed: {exc}"
        with self._state:
            if not self.error:
                self.error = reason
            self._closed.set()
        self._cancel_images.set()

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
    recording: dict[str, Any] = field(default_factory=dict)


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
    recording: dict[str, Any] = field(default_factory=dict)

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
        _resolve_frames(steps, records, finished=bool(ends))
        recording = _recording_status(records, finished=bool(ends))
        if header["version"] >= 2 and not ends:
            recording.update(telemetry_complete=False, frames_complete=False)

        width = len(joints)
        loop_t = np.array([_number(record.get("t")) for record in loops], float)
        sent = _rows([record.get("sent") for record in loops], width)
        measured = _rows([record.get("measured") for record in loops], width)
        measured_t = np.array([_number(record.get("measured_t")) for record in loops], float)
        at_limit = np.array([[joint in (record.get("at_limit") or ()) for joint in joints] for record in loops],
                            bool).reshape(len(loops), width)

        return EpisodeLog(path=path, header=header, joints=joints, loop_t=loop_t, sent=sent, measured=measured,
                          measured_t=measured_t, at_limit=at_limit, steps=steps,
                          end=_end(ends[-1] if ends else None, steps), recording=recording)

    @staticmethod
    def summary(path: str | Path) -> EpisodeSummary:
        """Stream the header and final status without retaining loop and step records."""
        path = Path(path)
        reader = _LogReader(path)
        header = None
        last = None
        last_step = []
        recording = {}
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
            if record is not None and record.get("kind") == "recording":
                recording = record
            if record is not None and record.get("kind") == "end":
                recording = record.get("recording") or recording
            previous_bad = record is None
        return EpisodeSummary(path=path, header=_header(path, header),
                              end=_end(last if reader.complete else None, last_step),
                              recording=_recording_status([{"kind": "recording", **recording}],
                                                          finished=reader.complete and last is not None))

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


def _resolve_frames(steps: list[dict[str, Any]], records: list[dict[str, Any]], *, finished: bool) -> None:
    results = {(record.get("step"), record.get("camera")): record for record in records
               if record.get("kind") == "frame"}
    for step in steps:
        for frame in step.get("frames") or ():
            if not isinstance(frame, dict):
                continue
            if not frame.pop("pending", False):
                continue
            result = results.get((step.get("step"), frame.get("camera")))
            if result is None:
                frame["file"] = None
                frame["drop_reason"] = ("image save did not finish before recording closed"
                                        if finished else "image save pending")
            else:
                frame["file"] = result.get("file")
                if not frame["file"]:
                    frame["drop_reason"] = result.get("drop_reason") or "image save failed"


def _recording_status(records: list[dict[str, Any]], *, finished: bool) -> dict[str, Any]:
    status = {}
    for record in records:
        if record.get("kind") == "recording":
            status = {key: value for key, value in record.items() if key not in ("kind", "t")}
        elif record.get("kind") == "end" and record.get("recording"):
            status = record["recording"]
    if status and not finished:
        status = {**status, "telemetry_complete": False, "frames_complete": False}
    return status


def _check_timeout(timeout: float) -> None:
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError("recording waits require a finite, non-negative timeout")


def _decision(decision: Any) -> dict[str, Any]:
    """Snapshot stored diagnostic fields without executing plugin properties."""
    def stored(name: str, default: Any = None) -> Any:
        value = inspect.getattr_static(decision, name, default)
        return value.__get__(decision) if isinstance(value, MemberDescriptorType) else value

    try:
        question, answer = stored("question", ""), stored("answer", "")
        state = stored("state")
        return {
            "question": question if isinstance(question, str) else _placeholder(question),
            "answer": answer if isinstance(answer, str) else _placeholder(answer),
            "state": _clean(state) if state is not None else {},
            "confidence": _rounded(stored("confidence"), SECOND_DIGITS),
            "response_s": _rounded(stored("response_s"), SECOND_DIGITS),
        }
    except Exception:  # noqa: BLE001 - unreadable diagnostic data costs only that decision
        return {"question": _placeholder(decision), "answer": "", "state": {}, "confidence": None,
                "response_s": None}


def _clean(value: Any, depth: int = 0) -> Any:
    """`value` as strict JSON values. It never raises: a non-finite number becomes null, a large
    array its type and shape, and unsupported objects a type placeholder. Custom mappings and
    representations are not evaluated on the producer thread."""
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
        return _placeholder(value)
    if isinstance(value, np.ndarray):
        if value.size > ARRAY_LIMIT:
            return f"<ndarray {value.dtype} {value.shape}>"
        return _clean(value.tolist(), depth + 1)
    if isinstance(value, Mapping):
        if type(value) is not dict:
            return _placeholder(value)
        return {str(key) if type(key) in (str, int, float, bool, type(None)) else _placeholder(key):
                _clean(item, depth + 1) for key, item in value.copy().items()}
    if type(value) in (list, tuple):
        return [_clean(item, depth + 1) for item in list(value)]
    if isinstance(value, PurePath):
        return str(value)
    return _placeholder(value)


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
