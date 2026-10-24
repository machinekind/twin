"""Detector threads in, `Observation.objects` out: `DetectorRunner` and `Perception`.

A detector (`machinekind_twin_api.Detector`) looks at camera frames and says what it sees where in
the image. `DetectorRunner` calls it on a daemon thread of its own, one call at a time, at the
detector's `interval_s`, and backs off after failures. Each answer goes to a sink: `Perception`
places it in the base frame (`objects.locate`) and hands it to the trackers (`tracks.Tracks`).

    perception = Perception(twin, ws)                       # headless: frames from the twin's cameras
    perception.start_detector(load_detector("cube"), {})     # its thread runs from here on
    runner = StrategyRunner(twin, strategy, objects=perception.objects)
    ...
    perception.stop_all("koniec")

Nobody waits for a detector. `Perception.objects()` steps the trackers with the newest placed
answers and returns at once, however slow the detector is; a slow detector only makes the poses
older (`ObjectPose.age_s`).

One clock drives everything: `Perception` takes `clock` (default `time.monotonic`), the frame
source stamps capture times on it, and the runners and the trackers get it from `Perception`. A
test injects a virtual clock together with a frame source stamping on it, and drives a runner with
`step_once` and an injected `wait`, so no test sleeps for pacing.

The detector, the API's request and answer types and the frame source are re-exported here, so
`machinekind_twin.detection.Detection is machinekind_twin_api.Detection`.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from machinekind_twin_api.detector import Detection, Detector, DetectorInfo, DetectRequest
from machinekind_twin_api.strategy import Frame, ObjectPose

from . import strategy
from .detectors import DetectorPlugin
from .kinematics import RobotKinematics
from .objects import CameraModel, camera_key, camera_models, locate, mapper_for
from .perception import CubeDetector, TableMapper
from .tracks import Hand, Tracks, TrackView
from .workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = [
    "DetectRequest",
    "Detection",
    "Detector",
    "DetectorInfo",
    "DetectorRunner",
    "DetectorStatus",
    "FrameSource",
    "OccluderSource",
    "Perception",
]

#: Where frames come from: camera name -> (RGB image or None, capture time on the Perception clock).
FrameSource = strategy.FrameSource
#: Arm masks (True where the arm hides the table) per camera, in the raw frame's geometry, for the
#: capture times {camera: time} of the frames used; None = no occluders.
OccluderSource = Callable[[dict[str, float]], dict[str, np.ndarray] | None]
#: What a runner hands its answers to: (detector name, detections, the frames read as
#: {camera: ((width, height), capture time)}, latency [s], interval [s]) -> detections dropped.
Sink = Callable[[str, list[Detection], dict[str, tuple[tuple[int, int], float]], float, float], int]


@dataclass
class DetectorStatus:
    """What the panel shows about one running detector."""

    running: bool = False
    #: `detector.start(info)` is running (a login check may take seconds).
    starting: bool = False
    #: Clock time the current call began; None when no call is in flight.
    call_started: float | None = None
    #: `detect` calls since start, failed ones included.
    calls: int = 0
    #: Consecutive failed calls; 0 after a success.
    failures: int = 0
    last_error: str = ""
    last_latency_s: float = 0.0
    #: Clock time the last call returned.
    last_done: float = 0.0
    #: Labels of the last successful answer, normalised.
    last_labels: tuple[str, ...] = ()
    #: Cameras offered in the last call.
    cameras: tuple[str, ...] = ()
    #: Detections dropped as invalid (or unplaceable) since start.
    dropped: int = 0
    #: Clock time of the next call.
    next_try: float = 0.0
    stopped_because: str = ""


@dataclass(eq=False)
class _Run:
    """One run of a detector, and its token: a thread holding an older run may not feed anything."""

    #: `detector.start` returned: from then on the detector gets exactly one `stop`.
    began: bool = False


class _LazyRequestFrames(Mapping[str, Frame]):
    """`DetectRequest.frames`: frames fetched on first read, with `strategy._LazyFrames`' semantics.

    Iteration and `len` list the offered cameras. Reading a camera without an image, or with one
    older than `max_age_s`, raises KeyError; `in` and `get` fetch and are therefore accurate.
    Every frame read is recorded in `read` as ((width, height), capture time) for the geometry.
    """

    def __init__(self, names: Sequence[str], fetch: FrameSource, max_age_s: float, clock: Callable[[], float]):
        self._names = tuple(names)
        self._fetch = fetch
        self._max_age_s = max_age_s
        self._clock = clock
        self._fetched: dict[str, Frame | None] = {}
        self.read: dict[str, tuple[tuple[int, int], float]] = {}

    def __getitem__(self, name: str) -> Frame:
        if name not in self._names:
            raise KeyError(name)
        if name not in self._fetched:
            self._fetched[name] = self._get(name)
        frame = self._fetched[name]
        if frame is None:
            raise KeyError(f"{name}: no fresh frame")
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def _get(self, name: str) -> Frame | None:
        try:
            image, captured = self._fetch(name)
        except Exception as exc:
            # A camera that went away between the listing and the read is a missing frame.
            logger.debug("reading frame %r failed: %s: %s", name, type(exc).__name__, exc)
            return None
        if image is None:
            return None
        age = max(0.0, self._clock() - float(captured))
        if age > self._max_age_s:
            return None
        h, w = image.shape[:2]
        self.read[name] = ((int(w), int(h)), float(captured))
        return Frame(image=image, age_s=age)


def _finite_non_negative(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


class DetectorRunner:
    """Calls one detector on a thread of its own and hands each answer to a sink."""

    #: Shortest time between the starts of two calls, whatever the detector asks [s].
    MIN_INTERVAL_S = 0.1
    #: An older frame is not offered (the panel's FRAME_MAX_AGE) [s].
    MAX_FRAME_AGE_S = 1.0
    #: Longest wait after consecutive failures [s].
    MAX_BACKOFF_S = 300.0
    #: How long `stop` waits for the thread [s]. A `detect` that ignores the stop is left behind.
    join_timeout_s = 2.0

    def __init__(self, name: str, detector: Detector[Any], *, info: DetectorInfo,
                 cameras: Callable[[], tuple[str, ...]], frame: FrameSource, sink: Sink,
                 clock: Callable[[], float], wait: Callable[[float], bool] | None = None):
        self.name = name
        self.detector = detector
        self.info = info
        self.status = DetectorStatus()
        self._cameras = cameras
        self._frame = frame
        self._sink = sink
        self._clock = clock
        self._stop = threading.Event()
        #: Waits up to the given time; True when the run is to end. Tests inject one that never sleeps.
        self._wait = wait if wait is not None else self._stop.wait
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._used = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ start and stop
    def start(self, *, threaded: bool = True) -> None:
        """Starts the detector on the calling thread (`detector.start(info)`, which may raise to
        refuse), then its thread. A runner starts once, and not after a `stop`. A bad `interval_s`
        or `pixel_error_px` is a ValueError, before the detector starts."""
        run = _Run()
        with self._lock:
            # One lock section from the check to the run token: a `stop` lands either before it
            # (and this start refuses) or after it (and the token cancels the run).
            if self._used:
                raise RuntimeError(f"detector runner {self.name!r} was already started or stopped")
            self._used = True
            for attr, default in (("interval_s", 1.0), ("pixel_error_px", 3.0)):
                value = getattr(self.detector, attr, default)
                if not _finite_non_negative(value):
                    self.status.stopped_because = f"zle ustawienia: {attr} = {value!r}"
                    raise ValueError(f"{attr} must be a finite non-negative number, got {value!r}")
            self._stop.clear()
            self._run = run
            self.status = DetectorStatus(starting=True)

        try:
            start = getattr(self.detector, "start", None)
            if start is not None:
                start(self.info)
        except Exception as exc:
            with self._lock:
                if run is self._run:
                    self._run = None
                self.status.starting = False
                self.status.stopped_because = f"detektor odmowil: {exc}"
            raise

        with self._lock:
            run.began = True
            cancelled = run is not self._run
            if not cancelled:
                self.status.starting = False
                self.status.running = True
                self.status.next_try = self._clock()
        if cancelled:
            # `stop` came while the detector was starting; it is owed its one `stop` now.
            self._call_stop(self.status.stopped_because or "zatrzymany w trakcie startu")
            raise RuntimeError("detektor zatrzymany w trakcie startu")

        if threaded:
            self._thread = threading.Thread(target=self._loop, args=(run,), name=f"detektor:{self.name}",
                                            daemon=True)
            self._thread.start()

    def stop(self, reason: str, *, wait: bool = True) -> None:
        """Ends the run: no answer is fed from now on. Calls `detector.stop(reason)` once, on this
        thread, and with `wait` joins the detector thread for up to `join_timeout_s`. A stop
        before `start` is kept: that start then refuses."""
        self._stop.set()
        with self._lock:
            run, self._run = self._run, None
            self.status.call_started = None
            if run is None:
                if not self._used:
                    self._used = True
                    self.status.stopped_because = reason
                return
            began = run.began
            self.status.running = False
            self.status.starting = False
            self.status.stopped_because = reason
        if began:
            self._call_stop(reason)
        if wait:
            self.join()

    def join(self, timeout: float | None = None) -> None:
        """Waits for the detector thread, at most `join_timeout_s` by default."""
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.join_timeout_s if timeout is None else timeout)

    def _call_stop(self, reason: str) -> None:
        stop = getattr(self.detector, "stop", None)
        if stop is None:
            return
        try:
            stop(reason)
        except Exception:
            logger.exception("stop() of detector %r raised", self.name)

    # ------------------------------------------------------------------ calls
    def step_once(self) -> float:
        """One call without sleeping. Returns the wait before the next call; inf when not running."""
        run = self._run
        if run is None or not run.began:
            return math.inf
        wait = self._step(run)
        return math.inf if wait is None else wait

    def _loop(self, run: _Run) -> None:
        while not self._stop.is_set() and run is self._run:
            wait = self._step(run)
            if wait is None or self._wait(wait):
                return

    def _step(self, run: _Run) -> float | None:
        """One `detect` call and its feed. None when the run ended meanwhile."""
        clock = self._clock
        with self._lock:
            if run is not self._run:
                return None
            t0 = clock()
            self.status.call_started = t0
            step = self.status.calls
        try:
            names = tuple(self._cameras())
        except Exception:
            logger.exception("listing the cameras for detector %r failed", self.name)
            names = ()
        frames = _LazyRequestFrames(names, self._frame, self.MAX_FRAME_AGE_S, clock)
        with self._lock:
            if run is not self._run:
                return None
            self.status.cameras = names
        error = ""
        bad = 0
        dets: list[Detection] | None
        try:
            answer = list(self.detector.detect(DetectRequest(frames=frames, step=step)))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
            dets = None
        else:
            dets = [d for d in answer if isinstance(d, Detection)]
            bad = len(answer) - len(dets)
            if bad:
                logger.debug("detector %r returned %d items that are not a Detection", self.name, bad)
        latency = clock() - t0
        interval = self._interval()

        with self._lock:
            if run is not self._run:
                return None                       # stopped meanwhile: the answer is discarded
            st = self.status
            st.call_started = None
            st.calls += 1
            st.last_latency_s = latency
            st.last_done = clock()
            if dets is None:
                st.failures += 1
                if st.failures == 1 or error != st.last_error:
                    logger.warning("detector %r failed (%d in a row): %s", self.name, st.failures, error)
                st.last_error = error
            else:
                st.failures = 0
                st.dropped += bad + self._feed(dets, frames.read, latency, interval)
                st.last_labels = tuple(sorted({d.label.strip().lower() for d in dets
                                               if isinstance(d.label, str) and d.label.strip()}))
            failures = st.failures
            wait = (min(self.MAX_BACKOFF_S, max(interval, 1.0) * 2 ** min(failures - 1, 8))
                    if failures else interval)
            st.next_try = t0 + wait
        return max(0.0, t0 + wait - clock())

    def _feed(self, dets: list[Detection], read: dict[str, tuple[tuple[int, int], float]],
              latency: float, interval: float) -> int:
        try:
            return int(self._sink(self.name, dets, read, latency, interval))
        except Exception:
            logger.exception("placing the answer of detector %r failed", self.name)
            return len(dets)

    def _interval(self) -> float:
        value = getattr(self.detector, "interval_s", 1.0)
        interval = float(value) if _finite_non_negative(value) else 1.0
        return max(self.MIN_INTERVAL_S, interval)


@dataclass(eq=False)
class _Models:
    key: tuple | None = None
    cameras: dict[str, CameraModel] = field(default_factory=dict)
    mapper: TableMapper | None = None


class _TwinHand:
    """The gripper of the twin's arm for the trackers (`tracks.Hand`), like the removed panel code.

    None when the robot has no gripper or the arm is not connected: without a real arm there is no
    hand to hold anything, and the tracker then gets `FAR_HAND`.
    """

    def __init__(self, twin: Any, ws: Workspace):
        self._twin = twin
        self._ws = ws
        self._lock = threading.Lock()
        self._kin: RobotKinematics | None = None

    def __call__(self) -> Hand | None:
        spec = self._ws.spec()
        grip = spec.gripper
        if grip is None or not self._twin.connected:
            return None
        with self._lock:
            if self._kin is None or self._kin.spec is not spec:
                self._kin = RobotKinematics(spec)
            kin = self._kin
            g = spec.joints.index(grip)
            status = self._twin.status
            joints = dict(status.measured) or self._twin.joints()
            q = kin.to_q(joints)
            sent = (status.command or {}).get(grip)
            cmd = kin.to_q({**joints, grip: float(sent)})[g] if sent is not None else q[g]
            return Hand(kin.tcp(joints), float(q[g]), float(cmd), self._closed(kin, joints, grip, g))

    def _closed(self, kin: RobotKinematics, joints: dict[str, float], grip: str, g: int) -> float:
        """The empty, closed jaw on this arm now: the gripper's lower limit as it holds (narrowed by
        the backend at connect), not the MJCF one. On the SO-101 the closed jaw stands at -5.4 deg
        after the servo-tick conversion while MJCF has -10 deg; with the MJCF value an empty hand
        came out as "w dloni". `to_q({grip: 0})` only when the limits do not tell."""
        try:
            lo = float(self._twin.joint_limits()[grip][0])
        except Exception as exc:
            logger.debug("no joint limits for the gripper, using 0: %s: %s", type(exc).__name__, exc)
            return float(kin.to_q({grip: 0.0})[g])
        return float(kin.to_q({**joints, grip: lo})[g])


class Perception:
    """The twin's perception: detector threads in, Observation.objects out. Never blocks on a detector.

    `frame` defaults to the twin's cameras (`twin.cameras.frame_t`); the panel injects a reader of
    its cached frames. Both stamp on `time.monotonic`, the default `clock`. `occluders` gives arm
    masks for the silhouette fit; it is optional and its failures cost only the occlusion handling
    of one call. `hand` defaults to the twin's gripper; pass `lambda: None` for none.
    """

    def __init__(self, twin: Any, ws: Workspace, *, frame: FrameSource | None = None,
                 occluders: OccluderSource | None = None, hand: Callable[[], Hand | None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.twin = twin
        self.ws = ws
        self.clock = clock
        self._frame: FrameSource = frame if frame is not None else twin.cameras.frame_t
        self._occluder_source = occluders
        self._hand: Callable[[], Hand | None] = hand if hand is not None else _TwinHand(twin, ws)
        self.tracks = Tracks(ws.spec(), clock)
        self._lock = threading.Lock()
        self._runners: dict[str, DetectorRunner] = {}
        self._models_lock = threading.Lock()
        self._models_cache = _Models()
        self._warned: set[type] = set()

    # ------------------------------------------------------------------ detectors
    def start_detector(self, plugin: DetectorPlugin, raw: Mapping[str, Any], *,
                       threaded: bool = True) -> DetectorRunner:
        """Creates the detector from its settings and starts it, on the calling thread.

        A running detector of the same name is stopped and replaced; a settings error leaves it
        running. Raises when the settings are refused or the detector refuses to start. Without
        `threaded` the runner gets no thread and the caller drives it (`step_once`), as tests do.
        """
        name = plugin.name
        detector = plugin.create(raw)
        cameras = self._models().cameras
        names = tuple(cameras)
        records = {c.name: c for c in self.ws.cameras}
        info = DetectorInfo(cameras=names, robot=self.ws.spec().name,
                            simulated=bool(names) and all(records[n].simulated for n in names if n in records))
        runner = DetectorRunner(name, detector, info=info, cameras=self._camera_names, frame=self._frame,
                                sink=self._sink_for(detector), clock=self.clock)

        with self._lock:
            old = self._runners.get(name)
            self._runners[name] = runner
        if old is not None:
            old.stop("ponownie z nowymi ustawieniami")
            self.tracks.clear(detector=name)

        try:
            runner.start(threaded=threaded)
        except BaseException:
            with self._lock:
                if self._runners.get(name) is runner:
                    del self._runners[name]
            raise
        return runner

    def stop_detector(self, name: str, reason: str) -> None:
        """Stops the detector `name` (also while it starts) and forgets its objects."""
        with self._lock:
            runner = self._runners.pop(name, None)
        if runner is not None:
            runner.stop(reason)
        self.tracks.clear(detector=name)

    def stop_all(self, reason: str, *, wait: bool = True) -> None:
        """Stops every detector, then joins their threads when `wait`, and forgets every object."""
        with self._lock:
            runners = list(self._runners.values())
            self._runners.clear()
        for runner in runners:
            runner.stop(reason, wait=False)
        if wait:
            for runner in runners:
                runner.join()
        self.tracks.clear()

    def running(self) -> dict[str, DetectorRunner]:
        """The detectors started and not stopped (those still starting too), by name."""
        with self._lock:
            return dict(self._runners)

    # ------------------------------------------------------------------ objects
    def objects(self) -> dict[str, ObjectPose]:
        """The objects now, by label: the `StrategyRunner` `objects=` callable. Steps the trackers."""
        return self.tracks.step(self._hand)

    def tick(self, idle_s: float = 0.15) -> None:
        """Steps the trackers when nobody did for `idle_s`, so tracking moves without a strategy."""
        if self.clock() - self.tracks.last_step() >= idle_s:
            self.tracks.step(self._hand)

    def snapshot(self) -> list[TrackView]:
        """The objects of the last step, for the panel. Never steps."""
        return self.tracks.snapshot()

    # ------------------------------------------------------------------ geometry
    def mapper(self) -> TableMapper | None:
        """The table map over the perception cameras, shared with the Map tab; None without cameras."""
        return self._models().mapper

    def cameras(self) -> dict[str, CameraModel]:
        """The cameras detections are placed from, by name."""
        return dict(self._models().cameras)

    def _models(self) -> _Models:
        """Camera models and the mapper, rebuilt once when the cameras change (`camera_key`)."""
        with self._models_lock:
            key = camera_key(self.ws)
            cached = self._models_cache
            if key != cached.key:
                cams = camera_models(self.ws)
                mapper = mapper_for(cams, self.ws.spec().desk) if cams else None
                cached = self._models_cache = _Models(key, cams, mapper)
            return cached

    def _camera_names(self) -> tuple[str, ...]:
        """The cameras offered to a detector in this call: the current set, not the one at start."""
        return tuple(self._models().cameras)

    def _occluders(self, times: dict[str, float]) -> dict[str, np.ndarray] | None:
        """The occluder source, never raising: a failure means no occluders for this call."""
        source = self._occluder_source
        if source is None:
            return None
        try:
            return source(times)
        except Exception as exc:
            (logger.debug if type(exc) in self._warned else logger.warning)(
                "arm masks failed, fitting without them: %s: %s", type(exc).__name__, exc)
            self._warned.add(type(exc))
            return None

    def _sink_for(self, detector: Detector[Any]) -> Sink:
        def sink(name: str, dets: list[Detection], read: dict[str, tuple[tuple[int, int], float]],
                 latency_s: float, interval_s: float) -> int:
            models = self._models()
            value = getattr(detector, "pixel_error_px", 3.0)
            pixel_error_px = float(value) if _finite_non_negative(value) else 3.0
            placed, dropped = locate(dets, models.cameras, read, pixel_error_px=pixel_error_px,
                                     mapper=models.mapper, occluders=self._occluders, cube=_cube)
            self.tracks.observe(name, placed, latency_s, interval_s)
            return dropped
        return sink


def _cube(side: float) -> CubeDetector:
    """The cube detector whose silhouette fit places a cube of this side."""
    return CubeDetector(cube_half=side / 2.0)
