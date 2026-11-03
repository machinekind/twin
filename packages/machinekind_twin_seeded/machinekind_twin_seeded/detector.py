"""The seeded detector: the LLM detector says which blob is which block, colour follows the blob.

Two threads share the work:

* the twin's runner thread calls `detect` every `interval_s` (0.2 s): it takes the LLM's latest
  answer, if one came, hands the worker a copy of the frames when a question is due, and grows
  every track by colour in the current frames;
* the worker thread `detektor-seed:<cli>` asks the LLM detector (`machinekind_twin_llm`) about the
  copied frames, one question at a time, at most every `seed_interval_s`, and leaves its answer
  for the next `detect`.

A label is "<colour> <shape>": the shape gives the block's size, the colour its HSV ranges
(`settings.resolve`). A label without a known shape and colour, or an override for them, is not
followed.

A track is one label in one camera. A fresh seed (the model's pixel for that label) snaps to the
colour blob within the seed's pixel error. A blob longer than `max_blob_ratio` times the model's
box (or the seed's search diameter without a box) is not one block: two touching blocks of one
colour, or a block the colour of the table (bare wood on wood). From then on the track follows the blob nearest its last
centroid, within `track_error_px`. A followed track outranks a new seed: the model only confirms it.
For every followed blob `detect` returns the blob's mask, box and centroid and the label's size, so
the twin's box fit places the block with its yaw. A track without a blob for `lost_after_s` is
dropped and waits for the next seed; one the model leaves out of three answers in a row is dropped
too.

`stop` returns at once: it cancels a running question (the CLI's process group is killed within
0.2 s, with up to 2 s grace) and wakes the worker, which ends by itself; nothing joins it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from machinekind_twin_api import Detection, Detector, DetectorInfo, DetectRequest, Frame
from machinekind_twin_llm import LlmDetector
from machinekind_twin_llm.cli import CliCancelled
from machinekind_twin_llm.prompt import normalise_label
from machinekind_twin_llm.settings import names

from .grow import blobs, colour_mask, nearest
from .settings import Block, SeededSettings, llm_settings, resolve, validate

__all__ = ["MAX_UNCONFIRMED", "PIXEL_ERROR_PX", "SeededDetector"]

logger = logging.getLogger(__name__)

#: How far a blob's centroid may lie from the block's centre as the twin reads a pixel with a size
#: (at half the block's height) [px]. Only the ray paths read it; the box fit uses the mask.
PIXEL_ERROR_PX = 6.0
#: Answers in a row that may leave out a followed block before its track is dropped.
MAX_UNCONFIRMED = 3


@dataclass
class _Job:
    """The frames of one question, copied on the runner thread."""

    images: dict[str, np.ndarray]
    #: The frames' ages when copied [s].
    ages: dict[str, float]
    #: The detector's clock when the question was handed over [s].
    t: float
    #: The question's number since start, from 0.
    step: int


@dataclass
class _Answer:
    """The worker's result: the LLM's detections in frame pixels, or why the question failed."""

    dets: list[Detection]
    t: float
    #: "" when the question was answered.
    error: str


@dataclass
class _Track:
    """One label followed in one camera."""

    #: Where the blob is looked for: the model's pixel until the first sighting, then the blob's
    #: last centroid [px].
    anchor: tuple[float, float]
    #: How far from `anchor` the blob's box may lie [px].
    radius: float
    #: The model's box of the seed, which bounds the blob's size; None = bounded by `seed_radius`.
    seed_box: tuple[float, float, float, float] | None
    #: The clock when the seed arrived: the lost deadline counts from it until the first sighting [s].
    seeded_at: float
    #: The seed's search radius [px]: bounds the blob's size when the model gave no box.
    seed_radius: float
    #: The clock at the last sighting [s]; None = not seen yet.
    seen_at: float | None = None
    #: The last grow found no blob.
    lost: bool = False
    #: Answers in a row since the model last named this track.
    unconfirmed: int = 0

    @property
    def live(self) -> bool:
        """Seen in the last grow."""
        return self.seen_at is not None and not self.lost


class _CopiedFrames(Mapping[str, Frame]):
    """A question's frames for the LLM detector: the copies of one `_Job`."""

    def __init__(self, job: _Job) -> None:
        self._job = job

    def __getitem__(self, name: str) -> Frame:
        return Frame(image=self._job.images[name], age_s=self._job.ages[name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._job.images)

    def __len__(self) -> int:
        return len(self._job.images)


class SeededDetector(Detector[SeededSettings]):
    """Grows colour tracks from the LLM detector's pixels and gives the twin each block's mask and size."""

    settings = SeededSettings

    def __init__(self, settings: SeededSettings | None = None, *,
                 clock: Callable[[], float] = time.monotonic):
        settings = settings if settings is not None else SeededSettings()
        self._tables = validate(settings)
        self.config = settings
        #: The labels asked for; empty = every block the model names.
        self._labels = self._tables.labels
        #: Every label met so far and its block; None = not a block the detector can follow.
        self._blocks: dict[str, Block | None] = {}
        self._cameras = names(settings.cameras)
        self._llm = LlmDetector(llm_settings(settings, self._labels))
        self.interval_s = float(settings.interval_s)
        # A fallback only: each Detection carries it explicitly.
        self.pixel_error_px = PIXEL_ERROR_PX
        #: Read in `detect` only, so a test may replace it after start.
        self._clock = clock
        # Shared between the runner thread and the worker, under the lock.
        self._lock = threading.Lock()
        self._job: _Job | None = None
        self._answer: _Answer | None = None
        self._busy = False
        self._asked_at = -math.inf
        self._asks = 0
        self._wake = threading.Event()
        self._stopping = threading.Event()
        # The runner thread's own, touched only by `detect`.
        self._tracks: dict[tuple[str, str], _Track] = {}
        self._too_big: set[tuple[str, str]] = set()
        self._last_error = ""
        # The worker's own.
        self._warned: set[str] = set()
        self._thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------ start and stop
    def start(self, info: DetectorInfo) -> None:
        self._llm.start(info)
        try:
            thread = threading.Thread(target=self._worker, name=f"detektor-seed:{self.config.cli}",
                                      daemon=True)
            thread.start()
        except BaseException:
            self._llm.stop("start failed")
            raise
        self._thread = thread
        self._started = True

    def stop(self, reason: str) -> None:
        self._stopping.set()
        try:
            self._llm.stop(reason)
        finally:
            self._wake.set()

    # ------------------------------------------------------------------ the runner thread
    def detect(self, request: DetectRequest) -> Sequence[Detection]:
        if self._stopping.is_set():
            raise RuntimeError("seeded: stopped")
        if not self._started:
            raise RuntimeError("seeded: detect before start")
        now = float(self._clock())
        cameras = self._cameras or tuple(request.frames)
        images: dict[str, np.ndarray] = {}
        ages: dict[str, float] = {}
        for name in cameras:
            frame = request.frames.get(name)
            if frame is not None:
                images[name] = frame.image
                ages[name] = float(frame.age_s)
        if not images:
            offered = ", ".join(cameras) or "(none offered)"
            raise RuntimeError(f"seeded: no fresh frame from cameras {offered}")

        answer = self._take_answer()
        if answer is not None:
            self._apply_answer(answer, images, now)
        self._submit(images, ages, now)
        found = self._grow(images, now)
        if self._last_error and not found and not self._tracks:
            # Raised only with nothing to follow: the runner backs off, and the next question is out.
            raise RuntimeError(f"seeded: {self._last_error}")
        return found

    def _take_answer(self) -> _Answer | None:
        with self._lock:
            answer, self._answer = self._answer, None
        return answer

    def _block(self, label: str) -> Block | None:
        """The block a label names, resolved once; None for a label not asked for or not a block."""
        if label not in self._blocks:
            block = None
            if not self._labels or label in self._labels:
                block = resolve(label, self._tables.sizes, self._tables.hsv)
            if block is None:
                logger.info("seeded: %r is not followed: not asked for, or no size or colour for it", label)
            self._blocks[label] = block
        return self._blocks[label]

    def _apply_answer(self, answer: _Answer, images: Mapping[str, np.ndarray], now: float) -> None:
        """Seeds from the model's pixels; a followed track is only confirmed, never moved."""
        if answer.error:
            self._last_error = answer.error
            return
        self._last_error = ""
        named: set[tuple[str, str]] = set()
        for det in answer.dets:                 # surest first
            label = normalise_label(det.label)
            key = (label, det.camera)
            if det.camera not in images or key in named or self._block(label) is None:
                continue
            named.add(key)
            track = self._tracks.get(key)
            if track is not None and track.live:
                track.unconfirmed = 0
                continue
            error = det.pixel_error_px
            radius = float(error if error is not None else self.config.seed_error_px)
            self._tracks[key] = _Track(anchor=(float(det.pixel[0]), float(det.pixel[1])), radius=radius,
                                       seed_box=det.box, seeded_at=now, seed_radius=radius)
        for key, track in list(self._tracks.items()):
            if key in named or not track.live:
                continue
            track.unconfirmed += 1
            if track.unconfirmed >= MAX_UNCONFIRMED:
                del self._tracks[key]

    def _submit(self, images: Mapping[str, np.ndarray], ages: Mapping[str, float], now: float) -> None:
        """Hands the worker a copy of the frames when no question runs and `seed_interval_s` passed."""
        with self._lock:
            due = not self._busy and now - self._asked_at >= self.config.seed_interval_s
        if not due:
            return
        # Copies: the worker reads them for seconds, the frame source may reuse its buffers.
        job = _Job({name: image.copy() for name, image in images.items()}, dict(ages), now, self._asks)
        with self._lock:
            self._job = job
            self._busy = True
            self._asked_at = now
            self._asks += 1
        self._wake.set()

    def _max_side(self, track: _Track) -> float:
        """The longest blob that can still be one block [px]: `max_blob_ratio` times the seed box's
        long side, or without a usable box times the seed's search diameter."""
        box = track.seed_box
        side = -1.0 if box is None else max(box[2] - box[0], box[3] - box[1])
        if not (math.isfinite(side) and side > 0):
            side = 2.0 * track.seed_radius
        return float(self.config.max_blob_ratio) * side

    def _grow(self, images: Mapping[str, np.ndarray], now: float) -> list[Detection]:
        """Every track grown in the current frames: the blob nearest its anchor, as a Detection."""
        config = self.config
        found: list[Detection] = []
        for camera, image in images.items():
            keys = [key for key in self._tracks if key[1] == camera]
            if not keys:
                continue
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for key in keys:
                label = key[0]
                track = self._tracks[key]
                block = self._blocks[label]
                assert block is not None
                labels, candidates = blobs(colour_mask(hsv, block.hsv), int(config.min_area_px))
                blob = nearest(candidates, track.anchor, track.radius, self._max_side(track))
                if blob is None:
                    if key not in self._too_big and nearest(candidates, track.anchor, track.radius):
                        # Two touching blocks of one colour, or a block the colour of the table.
                        self._too_big.add(key)
                        logger.info("seeded: %r in %s: the colour's blob is larger than one block; "
                                    "not followed there", label, camera)
                    track.lost = True
                    if self._expired(track, now):
                        del self._tracks[key]
                    continue
                track.anchor, track.radius = blob.centroid, float(config.track_error_px)
                track.seen_at, track.lost = now, False
                u0, v0, u1, v1 = blob.box
                found.append(Detection(
                    label=label, camera=camera, pixel=blob.centroid,
                    box=(float(u0), float(v0), float(u1), float(v1)), confidence=1.0,
                    mask=np.asarray(labels == blob.index, bool), size_m=block.size,
                    pixel_error_px=PIXEL_ERROR_PX))
        # A camera without a frame in this call grows nothing; its tracks expire like lost ones.
        for key, track in list(self._tracks.items()):
            if key[1] not in images and self._expired(track, now):
                del self._tracks[key]
        return found

    def _expired(self, track: _Track, now: float) -> bool:
        since = track.seen_at if track.seen_at is not None else track.seeded_at
        return now - since > self.config.lost_after_s

    # ------------------------------------------------------------------ the worker thread
    def _worker(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping.is_set():
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            answer: _Answer | None = None
            cancelled = False
            try:
                answer = self._ask(job)
                cancelled = answer is None
            finally:
                with self._lock:
                    if answer is not None and not self._stopping.is_set():
                        self._answer = answer
                    self._busy = False
            if cancelled:
                return

    def _ask(self, job: _Job) -> _Answer | None:
        """The LLM detector's answer about the job's frames; None when the question was cancelled."""
        try:
            dets = self._llm.detect(DetectRequest(frames=_CopiedFrames(job), step=job.step))
        except CliCancelled:
            return None
        except Exception as exc:  # noqa: BLE001 - any failure is the answer `detect` reports
            error = f"{type(exc).__name__}: {exc}"[:300]
            log = logger.debug if error in self._warned else logger.warning
            log("seeded: the seed question failed: %s", error)
            self._warned.add(error)
            return _Answer([], job.t, error)
        return _Answer(list(dets), job.t, "")
