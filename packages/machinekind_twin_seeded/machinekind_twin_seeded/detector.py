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

A track is one block in one camera; several blocks may share a label. A fresh seed (the model's
pixel for one block) snaps to the colour blob within the seed's pixel error. From then on the
track follows the blob nearest its last centroid, within `track_error_px`. The model's seeds of a
label in a camera go to that label's tracks there by distance, closest first, whatever order the
model lists them in: a seed near a followed track only confirms it, and a seed no track is near
starts a new track. One seed and one track of a label always pair, as one label always did.

A blob that several tracks follow holds several blocks: touching ones of one colour, or one on
another. It is split among them, each pixel going to the nearest track's centre (`grow.split`); the
twin then fits the blocks' boxes to their masks together. While tracks share a blob their centres
move with the blob, so the cut between them stays where it was. A blob with fewer pixels than its
tracks had together (`CROWDED`) cannot hold them all: a block left and its track reached its
neighbour. The track nearest the blob keeps it.

The model answers about frames some seconds old. A new seed that would take a share of a blob a
seen track follows is first checked against the frame the model saw: when its box holds clearly
less of the colour now than then (`CROWDED`), its block has left since, and the seed is dropped.

A blob, or a share of one, longer than `max_blob_ratio` times the model's box (or the seed's
search diameter without a box) is not one block and is not followed: a block the colour of the
table (bare wood on wood), or two of one colour while the model has named only one of them.

For every followed block `detect` returns its mask, box and centroid and the label's size, so the
twin's box fit places the block with its yaw. A track without a blob for `lost_after_s` is
dropped and waits for the next seed; one the model leaves out of three answers in a row is
dropped too.

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
from machinekind_twin_api import Detection, Detector, DetectorInfo, DetectRequest, Frame, base_label
from machinekind_twin_llm import LlmDetector
from machinekind_twin_llm.cli import CliCancelled
from machinekind_twin_llm.settings import names

from .grow import Blob, assign, blobs, colour_mask, nearest, part, split
from .settings import Block, HsvRange, SeededSettings, llm_settings, resolve, validate

__all__ = ["MAX_UNCONFIRMED", "PIXEL_ERROR_PX", "SeededDetector"]

logger = logging.getLogger(__name__)

#: How far a blob's centroid may lie from the block's centre as the twin reads a pixel with a size
#: (at half the block's height) [px]. Only the ray paths read it; the box fit uses the mask.
PIXEL_ERROR_PX = 6.0
#: Answers in a row that may leave out a followed block before its track is dropped.
MAX_UNCONFIRMED = 3
#: Two seeds of one label in one camera name one block twice when their boxes overlap this much
#: (IoU) and the boxes' centres lie within `SAME_SHIFT` of the shorter box side, or without boxes
#: when their pixels lie this close [px]. The box of a block resting on another of its size
#: overlaps the lower one's by most of its area, its centre a block's height higher.
SAME_IOU, SAME_SHIFT, SAME_PX = 0.5, 0.15, 2.0
#: A blob several tracks reach holds their blocks only when it has at least this share of the
#: pixels they had when last seen.
CROWDED = 0.75


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
    #: The frames the model was asked about, by camera.
    images: Mapping[str, np.ndarray]


@dataclass(eq=False)
class _Track:
    """One block followed in one camera."""

    label: str
    camera: str
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
    #: Its pixels at the last sighting; None = not seen yet.
    area: int | None = None
    #: The centroid (u, v) [px] and pixels of the blob it shared with other tracks at the last
    #: sighting; None = it had a blob of its own.
    shared: tuple[float, float, int] | None = None
    #: A seed's box (u0, v0, u1, v1) [px] and its pixels of the colour in the frame the model saw;
    #: None once seen, or without the frame.
    asked: tuple[tuple[int, int, int, int], int] | None = None

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
        self._tracks: list[_Track] = []
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
        """The block a label (`base_label`) names, resolved once; None for a label not asked for or
        not a block."""
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
        seeds: dict[tuple[str, str], list[Detection]] = {}
        colours: dict[tuple[str, tuple[HsvRange, ...]], np.ndarray] = {}
        for det in answer.dets:                 # surest first
            label = base_label(det.label)
            if det.camera not in images or self._block(label) is None:
                continue
            found = seeds.setdefault((label, det.camera), [])
            if not any(_same(det, other) for other in found):
                found.append(det)
        named: set[int] = set()
        for (label, camera), found in seeds.items():
            mine = [t for t in self._tracks if t.label == label and t.camera == camera]
            paired = self._pair(found, mine)
            for a, det in enumerate(found):
                track = mine[paired[a]] if a in paired else None
                if track is not None and track.live:
                    track.unconfirmed = 0
                    named.add(id(track))
                    continue
                error = det.pixel_error_px
                radius = float(error if error is not None else self.config.seed_error_px)
                fresh = _Track(label, camera, anchor=(float(det.pixel[0]), float(det.pixel[1])), radius=radius,
                               seed_box=det.box, seeded_at=now, seed_radius=radius)
                fresh.asked = self._asked(fresh, answer.images.get(camera), colours)
                if track is not None:
                    self._tracks[self._tracks.index(track)] = fresh
                else:
                    self._tracks.append(fresh)
                named.add(id(fresh))
        for track in list(self._tracks):
            if id(track) in named or not track.live:
                continue
            track.unconfirmed += 1
            if track.unconfirmed >= MAX_UNCONFIRMED:
                self._tracks.remove(track)

    def _asked(self, track: _Track, image: np.ndarray | None,
               colours: dict[tuple[str, tuple[HsvRange, ...]], np.ndarray]) -> tuple[tuple[int, int, int, int], int] | None:
        """The seed's box, clipped to the frame, and its pixels of the colour in the frame the model
        saw (`image`); without a box, the square of the seed's radius around its pixel."""
        if image is None:
            return None
        block = self._blocks[track.label]
        assert block is not None
        h, w = image.shape[:2]
        box = track.seed_box
        if box is None or not all(math.isfinite(x) for x in box):
            (u, v), r = track.anchor, track.seed_radius
            box = (u - r, v - r, u + r, v + r)
        u0, v0 = max(math.floor(box[0]), 0), max(math.floor(box[1]), 0)
        u1, v1 = min(math.ceil(box[2]), w - 1), min(math.ceil(box[3]), h - 1)
        if u1 < u0 or v1 < v0:
            return None
        key = (track.camera, block.hsv)
        if key not in colours:
            colours[key] = colour_mask(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), block.hsv) > 0
        return (u0, v0, u1, v1), int(np.count_nonzero(colours[key][v0:v1 + 1, u0:u1 + 1]))

    def _pair(self, seeds: list[Detection], tracks: list[_Track]) -> dict[int, int]:
        """Which track each seed of one label in one camera is, {seed: track}: one seed and one
        track always pair; otherwise the closest pairs first, each within the seed's pixel error
        plus `track_error_px`."""
        if len(seeds) == 1 and len(tracks) == 1:
            return {0: 0}
        cost = np.zeros((len(seeds), len(tracks)))
        for a, det in enumerate(seeds):
            error = det.pixel_error_px
            reach = float(error if error is not None else self.config.seed_error_px) + self.config.track_error_px
            for b, track in enumerate(tracks):
                cost[a, b] = math.hypot(det.pixel[0] - track.anchor[0], det.pixel[1] - track.anchor[1]) / reach
        return dict(assign(cost, 1.0))

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
        """Every track grown in the current frames: the blob nearest its anchor, or its share of a
        blob several tracks follow, as a Detection."""
        found: list[Detection] = []
        for camera, image in images.items():
            mine = [t for t in self._tracks if t.camera == camera]
            if not mine:
                continue
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            by_colour: dict[tuple[HsvRange, ...], list[_Track]] = {}
            for track in mine:
                block = self._blocks[track.label]
                assert block is not None
                by_colour.setdefault(block.hsv, []).append(track)
            for ranges, tracks in by_colour.items():
                labels, candidates = blobs(colour_mask(hsv, ranges), int(self.config.min_area_px))
                found += self._follow(tracks, labels, candidates, now)
        # A camera without a frame in this call grows nothing; its tracks expire like lost ones.
        for track in list(self._tracks):
            if (track.lost or track.camera not in images) and self._expired(track, now):
                self._tracks.remove(track)
        return found

    def _follow(self, tracks: list[_Track], labels: np.ndarray, candidates: list[Blob],
                now: float) -> list[Detection]:
        """One camera's tracks of one colour in its blobs: each takes the blob nearest its anchor,
        and a blob several take is split among them."""
        claims: dict[int, list[_Track]] = {}
        for track in tracks:
            blob = nearest(candidates, track.anchor, track.radius)
            if blob is None:
                track.lost = True
            else:
                claims.setdefault(blob.index, []).append(track)
        for index, owners in list(claims.items()):
            held = next(b for b in candidates if b.index == index)
            if any(t.seen_at is not None for t in owners):
                for track in [t for t in owners if self._stale(t, labels)]:
                    logger.debug("seeded: %r in %s: the block left its seed since the question; seed dropped",
                                 track.label, track.camera)
                    owners.remove(track)
                    self._tracks.remove(track)
            known = [t.area for t in owners if t.area is not None]
            if len(owners) > 1 and len(known) == len(owners) and held.area < CROWDED * sum(known):
                # Too small for all of them: the one nearest the blob keeps it.
                keep = min(owners, key=lambda t: math.hypot(t.anchor[0] - held.centroid[0],
                                                            t.anchor[1] - held.centroid[1]))
                for track in owners:
                    if track is not keep:
                        track.lost = True
                claims[index] = owners = [keep]
            if len(owners) == 1 and held.side > self._max_side(owners[0]):
                # Too long for one block: another blob within reach may still be it.
                free = [b for b in candidates if b.index not in claims]
                other = nearest(free, owners[0].anchor, owners[0].radius, self._max_side(owners[0]))
                if other is not None:
                    del claims[index]
                    claims[other.index] = owners
        found: list[Detection] = []
        for index, owners in claims.items():
            if not owners:
                continue
            mask = labels == index
            held = next(b for b in candidates if b.index == index)
            if len(owners) == 1:
                parts, centres, shared = [mask], [None], None
            else:
                anchors, free = zip(*(self._anchor(t, held) for t in owners), strict=True)
                parts, centres = split(mask, anchors, free)
                centres = [c if f else a for c, a, f in zip(centres, anchors, free, strict=True)]
                shared = (held.centroid[0], held.centroid[1], held.area)
            for track, share, centre in zip(owners, parts, centres, strict=True):
                got = part(share)
                if got is None or got[2] > self._max_side(track):
                    if got is not None and (track.label, track.camera) not in self._too_big:
                        self._too_big.add((track.label, track.camera))
                        logger.info("seeded: %r in %s: the colour's blob is larger than one block; "
                                    "not followed there", track.label, track.camera)
                    track.lost = True
                    continue
                centroid, (u0, v0, u1, v1), _ = got
                track.anchor, track.radius = centre or centroid, float(self.config.track_error_px)
                track.seen_at, track.lost, track.area = now, False, int(np.count_nonzero(share))
                track.shared, track.asked = shared, None
                block = self._blocks[track.label]
                assert block is not None
                found.append(Detection(
                    label=track.label, camera=track.camera, pixel=centroid,
                    box=(float(u0), float(v0), float(u1), float(v1)), confidence=1.0,
                    mask=share, size_m=block.size, pixel_error_px=PIXEL_ERROR_PX))
        return found

    def _stale(self, track: _Track, labels: np.ndarray) -> bool:
        """Whether a seed not seen yet names a block that has left since the question: its box holds
        less than `CROWDED` of the colour it held in the frame the model saw."""
        if track.seen_at is not None or track.asked is None:
            return False
        (u0, v0, u1, v1), then = track.asked
        return int(np.count_nonzero(labels[v0:v1 + 1, u0:u1 + 1])) < CROWDED * then

    def _anchor(self, track: _Track, held: Blob) -> tuple[tuple[float, float], bool]:
        """Where a track sharing `held` with others starts the split, and whether that start is
        free to move. A track that shared this blob before keeps its place on it and moves with it;
        one that had the whole blob to itself starts at the blob's centroid, not its block's, and
        is free; a new seed, or a track coming from a blob of its own, starts where it is."""
        if track.shared is not None:
            u, v, area = track.shared
            if min(area, held.area) >= CROWDED * max(area, held.area):
                du, dv = held.centroid[0] - u, held.centroid[1] - v
                return (track.anchor[0] + du, track.anchor[1] + dv), False
            return track.anchor, False
        alone = track.seen_at is not None and track.area is not None and track.area >= CROWDED * held.area
        return track.anchor, alone

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
            return _Answer([], job.t, error, {})
        return _Answer(list(dets), job.t, "", job.images)


def _same(a: Detection, b: Detection) -> bool:
    """Whether two seeds of one label in one camera name one block: their boxes overlap by
    `SAME_IOU` with centres within `SAME_SHIFT`, or without boxes their pixels lie within `SAME_PX`."""
    if a.box is not None and b.box is not None:
        au0, av0, au1, av1 = a.box
        bu0, bv0, bu1, bv1 = b.box
        inter = max(min(au1, bu1) - max(au0, bu0), 0.0) * max(min(av1, bv1) - max(av0, bv0), 0.0)
        union = (au1 - au0) * (av1 - av0) + (bu1 - bu0) * (bv1 - bv0) - inter
        side = min(au1 - au0, av1 - av0, bu1 - bu0, bv1 - bv0)
        shift = math.hypot((au0 + au1 - bu0 - bu1) / 2, (av0 + av1 - bv0 - bv1) / 2)
        return union > 0 and inter / union >= SAME_IOU and side > 0 and shift <= SAME_SHIFT * side
    return math.hypot(a.pixel[0] - b.pixel[0], a.pixel[1] - b.pixel[1]) <= SAME_PX
