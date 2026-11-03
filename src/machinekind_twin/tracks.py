"""Tracking: what the detectors placed, per label, through the gripper's hold, as `Observation.objects`.

Each label gets the twin's old cube tracker (`perception.ObjectTracker`), unchanged. A detector
thread hands its placed labels to `Tracks.observe`; whoever reads the objects (a running strategy
through `objects=`, or the panel when idle) calls `Tracks.step`, which feeds each tracker its newest
measurement once, together with the hand (TCP pose and jaw angles), and publishes one `ObjectPose`
per label.

Every published pose carries `age_s`: seconds since the frames it was measured from were captured,
0 while it follows the jaws. The tracker's `source` passes through with one bounded rewrite, so the
word matches the age:

- a plain "ostatnie widziane" becomes "kamery" while the label is in its detector's latest answer
  and `age_s <= fresh_s`; the tracker says "kamery" only on the step that accepts a measurement,
  and a strategy stepping faster than the detector answers would otherwise see it flicker;
- a "kamery" becomes "ostatnie widziane" once `age_s > fresh_s`, so a measurement that is old on
  arrival (a language model answering in 20 s) is never presented as what the cameras see now.

`fresh_s` is twice the detector's interval, kept within 0.5 to 1.0 s: the cube detector at 0.33 s
keeps "kamery" within 0.66 s of capture, close to the old panel's 0.5 s. Annotated sources
("ostatnie widziane (upuszczona)", "brak (...)") and "w dloni" are never rewritten.

The tracker's thresholds on the jaw (`still` in particular) are per update, so they depend on who
steps the tracks and how often, as they did with the policy's own rate before.

Blocks may rest on blocks. A closing jaw takes only the object nearest the TCP, never the block
under it (`ObjectTracker.may_grab`). An object that leaves the jaws comes to rest on the highest
other tracked block under it, or on the table (`ObjectTracker.rest_on`). `resting` gives where
each block is, for the box fit: its height when the cameras cannot tell it, and what holds whom.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from .kinematics import pose
from .objects import Located, Resting
from .perception import CubeDetection, ObjectTracker
from .robots import RobotSpec

logger = logging.getLogger(__name__)

__all__ = ["FAR_HAND", "Hand", "TrackView", "Tracks"]


@dataclass(frozen=True, eq=False)
class Hand:
    """The gripper as the tracker needs it: the TCP in the base frame and the jaw angles.

    The jaw values are in the gripper joint's MJCF units (rad for a hinge, m for a slide), like
    `ObjectTracker.update` takes them.
    """

    #: 4x4 TCP pose in the base frame.
    T_tcp: np.ndarray
    #: The jaw as measured.
    grip_q: float
    #: The jaw as commanded (what is sent to the servo).
    grip_cmd: float
    #: The empty, closed jaw on this arm now (the hardware limit, not the MJCF one).
    grip_closed: float


#: What the tracker gets when there is no hand: a TCP 10 m above the base and a jaw that is
#: neither squeezing nor open, so the single-camera gate near the hand never fires.
FAR_HAND = Hand(pose(np.eye(3), np.array([0.0, 0.0, 10.0])), 0.0, 0.0, 0.0)


@dataclass(frozen=True, eq=False)
class TrackView:
    """One published object as the panel shows it: the state of the last step."""

    label: str
    pos: np.ndarray
    rot: np.ndarray
    #: As published in `ObjectPose.source` (Polish, panel text).
    source: str
    age_s: float
    n_cameras: int
    confidence: float
    size_m: tuple[float, float, float] | None
    #: How the last measurement was placed: "sylwetka", "triangulacja" or "promien".
    method: str
    #: The detector whose measurement the track follows.
    detector: str


@dataclass(frozen=True)
class _Answer:
    """A detector's latest answer, as far as tracking cares."""

    labels: frozenset[str]
    hold_s: float
    fresh_s: float


@dataclass(eq=False)
class _Track:
    tracker: ObjectTracker
    detector: str
    #: The newest measurement handed in; fed to the tracker once (`used_t`).
    pending: Located
    used_t: float = -math.inf
    #: The last measurement fed to the tracker: what the panel shows about it.
    fed: Located | None = None
    #: Since when the tracker has given no pose; None while it gives one.
    none_since: float | None = None
    view: TrackView | None = field(default=None)


class Tracks:
    """Per-label trackers fed by detector threads and stepped by the readers. Thread-safe."""

    #: Lowest confidence a measurement must have (the old panel's gate).
    MIN_CONFIDENCE = 0.4
    #: Shortest time a pose is held after the last measurement [s].
    HOLD_S = 6.0
    #: Bounds of the age under which an object counts as seen by the cameras now [s].
    FRESH_MIN_S, FRESH_MAX_S = 0.5, 1.0
    #: A second `step` sooner than this after the previous one gets its result [s].
    MIN_STEP_S = 0.01
    #: How far past a block's footprint a dropped object still lands on it [m].
    REST_MARGIN_M = 0.005
    #: How far below a block's top a dropped object's bottom may be and still land on it [m].
    REST_SLACK_M = 0.01

    def __init__(self, spec: RobotSpec, clock: Callable[[], float]):
        self._spec = spec
        self._clock = clock
        self._lock = threading.Lock()
        self._tracks: dict[str, _Track] = {}
        self._answers: dict[str, _Answer] = {}
        self._last_step = -math.inf
        self._result: dict[str, ObjectPose] = {}
        self._warned: set[type] = set()

    # ------------------------------------------------------------------ timing rules
    @classmethod
    def hold_s_for(cls, interval_s: float, latency_s: float) -> float:
        """How long a detector's pose is held: long enough that a slow answer meets the next one."""
        return max(cls.HOLD_S, 3.0 * (interval_s + latency_s))

    @classmethod
    def fresh_s_for(cls, interval_s: float) -> float:
        """Up to which age a detector's object counts as seen by the cameras now."""
        return min(cls.FRESH_MAX_S, max(cls.FRESH_MIN_S, 2.0 * interval_s))

    # ------------------------------------------------------------------ the detector thread
    def observe(self, detector: str, batch: list[Located], latency_s: float, interval_s: float) -> None:
        """One detector call's placed labels. The newer capture time wins a label two detectors share."""
        with self._lock:
            self._answers[detector] = _Answer(frozenset(loc.label for loc in batch),
                                              self.hold_s_for(interval_s, latency_s),
                                              self.fresh_s_for(interval_s))
            for loc in batch:
                track = self._tracks.get(loc.label)
                if track is None:
                    self._tracks[loc.label] = _Track(ObjectTracker.for_spec(self._spec), detector, loc)
                elif loc.t >= track.pending.t:
                    track.pending, track.detector = loc, detector

    # ------------------------------------------------------------------ the readers
    def step(self, hand: Hand | Callable[[], Hand | None] | None = None) -> dict[str, ObjectPose]:
        """Feed each tracker its newest measurement and publish the objects.

        `hand` is the gripper now, None for no gripper (`FAR_HAND`), or a callable giving it, which
        is called only when the step runs. A step sooner than `MIN_STEP_S` after the previous one
        returns that one's result.
        """
        with self._lock:
            now = self._clock()
            if now - self._last_step < self.MIN_STEP_S:
                return dict(self._result)
            self._last_step = now
            h = self._hand(hand) if self._tracks else FAR_HAND
            nearest = self._nearest(h)
            result: dict[str, ObjectPose] = {}
            for label in sorted(self._tracks):
                track = self._tracks[label]
                track.tracker.may_grab = label == nearest
                if self._step_track(label, track, h, now):
                    del self._tracks[label]
                elif track.view is not None:
                    v = track.view
                    result[label] = ObjectPose(v.pos.copy(), v.rot.copy(), v.source, v.age_s)
            self._result = result
            return dict(result)

    def last_step(self) -> float:
        """Clock time of the last step that ran; -inf before the first."""
        with self._lock:
            return self._last_step

    def snapshot(self) -> list[TrackView]:
        """The published objects of the last step, by label. Never steps."""
        with self._lock:
            return [t.view for _, t in sorted(self._tracks.items()) if t.view is not None]

    def resting(self) -> dict[str, Resting]:
        """Per label with a size, where its block is: the published pose, or the newest measurement
        before the first step. A block in the hand rests on nothing and is left out."""
        with self._lock:
            out = {}
            for label, track in self._tracks.items():
                v = track.view if track.view is not None else track.pending
                if v.size_m is None or (track.view is not None and track.view.source == "w dloni"):
                    continue
                out[label] = Resting(np.array(v.pos, float), np.array(v.rot, float), v.size_m)
            return out

    def clear(self, detector: str | None = None) -> None:
        """Forget every track, or only those following `detector`."""
        with self._lock:
            if detector is None:
                self._tracks.clear()
                self._answers.clear()
                self._result = {}
                return
            self._answers.pop(detector, None)
            gone = [label for label, t in self._tracks.items() if t.detector == detector]
            for label in gone:
                del self._tracks[label]
                self._result.pop(label, None)

    # ------------------------------------------------------------------ internals
    def _hand(self, hand: Hand | Callable[[], Hand | None] | None) -> Hand:
        if callable(hand):
            try:
                hand = hand()
            except Exception as exc:
                # Without the hand nothing can be "w dloni"; the far hand is the safe reading.
                (logger.debug if type(exc) in self._warned else logger.warning)(
                    "reading the hand failed, tracking without it: %s: %s", type(exc).__name__, exc)
                self._warned.add(type(exc))
                hand = None
        return hand if hand is not None else FAR_HAND

    def _nearest(self, hand: Hand) -> str | None:
        """The label in the hand, else the one whose last pose lies nearest the TCP; None without
        poses."""
        for name, track in self._tracks.items():
            if track.tracker.in_hand is not None:
                return name
        tcp = hand.T_tcp[:3, 3]
        best, label = math.inf, None
        for name, track in self._tracks.items():
            last = track.tracker.last
            if last is not None and (d := float(np.linalg.norm(last[0] - tcp))) < best:
                best, label = d, name
        return label

    def _rest_on(self, label: str, half_h: float, p: np.ndarray) -> float:
        """The centre height of `label`'s box, `half_h` half its height, come to rest under `p`: on
        the highest other published block whose footprint holds p's xy and whose top is below the
        box, else on the table."""
        top = 0.0
        for other, track in self._tracks.items():
            v = track.view
            if other == label or v is None or v.size_m is None or v.source == "w dloni":
                continue
            local = v.rot.T @ (np.asarray(p, float) - v.pos)
            hx, hy, hz = (s / 2 for s in v.size_m)
            m = self.REST_MARGIN_M
            other_top = float(v.pos[2]) + hz
            below = other_top <= p[2] - half_h + self.REST_SLACK_M
            if abs(local[0]) <= hx + m and abs(local[1]) <= hy + m and below:
                top = max(top, other_top)
        return top + half_h

    def _step_track(self, label: str, track: _Track, hand: Hand, now: float) -> bool:
        """One tracker update; True when the track is to be deleted."""
        answer = self._answers.get(track.detector) or _Answer(frozenset(), self.HOLD_S, self.FRESH_MIN_S)
        p = track.pending
        det = None
        if p.t > track.used_t and p.confidence > self.MIN_CONFIDENCE and now - p.t <= answer.hold_s:
            track.used_t, track.fed = p.t, p
            det = CubeDetection(np.asarray(p.pos, float).copy(), np.asarray(p.rot, float).copy(),
                                area_px=p.confidence, confidence=p.confidence, n_cameras=p.n_cameras, t=p.t)
        tracker = track.tracker
        # A slow detector's pose must outlive the wait for its next answer.
        tracker.hold_s = answer.hold_s
        size = track.fed.size_m if track.fed is not None else None
        tracker.rest_on = None if size is None else partial(self._rest_on, label, size[2] / 2)
        got = tracker.update(det, hand.T_tcp, hand.grip_q, hand.grip_cmd, hand.grip_closed, now)
        if got is None or track.fed is None:
            track.view = None
            if track.none_since is None:
                track.none_since = now
            return now - track.none_since > answer.hold_s
        track.none_since = None
        source = tracker.source
        if source == "w dloni":
            age = 0.0
        else:
            age = max(0.0, now - tracker.t_seen) if math.isfinite(tracker.t_seen) else 0.0
            if source == "ostatnie widziane" and label in answer.labels and age <= answer.fresh_s:
                source = "kamery"
            elif source == "kamery" and age > answer.fresh_s:
                source = "ostatnie widziane"
        fed = track.fed
        track.view = TrackView(label, np.asarray(got[0], float).copy(), np.asarray(got[1], float).copy(),
                               source, age, fed.n_cameras, fed.confidence, fed.size_m, fed.method,
                               track.detector)
        return False
