"""Spectacles glasses -> a target in the reach box -> arm motion, backing off gently on impact.

    drv = SpecsReach(twin, relay_url="ws://127.0.0.1:8080/twin", mode="sim")
    drv.start()                                        # network thread + worker thread
    ...
    drv.stop()

The twin connects to the relay (`server.js` in the specs-lens repo) as `/twin`:

* on connect it sends the overlay: the reach box and the grab zone (`kind: "volume"`, ids `reach` and
  `grab`) and a TCP sphere, then the TCP sphere every 0.2 s, and a `target` sphere for an accepted target;
* it receives `registration` (the glasses' session and whether it is trusted),
  `target` (a hand point or an object centre, base frame, metres), `gripper` (open or close the
  gripper), `follow` (a point streamed at 20 Hz in teleop: the arm follows it) and `grab` (take an
  object from the table, top-down);
* it answers every target, stream and grab with `targetState`: accepted / rejected / moving / reached /
  backoff / hold, plus held / missed for a grab, and every gripper command with `gripperState`:
  accepted / rejected / done / hold. `type` must be the first key: the relay reads it with a regex.

The network thread only receives and queues targets and gripper commands. One worker
thread does everything else: validation, IK, collisions, motion, backoff. The checks run
in this order, and their names are the `reason` sent back: not connected, stop, busy,
not armed, untrusted, session, outside volume, no depth, stale, ik, jump, collision.

A gripper command gets the arm checks only (not connected, stop, busy, not armed, plus
`no gripper`, also when the backend can't drive the gripper: `gripper_drivable()` False,
the A1X over macOS XCAN). It does not interrupt a move: it waits in the queue and runs after the
target, so "pinch a point, then a left-hand pinch" closes the gripper at the point. It takes the arm, sets
the gripper joint to open or closed, waits `grip_wait_s` (or the backend's `gripper_travel_s()`)
for it to settle and releases the arm (`done`). The TCP sphere turns red with the label "closed"
while the gripper's command is nearer closed than open, whoever set it.

Motion is our own 50 Hz smoothstep ramp rather than `Twin.move`, because every tick must
be interruptible: by a new target (continuing from the current command), an impact
(`ImpactGuard` on the loop's tick listener) or losing the arm (STOP, Home). Ownership works
as in `Twin.move`: `claim` with `gen`, engage, `set_target` as owner on every tick, and the
ramp clock pauses while the loop does. On arrival the arm is released but stays engaged,
holding the target.

A follow stream (`follow`, one `stream` id per entry into the box, `seq` from 0) is not a queue of
targets: its start and end are worker items (`FollowStart` / `FollowStop`: the checks, the hold-off, taking
the arm, the follower's own IK and collision checker, one `ImpactGuard` per stream, releasing), and the
stepping is a tick listener on the twin's loop (`twin/follow.py`: a bounded tracker and one warm-started IK
step per tick; all times from the loop's `TickSample.t`). The network thread only writes the newest sample
into the stream's slot. A stream ends with `hold`: exit (`active: false`), timeout (no sample for 0.6 s; the
arm is released 2 s after the last one), outside volume, ik / collision (1 s / 2 s of failed ticks; shorter
failures only pause it: `moving` with detail "paused: ik" / "paused: collision"), impact (the loop thread
holds the measured pose at once, then the worker backs off), retargeted (a new stream, target or grab),
stop, preempted, mode, relay. At its end the arm stops where its command is (`Twin.hold_command`). A new
stream starts only from a new stream id. The driver holds the arm for "worker" and "follow" (`_holders`)
and releases it when neither needs it, so a gripper command during a stream runs without ending it.

A grab (`policy: "topdown"`) is a worker item like a target: gate (grab zone, width, gripper), then
open, hover, descend, close, lift, and `held` or `missed` (`held` with detail "assumed" when the backend
does not measure the gripper, the A1X today). It never re-opens after `missed`.

Modes: `dry` runs every check and shows the planned target without moving (accepted, then
hold); `sim` moves only a simulated arm; `real` moves the real arm, only with `real_ok`,
and waits `hold_off_s` after a hand target (and once at the start of a follow stream). While moving,
the supervisor speed cap (`Twin.set_speed`) drops to ~1.5x the ramp speed + 5 deg/s (60 deg/s while
following), and to 15 deg/s during backoff; it is restored afterwards. In `real`, losing the relay stops
the motion (hold). The glasses have no STOP: use Enter/Ctrl-C in the console or STOP in the panel.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args

import numpy as np

from . import scene as sc
from .collision import CollisionChecker
from .follow import Follower, down_rotation
from .impact import Impact, ImpactGuard
from .kinematics import RobotKinematics
from .reach import ReachVolume, grab_zone, grab_zone_object
from .runtime import Twin, smoothstep

logger = logging.getLogger(__name__)

Mode = Literal["dry", "sim", "real"]
Source = Literal["hand", "object"]
#: `targetState.state` values: a contract with the glasses (held / missed: grab only).
State = Literal["accepted", "rejected", "moving", "reached", "backoff", "hold", "held", "missed"]
#: `gripper.action` and `gripperState.state` values: the same contract.
GripAction = Literal["open", "close"]
GripState = Literal["accepted", "rejected", "done", "hold"]
#: "descend" is slower with a tighter guard: going down to an object and back up.
Profile = Literal["normal", "descend"]
#: Read from other threads (network, panel).
Phase = Literal["idle", "moving", "backoff"]
Cause = Literal["", "retarget", "impact", "preempt", "stop", "mode", "relay"]

MODES: tuple[Mode, ...] = get_args(Mode)
SOURCES: tuple[Source, ...] = get_args(Source)
GRIP_ACTIONS: tuple[GripAction, ...] = get_args(GripAction)
#: `grab.policy` values. Only the scripted top-down grasp for now; the field keeps the seam for others.
GRAB_POLICIES = ("topdown",)
OWNER = "specs"
#: The A1X `tcp` site sits this far before the fingertips along the approach axis [m].
TIP_AHEAD = 0.033

# Overlay colours, RGB 0..1.
TCP_COLOR = (1.0, 0.8, 0.1)
TCP_CLOSED_COLOR = (1.0, 0.3, 0.1)
TARGET_COLOR = (0.2, 0.9, 0.35)

Joints = dict[str, float]


@dataclass
class Target:
    """A target from the glasses, as received (only the types are checked)."""

    source: Source
    pos: np.ndarray                   # base frame, m
    session: str
    trusted: bool
    ts: Any                           # the glasses' clock: echoed back, never compared with ours
    depth: bool = False               # object depth was measured, not the 150 cm default
    label: str = ""
    t_rx: float = 0.0                 # receive time, our monotonic clock
    #: Objects: seconds since the glasses last saw the detection; None = not sent (fake glasses).
    age: float | None = None


@dataclass
class GripCommand:
    """A gripper command from the glasses (a left-hand pinch), as received."""

    action: GripAction
    ts: Any                           # the glasses' clock: echoed back
    t_rx: float = 0.0


@dataclass
class Grab(Target):
    """A `grab` from the glasses: an object to take from the table (source "object")."""

    stream: Any = None                # the glasses' id of the grab (= its ts), echoed in every targetState
    policy: str = "topdown"
    size: tuple[float, float] | None = None     # [width, height] m, or None
    yaw: float | None = None                    # jaw closing axis in the base xy plane, rad; None = bearing
    surface: np.ndarray | None = None           # the depth-measured point on the visible face (diagnostic)
    ray: np.ndarray | None = None               # unit vector camera -> surface (diagnostic)
    obj_id: Any = None


@dataclass
class FollowSample:
    """One `follow` message, as received."""

    stream: Any                       # ts of the stream's first active sample
    seq: int
    active: bool
    pos: np.ndarray | None            # base frame, m; None when inactive
    yaw: float | None
    ts: Any
    session: str
    trusted: bool


@dataclass(eq=False)
class _Stream:
    """A follow stream: written by the network thread (under `_follow_lock`), stepped by the tick listener."""

    id: Any
    sample: FollowSample              # the newest active sample (the slot)
    last_seq: int
    version: int = 0                  # bumped by each accepted sample
    ended: bool = False
    reason: str = ""
    # Set by the worker at the start, then used only on the loop thread.
    follower: Follower | None = None
    guard: ImpactGuard | None = None
    gen: int = 0
    seen: int = -1
    ts: Any = None                    # ts of the newest sample the listener has taken
    t_rx: float | None = None         # loop time the listener first saw the newest sample
    t_prev: float | None = None
    setpoint: np.ndarray | None = None
    yaw: float | None = None
    fail: str = ""
    fail_since: float = 0.0
    last_report: float = -math.inf
    last_sig: tuple = ()
    grip_rest: float | None = None    # the gripper's |effort| when the stream started (None = not yet)
    grip_over: int = 0                # consecutive ticks above rest + gripper_force


@dataclass
class FollowStart:
    """Worker item: a new stream's first sample arrived."""

    stream: _Stream


@dataclass
class FollowStop:
    """Worker item: a running stream ended (already reported); remove the listener, back off, release."""

    stream: _Stream
    hit: Impact | None = None
    #: Set by `_latest`: the item this stop overtook in the queue; it runs after the stop.
    then: Item | None = None


@dataclass
class _FollowReport:
    """A stream's `targetState`, queued in order and sent by the network thread (`tcp` filled in there)."""

    state: State
    reason: str | None
    detail: str
    pos: list[float] | None
    stream: Any
    ts: Any
    t: float


#: What the worker's queue holds.
Item = Target | GripCommand | FollowStart | FollowStop


@dataclass(frozen=True, eq=False)
class _Waypoint:
    pos: np.ndarray
    what: str
    rotation: np.ndarray | None = None
    #: Require the gripper's approach axis to point straight down.
    check_down: bool = False
    speed_factor: float = 1.0
    profile: Profile = "normal"


@dataclass
class _Leg:
    goal: Joints
    speed: float                      # deg/s of the fastest joint
    profile: Profile
    what: str
    close_gripper: bool = False


class _Sample(NamedTuple):
    t: float
    joints: Joints


class _Abort(Exception):
    """A refused or interrupted target; state, reason and detail go to `targetState`."""

    def __init__(self, state: State, reason: str, detail: str = ""):
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.detail = detail


class _Rejected(_Abort):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__("rejected", reason, detail)


class _LoopClock:
    """Ramp time that stops while the twin's loop is frozen."""

    def __init__(self, twin: Twin):
        self._frozen: Callable[[], bool] | None = getattr(twin, "_frozen", None)
        self._start = self._last = time.monotonic()
        self._paused = 0.0

    def elapsed(self) -> float:
        now = time.monotonic()
        if self._frozen is not None and self._frozen():
            self._paused += now - self._last
        self._last = now
        return now - self._start - self._paused


def parse_target(msg: Mapping[str, Any], now: float) -> Target | None:
    """Relay `target` message -> `Target`; None if it is malformed."""
    try:
        pos = np.asarray(msg["pos"], float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None

    source = msg.get("source")
    if source not in SOURCES:
        return None

    age = msg.get("age")
    if age is not None:
        # A malformed age becomes NaN, so the target is refused as "stale" instead of silently dropped.
        is_number = isinstance(age, (int, float)) and not isinstance(age, bool)
        age = float(age) if is_number else math.nan

    return Target(
        source=source,
        pos=pos,
        session=str(msg.get("session") or ""),
        trusted=msg.get("trusted") is True,
        ts=msg.get("ts"),
        depth=msg.get("depthMeasured") is True,
        label=str(msg.get("label") or ""),
        t_rx=now,
        age=age,
    )


def parse_grip(msg: Mapping[str, Any], now: float) -> GripCommand | None:
    """Relay `gripper` message -> `GripCommand`; None if it is malformed."""
    action = msg.get("action")
    if action not in GRIP_ACTIONS:
        return None
    return GripCommand(action=action, ts=msg.get("ts"), t_rx=now)


def _number(v: Any) -> float | None:
    """A finite JSON number, else None (booleans are not numbers here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _vector(v: Any, n: int) -> np.ndarray | None:
    try:
        a = np.asarray(v, float).reshape(n)
    except (TypeError, ValueError):
        return None
    return a if np.all(np.isfinite(a)) else None


def parse_follow(msg: Mapping[str, Any]) -> FollowSample | None:
    """Relay `follow` message -> `FollowSample`; None if it is malformed."""
    stream = msg.get("stream")
    if stream is None or isinstance(stream, bool) or not isinstance(stream, (int, float, str)):
        return None
    seq = msg.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, (int, float)) or seq != int(seq):
        return None
    active = msg.get("active")
    if not isinstance(active, bool):
        return None
    pos = None
    if active:
        pos = _vector(msg.get("pos"), 3)
        if pos is None:
            return None
    return FollowSample(stream=stream, seq=int(seq), active=active, pos=pos, yaw=_number(msg.get("yaw")),
                        ts=msg.get("ts"), session=str(msg.get("session") or ""),
                        trusted=msg.get("trusted") is True)


def parse_grab(msg: Mapping[str, Any], now: float) -> Grab | None:
    """Relay `grab` message -> `Grab`; None if it is malformed (or asks for a policy we don't have)."""
    obj = msg.get("object")
    if not isinstance(obj, Mapping):
        return None
    pos = _vector(obj.get("pos"), 3)
    if pos is None:
        return None
    policy = msg.get("policy", "topdown")
    if policy not in GRAB_POLICIES:
        return None

    size = None
    raw = obj.get("size")
    if raw is not None:
        wh = _vector(raw, 2)
        if wh is None or np.any(wh <= 0):
            return None
        size = (float(wh[0]), float(wh[1]))

    age = msg.get("age")
    if age is not None:
        is_number = isinstance(age, (int, float)) and not isinstance(age, bool)
        age = float(age) if is_number else math.nan

    ts = msg.get("ts")
    stream = msg.get("stream")
    return Grab(
        source="object",
        pos=pos,
        session=str(msg.get("session") or ""),
        trusted=msg.get("trusted") is True,
        ts=ts,
        depth=msg.get("depthMeasured") is True,
        label=str(obj.get("label") or ""),
        t_rx=now,
        age=age,
        stream=stream if stream is not None else ts,
        policy=str(policy),
        size=size,
        yaw=_number(obj.get("yaw")),
        surface=_vector(obj.get("surface"), 3),
        ray=_vector(obj.get("ray"), 3),
        obj_id=obj.get("id"),
    )


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"tryb {mode!r}: dozwolone {', '.join(MODES)}")


def _rounded(v: np.ndarray | list[float]) -> list[float]:
    return [round(float(x), 4) for x in v]


class SpecsReach:
    """Drives the arm from the glasses: a relay client plus one worker thread."""

    # Arrival: every arm joint within `arrive_deg` of the goal and slower than `still_deg_s`.
    tick_s = 0.02
    arrive_deg = 2.0
    still_deg_s = 10.0
    arrive_timeout_s = 3.0

    # A servo under load settles 1-2 deg off its command (in sim the TCP ends 8 mm low), so after
    # arrival the command is nudged by the error, at most `trim_max_deg` away from the IK goal.
    trim_steps = 4
    trim_deg = 0.2
    trim_max_deg = 2.5

    lag_deg = 6.0
    lag_deg_descend = 3.0
    lag_s = 0.15

    # Tighter guard once the ramp ends, so a joint stalled at the goal backs off before the A1X
    # backend declares it "deaf" (3 deg for 1 s) and stops.
    arrive_lag_deg = 2.5
    arrive_lag_s = 0.3

    # `start_center` (10 deg/s): a joint stalled with 3-6 deg of ramp ahead never exceeds 6 deg,
    # but the backend's deaf timer already counts above 3 deg. 4 deg during the ramp and 0.2 s on
    # arrival trip at least 0.3 s before the backend faults.
    center_lag_deg = 4.0
    center_arrive_lag_s = 0.2

    tcp_period_s = 0.2
    target_linger_s = 2.0
    reconnect_s = 2.0

    ik_tol = 0.003                    # m
    approach_tol = 0.26               # rad, object targets only
    # Keep the hover point this far below the box top: higher, IK with the tool pointing down often fails.
    hover_ceiling_gap = 0.03
    # A command that moved less than this since validation counts as unchanged (no recheck).
    unchanged_deg = 0.5

    max_object_age_s = 0.5

    # The gripper joint in the app's units: 0 closed, 100 open (A1X and SO-101 alike).
    grip_closed = 0.0
    grip_open = 100.0
    grip_wait_s = 1.2

    # Sim effort threshold as a fraction of the actuator's forcerange (normal / descending).
    sim_effort_frac = 0.8
    sim_effort_frac_descend = 0.5
    # No effort rule in sim for these robots, lag rule only. SO-101: kp 998 N m/rad against a
    # 2.94 N m forcerange saturates at 0.17 deg of error, so normal motion already sits at the force
    # limit (measured on 4 of 5 joints) and no threshold has headroom. The lag rule catches walls.
    sim_effort_off = frozenset({"so101"})
    # A1X thresholds in raw units of CAN frame 0x052. Guessed; see docs/TWIN.md.
    real_effort_abs = 20.0
    real_effort_abs_descend = 12.0
    # A joint already above this fraction of its threshold at rest gets no effort rule for that move.
    effort_rest_frac = 0.5

    # The backoff guard starts late: a joint pressed against an obstacle still shows effort above
    # the threshold for a moment (27 N m on J1 in sim right after the impact).
    backoff_grace_s = 0.3
    backoff_cap_deg_s = 15.0

    # dry mode logs joint efforts this often, to pick thresholds on the first real session.
    dry_effort_period_s = 1.0

    # Follow streams. Speed of the commanded TCP point [m/s] (acceleration 4x that per s) unless given.
    follow_speed_sim = 0.20
    follow_speed_real = 0.10
    follow_cap_deg_s = 60.0
    # The gripper's |effort| (backends with `gripper_effort`, the A1X) above its value at the stream's start
    # by more than the backend's `gripper_force` for this many ticks ends the stream (`impact`, "gripper
    # load"): the jaws press on something, e.g. the fingertips on the table (2026-09-27: the gripper motor
    # overheated and the bus went error-passive although the twin never commanded the gripper).
    follow_grip_ticks = 3
    # A stream may start within this distance outside the box (the glasses' smoothed cursor lags the
    # entry), and ends when a sample is farther outside than `follow_outside_m`; in between it is clamped.
    follow_margin_m = 0.02
    follow_outside_m = 0.05
    # No sample for this long ends the stream (the glasses send every 50 ms, less than half of it);
    # the arm is released `follow_release_s` after the last sample.
    follow_timeout_s = 0.6
    follow_release_s = 2.0
    # Failed ticks pause the stream (`moving`, detail "paused: ik" / "paused: collision"); this long
    # without a good tick ends it with `hold`.
    follow_ik_fail_s = 1.0
    follow_collision_s = 2.0
    # `moving` at most every `follow_report_s`, and at least every `follow_heartbeat_s`.
    follow_report_s = 0.5
    follow_heartbeat_s = 1.0
    follow_tilt_deg = 15.0            # report the tool's tilt from straight down above this
    # Real arm: the stream's guard is tighter than for ramps, well before the A1X backend's "deaf" fault
    # (3 deg for 1 s). A joint its lag rule is timing (more than `follow_guard_lag_deg` behind, nearly
    # still) stops the follower, point and rotation: the command must not run ahead of a stalled joint.
    # Stalled for `follow_guard_lag_s` = the guard's `lag` impact. A plain lag under load (the servo's
    # steady-state error of 1-2 deg) or a joint still moving does not stop it.
    follow_guard_lag_deg = 2.5
    follow_guard_lag_s = 0.3
    follow_guard_vel_still = 15.0
    # After a stream: how long to wait for the status to show the frozen command.
    freeze_wait_s = 0.2

    # Grab (top-down). Height when the glasses don't know the object's size, the tallest height used,
    # grasp tip height limits, widest object the jaws take (80 % of its width must fit in 90 mm).
    grab_default_h = 0.04
    grab_max_h = 0.12
    grab_tip_z = (0.012, 0.06)
    grab_max_width = 0.09
    grab_width_frac = 0.8

    def __init__(self, twin: Twin, *, relay_url: str, volume: ReachVolume | None = None, mode: Mode = "dry",
                 hover: float = 0.08, touch_dz: float = 0.01, speed_deg_s: float = 20.0,
                 backoff_speed_deg_s: float = 8.0, max_jump_deg: float = 150.0, grasp: bool = False,
                 allow_untrusted: bool = False, real_ok: bool = False, hold_off_s: float = 1.0,
                 follow_speed: float | None = None, follow_floor_m: float = 0.01,
                 on_state: Callable[[dict], None] | None = None):
        _check_mode(mode)
        self.twin = twin
        self.spec = twin.spec
        self.arm = tuple(j for j in self.spec.joints if j != self.spec.gripper)
        self.relay_url = relay_url
        self.volume = volume if volume is not None else ReachVolume.from_workspace(twin.workspace)
        self.mode: Mode = mode
        self.real_ok = bool(real_ok)
        self.hover = float(hover)
        self.touch_dz = float(touch_dz)
        self.speed = float(speed_deg_s)
        self.backoff_speed = float(backoff_speed_deg_s)
        self.max_jump = float(max_jump_deg)
        self.grasp = bool(grasp)
        self.allow_untrusted = bool(allow_untrusted)
        self.hold_off_s = float(hold_off_s)
        self.follow_speed = None if follow_speed is None else float(follow_speed)
        #: Setpoints stay this far above the box floor, in every mode (the follower's `z_floor`).
        self.follow_floor_m = float(follow_floor_m)
        self.grab_zone = grab_zone(self.spec)
        self.on_state = on_state

        self._stop = threading.Event()
        self._net: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue[Item] = queue.Queue()
        self._ws = None
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()

        # Session of the last trusted registration, and of the last one of any kind (`allow_untrusted`).
        self._session: str | None = None
        self._session_any: str | None = None

        self._cancel = threading.Event()
        self._cause: Cause = ""
        self._phase: Phase = "idle"

        # Read by `_on_tick` on the twin's loop thread.
        self._guard: ImpactGuard | None = None
        self._guard_from = 0.0
        self._impact: Impact | None = None

        self._last_state: dict | None = None
        self._last_grip: dict | None = None
        self._target_shown_until: float | None = None

        # The network thread has its own kinematics for the TCP sphere; IK and collisions run on the worker.
        self._kin_net = RobotKinematics(self.spec)
        self._kin_worker = RobotKinematics(self.spec)
        self._checker: CollisionChecker | None = None
        self._checker_version = -1
        self._kin: RobotKinematics | None = None
        self._effort_ids: np.ndarray | None = None

        # The panel's speed cap from before our first move, restored afterwards.
        self._prev_cap: float | None = None
        self._capped = False

        self._current: Target | None = None
        self._current_grip: GripCommand | None = None
        # The command the target was validated from; `_recheck` compares against it.
        self._validated_from: Joints | None = None

        # Who in the driver needs the arm ("worker", "follow"): released only when nobody does.
        self._holders: set[str] = set()

        # Follow streams: the one waiting for the worker and the running one, and the ids that ended (their
        # samples are ignored). Reports go through `_outbox` in order; the network thread sends them.
        self._follow_lock = threading.Lock()
        self._pending: _Stream | None = None
        self._stream: _Stream | None = None
        self._ended: deque = deque(maxlen=64)
        self._follow_stop = threading.Event()
        # Held by the tick listener during a step; `_drop_follow` waits for it (lock order: step, follow).
        self._step_lock = threading.Lock()
        self._follow_handle: int | None = None
        self._follow_release_at: float | None = None
        self._outbox: queue.SimpleQueue[_FollowReport] = queue.SimpleQueue()
        self._last_follow: dict | None = None
        self._last_grab: dict | None = None
        # The follower's own IK and collision checker (built on the worker, then used on the loop thread).
        self._f_checker: CollisionChecker | None = None
        self._f_kin: RobotKinematics | None = None
        self._f_version = -1

    def start(self) -> None:
        if self._net is not None:
            return
        self._stop.clear()
        self._net = threading.Thread(target=self._net_loop, name="okulary-siec", daemon=True)
        self._worker = threading.Thread(target=self._work_loop, name="okulary-ruch", daemon=True)
        self._net.start()
        self._worker.start()

    def stop(self) -> None:
        """Stops both threads. A moving arm holds its measured pose, a running stream stops where its
        command is (the worker's exit), and the arm is released."""
        self._stop.set()
        self._interrupt("stop")

        ws = self._ws
        if ws is not None:
            with suppress(Exception):
                ws.close()

        for thread in (self._worker, self._net):
            if thread is not None:
                thread.join(timeout=5.0)
        self._net = self._worker = None

    def set_mode(self, mode: Mode, real_ok: bool) -> None:
        """Mode change from the panel; stops a move in progress, including the hold-off wait."""
        _check_mode(mode)
        changed = (mode, bool(real_ok)) != (self.mode, self.real_ok)
        self.mode, self.real_ok = mode, bool(real_ok)
        if changed and (self._phase == "moving" or self._following()):
            self._interrupt("mode")

    @property
    def follow_v_max(self) -> float:
        """Speed of the followed TCP point [m/s]: `follow_speed`, else by mode (slower on the real arm)."""
        if self.follow_speed is not None:
            return self.follow_speed
        return self.follow_speed_real if self.mode == "real" else self.follow_speed_sim

    def set_follow_speed(self, v: float | None) -> None:
        """From the panel's slider; a running stream picks it up on its next tick."""
        self.follow_speed = None if v is None else max(0.01, float(v))

    def gripper_drivable(self) -> bool:
        """False when the backend says it can't drive the gripper (A1X over macOS XCAN: the 10-byte
        0x051 frame is not deliverable). A backend without `gripper_drivable` can."""
        fn = getattr(self.twin.backend, "gripper_drivable", None)
        if fn is None:
            return True
        try:
            return bool(fn())
        except Exception:
            return True

    def _grip_wait(self) -> float:
        """How long a gripper command takes: the backend's `gripper_travel_s()`, else `grip_wait_s`."""
        fn = getattr(self.twin.backend, "gripper_travel_s", None)
        if fn is not None:
            try:
                v = float(fn())
            except Exception:
                v = 0.0
            if math.isfinite(v) and v > 0:
                return v
        return self.grip_wait_s

    def _following(self) -> bool:
        return self._stream is not None or self._pending is not None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def last_state(self) -> dict | None:
        return self._last_state

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "mode": self.mode,
            "session": self._session_any if self.allow_untrusted else self._session,
            "phase": self._phase,
            "last": self._last_state,
            "grip": self._last_grip,
            "follow": self._last_follow,
            "grab": self._last_grab,
            "gripper_drivable": self.gripper_drivable(),
        }

    def _net_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ws = self._connect(quiet=failures > 0)
            if ws is None:
                failures += 1
            else:
                failures = 0
                self._serve(ws)
            self._stop.wait(self.reconnect_s)

    def _connect(self, *, quiet: bool):
        from websockets.sync.client import connect

        try:
            ws = connect(self.relay_url, proxy=None, open_timeout=3.0, close_timeout=1.0)
        except Exception as exc:
            log = logger.info if quiet else logger.warning
            log("Przekaznik %s niedostepny (%s) - ponowie za %.0f s", self.relay_url, exc, self.reconnect_s)
            return None
        logger.info("Polaczono z przekaznikiem %s", self.relay_url)
        return ws

    def _serve(self, ws) -> None:
        self._ws = ws
        try:
            objects = [self.volume.overlay_object()]
            if self.grab_zone is not None:
                objects.append(grab_zone_object(self.grab_zone))
            self._send({"type": "overlay", "objects": [*objects, self._tcp_object()]})
            self._pump(ws)
        except Exception as exc:
            if not self._stop.is_set():
                logger.warning("Przekaznik rozlaczony (%s) - ponowie za %.0f s", exc, self.reconnect_s)
        finally:
            self._ws = None
            # A real arm doesn't keep moving without the relay: the glasses can no longer see its state.
            # (In dry and sim a stream ends by its timeout: no more samples.)
            moving = self._phase == "moving" or self._following()
            if self.mode == "real" and moving and not self._stop.is_set():
                self._interrupt("relay")
            with suppress(Exception):
                ws.close()

    def _pump(self, ws) -> None:
        """Receives messages, sending the TCP sphere every `tcp_period_s` in between."""
        next_tcp = time.monotonic() + self.tcp_period_s
        while not self._stop.is_set():
            wait = max(0.0, next_tcp - time.monotonic())
            if self._following():
                wait = min(wait, 0.05)                  # stream reports come from the loop thread
            try:
                text = ws.recv(timeout=wait)
            except TimeoutError:
                text = None
            if text is not None:
                self._on_message(text)
            self._flush_follow_reports()

            now = time.monotonic()
            if now >= next_tcp:
                next_tcp = now + self.tcp_period_s
                self._send_tcp(now)

    def _send(self, msg: dict) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("Wysylka do przekaznika nieudana: %s", exc)
            return False
        return True

    def _tcp_now(self, kin: RobotKinematics | None = None) -> list[float]:
        kin = kin or self._kin_net
        return _rounded(kin.tcp(self.twin.joints())[:3, 3])

    def _tcp_object(self) -> dict:
        obj: dict[str, Any] = {"id": "tcp", "kind": "sphere", "pos": self._tcp_now(), "size": [0.02],
                               "color": list(TCP_COLOR)}
        if self._gripper_closed():
            obj["color"] = list(TCP_CLOSED_COLOR)
            obj["label"] = "closed"
        return obj

    def _gripper_closed(self) -> bool:
        """The gripper's command is nearer closed than open (whoever set it)."""
        g = self.spec.gripper
        if not g:
            return False
        cmd = self.twin.status.command
        v = cmd.get(g) if cmd else None
        return v is not None and float(v) < (self.grip_closed + self.grip_open) / 2

    def _send_tcp(self, now: float) -> None:
        msg: dict[str, Any] = {"type": "overlay", "objects": [self._tcp_object()]}
        with self._lock:
            until = self._target_shown_until
            if until is not None and now >= until:
                self._target_shown_until = None
                msg["remove"] = ["target"]
        self._send(msg)

    def _on_message(self, text: str | bytes) -> None:
        try:
            msg = json.loads(text)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return

        kind = msg.get("type")
        if kind == "registration":
            self._on_registration(msg)
        elif kind == "target":
            self._on_target(msg, text)
        elif kind == "gripper":
            self._on_gripper(msg, text)
        elif kind == "follow":
            self._on_follow(msg, text)
        elif kind == "grab":
            self._on_grab(msg, text)

    def _on_registration(self, msg: dict) -> None:
        session = msg.get("session")
        if session is None:
            return
        with self._lock:
            self._session_any = str(session)
            # Trust holds for the Lens session once earned, like the glasses keep drawing with the
            # last trusted transform: a wobbly update in the same session must not revoke silently.
            # A new session (the Lens restarted) starts without trust.
            if msg.get("trusted") is True:
                self._session = str(session)
            elif self._session != str(session):
                self._session = None

    def _on_target(self, msg: dict, text: str | bytes) -> None:
        t = parse_target(msg, time.monotonic())
        if t is None:
            logger.warning("Zepsuty cel z okularow: %.200s", text)
            return
        self._queue_target(t)

    def _on_grab(self, msg: dict, text: str | bytes) -> None:
        g = parse_grab(msg, time.monotonic())
        if g is None:
            logger.warning("Zepsuty chwyt z okularow: %.200s", text)
            return
        logger.info("Chwyt z okularow: %s %s", g.label or "obiekt", np.round(g.pos, 3).tolist())
        self._queue_target(g)

    def _queue_target(self, t: Target) -> None:
        # A target or a grab ends a follow stream (its stop is queued first) and interrupts a move;
        # the worker continues from the current command.
        self._end_streams("retargeted")
        self._queue.put(t)
        if self._phase == "moving":
            self._interrupt("retarget")

    def _on_follow(self, msg: dict, text: str | bytes) -> None:
        """A stream sample: into its stream's slot; a new stream id queues its start (ending the old one)."""
        s = parse_follow(msg)
        if s is None:
            logger.warning("Zepsuty punkt sledzenia z okularow: %.200s", text)
            return
        with self._follow_lock:
            if s.stream in self._ended:
                return
            for st in (self._stream, self._pending):
                if st is None or st.id != s.stream:
                    continue
                if s.seq <= st.last_seq:                # reordered or repeated (relay reconnect)
                    return
                st.last_seq = s.seq
                if s.active:
                    st.sample = s
                    st.version += 1
                else:
                    self._end_stream(st, "exit", ts=s.ts)
                return
            if not s.active:
                return                                  # the end of a stream we never saw start
            for st in (self._stream, self._pending):
                if st is not None:
                    self._end_stream(st, "retargeted")
            st = _Stream(id=s.stream, sample=s, last_seq=s.seq)
            self._pending = st
            self._queue.put(FollowStart(st))
        # A new stream replaces a target in progress (not a gripper command: that one finishes first).
        if self._current is not None and self._phase == "moving":
            self._interrupt("retarget")

    def _on_gripper(self, msg: dict, text: str | bytes) -> None:
        cmd = parse_grip(msg, time.monotonic())
        if cmd is None:
            logger.warning("Zepsuty rozkaz chwytaka z okularow: %.200s", text)
            return
        # No interrupt: a gripper command from the glasses (a left-hand pinch) during a move closes the
        # gripper once the arm is at the point.
        self._queue.put(cmd)

    def _target_queued(self) -> bool:
        with self._queue.mutex:
            return any(isinstance(item, Target) for item in self._queue.queue)

    def _interrupt(self, cause: Cause) -> None:
        if cause in ("stop", "preempt", "mode", "relay"):
            self._end_streams(self._lost_reason() if cause == "preempt" else cause)
        with self._lock:
            if not self._cancel.is_set():
                self._cause = cause
                self._cancel.set()

    def _reset_cancel(self) -> None:
        with self._lock:
            if not self._stop.is_set():                 # a `stop()` landing right now must not be lost
                self._cancel.clear()
                self._cause = ""

    @property
    def _cancel_cause(self) -> Cause:
        with self._lock:
            return self._cause

    def _lost_reason(self) -> str:
        return "stop" if self._estopped() else "preempted"

    def _report(self, t: Target, state: State, reason: str | None = None, *, detail: str = "") -> None:
        msg: dict[str, Any] = {"type": "targetState", "state": state}
        if reason:
            msg["reason"] = reason
        if detail:
            msg["detail"] = detail
        msg["pos"] = _rounded(t.pos)
        with suppress(Exception):                       # diagnostic only
            msg["tcp"] = self._tcp_now(self._kin_worker)
        grab = isinstance(t, Grab)
        if grab:
            msg["stream"] = t.stream
        msg["ts"] = t.ts
        msg["mode"] = self.mode
        self._last_state = msg
        if grab:
            self._last_grab = msg

        log = logger.warning if state in ("rejected", "backoff", "missed") else logger.info
        because = f" ({reason})" if reason else ""
        more = f" - {detail}" if detail else ""
        what = f"Chwyt {t.label or 'obiekt'}" if grab else f"Cel {t.source}"
        log("%s %s: %s%s%s", what, np.round(t.pos, 3).tolist(), state, because, more)

        self._send(msg)
        if self.on_state is not None:
            try:
                self.on_state(msg)
            except Exception:
                logger.exception("on_state rzucilo wyjatek")

    def _report_grip(self, cmd: GripCommand, state: GripState, reason: str | None = None, *, detail: str = "") -> None:
        msg: dict[str, Any] = {"type": "gripperState", "action": cmd.action, "state": state}
        if reason:
            msg["reason"] = reason
        if detail:
            msg["detail"] = detail
        msg["ts"] = cmd.ts
        msg["mode"] = self.mode
        self._last_grip = msg

        log = logger.warning if state == "rejected" else logger.info
        because = f" ({reason})" if reason else ""
        more = f" - {detail}" if detail else ""
        log("Chwytak %s: %s%s%s", cmd.action, state, because, more)
        self._send(msg)

    # ------------------------------------------------------------------ follow streams: state and reports
    def _emit_follow(self, st: _Stream, state: State, reason: str | None = None, detail: str = "", *,
                     ts: Any = None) -> None:
        """Queues a stream's `targetState`. Call with `_follow_lock` held: the order is the order of events."""
        pos = _rounded(st.sample.pos) if st.sample.pos is not None else None
        self._outbox.put(_FollowReport(state, reason, detail, pos, st.id, st.sample.ts if ts is None else ts,
                                       time.monotonic()))

    def _end_stream(self, st: _Stream, reason: str | None, detail: str = "", *, ts: Any = None,
                    hit: Impact | None = None) -> bool:
        """Ends a stream with `hold` (call with `_follow_lock` held). A running one also gets a `FollowStop`
        for the worker. False if it had already ended."""
        if st.ended:
            return False
        st.ended, st.reason = True, reason or ""
        self._ended.append(st.id)
        self._emit_follow(st, "hold", reason, detail, ts=ts)
        self._follow_stop.set()                         # the listener, or the start's hold-off, stops
        if st is self._pending:
            self._pending = None
        if st is self._stream:
            self._queue.put(FollowStop(st, hit))
        return True

    def _end_streams(self, reason: str) -> None:
        """Ends the running and the waiting stream (a target, STOP, a mode change...)."""
        with self._follow_lock:
            for st in (self._stream, self._pending):
                if st is not None:
                    self._end_stream(st, reason)

    def _reject_stream(self, st: _Stream, reason: str, detail: str = "") -> None:
        with self._follow_lock:
            if st.ended:
                return
            st.ended, st.reason = True, reason
            self._ended.append(st.id)
            self._emit_follow(st, "rejected", reason, detail)
            if st is self._pending:
                self._pending = None

    def _flush_follow_reports(self) -> None:
        """Network thread: sends the queued stream reports (stale ones, from before a reconnect, are dropped)."""
        while True:
            try:
                rep = self._outbox.get_nowait()
            except queue.Empty:
                return
            if time.monotonic() - rep.t > 5.0:
                continue
            msg: dict[str, Any] = {"type": "targetState", "state": rep.state}
            if rep.reason:
                msg["reason"] = rep.reason
            if rep.detail:
                msg["detail"] = rep.detail
            if rep.pos is not None:
                msg["pos"] = rep.pos
            with suppress(Exception):                   # diagnostic only
                msg["tcp"] = self._tcp_now()
            msg["stream"] = rep.stream
            msg["ts"] = rep.ts
            msg["mode"] = self.mode
            self._last_state = self._last_follow = msg

            if rep.state != "moving":
                log = logger.warning if rep.state == "rejected" else logger.info
                because = f" ({rep.reason})" if rep.reason else ""
                more = f" - {rep.detail}" if rep.detail else ""
                log("Sledzenie %s: %s%s%s", rep.stream, rep.state, because, more)
            self._send(msg)
            if self.on_state is not None:
                try:
                    self.on_state(msg)
                except Exception:
                    logger.exception("on_state rzucilo wyjatek")

    def _finish(self, t: Target, state: State, reason: str | None = None, detail: str = "") -> None:
        """Final state of a target: report it, fade the target sphere, release the arm."""
        self._report(t, state, reason, detail=detail)
        self._let_go()

    def _let_go(self) -> None:
        self._hide_target_later()
        self._release()

    def _show_target(self, p: np.ndarray) -> None:
        with self._lock:
            self._target_shown_until = None
        sphere = {"id": "target", "kind": "sphere", "pos": _rounded(p), "size": [0.025], "color": list(TARGET_COLOR)}
        self._send({"type": "overlay", "objects": [sphere]})

    def _hide_target_later(self) -> None:
        with self._lock:
            self._target_shown_until = time.monotonic() + self.target_linger_s

    def _work_loop(self) -> None:
        next_effort_log = 0.0
        while not self._stop.is_set():
            self._release_follow_when_due()
            try:
                t = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self.mode == "dry" and time.monotonic() >= next_effort_log:
                    next_effort_log = time.monotonic() + self.dry_effort_period_s
                    self._log_efforts("dry")
                continue
            self._run(self._latest(t))
        # `stop()`: a running stream's `FollowStop` is left in the queue, so the arm stops here, where its
        # command is, before it is released (else it would finish its way to the follower's last target).
        running = self._stream is not None
        self._drop_follow()
        if running:
            self._freeze()
        self._holders.clear()
        self._release()
        self._restore_speed()

    def _run(self, item: Item | None) -> None:
        """Runs a target or gripper command and any that replace it. An error leaves the arm holding
        its measured pose."""
        running = item
        try:
            while item is not None and not self._stop.is_set():
                running = item
                if isinstance(item, GripCommand):
                    item = self._execute_grip(item)
                elif isinstance(item, FollowStart):
                    item = self._execute_follow_start(item)
                elif isinstance(item, FollowStop):
                    item = self._execute_follow_stop(item)
                elif isinstance(item, Grab):
                    item = self._execute_grab(item)
                else:
                    item = self._execute(item)
        except Exception as exc:
            logger.exception("Kierowca okularow: blad wykonania celu")
            self._hold_after_error(exc)
            if isinstance(running, FollowStop):         # e.g. in its backoff: the stream's hold goes too
                self._drop_follow()
                self._follow_release_at = None
                self._holders.discard("follow")
                if running.then is not None:
                    self._report_busy(running.then)
        finally:
            self._current = None
            self._current_grip = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _hold_after_error(self, exc: Exception) -> None:
        if self.twin.owner == OWNER:
            try:
                self.twin.hold_measured()               # stop pushing towards the last ramp command
            except Exception:
                logger.exception("hold_measured po bledzie")
        if self._current is not None:
            self._report(self._current, "hold", "error", detail=f"{type(exc).__name__}: {exc}"[:200])
        if self._current_grip is not None:
            self._report_grip(self._current_grip, "hold", "error", detail=f"{type(exc).__name__}: {exc}"[:200])

    def _latest(self, item: Item) -> Item:
        """Takes the newest queued item; older unhandled ones are rejected as "busy". A stream's stop
        is never skipped: it is returned to run first (in `_run`, so an error in its backoff holds the
        arm like any other), and the item it overtook runs after it (`FollowStop.then`)."""
        if isinstance(item, FollowStop):
            return item
        while True:
            try:
                newer = self._queue.get_nowait()
            except queue.Empty:
                return item
            if isinstance(newer, FollowStop):
                newer.then = item
                return newer
            self._report_busy(item)
            item = newer

    def _reject_queued(self) -> None:
        """Rejects every queued item as "busy", so the glasses ask the user to point again. A stream's
        stop is not an answer to the glasses: it goes back to the queue and runs later."""
        stops: list[FollowStop] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, FollowStop):
                stops.append(item)
            else:
                self._report_busy(item)
        for stop in stops:
            self._queue.put(stop)

    def _report_busy(self, item: Item) -> None:
        if isinstance(item, GripCommand):
            self._report_grip(item, "rejected", "busy")
        elif isinstance(item, FollowStart):
            self._reject_stream(item.stream, "busy")
        elif isinstance(item, FollowStop):
            self._queue.put(item)                       # never rejected: it runs from the queue
        else:
            self._report(item, "rejected", "busy")

    def _execute_grip(self, cmd: GripCommand) -> Item | None:
        """One gripper command: the arm checks, then the gripper joint to open or closed. Returns the
        next item if a new target interrupted the settle wait."""
        self._current_grip = cmd
        self._reset_cancel()                            # the follower has its own `_follow_stop`
        self._phase = "moving"                          # so a new target interrupts the settle wait
        try:
            self._gate_arm()
            self._gate_gripper()
        except _Abort as exc:
            self._report_grip(cmd, exc.state, exc.reason, detail=exc.detail)
            return None

        self._report_grip(cmd, "accepted")
        if self.mode == "dry":
            self._report_grip(cmd, "hold")
            return None

        value = self.grip_closed if cmd.action == "close" else self.grip_open
        try:
            gen = self._take()
            self._command_arm({self.spec.gripper: value})
            if self._cancel.wait(self._grip_wait()):
                return self._after_cancel_grip(cmd)
            self._check(gen)
        except _Abort as exc:
            self._report_grip(cmd, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None
        self._report_grip(cmd, "done")
        self._release()                                 # stays engaged, holding the gripper
        return None

    def _after_cancel_grip(self, cmd: GripCommand) -> Item | None:
        """The settle wait was interrupted: a new target takes over, anything else holds."""
        cause = self._cancel_cause
        if cause == "retarget":
            try:
                newer = self._latest(self._queue.get_nowait())
            except queue.Empty:
                newer = None
            self._report_grip(cmd, "hold", "retargeted")
            return newer
        if cause in ("relay", "mode"):
            reason = cause
        elif cause == "stop" and self._stop.is_set():
            reason = "stop"
        else:
            reason = self._lost_reason()
        self._report_grip(cmd, "hold", reason)
        self._release()
        return None

    def _execute(self, t: Target) -> Item | None:
        """Handles one target; returns the next item if a new target interrupted this one."""
        self._current = t
        self._reset_cancel()
        # "moving" already during validation (IK and collision checks take time), so a target arriving
        # meanwhile replaces this one instead of queueing and moving the arm after it. A queued
        # gripper command waits: it runs once this target is done.
        self._phase = "moving"
        if self._target_queued():
            self._interrupt("retarget")

        try:
            legs = self._plan(t)
        except _Abort as exc:
            self._report(t, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._report(t, "accepted")
        self._show_target(t.pos if t.source == "hand" else self._hover_point(t.pos))
        if self.mode == "dry":
            self._report(t, "hold")
            return None
        if self._cancel.is_set() or self._held_off(t):
            return self._after_cancel(t)

        try:
            gen = self._take()
            self._recheck(legs)
        except _Abort as exc:
            # Nothing has moved yet. Note: unlike the other paths, this one doesn't fade the target sphere.
            self._report(t, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._cap_speed(1.5 * self.speed + 5.0)         # smoothstep peaks at 1.5x its average speed
        self._guard = ImpactGuard(self.arm)
        self._report(t, "moving")
        try:
            arrived = self._drive(t, legs, gen)
        except _Abort as exc:
            self._finish(t, exc.state, exc.reason, exc.detail)
            return None
        if not arrived:
            return self._after_cancel(t)

        self._finish(t, "reached")                      # stays engaged, holding the target
        return None

    def _held_off(self, t: Target) -> bool:
        """In `real`, a hand target waits `hold_off_s` before moving. True = interrupted meanwhile."""
        if self.mode == "real" and t.source == "hand" and self.hold_off_s > 0:
            return self._cancel.wait(self.hold_off_s)
        return False

    def _drive(self, t: Target, legs: list[_Leg], gen: int) -> bool:
        """Drives the legs in order. False = interrupted (see `_cause`)."""
        for i, leg in enumerate(legs):
            if i > 0:
                self._report(t, "moving")
            if leg.close_gripper:
                self._close_gripper(gen)
            self._tune_guard(self._guard, leg.profile)
            if not self._ramp(leg.goal, leg.speed, gen, profile=leg.profile):
                return False
        return True

    def _after_cancel(self, t: Target) -> Item | None:
        """After an interrupted ramp: a new target takes over, an impact backs off, anything else holds."""
        cause = self._cancel_cause
        if cause == "retarget":
            try:
                newer = self._latest(self._queue.get_nowait())
            except queue.Empty:
                newer = None
            self._report(t, "hold", "retargeted")
            return newer

        if cause == "impact":
            self._backoff(t)
            return None

        if cause in ("relay", "mode"):
            reason = cause
        elif cause == "stop" and self._stop.is_set():
            reason = "stop"
        else:
            reason = self._lost_reason()
        self._finish(t, "hold", reason)
        return None

    # ------------------------------------------------------------------ follow streams: worker items
    def _execute_follow_start(self, item: FollowStart) -> Item | None:
        """A new stream: the checks, the hold-off (real), taking the arm, the follower; then the tick
        listener steps it. Nothing is returned: the stream runs on the loop thread."""
        st = item.stream
        with self._follow_lock:
            if st.ended or st is not self._pending:
                return None                             # replaced, or ended before we got to it
            self._follow_stop.clear()
            sample = st.sample
        try:
            self._gate_follow(sample)
        except _Abort as exc:
            self._reject_stream(st, exc.reason, exc.detail)
            return None

        if self.mode == "dry":
            with self._follow_lock:
                if not st.ended:
                    self._emit_follow(st, "accepted")
                    self._end_stream(st, None)          # dry: checks only, nothing moves
            return None
        if self.mode == "real" and self.hold_off_s > 0 and self._follow_stop.wait(self.hold_off_s):
            return None                                 # ended meanwhile (exit, a target, STOP...)

        try:
            gen = self._take("follow")
        except _Abort as exc:
            self._reject_stream(st, exc.reason, exc.detail)
            return None
        try:
            kin, checker = self._follow_tools()
            follower = Follower(kin, checker, self.spec, limits=self.twin.joint_limits(),
                                joints=self._command(), volume=self.volume, v_max=self.follow_v_max,
                                z_floor=self.follow_floor_m, ik_tol=self.ik_tol)
            guard = ImpactGuard(self.arm)
            self._tune_guard(guard, "normal")
            if not self.twin.status.simulated:
                guard.lag_deg = self.follow_guard_lag_deg
                guard.lag_s = self.follow_guard_lag_s
                guard.vel_still = self.follow_guard_vel_still
        except Exception as exc:
            logger.exception("Sledzenie: start nieudany")
            with self._follow_lock:
                self._end_stream(st, "error", f"{type(exc).__name__}: {exc}"[:200])
            self._release("follow")
            self._restore_speed()
            return None
        self._cap_speed(self.follow_cap_deg_s)

        with self._follow_lock:
            started = not st.ended and st is self._pending
            if started:
                st.follower, st.guard, st.gen = follower, guard, gen
                self._pending, self._stream = None, st
                self._emit_follow(st, "accepted")
        if not started:
            self._release("follow")
            self._restore_speed()
            return None
        self._follow_handle = self.twin.add_tick_listener(self._on_follow_tick)
        return None

    def _gate_follow(self, s: FollowSample) -> None:
        self._gate_arm()
        self._gate_trust(s.trusted, s.session)
        if s.pos is None or not self.volume.contains(s.pos, margin=-self.follow_margin_m):
            raise _Rejected("outside volume")

    def _execute_follow_stop(self, item: FollowStop) -> Item | None:
        """A stream ended (already reported): the listener goes, an impact backs off, the arm is released
        (after a timeout only `follow_release_s` after the last sample). Returns the item the stop
        overtook in the queue, if any."""
        self._stop_stream(item)
        return None if item.then is None else self._latest(item.then)

    def _stop_stream(self, item: FollowStop) -> None:
        st = item.stream
        if st is not self._stream:
            return
        self._drop_follow()
        if item.hit is not None:
            self._follow_backoff(st, item.hit)
        else:
            self._freeze()
        if st.reason == "timeout":
            later = max(0.0, self.follow_release_s - self.follow_timeout_s)
            self._follow_release_at = time.monotonic() + later
            return
        self._release("follow")
        self._restore_speed()

    def _drop_follow(self) -> None:
        """No stream runs any more: the tick listener goes (and a step running right now finishes first,
        so no `set_target` of the stream comes after this)."""
        handle, self._follow_handle = self._follow_handle, None
        if handle is not None:
            self.twin.remove_tick_listener(handle)
        with self._step_lock, self._follow_lock:
            self._stream = None

    def _freeze(self) -> None:
        """The arm's target := its current command, so the supervisor stops where it is now.

        The follower's last `set_target` stays pending after the stream ends, and the supervisor would
        keep chasing it (the wrist turning towards tool-down at 1 rad/s even at a still point): a target
        or grab validated right after the stream saw the command move and held with "moved". The
        gripper keeps its own goal (`Twin.hold_command` only sets the arm joints).
        """
        if self.twin.owner != OWNER:
            return
        try:
            target = self.twin.hold_command(owner=OWNER)
        except RuntimeError:
            return
        if target is None:
            return
        # Until the status shows it (a tick in flight publishes its command after `hold_command`): the
        # next target's validation reads `status.command`.
        want = self._arm_joints(target)
        deadline = time.monotonic() + self.freeze_wait_s
        while time.monotonic() < deadline:
            now = self._arm_joints(self._command())
            if all(abs(now.get(k, math.inf) - v) < 1e-3 for k, v in want.items()):
                return
            if self._stop.wait(0.005):
                return

    def _release_follow_when_due(self) -> None:
        at = self._follow_release_at
        if at is None or time.monotonic() < at or self._stream is not None:
            return
        self._follow_release_at = None
        self._release("follow")
        self._restore_speed()

    def _follow_backoff(self, st: _Stream, hit: Impact) -> None:
        """After an impact while following: hold the measured pose, then back off slowly (as `_backoff`)."""
        tw = self.twin
        tw.hold_measured(dict(tw.status.measured) or None)
        logger.warning("Sledzenie: uderzenie %s %s (%.1f) - wycofanie", hit.kind, hit.joint, hit.value)
        pose = dict(hit.backoff_pose or {})
        if not pose or tw.owner != OWNER:
            return
        self._reset_cancel()
        self._cap_speed(self.backoff_cap_deg_s)
        try:
            arrived = self._retreat(pose, st.gen)
        except _Abort as exc:
            logger.warning("Sledzenie: wycofanie przerwane (%s)", exc.reason)
            return
        if not arrived and self._cancel_cause == "impact":
            tw.hold_measured()                          # second impact: no further backoff
            self._impact = None
        self._reject_queued()

    def _follow_tools(self) -> tuple[RobotKinematics, CollisionChecker]:
        """The follower's own IK and collision checker (as `_tools`: the bench scene without extras).

        Built on the worker when a stream starts, then used only by the tick listener: the worker's
        own `_kin` / `_checker` are not thread-safe.
        """
        version = self.twin.version
        if self._f_checker is None or self._f_version != version:
            scene = sc.build(self.twin.workspace.scene_config())
            self._f_checker = CollisionChecker(scene)
            self._f_kin = RobotKinematics(self.spec, scene.model, sc.PREFIX)
            self._f_version = version
        assert self._f_kin is not None
        return self._f_kin, self._f_checker

    # ------------------------------------------------------------------ follow streams: the loop thread
    def _on_follow_tick(self, sample) -> None:
        """Runs on the twin's loop thread after each tick: one step of the running stream."""
        with self._step_lock:
            st = self._stream
            if st is None or st.ended or st.follower is None or self._follow_stop.is_set():
                return
            try:
                self._follow_step(st, sample)
            except Exception as exc:                    # a listener that raises is removed: end cleanly
                logger.exception("Sledzenie: blad kroku")
                with self._follow_lock:
                    self._end_stream(st, "error", f"{type(exc).__name__}: {exc}"[:200])

    def _follow_step(self, st: _Stream, s) -> None:
        f = st.follower
        assert f is not None and st.guard is not None
        t = float(s.t)
        with self._follow_lock:
            newest, version = st.sample, st.version
        if version != st.seen:
            st.seen, st.t_rx, st.ts = version, t, newest.ts
            if not self.volume.contains(newest.pos, margin=-self.follow_outside_m):
                self._end_follow(st, "outside volume", f"{self.follow_outside_m * 100:.0f} cm poza pudelkiem")
                return
            st.setpoint = f.setpoint(newest.pos)
            if newest.yaw is not None:
                st.yaw = newest.yaw
        if st.t_rx is None:
            st.t_rx = t
        if t - st.t_rx > self.follow_timeout_s:
            self._end_follow(st, "timeout", f"brak punktu od {t - st.t_rx:.1f} s")
            return

        hit = st.guard.feed(t, s.sent, s.measured, s.fresh, efforts=self._efforts(),
                            velocities=self._velocities())
        if hit is not None:
            self._hold_now()
            self._end_follow(st, "impact", f"{hit.kind} {hit.joint}", hit=hit)
            return
        hit = self._gripper_load(st, t, s.measured)
        if hit is not None:
            self._hold_now()
            self._end_follow(st, "impact", "gripper load", hit=hit)
            return

        advance = bool(s.sent)                          # the loop sent nothing (link frozen): don't run ahead
        if not self.twin.status.simulated and st.guard.lagging:
            advance = False                             # a stalled joint: wait for it, or the guard's impact

        dt = self.tick_s if st.t_prev is None else min(max(t - st.t_prev, 0.0), 2.5 * self.tick_s)
        st.t_prev = t
        f.v_max = self.follow_v_max
        out = f.step(dt, st.setpoint if st.setpoint is not None else f.point, st.yaw, advance)
        if out.joints is None:
            if st.fail != out.fail:
                st.fail, st.fail_since = out.fail, t
            limit = self.follow_ik_fail_s if out.fail == "ik" else self.follow_collision_s
            if t - st.fail_since >= limit:
                self._end_follow(st, out.fail, out.detail)
            else:
                # A pause, not a hold: `hold` always ends a stream (the glasses then stop sending).
                self._follow_report(st, t, "moving", None, f"paused: {out.fail}")
            return
        st.fail = ""

        try:
            self.twin.set_target(out.joints, owner=OWNER)
        except RuntimeError:
            self._end_follow(st, self._lost_reason())
            return
        tilt = math.degrees(out.tilt)
        detail = f"tilt {tilt:.0f} deg" if tilt > self.follow_tilt_deg else ""
        self._follow_report(st, t, "moving", None, detail)

    def _gripper_load(self, st: _Stream, t: float, measured) -> Impact | None:
        """The gripper's load during a stream: an `Impact` when its |effort| stays above the stream's rest
        + `gripper_force` for `follow_grip_ticks` ticks. Backends without `gripper_effort`: never. While a
        gripper command runs (closing on an object loads the jaws on purpose) the rest is taken again."""
        b = self.twin.backend
        read = getattr(b, "gripper_effort", None)
        if b is None or read is None:
            return None
        if self._current_grip is not None:
            st.grip_rest, st.grip_over = None, 0
            return None
        try:
            eff = read()
        except Exception:
            return None
        if eff is None:
            return None
        eff = float(eff)
        if st.grip_rest is None:
            st.grip_rest = eff
            logger.info("Sledzenie: wysilek chwytaka w spoczynku %.2f", eff)
            return None
        limit = st.grip_rest + float(getattr(b, "gripper_force", 1.2))
        st.grip_over = st.grip_over + 1 if eff > limit else 0
        if st.grip_over < self.follow_grip_ticks:
            return None
        logger.warning("Sledzenie: obciazenie chwytaka %.2f > %.2f (spoczynek %.2f) przez %d tickow: stop",
                       eff, limit, st.grip_rest, st.grip_over)
        st.grip_over = 0
        assert st.guard is not None
        return Impact(t=t, joint=str(self.spec.gripper), kind="gripper", value=eff,
                      backoff_pose=st.guard.backoff_pose(t, dict(measured or {})))

    def _hold_now(self) -> None:
        """Impact while following: hold the measured pose on the loop thread, right now. The backoff
        waits for the worker, which may be busy (a gripper's settle wait): until then the supervisor
        would keep pushing towards the follower's last target."""
        if self.twin.owner != OWNER:
            return
        try:
            self.twin.hold_measured()
        except Exception:
            logger.exception("Sledzenie: hold_measured po uderzeniu")

    def _end_follow(self, st: _Stream, reason: str, detail: str = "", hit: Impact | None = None) -> None:
        with self._follow_lock:
            self._end_stream(st, reason, detail, ts=st.ts, hit=hit)

    def _follow_report(self, st: _Stream, t: float, state: State, reason: str | None, detail: str) -> None:
        """`moving` from the loop thread (detail: a pause, the tool's tilt): on change, at most every
        `follow_report_s`; again every `follow_heartbeat_s` (loop time)."""
        sig = (state, reason, detail)
        since = t - st.last_report
        if since < self.follow_report_s:
            return
        if sig == st.last_sig and since < self.follow_heartbeat_s:
            return
        with self._follow_lock:
            if st.ended:
                return
            self._emit_follow(st, state, reason, detail, ts=st.ts)
        st.last_report, st.last_sig = t, sig

    # ------------------------------------------------------------------ grab
    def _execute_grab(self, g: Grab) -> Item | None:
        """A grab: the checks, then open, hover, descend, close, lift, and `held` / `missed`.
        Returns the next item if a new target interrupted it."""
        self._current = g
        self._reset_cancel()
        self._phase = "moving"                          # validation takes time: a new target replaces it
        if self._target_queued():
            self._interrupt("retarget")

        try:
            legs = self._plan_grab(g)
        except _Abort as exc:
            self._report(g, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._report(g, "accepted")
        self._show_target(self._grab_points(g)[1])
        if self.mode == "dry":
            self._report(g, "hold")
            return None
        if self._cancel.is_set():
            return self._after_cancel(g)

        try:
            gen = self._take()
            self._recheck(legs)
        except _Abort as exc:
            self._report(g, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._cap_speed(1.5 * self.speed + 5.0)
        self._guard = ImpactGuard(self.arm)
        try:
            arrived = self._drive_grab(g, legs, gen)
            if not arrived:
                return self._after_cancel(g)
            state, detail = self._verify_grab()
        except _Abort as exc:
            self._finish(g, exc.state, exc.reason, exc.detail)
            return None
        self._finish(g, state, detail=detail)           # stays engaged, holding the object up
        return None

    def _gate_grab(self, g: Grab) -> None:
        self._gate_arm()
        self._gate_trust(g.trusted, g.session)
        if self._is_stale(g):
            raise _Rejected("stale", f"wykrycie sprzed {g.age:.2f} s > {self.max_object_age_s:.1f} s")
        if not g.depth:
            raise _Rejected("no depth")
        if self.grab_zone is None or not self.grab_zone.contains(g.pos):
            raise _Rejected("outside grab zone")
        if g.size is not None and g.size[0] * self.grab_width_frac > self.grab_max_width:
            raise _Rejected("too wide", f"szerokosc {g.size[0] * 100:.0f} cm")
        self._gate_gripper()

    def _grab_points(self, g: Grab) -> tuple[np.ndarray, np.ndarray]:
        """(grasp TCP point, hover point). The object stands on the table (z = 0): the tips go to half its
        height, at most its centre, within `grab_tip_z`; the TCP is `TIP_AHEAD` above the tips."""
        h = min(g.size[1], self.grab_max_h) if g.size is not None else self.grab_default_h
        lo, hi = self.grab_tip_z
        tip_z = float(np.clip(min(h / 2.0, float(g.pos[2])), lo, hi))
        grasp = np.array([float(g.pos[0]), float(g.pos[1]), tip_z + TIP_AHEAD])
        hover = grasp.copy()
        hover[2] = min(grasp[2] + self.hover, float(self.volume.hi[2]) - self.hover_ceiling_gap)
        return grasp, hover

    def _plan_grab(self, g: Grab) -> list[_Leg]:
        """Hover, descend and lift legs, each checked (IK tool-down, jumps, collisions)."""
        self._gate_grab(g)
        kin, checker = self._tools()
        command = self._command()
        self._validated_from = self._arm_joints(command)
        prev = self._validated_from

        grasp, hover = self._grab_points(g)
        rotation = down_rotation(self.spec, grasp, g.yaw) if len(self.arm) >= 6 else None
        above = self._solve(kin, _Waypoint(hover, "nad obiekt", rotation, check_down=True),
                            seed={**command, **prev})
        self._check_leg(checker, prev, above, "nad obiekt")
        down = self._solve(kin, _Waypoint(grasp, "w dol", rotation, check_down=True),
                           seed={**command, **above})
        self._check_leg(checker, above, down, "w dol")

        lift = dict(above)
        lift[self.spec.gripper] = self.grip_closed
        return [_Leg(above, self.speed, "normal", "hover"),
                _Leg(down, self.speed * 0.5, "descend", "descend"),
                _Leg(lift, self.speed * 0.5, "descend", "lift")]

    def _drive_grab(self, g: Grab, legs: list[_Leg], gen: int) -> bool:
        """Open, hover, descend, close, lift; `moving` with the leg as `detail`. False = interrupted."""
        grip = self.spec.gripper
        above, down, lift = legs
        self._report(g, "moving", detail="open")
        self._command_arm({grip: self.grip_open})
        if self._cancel.wait(self._grip_wait()):
            return False
        self._check(gen)

        for leg in (above, down):
            self._report(g, "moving", detail=leg.what)
            self._tune_guard(self._guard, leg.profile)
            if not self._ramp(leg.goal, leg.speed, gen, profile=leg.profile):
                return False

        self._report(g, "moving", detail="close")
        self._command_arm({grip: self.grip_closed})
        if self._cancel.wait(self._grip_wait()):
            return False
        self._check(gen)

        self._report(g, "moving", detail=lift.what)
        self._tune_guard(self._guard, lift.profile)
        return self._ramp(lift.goal, lift.speed, gen, profile=lift.profile)

    def _verify_grab(self) -> tuple[State, str]:
        """`held` if the measured jaws stand still wider than empty-closed (`spec.jaw`), else `missed`.
        A backend that doesn't measure the gripper (the A1X today): `held`, detail "assumed"."""
        measured_fn = getattr(self.twin.backend, "gripper_measured", None)
        if measured_fn is not None and not measured_fn():
            return "held", "assumed"
        g = self.spec.gripper
        k = self.spec.joints.index(g)
        kin = self._kin_worker
        closed = float(kin.to_q({g: self.grip_closed})[k])
        first = float(kin.to_q(dict(self.twin.status.measured))[k])
        self._stop.wait(0.1)
        q = float(kin.to_q(dict(self.twin.status.measured))[k])
        jaw = self.spec.jaw
        if q - closed > jaw.margin and abs(q - first) < jaw.still:
            return "held", ""
        return "missed", f"szczeka {q - closed:.4f} od zamknietej"

    def _plan(self, t: Target) -> list[_Leg]:
        self._gate(t)
        return self._legs(t)

    def _gate_arm(self) -> None:
        """The arm checks, shared by targets and gripper commands."""
        tw = self.twin
        if not tw.connected:
            raise _Rejected("not connected")
        if self._estopped():
            raise _Rejected("stop")
        if tw.owner not in (None, OWNER):
            raise _Rejected("busy")
        if self.mode == "real" and not self.real_ok:
            raise _Rejected("not armed")
        if self.mode == "sim" and not tw.status.simulated:
            raise _Rejected("not armed")

    def _gate_gripper(self) -> None:
        if not self.spec.gripper:
            raise _Rejected("no gripper")
        if not self.gripper_drivable():
            raise _Rejected("no gripper", "xcan: 0x051 not deliverable")

    def _gate_trust(self, trusted: bool, session_of: str) -> None:
        with self._lock:
            session = self._session_any if self.allow_untrusted else self._session
        if not (trusted or self.allow_untrusted) or session is None:
            raise _Rejected("untrusted")
        if session_of != session:
            raise _Rejected("session")

    def _gate(self, t: Target) -> None:
        """The checks that need no kinematics, in the order listed in the module docstring."""
        self._gate_arm()
        self._gate_trust(t.trusted, t.session)

        if not self.volume.contains(t.pos):
            raise _Rejected("outside volume")
        if t.source == "object" and not t.depth:
            raise _Rejected("no depth")
        if t.source == "object" and self._is_stale(t):
            raise _Rejected("stale", f"wykrycie sprzed {t.age:.2f} s > {self.max_object_age_s:.1f} s")

    def _is_stale(self, t: Target) -> bool:
        return t.age is not None and (math.isnan(t.age) or t.age > self.max_object_age_s)

    def _legs(self, t: Target) -> list[_Leg]:
        """Target points -> joints (IK) -> legs, each checked for jumps and collisions."""
        kin, checker = self._tools()
        command = self._command()
        self._validated_from = self._arm_joints(command)
        prev = self._validated_from

        legs: list[_Leg] = []
        for wp in self._waypoints(t):
            goal = self._solve(kin, wp, seed={**command, **prev})
            self._check_leg(checker, prev, goal, wp.what)
            legs.append(_Leg(goal, self.speed * wp.speed_factor, wp.profile, wp.what))
            prev = goal

        if t.source == "object" and self.grasp:
            legs.append(self._lift_leg(legs[0].goal))
        return legs

    def _waypoints(self, t: Target) -> list[_Waypoint]:
        if t.source == "hand":
            return [_Waypoint(t.pos, "dlon")]

        # An arm with fewer than 6 joints (SO-101) can't reach an arbitrary rotation: IK solves position
        # only, and the approach axis is checked only before descending (--grasp).
        full_rotation = len(self.arm) >= 6
        rotation = down_rotation(self.spec, t.pos) if full_rotation else None
        hover = _Waypoint(self._hover_point(t.pos), "nad obiekt", rotation,
                          check_down=full_rotation or self.grasp)
        if not self.grasp:
            return [hover]

        touch = _Waypoint(t.pos + np.array([0.0, 0.0, self.touch_dz]), "w dol", rotation,
                          check_down=True, speed_factor=0.5, profile="descend")
        return [hover, touch]

    def _lift_leg(self, hover_goal: Joints) -> _Leg:
        """After the grasp, back up to the hover pose along the same (already checked) path."""
        back = dict(hover_goal)
        gripper = self.spec.gripper
        if gripper:
            back[gripper] = self.grip_closed
        return _Leg(back, self.speed * 0.5, "descend", "w gore", close_gripper=bool(gripper))

    def _solve(self, kin: RobotKinematics, wp: _Waypoint, seed: Joints) -> Joints:
        _, checker = self._tools()
        sol = kin.ik(wp.pos, rotation=wp.rotation, seed=seed, tol=self.ik_tol,
                     accept=lambda joints: checker.config_clear(joints) is None)
        if not sol.ok:
            raise _Rejected("ik", f"IK {wp.what}: blad {sol.pos_err * 1000:.1f} mm")

        if wp.check_down:
            approach = kin.tool_axes(sol.joints)[0]
            tilt = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
            if tilt > self.approach_tol:
                raise _Rejected("ik", f"IK {wp.what}: os podejscia {math.degrees(tilt):.0f} st. od pionu")

        return {k: float(v) for k, v in sol.joints.items() if k in self.arm}

    def _check_leg(self, checker: CollisionChecker, start: Joints, goal: Joints, what: str) -> None:
        jump = self._largest_step(start, goal)
        if jump > self.max_jump:
            raise _Rejected("jump", f"skok {what} {jump:.0f} st. > {self.max_jump:.0f}")
        collision = checker.path_clear(start, goal)
        if collision is not None:
            raise _Rejected("collision", f"kolizja {what}: {collision}")

    def _recheck(self, legs: list[_Leg]) -> None:
        """Someone moved the arm during the hold-off (panel sliders, a jog): the plan was validated
        from a pose the arm no longer has, so hold "moved" and let the wearer pinch again."""
        if not legs:
            return
        start = self._arm_joints(self._command())
        validated = self._validated_from
        if validated is not None and all(abs(v - validated.get(k, math.inf)) < self.unchanged_deg
                                         for k, v in start.items()):
            return
        moved = self._largest_step(validated or {}, start) if validated else math.inf
        raise _Abort("hold", "moved", f"rozkaz zmieniony w czasie odczekania: o {moved:.0f} st.")

    def _hover_point(self, pos: np.ndarray) -> np.ndarray:
        p = np.array(pos, float)
        p[2] = min(float(pos[2]) + self.hover, float(self.volume.hi[2]) - self.hover_ceiling_gap)
        return p

    def _take(self, who: str = "worker") -> int:
        """Takes the arm as `Twin.move` does: wait for homing, `claim` with `gen`, engage. `who` in the
        driver ("worker", "follow") holds it until `_release(who)`."""
        if self._stream is None and self._follow_release_at is not None:
            self._follow_release_at = None              # an ended stream's delayed release: taken over now
            self._holders.discard("follow")
        tw = self.twin
        try:
            tw._wait_for_homing()
        except RuntimeError:
            raise _Rejected("busy") from None

        gen = tw.preempt_gen
        try:
            tw.claim(OWNER, preempt=lambda: self._interrupt("preempt"), gen=gen)
        except RuntimeError:
            raise _Rejected("stop" if self._estopped() else "busy") from None
        if not tw.set_engaged(True, owner=OWNER):
            self._release(who)
            raise _Rejected("stop" if self._estopped() else "busy")
        self._holders.add(who)
        return gen

    def _release(self, who: str = "worker") -> None:
        """`who` no longer needs the arm; the twin gets it back when nobody in the driver does."""
        self._holders.discard(who)
        if self._holders:
            return
        with suppress(Exception):
            self.twin.release(OWNER)

    def _command_arm(self, target: Joints) -> None:
        """`set_target` as owner; a refusal means the arm was taken away (STOP or another owner)."""
        try:
            self.twin.set_target(target, owner=OWNER)
        except RuntimeError:
            raise _Abort("hold", self._lost_reason()) from None

    def _check(self, gen: int) -> None:
        tw = self.twin
        if self._estopped():
            raise _Abort("hold", "stop")
        if tw.preempt_gen != gen:
            raise _Abort("hold", "preempted")
        if not tw.connected:
            raise _Abort("hold", "not connected")

    def _close_gripper(self, gen: int) -> None:
        self.twin.set_target({self.spec.gripper: self.grip_closed}, owner=OWNER)
        if self._cancel.wait(self._grip_wait()):
            return
        if self.twin.preempt_gen != gen:
            raise _Abort("hold", self._lost_reason())

    def _cap_speed(self, deg_s: float) -> None:
        """Caps the supervisor speed while we move, never above the panel's setting."""
        if not self._capped:
            self._prev_cap = getattr(self.twin, "_speed_cap", None)
            self._capped = True
        panel_cap = self._prev_cap if self._prev_cap is not None else math.inf
        self.twin.set_speed(min(panel_cap, float(deg_s)))

    def _restore_speed(self) -> None:
        if not self._capped or self._holders:
            return
        self._capped = False
        try:
            self.twin.set_speed(self._prev_cap)
        except Exception:
            logger.exception("Przywracanie granicy predkosci")

    def _tools(self) -> tuple[RobotKinematics, CollisionChecker]:
        """IK and collision checker for the bench scene WITHOUT extras (task objects, the card).

        Built on the worker thread and rebuilt when the twin's scene changes (`twin.version`).
        """
        version = self.twin.version
        if self._checker is None or self._checker_version != version:
            scene = sc.build(self.twin.workspace.scene_config())
            self._checker = CollisionChecker(scene)
            self._kin = RobotKinematics(self.spec, scene.model, sc.PREFIX)
            self._checker_version = version
        assert self._kin is not None
        return self._kin, self._checker

    def _estopped(self) -> bool:
        from ..control.safety import SafetyState

        return self.twin.safety_state == SafetyState.ESTOP

    def _command(self) -> Joints:
        tw = self.twin
        return {k: float(v) for k, v in (tw.status.command or tw.joints()).items()}

    def _arm_joints(self, joints: Mapping[str, float]) -> Joints:
        """Arm joints only, without the gripper."""
        return {k: float(joints[k]) for k in self.arm if k in joints}

    def _largest_step(self, start: Mapping[str, float], goal: Mapping[str, float]) -> float:
        """Largest arm-joint move between two poses, deg."""
        return max((abs(v - start.get(k, v)) for k, v in goal.items() if k in self.arm), default=0.0)

    def center_pose(self) -> Joints | None:
        """Arm joints with the TCP above the centre of the reach box (IK from home); None if IK fails."""
        kin, _ = self._tools()
        sol = kin.ik(self.volume.center, seed=self.spec.home, tol=self.ik_tol)
        if not sol.ok:
            return None
        return {k: float(v) for k, v in sol.joints.items() if k in self.arm}

    def start_center(self, speed_deg_s: float = 10.0) -> str | None:
        """Moves above the centre of the reach box in stages (`spec.home_stages`), under the impact guard.
        From home, every point in the box is a jump over 45 deg, so the first target would be refused.

        Call BEFORE `start()`, from the thread that then starts the driver. An impact holds the measured
        pose without backing off. Returns None on arrival, otherwise the reason.
        """
        tw = self.twin
        if not tw.connected:
            return "not connected"
        if self._estopped():
            return "stop"
        if not (tw.status.simulated or (self.mode == "real" and self.real_ok)):
            return "not armed"
        goal = self.center_pose()
        if goal is None:
            return "ik"

        _, checker = self._tools()
        steps, collision = self._stage_plan(self._arm_joints(self._command()), goal, checker)
        if collision is not None:
            logger.warning("Start nad srodek pudelka: %s", collision)
            return "collision"

        with self._lock:
            self._cancel.clear()
            self._cause = ""
        try:
            gen = self._take()
        except _Abort as exc:
            return exc.reason

        self._cap_speed(1.5 * speed_deg_s + 5.0)
        self._guard = ImpactGuard(self.arm)
        try:
            for step in steps:
                self._tune_guard(self._guard, "normal")
                self._guard.lag_deg = self.center_lag_deg
                if not self._ramp(step, speed_deg_s, gen, arrive_lag_s=self.center_arrive_lag_s):
                    return self._center_interrupted()
            return None
        except _Abort as exc:
            return exc.reason
        finally:
            self._guard = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _center_interrupted(self) -> str:
        cause = self._cancel_cause
        self.twin.hold_measured()
        hit, self._impact = self._impact, None
        if cause == "impact" and hit is not None:
            logger.warning("Start nad srodek pudelka: uderzenie %s %s - trzymanie", hit.kind, hit.joint)
            return f"impact {hit.kind} {hit.joint}"
        return cause or "preempted"

    def _stage_plan(self, start: Joints, goal: Joints,
                    checker: CollisionChecker) -> tuple[list[Joints], str | None]:
        """Intermediate poses as in staged homing: joint groups one after another (also in reverse),
        then the whole goal. The first plan with clear paths wins; with no stages, a straight move.
        Returns (poses, None) or ([], the first collision)."""
        stages = [tuple(j for j in stage if j in self.arm) for stage in self.spec.home_stages]
        stages = [stage for stage in stages if stage]
        orders = [stages, stages[::-1]] if stages else []

        first_collision: str | None = None
        for order in [*orders, []]:
            poses, collision = self._walk_stages(start, goal, [*order, self.arm], checker)
            if collision is None:
                return poses, None
            first_collision = first_collision or collision
        return [], first_collision

    def _walk_stages(self, start: Joints, goal: Joints, stages: list[tuple[str, ...]],
                     checker: CollisionChecker) -> tuple[list[Joints], str | None]:
        pose = dict(start)
        poses: list[Joints] = []
        for stage in stages:
            nxt = {**pose, **{j: goal[j] for j in stage}}
            if self._largest_step(pose, nxt) < 1e-3:
                continue
            collision = checker.path_clear(pose, nxt)
            if collision is not None:
                return poses, collision
            poses.append(nxt)
            pose = nxt
        return poses, None

    def _ramp(self, goal: Mapping[str, float], speed: float, gen: int, *, guarded: bool = True,
              profile: Profile = "normal", arrive_lag_s: float | None = None) -> bool:
        """Smoothstep ramp to `goal`, then wait for arrival. True = arrived, False = interrupted (`_cause`).
        Raises `_Abort` if the arm is taken away or doesn't arrive (it then holds its measured pose)."""
        command = self._command()
        start = {k: command.get(k, float(v)) for k, v in goal.items()}
        goal = {k: float(v) for k, v in goal.items()}
        duration = max(0.5, self._largest_step(start, goal) / max(speed, 1e-3))
        with self._listening(guarded):
            self._phase = "moving" if guarded else "backoff"
            if not self._sweep(start, goal, duration, gen):
                return False
            self._tighten_guard_for_arrival(arrive_lag_s)
            return self._settle(goal, gen, profile)

    def _sweep(self, start: Joints, goal: Joints, duration: float, gen: int) -> bool:
        clock = _LoopClock(self.twin)
        while True:
            if self._cancel.is_set():
                return False
            self._check(gen)

            s = smoothstep(clock.elapsed() / duration)
            self._command_arm({k: start[k] + s * (goal[k] - start[k]) for k in goal})
            if s >= 1.0:
                return True

            if self._cancel.wait(self.tick_s):
                return False

    def _tighten_guard_for_arrival(self, arrive_lag_s: float | None) -> None:
        guard = self._guard
        if guard is None:
            return
        guard.lag_deg = self.arrive_lag_deg
        guard.lag_s = self.arrive_lag_s if arrive_lag_s is None else float(arrive_lag_s)

    def _settle(self, goal: Joints, gen: int, profile: Profile) -> bool:
        """Waits until every arm joint is within `arrive_deg` of the goal and still. After
        `arrive_timeout_s`: holds the measured pose and raises `_Abort("hold", "not arrived")`."""
        deadline = time.monotonic() + self.arrive_timeout_s
        previous: _Sample | None = None
        while True:
            if self._cancel.wait(0.05):
                return False
            self._check(gen)

            status = self.twin.status
            sample = _Sample(status.measured_t, dict(status.measured))
            still = False
            if previous is None or sample.t > previous.t:
                still = previous is not None and self._stood_still(previous, sample)
                previous = sample

            far = self._far_joints(goal, sample.joints)
            if not far and still:
                if profile == "descend":
                    return True                         # no trim when descending: keep pressing on the object
                return self._trim(goal, gen)

            if time.monotonic() > deadline:
                # A stalled joint must not keep pushing into the obstacle towards a goal it can't reach.
                self.twin.hold_measured()
                misses = ", ".join(f"{k} {sample.joints.get(k, math.nan) - goal[k]:+.1f} st." for k in far)
                raise _Abort("hold", "not arrived", misses)

    def _far_joints(self, goal: Joints, measured: Joints) -> list[str]:
        return [k for k in goal if k in self.arm and abs(measured.get(k, goal[k]) - goal[k]) > self.arrive_deg]

    def _stood_still(self, before: _Sample, after: _Sample) -> bool:
        dt = after.t - before.t
        return all(abs(v - before.joints.get(k, v)) / dt < self.still_deg_s
                   for k, v in after.joints.items() if k in self.arm)

    def _trim(self, goal: Joints, gen: int) -> bool:
        """Commands goal + error, a few times, so the measured pose lands on the IK goal. False = interrupted."""
        guard = self._guard
        if guard is not None and guard.effort_abs:
            loaded = self._loaded_joints(self._efforts() or {}, guard.effort_abs)
            if loaded:
                logger.info("Bez docinania: wysilek %s ponad polowe progu", loaded)
                return True

        command = dict(goal)
        for _ in range(self.trim_steps):
            measured = dict(self.twin.status.measured)
            error = {k: goal[k] - measured[k] for k in goal if k in self.arm and k in measured}
            if not error or max(abs(v) for v in error.values()) < self.trim_deg:
                break
            for k, v in error.items():
                command[k] = float(np.clip(command[k] + v, goal[k] - self.trim_max_deg, goal[k] + self.trim_max_deg))
            self._command_arm(command)
            if self._cancel.wait(0.4):
                return False
            self._check(gen)
        return True

    @contextmanager
    def _listening(self, enabled: bool = True) -> Iterator[None]:
        """The impact guard listens to the twin's loop ticks for the duration of the block."""
        handle = self.twin.add_tick_listener(self._on_tick) if enabled else None
        try:
            yield
        finally:
            if handle is not None:
                self.twin.remove_tick_listener(handle)

    def _on_tick(self, sample) -> None:
        """Runs on the twin's loop thread: only records the impact and interrupts the ramp."""
        guard = self._guard
        if guard is None or self._cancel.is_set() or time.monotonic() < self._guard_from:
            return
        hit = guard.feed(sample.t, sample.sent, sample.measured, sample.fresh,
                         efforts=self._efforts(), velocities=self._velocities())
        if hit is not None:
            self._impact = hit
            self._interrupt("impact")

    def _tune_guard(self, guard: ImpactGuard, profile: Profile) -> None:
        """Guard thresholds for one leg. One guard serves the whole target, so its buffer of good
        samples (the backoff poses) carries across legs."""
        descend = profile == "descend"
        guard.lag_deg = self.lag_deg_descend if descend else self.lag_deg
        guard.lag_s = self.lag_s

        limits = self._effort_limits(descend) or {}
        # A joint already loaded at rest (weight, spring) would trip on every move: lag rule only for it.
        at_rest = (self._log_efforts("start odcinka") or {}) if limits else {}
        for joint in self._loaded_joints(at_rest, limits):
            logger.warning("Wysilek %s w spoczynku %.1f > %.0f%% progu %.1f - regula wysilku wylaczona "
                           "dla tego stawu w tym ruchu", joint, at_rest[joint], 100 * self.effort_rest_frac,
                           limits[joint])
            del limits[joint]
        guard.effort_abs = limits

    def _loaded_joints(self, efforts: Mapping[str, float], limits: Mapping[str, float]) -> list[str]:
        return [j for j, limit in limits.items() if j in efforts and abs(efforts[j]) > self.effort_rest_frac * limit]

    def _backoff(self, t: Target) -> None:
        """After an impact: hold the measured pose, then move slowly to the backoff pose.
        A second impact during the backoff only holds."""
        tw = self.twin
        hit, self._impact = self._impact, None
        # The measured pose, not the braking target: a joint pressed against an obstacle stops pushing.
        tw.hold_measured(dict(tw.status.measured) or None)

        why = f"impact {hit.kind} {hit.joint}" if hit is not None else "impact"
        logger.warning("Uderzenie: %s (%.1f) - wycofanie", why, hit.value if hit is not None else math.nan)
        pose = dict(hit.backoff_pose or {}) if hit is not None else {}
        gen = tw.preempt_gen
        if not pose or tw.owner != OWNER:
            self._finish(t, "hold", why)
            return

        self._reset_cancel()
        self._cap_speed(self.backoff_cap_deg_s)
        try:
            arrived = self._retreat(pose, gen)
        except _Abort as exc:
            self._finish(t, "hold", f"{why}; {exc.reason}", exc.detail)
            return

        cause = self._cancel_cause
        if arrived:
            self._report(t, "backoff", why)
        elif cause == "impact":
            tw.hold_measured()                          # second impact: no further backoff
            self._impact = None
            self._report(t, "hold", why)
        else:
            self._report(t, "hold", cause)
        self._reject_queued()
        self._let_go()

    def _retreat(self, pose: Joints, gen: int) -> bool:
        """Ramps to the backoff pose under a fresh guard that starts late. False = interrupted."""
        self._guard = ImpactGuard(self.arm, effort_abs=self._effort_limits(False))
        self._guard_from = time.monotonic() + self.backoff_grace_s
        try:
            with self._listening():
                if self._cancel.wait(0.06):             # ramp from the command AFTER the hold (next loop tick)
                    raise _Abort("hold", self._cancel_cause or "preempted")
                return self._ramp(pose, self.backoff_speed, gen, guarded=False)
        finally:
            self._guard = None

    def _effort_limits(self, descend: bool) -> Joints | None:
        """Per-joint effort thresholds; {} = no effort rule, None = the backend can't measure effort."""
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated:
            return self._sim_effort_limits(descend)
        try:
            probe = b.efforts()
        except Exception:
            probe = None
        if probe is None:
            return None
        limit = self.real_effort_abs_descend if descend else self.real_effort_abs
        return dict.fromkeys(self.arm, limit)

    def _sim_effort_limits(self, descend: bool) -> Joints:
        model = self.twin.scene.model
        ids = [model.actuator(sc.PREFIX + j).id for j in self.arm]
        self._effort_ids = np.array(ids)
        if self.spec.name in self.sim_effort_off:
            return {}

        frac = self.sim_effort_frac_descend if descend else self.sim_effort_frac
        ranges = model.actuator_forcerange[ids]
        limited = model.actuator_forcelimited[ids]
        return {j: frac * float(max(abs(lo), abs(hi)))
                for j, (lo, hi), is_limited in zip(self.arm, ranges, limited, strict=True)
                if is_limited and hi > lo}

    def _efforts(self) -> Joints | None:
        """Joint efforts from the backend; in sim, the arm actuators' `actuator_force` (N m)."""
        tw = self.twin
        b = tw.backend
        if b is None:
            return None
        if not b.info.simulated:
            try:
                return b.efforts()
            except Exception:
                return None

        ids = self._effort_ids
        if ids is None or not tw.lock.acquire(timeout=0.005):
            return None
        try:
            forces = tw.scene.data.actuator_force[ids]
            return {j: float(f) for j, f in zip(self.arm, forces, strict=True)}
        finally:
            tw.lock.release()

    def _velocities(self) -> dict[str, float] | None:
        b = self.twin.backend
        if b is None or b.info.simulated:
            return None
        try:
            return b.velocities()
        except Exception:
            return None

    def _log_efforts(self, what: str) -> Joints | None:
        """Logs arm joint efforts: the A1X guard thresholds are guesses, and this gives real numbers."""
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated and self._effort_ids is None:
            self._effort_limits(False)                  # sets up `_effort_ids`
        efforts = self._efforts()
        if efforts:
            logger.info("Wysilek stawow (%s): %s", what, {k: round(float(v), 2) for k, v in efforts.items()})
        return efforts
