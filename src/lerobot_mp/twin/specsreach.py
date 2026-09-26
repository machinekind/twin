"""Spectacles glasses -> a target in the reach box -> arm motion, backing off gently on impact.

    drv = SpecsReach(twin, relay_url="ws://127.0.0.1:8080/twin", mode="sim")
    drv.start()                                        # network thread + worker thread
    ...
    drv.stop()

The twin connects to the relay (`server.js` in the specs-lens repo) as `/twin`:

* on connect it sends the overlay: the reach box (`kind: "volume"`) and a TCP sphere,
  then the TCP sphere every 0.2 s, and a `target` sphere for an accepted target;
* it receives `registration` (the glasses' session and whether it is trusted)
  and `target` (a hand point or an object centre, base frame, metres);
* it answers every target with `targetState`: accepted / rejected / moving / reached /
  backoff / hold. `type` must be the first key: the relay reads it with a regex.

The network thread only receives and queues targets. One worker thread does everything
else: validation, IK, collisions, motion, backoff. The checks run in this order, and their
names are the `reason` sent back: not connected, stop, busy, not armed, untrusted,
session, outside volume, no depth, stale, ik, jump, collision.

Motion is our own 50 Hz smoothstep ramp rather than `Twin.move`, because every tick must
be interruptible: by a new target (continuing from the current command), an impact
(`ImpactGuard` on the loop's tick listener) or losing the arm (STOP, Home). Ownership works
as in `Twin.move`: `claim` with `gen`, engage, `set_target` as owner on every tick, and the
ramp clock pauses while the loop does. On arrival the arm is released but stays engaged,
holding the target.

Modes: `dry` runs every check and shows the planned target without moving (accepted, then
hold); `sim` moves only a simulated arm; `real` moves the real arm, only with `real_ok`,
and waits `hold_off_s` after a hand target. While moving, the supervisor speed cap
(`Twin.set_speed`) drops to ~1.5x the ramp speed + 5 deg/s, and to 15 deg/s during
backoff; it is restored afterwards. In `real`, losing the relay stops the motion (hold).
The glasses have no STOP: use Enter/Ctrl-C in the console or STOP in the panel.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args

import numpy as np

from . import scene as sc
from .collision import CollisionChecker
from .impact import Impact, ImpactGuard
from .kinematics import RobotKinematics
from .reach import ReachVolume
from .runtime import Twin, smoothstep

logger = logging.getLogger(__name__)

Mode = Literal["dry", "sim", "real"]
Source = Literal["hand", "object"]
#: `targetState.state` values: a contract with the glasses.
State = Literal["accepted", "rejected", "moving", "reached", "backoff", "hold"]
#: "descend" is slower with a tighter guard: going down to an object and back up.
Profile = Literal["normal", "descend"]
#: Read from other threads (network, panel).
Phase = Literal["idle", "moving", "backoff"]
Cause = Literal["", "retarget", "impact", "preempt", "stop", "mode", "relay"]

MODES: tuple[Mode, ...] = get_args(Mode)
SOURCES: tuple[Source, ...] = get_args(Source)
OWNER = "specs"

# Overlay colours, RGB 0..1.
TCP_COLOR = (1.0, 0.8, 0.1)
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


def down_rotation(spec, p: np.ndarray) -> np.ndarray:
    """TCP rotation in the base frame: approach axis pointing down, jaws closing horizontally,
    across the line from the base to `p`."""
    a = np.asarray(spec.tcp_approach, float)
    c = np.asarray(spec.tcp_closing, float)
    local = np.column_stack([a, c, np.cross(a, c)])

    bearing = math.atan2(float(p[1]), float(p[0]))
    down = np.array([0.0, 0.0, -1.0])
    side = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
    world = np.column_stack([down, side, np.cross(down, side)])

    return world @ local.T


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

    grip_closed = 0.0
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

    def __init__(self, twin: Twin, *, relay_url: str, volume: ReachVolume | None = None, mode: Mode = "dry",
                 hover: float = 0.08, touch_dz: float = 0.01, speed_deg_s: float = 20.0,
                 backoff_speed_deg_s: float = 8.0, max_jump_deg: float = 150.0, grasp: bool = False,
                 allow_untrusted: bool = False, real_ok: bool = False, hold_off_s: float = 1.0,
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
        self.on_state = on_state

        self._stop = threading.Event()
        self._net: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue[Target] = queue.Queue()
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
        # The command the target was validated from; `_recheck` compares against it.
        self._validated_from: Joints | None = None

    def start(self) -> None:
        if self._net is not None:
            return
        self._stop.clear()
        self._net = threading.Thread(target=self._net_loop, name="okulary-siec", daemon=True)
        self._worker = threading.Thread(target=self._work_loop, name="okulary-ruch", daemon=True)
        self._net.start()
        self._worker.start()

    def stop(self) -> None:
        """Stops both threads. A moving arm holds its measured pose and is released."""
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
        if changed and self._phase == "moving":
            self._interrupt("mode")

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
            "session": self._session,
            "phase": self._phase,
            "last": self._last_state,
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
            self._send({"type": "overlay", "objects": [self.volume.overlay_object(), self._tcp_object()]})
            self._pump(ws)
        except Exception as exc:
            if not self._stop.is_set():
                logger.warning("Przekaznik rozlaczony (%s) - ponowie za %.0f s", exc, self.reconnect_s)
        finally:
            self._ws = None
            # A real arm doesn't keep moving without the relay: the glasses can no longer see its state.
            if self.mode == "real" and self._phase == "moving" and not self._stop.is_set():
                self._interrupt("relay")
            with suppress(Exception):
                ws.close()

    def _pump(self, ws) -> None:
        """Receives messages, sending the TCP sphere every `tcp_period_s` in between."""
        next_tcp = time.monotonic() + self.tcp_period_s
        while not self._stop.is_set():
            try:
                text = ws.recv(timeout=max(0.0, next_tcp - time.monotonic()))
            except TimeoutError:
                text = None
            if text is not None:
                self._on_message(text)

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
        return {"id": "tcp", "kind": "sphere", "pos": self._tcp_now(), "size": [0.02], "color": list(TCP_COLOR)}

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

    def _on_registration(self, msg: dict) -> None:
        session = msg.get("session")
        if session is None:
            return
        with self._lock:
            self._session_any = str(session)
            # Any registration without `trusted: true`, even a repeat of the same session, revokes trust.
            self._session = str(session) if msg.get("trusted") is True else None

    def _on_target(self, msg: dict, text: str | bytes) -> None:
        t = parse_target(msg, time.monotonic())
        if t is None:
            logger.warning("Zepsuty cel z okularow: %.200s", text)
            return
        self._queue.put(t)
        # A new target interrupts the move; the worker continues from the current command.
        if self._phase == "moving":
            self._interrupt("retarget")

    def _interrupt(self, cause: Cause) -> None:
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
        msg["ts"] = t.ts
        msg["mode"] = self.mode
        self._last_state = msg

        log = logger.warning if state in ("rejected", "backoff") else logger.info
        because = f" ({reason})" if reason else ""
        more = f" - {detail}" if detail else ""
        log("Cel %s %s: %s%s%s", t.source, np.round(t.pos, 3).tolist(), state, because, more)

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
            try:
                t = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self.mode == "dry" and time.monotonic() >= next_effort_log:
                    next_effort_log = time.monotonic() + self.dry_effort_period_s
                    self._log_efforts("dry")
                continue
            self._run(self._latest(t))
        self._release()
        self._restore_speed()

    def _run(self, t: Target | None) -> None:
        """Runs a target and any that replace it. An error leaves the arm holding its measured pose."""
        try:
            while t is not None and not self._stop.is_set():
                t = self._execute(t)
        except Exception as exc:
            logger.exception("Kierowca okularow: blad wykonania celu")
            self._hold_after_error(exc)
        finally:
            self._current = None
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

    def _latest(self, t: Target) -> Target:
        """Takes the newest queued target; older unhandled ones are rejected as "busy"."""
        while True:
            try:
                newer = self._queue.get_nowait()
            except queue.Empty:
                return t
            self._report(t, "rejected", "busy")
            t = newer

    def _reject_queued(self) -> None:
        """Rejects every queued target as "busy", so the glasses ask the user to point again."""
        while True:
            try:
                t = self._queue.get_nowait()
            except queue.Empty:
                return
            self._report(t, "rejected", "busy")

    def _execute(self, t: Target) -> Target | None:
        """Handles one target; returns the next one if a new target interrupted this one."""
        self._current = t
        self._reset_cancel()
        # "moving" already during validation (IK and collision checks take time), so a target arriving
        # meanwhile replaces this one instead of queueing and moving the arm after it.
        self._phase = "moving"
        if not self._queue.empty():
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

    def _after_cancel(self, t: Target) -> Target | None:
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

    def _plan(self, t: Target) -> list[_Leg]:
        self._gate(t)
        return self._legs(t)

    def _gate(self, t: Target) -> None:
        """The checks that need no kinematics, in the order listed in the module docstring."""
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

        with self._lock:
            session = self._session_any if self.allow_untrusted else self._session
        if not (t.trusted or self.allow_untrusted) or session is None:
            raise _Rejected("untrusted")
        if t.session != session:
            raise _Rejected("session")

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
        sol = kin.ik(wp.pos, rotation=wp.rotation, seed=seed, tol=self.ik_tol)
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

    def _take(self) -> int:
        """Takes the arm as `Twin.move` does: wait for homing, `claim` with `gen`, engage."""
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
            self._release()
            raise _Rejected("stop" if self._estopped() else "busy")
        return gen

    def _release(self) -> None:
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
        if self._cancel.wait(self.grip_wait_s):
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
        if not self._capped:
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
