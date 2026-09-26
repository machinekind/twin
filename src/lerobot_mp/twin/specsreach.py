"""Spectacles glasses -> a target in the A1X reach box -> a guarded arm move through the twin.

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
else. The checks run in this order, and their names are the `reason` sent back: not
connected, stop, busy, not armed, untrusted, session, outside volume, no depth, stale,
ik, jump, collision.

Everything about the arm comes from `galaxeo.arm`: the reach box (`ReachBox`), IK and FK
of the tool frame, the self/table collision checks, and the move itself (`Motion.move`:
smoothstep ramp, impact guard, arrival, backoff). This module only supplies targets:
parsing, session trust, hover/descend/lift legs, grasp, modes and the relay protocol.
Moves go through `armlink.TwinTransport`, i.e. `Twin.set_target` as owner "specs", so
STOP, Home, the panel, the supervisor and ownership work as for any other owner, and
the twin's backend stays the only writer on the CAN bus.

Modes: `dry` runs every check and shows the planned target without moving (accepted, then
hold); `sim` moves only a simulated arm; `real` moves the real arm, only with `real_ok`,
and waits `hold_off_s` after a hand target. In `real`, losing the relay stops the motion
(hold). The glasses have no STOP: use Enter/Ctrl-C in the console or STOP in the panel.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np
from galaxeo.arm import JOINTS, Arm, Kinematics, Motion, MoveResult, ReachBox, down_rotation
from galaxeo.arm.motion import BACKOFF_CAP_DEG_S, MAX_JUMP_DEG

from .armlink import TwinTransport, to_joints, to_q
from .runtime import Twin

logger = logging.getLogger(__name__)

Mode = Literal["dry", "sim", "real"]
Source = Literal["hand", "object"]
#: `targetState.state` values: a contract with the glasses.
State = Literal["accepted", "rejected", "moving", "reached", "backoff", "hold"]
#: "descend" is slower with a tighter guard: going down to an object and back up.
Profile = Literal["normal", "descend"]
#: Read from other threads (network, panel).
Phase = Literal["idle", "moving", "backoff"]
Cause = Literal["", "retarget", "preempt", "stop", "mode", "relay", "not connected"]

MODES: tuple[Mode, ...] = get_args(Mode)
SOURCES: tuple[Source, ...] = get_args(Source)
OWNER = "specs"
#: The only arm galaxeo.arm models.
ROBOT = "a1x"
#: Longest piece of one guarded move [deg]: below galaxeo's 45 deg jump limit, with room
#: for the measured start sagging a degree or two off the previous goal.
PIECE_DEG = 40.0
assert PIECE_DEG < MAX_JUMP_DEG

# Overlay colours, RGB 0..1.
REACH_COLOR = (0.2, 0.65, 1.0)
TCP_COLOR = (1.0, 0.8, 0.1)
TARGET_COLOR = (0.2, 0.9, 0.35)


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
    start: np.ndarray                 # rad; the path was collision-checked from here
    goal: np.ndarray                  # rad
    speed: float                      # deg/s of the fastest joint
    profile: Profile
    what: str
    close_gripper: bool = False


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


def overlay_object(box: ReachBox, alpha: float = 0.35) -> dict[str, Any]:
    """The reach box as the glasses' overlay object of kind `volume` (base frame, m)."""
    return {
        "id": "reach",
        "kind": "volume",
        "pos": [round(float(v), 4) for v in box.center],
        "size": [round(float(v), 4) for v in box.size],
        "color": list(REACH_COLOR),
        "alpha": float(alpha),
        "label": "zasieg",
    }


def pieces(start: np.ndarray, goal: np.ndarray, max_deg: float = PIECE_DEG) -> list[np.ndarray]:
    """The straight joint-space path start -> goal cut into equal pieces of at most `max_deg`."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    n = max(1, math.ceil(math.degrees(float(np.max(np.abs(goal - start)))) / max_deg - 1e-9))
    return [start + (goal - start) * k / n for k in range(1, n + 1)]


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


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"tryb {mode!r}: dozwolone {', '.join(MODES)}")


def _rounded(v: np.ndarray | list[float]) -> list[float]:
    return [round(float(x), 4) for x in v]


def _deg(q: np.ndarray) -> float:
    """Largest joint move [deg] of a joint-space difference [rad]."""
    return math.degrees(float(np.max(np.abs(q)))) if len(q) else 0.0


class SpecsReach:
    """Drives the A1X from the glasses: a relay client plus one worker thread."""

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

    # `start_center` runs at 10 deg/s: a joint stalled with 3-6 deg of ramp ahead never exceeds
    # galaxeo's 6 deg lag threshold, while the A1X backend's "deaf" timer (3 deg for 1 s,
    # robot/a1x.py) already counts above 3 deg. 4 deg during the ramp and 0.2 s on arrival
    # trip at least 0.3 s before the backend faults.
    center_lag_deg = 4.0
    center_arrive_lag_s = 0.2

    # dry mode logs joint efforts this often, to pick thresholds on the first real session.
    dry_effort_period_s = 1.0

    def __init__(self, twin: Twin, *, relay_url: str, volume: ReachBox | None = None, mode: Mode = "dry",
                 hover: float = 0.08, touch_dz: float = 0.01, speed_deg_s: float = 20.0,
                 max_jump_deg: float = 150.0, grasp: bool = False, allow_untrusted: bool = False,
                 real_ok: bool = False, hold_off_s: float = 1.0, on_state: Callable[[dict], None] | None = None):
        _check_mode(mode)
        if twin.spec.name != ROBOT:
            raise ValueError(f"okulary prowadza tylko A1X (galaxeo.arm), a stanowisko ma ramie {twin.spec.name!r}")
        self.twin = twin
        self.spec = twin.spec
        self.relay_url = relay_url
        # Planning only (reach box, IK, FK, collisions); moves go through `link`.
        self.arm = Arm()
        self.volume = volume if volume is not None else self.arm.reach
        self.mode: Mode = mode
        self.real_ok = bool(real_ok)
        self.hover = float(hover)
        self.touch_dz = float(touch_dz)
        self.speed = float(speed_deg_s)
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
        # `preempt_gen` of the arm while we own it; checked on every tick of a move.
        self._gen: int | None = None

        self.link = TwinTransport(twin, OWNER, cancel=self._cancel, on_lost=lambda: self._interrupt("preempt"))
        self.motion = Motion(self.link, self.arm.lo, self.arm.hi, on_tick=self._on_tick)
        #: The last guarded move's result (diagnostics and tests).
        self.last_result: MoveResult | None = None

        self._last_state: dict | None = None
        self._target_shown_until: float | None = None

        # The network thread has its own kinematics (own MjData) for the TCP sphere.
        self._kin_net = Kinematics()

        # The panel's speed cap from before our first move, restored afterwards.
        self._prev_cap: float | None = None
        self._capped = False

        self._current: Target | None = None
        # The command the target was validated from [rad]; `_recheck` compares against it.
        self._validated_from: np.ndarray | None = None

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
            self._send({"type": "overlay", "objects": [overlay_object(self.volume), self._tcp_object()]})
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

    def _tcp_now(self, kin: Kinematics | None = None) -> list[float]:
        kin = kin or self._kin_net
        return _rounded(kin.fk(to_q(self.twin.joints()))[:3, 3])

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
        self._queue.put(t)
        # A new target interrupts the move; the worker continues from the current command.
        if self._phase == "moving":
            self._interrupt("retarget")

    def _interrupt(self, cause: Cause) -> None:
        with self._lock:
            if not self._cancel.is_set():
                self._cause = cause
                self._cancel.set()
            cause = self._cause
        self.motion.cancel(cause)

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
            msg["tcp"] = self._tcp_now(self.arm.kin)
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
            self._gen = None
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
            self._gen = self._take()
            self._recheck(legs)
        except _Abort as exc:
            # Nothing has moved yet. Note: unlike the other paths, this one doesn't fade the target sphere.
            self._report(t, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        # The supervisor must never lag galaxeo's slew cap (1.5x speed + 5), or the lag guard trips.
        self._cap_speed(1.5 * self.speed + 5.0)
        self._report(t, "moving")
        try:
            result = self._drive(t, legs)
        finally:
            self._gen = None
            self.link.grip(None)
        return self._after_move(t, result)

    def _held_off(self, t: Target) -> bool:
        """In `real`, a hand target waits `hold_off_s` before moving. True = interrupted meanwhile."""
        if self.mode == "real" and t.source == "hand" and self.hold_off_s > 0:
            return self._cancel.wait(self.hold_off_s)
        return False

    def _drive(self, t: Target, legs: list[_Leg]) -> MoveResult:
        """Drives the legs in order; stops at the first one that doesn't arrive."""
        result = MoveResult("reached")
        for i, leg in enumerate(legs):
            if i > 0:
                self._report(t, "moving")
            if leg.close_gripper:
                self.link.grip(self.grip_closed)
                if self._cancel.wait(self.grip_wait_s):
                    return MoveResult("hold", self._cancel_cause or "stop")
            result = self._move(leg.start, leg.goal, leg.speed, profile=leg.profile)
            if not result:
                return result
        return result

    def _move(self, start: np.ndarray, goal: np.ndarray, speed: float, *, profile: Profile = "normal",
              lag_deg: float | None = None, arrive_lag_s: float | None = None) -> MoveResult:
        """One leg as consecutive guarded moves along the checked straight path (`pieces`)."""
        result = MoveResult("reached")
        for piece in pieces(start, goal):
            if self._cancel.is_set():
                return MoveResult("hold", self._cancel_cause or "stop")
            self.motion.begin()
            if self._cancel.is_set():                   # an interrupt that `begin` just cleared
                self.motion.cancel(self._cancel_cause or "stop")
            self._phase = "moving"
            result = self.motion.move(piece, speed, profile=profile, lag_deg=lag_deg, arrive_lag_s=arrive_lag_s)
            self.last_result = result
            if not result:
                return result
        return result

    def _on_tick(self, now: float, cmd: np.ndarray, reading: Any, phase: str) -> None:
        """Every tick of a guarded move, on the worker thread."""
        gen = self._gen
        if gen is not None:
            cause = self._check(gen)
            if cause:
                self._interrupt(cause)
        if phase == "backoff" and self._phase != "backoff":
            self._phase = "backoff"
            self._cap_speed(BACKOFF_CAP_DEG_S)

    def _after_move(self, t: Target, result: MoveResult) -> Target | None:
        if result.state == "reached":
            self._finish(t, "reached")                  # stays engaged, holding the target
            return None
        if result.impact is not None:
            hit = result.impact
            logger.warning("Uderzenie: %s %s (%.1f) - %s", hit.kind, hit.joint, hit.value, result.state)
            self._report(t, "backoff" if result.state == "backoff" else "hold", result.reason, detail=result.detail)
            self._reject_queued()
            self._let_go()
            return None
        if self._cancel.is_set():
            return self._after_cancel(t)
        # Stale or frozen feedback, not arrived, a refused piece: the arm holds where it is.
        self._finish(t, "hold", result.reason, result.detail)
        return None

    def _after_cancel(self, t: Target) -> Target | None:
        """After an interrupted move: a new target takes over, anything else holds."""
        cause = self._cancel_cause
        if cause == "retarget":
            try:
                newer = self._latest(self._queue.get_nowait())
            except queue.Empty:
                newer = None
            self._report(t, "hold", "retargeted")
            return newer

        if cause in ("relay", "mode", "not connected"):
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
        self._validated_from = prev = self.link.command_q()

        legs: list[_Leg] = []
        for wp in self._waypoints(t):
            goal = self._solve(wp, seed=prev)
            self._check_leg(prev, goal, wp.what)
            legs.append(_Leg(prev, goal, self.speed * wp.speed_factor, wp.profile, wp.what))
            prev = goal

        if t.source == "object" and self.grasp:
            # After the grasp, back up to the hover pose along the same (already checked) path.
            legs.append(_Leg(prev, legs[0].goal, self.speed * 0.5, "descend", "w gore", close_gripper=True))
        return legs

    def _waypoints(self, t: Target) -> list[_Waypoint]:
        if t.source == "hand":
            return [_Waypoint(t.pos, "dlon")]

        rotation = down_rotation(t.pos)
        hover = _Waypoint(self._hover_point(t.pos), "nad obiekt", rotation, check_down=True)
        if not self.grasp:
            return [hover]

        touch = _Waypoint(t.pos + np.array([0.0, 0.0, self.touch_dz]), "w dol", rotation,
                          check_down=True, speed_factor=0.5, profile="descend")
        return [hover, touch]

    def _solve(self, wp: _Waypoint, seed: np.ndarray) -> np.ndarray:
        kin = self.arm.kin
        sol = kin.ik(wp.pos, wp.rotation, seed=seed, tol=self.ik_tol)
        if not sol.ok:
            raise _Rejected("ik", f"IK {wp.what}: blad {sol.pos_err * 1000:.1f} mm")

        if wp.check_down:
            approach = kin.fk(sol.q)[:3, 0]             # tool x is the approach axis
            tilt = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
            if tilt > self.approach_tol:
                raise _Rejected("ik", f"IK {wp.what}: os podejscia {math.degrees(tilt):.0f} st. od pionu")
        return sol.q

    def _check_leg(self, start: np.ndarray, goal: np.ndarray, what: str) -> None:
        jump = _deg(goal - start)
        if jump > self.max_jump:
            raise _Rejected("jump", f"skok {what} {jump:.0f} st. > {self.max_jump:.0f}")
        collision = self.arm.collision.path_clear(start, goal)
        if collision is not None:
            raise _Rejected("collision", f"kolizja {what}: {collision}")

    def _recheck(self, legs: list[_Leg]) -> None:
        """Someone moved the arm during the hold-off (panel sliders, a jog): the plan was validated
        from a pose the arm no longer has, so hold "moved" and let the wearer pinch again."""
        if not legs or self._validated_from is None:
            return
        moved = _deg(self.link.command_q() - self._validated_from)
        if moved < self.unchanged_deg:
            return
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

    def _check(self, gen: int) -> Cause | None:
        """Why the arm is no longer ours, or None."""
        tw = self.twin
        if self._estopped() or tw.preempt_gen != gen:
            return "preempt"
        if not tw.connected:
            return "not connected"
        return None

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

    def _estopped(self) -> bool:
        from ..control.safety import SafetyState

        return self.twin.safety_state == SafetyState.ESTOP

    def _center_q(self) -> np.ndarray | None:
        kin = self.arm.kin
        sol = kin.ik(self.volume.center, seed=kin.home, tol=self.ik_tol)
        return sol.q if sol.ok else None

    def center_pose(self) -> dict[str, float] | None:
        """Arm joints [deg] with the TCP above the centre of the reach box (IK from home); None if IK fails."""
        q = self._center_q()
        return None if q is None else to_joints(q)

    def start_center(self, speed_deg_s: float = 10.0) -> str | None:
        """Moves above the centre of the reach box in stages (`spec.home_stages`), each piece a guarded
        move. From home, every point in the box is a jump over 45 deg, so the first target would be refused.

        Call BEFORE `start()`, from the thread that then starts the driver. An impact backs off and
        stops there. Returns None on arrival, otherwise the reason.
        """
        tw = self.twin
        if not tw.connected:
            return "not connected"
        if self._estopped():
            return "stop"
        if not (tw.status.simulated or (self.mode == "real" and self.real_ok)):
            return "not armed"
        goal = self._center_q()
        if goal is None:
            return "ik"

        start = self.link.command_q()
        steps, collision = self._stage_plan(start, goal)
        if collision is not None:
            logger.warning("Start nad srodek pudelka: %s", collision)
            return "collision"

        with self._lock:
            self._cancel.clear()
            self._cause = ""
        try:
            self._gen = self._take()
        except _Abort as exc:
            return exc.reason

        self._cap_speed(1.5 * speed_deg_s + 5.0)
        try:
            prev = start
            for step in steps:
                result = self._move(prev, step, speed_deg_s, lag_deg=self.center_lag_deg,
                                    arrive_lag_s=self.center_arrive_lag_s)
                prev = step
                if not result:
                    if result.impact is not None:
                        logger.warning("Start nad srodek pudelka: %s - %s", result.reason, result.state)
                    elif self._cancel.is_set():
                        return self._cancel_cause or "preempted"
                    return result.reason
            return None
        finally:
            self._gen = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _stage_plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[list[np.ndarray], str | None]:
        """Intermediate poses as in staged homing: joint groups one after another (also in reverse),
        then the whole goal. The first plan with clear paths wins; with no stages, a straight move.
        Returns (poses, None) or ([], the first collision)."""
        index = {j: k for k, j in enumerate(JOINTS)}
        stages = [tuple(index[j] for j in stage if j in index) for stage in self.spec.home_stages]
        stages = [stage for stage in stages if stage]
        orders = [stages, stages[::-1]] if stages else []
        every = tuple(range(len(JOINTS)))

        first_collision: str | None = None
        for order in [*orders, []]:
            poses, collision = self._walk_stages(start, goal, [*order, every])
            if collision is None:
                return poses, None
            first_collision = first_collision or collision
        return [], first_collision

    def _walk_stages(self, start: np.ndarray, goal: np.ndarray,
                     stages: list[tuple[int, ...]]) -> tuple[list[np.ndarray], str | None]:
        pose = start.copy()
        poses: list[np.ndarray] = []
        for stage in stages:
            nxt = pose.copy()
            nxt[list(stage)] = goal[list(stage)]
            if _deg(nxt - pose) < 1e-3:
                continue
            collision = self.arm.collision.path_clear(pose, nxt)
            if collision is not None:
                return poses, collision
            poses.append(nxt)
            pose = nxt
        return poses, None

    def _log_efforts(self, what: str) -> None:
        """Logs arm joint efforts: the A1X guard thresholds are guesses, and this gives real numbers."""
        try:
            reading = self.link.read()
        except Exception:
            return
        if reading is not None:
            logger.info("Wysilek stawow (%s): %s", what,
                        {j: round(float(v), 2) for j, v in zip(JOINTS, reading.effort, strict=True)})
