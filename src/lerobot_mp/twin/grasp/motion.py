"""Taking, moving and holding the arm for the grasp commands, without a policy.

    arm = ArmSession(twin, kin, rig)
    arm.check()                                # refusal reason or None; listens for another CAN sender
    arm.take()                                 # claim as OWNER, engage, speed check, collision checker
    reason, sag = arm.preposition(hover)       # straight up when low, staged joint plan, open the jaw
    q, step, why = arm.accept(p, R, q_seed)    # one IK step plus one collision question
    arm.command(target)                        # all seven joints, as OWNER; raises Ended
    arm.hold(ending)                           # ONE hold call, with an explicit jaw value
    arm.close()

Shared by the grasp driver and the bench jog. One claim per session under one owner name;
the session releases only in `close()`. The tick listener fills the joint history and feeds
the impact guard on the twin's loop thread; everything else runs on the caller's thread.

Pre-positioning is isolated in `ArmSession.preposition` so it can move to the arm library's
guarded move later. The collision question is the twin's existing `CollisionChecker`.

Every time, age, ramp, wait and timeout goes through the injected `clock` and `sleep`
(`sleep` None: wait on the cancel event), so tests on a stepped twin run on the twin's clock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...control.safety import SafetyState
from ...robot.base import GripperState
from .. import robots
from .. import scene as sc
from ..collision import CollisionChecker
from ..impact import Impact, ImpactGuard
from ..kinematics import IKStep, RobotKinematics
from ..runtime import TickSample, Twin, smoothstep
from ..workspace import Workspace, default_path
from .contract import (
    ARM_SPEED_DEG_S,
    ARRIVE_DEG,
    ARRIVE_M,
    ARRIVE_STILL_DEG_S,
    ARRIVE_TIMEOUT_S,
    EFFORT_REST_FRAC,
    FOREIGN_LISTEN_S,
    HOVER_J5_MEASURED_MAX_DEG,
    IK_MAX_JUMP_DEG,
    JOINT_AGE_MAX_S,
    OBJECT_HEIGHT_DEFAULT,
    OWNER,
    PREPOSITION_SPEED_DEG_S,
    RETREAT_BELOW,
    RETREAT_CLEAR,
    RETREAT_NEAR_XY,
    RETREAT_STEP,
    RETREAT_TILT_MAX_DEG,
    SAG_MAX,
)
from .hover import Hover, stage_plan
from .rig import Rig

logger = logging.getLogger(__name__)

KEEP_GRIP = ("lifted", "tick cap", "time cap", "success")     # endings after which the jaw keeps its command
MEASURED_HOLD = ("lag", "table")                              # plus every ending starting with "impact"

RAMP_WRITE_S = 0.02                    # one target per ramp step
POLL_S = 0.05                          # arrival and sag sampling
GRIP_STAGE = 99.0                      # the jaw's first command of a session: differs from a reported 100
GRIP_OPEN = 100.0
GRIP_FIRST_S = 0.1
GRIP_WAIT_S = 2.0
SAG_AVERAGE_S = 0.3
SIM_EFFORT_FRAC = 0.5                  # of the arm actuators' forcerange
CLOSE_WAIT_S = 1.0                     # close(): how long a running move gets to end and hold by itself


class Ended(Exception):
    """Raised inside a move or a tick to end it with a reason code."""

    def __init__(self, ending: str, detail: str = "") -> None:
        super().__init__(f"{ending}: {detail}" if detail else ending)
        self.ending = ending
        self.detail = detail


@dataclass(frozen=True)
class JointSample:
    t: float                          # measured_t of the twin loop (the twin's clock)
    joints: dict[str, float]          # all joints, app units


class JointHistory:
    """Fresh joint samples of the last `keep_s` seconds, filled on the twin's loop thread."""

    def __init__(self, keep_s: float = 2.0) -> None:
        self.keep_s = float(keep_s)
        self._samples: deque[JointSample] = deque()
        self._lock = threading.Lock()

    def add(self, sample: TickSample) -> None:
        if not sample.fresh or not sample.measured:
            return
        t = float(sample.measured_t)
        with self._lock:
            if self._samples and t <= self._samples[-1].t:
                return
            self._samples.append(JointSample(t, {k: float(v) for k, v in sample.measured.items()}))
            while self._samples and self._samples[0].t < t - self.keep_s:
                self._samples.popleft()

    def newest(self) -> JointSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def nearest(self, t: float) -> JointSample | None:
        with self._lock:
            if not self._samples:
                return None
            return min(self._samples, key=lambda s: abs(s.t - t))

    def speed_deg_s(self) -> dict[str, float]:
        with self._lock:
            if len(self._samples) < 2:
                return {}
            a, b = self._samples[-2], self._samples[-1]
        dt = b.t - a.t
        if dt <= 1e-9:
            return {}
        return {k: (v - a.joints[k]) / dt for k, v in b.joints.items() if k in a.joints}

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


def load_workspace(path: str | Path | None) -> Workspace | None:
    """The A1X workspace of a grasp command; None when the file describes another robot.

    Without a path and without a file at the default location: an in-memory A1X workspace,
    never saved (`Workspace.load()` would give an SO-101).
    """
    if path is None and not default_path().is_file():
        return Workspace(robot="a1x")
    ws = Workspace.load(path)
    return ws if ws.robot == "a1x" else None


def enable_refusal(ws: Workspace, backend: str, port: str | None) -> str | None:
    """"enable on connect" when connecting would send the A1X enable sequence; ask BEFORE connecting."""
    if backend == "sim":
        return None
    cfg = robots.robot_config(ws.spec(), overrides={"robot": {"backend": backend, "port": port}})
    return "enable on connect" if cfg.a1x.enable_on_connect else None


def jaw_hold_value(gs: GripperState | None, *, keep_grip: bool) -> float | None:
    """The gripper value to hold after an ending; None leaves it to the twin's own rule (measured jaw)."""
    if gs is None or gs.measured:
        return None
    if keep_grip:
        return gs.commanded
    return gs.streamed if gs.streamed is not None else gs.commanded


def _estopped(twin: Twin) -> bool:
    return twin.safety_state == SafetyState.ESTOP


def stop_jaw(twin: Twin) -> bool:
    """Re-hold a jaw in mid-stroke at the opening the stream has reached, after the twin took the arm.

    The twin's STOP and preempt hold the A1X's "measured" gripper, which is its last command,
    so a closing jaw would finish its stroke. Works under STOP and needs no owner.
    """
    b = twin.backend
    if b is None or not (twin.owner is None or _estopped(twin)):
        return False
    gs = b.gripper_state()
    if gs is None or gs.measured or gs.streamed is None or gs.commanded is None:
        return False
    if abs(gs.streamed - gs.commanded) <= 1.0:
        return False
    logger.warning("jaw stopped at the streamed opening %.1f (command was %.1f)", gs.streamed, gs.commanded)
    twin.hold_measured(gripper=gs.streamed)
    return True


def stop_now(twin: Twin, reason: str) -> None:
    """STOP, then stop the jaw where it is."""
    twin.estop(reason)
    stop_jaw(twin)


class ArmSession:
    """One owner's hold on the arm: claim, clutch, tick listener, joint history, impact guard, cancel."""

    def __init__(self, twin: Twin, kin: RobotKinematics, rig: Rig, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.twin = twin
        self.kin = kin
        self.rig = rig
        self.spec = twin.spec
        self.arm: tuple[str, ...] = tuple(self.spec.arm_joints)
        self.grip: str | None = self.spec.gripper
        self.clock = clock
        self._sleep = sleep
        self.history = JointHistory()
        self.cancel = threading.Event()
        self.gen: int | None = None
        self.checker: CollisionChecker | None = None
        self.no_effort_rule: list[str] = []
        self.moved = False
        #: The last impact the guard reported, for the ending's name.
        self.impact: Impact | None = None
        #: Detail of the last refusal of `check` or `preposition`, for the operator.
        self.detail = ""
        #: (clock time, arm efforts) logged by `log_efforts`, so `guard.effort_abs` can be set from data.
        self.efforts_seen: list[tuple[float, dict[str, float]]] = []
        self._cause = ""
        self._lock = threading.Lock()
        self._listener: int | None = None
        self._guard: ImpactGuard | None = None
        self._took = False
        self._ever_took = False                # not cleared on release: a late hold still stops the jaw
        self._idle = threading.Event()         # no move or episode runs
        self._idle.set()
        self._mover: int | None = None         # the thread running it
        self._prev_cap: float | None = None
        self._last_grip: float | None = None
        self._effort_ids: tuple[int, np.ndarray] | None = None

    # ------------------------------------------------------------ clock and cancel
    def wait(self, dt: float) -> bool:
        """Wait `dt` seconds through the injected sleep; True when the cancel event is set."""
        if dt > 0.0:
            if self._sleep is not None:
                self._sleep(dt)
            else:
                self.cancel.wait(dt)
        return self.cancel.is_set()

    def interrupt(self, cause: str) -> None:
        """Ends the running move or episode with `cause`; the first cause wins. Any thread."""
        with self._lock:
            if not self.cancel.is_set():
                self._cause = cause
                self.cancel.set()

    def cause(self) -> str:
        return self._cause

    def begin(self) -> None:
        """A new move or episode: a cancel left over from the previous one no longer applies."""
        with self._lock:
            self.cancel.clear()
            self._cause = ""
        self.impact = None
        self.detail = ""

    def _estopped(self) -> bool:
        return _estopped(self.twin)

    def lost(self) -> str:
        """Why the arm is no longer ours."""
        if not self.twin.connected:
            return "not connected"
        return "stop" if self._estopped() else "preempted"

    def ending_of_cause(self) -> str:
        cause = self._cause
        if cause == "preempt":
            return self.lost()
        if cause == "impact":
            hit = self.impact
            return f"impact {hit.kind} {hit.joint}" if hit is not None else "impact"
        return cause or "stop"

    def check_arm(self) -> None:
        """Raises Ended when the move or episode must end: cancel, lost arm, STOP, disconnect."""
        if self.cancel.is_set():
            raise Ended(self.ending_of_cause())
        tw = self.twin
        if self.gen is None or tw.preempt_gen != self.gen:
            raise Ended(self.lost())
        if self._estopped():
            raise Ended("stop")
        if not tw.connected:
            raise Ended("not connected")

    # ------------------------------------------------------------------ checks
    def check_basic(self) -> str | None:
        tw = self.twin
        if not tw.connected:
            return "not connected"
        if self._estopped():
            return "stop"
        if tw.owner not in (None, OWNER):
            self.detail = f"ramie trzyma {tw.owner}"
            return "busy"
        return None

    def check_other_sender(self) -> str | None:
        """On a real backend: listen `FOREIGN_LISTEN_S` for command frames of another sender."""
        b = self.twin.backend
        if b is None or b.info.simulated:
            return None
        n0 = b.foreign_frames()
        t_end = self.clock() + FOREIGN_LISTEN_S
        while True:
            left = t_end - self.clock()
            if left <= 0.0:
                break
            if self.wait(min(left, POLL_S)):
                return self.ending_of_cause()
        if b.foreign_frames() != n0:
            self.detail = "inny nadawca na laczu CAN"
            return "busy"
        return None

    def check(self) -> str | None:
        """Refusal reason or None; moves nothing. On a real backend it listens 0.3 s first."""
        self.detail = ""
        return self.check_basic() or self.check_other_sender()

    # ------------------------------------------------------------------ taking
    def _build_checker(self) -> None:
        if self.checker is not None:
            return
        tw = self.twin
        extras = {}
        b = tw.backend
        # The simulated arm stands on the table of the scene's extras (the sim evaluation's table).
        if b is not None and b.info.simulated and "table" in tw.extras:
            extras["table"] = tw.extras["table"]
        self.checker = CollisionChecker(sc.build(tw.workspace.scene_config(**extras)))

    def listen(self) -> None:
        """Attach the tick listener only (dry mode); sends nothing."""
        if self._listener is None:
            self._listener = self.twin.add_tick_listener(self._on_tick)

    def take(self) -> str | None:
        """Claim as OWNER and engage; the speed check runs before any motion. None on success."""
        tw = self.twin
        if self.gen is not None and self.gen == tw.preempt_gen and tw.owner == OWNER:
            return None                                    # one claim per session
        try:
            tw._wait_for_homing()
        except RuntimeError:
            return "busy"
        gen = tw.preempt_gen
        try:
            tw.claim(OWNER, preempt=lambda: self.interrupt("preempt"), gen=gen)
        except RuntimeError as exc:
            self.detail = str(exc)
            return "stop" if self._estopped() else "busy"
        if not tw.set_engaged(True, owner=OWNER):
            tw.release(OWNER)
            return "stop" if self._estopped() else "busy"
        self.gen = gen
        self.listen()
        if not self._took:
            self._prev_cap = getattr(tw, "_speed_cap", None)
            self._took = True
        slow = self.slow_joints()
        tw.set_speed(PREPOSITION_SPEED_DEG_S)
        if slow:
            self.detail = slow
            logger.warning("speed limited below %.0f deg/s: %s", ARM_SPEED_DEG_S, self.detail)
            self._let_go()
            return "speed limited"
        self._ever_took = True
        self._build_checker()
        return None

    def slow_joints(self) -> str | None:
        """The arm joints the backend lets move slower than ARM_SPEED_DEG_S, for the operator; None when
        none. Leaves the twin's speed cap off."""
        tw = self.twin
        tw.set_speed(None)
        limits = tw.speed_limits()
        slow = {j: limits.get(j, 0.0) for j in self.arm if limits.get(j, 0.0) < ARM_SPEED_DEG_S - 1e-6}
        return ", ".join(f"{j} {v:.0f} st./s" for j, v in slow.items()) or None

    def _let_go(self) -> None:
        tw = self.twin
        if self._listener is not None:
            tw.remove_tick_listener(self._listener)
            self._listener = None
        if self._took:
            tw.set_speed(self._prev_cap)
            self._took = False
        if self.gen is not None:
            tw.release(OWNER)
        self.gen = None

    # ------------------------------------------------------------ tick listener, guard
    def _on_tick(self, sample: TickSample) -> None:
        """Twin loop thread: the joint history and the impact guard, nothing else."""
        self.history.add(sample)
        guard = self._guard
        if guard is None or self.cancel.is_set():
            return
        hit = guard.feed(sample.t, sample.sent, sample.measured, sample.fresh,
                         efforts=self._efforts(loop=True), velocities=self._velocities())
        if hit is not None:
            self.impact = hit
            logger.warning("impact %s %s (%.1f)", hit.kind, hit.joint, hit.value)
            self.interrupt("impact")

    def _sim_ids(self) -> np.ndarray:
        scene = self.twin.scene
        cached = self._effort_ids
        if cached is None or cached[0] != id(scene.model):
            ids = np.array([scene.model.actuator(sc.PREFIX + j).id for j in self.arm])
            self._effort_ids = cached = (id(scene.model), ids)
        return cached[1]

    def _effort_limits(self) -> dict[str, float] | None:
        """Per-joint effort limits; None = the backend cannot measure effort (lag rule only)."""
        tw = self.twin
        b = tw.backend
        if b is None:
            return None
        if b.info.simulated:
            with tw.lock:
                model = tw.scene.model
                ids = self._sim_ids()
                ranges = model.actuator_forcerange[ids].copy()
                limited = model.actuator_forcelimited[ids].copy()
            return {j: SIM_EFFORT_FRAC * float(max(abs(lo), abs(hi)))
                    for j, (lo, hi), on in zip(self.arm, ranges, limited, strict=True) if on and hi > lo}
        try:
            probe = b.efforts()
        except Exception:
            probe = None
        if probe is None:
            return None
        return dict.fromkeys(self.arm, float(self.rig.effort_abs()))

    def _efforts(self, *, loop: bool = False) -> dict[str, float] | None:
        """Arm efforts now. On the loop thread (`loop`) a sim read never waits for the twin's lock:
        that sample goes without efforts, which neither advances nor resets the effort rule."""
        tw = self.twin
        b = tw.backend
        if b is None:
            return None
        if not b.info.simulated:
            try:
                return b.efforts()
            except Exception:
                return None
        if not tw.lock.acquire(blocking=not loop):
            return None
        try:
            forces = tw.scene.data.actuator_force[self._sim_ids()]
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

    def log_efforts(self, what: str) -> dict[str, float] | None:
        efforts = self._efforts()
        if efforts:
            rounded = {k: round(float(v), 2) for k, v in efforts.items()}
            self.efforts_seen.append((self.clock(), rounded))
            logger.info("arm efforts (%s): %s", what, rounded)
        return efforts

    def guard(self, *, lag_deg: float, lag_s: float = 0.15) -> None:
        """Arm the impact guard for the next move; a joint loaded at rest gets the lag rule only."""
        limits = self._effort_limits()
        self.no_effort_rule = []
        b = self.twin.backend
        if limits is None and b is not None and not b.info.simulated:
            # One stale read at this moment would otherwise leave the whole move without the rule, unseen.
            logger.warning("no arm efforts from the backend: lag rule only for this move")
            self.no_effort_rule = list(self.arm)
        if limits:
            at_rest = self._efforts() or {}
            for j in list(limits):
                if j in at_rest and abs(at_rest[j]) > EFFORT_REST_FRAC * limits[j]:
                    logger.warning("%s effort at rest %.2f is above %.0f %% of its limit %.2f: lag rule only "
                                   "for this move", j, at_rest[j], 100 * EFFORT_REST_FRAC, limits[j])
                    del limits[j]
                    self.no_effort_rule.append(j)
        self.impact = None
        self._guard = ImpactGuard(self.arm, lag_deg=lag_deg, lag_s=lag_s, effort_abs=limits)

    def disarm(self) -> None:
        self._guard = None

    # ------------------------------------------------------------------ commands
    def command(self, target: Mapping[str, float]) -> None:
        """One target of all seven joints, as OWNER. Raises Ended when it cannot be sent."""
        missing = [j for j in self.spec.joints if j not in target]
        if missing:
            raise ValueError(f"partial target, missing {missing}")
        if not all(math.isfinite(float(v)) for v in target.values()):
            raise Ended("non-finite", str({k: float(v) for k, v in target.items()}))
        try:
            self.twin.set_target(target, owner=OWNER)
        except RuntimeError:
            raise Ended(self.lost()) from None
        except ValueError as exc:
            raise Ended("non-finite", str(exc)) from None
        self.moved = True
        if self.grip is not None:
            self._last_grip = float(target[self.grip])

    def jaw_command(self) -> float:
        """The jaw's current command: the backend's, else the last one this session sent, else open."""
        b = self.twin.backend
        gs = b.gripper_state() if b is not None else None
        if gs is not None and gs.commanded is not None:
            return float(gs.commanded)
        return self._last_opening()

    def _last_opening(self) -> float:
        return self._last_grip if self._last_grip is not None else GRIP_OPEN

    def widest_opening(self, extra: float | None = None) -> float:
        """The widest jaw opening that can be true now, `extra` included when given.

        On the A1X the command leads the jaw by up to 1.3 s: while a close is commanded the fingers
        are still open, and an open finger is what meets the arm's base or sits lowest over the table."""
        values = [self._last_opening()]
        if extra is not None:
            values.append(float(extra))
        b = self.twin.backend
        gs = b.gripper_state() if b is not None else None
        if gs is not None:
            values += [float(v) for v in (gs.commanded, gs.streamed, gs.position) if v is not None]
        return max(values)

    def accept(self, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray, *,
               collision: bool = True) -> tuple[np.ndarray | None, IKStep, str]:
        """One warm IK step from `q_seed` (rad) and, on the accepted joints, one collision question.

        Returns (joints in rad or None, the step, "" | "ik" | "jump" | "collision").
        """
        seed = {j: math.degrees(float(q)) for j, q in zip(self.arm, np.asarray(q_seed, float), strict=True)}
        if self.grip is not None:
            seed[self.grip] = self.widest_opening() if collision else self._last_opening()
        step = self.kin.ik_step(np.asarray(p, float), np.asarray(R, float), seed,
                                limits=self.twin.joint_limits())
        if not step.converged:
            return None, step, "ik"
        if step.jump_deg > IK_MAX_JUMP_DEG:
            return None, step, "jump"
        if collision:
            self._build_checker()
            assert self.checker is not None
            if self.checker.config_clear(step.joints) is not None:
                return None, step, "collision"
        return np.radians([step.joints[j] for j in self.arm]), step, ""

    def _joint_age(self) -> float:
        sample = self.history.newest()
        return math.inf if sample is None else self.clock() - sample.t

    def _measured(self) -> dict[str, float]:
        """The newest fresh joint sample, else the twin's last reading."""
        sample = self.history.newest()
        if sample is not None and self.clock() - sample.t <= JOINT_AGE_MAX_S:
            return dict(sample.joints)
        return {k: float(v) for k, v in self.twin.status.measured.items()}

    def _ramp(self, start: Mapping[str, float], goal: Mapping[str, float], grip: float,
              speed_deg_s: float) -> None:
        """Smoothstep from `start` to `goal` (arm joints), one target every 0.02 s.

        The ramp's clock pauses while the twin's loop does."""
        largest = max((abs(float(goal[j]) - float(start[j])) for j in self.arm), default=0.0)
        duration = max(0.5, largest / max(speed_deg_s, 1e-3))
        frozen = getattr(self.twin, "_frozen", None)
        elapsed, t_prev = 0.0, self.clock()
        while True:
            self.check_arm()
            now = self.clock()
            if frozen is None or not frozen():
                elapsed += now - t_prev
            t_prev = now
            s = smoothstep(elapsed / duration)
            target = {j: float(start[j]) + s * (float(goal[j]) - float(start[j])) for j in self.arm}
            if self.grip is not None:
                target[self.grip] = grip
            self.command(target)
            if s >= 1.0:
                return
            self.wait(RAMP_WRITE_S)

    @contextmanager
    def busy(self) -> Iterator[None]:
        """A move or an episode runs on this thread; `close()` from another thread lets it end and hold."""
        self._mover = threading.get_ident()
        self._idle.clear()
        try:
            yield
        finally:
            self._mover = None
            self._idle.set()

    # ------------------------------------------------------------------ pre-positioning
    def preposition(self, hover: Hover,
                    speed_deg_s: float = PREPOSITION_SPEED_DEG_S) -> tuple[str | None, float]:
        """Move to the hover and open the jaw there. Returns (reason, sag); reason None on success.

        The only pre-positioning code: when the arm library gets a guarded move, this is replaced.
        """
        if self.gen is None:
            raise RuntimeError("preposition() before take()")
        with self.busy():
            self.begin()
            try:
                self.check_arm()
                sag = self._preposition(hover, speed_deg_s)
            except Ended as e:
                self.detail = e.detail
                logger.warning("pre-positioning ended: %s%s", e.ending, f" ({e.detail})" if e.detail else "")
                self.hold(e.ending)
                return e.ending, 0.0
            except Exception as exc:
                logger.exception("pre-positioning failed")
                self.detail = f"{type(exc).__name__}: {exc}"[:200]
                self.hold("error")
                return "error", 0.0
            except BaseException:
                self.hold("stop")
                raise
            self.disarm()
            self.moved = True
            return None, sag

    def _preposition(self, hover: Hover, speed_deg_s: float) -> float:
        tw = self.twin
        # `moved` becomes true with this move's first target; the jaw rule of the stages asks
        # about earlier ones.
        moved = self.moved
        tw.set_speed(speed_deg_s)
        self.guard(lag_deg=4.0, lag_s=0.15)
        measured = self._measured()
        start = {j: float(measured[j]) for j in self.arm}
        keep = self.jaw_command()

        start = self._retreat(hover, start, keep, speed_deg_s, moved)

        self._build_checker()
        assert self.checker is not None
        goal = {j: float(hover.joints[j]) for j in self.arm}
        poses, collision = stage_plan(self.spec, start, goal, self.checker.path_clear)
        if collision is not None:
            raise Ended("collision", collision)
        grip = keep if moved else GRIP_STAGE
        pose = start
        for nxt in poses:
            self._ramp(pose, nxt, grip, speed_deg_s)
            pose = nxt
        if not poses:
            self._ramp(pose, goal, grip, speed_deg_s)

        self._open_jaw(goal)
        self._arrive(hover)
        if not tw.status.simulated:
            j5 = self._measured()[self.arm[4]]
            if abs(j5) > HOVER_J5_MEASURED_MAX_DEG:
                raise Ended("j5 offset", f"{self.arm[4]} {j5:.2f} deg at the hover")
        return self._sag(hover)

    def _retreat(self, hover: Hover, start: dict[str, float], keep: float,
                 speed_deg_s: float, moved: bool) -> dict[str, float]:
        """Straight up to the hover height when the tool is low, jaw unchanged.

        Returns where the arm then stands."""
        T = self.kin.tcp(start)
        approach = T[:3, :3] @ np.asarray(self.spec.tcp_approach, float)
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -float(approach[2])))))
        p = T[:3, 3]
        near = math.hypot(*(p[:2] - np.asarray(hover.p[:2], float))) < RETREAT_NEAR_XY
        if not (tilt <= RETREAT_TILT_MAX_DEG and p[2] < hover.p[2] - RETREAT_BELOW and (moved or near)):
            return start
        logger.info("straight up from TCP z %.3f m to %.3f m before pre-positioning", p[2], hover.p[2])
        # The tool keeps its measured orientation: squaring a few degrees of tilt in the first 5 mm
        # step needs more than IK_MAX_JUMP_DEG from one joint in 10 to 25 % of low poses, and would
        # swing the fingertips sideways past the cube. The staged plan squares it at the hover height.
        R = T[:3, :3].copy()
        q = np.radians([start[j] for j in self.arm])
        z, pose = float(p[2]), dict(start)
        while z < hover.p[2] - 1e-9:
            z_ok, z = z, min(z + RETREAT_STEP, float(hover.p[2]))
            q_new, step, why = self.accept(np.array([p[0], p[1], z]), R, q, collision=False)
            if q_new is None:
                detail = (f"straight-up step to z {z:.3f} m refused ({why}, {step.pos_err * 1000:.2f} mm, "
                          f"{step.rot_err * 1000:.2f} mrad, {step.jump_deg:.1f} deg)")
                # At long reach a vertical line leaves the warm solver's branch (joint 4 at its limit)
                # before the hover height. Well above the object the collision-checked staged plan can
                # take over; ending here instead left the arm there and refused every later episode.
                if z_ok >= self._retreat_clear():
                    logger.warning("%s; the staged plan goes on from TCP z %.3f m", detail, z_ok)
                    return pose
                raise Ended("ik", detail)
            nxt = {j: math.degrees(float(v)) for j, v in zip(self.arm, q_new, strict=True)}
            self._ramp(pose, nxt, keep, speed_deg_s)
            q, pose = q_new, nxt
        return pose

    def _retreat_clear(self) -> float:
        touch, height = self.rig.touch_tcp_z(), self.rig.object_height()
        if touch is None:
            return math.inf
        return touch + (OBJECT_HEIGHT_DEFAULT if height is None else height) + RETREAT_CLEAR

    def _open_jaw(self, goal: Mapping[str, float]) -> None:
        """At the hover, never earlier: 99 first when the last command was not 99, so the A1X's
        first-command gate lets it through, then 100; wait for the stream."""
        if self.grip is None:
            return
        target = {j: float(goal[j]) for j in self.arm}
        last = self._last_grip if self._last_grip is not None else self.jaw_command()
        if last != GRIP_STAGE:
            self.command({**target, self.grip: GRIP_STAGE})
            t_end = self.clock() + GRIP_FIRST_S
            while self.clock() < t_end:
                self.check_arm()
                self.wait(min(RAMP_WRITE_S, t_end - self.clock()))
        self.command({**target, self.grip: GRIP_OPEN})
        b = self.twin.backend
        t_end = self.clock() + GRIP_WAIT_S
        while self.clock() < t_end:
            self.check_arm()
            gs = b.gripper_state() if b is not None else None
            if gs is None or (gs.streamed is not None and gs.streamed >= GRIP_STAGE):
                return
            self.wait(RAMP_WRITE_S)
        logger.warning("jaw stream did not reach %.0f within %.1f s", GRIP_STAGE, GRIP_WAIT_S)

    def _misses(self, hover: Hover) -> list[str]:
        age = self._joint_age()
        if age > JOINT_AGE_MAX_S:
            return [f"joint feedback {age:.2f} s old"]
        m = self._measured()
        misses = [f"{j} {m[j] - hover.joints[j]:+.1f} deg" for j in self.arm
                  if abs(m[j] - hover.joints[j]) > ARRIVE_DEG]
        speed = self.history.speed_deg_s()
        misses += [f"{j} moving {speed[j]:.0f} deg/s" for j in self.arm
                   if abs(speed.get(j, 0.0)) >= ARRIVE_STILL_DEG_S]
        p = self.kin.tcp({j: m[j] for j in self.arm})[:3, 3]
        d = p - np.asarray(hover.p, float)
        if abs(d[0]) > ARRIVE_M or abs(d[1]) > ARRIVE_M:
            misses.append(f"TCP xy off by {d[0] * 1000:+.1f} {d[1] * 1000:+.1f} mm")
        if not (-ARRIVE_M - SAG_MAX <= d[2] <= ARRIVE_M):
            misses.append(f"TCP z off by {d[2] * 1000:+.1f} mm")
        return misses

    def _arrive(self, hover: Hover) -> None:
        t_end = self.clock() + ARRIVE_TIMEOUT_S
        while True:
            self.check_arm()
            misses = self._misses(hover)
            if not misses:
                return
            if self.clock() >= t_end:
                raise Ended("not arrived", "; ".join(misses))
            self.wait(POLL_S)

    def _sag(self, hover: Hover) -> float:
        values: list[float] = []
        t_end = self.clock() + SAG_AVERAGE_S
        while True:
            self.check_arm()
            m = self._measured()
            z = float(self.kin.tcp({j: m[j] for j in self.arm})[2, 3])
            values.append(min(max(float(hover.p[2]) - z, 0.0), SAG_MAX))
            if self.clock() >= t_end:
                break
            self.wait(POLL_S)
        return float(np.mean(values))

    # ------------------------------------------------------------------ holding
    def hold(self, ending: str) -> None:
        """Every ending holds in ONE call, with the jaw value given explicitly."""
        tw = self.twin
        self.disarm()
        b = tw.backend
        gs = b.gripper_state() if b is not None else None
        try:
            if self._took:
                # Whatever follows an ending moves at 20 deg/s, also after the twin took the arm.
                tw.set_speed(PREPOSITION_SPEED_DEG_S)
            if tw.owner == OWNER:
                grip = jaw_hold_value(gs, keep_grip=ending in KEEP_GRIP)
                if ending.startswith("impact") or ending in MEASURED_HOLD:
                    # The plain reading: a joint pressed against something stops pushing.
                    tw.hold_measured(dict(tw.status.measured) or None, gripper=grip)
                else:
                    # The twin's braking pose: a joint in motion does not reverse.
                    tw.hold_measured(gripper=grip)
            elif self._ever_took:
                # Only a session that had taken the arm, also after `close()` released it under a
                # running move: dry mode and a refused take write nothing.
                stop_jaw(tw)
        except Exception:
            logger.exception("holding after %r failed; holding the twin's own pose", ending)
            try:
                tw.hold_measured()
            except Exception:
                logger.exception("holding the twin's own pose failed too")

    def close(self) -> None:
        """End a running move or episode with a hold, remove the tick listener, restore the speed cap,
        release the arm."""
        if not self._idle.is_set():
            self.interrupt("stop")
            # The move's own thread holds at its next check; holding here first could let one more
            # target land between this hold and the release.
            own = self._mover == threading.get_ident()
            if own or not self._idle.wait(CLOSE_WAIT_S):
                self.hold("stop")
        self.disarm()
        self._let_go()
