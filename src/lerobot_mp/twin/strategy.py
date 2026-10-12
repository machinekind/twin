"""The control-strategy contract: one way for any strategy to drive the arm, in simulation and on hardware.

A strategy (an RL policy, a modular planner, a VLA model) turns what the twin sees into joint
targets. It never touches the backend: `StrategyRunner` claims the arm (`Twin.claim`), builds an
`Observation` every tick, calls the strategy, and sends its target through `Twin.set_target`, so
every command passes the safety supervisor. Simulation and hardware run the same code path.

A strategy from another package needs nothing from the twin: it follows the `Strategy` protocol
and reads the observation's attributes. It returns an `Action`, any object with the same
attributes, or a plain mapping of joint targets.

    runner = StrategyRunner(twin, Waypoints([pose_a, pose_b]), goal=panel_goal, objects=cube_poses)
    runner.start()            # own thread at `strategy.hz`; tests: `start(threaded=False)` + `step_once`
    ...
    runner.stop()             # the arm holds its last target

What the runner guards, for every strategy:

* targets: only the robot's joints, only finite numbers - anything else stops the run with the
  arm held where it measures;
* the arm taken away (Home, STOP, reconnect, another owner) stops the run;
* a disconnected arm stops the run;
* `time_limit_s` (the strategy's, or the runner's) ends the run.

A joint that cannot follow its command (collision, blocked or limp servo) is the twin's own
tracking guard, for every owner: it triggers STOP, which takes the arm from the runner.
What counts as success and how far a goal may be is the task's business, inside the strategy.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

#: Default owner name of the arm while a strategy runs (`Twin.claim`).
OWNER = "strategia"


@dataclass(frozen=True)
class ObjectPose:
    """Where an object is, in the robot's base frame."""

    #: Position [m], shape (3,).
    position: np.ndarray
    #: Rotation matrix, shape (3, 3).
    rotation: np.ndarray
    #: Where the pose comes from, e.g. "kamery", "w dloni", "ostatnie widziane", "symulacja".
    source: str = ""
    #: Seconds since the pose was measured; 0 = this tick.
    age_s: float = 0.0


@dataclass(frozen=True)
class RunInfo:
    """What a strategy learns once, when the run starts."""

    robot: str
    #: Joints in app order: degrees, the gripper 0..100.
    joints: tuple[str, ...]
    gripper: str | None
    #: The limits that hold on this arm now [app units]: the description's, narrowed to the hardware's.
    limits: Mapping[str, tuple[float, float]]
    simulated: bool


@dataclass(frozen=True)
class Observation:
    """What the strategy sees in one tick. Units: degrees, gripper 0..100, metres, base frame."""

    #: Seconds since the run started, and the tick number (0 = first).
    t: float
    step: int
    #: Measured joint positions.
    joints: Mapping[str, float]
    #: The supervisor's latest command: where the servos are being told to go (it trails the target).
    command: Mapping[str, float]
    #: The goal set in the panel, shape (3,); None = no goal.
    goal: np.ndarray | None = None
    #: Objects the perception knows about, by name (e.g. "cube").
    objects: Mapping[str, ObjectPose] = field(default_factory=dict)
    #: Camera frames by camera name, fetched only when read (a strategy without vision pays nothing).
    frames: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """What the strategy wants after one tick."""

    #: Joint targets [app units]. Joints left out keep their target; empty = hold.
    target: Mapping[str, float] = field(default_factory=dict)
    #: The run is over (the task is done, or the strategy gives up).
    done: bool = False
    #: The task's result when known: success, and the task's error (e.g. metres from the goal).
    success: bool | None = None
    error: float | None = None
    #: A short status line for the panel.
    note: str = ""


@runtime_checkable
class Strategy(Protocol):
    """A control strategy. Required: `hz` and `act`. Optional: `start(run, obs)` (may raise to
    refuse the run, e.g. a policy trained on another robot), `stop(reason)`, `time_limit_s`."""

    hz: float

    def act(self, obs: Observation) -> Action | Mapping[str, float]: ...


@dataclass
class RunnerStatus:
    running: bool = False
    step: int = 0
    hz: float = 0.0
    stopped_because: str = ""
    done: bool = False
    success: bool | None = None
    error: float | None = None
    note: str = ""


class _Frames(Mapping[str, np.ndarray]):
    """Camera frames fetched on first read, once per tick."""

    def __init__(self, names: Sequence[str], fetch: Callable[[str], np.ndarray]):
        self._names = tuple(names)
        self._fetch = fetch
        self._got: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._names:
            raise KeyError(name)
        if name not in self._got:
            self._got[name] = self._fetch(name)
        return self._got[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class StrategyRunner:
    """Runs one strategy on the twin's arm. See the module docstring for what it guards."""

    def __init__(self, twin, strategy: Strategy, *, owner: str = OWNER,
                 goal: Callable[[], Any] | None = None,
                 objects: Callable[[], Mapping[str, ObjectPose]] | None = None,
                 cameras: Sequence[str] = (), frame: Callable[[str], np.ndarray] | None = None,
                 time_limit_s: float | None = None):
        if not isinstance(strategy, Strategy):
            raise TypeError(f"{type(strategy).__name__} nie jest strategia: potrzeba `hz` i `act(obs)`")
        if not (isinstance(strategy.hz, (int, float)) and math.isfinite(strategy.hz) and strategy.hz > 0):
            raise ValueError(f"strategia: hz musi byc dodatnie, jest {strategy.hz!r}")
        self.twin = twin
        self.strategy = strategy
        self.owner = owner
        self._goal = goal
        self._objects = objects
        self._cameras = tuple(cameras)
        self._frame = frame if frame is not None else twin.render
        limit = getattr(strategy, "time_limit_s", None) if time_limit_s is None else time_limit_s
        self.time_limit_s = float(limit) if limit is not None else None
        self.status = RunnerStatus()
        self._joints: tuple[str, ...] = ()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float | None = None
        self._t_hz, self._ticks = 0.0, 0

    # ------------------------------------------------------------------ start
    def start(self, *, threaded: bool = True, gen: int | None = None,
              guard: Callable[[], bool] | None = None) -> None:
        """Takes the arm and starts. `gen` and `guard` go to `Twin.claim` (see `PolicyRunner.start`):
        a STOP or Home pressed while the strategy was loading must not be overridden by it."""
        if not self.twin.connected:
            raise RuntimeError("ramie blizniaka nie jest polaczone")
        self.stop()
        kw = {k: v for k, v in (("gen", gen), ("guard", guard)) if v is not None}
        self.twin.claim(self.owner, preempt=self._preempted, **kw)
        try:
            spec = self.twin.workspace.spec()
            self._joints = tuple(spec.joints)
            run = RunInfo(robot=spec.name, joints=self._joints, gripper=spec.gripper,
                          limits=dict(self.twin.joint_limits()), simulated=bool(self.twin.status.simulated))
            begin = getattr(self.strategy, "start", None)
            if begin is not None:
                begin(run, self._observe(0.0, 0))
        except Exception:
            self.twin.release(self.owner)
            raise
        self.status = RunnerStatus(running=True)
        self._stop.clear()
        self._t_start, self._ticks = None, 0
        if not self.twin.set_engaged(True, owner=self.owner):
            reason = getattr(self.twin, "preempt_reason", "") or f"ramie ma: {self.twin.owner or 'nikt'}"
            self.status.running = False
            self.status.stopped_because = f"przerwana: {reason}"
            raise RuntimeError(f"ramie odebrane ({reason})")
        if threaded:
            th = threading.Thread(target=self._loop, name=self.owner, daemon=True)
            self._thread = th
            th.start()

    # ------------------------------------------------------------------- loop
    def _loop(self) -> None:
        period = 1.0 / float(self.strategy.hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self.step_once(t0):
                return
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def step_once(self, now: float | None = None) -> bool:
        """One tick. False = the run is over (the reason in `status.stopped_because`).

        `now` [s]: the thread passes the wall clock, a test the simulated time.
        """
        if not self.status.running:
            return False
        now = time.monotonic() if now is None else now
        if self._t_start is None:
            self._t_start = self._t_hz = now
        st = self.status
        try:
            if not self.twin.connected:
                self._halt("ramie rozlaczone")
                return False
            t = now - self._t_start
            if self.time_limit_s is not None and t >= self.time_limit_s:
                self._halt("limit czasu")
                return False
            action = _as_action(self.strategy.act(self._observe(t, st.step)))
            target = self._checked(action.target)
            if target:
                self.twin.set_target(target, owner=self.owner)
            st.step += 1
            st.stopped_because = ""
            st.note, st.success, st.error = action.note, action.success, action.error
            self._ticks += 1
            if now - self._t_hz >= 1.0:
                st.hz, self._ticks, self._t_hz = self._ticks / (now - self._t_hz), 0, now
            if action.done:
                st.done = True
                self._halt("zadanie wykonane" if action.success else "strategia skonczyla")
                return False
        except _BadTarget as exc:
            self._halt(f"zly cel strategii: {exc}", hold_measured=True)
            return False
        except Exception as exc:
            if self.twin.owner != self.owner and self._stop.is_set():
                return False                                # stopped by `stop` during the tick
            logger.exception("Strategia przerwana")
            self._halt(f"{type(exc).__name__}: {exc}", hold_measured=True)
            return False
        return True

    def _observe(self, t: float, step: int) -> Observation:
        status = self.twin.status
        measured = dict(status.measured) or self.twin.joints()
        goal = self._goal() if self._goal is not None else None
        return Observation(
            t=t, step=step, joints=measured, command=dict(status.command or {}),
            goal=None if goal is None else np.asarray(goal, float),
            objects=dict(self._objects() or {}) if self._objects is not None else {},
            frames=_Frames(self._cameras, self._frame),
        )

    def _checked(self, target: Mapping[str, float]) -> dict[str, float]:
        # A NaN reaching the supervisor drove a gripper past closed (grasp student, 2026-09-27).
        out = {}
        for name, value in target.items():
            if name not in self._joints:
                raise _BadTarget(f"staw {name!r} spoza ramienia")
            v = float(value)
            if not math.isfinite(v):
                raise _BadTarget(f"{name} = {value!r}")
            out[name] = v
        return out

    # ------------------------------------------------------------------- stop
    def _preempted(self) -> None:
        """The twin took the arm (Home, STOP, reconnect, another owner): only stop the thread."""
        self.stop(f"przerwana: {getattr(self.twin, 'preempt_reason', '') or 'ramie odebrane'}")

    def stop(self, reason: str = "") -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th is not threading.current_thread() and th.is_alive():
            th.join(timeout=2.0)
        self._thread = None
        if self.status.running:
            self.status.running = False
            self.status.stopped_because = reason or "zatrzymana"
            self._let_go(hold_measured=False)
            self._tell_strategy(self.status.stopped_because)

    def _halt(self, reason: str, hold_measured: bool = False) -> None:
        self.status.running = False
        self.status.stopped_because = reason
        self._let_go(hold_measured)
        self._stop.set()
        self._tell_strategy(reason)
        logger.info("Strategia zatrzymana: %s", reason)

    def _let_go(self, hold_measured: bool) -> None:
        """Gives the arm back, only if it is still ours. `hold_measured`: the arm stops where it IS,
        not at a target that may lie behind an obstacle."""
        if self.twin.owner != self.owner:
            return
        if hold_measured:
            self.twin.hold_measured()
        self.twin.set_engaged(False, owner=self.owner)
        self.twin.release(self.owner)

    def _tell_strategy(self, reason: str) -> None:
        end = getattr(self.strategy, "stop", None)
        if end is None:
            return
        try:
            end(reason)
        except Exception:                                    # the arm is already safe; only report
            logger.exception("stop() strategii rzucilo wyjatek")


class _BadTarget(ValueError):
    pass


def _as_action(out: Any) -> Action:
    """The strategy's answer as an `Action`: an `Action`, an object with its attributes, or a mapping."""
    if isinstance(out, Action):
        return out
    if isinstance(out, Mapping):
        return Action(target=dict(out))
    target = getattr(out, "target", None)
    if not isinstance(target, Mapping):
        raise TypeError(f"strategia zwrocila {type(out).__name__}: potrzeba Action, obiektu z `target` "
                        "albo mapowania staw -> cel")
    return Action(target=dict(target), done=bool(getattr(out, "done", False)),
                  success=getattr(out, "success", None), error=getattr(out, "error", None),
                  note=str(getattr(out, "note", "") or ""))
