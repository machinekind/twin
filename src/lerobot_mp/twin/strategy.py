"""The control-strategy runner: one way for any strategy to drive the arm, in simulation and on hardware.

A strategy (an RL policy, a modular planner, a VLA model) turns what the twin sees into joint
targets. It never touches the backend: `StrategyRunner` claims the arm, builds an `Observation`
every tick, calls the strategy, and sends its target through `Twin.set_target`, so every command
passes the safety supervisor. Simulation and hardware run the same code path.

The contract a strategy follows (`Observation`, `Action`, `ActionLike`, `RunInfo`, `ObjectPose`,
`Frame` and the `Strategy` protocol) lives in the `lerobot-mp-api` package, `lerobot_mp_api.strategy`.
A strategy from another package depends on that package, never on the twin. This module re-exports
the same classes, so `lerobot_mp.twin.strategy.Action is lerobot_mp_api.Action`. Such packages
register their strategies through entry points, and the twin builds each one from its settings:
see `strategies`.

    runner = StrategyRunner(twin, Waypoints([pose_a, pose_b]), goal=panel_goal, objects=cube_poses)
    runner.start()      # its own thread at `strategy.hz`; tests: `start(threaded=False)` + `step_once`
    ...
    runner.stop()       # the arm holds its last target

The runner ends a run, for every strategy, when:

* `act` raises, or a target names a joint the robot does not have or is not a finite number (the
  arm then holds where it measures);
* the twin takes the arm away (Home, STOP, reconnect, another owner);
* the arm disconnects;
* the time limit runs out.

A joint that cannot follow its command (a collision, a blocked or limp servo) is the twin's own
tracking guard, for every owner: it triggers STOP, which takes the arm from the runner. What counts
as success, and how far a goal may be, is the task's business inside the strategy.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot_mp_api.strategy import Action, ActionLike, Frame, ObjectPose, Observation, RunInfo, S, Strategy

__all__ = [
    "OWNER", "Action", "ActionLike", "Frame", "FrameSource", "ObjectPose", "Observation", "RunInfo",
    "RunnerStatus", "S", "Strategy", "StrategyRunner",
]

logger = logging.getLogger(__name__)

#: Default owner name of the arm while a strategy runs (`Twin.claim`).
OWNER = "strategia"

#: Where frames come from: camera name -> (RGB image or None, capture time on `time.monotonic`).
#: `CameraHub.arm_frame_t` is the default: it serves real cameras' streams and renders simulated
#: ones, and a camera on the arm by the arm: rendered on a simulated arm, the workspace camera of
#: its name on a real one.
FrameSource = Callable[[str], tuple[np.ndarray | None, float]]


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


# --------------------------------------------------------------------------- the runner


@dataclass(eq=False)
class _Run:
    """One run of the strategy. It is also its token: a loop thread holding an older run - a slow
    `act` that outlived `stop` - may not send or finish anything, because that run is over."""

    #: `strategy.start` returned: from then on the strategy gets exactly one `stop`.
    began: bool = False
    t_start: float | None = None
    rate_since: float = 0.0
    rate_ticks: int = 0


class StrategyRunner:
    """Runs one strategy on the twin's arm. The module docstring lists when it ends a run."""

    #: How long `stop` waits for the loop thread [s]. A slower `act` is left behind: its run is over.
    join_timeout_s = 2.0

    def __init__(
        self,
        twin,
        strategy: Strategy[Any],
        *,
        owner: str = OWNER,
        goal: Callable[[], Any] | None = None,
        objects: Callable[[], Mapping[str, ObjectPose]] | None = None,
        cameras: Sequence[str] = (),
        frame: FrameSource | None = None,
        time_limit_s: float | None = None,
    ):
        _check_strategy(strategy)

        self.twin = twin
        self.strategy = strategy
        self.owner = owner
        self.status = RunnerStatus()

        if time_limit_s is None:
            time_limit_s = getattr(strategy, "time_limit_s", None)
        self.time_limit_s = None if time_limit_s is None else float(time_limit_s)

        self._goal = goal
        self._objects = objects
        self._cameras = tuple(cameras)
        self._frame = frame if frame is not None else twin.cameras.arm_frame_t
        self._joints: tuple[str, ...] = ()

        self._run: _Run | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Re-entrant: a preempt fired from inside `set_target` (same thread) ends the run under this lock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ start

    def start(self, *, threaded: bool = True, gen: int | None = None,
              guard: Callable[[], bool] | None = None) -> None:
        """Takes the arm and starts the run.

        `gen` and `guard` go to `Twin.claim` (see `TwinApp._claim_kw`): a STOP or Home pressed while
        the strategy was loading must not be overridden by it.
        """
        if not self.twin.connected:
            raise RuntimeError("ramie blizniaka nie jest polaczone")

        self.stop()

        run = _Run()
        self._claim(gen, guard)
        self._begin(run)
        self._engage(run)

        if threaded:
            self._thread = threading.Thread(target=self._loop, args=(run,), name=self.owner, daemon=True)
            self._thread.start()

    def _claim(self, gen: int | None, guard: Callable[[], bool] | None) -> None:
        conditions = {}
        if gen is not None:
            conditions["gen"] = gen
        if guard is not None:
            conditions["guard"] = guard

        self.twin.claim(self.owner, preempt=self._preempted, **conditions)

    def _begin(self, run: _Run) -> None:
        """Makes `run` the current one and lets the strategy start it (or refuse it)."""
        with self._lock:
            self._run = run
            self.status = RunnerStatus(running=True)
        self._stop.clear()

        try:
            info = self._run_info()
            start = getattr(self.strategy, "start", None)
            if start is not None:
                start(info, self._observe(t=0.0, step=0))
        except Exception as exc:
            self._finish(run, f"strategia odmowila: {exc}")
            raise

        run.began = True

    def _run_info(self) -> RunInfo:
        spec = self.twin.workspace.spec()
        self._joints = tuple(spec.joints)

        return RunInfo(
            robot=spec.name,
            joints=self._joints,
            gripper=spec.gripper,
            limits=dict(self.twin.joint_limits()),
            max_vel=self.twin.speed_limits(),
            folder=spec.root,
            simulated=bool(self.twin.status.simulated),
        )

    def _engage(self, run: _Run) -> None:
        # The arm may have been taken while the strategy was starting.
        if not self.status.running:
            self._tell_strategy(self.status.stopped_because)
            raise RuntimeError(f"ramie odebrane ({self.status.stopped_because})")

        if not self.twin.set_engaged(True, owner=self.owner):
            reason = self._preempt_reason() or f"ramie ma: {self.twin.owner or 'nikt'}"
            self._finish(run, f"przerwana: {reason}")
            raise RuntimeError(f"ramie odebrane ({reason})")

    # ------------------------------------------------------------------- loop

    def _loop(self, run: _Run) -> None:
        period = 1.0 / float(self.strategy.hz)

        while not self._stop.is_set() and run is self._run:
            started = time.monotonic()
            if not self.step_once(started, _run=run):
                return
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    def step_once(self, now: float | None = None, *, _run: _Run | None = None) -> bool:
        """One tick. False = the run is over (the reason is in `status.stopped_because`).

        `now` [s]: the loop thread passes `time.monotonic()`, a test the simulated time.
        """
        run = _run or self._run
        if run is None or run is not self._run or not self.status.running:
            return False

        if now is None:
            now = time.monotonic()

        try:
            return self._tick(run, now)
        except _BadTarget as exc:
            self._finish(run, f"zly cel strategii: {exc}", hold_measured=True)
        except Exception as exc:
            self._fail(run, exc)
        return False

    def _tick(self, run: _Run, now: float) -> bool:
        if run.t_start is None:
            run.t_start = run.rate_since = now
        t = now - run.t_start

        if not self.twin.connected:
            self._finish(run, "ramie rozlaczone")
            return False

        if self.time_limit_s is not None and t >= self.time_limit_s:
            self._finish(run, "limit czasu")
            return False

        answer = self.strategy.act(self._observe(t=t, step=self.status.step))
        action = _as_action(answer)
        target = self._checked(action.target)

        if not self._send(run, target, action):
            return False
        self._count_rate(run, now)

        if action.done:
            self.status.done = True
            self._finish(run, "zadanie wykonane" if action.success else "strategia skonczyla")
            return False

        return True

    def _send(self, run: _Run, target: dict[str, float], action: Action) -> bool:
        """Sends the target and records the tick, unless the run ended while `act` was running."""
        with self._lock:
            if run is not self._run or not self.status.running:
                return False

            if target:
                self.twin.set_target(target, owner=self.owner)

            status = self.status
            status.step += 1
            status.stopped_because = ""
            status.note = action.note
            status.success = action.success
            status.error = action.error

        return True

    def _count_rate(self, run: _Run, now: float) -> None:
        run.rate_ticks += 1
        elapsed = now - run.rate_since
        if elapsed >= 1.0:
            self.status.hz = run.rate_ticks / elapsed
            run.rate_ticks = 0
            run.rate_since = now

    def _fail(self, run: _Run, exc: Exception) -> None:
        if self.twin.owner != self.owner:
            # The arm was taken during the tick (Home, STOP, reconnect): that is the reason, not `exc`.
            self._finish(run, f"przerwana: {self._preempt_reason() or 'ramie odebrane'}")
            return

        logger.exception("Strategia przerwana")
        self._finish(run, f"{type(exc).__name__}: {exc}", hold_measured=True)

    def _observe(self, *, t: float, step: int) -> Observation:
        status = self.twin.status
        goal = self._goal() if self._goal is not None else None
        objects = self._objects() if self._objects is not None else None

        return Observation(
            t=t,
            step=step,
            joints=dict(status.measured) or self.twin.joints(),
            command=dict(status.command or {}),
            goal=None if goal is None else np.asarray(goal, float),
            objects=dict(objects or {}),
            frames=_LazyFrames(self._cameras, self._frame),
        )

    def _checked(self, target: Mapping[str, float]) -> dict[str, float]:
        """The target as floats, or `_BadTarget`. A NaN must not reach the supervisor: it can drive
        a gripper past closed."""
        checked = {}

        for name, value in target.items():
            if name not in self._joints:
                raise _BadTarget(f"staw {name!r} spoza ramienia")
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise _BadTarget(f"{name} = {value!r} to nie liczba")
            if not math.isfinite(value):
                raise _BadTarget(f"{name} = {value!r}")

            checked[name] = float(value)

        return checked

    # ------------------------------------------------------------------- stop

    def stop(self, reason: str = "") -> None:
        run = self._run
        self._stop.set()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=self.join_timeout_s)
        self._thread = None

        self._finish(run, reason or "zatrzymana")

    def _preempted(self) -> None:
        """The twin took the arm (Home, STOP, reconnect, another owner)."""
        self.stop(f"przerwana: {self._preempt_reason() or 'ramie odebrane'}")

    def _preempt_reason(self) -> str:
        return getattr(self.twin, "preempt_reason", "")

    def _finish(self, run: _Run | None, reason: str, hold_measured: bool = False) -> None:
        """Ends `run` once, whoever gets here first: a tick, `stop`, or the twin taking the arm."""
        with self._lock:
            if run is None or run is not self._run or not self.status.running:
                return

            self.status.running = False
            self.status.stopped_because = reason
            self._run = None
            self._stop.set()

        # Outside the lock: the twin's calls take its own locks.
        self._let_go(hold_measured)
        if run.began:
            self._tell_strategy(reason)

        logger.info("Strategia zatrzymana: %s", reason)

    def _let_go(self, hold_measured: bool) -> None:
        """Gives the arm back, only if it is still ours.

        `hold_measured`: the arm stops where it IS, not at a target that may lie behind an obstacle.
        """
        if self.twin.owner != self.owner:
            return

        if hold_measured:
            self.twin.hold_measured()
        self.twin.set_engaged(False, owner=self.owner)
        self.twin.release(self.owner)

    def _tell_strategy(self, reason: str) -> None:
        stop = getattr(self.strategy, "stop", None)
        if stop is None:
            return

        try:
            stop(reason)
        except Exception:
            # The arm is already safe; a failing `stop` is only worth a log line.
            logger.exception("stop() strategii rzucilo wyjatek")


# --------------------------------------------------------------------------- helpers


class _BadTarget(ValueError):
    pass


class _LazyFrames(Mapping[str, Frame]):
    """Camera frames fetched on first read, once per tick. A camera without a frame is missing."""

    def __init__(self, names: Sequence[str], fetch: FrameSource):
        self._names = tuple(names)
        self._fetch = fetch
        self._fetched: dict[str, Frame | None] = {}

    def __getitem__(self, name: str) -> Frame:
        if name not in self._names:
            raise KeyError(name)

        if name not in self._fetched:
            self._fetched[name] = self._read(name)

        frame = self._fetched[name]
        if frame is None:
            raise KeyError(f"{name}: brak klatki z kamery")
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def _read(self, name: str) -> Frame | None:
        image, captured = self._fetch(name)
        if image is None:
            return None

        return Frame(image=image, age_s=max(0.0, time.monotonic() - captured))


def _check_strategy(strategy: Any) -> None:
    # By attributes, not `isinstance`: a strategy that does not subclass `Strategy` needs only these.
    if not hasattr(strategy, "hz") or not callable(getattr(strategy, "act", None)):
        raise TypeError(f"{type(strategy).__name__} nie jest strategia: potrzeba `hz` i `act(obs)`")

    hz = strategy.hz
    if not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError(f"strategia: hz musi byc dodatnie, jest {hz!r}")


def _as_action(answer: Any) -> Action:
    """The strategy's answer as an `Action`: an `Action`, an object with its attributes, or a mapping."""
    if isinstance(answer, Action):
        return answer

    if isinstance(answer, Mapping):
        return Action(target=dict(answer))

    target = getattr(answer, "target", None)
    if not isinstance(target, Mapping):
        raise TypeError(f"strategia zwrocila {type(answer).__name__}: potrzeba Action, "
                        "obiektu z `target` albo mapowania staw -> cel")

    return Action(
        target=dict(target),
        done=bool(getattr(answer, "done", False)),
        success=getattr(answer, "success", None),
        error=getattr(answer, "error", None),
        note=str(getattr(answer, "note", "") or ""),
    )
