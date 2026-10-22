"""The control-strategy contract: one way for any strategy to drive the arm, in simulation and on hardware.

A strategy (an RL policy, a modular planner, a VLA model) turns what the twin sees into joint
targets. It never touches the backend: the twin's runner claims the arm, builds an `Observation`
every tick, calls the strategy, and sends its target through the twin's safety supervisor.
Simulation and hardware run the same code path.

A strategy from another package depends on this package, never on the twin. It subclasses
`Strategy`, reads the observation's attributes, and returns an `Action`, any object with the same
attributes, or a plain mapping of joint targets. Such packages register their strategies
under the `machinekind_twin.strategies` entry point group (`entry_points.STRATEGIES`), and the twin
builds each one from its settings.

The twin's runner ends a run when `act` raises or a target names a joint the robot does not have
or is not a finite number (the arm then holds where it measures), when the twin takes the arm away
(Home, STOP, reconnect, another owner), when the arm disconnects, and when the time limit runs out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

__all__ = ["Action", "ActionLike", "Frame", "ObjectPose", "Observation", "RunInfo", "S", "Strategy"]

#: The type of a strategy's settings (`Strategy.settings`). Invariant, because `settings` is a
#: writable class attribute: code that takes any strategy annotates it `Strategy[Any]`.
S = TypeVar("S")


@dataclass(frozen=True)
class ObjectPose:
    """Where an object is, in the robot's base frame."""

    #: Position [m], shape (3,).
    position: np.ndarray
    #: Rotation matrix, shape (3, 3). Identity when the perception knows no orientation.
    rotation: np.ndarray
    #: Where the pose comes from, as the twin labels it (in Polish): e.g. "kamery" (the cameras),
    #: "w dloni" (in the gripper), "ostatnie widziane" (last seen), "symulacja" (the simulation).
    #: A detector's object is "kamery" while the cameras saw it within the last fraction of a
    #: second, "ostatnie widziane" when the pose is older, "w dloni" while the gripper holds it.
    source: str = ""
    #: Seconds since the frames the pose was measured from were captured, at the moment the
    #: observation was built; 0 for a pose that follows the jaws.
    age_s: float = 0.0


@dataclass(frozen=True)
class Frame:
    """One camera frame."""

    #: RGB image, shape (height, width, 3).
    image: np.ndarray
    #: Seconds since the frame was captured, at the moment it was read. A simulated camera renders
    #: on read, so its age is the render time; a real camera's is the age of its stream's latest frame.
    age_s: float


@dataclass(frozen=True)
class RunInfo:
    """What a strategy learns once, when the run starts."""

    robot: str
    #: Joints in app order: degrees, the gripper 0..100.
    joints: tuple[str, ...]
    gripper: str | None
    #: The limits that hold on this arm now [app units]: the description's, narrowed to the hardware's.
    limits: Mapping[str, tuple[float, float]]
    #: Highest commanded speed [app units / s] when the run starts: the supervisor's limit (the description's
    #: `max_vel` x `velocity_scale`, under the panel's speed cap). The operator may lower the cap during the run.
    max_vel: Mapping[str, float]
    #: The robot's folder: `robot.yaml` and the model, for a strategy with its own kinematics.
    folder: Path
    simulated: bool


@dataclass(frozen=True)
class Observation:
    """What the strategy sees in one tick. Units: degrees, gripper 0..100, metres, base frame."""

    #: Seconds since the run started.
    t: float
    #: Tick number, 0 = the first.
    step: int
    #: Measured joint positions.
    joints: Mapping[str, float]
    #: The supervisor's latest command: where the servos are being told to go (it trails the target).
    command: Mapping[str, float]
    #: The task's goal point, shape (3,); None = no goal.
    goal: np.ndarray | None = None
    #: Objects the perception knows about, by a detector's label, e.g. "cube" or "red pen"; empty
    #: when no detector runs.
    objects: Mapping[str, ObjectPose] = field(default_factory=dict)
    #: Camera frames by camera name, fetched only when read: a strategy without vision pays nothing.
    #: A camera without a frame (not open yet, stream broken) is missing.
    frames: Mapping[str, Frame] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """What the strategy wants after one tick."""

    #: Joint targets [app units]. Joints left out, or an empty target, keep driving to the previous one.
    target: Mapping[str, float] = field(default_factory=dict)
    #: The run is over: the task is done, or the strategy gives up.
    done: bool = False
    #: Whether the task succeeded, when known.
    success: bool | None = None
    #: The task's error, when known (e.g. metres from the goal).
    error: float | None = None
    #: A short status line for the panel.
    note: str = ""


class ActionLike(Protocol):
    """A strategy's own answer object with `Action`'s attributes. Only `target` is required."""

    @property
    def target(self) -> Mapping[str, float]: ...


class Strategy(Protocol[S]):
    """A control strategy: a protocol with defaults, meant to be subclassed.

    Subclass it with the settings type as the parameter, `class Mine(Strategy[MineSettings])`, and
    define `hz` and `act`. The base supplies defaults for the rest: `settings` is None, `start` and
    `stop` do nothing, `time_limit_s` is None. A type checker checks each member a subclass defines
    against this contract.

    `settings` is a plain class attribute holding the dataclass of what the operator sets for one
    run, so a subclass overrides it with `settings = MineSettings`; `S` is invariant for that. The
    twin's `strategies` builds the dataclass and creates the strategy as `cls(settings)`, or
    `cls()` when `settings` is None. `start(run, obs)` may raise to refuse the run (e.g. a policy
    trained on another robot). `stop(reason)` is called exactly once after a successful `start`.

    At runtime the twin needs only the attributes, not the base class: a class with `hz` and `act`
    runs. A type checker, though, matches such a class against the protocol only when it defines
    every member here, with its type; subclassing is simpler.
    """

    #: Ticks per second: how often the twin calls `act`.
    hz: float
    #: The settings dataclass; None = the strategy takes no settings.
    settings: type[S] | None = None
    #: End the run after this many seconds; None = no limit of the strategy's own.
    time_limit_s: float | None = None

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Before the first tick. Raise to refuse the run."""

    def stop(self, reason: str) -> None:
        """After the last tick, once, with why the run ended."""

    def act(self, obs: Observation) -> Action | ActionLike | Mapping[str, float]: ...
