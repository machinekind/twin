"""The control-strategy contract: one way for any strategy to drive the arm, in simulation and on hardware.

A strategy (an RL policy, a modular planner, a VLA model) turns what the twin sees into joint
targets. It never touches the backend: the twin's runner claims the arm, builds an `Observation`
every tick, calls the strategy, and sends its target through the twin's safety supervisor.
Simulation and hardware run the same code path.

A strategy from another package depends on this package, never on the twin. It follows the
`Strategy` protocol, reads the observation's attributes, and returns an `Action`, any object with
the same attributes, or a plain mapping of joint targets. Such packages register their strategies
under the `lerobot_mp.strategies` entry point group (`entry_points.STRATEGIES`), and the twin
builds each one from its settings.

The twin's runner ends a run when a target names a joint the robot does not have or is not a
finite number (the arm then holds where it measures), when the twin takes the arm away (Home,
STOP, reconnect, another owner), when the arm disconnects, and when the time limit runs out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

__all__ = ["Action", "ActionLike", "Frame", "ObjectPose", "Observation", "RunInfo", "S", "Strategy"]

#: The type of a strategy's settings (`Strategy.settings`).
S = TypeVar("S", covariant=True)


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
    #: Highest commanded speed [app units / s]: the supervisor never goes faster.
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
    #: The goal set in the panel, shape (3,); None = no goal.
    goal: np.ndarray | None = None
    #: Objects the perception knows about, by name (e.g. "cube").
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
    """A control strategy.

    Required: `hz` and `act`. Optional: `settings`, a class attribute holding the dataclass of what
    the operator sets for one run (the twin's `strategies` builds it and creates the strategy as
    `cls(settings)`); `start(run, obs)`, which may raise to refuse the run (e.g. a policy trained
    on another robot); `stop(reason)`, called exactly once after a successful `start`; and
    `time_limit_s`.

    The generic `S` is the settings type, and it exists only for the author's type checker. A
    strategy need not inherit from this protocol: the same attributes are enough. A package that
    wants the check writes `_: type[Strategy[MySettings]] = MyStrategy`. A type checker counts
    every protocol member as required, so a strategy without settings says `settings = None` to
    pass it.
    """

    hz: float

    # A read-only property here, so that a class attribute `settings = MySettings` matches it.
    @property
    def settings(self) -> type[S] | None: ...

    def act(self, obs: Observation) -> Action | ActionLike | Mapping[str, float]: ...
