"""The robot backend contract: how the twin talks to an arm.

The twin talks to every arm through `RobotBackend`, so its simulated arm and a real one are
interchangeable. A backend from another package (a robot's own repository) depends on this
package, never on the twin: it implements the `BackendPlugin` protocol (or subclasses
`RobotBackend`), and the twin's factory wraps a plugin into a `RobotBackend`. Such a package
registers a factory under the `machinekind_twin.backends` entry point group (`entry_points.BACKENDS`);
the twin calls it with a `BackendContext`, plain data with everything the backend may use.

The twin may also open a backend only to watch an arm it does not drive (another robot in the
scene): `BackendContext.read_only`, set only for a factory marked with `read_only_capable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo", "read_only_capable",
           "supports_read_only"]

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class BackendContext:
    """What the twin hands a backend factory. Plain data: a plugin never imports the twin."""

    #: Name of the robot description the twin runs (`robot.yaml`'s `name`).
    robot: str
    #: Joints in app order: degrees, the gripper 0..100.
    joints: tuple[str, ...]
    #: The gripper joint, None without one.
    gripper: str | None
    #: Joint limits [app units] at connect time; the twin narrows or widens them later through
    #: `set_joint_limits`.
    limits: Mapping[str, tuple[float, float]]
    #: Highest commanded speed [app units / s].
    max_vel: Mapping[str, float]
    #: Where to find the arm (serial port, CAN interface); None = the backend's default.
    port: str | None
    #: The backend's own section of the twin's YAML config, unvalidated: the plugin checks it.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: True: the twin only reads this arm. The backend then writes nothing to the hardware, also not
    #: in `connect` and `disconnect` (no torque, no goals, no settings), and `send_joints` raises.
    #: The twin sets it only for a factory marked with `read_only_capable`.
    read_only: bool = False


def read_only_capable(factory: _F) -> _F:
    """Marks a backend factory that honours `BackendContext.read_only`."""
    factory.supports_read_only = True  # type: ignore[attr-defined]
    return factory


def supports_read_only(factory: Any) -> bool:
    """Whether `factory` is marked with `read_only_capable`."""
    return getattr(factory, "supports_read_only", False) is True


@runtime_checkable
class BackendPlugin(Protocol):
    """What a backend from another package implements; units as in `BackendContext`.

    Required: `connect`, `disconnect`, `read_joints`, `send_joints`, `is_connected`. Optional,
    with the meaning given on `RobotBackend`: `info` (an object with `name`, `description`,
    `simulated`), `link_silent` (or `_link_silent`), `faults()`, `joint_limits()`, `set_joint_limits(limits)`,
    `gripper_range_q()`, `calibration_warnings()`, `efforts()`, `velocities()`, `set_grip_strength(strength)`.
    Any other public method stays reachable through the wrapper (the panel looks some up by name).
    """

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joints(self) -> dict[str, float]: ...

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]: ...

    @property
    def is_connected(self) -> bool: ...


@dataclass
class RobotInfo:
    """The connected robot, for the HUD and the logs."""

    name: str
    description: str = ""
    simulated: bool = True


class RobotBackend(ABC):
    """How the twin talks to an arm: connect, read joint positions, send joint targets, disconnect.

    A subclass implements the abstract members; every other method has a default.
    """

    info: RobotInfo

    @abstractmethod
    def connect(self) -> None:
        """Connects to the arm (a simulator initialises its state)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection. Must be safe to call more than once."""

    @abstractmethod
    def read_joints(self) -> dict[str, float]:
        """The current joint positions [app units]: {joint name: value}."""

    @abstractmethod
    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Sends joint targets [app units]. Returns what actually went to the robot.

        It may return fewer joints than it got, or none when the backend holds the send back (e.g.
        `feetech` on a broken link, so that commands do not queue up in the network and later reach
        the servos in a burst). The caller then stays with its previous command.
        """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    def link_silent(self) -> bool:
        """The last `read_joints` got NO fresh reply and returned stale positions.

        The twin then does not count that reading as a measurement and pauses supervision. A
        backend without a link (the simulator): always False. By default it reads `_link_silent`,
        which `feetech`, `lerobot` and `a1x` set (and the test doubles).
        """
        return bool(getattr(self, "_link_silent", False))

    def faults(self) -> list[str]:
        """Hardware faults seen in the latest replies, as readable sentences in Polish, like the panel.

        An empty list = all is well. A servo in protection (overload, overheating, voltage) releases
        its torque but keeps replying normally, so without this it looks healthy while the rest of
        the arm drives on. Never raises: the control loop calls it every cycle.
        """
        return []

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """The hardware's own hard limits [app units]: {joint: (min, max)}.

        Only the joints the hardware really limits (e.g. angle limits in the servos' EEPROM). The
        layers above narrow their ranges to these, so that a servo never silently clips a command.
        """
        return {}

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:  # noqa: B027
        """The limits [app units] `send_joints` clamps to from now on; joints not given keep theirs.

        The twin narrows them to the hardware's own limits and widens them for a joint that stands
        outside its range and is being led back. Without this, a backend that clamps to its
        connect-time limits makes the first command of such a joint jump to the limit at full servo
        speed.
        """

    def gripper_range_q(self) -> tuple[float, float] | None:
        """(closed, open) of the gripper joint [rad of the jaw joint] that THIS backend maps 0..100 onto.

        The twin compares it with the robot description's `gripper_range_q`: a backend that maps
        0..100 differently (LeRobot, by its own calibration range) silently gives the policy and the
        cube tracking a jaw angle other than the real one. None = no mapping to report.
        """
        return None

    def calibration_warnings(self) -> list[str]:
        """Non-fatal concerns about the arm's calibration (sentences for the panel). Never raises.

        E.g. `lerobot` counts degrees from the middle of its calibrated range and the twin from the
        `center_ticks` tick: the arm moves, but its angles are offset from the model's.
        """
        return []

    def efforts(self) -> dict[str, float] | None:
        """Arm joint efforts (torque) from the latest fresh reading; gripper excluded. Must not raise.

        Units depend on the hardware: A1X uses the raw 0x052 scale (after galaxeo's `EFF_SCALE`, not N*m).
        None = not measured by this backend, or the reading is stale/frozen.
        """
        return None

    def velocities(self) -> dict[str, float] | None:
        """Arm joint velocities in deg/s from the latest fresh reading; gripper excluded. Must not raise.

        None = not measured by this backend, or the reading is stale/frozen (callers then differentiate positions).
        """
        return None

    def set_grip_strength(self, strength: int | None) -> bool:
        """How hard the gripper holds from now on, in this backend's scale; None puts its own back.

        The scale is the hardware's. On a position-controlled servo it is the position loop's P gain:
        the squeeze on a held object is about P times how far the goal lies past it (Feetech STS:
        register 21, 1..254). The backend's own value is the one it had before the first call.

        The twin calls this from its control loop, between `read_joints` and `send_joints`, so a backend
        needs no lock for it. False = this backend or arm cannot (no gripper, no such setting,
        read-only, the servo did not take it): the twin logs it and goes on. Must not raise.
        """
        return False

    def step(self, dt: float) -> None:  # noqa: B027 - does nothing by default
        """One simulation step. A real robot ignores it: reality advances on its own."""

    def __enter__(self) -> "RobotBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
