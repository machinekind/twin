"""The robot backend contract: how the twin talks to an arm.

The twin talks to every arm through `RobotBackend`, so its simulated arm and a real one are
interchangeable. A backend from another package (a robot's own repository) depends on this
package, never on the twin: it implements the `BackendPlugin` protocol (or subclasses
`RobotBackend`), and the twin's factory wraps a plugin into a `RobotBackend`. Such a package
registers a factory under the `lerobot_mp.backends` entry point group (`entry_points.BACKENDS`);
the twin calls it with a `BackendContext`, plain data with everything the backend may use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["BackendContext", "BackendPlugin", "RobotBackend", "RobotInfo"]


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


@runtime_checkable
class BackendPlugin(Protocol):
    """What a backend from another package implements; units as in `BackendContext`.

    Required: `connect`, `disconnect`, `read_joints`, `send_joints`, `is_connected`. Optional,
    with the meaning given on `RobotBackend`: `info` (an object with `name`, `description`,
    `simulated`), `link_silent` (or `_link_silent`), `faults()`, `joint_limits()`, `set_joint_limits(limits)`,
    `gripper_range_q()`, `calibration_warnings()`, `efforts()`, `velocities()`. Any other public
    method stays reachable through the wrapper (the panel looks some up by name).
    """

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joints(self) -> dict[str, float]: ...

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]: ...

    @property
    def is_connected(self) -> bool: ...


@dataclass
class RobotInfo:
    """Opis podlaczonego robota (do HUD i logow)."""

    name: str
    description: str = ""
    simulated: bool = True


class RobotBackend(ABC):
    """Minimalny interfejs: polacz, czytaj pozycje, wysylaj pozycje, rozlacz."""

    info: RobotInfo

    @abstractmethod
    def connect(self) -> None:
        """Nawiazuje polaczenie (dla symulatora - inicjalizuje stan)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Zamyka polaczenie; powinno byc bezpieczne do wielokrotnego wywolania."""

    @abstractmethod
    def read_joints(self) -> dict[str, float]:
        """Zwraca aktualne pozycje stawow: {nazwa_stawu: wartosc}."""

    @abstractmethod
    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Wysyla zadane pozycje stawow. Zwraca to, co faktycznie poszlo do robota.

        Moze zwrocic mniej stawow niz dostal - albo nic, gdy backend wstrzymal
        wysylke (np. `feetech` przy zerwanym laczu, zeby rozkazy nie czekaly
        w kolejce sieci i nie dojechaly potem do serw seria). Wolajacy zostaje
        wtedy przy poprzednim rozkazie.
        """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    def link_silent(self) -> bool:
        """Ostatni `read_joints` nie dostal ZADNEJ swiezej odpowiedzi - oddal stare pozycje.

        Blizniak nie liczy wtedy takiego odczytu jako pomiaru i wstrzymuje nadzor.
        Backend bez lacza (symulator) - zawsze False. Domyslnie czyta `_link_silent`,
        ktore ustawiaja `feetech` i `lerobot` (i atrapy w testach).
        """
        return bool(getattr(self, "_link_silent", False))

    def faults(self) -> list[str]:
        """Usterki sprzetu widziane w ostatnich odpowiedziach - czytelne zdania po polsku.

        Pusta lista = wszystko w porzadku. Serwo w ochronie (przeciazenie,
        przegrzanie, napiecie) zwalnia moment, ale dalej grzecznie odpowiada,
        wiec bez tego wyglada na zdrowe, a reszta ramienia jedzie dalej.
        Nigdy nie rzuca wyjatkiem - wola to petla sterowania co cykl.
        """
        return []

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Twarde limity samego sprzetu w jednostkach aplikacji: {staw: (min, max)}.

        Tylko stawy, ktore sprzet naprawde ogranicza (np. limity kata w EEPROM
        serw). Wyzsze warstwy zawezaja nimi swoje zakresy, zeby rozkaz nigdy
        nie byl przycinany po cichu przez serwo.
        """
        return {}

    def set_joint_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:  # noqa: B027
        """The limits [app units] `send_joints` clamps to from now on; joints not given keep theirs.

        The twin narrows them to the hardware's own limits and widens them for a joint that stands
        outside its range and is being led back. Without it a backend clamping to its connect-time
        limits made the first command of such a joint jump to the limit at full servo speed.
        """

    def gripper_range_q(self) -> tuple[float, float] | None:
        """(closed, open) of the gripper joint [rad of the jaw joint] that THIS backend maps 0..100 onto.

        The twin compares it with the robot description's `gripper_range_q`. A backend mapping 0..100
        differently (LeRobot, by its own calibration range) gave the policy and the cube tracking a
        different jaw angle than the real one, and nothing showed it. None = no mapping to report.
        """
        return None

    def calibration_warnings(self) -> list[str]:
        """Niefatalne zastrzezenia do kalibracji ramienia (zdania dla panelu). Nigdy nie rzuca.

        Np. `lerobot` liczy stopnie od srodka zakresu z kalibracji, a blizniak od
        tiku `center_ticks` - ramie jedzie, ale katy sa przesuniete wzgledem modelu.
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

    def step(self, dt: float) -> None:  # noqa: B027 - domyslnie nic nie robi
        """Krok symulacji. Prawdziwy robot ignoruje (rzeczywistosc liczy sie sama)."""

    def __enter__(self) -> "RobotBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
