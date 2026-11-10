"""Joints of the robots the twin only watches, read from their backends opened read-only.

Each such robot gets its own thread (`PassiveReader`): open the backend read-only, read it about ten
times a second, hand every fresh pose on. A robot that reports where its base is (`base_pose`) walks
or drives, so it is read thirty times a second. When the backend cannot be opened or a read fails, the
robot keeps its last pose, `status` says why, and the reader tries again a few seconds later.

A robot the panel may drive (`RobotRecord.base_drive`) gets its base velocity from the same thread,
after each read: only that thread talks to the backend.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from .backends.factory import BaseDriveArm, ReadOnlyArm

logger = logging.getLogger(__name__)


class PassiveReader:
    """Reads one robot's joints and base pose in a thread and hands them to `apply`. Sends nothing but the
    base velocity set by `drive`."""

    #: Pause between reads [s].
    period = 0.1
    #: Pause after a read that brought a base pose, or while a base velocity command runs [s].
    base_period = 1 / 30
    #: Pause before opening the backend again after a failure [s].
    retry_s = 3.0
    #: Zero velocities sent after a command ends. Then the reader goes silent: on some robots a zero
    #: keeps the twin in command, and an idle panel must not hold it.
    trailing_zeros = 3

    def __init__(self, name: str, open_arm: Callable[[], ReadOnlyArm],
                 apply: Callable[[dict[str, float], tuple[np.ndarray, float] | None], None]):
        self.name = name
        self._open = open_arm
        self._apply = apply
        self._arm: ReadOnlyArm | None = None
        #: The last read was fresh and applied.
        self.ok = False
        #: What the panel shows about this robot's pose.
        self.status = "laczenie..."
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: The base velocity to hold [m/s, m/s, rad/s] until the `time.monotonic()` moment `_until`.
        self._velocity = (0.0, 0.0, 0.0)
        self._until = 0.0
        #: Zero velocities still to send since the last command ended.
        self._zeros_left = 0
        #: The twin's STOP (`halt`): no command goes out, whatever `drive` sets meanwhile.
        self.halted = False

    def drive(self, forward: float, left: float, turn: float, seconds: float) -> None:
        """Hold this base velocity for `seconds` from now, in place of the running command."""
        self._velocity = (forward, left, turn)
        self._until = time.monotonic() + seconds

    def halt(self) -> None:
        """The twin's STOP: from the next cycle on only the trailing zeros go out, until `resume`."""
        self.halted = True
        self.drive(0.0, 0.0, 0.0, 0.0)

    def resume(self) -> None:
        """The STOP cleared: the robot stays still until the next command."""
        # Zero first, lift second: a cycle that sees `halted` lifted already sees the zero, so a command set
        # while halted never starts. The cycle reads `halted` before the command, so no lock is needed.
        self.drive(0.0, 0.0, 0.0, 0.0)
        self.halted = False

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"robot-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.drive(0.0, 0.0, 0.0, 0.0)                  # whatever the thread still sends is a zero
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            # Still inside a backend call: it sends the last zero and disconnects once that call returns.
            # Closing from here could put both before the command it is sending.
            if self._thread.is_alive():
                return
            self._thread = None
        self._close()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.read_once())
        self._close()

    def read_once(self) -> float:
        """One read (opening the backend first when needed). Returns the pause before the next one."""
        try:
            if self._arm is None:
                arm = self._open()
                self._arm = arm
                arm.connect()
            joints = self._arm.read_joints()
            if self._arm.link_silent:
                # Unheard, the robot walks no further: with the link back it waits for a new command.
                self.drive(0.0, 0.0, 0.0, 0.0)
                self._set(False, "brak odpowiedzi - zostaje ostatnia poza")
                return self.period
            base = self._arm.base_pose()
            self._apply(joints, base)

            live = not self.halted and time.monotonic() < self._until
            if isinstance(self._arm, BaseDriveArm) and live:
                self._zeros_left = self.trailing_zeros
                self._arm.send_velocity(*self._velocity)
            elif isinstance(self._arm, BaseDriveArm) and self._zeros_left > 0:
                self._zeros_left -= 1
                self._arm.send_velocity(0.0, 0.0, 0.0)

            faults = self._arm.faults()
            self._set(True, "odczyt na zywo" + (f" ({'; '.join(faults)})" if faults else ""))
            return self.base_period if base is not None or live else self.period
        except Exception as exc:  # noqa: BLE001 - any failure keeps the last pose and retries
            self.drive(0.0, 0.0, 0.0, 0.0)              # as with a silent link: no old command resumes
            self._set(False, f"{type(exc).__name__}: {exc} - zostaje ostatnia poza")
            self._close()
            return self.retry_s

    def _set(self, ok: bool, status: str) -> None:
        if status != self.status and not ok:
            logger.warning("Robot %s: %s", self.name, status)
        self.ok, self.status = ok, status

    def _close(self) -> None:
        arm, self._arm = self._arm, None
        if arm is None:
            return
        if isinstance(arm, BaseDriveArm) and self._zeros_left > 0:
            try:
                arm.send_velocity(0.0, 0.0, 0.0)            # stop it now, not at the robot's own timeout
            except Exception:
                logger.exception("Robot %s: blad przy zatrzymaniu podstawy", self.name)
        try:
            arm.disconnect()
        except Exception:
            logger.exception("Robot %s: blad przy zamykaniu backendu", self.name)
