"""Joints of the robots the twin only watches, read from their backends opened read-only.

Each such robot gets its own thread (`PassiveReader`): open the backend read-only, read it about ten
times a second, hand every fresh pose on. When the backend cannot be opened or a read fails, the
robot keeps its last pose, `status` says why, and the reader tries again a few seconds later.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .backends.factory import ReadOnlyArm

logger = logging.getLogger(__name__)


class PassiveReader:
    """Reads one robot's joints in a thread and hands them to `apply`. Never sends anything."""

    #: Pause between reads [s].
    period = 0.1
    #: Pause before opening the backend again after a failure [s].
    retry_s = 3.0

    def __init__(self, name: str, open_arm: Callable[[], ReadOnlyArm],
                 apply: Callable[[dict[str, float]], None]):
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

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"robot-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
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
                self._set(False, "brak odpowiedzi - zostaje ostatnia poza")
                return self.period
            self._apply(joints)
            faults = self._arm.faults()
            self._set(True, "odczyt na zywo" + (f" ({'; '.join(faults)})" if faults else ""))
            return self.period
        except Exception as exc:  # noqa: BLE001 - any failure keeps the last pose and retries
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
        try:
            arm.disconnect()
        except Exception:
            logger.exception("Robot %s: blad przy zamykaniu backendu", self.name)
