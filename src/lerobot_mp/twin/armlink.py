"""`galaxeo.arm` transport over the twin runtime: guarded moves go through the twin, never around it.

    link = TwinTransport(twin, owner="specs", cancel=driver_cancel_event)
    motion = Motion(link, arm.lo, arm.hi)
    motion.move(goal_rad, 20.0)

`write` is `Twin.set_target` as `owner`, so the panel, STOP, Home, the supervisor and
arm ownership keep working, and the twin's backend stays the only writer on the CAN bus.
`read` is the twin loop's latest measurement. This module is the one place where the
twin's app units (degrees, gripper 0..100) meet galaxeo's radians (arm joints only).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping

import numpy as np
from galaxeo.arm import JOINTS, Reading
from galaxeo.arm.transport import EFFORT_DESCEND, EFFORT_MOVE, SIM_EFFORT_FRAC, SIM_EFFORT_FRAC_DESCEND

from . import scene as sc
from .runtime import Twin


def to_q(joints: Mapping[str, float]) -> np.ndarray:
    """Twin joints [deg] -> galaxeo arm vector [rad], `JOINTS` order."""
    return np.radians([float(joints[j]) for j in JOINTS])


def to_joints(q) -> dict[str, float]:
    """galaxeo arm vector [rad] -> twin joints [deg]."""
    return {j: math.degrees(float(v)) for j, v in zip(JOINTS, q, strict=True)}


class TwinTransport:
    """`galaxeo.arm` transport (now / sleep / read / write / effort_limits) on a `Twin`."""

    sim_effort_frac = SIM_EFFORT_FRAC
    sim_effort_frac_descend = SIM_EFFORT_FRAC_DESCEND

    def __init__(self, twin: Twin, owner: str, *, cancel: threading.Event | None = None,
                 on_lost: Callable[[], None] | None = None):
        self.twin = twin
        self.owner = owner
        self.rate_hz = float(twin.loop_hz)
        self._cancel = cancel or threading.Event()
        self._on_lost = on_lost
        self._start = self._last = time.monotonic()
        self._paused = 0.0
        self._prev: tuple[float, np.ndarray] | None = None
        self._dq = np.zeros(len(JOINTS))
        self._effort = np.zeros(len(JOINTS))
        self._grip: float | None = None

    # ---- clock: stands still while the twin loop is frozen (link unconfirmed) -----------
    def now(self) -> float:
        wall = time.monotonic()
        if self.twin._frozen():
            self._paused += wall - self._last
        self._last = wall
        return wall - self._start - self._paused

    def sleep(self, secs: float) -> None:
        # Bounded in wall time: a loop frozen for good must end the move on stale feedback, not hang it.
        until = self.now() + secs
        wall_end = time.monotonic() + secs + 0.2
        while not self._cancel.is_set():
            left = until - self.now()
            if left <= 0 or time.monotonic() >= wall_end:
                return
            time.sleep(min(left, 0.005))

    # ---- feedback -------------------------------------------------------------------------
    def _loop_now(self) -> float:
        return self.twin._clock if self.twin._manual else time.monotonic()

    def read(self) -> Reading | None:
        tw = self.twin
        b = tw.backend
        st = tw.status
        if b is None or not st.connected or not st.measured:
            return None
        q = to_q(st.measured)
        t = self.now() - max(0.0, self._loop_now() - st.measured_t)
        frozen = tw._frozen()

        dq = None
        if not b.info.simulated:
            vel = b.velocities()
            eff = b.efforts()
            if eff is None:
                frozen = True                            # the backend has no fresh frame
            else:
                self._effort = np.array([float(eff[j]) for j in JOINTS])
            if vel is not None:
                dq = np.radians([float(vel[j]) for j in JOINTS])
        else:
            self._effort = self._sim_efforts(self._effort)
        if dq is None:
            dq = self._differentiate(st.measured_t, q)
        return Reading(t, q, dq, self._effort.copy(), frozen)

    def _differentiate(self, t: float, q: np.ndarray) -> np.ndarray:
        prev = self._prev
        if prev is None or t != prev[0]:
            if prev is not None and t > prev[0]:
                self._dq = (q - prev[1]) / (t - prev[0])
            self._prev = (t, q)
        return self._dq.copy()

    def _sim_ids(self) -> list[int]:
        model = self.twin.scene.model
        return [model.actuator(sc.PREFIX + j).id for j in JOINTS]

    def _sim_efforts(self, last: np.ndarray) -> np.ndarray:
        tw = self.twin
        if not tw.lock.acquire(timeout=0.005):
            return last
        try:
            return np.array(tw.scene.data.actuator_force[self._sim_ids()], float)
        finally:
            tw.lock.release()

    def effort_limits(self, descend: bool) -> np.ndarray | None:
        b = self.twin.backend
        if b is None:
            return None
        if not b.info.simulated:
            if b.efforts() is None:
                return None
            return np.full(len(JOINTS), EFFORT_DESCEND if descend else EFFORT_MOVE)
        frac = self.sim_effort_frac_descend if descend else self.sim_effort_frac
        with self.twin.lock:
            model = self.twin.scene.model
            r = model.actuator_forcerange[self._sim_ids()]
        return frac * np.maximum(np.abs(r[:, 0]), np.abs(r[:, 1]))

    # ---- command --------------------------------------------------------------------------
    def write(self, p_des) -> None:
        target = to_joints(p_des)
        if self._grip is not None and self.twin.spec.gripper:
            target[self.twin.spec.gripper] = self._grip
        self._set(target)

    def grip(self, value: float | None) -> None:
        """Gripper target [0..100] sent with every write from now on (None = leave it alone)."""
        self._grip = None if value is None else float(value)
        if self._grip is not None:
            self._set({self.twin.spec.gripper: self._grip})

    def _set(self, target: Mapping[str, float]) -> None:
        try:
            self.twin.set_target(target, owner=self.owner)
        except RuntimeError:
            # The arm was taken away (STOP, Home, another owner): it already holds its measured pose.
            if self._on_lost is not None:
                self._on_lost()

    def command_q(self) -> np.ndarray:
        """The twin's current command (else the scene pose) [rad]."""
        tw = self.twin
        return to_q(tw.status.command or tw.joints())
